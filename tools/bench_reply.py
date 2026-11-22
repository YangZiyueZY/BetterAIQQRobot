"""生产链路基准：让每个模型真的走一遍群聊回复，量实际延迟与可用性。

为什么不能只看短 prompt 的延迟：生产 prompt 带着人设 + 记忆 + 群画像 + 上下文，
输入长度差一个量级，而推理模型的耗时很大一部分花在"读完再想"上。
实测 step-5-preview 短 prompt 1.7s、生产 prompt 却要 10~30s。

用法：
    ./.venv/bin/python -m tools.bench_reply
输出：data/model_reply_bench.json
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qqbot.config import load_config  # noqa: E402
from qqbot.generate.llm import LLMError, LLMTimeoutError  # noqa: E402
from qqbot.generate.prompt import build_system, build_user  # noqa: E402
from qqbot.face_catalog import catalog  # noqa: E402
from qqbot.models import Conversation, Message  # noqa: E402
from qqbot.persona import load as load_persona  # noqa: E402

MODELS = [
    "deepseek-flash",
    "step-5-preview",
    "step-3.7-flash",
    "mimo-v2.6-flash",
    "deepseek-v4-flash-0731",
    "qwen3.8-27b",
]

#: 三种典型场景：寒暄 / 需要接上下文 / 时效性（会触发搜索资料进 prompt）
CASES = [
    ("寒暄", "小七在吗"),
    ("闲聊", "哈哈哈哈哈笑死我了"),
    ("接上下文", "那你刚才说的那个呢"),
    ("技术问题", "小七python怎么合并两个字典"),
    ("表态", "小七你觉得原神和崩铁哪个好玩"),
    ("追问", "为什么啊"),
]

#: 虚构的群号，基准测试只关心 prompt 形状
GROUP = 100001


async def main() -> None:
    cfg = load_config()
    persona = load_persona(cfg.root / "persona.yaml")

    from qqbot.store import Database, Repo
    from qqbot.store.memories import MemoryRepo
    from qqbot.gateway.api import OneBotAPI

    db = await Database.open(cfg.storage.db_path)
    repo = Repo(db)

    class Stub:
        async def send_group_segments(self, *a, **k): return 1
        async def send_group_msg(self, *a, **k): return 1
        async def set_msg_emoji_like(self, *a, **k): pass
        async def get_emoji_likes(self, m): return []
        async def get_login_info(self): return {"nickname": "x", "user_id": 1}
        async def get_group_list(self): return []

    targets = sys.argv[1:] or MODELS
    results = []
    for model in targets:
        cfg.llm.model = model
        cfg.llm.reply_model = model
        cfg.llm.analysis_model = model

        from qqbot.pipeline import Pipeline
        pipe = Pipeline(cfg, repo, Stub())
        rows = []
        for label, text in CASES:
            t0 = time.monotonic()
            try:
                draft = await pipe.generate(
                    _conv(text, persona)
                )
                ms = int((time.monotonic() - t0) * 1000)
                bubbles = [b.render_text() for b in draft.bubbles] if draft else []
                rows.append({
                    "case": label, "ms": ms,
                    "ok": bool(bubbles), "draft": bubbles,
                    "error": "" if bubbles else "清洗后没有可发内容",
                })
            except (LLMError, LLMTimeoutError) as exc:
                ms = int((time.monotonic() - t0) * 1000)
                rows.append({"case": label, "ms": ms, "ok": False,
                             "draft": [], "error": str(exc)[:150]})
            except Exception as exc:  # noqa: BLE001
                ms = int((time.monotonic() - t0) * 1000)
                rows.append({"case": label, "ms": ms, "ok": False,
                             "draft": [], "error": f"{type(exc).__name__}: {exc}"[:150]})
        await pipe.close()

        good = [r for r in rows if r["ok"]]
        lats = [r["ms"] for r in rows]
        summary = {
            "model": model,
            "ok_rate": round(len(good) / len(rows), 2),
            "latency_ms_median": int(statistics.median(lats)) if lats else 0,
            "latency_ms_max": max(lats) if lats else 0,
            "reply_ready": len(good) == len(rows) and statistics.median(lats) < 6000,
            "rows": rows,
        }
        results.append(summary)
        print(f"  ✓ {model:24} 成功率 {summary['ok_rate']:.0%}  "
              f"中位 {summary['latency_ms_median']}ms  最慢 {summary['latency_ms_max']}ms",
              flush=True)

    await db.close()
    dest = pathlib.Path(sys.argv[0] and "data/model_reply_bench.json")
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'模型':<24}{'成功率':<8}{'中位延迟':<10}{'最慢':<10}{'可做回复档'}")
    print("─" * 66)
    for r in results:
        print(f"{r['model']:<24}{r['ok_rate']:.0%}{'':<5}"
              f"{r['latency_ms_median']}ms{'':<4}{r['latency_ms_max']}ms{'':<4}"
              f"{'✓' if r['reply_ready'] else '✗'}")
    print(f"\n明细 -> {dest}")


def _conv(text: str, persona) -> Conversation:
    msg = Message(group_id=GROUP, message_id=1, user_id=11, nickname="测试用户",
                  text=text, ts=int(time.time()),
                  mentions_self=persona.name in text)
    return Conversation(group_id=GROUP, messages=[msg], trigger=msg)


if __name__ == "__main__":
    asyncio.run(main())
