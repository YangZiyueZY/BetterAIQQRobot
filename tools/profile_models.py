"""模型画像探针：把每个模型的"脾气"测出来。

为什么要这个：同一个 OpenAI 兼容接口下，各家模型的行为差别很大 ——
  · 有的返回 reasoning_content，有的把思考混在正文里
  · 有的认 reasoning_effort，有的**静默忽略**（参数发了等于没发）
  · 有的会不填 usage.reasoning_tokens（统计会失真）
  · 有的思维链很重，预算给少了会返回**空正文**
  · 有的不支持 response_format

适配好这些差别，才能把模型放到对的位置（回复档要快、分析档要深）。

用法：
    ./.venv/bin/python -m tools.profile_models                # 全部
    ./.venv/bin/python -m tools.profile_models deepseek-flash  # 指定
输出：data/model_profiles.json
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time

import aiohttp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qqbot.config import load_config  # noqa: E402

#: 需要画像的模型
TARGETS = [
    "deepseek-flash",
    "step-5-preview",
    "step-3.7-flash",
    "mimo-v2.6-flash",
    "deepseek-v4-flash-0731",
    "qwen3.8-27b",
]

#: 单次请求最长等多久
CALL_TIMEOUT = 90.0

#: 同一个模型的并发数（代理那边有限流，别太激进）
CONCURRENCY = 2

# --- 探针用的问题 ------------------------------------------------------------

#: 需要一点推理，但答案唯一、解析快
REASONING_Q = ("一个班 40 人，会篮球 25 人，会足球 20 人，都会 10 人。"
               "两样都不会的有多少人？只回答数字。")

#: 纯闲聊，用来测基础延迟和上下文理解
CHAT_Q = "用一句话回答：今天心情怎么样？"

#: 需要真的思考才会答对
LOGIC_Q = ("如果 3 台机器 3 分钟做 3 个零件，那么 100 台机器做 100 个零件要几分钟？"
           "只回答数字。")


async def call(session, cfg, model, messages, *, max_tokens=800,
               reasoning_effort=None, json_mode=False, temperature=0.7):
    """发一次请求，返回结构化结果。绝不抛异常。"""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {cfg.api_key}",
               "Content-Type": "application/json"}
    started = time.monotonic()
    try:
        async with session.post(f"{cfg.base_url}/chat/completions",
                                json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=CALL_TIMEOUT)) as r:
            raw = await r.text()
            status = r.status
    except asyncio.TimeoutError:
        return {"ok": False, "ms": int((time.monotonic() - started) * 1000),
                "error": f"超时(>{CALL_TIMEOUT:.0f}s)"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ms": int((time.monotonic() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"[:200]}

    ms = int((time.monotonic() - started) * 1000)
    if status != 200:
        return {"ok": False, "ms": ms, "status": status, "error": raw[:220]}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "ms": ms, "error": "响应不是 JSON"}

    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    return {
        "ok": True,
        "ms": ms,
        "content": (msg.get("content") or "").strip(),
        "reasoning_chars": len(reasoning),
        "has_reasoning_field": bool(msg.get("reasoning_content") or msg.get("reasoning")),
        "reasoning_tokens_reported": int(details.get("reasoning_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "finish": str(choice.get("finish_reason") or ""),
    }


async def profile_model(session, cfg, model) -> dict:
    out: dict = {"model": model}

    # ---- 1. 基础：延迟 + 是否返回思考字段 ----
    basic = await call(session, cfg, model,
                       [{"role": "user", "content": CHAT_Q}], max_tokens=600)
    out["basic"] = basic

    # ---- 2. 推理题：思考成本 vs 是否给出正文 ----
    reason = await call(session, cfg, model,
                        [{"role": "user", "content": REASONING_Q}], max_tokens=2000)
    out["reasoning"] = reason

    # ---- 3. reasoning_effort 到底认不认 ----
    low = await call(session, cfg, model,
                     [{"role": "user", "content": LOGIC_Q}], max_tokens=2500,
                     reasoning_effort="low")
    high = await call(session, cfg, model,
                      [{"role": "user", "content": LOGIC_Q}], max_tokens=2500,
                      reasoning_effort="high")
    out["effort_low"] = low
    out["effort_high"] = high
    if low.get("ok") and high.get("ok"):
        # 思考量或耗时明显不同才算"参数生效"
        d_chars = abs(high.get("reasoning_chars", 0) - low.get("reasoning_chars", 0))
        d_ms = abs(high.get("ms", 0) - low.get("ms", 0))
        same_content = (low.get("content") == high.get("content"))
        out["effort_effective"] = bool(
            d_chars > 50 or d_ms > 800 or not same_content
        )
    else:
        out["effort_effective"] = None

    # ---- 4. JSON 模式 ----
    js = await call(session, cfg, model,
                    [{"role": "user",
                      "content": '返回一个 JSON：{"ok": true, "n": 3}。只返回 JSON。'}],
                    max_tokens=400, json_mode=True)
    if js.get("ok"):
        try:
            json.loads(js["content"])
            js["valid_json"] = True
        except Exception:
            js["valid_json"] = False
    out["json_mode"] = js

    # ---- 5. 紧预算下的截断行为（会不会返回空正文）----
    tight = await call(session, cfg, model,
                       [{"role": "user", "content": LOGIC_Q}], max_tokens=120)
    out["tight_budget"] = tight

    # ---- 6. 宽松预算下能否答对逻辑题 ----
    out["wide_budget"] = await call(
        session, cfg, model, [{"role": "user", "content": LOGIC_Q}], max_tokens=6000)

    return out


def summarize(p: dict) -> dict:
    """把原始探针结果提炼成可用的画像。"""
    b = p.get("basic") or {}
    r = p.get("reasoning") or {}
    tight = p.get("tight_budget") or {}
    wide = p.get("wide_budget") or {}
    js = p.get("json_mode") or {}

    ok_any = bool(b.get("ok"))
    # 思维链：有 reasoning 字段，或者思考字符数明显
    thinks = bool(r.get("has_reasoning_field") and r.get("reasoning_chars", 0) > 30)
    # 是否填报 reasoning_tokens
    reports_rt = bool((r.get("reasoning_tokens_reported") or 0) > 0)
    # 紧预算下是否空正文（说明思考吃光预算）
    empty_when_tight = bool(
        tight.get("ok") and not (tight.get("content") or "").strip()
        and tight.get("finish") == "length"
    )
    # 逻辑题答对没（3 台 3 分钟 3 个 → 100 台 100 个也是 3 分钟）
    wide_text = (wide.get("content") or "").strip()
    logic_ok = "3" in wide_text[:12] if wide_text else None

    lat = b.get("ms") or 0
    return {
        "available": ok_any,
        "thinks": thinks,
        "reports_reasoning_tokens": reports_rt,
        "effort_effective": p.get("effort_effective"),
        "json_mode_ok": bool(js.get("ok") and js.get("valid_json")),
        "empty_when_tight": empty_when_tight,
        "reasoning_chars_typical": r.get("reasoning_chars", 0),
        "completion_tokens_typical": r.get("completion_tokens", 0),
        "latency_ms": lat,
        "latency_s": round(lat / 1000, 1),
        "logic_correct": logic_ok,
        "notes": _notes(thinks, reports_rt, empty_when_tight, lat,
                        p.get("effort_effective"), logic_ok),
    }


def _notes(thinks, reports_rt, empty_tight, lat, effort, logic) -> list[str]:
    n = []
    if thinks:
        n.append("有思维链")
    else:
        n.append("无思维链（或未返回 reasoning 字段）")
    if thinks and not reports_rt:
        n.append("不填 reasoning_tokens，统计需按字符数推断")
    if effort is False:
        n.append("reasoning_effort 被静默忽略")
    elif effort is True:
        n.append("reasoning_effort 生效")
    if empty_tight:
        n.append("预算紧时会返回空正文（思考吃光 token）")
    if lat >= 8000:
        n.append("延迟偏高，不适合群聊回复档")
    elif lat and lat <= 3000:
        n.append("延迟低，适合回复档")
    if logic is False:
        n.append("宽松预算下逻辑题答错")
    return n


async def main() -> None:
    cfg = load_config().llm
    targets = sys.argv[1:] or TARGETS

    print(f"画像 {len(targets)} 个模型（单个请求上限 {CALL_TIMEOUT:.0f}s）")
    print(f"endpoint: {cfg.base_url}\n")

    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        async def one(model: str) -> dict:
            async with sem:
                t0 = time.monotonic()
                print(f"  ▶ {model} ...", flush=True)
                raw = await profile_model(session, cfg, model)
                raw["elapsed_s"] = round(time.monotonic() - t0, 1)
                raw["summary"] = summarize(raw)
                s = raw["summary"]
                print(f"  ✓ {model}  {raw['elapsed_s']}s  "
                      f"可用={s['available']} 思维链={s['thinks']} "
                      f"延迟={s['latency_s']}s effort生效={s['effort_effective']}",
                      flush=True)
                return raw

        results = await asyncio.gather(*(one(m) for m in targets))

    dest = pathlib.Path("data/model_profiles.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'模型':<24}{'可用':<6}{'思维链':<8}{'延迟':<8}{'effort':<9}{'JSON':<6}{'紧预算空':<9}")
    print("─" * 72)
    for r in results:
        s = r["summary"]
        print(f"{r['model']:<24}"
              f"{'✓' if s['available'] else '✗':<6}"
              f"{'✓' if s['thinks'] else '—':<8}"
              f"{str(s['latency_s']) + 's':<8}"
              f"{('生效' if s['effort_effective'] else '忽略' if s['effort_effective'] is False else '?'):<9}"
              f"{'✓' if s['json_mode_ok'] else '—':<6}"
              f"{'是' if s['empty_when_tight'] else '否':<9}")
    print(f"\n明细已写入 {dest}")


if __name__ == "__main__":
    asyncio.run(main())
