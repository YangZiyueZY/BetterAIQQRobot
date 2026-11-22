"""新群冷启动：先读一遍群历史，再决定怎么融进去。

为什么需要它
------------
一个真人进新群，不会上来就瞎接话 —— 他会先潜水看几天，搞清楚这是个什么群、
大家在聊什么、有什么梗，然后才偶尔插一句。机器人也应该这样。

所以打开发言权限时先做三件事：

  1. **拉全量历史**：把能拿到的群记录尽量往前拉，形成样本
  2. **分析群偏好**：话题分布、黑话、氛围、常出现的人、活跃时段
  3. **生成"看不懂就问"的问题**：这是最关键的一步 —— 与其硬接话，
     不如像新人一样真诚地问一句。真人就是这么融入的。

问题不会立刻抛出去，而是存起来，等它在线且群里有动静时慢慢问掉
（见 `Pipeline._maybe_ask_question`）。一次只问一个，问完记下来不再问。

关于历史能拉多少
----------------
NapCat 的 message_id 是**内存里的短 id**（LRU 5000 条、无 TTL、重启即清空），
所以往前翻有硬上限。翻到底时它**不返回空数组，而是 retcode 1200**，
这一点在 `api.get_group_msg_history` 里已经处理成返回 None。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .config import Config
from .gateway.api import OneBotAPI
from .gateway.events import parse_group_message
from .models import Message
from .store.onboard import (STATUS_ANALYZING, STATUS_FAILED, STATUS_PULLING,
                            STATUS_READY, OnboardRepo, OnboardRow)

log = logging.getLogger(__name__)

#: 一次拉多少条（NapCat 侧默认 20，给大一点减少往返）
_PAGE = 200

#: 送进 LLM 的转录上限（字符）。太长既贵又超出上下文。
_TRANSCRIPT_CAP = 12000


@dataclass
class PullResult:
    messages: list[Message] = field(default_factory=list)
    pages: int = 0
    exhausted: bool = False

    @property
    def count(self) -> int:
        return len(self.messages)


class Onboarder:
    def __init__(self, cfg: Config, api: OneBotAPI, repo, onboard: OnboardRepo,
                 llm, profile_builder=None, persona_for=None) -> None:
        self.cfg = cfg
        self.api = api
        self.repo = repo
        self.onboard = onboard
        self.llm = llm
        self.profile_builder = profile_builder
        """ProfileBuilder，分析完顺手把统计画像和 LLM 主题都写回去。"""
        self.persona_for = persona_for
        """(group_id) -> Persona，用来让问题带上角色口吻。"""
        self._tasks: dict[int, asyncio.Task] = {}

    # ---- 启动 ---------------------------------------------------------

    async def start(self, group_id: int, *, force: bool = False) -> dict:
        """打开发言权限时调用。幂等：已经分析过的不重跑（除非 force）。"""
        if not getattr(self.cfg.behavior, "onboard_enabled", True):
            return {"started": False, "reason": "冷启动分析已关闭"}
        row = await self.onboard.get(group_id)
        if not force:
            if row.status in (STATUS_PULLING, STATUS_ANALYZING):
                return {"started": False, "reason": "已经在分析了"}
            if row.status == STATUS_READY:
                return {"started": False, "reason": "已经分析过了",
                        "questions": len(row.questions)}

        existing = self._tasks.get(group_id)
        if existing and not existing.done():
            return {"started": False, "reason": "任务正在跑"}

        await self.onboard.set_status(group_id, STATUS_PULLING)
        self._tasks[group_id] = asyncio.ensure_future(self._run(group_id))
        log.info("群 %s 开始冷启动分析", group_id)
        return {"started": True, "reason": "已开始拉取群历史"}

    async def _run(self, group_id: int) -> None:
        try:
            pulled = await self.pull_history(group_id)
            # 把拉到的历史落库：画像统计、记忆抽取、时间线页都靠 messages 表。
            # save_message 自带去重，重复的会返回 False，所以可以无脑写。
            stored = await self._store(group_id, pulled.messages)
            await self.onboard.set_status(
                group_id, STATUS_ANALYZING, pulled=pulled.count,
                note=f"拉了 {pulled.count} 条（新入库 {stored}）")
            await self.analyze(group_id, pulled.messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("群 %s 冷启动分析失败: %s", group_id, exc, exc_info=True)
            await self.onboard.set_status(group_id, STATUS_FAILED, note=str(exc)[:200])
        finally:
            self._tasks.pop(group_id, None)

    async def close(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()

    # ---- 拉历史 -------------------------------------------------------

    async def pull_history(self, group_id: int, limit: int | None = None) -> PullResult:
        """尽可能往前拉群历史。

        分页要点（实测 NapCat 4.18.28）：
          · 往前翻是 `message_seq=<当前最旧的 seq> + reverse_order=True`
          · 返回结果**以该 seq 结尾**（含它），所以要跳过重复的那条，否则死循环
          · **翻到底返回 retcode 1200 而不是空数组** —— 由网关层翻成 None
        """
        limit = int(limit or getattr(self.cfg.behavior, "onboard_history", 1000))
        result = PullResult()
        cursor: str | None = None
        seen: set[int] = set()

        while result.count < limit:
            want = min(_PAGE, limit - result.count)
            batch = await self.api.get_group_msg_history(
                group_id, count=want, message_seq=cursor, reverse_order=True)
            result.pages += 1
            if batch is None:
                result.exhausted = True
                break
            if not batch:
                break

            fresh = 0
            oldest: int | None = None
            for raw in batch:
                msg = parse_group_message(raw, 0)
                if msg is None:
                    continue
                if msg.message_id in seen:
                    continue
                seen.add(msg.message_id)
                result.messages.append(msg)
                fresh += 1
                if oldest is None or msg.message_id < oldest:
                    oldest = msg.message_id
            if fresh == 0 or oldest is None:
                break            # 没有新消息了，别再来一轮

            # 用最旧的那条做下一页的锚点；它本身会在下一页重复出现，靠 seen 去重
            cursor = str(oldest)
            if len(batch) < want:
                # 这一页没拿满，说明到边界了
                result.exhausted = True
                break

        result.messages.sort(key=lambda m: (m.ts, m.message_id))
        log.info("群 %s 拉到 %d 条历史（%d 页，到底=%s）",
                 group_id, result.count, result.pages, result.exhausted)
        return result

    async def _store(self, group_id: int, messages: list[Message]) -> int:
        """把历史消息写进本地库，已存在的跳过。返回新入库条数。"""
        fresh = 0
        for msg in messages:
            try:
                if await self.repo.save_message(msg):
                    fresh += 1
            except Exception:  # noqa: BLE001
                continue
        return fresh

    # ---- 分析 ---------------------------------------------------------

    async def analyze(self, group_id: int, messages: list[Message]) -> dict:
        """把历史交给 LLM 分析，产出画像与"看不懂就问"的问题。"""
        from . import prompts as prompt_registry
        from .generate.llm import LLMError, extract_json

        if not messages:
            await self.onboard.save_analysis(group_id, [], note="没有历史可分析")
            return {"topics": {}, "questions": []}

        persona = self.persona_for(group_id) if self.persona_for else None
        name = persona.name if persona else "机器人"
        transcript = _render_transcript(messages)[-_TRANSCRIPT_CAP:]

        body = prompt_registry.render_part(
            "task.onboard", name=name, group=str(group_id),
            count=len(messages), transcript=transcript)

        try:
            text = await self.llm.chat(
                [{"role": "user", "content": body}],
                temperature=0.4, max_tokens=1800, json_mode=True,
                profile="analysis",
            )
            payload = extract_json(text)
        except LLMError as exc:
            log.warning("群 %s 冷启动分析调用失败: %s", group_id, exc)
            await self.onboard.set_status(group_id, STATUS_FAILED, note=str(exc)[:200])
            return {"topics": {}, "questions": []}

        questions = _clean_questions(payload.get("questions"), limit=6)
        note = str(payload.get("note") or "")[:200]
        await self.onboard.save_analysis(group_id, questions, note=note)

        # 顺手把画像更新掉 —— 冷启动正是样本最全的时候。
        # 先 rebuild 出统计量（话题分布/表情率/活跃时段），
        # 再用 apply_topics 把 LLM 看懂的主题/黑话/氛围合进去。
        if self.profile_builder is not None:
            try:
                await self.profile_builder.rebuild(group_id)
            except Exception:  # noqa: BLE001
                log.debug("冷启动后重算统计画像失败", exc_info=True)
            try:
                await self.profile_builder.apply_topics(group_id, payload)
            except Exception:  # noqa: BLE001
                log.debug("写入 LLM 画像失败", exc_info=True)

        log.info("群 %s 冷启动完成：话题 %d 个，问题 %d 条",
                 group_id, len(payload.get("topics") or {}), len(questions))
        return payload

    # ---- 提问 ---------------------------------------------------------

    async def pending_question(self, group_id: int) -> dict | None:
        row = await self.onboard.get(group_id)
        if row.status != STATUS_READY:
            return None
        pending = row.pending
        return pending[0] if pending else None

    async def mark_asked(self, group_id: int, text: str) -> None:
        await self.onboard.mark_asked(group_id, text)

    async def snapshot(self, group_id: int) -> dict:
        row = await self.onboard.get(group_id)
        return {
            "status": row.status or "idle",
            "pulled": row.pulled,
            "note": row.note,
            "analyzed_ts": row.analyzed_ts,
            "questions": row.questions,
            "asked": row.asked,
            "pending": len(row.pending),
        }


def _render_transcript(messages: list[Message]) -> str:
    """把历史渲染成转录文本。

    带上时间（按天分段）—— **活跃时段**是分析的一个重要维度，
    没有时间信息就看不出来"这个群晚上才热闹"。
    """
    lines: list[str] = []
    last_day = ""
    for msg in messages:
        day = time.strftime("%m-%d", time.localtime(msg.ts or 0))
        if day != last_day:
            lines.append(f"\n=== {day} ===")
            last_day = day
        clock = time.strftime("%H:%M", time.localtime(msg.ts or 0))
        who = "我" if msg.is_self else (msg.nickname or "群友")
        text = (msg.text or "").replace("\n", " ")[:120]
        lines.append(f"[{clock}] {who}: {text}")
    return "\n".join(lines)


def _clean_questions(raw, limit: int = 6) -> list[dict]:
    """清洗模型给的问题清单。

    模型有时会把整段话塞进 text，或者混进"问题：""问："这种前缀 ——
    发到群里之前必须收拾干净，否则一眼就是机器生成的。
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            text, why = item, ""
        elif isinstance(item, dict):
            text = str(item.get("text") or "")
            why = str(item.get("why") or "")
        else:
            continue
        text = text.strip().strip('"“”「」')
        for prefix in ("问题：", "问题:", "问：", "问:", "Q:", "Q："):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        if not text or len(text) > 40:
            continue
        out.append({"text": text, "why": why[:80]})
        if len(out) >= limit:
            break
    return out
