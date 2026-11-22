"""自进化引擎：收获反馈 -> 更新策略。

自进化的关键不是"让它自己改 prompt"（那很危险），而是找到**可测量的奖励
信号**并据此微调数值参数。QQ 恰好给了我们一个金矿：
`get_emoji_likes(msg_id)` —— 有人给它贴表情，就是明确的"说得好"。

信号优先级（取最强的一个）：
    被制止 > 被贴表情 > 被回复 / 被@ > 话题延续 > 被无视 > 冷场

安全阀（都在这里）：
    · 冷启动前 N 条发言不收反馈（样本太少必然误判）
    · 标量参数走 EMA，一次只动一点点
    · 所有参数夹在 BOUNDS 内，不能无限漂
    · at_all 永远默认关
"""

from __future__ import annotations

import json
import logging
import time

from .config import EvolutionConfig
from .memory import tokenize
from .store.evolution import (
    GroupPolicy,
    FeedbackRepo,
    PolicyRepo,
    SIGNAL_WEIGHTS,
)
from .store.repo import Repo

log = logging.getLogger(__name__)

#: 判定"被制止"的词。命中就是最强负反馈，直接收敛。
STOP_WORDS = ("别刷", "别发了", "闭嘴", "烦不烦", "机器人滚", "少说两句", "安静点")

#: 判定"话题延续"的重合度阈值
CONTINUE_THRESHOLD = 0.28


class EvolutionEngine:
    def __init__(
        self,
        cfg: EvolutionConfig,
        feedback: FeedbackRepo,
        policies: PolicyRepo,
        repo: Repo,
        api,
        mood_store,
    ) -> None:
        self.cfg = cfg
        self.feedback = feedback
        self.policies = policies
        self.repo = repo
        self.api = api
        self.mood = mood_store
        self.harvested = 0
        self.by_kind: dict[str, int] = {}

    # ---- 登记 ---------------------------------------------------------

    async def open_tracking(
        self, group_id: int, msg_ids: list[int], attributes: dict,
    ) -> int | None:
        if not self.cfg.enabled or not self.cfg.feedback_enabled or not msg_ids:
            return None
        return await self.feedback.open_event(group_id, msg_ids, attributes=attributes)

    # ---- 收获 ---------------------------------------------------------

    async def harvest(self) -> dict:
        """给到期的发言事件判定奖励。由定时任务调用。"""
        if not self.cfg.enabled or not self.cfg.feedback_enabled:
            return {}

        pending = await self.feedback.pending(older_than=self.cfg.harvest_delay)
        stale = await self.feedback.expired_pending(older_than=3600.0)

        done = 0
        for event in pending:
            kind, detail = await self._evaluate(event)
            await self.feedback.record(int(event["id"]), kind, detail)
            await self._apply(event, kind, detail)
            done += 1

        for event in stale:
            # 一小时都没动静，按"被无视"结案，别再挂着
            await self.feedback.mark_harvested(int(event["id"]), "ignored")
            await self._apply(event, "ignored", "超过一小时无任何回应")
            done += 1

        if done:
            log.info("反馈收获 %d 条: %s", done, dict(self.by_kind))
        return {"harvested": done, "by_kind": dict(self.by_kind)}

    async def _evaluate(self, event: dict) -> tuple[str, str]:
        group_id = int(event["group_id"])
        sent_ts = int(event["sent_ts"])
        try:
            msg_ids = {int(x) for x in json.loads(event.get("msg_ids") or "[]")}
        except (json.JSONDecodeError, TypeError, ValueError):
            msg_ids = set()

        # 最强正反馈：有人给它贴表情
        for message_id in msg_ids:
            likes = await self.api.get_emoji_likes(message_id)
            if likes:
                names = "、".join(str(x.get("nick_name") or "?") for x in likes[:3])
                return "emoji_like", f"{names} 贴了表情"

        after = await self.repo.messages_since(group_id, sent_ts + 1, limit=30)
        others = [m for m in after if not m.is_self]
        if not others:
            return "cold", "说完之后没人接话"

        for msg in others:
            if any(word in msg.text for word in STOP_WORDS):
                return "stopped", f"被制止: {msg.text[:40]}"
        for msg in others:
            if msg.reply_to in msg_ids:
                return "reply", f"{msg.nickname} 回复了它"
        for msg in others:
            if msg.mentions_self:
                return "mention", f"{msg.nickname} 接着 @ 它"

        # 话题是否延续：拿它说的话和后继消息比字符重合度
        said = " ".join(
            (await self.repo.message_by_id(group_id, mid) or _EMPTY).text  # type: ignore[union-attr]
            for mid in list(msg_ids)[:2]
        )
        said_tokens = tokenize(said)
        if said_tokens:
            best = 0.0
            for msg in others[:6]:
                tokens = tokenize(msg.text)
                if not tokens:
                    continue
                score = len(said_tokens & tokens) / max(1, len(said_tokens | tokens))
                best = max(best, score)
            if best >= CONTINUE_THRESHOLD:
                return "topic_continue", f"话题还在继续(相似度{best:.2f})"

        return "ignored", f"下一条无关: {others[0].text[:30]}"

    # ---- 更新策略 -----------------------------------------------------

    async def _apply(self, event: dict, kind: str, detail: str) -> None:
        weight = SIGNAL_WEIGHTS.get(kind, 0.0)
        if weight == 0:
            return

        group_id = int(event["group_id"])
        try:
            attributes = json.loads(event.get("attributes") or "{}")
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        policy = await self.policies.get(group_id)
        positive = weight > 0

        # 标量参数：EMA + 边界
        policy.reward_scalar("threshold_delta", weight)
        policy.reward_scalar("initiative", weight)
        if attributes.get("used_sticker"):
            policy.reward_scalar("sticker_rate", weight)
        if attributes.get("used_face"):
            policy.reward_scalar("emoji_rate", weight)

        # Beta 后验：只奖励当时真的用过的形式，避免归因错位
        if attributes.get("used_quote"):
            policy.reward_quote(positive)
        if attributes.get("used_at"):
            policy.reward_at(positive)

        await self.policies.save(policy)

        # 内在状态也跟着变
        if kind == "stopped":
            await self.mood.update(on_negative=1.5)
        elif positive:
            await self.mood.update(on_positive=min(1.5, weight))
        else:
            await self.mood.update(on_negative=abs(weight))

        self.harvested += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        log.debug("反馈 [%s] %s (%.1f) %s", group_id, kind, weight, detail)

    # ---- 对外 ---------------------------------------------------------

    async def policy_for(self, group_id: int) -> GroupPolicy:
        return await self.policies.get(group_id)

    async def note_spoke(self, group_id: int) -> GroupPolicy:
        """每次发言后累加样本数，用于判断是否过了冷启动。"""
        await self.policies.bump_samples(group_id)
        return await self.policies.get(group_id)

    def stats(self) -> dict:
        return {
            "feedback_harvested": self.harvested,
            "feedback_by_kind": dict(self.by_kind),
        }


class _Empty:
    text = ""
    nickname = ""


_EMPTY = _Empty()
