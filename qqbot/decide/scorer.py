"""决定层：判断"要不要开口"。

这是整套系统里最影响拟真度的模块，占 80% 的成败。

刻意不用大模型 —— 规则打分便宜、可控、可解释，而且每次判断都能落库，
事后可以拿 decisions 表复盘"为什么它当时没说话"。

评分（可累加，最终 clamp 到 0~1）：
    被 @ / 被回复            1.00      被点名了，基本要回
    出现名字或外号            0.60
    命中精通领域              0.50
    是个问句                  0.15      有人提问，插话合理
    群冷场                    0.20

一票否决（在分数之前）：
    正在睡觉 / 外出           直接否决
    mention_only 且没被 @     直接否决
    本群冷却期内              直接否决
    超出时/日额度             直接否决

过了阈值之后还有一次**随机放弃** —— 真人不是有求必应，这一步不能省。
"""

from __future__ import annotations

import logging
import random
import re
import time

from ..config import BehaviorConfig
from ..models import Conversation, Decision
from ..persona import Persona

log = logging.getLogger(__name__)

QUESTION_MARK = re.compile(r"[?？]$|吗|呢|怎么|为什么|如何|是不是|有没有|能不能")

#: 被点名时的固定高分，不做成可配置项 —— 这类机会不该被调低。
MENTION_SCORE = 1.0
REPLY_TO_SELF_SCORE = 0.85


class Decider:
    def __init__(self, cfg: BehaviorConfig, debug=None) -> None:
        self.cfg = cfg
        self.debug = debug
        self._last_spoke: dict[int, float] = {}
        """group_id -> monotonic 时间戳，用于群冷却。"""
        self.give_ups = 0
        self.rejected: dict[str, int] = {}

    # ---- 记录 ---------------------------------------------------------

    def note_spoke(self, group_id: int, now: float | None = None) -> None:
        self._last_spoke[group_id] = now if now is not None else time.monotonic()

    @property
    def forced(self) -> bool:
        return bool(self.debug is not None and self.debug.enabled)

    def cooldown_remaining(self, group_id: int, now: float | None = None) -> float:
        if self.forced:
            return 0.0
        now = now if now is not None else time.monotonic()
        last = self._last_spoke.get(group_id)
        if last is None:
            return 0.0
        return max(0.0, self.cfg.group_cooldown - (now - last))

    # ---- 特征 ---------------------------------------------------------

    def features(self, conv: Conversation, persona: Persona,
                 extra_topics: list[str] | None = None) -> dict:
        # 文本类特征看完整上下文：跨批次的"回复我""提到名字"才算得出来。
        # 但 batch_size 仍然只看本批 —— 那是"群里是不是在刷屏"的语义。
        all_msgs = conv.all_messages
        recent = "\n".join(m.text for m in all_msgs[-8:])
        trigger = conv.trigger

        self_message_ids = {m.message_id for m in all_msgs if m.is_self}
        reply_to_self = bool(trigger.reply_to and trigger.reply_to in self_message_ids)

        mentioned = any(m.mentions_self for m in all_msgs[-5:]) or trigger.mentions_self

        name_hit = any(word in recent for word in persona.trigger_words()) if persona.trigger_words() else False

        # 命中领域：人设自带的 + 群画像学到的主题
        keywords = list(persona.topic_keywords())
        for topic in (extra_topics or []):
            if topic and topic not in keywords:
                keywords.append(topic)
        topic_hits = [k for k in keywords if k and k in recent]

        question = bool(QUESTION_MARK.search(trigger.text))

        batch_size = len(conv.messages)
        someone_else_spoke = any(not m.is_self for m in all_msgs)

        return {
            "mentioned": mentioned,
            "reply_to_self": reply_to_self,
            "name_hit": name_hit,
            "topic_hits": topic_hits[:5],
            "question": question,
            "batch_size": batch_size,
            "someone_else_spoke": someone_else_spoke,
            "trigger_user": trigger.user_id,
        }

    # ---- 打分 ---------------------------------------------------------

    def score(self, features: dict, *, excitement: float = 0.0) -> tuple[float, list[str]]:
        cfg = self.cfg
        total = 0.0
        reasons: list[str] = []

        if features["mentioned"]:
            total += MENTION_SCORE
            reasons.append("被@")
        elif features["reply_to_self"]:
            total += REPLY_TO_SELF_SCORE
            reasons.append("回复了我")

        if features["name_hit"]:
            total += cfg.name_priority
            reasons.append("提到名字")

        if features["topic_hits"]:
            total += cfg.topic_priority
            reasons.append(f"话题命中({','.join(features['topic_hits'][:2])})")

        if features["question"] and not features["mentioned"]:
            total += 0.15
            reasons.append("有人在提问")

        if features["batch_size"] >= 4 and not features["mentioned"]:
            total += cfg.idle_priority
            reasons.append("群在刷屏")

        # 兴奋度直接加分 —— 这就是"遇到喜欢的话题就跟着聊"
        if excitement > 0:
            total += min(0.45, excitement * 0.5)
            reasons.append(f"兴致高({excitement:.2f})")

        return min(1.0, total), reasons

    # ---- 决策 ---------------------------------------------------------

    def decide(
        self,
        conv: Conversation,
        *,
        persona: Persona,
        presence: str,
        budget_ok: bool = True,
        budget_reason: str = "",
        now: float | None = None,
        threshold_delta: float = 0.0,
        excitement: float = 0.0,
        mood: float = 0.0,
        extra_topics: list[str] | None = None,
    ) -> Decision:
        """threshold_delta 来自群策略（自进化），excitement 来自内在状态。

        **聊到喜欢的话题时阈值会被显著拉低**，这就是"遇到喜欢的话题就
        直接跟着聊"的实现 —— 加上随机手气，保证不是每次都一定接。
        """
        now = now if now is not None else time.monotonic()
        features = self.features(conv, persona, extra_topics=extra_topics)
        score, reasons = self.score(features, excitement=excitement)

        decision = Decision(
            should_reply=False,
            priority=score,
            reason="",
            features=features,
            ts=int(time.time()),
            group_id=conv.group_id,
            text=conv.trigger.text,
        )

        def reject(reason: str) -> Decision:
            decision.reason = reason
            self.rejected[reason] = self.rejected.get(reason, 0) + 1
            return decision

        if score <= 0:
            if self.forced and self.debug.respond_to_all:
                # 压力测试：既不相关也没人叫它，也照样接
                score = 0.2
                reasons.append("测试模式(响应所有)")
            else:
                return reject("不值得插话")

        # --- 一票否决 ---
        if presence in ("sleeping", "away"):
            return reject(f"当前状态 {presence}")

        if self.cfg.mention_only and not (features["mentioned"] or features["reply_to_self"]):
            return reject("仅@模式")

        # 兴奋时连冷却都能豁免一部分 —— 真人聊到兴头上不会看表
        remaining = self.cooldown_remaining(conv.group_id, now)
        if remaining > 0 and excitement < 0.7:
            return reject(f"群冷却中({remaining:.0f}s)")

        if not budget_ok:
            return reject(budget_reason or "超出额度")

        # 阈值 = 基础 + 群策略修正 - 兴奋加成 - 心情加成
        threshold = (
            self.cfg.threshold
            + threshold_delta
            - excitement * 0.30
            - max(0.0, mood) * 0.06
        )
        threshold = max(0.15, min(0.95, threshold))
        if self.forced:
            # 测试模式：门槛压到配置的下限，让它更容易开口
            threshold = min(threshold, max(0.0, self.debug.relax_threshold))

        if score < threshold:
            return reject(f"分数不足({score:.2f}<{threshold:.2f})")

        # --- 随机放弃 ---
        give_up = self.cfg.give_up_prob
        if features["mentioned"] or features["reply_to_self"]:
            give_up *= 0.25
        elif excitement > 0.7:
            give_up *= 0.4      # 兴头上更愿意说话
        if self.forced:
            give_up = 0.0       # 测试模式不随机放弃，否则复现问题很烦
        if random.random() < give_up:
            self.give_ups += 1
            return reject("随机放弃")

        decision.should_reply = True
        decision.reason = " + ".join(reasons) if reasons else "主动接话"

        if features["mentioned"] or features["reply_to_self"]:
            decision.delay = random.uniform(0.8, 3.0)
        elif excitement > 0.7:
            decision.delay = random.uniform(1.0, 4.0)
        else:
            decision.delay = random.uniform(2.0, 8.0)

        return decision
