"""决策模型：把"要不要接这句话"从固定规则升级成 LLM 判定。

为什么需要它
------------
原来的决定层是手调线性打分：关键词命中 + 提及 + 相似度，再过阈值。
它的问题不是不准，而是**看不懂语义** —— 分不清"在说给别人听"和
"在跟我说"，也看不出"群里已经有人接过这话了"。这些恰恰是真人判断的
依据，也是"像不像人"的关键。

两阶段设计
----------
    ① 廉价门控（scorer）—— 免费、瞬时
       一票否决：睡着/外出、额度用完、冷却中、仅@模式
       给出先验分数
    ② LLM 判定 —— 只对**灰区分数**的候选调用
       问它"这句该不该接"，返回 speak / reason / tone / confidence

为什么不全交给 LLM：每条群消息都调一次又贵又慢（群里一分钟几十条），
而灰区之外打分器已经很有把握。LLM 的价值在灰区。

三条硬性要求（否则决策模型会收敛到"永远不说话"）
------------------------------------------------
1. **被点名必须理**：@ 它 / 回复它时强制发言，不允许 LLM 说不
2. **冷场要主动**：在群里沉默超过 `silence_nudge_minutes` 就压低门槛
3. **保留探索**：LLM 说不、但打分器分数不错时，仍有 `explore_prob` 概率说

没有这三条，系统的局部最优就是闭嘴 —— 不说话当然不会被认出来。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field

from .. import prompts
from ..config import DeciderConfig
from ..models import Conversation, Decision
from ..persona import Persona
from .scorer import Decider

log = logging.getLogger(__name__)

#: 这些拒绝理由属于"一票否决"，跟分数无关 —— 不进入 LLM 判定
HARD_VETO = ("当前状态", "仅@模式", "额度", "冷却中")


@dataclass
class LLMVerdict:
    speak: bool
    reason: str = ""
    confidence: float = 0.5
    tone: str = "正常"
    raw: str = ""
    error: str = ""


def _persona_brief(persona: Persona) -> str:
    """给决策模型的角色摘要。**故意很短** —— 它只需要知道"这是谁、
    说话什么调性"，不需要完整人设（那是生成阶段的事，这里省 token）。"""
    bits = [f"名字「{persona.name}」"]
    ident = persona.identity
    for key, fmt in (("age", "{}岁"), ("gender", "{}"), ("occupation", "{}")):
        if ident.get(key):
            bits.append(fmt.format(ident[key]))
    traits = [str(t.get("name", "")) for t in (persona.data.get("traits") or [])
              if isinstance(t, dict)]
    if traits:
        bits.append("性格：" + "、".join(traits[:4]))
    style = persona.data.get("style") or {}
    if style.get("sentence_length"):
        bits.append(f"说话：{style['sentence_length']}")
    return "；".join(b for b in bits if b)


class LLMDecider:
    """问 LLM"这句该不该接"。"""

    def __init__(self, cfg: DeciderConfig, llm) -> None:
        self.cfg = cfg
        self.llm = llm
        self.calls = 0
        self.errors = 0
        self.said_no = 0

    async def consult(
        self,
        conv: Conversation,
        persona: Persona,
        *,
        presence: str,
        score: float,
        reasons: list[str],
        silence_minutes: float = 0.0,
        profile_note: str = "",
    ) -> LLMVerdict | None:
        """返回 None 表示"放弃了，用打分器的结论"。"""
        if self.llm is None or not getattr(self.llm, "configured", True):
            return None

        trigger = conv.trigger
        body = prompts.render_part(
            "task.decide",
            persona=_persona_brief(persona),
            group=profile_note or f"QQ 群 {conv.group_id}",
            recent=_render_recent(conv),
            trigger=f"{trigger.nickname}：{trigger.text}",
            state=_state_text(presence, silence_minutes),
            score=(f"{score:.2f}"
                   + (f"（{'；'.join(reasons[:3])}）" if reasons else "")),
        )
        try:
            text = await asyncio.wait_for(
                self.llm.chat(
                    [{"role": "user", "content": body}],
                    temperature=0.3, max_tokens=200, json_mode=True,
                    profile="reply",
                ),
                timeout=max(1.0, float(self.cfg.timeout)),
            )
        except asyncio.TimeoutError:
            self.errors += 1
            log.info("决策判定超时 %.0fs，回退打分器", self.cfg.timeout)
            return None
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            log.info("决策判定失败，回退打分器: %s", exc)
            return None

        self.calls += 1
        verdict = _parse_verdict(text)
        if verdict is None:
            self.errors += 1
            return None
        if not verdict.speak:
            self.said_no += 1
        return verdict

    def stats(self) -> dict:
        return {"decide_calls": self.calls, "decide_errors": self.errors,
                "decide_said_no": self.said_no}


class HybridDecider:
    """打分器 + LLM 判定的组合。管线只跟这个打交道。"""

    def __init__(self, cfg: DeciderConfig, base: Decider, llm_decider: LLMDecider | None,
                 repo=None, debug=None) -> None:
        self.cfg = cfg
        self.base = base
        self.llm = llm_decider
        self.repo = repo
        self.debug = debug
        self.consulted = 0
        self.overridden = 0
        self.llm_vetoed = 0
        self.explored = 0

    @property
    def effective_mode(self) -> str:
        """测试模式下强制退回打分器 —— 调试时别让 LLM 判定添乱。"""
        if self.debug is not None and self.debug.enabled:
            return "score"
        return (self.cfg.mode or "hybrid").lower()

    def note_spoke(self, group_id: int, now: float | None = None) -> None:
        self.base.note_spoke(group_id, now)

    def cooldown_remaining(self, group_id: int, now: float | None = None) -> float:
        return self.base.cooldown_remaining(group_id, now)

    @property
    def rejected(self) -> dict:
        return self.base.rejected

    @property
    def give_ups(self) -> int:
        return getattr(self.base, "give_ups", 0)

    def __getattr__(self, name: str):
        """没显式转发的属性一律透传给打分器。

        为什么需要：HybridDecider 是 scorer.Decider 的包装，
        外层代码仍在读打分器的属性（曾经漏了 `give_ups`，导致
        `/api/status` 抛 AttributeError -> 面板 500，看起来像"连不上"）。
        显式转发容易漏，交给 __getattr__ 兜底 —— 包装类的语义就是
        "除了我改写的，其余都跟被包的一样"。
        """
        if name.startswith("_"):
            raise AttributeError(name)      # 别干扰 pickle / 复制等内部协议
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    async def decide(self, conv: Conversation, *, persona: Persona,
                     presence: str, **kwargs) -> Decision:
        """先跑打分器，再按情况叠加 LLM 判定与硬性要求。"""
        decision = self.base.decide(conv, persona=persona, presence=presence, **kwargs)
        score = float(decision.priority or 0.0)
        mode = self.effective_mode

        # 打分器给的理由，LLM 判定时作为参考
        reasons = [decision.reason] if decision.reason else []
        features = decision.features or {}
        mentioned = bool(features.get("mentioned") or features.get("reply_to_self"))

        # 一票否决不进入 LLM —— 睡着/没额度/冷却中时说与不说都没意义
        if not decision.should_reply and any(
                decision.reason.startswith(k) for k in HARD_VETO):
            return decision

        # --- 硬性要求 ①：被点名必须理 ---
        if mentioned and self.cfg.force_on_mention and not decision.should_reply:
            if not any(decision.reason.startswith(k) for k in HARD_VETO):
                decision.should_reply = True
                decision.reason = "被点名，必须理（硬性要求）"
                decision.features = {**features, "decided_by": "force_mention"}
                self.overridden += 1
                log.info("决策覆盖：被点名必须理 [%s]", conv.group_id)
                return decision

        silence_min = await self._silence_minutes(conv.group_id)

        # --- 硬性要求 ②：冷场太久就主动开口 ---
        # 沉默成本随时间上升，压掉一部分阈值。这是防"越不说越安全"的关键。
        nudged = False
        if (not decision.should_reply and self.cfg.silence_nudge_minutes > 0
                and silence_min >= self.cfg.silence_nudge_minutes
                and not any(decision.reason.startswith(k) for k in HARD_VETO)):
            decision.should_reply = True
            decision.reason = (f"在群里沉默 {_human_minutes(silence_min)}，"
                               f"该主动说一句了（硬性要求）")
            decision.features = {**features, "decided_by": "silence_nudge"}
            self.overridden += 1
            nudged = True
            log.info("决策覆盖：冷场 %d 分钟，催它开口 [%s]",
                     int(silence_min), conv.group_id)

        # --- 灰区才问 LLM ---
        gray = self.cfg.band_low <= score <= self.cfg.band_high
        want_llm = mode == "llm" or (mode == "hybrid" and gray)
        if nudged or not want_llm or self.llm is None:
            return decision

        self.consulted += 1
        verdict = await self.llm.consult(
            conv, persona, presence=presence, score=score, reasons=reasons,
            silence_minutes=silence_min,
        )
        if verdict is None:
            return decision        # 失败就沿用打分器的结论

        if verdict.speak:
            if not decision.should_reply:
                decision.should_reply = True
                decision.reason = f"LLM 判定该接：{verdict.reason}"
            else:
                decision.reason = f"{decision.reason}；LLM 同意：{verdict.reason}"
            decision.features = {**features, "decided_by": "llm",
                                 "llm_tone": verdict.tone,
                                 "llm_conf": verdict.confidence}
            return decision

        # LLM 说不说 —— 但保留探索，别让它一路沉默下去
        if self.cfg.explore_prob > 0 and random.random() < self.cfg.explore_prob:
            self.explored += 1
            decision.should_reply = True
            decision.reason = (f"LLM 觉得不必接（{verdict.reason}），"
                               f"但按探索概率还是接了")
            decision.features = {**features, "decided_by": "explore",
                                 "llm_tone": verdict.tone}
            return decision

        self.llm_vetoed += 1
        decision.should_reply = False
        decision.reason = f"LLM 判定不必接：{verdict.reason}"
        decision.features = {**features, "decided_by": "llm",
                             "llm_conf": verdict.confidence}
        return decision

    async def _silence_minutes(self, group_id: int) -> float:
        """它在群里沉默多久了。查不到就当刚说过（不触发冷场覆盖）。"""
        if self.repo is None:
            return 0.0
        try:
            last = await self.repo.last_self_message_ts(group_id)
        except Exception:  # noqa: BLE001
            return 0.0
        if not last:
            return 0.0
        return max(0.0, (time.time() - float(last)) / 60.0)

    def stats(self) -> dict:
        out = {
            "decide_mode": self.effective_mode,
            "decide_consulted": self.consulted,
            "decide_overridden": self.overridden,
            "decide_llm_vetoed": self.llm_vetoed,
            "decide_explored": self.explored,
        }
        if self.llm is not None:
            out.update(self.llm.stats())
        return out


def _human_minutes(minutes: float) -> str:
    """沉默时长的中文写法。别把 240 分钟写成"4 分钟"。"""
    if minutes < 60:
        return f"{minutes:.0f} 分钟"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f} 小时"
    return f"{minutes / 1440:.1f} 天"


def _state_text(presence: str, silence_minutes: float) -> str:
    label = {"online": "在线", "busy": "忙碌", "away": "外出",
             "sleeping": "睡觉"}.get(presence, presence)
    if silence_minutes <= 0:
        return f"{label}；刚在群里说过话"
    return f"{label}；已经 {_human_minutes(silence_minutes)}没在群里说话了"


def _render_recent(conv: Conversation, limit: int = 14) -> str:
    """渲染最近对话。它自己说的话标成「我」，方便 LLM 判断"接没接过"。"""
    lines: list[str] = []
    for msg in (conv.messages or [])[-limit:]:
        who = "我" if msg.is_self else msg.nickname
        text = (msg.text or "").replace("\n", " ")[:80]
        lines.append(f"{who}: {text}")
    return "\n".join(lines) or "（没有上下文）"


def _parse_verdict(text: str) -> LLMVerdict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or "speak" not in payload:
        return None
    try:
        conf = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return LLMVerdict(
        speak=bool(payload.get("speak")),
        reason=str(payload.get("reason") or "")[:120],
        confidence=max(0.0, min(1.0, conf)),
        tone=str(payload.get("tone") or "正常")[:8],
        raw=raw[:300],
    )
