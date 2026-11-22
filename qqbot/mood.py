"""内在状态：让行为从"阈值机器"变成"有脾气的人"。

v1.0 的决定层是一串硬门（睡觉→拒、冷却→拒、分数<0.55→拒），结果行为可预测 ——
而**可预测本身就是最大的破绽**。

这里维护一个随时间漂移、被互动影响的隐状态，所有模块都从它读：

    energy    精力   高→话多句长回得快；低→惜字如金
    mood      心情   -1~+1，影响语气和表情使用率
    battery   社交电量 说话消耗、独处恢复；见底就"不想说话"
    luck      当天手气 整体松紧，保证再好的话题也有"今天就是不想说"的可能
    excite    分话题兴奋度 聊到喜欢的就回血、且更愿意开口

状态持久化到 `agent_state` 表，重启后性格连贯，面板上还能回看
"她那天为什么那么冷淡"。
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)

KEY = "mood"
STATE_ID = 0

#: 各项的自然回归目标与速度（每秒）
BASELINE = {"energy": 0.70, "mood": 0.05, "battery": 0.65, "luck": 0.50}

#: 回归半衰期（秒）——离基线越远回得越快
HALF_LIFE = {"energy": 3 * 3600.0, "mood": 6 * 3600.0,
             "battery": 4 * 3600.0, "luck": 8 * 3600.0}

#: 单次发言的社交电量消耗
SPEAK_COST = 0.035


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _decay_toward(current: float, target: float, half_life: float, dt: float) -> float:
    if dt <= 0 or half_life <= 0:
        return current
    factor = math.pow(0.5, dt / half_life)
    return target + (current - target) * factor


@dataclass
class MoodState:
    energy: float = BASELINE["energy"]
    mood: float = BASELINE["mood"]
    battery: float = BASELINE["battery"]
    luck: float = BASELINE["luck"]
    excite: dict[str, float] = field(default_factory=dict)
    """话题 -> 兴奋度 0~1。聊到就上升，久不提就回落。"""
    updated_at: int = 0

    # ---- 时间演化 -----------------------------------------------------

    def tick(self, now: int | None = None) -> None:
        """随时间回归基线。跳跃式漂移会让性格不连贯，所以走半衰期。"""
        now = int(now or time.time())
        if not self.updated_at:
            self.updated_at = now
            return
        dt = max(0, now - self.updated_at)
        if dt == 0:
            return

        for key, target in BASELINE.items():
            setattr(self, key, _decay_toward(getattr(self, key), target, HALF_LIFE[key], dt))

        # 兴奋度衰减快得多（几小时就淡）
        for topic in list(self.excite):
            self.excite[topic] = _decay_toward(self.excite[topic], 0.0, 4 * 3600.0, dt)
            if self.excite[topic] < 0.02:
                del self.excite[topic]

        # 电量只靠时间缓慢回血（独处恢复）
        self.battery = _clamp(self.battery + dt / (6 * 3600.0) * 0.25)

        # 手气每天换一次
        if now // 86400 != self.updated_at // 86400:
            self.luck = random.betavariate(2.2, 2.2)
            log.debug("新的一天，手气刷新为 %.2f", self.luck)

        self.updated_at = now

    # ---- 互动影响 -----------------------------------------------------

    def on_spoke(self) -> None:
        """说一次话消耗电量，轻微提升心情（社交正反馈）。"""
        self.battery = _clamp(self.battery - SPEAK_COST)
        self.mood = _clamp(self.mood + 0.01, -1.0, 1.0)

    def on_positive(self, weight: float = 1.0) -> None:
        """被贴表情 / 被回复 / 被 @ —— 明确的社交正反馈。"""
        self.mood = _clamp(self.mood + 0.08 * weight, -1.0, 1.0)
        self.energy = _clamp(self.energy + 0.05 * weight)
        self.battery = _clamp(self.battery + 0.04 * weight)

    def on_negative(self, weight: float = 1.0) -> None:
        """被无视 / 冷场 / 被制止。"""
        self.mood = _clamp(self.mood - 0.10 * weight, -1.0, 1.0)
        self.battery = _clamp(self.battery - 0.08 * weight)

    def on_topic(self, topics: list[str], boost: float = 0.25) -> None:
        """聊到（或看到）感兴趣的话题 → 兴奋度上升。"""
        for topic in topics:
            current = self.excite.get(topic, 0.0)
            self.excite[topic] = _clamp(current + boost)
        if topics:
            self.energy = _clamp(self.energy + 0.04 * len(topics))

    # ---- 派生量 -------------------------------------------------------

    @property
    def excitement(self) -> float:
        """当前最高的那个兴奋度 —— 用于"遇到喜欢的话题就跟着聊"。"""
        return max(self.excite.values()) if self.excite else 0.0

    @property
    def is_excited(self) -> bool:
        return self.excitement >= 0.60

    def effective_energy(self) -> float:
        """精力还要看电量：电量见底时精力再好也懒得说。"""
        return _clamp(self.energy * (0.45 + 0.55 * self.battery))

    def describe(self) -> str:
        """给 prompt 用的一句话状态描述。"""
        e = self.effective_energy()
        if self.battery < 0.20:
            return "有点累，不想多说话"
        if self.is_excited:
            return "聊到感兴趣的话题，挺来劲"
        if self.mood > 0.35:
            return "心情不错"
        if self.mood < -0.30:
            return "心情一般，不太想搭话"
        if e > 0.70:
            return "精神不错"
        if e < 0.35:
            return "有点没劲"
        return "普通状态"

    def snapshot(self) -> dict:
        return {
            "energy": round(self.energy, 3),
            "effective_energy": round(self.effective_energy(), 3),
            "mood": round(self.mood, 3),
            "battery": round(self.battery, 3),
            "luck": round(self.luck, 3),
            "excite": {k: round(v, 3) for k, v in
                       sorted(self.excite.items(), key=lambda kv: -kv[1])[:8]},
            "excitement": round(self.excitement, 3),
            "describe": self.describe(),
        }

    # ---- 序列化 -------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "MoodState":
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class MoodStore:
    """状态的长久化 + 进程内缓存。"""

    def __init__(self, repo) -> None:
        self.repo = repo
        self.state = MoodState()
        self._loaded = False

    async def load(self) -> MoodState:
        raw = await self.repo.get_meta(KEY)
        if raw:
            self.state = MoodState.from_json(raw)
        self.state.tick()
        self._loaded = True
        return self.state

    async def save(self) -> None:
        await self.repo.set_meta(KEY, self.state.to_json())

    async def current(self) -> MoodState:
        if not self._loaded:
            await self.load()
        self.state.tick()
        return self.state

    async def update(self, **effects) -> MoodState:
        """effects: on_spoke=True / on_positive=1.0 / on_negative=1.0 / on_topic=[...]"""
        state = await self.current()
        if effects.get("on_spoke"):
            state.on_spoke()
        if effects.get("on_positive"):
            state.on_positive(float(effects["on_positive"]))
        if effects.get("on_negative"):
            state.on_negative(float(effects["on_negative"]))
        if effects.get("on_topic"):
            state.on_topic(list(effects["on_topic"]))
        state.updated_at = int(time.time())
        await self.save()
        return state
