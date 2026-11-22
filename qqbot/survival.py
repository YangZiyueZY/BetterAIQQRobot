"""伪装存活机制 —— 把"装得像不像真人"变成一个可以结算的奖励。

目标
----
这个机器人存在的意义是**不被认出来是机器人**。所以要有一个贯穿始终的
奖励信号，而不是只看"这条回复有没有人接"。

三个组成部分
------------
1. **存活天数**：一天下来没被质疑，就算活过一天，连续天数越多奖励越高。
2. **识破惩罚**：有人质疑"你是机器人吧"，扣分并清零连胜。
3. **防捷径硬下限**（最重要）：`不说话的机器人最不容易被认出来` ——
   如果只按"没被质疑"给奖励，最优策略就是**永远不说话**，
   这会让决策模型一路收敛到沉默。所以：
     · 群里有正常聊天量、而它一天说得太少 → 判定为**失败日**（扣分，且不算存活）
     · 这样"少说话"从"安全策略"变成"明确的亏损"

关于误判
--------
"怀疑"必须**指向自己**才算：@ 了它、回复了它、或者它刚说完话有人提这茬。
不然群里聊到别的 AI 产品就会误伤，把正常的讨论记成"被识破"。
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import EvolutionConfig
from .models import Message
from .store.repo import Repo
from .store.survival import SurvivalRepo

log = logging.getLogger(__name__)

DAY = 86400


@dataclass
class SuspicionHit:
    message_id: int
    user_id: int
    nickname: str
    text: str
    pattern: str
    strength: float


#: 怀疑模式 -> 强度。
#:
#: 强度用来区分"随口一问"和"咬定它是机器人"：
#: 前者扣一点，后者扣得多，而且要清零连胜。
SUSPICION_PATTERNS: list[tuple[str, float]] = [
    # —— 直接下判断，最重的信号 ——
    (r"你是(个)?(机器人|ai|人工智能|程序|脚本|假人)", 1.0),
    (r"你是不是(个)?(机器人|ai|人工智能|真人)", 1.0),
    (r"(机器人|ai)吧[?？]?$", 0.9),
    (r"就是个(机器人|ai)", 0.9),
    (r"不是真人", 0.9),
    (r"(一看|一眼)就是(机器人|ai)", 1.0),
    (r"(你|这)是(不是)?(自动回复|机器人回复)", 0.9),
    # —— 从行为推断，中等信号 ——
    (r"怎么(可能|会)?这么?快", 0.6),
    (r"秒回", 0.6),
    (r"24\s*小时在线|二十四小时在线", 0.7),
    (r"不用睡觉|不睡觉|不用休息", 0.6),
    (r"谁问都(回|答)", 0.6),
    (r"复读机", 0.5),
    (r"(太|好)机械(了|吧)", 0.7),
    (r"像(个)?机器", 0.7),
    # —— 弱信号：可能是玩笑，扣得少 ——
    (r"(什么|哪个|这)(ai|人工智能)", 0.3),
    (r"这(是)?程序", 0.4),
]

_COMPILED = [(re.compile(p, re.I), s) for p, s in SUSPICION_PATTERNS]

#: 短消息里出现这些词基本是玩笑/口头禅，不算质疑
_NEGATIVE = ("哈哈", "笑死", "别闹", "你猜", "开玩笑")


class SuspicionDetector:
    """从消息里识别"有人怀疑它是机器人"。"""

    @staticmethod
    def scan(msg: Message, *, directed: bool) -> SuspicionHit | None:
        """`directed` = 这条消息是否指向机器人（@ / 回复 / 它刚说完话）。

        **不做指向判断就不检测** —— 否则"原神新出的那个 AI 挺好用"
        会被记成一次质疑。
        """
        if not directed or msg.is_self:
            return None
        text = (msg.text or "").strip()
        if not text or len(text) > 120:
            return None

        for rx, strength in _COMPILED:
            m = rx.search(text)
            if not m:
                continue
            # 玩笑口吻降权：真人互相打趣"你是机器人吧哈哈"不该重罚
            if any(w in text for w in _NEGATIVE):
                strength *= 0.4
            return SuspicionHit(
                message_id=msg.message_id, user_id=msg.user_id,
                nickname=msg.nickname, text=text,
                pattern=m.group(0), strength=round(strength, 2),
            )
        return None


@dataclass
class DayVerdict:
    group_id: int
    day_key: str
    verdict: str
    """survived | suspected | too_quiet | idle"""
    reward: float
    streak: int
    spoke_count: int
    group_msg_count: int
    suspicion_count: int

    @property
    def label(self) -> str:
        return {
            "survived": "装住了",
            "suspected": "被识破",
            "too_quiet": "该说话没说",
            "idle": "群里没人说话",
        }.get(self.verdict, self.verdict)


class SurvivalTracker:
    """按天结算"伪装存活"，产出奖励信号。"""

    def __init__(self, cfg: EvolutionConfig, repo: Repo,
                 survival: SurvivalRepo) -> None:
        self.cfg = cfg
        self.repo = repo          # 用来查当天群消息量
        self.survival = survival

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "survival_enabled", True))

    # ---- 采集 ---------------------------------------------------------

    async def note_message(self, msg: Message, *, directed: bool) -> SuspicionHit | None:
        """收到消息时调用。命中怀疑就记一笔。"""
        if not self.enabled:
            return None
        hit = SuspicionDetector.scan(msg, directed=directed)
        if hit is None:
            return None
        try:
            await self.survival.add_suspicion(
                msg.group_id, hit.message_id, hit.user_id, hit.nickname,
                hit.text, hit.pattern, hit.strength, msg.ts or int(time.time()),
            )
        except Exception:  # noqa: BLE001
            log.debug("记录怀疑事件失败", exc_info=True)
        else:
            log.info("疑似被识破 [%s] %s: %s（强度 %.1f）",
                     msg.group_id, hit.nickname, hit.pattern, hit.strength)
        return hit

    async def note_spoke(self, group_id: int, ts: int | None = None) -> None:
        """成功发言后调用，累计当天发言数。"""
        if not self.enabled:
            return
        day_key = _day_key(ts or int(time.time()))
        try:
            await self.survival.bump_spoke(group_id, day_key)
        except Exception:  # noqa: BLE001
            log.debug("累计发言数失败", exc_info=True)

    # ---- 结算 ---------------------------------------------------------

    async def settle(self, group_id: int, day_key: str,
                     now: int | None = None) -> DayVerdict:
        """结算某一天。

        判定顺序：识破 > 该说话没说 > 群里没人说话 > 装住了。
        先判识破是因为"今天被识破了"这件事比"今天说得少"重要得多。
        """
        now = int(now or time.time())
        start, end = _day_bounds(day_key)

        row = await self.survival.get_day(group_id, day_key)
        spoke = row.spoke_count if row else 0
        susp = await self.survival.suspicion_count_between(group_id, start, end)
        group_msgs = await self._group_msg_count(group_id, start, end)

        min_msgs = max(1, int(getattr(self.cfg, "min_msgs_per_active_day", 3)))
        active_floor = max(1, int(getattr(self.cfg, "min_group_msgs_for_active_day", 20)))
        # 连胜要看前一天的值，不能看日期最大的那天：
        # 补算历史时结算顺序是从新往旧，读最新记录会把已清零的旧连胜读回来。
        streak = await self.survival.streak_before(group_id, day_key)

        if susp > 0:
            verdict = "suspected"
            streak = 0
            reward = -abs(float(getattr(self.cfg, "suspicion_penalty", 3.0))) * susp
        elif group_msgs < active_floor:
            # 群里本来就没人说话，不算它失职
            verdict = "idle"
            reward = 0.0
        elif spoke < min_msgs:
            # ★ 防捷径：群里有正常聊天量，它却几乎不说话 —— 这是失败，不是安全
            verdict = "too_quiet"
            streak = 0
            reward = -abs(float(getattr(self.cfg, "silent_penalty", 1.5)))
        else:
            verdict = "survived"
            streak += 1
            # 连胜奖励递增但增长放缓：活 30 天不该比活 3 天多 10 倍
            base = float(getattr(self.cfg, "survival_reward", 1.0))
            cap = max(1.0, float(getattr(self.cfg, "streak_reward_cap", 3.0)))
            reward = base * min(cap, 1.0 + math.log2(max(1, streak)))

        await self.survival.settle_day(
            group_id, day_key, group_msg_count=group_msgs,
            suspicion_count=susp, verdict=verdict,
            reward=round(reward, 3), streak=streak, now=now,
        )
        log.info("存活结算 [%s] %s -> %s（发言 %d / 群消息 %d / 怀疑 %d / 连胜 %d / 奖励 %+.2f）",
                 group_id, day_key, verdict, spoke, group_msgs, susp, streak, reward)
        return DayVerdict(group_id, day_key, verdict, round(reward, 3), streak,
                          spoke, group_msgs, susp)

    async def settle_due(self, now: int | None = None) -> list[DayVerdict]:
        """结算所有"已经过完但还没结算"的日子。

        只结算**完整过去**的自然日（今天不结，它还没过完）。
        补算最近 `lookback_days` 天，避免停机几天后出现空洞。
        """
        if not self.enabled:
            return []
        now = int(now or time.time())
        lookback = max(1, int(getattr(self.cfg, "survival_lookback_days", 7)))
        out: list[DayVerdict] = []

        for group in await self.repo.list_groups():
            group_id = int(group["group_id"])
            done = await self.survival.settled_day_keys(group_id)
            # 从旧到新结算：连胜是递推量，倒着算会算错
            for offset in range(lookback, 0, -1):
                day_key = _day_key(now - offset * DAY)
                if day_key in done:
                    continue
                try:
                    out.append(await self.settle(group_id, day_key, now=now))
                except Exception:  # noqa: BLE001
                    log.debug("结算 %s/%s 失败", group_id, day_key, exc_info=True)
        return out

    async def _group_msg_count(self, group_id: int, start: int, end: int) -> int:
        async with self.repo.conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE group_id = ? AND is_self = 0 AND ts >= ? AND ts < ?",
            (group_id, start, end),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    # ---- 面板 ---------------------------------------------------------

    async def snapshot(self, group_id: int) -> dict:
        totals = await self.survival.totals(group_id)
        streak = await self.survival.latest_streak(group_id)
        days = await self.survival.settled_days(group_id, limit=30)
        recent = await self.survival.recent_suspicions(group_id, limit=10)
        return {
            "enabled": self.enabled,
            "streak": streak,
            "days": totals["days"],
            "reward_total": round(totals["reward"], 2),
            "spoke_total": totals["spoke"],
            "suspicion_total": totals["suspicion"],
            "history": [
                {
                    "day_key": d.day_key,
                    "verdict": d.verdict,
                    "label": DayVerdict(
                        d.group_id, d.day_key, d.verdict, d.reward, d.streak,
                        d.spoke_count, d.group_msg_count, d.suspicion_count,
                    ).label,
                    "reward": round(d.reward, 2),
                    "spoke": d.spoke_count,
                    "group_msgs": d.group_msg_count,
                    "suspicion": d.suspicion_count,
                }
                for d in days
            ],
            "recent_suspicions": [
                {
                    "nickname": r.get("nickname", ""),
                    "text": r.get("text", ""),
                    "pattern": r.get("pattern", ""),
                    "strength": r.get("strength", 0),
                    "ts": r.get("ts", 0),
                }
                for r in recent
            ],
        }

    async def suspicion_snapshot(self, group_id: int) -> dict:
        """给决策模型看的轻量风险信号。"""
        now = int(time.time())
        last_day = await self.survival.suspicion_count_between(
            group_id, now - DAY, now)
        last_week = await self.survival.suspicion_count_between(
            group_id, now - 7 * DAY, now)
        return {
            "today": last_day,
            "week": last_week,
            "streak": await self.survival.latest_streak(group_id),
        }


def _day_key(ts: int) -> str:
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")


def _day_bounds(day_key: str) -> tuple[int, int]:
    dt = datetime.strptime(day_key, "%Y-%m-%d")
    start = dt.timestamp()
    return int(start), int(start + DAY)
