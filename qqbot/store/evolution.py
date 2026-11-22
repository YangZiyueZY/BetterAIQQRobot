"""群档案与自进化的持久化。

三张表各司其职：
  group_profiles   客观统计画像（地基，不用 LLM，零成本）
  group_policies   策略参数（自进化产物，带边界约束）
  feedback_events  每次发言的奖励信号（引擎的燃料）

设计上刻意把"画像"和"策略"分开：画像是**事实**（这个群说什么话），
策略是**选择**（我在这个群该怎么说）。前者可以随便更新，后者必须
慢速、有界、可回滚 —— 否则系统会为了短期奖励漂成讨好型话痨。
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field

from .db import Database

log = logging.getLogger(__name__)

#: 策略参数的硬边界。这是安全阀，不能删。
BOUNDS: dict[str, tuple[float, float]] = {
    "threshold_delta": (-0.15, 0.20),
    "initiative": (0.05, 0.60),
    "sticker_rate": (0.03, 0.30),
    "emoji_rate": (0.05, 0.35),
}

#: 各属性的学习率（EMA）。越小越稳，越大越快但越容易漂。
LEARNING_RATE = {
    "threshold_delta": 0.04,
    "initiative": 0.05,
    "sticker_rate": 0.06,
    "emoji_rate": 0.06,
    "quote": 1.0,   # Beta 计数直接累加
    "at": 1.0,
}

#: 冷启动：前这么多次发言不收反馈（样本太少必然误判）
COLD_START_SAMPLES = 100


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


# ==========================================================================
# 群画像
# ==========================================================================


@dataclass
class GroupProfile:
    group_id: int = 0
    topics: dict = field(default_factory=dict)
    """{"A游戏": 0.42, "技术": 0.21}"""
    avg_msg_len: float = 0.0
    """群友平均说多长 —— 最有效的融入手段：他们说 8 字，你就别说 40 字。"""
    emoji_rate: float = 0.0
    sticker_rate: float = 0.0
    question_rate: float = 0.0
    hourly: list = field(default_factory=lambda: [0.0] * 24)
    """24 个浮点，一天中各小时的活跃度。比全局作息精确得多。"""
    slang: dict = field(default_factory=dict)
    """群内黑话 -> 含义。听不懂黑话就永远是外人。"""
    core_members: list = field(default_factory=list)
    """[{"user_id":.., "nickname":.., "count":..}] 说话最多的几个人 = 意见领袖。"""
    atmosphere: float = 0.0
    """-1~+1，负向词比例推出来的和谐度。"""
    sample_size: int = 0
    updated_ts: int = 0

    def peak_hours(self, top: int = 3) -> list[int]:
        pairs = sorted(enumerate(self.hourly), key=lambda kv: -kv[1])[:top]
        return sorted(h for h, v in pairs if v > 0)

    def summary(self) -> dict:
        top_topics = sorted(self.topics.items(), key=lambda kv: -kv[1])[:5]
        return {
            "group_id": self.group_id,
            "top_topics": top_topics,
            "avg_msg_len": round(self.avg_msg_len, 1),
            "emoji_rate": round(self.emoji_rate, 3),
            "sticker_rate": round(self.sticker_rate, 3),
            "peak_hours": self.peak_hours(),
            "core_members": self.core_members[:5],
            "slang": dict(list(self.slang.items())[:20]),
            "atmosphere": round(self.atmosphere, 3),
            "sample_size": self.sample_size,
        }


class ProfileRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def get(self, group_id: int) -> GroupProfile:
        async with self.conn.execute(
            "SELECT * FROM group_profiles WHERE group_id = ?", (group_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return GroupProfile(group_id=group_id)

        def jload(key, default):
            try:
                return json.loads(row[key] or default)
            except (json.JSONDecodeError, TypeError, IndexError):
                return json.loads(default)

        return GroupProfile(
            group_id=row["group_id"],
            topics=jload("topics", "{}"),
            avg_msg_len=row["avg_msg_len"],
            emoji_rate=row["emoji_rate"],
            sticker_rate=row["sticker_rate"],
            question_rate=row["question_rate"],
            hourly=jload("hourly", "[]") or [0.0] * 24,
            slang=jload("slang", "{}"),
            core_members=jload("core_members", "[]"),
            atmosphere=row["atmosphere"],
            sample_size=row["sample_size"],
            updated_ts=row["updated_ts"],
        )

    async def save(self, p: GroupProfile) -> None:
        p.updated_ts = int(time.time())
        await self.conn.execute(
            """
            INSERT INTO group_profiles
                (group_id, topics, avg_msg_len, emoji_rate, sticker_rate,
                 question_rate, hourly, slang, core_members, atmosphere,
                 sample_size, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (group_id) DO UPDATE SET
                topics=excluded.topics, avg_msg_len=excluded.avg_msg_len,
                emoji_rate=excluded.emoji_rate, sticker_rate=excluded.sticker_rate,
                question_rate=excluded.question_rate, hourly=excluded.hourly,
                slang=excluded.slang, core_members=excluded.core_members,
                atmosphere=excluded.atmosphere, sample_size=excluded.sample_size,
                updated_ts=excluded.updated_ts
            """,
            (
                p.group_id, json.dumps(p.topics, ensure_ascii=False),
                p.avg_msg_len, p.emoji_rate, p.sticker_rate, p.question_rate,
                json.dumps(p.hourly), json.dumps(p.slang, ensure_ascii=False),
                json.dumps(p.core_members, ensure_ascii=False),
                p.atmosphere, p.sample_size, p.updated_ts,
            ),
        )
        await self.conn.commit()

    async def list_all(self) -> list[GroupProfile]:
        async with self.conn.execute(
            "SELECT group_id FROM group_profiles ORDER BY updated_ts DESC"
        ) as cur:
            ids = [r["group_id"] for r in await cur.fetchall()]
        return [await self.get(gid) for gid in ids]


# ==========================================================================
# 群策略（自进化产物）
# ==========================================================================


@dataclass
class GroupPolicy:
    group_id: int = 0
    threshold_delta: float = 0.0
    initiative: float = 0.30
    sticker_rate: float = 0.12
    emoji_rate: float = 0.15
    quote_alpha: float = 2.0
    quote_beta: float = 2.0
    at_alpha: float = 1.0
    at_beta: float = 3.0
    at_all_enabled: bool = False
    tone: dict = field(default_factory=dict)
    samples: int = 0
    updated_ts: int = 0

    # ---- Thompson 采样 -------------------------------------------------

    def sample_quote(self) -> bool:
        """这个群吃不吃"引用"？Beta 后验采样，天然带随机性。"""
        return random.betavariate(
            max(0.2, self.quote_alpha), max(0.2, self.quote_beta)
        ) > 0.5

    def sample_at(self) -> bool:
        return random.betavariate(
            max(0.2, self.at_alpha), max(0.2, self.at_beta)
        ) > 0.5

    # ---- 更新 ---------------------------------------------------------

    def reward_quote(self, positive: bool) -> None:
        if self.samples < COLD_START_SAMPLES:
            return
        if positive:
            self.quote_alpha += LEARNING_RATE["quote"]
        else:
            self.quote_beta += LEARNING_RATE["quote"]

    def reward_at(self, positive: bool) -> None:
        if self.samples < COLD_START_SAMPLES:
            return
        if positive:
            self.at_alpha += LEARNING_RATE["at"]
        else:
            self.at_beta += LEARNING_RATE["at"]

    def reward_scalar(self, attribute: str, reward: float) -> None:
        """标量参数走 EMA，一次反馈只动一点点，且始终夹在边界内。"""
        if self.samples < COLD_START_SAMPLES:
            return
        if attribute not in BOUNDS:
            return
        rate = LEARNING_RATE.get(attribute, 0.05)

        if attribute == "threshold_delta":
            # 正奖励 → 门槛下调（更敢说）；负奖励 → 上调（收敛）
            target = getattr(self, attribute) - reward * 0.05
        elif attribute == "initiative":
            target = getattr(self, attribute) + reward * 0.05
        else:
            target = getattr(self, attribute) + reward * 0.03

        current = getattr(self, attribute)
        updated = current + (target - current) * rate
        setattr(self, attribute, _clamp(updated, BOUNDS[attribute]))

    def snapshot(self) -> dict:
        return {
            "group_id": self.group_id,
            "threshold_delta": round(self.threshold_delta, 4),
            "initiative": round(self.initiative, 4),
            "sticker_rate": round(self.sticker_rate, 4),
            "emoji_rate": round(self.emoji_rate, 4),
            "quote_p": round(self.quote_alpha / (self.quote_alpha + self.quote_beta), 3),
            "at_p": round(self.at_alpha / (self.at_alpha + self.at_beta), 3),
            "samples": self.samples,
            "cold_start": self.samples < COLD_START_SAMPLES,
        }

    def reset_learned(self) -> None:
        """回滚到出厂设置。面板上的"重置策略"按钮。"""
        self.threshold_delta = 0.0
        self.initiative = 0.30
        self.sticker_rate = 0.12
        self.emoji_rate = 0.15
        self.quote_alpha, self.quote_beta = 2.0, 2.0
        self.at_alpha, self.at_beta = 1.0, 3.0
        self.samples = 0


class PolicyRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def get(self, group_id: int) -> GroupPolicy:
        async with self.conn.execute(
            "SELECT * FROM group_policies WHERE group_id = ?", (group_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return GroupPolicy(group_id=group_id)
        try:
            tone = json.loads(row["tone"] or "{}")
        except (json.JSONDecodeError, TypeError):
            tone = {}
        return GroupPolicy(
            group_id=row["group_id"],
            threshold_delta=row["threshold_delta"],
            initiative=row["initiative"],
            sticker_rate=row["sticker_rate"],
            emoji_rate=row["emoji_rate"],
            quote_alpha=row["quote_alpha"],
            quote_beta=row["quote_beta"],
            at_alpha=row["at_alpha"],
            at_beta=row["at_beta"],
            at_all_enabled=bool(row["at_all_enabled"]),
            tone=tone,
            samples=row["samples"],
            updated_ts=row["updated_ts"],
        )

    async def save(self, p: GroupPolicy) -> None:
        p.updated_ts = int(time.time())
        await self.conn.execute(
            """
            INSERT INTO group_policies
                (group_id, threshold_delta, initiative, sticker_rate, emoji_rate,
                 quote_alpha, quote_beta, at_alpha, at_beta, at_all_enabled,
                 tone, samples, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (group_id) DO UPDATE SET
                threshold_delta=excluded.threshold_delta,
                initiative=excluded.initiative, sticker_rate=excluded.sticker_rate,
                emoji_rate=excluded.emoji_rate, quote_alpha=excluded.quote_alpha,
                quote_beta=excluded.quote_beta, at_alpha=excluded.at_alpha,
                at_beta=excluded.at_beta, at_all_enabled=excluded.at_all_enabled,
                tone=excluded.tone, samples=excluded.samples,
                updated_ts=excluded.updated_ts
            """,
            (
                p.group_id, p.threshold_delta, p.initiative, p.sticker_rate,
                p.emoji_rate, p.quote_alpha, p.quote_beta, p.at_alpha, p.at_beta,
                int(p.at_all_enabled), json.dumps(p.tone, ensure_ascii=False),
                p.samples, p.updated_ts,
            ),
        )
        await self.conn.commit()

    async def bump_samples(self, group_id: int) -> int:
        await self.conn.execute(
            "INSERT INTO group_policies (group_id, samples) VALUES (?, 1) "
            "ON CONFLICT (group_id) DO UPDATE SET samples = samples + 1",
            (group_id,),
        )
        await self.conn.commit()
        return (await self.get(group_id)).samples


# ==========================================================================
# 奖励信号
# ==========================================================================

#: 各类信号的权重。这是自进化的"价值观" —— 改这里等于改它的追求。
SIGNAL_WEIGHTS: dict[str, float] = {
    "emoji_like": 1.0,      # 有人给它贴表情：最强的正反馈
    "reply": 0.8,           # 有人回复它的消息
    "mention": 0.6,         # 有人紧接着 @ 它继续问
    "topic_continue": 0.4,  # 说完后话题还在延续
    "cold": -0.5,           # 说完后冷场
    "ignored": -0.3,        # 下一条完全无关
    "stopped": -1.0,        # 被明确制止（别刷了/被禁言）
}


class FeedbackRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def open_event(
        self,
        group_id: int,
        msg_ids: list[int],
        *,
        attributes: dict | None = None,
        now: int | None = None,
    ) -> int:
        now = int(now or time.time())
        cur = await self.conn.execute(
            "INSERT INTO feedback_events (group_id, msg_ids, sent_ts, kind, attributes, ts) "
            "VALUES (?, ?, ?, 'pending', ?, ?)",
            (group_id, json.dumps(msg_ids), now,
             json.dumps(attributes or {}, ensure_ascii=False), now),
        )
        await self.conn.commit()
        return int(cur.lastrowid or 0)

    async def record(self, event_id: int, kind: str, detail: str = "",
                     weight: float | None = None) -> None:
        w = SIGNAL_WEIGHTS.get(kind, 0.0) if weight is None else weight
        await self.conn.execute(
            "UPDATE feedback_events SET kind = ?, weight = ?, detail = ?, harvested = 1 "
            "WHERE id = ?",
            (kind, w, detail[:300], event_id),
        )
        await self.conn.commit()

    async def pending(self, older_than: float = 300.0, limit: int = 50) -> list[dict]:
        """发言后已过 `older_than` 秒但还没收获信号的。"""
        cutoff = int(time.time() - older_than)
        async with self.conn.execute(
            "SELECT * FROM feedback_events WHERE kind = 'pending' AND sent_ts <= ? "
            "ORDER BY sent_ts ASC LIMIT ?",
            (cutoff, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def expired_pending(self, older_than: float = 3600.0) -> list[dict]:
        """超过一小时也没信号的，直接按"被无视"结案。"""
        cutoff = int(time.time() - older_than)
        async with self.conn.execute(
            "SELECT * FROM feedback_events WHERE kind = 'pending' AND sent_ts <= ? "
            "ORDER BY sent_ts ASC LIMIT 100",
            (cutoff,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def mark_harvested(self, event_id: int, kind: str = "ignored") -> None:
        await self.conn.execute(
            "UPDATE feedback_events SET harvested = 1, kind = ? WHERE id = ?",
            (kind, event_id),
        )
        await self.conn.commit()

    async def recent(self, group_id: int | None = None, limit: int = 100) -> list[dict]:
        if group_id:
            sql = ("SELECT * FROM feedback_events WHERE group_id = ? "
                   "AND kind != 'pending' ORDER BY sent_ts DESC LIMIT ?")
            args: tuple = (group_id, limit)
        else:
            sql = ("SELECT * FROM feedback_events WHERE kind != 'pending' "
                   "ORDER BY sent_ts DESC LIMIT ?")
            args = (limit,)
        async with self.conn.execute(sql, args) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def summary(self, group_id: int | None = None, days: int = 7) -> dict:
        since = int(time.time() - days * 86400)
        sql = ("SELECT kind, COUNT(*) AS n, AVG(weight) AS w FROM feedback_events "
               "WHERE kind != 'pending' AND sent_ts >= ?")
        args: list = [since]
        if group_id:
            sql += " AND group_id = ?"
            args.append(group_id)
        sql += " GROUP BY kind"
        async with self.conn.execute(sql, tuple(args)) as cur:
            rows = await cur.fetchall()

        by_kind = {r["kind"]: int(r["n"] or 0) for r in rows}
        total = sum(by_kind.values())
        positive = sum(by_kind.get(k, 0) for k in ("emoji_like", "reply", "mention", "topic_continue"))
        return {
            "signals": by_kind,
            "signals_total": total,
            "positive_rate": (positive / total) if total else 0.0,
            "window_days": days,
        }

    async def timeseries(self, group_id: int | None = None, days: int = 14) -> list[dict]:
        """按天汇总奖励均值 —— 面板上画"进化曲线"用。"""
        since = int(time.time() - days * 86400)
        sql = ("SELECT (sent_ts / 86400) * 86400 AS day, "
               "AVG(weight) AS w, COUNT(*) AS n FROM feedback_events "
               "WHERE kind != 'pending' AND sent_ts >= ?")
        args: list = [since]
        if group_id:
            sql += " AND group_id = ?"
            args.append(group_id)
        sql += " GROUP BY day ORDER BY day"
        async with self.conn.execute(sql, tuple(args)) as cur:
            rows = await cur.fetchall()
        return [
            {"ts": int(r["day"]), "reward": round(float(r["w"] or 0), 3),
             "count": int(r["n"] or 0)}
            for r in rows
        ]
