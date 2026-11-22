"""伪装存活与怀疑事件的存储。

`survival_days` 是**按天结算**的账本，`suspicion_events` 是明细。
分开存是因为：结算要按天聚合（快），而明细要能翻出来看"到底哪句话露馅了"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .db import Database

log = logging.getLogger(__name__)


@dataclass
class SurvivalRow:
    group_id: int
    day_key: str
    spoke_count: int
    group_msg_count: int
    suspicion_count: int
    verdict: str
    reward: float
    streak: int
    settled_ts: int


class SurvivalRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    # ---- 明细 ---------------------------------------------------------

    async def add_suspicion(self, group_id: int, message_id: int, user_id: int,
                            nickname: str, text: str, pattern: str,
                            strength: float, ts: int) -> None:
        await self.conn.execute(
            "INSERT INTO suspicion_events "
            "(group_id, message_id, user_id, nickname, text, pattern, strength, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (group_id, message_id, user_id, nickname, text[:200], pattern,
             float(strength), ts),
        )
        await self.conn.commit()

    async def recent_suspicions(self, group_id: int, limit: int = 20) -> list[dict]:
        async with self.conn.execute(
            "SELECT * FROM suspicion_events WHERE group_id = ? "
            "ORDER BY ts DESC LIMIT ?", (group_id, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def suspicion_count_between(self, group_id: int, start: int, end: int) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM suspicion_events "
            "WHERE group_id = ? AND ts >= ? AND ts < ?", (group_id, start, end),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    # ---- 账本 ---------------------------------------------------------

    async def bump_spoke(self, group_id: int, day_key: str) -> None:
        """当天发言 +1。用 upsert 而不是先读后写，避免并发丢计数。"""
        await self.conn.execute(
            "INSERT INTO survival_days (group_id, day_key, spoke_count) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(group_id, day_key) DO UPDATE SET "
            "spoke_count = spoke_count + 1",
            (group_id, day_key),
        )
        await self.conn.commit()

    async def get_day(self, group_id: int, day_key: str) -> SurvivalRow | None:
        async with self.conn.execute(
            "SELECT * FROM survival_days WHERE group_id = ? AND day_key = ?",
            (group_id, day_key),
        ) as cur:
            row = await cur.fetchone()
        return SurvivalRow(**dict(row)) if row else None

    async def settle_day(self, group_id: int, day_key: str, *,
                         group_msg_count: int, suspicion_count: int,
                         verdict: str, reward: float, streak: int,
                         now: int) -> SurvivalRow:
        await self.conn.execute(
            "INSERT INTO survival_days "
            "(group_id, day_key, spoke_count, group_msg_count, suspicion_count, "
            " verdict, reward, streak, settled_ts) "
            "VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(group_id, day_key) DO UPDATE SET "
            "group_msg_count = excluded.group_msg_count, "
            "suspicion_count = excluded.suspicion_count, "
            "verdict = excluded.verdict, reward = excluded.reward, "
            "streak = excluded.streak, settled_ts = excluded.settled_ts",
            (group_id, day_key, int(group_msg_count), int(suspicion_count),
             verdict, float(reward), int(streak), int(now)),
        )
        await self.conn.commit()
        row = await self.get_day(group_id, day_key)
        assert row is not None
        return row

    async def settled_days(self, group_id: int, limit: int = 30) -> list[SurvivalRow]:
        async with self.conn.execute(
            "SELECT * FROM survival_days WHERE group_id = ? "
            "AND settled_ts > 0 ORDER BY day_key DESC LIMIT ?", (group_id, limit),
        ) as cur:
            return [SurvivalRow(**dict(r)) for r in await cur.fetchall()]

    async def settled_day_keys(self, group_id: int) -> set[str]:
        async with self.conn.execute(
            "SELECT day_key FROM survival_days WHERE group_id = ? AND settled_ts > 0",
            (group_id,),
        ) as cur:
            return {str(r["day_key"]) for r in await cur.fetchall()}

    async def streak_before(self, group_id: int, day_key: str) -> int:
        """取这一天之前最近一个已结算日的连胜值。

        不能直接用 latest_streak（日期最大的那天）：补算历史时结算顺序是从新往旧，
        先写进去的可能是较晚那天 —— 用日期最大的记录，会把已经被清零的旧连胜读回来。
        """
        async with self.conn.execute(
            "SELECT streak FROM survival_days WHERE group_id = ? "
            "AND settled_ts > 0 AND day_key < ? ORDER BY day_key DESC LIMIT 1",
            (group_id, day_key),
        ) as cur:
            row = await cur.fetchone()
        return int(row["streak"]) if row else 0

    async def latest_streak(self, group_id: int) -> int:
        async with self.conn.execute(
            "SELECT streak FROM survival_days WHERE group_id = ? "
            "AND settled_ts > 0 ORDER BY day_key DESC LIMIT 1", (group_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["streak"]) if row else 0

    async def totals(self, group_id: int) -> dict:
        async with self.conn.execute(
            "SELECT COUNT(*) AS days, COALESCE(SUM(reward), 0) AS reward, "
            "COALESCE(SUM(spoke_count), 0) AS spoke, "
            "COALESCE(SUM(suspicion_count), 0) AS susp "
            "FROM survival_days WHERE group_id = ? AND settled_ts > 0", (group_id,),
        ) as cur:
            row = await cur.fetchone()
        return {
            "days": int(row["days"] or 0),
            "reward": float(row["reward"] or 0),
            "spoke": int(row["spoke"] or 0),
            "suspicion": int(row["susp"] or 0),
        }

    async def stats(self) -> dict:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM suspicion_events") as cur:
            susp = int((await cur.fetchone())["n"])
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM survival_days WHERE settled_ts > 0") as cur:
            days = int((await cur.fetchone())["n"])
        return {"suspicion_events": susp, "survival_days": days}
