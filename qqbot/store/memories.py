"""带类型的长期记忆。

和 v1.0 的扁平记忆不同，这里的每条记忆都有：
  type        决定半衰期（culture/slang 近乎永久，event 只有几天）
  confidence  别人明确说的=0.9，自己推断的=0.5；召回时降权
  ttl_days    过期时间
  invalid_at  被新记忆推翻的时间戳（不删除，保留演化历史）
  supersedes 指向被它推翻的那条

"自己判断是否有用"用三维打分实现，见 memory.py 的 USEFULNESS_PROMPT。
"""

from __future__ import annotations

import json
import logging
import time

from .db import Database

log = logging.getLogger(__name__)

#: 各类型记忆的默认半衰期（天）。真人不会用同一个速度忘记所有事。
TTL_BY_TYPE: dict[str, int] = {
    "self": 3650,
    "culture": 3650,
    "slang": 3650,
    "relation": 365,
    "fact": 180,
    "preference": 90,
    "topic": 30,
    "emotion": 14,
    "event": 7,
}

VALID_TYPES = tuple(TTL_BY_TYPE)


def _row(row) -> dict:
    out = dict(row)
    for key in ("source_msg_ids",):
        try:
            out[key] = json.loads(out.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            out[key] = []
    return out


class MemoryRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    # ---- 写入 ---------------------------------------------------------

    async def add(
        self,
        scope: str,
        scope_id: int,
        content: str,
        *,
        type: str = "fact",  # noqa: A002 - 与 DB 列名一致
        confidence: float = 1.0,
        source: str = "observed",
        ttl_days: int | None = None,
        group_id: int | None = None,
        source_msg_ids: list[int] | None = None,
        supersedes_id: int | None = None,
        now: int | None = None,
    ) -> int:
        now = int(now or time.time())
        if type not in TTL_BY_TYPE:
            type = "fact"
        ttl = ttl_days if ttl_days is not None else TTL_BY_TYPE[type]  # noqa: A001

        cur = await self.conn.execute(
            """
            INSERT INTO memories
                (scope, scope_id, content, created_ts, type, ttl_days,
                 confidence, source, group_id, source_msg_ids, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scope, scope_id, content, now, type, ttl,
                confidence, source, group_id,
                json.dumps(source_msg_ids or []),
                supersedes_id,
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid or 0)

    async def invalidate(self, memory_id: int, now: int | None = None) -> None:
        """被新记忆推翻。不删除 —— 演化历史本身有价值。"""
        await self.conn.execute(
            "UPDATE memories SET invalid_at = ? WHERE id = ?",
            (int(now or time.time()), memory_id),
        )
        await self.conn.commit()

    async def touch(self, memory_id: int, now: int | None = None) -> None:
        now = int(now or time.time())
        await self.conn.execute(
            "UPDATE memories SET last_hit_ts = ?, strength = MIN(2.0, strength + 0.05) "
            "WHERE id = ?",
            (now, memory_id),
        )
        await self.conn.commit()

    async def delete(self, memory_id: int) -> bool:
        cur = await self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    # ---- 查询 ---------------------------------------------------------

    async def active(self, scope: str, scope_id: int, limit: int = 300) -> list[dict]:
        """取未失效且未过期的记忆。"""
        now = int(time.time())
        async with self.conn.execute(
            """
            SELECT * FROM memories
            WHERE scope = ? AND scope_id = ?
              AND invalid_at IS NULL
              AND (created_ts + ttl_days * 86400) > ?
            ORDER BY created_ts DESC LIMIT ?
            """,
            (scope, scope_id, now, limit),
        ) as cur:
            return [_row(r) for r in await cur.fetchall()]

    async def all_for(self, scope: str, scope_id: int, limit: int = 300) -> list[dict]:
        """含已失效的，供面板回看演化历史。"""
        async with self.conn.execute(
            "SELECT * FROM memories WHERE scope = ? AND scope_id = ? "
            "ORDER BY created_ts DESC LIMIT ?",
            (scope, scope_id, limit),
        ) as cur:
            return [_row(r) for r in await cur.fetchall()]

    async def list_all(self, scope: str | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM memories"
        args: list = []
        if scope:
            sql += " WHERE scope = ?"
            args.append(scope)
        sql += " ORDER BY created_ts DESC LIMIT ?"
        args.append(limit)
        async with self.conn.execute(sql, tuple(args)) as cur:
            return [_row(r) for r in await cur.fetchall()]

    async def count(self, scope: str | None = None) -> int:
        if scope:
            sql, args = "SELECT COUNT(*) AS n FROM memories WHERE scope = ?", (scope,)
        else:
            sql, args = "SELECT COUNT(*) AS n FROM memories", ()
        async with self.conn.execute(sql, args) as cur:
            return int((await cur.fetchone())["n"] or 0)

    # ---- 维护 ---------------------------------------------------------

    async def evict_over_capacity(self, scope: str, scope_id: int, capacity: int) -> int:
        """超容量时淘汰分数最低的。

        分数 = strength × 半衰期衰减 × (1 + 命中次数)
        """
        rows = await self.all_for(scope, scope_id, limit=capacity + 200)
        if len(rows) <= capacity:
            return 0

        now = time.time()

        def score(m: dict) -> float:
            ttl = max(1, int(m.get("ttl_days") or 180))
            age_days = max(0.0, (now - float(m.get("created_ts") or now)) / 86400.0)
            decay = 0.5 ** (age_days / ttl)
            hits = 1.0 if m.get("last_hit_ts") else 0.0
            return float(m.get("strength") or 1.0) * decay * (1.0 + hits)

        rows.sort(key=score)
        victims = rows[: len(rows) - capacity]
        for m in victims:
            await self.delete(int(m["id"]))
        if victims:
            log.info("记忆淘汰: %s/%s 移除 %d 条", scope, scope_id, len(victims))
        return len(victims)

    async def counts_by_scope(self, scope: str) -> list[dict]:
        """每个 scope_id 有多少条记忆 —— 面板的群列表要用。"""
        async with self.conn.execute(
            """
            SELECT scope_id,
                   COUNT(*) AS total,
                   SUM(CASE WHEN invalid_at IS NULL THEN 1 ELSE 0 END) AS active,
                   SUM(CASE WHEN invalid_at IS NOT NULL THEN 1 ELSE 0 END) AS invalidated,
                   MAX(created_ts) AS last_ts
            FROM memories WHERE scope = ?
            GROUP BY scope_id ORDER BY total DESC
            """,
            (scope,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def type_counts(self, scope: str | None = None,
                          scope_id: int | None = None) -> dict[str, int]:
        sql = "SELECT type, COUNT(*) AS n FROM memories WHERE invalid_at IS NULL"
        args: list = []
        if scope:
            sql += " AND scope = ?"
            args.append(scope)
        if scope_id is not None:
            sql += " AND scope_id = ?"
            args.append(scope_id)
        sql += " GROUP BY type ORDER BY n DESC"
        async with self.conn.execute(sql, tuple(args)) as cur:
            return {r["type"]: int(r["n"]) for r in await cur.fetchall()}

    async def stats(self) -> dict:
        async with self.conn.execute(
            "SELECT type, COUNT(*) AS n FROM memories "
            "WHERE invalid_at IS NULL GROUP BY type ORDER BY n DESC"
        ) as cur:
            by_type = {r["type"]: r["n"] for r in await cur.fetchall()}
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE invalid_at IS NOT NULL"
        ) as cur:
            invalidated = int((await cur.fetchone())["n"] or 0)
        return {
            "memories": sum(by_type.values()),
            "memories_by_type": by_type,
            "memories_invalidated": invalidated,
        }
