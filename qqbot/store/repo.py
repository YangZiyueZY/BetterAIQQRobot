"""数据读写封装。上层只碰这里，不写裸 SQL。"""

from __future__ import annotations

import json
import logging
import time

import aiosqlite

from ..models import Decision, Message
from .db import Database

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400
GLOBAL_SCOPE_ID = 0
"""budget 表里代表全局额度的伪 group_id。"""


def _row_to_message(row: aiosqlite.Row) -> Message:
    try:
        segments = json.loads(row["segments"] or "[]")
    except json.JSONDecodeError:
        segments = []
    return Message(
        group_id=row["group_id"],
        message_id=row["message_id"],
        user_id=row["user_id"],
        nickname=row["nickname"],
        text=row["text"],
        segments=segments,
        ts=row["ts"],
        is_self=bool(row["is_self"]),
        reply_to=row["reply_to"],
        mentions_self=bool(row["mentions_self"]),
        has_image=bool(row["has_image"]),
        media_hash=row["media_hash"] if "media_hash" in row.keys() else None,
        reply_to_self=bool(row["reply_to_self"]) if "reply_to_self" in row.keys() else False,
    )


def _row_to_decision(row: aiosqlite.Row) -> Decision:
    try:
        features = json.loads(row["features"] or "{}")
    except json.JSONDecodeError:
        features = {}
    return Decision(
        id=row["id"],
        should_reply=bool(row["should_reply"]),
        priority=row["priority"],
        reason=row["reason"],
        features=features,
        ts=row["ts"],
        group_id=row["group_id"],
        text=row["text"],
        draft=row["draft"],
    )


class Repo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self) -> aiosqlite.Connection:
        return self.db.conn

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------

    async def save_message(self, m: Message) -> bool:
        """写入一条消息。返回 False 表示重复（group_id + message_id 已存在）。"""
        cur = await self.conn.execute(
            """
            INSERT OR IGNORE INTO messages
                (group_id, message_id, user_id, nickname, text, segments,
                 ts, is_self, reply_to, mentions_self, has_image,
                 media_hash, reply_to_self)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.group_id, m.message_id, m.user_id, m.nickname, m.text,
                json.dumps(m.segments, ensure_ascii=False),
                m.ts or int(time.time()),
                int(m.is_self), m.reply_to, int(m.mentions_self), int(m.has_image),
                m.media_hash, int(m.reply_to_self),
            ),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    # ---- meta（键值对，存内在状态等） ---------------------------------

    async def get_meta(self, key: str) -> str | None:
        async with self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        return row["value"] if row else None

    async def set_meta(self, key: str, value: str) -> None:
        await self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self.conn.commit()

    # ---- 反馈归因所需的查询 -------------------------------------------

    async def message_by_id(self, group_id: int, message_id: int) -> Message | None:
        async with self.conn.execute(
            "SELECT * FROM messages WHERE group_id = ? AND message_id = ?",
            (group_id, message_id),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_message(row) if row else None

    async def set_media_hash(self, group_id: int, message_id: int, media_hash: str) -> None:
        """图片下载后回填哈希，用于去重与多模态缓存命中。"""
        await self.conn.execute(
            "UPDATE messages SET media_hash = ? WHERE group_id = ? AND message_id = ?",
            (media_hash, group_id, message_id),
        )
        await self.conn.commit()

    async def mark_reply_to_self(self, group_id: int, message_id: int) -> None:
        await self.conn.execute(
            "UPDATE messages SET reply_to_self = 1 WHERE group_id = ? AND message_id = ?",
            (group_id, message_id),
        )
        await self.conn.commit()

    async def messages_since(self, group_id: int, since_ts: int, limit: int = 200) -> list[Message]:
        async with self.conn.execute(
            "SELECT * FROM messages WHERE group_id = ? AND ts >= ? "
            "ORDER BY ts ASC, id ASC LIMIT ?",
            (group_id, since_ts, limit),
        ) as cur:
            return [_row_to_message(r) for r in await cur.fetchall()]

    async def recent_messages(self, group_id: int, limit: int = 30) -> list[Message]:
        async with self.conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM messages WHERE group_id = ?
                ORDER BY ts DESC, id DESC LIMIT ?
            ) ORDER BY ts ASC, id ASC
            """,
            (group_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_message(r) for r in rows]

    async def paged_messages(
        self, group_id: int, limit: int = 50, before_id: int | None = None
    ) -> list[Message]:
        """给面板的时间线用：按 id 倒序翻页，返回时再正序。"""
        if before_id:
            sql = ("SELECT * FROM messages WHERE group_id = ? AND id < ? "
                   "ORDER BY id DESC LIMIT ?")
            args = (group_id, before_id, limit)
        else:
            sql = "SELECT * FROM messages WHERE group_id = ? ORDER BY id DESC LIMIT ?"
            args = (group_id, limit)
        async with self.conn.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [_row_to_message(r) for r in reversed(rows)]

    async def last_self_message_ts(self, group_id: int) -> int:
        async with self.conn.execute(
            "SELECT MAX(ts) AS ts FROM messages WHERE group_id = ? AND is_self = 1",
            (group_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["ts"] or 0) if row else 0

    async def recent_self_texts(self, group_id: int, limit: int = 10) -> list[str]:
        """给 safety 层做复读检测。"""
        async with self.conn.execute(
            "SELECT text FROM messages WHERE group_id = ? AND is_self = 1 "
            "ORDER BY id DESC LIMIT ?",
            (group_id, limit),
        ) as cur:
            return [r["text"] for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # 群 / 成员
    # ------------------------------------------------------------------

    async def upsert_group(self, group_id: int, name: str | None = None,
                           *, default_enabled: bool = False) -> None:
        now = int(time.time())
        await self.conn.execute(
            """
            INSERT INTO chat_groups (group_id, name, enabled, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (group_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                name      = COALESCE(excluded.name, chat_groups.name)
            """,
            (group_id, name, int(default_enabled), now, now),
        )
        await self.conn.commit()

    async def set_group_enabled(self, group_id: int, enabled: bool) -> None:
        await self.conn.execute(
            "UPDATE chat_groups SET enabled = ? WHERE group_id = ?",
            (int(enabled), group_id),
        )
        await self.conn.commit()

    async def group_enabled(self, group_id: int) -> bool:
        async with self.conn.execute(
            "SELECT enabled FROM chat_groups WHERE group_id = ?", (group_id,)
        ) as cur:
            row = await cur.fetchone()
        return bool(row["enabled"]) if row else False

    async def list_groups(self) -> list[dict]:
        async with self.conn.execute(
            """
            SELECT g.group_id, g.name, g.enabled, g.last_seen,
                   (SELECT COUNT(*) FROM messages m WHERE m.group_id = g.group_id) AS msg_count,
                   (SELECT COUNT(*) FROM group_members gm WHERE gm.group_id = g.group_id) AS member_count
            FROM chat_groups g
            ORDER BY g.enabled DESC, msg_count DESC
            """
        ) as cur:
            rows = await cur.fetchall()

        out = []
        for row in rows:
            budget = await self.budget_state(row["group_id"])
            out.append({
                "group_id": row["group_id"],
                "name": row["name"] or f"群 {row['group_id']}",
                "enabled": bool(row["enabled"]),
                "msg_count": row["msg_count"],
                "member_count": row["member_count"],
                "last_seen": row["last_seen"] or 0,
                "hour_used": budget["hour_used"],
                "day_used": budget["day_used"],
            })
        return out

    async def get_group(self, group_id: int) -> dict | None:
        for item in await self.list_groups():
            if item["group_id"] == group_id:
                return item
        return None

    async def upsert_member(self, group_id: int, user_id: int,
                            nickname: str, card: str | None = None) -> None:
        now = int(time.time())
        await self.conn.execute(
            """
            INSERT INTO group_members (group_id, user_id, nickname, card, last_seen, msg_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT (group_id, user_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                msg_count = group_members.msg_count + 1,
                nickname  = excluded.nickname,
                card      = COALESCE(excluded.card, group_members.card)
            """,
            (group_id, user_id, nickname, card, now),
        )
        await self.conn.commit()

    async def member_names(self, group_id: int) -> dict[int, str]:
        async with self.conn.execute(
            "SELECT user_id, nickname, card FROM group_members WHERE group_id = ?",
            (group_id,),
        ) as cur:
            rows = await cur.fetchall()
        return {r["user_id"]: (r["card"] or r["nickname"] or str(r["user_id"])) for r in rows}

    # ------------------------------------------------------------------
    # 决策留痕
    # ------------------------------------------------------------------

    async def save_decision(self, group_id: int, decision: Decision) -> None:
        await self.conn.execute(
            """
            INSERT INTO decisions
                (group_id, ts, should_reply, priority, reason, features, text, draft)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                decision.ts or int(time.time()),
                int(decision.should_reply),
                decision.priority,
                decision.reason,
                json.dumps(decision.features, ensure_ascii=False),
                decision.text,
                decision.draft,
            ),
        )
        await self.conn.commit()

    async def list_decisions(
        self, limit: int = 50, group_id: int | None = None
    ) -> list[Decision]:
        if group_id:
            sql = "SELECT * FROM decisions WHERE group_id = ? ORDER BY id DESC LIMIT ?"
            args: tuple = (group_id, limit)
        else:
            sql = "SELECT * FROM decisions ORDER BY id DESC LIMIT ?"
            args = (limit,)
        async with self.conn.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [_row_to_decision(r) for r in rows]

    async def decision_stats(self) -> dict:
        async with self.conn.execute(
            "SELECT COUNT(*) AS total, SUM(should_reply) AS replied FROM decisions"
        ) as cur:
            row = await cur.fetchone()
        total = int(row["total"] or 0)
        replied = int(row["replied"] or 0)
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE is_self = 1"
        ) as cur:
            sent = int((await cur.fetchone())["n"] or 0)
        return {
            "decisions": total,
            "replies": replied,
            "messages_sent": sent,
            "spoke_rate": (replied / total) if total else 0.0,
        }

    # ------------------------------------------------------------------
    # 额度（budget）
    # ------------------------------------------------------------------

    async def budget_state(self, group_id: int, now: int | None = None) -> dict:
        """读当前额度窗口，惰性重置过期窗口。"""
        now = int(now or time.time())
        async with self.conn.execute(
            "SELECT * FROM budget WHERE group_id = ?", (group_id,)
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return {"hour_used": 0, "day_used": 0, "window_start": now, "day_start": now}

        hour_used = row["hour_used"]
        day_used = row["day_used"]
        window_start = row["window_start"] or now
        day_start = row["day_start"] or now

        if now - window_start >= HOUR:
            hour_used = 0
            window_start = now
        if now - day_start >= DAY:
            day_used = 0
            day_start = now

        return {
            "hour_used": hour_used,
            "day_used": day_used,
            "window_start": window_start,
            "day_start": day_start,
        }

    async def bump_budget(self, group_id: int, now: int | None = None) -> None:
        now = int(now or time.time())
        state = await self.budget_state(group_id, now)
        await self.conn.execute(
            """
            INSERT INTO budget (group_id, window_start, hour_used, day_used, day_start)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (group_id) DO UPDATE SET
                window_start = excluded.window_start,
                hour_used    = excluded.hour_used,
                day_used     = excluded.day_used,
                day_start    = excluded.day_start
            """,
            (
                group_id,
                state["window_start"],
                state["hour_used"] + 1,
                state["day_used"] + 1,
                state["day_start"],
            ),
        )
        await self.conn.commit()

    async def global_day_used(self, now: int | None = None) -> int:
        return (await self.budget_state(GLOBAL_SCOPE_ID, now))["day_used"]

    # ------------------------------------------------------------------
    # 记忆
    # ------------------------------------------------------------------

    async def add_memory(self, scope: str, scope_id: int, content: str,
                         now: int | None = None) -> int:
        now = int(now or time.time())
        cur = await self.conn.execute(
            "INSERT INTO memories (scope, scope_id, content, created_ts) VALUES (?, ?, ?, ?)",
            (scope, scope_id, content, now),
        )
        await self.conn.commit()
        return int(cur.lastrowid or 0)

    async def list_memories(self, scope: str | None = None,
                            scope_id: int | None = None,
                            limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM memories"
        args: list = []
        where: list[str] = []
        if scope:
            where.append("scope = ?")
            args.append(scope)
        if scope_id is not None:
            where.append("scope_id = ?")
            args.append(scope_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_ts DESC LIMIT ?"
        args.append(limit)
        async with self.conn.execute(sql, tuple(args)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def recall_memories(self, scope: str, scope_id: int,
                              limit: int = 200) -> list[dict]:
        async with self.conn.execute(
            "SELECT * FROM memories WHERE scope = ? AND scope_id = ? "
            "ORDER BY created_ts DESC LIMIT ?",
            (scope, scope_id, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def touch_memory(self, memory_id: int, now: int | None = None) -> None:
        now = int(now or time.time())
        await self.conn.execute(
            "UPDATE memories SET last_hit_ts = ?, strength = MIN(2.0, strength + 0.05) "
            "WHERE id = ?",
            (now, memory_id),
        )
        await self.conn.commit()

    async def delete_memory(self, memory_id: int) -> bool:
        cur = await self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def memory_count(self) -> int:
        async with self.conn.execute("SELECT COUNT(*) AS n FROM memories") as cur:
            return int((await cur.fetchone())["n"] or 0)

    # ------------------------------------------------------------------
    # 统计 / 维护
    # ------------------------------------------------------------------

    async def message_stats(self) -> dict:
        async with self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN is_self = 0 THEN 1 ELSE 0 END) AS received, "
            "SUM(CASE WHEN is_self = 1 THEN 1 ELSE 0 END) AS sent FROM messages"
        ) as cur:
            row = await cur.fetchone()
        return {
            "messages_total": int(row["total"] or 0),
            "messages_received": int(row["received"] or 0),
            "messages_sent": int(row["sent"] or 0),
        }

    async def timeseries(self, hours: int = 24, now: int | None = None) -> list[dict]:
        now = int(now or time.time())
        since = now - hours * HOUR
        async with self.conn.execute(
            """
            SELECT (ts / 3600) * 3600 AS bucket,
                   SUM(CASE WHEN is_self = 0 THEN 1 ELSE 0 END) AS in_count,
                   SUM(CASE WHEN is_self = 1 THEN 1 ELSE 0 END) AS out_count
            FROM messages WHERE ts >= ?
            GROUP BY bucket ORDER BY bucket
            """,
            (since,),
        ) as cur:
            rows = await cur.fetchall()

        by_bucket = {int(r["bucket"]): (int(r["in_count"] or 0), int(r["out_count"] or 0))
                     for r in rows}
        current = (now // HOUR) * HOUR
        points = []
        for step in range(hours - 1, -1, -1):
            bucket = current - step * HOUR
            in_count, out_count = by_bucket.get(bucket, (0, 0))
            points.append({"ts": bucket, "in": in_count, "out": out_count})
        return points

    async def stats(self) -> dict:
        async with self.conn.execute("SELECT COUNT(*) AS n FROM messages") as cur:
            total = int((await cur.fetchone())["n"] or 0)
        async with self.conn.execute(
            "SELECT group_id, COUNT(*) AS n FROM messages "
            "GROUP BY group_id ORDER BY n DESC LIMIT 10"
        ) as cur:
            per_group = [(r["group_id"], r["n"]) for r in await cur.fetchall()]
        return {"messages": total, "per_group": per_group}

    async def purge_old_messages(self, retention_days: int, now: int | None = None) -> int:
        cutoff = int(now or time.time()) - retention_days * DAY
        cur = await self.conn.execute(
            "DELETE FROM messages WHERE ts < ? AND is_self = 0", (cutoff,)
        )
        await self.conn.commit()
        return cur.rowcount
