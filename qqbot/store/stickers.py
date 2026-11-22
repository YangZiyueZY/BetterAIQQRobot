"""表情库。

文件按内容哈希命名（`data/stickers/<sha256>.<ext>`），所以同一张图重复收
也不会存两份。语义标签由"观察人类怎么用"产出（见 sticker/tagger.py）。
"""

from __future__ import annotations

import json
import logging
import time

from .db import Database

log = logging.getLogger(__name__)


def _row(r) -> dict:
    out = dict(r)
    for key in ("mood_tags", "content_tags"):
        try:
            out[key] = json.loads(out.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            out[key] = []
    return out


class StickerRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    # ---- 入库 ---------------------------------------------------------

    async def add(
        self,
        *,
        hash: str,  # noqa: A002
        ext: str = "png",
        path: str = "",
        size: int = 0,
        source_group: int | None = None,
        source_user: int | None = None,
        source_msg_id: int | None = None,
        is_animated: bool = False,
        is_mark_face: bool = False,
        now: int | None = None,
    ) -> tuple[int, bool]:
        """返回 (sticker_id, 是否新收藏)。已存在则只累加 seen_count。"""
        now = int(now or time.time())
        async with self.conn.execute(
            "SELECT id FROM stickers WHERE hash = ?", (hash,)
        ) as cur:
            row = await cur.fetchone()

        if row:
            await self.conn.execute(
                "UPDATE stickers SET seen_count = seen_count + 1 WHERE id = ?",
                (row["id"],),
            )
            await self.conn.commit()
            return int(row["id"]), False

        cur = await self.conn.execute(
            """
            INSERT INTO stickers
                (hash, ext, path, size, is_animated, is_mark_face,
                 source_group, source_user, source_msg_id, collected_ts, seen_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (hash, ext, path, size, int(is_animated), int(is_mark_face),
             source_group, source_user, source_msg_id, now),
        )
        await self.conn.commit()
        return int(cur.lastrowid or 0), True

    async def mark_collected(self, sticker_id: int, reason: str = "",
                             now: int | None = None) -> None:
        """把"见过"升级成"收藏"。这才是可以拿来发的前提。"""
        await self.conn.execute(
            "UPDATE stickers SET collected = 1, collect_reason = ?, collected_ts = ? "
            "WHERE id = ?",
            (reason[:120], int(now or time.time()), sticker_id),
        )
        await self.conn.commit()

    async def set_after_text(self, group_id: int, text: str) -> None:
        """给本群最近一条 after_text 为空的观察补上后文。

        表情的语义要靠"它被用在哪句话之后"来学，但后文要等下一条消息才知道，
        所以入库时先留空，下一条消息到达时回填。
        """
        if not text.strip():
            return
        await self.conn.execute(
            """
            UPDATE sticker_observations SET after_text = ?
            WHERE id = (
                SELECT id FROM sticker_observations
                WHERE group_id = ? AND after_text = '' AND tagged = 0
                ORDER BY ts DESC LIMIT 1
            )
            """,
            (text[:200], group_id),
        )
        await self.conn.commit()

    async def bump_seen(self, sticker_hash: str) -> None:
        await self.conn.execute(
            "UPDATE stickers SET seen_count = seen_count + 1 WHERE hash = ?",
            (sticker_hash,),
        )
        await self.conn.commit()

    async def bump_use(self, sticker_id: int, *, win: bool = False,
                       now: int | None = None) -> None:
        now = int(now or time.time())
        await self.conn.execute(
            "UPDATE stickers SET use_count = use_count + 1, "
            "win_count = win_count + ?, last_used_ts = ? WHERE id = ?",
            (int(win), now, sticker_id),
        )
        await self.conn.commit()

    async def set_tags(self, sticker_id: int, mood_tags: list[str],
                       content_tags: list[str] | None = None,
                       desc: str | None = None) -> None:
        sets = ["mood_tags = ?"]
        args: list = [json.dumps(mood_tags, ensure_ascii=False)]
        if content_tags is not None:
            sets.append("content_tags = ?")
            args.append(json.dumps(content_tags, ensure_ascii=False))
        if desc is not None:
            sets.append("desc = ?")
            args.append(desc)
        args.append(sticker_id)
        await self.conn.execute(
            f"UPDATE stickers SET {', '.join(sets)} WHERE id = ?", tuple(args)
        )
        await self.conn.commit()

    async def set_qq_res_id(self, sticker_id: int, res_id: str) -> None:
        await self.conn.execute(
            "UPDATE stickers SET qq_res_id = ? WHERE id = ?", (res_id, sticker_id)
        )
        await self.conn.commit()

    async def set_blocked(self, sticker_id: int, blocked: bool) -> None:
        await self.conn.execute(
            "UPDATE stickers SET blocked = ? WHERE id = ?", (int(blocked), sticker_id)
        )
        await self.conn.commit()

    async def delete(self, sticker_id: int) -> bool:
        cur = await self.conn.execute("DELETE FROM stickers WHERE id = ?", (sticker_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    # ---- 查询 ---------------------------------------------------------

    async def get(self, sticker_id: int) -> dict | None:
        async with self.conn.execute(
            "SELECT * FROM stickers WHERE id = ?", (sticker_id,)
        ) as cur:
            row = await cur.fetchone()
        return _row(row) if row else None

    async def by_hash(self, sticker_hash: str) -> dict | None:
        async with self.conn.execute(
            "SELECT * FROM stickers WHERE hash = ?", (sticker_hash,)
        ) as cur:
            row = await cur.fetchone()
        return _row(row) if row else None

    async def list_all(self, limit: int = 300, include_blocked: bool = False,
                       collected_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM stickers"
        conds = []
        if not include_blocked:
            conds.append("blocked = 0")
        if collected_only:
            conds.append("collected = 1")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY use_count DESC, collected_ts DESC LIMIT ?"
        async with self.conn.execute(sql, (limit,)) as cur:
            return [_row(r) for r in await cur.fetchall()]

    async def count(self, include_blocked: bool = False,
                    collected_only: bool = True) -> int:
        sql = "SELECT COUNT(*) AS n FROM stickers"
        conds = []
        if not include_blocked:
            conds.append("blocked = 0")
        if collected_only:
            conds.append("collected = 1")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        async with self.conn.execute(sql) as cur:
            return int((await cur.fetchone())["n"] or 0)

    async def untagged(self, limit: int = 40, min_obs: int = 3) -> list[dict]:
        """攒够观察样本但还没打标签的表情 —— tagger 的输入。"""
        async with self.conn.execute(
            """
            SELECT s.*, COUNT(o.id) AS obs_count
            FROM stickers s
            JOIN sticker_observations o ON o.sticker_id = s.id AND o.tagged = 0
            WHERE s.mood_tags = '[]'
            GROUP BY s.id
            HAVING obs_count >= ?
            ORDER BY obs_count DESC LIMIT ?
            """,
            (min_obs, limit),
        ) as cur:
            return [_row(r) for r in await cur.fetchall()]

    async def candidates(self, mood_tags: list[str] | None = None,
                         limit: int = 80) -> list[dict]:
        """给选用器的候选池：未封禁、未在冷却期。"""
        async with self.conn.execute(
            "SELECT * FROM stickers WHERE blocked = 0 AND collected = 1 "
            "ORDER BY use_count DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = [_row(r) for r in await cur.fetchall()]

        if not mood_tags:
            return rows
        wanted = set(mood_tags)
        scored = [(len(wanted & set(r["mood_tags"])), r) for r in rows]
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored]

    async def recently_used(self, group_id: int, since_ts: int) -> set[int]:
        """本群近期用过的表情 id，用于冷却。"""
        async with self.conn.execute(
            "SELECT DISTINCT sticker_id FROM sticker_usage "
            "WHERE group_id = ? AND ts >= ?",
            (group_id, since_ts),
        ) as cur:
            return {int(r["sticker_id"]) for r in await cur.fetchall()}

    # ---- 收藏规则要用到的计数 -----------------------------------------

    async def distinct_observers(self, sticker_id: int) -> int:
        """有几个人用过这个表情 —— 判断它是不是"本群通用梗"。"""
        async with self.conn.execute(
            "SELECT COUNT(DISTINCT user_id) AS n FROM sticker_observations "
            "WHERE sticker_id = ?",
            (sticker_id,),
        ) as cur:
            return int((await cur.fetchone())["n"] or 0)

    async def collected_count(self, group_id: int, since_ts: int) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM stickers "
            "WHERE collected = 1 AND source_group = ? AND collected_ts >= ?",
            (group_id, since_ts),
        ) as cur:
            return int((await cur.fetchone())["n"] or 0)

    async def collected_from_user(self, group_id: int, user_id: int,
                                  since_ts: int) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM stickers "
            "WHERE collected = 1 AND source_group = ? AND source_user = ? "
            "AND collected_ts >= ?",
            (group_id, user_id, since_ts),
        ) as cur:
            return int((await cur.fetchone())["n"] or 0)

    async def tag_counts(self) -> dict[str, int]:
        """库里出现过的情绪标签及频次 —— 给 prompt 列"可用情绪"。"""
        async with self.conn.execute(
            "SELECT mood_tags FROM stickers WHERE blocked = 0 AND collected = 1 "
            "AND mood_tags != '[]'"
        ) as cur:
            rows = await cur.fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            try:
                tags = json.loads(r["mood_tags"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    async def oldest_by_score(self, limit: int = 50) -> list[dict]:
        """淘汰候选：用得少、赢得少的排前面。"""
        async with self.conn.execute(
            "SELECT * FROM stickers WHERE collected = 1 "
            "ORDER BY (use_count * 2 + win_count) ASC, "
            "collected_ts ASC LIMIT ?",
            (limit,),
        ) as cur:
            return [_row(r) for r in await cur.fetchall()]

    async def last_use_ts(self, sticker_id: int, group_id: int | None = None) -> int:
        sql = "SELECT MAX(ts) AS ts FROM sticker_usage WHERE sticker_id = ?"
        args: tuple = (sticker_id,)
        if group_id is not None:
            sql += " AND group_id = ?"
            args = (sticker_id, group_id)
        async with self.conn.execute(sql, args) as cur:
            row = await cur.fetchone()
        return int((row["ts"] if row else 0) or 0)

    # ---- 使用记录 / 观察 ----------------------------------------------

    async def record_usage(self, sticker_id: int, group_id: int,
                           msg_id: int | None, context: str = "",
                           now: int | None = None) -> int:
        cur = await self.conn.execute(
            "INSERT INTO sticker_usage (sticker_id, group_id, msg_id, ts, context) "
            "VALUES (?, ?, ?, ?, ?)",
            (sticker_id, group_id, msg_id, int(now or time.time()), context[:300]),
        )
        await self.conn.commit()
        return int(cur.lastrowid or 0)

    async def observe(self, sticker_id: int, group_id: int, user_id: int,
                      before_text: str, after_text: str,
                      now: int | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO sticker_observations "
            "(sticker_id, group_id, user_id, before_text, after_text, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sticker_id, group_id, user_id, before_text[:200], after_text[:200],
             int(now or time.time())),
        )
        await self.conn.commit()

    async def observations(self, sticker_id: int, limit: int = 20) -> list[dict]:
        async with self.conn.execute(
            "SELECT * FROM sticker_observations WHERE sticker_id = ? "
            "ORDER BY ts DESC LIMIT ?",
            (sticker_id, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def mark_observed_tagged(self, sticker_id: int) -> None:
        await self.conn.execute(
            "UPDATE sticker_observations SET tagged = 1 WHERE sticker_id = ?",
            (sticker_id,),
        )
        await self.conn.commit()

    # ---- 统计 ---------------------------------------------------------

    async def stats(self) -> dict:
        async with self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(collected) AS collected, "
            "SUM(CASE WHEN mood_tags != '[]' AND collected = 1 THEN 1 ELSE 0 END) AS tagged, "
            "SUM(use_count) AS uses, "
            "SUM(CASE WHEN blocked = 1 THEN 1 ELSE 0 END) AS blocked "
            "FROM stickers"
        ) as cur:
            row = await cur.fetchone()
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM sticker_observations WHERE tagged = 0"
        ) as cur:
            pending = int((await cur.fetchone())["n"] or 0)
        return {
            "stickers_seen": int(row["total"] or 0),
            "stickers": int(row["collected"] or 0),
            "stickers_tagged": int(row["tagged"] or 0),
            "sticker_uses": int(row["uses"] or 0),
            "stickers_blocked": int(row["blocked"] or 0),
            "sticker_obs_pending": pending,
        }
