"""多模态缓存：图片/语音的内容哈希 -> 理解结果。

群里的表情包和梗图高度重复，同一张图发十次很常见。按哈希缓存能让
VLM 的实际调用量远低于"每张都算"，成本直接降一个量级。
"""

from __future__ import annotations

import json
import logging
import time

from .db import Database

log = logging.getLogger(__name__)


def _row(r) -> dict:
    out = dict(r)
    try:
        out["caption"] = json.loads(out.get("caption") or "{}")
    except (json.JSONDecodeError, TypeError):
        out["caption"] = {}
    return out


class MediaRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def get(self, hash: str) -> dict | None:  # noqa: A002
        async with self.conn.execute(
            "SELECT * FROM media_cache WHERE hash = ?", (hash,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        await self.conn.execute(
            "UPDATE media_cache SET hits = hits + 1, last_hit_ts = ? WHERE hash = ?",
            (int(time.time()), hash),
        )
        await self.conn.commit()
        return _row(row)

    async def upsert(
        self,
        *,
        hash: str,  # noqa: A002
        kind: str = "image",
        path: str = "",
        summary: str = "",
        caption: dict | None = None,
        ocr_text: str = "",
        ptt_text: str = "",
        sensitive: bool = False,
        now: int | None = None,
    ) -> None:
        now = int(now or time.time())
        await self.conn.execute(
            """
            INSERT INTO media_cache
                (hash, kind, path, summary, caption, ocr_text, ptt_text,
                 sensitive, created_ts, last_hit_ts, hits)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT (hash) DO UPDATE SET
                path      = COALESCE(NULLIF(excluded.path, ''), media_cache.path),
                summary   = COALESCE(NULLIF(excluded.summary, ''), media_cache.summary),
                caption   = CASE WHEN excluded.caption != '{}'
                                 THEN excluded.caption ELSE media_cache.caption END,
                ocr_text  = COALESCE(NULLIF(excluded.ocr_text, ''), media_cache.ocr_text),
                ptt_text  = COALESCE(NULLIF(excluded.ptt_text, ''), media_cache.ptt_text),
                sensitive = MAX(excluded.sensitive, media_cache.sensitive),
                last_hit_ts = excluded.last_hit_ts
            """,
            (hash, kind, path, summary,
             json.dumps(caption or {}, ensure_ascii=False),
             ocr_text, ptt_text, int(sensitive), now, now),
        )
        await self.conn.commit()

    async def count(self) -> int:
        async with self.conn.execute("SELECT COUNT(*) AS n FROM media_cache") as cur:
            return int((await cur.fetchone())["n"] or 0)

    async def stats(self) -> dict:
        async with self.conn.execute(
            "SELECT kind, COUNT(*) AS n, SUM(hits) AS hits FROM media_cache GROUP BY kind"
        ) as cur:
            rows = await cur.fetchall()
        total_hits = sum(int(r["hits"] or 0) for r in rows)
        return {
            "media_cached": sum(int(r["n"] or 0) for r in rows),
            "media_cache_hits": total_hits,
        }

    async def purge(self, keep_days: int = 60) -> int:
        cutoff = int(time.time()) - keep_days * 86400
        cur = await self.conn.execute(
            "DELETE FROM media_cache WHERE COALESCE(last_hit_ts, created_ts) < ?",
            (cutoff,),
        )
        await self.conn.commit()
        return cur.rowcount
