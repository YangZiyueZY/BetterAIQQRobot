"""云端 OCR 任务台账。

PaddleOCR 是任务制的：提交拿到 jobId → 轮询 → 取 JSONL 结果。
整个过程可能跑几分钟，所以必须落库 —— 否则进程一重启，在途任务就白跑了。
"""

from __future__ import annotations

import logging
import time

from .db import Database

log = logging.getLogger(__name__)

PENDING_STATES = ("pending", "running")


class OcrJobRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def create(self, job_id: str, media_hash: str,
                     backend: str = "paddleocr",
                     now: int | None = None) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO ocr_jobs "
            "(job_id, media_hash, backend, state, submitted_ts) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (job_id, media_hash, backend, int(now or time.time())),
        )
        await self.conn.commit()

    async def set_state(self, job_id: str, state: str, *,
                        error: str = "", chars: int = 0,
                        now: int | None = None) -> None:
        await self.conn.execute(
            "UPDATE ocr_jobs SET state = ?, error = ?, chars = ?, finished_ts = ? "
            "WHERE job_id = ?",
            (state, error[:300], chars,
             int(now or time.time()) if state not in PENDING_STATES else None,
             job_id),
        )
        await self.conn.commit()

    async def get(self, job_id: str) -> dict | None:
        async with self.conn.execute(
            "SELECT * FROM ocr_jobs WHERE job_id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def pending(self, limit: int = 50) -> list[dict]:
        """还没结束的任务 —— 启动时用来接着收结果。"""
        placeholders = ",".join("?" * len(PENDING_STATES))
        async with self.conn.execute(
            f"SELECT * FROM ocr_jobs WHERE state IN ({placeholders}) "
            "ORDER BY submitted_ts ASC LIMIT ?",
            (*PENDING_STATES, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def has_pending_for(self, media_hash: str) -> bool:
        placeholders = ",".join("?" * len(PENDING_STATES))
        async with self.conn.execute(
            f"SELECT 1 FROM ocr_jobs WHERE media_hash = ? "
            f"AND state IN ({placeholders}) LIMIT 1",
            (media_hash, *PENDING_STATES),
        ) as cur:
            return await cur.fetchone() is not None

    async def stats(self) -> dict:
        async with self.conn.execute(
            "SELECT state, COUNT(*) AS n FROM ocr_jobs GROUP BY state"
        ) as cur:
            by_state = {r["state"]: int(r["n"]) for r in await cur.fetchall()}
        return {
            "ocr_jobs": sum(by_state.values()),
            "ocr_jobs_by_state": by_state,
            "ocr_jobs_pending": sum(by_state.get(s, 0) for s in PENDING_STATES),
        }

    async def purge(self, keep_days: int = 30) -> int:
        cutoff = int(time.time()) - keep_days * 86400
        cur = await self.conn.execute(
            "DELETE FROM ocr_jobs WHERE state NOT IN ('pending','running') "
            "AND COALESCE(finished_ts, submitted_ts) < ?",
            (cutoff,),
        )
        await self.conn.commit()
        return cur.rowcount
