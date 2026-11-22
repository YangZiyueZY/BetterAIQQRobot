"""冷启动分析的存储。

`questions` 和 `asked` 都存 JSON 字符串 —— 它们只在冷启动阶段用，
为它单开一张表不值得。`asked` 单独记是为了"问过的不再问"。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .db import Database

log = logging.getLogger(__name__)

STATUS_IDLE = ""
STATUS_PULLING = "pulling"
STATUS_ANALYZING = "analyzing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"


@dataclass
class OnboardRow:
    group_id: int
    status: str = STATUS_IDLE
    pulled: int = 0
    analyzed_ts: int = 0
    questions: list[dict] = field(default_factory=list)
    asked: list[dict] = field(default_factory=list)
    note: str = ""
    updated_ts: int = 0

    @property
    def pending(self) -> list[dict]:
        """还没问过的问题。"""
        done = {str(a.get("text")) for a in self.asked}
        return [q for q in self.questions if str(q.get("text")) not in done]


class OnboardRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def get(self, group_id: int) -> OnboardRow:
        async with self.conn.execute(
            "SELECT * FROM onboard_state WHERE group_id = ?", (group_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return OnboardRow(group_id=group_id)
        return _to_row(dict(row))

    async def set_status(self, group_id: int, status: str, *,
                         pulled: int | None = None,
                         note: str | None = None) -> None:
        now = int(time.time())
        await self.conn.execute(
            "INSERT INTO onboard_state (group_id, status, pulled, note, updated_ts) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(group_id) DO UPDATE SET "
            "status = excluded.status, "
            "pulled = COALESCE(?, onboard_state.pulled), "
            "note = COALESCE(?, onboard_state.note), "
            "updated_ts = excluded.updated_ts",
            (group_id, status, int(pulled or 0), note or "", now,
             pulled, note),
        )
        await self.conn.commit()

    async def save_analysis(self, group_id: int, questions: list[dict],
                            note: str = "") -> None:
        """分析完成：写入问题清单，状态转 ready。

        注意 **不覆盖 already asked** —— 重新分析不该让问过的问题复活。
        """
        now = int(time.time())
        await self.conn.execute(
            "INSERT INTO onboard_state "
            "(group_id, status, questions, note, analyzed_ts, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(group_id) DO UPDATE SET "
            "status = excluded.status, questions = excluded.questions, "
            "note = excluded.note, analyzed_ts = excluded.analyzed_ts, "
            "updated_ts = excluded.updated_ts",
            (group_id, STATUS_READY, json.dumps(questions, ensure_ascii=False),
             note, now, now),
        )
        await self.conn.commit()

    async def mark_asked(self, group_id: int, text: str) -> None:
        row = await self.get(group_id)
        asked = list(row.asked)
        asked.append({"text": text, "ts": int(time.time())})
        await self.conn.execute(
            "UPDATE onboard_state SET asked = ?, updated_ts = ? WHERE group_id = ?",
            (json.dumps(asked[-50:], ensure_ascii=False), int(time.time()), group_id),
        )
        await self.conn.commit()

    async def reset(self, group_id: int) -> None:
        await self.conn.execute(
            "DELETE FROM onboard_state WHERE group_id = ?", (group_id,))
        await self.conn.commit()

    async def all_rows(self) -> list[OnboardRow]:
        async with self.conn.execute("SELECT * FROM onboard_state") as cur:
            return [_to_row(dict(r)) for r in await cur.fetchall()]

    async def stats(self) -> dict:
        async with self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM onboard_state GROUP BY status"
        ) as cur:
            rows = await cur.fetchall()
        return {f"onboard_{r['status'] or 'idle'}": int(r["n"]) for r in rows}


def _to_row(raw: dict) -> OnboardRow:
    def loads(key: str) -> list[dict]:
        try:
            data = json.loads(raw.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            return []
        return [d for d in data if isinstance(d, dict)]

    return OnboardRow(
        group_id=int(raw["group_id"]),
        status=str(raw.get("status") or ""),
        pulled=int(raw.get("pulled") or 0),
        analyzed_ts=int(raw.get("analyzed_ts") or 0),
        questions=loads("questions"),
        asked=loads("asked"),
        note=str(raw.get("note") or ""),
        updated_ts=int(raw.get("updated_ts") or 0),
    )
