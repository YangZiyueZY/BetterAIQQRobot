"""逐群人设的存储。

**没有记录 = 跟随全局默认**（persona.yaml）。所以"差异化"是显式动作，
不是默认状态 —— 用户不动就永远是统一人设。
"""

from __future__ import annotations

import logging
import time

from .db import Database

log = logging.getLogger(__name__)


class PersonaRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def get(self, group_id: int) -> str:
        """取该群的 YAML。没有自定义就返回空串。"""
        async with self.conn.execute(
            "SELECT yaml FROM group_personas WHERE group_id = ?", (group_id,)
        ) as cur:
            row = await cur.fetchone()
        return str(row["yaml"]) if row else ""

    async def set(self, group_id: int, yaml_text: str,
                  now: int | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO group_personas (group_id, yaml, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(group_id) DO UPDATE SET yaml = excluded.yaml, "
            "updated_ts = excluded.updated_ts",
            (group_id, yaml_text, int(now or time.time())),
        )
        await self.conn.commit()

    async def reset(self, group_id: int) -> bool:
        """删掉自定义，回到跟随默认。返回是否真的有东西被删。"""
        cur = await self.conn.execute(
            "DELETE FROM group_personas WHERE group_id = ?", (group_id,)
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def custom_ids(self) -> list[int]:
        """哪些群有自定义人设 —— 面板上要打标记。"""
        async with self.conn.execute(
            "SELECT group_id FROM group_personas ORDER BY group_id"
        ) as cur:
            return [int(r["group_id"]) for r in await cur.fetchall()]

    async def count(self) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM group_personas"
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def stats(self) -> dict:
        return {"group_personas": await self.count()}
