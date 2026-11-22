"""提示词覆盖层的存储。

只存**改过的**条目。代码里的默认值始终是事实来源，
重置 = 删记录，不需要把默认值复制一份到库里。

**两个层级**：
    group_id = 0   全局覆盖（对所有群生效）
    group_id > 0   只对该群生效，优先级更高

需要按群是因为有些提示词与具体人设强相关 —— 比如「沉默是常态」适合潜水的
群友，但完全不适合一个讲解题目的角色。全局只能改一处，按群才能各用各的。
"""

from __future__ import annotations

import logging
import time

from .db import Database

log = logging.getLogger(__name__)

GLOBAL = 0


class PromptRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    async def all(self, group_id: int = GLOBAL) -> dict[str, str]:
        """取该群**当前生效**的覆盖（群覆盖 + 全局覆盖，群优先）。"""
        async with self.conn.execute(
            "SELECT key, group_id, text FROM prompt_overrides"
        ) as cur:
            rows = await cur.fetchall()
        out: dict[str, str] = {}
        for r in rows:                       # 先铺全局
            if int(r["group_id"]) == GLOBAL:
                out[str(r["key"])] = str(r["text"])
        for r in rows:                       # 再盖群级
            if int(r["group_id"]) == group_id and group_id != GLOBAL:
                out[str(r["key"])] = str(r["text"])
        return out

    async def all_tiers(self) -> dict[int, dict[str, str]]:
        """一次取回所有层，启动时灌进进程缓存用。"""
        async with self.conn.execute(
            "SELECT key, group_id, text FROM prompt_overrides"
        ) as cur:
            rows = await cur.fetchall()
        out: dict[int, dict[str, str]] = {}
        for r in rows:
            out.setdefault(int(r["group_id"]), {})[str(r["key"])] = str(r["text"])
        return out

    async def own(self, group_id: int = GLOBAL) -> dict[str, str]:
        """只取该层自己改过的条目（面板要区分"这一层改了什么"）。"""
        async with self.conn.execute(
            "SELECT key, text FROM prompt_overrides WHERE group_id = ?",
            (group_id,),
        ) as cur:
            return {str(r["key"]): str(r["text"]) for r in await cur.fetchall()}

    async def set(self, key: str, text: str, group_id: int = GLOBAL,
                  now: int | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO prompt_overrides (key, group_id, text, updated_ts) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key, group_id) DO UPDATE SET text = excluded.text, "
            "updated_ts = excluded.updated_ts",
            (key, group_id, text, int(now or time.time())),
        )
        await self.conn.commit()

    async def reset(self, key: str, group_id: int = GLOBAL) -> bool:
        cur = await self.conn.execute(
            "DELETE FROM prompt_overrides WHERE key = ? AND group_id = ?",
            (key, group_id),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def reset_all(self, group_id: int | None = None) -> int:
        """不传 group_id 就清空全部（两层都清）。"""
        if group_id is None:
            cur = await self.conn.execute("DELETE FROM prompt_overrides")
        else:
            cur = await self.conn.execute(
                "DELETE FROM prompt_overrides WHERE group_id = ?", (group_id,)
            )
        await self.conn.commit()
        return cur.rowcount

    async def reset_group(self, group_id: int) -> int:
        """删掉某群的全部群级覆盖（不动全局）。"""
        return await self.reset_all(group_id)

    async def tiers(self) -> dict[str, int]:
        """两层各有几条 —— 面板显示用。"""
        async with self.conn.execute(
            "SELECT group_id, COUNT(*) AS n FROM prompt_overrides GROUP BY group_id"
        ) as cur:
            rows = await cur.fetchall()
        out = {"global": 0, "by_group": 0}
        for r in rows:
            if int(r["group_id"]) == GLOBAL:
                out["global"] = int(r["n"])
            else:
                out["by_group"] += int(r["n"])
        return out

    async def count(self) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM prompt_overrides"
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def stats(self) -> dict:
        t = await self.tiers()
        return {"prompt_overrides": t["global"] + t["by_group"],
                "prompt_overrides_global": t["global"],
                "prompt_overrides_group": t["by_group"]}
