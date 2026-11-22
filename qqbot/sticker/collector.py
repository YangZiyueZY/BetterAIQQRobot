"""收藏规则引擎。

真人不糊弄：不会把看到的每个表情都存下来。库一旦被垃圾塞满，
"能用对表情"这个目标就彻底失败了。所以这里的原则是**克制**，
而且每一条判断都给出理由（`reason`），面板上能复盘"它当时为什么没收"。

判定顺序刻意从便宜到贵：
    开关 -> 是否已收 -> 每日上限 -> 单人上限 -> 容量 -> 通用性 -> 随机 -> 心情
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from ..config import StickerConfig
from ..store.stickers import StickerRepo

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400


@dataclass(slots=True)
class CollectVerdict:
    collect: bool
    reason: str

    def __bool__(self) -> bool:
        return self.collect


class StickerCollector:
    def __init__(self, cfg: StickerConfig, repo: StickerRepo) -> None:
        self.cfg = cfg
        self.repo = repo
        self.collected = 0
        self.rejected: dict[str, int] = {}

    async def consider(
        self,
        *,
        sticker_id: int,
        group_id: int,
        user_id: int,
        is_new: bool = True,
        mood: float = 0.0,
        now: int | None = None,
    ) -> CollectVerdict:
        """判断这张表情值不值得收藏。

        注意 `is_new` 只表示"这是第一次见到这个哈希"，**不是**"已经收藏过"。
        门禁判的是 `collected` 字段 —— 否则会出现这个死结：
            第一次见 -> 不够通用 -> 拒
            之后再见 -> is_new=False -> 拒
        结果是永远收不了任何一张。
        """
        now = int(now or time.time())
        cfg = self.cfg

        def no(reason: str) -> CollectVerdict:
            self.rejected[reason] = self.rejected.get(reason, 0) + 1
            return CollectVerdict(False, reason)

        if not cfg.enabled or not cfg.auto_collect:
            return no("自动收藏已关闭")

        row = await self.repo.get(sticker_id)
        if row is None:
            return no("表情不存在")
        if row.get("collected"):
            return no("库里已经有了")

        # 每日上限 —— 真人一天存 3 个已经很多了
        day_start = now - (now % DAY)
        if await self.repo.collected_count(group_id, day_start) >= cfg.collect_daily_limit:
            return no(f"本群今日收藏已达上限({cfg.collect_daily_limit})")

        # 同一个来源一小时内最多贡献一张，防止某人刷屏把库塞爆
        if await self.repo.collected_from_user(group_id, user_id, now - HOUR) >= cfg.per_user_hourly:
            return no("该群友一小时内已贡献过表情")

        # 通用性：要么被看到多次，要么有多个不同的人用过
        # —— 这才是"本群的梗"，而不是某人的私货
        seen = int(row.get("seen_count") or 1)
        users = await self.repo.distinct_observers(sticker_id)
        if seen < cfg.collect_min_seen and users < cfg.collect_min_users:
            return no(f"还不够通用(见到{seen}次/{users}人)")

        # 心情门槛：她情绪低落时不会注意到好玩的梗
        if mood < -0.35:
            return no("心情不好，没兴趣收")

        # 满足条件也不是必收 —— 保留随机性
        if random.random() > cfg.collect_prob:
            return no("随机跳过")

        # 容量：满了就淘汰最没用的，而不是拒收新的
        await self._enforce_capacity()

        self.collected += 1
        return CollectVerdict(True, f"通用梗(见到{seen}次/{users}人)")

    async def _enforce_capacity(self) -> int:
        """超容量时淘汰得分最低的。文件保留 —— 哈希命名的文件再收一次是零成本的。"""
        total = await self.repo.count(include_blocked=True)
        if total < self.cfg.capacity:
            return 0
        overflow = total - self.cfg.capacity + 1
        from ..store.evolution import BOUNDS  # noqa: F401  (仅确保依赖健康，无副作用)

        victims = await self.repo.oldest_by_score(limit=max(overflow, 1))
        removed = 0
        for v in victims[:overflow]:
            if await self.repo.delete(int(v["id"])):
                removed += 1
        if removed:
            log.info("表情库已满，淘汰 %d 张", removed)
        return removed

    def stats(self) -> dict:
        return {
            "stickers_collected": self.collected,
            "stickers_rejected": sum(self.rejected.values()),
            "collect_reject_reasons": dict(
                sorted(self.rejected.items(), key=lambda kv: -kv[1])[:6]
            ),
        }
