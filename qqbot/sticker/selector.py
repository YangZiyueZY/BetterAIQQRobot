"""表情选用器。

"能用对"比"能发"难得多。这里做三件事：

  1. **情绪匹配**：按当前对话情绪挑 tag 命中的
  2. **偏爱常用**：自己喜欢的表情用得更多（win_count 加权），真人都有口头禅表情
  3. **冷却**：同一张短期内不重复，本群最近若干条里出现过就不发

排序用 win_rate 而不是纯 use_count —— 发得多但没人理的表情应该降权。
ε-greedy 保留探索：偶尔翻出很久没用的老表情。
"""

from __future__ import annotations

import logging
import random
import time

from ..config import StickerConfig
from ..models import StickerRef
from ..store.stickers import StickerRepo

log = logging.getLogger(__name__)

DAY = 86400

#: 语境情绪 -> 可能的标签。模型给的是宽泛情绪，库里标签更具体。
MOOD_ALIASES: dict[str, tuple[str, ...]] = {
    "搞笑": ("搞笑", "大笑", "笑死", "欢乐", "沙雕", "哈哈哈"),
    "无语": ("无语", "嫌弃", "冷漠", "呵", "白眼", "服了"),
    "赞同": ("赞同", "赞", "牛", "强", "支持", "666", "佩服"),
    "开心": ("开心", "快乐", "高兴", "庆祝", "撒花"),
    "哭泣": ("哭", "流泪", "委屈", "可怜", "难过", "泪"),
    "生气": ("生气", "愤怒", "怒", "凶", "炸"),
    "害羞": ("害羞", "羞", "脸红", "可爱", "萌", "撒娇"),
    "惊讶": ("惊讶", "震惊", "惊", "呆", "愕然"),
    "得意": ("得意", "骄傲", "嚣张", "酷", "拽"),
    "疑惑": ("疑惑", "问号", "迷惑", "不解", "懵"),
    "累": ("累", "困", "疲惫", "摆烂", "摸鱼", "躺"),
    "晚安": ("晚安", "睡", "拜拜", "下线"),
}

#: 兜底标签池：库里没有明确标签时也不至于没得挑
FALLBACK_MOODS = ("搞笑", "赞同", "无语", "开心")


class StickerSelector:
    def __init__(self, cfg: StickerConfig, repo: StickerRepo) -> None:
        self.cfg = cfg
        self.repo = repo
        self.sent = 0

    # ---- 给 prompt 用 -------------------------------------------------

    async def mood_labels(self, limit: int = 14) -> list[str]:
        """库里实际有的情绪标签，按数量排序。"""
        counts = await self.repo.tag_counts()
        if not counts:
            return []
        return list(counts.keys())[:limit]

    # ---- 选一张 -------------------------------------------------------

    async def resolve(
        self,
        mood: str | None,
        group_id: int,
        *,
        context: str = "",
        now: int | None = None,
    ) -> StickerRef | None:
        """把模型给的情绪标签解析成一张具体的表情包。没有合适的就返回 None。"""
        if not self.cfg.enabled:
            return None
        now = int(now or time.time())

        tags = self._expand_tags(mood)
        candidates = await self.repo.candidates(tags, limit=120)
        if not candidates:
            return None

        # 冷却过滤：本群最近用过的 + 近期本群出现过的
        cooldown_since = now - self.cfg.group_cooldown_days * DAY
        used_recently = await self.repo.recently_used(group_id, cooldown_since)
        pool = [c for c in candidates if int(c["id"]) not in used_recently]
        if not pool:
            # 冷却把库掏空了就放宽到"最近没在本群发过"
            pool = [c for c in candidates
                    if await self.repo.last_use_ts(int(c["id"]), group_id) < now - DAY]
        if not pool:
            return None

        chosen = self._pick(pool, tags)
        if chosen is None:
            return None

        self.sent += 1
        return StickerRef(
            id=int(chosen["id"]),
            hash=chosen["hash"],
            path=chosen.get("path") or "",
            mood_tags=chosen.get("mood_tags") or [],
            desc=chosen.get("desc") or "",
        )

    def _expand_tags(self, mood: str | None) -> list[str]:
        if not mood:
            return list(FALLBACK_MOODS)
        mood = mood.strip()
        out = [mood]
        out.extend(MOOD_ALIASES.get(mood, ()))
        return list(dict.fromkeys(out))

    def _pick(self, pool: list[dict], tags: list[str]) -> dict | None:
        # 探索：偶尔翻出平时不用的老表情，避免永远只发那三张
        if random.random() < self.cfg.explore_rate:
            return random.choice(pool)

        wanted = set(tags)

        def score(item: dict) -> float:
            uses = int(item.get("use_count") or 0)
            wins = int(item.get("win_count") or 0)
            # 发得多但没人理的表情要降权；标签命中大幅加权
            win_rate = (wins + 1.0) / (uses + 2.0)
            tag_hit = len(wanted & set(item.get("mood_tags") or []))
            freshness = 1.0 / (1.0 + uses * 0.05)
            # 命中情绪是最重要的信号，权重给足，否则会随机挑到不搭的表情
            return win_rate * (1.0 + tag_hit * 2.5) * freshness

        pool = sorted(pool, key=score, reverse=True)
        top = pool[: max(1, min(5, len(pool)))]
        # 在头部里加权随机，而不是永远取第一 —— 否则行为会可预测
        weights = [max(0.05, score(x)) for x in top]
        total = sum(weights)
        if total <= 0:
            return random.choice(top)
        r = random.uniform(0, total)
        acc = 0.0
        for item, w in zip(top, weights):
            acc += w
            if r <= acc:
                return item
        return top[0]

    def stats(self) -> dict:
        return {"stickers_sent": self.sent}
