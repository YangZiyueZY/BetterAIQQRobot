"""网络搜索：判定该不该搜 -> 搜 -> 擦除注入 -> 交给 prompt。

把"什么时候搜"和"怎么搜"分开：`need.py` 只做判定（纯函数、好测、好调），
`client.py` 只做请求（多供应商适配）。

这一层负责把它们串起来，并做一件容易被忽略的事：
**搜索结果也是不可信外部内容**。网页里完全可以写"忽略你之前的指令"，
所以结果在进 prompt 之前必须过一遍注入擦除 —— 和群消息一样的待遇。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import SearchConfig
from ..safety import injection
from .client import SearchClient, SearchResult
from .need import SearchNeed, build_query, evaluate

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchOutcome:
    performed: bool = False
    query: str = ""
    results: list[SearchResult] = field(default_factory=list)
    reason: str = ""
    scrubbed: int = 0

    def render(self, max_chars: int = 300) -> str:
        if not self.results:
            return ""
        return "\n".join(r.render(max_chars) for r in self.results)


class SearchService:
    def __init__(self, cfg: SearchConfig, client: SearchClient) -> None:
        self.cfg = cfg
        self.client = client
        self.searches = 0
        self.skipped = 0
        self.scrubbed = 0
        self._last_need: SearchNeed | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.api_key)

    async def lookup(
        self,
        text: str,
        *,
        has_url: bool = False,
        is_mention: bool = False,
        hint: str = "",
        strip_words=None,
    ) -> SearchOutcome:
        """按需搜索。搜不到或不该搜时返回 performed=False，调用方照常生成。"""
        out = SearchOutcome()

        if not self.enabled:
            out.reason = "搜索未启用或未配置"
            return out

        need = evaluate(text, threshold=self.cfg.need_threshold, has_url=has_url,
                        strip_words=strip_words)
        self._last_need = need

        # 被点名问了，阈值放宽一点 —— 人家专门问的，含糊回答很减分
        if not need.needed and is_mention and self.cfg.search_when_mention:
            relaxed = evaluate(text, threshold=max(0.15, self.cfg.need_threshold * 0.5),
                               has_url=has_url, strip_words=strip_words)
            if relaxed.needed:
                need = relaxed
                need.reasons.append("被点名追问")

        if not need.needed:
            self.skipped += 1
            out.reason = f"不需要搜（{'、'.join(need.reasons) or '无信号'}）"
            log.debug("不搜索: %s", out.reason)
            return out

        query = need.query or build_query(text, hint, strip_words=strip_words)
        out.query = query
        out.reason = "、".join(need.reasons)

        results = await self.client.search(query)
        if not results:
            out.reason = f"{out.reason} -> 无结果或失败"
            return out

        # 外部内容一律过注入擦除 —— 网页里也可能藏着越狱文本
        cleaned: list[SearchResult] = []
        for item in results:
            title, h1 = injection.scrub(item.title)
            snippet, h2 = injection.scrub(item.snippet)
            if h1 or h2:
                self.scrubbed += 1
                out.scrubbed += 1
                log.warning("搜索结果中擦除疑似注入：%s", item.url or item.title[:40])
            cleaned.append(SearchResult(title=title, url=item.url, snippet=snippet))

        out.performed = True
        out.results = cleaned
        self.searches += 1
        return out

    def stats(self) -> dict:
        base = self.client.stats()
        base.update({
            "search_performed": self.searches,
            "search_skipped": self.skipped,
            "search_scrubbed": self.scrubbed,
        })
        if self._last_need is not None:
            base["search_last_score"] = round(self._last_need.score, 2)
        return base
