"""网络搜索客户端。

支持几家常见供应商，每种一个适配器（请求体形状和响应结构都不一样）。
另外有个 `generic` 兜底：用宽容的解析去猜常见字段路径，
这样接一个没预设过的接口时不用改代码。

**搜索结果是不可信的外部内容** —— 网页里完全可以写"忽略你之前的指令"。
所以结果在进 prompt 之前会过一遍注入擦除（见 search/__init__.py 的调用点）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from ..config import SearchConfig

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400

#: 各供应商的默认地址
DEFAULT_BASE: dict[str, str] = {
    # 注意是 .com 不是 .cn —— 见 PROVIDER_HINTS
    "ollama": "https://ollama.com/api/web_search",
    "tavily": "https://api.tavily.com/search",
    "bocha": "https://api.bochaai.com/v1/web-search",
    "serper": "https://google.serper.dev/search",
    "searxng": "",           # 自建，必须自己填
    "generic": "",
}

#: 常见踩坑提示。测试失败时一并返回，省得用户对着 404 干瞪眼。
PROVIDER_HINTS: dict[str, str] = {
    "ollama": (
        "Ollama 的文档在 ollama.ac.cn（中文镜像），但 **API 在 ollama.com**。"
        "镜像站的 /api/* 对 POST 一律返回 404，所以 base_url 必须用 "
        "https://ollama.com/api/web_search"
    ),
    "tavily": "Tavily 的 key 形如 tvly-xxx，接口是 https://api.tavily.com/search",
    "bocha": "博查的 key 在 open.bochaai.com 申请，接口是 https://api.bochaai.com/v1/web-search",
    "serper": "serper 用的是 X-API-KEY 请求头（不是 Bearer），key 在 serper.dev 申请",
    "searxng": "searxng 是自建服务，必须在 base_url 填你自己的地址，并在 settings.yml 里开 json 格式",
}

#: 各家对 max_results 的硬上限
MAX_RESULTS_CAP: dict[str, int] = {"ollama": 10}


def host_warning(provider: str, base_url: str) -> str:
    """配置层面的已知陷阱 —— 在请求之前就能拦下来。"""
    url = (base_url or "").lower()
    if provider == "ollama" or "ollama" in url:
        if "ollama.ac.cn" in url:
            return ("base_url 指向的是文档镜像站 ollama.ac.cn —— "
                    "它的 /api/* 对 POST 全部返回 404。请改成 "
                    "https://ollama.com/api/web_search")
    return ""


@dataclass(slots=True)
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""

    def render(self, max_chars: int = 300) -> str:
        text = (self.snippet or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        parts = [f"· {self.title or '(无标题)'}"]
        if text:
            parts.append(f"  {text}")
        if self.url:
            parts.append(f"  来源: {self.url}")
        return "\n".join(parts)


class SearchError(RuntimeError):
    pass


def _first_str(data: dict, *paths: str) -> str:
    """按路径列表取第一个非空字符串。路径形如 'a.b.c'。"""
    for path in paths:
        cur: Any = data
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
            else:
                cur = None
                break
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    return ""


def parse_results(data: Any, limit: int) -> list[SearchResult]:
    """宽容地从各家响应里抠出结果列表。

    试过的路径覆盖：tavily / bocha / serper / searxng / 一些自建代理。
    """
    if not isinstance(data, dict):
        return []

    # 找到主数组
    candidates: list[list] = []
    for path in ("results", "data.webPages.value", "data.results",
                 "organic", "items", "data.items", "webPages.value"):
        cur: Any = data
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, list):
            candidates.append(cur)

    rows = candidates[0] if candidates else []
    out: list[SearchResult] = []
    for row in rows[: limit * 3]:
        if not isinstance(row, dict):
            continue
        item = SearchResult(
            title=_first_str(row, "title", "name", "heading"),
            url=_first_str(row, "url", "link", "href", "sourceUrl"),
            snippet=_first_str(row, "content", "snippet", "summary",
                               "description", "text", "abstract"),
        )
        if item.title or item.snippet:
            out.append(item)
        if len(out) >= limit:
            break
    return out


class SearchClient:
    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        # 预算是"防跑飞"的护栏，不追求精确，所以放内存即可
        self._hour_bucket = 0
        self._hour_used = 0
        self._day_bucket = 0
        self._day_used = 0
        self.calls = 0
        self.failures = 0
        self.blocked = 0

    # ---- 生命周期 -----------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    @property
    def endpoint(self) -> str:
        return self.cfg.base_url or DEFAULT_BASE.get(self.cfg.provider, "")

    # ---- 预算 ---------------------------------------------------------

    def allow(self, now: int | None = None) -> tuple[bool, str]:
        now = int(now or time.time())
        hour, day = now // HOUR, now // DAY
        if hour != self._hour_bucket:
            self._hour_bucket, self._hour_used = hour, 0
        if day != self._day_bucket:
            self._day_bucket, self._day_used = day, 0

        if self.cfg.hourly_budget > 0 and self._hour_used >= self.cfg.hourly_budget:
            return False, f"本小时搜索额度已用完({self._hour_used}/{self.cfg.hourly_budget})"
        if self.cfg.daily_budget > 0 and self._day_used >= self.cfg.daily_budget:
            return False, f"今日搜索额度已用完({self._day_used}/{self.cfg.daily_budget})"
        return True, ""

    def _spend(self) -> None:
        self._hour_used += 1
        self._day_used += 1

    # ---- 请求 ---------------------------------------------------------

    def _build_request(self, query: str) -> tuple[str, dict, dict, dict | None]:
        """按供应商拼 (method, url, headers, json_body)。"""
        cfg = self.cfg
        url = self.endpoint
        limit = max(1, cfg.max_results)

        if cfg.provider == "ollama":
            # Ollama 的网页搜索：只要 query 和 max_results，上限 10
            headers = {"Authorization": f"Bearer {cfg.api_key}",
                       "Content-Type": "application/json"}
            body = {"query": query, "max_results": min(limit, 10)}
            return "POST", url, headers, body

        if cfg.provider == "tavily":
            headers = {"Authorization": f"Bearer {cfg.api_key}",
                       "Content-Type": "application/json"}
            body = {"api_key": cfg.api_key, "query": query,
                    "max_results": limit, "search_depth": "basic"}
            return "POST", url, headers, body

        if cfg.provider == "bocha":
            headers = {"Authorization": f"Bearer {cfg.api_key}",
                       "Content-Type": "application/json"}
            body = {"query": query, "count": limit, "summary": True}
            return "POST", url, headers, body

        if cfg.provider == "serper":
            headers = {"X-API-KEY": cfg.api_key, "Content-Type": "application/json"}
            body = {"q": query, "num": limit}
            return "POST", url, headers, body

        if cfg.provider == "searxng":
            headers = {"Accept": "application/json"}
            sep = "&" if "?" in url else "?"
            full = f"{url}{sep}q={aiohttp.helpers.quote(query)}&format=json"
            return "GET", full, headers, None

        # generic：POST + Bearer，字段名用最常见的
        headers = {"Authorization": f"Bearer {cfg.api_key}",
                   "Content-Type": "application/json"}
        body = {"query": query, "max_results": limit, "count": limit}
        return "POST", url, headers, body

    async def search(self, query: str) -> list[SearchResult]:
        """搜一次。任何失败都返回空列表 —— 搜不到不该影响回复。"""
        if not self.cfg.enabled or not self.cfg.api_key:
            return []
        if not self.endpoint:
            log.warning("搜索未配置 base_url（provider=%s）", self.cfg.provider)
            return []

        ok, reason = self.allow()
        if not ok:
            self.blocked += 1
            log.info("跳过搜索：%s", reason)
            return []

        warn = host_warning(self.cfg.provider, self.endpoint)
        if warn:
            log.warning("搜索配置有问题：%s", warn)

        method, url, headers, body = self._build_request(query)
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout)
        started = time.monotonic()

        try:
            if method == "GET":
                ctx = session.get(url, headers=headers, timeout=timeout)
            else:
                ctx = session.post(url, headers=headers, json=body, timeout=timeout)
            async with ctx as resp:
                text = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            log.warning("搜索失败（%s）: %s", self.cfg.provider, exc)
            return []

        self.calls += 1
        self._spend()

        if status != 200:
            self.failures += 1
            log.warning("搜索返回 HTTP %s: %s", status, text[:200])
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self.failures += 1
            log.warning("搜索响应不是 JSON: %s", text[:200])
            return []

        results = parse_results(data, max(1, self.cfg.max_results))
        log.info("搜索「%s」-> %d 条结果（%dms）",
                 query[:40], len(results), int((time.monotonic() - started) * 1000))
        return results

    # ---- 观测 ---------------------------------------------------------

    def stats(self) -> dict:
        return {
            "search_calls": self.calls,
            "search_failures": self.failures,
            "search_blocked": self.blocked,
            "search_used_hour": self._hour_used,
            "search_used_day": self._day_used,
        }
