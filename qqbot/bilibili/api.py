"""B 站 API 客户端。

要点（全部来自实测）：
  · **AI 字幕必须登录态**：游客态 `subtitles: []` + `need_login_subtitle: true`，
    连上 buvid3/buvid4 也没用，不存在免登录取巧路径
  · `subtitle_url` 以 `//` 开头，要补 `https:`
  · 下载字幕/音频**必须带 Referer**，否则 403
  · `auth_key` 约 1 小时失效 —— 所以不能长期缓存 subtitle_url，每次现取
  · 搜索**必须带 buvid3 或 Cookie**，裸请求 412 风控；Referer 用 search.bilibili.com
  · 搜索结果的 title 含 `<em class="keyword">` 高亮标签，要清洗
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

import aiohttp

from .wbi import WbiKeys

log = logging.getLogger(__name__)

API = "https://api.bilibili.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

#: 从文本里抠 B 站链接。短链、BV、av 三种形态。
LINK_RE = re.compile(
    r"(?:https?://)?(?:b23\.tv/[A-Za-z0-9]+"
    r"|(?:www\.)?bilibili\.com/video/(?:BV[A-Za-z0-9]{8,12}|av\d+)"
    r"|(?:www\.)?bilibili\.com/(?:bangumi/play/)?(?:ep|ss)\d+"
    r"|BV[A-Za-z0-9]{8,12})",
    re.IGNORECASE,
)

BV_RE = re.compile(r"(BV[A-Za-z0-9]{8,12})", re.IGNORECASE)
AV_RE = re.compile(r"av(\d+)", re.IGNORECASE)

EM_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(slots=True)
class VideoInfo:
    bvid: str = ""
    aid: int = 0
    cid: int = 0
    title: str = ""
    author: str = ""
    duration: int = 0
    desc: str = ""
    pubdate: int = 0
    view: int = 0
    pages: list[dict] = field(default_factory=list)
    url: str = ""

    @property
    def has_multi_part(self) -> bool:
        return len(self.pages) > 1

    @property
    def media_duration(self) -> int:
        """**当前 cid 那一段**的实际时长（秒）—— 长度守卫必须用这个。

        坑：多 P 视频的顶层 `duration` 是**所有分P之和**。
        实测《原神》沃雅妮莎角色PV（BV13Xew6LE7T）`videos=4`、顶层 `duration=713`，
        但 `cid` 指向的 P1 只有 176s，音轨下载下来的也只有 176s。
        拿 713 去比 `max_asr_seconds=600`，会把一个实际只需转 176s 的视频
        直接误杀成"视频太长"。
        """
        for p in self.pages:
            if int(p.get("cid") or 0) == int(self.cid or 0):
                return int(p.get("duration") or 0) or self.duration
        return self.duration


@dataclass(slots=True)
class VideoHit:
    bvid: str = ""
    title: str = ""
    author: str = ""
    duration: str = ""
    play: int = 0
    pubdate: int = 0
    description: str = ""

    def render(self) -> str:
        bits = [f"· {self.title}"]
        meta = []
        if self.author:
            meta.append(f"UP: {self.author}")
        if self.duration:
            meta.append(self.duration)
        if self.play:
            meta.append(f"{self.play} 播放")
        if meta:
            bits.append("  " + " · ".join(meta))
        if self.bvid:
            bits.append(f"  https://www.bilibili.com/video/{self.bvid}")
        return "\n".join(bits)


@dataclass(slots=True)
class SubtitleTrack:
    lan: str = ""
    lan_doc: str = ""
    url: str = ""
    kind: str = "ai"     # ai / cc
    ai_status: int = 0

    @property
    def is_ai(self) -> bool:
        return self.kind == "ai"

    def label(self) -> str:
        return f"{self.lan_doc or self.lan}（{'AI' if self.is_ai else 'CC'} 字幕）"


class BiliError(RuntimeError):
    pass


def extract_links(text: str) -> list[str]:
    """从一段文本里取出所有 B 站链接 / BV 号，保持出现顺序、去重。"""
    out: list[str] = []
    for match in LINK_RE.finditer(text or ""):
        link = match.group(0).strip()
        if link not in out:
            out.append(link)
    return out


def normalize_link(link: str) -> str:
    if link.lower().startswith(("http://", "https://")):
        return link
    if link.lower().startswith("b23.tv"):
        return "https://" + link
    if link.lower().startswith("bv") or link.lower().startswith("av"):
        return f"https://www.bilibili.com/video/{link}"
    return "https://" + link


class BiliClient:
    def __init__(self, *, cookie: str = "", buvid3: str = "",
                 buvid4: str = "", timeout: float = 20.0) -> None:
        self.cookie = cookie
        self.buvid3 = buvid3
        self.buvid4 = buvid4
        self.timeout = timeout
        self.wbi = WbiKeys()
        self._session: aiohttp.ClientSession | None = None
        self.calls = 0
        self.failures = 0
        self.subtitle_hits = 0
        #: 拿到 need_login_subtitle 或字幕空数组时置位 —— 面板据此提示"该扫码了"
        self.auth_suspect = False
        self.auth_suspect_ts = 0.0
        self.last_auth_error = ""

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

    def _headers(self, referer: str = "https://www.bilibili.com/") -> dict:
        h = {"User-Agent": UA, "Referer": referer}
        # 没有登录 Cookie 时，用游客指纹兜底（buvid3 + buvid4 一起才稳）
        guest = "; ".join(
            f"{k}={v}" for k, v in (("buvid3", self.buvid3), ("buvid4", self.buvid4)) if v
        )
        cookie = self.cookie or guest
        if cookie:
            h["Cookie"] = cookie
        return h

    async def _get_json(self, url: str, *, referer: str | None = None,
                        timeout: float | None = None) -> dict:
        session = await self._get_session()
        t = aiohttp.ClientTimeout(total=timeout or self.timeout)
        try:
            async with session.get(url, headers=self._headers(referer or ""),
                                   timeout=t) as resp:
                text = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            raise BiliError(f"请求失败: {exc}") from exc

        self.calls += 1
        if status != 200:
            self.failures += 1
            raise BiliError(f"HTTP {status}: {text[:150]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self.failures += 1
            raise BiliError(f"响应不是 JSON: {text[:150]}") from exc

        if data.get("code") not in (0, None):
            raise BiliError(f"接口返回 code={data.get('code')} {data.get('message')}")
        return data.get("data") or {}

    # ---- wbi ----------------------------------------------------------

    async def _ensure_wbi(self) -> None:
        if self.wbi.valid:
            return
        try:
            data = await self._get_json(f"{API}/x/web-interface/nav")
            img = ((data.get("wbi_img") or {}).get("img_url")) or ""
            sub = ((data.get("wbi_img") or {}).get("sub_url")) or ""
            if img and sub:
                self.wbi.update(img, sub)
        except BiliError as exc:
            log.debug("取 wbi key 失败（非致命，会回退到无签名接口）: %s", exc)

    # ---- 解析链接 -----------------------------------------------------

    async def resolve(self, link: str) -> str:
        """短链 / 完整链接 / BV / av 号 -> BV 号。"""
        link = normalize_link(link)

        m = BV_RE.search(link)
        if m:
            return m.group(1)

        m = AV_RE.search(link)
        if m:
            data = await self._get_json(f"{API}/x/web-interface/view?aid={m.group(1)}")
            return str(data.get("bvid") or "")

        # b23.tv 短链：跟随重定向拿 Location
        session = await self._get_session()
        try:
            async with session.get(link, headers=self._headers(),
                                   allow_redirects=False,
                                   timeout=aiohttp.ClientTimeout(total=self.timeout)) as r:
                location = r.headers.get("Location") or ""
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            raise BiliError(f"短链解析失败: {exc}") from exc

        m = BV_RE.search(location)
        if m:
            return m.group(1)
        raise BiliError(f"短链里没找到 BV 号: {location[:120]}")

    # ---- 视频信息 -----------------------------------------------------

    async def view(self, bvid: str) -> VideoInfo:
        data = await self._get_json(f"{API}/x/web-interface/view?bvid={bvid}")
        # 每个分P的时长必须存下来：顶层 duration 是各 P 之和，
        # 拿它做长度判断会误杀多 P 视频（见 VideoInfo.media_duration）
        pages = [{"cid": p.get("cid"), "part": p.get("part"),
                  "page": int(p.get("page") or 0),
                  "duration": int(p.get("duration") or 0)}
                 for p in (data.get("pages") or [])]
        return VideoInfo(
            bvid=str(data.get("bvid") or bvid),
            aid=int(data.get("aid") or 0),
            cid=int(data.get("cid") or 0),
            title=str(data.get("title") or ""),
            author=str(((data.get("owner") or {}).get("name")) or ""),
            duration=int(data.get("duration") or 0),
            desc=str(data.get("desc") or "")[:400],
            pubdate=int(data.get("pubdate") or 0),
            view=int(((data.get("stat") or {}).get("view")) or 0),
            pages=pages,
            url=f"https://www.bilibili.com/video/{data.get('bvid') or bvid}",
        )

    # ---- 字幕 ---------------------------------------------------------

    async def subtitles(self, aid: int, cid: int, bvid: str = "") -> list[SubtitleTrack]:
        """取字幕列表。

        `x/player/wbi/v2` 在部分视频会要求签名（-403），所以失败就回退到
        `x/player/v2`（结构完全一样）。
        """
        referer = f"https://www.bilibili.com/video/{bvid}/" if bvid else ""

        async def fetch(url: str) -> dict:
            return await self._get_json(url, referer=referer)

        data: dict = {}
        try:
            await self._ensure_wbi()
            if self.wbi.valid:
                query = self.wbi.sign({"aid": aid, "cid": cid})
                data = await fetch(f"{API}/x/player/wbi/v2?{query}")
            else:
                raise BiliError("no wbi key")
        except BiliError as exc:
            log.debug("wbi/v2 失败，回退无签名接口: %s", exc)

        if not data:
            data = await fetch(f"{API}/x/player/v2?aid={aid}&cid={cid}")

        if data.get("need_login_subtitle") and not (data.get("subtitle") or {}).get("subtitles"):
            # 这是"Cookie 掉了"最可靠的信号：不是这个视频没字幕，
            # 而是接口明确在说"你得登录"。面板据此提示扫码。
            self.auth_suspect = True
            self.auth_suspect_ts = time.time()
            self.last_auth_error = "AI 字幕需要登录态，Cookie 可能已失效"
            log.warning("B站 Cookie 可能已失效：接口返回 need_login_subtitle")

        tracks: list[SubtitleTrack] = []
        for item in ((data.get("subtitle") or {}).get("subtitles") or []):
            url = str(item.get("subtitle_url") or "")
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("http://"):
                url = "https://" + url[len("http://"):]
            stype = int(item.get("type") or 0)
            tracks.append(SubtitleTrack(
                lan=str(item.get("lan") or ""),
                lan_doc=str(item.get("lan_doc") or ""),
                url=url,
                kind="ai" if (stype == 1 or str(item.get("lan", "")).startswith("ai-")) else "cc",
                ai_status=int(item.get("ai_status") or 0),
            ))
        return [t for t in tracks if t.url]

    def pick_track(self, tracks: list[SubtitleTrack],
                   prefer: str = "zh") -> SubtitleTrack | None:
        """优先中文、优先人工 CC（断句准、错字少）。"""
        if not tracks:
            return None

        def score(t: SubtitleTrack) -> tuple:
            chinese = t.lan.startswith("zh") or t.lan.startswith("ai-zh")
            return (not chinese, t.is_ai)

        return sorted(tracks, key=score)[0]

    async def download_subtitle(self, track: SubtitleTrack) -> list[tuple[float, str]]:
        """下载字幕内容，返回 [(起始秒, 文本)]。"""
        session = await self._get_session()
        try:
            async with session.get(
                track.url,
                headers=self._headers("https://www.bilibili.com/"),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                text = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            raise BiliError(f"字幕下载失败: {exc}") from exc

        if status != 200:
            self.failures += 1
            raise BiliError(f"字幕下载 HTTP {status}")

        try:
            body = json.loads(text).get("body") or []
        except json.JSONDecodeError as exc:
            raise BiliError("字幕内容不是 JSON") from exc

        self.subtitle_hits += 1
        return [(float(row.get("from") or 0), str(row.get("content") or ""))
                for row in body if row.get("content")]

    # ---- 音频流 -------------------------------------------------------

    async def audio_url(self, aid: int, cid: int, bvid: str = "") -> str:
        """取最高码率的音频直链。

        注意用 `/x/player/playurl` 而**不是** wbi 版 —— 后者实测返回 HTML。
        另外 DASH 下档位与 qn 无关，挑流只看 bandwidth。
        """
        url = (f"{API}/x/player/playurl?avid={aid}&cid={cid}&fnval=16&fourk=1")
        data = await self._get_json(
            url, referer=f"https://www.bilibili.com/video/{bvid}/" if bvid else ""
        )
        tracks = ((data.get("dash") or {}).get("audio")) or []
        if not tracks:
            raise BiliError("没有可用的音频流")
        best = max(tracks, key=lambda x: int(x.get("bandwidth") or 0))
        return str(best.get("baseUrl") or best.get("base_url") or "")

    async def download(self, url: str, dest) -> int:
        """下载（音频/其它），必须带 Referer，否则 403。"""
        session = await self._get_session()
        try:
            async with session.get(
                url, headers=self._headers("https://www.bilibili.com/"),
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            ) as resp:
                if resp.status != 200:
                    raise BiliError(f"下载 HTTP {resp.status}")
                size = 0
                with open(dest, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        fh.write(chunk)
                        size += len(chunk)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            raise BiliError(f"下载失败: {exc}") from exc
        return size

    # ---- 搜索 ---------------------------------------------------------

    async def search(self, keyword: str, *, limit: int = 5,
                     order: str = "totalrank") -> list[VideoHit]:
        """关键词搜视频。必须带 buvid3 或 Cookie，否则 412 风控。"""
        import urllib.parse

        params = {"search_type": "video", "keyword": keyword, "page": 1,
                  "page_size": max(1, min(limit, 50)), "order": order}

        await self._ensure_wbi()
        if self.wbi.valid:
            url = f"{API}/x/web-interface/wbi/search/type?{self.wbi.sign(params)}"
        else:
            url = (f"{API}/x/web-interface/search/type?"
                   + urllib.parse.urlencode({**params, "keyword": keyword}))

        data = await self._get_json(url, referer="https://search.bilibili.com/")

        hits: list[VideoHit] = []
        for row in (data.get("result") or [])[:limit]:
            if not isinstance(row, dict):
                continue
            hits.append(VideoHit(
                bvid=str(row.get("bvid") or ""),
                # title 里带 <em class="keyword"> 高亮标签，必须清掉
                title=EM_TAG_RE.sub("", str(row.get("title") or "")).strip(),
                author=str(row.get("author") or ""),
                duration=str(row.get("duration") or ""),
                play=int(row.get("play") or 0),
                pubdate=int(row.get("pubdate") or 0),
                description=EM_TAG_RE.sub("", str(row.get("description") or ""))[:200],
            ))
        return [h for h in hits if h.bvid]

    # ---- 观测 ---------------------------------------------------------

    def set_cookie(self, cookie: str) -> None:
        """换新 Cookie。扫码登录成功后热替换，不用重启进程。"""
        self.cookie = cookie
        self.auth_suspect = False
        self.auth_suspect_ts = 0.0
        self.last_auth_error = ""
        log.info("B站 Cookie 已更新")

    def auth_state(self) -> dict:
        return {
            "auth_suspect": self.auth_suspect,
            "auth_suspect_age": (int(time.time() - self.auth_suspect_ts)
                                 if self.auth_suspect_ts else 0),
            "auth_error": self.last_auth_error,
            "has_cookie": bool(self.cookie),
            "has_sessdata": "SESSDATA=" in (self.cookie or ""),
        }

    def stats(self) -> dict:
        return {
            "bili_calls": self.calls,
            "bili_failures": self.failures,
            "bili_subtitle_hits": self.subtitle_hits,
        }
