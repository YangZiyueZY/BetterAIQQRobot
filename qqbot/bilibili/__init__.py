"""B 站视频理解服务：链接/搜索 -> 可读文本。

## 视频解析工作流（固定这个顺序，别绕过去）

```
链接/关键词
   │
   ├─① resolve   b23.tv 短链 / 裸 BV 号 -> bvid
   ├─② view      -> aid / cid / title / pages（**每 P 的时长要留着**）
   │
   ├─③ 字幕优先  player/v2 -> 字幕轨列表 -> 下载 JSON -> 拼文本
   │             命中 => source="subtitle"，结束（最快、最准、不花钱）
   │
   └─ 无字幕才走 ASR 兜底
                 取音轨直链（最高码率）-> 下载 m4s
                 -> ffmpeg 转 16k/16bit/mono wav
                 -> StepFun ASR（通道由 transport=auto 决定：
                    step_plan 走 /audio/asr/sse，/v1 走 /audio/transcriptions）
                 命中 => source="asr"
```

三条硬规则：

1. **字幕永远优先**。AI 字幕有音近错字（实测 VENUS斯→VERS、质朴GLM→智谱GLM），
   但它不需要下载音轨，快几个数量级；ASR 只是"没字幕"时的兜底，不是默认路径。
2. **长度守卫用 `info.media_duration`（当前 cid 那一段），不是 `info.duration`。**
   多 P 视频的顶层 duration 是各 P 之和 —— 实测《原神》沃雅妮莎 PV 顶层 713s、
   但实际只转 P1 的 176s，用总时长比 `max_asr_seconds=600` 会把它误杀。
3. **多 P 只转写 cid 指向的那一段（即 P1）**，并在结果里标出来，
   免得 LLM 以为拿到了全片。要覆盖所有 P 得单独设计（每 P 一次 ASR + 拼接），
   现在不做：PV 的 P2~P4 是日/英/韩同内容，转了纯属浪费。

字幕/转写文本都是**外部内容**，进 prompt 前要过注入擦除（由调用方做）。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from ..asr.client import StepFunASR, to_wav
from ..config import BilibiliConfig
from .api import BiliClient, BiliError, SubtitleTrack, VideoHit, VideoInfo, extract_links

log = logging.getLogger(__name__)

#: 提取结果的缓存时长。字幕内容不会变，缓存久一点没关系。
CACHE_TTL = 3600.0

#: 字幕正文最多取多少字进 prompt
DEFAULT_MAX_CHARS = 3000


@dataclass(slots=True)
class VideoContent:
    ok: bool = False
    info: VideoInfo | None = None
    text: str = ""
    source: str = ""          # subtitle / asr / none
    track_label: str = ""
    from_cache: bool = False
    error: str = ""
    elapsed_ms: int = 0

    def render(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """渲染成给 LLM 看的块。"""
        if not self.ok or not self.text:
            return ""
        info = self.info
        lines = []
        if info:
            bits = [f"《{info.title}》"]
            if info.author:
                bits.append(f"UP: {info.author}")
            if info.duration:
                bits.append(f"{info.duration // 60}分{info.duration % 60}秒")
            lines.append("".join(bits))
            if info.url:
                lines.append(f"链接：{info.url}")
        # track_label 里带着"仅 P1/共 N 段"这类信息，别在渲染时丢掉
        tag = {"subtitle": self.track_label or "字幕",
               "asr": self.track_label or "音频转写"}.get(self.source, self.source)
        lines.append(f"（内容来自{tag}，是机器识别的，可能有同音错字）")
        body = self.text[:max_chars]
        if len(self.text) > max_chars:
            body += "……（后略）"
        lines.append(body)
        return "\n".join(lines)


@dataclass(slots=True)
class SearchOutcome:
    ok: bool = False
    keyword: str = ""
    hits: list[VideoHit] = field(default_factory=list)
    error: str = ""

    def render(self, limit: int = 4) -> str:
        if not self.hits:
            return ""
        return "\n".join(h.render() for h in self.hits[:limit])


class BiliService:
    def __init__(self, cfg: BilibiliConfig, client: BiliClient,
                 asr: StepFunASR | None = None) -> None:
        self.cfg = cfg
        self.client = client
        self.asr = asr
        self._cache: dict[str, tuple[float, VideoContent]] = {}
        self.fetched = 0
        self.asr_fallbacks = 0
        self.cache_hits = 0

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    # ---- 链接 ---------------------------------------------------------

    def links_in(self, text: str) -> list[str]:
        if not self.enabled or not self.cfg.enabled_link:
            return []
        return extract_links(text)[: self.cfg.max_links_per_message]

    async def fetch(self, link_or_bvid: str) -> VideoContent:
        """拿一个视频的可读内容（字幕优先，ASR 兜底）。"""
        started = time.monotonic()

        try:
            bvid = await self.client.resolve(link_or_bvid)
        except BiliError as exc:
            return VideoContent(ok=False, error=f"解析链接失败: {exc}")

        if not bvid:
            return VideoContent(ok=False, error="没解析出 BV 号")

        cached = self._cache.get(bvid)
        if cached and (time.time() - cached[0]) < CACHE_TTL:
            self.cache_hits += 1
            content = cached[1]
            content.from_cache = True
            return content

        try:
            info = await self.client.view(bvid)
        except BiliError as exc:
            return VideoContent(ok=False, error=f"取视频信息失败: {exc}")

        content = VideoContent(ok=False, info=info)

        # ① 字幕优先
        text, track = await self._try_subtitle(info)
        if text:
            content.ok = True
            content.text = text
            content.source = "subtitle"
            content.track_label = track.label() if track else "字幕"
            content.elapsed_ms = int((time.monotonic() - started) * 1000)
            self.fetched += 1
            self._cache[bvid] = (time.time(), content)
            log.info("B站字幕《%s》-> %d 字（%s）",
                     info.title[:30], len(text), content.track_label)
            return content

        # ② 无字幕 -> 音频 ASR 兜底
        if self.cfg.asr_fallback and self.asr is not None and self.asr.configured:
            text, err = await self._try_asr(info)
            if text:
                content.ok = True
                content.text = text
                content.source = "asr"
                # 多 P 时必须标清楚只转了一段，否则 LLM 会以为拿到了全片
                content.track_label = ("音频转写" + (
                    f"（仅 P{int(info.pages[0].get('page') or 1)}/共 {len(info.pages)} 段）"
                    if info.has_multi_part else ""))
                content.elapsed_ms = int((time.monotonic() - started) * 1000)
                self.fetched += 1
                self.asr_fallbacks += 1
                self._cache[bvid] = (time.time(), content)
                log.info("B站音频转写《%s》-> %d 字（%s）",
                         info.title[:30], len(text), content.track_label)
                return content
            content.error = f"无字幕，音频转写也没成功：{err}"
        else:
            content.error = ("该视频没有字幕"
                             + ("（音频转写未启用）" if not self.cfg.asr_fallback
                                else "（ASR 未配置）"))

        content.elapsed_ms = int((time.monotonic() - started) * 1000)
        return content

    async def _try_subtitle(self, info: VideoInfo) -> tuple[str, SubtitleTrack | None]:
        try:
            tracks = await self.client.subtitles(info.aid, info.cid, info.bvid)
        except BiliError as exc:
            log.debug("取字幕列表失败: %s", exc)
            return "", None

        track = self.client.pick_track(tracks, prefer=self.cfg.prefer_lan)
        if track is None:
            return "", None

        try:
            rows = await self.client.download_subtitle(track)
        except BiliError as exc:
            log.debug("下载字幕失败: %s", exc)
            return "", None

        text = "".join(content for _, content in rows).strip()
        if self.cfg.dedupe_repeat:
            text = _collapse_repeats(text)
        return text, track

    async def _try_asr(self, info: VideoInfo) -> tuple[str, str]:
        """音频兜底：取音轨 -> 下载 -> ffmpeg 转 wav -> ASR。

        只转写 `info.cid` 指向的那一段（多 P 时就是 P1），
        所以长度判断也必须用那一段的时长，见 `VideoInfo.media_duration`。
        """
        import tempfile
        from pathlib import Path

        dur = info.media_duration
        if dur and dur > self.cfg.max_asr_seconds:
            # 只报当前这段的时长 —— 报全片总时长会误导人以为放宽上限没用
            part = f"P{int(info.pages[0].get('page') or 1)} " if info.has_multi_part else ""
            return "", (f"音频太长（{part}这一段 {dur}s > "
                        f"{self.cfg.max_asr_seconds}s）")

        try:
            url = await self.client.audio_url(info.aid, info.cid, info.bvid)
        except BiliError as exc:
            return "", f"取音轨失败: {exc}"

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "audio.m4s"
            wav = Path(tmp) / "audio.wav"
            try:
                size = await self.client.download(url, raw)
            except BiliError as exc:
                return "", f"下载音轨失败: {exc}"
            if size <= 0:
                return "", "音轨为空"
            log.debug("音轨下载完成 %.2fMB", size / 1048576)

            if not await __import__("asyncio").to_thread(to_wav, raw, wav):
                return "", "ffmpeg 转码失败"

            result = await self.asr.transcribe_file(wav)
            if result.ok:
                return result.text, ""
            return "", result.error

    # ---- 搜索 ---------------------------------------------------------

    async def search(self, keyword: str, *, limit: int | None = None,
                     order: str = "totalrank") -> SearchOutcome:
        if not self.enabled or not self.cfg.enabled_search:
            return SearchOutcome(ok=False, keyword=keyword, error="B站搜索未启用")
        keyword = (keyword or "").strip()
        if not keyword:
            return SearchOutcome(ok=False, error="关键词为空")

        try:
            hits = await self.client.search(
                keyword, limit=limit or self.cfg.search_results, order=order
            )
        except BiliError as exc:
            return SearchOutcome(ok=False, keyword=keyword, error=str(exc))

        return SearchOutcome(ok=bool(hits), keyword=keyword, hits=hits)

    # ---- 观测 ---------------------------------------------------------

    def stats(self) -> dict:
        base = self.client.stats()
        base.update({
            "bili_fetched": self.fetched,
            "bili_cache_hits": self.cache_hits,
            "bili_asr_fallbacks": self.asr_fallbacks,
            "bili_cached": len(self._cache),
        })
        return base


#: 字幕里常见的整句重复（B 站滚动字幕有时会重复推送同一句）
def _collapse_repeats(text: str) -> str:
    if not text:
        return text
    # 连续重复的两字以上片段，压成一次
    return re.sub(r"(.{4,20})\1+", r"\1", text)
