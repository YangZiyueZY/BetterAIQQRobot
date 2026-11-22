"""音频转文字：StepFun ASR。

三条路，能力**不一样**（这里栽过一次，见下）：

  `/audio/asr/sse`             **JSON + base64 音频，SSE 流式返回** ← step_plan 只有这条
      · 请求体是 JSON：`audio.data` 放 base64，`audio.input` 放识别与格式配置
      · 响应是 `transcript.text.delta` / `transcript.text.done` 事件流
      · 支持 ogg / mp3 / wav / pcm / m4a
      · `/v1` 和 `/step_plan/v1` 两种前缀都能用

  `/audio/transcriptions`      **同步 + multipart 文件上传**（OpenAI 兼容形状）
      · 一次性返回全文，最简；上传限制 mp3 / pcm / ogg / wav，< 100MB
      · ⚠️ **step_plan 前缀下这条路由根本没注册**，网关直接回 `404 page not found`

  `/audio/asr/file/submit`     异步任务 + `audio.url`
      · 服务端自己去下载，所以 URL 必须**公网可达**
      · 换来分句、字级时间戳、双声道分轨、说话人识别
      · 适合长音频；但我们的音频是本地下载的，除非有公网地址否则用不上

#### ⚠️ 曾经的误判（别再犯）

2026-10 排查 step_plan 时只试了 OpenAI 兼容的 `/step_plan/v1/audio/transcriptions`
和 `/step_plan/v1/audio/asr/file/submit`，两条都是 404，于是写下了
"step_plan 不提供 ASR"的结论 —— **那是错的**。step_plan 的 ASR 走的是
`/step_plan/v1/audio/asr/sse`，请求形状完全不同（JSON + base64，不是 multipart）。
实测 2026-10-10 13:19:41.562 +08:00：同一个 key、同一句音频，
`POST /step_plan/v1/audio/asr/sse` 返回 200 并正确识别出文本。
`GET /step_plan/v1/models` 里也**确实列着** `stepaudio-2.5-asr`。

教训：**404 只代表"这条路径没有"，不代表"这个能力没有"**。
判定一个套餐提不提供某项能力，得先翻它的套餐专属文档，别拿开放平台的接口形状去套。

所以 `transport="auto"` 的规则是：step_plan → 必走 SSE；
其他前缀 → 优先 multipart 上传（大文件友好、一次拿全文），
撞上 404 再自动回退 SSE。

注意：StepFun 只支持中英文，不支持中文方言。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from ..config import AsrConfig

log = logging.getLogger(__name__)

#: 同步接口支持的上传格式
UPLOAD_FORMATS = ("mp3", "pcm", "ogg", "wav")

#: 上传大小上限（官方 <100MB，留点余量）
MAX_UPLOAD_BYTES = 95 * 1024 * 1024

#: SSE 通道的上限。它要把整段音频 base64 塞进一个 JSON body，
#: 膨胀 4/3 之后还是一次性提交，所以比 multipart 那条保守得多。
SSE_MAX_BYTES = 20 * 1024 * 1024

#: 转码目标：16kHz / 16bit / 单声道（官方推荐）
TARGET_RATE = 16000

#: 用户可能把完整 endpoint 粘进 base_url，这里把尾巴剥掉
_ENDPOINT_SUFFIXES = (
    "/chat/completions", "/audio/transcriptions", "/audio/asr/file/submit",
    "/audio/asr/file/query", "/audio/asr/sse", "/audio/speech", "/audio/voices",
    "/completions", "/embeddings", "/models",
)

#: 两条同步路径的名字（transport 取值就用它们）
UPLOAD_PATH = "/audio/transcriptions"
SSE_PATH = "/audio/asr/sse"

#: SSE 接口认的音频容器格式（比 multipart 那条多一个 m4a）
SSE_FORMATS = ("ogg", "mp3", "wav", "pcm", "m4a")

DEFAULT_BASE = "https://api.stepfun.com/v1"


def normalize_base_url(raw: str, default: str = DEFAULT_BASE) -> str:
    """把粘贴的完整 endpoint 归一到 API 根。

    文档里给的可能是 `.../chat/completions` 这种完整地址，
    但 ASR 要的是 `{root}/audio/transcriptions`。
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return default
    for suffix in _ENDPOINT_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    return url or default


def is_step_plan(raw: str) -> bool:
    """base_url 是不是 step_plan 套餐前缀。"""
    url = (raw or "").lower().rstrip("/")
    return "/step_plan/" in url or url.endswith("/step_plan")


def base_url_warning(raw: str) -> str:
    """已知的 base_url 陷阱 —— **提示信息，不是拦截**。

    这里改过一次：原先写的是"step_plan 不提供音频转写，直接拒绝发请求"，
    那是误判（见模块头的踩坑说明）。step_plan 有 ASR，只是**只有 SSE 那一条**：

      /step_plan/v1/audio/asr/sse          -> 200  ✓ 实测可用
      /step_plan/v1/audio/transcriptions   -> 404  （网关没注册这条路由）
      /step_plan/v1/audio/speech           -> 400  （存在，TTS）

    所以代码会自动把 step_plan 的转写改走 SSE，这里只负责把这件事说清楚，
    省得用户对着 multipart 的 404 去查自己的 key。
    """
    if is_step_plan(raw):
        return ("base_url 指向 step_plan 套餐 —— 它的音频转写**只有 SSE 一条路**："
                f"{SSE_PATH}（JSON + base64）。OpenAI 兼容的 {UPLOAD_PATH}"
                " 在这个前缀下网关直接 404，已自动改走 SSE，不是 key 的问题。")
    return ""


@dataclass(slots=True)
class AsrResult:
    ok: bool
    text: str = ""
    duration: float = 0.0
    error: str = ""
    ms: int = 0
    utterances: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.utterances is None:
            self.utterances = []


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=8)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def to_wav(src: Path, dest: Path, *, rate: int = TARGET_RATE) -> bool:
    """转成 ASR 友好的 16kHz / 16bit / 单声道 wav。

    B 站音轨是 m4s（实际是 mp4 容器），必须转码才能上传。
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vn",                     # 丢掉视频轨
        "-ac", "1",                # 单声道
        "-ar", str(rate),          # 采样率
        "-acodec", "pcm_s16le",    # 16bit PCM
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("ffmpeg 执行失败: %s", exc)
        return False
    if proc.returncode != 0:
        log.warning("ffmpeg 转码失败: %s", proc.stderr.decode("utf-8", "ignore")[:200])
        return False
    return dest.is_file() and dest.stat().st_size > 0


class AsrError(RuntimeError):
    pass


def _explain(status: int, body: str) -> str:
    """把 HTTP 错误翻译成能照着做的说明。

    402 是最容易误判的 —— 它说明 key 是**对的**（错的 key 会返回 401），
    只是账户没额度。不写清楚的话会让人一直去查 key。
    """
    if status == 401:
        return "鉴权失败（401）—— STEPFUN_API_KEY 不对或已失效"
    if status == 402:
        return ("账户额度不足（402 quota_exceeded）—— key 本身是有效的"
                "（错误 key 会返回 401），是账户没有余额/未开通，"
                "需要去 platform.stepfun.com 充值")
    if status == 429:
        return "请求被限流（429），稍后再试或降低频率"
    if status == 404:
        # 实测：step_plan 前缀下 multipart 上传那条是网关直出的 `404 page not found`
        # （响应头 X-Router-Id 命中 path:/step_plan/v1/audio/ 但 X-Upstream-Latency=0，
        # 且没有 x-trace-id —— 请求根本没进后端）。这种 404 跟 key 无关，换 SSE 那条即可。
        return ("接口路径不存在（404）—— 这个前缀下没有注册这条路由，"
                "跟 key 无关。step_plan 的音频转写只有 /audio/asr/sse 一条，"
                "把 transport 设为 sse（或 auto）即可")
    return f"HTTP {status}: {body[:200]}"


def _parse_sse_events(raw: str) -> tuple[str, str]:
    """从 SSE 响应文本里抠出最终全文。返回 `(text, error)`。

    不逐行流式读，是因为我们**一次性提交整段音频、只要最终全文**：
    `resp.text()` 会等流关闭，代码简单一档，测试也能直接塞假 session。
    要边说边出字幕的话再改成读 `content` 迭代。

    事件类型（官方）：`transcript.text.delta` / `transcript.text.done` / `error`。
    优先取 done 的 `text`（服务端给的完整结果），没有才把 delta 拼起来。
    """
    deltas: list[str] = []
    final = ""
    err = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue          # event:/id:/心跳注释都不认
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type") or "")
        if et == "transcript.text.done":
            final = str(ev.get("text") or "") or final
        elif et == "transcript.text.delta":
            deltas.append(str(ev.get("delta") or ""))
        elif et == "error":
            err = str(ev.get("message") or "识别出错")
    return (final or "".join(deltas)).strip(), err


def _wav_seconds_and_rate(path: Path) -> tuple[float, int]:
    """返回 (秒, 每秒字节数)。只认带标准头的 wav —— 切段就靠这两个数。"""
    try:
        with wave.open(str(path), "rb") as w:
            bps = w.getframerate() * w.getsampwidth() * w.getnchannels()
            return (w.getnframes() / float(w.getframerate()) if w.getframerate() else 0.0), bps
    except (OSError, wave.Error):
        return 0.0, 0


def split_wav(src: Path, dest_dir: Path, *, max_bytes: int | None = None) -> list[Path]:
    """把过长的 wav 按时间切成多段，每段原始大小 < max_bytes。

    SSE 通道要把**整段**音频 base64 塞进一个 JSON body，所以长音频必须切开；
    不切的话 step_plan 免费通道就只能处理 ~11 分钟（20MB / 32KB每秒）。

    用 `-c copy` 直接按字节切 PCM，不重编码 —— 快且不掉音质。
    边界可能把一句话劈成两半，这是切段方案的固有代价，调用方按顺序拼回去。

    `max_bytes` 默认在**调用时**读模块常量（不是 def 时绑定），测试才好覆盖。
    """
    max_bytes = max_bytes or SSE_MAX_BYTES
    seconds, bps = _wav_seconds_and_rate(src)
    if not seconds or not bps:
        return []
    # 下限只到 1 秒：留 10 秒这种"整数好看"的下限会静默吞掉小的 max_bytes，
    # 结果就是明明说好了上限却切出超限的段（测试里就是这么暴露的）
    max_seconds = max(1, int(max_bytes * 0.9 / bps))
    if seconds <= max_seconds:
        return [src]

    parts: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)   # ffmpeg 不会自己建目录
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-f", "segment", "-segment_time", str(max_seconds),
         "-reset_timestamps", "1", "-c", "copy",
         str(dest_dir / "part%03d.wav")],
        capture_output=True, timeout=300)
    if proc.returncode != 0:
        log.warning("切段失败: %s", proc.stderr.decode("utf-8", "ignore")[:200])
        return []
    parts = sorted(dest_dir.glob("part*.wav"))
    if not parts:
        return []
    log.info("音频 %.0fs 切成 %d 段（每段 <= %ds）", seconds, len(parts), max_seconds)
    return parts


class StepFunASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        self.calls = 0
        self.failures = 0
        self.chars = 0
        self.audio_seconds = 0.0

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
    def configured(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.api_key)

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.api_key}"}

    # ---- 入口：按套餐自动挑通道 -----------------------------------------

    def _resolve_transport(self) -> str:
        """返回 "sse" 或 "upload"。

        `auto` 的规则：step_plan 前缀**只有 SSE 那条**（multipart 的
        /audio/transcriptions 在网关上就是 404），所以必须走 SSE；
        开放平台 /v1 两条都有，优先 multipart 上传 —— 大文件友好且一次拿全文。
        """
        t = (self.cfg.transport or "auto").strip().lower()
        if t in ("sse", "upload"):
            return t
        return "sse" if is_step_plan(self.cfg.base_url) else "upload"

    async def transcribe_file(self, audio: Path, *, hotwords: list[str] | None = None) -> AsrResult:
        """本地音频 -> 全文。不需要公网地址，通道由 `transport` 决定。"""
        if self._resolve_transport() == "sse":
            return await self.transcribe_sse(audio, hotwords=hotwords)

        r = await self.transcribe_upload(audio, hotwords=hotwords)
        # 兜底：自定义代理/网关下 upload 那条可能压根没注册（404）。
        # 换 SSE 再试一次 —— 404 只代表路径没有，不代表能力没有（这条踩过）。
        if not r.ok and "404" in r.error:
            log.info("upload 通道 404，回退 SSE：%s", r.error)
            r2 = await self.transcribe_sse(audio, hotwords=hotwords)
            if r2.ok:
                return r2
        return r

    # ---- 通道 A：multipart 同步上传 ------------------------------------

    async def transcribe_upload(self, audio: Path, *, hotwords: list[str] | None = None) -> AsrResult:
        """`{root}/audio/transcriptions`：上传本地音频，同步拿回全文。"""
        started = time.monotonic()
        # 调用方没显式传就用配置里的 —— 之前 cfg.hotwords 是个没人读的死字段
        hotwords = hotwords or self.cfg.hotwords
        if not self.configured:
            return AsrResult(ok=False, error="ASR 未启用或未配置 STEPFUN_API_KEY")
        if not audio.is_file():
            return AsrResult(ok=False, error=f"音频文件不存在: {audio}")

        size = audio.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            return AsrResult(ok=False, error=f"文件过大（{size / 1048576:.0f}MB > 95MB）")

        suffix = audio.suffix.lstrip(".").lower()
        if suffix not in UPLOAD_FORMATS:
            return AsrResult(ok=False, error=f"不支持的格式 {suffix}，需为 {UPLOAD_FORMATS}")

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("model", self.cfg.model)
        form.add_field("response_format", "json")
        if hotwords:
            form.add_field("hotwords", json.dumps(hotwords[:100], ensure_ascii=False))
        try:
            form.add_field("file", audio.read_bytes(),
                           filename=audio.name,
                           content_type="audio/wav" if suffix == "wav" else f"audio/{suffix}")
        except OSError as exc:
            return AsrResult(ok=False, error=f"读文件失败: {exc}")

        url = f"{normalize_base_url(self.cfg.base_url)}{UPLOAD_PATH}"
        try:
            async with session.post(
                url, data=form, headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout),
            ) as resp:
                body = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            return AsrResult(ok=False, error=f"请求失败: {exc}",
                             ms=int((time.monotonic() - started) * 1000))

        ms = int((time.monotonic() - started) * 1000)
        self.calls += 1

        if status != 200:
            self.failures += 1
            return AsrResult(ok=False, error=_explain(status, body), ms=ms)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # response_format=json 也会有些实现直接返回纯文本，兜一下
            text = body.strip()
            if text:
                self.chars += len(text)
                return AsrResult(ok=True, text=text, ms=ms)
            self.failures += 1
            return AsrResult(ok=False, error="响应为空", ms=ms)

        text = str(data.get("text") or "").strip()
        if not text:
            self.failures += 1
            return AsrResult(ok=False, error="识别结果为空（可能是纯音乐或噪音）", ms=ms)

        self.chars += len(text)
        log.info("ASR 完成：%s -> %d 字（%dms）", audio.name, len(text), ms)
        return AsrResult(ok=True, text=text, ms=ms)

    # ---- 通道 B：HTTP + SSE（step_plan 唯一可用的那条） ------------------

    async def transcribe_sse(self, audio: Path, *, hotwords: list[str] | None = None,
                             language: str | None = None) -> AsrResult:
        """`{root}/audio/asr/sse`：JSON + base64 提交，SSE 返回增量与全文。

        请求形状和 OpenAI 兼容那条完全不同 —— 别把 multipart 的写法套过来，
        那正是当初误判 "step_plan 没有 ASR" 的原因。

        这条要把**整段**音频 base64 塞进一个 JSON body，所以超过 `SSE_MAX_BYTES`
        的长音频会自动按时间切段、逐段识别再拼回去（只有 wav 能切，
        因为只有它带标准头可以算出每秒字节数）。
        """
        hotwords = hotwords or self.cfg.hotwords
        started = time.monotonic()

        try:
            size = audio.stat().st_size
        except OSError:
            size = 0
        if size > SSE_MAX_BYTES and audio.suffix.lower() == ".wav":
            with tempfile.TemporaryDirectory() as td:
                parts = split_wav(audio, Path(td))
                if len(parts) > 1:
                    texts: list[str] = []
                    worst: AsrResult | None = None
                    for i, p in enumerate(parts):
                        r = await self._sse_one(p, hotwords=hotwords, language=language)
                        if r.ok:
                            texts.append(r.text)
                        else:
                            worst = worst or r
                            log.warning("第 %d/%d 段识别失败：%s", i + 1, len(parts), r.error)
                    if texts:
                        text = "".join(texts).strip()
                        ms = int((time.monotonic() - started) * 1000)
                        self.chars += len(text)
                        log.info("ASR(SSE) 分段完成：%d 段 -> %d 字（%dms）",
                                 len(parts), len(text), ms)
                        return AsrResult(ok=True, text=text, ms=ms)
                    return worst or AsrResult(ok=False, error="切段后一段都没识别出来",
                                              ms=int((time.monotonic() - started) * 1000))

        return await self._sse_one(audio, hotwords=hotwords, language=language)

    async def _sse_one(self, audio: Path, *, hotwords: list[str] | None = None,
                       language: str | None = None) -> AsrResult:
        """SSE 单段识别（不切段）。"""
        started = time.monotonic()
        if not self.configured:
            return AsrResult(ok=False, error="ASR 未启用或未配置 STEPFUN_API_KEY")
        if not audio.is_file():
            return AsrResult(ok=False, error=f"音频文件不存在: {audio}")

        suffix = audio.suffix.lstrip(".").lower()
        if suffix not in SSE_FORMATS:
            return AsrResult(ok=False, error=f"SSE 通道不支持格式 {suffix}，需为 {SSE_FORMATS}")
        try:
            raw = audio.read_bytes()
        except OSError as exc:
            return AsrResult(ok=False, error=f"读文件失败: {exc}")
        if len(raw) > SSE_MAX_BYTES:
            # 切不了段（不是 wav）或单段仍然过大 —— 明确说清上限在哪
            return AsrResult(
                ok=False,
                error=f"文件过大（{len(raw) / 1048576:.1f}MB > "
                      f"{SSE_MAX_BYTES / 1048576:.1f}MB），SSE 通道是整段 base64 提交，"
                      "只有 wav 能自动切段；长音频请改用 /v1 + transport=upload "
                      "或异步 submit")

        transcription: dict = {"model": self.cfg.model,
                               "language": language or self.cfg.language,
                               "enable_itn": True}
        if hotwords:
            transcription["hotwords"] = list(hotwords[:100])
        fmt: dict = {"type": suffix}
        if suffix == "pcm":
            # 文档：pcm 的 rate/bits/channel 是必填，其他容器格式选填
            fmt.update({"codec": "pcm_s16le", "rate": TARGET_RATE,
                        "bits": 16, "channel": 1})

        payload = {"audio": {"data": base64.b64encode(raw).decode("ascii"),
                             "input": {"transcription": transcription, "format": fmt}}}
        url = f"{normalize_base_url(self.cfg.base_url)}{SSE_PATH}"
        session = await self._get_session()
        try:
            async with session.post(
                url, json=payload,
                headers={**self._headers, "Content-Type": "application/json",
                         "Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout),
            ) as resp:
                body = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            return AsrResult(ok=False, error=f"请求失败: {exc}",
                             ms=int((time.monotonic() - started) * 1000))

        ms = int((time.monotonic() - started) * 1000)
        self.calls += 1
        if status != 200:
            self.failures += 1
            return AsrResult(ok=False, error=_explain(status, body), ms=ms)

        text, err = _parse_sse_events(body)
        if err:
            self.failures += 1
            return AsrResult(ok=False, error=f"SSE 报错：{err}", ms=ms)
        if not text:
            self.failures += 1
            return AsrResult(ok=False, error="识别结果为空（可能是纯音乐或噪音）", ms=ms)

        self.chars += len(text)
        log.info("ASR(SSE) 完成：%s -> %d 字（%dms）", audio.name, len(text), ms)
        return AsrResult(ok=True, text=text, ms=ms)

        transcription: dict = {"model": self.cfg.model,
                               "language": language or self.cfg.language,
                               "enable_itn": True}
        if hotwords:
            transcription["hotwords"] = list(hotwords[:100])
        fmt: dict = {"type": suffix}
        if suffix == "pcm":
            # 文档：pcm 的 rate/bits/channel 是必填，其他容器格式选填
            fmt.update({"codec": "pcm_s16le", "rate": TARGET_RATE,
                        "bits": 16, "channel": 1})

        payload = {"audio": {"data": base64.b64encode(raw).decode("ascii"),
                             "input": {"transcription": transcription, "format": fmt}}}
        url = f"{normalize_base_url(self.cfg.base_url)}{SSE_PATH}"
        session = await self._get_session()
        try:
            async with session.post(
                url, json=payload,
                headers={**self._headers, "Content-Type": "application/json",
                         "Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout),
            ) as resp:
                body = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.failures += 1
            return AsrResult(ok=False, error=f"请求失败: {exc}",
                             ms=int((time.monotonic() - started) * 1000))

        ms = int((time.monotonic() - started) * 1000)
        self.calls += 1
        if status != 200:
            self.failures += 1
            return AsrResult(ok=False, error=_explain(status, body), ms=ms)

        text, err = _parse_sse_events(body)
        if err:
            self.failures += 1
            return AsrResult(ok=False, error=f"SSE 报错：{err}", ms=ms)
        if not text:
            self.failures += 1
            return AsrResult(ok=False, error="识别结果为空（可能是纯音乐或噪音）", ms=ms)

        self.chars += len(text)
        log.info("ASR(SSE) 完成：%s -> %d 字（%dms）", audio.name, len(text), ms)
        return AsrResult(ok=True, text=text, ms=ms)

    # ---- 异步任务（需要公网 URL） -------------------------------------

    async def submit_url(self, audio_url: str, *, fmt: str = "wav", rate: int = TARGET_RATE,
                         channel: int = 1, utterances: bool = False) -> str:
        """提交一个公网可达的音频 URL，返回 task_id。"""
        if not self.configured:
            raise AsrError("ASR 未启用或未配置 STEPFUN_API_KEY")

        payload = {
            "audio": {"format": fmt, "rate": rate, "bits": 16,
                      "channel": channel, "url": audio_url},
            "request": {
                "model_name": self.cfg.model,
                "enable_channel_split": False,
                "show_utterances": bool(utterances),
                "enable_speaker_info": False,
            },
        }
        session = await self._get_session()
        try:
            async with session.post(
                f"{normalize_base_url(self.cfg.base_url)}/audio/asr/file/submit",
                json=payload, headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout),
            ) as resp:
                body = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise AsrError(f"提交失败: {exc}") from exc

        if status != 200:
            raise AsrError(f"提交 HTTP {status}: {body[:200]}")
        try:
            return str(json.loads(body)["task_id"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise AsrError(f"提交响应异常: {body[:200]}") from exc

    async def query(self, task_id: str) -> dict:
        session = await self._get_session()
        try:
            async with session.post(
                f"{normalize_base_url(self.cfg.base_url)}/audio/asr/file/query",
                json={"task_id": task_id}, headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout),
            ) as resp:
                body = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return {"status": "RUNNING", "error": {"message": str(exc)}}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"status": "RUNNING"}

    async def wait_url(self, task_id: str, *, timeout: float | None = None) -> AsrResult:
        """轮询到出结果。官方建议 1~3 秒一次。"""
        deadline = time.monotonic() + (timeout or self.cfg.job_timeout)
        while time.monotonic() < deadline:
            await asyncio.sleep(self.cfg.poll_interval)
            data = await self.query(task_id)
            status = str(data.get("status") or "").upper()
            if status == "FAILED":
                err = data.get("error") or {}
                self.failures += 1
                return AsrResult(ok=False,
                                 error=f"{err.get('stage', '')}: {err.get('message', '任务失败')}")
            if status and status not in ("PENDING", "RUNNING"):
                results = data.get("result") or []
                texts = [str(r.get("text") or "") for r in results if isinstance(r, dict)]
                text = "".join(texts).strip()
                if not text:
                    self.failures += 1
                    return AsrResult(ok=False, error="识别结果为空")
                self.calls += 1
                self.chars += len(text)
                return AsrResult(ok=True, text=text,
                                 duration=float(data.get("duration") or 0),
                                 utterances=results[0].get("utterances") if results else [])
        return AsrResult(ok=False, error="等待超时")

    # ---- 观测 ---------------------------------------------------------

    def stats(self) -> dict:
        return {
            "asr_calls": self.calls,
            "asr_failures": self.failures,
            "asr_chars": self.chars,
        }
