"""非文字消息段的解析：语音 / 视频 / 卡片 / 合并转发 / 文件 / 表情包。

`gateway/events.py` 只给这些段落了个占位符 —— 群友发语音、甩卡片、合并转发时，
模型看到的就只有 `[语音]` `[卡片]` 三个字，等于完全不知道内容。这个模块把它们
变成**给 LLM 看的备注**：自然语言、带关键信息，每条消息一段字符串。

设计原则（都是踩坑换来的）：

  · **绝不抛异常**。任何一段解析失败都降级成有信息量的占位符（如
    `[语音]（没听清）`），不能让整条消息因为一段解析不了而丢掉。
    每一段还额外包一层 `cfg.ingest_timeout`，卡住的调用直接放弃。
  · 只做"把内容变成一句话"，**不看图**（看图是 MediaUnderstander 的活）。
  · 各段类型的事实都写在对应函数上面，全部来自实测，别照文档猜。

实测事实（NapCat 4.18.28，messagePostFormat=array）：

  record   只有 file / path / url / file_size。`path` 指向的文件**实测不存在**；
           url 是 http + rkey，**会过期，收到就得下载**；没有 duration。
           若 packet 后端不可用，`url` 键**直接不存在**（只有 file/path/file_size）。
  video    只有 file / url / file_size，没有 duration / thumb / path；
           实测 4 条里 3 条 `url` 是 NapCat 本机绝对路径 → 必须先判 scheme。
  json     `data.data` 是**一个 JSON 字符串**，要二次 `json.loads`；
           顶层有 app / meta / prompt / ver / view / desc / config；
           **meta 的第一个 key 决定子类型**（如 `Location.Search` 是位置分享）。
  forward  推送里**只有 id**（19 位字符串），内容要调 `get_forward_msg` 拉。
  file     字段是 file / file_id / file_size / url —— **没有 name / size**。
  image    `sub_type` 是**数字**（0 普通图 / 1 动画表情 / 7、11 含义未确认）；
           商城表情（mface）入站**根本不产生 mface 段**，NapCat 转成 image
           并带 `emoji_id` → 判定必须是 `sub_type == 1 or "emoji_id" in data`；
           未知 sub_type（7/11）不要当表情包。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from ..gateway.events import SEGMENT_PLACEHOLDER
from ..models import Message

log = logging.getLogger(__name__)

#: 单个音频/视频的下载上限。语音和视频音轨都在这个量级以内，超过多半是异常链接。
MAX_DOWNLOAD_BYTES = 24 * 1024 * 1024

#: ASR 能直接吃的容器格式（asr.client 的 UPLOAD_FORMATS 与 SSE_FORMATS 并集）。
#: 注意 **实测语音是 .amr，不在这张表里** —— 不转码就是白下载。
ASR_SUFFIXES = ("mp3", "wav", "ogg", "pcm", "m4a")

#: 卡片里值得摘出来给模型看的字段。不是遍历顺序，输出时按语义各取所需。
CARD_KEYS = (
    "title", "desc", "jumpUrl", "qqdocurl", "address", "name", "nickname",
    "brief", "summary", "tag",
)

#: 合并转发正文的长度上限，防止一条转发把上下文整个吃掉。
FORWARD_BODY_LIMIT = 1200


# --------------------------------------------------------------------------
# 小工具（全部要求：给什么怪输入都不许抛）
# --------------------------------------------------------------------------


def _as_dict(value: object) -> dict:
    """段 data / 卡片 payload 可能是 None、字符串、数组 —— 一律当空字典。"""
    return value if isinstance(value, dict) else {}


def is_http(url: object) -> bool:
    """只有 http(s) 才下载。

    实测 `video.url` 大概率是 NapCat 本机绝对路径（4 条里 3 条），语音段在
    packet 后端不可用时干脆没有 url。把本机路径丢给 aiohttp 只会报错，
    所以下载前一律先判 scheme。
    """
    if not isinstance(url, str) or not url:
        return False
    try:
        return urlparse(url).scheme in ("http", "https")
    except ValueError:
        return False


def ffmpeg_available() -> bool:
    """独立包一层，方便测试离线替换（不去真的找系统 ffmpeg）。"""
    return shutil.which("ffmpeg") is not None


def size_human(value: object) -> str:
    """字节数转人类可读。file_size 是字符串，也可能是空串/None/怪值。"""
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "大小未知"
    if size < 0 or size != size:  # 负数、NaN
        return "大小未知"
    if size < 1024:
        return f"{int(size)}B"
    for unit, div in (("KB", 1024), ("MB", 1024 ** 2), ("GB", 1024 ** 3)):
        if size < div * 1024:
            scaled = size / div
            return f"{scaled:.1f}{unit}" if scaled < 100 else f"{scaled:.0f}{unit}"
    return f"{size / 1024 ** 4:.1f}TB"


def sniff_audio_ext(raw: bytes) -> str:
    """从内容猜音频后缀。

    语音的下载 url 路径是 `/download`（真文件名在 query 里），拿不到后缀；
    而 ASR 是按后缀挑格式的，猜对了才能免掉一次 ffmpeg 转码。
    """
    if raw.startswith(b"#!AMR"):
        return ".amr"
    if raw.startswith(b"\x02#!SILK") or raw.startswith(b"\x02#!"):
        return ".silk"
    if raw.startswith(b"ID3") or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if raw.startswith(b"OggS"):
        return ".ogg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        return ".wav"
    if raw[4:8] == b"ftyp":
        return ".m4a"
    return ""


def is_sticker_image(data: object) -> bool:
    """这个 image 段是不是表情包。

    实测两个坑，缺一个就会把商城表情当成真图片送给视觉模型（白花钱还答非所问）：
      · `sub_type` 是**数字**不是字符串（0 普通图 / 1 动画表情 / 7、11 未确认）
      · 商城表情（mface）入站**不产生 mface 段**，NapCat 转成 image 且带 emoji_id
    所以判定是 `sub_type == 1 or "emoji_id" in data`；7/11 这类未知值**不算**表情包。
    """
    d = _as_dict(data)
    if d.get("emoji_id"):
        return True
    sub = d.get("sub_type")
    if isinstance(sub, str):
        return sub.strip() == "1"
    if isinstance(sub, bool):  # True 会和 1 相等，这里不是我们想要的语义
        return False
    return sub == 1


def sticker_note(data: object) -> str:
    d = _as_dict(data)
    name = (
        str(d.get("summary") or "").strip()
        or str(d.get("emoji_id") or "").strip()
        or str(d.get("file") or "").strip()
    )
    return f"[表情包] {name[:40]}" if name else "[表情包]"


def file_note(data: object) -> str:
    """`[文件] 报文名.pdf（1.1MB）`。

    **实测字段是 `file` / `file_size`，没有 `name` / `size`** ——
    照直觉写 data["name"] 会永远拿到空文件名。
    """
    d = _as_dict(data)
    name = str(d.get("file") or d.get("file_name") or "").strip() or "未知文件"
    size = size_human(d.get("file_size") or d.get("size"))
    return f"[文件] {name}（{size}）"


# --------------------------------------------------------------------------
# 卡片
# --------------------------------------------------------------------------


def _load_card_payload(d: dict) -> dict:
    """`data.data` 是一个 JSON **字符串**，要二次 loads。

    解析不出来（实测见过截断的卡片）就当空字典，外面还有 prompt 兜底。
    """
    raw = d.get("data")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _walk_card_fields(value: object, out: dict[str, str], depth: int = 0) -> None:
    """递归扫 meta，收 CARD_KEYS 里第一个非空值。

    不能假设第一个 key 是 news —— 实测位置分享是 `meta["Location.Search"]`。
    也不能假设一定有 qqdocurl（小程序字段，本机没观测到）。
    深度卡在 3 层，够覆盖 `meta.news.*` / `meta.detail_1.*`，又不会被深结构拖住。
    """
    if depth > 3 or not isinstance(value, dict):
        return
    for key, val in value.items():
        if isinstance(val, dict):
            _walk_card_fields(val, out, depth + 1)
        elif isinstance(val, (list, tuple)):
            for item in val[:3]:
                if isinstance(item, dict):
                    _walk_card_fields(item, out, depth + 1)
        elif key in CARD_KEYS and val not in (None, ""):
            text = str(val).strip()
            if text and key not in out:
                out[key] = text


def card_note(data: object) -> str:
    """QQ 卡片（分享链接 / 位置 / 小程序）-> 一句话。

    `meta` 的第一个 key 决定子类型，但我们不按子类型分支 ——
    实测见过的和没见过的（structmsg + news、detail_1 小程序）字段名高度重合，
    统一收集再拼句子，反而不会被"没见过的子类型"卡住。
    """
    d = _as_dict(data)
    payload = _load_card_payload(d)

    fields: dict[str, str] = {}
    _walk_card_fields(payload.get("meta"), fields)
    # 少数卡片把关键字段放在顶层，有就一并取（meta 里已有的优先）
    for key in CARD_KEYS:
        val = payload.get(key)
        if key not in fields and isinstance(val, (str, int, float)) and str(val).strip():
            fields[key] = str(val).strip()

    app = str(payload.get("app") or "").strip()
    view = str(payload.get("view") or "").strip()

    if fields.get("address") or "map" in app.lower() or "location" in view.lower():
        name = fields.get("name") or fields.get("title") or ""
        addr = fields.get("address") or fields.get("desc") or ""
        bits = []
        if name:
            bits.append(f"名称「{name}」")
        if addr:
            bits.append(addr)
        return "[位置] " + " ".join(bits) if bits else "[位置]"

    title = fields.get("title") or fields.get("name") or fields.get("nickname") or ""
    desc = fields.get("desc") or fields.get("brief") or fields.get("summary") or ""
    link = fields.get("jumpUrl") or fields.get("qqdocurl") or ""
    if title or desc or link:
        bits = []
        if title:
            bits.append(f"标题「{title}」")
        if desc:
            bits.append(f"简介「{desc}」")
        if link:
            bits.append(f"链接 {link}")
        return "[分享] " + " ".join(bits)

    # 兜底：卡片自带的一句提示语，比只有一个 [卡片] 强
    prompt = str(payload.get("prompt") or d.get("prompt") or "").strip()
    if prompt:
        return f"[卡片] {prompt[:150]}"
    if app or view:
        return f"[卡片] {app} {view}".strip()
    return "[卡片]"


# --------------------------------------------------------------------------
# 合并转发
# --------------------------------------------------------------------------


def render_segments(segments: object) -> str:
    """把 OB11 段数组压成一行文本（合并转发里的每条消息用）。

    文字段直接相接；占位符前面补一个空格 —— 否则"看图"+"[图片]"会粘成
    "看图[图片]"，读起来像另一个词。
    """
    if not isinstance(segments, (list, tuple)):
        return ""
    parts: list[str] = []

    def push(text: str, *, spaced: bool = False) -> None:
        if spaced and parts and not parts[-1].endswith((" ", "\n")):
            parts.append(" ")
        parts.append(text)

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        stype = str(seg.get("type") or "")
        data = _as_dict(seg.get("data"))
        if stype == "text":
            push(str(data.get("text") or ""))
        elif stype == "at":
            qq = str(data.get("qq") or "")
            push("@全体成员" if qq == "all" else f"@{qq}", spaced=True)
        elif stype == "image":
            push(sticker_note(data) if is_sticker_image(data) else "[图片]",
                 spaced=True)
        elif stype in ("mface", "bface"):
            push(sticker_note(data), spaced=True)
        elif stype == "face":
            fid = data.get("id")
            push(f"[表情{fid}]" if fid not in (None, "") else "[表情]", spaced=True)
        else:
            push(SEGMENT_PLACEHOLDER.get(stype, f"[{stype}]"), spaced=True)
    return "".join(parts).strip()


def _forward_node(node: object) -> tuple[str, str]:
    """从 `get_forward_msg` 返回的一个消息对象里取 (说话人, 内容)。

    实测它返回的是**完整 OB11 消息对象**（不是 node 段包装），
    发送者在 `sender.card/nickname`，内容在 `message` 段数组里。
    也见过 `content` / `raw_message` 这种简写形状，一并兜住。
    """
    if not isinstance(node, dict):
        return "", ""
    sender = _as_dict(node.get("sender"))
    who = str(sender.get("card") or sender.get("nickname") or "").strip()
    if not who and sender.get("user_id") is not None:
        who = str(sender.get("user_id"))

    message = node.get("message")
    content = message if isinstance(message, str) else render_segments(message)
    if not content:
        content = str(node.get("content") or node.get("raw_message") or "")
    return who, " ".join(content.split())


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------


async def download_to_file(url: str, dest_dir: Path, *, timeout: float = 20.0,
                           max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path | None:
    """下载到本地，按内容哈希命名（同一份音频重复推送只落一次盘）。

    再判一次 scheme 是保险：本机绝对路径进了这里，aiohttp 只会抛异常。
    **绝不记录完整 url** —— 语音 url 带 rkey，日志泄漏等于把语音权限漏出去。
    """
    if not is_http(url):
        return None
    path_part = urlparse(url).path
    suffix = Path(path_part).suffix
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,5}", suffix or ""):
        suffix = ""

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=8)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    log.info("下载失败 HTTP %s", resp.status)
                    return None
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        log.info("下载超过上限 %d 字节，放弃", max_bytes)
                        return None
                    chunks.append(chunk)
        raw = b"".join(chunks)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as exc:
        log.info("下载异常：%s", exc)          # 只记异常类型，不记 url
        return None
    if not raw:
        return None

    suffix = sniff_audio_ext(raw) or suffix or ".bin"
    dest = dest_dir / f"{hashlib.sha256(raw).hexdigest()}{suffix}"
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            tmp.write_bytes(raw)
            tmp.replace(dest)
        except OSError as exc:
            log.info("写临时文件失败: %s", exc)
            return None
    return dest


# --------------------------------------------------------------------------
# 主体
# --------------------------------------------------------------------------


async def _call(fn, *args):
    """把纯函数包成可 await 的形状，好统一走 wait_for。"""
    return fn(*args)


class SegmentIngestor:
    """把一条消息里的非文字段翻译成给 LLM 的备注。

    `cfg` 是 MediaConfig，`api` 是 OneBotAPI，`asr` 是 StepFunASR（可 None），
    `store` 是 ImageStore，`root` 是项目根，临时音频落在 `root/data/tmp/ingest`。

    调用方式：`notes = await ingestor.ingest(msg)` —— 返回的字符串由 pipeline
    追加到给模型看的上下文里；没有可解析的段就返回 `[]`。
    """

    def __init__(self, cfg, api, asr=None, store=None, root=None) -> None:
        self.cfg = cfg
        self.api = api
        self.asr = asr
        self.store = store
        self.root = Path(root) if root else Path.cwd()
        self._tmp: Path | None = None
        self._stats: dict[str, int] = {
            "voice": 0, "voice_text": 0, "voice_failed": 0,
            "video": 0, "video_text": 0,
            "card": 0, "forward": 0, "forward_nodes": 0,
            "file": 0, "sticker": 0,
            "timeouts": 0, "errors": 0,
        }

    # ---- 对外 -----------------------------------------------------------

    async def ingest(self, msg: Message) -> list[str]:
        notes: list[str] = []
        segments = getattr(msg, "segments", None)
        if not isinstance(segments, (list, tuple)):
            return notes
        for seg in segments:
            try:
                note = await self._segment(seg, msg)
            except Exception as exc:  # noqa: BLE001 - 解析失败绝不能让整条消息丢掉
                self._stats["errors"] += 1
                log.info("消息段解析异常，跳过：%s", exc)
                note = ""
            if note:
                notes.append(str(note))
        return notes

    def stats(self) -> dict:
        return dict(self._stats)

    # ---- 单段分发 -------------------------------------------------------

    async def _segment(self, seg: object, msg: Message) -> str:
        if not isinstance(seg, dict):
            return ""
        stype = str(seg.get("type") or "")
        data = _as_dict(seg.get("data"))

        if stype == "record":
            if not self._flag("voice_enabled"):
                return ""
            self._stats["voice"] += 1
            return await self._guarded(lambda: self._voice(data, msg), "[语音]（没听清）")

        if stype == "video":
            if not self._flag("video_enabled"):
                return ""
            self._stats["video"] += 1
            return await self._guarded(lambda: self._video(data), "[视频]")

        if stype == "json":
            if not self._flag("card_enabled"):
                return ""
            self._stats["card"] += 1
            return await self._guarded(lambda: _call(card_note, data), "[卡片]")

        if stype == "forward":
            if not self._flag("forward_enabled"):
                return ""
            self._stats["forward"] += 1
            return await self._guarded(
                lambda: self._forward(data), "[合并转发]（取不到内容）")

        if stype == "file":
            if not self._flag("file_enabled"):
                return ""
            self._stats["file"] += 1
            return await self._guarded(lambda: _call(file_note, data), "[文件]")

        if stype in ("image", "mface"):
            # 普通图片交给 MediaUnderstander（看图是它的活），这里只认表情包。
            # 商城表情入站是带 emoji_id 的 image 段，事件层会错标成 [图片]，
            # 模型就把它当成一张真照片了 —— 这里补上正确的标签。
            if stype == "mface" or is_sticker_image(data):
                self._stats["sticker"] += 1
                return sticker_note(data)
            return ""

        # face / poke / music / text / at / reply：事件层已给占位符，
        # 段里也没有更多可读内容（music 段实测只有 type+id），不重复输出。
        return ""

    async def _guarded(self, factory, fallback: str) -> str:
        """统一超时 + 兜底。`factory` 返回协程。"""
        try:
            note = await asyncio.wait_for(factory(), timeout=self._timeout())
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            log.info("消息段解析超时（%.1fs），降级成占位符", self._timeout())
            return fallback
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            log.info("消息段解析失败，降级成占位符：%s", exc)
            return fallback
        text = str(note or "").strip()
        return text or fallback

    # ---- 语音 -----------------------------------------------------------

    async def _voice(self, data: dict, msg: Message) -> str:
        # 顺序是权衡过的：fetch_ptt_text 快且免费，但**实测 4 条里 3 条失败返回空串**，
        # 所以空串是常态，必须继续走"下载 + ASR"，不能就此收工。
        text = await self._ptt_text(msg)
        if text:
            self._stats["voice_text"] += 1
            return f"[语音] 转文字：{text}"

        # 没有 ASR 就别下载了：语音 url 带 rkey、会过期，下下来也没人用。
        if self.asr is None:
            self._stats["voice_failed"] += 1
            return "[语音]（没听清）"

        audio = await self._voice_audio(data)
        if audio is not None:
            text = await self._transcribe(audio)
            if text:
                self._stats["voice_text"] += 1
                return f"[语音] 转文字：{text}"

        self._stats["voice_failed"] += 1
        return "[语音]（没听清）"

    async def _ptt_text(self, msg: Message) -> str:
        mid = getattr(msg, "message_id", None)
        if mid is None or self.api is None:
            return ""
        try:
            return str(await self.api.fetch_ptt_text(mid) or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.debug("fetch_ptt_text 失败：%s", exc)
            return ""

    async def _voice_audio(self, data: dict) -> Path | None:
        url = str(data.get("url") or "")
        if is_http(url):
            return await self._grab(url)
        # 实测：packet 后端不可用时 record 段**根本没有 url 键**（只有 file/path），
        # 而 path 指向的文件又不存在。这时只能把 file 当 token 交给 NapCat 的
        # get_record，让它代下/转码。
        token = str(data.get("file") or "").strip()
        if token and self.api is not None:
            info = await self._get_record(token)
            rec_url = str(info.get("url") or "")
            if is_http(rec_url):
                return await self._grab(rec_url)
        return None

    async def _get_record(self, file_token: str) -> dict:
        try:
            info = await self.api.get_record(file_token, out_format="mp3")
        except Exception as exc:  # noqa: BLE001
            log.info("get_record 失败：%s", exc)
            return {}
        return info if isinstance(info, dict) else {}

    # ---- 视频 -----------------------------------------------------------

    async def _video(self, data: dict) -> str:
        url = str(data.get("url") or "")
        # 实测 video.url **大概率是 NapCat 本机绝对路径**（4 条里 3 条），
        # 必须先判 scheme，否则会拿本机路径去发 HTTP 请求。
        if not is_http(url):
            return "[视频]（本机文件，读不到内容）"
        if self.asr is None:
            # 没有 ASR 就没有任何能读内容的手段，别白下载几十兆
            return "[视频]（没配置语音识别）"

        path = await self._grab(url)
        if path is None:
            return "[视频]（下载失败，读不到内容）"
        if not ffmpeg_available():
            # 视频容器 ASR 不认，必须靠 ffmpeg 抽音轨
            return "[视频]（没有 ffmpeg，提取不了声音）"

        text = await self._transcribe(path)
        if text:
            self._stats["video_text"] += 1
            return f"[视频] 语音内容：{text}"
        return "[视频]（听不出内容）"

    # ---- 合并转发 -------------------------------------------------------

    async def _forward(self, data: dict) -> str:
        fid = str(data.get("id") or "").strip()
        if not fid or self.api is None:
            return "[合并转发]（取不到内容）"
        try:
            nodes = await self.api.get_forward_msg(fid)
        except Exception as exc:  # noqa: BLE001 - api 层一般自己吞了，这里再兜一层
            log.info("取合并转发失败：%s", exc)
            return "[合并转发]（取不到内容）"
        if not isinstance(nodes, (list, tuple)) or not nodes:
            return "[合并转发]（取不到内容）"

        total = len(nodes)
        self._stats["forward_nodes"] += total
        limit = max(1, int(getattr(self.cfg, "forward_max_nodes", 30) or 30))
        shown = list(nodes[:limit])

        parts: list[str] = []
        for node in shown:
            who, content = _forward_node(node)
            content = content or "[空消息]"
            parts.append(f"{who}: {content}" if who else content)
        body = " / ".join(parts)
        if len(body) > FORWARD_BODY_LIMIT:
            body = body[:FORWARD_BODY_LIMIT] + "…"

        note = f"[合并转发] 共 {total} 条：{body}"
        rest = total - len(shown)
        if rest > 0:
            note += f"（还有 {rest} 条省略）"
        return note

    # ---- 音频准备 / 转写 -------------------------------------------------

    async def _transcribe(self, audio: Path) -> str:
        if self.asr is None:
            return ""
        ready = await self._prepare_audio(audio)
        if ready is None:
            return ""
        try:
            result = await self.asr.transcribe_file(ready)
        except Exception as exc:  # noqa: BLE001
            log.info("ASR 转写失败：%s", exc)
            return ""
        if isinstance(result, str):
            return result.strip()
        if not getattr(result, "ok", True):
            return ""
        return str(getattr(result, "text", "") or "").strip()

    async def _prepare_audio(self, path: Path) -> Path | None:
        """转成 ASR 认的格式。

        **实测语音是 .amr**，而 StepFun 只认 mp3/wav/ogg/pcm/m4a ——
        不转码就是白下载。目标统一用 16kHz 单声道 wav（见 asr.client.to_wav）。
        没装 ffmpeg 且格式不认时只能放弃，降级成"没听清"。
        """
        try:
            suffix = path.suffix.lstrip(".").lower()
        except AttributeError:
            return None
        if suffix in ASR_SUFFIXES:
            return path
        if not ffmpeg_available():
            log.info("音频格式 %s 不被 ASR 支持且没有 ffmpeg，跳过转写", suffix)
            return None
        from ..asr.client import to_wav  # 局部导入：只有真要转码时才需要它

        dest = path.with_suffix(".wav")
        try:
            ok = await asyncio.to_thread(to_wav, path, dest)
        except Exception as exc:  # noqa: BLE001
            log.info("转码失败：%s", exc)
            return None
        return dest if ok else None

    # ---- 杂项 -----------------------------------------------------------

    async def _grab(self, url: str) -> Path | None:
        try:
            return await download_to_file(url, self._tmp_dir(), timeout=self._timeout())
        except Exception as exc:  # noqa: BLE001
            log.info("音频下载失败：%s", exc)   # 不带 url（含 rkey）
            return None

    def _tmp_dir(self) -> Path:
        if self._tmp is None:
            self._tmp = self.root / "data" / "tmp" / "ingest"
            self._tmp.mkdir(parents=True, exist_ok=True)
        return self._tmp

    def _timeout(self) -> float:
        try:
            value = float(getattr(self.cfg, "ingest_timeout", 25.0) or 25.0)
        except (TypeError, ValueError):
            value = 25.0
        return max(0.1, value)

    def _flag(self, name: str) -> bool:
        """开关默认视为开 —— 配置缺字段时宁可多解析，也不要静默什么都不做。"""
        return bool(getattr(self.cfg, name, True))
