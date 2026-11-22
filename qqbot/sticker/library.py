"""表情库的文件层：下载、识别格式、按内容哈希落盘。

**收到就要立刻下载**。QQ CDN 的 `rkey` 是时效性的，拖几个小时链接就死了，
这是整个收藏链路里最容易踩的坑。所以这里只做"尽力而为 + 快速失败"，
不重试到天荒地老。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

#: 魔数 -> (扩展名, 是否动图)
MAGIC: list[tuple[bytes, str, bool]] = [
    (b"GIF87a", "gif", True),
    (b"GIF89a", "gif", True),
    (b"\x89PNG\r\n\x1a\n", "png", False),
    (b"\xff\xd8\xff", "jpg", False),
    (b"BM", "bmp", False),
]

DOWNLOAD_TIMEOUT = 20.0
MAX_BYTES = 8 * 1024 * 1024


def sniff(data: bytes) -> tuple[str, bool]:
    """从魔数判断扩展名和是否动图。比看 URL 后缀可靠。"""
    for magic, ext, animated in MAGIC:
        if data.startswith(magic):
            return ext, animated
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        # WEBP 可能是动图，粗略按前 200 字节里有没有 ANIM 判断
        return "webp", b"ANIM" in data[:200]
    return "png", False


@dataclass
class FetchedImage:
    hash: str
    path: Path
    ext: str
    size: int
    is_animated: bool


class StickerLibrary:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT, sock_connect=8)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ---- 落盘 ---------------------------------------------------------

    def path_for(self, content_hash: str, ext: str) -> Path:
        return self.base_dir / f"{content_hash}.{ext}"

    def store_bytes(self, data: bytes) -> FetchedImage | None:
        if not data or len(data) > MAX_BYTES:
            return None
        ext, animated = sniff(data)
        content_hash = hashlib.sha256(data).hexdigest()
        path = self.path_for(content_hash, ext)

        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # 原子替换，避免半截文件被当成成品

        return FetchedImage(
            hash=content_hash, path=path, ext=ext,
            size=len(data), is_animated=animated,
        )

    async def fetch(self, url: str) -> FetchedImage | None:
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    log.debug("表情下载失败 HTTP %s", resp.status)
                    return None
                data = await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.debug("表情下载异常: %s", exc)
            return None

        result = self.store_bytes(data)
        if result is None:
            log.debug("表情内容不可用（空或过大）")
        return result

    # ---- 从消息段取图 ------------------------------------------------

    @staticmethod
    def extract_image_segments(segments: list[dict]) -> list[dict]:
        """挑出表情包类型的图片段。

        sub_type 语义：0 普通图片 / 1 表情包。mface 是商城表情，另算。
        """
        found: list[dict] = []
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            stype = seg.get("type")
            data = seg.get("data") or {}
            if stype == "image":
                if str(data.get("sub_type")) == "1":
                    found.append({"kind": "image", "data": data})
            elif stype in ("mface", "marketface"):
                found.append({"kind": "mark_face", "data": data})
        return found

    async def ingest(self, segments: list[dict]) -> list[tuple[FetchedImage, bool]]:
        """把消息里的表情包下载并入库到磁盘。返回 [(图, 是否商城表情)]。"""
        out: list[tuple[FetchedImage, bool]] = []
        for item in self.extract_image_segments(segments):
            data = item["data"]
            url = data.get("url") or ""
            is_mark = item["kind"] == "mark_face"

            if url:
                fetched = await self.fetch(url)
            else:
                # 没有 url 时退而求其次：本地文件已经有了就直接用
                fetched = None

            if fetched is not None:
                out.append((fetched, is_mark))
        return out
