"""图片获取、去重、压缩。

压缩这一步直接决定 VLM 的成本：视觉模型按分辨率计费，一张 2000px 的图
可能是 768px 的 4 倍 token。所以送模型前一律先压。原图留在磁盘上，
需要时可以重新处理。
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 20.0
MAX_BYTES = 12 * 1024 * 1024

MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
]


def sniff_ext(data: bytes) -> str:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "jpg"


@dataclass
class LocalImage:
    hash: str
    path: Path
    ext: str
    size: int
    width: int = 0
    height: int = 0
    is_animated: bool = False


class ImageStore:
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

    def store_bytes(self, data: bytes) -> LocalImage | None:
        if not data or len(data) > MAX_BYTES:
            return None
        ext = sniff_ext(data)
        content_hash = hashlib.sha256(data).hexdigest()
        path = self.base_dir / f"{content_hash}.{ext}"

        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)

        width, height, animated = _probe(path)
        return LocalImage(hash=content_hash, path=path, ext=ext,
                          size=len(data), width=width, height=height,
                          is_animated=animated)

    async def fetch(self, url: str) -> LocalImage | None:
        if not url:
            return None
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    log.debug("图片下载失败 HTTP %s", resp.status)
                    return None
                return self.store_bytes(await resp.read())
        except (aiohttp.ClientError, TimeoutError) as exc:
            log.debug("图片下载异常: %s", exc)
            return None

    # ---- 压缩 + 编码 --------------------------------------------------

    @staticmethod
    def to_data_uri(path: Path, max_side: int = 768, quality: int = 82) -> str | None:
        """压到指定边长并转成 data URI。

        动图只取第一帧 —— 视觉模型本来也只能看静帧。
        """
        try:
            from PIL import Image
        except ImportError:
            log.warning("未安装 Pillow，无法压缩图片，将直接读原图")
            try:
                raw = Path(path).read_bytes()
            except OSError:
                return None
            ext = Path(path).suffix.lstrip(".").lower() or "jpeg"
            mime = "image/gif" if ext == "gif" else f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

        try:
            with Image.open(path) as img:
                if getattr(img, "is_animated", False):
                    img.seek(0)
                img = img.convert("RGB")
                if max(img.size) > max_side:
                    ratio = max_side / max(img.size)
                    new_size = (max(1, int(img.width * ratio)),
                                max(1, int(img.height * ratio)))
                    img = img.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
        except Exception as exc:  # noqa: BLE001 - 压缩失败就放弃，不值得拖垮主流程
            log.debug("图片压缩失败 %s: %s", path, exc)
            return None

        encoded = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{encoded}"

    def local_path(self, content_hash: str) -> Path | None:
        for ext in ("png", "jpg", "gif", "webp", "bmp"):
            candidate = self.base_dir / f"{content_hash}.{ext}"
            if candidate.is_file():
                return candidate
        return None


def _probe(path: Path) -> tuple[int, int, bool]:
    """读尺寸和是否动图。失败就返回 0，不影响主流程。"""
    try:
        from PIL import Image
    except ImportError:
        return 0, 0, path.suffix.lower() == ".gif"
    try:
        with Image.open(path) as img:
            return img.width, img.height, bool(getattr(img, "is_animated", False))
    except Exception:  # noqa: BLE001
        return 0, 0, False
