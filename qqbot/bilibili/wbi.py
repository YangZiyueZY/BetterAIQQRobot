"""B 站 wbi 签名。

部分接口（`x/player/wbi/v2`、`wbi/search/type`）要求带 wbi 签名，
算法是固定的：从 nav 接口拿 img_key/sub_key，按一张 64 位索引表重排取前 32 位，
再把参数排序编码后拼上 mixin_key 做 md5。

签名的 key 会轮换，所以**缓存要有有效期**，不能启动时取一次用一辈子。
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.parse

log = logging.getLogger(__name__)

#: 官方的 64 位重排索引表，写死即可
MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

#: wbi key 的缓存时长。官方会轮换，保守取 1 小时。
KEY_TTL = 3600.0

#: 这些字符在签名前必须从参数值里过滤掉
_FILTER = "!'()*"


def mixin_key(img_key: str, sub_key: str) -> str:
    raw = img_key + sub_key
    return "".join(raw[i] for i in MIXIN_KEY_ENC_TAB if i < len(raw))[:32]


def sign(params: dict, img_key: str, sub_key: str,
         now: float | None = None) -> str:
    """返回签好名的 query string（已含 w_rid 和 wts）。"""
    key = mixin_key(img_key, sub_key)
    signed = dict(params)
    signed["wts"] = int(now or time.time())

    cleaned = {
        k: "".join(c for c in str(v) if c not in _FILTER)
        for k, v in sorted(signed.items())
    }
    query = urllib.parse.urlencode(cleaned)
    signed["w_rid"] = hashlib.md5((query + key).encode()).hexdigest()
    return urllib.parse.urlencode(sorted(signed.items()))


class WbiKeys:
    """img_key / sub_key 的带缓存持有者。"""

    def __init__(self) -> None:
        self.img_key = ""
        self.sub_key = ""
        self._fetched_at = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.img_key and self.sub_key
                    and (time.time() - self._fetched_at) < KEY_TTL)

    def update(self, img_url: str, sub_url: str) -> None:
        def key_of(url: str) -> str:
            return url.rsplit("/", 1)[-1].split(".")[0] if url else ""

        self.img_key = key_of(img_url)
        self.sub_key = key_of(sub_url)
        self._fetched_at = time.time()
        if not (self.img_key and self.sub_key):
            log.warning("wbi key 解析失败: img=%r sub=%r", img_url[:40], sub_url[:40])

    def sign(self, params: dict) -> str:
        return sign(params, self.img_key, self.sub_key)
