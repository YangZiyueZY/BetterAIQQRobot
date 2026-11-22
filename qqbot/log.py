"""日志配置。"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", extra_handler: logging.Handler | None = None) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-20s %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    if extra_handler is not None:
        # 面板日志流：级别放开到 DEBUG，实际过滤交给 root 的 level
        extra_handler.setLevel(logging.DEBUG)
        root.addHandler(extra_handler)

    # 第三方库降噪
    for noisy in ("aiosqlite", "asyncio", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
