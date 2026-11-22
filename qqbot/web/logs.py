"""日志环形缓冲 + 订阅，供管理面板的实时日志流使用。"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections import deque
from typing import Any

CAPACITY = 800


class RingLogHandler(logging.Handler):
    """把日志同时留在内存环里（补历史）和推给所有订阅者（实时）。

    emit() 可能在事件循环之外的线程被调用（例如 aiosqlite 内部），所以
    入队用 call_soon_threadsafe 兜底。
    """

    def __init__(self, capacity: int = CAPACITY) -> None:
        super().__init__()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---- logging.Handler ---------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": int(record.created),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                exc_text = "".join(traceback.format_exception(*record.exc_info))
                entry["message"] = f"{entry['message']}\n{exc_text.strip()}"
        except Exception:  # noqa: BLE001 - 日志器自身绝不能抛
            return

        self._buffer.append(entry)
        self._publish(entry)

    def _publish(self, entry: dict) -> None:
        if not self._subscribers:
            return
        for queue in list(self._subscribers):
            try:
                if self._loop is not None and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._offer, queue, entry)
                else:
                    self._offer(queue, entry)
            except RuntimeError:
                # 循环已关闭
                self._subscribers.discard(queue)

    @staticmethod
    def _offer(queue: asyncio.Queue, entry: dict) -> None:
        try:
            queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass

    # ---- 对外接口 -----------------------------------------------------

    def recent(self, limit: int = 200) -> list[dict]:
        items = list(self._buffer)
        return items[-limit:] if limit > 0 else items

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
