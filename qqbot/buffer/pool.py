"""蓄水池：把连续的群消息攒成一段对话再交给下游。

真人不会收到一条就回一条。这里的三个参数决定了"节奏感"：

  debounce   新消息到达后等这么久；期间又来人说话就重置计时。
             这就是"等对方把话说完"。
  max_wait   距本批第一条消息最多等这么久，防止刷屏时永远触发不了。
  max_batch  一批最多攒多少条，超了立刻触发，防止内存无界增长。

每个群一个独立计时器。触发后回调 on_conversation，并清空该群缓冲。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from ..config import BehaviorConfig
from ..models import Conversation, Message

log = logging.getLogger(__name__)

ConversationHandler = Callable[[Conversation], Awaitable[None]]


class MessagePool:
    def __init__(
        self,
        on_conversation: ConversationHandler,
        cfg: BehaviorConfig,
    ) -> None:
        self._on_conversation = on_conversation
        self.cfg = cfg
        self._buffers: dict[int, list[Message]] = {}
        self._started: dict[int, float] = {}
        self._timers: dict[int, asyncio.Task] = {}
        self.flushed = 0

    # ---- 入口 ---------------------------------------------------------

    async def push(self, msg: Message) -> None:
        gid = msg.group_id
        loop = asyncio.get_running_loop()
        now = loop.time()

        buf = self._buffers.setdefault(gid, [])
        buf.append(msg)
        self._started.setdefault(gid, now)

        elapsed = now - self._started[gid]
        if len(buf) >= max(1, self.cfg.max_batch) or elapsed >= self.cfg.max_wait:
            await self.flush(gid)
            return

        self._arm(gid)

    def _arm(self, gid: int) -> None:
        """（重新）武装该群的防抖计时器。"""
        old = self._timers.pop(gid, None)
        if old is not None:
            old.cancel()

        loop = asyncio.get_running_loop()
        elapsed = loop.time() - self._started.get(gid, loop.time())
        delay = max(0.05, min(self.cfg.debounce, self.cfg.max_wait - elapsed))
        self._timers[gid] = asyncio.create_task(self._fire(gid, delay))

    async def _fire(self, gid: int, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        await self.flush(gid, from_timer=True)

    # ---- 触发 ---------------------------------------------------------

    async def flush(self, gid: int, *, from_timer: bool = False) -> None:
        timer = self._timers.pop(gid, None)
        if timer is not None and not from_timer:
            timer.cancel()

        buf = self._buffers.pop(gid, None)
        self._started.pop(gid, None)
        if not buf:
            return

        self.flushed += 1
        conv = Conversation(group_id=gid, messages=list(buf), trigger=buf[-1])
        try:
            await self._on_conversation(conv)
        except Exception:  # noqa: BLE001 - 单批失败不能拖垮整个池
            log.exception("处理会话失败 group=%s", gid)

    async def close(self) -> None:
        for task in list(self._timers.values()):
            task.cancel()
        self._timers.clear()
        self._buffers.clear()
        self._started.clear()

    # ---- 观测 ---------------------------------------------------------

    @property
    def pending(self) -> int:
        return sum(len(b) for b in self._buffers.values())
