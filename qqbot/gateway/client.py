"""NapCat 正向 WebSocket 客户端：断线重连 + 事件分发。

关键约束：**读循环永不阻塞**。
事件回调（落库、生成、发送）都可能耗时，甚至反过来要在这条连接上发
API 请求。若在读循环里 inline await 回调，就会形成自锁：
回调等 API 响应 -> 响应要从读循环取 -> 读循环被回调堵住。

所以这里用「读循环只解析 + 入队，独立 worker 消费」的结构，
on_ready 之类的回调则 schedule 成独立 task。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import aiohttp

from ..config import OneBotConfig
from ..models import Message
from .api import OneBotAPI
from .events import parse_group_message

log = logging.getLogger(__name__)

MessageHandler = Callable[[Message], Awaitable[None]]

QUEUE_MAX = 2000


class Gateway:
    def __init__(
        self,
        cfg: OneBotConfig,
        api: OneBotAPI,
        on_message: MessageHandler,
        on_ready: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._cfg = cfg
        self._api = api
        self._on_message = on_message
        self._on_ready = on_ready

        self.self_id: int = 0
        self.connected = asyncio.Event()

        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=QUEUE_MAX)
        self._worker: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._last_event = 0.0
        self._ready_announced = False
        self._closing = False
        self.dropped = 0

    # ---- 主循环 -------------------------------------------------------

    async def run(self) -> None:
        """永不返回（除非被取消）：连上 -> 读事件 -> 断了就退避重连。"""
        backoff = self._cfg.reconnect_min
        while not self._closing:
            try:
                await self._session_once()
                backoff = self._cfg.reconnect_min
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 任何异常都要能重连
                log.warning("连接异常: %s: %s", type(exc).__name__, exc)

            self._api.unbind()
            self.connected.clear()
            self._ready_announced = False

            if self._closing:
                break
            log.info("%.1fs 后重连…", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._cfg.reconnect_max)

    async def _session_once(self) -> None:
        headers = {}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                self._cfg.ws_url, headers=headers, heartbeat=30
            ) as ws:
                self._api.bind(ws)
                self._last_event = asyncio.get_running_loop().time()
                self.connected.set()
                self._worker = asyncio.create_task(self._consume())
                watchdog = asyncio.create_task(self._watchdog(ws))
                log.info("已连接 %s", self._cfg.ws_url)

                try:
                    async for raw in ws:
                        if raw.type == aiohttp.WSMsgType.TEXT:
                            self._last_event = asyncio.get_running_loop().time()
                            # 只解析 + 入队，绝不 await 业务逻辑
                            self._dispatch(raw.data)
                        elif raw.type == aiohttp.WSMsgType.ERROR:
                            log.warning("WebSocket 错误: %s", ws.exception())
                            break
                        elif raw.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            log.info("WebSocket 已关闭 (code=%s)", ws.close_code)
                            break
                finally:
                    for task in (watchdog, self._worker):
                        if task is not None:
                            task.cancel()
                    for task in (watchdog, self._worker):
                        if task is None:
                            continue
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    self._worker = None

    async def _watchdog(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """心跳超时保护：NapCat 卡死但 TCP 未断时，强制重连。"""
        interval = min(30.0, max(5.0, self._cfg.heartbeat_timeout / 3))
        try:
            while True:
                await asyncio.sleep(interval)
                idle = asyncio.get_running_loop().time() - self._last_event
                if idle > self._cfg.heartbeat_timeout:
                    log.warning("%.0fs 无任何事件，判定连接已死，主动重连", idle)
                    await ws.close()
                    return
        except asyncio.CancelledError:
            pass

    async def _consume(self) -> None:
        """独立 worker：串行消费消息，保证落库顺序。"""
        try:
            while True:
                msg = await self._queue.get()
                try:
                    await self._on_message(msg)
                except Exception:  # noqa: BLE001 - 单条失败不能拖垮 worker
                    log.exception("处理消息失败 message_id=%s", msg.message_id)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise

    # ---- 分发（同步，不允许 await 业务逻辑） --------------------------

    def _dispatch(self, data: str) -> None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            log.warning("收到非 JSON 数据: %.200s", data)
            return

        # API 响应（带 echo）优先，必须最先处理，否则调用方会一直等
        if payload.get("echo") is not None and payload.get("post_type") is None:
            self._api.resolve(payload)
            return

        # 任何事件都可能带 self_id，尽早拿到 UIN，不必等第一条群消息
        self_id = payload.get("self_id")
        if self_id and int(self_id) != self.self_id:
            self.self_id = int(self_id)
            log.info("账号 UIN: %s", self.self_id)
            self._schedule_ready()

        post_type = payload.get("post_type")
        if post_type == "meta_event":
            if payload.get("meta_event_type") == "lifecycle":
                log.info("NapCat 生命周期事件: %s", payload.get("sub_type"))
            return

        if post_type == "message":
            self._enqueue_message(payload)
            return

        if post_type == "notice":
            log.debug("notice: %s", payload.get("notice_type"))

    def _enqueue_message(self, payload: dict) -> None:
        if self.self_id == 0:
            return
        msg = parse_group_message(payload, self.self_id)
        if msg is None:
            return
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("消息队列已满，累计丢弃 %d 条", self.dropped)

    # ---- 就绪回调 -----------------------------------------------------

    def _schedule_ready(self) -> None:
        """把 on_ready 丢到独立 task，避免它 await API 时堵住读循环。"""
        if self._ready_announced or self.self_id == 0 or self._on_ready is None:
            return
        self._ready_announced = True
        task = asyncio.create_task(self._run_ready())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_ready(self) -> None:
        try:
            await self._on_ready(self.self_id)  # type: ignore[misc]
        except Exception:  # noqa: BLE001
            log.exception("on_ready 回调失败")

    async def close(self) -> None:
        self._closing = True
        for task in list(self._tasks):
            task.cancel()
