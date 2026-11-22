"""OneBot v11 API 封装。

正向 WebSocket（NapCat 当服务端）下，事件推送和 API 调用共用同一条连接：
发 {"action", "params", "echo"}，按 echo 匹配响应。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

log = logging.getLogger(__name__)


class OneBotError(RuntimeError):
    def __init__(self, action: str, payload: dict) -> None:
        self.action = action
        self.payload = payload
        super().__init__(
            f"{action} 失败: retcode={payload.get('retcode')} "
            f"status={payload.get('status')} msg={payload.get('message') or payload.get('wording')}"
        )


class OneBotAPI:
    """通过 WebSocket 调用 OneBot API，按 echo 做请求-响应配对。"""

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout
        self._ws: Any = None
        self._pending: dict[str, asyncio.Future] = {}

    # ---- 连接生命周期 -------------------------------------------------

    def bind(self, ws: Any) -> None:
        self._ws = ws

    def unbind(self) -> None:
        self._ws = None
        # 连接断了，所有待响应请求立刻失败，避免调用方永久挂起
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("WebSocket 已断开"))
        self._pending.clear()

    # ---- 请求 ---------------------------------------------------------

    async def call(self, action: str, **params: Any) -> Any:
        ws = self._ws
        if ws is None or getattr(ws, "closed", False):
            raise ConnectionError("WebSocket 未连接")

        echo = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        try:
            await ws.send_json({"action": action, "params": params, "echo": echo})
            payload = await asyncio.wait_for(fut, self._timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"{action} 超时 ({self._timeout}s)") from None
        finally:
            self._pending.pop(echo, None)
        return payload

    def resolve(self, payload: dict) -> None:
        """由 client 在收到带 echo 的响应时调用。"""
        fut = self._pending.get(payload.get("echo"))
        if fut is None or fut.done():
            return
        status = payload.get("status")
        retcode = payload.get("retcode")
        if status == "ok" or retcode in (0, 1):
            fut.set_result(payload.get("data"))
        else:
            fut.set_exception(OneBotError(str(payload.get("echo")), payload))

    # ---- 常用接口 -----------------------------------------------------

    async def get_login_info(self) -> dict:
        return await self.call("get_login_info")

    async def get_group_list(self) -> list[dict]:
        return await self.call("get_group_list") or []

    async def get_group_member_list(self, group_id: int) -> list[dict]:
        return await self.call("get_group_member_list", group_id=group_id) or []

    async def get_msg(self, message_id: int) -> dict | None:
        try:
            return await self.call("get_msg", message_id=message_id)
        except (OneBotError, TimeoutError):
            return None

    # ---- 历史与转发（冷启动分析、合并转发要用）----------------------

    async def get_group_msg_history(self, group_id: int, *,
                                    count: int = 20,
                                    message_seq: str | None = None,
                                    reverse_order: bool = False) -> list[dict] | None:
        """拉群历史消息。**返回 None 表示翻到底了**，空列表表示这次没数据。

        实测（NapCat 4.18.28）两个坑：
          · 往前翻页要 `message_seq=<当前最旧的 seq> + reverse_order=True`，
            结果**以该 seq 结尾**（含它），所以下一页要跳过最后一条，否则死循环
          · **翻到底不是返回空数组，而是 retcode 1200「消息X不存在」** ——
            不认这个错误码就会把"到底了"当成异常，永远拉不完
        """
        params: dict[str, Any] = {"group_id": str(group_id), "count": int(count)}
        if message_seq is not None:
            params["message_seq"] = str(message_seq)
            params["reverse_order"] = bool(reverse_order)
        try:
            data = await self.call("get_group_msg_history", **params)
        except OneBotError as exc:
            msg = str(exc)
            if "不存在" in msg or "1200" in msg:
                return None          # 到底了，不是错误
            log.warning("拉群历史失败 group=%s: %s", group_id, exc)
            return []
        except (TimeoutError, ConnectionError):
            return []
        if isinstance(data, dict):
            msgs = data.get("messages")
            return list(msgs) if isinstance(msgs, list) else []
        return list(data) if isinstance(data, list) else []

    async def get_forward_msg(self, forward_id: str) -> list[dict]:
        """取合并转发的内容。

        实测：push 里的 forward 段**只有 `id`**（NapCat 配置 `parseMultMsg:false`），
        内容得自己拉。返回的是完整 OB11 消息对象数组（不是 node 段包装）。
        """
        try:
            data = await self.call("get_forward_msg", id=str(forward_id))
        except (OneBotError, TimeoutError, ConnectionError) as exc:
            log.info("取合并转发失败 id=%s: %s", forward_id, exc)
            return []
        if isinstance(data, dict):
            msgs = data.get("messages")
            return list(msgs) if isinstance(msgs, list) else []
        return list(data) if isinstance(data, list) else []

    async def get_record(self, file: str, *, out_format: str = "mp3") -> dict:
        """让 NapCat 把语音下载/转码出来。

        实测：record 段的 `path` **指向的文件并不存在**（QQ 只存元数据），
        所以只能靠 `url` 下载，或把 `file` 当 token 交给这个 action 代下。
        """
        try:
            data = await self.call("get_record", file=str(file),
                                   out_format=str(out_format))
        except (OneBotError, TimeoutError, ConnectionError) as exc:
            log.info("取语音文件失败 file=%s: %s", file, exc)
            return {}
        return data if isinstance(data, dict) else {}

    async def send_group_msg(self, group_id: int, text: str) -> int | None:
        data = await self.call("send_group_msg", group_id=group_id, message=text)
        return (data or {}).get("message_id")

    async def send_group_segments(self, group_id: int, segments: list[dict]) -> int | None:
        data = await self.call("send_group_msg", group_id=group_id, message=segments)
        return (data or {}).get("message_id")

    async def set_msg_emoji_like(self, message_id: int, emoji_id: str = "128077") -> None:
        try:
            await self.call("set_msg_emoji_like", message_id=message_id, emoji_id=emoji_id)
        except (OneBotError, TimeoutError):
            pass

    async def get_emoji_likes(self, message_id: int) -> list[dict]:
        """取某条消息的表情回应列表。

        注意 message_id 必须是**字符串** —— 传 int 会被 schema 校验拒掉
        （NapCat 返回 "Schema compilation error: Expected required property"）。
        """
        try:
            data = await self.call(
                "get_emoji_likes", message_id=str(message_id), count=0
            )
        except (OneBotError, TimeoutError):
            return []
        if isinstance(data, dict):
            return data.get("emoji_like_list") or []
        return []

    async def fetch_custom_face(self, count: int = 48) -> list[str]:
        """取 QQ 收藏夹里的表情 URL 列表。"""
        try:
            return await self.call("fetch_custom_face", count=count) or []
        except (OneBotError, TimeoutError):
            return []

    async def fetch_custom_face_detail(self, count: int = 48) -> list[dict]:
        """取收藏夹详情（含 res_id / md5 / desc），比只取 URL 有用。"""
        try:
            return await self.call("fetch_custom_face_detail", count=count) or []
        except (OneBotError, TimeoutError):
            return []

    async def add_custom_face(self, file: str, *, is_origin: bool = True) -> None:
        """把本地文件加入 QQ 收藏夹。

        **这是写操作，会改动账号状态。** 调用前务必确认开关打开。
        file 用本地绝对路径；md5 不传时 NapCat 会自己算。
        """
        await self.call("add_custom_face", file=str(file), is_origin=is_origin)

    async def delete_custom_face(self, res_id: str) -> None:
        await self.call("delete_custom_face", res_id=res_id)

    async def ocr_image(self, image: str) -> str:
        """图片 OCR。返回识别出的文字（可能为空）。"""
        try:
            data = await self.call("ocr_image", image=image)
        except (OneBotError, TimeoutError):
            return ""
        if isinstance(data, dict):
            return str(data.get("text") or data.get("result") or "")
        return str(data or "")

    async def fetch_ptt_text(self, message_id: int) -> str:
        """取语音转文字结果。message_id 必须是字符串。"""
        try:
            data = await self.call("fetch_ptt_text", message_id=str(message_id))
        except (OneBotError, TimeoutError):
            return ""
        if isinstance(data, dict):
            return str(data.get("text") or "")
        return str(data or "")

    async def set_online_status(self, status: int, ext_status: int = 0,
                                battery_status: int = 100) -> None:
        """设置 QQ 在线状态。10=在线 30=离开 50=忙碌 60=Q我吧。

        让 QQ 上显示的在线状态跟着作息走，是真的拟真度，白送的。
        """
        try:
            await self.call(
                "set_online_status",
                status=status, ext_status=ext_status, battery_status=battery_status,
            )
        except (OneBotError, TimeoutError):
            pass

