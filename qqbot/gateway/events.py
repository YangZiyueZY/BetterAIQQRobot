"""OneBot v11 事件解析 -> 内部模型。

NapCat 侧 messagePostFormat 需为 array，我们按消息段数组解析。
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Message

log = logging.getLogger(__name__)

# 段类型 -> 占位文本，让 LLM 知道这里原本有东西
SEGMENT_PLACEHOLDER = {
    "image": "[图片]",
    "face": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "json": "[卡片]",
    "music": "[音乐]",
    "forward": "[合并转发]",
    "mface": "[表情包]",
    "poke": "[戳一戳]",
}


def _nickname(payload: dict) -> str:
    sender = payload.get("sender") or {}
    return (
        sender.get("card")
        or sender.get("nickname")
        or str(sender.get("user_id", ""))
    ).strip()


def parse_group_message(payload: dict, self_id: int) -> Message | None:
    """把一条群消息事件转成 Message；不是群消息或无法解析时返回 None。"""
    if payload.get("post_type") != "message":
        return None
    if payload.get("message_type") != "group":
        return None

    group_id = payload.get("group_id")
    message_id = payload.get("message_id")
    user_id = payload.get("user_id")
    if group_id is None or message_id is None or user_id is None:
        log.debug("丢弃字段不全的消息事件: %s", payload)
        return None

    raw_segments = payload.get("message")
    if isinstance(raw_segments, str):
        # 兜底：万一 NapCat 侧仍是 string 格式
        segments: list[dict] = [{"type": "text", "data": {"text": raw_segments}}]
    elif isinstance(raw_segments, list):
        segments = raw_segments
    else:
        segments = []

    parts: list[str] = []
    mentions_self = False
    reply_to: int | None = None
    has_image = False

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        stype = seg.get("type")
        sdata = seg.get("data") or {}

        if stype == "text":
            parts.append(str(sdata.get("text", "")))
        elif stype == "at":
            target = str(sdata.get("qq", ""))
            if target in ("all", str(self_id)):
                mentions_self = mentions_self or target == str(self_id)
                parts.append("@全体成员" if target == "all" else "@我")
            else:
                parts.append(f"@{target}")
        elif stype == "reply":
            try:
                reply_to = int(sdata.get("id"))
            except (TypeError, ValueError):
                reply_to = None
        else:
            if stype == "image":
                has_image = True
            parts.append(SEGMENT_PLACEHOLDER.get(str(stype), f"[{stype}]"))

    text = "".join(parts).strip()
    if not text:
        return None

    is_self = str(user_id) == str(self_id)

    return Message(
        group_id=int(group_id),
        message_id=int(message_id),
        user_id=int(user_id),
        nickname="我" if is_self else _nickname(payload),
        text=text,
        segments=segments,
        ts=int(payload.get("time") or 0),
        is_self=is_self,
        reply_to=reply_to,
        mentions_self=mentions_self,
        has_image=has_image,
    )
