"""OneBot v11 消息段构建与渲染。

把"生成层的意图"翻译成"QQ 认的段数组"。几条硬性约束：

  · `reply` 段**必须放在数组第一位**，否则 QQ 不认这个引用
  · `at` 要放在文本之前，否则渲染成"文字@某人"
  · `face` 的 id 用 QQ 系统表情编号（见 face_catalog）
  · 发自定义表情走 `image` + `sub_type: 1`，QQ 才当表情显示而不是普通图片

生成层不直接产出段，而是产出带语义的 Bubble（"文字 笑死 + 表情 笑哭"），
在这里统一渲染。这样换协议实现时只改这一个文件。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .face_catalog import FaceCatalog

log = logging.getLogger(__name__)

#: 文本里残留的 [face:xxx] 标记（模型有时会把 token 写在正文里）
FACE_TOKEN_RE = re.compile(r"\[face:([^\[\]:]{1,12})\]")
STICKER_TOKEN_RE = re.compile(r"\[sticker:([^\[\]:]{1,16})\]")

#: OneBot image 段的 sub_type：1 = 表情包
SUBTYPE_STICKER = "1"
SUBTYPE_NORMAL = "0"


# --------------------------------------------------------------------------
# 基础构造函数
# --------------------------------------------------------------------------


def text(content: str) -> dict:
    return {"type": "text", "data": {"text": content}}


def at(user_id: int | str) -> dict:
    return {"type": "at", "data": {"qq": str(user_id)}}


def at_all() -> dict:
    return {"type": "at", "data": {"qq": "all"}}
    # 注意：@全体需要管理员权限，且极易招人烦。默认禁用，见 GroupPolicy.at_all_enabled。


def reply(message_id: int | str) -> dict:
    return {"type": "reply", "data": {"id": str(message_id)}}


def face(face_id: int) -> dict:
    return {"type": "face", "data": {"id": str(face_id)}}


def image(path_or_uri: str, *, as_sticker: bool = True) -> dict:
    """图片段。`file` 用 file:// URI，NapCat 会自己读本地文件上传。"""
    target = path_or_uri
    if not target.startswith(("http://", "https://", "file://", "base64://")):
        target = Path(target).resolve().as_uri()
    return {
        "type": "image",
        "data": {
            "file": target,
            "sub_type": SUBTYPE_STICKER if as_sticker else SUBTYPE_NORMAL,
        },
    }


def poke(user_id: int | str) -> dict:
    return {"type": "poke", "data": {"qq": str(user_id)}}


# --------------------------------------------------------------------------
# Bubble -> 段数组
# --------------------------------------------------------------------------


def render_bubble(bubble, catalog: FaceCatalog, *, sticker_base: Path | None = None) -> list[dict]:
    """把一个 Bubble 渲染成段数组。返回空列表表示这条没东西可发。"""
    segs: list[dict] = []

    # reply 必须在最前
    if getattr(bubble, "reply_to", None):
        segs.append(reply(bubble.reply_to))

    if getattr(bubble, "at_user", None):
        segs.append(at(bubble.at_user))

    body = (getattr(bubble, "text", "") or "").strip()
    face_name = getattr(bubble, "face", None)

    # 正文里内联的表情（[困] / [face:困]）就地拆成独立的 face 段，
    # 顺序保留 —— 直接 append 一个 text 段会让表情退化成字面量。
    inline_used = False
    if body:
        for kind, value in split_inline_faces(body, catalog):
            if kind == "face":
                resolved = catalog.by_token(value)
                if resolved is not None:
                    segs.append(face(resolved.id))
                    inline_used = True
                    continue
            segs.append(text(value))

    if face_name and not inline_used:
        resolved = catalog.by_token(face_name)
        if resolved is None:
            # 名字对不上就退化成一个方括号文字，总比整条发不出去好
            log.debug("未知表情名 %r，降级为文字", face_name)
            segs.append(text(f"[{face_name}]"))
        else:
            segs.append(face(resolved.id))

    # 自定义表情包
    path = getattr(bubble, "sticker_path", None)
    if path:
        try:
            segs.append(image(path, as_sticker=True))
        except (OSError, ValueError) as exc:
            log.warning("表情包路径不可用，跳过: %s (%s)", path, exc)

    return segs


#: 正文里内联的 QQ 表情写法。两种都收：
#:     [face:困]   显式标记（模型偶尔会写）
#:     [困]        裸名字 —— **模型最自然的写法**，因为 prompt 里的清单就是这个名字
_FACE_INLINE_RE = re.compile(r"\[(?:face:)?([^\[\]:]{1,12})\]")


def split_inline_faces(body: str, catalog) -> list[tuple[str, str]]:
    """把正文切成 [(kind, value)]，kind 是 "text" 或 "face"，顺序保留。

    只把**目录里认得的**名字当表情 —— `[图片]`、`[链接]` 这类 `by_token` 返回 None，
    会原样留作文字，不会误伤。

    为什么需要这个：模型看到 prompt 里的表情清单写的就是「微笑、呲牙、困…」，
    很自然会把 `[困]` 直接写进正文。以前这种写法不会被转成 QQ 表情，
    会原样发出字面量 `[困]` —— 用户看到的是方括号加两个字。
    """
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _FACE_INLINE_RE.finditer(body or ""):
        resolved = catalog.by_token(m.group(1))
        if resolved is None:
            continue                       # 不是已知表情，留着当普通文字
        if m.start() > pos:
            chunk = body[pos:m.start()].strip()
            if chunk:
                out.append(("text", chunk))
        out.append(("face", resolved.name))
        pos = m.end()
    tail = (body or "")[pos:].strip()
    if tail:
        out.append(("text", tail))
    if not out:
        stripped = (body or "").strip()
        if stripped:
            out.append(("text", stripped))
    return out


def split_face_tokens(body: str) -> tuple[str, list[str]]:
    """把正文里的 [face:x] 拆成 (纯文本, [名字...])，保留出现顺序。"""
    names = FACE_TOKEN_RE.findall(body)
    if not names:
        return body, []
    cleaned = FACE_TOKEN_RE.sub("", body).strip()
    return cleaned, names


def extract_sticker_token(body: str) -> tuple[str, str | None]:
    """把正文里的 [sticker:标签] 拆出来。"""
    m = STICKER_TOKEN_RE.search(body or "")
    if not m:
        return body, None
    return STICKER_TOKEN_RE.sub("", body).strip(), m.group(1).strip()


def describe(segments: list[dict]) -> str:
    """把段数组渲染成可读文本，用于日志和回填的消息文本。"""
    parts: list[str] = []
    for seg in segments:
        stype = seg.get("type")
        data = seg.get("data") or {}
        if stype == "text":
            parts.append(str(data.get("text", "")))
        elif stype == "at":
            parts.append("@全体成员" if data.get("qq") == "all" else f"@{data.get('qq')}")
        elif stype == "face":
            parts.append(f"[表情:{data.get('id')}]")
        elif stype == "image":
            parts.append("[表情包]" if str(data.get("sub_type")) == SUBTYPE_STICKER else "[图片]")
        elif stype == "reply":
            parts.append("")  # 引用不占正文
        else:
            parts.append(f"[{stype}]")
    return "".join(parts).strip()
