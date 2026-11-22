"""内部领域模型。

和 OneBot 协议解耦：协议层只负责把它们造出来，其余模块只认这些类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Message:
    """一条群消息。"""

    group_id: int
    message_id: int
    user_id: int
    nickname: str
    text: str
    segments: list[dict] = field(default_factory=list)
    ts: int = 0
    is_self: bool = False
    reply_to: int | None = None
    """被引用消息的 message_id。"""
    mentions_self: bool = False
    """@ 了机器人自己。"""
    has_image: bool = False
    media_hash: str | None = None
    """图片/语音的内容哈希，用于去重与多模态缓存命中。"""
    reply_to_self: bool = False
    """这条消息的 reply 段指向的是机器人自己发的消息。"""

    @property
    def display(self) -> str:
        return self.text


@dataclass(slots=True)
class StickerRef:
    """一张表情包。`path` 已解析到本地文件。"""

    id: int
    hash: str
    path: str
    mood_tags: list[str] = field(default_factory=list)
    desc: str = ""


@dataclass(slots=True)
class Bubble:
    """一次发言里的**一条**消息。

    真人会分多条发，每条可以是纯文字、文字+QQ表情、或者只甩一个表情包。
    这个结构就是为了表达这件事。
    """

    text: str = ""
    face: str | None = None
    """QQ 系统表情的中文名，如 "笑哭"。见 face_catalog。"""
    sticker: str | None = None
    """模型给出的表情包意图（情绪标签），由 selector 解析成 sticker_ref。"""
    sticker_ref: StickerRef | None = None
    """selector 解析后的具体表情包。"""
    at_user: int | None = None
    reply_to: int | None = None

    @property
    def is_sticker_only(self) -> bool:
        return bool(self.sticker_ref or self.sticker) and not self.text.strip()

    @property
    def sticker_path(self) -> str | None:
        return self.sticker_ref.path if self.sticker_ref else None

    def __bool__(self) -> bool:
        return bool(self.text.strip() or self.face or self.sticker or self.sticker_path)

    def render_text(self) -> str:
        """给日志/回填用的可读形式。"""
        parts = []
        if self.text.strip():
            parts.append(self.text.strip())
        if self.face:
            parts.append(f"[{self.face}]")
        if self.sticker_ref:
            parts.append(f"[表情包#{self.sticker_ref.id}]")
        elif self.sticker:
            parts.append(f"[表情包:{self.sticker}]")
        return " ".join(parts)


@dataclass(slots=True)
class Conversation:
    """蓄水池攒出来的一批上下文，交给决定层/生成层。

    `messages` 是**本批**（防抖窗口内攒的那几条），`history` 是本批之前
    从数据库捞回来的近期消息 —— 含机器人自己发过的。

    分开是因为两者语义不同：决定层要用"本批有几条"判断是不是在刷屏，
    而生成层需要的是连贯的对话。只给批次的话，它隔半分钟就"失忆"了。
    """

    group_id: int
    messages: list[Message]
    trigger: Message
    """触发这次决策的那条消息。"""
    history: list[Message] = field(default_factory=list)
    """本批之前的近期消息，由 pipeline 从库里补齐。"""

    @property
    def last_ts(self) -> int:
        return self.messages[-1].ts if self.messages else 0

    @property
    def all_messages(self) -> list[Message]:
        """历史 + 本批，按时间顺序，按 message_id 去重。"""
        seen: set[int] = set()
        out: list[Message] = []
        for msg in (*self.history, *self.messages):
            if msg.message_id in seen:
                continue
            seen.add(msg.message_id)
            out.append(msg)
        return out

    def context_text(self, limit: int = 6) -> str:
        """最近几条的纯文本，用于话题匹配和记忆召回。"""
        return "\n".join(m.text for m in self.all_messages[-limit:])

    def render(self) -> str:
        lines = []
        for m in self.all_messages:
            who = "我" if m.is_self else m.nickname
            lines.append(f"{who}: {m.text}")
        return "\n".join(lines)


@dataclass(slots=True)
class Decision:
    """决定层的输出 + 留痕字段。"""

    should_reply: bool
    priority: float = 0.0
    delay: float = 0.0
    """发言前额外等待的秒数。"""
    reason: str = ""
    features: dict = field(default_factory=dict)

    # 留痕（写 decisions 表用）
    id: int = 0
    ts: int = 0
    group_id: int = 0
    text: str = ""
    """触发这次决策的消息文本。"""
    draft: str = ""
    """实际发出的内容，多条用换行分隔。"""

    def __bool__(self) -> bool:
        return self.should_reply


@dataclass(slots=True)
class Draft:
    """生成层的输出：一次可以分多条发出。"""

    bubbles: list[Bubble] = field(default_factory=list)
    thought: str = ""
    quote: bool = False
    """模型建议引用触发消息。最终是否引用由群策略的 Thompson 采样决定。"""
    at: bool = False
    """模型建议 @ 触发者。同上。"""
    emoji_like: str | None = None
    """给触发消息贴的表情回应（Unicode 码点字符串）。"""

    @property
    def messages(self) -> list[str]:
        """兼容旧接口：纯文本视图。"""
        return [b.render_text() for b in self.bubbles]

    @property
    def text_bubbles(self) -> list[str]:
        return [b.text for b in self.bubbles if b.text.strip()]

    def __bool__(self) -> bool:
        return any(self.bubbles)
