"""生成结果的后处理 —— "去 AI 味"。

这一步的效果往往比换模型更明显。规则都是针对中文 LLM 的典型口癖，
按「删 -> 改 -> 判」的顺序处理：

  1. 删除模型爱加的包装（引号、前缀、"回复："）
  2. 删除书面语路标词，压掉重复标点
  3. 切分过长气泡
  4. 判定是否触发"重生成"（AI 味太重）

所有规则都是保守的：宁可少改，不要把正常口语改坏。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..models import Bubble

#: 直接删除的路标词/开场白。**整词删除**，不做替换。
AI_TELLS = (
    "作为一个",
    "作为一个AI",
    "作为人工智能",
    "总的来说",
    "总而言之",
    "需要注意的是",
    "值得一提的是",
    "首先，",
    "其次，",
    "最后，",
    "换句话说",
    "综上所述",
    "希望这能帮到你",
    "希望对你有帮助",
    "有什么我可以帮",
    "如果你还有其他问题",
    "让我知道",
    "不过话说回来，",
)

#: 命中就认为这轮不该发出去，交给上层重生成。
#: 覆盖各种自称变体 —— 中文模型的说法五花八门，漏一个就漏一类。
REGEN_MARKERS = (
    "作为AI",
    "作为一个AI",
    "作为人工智能",
    "作为一个人工智能",
    "作为语言模型",
    "我是AI",
    "我是一个AI",
    "我是人工智能",
    "我是一个人工智能",
    "我是语言模型",
    "我是一个语言模型",
    "我是机器人",
    "我是一个机器人",
    "我是虚拟",
    "我无法",
    "我不能参与",
    "让我们",
    "请注意",
)

#: 句末标点里，真人聊天几乎不用的（保留问号感叹号）。
TRAILING_PERIOD = re.compile(r"[。.]$")

#: 句内切分点：优先在标点处断，其次在空格。
_SPLIT_AT = re.compile(r"(?<=[。！？!?~～…，,、；;])")

WRAPPER_PREFIX = re.compile(
    r"^\s*(回复|答复|回答|我说|消息)\s*[:：]\s*"
)
LEADING_QUOTE = re.compile(r'^[\s"“”\'‘’「『]+')
TRAILING_QUOTE = re.compile(r'[\s"“”\'‘’」』]+$')
REPEAT_PUNCT = re.compile(r"([!！?？~～。.])\1{2,}")


@dataclass
class PostResult:
    bubbles: list = field(default_factory=list)
    dropped: int = 0
    """被丢弃的气泡数。"""
    regenerated: bool = False
    reasons: list[str] = field(default_factory=list)
    max_chars: int = 40
    max_bubbles: int = 3

    @property
    def messages(self) -> list[str]:
        """兼容旧接口：渲染成文本。"""
        return [b.render_text() for b in self.bubbles]

    @property
    def ok(self) -> bool:
        return bool(self.bubbles)


def clean_text(text: str) -> str:
    """单条气泡的清洗。"""
    if not text:
        return ""
    out = text.strip()
    out = LEADING_QUOTE.sub("", out)
    out = TRAILING_QUOTE.sub("", out)
    out = WRAPPER_PREFIX.sub("", out)

    for tell in AI_TELLS:
        out = out.replace(tell, "")

    # 折叠过度重复的标点：!!! -> !
    out = REPEAT_PUNCT.sub(r"\1", out)

    # 连续空白压成一个空格
    out = re.sub(r"[ \t]{2,}", " ", out)
    # 三条以上换行压成一条
    out = re.sub(r"\n{2,}", "\n", out)
    out = out.strip()

    # 中文聊天几乎不打句末句号
    out = TRAILING_PERIOD.sub("", out)
    return out.strip()


def split_bubble(text: str, max_chars: int) -> list[str]:
    """把过长的一段按标点切成多条。切不动就硬切。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    buf = ""
    for token in _SPLIT_AT.split(text):
        if not token:
            continue
        if len(buf) + len(token) <= max_chars:
            buf += token
        else:
            if buf:
                pieces.append(buf)
            # 单段本身超长：硬切
            while len(token) > max_chars:
                pieces.append(token[:max_chars])
                token = token[max_chars:]
            buf = token
    if buf:
        pieces.append(buf)
    return [p.strip() for p in pieces if p.strip()]


def postprocess_bubbles(
    bubbles: list,
    *,
    max_chars: int = 40,
    max_bubbles: int = 3,
    check_regen: bool = True,
    face_keep_prob: float = 1.0,
) -> PostResult:
    """清洗 + 切分 + 判定。表情包气泡不参与切分，独立成条。

    face_keep_prob < 1 时会对本轮的表情做一次概率筛选 —— 见文件末尾的说明。
    """
    result = PostResult(max_chars=max_chars, max_bubbles=max_bubbles)

    if check_regen:
        joined = "".join(getattr(b, "text", "") or "" for b in bubbles)
        for marker in REGEN_MARKERS:
            if marker in joined:
                result.regenerated = True
                result.reasons.append(f"AI 味标记: {marker}")
                return result

    out: list[Bubble] = []
    for bubble in bubbles:
        text = clean_text(getattr(bubble, "text", ""))
        face = getattr(bubble, "face", None)
        sticker = getattr(bubble, "sticker", None)
        sticker_ref = getattr(bubble, "sticker_ref", None)

        # 表情包独占一条，不做长度切分
        if sticker or sticker_ref:
            out.append(Bubble(text=text, face=face, sticker=sticker, sticker_ref=sticker_ref))
            continue

        # 只有 QQ 表情没有文字 —— 合法的"只回个表情"
        if not text:
            if face:
                out.append(Bubble(face=face))
            else:
                result.dropped += 1
            continue

        pieces = split_bubble(text, max_chars)
        for index, piece in enumerate(pieces):
            # 表情跟着最后一段走（"确实" + [笑哭] 的语序更自然）
            out.append(Bubble(text=piece, face=face if index == len(pieces) - 1 else None))

    if len(out) > max_bubbles:
        result.dropped += len(out) - max_bubbles
        result.reasons.append(f"气泡数超限，截断到 {max_bubbles} 条")
        out = out[:max_bubbles]

    # 纯标点/空白的废话丢掉
    out = [b for b in out if b.sticker or b.sticker_ref or b.face
           or re.search(r"[\w\u4e00-\u9fff]", b.text or "")]

    # 表情频率闸门。
    #
    # 模型倾向于**每轮都配一个表情**：人设例句里带表情，它就会照抄这个模式。
    # 实测连续 6 条回复里 4 条带表情 —— 真人不会这样，表情刷屏比文字还烦。
    # prompt 里写"十条里最多一条"压不住，所以在代码层兜一道。
    #
    # 注意是**整轮保留或整轮不保留**，而不是逐条筛：一轮里配两个表情本来就少见，
    # 逐条筛反而会出现"第一条有第二条没有"的割裂感。
    # 一轮最多一个表情 —— 这条无条件生效，和概率无关。
    # 一轮里连配两个表情很怪，真人不会那么干。
    seen_face = False
    extra_faces = 0
    for b in out:
        if not b.face:
            continue
        if seen_face:
            b.face = None
            extra_faces += 1
        seen_face = True
    if extra_faces:
        result.reasons.append(f"一轮多个表情，多余 {extra_faces} 个已去掉")

    # 概率闸门：模型每轮都想配，但真人不是。
    if face_keep_prob < 1.0 and seen_face:
        if random.random() > max(0.0, face_keep_prob):
            for b in out:
                b.face = None
            result.reasons.append("表情频率闸门：本轮不配表情")

    result.bubbles = out
    if not out:
        result.reasons.append("清洗后没有可发内容")
    return result


def postprocess(
    messages: list[str],
    *,
    max_chars: int = 40,
    max_bubbles: int = 3,
    check_regen: bool = True,
) -> PostResult:
    """兼容接口：纯文本输入。"""
    return postprocess_bubbles(
        [Bubble(text=m) for m in (messages or [])],
        max_chars=max_chars,
        max_bubbles=max_bubbles,
        check_regen=check_regen,
    )
