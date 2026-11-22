"""Prompt 组装。

分两层：
  - system：人设 + 硬性约束 + 可用表情清单 + 输出格式。**尽量静态**，便于供应商侧缓存。
  - user：当前场景 + 群画像 + 记忆 + 对话上下文 + 触发消息。

输出是**结构化 JSON**，能表达真人发言的全部形态：
    纯文字 / 文字+QQ表情 / 只发QQ表情 / 只甩表情包 / 分多条 / 引用 / @
"""

from __future__ import annotations

import logging

from .. import prompts
from ..config import BehaviorConfig
from ..face_catalog import FaceCatalog
from ..models import Conversation, Message
from ..persona import Persona
from ..safety import injection

log = logging.getLogger(__name__)

#: 已经告警过的 message_id。
#: 注入文本一旦进了历史，会在 30 条窗口里被反复擦除、反复触发日志 ——
#: 每条消息只告警一次，否则会把日志刷满，还会让人误以为是自己刚发的话被拦了。
_warned: set[int] = set()
_WARNED_MAX = 500


def _warn_injection_once(message_id: int, who: str, snippet: str) -> None:
    if message_id in _warned:
        return
    if len(_warned) >= _WARNED_MAX:
        _warned.clear()
    _warned.add(message_id)
    # 说清是"哪条消息"被擦 —— 含糊的告警会让人误判成自己刚发的消息被拦了
    log.warning(
        "擦除提示词注入：消息 #%s 来自「%s」，内容开头 %r（原文仍在数据库，仅供参考）",
        message_id, who, snippet,
    )

def build_system(
    persona: Persona,
    behavior: BehaviorConfig,
    face_catalog: FaceCatalog | None = None,
    *,
    sticker_moods: list[str] | None = None,
    group_id: int = 0,
) -> str:
    """拼系统提示词。

    每段的正文都来自 prompts 注册表（可在面板查看/修改），
    这里只负责**填数据**和**决定顺序**。

    group_id 用于取该群的提示词覆盖 —— 不同群可以用不同的长度约束、
    不同的行为规则（比如讲解型人设不能有"沉默是常态"）。
    """
    parts: list[str] = []

    # 身份：把 persona.identity 各字段拼成一句话
    ident = persona.identity
    identity_bits: list[str] = []
    if ident.get("age"):
        identity_bits.append(f"你 {ident['age']} 岁。")
    if ident.get("gender"):
        identity_bits.append(f"性别{ident['gender']}。")
    if ident.get("city"):
        identity_bits.append(f"在{ident['city']}。")
    if ident.get("occupation"):
        identity_bits.append(f"{ident['occupation']}。")
    if ident.get("backstory"):
        identity_bits.append(str(ident["backstory"]).strip())
    parts.append(prompts.render_part(
        "sys.identity", group_id,
        name=persona.name,
        identity="".join(identity_bits),
    ))

    # 下面这些是 persona.yaml 的结构化字段，由 Persona 负责渲染成文本
    traits = persona.traits_block()
    if traits:
        parts.append("【性格】\n" + traits)

    style = persona.style_block()
    if style:
        parts.append("【说话风格】\n" + style)

    knowledge = persona.knowledge_block()
    if knowledge:
        parts.append("【你懂的】\n" + knowledge)

    few_shot = persona.few_shot_block()
    if few_shot:
        parts.append("【模仿这些例子说话】\n" + few_shot)

    # 任务与行为规则放在性格之后 —— 先知道她是谁，再知道她要干什么
    mission = persona.mission_block()
    if mission:
        parts.append("【你的任务】\n" + mission)

    parts.append(prompts.render_part("sys.human_rules", group_id))

    forbidden = persona.forbidden
    if forbidden:
        parts.append("【禁止】\n" + "\n".join(f"- {x}" for x in forbidden))

    parts.append(prompts.render_part(
        "sys.length_rules", group_id,
        max_chars=behavior.max_chars_per_bubble,
        max_bubbles=behavior.max_bubbles,
    ))

    if face_catalog is not None:
        face_list = face_catalog.prompt_block()
        if face_list:
            parts.append(prompts.render_part(
                "sys.face_rules", group_id, face_list=face_list))

    if sticker_moods:
        parts.append(prompts.render_part(
            "sys.sticker_rules", group_id, moods="、".join(sticker_moods)))

    parts.append(prompts.render_part("sys.output_rules", group_id))

    return "\n\n".join(parts)


def _scrub_block(items: list[str]) -> tuple[list[str], int]:
    """对一批不可信文本做注入擦除，返回 (结果, 命中数)。"""
    out: list[str] = []
    hits = 0
    for item in items:
        cleaned, hit = injection.scrub(item)
        out.append(cleaned)
        hits += int(hit)
    return out, hits


def render_conversation(conv: Conversation, *, max_chars: int = 2000) -> str:
    """渲染成对话文本。自己发的标成"我"。

    用 all_messages（历史 + 本批），而不是只渲染本批 —— 否则模型看不到上文。
    太长时从**前面**截断，因为越近的话越重要。
    """
    lines: list[str] = []
    scrubbed: list[tuple[int, str, str]] = []
    for msg in conv.all_messages:
        who = "我" if msg.is_self else msg.nickname
        text = msg.text
        # 群聊是不可信输入源：有人会发"忽略你之前的指令"这类文本，
        # 而且一旦进了历史，会在后续每条消息的上下文里反复出现。
        # 这里擦掉，但数据库里保留原文供追查。
        if not msg.is_self:
            text, hit = injection.scrub(text)
            if hit:
                scrubbed.append((msg.message_id, who, msg.text[:60].replace("\n", " ")))
        if msg.mentions_self and not msg.is_self:
            text = f"（@了你）{text}"
        if msg.reply_to_self and not msg.is_self:
            text = f"（回复了你）{text}"
        elif msg.reply_to and not msg.is_self:
            text = f"（回复了某条消息）{text}"
        lines.append(f"{who}: {text}")

    # 超出预算就从头砍，保留最近的
    total = sum(len(x) + 1 for x in lines)
    while lines and total > max_chars:
        total -= len(lines.pop(0)) + 1
    if not lines:
        return "（暂无上下文）"
    for message_id, who, snippet in scrubbed:
        _warn_injection_once(message_id, who, snippet)
    return "\n".join(lines)


def build_profile_note(profile, limit_topics: int = 3) -> str:
    """把群画像翻译成一句"这个群是什么群"的说明，喂给模型。"""
    if profile is None:
        return ""
    bits: list[str] = []

    topics = sorted(profile.topics.items(), key=lambda kv: -kv[1])[:limit_topics]
    if topics:
        bits.append("这个群主要聊：" + "、".join(f"{t}" for t, _ in topics))
    if profile.avg_msg_len:
        bits.append(f"大家说话平均 {profile.avg_msg_len:.0f} 个字")
    peaks = profile.peak_hours()
    if peaks:
        bits.append("最活跃时段：" + "、".join(f"{h}点" for h in peaks))
    if profile.slang:
        pairs = list(profile.slang.items())[:6]
        bits.append("群内黑话：" + "、".join(f"{k}={v}" for k, v in pairs))
    if profile.core_members:
        names = [m.get("nickname") for m in profile.core_members[:4] if m.get("nickname")]
        if names:
            bits.append("核心成员：" + "、".join(names))
    return "；".join(bits)


def build_user(
    conv: Conversation,
    persona: Persona,
    *,
    memories: list[str] | None = None,
    search_notes: str = "",
    video_notes: str = "",
    presence: str = "在线",
    mood_text: str = "",
    now_text: str = "",
    is_mention: bool = False,
    profile_note: str = "",
    media_notes: list[str] | None = None,
    sticker_moods: list[str] | None = None,
    group_id: int | None = None,
) -> str:
    gid = conv.group_id if group_id is None else group_id
    parts: list[str] = []

    scene: list[str] = []
    if now_text:
        scene.append(f"现在是 {now_text}。")
    scene.append(f"你当前的状态：{presence}。")
    if mood_text:
        scene.append(f"你的心情：{mood_text}。")
    parts.append("【场景】" + " ".join(scene))

    if profile_note:
        # 群画像里的主题/黑话是 LLM 从群聊里总结的，同样可能被投毒
        safe_note, dropped = _scrub_block([profile_note])
        if dropped:
            log.warning("群画像文本中擦除了疑似注入")
        parts.append("【这是个什么群】" + safe_note[0])

    # 下面三块的内容都源自不可信输入，逐条过一遍注入过滤。
    # 尤其是 media_notes —— 图片里的文字（OCR）是完全由群友控制的，
    # 在图片里塞越狱文本能绕过"只检查消息文本"的那道闸门。
    if memories:
        safe_mem, dropped = _scrub_block(memories)
        if dropped:
            log.warning("记忆召回中擦除了 %d 条疑似注入", dropped)
        parts.append("【你还记得的事】\n" + "\n".join(f"- {m}" for m in safe_mem))

    if media_notes:
        safe_media, dropped = _scrub_block(media_notes)
        if dropped:
            log.warning("图片文字中擦除了 %d 条疑似注入（OCR 也可能是攻击载荷）", dropped)
        parts.append("【群友发的图】\n" + "\n".join(f"- {m}" for m in safe_media))

    parts.append("【最近的群聊】\n" + render_conversation(conv))

    if video_notes:
        # 措辞很关键。第一版写的是"用来理解他们在聊什么"+"不要照搬"，
        # 结果模型读成"这是背景资料，别用"，回了一句"我这打不开啊"。
        # 必须明确：**你已经看过了**，而且**不许装没看过**。
        parts.append(
            "【群友发的视频内容】（你已经看过了）\n" + video_notes + "\n"
            + prompts.render_part("user.video_note", gid)
        )

    if search_notes:
        # 位置很关键：放在对话之后、提问之前 —— 离触发最近，最不容易被忽略。
        # 措辞也要明确"该用就用"，否则人设里那句"不懂就承认不懂"会压过它，
        # 出现"查了但还说不知道"的浪费。
        parts.append(
            "【刚查到的实时资料】\n" + search_notes + "\n"
            + prompts.render_part("user.search_note", gid)
        )

    trigger = conv.trigger
    key = "user.mention_hint" if is_mention else "user.normal_hint"
    parts.append(prompts.render_part(
        key, gid, nickname=trigger.nickname, text=trigger.text))

    if sticker_moods:
        parts.append(prompts.render_part(
            "user.sticker_hint", gid, moods="、".join(sticker_moods)))

    return "\n\n".join(parts)


def build_memory_prompt(existing: list[str], transcript: str, persona: Persona,
                        group_id: int = 0) -> tuple[str, str]:
    """长期记忆抽取。要求三维打分而不是简单的"有用/没用"。"""
    system = prompts.render_part("task.memory", group_id, name=persona.name)
    user = ""
    if existing:
        user += "【已经记住的，不要重复】\n" + "\n".join(f"- {m}" for m in existing) + "\n\n"
    user += "【新的群聊片段】\n" + transcript
    return system, user


def build_profile_prompt(transcript: str, current_topics: dict,
                         group_id: int = 0) -> tuple[str, str]:
    """群画像复盘的 prompt：识别主题、黑话、氛围。"""
    system = prompts.render_part("task.profile", group_id)
    user = ""
    if current_topics:
        user += f"【当前的主题分布，供参考】{current_topics}\n\n"
    user += "【群聊内容】\n" + transcript
    return system, user


def summarize_message(msg: Message, limit: int = 60) -> str:
    text = msg.text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
