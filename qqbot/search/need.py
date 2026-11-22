"""时效性判断：这条消息值不值得去搜。

这是整个搜索功能里最关键的模块 —— 不是"能不能搜"，而是"**什么时候该搜**"。
群里每句话都去搜既慢又蠢（搜索挂在一句话的回复链路上，慢了这个回复就不像人），
真人也不会为了回一句"哈哈"去查百度。

判定思路：**宁可漏搜，不要乱搜**。
  漏搜的代价 = 回答不够新（模型还能糊弄过去）
  乱搜的代价 = 每条消息慢两三秒 + 花钱 + 群里显得很怪

所以阈值偏高，且只有"明确的信息需求"才过。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 明确让人去查 —— 命中直接搜，不用算分
EXPLICIT_RE = re.compile(
    r"(搜一下|搜搜|搜索一下|查一下|查查|帮我查|帮我搜|你查查|百度一下|百度下"
    r"|谷歌一下|google 一下|上网查|网上查|查资料|搜资料"
    r"|搜\s*索|look ?it ?up|search for)",
    re.IGNORECASE,
)

#: 时间敏感词：这些话题上，模型的知识很可能已经过期
TIME_SENSITIVE: tuple[str, ...] = (
    # 时间指向
    "今天", "昨天", "明天", "现在", "目前", "当前", "这几天", "最近",
    "这周", "本周", "这个月", "本月", "今年", "上周", "上个月", "刚才", "刚刚",
    # 实时数据
    "天气", "气温", "下雨", "台风", "空气质量", "限号",
    "股价", "汇率", "油价", "房价", "票价", "多少钱", "价格多少",
    "比分", "赛程", "战况", "排名", "榜", "销量", "票房",
    # 新闻与事件
    "新闻", "发生了什么", "怎么回事", "出事了", "最新消息", "官宣",
    # 版本与发布
    "最新版本", "最新版", "更新了", "发布了", "上线了", "什么时候出",
    "什么时候发布", "出了吗", "新版", "更新的",
)

#: 问句形态：同样长度下，问句更可能需要外部信息
QUESTION_RE = re.compile(
    r"[?？]|吗$|吗[，,。！!]|呢$|呢[，,。！!]"
    r"|怎么|为什么|为啥|什么时候|多少钱|多少|是不是|有没有|能不能|会不会|哪个"
)

#: 明确不需要搜的：纯情绪、纯接梗
NOISE_RE = re.compile(
    r"^(哈+|草+|草|6+|牛+|好+|行+|嗯+|哦+|在吗|在不在|早|晚安|谢谢|谢了|收到|好滴|好嘞)+$"
)

#: 消息里已经有链接，多半是在分享而不是在问
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

#: 太短的消息不搜（"在吗""咋了"这种）
MIN_LEN = 5

#: 一条消息里最多认几个时间敏感词，避免长文本堆分
MAX_TIME_HITS = 3

WEIGHT_EXPLICIT = 1.0
WEIGHT_TIME = 0.5
WEIGHT_QUESTION = 0.25
PENALTY_NOISE = -0.5
PENALTY_URL = -0.3


@dataclass(slots=True)
class SearchNeed:
    needed: bool
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    query: str = ""

    def __bool__(self) -> bool:
        return self.needed


def build_query(text: str, extra: str = "", strip_words=None) -> str:
    """把消息整理成搜索词。

    去噪三件事：
      1. @ 和表情占位符
      2. **机器人自己的名字** —— 「小七今天天气怎么样」拿去搜，
         "小七"会污染结果；真人是「今天天气怎么样」
      3. 多余空白
    """
    q = text or ""
    q = re.sub(r"@全体成员|@我|@\d+", " ", q)
    q = q.replace("@", " ")        # 剩下的裸 @昵称：留着 @ 对搜索没意义
    q = re.sub(r"\[[^\]]{1,12}\]", " ", q)          # [图片] [表情] 这类占位符

    # 长词优先，避免「小七」先被删掉导致「七七」残留之类的问题
    for word in sorted(strip_words or [], key=len, reverse=True):
        word = (word or "").strip()
        if len(word) >= 2:
            q = q.replace(word, " ")

    q = re.sub(r"\s+", " ", q).strip()
    if extra:
        q = f"{q} {extra}".strip()
    return q[:100] or (text or "")[:100]


def evaluate(text: str, *, threshold: float = 0.5,
             has_url: bool = False, strip_words=None) -> SearchNeed:
    """判断这条消息是否需要联网搜索。

    保守优先：只有分数过阈值才搜。
    """
    raw = (text or "").strip()
    need = SearchNeed(needed=False, query=build_query(raw, strip_words=strip_words))

    # 明确让人去查：直接通过。
    # **必须排在长度门槛之前** —— "百度一下""查查呗"都很短，
    # 按长度先拦掉的话，最该搜的那些话反而搜不了。
    if EXPLICIT_RE.search(raw):
        need.needed = True
        need.score = WEIGHT_EXPLICIT
        need.reasons.append("明确要求搜索")
        return need

    if len(raw) < MIN_LEN:
        need.reasons.append(f"太短({len(raw)}字)")
        return need

    # 已经在分享链接了，多半不是要查
    if has_url or URL_RE.search(raw):
        need.reasons.append("消息里已有链接")
        return need

    score = 0.0
    reasons: list[str] = []

    hits = [w for w in TIME_SENSITIVE if w in raw][:MAX_TIME_HITS]
    if hits:
        score += WEIGHT_TIME * len(hits)
        reasons.append("时效性词(" + "、".join(hits) + ")")

    if QUESTION_RE.search(raw):
        score += WEIGHT_QUESTION
        reasons.append("疑问句")

    if NOISE_RE.match(raw):
        score += PENALTY_NOISE
        reasons.append("纯情绪/寒暄")

    need.score = max(0.0, score)
    need.reasons = reasons
    need.needed = need.score >= threshold
    if not need.needed and not reasons:
        reasons.append("没有信息需求")
    return need


def describe(need: SearchNeed) -> str:
    if need.needed:
        return f"要搜(分数{need.score:.2f}: {'、'.join(need.reasons)})"
    return f"不搜({need.reasons[0] if need.reasons else '无信号'})"
