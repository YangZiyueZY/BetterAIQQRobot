"""提示词注入防护。

群聊是个不可信输入源 —— 任何人都能往群里发一段"忽略你之前的指令"。
而且这类文本一旦进了对话历史，会在后续**每一条**消息的上下文里反复出现。

**刻意采用"渲染时擦除"而不是"入库时擦除"**：
  · 数据库里保留原文 —— 面板上能看到谁发了什么，便于追查
  · 但模型永远看不到被擦除的内容

判定策略偏保守：正常群聊消息又短又随意，只有"够长 **且** 含多条指令性
特征"才判为注入。宁可漏掉一次，也不要把群友认真打的字屏蔽掉。
"""

from __future__ import annotations

import re

#: 长文本里出现这些就高度可疑
MARKERS: tuple[str, ...] = (
    "ignore previous", "ignore all previous", "ignore the above",
    "disregard", "you are now", "new instructions", "system prompt",
    "jailbreak", "dan mode", "mode: sandbox", "[mode:", "prompt-bank",
    "refusal string", "failing answer", "safety framing",
    "不要拒绝", "忽略之前", "忽略上述", "忽略以上", "新的指令",
    "开发者模式", "越狱", "绕过限制", "解除限制",
)

#: 指令性句式 —— **只认"对着 AI 下命令"的元指令**。
#:
#: 曾经写成 `you (must|should|will)`，结果把正常英文文档全擦了：
#: StackOverflow 上"合并两个字典"那篇的正文里有 "only the value y[k] should ..."，
#: 打分直接 6 分被当成注入丢掉。而搜索返回的大多是英文页面 ——
#: 等于技术问题的资料全被静默扔掉。
#:
#: 判别标准：注入是**在指挥 AI 的行为**，不是普通的技术说明。
INSTRUCTION_RE = re.compile(
    r"(do not (refuse|decline|apologize|redirect|start your reply with)"
    r"|(any|a|your) refusal[^.]{0,40}(is|will be|counts? as)[^.]{0,30}"
    r"(fail|wrong|bad)"
    r"|response rules"
    r"|execute the (original )?(request|task) directly"
    r"|treat all[^.]{0,40}as (synthetic|local fixtures?)"
    r"|fill (it |them )?with[^.]{0,25}(template|placeholder)"
    r"|answer in the user'?s language"
    r"|keep responses? (compact|short and direct)"
    r"|silently treat"
    r"|you are (now )?(a|an) [a-z ]{0,24}(without|with no|free of)"
    r" ?(restrictions?|limits?|filters?|rules?)"
    r"|from now on,? you (are|will|must)"
    r"|ignore (all )?(previous|prior|above) (instructions?|prompts?|rules?))",
    re.IGNORECASE,
)

#: 高特异性句式：**不看长度**，命中即擦。
#: 关键是必须同时含"指令/提示/规则"这类宾语 ——
#: "忽略我刚才说的"是正常更正，"忽略之前的所有指令"才是攻击。
ALWAYS_SCRUB_RE = re.compile(
    r"忽略(之前|上述|以上|前面|所有)?的?(所有)?(指令|提示词|提示|设定|规则|约束)"
    r"|无视(之前|上述|以上)(的)?(指令|提示|设定|规则)"
    r"|(解除|绕过|关闭|去掉)(你的)?(所有)?(限制|安全|约束|审查)"
    r"|(进入|开启)(开发者|上帝|无限制)模式"
    r"|jailbreak|\bdan mode\b|mode:\s*sandbox"
    r"|ignore (all )?(previous|prior|above) (instructions?|prompts?|rules?)",
    re.IGNORECASE,
)

#: 短于这个长度的一律不判（闲聊本来就短，不值得动它）
MIN_SUSPECT_LEN = 80

#: 命中阈值：标记数 + 句式数 达到这个值才擦除。
#: 单独的 'you should' 之类普通英文一律不算信号 —— 见 INSTRUCTION_RE 的说明。
HIT_THRESHOLD = 2

PLACEHOLDER = "［这段文字与聊天无关，已忽略］"


def score(text: str) -> int:
    """可疑度打分。0 表示正常。"""
    if not text or len(text) < MIN_SUSPECT_LEN:
        return 0
    low = text.lower()
    hits = sum(1 for m in MARKERS if m in low)
    hits += len(INSTRUCTION_RE.findall(text))
    return hits


def looks_like_injection(text: str) -> bool:
    if not text:
        return False
    # 高特异性句式不看长度
    if ALWAYS_SCRUB_RE.search(text):
        return True
    return score(text) >= HIT_THRESHOLD


def scrub(text: str) -> tuple[str, bool]:
    """返回 (擦除后的文本, 是否发生了擦除)。"""
    if not looks_like_injection(text):
        return text, False
    return PLACEHOLDER, True


def scrub_many(texts: list[str]) -> tuple[list[str], int]:
    """批量擦除，返回 (结果, 擦除条数)。"""
    out: list[str] = []
    removed = 0
    for text in texts:
        cleaned, hit = scrub(text)
        out.append(cleaned)
        removed += int(hit)
    return out, removed
