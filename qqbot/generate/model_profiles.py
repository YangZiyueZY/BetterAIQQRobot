"""模型预适配：把每个模型的"脾气"固化下来。

同一套 OpenAI 兼容协议下，各家行为差别很大。实测（tools/profile_models.py）：

  · `deepseek-flash`        有思维链、填报 reasoning_tokens、**忽略 reasoning_effort**
  · `step-5-preview`        有思维链、**不填 reasoning_tokens**、effort 生效、生产延迟高
  · `step-3.7-flash`        有思维链、**预算紧时返回空正文**（思考吃光 token）
  · `mimo-v2.6-flash`       思维链不稳定、延迟 9s+
  · `deepseek-v4-flash-0731` 最快，但思维链时有时无、逻辑题答错
  · `qwen3.8-27b`           有思维链、填报 tokens、effort 生效、快且逻辑正确

适配的意义：把这些差异编码成"该放哪个档、预算给多少"，
避免每换一个模型都要重新踩一遍坑。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: 延迟档位
FAST = "fast"        # 中位 < 3s，适合群聊回复
MEDIUM = "medium"    # 3~8s，回复勉强、分析合适
SLOW = "slow"        # > 8s，只适合分析或不适合

#: 推荐用途
REPLY = "reply"
ANALYSIS = "analysis"
BOTH = "both"
AVOID = "avoid"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    family: str = ""
    #: 是否返回 reasoning_content
    thinks: bool = False
    #: 是否填报 usage.reasoning_tokens（不填的话统计要按字符数推断）
    reports_reasoning_tokens: bool = False
    #: reasoning_effort 是否真的改变行为（False = 静默忽略，发了等于没发）
    effort_effective: bool = False
    #: 是否支持 response_format=json_object
    json_mode: bool = True
    #: 预算给紧时是否返回空正文（思考吃光 token）→ 必须留足 headroom
    empty_when_tight: bool = False
    #: 基础延迟档（短 prompt）
    latency: str = FAST
    #: 生产链路（带人设+记忆+上下文）的中位延迟，毫秒
    reply_latency_ms: int = 0
    #: 群聊回复的成功率（0~1）
    reply_ok_rate: float = 1.0
    #: 推荐用途
    recommended: str = BOTH
    #: 建议预算
    reply_max_tokens: int = 1200
    analysis_max_tokens: int = 4000
    #: 建议的 reasoning_effort（effort 不生效的模型填了也没用）
    reasoning_effort: str = "high"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def effort_useful(self) -> bool:
        return self.thinks and self.effort_effective

    #: 回复档的成功率门槛。群聊里"偶尔不吭声"和"不说话"是两回事 ——
    #: 6 次里有 1 次空回复就已经很显眼了，所以卡在 95%。
    REPLY_OK_FLOOR = 0.95

    @property
    def reply_ready(self) -> bool:
        return (self.recommended in (REPLY, BOTH)
                and self.reply_ok_rate >= self.REPLY_OK_FLOOR)

    def describe(self) -> str:
        bits = []
        bits.append("有思维链" if self.thinks else "无思维链")
        if self.thinks and not self.reports_reasoning_tokens:
            bits.append("不填报思考token")
        bits.append("effort生效" if self.effort_effective else "effort被忽略")
        if self.empty_when_tight:
            bits.append("预算紧会空正文")
        if self.reply_latency_ms:
            bits.append(f"回复中位{self.reply_latency_ms}ms")
        return "，".join(bits)


#: 实测画像。值来自 tools/profile_models.py 与 tools/bench_reply.py。
PROFILES: dict[str, ModelProfile] = {
    "deepseek-flash": ModelProfile(
        name="deepseek-flash", family="deepseek",
        thinks=True, reports_reasoning_tokens=True, effort_effective=False,
        json_mode=True, empty_when_tight=False, latency=FAST, reply_latency_ms=2895, reply_ok_rate=1.0,
        recommended=BOTH, reply_max_tokens=1200, analysis_max_tokens=4000,
        notes=("n=6 全过，唯一稳定可做回复档的","reasoning_effort 被静默忽略，调它没用",
               "填报 reasoning_tokens，统计准确",
               "综合表现最稳，默认首选"),
    ),
    "step-5-preview": ModelProfile(
        name="step-5-preview", family="step",
        thinks=True, reports_reasoning_tokens=False, effort_effective=True,
        json_mode=True, empty_when_tight=False, latency=SLOW, reply_latency_ms=9500, reply_ok_rate=1.0,
        recommended=ANALYSIS, reply_max_tokens=3000, analysis_max_tokens=8000,
        notes=("n=3 全过但中位 9.5s、最慢 27s","reasoning_effort 真的生效，可以用它控制思考深度",
               "不填报 reasoning_tokens，统计按字符数推断",
               "生产 prompt 下思考很重：短 prompt 1.7s，带人设+上下文可到 10~30s",
               "1200 token 预算会被思考吃光导致空正文 → 回复档别用它",
               "适合分析档（记忆抽取/群复盘），或给足预算"),
    ),
    "step-3.7-flash": ModelProfile(
        name="step-3.7-flash", family="step",
        thinks=True, reports_reasoning_tokens=False, effort_effective=False,
        json_mode=True, empty_when_tight=True, latency=FAST, reply_latency_ms=4990, reply_ok_rate=1.0,
        recommended=BOTH, reply_max_tokens=1800, analysis_max_tokens=5000,
        notes=("n=3 全过；出现过英文碎片（'That？'）","★ 实测预算 120 token 时会返回空正文（finish=length，思考吃光）",
               "所以回复档预算要给到 1500 以上，别贴边",
               "reasoning_effort 被静默忽略"),
    ),
    "mimo-v2.6-flash": ModelProfile(
        name="mimo-v2.6-flash", family="mimo",
        thinks=False, reports_reasoning_tokens=False, effort_effective=False,
        json_mode=True, empty_when_tight=False, latency=SLOW, reply_latency_ms=5011, reply_ok_rate=1.0,
        recommended=ANALYSIS,
        notes=("n=3 全过；闲聊场景有轻微跑题","基础延迟 9s+，群聊回复等不起",
               "思维链字段时有时无，不稳定",
               "放着做分析或备用"),
    ),
    "deepseek-v4-flash-0731": ModelProfile(
        name="deepseek-v4-flash-0731", family="deepseek",
        thinks=False, reports_reasoning_tokens=False, effort_effective=True,
        json_mode=True, empty_when_tight=False, latency=FAST, reply_latency_ms=3379, reply_ok_rate=0.67,
        recommended=AVOID,
        notes=("n=6 有 2 次空正文；逻辑题也答错","最快（短 prompt 0.7s）",
               "⚠️ 实测逻辑题答错（问 100 台机器做 100 个零件几分钟，答 1 分钟）",
               "思维链时有时无，行为不稳定",
               "快但不可靠 —— 不建议放进生产值班"),
    ),
    "qwen3.8-27b": ModelProfile(
        name="qwen3.8-27b", family="qwen",
        thinks=True, reports_reasoning_tokens=True, effort_effective=True,
        json_mode=True, empty_when_tight=False, latency=FAST, reply_latency_ms=2513, reply_ok_rate=0.83,
        recommended=BOTH, reply_max_tokens=1200, analysis_max_tokens=4000,
        notes=("n=6 有 1 次空正文","有思维链 + 填报 token + effort 真生效（159→453 字，2.85 倍）",
               "基础延迟 0.9s，逻辑题答对",
               "回复档的强候选"),
    ),
}

#: 没收录的模型用这个兜底
DEFAULT_PROFILE = ModelProfile(
    name="(未知模型)",
    thinks=True, reports_reasoning_tokens=True, effort_effective=True,
    json_mode=True, empty_when_tight=False, latency=MEDIUM,
    recommended=BOTH,
    notes=("没有实测画像，按通用保守值处理",
           "建议跑一遍 tools/profile_models.py 补上"),
)


def profile_for(model: str) -> ModelProfile:
    """取模型画像。支持大小写/前后缀宽松匹配。"""
    if not model:
        return DEFAULT_PROFILE
    key = model.strip()
    if key in PROFILES:
        return PROFILES[key]
    low = key.lower()
    for name, prof in PROFILES.items():
        if name.lower() == low:
            return prof
    # 宽松：模型名常带版本后缀
    for name, prof in PROFILES.items():
        if low.startswith(name.lower()) or name.lower().startswith(low):
            return prof
    return DEFAULT_PROFILE


def recommended_overrides(model: str) -> dict:
    """把画像转成可以直接写进 LLMConfig 的推荐值。"""
    p = profile_for(model)
    out = {
        "reply_max_tokens": p.reply_max_tokens,
        "analysis_max_tokens": p.analysis_max_tokens,
    }
    # effort 不生效的模型，填 high 只是自欺欺人（有些代理还会因此报 400）
    if p.thinks and not p.effort_effective:
        out["reasoning_effort"] = "low"
    elif p.effort_effective:
        out["reasoning_effort"] = p.reasoning_effort
    return out


def advise(reply_model: str, analysis_model: str) -> list[str]:
    """根据当前配置给出警告，供面板和启动日志使用。"""
    warns: list[str] = []
    rp = profile_for(reply_model)
    ap = profile_for(analysis_model)

    if rp.recommended == AVOID:
        warns.append(f"回复档用了 {rp.name}，实测不可靠：{rp.notes[0] if rp.notes else ''}")
    elif rp.recommended == ANALYSIS:
        warns.append(f"回复档用了 {rp.name}，它更适合分析档（延迟高）")

    if rp.latency == SLOW:
        warns.append(f"{rp.name} 延迟偏高，群聊回复会显得很迟钝")

    if rp.reply_ok_rate < ModelProfile.REPLY_OK_FLOOR:
        warns.append(f"回复档用了 {rp.name}，实测成功率只有 {rp.reply_ok_rate:.0%}"
                     f"（{rp.notes[0] if rp.notes else ''}）—— 会出现空回复")

    if rp.empty_when_tight:
        warns.append(f"{rp.name} 在预算紧时会返回空正文，"
                     f"回复预算建议 ≥ {rp.reply_max_tokens}（当前建议值）")

    if ap.recommended == REPLY:
        warns.append(f"分析档用了 {ap.name}，它偏快但对复杂推理可能不够深")

    return warns


def all_profiles() -> list[dict]:
    """给面板用的清单。"""
    out = []
    for p in PROFILES.values():
        out.append({
            "name": p.name,
            "family": p.family,
            "thinks": p.thinks,
            "reports_reasoning_tokens": p.reports_reasoning_tokens,
            "effort_effective": p.effort_effective,
            "json_mode": p.json_mode,
            "empty_when_tight": p.empty_when_tight,
            "latency": p.latency,
            "reply_latency_ms": p.reply_latency_ms,
            "reply_ok_rate": p.reply_ok_rate,
            "recommended": p.recommended,
            "reply_max_tokens": p.reply_max_tokens,
            "analysis_max_tokens": p.analysis_max_tokens,
            "describe": p.describe(),
            "notes": list(p.notes),
        })
    return out
