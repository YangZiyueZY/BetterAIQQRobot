"""人设加载、校验与派生。

人设是结构化 YAML（见 persona.yaml）。这里负责：
  - 加载 + 解析 + 校验
  - 派生决定层要用的触发词（名字、外号、口头禅）
  - 派生 prompt 要用的文本块
  - 给面板用的瘦身摘要
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

REQUIRED_TOP = ("identity",)


class PersonaError(ValueError):
    pass


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class Persona:
    raw: str
    data: dict = field(default_factory=dict)
    path: Path | None = None

    # ---- 基本信息 -----------------------------------------------------

    @property
    def identity(self) -> dict:
        return self.data.get("identity") or {}

    @property
    def name(self) -> str:
        return str(self.identity.get("name") or "机器人")

    @property
    def aliases(self) -> list[str]:
        raw = [self.name, *_as_list(self.identity.get("aliases"))]
        out: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        return out

    @property
    def catchphrases(self) -> list[str]:
        style = self.data.get("style") or {}
        return [str(x) for x in _as_list(style.get("catchphrases")) if str(x).strip()]

    @property
    def expert_topics(self) -> list[str]:
        knowledge = self.data.get("knowledge") or {}
        return [str(x) for x in _as_list(knowledge.get("expert")) if str(x).strip()]

    @property
    def basic_topics(self) -> list[str]:
        knowledge = self.data.get("knowledge") or {}
        return [str(x) for x in _as_list(knowledge.get("basic")) if str(x).strip()]

    @property
    def few_shots(self) -> list[dict]:
        return [x for x in _as_list(self.data.get("few_shot")) if isinstance(x, dict)]

    @property
    def forbidden(self) -> list[str]:
        return [str(x).strip() for x in _as_list(self.data.get("forbidden")) if str(x).strip()]

    # ---- 派生 ---------------------------------------------------------

    def trigger_words(self) -> list[str]:
        """决定层用来判断"有人在叫我"的词表。"""
        words = set(self.aliases) | set(self.catchphrases)
        return sorted((w for w in words if len(w) >= 2), key=len, reverse=True)

    def topic_keywords(self) -> list[str]:
        """命中"精通领域"的粗粒度关键词。"""
        out: list[str] = []
        for topic in self.expert_topics:
            for part in str(topic).replace("、", ",").replace("/", ",").split(","):
                part = part.strip()
                if part and part not in out:
                    out.append(part)
        return out

    def summary(self) -> dict:
        """给管理面板的瘦身视图，不把整个 YAML 塞过去。"""
        return {
            "name": self.name,
            "age": self.identity.get("age"),
            "city": self.identity.get("city"),
            "occupation": self.identity.get("occupation"),
            "traits": len(_as_list(self.data.get("traits"))),
            "few_shot": len(self.few_shots),
            "expert": len(self.expert_topics),
            "aliases": self.aliases,
            "catchphrases": self.catchphrases,
            "forbidden": len(self.forbidden),
            "mission": 1 if (self.data.get("mission") or {}).get("goal") else 0,
        }

    # ---- prompt 素材 --------------------------------------------------

    def style_block(self) -> str:
        """给人设风格做自然语言化，直接拼进 system prompt。"""
        style = self.data.get("style") or {}
        lines: list[str] = []
        if style.get("sentence_length"):
            lines.append(f"- 句子长度：{style['sentence_length']}")
        if style.get("bubbles"):
            lines.append(f"- 分条习惯：{style['bubbles']}")
        punct = style.get("punctuation")
        if isinstance(punct, dict):
            for key, value in punct.items():
                lines.append(f"- 标点「{key}」：{value}")
        if style.get("emoji_frequency"):
            lines.append(f"- 表情使用：{style['emoji_frequency']}")
        if self.catchphrases:
            lines.append(f"- 口头禅：{'、'.join(self.catchphrases)}")
        if style.get("disfluency"):
            lines.append(f"- 口语特征：{style['disfluency']}")
        for item in _as_list(style.get("avoid")):
            lines.append(f"- 绝对不要：{item}")
        return "\n".join(lines)

    def traits_block(self) -> str:
        lines: list[str] = []
        for item in _as_list(self.data.get("traits")):
            if isinstance(item, dict):
                lines.append(f"- {item.get('name', '')}")
                if item.get("do"):
                    lines.append(f"  要：{item['do']}")
                if item.get("dont"):
                    lines.append(f"  不要：{item['dont']}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    def few_shot_block(self) -> str:
        lines: list[str] = []
        for item in self.few_shots:
            ctx = item.get("context", "")
            replies = _as_list(item.get("replies"))
            rendered = " / ".join(f"「{r}」" for r in replies)
            lines.append(f"- 当{ctx}时，你会说：{rendered}")
        return "\n".join(lines)

    def mission_block(self) -> str:
        """任务与行为规则。

        为什么单独一段：`traits` 是"她是什么样的人"（静态性格），
        而这里是"她要做什么、按什么流程做"（可执行的行为约定）。
        像讲解型角色，"先确认题干 → 拆考点 → 分步讲 → 提醒易错点"
        这类流程放性格里会被淹没。

        mission:
          goal: 一句话核心任务
          behavior: [行为规则…]
          stages: [初识：… / 熟悉：… / 亲密：…]
        """
        m = self.data.get("mission") or {}
        lines: list[str] = []
        goal = str(m.get("goal") or "").strip()
        if goal:
            lines.append(goal)
        behavior = _as_list(m.get("behavior"))
        if behavior:
            if lines:
                lines.append("")
            lines.append("【具体怎么做】")
            for item in behavior:
                lines.append(f"- {item}")
        stages = _as_list(m.get("stages"))
        if stages:
            if lines:
                lines.append("")
            lines.append("【关系递进】")
            for item in stages:
                lines.append(f"- {item}")
        return "\n".join(lines)

    def knowledge_block(self) -> str:
        knowledge = self.data.get("knowledge") or {}
        lines: list[str] = []
        if self.expert_topics:
            lines.append(f"- 精通：{'、'.join(self.expert_topics)}")
        if self.basic_topics:
            lines.append(f"- 略懂：{'、'.join(self.basic_topics)}")
        if knowledge.get("unknown_behavior"):
            lines.append(f"- 遇到不懂的：{knowledge['unknown_behavior']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 解析 / 校验
# --------------------------------------------------------------------------


def parse(raw: str) -> Persona:
    """解析 YAML 文本，失败抛 PersonaError。"""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PersonaError(f"YAML 语法错误: {exc}") from exc

    if data is None:
        raise PersonaError("人设内容为空")
    if not isinstance(data, dict):
        raise PersonaError("人设顶层必须是映射（key: value）")

    for key in REQUIRED_TOP:
        if key not in data:
            raise PersonaError(f"缺少必需字段: {key}")

    identity = data.get("identity")
    if not isinstance(identity, dict):
        raise PersonaError("identity 必须是映射")
    if not str(identity.get("name") or "").strip():
        raise PersonaError("identity.name 不能为空")

    for optional in ("traits", "style", "knowledge", "few_shot", "forbidden"):
        if optional in data and data[optional] is not None:
            if not isinstance(data[optional], (list, dict)):
                raise PersonaError(f"{optional} 必须是列表或映射")

    return Persona(raw=raw, data=data)


def validate(raw: str) -> tuple[bool, str | None, Persona | None]:
    try:
        return True, None, parse(raw)
    except PersonaError as exc:
        return False, str(exc), None


def load(path: Path) -> Persona:
    if not path.is_file():
        raise PersonaError(f"人设文件不存在: {path}")
    persona = parse(path.read_text(encoding="utf-8"))
    persona.path = path
    log.info("人设已加载: %s（%s）", persona.name, path)
    return persona


def save(path: Path, raw: str) -> Persona:
    """先校验再落盘，避免把坏 YAML 写进去。"""
    persona = parse(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(path)
    persona.path = path
    log.info("人设已保存: %s", path)
    return persona
