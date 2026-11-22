"""发送前的硬拦截层。

这是代码层面的最后一道闸，**不能指望 prompt 兜底**。原则是 fail-closed：
审核组件自身出任何异常，宁可这一轮不说话。

拦截顺序：空白 -> 黑名单 -> 长度/条数 -> 复读
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import BehaviorConfig

log = logging.getLogger(__name__)

#: 内置兜底词条。刻意保持宽泛，具体边界交给用户可编辑的 safety.txt 扩展。
#: 这里侧重"诈骗话术"和"危险内容"，而不是观点表达。
BUILTIN_BLOCKLIST: tuple[str, ...] = (
    "自杀",
    "自残",
    "毒品",
    "冰毒",
    "海洛因",
    "博彩",
    "赌博网站",
    "洗钱",
    "代开发票",
    "刷单返利",
    "包过",
    "帮你贷款",
    "帮我贷款",
    "加微信转账",
    "扫码转账",
    "先转账",
    "裸聊",
    "约炮",
    "开房",
    "身份证号",
    "银行卡号",
    "验证码发我",
    "把验证码",
)

URL_RE = re.compile(r"https?://|www\.|\.com/|\.cn/")
HAS_WORD = re.compile(r"[\w\u4e00-\u9fff]")


@dataclass
class SafetyResult:
    ok: bool
    messages: list[str] = field(default_factory=list)
    reason: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


class SafetyFilter:
    def __init__(
        self,
        cfg: BehaviorConfig,
        blocklist: set[str] | None = None,
        *,
        block_urls: bool = True,
    ) -> None:
        self.cfg = cfg
        self.blocklist = set(BUILTIN_BLOCKLIST) | (blocklist or set())
        self.block_urls = block_urls
        self.blocked_count = 0

    # ---- 装载 ---------------------------------------------------------

    @classmethod
    def load(cls, cfg: BehaviorConfig, path: Path | None = None) -> "SafetyFilter":
        """从 safety.txt 读取用户自定义词条，每行一个，# 开头为注释。"""
        words: set[str] = set()
        if path is not None and path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                words.add(line)
            log.info("安全词库已加载: %d 条自定义词条（%s）", len(words), path)
        return cls(cfg, words)

    # ---- 检查 ---------------------------------------------------------

    def check(self, messages: list[str], recent_self: list[str] | None = None) -> SafetyResult:
        try:
            return self._check(messages, recent_self or [])
        except Exception as exc:  # noqa: BLE001 - fail-closed
            log.exception("安全过滤异常，本轮静默")
            return SafetyResult(ok=False, reason=f"过滤器异常: {exc}")

    def _check(self, messages: list[str], recent_self: list[str]) -> SafetyResult:
        cleaned = [m.strip() for m in (messages or []) if m and m.strip()]
        if not cleaned:
            return SafetyResult(ok=False, reason="没有可发内容")

        max_chars = max(1, self.cfg.max_chars_per_bubble)
        max_bubbles = max(1, self.cfg.max_bubbles)

        kept: list[str] = []
        for text in cleaned[:max_bubbles]:
            haystack = _normalize(text)
            for word in self.blocklist:
                if word and _normalize(word) in haystack:
                    self.blocked_count += 1
                    log.warning("命中黑名单词「%s」，丢弃本轮", word)
                    return SafetyResult(ok=False, reason=f"命中黑名单（{word}）")
            if self.block_urls and URL_RE.search(text):
                self.blocked_count += 1
                return SafetyResult(ok=False, reason="包含链接")
            if len(text) > max_chars:
                text = text[:max_chars]
            if not HAS_WORD.search(text):
                continue
            kept.append(text)

        if not kept:
            return SafetyResult(ok=False, reason="清洗后无有效内容")

        # 复读检测：与最近自己说过的话高度重合就丢弃
        recent_norm = {_normalize(t) for t in recent_self}
        deduped = [t for t in kept if _normalize(t) not in recent_norm]
        if not deduped:
            return SafetyResult(ok=False, reason="与最近发言重复")

        return SafetyResult(ok=True, messages=deduped)
