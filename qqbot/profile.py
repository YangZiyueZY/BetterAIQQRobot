"""群画像：客观统计，不用 LLM，零成本。

这是自进化的地基 —— 不认识这个群，就谈不上融入。

    avg_msg_len    群友平均说多长。**最有效的融入手段**：他们说 8 字，
                   你就别说 40 字。
    hourly         24 小时活跃度分布，比全局作息精确得多。
    core_members   说话最多的几个人 = 意见领袖，他们带节奏时更容易接话。
    emoji_rate     多少比例的消息带 QQ 表情 —— 决定她用不用表情。
    topics         主题分布，由 LLM 复盘补充（refresh_topics）。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter

from .config import EvolutionConfig
from .store.evolution import GroupProfile, ProfileRepo
from .store.repo import Repo

log = logging.getLogger(__name__)

SAMPLE_SIZE = 2000

#: 极简情绪词典。够用就好 —— 精准情感分析不值得在这里花成本。
NEGATIVE_WORDS = ("烦", "恶心", "滚", "骂", "傻", "蠢", "垃圾", "吵", "闭嘴",
                  "无语", "崩", "难受", "生气", "气死", "讨厌", "失望", "算了")
POSITIVE_WORDS = ("哈哈", "笑死", "好耶", "开心", "赞", "牛", "厉害", "喜欢",
                  "谢谢", "可爱", "舒服", "爽", "绝了", "可以", "不错")

QUESTION_RE = re.compile(r"[?？]|吗|呢|怎么|为什么|如何|是不是|有没有|能不能|几个|多少")


def _is_question(text: str) -> bool:
    return bool(QUESTION_RE.search(text or ""))


class ProfileBuilder:
    def __init__(self, cfg: EvolutionConfig, profiles: ProfileRepo, repo: Repo) -> None:
        self.cfg = cfg
        self.profiles = profiles
        self.repo = repo
        self.rebuilds = 0

    async def rebuild(self, group_id: int) -> GroupProfile | None:
        """从最近的群消息重算画像。"""
        messages = await self.repo.messages_since(
            group_id, 0, limit=SAMPLE_SIZE
        )
        # messages_since 是升序取前 N 条；我们要的是最近 N 条
        messages = messages[-SAMPLE_SIZE:]

        human = [m for m in messages if not m.is_self and m.text.strip()]
        if len(human) < 20:
            return None

        profile = await self.profiles.get(group_id)
        profile.group_id = group_id
        profile.sample_size = len(human)

        lengths = [len(m.text) for m in human]
        profile.avg_msg_len = sum(lengths) / len(lengths)

        emoji = sticker = question = 0
        speaker = Counter()
        names: dict[int, str] = {}
        hourly = [0.0] * 24
        pos = neg = 0
        word_counter: Counter = Counter()

        for msg in human:
            speaker[msg.user_id] += 1
            names.setdefault(msg.user_id, msg.nickname)
            hour = (msg.ts // 3600) % 24
            hourly[hour] += 1

            segs = msg.segments or []
            if any(s.get("type") == "face" for s in segs if isinstance(s, dict)):
                emoji += 1
            if any(isinstance(s, dict) and s.get("type") == "image"
                   and str((s.get("data") or {}).get("sub_type")) == "1" for s in segs):
                sticker += 1
            if _is_question(msg.text):
                question += 1

            pos += sum(1 for w in POSITIVE_WORDS if w in msg.text)
            neg += sum(1 for w in NEGATIVE_WORDS if w in msg.text)

            for token in re.findall(r"[\u4e00-\u9fff]{2,6}|[a-zA-Z]{3,}", msg.text):
                word_counter[token] += 1

        total = len(human)
        profile.emoji_rate = emoji / total
        profile.sticker_rate = sticker / total
        profile.question_rate = question / total
        profile.hourly = hourly
        profile.atmosphere = max(-1.0, min(1.0, (pos - neg) / max(6.0, total * 0.15)))

        profile.core_members = [
            {"user_id": uid, "nickname": names.get(uid, str(uid)), "count": n}
            for uid, n in speaker.most_common(8)
        ]

        # 高频词里挑出"不像通用中文"的，当黑话候选
        profile.slang = self._slang_candidates(word_counter, total, profile.slang)

        await self.profiles.save(profile)
        self.rebuilds += 1
        log.info(
            "群画像已更新 [%s] 样本%d 平均%.0f字 表情%.0f%% 表情包%.0f%% 氛围%.2f",
            group_id, total, profile.avg_msg_len,
            profile.emoji_rate * 100, profile.sticker_rate * 100, profile.atmosphere,
        )
        return profile

    @staticmethod
    def _slang_candidates(counter: Counter, total: int,
                          existing: dict) -> dict:
        """把出现频率异常高、但明显不是常用词的挑出来做黑话候选。

        这里只做**候选**筛选，真正的释义交给 LLM 复盘补。已有释义保留。
        """
        out = dict(existing or {})
        for word, count in counter.most_common(40):
            if word in out:
                continue
            # 一个词被反复说，且它不是那种一眼就懂的通用词
            if count >= max(6, total * 0.02) and len(word) <= 6:
                out.setdefault(word, "")
        # 丢掉释义为空的（LLM 还没来得及补），保持 prompt 干净
        return {k: v for k, v in out.items() if v}

    def has_topics(self, profile: GroupProfile) -> bool:
        return bool(profile.topics)

    async def apply_topics(self, group_id: int, payload: dict) -> GroupProfile:
        """把 LLM 复盘出的主题/黑话/氛围写回画像。"""
        profile = await self.profiles.get(group_id)
        topics = payload.get("topics")
        if isinstance(topics, dict) and topics:
            cleaned = {}
            for key, value in list(topics.items())[:6]:
                try:
                    cleaned[str(key)[:20]] = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
            if cleaned:
                profile.topics = cleaned

        slang = payload.get("slang")
        if isinstance(slang, dict):
            merged = dict(profile.slang)
            for key, value in list(slang.items())[:10]:
                if str(key).strip() and str(value).strip():
                    merged[str(key)[:20]] = str(value)[:40]
            profile.slang = merged

        try:
            atmosphere = float(payload.get("atmosphere"))
            profile.atmosphere = max(-1.0, min(1.0, atmosphere))
        except (TypeError, ValueError):
            pass

        await self.profiles.save(profile)
        return profile

    def stats(self) -> dict:
        return {"profile_rebuilds": self.rebuilds}
