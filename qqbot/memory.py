"""长期记忆 2.0：带类型、会遗忘、能自我修正。

和 v1.0 的扁平列表最大的区别：

  · **分类型**：不同类型半衰期不同。"本群管打游戏叫上班"近乎永久，
    "上周六联机到两点"只有几天。
  · **三维打分**：不是"有用/没用"二分类，而是
        persistence × reusability × non_triviality
    低于阈值直接丢弃 —— 宁可少记，不要凑数。
  · **矛盾消解**：新人说"张三考完了"，旧记忆"张三在准备考研"就该失效。
    真人不会同时相信两件矛盾的事。失效不删除，保留演化历史。
  · **来源置信度**：别人明确说的=0.9，自己推断的=0.5，召回时降权。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time

from .config import BehaviorConfig, EvolutionConfig
from .generate.llm import LLMError, extract_json
from .generate.prompt import build_memory_prompt
from .models import Message
from .store.memories import TTL_BY_TYPE, MemoryRepo

log = logging.getLogger(__name__)

ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
CJK = re.compile(r"[\u4e00-\u9fff]")

#: 低于这个 usefulness 丢弃
MIN_USEFULNESS = 0.45

#: 判定语义相近的阈值（用于矛盾检测）
SIMILAR_THRESHOLD = 0.55
DUP_THRESHOLD = 0.85

#: 记忆强度的半衰期天数的基准
HALF_LIFE_SCALE = 1.0


def tokenize(text: str) -> set[str]:
    """中文按字符 bigram，英文/数字按词。够用且无需分词库。"""
    tokens: set[str] = set()
    lowered = (text or "").lower()

    for word in ASCII_WORD.findall(lowered):
        if len(word) >= 2:
            tokens.add(word)

    chars = CJK.findall(lowered)
    if len(chars) == 1:
        tokens.add(chars[0])
    for i in range(len(chars) - 1):
        tokens.add(chars[i] + chars[i + 1])

    return tokens


def similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class MemoryManager:
    def __init__(
        self,
        repo: MemoryRepo,
        cfg: BehaviorConfig,
        evolution: EvolutionConfig,
        llm,
    ) -> None:
        self.repo = repo
        self.cfg = cfg
        self.evolution = evolution
        self.llm = llm
        self.extracted = 0
        self.rejected = 0
        self.superseded = 0

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    async def extract(self, group_id: int, messages: list[Message]) -> int:
        """从一批消息里抽取值得长期记住的东西。尽力而为，失败不影响主流程。"""
        if not self.cfg.memory_enabled or len(messages) < 8:
            return 0

        existing = await self.repo.active("group", group_id, limit=60)
        existing_texts = [m["content"] for m in existing]

        transcript = "\n".join(
            f"{'我' if m.is_self else m.nickname}: {m.text}" for m in messages
        )
        system, user = build_memory_prompt(existing_texts, transcript, _PersonaStub())

        try:
            text = await self.llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.25, json_mode=True, profile="analysis",
            )
            payload = extract_json(text)
        except LLMError as exc:
            log.warning("记忆抽取失败: %s", exc)
            return 0

        items = payload.get("memories")
        if not isinstance(items, list):
            return 0

        saved = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not (2 <= len(content) <= 60):
                self.rejected += 1
                continue

            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score < MIN_USEFULNESS:
                self.rejected += 1
                continue

            if any(similarity(content, e) >= DUP_THRESHOLD for e in existing_texts):
                self.rejected += 1
                continue

            mtype = str(item.get("type") or "fact").strip()
            if mtype not in TTL_BY_TYPE:
                mtype = "fact"
            try:
                ttl = int(item.get("ttl_days") or TTL_BY_TYPE[mtype])
            except (TypeError, ValueError):
                ttl = TTL_BY_TYPE[mtype]

            # 矛盾消解：语义相近的旧记忆可能被推翻
            superseded_id = await self._resolve_conflict(
                group_id, content, existing
            )

            await self.repo.add(
                "group", group_id, content,
                type=mtype, confidence=0.9, source="observed",
                ttl_days=ttl, group_id=group_id,
                supersedes_id=superseded_id,
            )
            existing_texts.append(content)
            saved += 1

        if saved:
            self.extracted += saved
            log.info("记忆 +%d 条 (群 %s)", saved, group_id)

        await self.repo.evict_over_capacity(
            "group", group_id, self.evolution.memory_capacity_per_group
        )
        return saved

    async def _resolve_conflict(self, group_id: int, content: str,
                                existing: list[dict]) -> int | None:
        """找出被这条新记忆推翻的旧记忆，标记失效。返回被推翻的 id。"""
        similar = [
            m for m in existing
            if m.get("content") and similarity(content, m["content"]) >= SIMILAR_THRESHOLD
        ]
        if not similar:
            return None

        # 只取最像的一条，避免误伤
        similar.sort(key=lambda m: -similarity(content, m["content"]))
        target = similar[0]

        try:
            text = await self.llm.chat(
                [
                    {"role": "system", "content":
                        "判断两条记忆的关系。只输出 JSON："
                        '{"relation": "contradict" | "refine" | "unrelated"}\n'
                        "contradict = 新信息让旧信息不再成立（如'考完了'推翻'在准备考试'）\n"
                        "refine = 新信息是旧信息的补充细节\n"
                        "unrelated = 只是用词相似"},
                    {"role": "user", "content":
                        f"旧：{target['content']}\n新：{content}"},
                ],
                # 输出只有几十个字符，但思考仍要空间 —— 给小了会截断
                temperature=0.1, max_tokens=600, json_mode=True, profile="analysis",
            )
            relation = str(extract_json(text).get("relation") or "").strip()
        except LLMError:
            return None

        if relation == "contradict":
            await self.repo.invalidate(int(target["id"]))
            self.superseded += 1
            log.info("记忆被推翻: %r -> %r", target["content"], content)
            return int(target["id"])
        return None

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------

    def _score(self, memory: dict, query_tokens: set[str], now: float) -> float:
        content = memory.get("content") or ""
        mem_tokens = tokenize(content)
        if not mem_tokens:
            return 0.0

        overlap = len(mem_tokens & query_tokens) / len(mem_tokens)
        if overlap <= 0:
            return 0.0

        strength = float(memory.get("strength") or 1.0)
        confidence = float(memory.get("confidence") or 1.0)
        ttl = max(1.0, float(memory.get("ttl_days") or 180))
        age_days = max(0.0, (now - float(memory.get("created_ts") or now)) / 86400.0)
        decay = math.pow(0.5, age_days / ttl)

        # 命中过的更容易再被想起
        last_hit = memory.get("last_hit_ts")
        if last_hit:
            hit_age = max(0.0, (now - float(last_hit)) / 86400.0)
            strength += 0.3 * math.pow(0.5, hit_age / 7.0)

        # 类型加权：群文化和黑话对"听懂"的帮助最大
        type_boost = {"culture": 1.25, "slang": 1.25, "relation": 1.15,
                      "self": 1.2, "preference": 1.1}.get(memory.get("type"), 1.0)

        return overlap * decay * strength * confidence * type_boost

    async def recall(
        self,
        group_id: int,
        query: str,
        *,
        user_ids: list[int] | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        if not self.cfg.memory_enabled:
            return []

        top_k = top_k or self.cfg.memory_recall_top
        now = time.time()
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidates: list[tuple[float, dict]] = []
        for memory in await self.repo.active("group", group_id, limit=250):
            score = self._score(memory, query_tokens, now)
            if score > 0:
                candidates.append((score, memory))

        for user_id in (user_ids or [])[:5]:
            for memory in await self.repo.active("user", user_id, limit=40):
                score = self._score(memory, query_tokens, now)
                if score > 0:
                    candidates.append((score * 0.8, memory))

        candidates.sort(key=lambda item: item[0], reverse=True)
        picked = candidates[:top_k]

        for _, memory in picked:
            try:
                await self.repo.touch(int(memory["id"]))
            except Exception:  # noqa: BLE001
                pass

        return [m["content"] for _, m in picked]

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    async def forget(self) -> int:
        """过期与低强度的记忆淡出。容量淘汰在 extract 里做。"""
        removed = 0
        for scope, scope_id in await self._scopes():
            removed += await self.repo.evict_over_capacity(
                scope, scope_id, self.evolution.memory_capacity_per_group
            )
        return removed

    async def _scopes(self) -> list[tuple[str, int]]:
        async with self.repo.conn.execute(
            "SELECT DISTINCT scope, scope_id FROM memories"
        ) as cur:
            return [(r["scope"], r["scope_id"]) for r in await cur.fetchall()]

    async def stats(self) -> dict:
        base = await self.repo.stats()
        base.update({
            "memories_extracted": self.extracted,
            "memories_rejected": self.rejected,
            "memories_superseded": self.superseded,
        })
        return base


class _PersonaStub:
    """build_memory_prompt 只需要 name。"""

    name = "机器人"
