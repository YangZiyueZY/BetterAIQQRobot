"""表情语义标注：从"人类怎么用"里学出标签。

这是整个表情系统里最巧的一环。不去调视觉模型猜图里画了什么，
而是**观察人类把它用在什么语境**：

    「笑死我了」→ [这张图] → 「哈哈哈」
    「草」      → [这张图] → 「绝了」

攒够几条后用 LLM 归纳："这张图总是配着笑点出现" → 标签 搞笑/大笑。
本质上是在模仿"看别人怎么用，就学会了"。

可选再叠一层 VLM 打内容标签（"猫""摇头"），但那是加分项，不是必需。
"""

from __future__ import annotations

import json
import logging

from .. import prompts

from ..config import StickerConfig
from ..generate.llm import LLMError, extract_json
from ..store.stickers import StickerRepo

log = logging.getLogger(__name__)




class StickerTagger:
    def __init__(self, cfg: StickerConfig, repo: StickerRepo, llm) -> None:
        self.cfg = cfg
        self.repo = repo
        self.llm = llm
        self.tagged = 0

    async def tag_pending(self) -> int:
        """给攒够观察样本的表情打标签。返回本次打了几张。"""
        if not self.cfg.enabled:
            return 0

        pending = await self.repo.untagged(
            limit=self.cfg.tag_batch, min_obs=self.cfg.tag_min_observations
        )
        if not pending:
            return 0

        blocks: list[str] = []
        for index, item in enumerate(pending):
            obs = await self.repo.observations(int(item["id"]), limit=6)
            lines = []
            for o in obs:
                before = (o.get("before_text") or "").strip()
                after = (o.get("after_text") or "").strip()
                if before or after:
                    lines.append(f"    {before or '（无语境）'} → 【这张图】 → {after or '（无后续）'}")
            if not lines:
                continue
            blocks.append(f"[{index}] 用过 {len(obs)} 次：\n" + "\n".join(lines))

        if not blocks:
            return 0

        try:
            text = await self.llm.chat(
                [
                    {"role": "system", "content": prompts.text("task.sticker_tag")},
                    {"role": "user", "content": "\n\n".join(blocks)},
                ],
                temperature=0.2,
                json_mode=True,
                profile="analysis",
            )
            payload = extract_json(text)
        except LLMError as exc:
            log.warning("表情打标失败: %s", exc)
            return 0

        results = payload.get("results")
        if not isinstance(results, list):
            return 0

        by_index = {
            int(r["index"]): r for r in results
            if isinstance(r, dict) and str(r.get("index", "")).lstrip("-").isdigit()
        }

        count = 0
        for index, item in enumerate(pending):
            result = by_index.get(index)
            if not result:
                continue
            mood_tags = [str(t).strip() for t in (result.get("mood_tags") or []) if str(t).strip()]
            if not mood_tags:
                continue
            content_tags = [str(t).strip() for t in (result.get("content_tags") or []) if str(t).strip()]
            await self.repo.set_tags(
                int(item["id"]), mood_tags[:4], content_tags[:4],
                str(result.get("desc") or "")[:60],
            )
            await self.repo.mark_observed_tagged(int(item["id"]))
            count += 1

        if count:
            self.tagged += count
            log.info("表情打标 +%d 张", count)
        return count

    def stats(self) -> dict:
        return {"stickers_tagged_by_bot": self.tagged}
