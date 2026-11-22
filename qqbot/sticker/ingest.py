"""表情包入库编排：下载 -> 记账 -> 观察语境 -> 决定收藏。

**必须收到就下**。QQ CDN 的 rkey 是时效性的，拖几小时链接就死了。
所以这里全走"尽力而为"：任何一步失败都只记日志，绝不影响消息主流程。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import StickerConfig
from ..models import Message
from ..store.stickers import StickerRepo
from .collector import StickerCollector
from .library import StickerLibrary

log = logging.getLogger(__name__)


class StickerIngest:
    def __init__(
        self,
        cfg: StickerConfig,
        repo: StickerRepo,
        library: StickerLibrary,
        collector: StickerCollector,
        root: Path,
    ) -> None:
        self.cfg = cfg
        self.repo = repo
        self.library = library
        self.collector = collector
        self.root = root
        self.api = None  # 需要写 QQ 收藏夹时由 pipeline 注入
        self.seen = 0
        self.new = 0

    async def handle_message(
        self,
        msg: Message,
        *,
        before_text: str = "",
        mood: float = 0.0,
    ) -> dict:
        """处理一条消息里的表情包。返回统计信息。"""
        if not self.cfg.enabled:
            return {}

        found = self.library.extract_image_segments(msg.segments)
        if not found:
            return {}

        images = await self.library.ingest(msg.segments)
        if not images:
            return {}

        collected_now: list[int] = []
        for fetched, is_mark in images:
            self.seen += 1
            sticker_id, is_new = await self.repo.add(
                hash=fetched.hash,
                ext=fetched.ext,
                path=str(fetched.path),
                size=fetched.size,
                source_group=msg.group_id,
                source_user=msg.user_id,
                source_msg_id=msg.message_id,
                is_animated=fetched.is_animated,
                is_mark_face=is_mark,
            )
            if is_new:
                self.new += 1

            # 记下"它被用在什么语境" —— 这是后面打标签的原料
            await self.repo.observe(
                sticker_id, msg.group_id, msg.user_id,
                before_text=before_text, after_text="",
            )

            verdict = await self.collector.consider(
                sticker_id=sticker_id,
                group_id=msg.group_id,
                user_id=msg.user_id,
                is_new=is_new,
                mood=mood,
            )
            if not verdict:
                log.debug("未收藏 %s: %s", fetched.hash[:8], verdict.reason)
                continue

            await self.repo.mark_collected(sticker_id, verdict.reason)
            collected_now.append(sticker_id)
            log.info("收藏表情 #%d (%s) —— %s", sticker_id, fetched.ext, verdict.reason)

            if self.cfg.write_to_qq:
                await self._push_to_qq(sticker_id, fetched.path)

        return {"seen": len(images), "collected": collected_now}

    async def _push_to_qq(self, sticker_id: int, path: Path) -> None:
        """把表情真的写进 QQ 收藏夹。

        这会**改动账号状态**，所以由 sticker.write_to_qq 单独开关控制，
        默认关闭。失败了也只是本地库少一个 qq_res_id，不影响使用。
        """
        if self.api is None:
            return
        try:
            await self.api.add_custom_face(str(path), is_origin=True)
            log.info("已写入 QQ 收藏夹: %s", path.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("写入 QQ 收藏夹失败（不影响本地使用）: %s", exc)

    def stats(self) -> dict:
        return {"stickers_seen_msgs": self.seen, "stickers_new": self.new}
