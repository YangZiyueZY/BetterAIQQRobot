"""发送层：分条发送 + 模拟打字延迟 + 段渲染。

注意：包名故意不叫 typing，避免遮蔽标准库 typing。

三件必须做对的事：

1. **分条发送**。真人常把一句话拆成 2~3 条发。一口气甩出 60 字是最明显
   的机器人特征。
2. **自我回填**。NapCat 侧 reportSelfMessage=false，机器人自己发的消息
   不会作为事件推回来。发出后必须手动写进 messages 表（is_self=1），
   否则下一轮 LLM 看不到自己说过什么 —— 会自相矛盾、会复读。
3. **引用要能降级**。太老的 message_id 会让引用失败，绝不能因此让整条
   消息发不出去：失败就脱掉 reply 段重发一次。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field

from ..config import BehaviorConfig
from ..face_catalog import catalog as face_catalog
from ..gateway.api import OneBotAPI, OneBotError
from ..models import Bubble, Draft, Message, StickerRef
from ..segments import describe, render_bubble
from ..store.repo import Repo

log = logging.getLogger(__name__)

#: 尝试脱掉 reply 段重试时，需要剔除的段类型
_REPLY_SEG = "reply"

#: 表情 token，如 [微笑] [抠鼻]。这是**点选**不是打字，不该计入打字时间。
_FACE_TOKEN_RE = re.compile(r"\[[^\]]{1,12}\]")

#: 贴纸（表情包）几乎是瞬时的：从收藏里点一下就发出去了
_STICKER_MIN = 0.5
_STICKER_MAX = 1.5


@dataclass
class SendResult:
    message_ids: list[int] = field(default_factory=list)
    used_stickers: list[int] = field(default_factory=list)
    used_faces: list[str] = field(default_factory=list)
    quote_used: bool = False
    at_used: bool = False
    failed: int = 0

    def __bool__(self) -> bool:
        return bool(self.message_ids)


class TypingSender:
    def __init__(self, api: OneBotAPI, repo: Repo, cfg: BehaviorConfig) -> None:
        self.api = api
        self.repo = repo
        self.cfg = cfg
        self.self_id = 0
        self.sent_count = 0
        self.failed_count = 0
        self.blocked_count = 0

    # ---- 时长估算 -----------------------------------------------------

    @staticmethod
    def typing_units(text: str) -> int:
        """数这条消息"要打几个字"。

        · 表情 token（`[微笑]`）是点选的，不算打字
        · 空白不算
        · 标点算：手机上打标点也要切一下键盘，但比打字快，所以按半个算
        """
        stripped = _FACE_TOKEN_RE.sub("", text or "")
        units = 0.0
        for ch in stripped:
            if ch.isspace():
                continue
            units += 0.5 if ch in "，。！？、；：…～~,.!?;:" else 1.0
        # 返回 0 表示"没有要打的字"（空消息、纯表情）。
        # 最小耗时不在这里兜 —— typing_delay 的 floor 负责。
        return round(units)

    def typing_delay(self, text: str, *, has_sticker: bool = False) -> float:
        """估算"把这条消息打出来"要多久。

        拟人的关键：真人不可能秒发。原来第一条消息**完全没有打字时间**，
        只有 0.4~1.6 秒反应时间就发出去了，这是最明显的机器人特征。
        """
        if has_sticker and not text.strip():
            # 表情包是点一下，不涉及打字
            return random.uniform(_STICKER_MIN, _STICKER_MAX)

        low = max(10.0, float(self.cfg.typing_ms_per_char_min))
        high = max(low, float(self.cfg.typing_ms_per_char_max))
        per_char = random.uniform(low, high) / 1000.0

        units = self.typing_units(text)
        # 乘一个抖动：真人打字节奏不会均匀，偶尔快偶尔卡一下
        delay = units * per_char * random.uniform(0.85, 1.2)

        floor = max(0.0, float(self.cfg.typing_min_delay))
        ceil = max(floor, float(self.cfg.typing_max_delay))
        if delay < floor:
            # 短消息走下限，但**下限本身也要抖** ——
            # 否则「在」「咋了」「确实」耗时一模一样（都是精确的 1.00s），
            # 比快还机械。
            delay = floor * random.uniform(0.85, 1.45)
        return min(ceil, delay)

    def reaction_delay(self) -> float:
        """看到消息 -> 开始打字。0.4 秒太像自动应答了，默认改成 0.6~2.0。"""
        low = max(0.0, float(self.cfg.reaction_delay_min))
        high = max(low, float(self.cfg.reaction_delay_max))
        return random.uniform(low, high)

    # ---- 发送 ---------------------------------------------------------

    async def send(
        self,
        group_id: int,
        draft: Draft,
        *,
        trigger: Message | None = None,
        use_quote: bool = False,
        use_at: bool = False,
    ) -> SendResult:
        """分条发出。quote/at 由调用方（群策略采样）决定，不交给模型。"""
        bubbles = [b for b in (draft.bubbles or []) if b][: max(1, self.cfg.max_bubbles)]
        if not bubbles:
            return SendResult()

        result = SendResult()
        catalog = face_catalog()

        # 引用和 @ 都只加在第一条上 —— QQ 的引用是挂在单条消息上的，
        # 连发三条时后两条挂引用反而奇怪。
        if trigger is not None:
            if use_quote and trigger.message_id:
                bubbles[0].reply_to = trigger.message_id
                result.quote_used = True
            if use_at and trigger.user_id:
                bubbles[0].at_user = trigger.user_id
                result.at_used = True

        # 反应时间：看到消息到开始打字（每条回复只算一次）。
        # 为 0 时不白调一次 sleep —— 测试和真实路径都更干净。
        reaction = self.reaction_delay()
        if reaction > 0:
            await asyncio.sleep(reaction)

        for index, bubble in enumerate(bubbles):
            # ★ 每一条都要等打字时间，**包括第一条**。
            #   原来写的是 `if index > 0`，第一条只有反应时间就发出去了 ——
            #   「在」这种两个字的消息 0.4 秒就到群里，一眼假。
            delay = self.typing_delay(
                bubble.text, has_sticker=bool(bubble.sticker_path)
            )
            if index == 0 and not self.cfg.typing_first_bubble:
                delay = 0.0
            if delay > 0:
                await asyncio.sleep(delay)

            segments = render_bubble(bubble, catalog, sticker_base=None)
            if not segments:
                continue

            message_id = await self._send_segments(group_id, segments)
            if message_id is None:
                result.failed += 1
                break

            result.message_ids.append(message_id)
            self.sent_count += 1

            if bubble.face:
                result.used_faces.append(bubble.face)
            if bubble.sticker_ref:
                result.used_stickers.append(bubble.sticker_ref.id)

            await self._backfill(group_id, message_id, segments, bubble)

        # 给触发消息贴个表情回应（相当于"我看到了/说得好"）
        if draft.emoji_like and trigger is not None and trigger.message_id:
            await self.api.set_msg_emoji_like(trigger.message_id, draft.emoji_like)

        return result

    async def _send_segments(self, group_id: int, segments: list[dict]) -> int | None:
        """发一次。带 reply 段失败时自动脱掉重试 —— 引用过期不能拖垮整条消息。"""
        try:
            message_id = await self.api.send_group_segments(group_id, segments)
            if message_id:
                return int(message_id)
            self.failed_count += 1
            log.warning("发送未返回 message_id")
            return None
        except (OneBotError, TimeoutError, ConnectionError) as exc:
            has_reply = any(s.get("type") == _REPLY_SEG for s in segments)
            if has_reply:
                log.info("带引用的发送失败（%s），脱掉引用重试", exc)
                stripped = [s for s in segments if s.get("type") != _REPLY_SEG]
                if stripped:
                    try:
                        message_id = await self.api.send_group_segments(group_id, stripped)
                        if message_id:
                            return int(message_id)
                    except (OneBotError, TimeoutError, ConnectionError) as exc2:
                        log.warning("重试仍失败: %s", exc2)
            else:
                log.warning("发送失败 group=%s: %s", group_id, exc)

            self.failed_count += 1
            return None

    async def _backfill(self, group_id: int, message_id: int,
                        segments: list[dict], bubble) -> None:
        """把自己发的消息写回时间线，补上 NapCat 不上报的那一侧。"""
        msg = Message(
            group_id=group_id,
            message_id=message_id,
            user_id=self.self_id,
            nickname="我",
            text=bubble.render_text() or describe(segments),
            segments=segments,
            ts=int(time.time()),
            is_self=True,
            has_image=bool(bubble.sticker_path),
        )
        try:
            await self.repo.save_message(msg)
        except Exception:  # noqa: BLE001 - 回填失败不影响发送本身
            log.exception("自我回填失败 message_id=%s", message_id)

    # ---- 手动发送（面板用） -------------------------------------------

    async def send_text(self, group_id: int, text: str) -> list[int]:
        """面板的"手动发一句"。不走 LLM、不加引用，但保留分条。"""
        from ..generate.postprocess import split_bubble

        pieces = split_bubble(text, self.cfg.max_chars_per_bubble) or [text]
        draft = Draft(bubbles=[Bubble(text=p) for p in pieces])
        result = await self.send(group_id, draft)
        return result.message_ids
