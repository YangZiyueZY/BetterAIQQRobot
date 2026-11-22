"""多模态理解：看图、看图里的字、听语音。

分层门控是成本控制的核心。**不是每张图都值得送模型**：

    L0 元信息   所有图       免费    尺寸/是否动图/sub_type/summary/哈希
    L1 OCR      截图类/被@    免费    NapCat 自带 ocr_image
    L2 VLM      见门控        花钱    云端多模态模型
    L3 语音     所有语音      免费    NapCat 自带 fetch_ptt_text

L2 的触发条件（要同时满足）：
    · 被 @ 或回复了她，**或** 图出现在她关注的话题上下文里
    · 本群本小时 / 全局本日的 VLM 预算没用完
    · 该图哈希没描述过（命中缓存直接复用）

群里表情包和梗图高度重复，缓存命中率很高，实际花费远低于"每张都算"。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import MediaConfig, PaddleOCRConfig
from ..generate.llm import LLMError, extract_json
from ..models import Message
from ..store.media import MediaRepo

from .images import ImageStore, LocalImage
from .paddle_ocr import DONE, FAILED, PaddleOCRClient

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400

VISION_SYSTEM = """你在看一张 QQ 群聊里别人发的图片。请客观描述，不要脑补。

**重点是画面本身**。图里的信息大部分不在文字上 ——
谁在做什么、什么表情、什么场景、什么氛围、有什么值得注意的细节。
不要只念图上的文字就交差。没有文字的图照样要好好看。

只输出 JSON：
{"summary": "一句话说清图里是什么，不超过 30 字",
 "detail": "画面里值得说的细节，2~4 句：主体、动作、场景、氛围、能看出的信息。看不清就留空",
 "type": "表情包|照片|截图|聊天记录截图|二维码|表格|题目|其他",
 "mood": "如果是表情包，它想表达什么情绪（搞笑/无语/赞同/哭泣/生气/害羞/惊讶/得意/疑惑）；不是表情包就留空",
 "text_in_image": "图里能看清的文字，原样抄下来；没有就留空",
 "mentions": ["图里出现的主要对象，如 猫、代码、试卷"],
 "sensitive": false}

注意：看不清就说看不清，不要猜测。不要描述人物身份。"""


def _parse_caption(payload: dict) -> dict:
    return {
        "summary": str(payload.get("summary") or "")[:200],
        "detail": str(payload.get("detail") or "")[:500],
        "type": str(payload.get("type") or "")[:20],
        "mood": str(payload.get("mood") or "")[:20],
        "text_in_image": str(payload.get("text_in_image") or "")[:600],
        "mentions": [str(m)[:20] for m in (payload.get("mentions") or [])][:6],
        "sensitive": bool(payload.get("sensitive")),
    }


@dataclass
class VisionBudget:
    """单群小时额度 + 全局日额度。花超了就自动降级到只 OCR。"""

    hourly_limit: int
    daily_limit: int
    hourly: dict[int, tuple[int, int]] = field(default_factory=dict)
    """group_id -> (hour_bucket, used)"""
    daily: tuple[int, int] = (0, 0)

    def allow(self, group_id: int, now: int | None = None) -> tuple[bool, str]:
        now = int(now or time.time())
        hour_bucket = now // HOUR
        day_bucket = now // DAY

        if day_bucket != self.daily[0]:
            self.daily = (day_bucket, 0)
        if self.daily_limit > 0 and self.daily[1] >= self.daily_limit:
            return False, f"全局今日看图额度已用完({self.daily[1]}/{self.daily_limit})"

        bucket, used = self.hourly.get(group_id, (hour_bucket, 0))
        if bucket != hour_bucket:
            bucket, used = hour_bucket, 0
        if self.hourly_limit > 0 and used >= self.hourly_limit:
            return False, f"本群本小时看图额度已用完({used}/{self.hourly_limit})"

        self.hourly[group_id] = (bucket, used + 1)
        self.daily = (day_bucket, self.daily[1] + 1)
        return True, ""

    def snapshot(self) -> dict:
        now = int(time.time())
        return {
            "hourly_used_total": sum(u for b, u in self.hourly.values()
                                     if b == now // HOUR),
            "daily_used": self.daily[1],
            "hourly_limit": self.hourly_limit,
            "daily_limit": self.daily_limit,
        }


class MediaUnderstander:
    def __init__(
        self,
        cfg: MediaConfig,
        repo: MediaRepo,
        images: ImageStore,
        llm,
        api,
        root: Path,
        *,
        paddle: PaddleOCRClient | None = None,
        ocr_jobs=None,
    ) -> None:
        self.cfg = cfg
        self.repo = repo
        self.images = images
        self.llm = llm
        self.api = api
        self.root = root
        self.paddle = paddle
        self.ocr_jobs = ocr_jobs
        self.budget = VisionBudget(cfg.vision_hourly_budget, cfg.vision_daily_budget)
        self.vision_calls = 0
        self.ocr_calls = 0
        self.ptt_calls = 0
        self.cache_hits = 0
        # message_id -> 后台理解完成后落下的备注，等生成时取
        self.pending_notes: dict[int, list[str]] = {}
        self.paddle_direct = 0
        self.rejected: dict[str, int] = {}

    # ---- 图片 ---------------------------------------------------------

    async def handle_message(
        self, msg: Message, *, force_vision: bool = False
    ) -> list[str]:
        """处理一条消息里的图片。返回给 LLM 看的备注列表。"""
        if not self.cfg.enabled:
            return []

        image_segs = [s for s in (msg.segments or [])
                      if isinstance(s, dict) and s.get("type") == "image"]
        if not image_segs:
            return []

        notes: list[str] = []
        for seg in image_segs[:2]:  # 一条消息最多看两张，防刷
            data = seg.get("data") or {}
            is_sticker = str(data.get("sub_type")) == "1"
            url = data.get("url") or ""

            local = await self.images.fetch(url)
            if local is None:
                notes.append("[图片]（下载失败）")
                continue

            note = await self._describe(local, msg, force_vision=force_vision,
                                        is_sticker=is_sticker)
            if note:
                notes.append(note)

        return notes

    async def collect(self, msg: Message) -> list[dict]:
        """**只做下载**，返回待理解的图片。

        为什么必须拆开：
          · 下载要立刻做 —— NapCat 给的图片 url 带 rkey，会过期
          · 理解很慢 —— 实测一张图 18.8 秒（多模模型 + 大图 base64），
            挂在消息主链路上会把整条消息处理卡住

        所以消息入口只下载，理解丢后台；生成回复前再按需等一小会儿。
        """
        if not self.cfg.enabled:
            return []
        image_segs = [s for s in (msg.segments or [])
                      if isinstance(s, dict) and s.get("type") == "image"]
        out: list[dict] = []
        for seg in image_segs[:2]:          # 一条消息最多看两张，防刷
            data = seg.get("data") or {}
            out.append({
                "is_sticker": str(data.get("sub_type")) == "1",
                "local": await self.images.fetch(data.get("url") or ""),
            })
        return out

    async def describe_pending(self, msg: Message, items: list[dict],
                               *, force_vision: bool = False) -> list[str]:
        """理解已下载的图片，返回备注。**慢，应该在后台跑。**"""
        notes: list[str] = []
        for item in items:
            local = item.get("local")
            if local is None:
                notes.append("[图片]（下载失败）")
                continue
            note = await self._describe(local, msg, force_vision=force_vision,
                                        is_sticker=bool(item.get("is_sticker")))
            if note:
                notes.append(note)
        if notes:
            self.pending_notes[msg.message_id] = notes
        return notes

    async def _describe(
        self, local: LocalImage, msg: Message, *,
        force_vision: bool, is_sticker: bool,
    ) -> str:
        cached = await self.repo.get(local.hash)
        if cached and cached.get("summary"):
            self.cache_hits += 1
            return self._render_note(cached)

        ocr_text = ""
        if self.cfg.ocr_enabled:
            ocr_text = await self._ocr(local)

        caption: dict = {}
        if self._should_use_vision(local, msg, force_vision, is_sticker, ocr_text):
            caption = await self._vision(local, msg)

        summary = caption.get("summary") or self._fallback_summary(is_sticker, ocr_text)
        # 把哈希回填到消息上 —— pipeline 会持久化它。
        # 没有这一步，消息和媒体缓存之间就断了链，后续查不到这张图的 OCR。
        if not msg.media_hash:
            msg.media_hash = local.hash
        await self.repo.upsert(
            hash=local.hash,
            kind="sticker" if is_sticker else "image",
            path=str(local.path),
            summary=summary,
            caption=caption,
            ocr_text=ocr_text,
            sensitive=bool(caption.get("sensitive")),
        )
        return self._render_note({
            "summary": summary, "caption": caption, "ocr_text": ocr_text,
            "kind": "sticker" if is_sticker else "image",
        })

    async def refresh_notes(self, msg: Message,
                            fallback: list[str] | None = None) -> list[str]:
        """生成前重读缓存，拿到最新的图片说明。

        为什么需要：图片理解是**收到消息那一刻**做的，而 PaddleOCR 是异步任务，
        实测结果可能比消息晚十几秒才入库（那次只差 1 秒）。定时的那份说明里
        只有"一张图片"，模型拿到空信息就开始编 —— 实测编出了
        "我这边加载不出来""你打出来得了"。

        生成前重读一次，晚到的那份 OCR 就能用上了。
        """
        if not msg.media_hash or self.repo is None:
            return list(fallback or [])
        try:
            record = await self.repo.get(msg.media_hash)
        except Exception:  # noqa: BLE001
            log.debug("重读媒体缓存失败", exc_info=True)
            return list(fallback or [])
        if not record:
            return list(fallback or [])
        return [self._render_note(record)]

    def _should_use_vision(self, local: LocalImage, msg: Message, force: bool,
                           is_sticker: bool, ocr_text: str) -> bool:
        if not self.cfg.vision_enabled:
            self._reject("看图功能已关闭")
            return False
        # 表情包的语义由 tagger 从使用语境里学，不必花视觉钱
        if is_sticker:
            self._reject("是表情包，交给语境标注")
            return False
        # 这里**曾经**有一条"OCR 已经拿到足够文字就跳过看图"的闸门，已经删掉。
        #
        # 理由是它建立在错误的前提上：OCR 只能读字，而日常图里大量信息藏在画面里
        # （表情、场景、动作、氛围），还有相当一部分图压根没有文字。
        # 更糟的是那个闸门看的是**原始长度**，PaddleOCR 没识别到文字时只返回
        # 图片占位标记（`<div><img src="imgs/img_in_image_box_...">`），
        # 一张没字的图能产出 1619 字符 —— 长度轻松骗过 `len() > 30`，
        # 于是"没有文字的图"被判定成"文字够了"。正好判反。
        #
        # 现在：图片一律交给多模态模型看，OCR 只作辅助补充（图上的文字细节）。
        # 成本由下面的额度兜着（单群小时额度 + 全局日额度）。
        _ = force   # 保留参数；是否看图不再由"有没有被点名"决定

        allowed, reason = self.budget.allow(msg.group_id)
        if not allowed:
            self._reject(reason)
            return False
        return True

    def _reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    # ---- OCR / 语音 / VLM ---------------------------------------------

    async def _ocr(self, local: LocalImage) -> str:
        """按配置的后端取文字。

        `both` 是推荐档：NapCat 的快速 OCR 立刻把消息回掉，
        PaddleOCR 只提交任务（不等待），结果由定时任务收回来写进缓存 ——
        实测 PaddleOCR 快则 6 秒慢则 94 秒，绝不能挂在回复链路上等它。
        """
        backend = (self.cfg.ocr_backend or "napcat").lower()
        text = ""

        if self.cfg.ocr_enabled and backend in ("napcat", "both"):
            text = await self._napcat_ocr(local)

        if backend in ("paddleocr", "both") and self._paddle_ready():
            if backend == "both":
                await self._submit_paddle(local)
            else:
                rich = await self.paddle.recognize(local.path)
                if rich:
                    text = rich
                    self.paddle_direct += 1

        return text

    def _paddle_ready(self) -> bool:
        return bool(self.paddle is not None and self.paddle.cfg.enabled
                    and self.paddle.cfg.configured)

    async def _napcat_ocr(self, local: LocalImage) -> str:
        try:
            text = await self.api.ocr_image(str(local.path))
        except Exception:  # noqa: BLE001
            return ""
        if text:
            self.ocr_calls += 1
        return text

    async def _submit_paddle(self, local: LocalImage) -> bool:
        """只提交，不等结果。"""
        if self.ocr_jobs is None:
            return False
        if local.width and local.height and min(local.width, local.height) < self.paddle.cfg.min_side:
            self._reject("图太小，不值得送云端 OCR")
            return False
        if await self.ocr_jobs.has_pending_for(local.hash):
            return False          # 已经在跑了，别重复提交
        job_id = await self.paddle.submit(local.path)
        if not job_id:
            return False
        await self.ocr_jobs.create(job_id, local.hash)
        return True

    async def poll_jobs(self, limit: int = 5) -> int:
        """收 PaddleOCR 的结果。由定时任务调用，每次只处理几个，别把 tick 拖长。"""
        if not self._paddle_ready() or self.ocr_jobs is None:
            return 0

        done = 0
        for job in await self.ocr_jobs.pending(limit=limit):
            job_id = job["job_id"]
            # 超时保护：超过总时限就别再挂着
            if (time.time() - int(job["submitted_ts"] or 0)) > self.paddle.cfg.job_timeout * 1.5:
                await self.ocr_jobs.set_state(job_id, FAILED, error="超时未完成")
                continue

            st = await self.paddle.poll(job_id)
            if st.state == DONE:
                md = await self.paddle.fetch_markdown(st.jsonl_url or "")
                if md:
                    await self.repo.upsert(hash=job["media_hash"], kind="image", ocr_text=md)
                    await self.ocr_jobs.set_state(job_id, DONE, chars=len(md))
                    done += 1
                    log.info("PaddleOCR 结果已入库: %s（%d 字）", job_id, len(md))
                else:
                    await self.ocr_jobs.set_state(job_id, FAILED, error="结果为空")
            elif st.state == FAILED:
                await self.ocr_jobs.set_state(job_id, FAILED, error=st.error or "任务失败")
        return done

    async def transcribe(self, msg: Message) -> str:
        """语音转文字。NapCat 自带，不花钱。"""
        if not self.cfg.ptt_enabled:
            return ""
        has_ptt = any(isinstance(s, dict) and s.get("type") == "record"
                      for s in (msg.segments or []))
        if not has_ptt:
            return ""
        text = await self.api.fetch_ptt_text(msg.message_id)
        if text:
            self.ptt_calls += 1
        return text

    async def _vision(self, local: LocalImage, msg: Message) -> dict:
        data_uri = self.images.to_data_uri(local.path, max_side=self.cfg.vision_max_side)
        if not data_uri:
            return {}
        try:
            text = await self.llm.chat_vision(VISION_SYSTEM, "看看这张图。", data_uri)
            payload = extract_json(text)
        except LLMError as exc:
            log.warning("看图失败: %s", exc)
            return {}
        self.vision_calls += 1
        return _parse_caption(payload)

    # ---- 渲染 ---------------------------------------------------------

    @staticmethod
    def _meaningful_text(text: str) -> str:
        """剥掉 markdown / HTML 噪音后剩下的真实文字。

        PaddleOCR 的 markdown 里混着 `<table style=...>`、`<img src=...>`
        这类标记，对模型是纯噪音，也骗得过 `len()` 判断。
        """
        if not text:
            return ""
        return MediaUnderstander._clean_ocr(text, limit=2000).strip()

    @staticmethod
    def _fallback_summary(is_sticker: bool, ocr_text: str) -> str:
        """没能看到图时的兜底。

        **必须说清"没看到"** —— 以前这里返回"一张图片"，模型拿到这种空信息
        就会自己编，实测编出了"我这边加载不出来""你打出来得了"。
        """
        usable = MediaUnderstander._meaningful_text(ocr_text)
        if usable:
            return f"一张带文字的图：{usable[:60]}"
        if is_sticker:
            return "表情包"
        # 措辞很重要：只写"没能看到内容"，模型会理解成技术故障，
        # 实测回出"我这边加载不出来""刷不出来""你打出来得了"——
        # 真人不会这么说话。必须明确禁止这类说法。
        return ("一张图（还没看清内容）。**不要编图里有什么**，"
                "也**不要说打不开 / 加载不出来 / 刷不出来**这类话 —— "
                "那不像人。就说'图我瞅瞅''等下我看看'这种，或者干脆先不接这茬")

    @staticmethod
    def _clean_ocr(text: str, limit: int = 500) -> str:
        """把 PaddleOCR 的 markdown 收拾成适合进 prompt 的纯文本。

        表格是 `<table border=1 style='...'><tr><td>姓名</td>...` 这种，
        对模型来说是纯噪音 —— 转成 `姓名 | 成绩 | 备注` 这种行式更省 token 也更好懂。
        LaTeX 公式原样保留（模型看得懂，而且那是信息本身）。
        """
        if not text:
            return ""
        out = text

        # 表格 -> 行式
        def table_to_lines(match: re.Match) -> str:
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(0), re.S | re.I)
            lines = []
            for row in rows:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                if any(cells):
                    lines.append(" | ".join(cells))
            return "\n".join(lines)

        out = re.sub(r"<table.*?</table>", table_to_lines, out, flags=re.S | re.I)
        out = re.sub(r"<[^>]+>", " ", out)          # 剩余标签清掉
        out = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", out)  # 图片占位
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out[:limit]

    @staticmethod
    def _render_note(record: dict) -> str:
        caption = record.get("caption") or {}
        if isinstance(caption, str):
            try:
                caption = json.loads(caption)
            except (json.JSONDecodeError, TypeError):
                caption = {}
        summary = record.get("summary") or caption.get("summary") or "一张图"
        bits = [summary]
        # 画面描述（多模态模型给的）比 OCR 文字重要得多，放前面
        detail = str(caption.get("detail") or "").strip()
        if detail:
            bits.append(f"画面：{detail}")
        if caption.get("mood"):
            bits.append(f"情绪：{caption['mood']}")
        if caption.get("mentions"):
            bits.append("画面里有：" + "、".join(caption["mentions"]))
        # OCR 只作辅助 —— 补充"图上的文字"这一项，不再是理解图片的主力
        raw_text = record.get("ocr_text") or caption.get("text_in_image") or ""
        text = MediaUnderstander._clean_ocr(raw_text) if raw_text else ""
        if text:
            bits.append(f"图上的文字：{text}")
        return "；".join(bits)

    # ---- 维护 ---------------------------------------------------------

    async def purge(self) -> int:
        removed = await self.repo.purge(self.cfg.retention_days)
        if removed:
            log.info("清理过期媒体缓存 %d 条", removed)
        return removed

    def stats(self) -> dict:
        return {
            "vision_calls": self.vision_calls,
            "ocr_calls": self.ocr_calls,
            "ptt_calls": self.ptt_calls,
            "vision_cache_hits": self.cache_hits,
            "vision_budget": self.budget.snapshot(),
            "paddle_ocr_direct": self.paddle_direct,
        }
