"""编排层：把 gateway 收到的消息串成完整链路。

    message ─┬─▶ 落库(去重)
             ├─▶ 表情包入库（**必须立刻下载**，QQ CDN 的 rkey 会过期）
             ├─▶ 观察语境回填（供表情打标学习语义）
             ├─▶ 群画像累积
             └─▶ 蓄水池(防抖) ──▶ 决定层
                                      │
                          ┌───────────┴────────────┐
                          ▼ 不说                    ▼ 说
                    写 decisions 表          记忆召回 + 图片理解
                                             群画像 + 内在状态
                                                  ↓
                                             prompt → LLM
                                                  ↓
                                            去 AI 味 → 表情包选用
                                                  ↓
                                             safety 硬拦截
                                                  ↓
                                    群策略决定引用/@ → 分条发送
                                                  ↓
                                      自我回填 + 开启反馈追踪

任何一环抛异常都只影响这一轮，绝不影响连接和后续消息。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime

from .asr import StepFunASR
from .bilibili import BiliService
from .bilibili.api import BiliClient
from .bilibili.login import BiliLogin
from .buffer.pool import MessagePool
from .config import Config
from .decide.llm_decider import HybridDecider, LLMDecider
from .decide.scorer import Decider
from .evolution import EvolutionEngine
from .face_catalog import catalog as face_catalog
from .gateway.api import OneBotAPI
from .generate.llm import LLMClient, LLMError
from .generate.postprocess import postprocess_bubbles
from .generate.prompt import (
    build_profile_note,
    render_conversation,
    build_profile_prompt,
    build_system,
    build_user,
)
from .generate.llm import extract_json
from .media import ImageStore, MediaUnderstander, SegmentIngestor
from .media.paddle_ocr import PaddleOCRClient
from .memory import MemoryManager
from .onboard import Onboarder
from .models import Bubble, Conversation, Decision, Draft, Message
from .persona import (Persona, PersonaError, load as load_persona,
                      parse as parse_persona)
from . import prompts
from .profile import ProfileBuilder
from .safety.filter import SafetyFilter
from .search import SearchService
from .search.client import SearchClient
from .schedule.state import Presence, Scheduler
from .sender.typing_sim import TypingSender
from .store.evolution import FeedbackRepo, PolicyRepo, ProfileRepo
from .store.media import MediaRepo
from .store.memories import MemoryRepo
from .store.ocr import OcrJobRepo
from .store.onboard import OnboardRepo
from .store.personas import PersonaRepo
from .store.prompts import PromptRepo
from .store.repo import Repo
from .store.survival import SurvivalRepo
from .store.stickers import StickerRepo
from .survival import SurvivalTracker
from .sticker import (
    StickerCollector,
    StickerIngest,
    StickerLibrary,
    StickerSelector,
    StickerTagger,
)
from .mood import MoodStore

log = logging.getLogger(__name__)

#: 每积累这么多条群消息，触发一次记忆抽取
MEMORY_EXTRACT_EVERY = 40
#: 每积累这么多条，重算一次统计画像
PROFILE_EVERY = 200
#: 每积累这么多条，跑一次 LLM 复盘（主题/黑话）
REFLECT_EVERY = 500


class Pipeline:
    def __init__(self, cfg: Config, repo: Repo, api: OneBotAPI) -> None:
        self.cfg = cfg
        self.repo = repo
        self.api = api
        self.self_id = 0
        self.paused = False
        self.started_at = time.time()
        self.nickname = ""

        db = repo.db
        # --- 领域 store ---
        self.memories = MemoryRepo(db)
        self.stickers = StickerRepo(db)
        self.media_repo = MediaRepo(db)
        self.ocr_jobs = OcrJobRepo(db)
        self.personas = PersonaRepo(db)
        self.prompt_repo = PromptRepo(db)
        # 逐群人设缓存：group_id -> Persona。
        # 没自定义的群也要缓存 —— 否则每条消息都要查一次库。
        self._persona_cache: dict[int, Persona] = {}
        # 图片理解的后台任务：message_id -> Task。
        # 下载要立刻做（rkey 会过期），理解很慢，所以拆开。
        self._media_tasks: dict[int, asyncio.Task] = {}
        self.profiles = ProfileRepo(db)
        self.policies = PolicyRepo(db)
        self.feedbacks = FeedbackRepo(db)

        # --- 人设与表情目录 ---
        self.persona = load_persona(cfg.persona_path)
        self.faces = face_catalog()

        # --- 各层 ---
        self.llm = LLMClient(cfg.llm)
        # 决定层包一层：打分器负责一票否决与先验分数，灰区交给 LLM 判定。
        # 管线只跟 HybridDecider 打交道。
        self.decider = HybridDecider(
            cfg.decider, Decider(cfg.behavior, cfg.debug),
            LLMDecider(cfg.decider, self.llm), repo=repo, debug=cfg.debug,
        )
        self.scheduler = Scheduler(cfg.behavior, repo, cfg.debug)
        self.sender = TypingSender(api, repo, cfg.behavior)
        self.safety = SafetyFilter.load(cfg.behavior, cfg.root / "safety.txt")
        self.mood = MoodStore(repo)
        self.memory = MemoryManager(self.memories, cfg.behavior, cfg.evolution, self.llm)

        # --- 表情包 ---
        self.sticker_lib = StickerLibrary(cfg.root / cfg.sticker.base_dir)
        self.collector = StickerCollector(cfg.sticker, self.stickers)
        self.tagger = StickerTagger(cfg.sticker, self.stickers, self.llm)
        self.selector = StickerSelector(cfg.sticker, self.stickers)
        self.sticker_ingest = StickerIngest(
            cfg.sticker, self.stickers, self.sticker_lib, self.collector, cfg.root
        )
        self.sticker_ingest.api = api

        # --- 多模态 ---
        self.image_store = ImageStore(cfg.root / cfg.media.base_dir)
        self.bili_client = BiliClient(
            cookie=cfg.bilibili.cookie, buvid3=cfg.bilibili.buvid3,
            buvid4=cfg.bilibili.buvid4,
            timeout=cfg.bilibili.timeout,
        )
        self.asr = StepFunASR(cfg.asr)
        self.bili = BiliService(cfg.bilibili, self.bili_client, self.asr)
        self.bili_login = BiliLogin()
        """扫码登录会话。Cookie 掉了时面板用它拿新的。"""
        self.search_client = SearchClient(cfg.search)
        self.search = SearchService(cfg.search, self.search_client)
        self.paddle = PaddleOCRClient(cfg.paddleocr)
        self.media = MediaUnderstander(
            cfg.media, self.media_repo, self.image_store, self.llm, api, cfg.root,
            paddle=self.paddle, ocr_jobs=self.ocr_jobs,
        )
        # 非文字消息段：语音/视频/卡片/合并转发/文件。
        # 和图片一样属于"理解消息"，但走的是各自的通道（ASR、API 展开）。
        self.ingest = SegmentIngestor(cfg.media, api, asr=self.asr,
                                      store=self.image_store, root=cfg.root)

        # --- 自进化 ---
        self.evolution = EvolutionEngine(
            cfg.evolution, self.feedbacks, self.policies, repo, api, self.mood
        )
        self.profile_builder = ProfileBuilder(cfg.evolution, self.profiles, repo)

        # 冷启动：打开发言权限时先读一遍群历史，再决定怎么融进去
        # 伪装存活：把"有没有被认出来"变成可结算的奖励。
        # 它独立于自进化（不改参数，只记账），但给决策层提供风险参考。
        self.survival_repo = SurvivalRepo(db)
        self.survival = SurvivalTracker(cfg.evolution, repo, self.survival_repo)

        self.onboard_repo = OnboardRepo(db)
        self.onboard = Onboarder(
            cfg, api, repo, self.onboard_repo, self.llm,
            profile_builder=self.profile_builder,
            persona_for=self.persona_for,
        )
        self._last_ask: dict[int, float] = {}

        self.pool = MessagePool(self._on_conversation, cfg.behavior)

        # 系统提示词按群缓存 —— 逐群人设生效后，不同群的 system 不一样了。
        # 以前是单个字符串，那样改了任一群的人设都会串味。
        self._system_cache: dict[int, str] = {}
        self._system_stale = True
        self._known_face_names = self.faces.prompt_block()

        self.stats = {
            "messages_received": 0,
            "duplicates_dropped": 0,
            "skipped_disabled": 0,
            "errors": 0,
            "memory_batches": 0,
            "profile_rebuilds": 0,
            "reflects": 0,
        }
        self._seen_groups: set[int] = set()
        self._seen_members: set[tuple[int, int]] = set()
        self._group_counter: dict[int, int] = {}
        self._last_text: dict[int, str] = {}
        """每群上一条人类消息的文本 —— 用作表情包的"前文语境"。"""
        self._pending_media: dict[int, list[str]] = {}
        # message_id -> (存入时刻, 已渲染好的 B 站视频内容块)。
        # 存时刻是因为：带了链接但始终没触发回复的消息，块会一直留着 ——
        # 不清理就是个慢性内存泄漏。
        self._pending_video: dict[int, tuple[float, str]] = {}
        """message_id -> (ts, B站视频内容块)"""

    # ------------------------------------------------------------------
    # 人设 / prompt
    # ------------------------------------------------------------------

    def persona_for(self, group_id: int) -> Persona:
        """取该群生效的人设（同步版，只看缓存）。

        默认所有群统一用 persona.yaml；只有显式给某个群存过自定义，
        那个群才走自己那份。
        """
        return self._persona_cache.get(group_id) or self.persona

    async def load_persona_for(self, group_id: int) -> Persona:
        """带 DB 查询的版本（消息处理路径用）。"""
        if group_id in self._persona_cache:
            return self._persona_cache[group_id]
        raw = ""
        try:
            raw = await self.personas.get(group_id)
        except Exception:  # noqa: BLE001
            log.debug("读群 %s 的自定义人设失败", group_id, exc_info=True)

        if raw.strip():
            try:
                persona = parse_persona(raw)
                self._persona_cache[group_id] = persona
                return persona
            except PersonaError as exc:
                # 自定义人设坏了不能拖垮整个群 —— 退回默认并告警
                log.warning("群 %s 的自定义人设解析失败，退回默认: %s", group_id, exc)

        self._persona_cache[group_id] = self.persona
        return self.persona

    def invalidate_persona(self, group_id: int | None = None) -> None:
        """人设改动后清缓存。传 None 清全部。"""
        if group_id is None:
            self._persona_cache.clear()
        else:
            self._persona_cache.pop(group_id, None)

    async def reload_prompts(self) -> None:
        """把 DB 里的提示词覆盖层（全局 + 各群）载入进程缓存。"""
        try:
            prompts.clear_overrides()
            for gid, layer in (await self.prompt_repo.all_tiers()).items():
                prompts.set_overrides(layer, gid)
        except Exception:  # noqa: BLE001
            log.warning("载入提示词覆盖失败，用默认", exc_info=True)
            prompts.clear_overrides()
        self.mark_prompt_stale()

    def reload_persona(self) -> None:
        """全局人设已更新，重新加载（面板改完默认人设时调）。"""
        self.persona = load_persona(self.cfg.persona_path)
        self.invalidate_persona()
        self.mark_prompt_stale()

    def mark_prompt_stale(self, group_id: int | None = None) -> None:
        """标记系统提示词需要重建。传 None 清全部群。"""
        if group_id is None:
            self._system_cache.clear()
            self._system_stale = True
        else:
            self._system_cache.pop(group_id, None)

    async def system_prompt(self, group_id: int) -> str:
        """取该群的系统提示词（按群缓存）。"""
        cached = self._system_cache.get(group_id)
        if cached and not self._system_stale:
            return cached

        persona = self.persona_for(group_id)
        moods = await self.selector.mood_labels()
        built = build_system(persona, self.cfg.behavior, self.faces,
                             sticker_moods=moods, group_id=group_id)
        self._system_cache[group_id] = built
        return built

    # ------------------------------------------------------------------
    # 接入
    # ------------------------------------------------------------------

    async def load_prompt_overrides(self) -> None:
        """启动时把提示词覆盖层载入（面板改动后也会再调）。"""
        await self.reload_prompts()

    async def on_ready(self, self_id: int) -> None:
        self.self_id = self_id
        self.sender.self_id = self_id
        try:
            info = await self.api.get_login_info()
            self.nickname = str(info.get("nickname") or "")
            log.info("登录账号: %s(%s)", info.get("nickname"), info.get("user_id"))
        except Exception as exc:  # noqa: BLE001
            log.warning("get_login_info 失败: %s", exc)

        try:
            await self.sync_groups(await self.api.get_group_list())
        except Exception as exc:  # noqa: BLE001
            log.warning("同步群列表失败: %s", exc)

        await self._sync_online_status()

    async def sync_groups(self, groups: list[dict]) -> None:
        for group in groups:
            group_id = group.get("group_id")
            if group_id is None:
                continue
            group_id = int(group_id)
            default_enabled = bool(self.cfg.groups and group_id in self.cfg.groups)
            await self.repo.upsert_group(
                group_id, group.get("group_name"), default_enabled=default_enabled
            )
        log.info("群列表已同步: %d 个", len(groups))

        # 模型画像建议：配置不理想就在启动日志里点明
        try:
            from .generate.model_profiles import advise
            for w in advise(self.cfg.llm.reply_model or self.cfg.llm.model,
                            self.cfg.llm.analysis_model or self.cfg.llm.model):
                log.warning("模型适配: %s", w)
        except Exception:  # noqa: BLE001
            log.debug("模型画像检查失败", exc_info=True)

        if self.paddle.cfg.enabled and not self.paddle.cfg.configured:
            log.warning("PaddleOCR 已启用但没配 token（.env 里的 PADDLEOCR_TOKEN）")
        pending = await self.ocr_jobs.pending(limit=100)
        if pending:
            log.info("有 %d 个云端 OCR 任务在途，会在定时任务里继续收结果", len(pending))

    async def _sync_online_status(self) -> None:
        """让 QQ 上显示的在线状态跟着作息走。"""
        from .schedule.state import Presence as P

        presence = self.scheduler.presence()
        if self.scheduler.forced:
            presence = P.ONLINE   # 测试模式：QQ 上也一直显示在线
        code = {
            P.ONLINE: 10,
            P.BUSY: 50,
            P.AWAY: 30,
            P.SLEEPING: 30,
        }.get(presence, 10)
        await self.api.set_online_status(code)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    async def handle_message(self, msg: Message) -> None:
        group_id = msg.group_id

        if group_id not in self._seen_groups:
            self._seen_groups.add(group_id)
            default_enabled = bool(self.cfg.groups and group_id in self.cfg.groups)
            await self.repo.upsert_group(group_id, default_enabled=default_enabled)

        # 解析该群生效的人设并缓存。放在这里是因为后面 decide/prompt 都走
        # 同步的 persona_for()，它只看缓存 —— 不预热的话第一个群会串到默认人设。
        try:
            await self.load_persona_for(group_id)
        except Exception:  # noqa: BLE001
            log.debug("预热群 %s 人设失败", group_id, exc_info=True)

        inserted = await self.repo.save_message(msg)
        if not inserted:
            self.stats["duplicates_dropped"] += 1
            return

        if msg.is_self:
            return

        self._seen_members.add((group_id, msg.user_id))
        await self.repo.upsert_member(group_id, msg.user_id, msg.nickname)
        self.stats["messages_received"] += 1

        # 怀疑检测：有人怀疑它是机器人就记一笔。
        # directed 判定很关键 —— 不判断指向，聊到别的 AI 会误伤。
        try:
            directed = bool(msg.mentions_self or msg.reply_to_self
                            or await self._bot_spoke_recently(group_id))
            hit = await self.survival.note_message(msg, directed=directed)
            if hit is not None and self.survival_repo is not None:
                self.stats["suspicions"] = self.stats.get("suspicions", 0) + 1
        except Exception:  # noqa: BLE001
            log.debug("怀疑检测失败", exc_info=True)
        self._group_counter[group_id] = self._group_counter.get(group_id, 0) + 1

        # 图片 / 语音的即时处理（下载必须现在做，rkey 会过期）
        await self._handle_media(group_id, msg)

        # 回填上一条表情观察的"后文"，然后记下这条作为下一条的前文
        await self.stickers.set_after_text(group_id, msg.text)
        self._last_text[group_id] = msg.text

        flag = "  << @我" if msg.mentions_self else ""
        log.info("[%s] %s: %s%s", group_id, msg.nickname, msg.text[:120], flag)

        if not await self.repo.group_enabled(group_id):
            self.stats["skipped_disabled"] += 1
            return
        if self.paused:
            return

        await self.pool.push(msg)
        await self._maybe_periodic(group_id)

    # ------------------------------------------------------------------
    # 新群冷启动
    # ------------------------------------------------------------------

    async def _bot_spoke_recently(self, group_id: int,
                                  within: float = 300.0) -> bool:
        """机器人最近 5 分钟内在这个群说过话吗。

        用来判断"有人提到机器人/AI"是不是在说它 —— 它刚说完话时
        别人接一句"这是机器人吧"，几乎肯定是在说它。
        """
        try:
            last = await self.repo.last_self_message_ts(group_id)
        except Exception:  # noqa: BLE001
            return False
        return bool(last) and (time.time() - float(last)) <= within

    async def on_group_enabled(self, group_id: int, *, force: bool = False) -> dict:
        """面板给某个群打开发言权限时调用。启动冷启动分析。"""
        try:
            return await self.onboard.start(group_id, force=force)
        except Exception as exc:  # noqa: BLE001
            log.warning("启动群 %s 冷启动失败: %s", group_id, exc)
            return {"started": False, "reason": str(exc)[:120]}

    async def onboard_snapshot(self, group_id: int) -> dict:
        snap = await self.onboard.snapshot(group_id)
        snap["last_ask_ts"] = int(self._last_ask.get(group_id, 0))
        return snap

    async def _maybe_ask_question(self) -> None:
        """在合适的时机，把冷启动时记下的疑问问掉。

        时机很讲究 —— 真人不会半夜突然发问，也不会在群里正聊得火热时
        插一句自己的困惑。所以要同时满足：在线、群里最近有动静、
        距上次提问够久。
        """
        if not getattr(self.cfg.behavior, "onboard_enabled", True):
            return
        if self.scheduler.presence() is not Presence.ONLINE:
            return

        gap = max(1, int(getattr(self.cfg.behavior, "onboard_ask_gap_minutes", 90))) * 60
        min_msgs = max(1, int(getattr(self.cfg.behavior,
                                      "onboard_ask_min_group_msgs", 3)))
        now = time.time()

        for group in await self.repo.list_groups():
            group_id = int(group["group_id"])
            if not group["enabled"] or group["paused"]:
                continue
            if now - self._last_ask.get(group_id, 0) < gap:
                continue
            try:
                question = await self.onboard.pending_question(group_id)
            except Exception:  # noqa: BLE001
                continue
            if not question:
                continue
            # 群里最近得有动静，冷场时提问很突兀
            recent = await self.repo.messages_since(group_id, int(now) - 600, limit=50)
            if len([m for m in recent if not m.is_self]) < min_msgs:
                continue
            ok, _reason = await self.scheduler.check_budget(group_id)
            if not ok:
                continue

            text = str(question.get("text") or "").strip()
            if not text:
                continue
            try:
                await self.sender.send_text(group_id, text)
            except Exception:  # noqa: BLE001
                log.warning("提问发送失败 group=%s", group_id, exc_info=True)
                continue
            self._last_ask[group_id] = now
            await self.onboard.mark_asked(group_id, text)
            await self.scheduler.note_spoke(group_id)
            await self.survival.note_spoke(group_id)
            log.info("群 %s 提了个问题：%s", group_id, text[:40])

    async def _understand_media(self, msg: Message, items: list[dict],
                                force_vision: bool) -> None:
        """后台理解图片。完成后备注会落进 media.pending_notes 和缓存。"""
        try:
            notes = await self.media.describe_pending(
                msg, items, force_vision=force_vision)
            # 兜底再回填一次 hash：万一 collect 那条路没写成功
            # （比如 collect 之后新加了别的下载入口）
            if msg.media_hash:
                try:
                    await self.repo.set_media_hash(
                        msg.group_id, msg.message_id, msg.media_hash)
                except Exception:  # noqa: BLE001
                    log.debug("后台回填 media_hash 失败", exc_info=True)
            if notes:
                self._pending_media[msg.message_id] = notes
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.warning("图片理解失败 message_id=%s", msg.message_id, exc_info=True)
        finally:
            self._media_tasks.pop(msg.message_id, None)

    async def _await_media(self, msg: Message) -> None:
        """生成回复前，给还在跑的看图任务一点时间。

        用户发图多半就是要聊这张图，等一小会儿值得；但要有上限 ——
        实测一张图 18.8 秒，无限等会把回复拖死。超时就先回，
        说明里会写"没能看到内容"，模型不会硬编。
        """
        task = self._media_tasks.get(msg.message_id)
        if task is None or task.done():
            return
        limit = float(getattr(self.cfg.media, "vision_wait", 8.0))
        if limit <= 0:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=limit)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            log.info("看图超时 %.0fs，先按没看清回 message_id=%s", limit, msg.message_id)
        except Exception:  # noqa: BLE001
            log.debug("等待看图任务失败", exc_info=True)

    async def _handle_media(self, group_id: int, msg: Message) -> None:
        """表情包入库 + 记录图片备注。全部尽力而为。"""
        # 被 @ 或回复她时，值得花钱看图
        force_vision = msg.mentions_self or msg.reply_to_self

        if self.cfg.sticker.enabled:
            try:
                result = await self.sticker_ingest.handle_message(
                    msg, before_text=self._last_text.get(group_id, ""),
                    mood=(await self.mood.current()).mood,
                )
                if result:
                    log.debug("表情包处理: %s", result)
            except Exception:  # noqa: BLE001
                log.warning("表情包入库失败", exc_info=True)

        # 非文字段（语音/视频/卡片/合并转发/文件）：解析成给 LLM 看的备注。
        # 这些也慢（语音要转写、转发要拉内容），但比看图快得多，
        # 而且不像图片有 rkey 过期问题（record 的 url 会过期，所以放前面）。
        try:
            seg_notes = await self.ingest.ingest(msg)
            if seg_notes:
                self._pending_media.setdefault(msg.message_id, []).extend(seg_notes)
        except Exception:  # noqa: BLE001
            log.warning("非文字段解析失败 message_id=%s", msg.message_id, exc_info=True)

        # 图片：**只下载**，理解丢后台。
        #
        # 下载必须现在做 —— 图片 url 带 rkey，会过期。
        # 但理解很慢（多模模型看图实测 18.8 秒），挂在消息主链路上会把
        # 整条消息处理卡住，后面的消息全部排队。所以拆成两步：
        #   这里下载 -> 后台 task 理解 -> 生成回复前有界等待 + 读缓存兜底
        try:
            items = await self.media.collect(msg)
            if items:
                # ★ 下载成功就**立刻**把 hash 写回库。
                #
                # 消息在上面已经落库了（那时还没有 hash），而理解是后台任务 ——
                # 等到后台跑完再回填就太晚了：生成回复时按 hash 查媒体缓存会查不到，
                # 等于图片白下载了（实测踩过：hash 恒为 NULL，机器人一直说"刷不出来"）。
                first = next((it for it in items if it.get("local")), None)
                if first is not None and not msg.media_hash:
                    msg.media_hash = first["local"].hash
                    await self.repo.set_media_hash(
                        group_id, msg.message_id, msg.media_hash)
                self._media_tasks[msg.message_id] = asyncio.ensure_future(
                    self._understand_media(msg, items, force_vision)
                )
        except Exception:  # noqa: BLE001
            log.warning("图片下载失败", exc_info=True)

        # B 站视频：拿字幕/转写。放在这里做是因为蓄水池有 5 秒防抖窗口，
        # 等 LLM 真要生成时内容已经就绪了。
        try:
            await self._handle_bilibili(msg)
        except Exception:  # noqa: BLE001
            log.warning("B站视频解析失败", exc_info=True)

        # 回填图片哈希。消息在上面已经落库了，所以这里要单独 update ——
        # 没有它，消息和媒体缓存就断链，云端 OCR 的结果永远找不回来。
        if msg.media_hash:
            try:
                await self.repo.set_media_hash(group_id, msg.message_id, msg.media_hash)
            except Exception:  # noqa: BLE001
                log.debug("回填 media_hash 失败", exc_info=True)

    async def _handle_bilibili(self, msg: Message) -> None:
        """消息里带 B 站链接就解析出内容，供生成时使用。

        链接可能藏在文本里，也可能在 json/xml 分享卡片里 —— 都扫一遍。
        """
        if not self.bili.enabled:
            return

        haystack = msg.text or ""
        for seg in msg.segments or []:
            if isinstance(seg, dict):
                data = seg.get("data") or {}
                for key in ("text", "json", "xml", "content"):
                    val = data.get(key)
                    if isinstance(val, str):
                        haystack += "\n" + val

        links = self.bili.links_in(haystack)
        if not links:
            return

        blocks: list[str] = []
        for link in links:
            content = await self.bili.fetch(link)
            if content.ok:
                blocks.append(content.render(self.cfg.bilibili.max_chars))
                log.info("B站视频已解析《%s》[%s] %d 字",
                         (content.info.title[:28] if content.info else ""),
                         content.source, len(content.text))
            else:
                log.info("B站视频解析未成功: %s", content.error)
                blocks.append(f"（群友发了个B站视频，但没能拿到内容：{content.error}）")

        if blocks:
            self._prune_video()
            self._pending_video[msg.message_id] = (time.monotonic(), "\n\n".join(blocks))

    def _prune_video(self, ttl: float = 600.0) -> None:
        """丢掉过期的视频块，别让没人回复的消息一直占内存。"""
        if len(self._pending_video) < 50:
            cutoff = time.monotonic() - ttl
            self._pending_video = {
                k: v for k, v in self._pending_video.items() if v[0] >= cutoff
            }
            return
        # 太多了就按时间排序只留最近的
        newest = sorted(self._pending_video.items(), key=lambda kv: -kv[1][0])[:50]
        self._pending_video = dict(newest)

    def _take_video(self, message_id: int) -> str:
        """取出并删除某个消息的视频块（内容会变，不用 .pop 的裸写法）。"""
        item = self._pending_video.pop(message_id, None)
        return item[1] if item else ""

    async def _check_bili_cookie(self, interval: float = 1800.0) -> None:
        now = time.monotonic()
        if now - getattr(self, "_bili_checked_at", 0.0) < interval:
            return
        self._bili_checked_at = now

        if not (self.bili.enabled and self.cfg.bilibili.cookie):
            return
        info = await self.bili_login.nav_info(self.cfg.bilibili.cookie)
        if info.get("is_login") is False:
            if not self.bili_client.auth_suspect:
                log.warning("B站 Cookie 已失效 —— 面板「设置 → B站视频」点「扫码登录」重新登录")
                self.bili_client.auth_suspect = True
                self.bili_client.auth_suspect_ts = time.time()
                self.bili_client.last_auth_error = "Cookie 已失效，需要重新扫码登录"
        else:
            self.bili_client.auth_suspect = False

    async def bili_qr_start(self) -> dict:
        session = await self.bili_login.start()
        return session.snapshot()

    async def bili_qr_poll(self) -> dict:
        """轮询扫码状态。成功后**当场**把新 Cookie 装进运行中的客户端。"""
        session = await self.bili_login.poll()
        if session.state == "confirmed" and session.cookie:
            self.bili_client.set_cookie(session.cookie)
            self.cfg.bilibili.cookie = session.cookie
            await self._persist_bili_cookie(session.cookie)
        return session.snapshot()

    async def _persist_bili_cookie(self, cookie: str) -> None:
        """写进 .env，重启后仍然有效。"""
        try:
            import asyncio
            from .config import _write_env_secret
            await asyncio.to_thread(
                _write_env_secret, self.cfg.root / ".env", "BILIBILI_COOKIE", cookie
            )
            log.info("B站 Cookie 已写入 .env")
        except Exception:  # noqa: BLE001
            log.warning("B站 Cookie 写 .env 失败", exc_info=True)

    async def bili_auth_state(self) -> dict:
        """面板用：当前 Cookie 还有效吗。"""
        state = self.bili_client.auth_state()
        state["enabled"] = self.bili.enabled
        if not state["has_cookie"]:
            state["ok"] = False
            state["need_login"] = True
            state["message"] = "还没配 Cookie —— AI 字幕拿不到，需要扫码登录"
            return state

        info = await self.bili_login.nav_info(self.cfg.bilibili.cookie)
        state.update(info)
        if info.get("is_login"):
            state["ok"] = True
            state["need_login"] = False
            state["message"] = f"已登录：{info.get('uname') or info.get('mid')}"
        elif info.get("is_login") is False:
            state["ok"] = False
            state["need_login"] = True
            state["message"] = "Cookie 已失效 —— 需要重新扫码登录"
        else:
            # 网络问题查不出来，退回本地信号
            state["ok"] = not state["auth_suspect"]
            state["need_login"] = state["auth_suspect"]
            state["message"] = ("网络原因没能校验登录态"
                                + ("（但本地已发现字幕接口要求登录）"
                                   if state["auth_suspect"] else ""))
        return state

    async def _bili_search(self, text: str) -> str:
        """消息里明确提到 B 站 + 有信息需求时，搜 B 站。"""
        if not (self.bili.enabled and self.cfg.bilibili.enabled_search):
            return ""
        lowered = (text or "").lower()
        if not any(k in lowered for k in ("b站", "bilibili", "哔哩", "小破站")):
            return ""
        # 去掉平台名本身再当关键词
        keyword = re.sub(r"b站|bilibili|哔哩哔哩|小破站", " ", text, flags=re.I).strip()
        keyword = re.sub(r"[?？!！。，,、]+$", "", keyword).strip()
        if len(keyword) < 2:
            return ""
        outcome = await self.bili.search(keyword)
        if not outcome.ok:
            return ""
        log.info("B站搜索「%s」-> %d 条", keyword[:30], len(outcome.hits))
        return outcome.render(4)

    async def _maybe_periodic(self, group_id: int) -> None:
        count = self._group_counter.get(group_id, 0)
        if count and count % MEMORY_EXTRACT_EVERY == 0:
            recent = await self.repo.recent_messages(group_id, limit=MEMORY_EXTRACT_EVERY)
            self.stats["memory_batches"] += 1
            await self.memory.extract(group_id, recent)

        if self.cfg.evolution.profile_enabled and count and count % PROFILE_EVERY == 0:
            profile = await self.profile_builder.rebuild(group_id)
            if profile and count % REFLECT_EVERY == 0:
                await self._reflect(group_id, profile)

    async def _reflect(self, group_id: int, profile) -> None:
        """LLM 复盘：识别群主题、黑话、氛围。产出建议而非自动改人设。"""
        recent = await self.repo.recent_messages(group_id, limit=300)
        if len(recent) < 50:
            return
        transcript = "\n".join(
            f"{'我' if m.is_self else m.nickname}: {m.text}" for m in recent
        )
        system, user = build_profile_prompt(transcript, profile.topics)
        try:
            text = await self.llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.3, json_mode=True, profile="analysis",
            )
            payload = extract_json(text)
        except LLMError as exc:
            log.warning("群复盘失败: %s", exc)
            return

        await self.profile_builder.apply_topics(group_id, payload)
        self.stats["reflects"] += 1
        log.info("群复盘完成 [%s] note=%s", group_id, str(payload.get("note") or "")[:60])

    # ------------------------------------------------------------------
    # 决策 -> 生成 -> 发送
    # ------------------------------------------------------------------

    async def _load_history(self, conv: Conversation) -> None:
        """把本批之前的历史补上。

        不做这一步的话，模型只看得到防抖窗口（几秒）内攒的那几条消息 ——
        隔半分钟再说一句，它就完全不记得刚才聊过什么，像个失忆的人。
        """
        limit = max(0, self.cfg.behavior.context_messages)
        if limit <= 0:
            return

        recent = await self.repo.recent_messages(
            conv.group_id, limit=limit + len(conv.messages)
        )
        batch_ids = {m.message_id for m in conv.messages}
        history = [m for m in recent if m.message_id not in batch_ids]
        conv.history = history[-limit:] if limit else []

    async def _on_conversation(self, conv: Conversation) -> None:
        if self.paused or conv.trigger.is_self:
            return
        try:
            await self._run_conversation(conv)
        except Exception:  # noqa: BLE001 - 单轮失败不影响后续
            self.stats["errors"] += 1
            log.exception("处理会话异常 group=%s", conv.group_id)

    async def _run_conversation(self, conv: Conversation) -> None:
        group_id = conv.group_id
        now_mono = time.monotonic()
        now_wall = int(time.time())

        # 必须在决定层之前补 —— 决定层也要看上下文才能判断"这条是不是在回我"
        await self._load_history(conv)

        state = await self.mood.current()
        presence = self.scheduler.presence(now_wall)
        budget_ok, budget_reason = await self.scheduler.check_budget(group_id, now_wall)
        policy = await self.policies.get(group_id)
        profile = await self.profiles.get(group_id)

        # 看到感兴趣的话题就"回血" —— 哪怕最后没说话
        hits = self._topic_hits(conv, profile)
        if hits:
            await self.mood.update(on_topic=hits)
            state = await self.mood.current()

        decision = await self.decider.decide(
            conv,
            persona=self.persona_for(conv.group_id),
            presence=presence.value,
            budget_ok=budget_ok,
            budget_reason=budget_reason,
            now=now_mono,
            threshold_delta=policy.threshold_delta,
            excitement=state.excitement,
            mood=state.mood,
            extra_topics=list(profile.topics.keys()),
        )

        if not decision.should_reply:
            log.info("不发言 [%s] %s (分数 %.2f)", group_id, decision.reason, decision.priority)
            await self.repo.save_decision(group_id, decision)
            return

        log.info("决定发言 [%s] %s (分数 %.2f, 延迟 %.1fs)",
                 group_id, decision.reason, decision.priority, decision.delay)

        draft = await self.generate(conv, presence=presence, state=state, policy=policy, profile=profile)
        if draft is None or not draft.bubbles:
            decision.should_reply = False
            decision.reason = f"{decision.reason} -> 生成失败/为空"
            await self.repo.save_decision(group_id, decision)
            return

        decision.draft = "\n".join(b.text for b in draft.bubbles)
        recent_self = await self.repo.recent_self_texts(group_id, limit=8)
        plain = [b.render_text() for b in draft.bubbles]
        verdict = self.safety.check(plain, recent_self)
        if not verdict.ok:
            decision.should_reply = False
            decision.reason = f"{decision.reason} -> 安全拦截({verdict.reason})"
            await self.repo.save_decision(group_id, decision)
            log.warning("安全拦截 [%s]: %s", group_id, verdict.reason)
            return

        if decision.delay > 0:
            await asyncio.sleep(decision.delay)

        # 引用 / @ 由群策略的 Thompson 采样决定，不交给模型。
        # @ 还要额外过一道语义判断：群里只有一个人在说话时不该 @ ——
        # 没有歧义，@ 人反而显得刻意。真人也只在人多、需要指明对象时 @。
        speakers = {m.user_id for m in conv.all_messages if not m.is_self}
        use_quote = policy.sample_quote() and not conv.trigger.is_self
        use_at = (
            policy.sample_at()
            and not conv.trigger.is_self
            and len(speakers) >= 2
        )

        result = await self.sender.send(
            group_id, draft, trigger=conv.trigger, use_quote=use_quote, use_at=use_at
        )
        await self.repo.save_decision(group_id, decision)

        if result:
            self.decider.note_spoke(group_id, time.monotonic())
            await self.scheduler.note_spoke(group_id, now_wall)
            # 存活账本：这天说够话了才不算失职（防“少说少错”的捷径）
            await self.survival.note_spoke(group_id)
            await self.mood.update(on_spoke=True)

            # 记录用了哪些表情，供反馈归因和冷却
            for sticker_id in result.used_stickers:
                await self.stickers.bump_use(sticker_id)
                await self.stickers.record_usage(
                    sticker_id, group_id, result.message_ids[0] if result.message_ids else None,
                    context=conv.trigger.text,
                )

            await self.policies.bump_samples(group_id)
            await self.evolution.open_tracking(
                group_id, result.message_ids,
                attributes={
                    "used_quote": result.quote_used,
                    "used_at": result.at_used,
                    "used_sticker": bool(result.used_stickers),
                    "used_face": bool(result.used_faces),
                    "reason": decision.reason,
                    "priority": round(decision.priority, 3),
                },
            )

            # 表情包用过之后，攒够观察就跑一次打标
            if result.used_stickers or self.stickers:
                try:
                    await self.tagger.tag_pending()
                except Exception:  # noqa: BLE001
                    log.debug("表情打标失败", exc_info=True)

    def _topic_hits(self, conv: Conversation, profile) -> list[str]:
        text = conv.context_text(limit=6)
        hits = [t for t in list(profile.topics.keys()) if t and t in text]
        hits += [k for k in self.persona_for(conv.group_id).expert_topics
                 if k in text]
        return list(dict.fromkeys(hits))[:3]

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    async def generate(
        self,
        conv: Conversation,
        *,
        presence: Presence | None = None,
        state=None,
        policy=None,
        profile=None,
    ) -> Draft | None:
        """完整生成流程：记忆召回 -> 图片理解 -> prompt -> LLM -> 去 AI 味 -> 表情选用。"""
        group_id = conv.group_id
        presence = presence or self.scheduler.presence()
        state = state or await self.mood.current()
        policy = policy or await self.policies.get(group_id)
        profile = profile or await self.profiles.get(group_id)

        # 记忆召回用完整上下文当 query，而不是只有本批
        user_ids = [m.user_id for m in conv.all_messages[-10:] if not m.is_self]
        memories = await self.memory.recall(
            group_id, conv.context_text(limit=8), user_ids=user_ids
        )

        # 看看图跑完没有 —— 用户发的图多半就是要聊的，值得等一小会儿
        await self._await_media(conv.trigger)

        media_notes = self._pending_media.pop(conv.trigger.message_id, [])
        media_notes = self.media.pending_notes.pop(conv.trigger.message_id,
                                                   media_notes)
        # 再重读一次缓存：理解是异步的，结果可能比消息晚到，
        # 只用在收到消息那一刻算出来的说明，晚到的内容就永远用不上了。
        if conv.trigger.media_hash:
            try:
                media_notes = await self.media.refresh_notes(conv.trigger, media_notes)
            except Exception:  # noqa: BLE001
                log.debug("刷新图片说明失败", exc_info=True)

        is_mention = conv.trigger.mentions_self or any(
            m.mentions_self for m in conv.all_messages[-3:]
        )

        # 群友发的 B 站视频内容（在收到消息时就解析好了）
        self._prune_video()
        video_notes = self._take_video(conv.trigger.message_id)
        if not video_notes:
            # 触发消息没有，就在最近几条里找 —— 分享卡片常在触发消息的前一条
            for m in reversed(conv.messages[-3:]):
                video_notes = self._take_video(m.message_id)
                if video_notes:
                    break

        # 明确提到 B 站时优先搜 B 站
        bili_notes = ""
        if not video_notes:
            bili_notes = await self._bili_search(conv.trigger.text)

        # 按需联网搜索。走独立的时效性判定，绝大多数消息不会触发。
        search_notes = bili_notes
        outcome = await self.search.lookup(
            conv.trigger.text,
            has_url=any("http" in m.text for m in conv.messages[-3:]),
            is_mention=is_mention,
            # 别把自己的名字搜进去 —— 「小七今天天气」会污染结果
            strip_words=self.persona_for(conv.group_id).trigger_words(),
        )
        if outcome.performed:
            search_notes = outcome.render(self.cfg.search.max_snippet_chars)
            log.info("联网搜索「%s」-> %d 条（%s）",
                     outcome.query[:40], len(outcome.results), outcome.reason)

        sticker_moods = await self.selector.mood_labels()
        user_prompt = build_user(
            conv,
            self.persona_for(group_id),
            memories=memories,
            presence=presence.label,
            mood_text=state.describe(),
            now_text=datetime.now().strftime("%H:%M"),
            is_mention=is_mention,
            profile_note=build_profile_note(profile),
            media_notes=media_notes,
            search_notes=search_notes,
            video_notes=video_notes,
            sticker_moods=sticker_moods,
        )

        try:
            # 显式走 reply 画像：群聊要快，预算和时间都收紧
            draft = await self.llm.draft(
                await self.system_prompt(group_id), user_prompt, profile="reply"
            )
        except LLMError as exc:
            log.warning("LLM 生成失败: %s", exc)
            return None

        result = postprocess_bubbles(
            draft.bubbles,
            max_chars=self.cfg.behavior.max_chars_per_bubble,
            max_bubbles=self.cfg.behavior.max_bubbles,
            face_keep_prob=self.cfg.behavior.face_keep_prob,
        )
        if result.regenerated:
            log.info("检测到 AI 味，丢弃本轮: %s", "; ".join(result.reasons))
            return None
        if not result.ok:
            log.info("后处理丢弃本轮: %s", "; ".join(result.reasons))
            return None

        # 解析表情包意图 -> 具体一张图
        for bubble in result.bubbles:
            if bubble.sticker and not bubble.sticker_ref:
                ref = await self.selector.resolve(
                    bubble.sticker, group_id, context=conv.trigger.text
                )
                if ref is None:
                    # 库里没有匹配的，且它是纯表情包气泡 -> 整条丢掉
                    bubble.sticker = None
                else:
                    bubble.sticker_ref = ref

        result.bubbles = [b for b in result.bubbles if b]
        if not result.bubbles:
            log.info("表情包解析后没有可发内容")
            return None

        return Draft(
            bubbles=result.bubbles,
            thought=draft.thought,
            quote=draft.quote,
            at=draft.at,
            emoji_like=draft.emoji_like,
        )

    # ------------------------------------------------------------------
    # 干跑（面板的"测试"按钮）
    # ------------------------------------------------------------------

    async def dry_run(self, group_id: int, text: str, *, force: bool = False) -> dict:
        msg = Message(
            group_id=group_id,
            message_id=0,
            user_id=0,
            nickname="测试用户",
            text=text,
            ts=int(time.time()),
        )
        # 先解析该群的人设 —— 下面 mentions_self / decide / build_user 都要用
        persona = await self.load_persona_for(group_id)
        msg.mentions_self = persona.name in text

        conv = Conversation(group_id=group_id, messages=[msg], trigger=msg)
        await self._load_history(conv)

        # 让"试生成"也能测 B 站链接解析 / 视频搜索
        try:
            await self._handle_bilibili(msg)
        except Exception:  # noqa: BLE001
            log.debug("试生成里的 B 站解析失败", exc_info=True)

        state = await self.mood.current()
        policy = await self.policies.get(group_id)
        profile = await self.profiles.get(group_id)
        decision = await self.decider.decide(
            conv,
            persona=self.persona_for(conv.group_id),
            presence=self.scheduler.presence().value,
            budget_ok=True,
            now=time.monotonic(),
            threshold_delta=policy.threshold_delta,
            excitement=state.excitement,
            mood=state.mood,
            extra_topics=list(profile.topics.keys()),
        )

        out = {
            "should_reply": decision.should_reply,
            "priority": round(decision.priority, 3),
            "reason": decision.reason,
            "draft": [],
            "forced": False,
            "context": render_conversation(conv),
            "context_count": len(conv.all_messages),
            # 用 get 不用 pop —— 下面 generate() 还要取这一份。
            # 写 pop 的话会在 generate() 之前把块取空，模型就真看不到视频内容了
            # （表现就是它说"打不开""懒得点"）。
            "video": (self._pending_video.get(msg.message_id) or (0, ""))[1],
            "error": None,
        }
        if force and not decision.should_reply:
            # 面板勾了"强制生成"：绕开决定层，单纯用来试模型与 prompt
            out["reason"] = f"{decision.reason}（已强制生成）"
            out["forced"] = True
            out["should_reply"] = True
            decision.should_reply = True

        if decision.should_reply:
            draft = await self.generate(conv)
            if draft is None:
                out["error"] = "生成失败或被后处理丢弃（详见日志）"
            else:
                out["draft"] = [b.render_text() for b in draft.bubbles]
        return out

    # ------------------------------------------------------------------
    # 定时任务
    # ------------------------------------------------------------------

    async def on_tick(self) -> None:
        """每分钟一次：反馈收获、记忆维护、在线状态同步、媒体清理。"""
        try:
            await self.evolution.harvest()
        except Exception:  # noqa: BLE001
            log.debug("反馈收获失败", exc_info=True)

        try:
            await self.memory.forget()
        except Exception:  # noqa: BLE001
            log.debug("记忆维护失败", exc_info=True)

        try:
            await self._sync_online_status()
        except Exception:  # noqa: BLE001
            pass

        # B 站 Cookie 健康检查。半小时一次，很便宜（一个 nav 请求）——
        # 但能在 Cookie 掉的时候主动记下来，用户打开面板就看到提示，
        # 不用等到某次解析视频失败才发现。
        try:
            await self._check_bili_cookie()
        except Exception:  # noqa: BLE001
            log.debug("B站 Cookie 检查失败", exc_info=True)

        # 收云端 OCR 结果。PaddleOCR 是任务制的，这是唯一的回收入口。
        try:
            await self.media.poll_jobs()
        except Exception:  # noqa: BLE001
            log.debug("PaddleOCR 结果回收失败", exc_info=True)

        # 伪装存活结算：结算“已经过完但还没结”的日子。
        # 放在 tick 里而不是定时器，是因为它要补算停机期间留下的空洞。
        try:
            await self.survival.settle_due()
        except Exception:  # noqa: BLE001
            log.debug("存活结算失败", exc_info=True)

        # 冷启动留下的疑问，找机会在群里问掉
        try:
            await self._maybe_ask_question()
        except Exception:  # noqa: BLE001
            log.debug("提问调度失败", exc_info=True)

        # 每 10 分钟打一次表情标签
        if int(time.time()) % 600 < 60:
            try:
                await self.tagger.tag_pending()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 面板快照
    # ------------------------------------------------------------------

    async def control_snapshot(self) -> dict:
        state = await self.mood.current()
        # 这两个 stats() 是异步的（要查库），不能直接 ** 展开
        survival_stats = await self.survival_repo.stats()
        onboard_stats = await self.onboard_repo.stats()
        return {
            "paused": self.paused,
            "uptime_seconds": int(time.time() - self.started_at),
            "started_at": int(self.started_at),
            "self_id": self.self_id,
            "queue_size": self.pool.pending,
            "presence": self.scheduler.describe(),
            "mood": state.snapshot(),
            "debug": {
                "enabled": bool(self.cfg.debug.enabled),
                "relax_threshold": self.cfg.debug.relax_threshold,
                "respond_to_all": bool(self.cfg.debug.respond_to_all),
                "affects": (["作息", "发言额度", "群冷却", "随机放弃", "发言门槛"]
                            + (["响应所有消息"] if self.cfg.debug.respond_to_all else [])),
            },
            "stats": {
                **self.stats,
                **self.llm.stats.as_dict(),
                **self.collector.stats(),
                **self.selector.stats(),
                **self.tagger.stats(),
                **self.media.stats(),
                **self.paddle.stats(),
                **self.search.stats(),
                **self.bili.stats(),
                **self.bili_login.stats(),
                **self.asr.stats(),
                **self.evolution.stats(),
                **self.profile_builder.stats(),
                # 这一轮新增的几层，统计也要露出来 —— 否则面板上看不到
                # 决策模式、存活结算、非文字段解析有没有在工作
                **self.decider.stats(),
                **survival_stats,
                **onboard_stats,
                **self.ingest.stats(),
                "pages_total": 0,
                "messages_sent": self.sender.sent_count,
                "send_failures": self.sender.failed_count,
                "send_blocked": self.safety.blocked_count,
                "gives_up": self.decider.give_ups,
                "conversations_flushed": self.pool.flushed,
            },
        }

    async def close(self) -> None:
        await self.pool.close()
        await self.sticker_lib.close()
        await self.image_store.close()
        await self.paddle.close()
        await self.bili_login.close()
        await self.bili_client.close()
        await self.asr.close()
        await self.search_client.close()
        await self.llm.close()
