"""测试套件。不依赖 pytest，直接跑：

    ./.venv/bin/python -m tests.test_qqbot
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qqbot.buffer.pool import MessagePool  # noqa: E402
from qqbot.config import BehaviorConfig, Config  # noqa: E402
from qqbot.decide.scorer import Decider  # noqa: E402
from qqbot.generate.llm import extract_json, to_draft  # noqa: E402
from qqbot.generate.postprocess import clean_text, postprocess, split_bubble  # noqa: E402
from qqbot.generate.prompt import build_system, build_user  # noqa: E402
from qqbot.memory import MemoryManager, similarity, tokenize  # noqa: E402
from qqbot.models import Bubble, Conversation, Draft, Message  # noqa: E402
from qqbot.persona import load as load_persona  # noqa: E402
from qqbot.pipeline import Pipeline  # noqa: E402
from qqbot.safety.filter import SafetyFilter  # noqa: E402
from qqbot.schedule.state import Presence, Scheduler  # noqa: E402
from qqbot.store import Database, MemoryRepo, Repo  # noqa: E402

TESTS: list = []


def test(fn):
    TESTS.append(fn)
    return fn


def eq(actual, expected, label: str = "") -> None:
    if actual != expected:
        raise AssertionError(f"{label} 期望 {expected!r}，实际 {actual!r}")


def ok(cond, label: str = "") -> None:
    if not cond:
        raise AssertionError(f"{label} 断言失败")


# ==========================================================================
# 后处理 —— 去 AI 味
# ==========================================================================


@test
def t_clean_removes_ai_tells() -> None:
    eq(clean_text("总的来说这个还行"), "这个还行", "删除路标词")
    eq(clean_text("作为一个AI，我不能"), "AI，我不能", "删除开场白")


@test
def t_clean_strips_wrappers() -> None:
    eq(clean_text("回复：你好啊"), "你好啊", "去掉 回复： 前缀")
    eq(clean_text('"笑死"'), "笑死", "去掉引号")
    eq(clean_text("确实。"), "确实", "去掉句末句号")
    eq(clean_text("牛！！！"), "牛！", "折叠重复标点")


@test
def t_clean_empty() -> None:
    eq(clean_text(""), "", "空串")
    eq(clean_text("   "), "", "纯空白")
    # 单个句末句号会被 TRAILING_PERIOD 剥掉，剩下的空串由 postprocess 兜底丢弃
    eq(clean_text("。"), "", "纯句号剥成空")


@test
def t_split_bubble() -> None:
    eq(split_bubble("短", 40), ["短"], "不超过上限不分")
    parts = split_bubble("第一句。第二句。第三句。", 6)
    ok(len(parts) >= 2, "长文本应被切分")
    ok(all(len(p) <= 6 or "。" not in p for p in parts), "每段尽量不超限")
    long_parts = split_bubble("啊" * 25, 10)
    ok(all(len(p) <= 10 for p in long_parts), "无标点时硬切")
    ok("".join(long_parts) == "啊" * 25, "硬切不丢字")


@test
def t_postprocess_caps_bubbles() -> None:
    r = postprocess(["一", "二", "三", "四", "五"], max_bubbles=3)
    eq(len(r.messages), 3, "气泡数截断")
    ok(r.dropped == 2, "记录丢弃数")


@test
def t_postprocess_regen_marker() -> None:
    r = postprocess(["我是一个人工智能助手"], check_regen=True)
    ok(r.regenerated, "命中 AI 味标记应触发重生成")
    eq(r.messages, [], "重生成时不应有内容")


@test
def t_postprocess_drops_placeholder() -> None:
    r = postprocess(["", "   ", "真的"])
    eq(r.messages, ["真的"], "丢弃空白")


# ==========================================================================
# 安全检查
# ==========================================================================


@test
def t_safety_blocklist() -> None:
    f = SafetyFilter(BehaviorConfig())
    ok(not f.check(["我们来聊聊自杀"]).ok, "命中黑名单应拒绝")
    ok(f.check(["今天天气不错"]).ok, "正常内容应通过")


@test
def t_safety_url() -> None:
    f = SafetyFilter(BehaviorConfig())
    ok(not f.check(["看这个 https://example.com"]).ok, "链接应被拦截")


@test
def t_safety_clamps_length() -> None:
    cfg = BehaviorConfig(max_chars_per_bubble=10)
    f = SafetyFilter(cfg)
    r = f.check(["啊" * 50])
    ok(r.ok, "应通过")
    ok(all(len(m) <= 10 for m in r.messages), "超长应被截断")


@test
def t_safety_dedup() -> None:
    f = SafetyFilter(BehaviorConfig())
    r = f.check(["笑死"], recent_self=["笑死"])
    ok(not r.ok, "与最近发言重复应丢弃")


@test
def t_safety_empty() -> None:
    f = SafetyFilter(BehaviorConfig())
    ok(not f.check([]).ok, "空列表拒绝")
    ok(not f.check(["   "]).ok, "纯空白拒绝")


@test
def t_safety_fail_closed() -> None:
    f = SafetyFilter(BehaviorConfig())
    f.cfg = None  # 制造内部异常
    r = f.check(["任意内容"])
    ok(not r.ok, "内部异常必须 fail-closed")


# ==========================================================================
# 决定层
# ==========================================================================


def _msg(text: str, *, uid: int = 1, nickname: str = "张三",
         mention: bool = False, is_self: bool = False,
         reply_to: int | None = None, ts: int | None = None) -> Message:
    return Message(
        group_id=100, message_id=ts or int(time.time() * 1000), user_id=uid,
        nickname=nickname, text=text, ts=ts or int(time.time()),
        is_self=is_self, mentions_self=mention, reply_to=reply_to,
    )


def _conv(messages: list[Message]) -> Conversation:
    return Conversation(group_id=100, messages=messages, trigger=messages[-1])


def _decider(**kwargs) -> Decider:
    base = dict(
        give_up_prob=0.0,          # 测试中关掉随机放弃
        group_cooldown=0.0,
        threshold=0.55,
    )
    base.update(kwargs)
    return Decider(BehaviorConfig(**base))


@test
def t_decide_mention() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider()
    conv = _conv([_msg("在吗", mention=True)])
    dec = d.decide(conv, persona=persona, presence="online", now=1000.0)
    ok(dec.should_reply, "被 @ 应该回复")
    eq(dec.priority, 1.0, "被 @ 满分")
    ok("被@" in dec.reason, f"原因应含被@，实际 {dec.reason}")


@test
def t_decide_mention_only_mode() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider(mention_only=True)
    conv = _conv([_msg("小七在吗")])  # 提到名字但没 @
    dec = d.decide(conv, persona=persona, presence="online", now=1000.0)
    ok(not dec.should_reply, "仅@模式下名字触发也应拒绝")
    eq(dec.reason, "仅@模式")


@test
def t_decide_sleeping() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider()
    dec = d.decide(_conv([_msg("在吗", mention=True)]), persona=persona,
                   presence="sleeping", now=1000.0)
    ok(not dec.should_reply, "睡觉时不应回复")


@test
def t_decide_cooldown() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider(group_cooldown=100.0)
    d.note_spoke(100, now=1000.0)
    dec = d.decide(_conv([_msg("在吗", mention=True)]), persona=persona,
                   presence="online", now=1050.0)
    ok(not dec.should_reply, "冷却期内应拒绝")
    ok("冷却" in dec.reason, f"原因应含冷却，实际 {dec.reason}")

    # 冷却结束后可以
    dec2 = d.decide(_conv([_msg("在吗", mention=True)]), persona=persona,
                    presence="online", now=1200.0)
    ok(dec2.should_reply, "冷却结束后应可回复")


@test
def t_decide_budget_exhausted() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider()
    dec = d.decide(_conv([_msg("在吗", mention=True)]), persona=persona,
                   presence="online", budget_ok=False,
                   budget_reason="本群今日额度用完", now=1000.0)
    ok(not dec.should_reply, "额度用完应拒绝")
    eq(dec.reason, "本群今日额度用完")


@test
def t_decide_low_score_rejected() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider()
    dec = d.decide(_conv([_msg("今天下雨了")]), persona=persona,
                   presence="online", now=1000.0)
    ok(not dec.should_reply, "无关闲聊不应插话")


@test
def t_decide_reply_to_self() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider()
    mine = _msg("我说了句话", is_self=True, ts=100)
    reply = _msg("回你一句", reply_to=mine.message_id, ts=200)
    dec = d.decide(_conv([mine, reply]), persona=persona, presence="online", now=1000.0)
    ok(dec.should_reply, "回复我的消息应该接话")
    ok("回复了我" in dec.reason, f"实际 {dec.reason}")


@test
def t_decide_features() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    d = _decider()
    f = d.features(_conv([_msg("小七 这个编程问题怎么搞？", mention=True)]), persona)
    ok(f["mentioned"], "mentioned")
    ok(f["name_hit"], "name_hit 应命中别名")
    ok(f["topic_hits"], "topic_hits 应命中'编程'")
    ok(f["question"], "question")


# ==========================================================================
# 蓄水池
# ==========================================================================


@test
async def t_pool_debounce() -> None:
    got: list[Conversation] = []

    async def cb(conv: Conversation) -> None:
        got.append(conv)

    cfg = BehaviorConfig(debounce=0.15, max_wait=5.0, max_batch=10)
    pool = MessagePool(cb, cfg)

    await pool.push(_msg("一", ts=1))
    await pool.push(_msg("二", ts=2))
    await pool.push(_msg("三", ts=3))
    eq(len(got), 0, "防抖未到期不应触发")

    await asyncio.sleep(0.4)
    eq(len(got), 1, "应触发一次")
    eq([m.text for m in got[0].messages], ["一", "二", "三"], "攒成一批")
    eq(got[0].trigger.text, "三", "触发消息是最后一条")
    await pool.close()


@test
async def t_pool_max_batch() -> None:
    got: list[Conversation] = []

    async def cb(conv: Conversation) -> None:
        got.append(conv)

    cfg = BehaviorConfig(debounce=10.0, max_wait=60.0, max_batch=3)
    pool = MessagePool(cb, cfg)
    for i in range(3):
        await pool.push(_msg(f"m{i}", ts=i))

    eq(len(got), 1, "达到 max_batch 应立即触发")
    eq(len(got[0].messages), 3, "整批交出")
    await pool.close()


@test
async def t_pool_separate_groups() -> None:
    got: list[Conversation] = []

    async def cb(conv: Conversation) -> None:
        got.append(conv)

    cfg = BehaviorConfig(debounce=0.1, max_wait=5.0, max_batch=10)
    pool = MessagePool(cb, cfg)

    a = _msg("a", ts=1)
    b = _msg("b", ts=2)
    b.group_id = 200
    await pool.push(a)
    await pool.push(b)

    await asyncio.sleep(0.35)
    eq(len(got), 2, "两个群应各自触发")
    eq(sorted(c.group_id for c in got), [100, 200], "群隔离")
    await pool.close()


# ==========================================================================
# 作息与额度
# ==========================================================================


async def _tmp_repo() -> tuple[Database, Repo]:
    tmp = Path(tempfile.mkdtemp()) / "t.db"
    db = await Database.open(tmp)
    return db, Repo(db)


@test
async def t_scheduler_roll_is_bounded() -> None:
    """掷骰子判定"外出/忙碌"要能确定性验证 —— 用极小的时间片推。

    单独测这个是因为上面的时段测试必须把概率归零（否则会随机失败），
    那样就覆盖不到 _maybe_roll 了。
    """
    db, repo = await _tmp_repo()
    try:
        import random as _r

        cfg = BehaviorConfig(active_from="00:00", active_to="23:59",
                             jitter_minutes=0, away_prob_per_hour=1000.0)
        s = Scheduler(cfg, repo)

        _r.seed(1234)
        now = 1_700_000_000.0
        # 推进一小时：概率 1000/小时，几乎必然进入外出
        eq(s.presence(now), Presence.ONLINE, "初始应在线")
        got = s.presence(now + 3600)
        ok(got in (Presence.AWAY, Presence.BUSY), f"高概率下应离开在线，实际 {got}")

        # 概率归零时必须是稳定的 ONLINE
        cfg2 = BehaviorConfig(active_from="00:00", active_to="23:59",
                              jitter_minutes=0, away_prob_per_hour=0.0)
        s2 = Scheduler(cfg2, repo)
        for _ in range(20):
            eq(s2.presence(now), Presence.ONLINE, "概率为 0 时应恒为在线")
    finally:
        await db.close()


@test
async def t_scheduler_window() -> None:
    db, repo = await _tmp_repo()
    try:
        # away/busy 的概率必须归零：这个测试验的是**时段判定**，
        # 而 presence() 在时段内还会掷骰子决定"外出/忙碌"。
        # 不关掉的话，noon 是未来时间戳（比如凌晨跑测试时），
        # _maybe_roll 会算出十几个小时的巨大间隔 → 约 43% 概率随机失败。
        # 注意：busy 的概率是 away 的 0.8 倍派生的，没有独立字段
        cfg = BehaviorConfig(active_from="09:00", active_to="18:00",
                             jitter_minutes=0, away_prob_per_hour=0.0)
        s = Scheduler(cfg, repo)

        # 用本地时间构造
        import datetime as dt

        noon = dt.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        midnight = dt.datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)

        ok(s.in_window(noon.timestamp()), "中午应在活动时段内")
        ok(not s.in_window(midnight.timestamp()), "凌晨应在时段外")
        eq(s.presence(midnight.timestamp()), Presence.SLEEPING, "时段外=睡觉")
        eq(s.presence(noon.timestamp()), Presence.ONLINE, "时段内=在线")
    finally:
        await db.close()


@test
async def t_scheduler_budget() -> None:
    db, repo = await _tmp_repo()
    try:
        cfg = BehaviorConfig(group_hour_budget=3, group_day_budget=10, global_day_budget=100)
        s = Scheduler(cfg, repo)

        ok((await s.check_budget(100))[0], "初始应允许")

        for _ in range(3):
            await s.note_spoke(100)

        allowed, reason = await s.check_budget(100)
        ok(not allowed, "超时额度应拒绝")
        ok("本群" in reason, f"原因应指明范围，实际 {reason}")
    finally:
        await db.close()


@test
async def t_scheduler_global_budget() -> None:
    db, repo = await _tmp_repo()
    try:
        cfg = BehaviorConfig(group_hour_budget=100, group_day_budget=100, global_day_budget=2)
        s = Scheduler(cfg, repo)
        await s.note_spoke(1)
        await s.note_spoke(2)
        allowed, reason = await s.check_budget(3)
        ok(not allowed, "全局额度用尽应拒绝")
        ok("全局" in reason, f"实际 {reason}")
    finally:
        await db.close()


# ==========================================================================
# 记忆
# ==========================================================================


@test
def t_tokenize_and_similarity() -> None:
    ok("编程" in tokenize("我在学编程"), "中文 bigram")
    ok("python" in tokenize("学 Python"), "英文词")
    ok(similarity("喜欢打游戏", "喜欢打游戏") == 1.0, "完全相同=1")
    ok(similarity("喜欢打游戏", "今天天气不错") < 0.1, "无关接近 0")
    ok(similarity("喜欢打游戏", "喜欢玩游戏") > 0.2, "部分重合")


@test
async def t_memory_typed_and_recall() -> None:
    db, _repo = await _tmp_repo()
    try:
        mem = MemoryRepo(db)
        manager = MemoryManager(mem, BehaviorConfig(memory_recall_top=3),
                                _evo_cfg(), llm=None)
        # 分类型写入：culture 半衰期远长于 event
        await mem.add("group", 100, "张三喜欢打游戏", type="fact", group_id=100)
        await mem.add("group", 100, "本群管打游戏叫上班", type="culture", group_id=100)
        await mem.add("group", 100, "上周六联机到两点", type="event", group_id=100)

        rows = await mem.active("group", 100)
        eq(len(rows), 3, "三条都应有效")
        types = {r["type"] for r in rows}
        ok("culture" in types and "event" in types, "类型应被保留")
        # TTL 按类型自动填
        by_type = {r["type"]: r["ttl_days"] for r in rows}
        ok(by_type["culture"] > by_type["event"], "culture 应比 event 记得久")

        recalled = await manager.recall(100, "张三在打什么游戏")
        ok(recalled, f"应召回相关内容，实际 {recalled}")
        ok(any("上班" in r or "游戏" in r for r in recalled), f"实际 {recalled}")
    finally:
        await db.close()


@test
async def t_memory_invalidate_not_delete() -> None:
    """矛盾消解：旧记忆应被标记失效，而不是删除（保留演化历史）。"""
    db, _repo = await _tmp_repo()
    try:
        mem = MemoryRepo(db)
        old = await mem.add("group", 100, "张三在准备考研", type="fact", group_id=100)
        await mem.invalidate(old, now=int(time.time()))

        eq(len(await mem.active("group", 100)), 0, "失效的不该出现在 active")
        eq(len(await mem.all_for("group", 100)), 1, "但历史里还能查到")
        stats = await mem.stats()
        eq(stats["memories_invalidated"], 1, "应记一笔失效")
    finally:
        await db.close()


@test
async def t_memory_ttl_expiry() -> None:
    """过期记忆自动淡出。"""
    db, _repo = await _tmp_repo()
    try:
        mem = MemoryRepo(db)
        old = int(time.time()) - 400 * 86400
        await mem.add("group", 100, "很久以前的事", type="event",
                      ttl_days=7, now=old)
        eq(len(await mem.active("group", 100)), 0, "过期的应被过滤")
    finally:
        await db.close()


def _evo_cfg():
    from qqbot.config import EvolutionConfig
    return EvolutionConfig()


# ==========================================================================
# LLM 输出解析
# ==========================================================================


@test
def t_extract_json_variants() -> None:
    eq(extract_json('{"a":1}')["a"], 1, "裸 JSON")
    eq(extract_json('```json\n{"a":2}\n```')["a"], 2, "markdown 围栏")
    eq(extract_json('好的，这是结果：{"a":3} 希望有帮助')["a"], 3, "前后有废话")
    eq(extract_json('{"a":{"b":{"c":4}}}')["a"]["b"]["c"], 4, "嵌套")
    eq(extract_json('{"text":"含 } 花括号"}')["text"], "含 } 花括号", "字符串里的花括号")


@test
def t_extract_json_invalid() -> None:
    for bad in ["", "没有任何 JSON", "{坏的"]:
        try:
            extract_json(bad)
            raise AssertionError(f"应当抛错: {bad!r}")
        except Exception as exc:
            ok("LLMError" in type(exc).__name__ or "解析" in str(exc), f"{bad!r} -> {exc}")


@test
def t_to_draft() -> None:
    d = to_draft({"thought": "想想", "messages": ["一", "二"], "emoji_like": None})
    eq(d.messages, ["一", "二"], "多条")
    eq(d.thought, "想想")

    d2 = to_draft({"messages": "单条字符串"})
    eq(d2.messages, ["单条字符串"], "字符串应包成列表")

    d3 = to_draft({"messages": []})
    eq(d3.messages, [], "空列表")
    ok(not d3, "空草稿为假值")


# ==========================================================================
# Prompt
# ==========================================================================


@test
def t_prompt_contains_constraints() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    cfg = BehaviorConfig(max_chars_per_bubble=25, max_bubbles=2)
    s = build_system(persona, cfg)
    ok("小七" in s, "含名字")
    ok("25" in s, "含长度上限")
    ok("JSON" in s, "含输出格式")
    ok(persona.catchphrases[0] in s, "含口头禅")


@test
def t_prompt_user_includes_context() -> None:
    persona = load_persona(ROOT / "persona.yaml")
    conv = _conv([_msg("你好"), _msg("小七", mention=True)])
    u = build_user(conv, persona, memories=["张三喜欢打游戏"],
                   presence="在线", now_text="21:30", is_mention=True)
    ok("张三喜欢打游戏" in u, "含记忆")
    ok("21:30" in u, "含时间")
    ok("你好" in u, "含上下文")
    ok("@ 了你" in u, "标记被@")
    ok("在线" in u, "含状态")


# ==========================================================================
# 端到端：完整管线（stub LLM）
# ==========================================================================


class StubLLM:
    """替身 LLM，返回固定草稿，避免测试依赖网络。"""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {
            "thought": "随便接一句",
            "messages": ["确实", "我也这么觉得"],
        }
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    async def draft(self, system: str, user: str, retries: int = 1,
                    profile: str = "reply") -> Draft:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        return to_draft(self.payload)

    async def chat(self, *a, **k):  # 记忆抽取用
        return '{"memories": []}'

    async def close(self) -> None:
        return None

    class _Stats:
        calls = 0
        errors = 0
        tokens_in = 0
        tokens_out = 0

        def as_dict(self) -> dict:
            return {}

    stats = _Stats()


class StubAPI:
    """替身 NapCat API：记录发出的段，不碰网络。"""

    def __init__(self) -> None:
        self.sent: list[tuple[int, list[dict]]] = []
        self._next_id = 500000

    async def send_group_segments(self, group_id: int, segments: list[dict]) -> int:
        self.sent.append((group_id, segments))
        self._next_id += 1
        return self._next_id

    async def send_group_msg(self, group_id: int, text: str) -> int:
        return await self.send_group_segments(group_id, [{"type": "text",
                                                          "data": {"text": text}}])

    @property
    def sent_texts(self) -> list[str]:
        from qqbot.segments import describe
        return [describe(segs) for _, segs in self.sent]

    @property
    def sent_bodies(self) -> list[str]:
        """只取 text 段 —— 不含 @ 前缀和引用，用于断言长度约束。"""
        out = []
        for _, segs in self.sent:
            body = "".join(str(s["data"].get("text", ""))
                           for s in segs if s.get("type") == "text")
            out.append(body)
        return out

    async def set_msg_emoji_like(self, *a, **k) -> None:
        return None

    async def get_emoji_likes(self, message_id) -> list:
        return []

    async def ocr_image(self, image: str) -> str:
        return ""

    async def fetch_ptt_text(self, message_id) -> str:
        return ""

    async def set_online_status(self, *a, **k) -> None:
        return None

    async def get_login_info(self) -> dict:
        return {"nickname": "测试号", "user_id": 999}

    async def get_group_list(self) -> list[dict]:
        return [{"group_id": 100, "group_name": "测试群"}]


async def _pipeline(tmp: Path, **behavior) -> tuple[Database, Repo, Pipeline, StubAPI, StubLLM]:
    db = await Database.open(tmp / "e2e.db")
    repo = Repo(db)
    api = StubAPI()

    cfg = Config(
        behavior=BehaviorConfig(
            give_up_prob=0.0,
            group_cooldown=0.0,
            active_from="00:00",
            active_to="23:59",
            jitter_minutes=0,
            away_prob_per_hour=0.0,
            debounce=0.05,
            max_wait=1.0,
            **behavior,
        ),
        persona_path=ROOT / "persona.yaml",
        root=ROOT,
    )
    cfg.storage.db_path = tmp / "e2e.db"

    pipeline = Pipeline(cfg, repo, api)
    stub = StubLLM()
    pipeline.llm = stub  # type: ignore[assignment]
    pipeline.self_id = 999
    pipeline.sender.self_id = 999

    await repo.upsert_group(100, "测试群", default_enabled=True)
    return db, repo, pipeline, api, stub


@test
async def t_e2e_mention_replies_and_backfills() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)
    try:
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)

        eq(len(api.sent), 2, "应发出两条（分条发送）")
        eq(api.sent_bodies, ["确实", "我也这么觉得"], "内容正确")
        eq(stub.calls, 1, "LLM 只调用一次")
        ok("小七" in stub.last_system, "system prompt 含人设")

        # 关键：自己的发言必须回填进时间线
        recent = await repo.recent_messages(100, limit=10)
        mine = [m for m in recent if m.is_self]
        eq(len(mine), 2, "自己发的两条应已回填")
        eq(mine[0].user_id, 999, "回填的 user_id 是 self_id")

        # 决策应留痕
        decisions = await repo.list_decisions(10, 100)
        eq(len(decisions), 1, "应有决策记录")
        ok(decisions[0].should_reply, "记录为已发言")
        ok("确实" in decisions[0].draft, "记录了草稿")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_disabled_group_silent() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)
    try:
        await repo.set_group_enabled(100, False)
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)
        eq(len(api.sent), 0, "关闭的群不应发言")
        eq(stub.calls, 0, "不应调用 LLM")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_paused_silent() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)
    try:
        pipeline.paused = True
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)
        eq(len(api.sent), 0, "暂停时不应发言")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_dedup() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)
    try:
        m = _msg("重复消息", ts=12345)
        await pipeline.handle_message(m)
        await pipeline.handle_message(m)  # 同 message_id 重推
        eq(pipeline.stats["duplicates_dropped"], 1, "应识别为重复")
        eq(pipeline.stats["messages_received"], 1, "只计一次")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_safety_blocks() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)
    try:
        stub.payload = {"messages": ["把验证码发我就能办"]}
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)
        eq(len(api.sent), 0, "命中黑名单不应发出")
        ok(pipeline.safety.blocked_count > 0, "应记录拦截次数")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_postprocess_trims() -> None:
    """LLM 写了小作文 -> 应被切成多条并截断到 max_bubbles。"""
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp, max_chars_per_bubble=12, max_bubbles=2)
    try:
        stub.payload = {
            "messages": ["第一句话挺长的需要切分。第二句也不短。第三句也长。第四句。"]
        }
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)

        ok(len(api.sent) <= 2, f"最多 2 条，实际 {len(api.sent)}")
        ok(all(len(t) <= 12 for t in api.sent_bodies), f"每条正文<=12字: {api.sent_bodies}")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_budget_consumed() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp, group_day_budget=1, group_hour_budget=0)
    try:
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)
        eq(len(api.sent), 2, "第一次应发言")

        await pipeline.handle_message(_msg("小七再问一次", mention=True, ts=int(time.time()) + 1))
        await pipeline.pool.flush(100)
        eq(len(api.sent), 2, "额度用尽后不应再发")

        state = await repo.budget_state(100)
        eq(state["day_used"], 1, "额度应被扣减")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_dry_run() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)
    try:
        out = await pipeline.dry_run(100, "小七在吗")
        ok(out["should_reply"], "干跑应判为要回复")
        ok(out["draft"], "干跑应返回草稿")
        eq(len(api.sent), 0, "干跑绝不能真的发送")
    finally:
        await pipeline.pool.close()
        await db.close()


@test
async def t_e2e_llm_failure_is_silent() -> None:
    tmp = Path(tempfile.mkdtemp())
    db, repo, pipeline, api, stub = await _pipeline(tmp)

    from qqbot.generate.llm import LLMError

    class FailingLLM(StubLLM):
        async def draft(self, system: str, user: str, retries: int = 1,
                    profile: str = "reply") -> Draft:
            raise LLMError("模拟上游 500")

    pipeline.llm = FailingLLM()  # type: ignore[assignment]
    try:
        await pipeline.handle_message(_msg("小七在吗", mention=True))
        await pipeline.pool.flush(100)
        eq(len(api.sent), 0, "LLM 失败应静默跳过")
        decisions = await repo.list_decisions(5, 100)
        eq(len(decisions), 1, "仍应留痕")
        ok(not decisions[0].should_reply, "记为未发言")
    finally:
        await pipeline.pool.close()
        await db.close()


# ==========================================================================
# 运行器
# ==========================================================================


async def run_all() -> int:
    print(f"\n运行 {len(TESTS)} 个测试…\n")
    passed = 0
    failures: list[tuple[str, str]] = []

    for fn in TESTS:
        name = fn.__name__
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            print(f"  \033[32m✓\033[0m {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  \033[31m✗\033[0m {name}: {exc}")
            failures.append((name, traceback.format_exc()))

    print(f"\n{'=' * 60}")
    print(f"通过 {passed}/{len(TESTS)}")
    if failures:
        print("\n失败详情:")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all()))
