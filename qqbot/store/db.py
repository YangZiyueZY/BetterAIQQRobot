"""SQLite 连接、建表与轻量迁移。

选 aiosqlite 而不是 ORM：表结构简单，手写 SQL 更透明。
表结构演进用 `_ensure_column` 做增量迁移，不丢历史数据。
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

# 注意：这是 SQL 字符串，注释只能用 -- ，不能用 Python 的 """ 文档字符串。
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- 群消息全量时间线，是所有上层功能的地基。
-- is_self=1 的行由 sender 发出后回填，因为 NapCat 的
-- reportSelfMessage=false 不会把机器人自己的消息推给我们。
CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id      INTEGER NOT NULL,
    message_id    INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    nickname      TEXT    NOT NULL DEFAULT '',
    text          TEXT    NOT NULL DEFAULT '',
    segments      TEXT    NOT NULL DEFAULT '[]',
    ts            INTEGER NOT NULL DEFAULT 0,
    is_self       INTEGER NOT NULL DEFAULT 0,
    reply_to      INTEGER,
    mentions_self INTEGER NOT NULL DEFAULT 0,
    has_image     INTEGER NOT NULL DEFAULT 0,
    media_hash    TEXT,
    reply_to_self INTEGER NOT NULL DEFAULT 0,
    UNIQUE (group_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_group_ts ON messages (group_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_user      ON messages (group_id, user_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_self      ON messages (group_id, is_self, ts);

CREATE TABLE IF NOT EXISTS chat_groups (
    group_id   INTEGER PRIMARY KEY,
    name       TEXT,
    enabled    INTEGER NOT NULL DEFAULT 0,
    first_seen INTEGER,
    last_seen  INTEGER
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id   INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    nickname   TEXT    NOT NULL DEFAULT '',
    card       TEXT,
    last_seen  INTEGER,
    msg_count  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (group_id, user_id)
);

-- 决定层每次判断都留痕，用于事后复盘调参。
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id     INTEGER NOT NULL,
    ts           INTEGER NOT NULL,
    should_reply INTEGER NOT NULL,
    priority     REAL    NOT NULL DEFAULT 0,
    reason       TEXT    NOT NULL DEFAULT '',
    features     TEXT    NOT NULL DEFAULT '{}',
    text         TEXT    NOT NULL DEFAULT '',
    draft        TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_decisions_group_ts ON decisions (group_id, ts);
CREATE INDEX IF NOT EXISTS idx_decisions_ts       ON decisions (ts);

-- 长期记忆。type 决定半衰期：culture/slang 近乎永久，event 只有几天。
CREATE TABLE IF NOT EXISTS memories (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scope          TEXT    NOT NULL,
    scope_id       INTEGER NOT NULL,
    content        TEXT    NOT NULL,
    strength       REAL    NOT NULL DEFAULT 1.0,
    created_ts     INTEGER NOT NULL,
    last_hit_ts    INTEGER,
    source_msg_ids TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (scope, scope_id);

CREATE TABLE IF NOT EXISTS budget (
    group_id     INTEGER PRIMARY KEY,
    window_start INTEGER NOT NULL DEFAULT 0,
    hour_used    INTEGER NOT NULL DEFAULT 0,
    day_used     INTEGER NOT NULL DEFAULT 0,
    day_start    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ======================================================================
-- 表情包系统
-- ======================================================================

-- 表情库。文件按内容哈希命名，天然去重。
CREATE TABLE IF NOT EXISTS stickers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hash          TEXT    NOT NULL UNIQUE,
    ext           TEXT    NOT NULL DEFAULT 'png',
    path          TEXT    NOT NULL DEFAULT '',
    size          INTEGER NOT NULL DEFAULT 0,
    width         INTEGER NOT NULL DEFAULT 0,
    height        INTEGER NOT NULL DEFAULT 0,
    is_animated   INTEGER NOT NULL DEFAULT 0,
    is_mark_face  INTEGER NOT NULL DEFAULT 0,
    -- collected=0 表示"见过但没收"（仍记 seen_count，用来判断是不是本群通用梗）
    -- collected=1 才是正式收藏、可以拿来发的
    collected     INTEGER NOT NULL DEFAULT 0,
    collect_reason TEXT   NOT NULL DEFAULT '',
    source_group  INTEGER,
    source_user   INTEGER,
    source_msg_id INTEGER,
    collected_ts  INTEGER NOT NULL DEFAULT 0,
    mood_tags     TEXT    NOT NULL DEFAULT '[]',
    content_tags  TEXT    NOT NULL DEFAULT '[]',
    desc          TEXT    NOT NULL DEFAULT '',
    qq_res_id     TEXT,
    seen_count    INTEGER NOT NULL DEFAULT 1,
    use_count     INTEGER NOT NULL DEFAULT 0,
    win_count     INTEGER NOT NULL DEFAULT 0,
    last_used_ts  INTEGER,
    blocked       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_stickers_use ON stickers (use_count DESC);
CREATE INDEX IF NOT EXISTS idx_stickers_src ON stickers (source_group, collected_ts);

-- 每次使用表情的留痕，用于冷却与效果归因
CREATE TABLE IF NOT EXISTS sticker_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sticker_id INTEGER NOT NULL,
    group_id   INTEGER NOT NULL,
    msg_id     INTEGER,
    ts         INTEGER NOT NULL,
    context    TEXT    NOT NULL DEFAULT '',
    reward     REAL
);
CREATE INDEX IF NOT EXISTS idx_sticker_usage ON sticker_usage (group_id, ts);

-- 观察到的使用语境：某人发了这个表情，前后说了什么。
-- 攒够样本后交给 LLM 归纳出 mood_tags —— 本质是"看别人怎么用就学会了"。
CREATE TABLE IF NOT EXISTS sticker_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sticker_id  INTEGER NOT NULL,
    group_id    INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    before_text TEXT    NOT NULL DEFAULT '',
    after_text  TEXT    NOT NULL DEFAULT '',
    ts          INTEGER NOT NULL,
    tagged      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sticker_obs ON sticker_observations (sticker_id, tagged);

-- ======================================================================
-- 多模态缓存
-- ======================================================================

CREATE TABLE IF NOT EXISTS media_cache (
    hash        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'image',
    path        TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    caption     TEXT NOT NULL DEFAULT '{}',
    ocr_text    TEXT NOT NULL DEFAULT '',
    ptt_text    TEXT NOT NULL DEFAULT '',
    sensitive   INTEGER NOT NULL DEFAULT 0,
    hits        INTEGER NOT NULL DEFAULT 0,
    created_ts  INTEGER NOT NULL DEFAULT 0,
    last_hit_ts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_media_kind ON media_cache (kind, last_hit_ts);

-- 云端 OCR 是任务制的（提交 -> 轮询 -> 取结果），整个流程可能跑几分钟。
-- 落库是为了重启后还能接着收结果，而不是把在途任务丢掉。
CREATE TABLE IF NOT EXISTS ocr_jobs (
    job_id       TEXT PRIMARY KEY,
    media_hash   TEXT    NOT NULL,
    backend      TEXT    NOT NULL DEFAULT 'paddleocr',
    state        TEXT    NOT NULL DEFAULT 'pending',
    submitted_ts INTEGER NOT NULL DEFAULT 0,
    finished_ts  INTEGER,
    error        TEXT    NOT NULL DEFAULT '',
    chars        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_state ON ocr_jobs (state, submitted_ts);

-- 逐群人设。**没有记录 = 跟随全局默认**（persona.yaml）。
-- 存整份 YAML 而不是差异：人设是结构化的，稀疏覆盖难编辑也难解释；
-- 整份副本语义明确（改了就是这份，全局变动不影响它），面板上给个「恢复默认」即可。
CREATE TABLE IF NOT EXISTS group_personas (
    group_id   INTEGER PRIMARY KEY,
    yaml       TEXT    NOT NULL,
    updated_ts INTEGER NOT NULL DEFAULT 0
);

-- 提示词覆盖层。**没有记录 = 用代码里的默认**。
-- 代码始终是事实来源，这里只存改动过的条目，重置就是删记录。
-- group_id = 0 表示全局；>0 表示只对该群生效。
-- 解析顺序：群覆盖 > 全局覆盖 > 代码默认。
-- 需要按群是因为有些提示词与具体人设强相关 —— 比如"沉默是常态"适合潜水的
-- 群友，但完全不适合一个讲解题目的角色。
CREATE TABLE IF NOT EXISTS prompt_overrides (
    key        TEXT    NOT NULL,
    group_id   INTEGER NOT NULL DEFAULT 0,
    text       TEXT    NOT NULL,
    updated_ts INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (key, group_id)
);

-- ======================================================================
-- 群档案与自进化
-- ======================================================================

-- 统计画像：纯统计，不用 LLM，零成本，是自进化的地基
CREATE TABLE IF NOT EXISTS group_profiles (
    group_id      INTEGER PRIMARY KEY,
    topics        TEXT    NOT NULL DEFAULT '{}',
    avg_msg_len   REAL    NOT NULL DEFAULT 0,
    emoji_rate    REAL    NOT NULL DEFAULT 0,
    sticker_rate  REAL    NOT NULL DEFAULT 0,
    question_rate REAL    NOT NULL DEFAULT 0,
    hourly        TEXT    NOT NULL DEFAULT '[]',
    slang         TEXT    NOT NULL DEFAULT '{}',
    core_members  TEXT    NOT NULL DEFAULT '[]',
    atmosphere    REAL    NOT NULL DEFAULT 0,
    sample_size   INTEGER NOT NULL DEFAULT 0,
    updated_ts    INTEGER NOT NULL DEFAULT 0
);

-- 群策略：自进化的产物。引用/@ 用 Beta 后验，Thompson 采样选动作。
CREATE TABLE IF NOT EXISTS group_policies (
    group_id        INTEGER PRIMARY KEY,
    threshold_delta REAL    NOT NULL DEFAULT 0,
    initiative      REAL    NOT NULL DEFAULT 0.30,
    sticker_rate    REAL    NOT NULL DEFAULT 0.12,
    emoji_rate      REAL    NOT NULL DEFAULT 0.15,
    quote_alpha     REAL    NOT NULL DEFAULT 2.0,
    quote_beta      REAL    NOT NULL DEFAULT 2.0,
    at_alpha        REAL    NOT NULL DEFAULT 1.0,
    at_beta         REAL    NOT NULL DEFAULT 3.0,
    at_all_enabled  INTEGER NOT NULL DEFAULT 0,
    tone            TEXT    NOT NULL DEFAULT '{}',
    samples         INTEGER NOT NULL DEFAULT 0,
    updated_ts      INTEGER NOT NULL DEFAULT 0
);

-- 每次发言的奖励信号留痕
CREATE TABLE IF NOT EXISTS feedback_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id   INTEGER NOT NULL,
    msg_ids    TEXT    NOT NULL DEFAULT '[]',
    sent_ts    INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    weight     REAL    NOT NULL DEFAULT 0,
    detail     TEXT    NOT NULL DEFAULT '',
    attributes TEXT    NOT NULL DEFAULT '{}',
    harvested  INTEGER NOT NULL DEFAULT 0,
    ts         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_feedback_group ON feedback_events (group_id, sent_ts);
CREATE INDEX IF NOT EXISTS idx_feedback_open  ON feedback_events (harvested, sent_ts);

-- 被怀疑是机器人的记录。这是**惩罚信号**的来源。
-- 只记"指向自己"的怀疑（@ 它 / 回复它 / 它刚说完话时有人提这茬），
-- 不然聊到别的 AI 产品会误伤。
CREATE TABLE IF NOT EXISTS suspicion_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id   INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id    INTEGER NOT NULL DEFAULT 0,
    nickname   TEXT    NOT NULL DEFAULT '',
    text       TEXT    NOT NULL DEFAULT '',
    pattern    TEXT    NOT NULL DEFAULT '',
    strength   REAL    NOT NULL DEFAULT 0,
    ts         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_suspicion_group ON suspicion_events (group_id, ts);

-- 按天结算的"伪装存活"记录。
-- reward 正数是装得像，负数是露馅或该说话没说。
CREATE TABLE IF NOT EXISTS survival_days (
    group_id        INTEGER NOT NULL,
    day_key         TEXT    NOT NULL,
    spoke_count     INTEGER NOT NULL DEFAULT 0,
    group_msg_count INTEGER NOT NULL DEFAULT 0,
    suspicion_count INTEGER NOT NULL DEFAULT 0,
    verdict         TEXT    NOT NULL DEFAULT '',
    reward          REAL    NOT NULL DEFAULT 0,
    streak          INTEGER NOT NULL DEFAULT 0,
    settled_ts      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (group_id, day_key)
);
CREATE INDEX IF NOT EXISTS idx_survival_group ON survival_days (group_id, day_key);

-- 新入群（或刚打开发言权限）时的冷启动分析状态。
-- 流程：拉全部历史 -> 分析群偏好 -> 生成"看不懂就问"的问题 -> 慢慢在群里问掉。
CREATE TABLE IF NOT EXISTS onboard_state (
    group_id    INTEGER PRIMARY KEY,
    status      TEXT    NOT NULL DEFAULT '',
    pulled      INTEGER NOT NULL DEFAULT 0,
    analyzed_ts INTEGER NOT NULL DEFAULT 0,
    questions   TEXT    NOT NULL DEFAULT '[]',
    asked       TEXT    NOT NULL DEFAULT '[]',
    note        TEXT    NOT NULL DEFAULT '',
    updated_ts  INTEGER NOT NULL DEFAULT 0
);
"""

#: 依赖迁移新增列的索引。
#: 必须**在迁移之后**执行 —— 老库的 messages 表建表时还没有 media_hash，
#: 如果把它写进 SCHEMA，executescript 会在 CREATE INDEX 处直接报
#: "no such column"，连迁移的机会都没有。
POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_messages_media     ON messages (media_hash);
CREATE INDEX IF NOT EXISTS idx_messages_replyself ON messages (group_id, reply_to_self, ts);
"""

#: 增量迁移：表名 -> [(列名, 列定义)]
MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "decisions": [
        ("features", "TEXT NOT NULL DEFAULT '{}'"),
        ("text", "TEXT NOT NULL DEFAULT ''"),
        ("draft", "TEXT NOT NULL DEFAULT ''"),
    ],
    "messages": [
        ("has_image", "INTEGER NOT NULL DEFAULT 0"),
        # 图片/语音的内容哈希，用于去重与缓存命中
        ("media_hash", "TEXT"),
        # 这条消息的 reply 段是否指向机器人自己的消息
        ("reply_to_self", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "memories": [
        # fact | relation | culture | slang | event | preference | topic | emotion | self
        ("type", "TEXT NOT NULL DEFAULT 'fact'"),
        ("ttl_days", "INTEGER NOT NULL DEFAULT 180"),
        # 别人明确说的=0.9，自己推断的=0.5；召回时按此降权
        ("confidence", "REAL NOT NULL DEFAULT 1.0"),
        ("source", "TEXT NOT NULL DEFAULT 'observed'"),
        # 被新记忆推翻时打时间戳，而不是删除 —— 保留演化历史
        ("invalid_at", "INTEGER"),
        ("supersedes_id", "INTEGER"),
        ("group_id", "INTEGER"),
    ],
    "group_members": [
        # 和机器人的亲疏：聊得多上升、被怼下降
        ("affinity", "REAL NOT NULL DEFAULT 0"),
        ("mention_count", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "stickers": [
        # 0 = 见过但未收藏（仍记 seen_count），1 = 正式收藏可用
        ("collected", "INTEGER NOT NULL DEFAULT 0"),
        ("collect_reason", "TEXT NOT NULL DEFAULT ''"),
    ],
}


async def _migrate_prompt_overrides(conn: aiosqlite.Connection) -> bool:
    """把 prompt_overrides 的主键从 (key) 改成 (key, group_id)。

    SQLite 改不了主键，只能重建表。这张表很小（最多几十行），
    而且能完整迁过来，所以重建是安全的。

    为什么要按群：有些提示词和具体人设强相关 —— 比如「沉默是常态」
    适合潜水的群友，但完全不适合一个讲解题目的角色。
    """
    async with conn.execute("PRAGMA table_info(prompt_overrides)") as cur:
        cols = {row[1] for row in await cur.fetchall()}
    if not cols or "group_id" in cols:
        return False        # 表不存在（由 SCHEMA 建）或已经是新结构

    await conn.executescript("""
        ALTER TABLE prompt_overrides RENAME TO prompt_overrides_legacy;
        CREATE TABLE prompt_overrides (
            key        TEXT    NOT NULL,
            group_id   INTEGER NOT NULL DEFAULT 0,
            text       TEXT    NOT NULL,
            updated_ts INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (key, group_id)
        );
        INSERT INTO prompt_overrides (key, group_id, text, updated_ts)
            SELECT key, 0, text, updated_ts FROM prompt_overrides_legacy;
        DROP TABLE prompt_overrides_legacy;
    """)
    log.info("prompt_overrides 已迁移：新增 group_id 维度（老数据归为全局）")
    return True


async def _ensure_column(conn: aiosqlite.Connection, table: str, column: str, ddl: str) -> bool:
    async with conn.execute(f"PRAGMA table_info({table})") as cur:
        cols = {row[1] for row in await cur.fetchall()}
    if not cols:
        return False  # 表还不存在，由 SCHEMA 负责
    if column in cols:
        return False
    await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    log.info("迁移: %s 增加列 %s", table, column)
    return True


class Database:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    @classmethod
    async def open(cls, path: Path | str) -> "Database":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path)
        conn.row_factory = aiosqlite.Row
        await conn.executescript(SCHEMA)

        migrated = False
        for table, columns in MIGRATIONS.items():
            for column, ddl in columns:
                migrated |= await _ensure_column(conn, table, column, ddl)
        migrated |= await _migrate_prompt_overrides(conn)
        if migrated:
            await conn.commit()

        # 依赖新列的索引，必须等列都加好
        await conn.executescript(POST_MIGRATION_INDEXES)

        await conn.commit()
        log.info("数据库就绪: %s", path)
        return cls(conn)

    async def close(self) -> None:
        await self.conn.close()
