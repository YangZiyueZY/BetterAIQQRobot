"""配置加载。

优先级（后者覆盖前者）：
    config.toml  <  data/settings.json  <  .env（仅敏感项）

data/settings.json 是管理面板写入的运行时覆盖层，让它不必去改写用户手写的
config.toml，也省掉一个 TOML 写入库的依赖。

本模块同时负责把面板传来的局部更新**热应用**到正在运行的配置对象上 ——
所有下游模块（Decider / Scheduler / LLMClient…）持有的是这些 dataclass 的
引用，改字段即刻生效，不需要重启进程。
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

VERSION = "1.1.0"


def load_dotenv(path: Path) -> None:
    """极简 .env 解析。已存在的环境变量优先，不覆盖。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# --------------------------------------------------------------------------
# 配置分节
# --------------------------------------------------------------------------


@dataclass(slots=True)
class OneBotConfig:
    ws_url: str = "ws://127.0.0.1:3001"
    token: str = ""
    request_timeout: float = 15.0
    heartbeat_timeout: float = 90.0
    reconnect_min: float = 1.0
    reconnect_max: float = 60.0


@dataclass(slots=True)
class StorageConfig:
    db_path: Path = ROOT / "data" / "qqbot.db"
    retention_days: int = 90


@dataclass(slots=True)
class WebConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8090
    token: str = ""


@dataclass(slots=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key: str = ""
    """走 .env 的 LLM_API_KEY，不存 settings.json。"""
    temperature: float = 0.9
    """口语化需要高温度。"""
    max_tokens: int = 800
    """注意：推理模型会先烧 reasoning token。400 对它们偏紧，容易 content 为空。"""
    timeout: float = 60.0
    max_concurrency: int = 2
    keep_thinking: bool = True
    """保留思维链。**别关** —— 关掉之后推理模型会把预算全烧在内部思考上，
    content 反而返回空（实测 deepseek-flash 就是这样）。"""
    reasoning_effort: str = "high"
    """思维链强度：low / medium / high。有些代理会静默忽略这个参数。"""
    reply_max_tokens: int = 1200
    reply_timeout: float = 30.0
    analysis_max_tokens: int = 4000
    analysis_timeout: float = 150.0
    reply_model: str = ""
    """群聊回复专用模型。留空则用 model。

    实测重推理模型（如 step-5-preview）回一句"在吗"要 10~30 秒，
    还会把 1200 token 预算全烧在思考上、content 返回空 —— 群聊里不可用。
    所以回复档要挑快的。
    """
    analysis_model: str = ""
    """分析专用模型（记忆抽取 / 群复盘 / 表情打标）。留空则用 model。

    这些场景不在回复链路上，慢几秒无所谓，但需要更强的推理。"""

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


@dataclass(slots=True)
class BehaviorConfig:
    # 决定层
    threshold: float = 0.55
    give_up_prob: float = 0.30
    mention_only: bool = False
    group_cooldown: float = 180.0
    name_priority: float = 0.60
    topic_priority: float = 0.50
    idle_priority: float = 0.20
    # 蓄水池
    debounce: float = 5.0
    max_wait: float = 30.0
    max_batch: int = 15
    # 作息
    active_from: str = "08:30"
    """单窗口作息的下界。**下面的窗口字段为空时才用它。**"""
    active_to: str = "23:30"
    """单窗口作息的上界。"""
    jitter_minutes: int = 45
    away_prob_per_hour: float = 0.35
    group_hour_budget: int = 8
    group_day_budget: int = 40
    global_day_budget: int = 200
    # --- 作息规律（工作日 / 周末 / 节假日分开配）---
    #
    # 真人每天的上线时间是有规律的：一个高中生工作日只可能中午和晚上出现，
    # 周末和节假日时间充裕。所以按日期类型给不同的时段表。
    #
    # 格式："11:30-13:30,17:30-23:30"，逗号分隔多个窗口，支持跨午夜（22:00-02:00）。
    # **留空 = 回落到 active_from~active_to 的单窗口**（老配置行为不变）。
    weekday_windows: str = ""
    """工作日窗口。高中生的典型值："11:30-13:30,17:30-23:30"。"""
    weekend_windows: str = ""
    """周末窗口。典型值："09:00-23:30"。"""
    holiday_windows: str = ""
    """节假日窗口。留空则用 weekend_windows（或单窗口回落的那个）。"""
    holidays: list[str] = field(default_factory=list)
    """节假日日期，写 "MM-DD" 或 "YYYY-MM-DD" 都认。"""
    off_schedule_prob: float = 0.06
    """工作日作息被打乱的概率（生病请假之类）。

    真人也会有意外 —— 某天不舒服就白天也在线。一天最多掷一次骰子，
    所以同一天内表现稳定，不会忽上忽下。
    """
    late_night_prob: float = 0.10
    """当晚熬夜（把最后一个窗口延到凌晨）的概率。"""
    # --- 新群冷启动 ---
    #
    # 刚打开发言权限时先潜水读一遍群历史，搞清楚这是什么群、大家在聊什么，
    # 把看不懂的记下来，之后慢慢问。真人进新群就是这么融入的 ——
    # 比上来就硬接话自然得多。
    onboard_enabled: bool = True
    onboard_history: int = 1000
    """尽量往前拉多少条群历史。NapCat 的 id 是内存 LRU（5000 条），有硬上限。"""
    onboard_ask_gap_minutes: int = 90
    """两次提问之间至少隔多久。别把新人问成了调查问卷。"""
    onboard_ask_min_group_msgs: int = 3
    """提问前这段时间群里至少要有这么多条消息 —— 冷场时提问很突兀。"""
    # 生成长度
    max_chars_per_bubble: int = 40
    max_bubbles: int = 3
    # --- 打字节奏（拟人的关键：真人不可能秒发）---
    #
    # 原来用"字/秒"，但 8~18 字/秒比人在键盘上还快，手机上打中文实际只有 2~4 字/秒。
    # 改成"每个字花多少毫秒"，更好对着真实手感调，也更有上下界的直觉。
    typing_ms_per_char_min: float = 180.0
    """打一个字最快花多少毫秒。180 ≈ 5.5 字/秒，手快的人。"""
    typing_ms_per_char_max: float = 420.0
    """打一个字最慢花多少毫秒。420 ≈ 2.4 字/秒，边想边打。"""
    typing_min_delay: float = 1.0
    """再短的消息也至少等这么久（秒）。打「在」两个字也要拿起手机、点开、打字。"""
    typing_max_delay: float = 12.0
    """单条消息的打字时间上限（秒）。超过就不像打字慢，像掉线了。"""
    reaction_delay_min: float = 0.6
    """看到消息到开始打字的反应时间下界（秒）。"""
    reaction_delay_max: float = 2.0
    """反应时间上界（秒）。0.4 秒就回一句「在」太快了，像自动应答。"""
    face_keep_prob: float = 0.35
    """模型给出 QQ 表情时，实际保留的概率。

    模型倾向于**每一轮都配一个表情**（人设例句里带一个它就会照抄），
    而真人不会每条都配 —— 表情刷屏比文字刷屏更烦人。
    光靠 prompt 里写"十条里最多一条"压不住（实测模型会无视），
    所以在后处理加一道概率闸门兜底。
    """
    typing_first_bubble: bool = True
    """第一条是否也计入打字时间。

    曾经是 False（只有第二条起才延迟），结果**第一条只有 0.4 秒反应时间就发出去了** ——
    真人不可能"看到消息 → 0.4 秒 → 消息已经在群里"。所以默认 True。
    """
    context_messages: int = 30
    # 记忆
    memory_enabled: bool = True
    memory_recall_top: int = 5


@dataclass(slots=True)
class StickerConfig:
    enabled: bool = True
    # --- 收藏 ---
    auto_collect: bool = True
    """是否自动收藏群友发的表情包（只存在本地库）。"""
    write_to_qq: bool = False
    """是否调用 add_custom_face 把它真的写进 QQ 收藏夹。**会改动账号状态，默认关。**"""
    collect_prob: float = 0.60
    """满足条件也不是必收 —— 真人不会每个都存。"""
    collect_min_seen: int = 2
    """至少在本群被看到这么多次，才算"通用梗"。"""
    collect_min_users: int = 2
    """或者至少这么多不同的人用过。"""
    collect_daily_limit: int = 3
    per_user_hourly: int = 1
    capacity: int = 300
    max_size_kb: int = 2048
    # --- 使用 ---
    send_rate: float = 0.12
    """发言中带自定义表情包的基础比例（还会被群策略调制）。"""
    group_cooldown_days: int = 3
    recent_window: int = 20
    """本群最近这么多条里出现过就不重复发。"""
    explore_rate: float = 0.15
    """ε-greedy 的探索率。"""
    base_dir: str = "data/stickers"
    # --- 标注 ---
    tag_min_observations: int = 3
    tag_batch: int = 5


@dataclass(slots=True)
class MediaConfig:
    enabled: bool = True
    base_dir: str = "data/media"
    # --- 看图（花钱的部分）---
    #
    # **这是理解图片的主力**：收到图就丢给多模态模型看。
    # OCR 只能读字，而日常图里大量信息藏在画面中（表情、场景、动作、氛围），
    # 还有相当一部分图压根没有文字 —— 光靠 OCR 等于没看图。
    vision_enabled: bool = True
    vision_hourly_budget: int = 10
    """单群每小时最多描述几张图。**看图是主力，这里就是唯一的成本闸门**。"""
    vision_daily_budget: int = 200
    vision_wait: float = 15.0
    """生成回复前最多等看图多久（秒）。

    实测看图 5~19 秒（取决图片大小和负载，小图 5~7 秒，大图接近 20 秒）。
    用户发的图多半就是要问的，所以值得等；但要有上限，别把回复拖死。
    """
    vision_max_side: int = 768
    """送进模型前把图压到这个边长，直接决定成本。"""
    # --- 非文字消息的解析（语音/视频/卡片/合并转发/文件）---
    #
    # 群友的表达方式不只是文字。只认得文字和图片，等于漏掉一大半上下文。
    # 每种都只做"能把内容变成给 LLM 看的备注"这一件事，失败就退化成占位符，
    # 绝不能让某条消息因为解析失败而整个丢掉。
    voice_enabled: bool = True
    """语音。先用 NapCat 的 fetch_ptt_text（免费快但实测失败率高），
    失败再下载音频走 ASR。"""
    video_enabled: bool = True
    """视频。注意 video.url 大概率是 NapCat 本机路径而不是 http，要判 scheme。"""
    card_enabled: bool = True
    """QQ 卡片（分享链接/位置/小程序）。内容在 data.data 那个 JSON 字符串里。"""
    forward_enabled: bool = True
    """合并转发。推送里只有 id，内容得调 get_forward_msg。"""
    file_enabled: bool = True
    """文件。只记文件名和大小，不下载。"""
    ingest_timeout: float = 25.0
    """单个非文字段解析的耗时上限（秒）。超时降级成占位符。"""
    forward_max_nodes: int = 30
    """合并转发最多展开多少条，防止超长刷屏。"""

    # --- OCR：**辅助**，补充"图上的文字"这一项 ---
    ocr_enabled: bool = True
    ocr_backend: str = "napcat"
    """OCR 后端（辅助手段，不承担理解图片的主要职责）：
        napcat     —— NapCat 自带的 ocr_image，同步、免费、秒回，只有纯文字
        paddleocr  —— 云端 PaddleOCR-VL，输出 markdown（公式/表格），**很慢**
        both       —— NapCat 先把消息回掉，PaddleOCR 后台慢慢跑并缓存结果
    """
    ptt_enabled: bool = True
    """语音转文字，用 NapCat 的 fetch_ptt_text，不花钱。"""
    retention_days: int = 60


@dataclass(slots=True)
class EvolutionConfig:
    enabled: bool = False
    """自进化总开关。建议先跑一周统计画像，确认数据合理再打开。"""
    profile_enabled: bool = True
    """群画像（纯统计，无风险）单独可控。"""
    feedback_enabled: bool = True
    """奖励信号采集。"""
    harvest_delay: float = 300.0
    """发言后多久去收获信号（秒）。QQ 的表情回应有保留期，别太晚。"""
    profile_every: int = 200
    """每积累这么多条群消息，重算一次画像。"""
    reflect_every: int = 500
    """每这么多条消息跑一次 LLM 复盘。"""
    memory_capacity_per_group: int = 500
    # --- 伪装存活（奖励机制）---
    #
    # 目标是"不被认出来是机器人"。所以给一个贯穿始终的奖励信号：
    # 一天没被质疑算活过一天，连续天数越多奖励越高；被质疑就清零扣分。
    #
    # **但只按"没被质疑"给奖励会出事**：最优策略会变成永远不说话。
    # 所以配了防捷径的硬下限 —— 群里有正常聊天量而它几乎不发言，
    # 判定为**失败日**（扣分且不算存活），让"少说话"从安全策略变成亏损。
    survival_enabled: bool = True
    """存活结算总开关。它独立于自进化 —— 结算本身不改任何参数，只记账。"""
    min_msgs_per_active_day: int = 3
    """群里有正常聊天量时，它一天至少要说这么多句才不算失职。"""
    min_group_msgs_for_active_day: int = 20
    """群消息少于这个数的日子算"群里没人说话"，不追究它。"""
    survival_reward: float = 1.0
    """活过一天的基础奖励。"""
    streak_reward_cap: float = 3.0
    """连胜奖励的倍数上限。活 30 天不该比活 3 天多 10 倍。"""
    suspicion_penalty: float = 3.0
    """每次被质疑扣多少分。"""
    silent_penalty: float = 1.5
    """该说话却没说（潜水）扣多少分 —— 防"少说少错"的捷径。"""
    survival_lookback_days: int = 7
    """补算最近几天的结算。停机几天后不该留空洞。"""


@dataclass(slots=True)
class DeciderConfig:
    """决策模型。

    目标是"更像真人地决定要不要接话"。设计成**两阶段**：

      1. 廉价门控（原有打分器）：免费、瞬时，负责一票否决
         （睡着/外出/额度/冷却/仅@模式）并给出先验分数
      2. LLM 判定：只对**分数落在灰区**的候选调用，问它"这句该不该接"

    为什么不全交给 LLM：每条群消息都调一次又贵又慢，而灰区之外的情况
    打分器已经很有把握（分数极低=不值得说；被@=必须理）。LLM 的价值
    恰恰在灰区 —— 语义接不接得上、群里是不是已经有人接了。
    """

    mode: str = "hybrid"
    """score（只用打分器）/ hybrid（灰区问 LLM）/ llm（全交给 LLM）。"""
    band_low: float = 0.25
    """低于这个分数不问 LLM —— 明显不值得说。"""
    band_high: float = 0.75
    """高于这个分数不问 LLM —— 明显该说。"""
    timeout: float = 8.0
    """LLM 判定超时就回退到打分器。它在回复链路上，不能拖。"""
    force_on_mention: bool = True
    """被 @ / 被回复时强制发言，不允许 LLM 说不。

    这是硬性要求之一：真人被点名一定会理。没有这条，决策模型会慢慢
    收敛到"什么都不说最安全"。
    """
    silence_nudge_minutes: int = 180
    """在群里沉默超过这么久，就压低门槛催它开口。防"越不说话越安全"。"""
    explore_prob: float = 0.15
    """LLM 说不说、但打分器分数较高时的探索概率。

    保留一点随机性 —— 它也是防止决策模型退化成一味沉默的保险。
    """


@dataclass(slots=True)
class DebugConfig:
    """测试模式：把"时间/额度/冷却"这些现实约束全部拿掉，方便随时验证。

    刻意**不绕过 safety 层** —— 就算在测试，也不该让它说出格的话。
    也不绕过群开关 —— 那是逐群的明确意图，面板上点一下就行。
    """

    enabled: bool = False
    """总开关。打开后：作息失效（永远在线）、额度不生效、群冷却归零、
    不随机放弃、发言门槛大幅降低。"""
    relax_threshold: float = 0.15
    """测试模式下的发言门槛。0.15 意味着"有人提问"或"群在刷屏"就会开口。"""
    respond_to_all: bool = False
    """连纯闲聊也回。评分本身为 0 的消息（既不相关也没人叫它）也会被强行放行，
    用来做压力测试。**会刷屏，只在你盯着的时候开。**"""


@dataclass(slots=True)
class PaddleOCRConfig:
    """PaddleOCR-VL 云端 OCR。

    和 NapCat 自带的 ocr_image 完全是两类东西：
      · ocr_image   同步、免费、秒回，只给纯文字（无版面/公式/表格）
      · PaddleOCR   异步任务制、走云端、**很慢**，但输出 markdown，
                    公式变 LaTeX、表格保留结构

    实测同一个接口快的时候 6 秒、慢的时候 94 秒 —— 所以**绝不能阻塞群聊回复**。
    默认只作为"后台补充"：先用 NapCat 的快速 OCR 把消息回掉，
    PaddleOCR 的结果慢慢跑，写进缓存供后续使用。
    """

    enabled: bool = False
    base_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    model: str = "PaddleOCR-VL-1.6"
    token: str = ""
    """走 .env 的 PADDLEOCR_TOKEN，不存 settings.json。"""
    poll_interval: float = 5.0
    submit_timeout: float = 60.0
    job_timeout: float = 300.0
    """单个任务最长等多久。实测 6~94 秒都有，给宽一点。"""
    doc_orientation: bool = False
    """版面方向分类（歪着的照片有用，普通截图没必要）。"""
    doc_unwarping: bool = False
    """文档去弯曲（拍纸质文档才有用）。"""
    chart_recognition: bool = False
    """图表识别。"""
    max_chars: int = 3000
    """markdown 太长就截断，别把上下文吃光。"""
    min_side: int = 200
    """太小的图（表情包之类）不值得送云端。"""

    @property
    def configured(self) -> bool:
        return bool(self.token and self.base_url and self.model)


@dataclass(slots=True)
class SearchConfig:
    """网络搜索。给模型补上"知识截止之后"的信息。

    关键不是"能搜"，而是**什么时候该搜** —— 群里每句话都去搜既慢又蠢。
    所以有独立的时效性判断器（search/need.py），只有命中才发请求。
    """

    enabled: bool = False
    provider: str = "tavily"
    """tavily / bocha（博查）/ serper / searxng / generic"""
    base_url: str = ""
    """留空则用该 provider 的默认地址。"""
    api_key: str = ""
    """走 .env 的 SEARCH_API_KEY。"""
    max_results: int = 3
    timeout: float = 12.0
    """搜索必须快 —— 它是挂在回复链路上的，慢了这个回复就不像人了。"""
    need_threshold: float = 0.5
    """时效性判定阈值。越高越少搜。"""
    hourly_budget: int = 10
    daily_budget: int = 80
    max_snippet_chars: int = 300
    """每条结果的摘要截断长度。"""
    search_when_mention: bool = True
    """被 @ 且有信息需求时优先搜（点名问了就别含糊）。"""
    country: str = "cn"
    language: str = "zh"

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.api_key)


@dataclass(slots=True)
class BilibiliConfig:
    """B 站视频理解 + 搜索。

    核心事实（实测）：**AI 字幕必须登录态**。游客态返回空列表 +
    `need_login_subtitle: true`，连 buvid3/buvid4 都没用，不存在免登录取巧路径。
    所以 SESSDATA 是必需的（走 .env 的 BILIBILI_COOKIE）。
    """

    enabled: bool = True
    cookie: str = ""
    """完整 Cookie 串，走 .env 的 BILIBILI_COOKIE（含 SESSDATA，是凭据）。"""
    buvid3: str = ""
    """游客指纹的一半。没有登录 Cookie 时，搜索至少需要它，否则 412 风控。"""
    buvid4: str = ""
    """游客指纹的另一半 —— 文档实测两个一起用才稳。"""
    enabled_link: bool = True
    """自动解析群里发的 B 站链接。"""
    enabled_search: bool = True
    """允许机器人主动搜 B 站。"""
    max_links_per_message: int = 1
    """一条消息里最多解析几个链接 —— 防刷屏。"""
    max_chars: int = 3000
    """字幕正文进 prompt 的截断长度。"""
    prefer_lan: str = "zh"
    dedupe_repeat: bool = True
    timeout: float = 20.0

    # --- 音频 ASR 兜底（无字幕时）---
    asr_fallback: bool = True
    max_asr_seconds: int = 600
    """超过这么长的视频不做音频转写（太慢太贵）。"""

    # --- 搜索 ---
    search_results: int = 5
    search_order: str = "totalrank"

    @property
    def configured(self) -> bool:
        return bool(self.cookie or self.buvid3)


@dataclass(slots=True)
class AsrConfig:
    """音频转文字。

    两条同步通道，**能用的范围不一样**（这里踩过坑）：

      upload  `{root}/audio/transcriptions`  multipart 上传，一次返回全文
      sse     `{root}/audio/asr/sse`         JSON + base64，SSE 返回全文

    `transport = "auto"`（默认）按 base_url 选：指向 **step_plan 时必走 sse** ——
    那个前缀下 multipart 那条路由网关直接 404，而 sse 是通的；
    开放平台 `/v1` 两条都有，优先 upload（大文件友好、一次拿全文）。

    异步的 `asr/file/submit` 需要 URL 公网可达，只在你有公网地址时用。
    注意：StepFun 只支持中英文，不支持中文方言。
    """

    enabled: bool = False
    api_key: str = ""
    """走 .env 的 STEPFUN_API_KEY。"""
    base_url: str = "https://api.stepfun.com/v1"
    model: str = "stepaudio-2.5-asr"
    transport: str = "auto"
    """auto = 按 base_url 自动挑；也可强制 sse / upload。"""
    language: str = "zh"
    """SSE 通道的识别语言。"""
    timeout: float = 300.0
    """音频转写比文本慢得多，超时给宽一点。"""
    poll_interval: float = 2.0
    job_timeout: float = 600.0
    """异步模式的总超时。"""
    hotwords: list[str] = field(default_factory=list)
    """热词，能显著改善专有名词识别。"""

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(slots=True)
class Config:
    onebot: OneBotConfig = field(default_factory=OneBotConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    web: WebConfig = field(default_factory=WebConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    sticker: StickerConfig = field(default_factory=StickerConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    decider: DeciderConfig = field(default_factory=DeciderConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    paddleocr: PaddleOCRConfig = field(default_factory=PaddleOCRConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    bilibili: BilibiliConfig = field(default_factory=BilibiliConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    log_level: str = "INFO"
    groups: list[int] = field(default_factory=list)
    persona_path: Path = ROOT / "persona.yaml"
    root: Path = ROOT

    @property
    def settings_path(self) -> Path:
        return self.root / "data" / "settings.json"


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------

#: 面板可写的节 -> dataclass。用于校验局部更新。
SECTIONS: dict[str, type] = {
    "llm": LLMConfig,
    "behavior": BehaviorConfig,
    "web": WebConfig,
    "sticker": StickerConfig,
    "media": MediaConfig,
    "evolution": EvolutionConfig,
    "decider": DeciderConfig,
    "debug": DebugConfig,
    "paddleocr": PaddleOCRConfig,
    "search": SearchConfig,
    "bilibili": BilibiliConfig,
    "asr": AsrConfig,
}


def _coerce(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """按 dataclass 字段类型过滤 + 转换，丢弃未知键。"""
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        try:
            if f.type in ("float", float):
                value = float(value)
            elif f.type in ("int", int):
                value = int(value)
            elif f.type in ("bool", bool):
                if isinstance(value, str):
                    value = value.strip().lower() not in ("false", "0", "no", "", "off")
                else:
                    value = bool(value)
            elif f.type in ("str", str):
                value = str(value)
        except (TypeError, ValueError):
            continue
        out[f.name] = value
    return out


def _apply_section(target: Any, data: dict[str, Any]) -> list[str]:
    """把 data 写进 dataclass 实例，返回实际改动的字段名。"""
    changed = []
    for key, value in _coerce(type(target), data).items():
        if getattr(target, key) != value:
            setattr(target, key, value)
            changed.append(key)
    return changed


def _write_env_secret(path: Path, key: str, value: str) -> None:
    """就地更新 .env 里的一行，保留注释和其它键。

    密钥必须有落盘的地方 —— 只放内存的话，用户以为配好了，
    一重启就静默丢失，然后看到"未配置"一头雾水。
    .env 本来就是放密钥的地方，而且已经在 .gitignore 里、权限 600。
    """
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()

    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def _read_settings_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config(root: Path = ROOT, *, apply_overlay: bool = True) -> Config:
    load_dotenv(root / ".env")

    data: dict = {}
    toml_path = root / "config.toml"
    if toml_path.is_file():
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)

    ob = data.get("onebot", {})
    st = data.get("storage", {})
    web_raw = dict(data.get("web", {}))
    llm_raw = dict(data.get("llm", {}))
    beh_raw = dict(data.get("behavior", {}))
    stk_raw = dict(data.get("sticker", {}))
    media_raw = dict(data.get("media", {}))
    evo_raw = dict(data.get("evolution", {}))
    dbg_raw = dict(data.get("debug", {}))
    ocr_raw = dict(data.get("paddleocr", {}))
    search_raw = dict(data.get("search", {}))
    bili_raw = dict(data.get("bilibili", {}))
    asr_raw = dict(data.get("asr", {}))
    bot = data.get("bot", {})

    overlay = _read_settings_file(root / "data" / "settings.json") if apply_overlay else {}
    for name, bucket in (
        ("llm", llm_raw), ("behavior", beh_raw), ("web", web_raw),
        ("sticker", stk_raw), ("media", media_raw), ("evolution", evo_raw),
        ("debug", dbg_raw), ("paddleocr", ocr_raw), ("search", search_raw), ("bilibili", bili_raw), ("asr", asr_raw),
    ):
        sub = overlay.get(name)
        if isinstance(sub, dict):
            bucket.update(sub)

    db_path = Path(st.get("db_path", "data/qqbot.db"))
    if not db_path.is_absolute():
        db_path = root / db_path
    # 环境变量可覆盖，方便跑演示/预发实例而不污染正式库
    if os.environ.get("QQBOT_DB_PATH"):
        db_path = Path(os.environ["QQBOT_DB_PATH"]).expanduser()

    persona = Path(bot.get("persona_path", "persona.yaml"))
    if not persona.is_absolute():
        persona = root / persona

    cfg = Config(
        onebot=OneBotConfig(
            ws_url=os.environ.get("ONEBOT_WS_URL") or ob.get("ws_url", "ws://127.0.0.1:3001"),
            token=os.environ.get("ONEBOT_TOKEN", ""),
            request_timeout=float(ob.get("request_timeout", 15.0)),
            heartbeat_timeout=float(ob.get("heartbeat_timeout", 90.0)),
            reconnect_min=float(ob.get("reconnect_min", 1.0)),
            reconnect_max=float(ob.get("reconnect_max", 60.0)),
        ),
        storage=StorageConfig(
            db_path=db_path,
            retention_days=int(st.get("retention_days", 90)),
        ),
        web=WebConfig(
            enabled=bool(web_raw.get("enabled", True)),
            host=os.environ.get("WEB_HOST") or str(web_raw.get("host", "127.0.0.1")),
            port=int(os.environ.get("QQBOT_WEB_PORT") or web_raw.get("port", 8090)),
            token=os.environ.get("WEB_TOKEN", ""),
        ),
        # 走 _coerce 而不是逐字段显式赋值 —— 老写法每加一个字段都要回来改这里，
        # 漏了就静默失效（实测 reply_model / analysis_model 就是这么丢的）。
        # base_url 要去尾斜杠，所以单独覆盖一下。
        llm=LLMConfig(
            **{
                **_coerce(LLMConfig, strip_secrets("llm", llm_raw)),
                "base_url": str(llm_raw.get(
                    "base_url", "https://api.deepseek.com/v1")).rstrip("/"),
            },
            api_key=os.environ.get("LLM_API_KEY", ""),
        ),
        behavior=BehaviorConfig(**_coerce(BehaviorConfig, beh_raw)),
        sticker=StickerConfig(**_coerce(StickerConfig, stk_raw)),
        media=MediaConfig(**_coerce(MediaConfig, media_raw)),
        evolution=EvolutionConfig(**_coerce(EvolutionConfig, evo_raw)),
        debug=DebugConfig(**_coerce(DebugConfig, dbg_raw)),
        paddleocr=PaddleOCRConfig(
            **_coerce(PaddleOCRConfig, strip_secrets("paddleocr", ocr_raw)),
            token=os.environ.get("PADDLEOCR_TOKEN", ""),
        ),
        search=SearchConfig(
            **_coerce(SearchConfig, strip_secrets("search", search_raw)),
            api_key=os.environ.get("SEARCH_API_KEY", ""),
        ),
        bilibili=BilibiliConfig(
            **_coerce(BilibiliConfig, strip_secrets("bilibili", bili_raw)),
            cookie=os.environ.get("BILIBILI_COOKIE", ""),
            buvid3=os.environ.get("BILIBILI_BUVID3", ""),
            buvid4=os.environ.get("BILIBILI_BUVID4", ""),
        ),
        asr=AsrConfig(
            **_coerce(AsrConfig, strip_secrets("asr", asr_raw)),
            api_key=os.environ.get("STEPFUN_API_KEY", ""),
        ),
        log_level=os.environ.get("LOG_LEVEL") or bot.get("log_level", "INFO"),
        groups=[int(g) for g in bot.get("groups", [])],
        persona_path=persona,
        root=root,
    )
    return cfg


# --------------------------------------------------------------------------
# 运行时更新（供管理面板使用）
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


#: 各配置节的敏感字段。
#: **只在这里声明一次** —— 之前是各处硬编码 pop("api_key")/pop("token")，
#: 结果新增 bilibili.cookie 时没人记得加，443 字符的登录凭据被写进了
#: settings.json，既泄漏又导致 load_config 里 "got multiple values" 崩溃。
SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "llm": ("api_key",),
    "paddleocr": ("token",),
    "search": ("api_key",),
    "bilibili": ("cookie", "buvid3", "buvid4"),
    "asr": ("api_key",),
}


def secrets_of(section: str) -> tuple[str, ...]:
    return SECRET_FIELDS.get(section, ())


def strip_secrets(section: str, raw: dict[str, Any]) -> dict[str, Any]:
    """剥掉敏感字段 —— 用于两类地方：
       1. 写 settings.json 前
       2. load_config 里给显式 kwargs 让路（否则会 got multiple values）
    """
    drop = secrets_of(section)
    return {k: v for k, v in raw.items() if k not in drop}


def _secret_snapshot(cfg: Any) -> dict[str, Any]:
    """快照里去掉所有敏感字段，只回显"有没有配"。"""
    data = serialize_section(cfg)
    filled = {}
    for name in secrets_of(type(cfg).__name__.replace("Config", "").lower()):
        filled[f"{name}_set"] = bool(getattr(cfg, name, ""))
        data.pop(name, None)
    # 兜底：按字段名再扫一遍，防止节名映射不上
    for name in ("api_key", "token", "cookie", "buvid3", "buvid4"):
        if name in data:
            filled.setdefault(f"{name}_set", bool(data[name]))
            data.pop(name, None)
    data.update(filled)
    return data


def _paddle_snapshot(cfg: "PaddleOCRConfig") -> dict[str, Any]:
    """PaddleOCR 的快照 —— token 只回显"有没有配"，绝不回显内容。"""
    data = serialize_section(cfg)
    data.pop("token", None)
    data["token_set"] = bool(cfg.token)
    data["configured"] = cfg.configured
    return data


def serialize_section(obj: Any) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in asdict(obj).items()}


def settings_snapshot(cfg: Config) -> dict[str, Any]:
    """给面板用的设置快照，**绝不包含密钥明文**。"""
    llm = serialize_section(cfg.llm)
    llm.pop("api_key", None)
    llm["api_key_set"] = bool(cfg.llm.api_key)

    return {
        "llm": llm,
        "behavior": serialize_section(cfg.behavior),
        "sticker": serialize_section(cfg.sticker),
        "media": serialize_section(cfg.media),
        "evolution": serialize_section(cfg.evolution),
        "debug": serialize_section(cfg.debug),
        "paddleocr": _secret_snapshot(cfg.paddleocr),
        "search": _secret_snapshot(cfg.search),
        "bilibili": _secret_snapshot(cfg.bilibili),
        "asr": _secret_snapshot(cfg.asr),
        "schedule": {
            k: getattr(cfg.behavior, k)
            for k in (
                "active_from",
                "active_to",
                "jitter_minutes",
                "away_prob_per_hour",
                "group_hour_budget",
                "group_day_budget",
                "global_day_budget",
            )
        },
        "safety": {
            k: getattr(cfg.behavior, k)
            for k in ("max_chars_per_bubble", "max_bubbles")
        },
        "web": {"host": cfg.web.host, "port": cfg.web.port,
                "enabled": cfg.web.enabled, "token_set": bool(cfg.web.token)},
    }


def update_settings(cfg: Config, patch: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """把面板传来的局部更新热应用到 cfg，并持久化到 data/settings.json。

    返回 (最新快照, 改动日志)。
    """
    if not isinstance(patch, dict):
        raise ValueError("settings patch 必须是对象")

    # schedule / safety 是 behavior 的展示用子集，这里折回去
    merged: dict[str, dict[str, Any]] = {}
    for section in ("llm", "behavior", "web", "sticker", "media",
                    "evolution", "debug", "paddleocr", "search",
                    "bilibili", "asr"):
        sub = patch.get(section)
        if isinstance(sub, dict):
            merged.setdefault(section, {}).update(sub)
    for alias, targets in (
        ("schedule", (
            "active_from", "active_to", "jitter_minutes", "away_prob_per_hour",
            "group_hour_budget", "group_day_budget", "global_day_budget",
        )),
        ("safety", ("max_chars_per_bubble", "max_bubbles")),
    ):
        sub = patch.get(alias)
        if isinstance(sub, dict):
            bucket = merged.setdefault("behavior", {})
            for key in targets:
                if key in sub:
                    bucket[key] = sub[key]

    changes: list[str] = []

    # api_key：只允许"设置新值"，绝不回显；空字符串表示不动
    llm_patch = merged.get("llm")
    if isinstance(llm_patch, dict):
        new_key = llm_patch.pop("api_key", None)
        if new_key:
            cfg.llm.api_key = str(new_key).strip()
            # 同步写进 .env，否则重启就丢了
            _write_env_secret(cfg.root / ".env", "LLM_API_KEY", cfg.llm.api_key)
            changes.append("llm.api_key")
        if "base_url" in llm_patch:
            llm_patch["base_url"] = str(llm_patch["base_url"]).rstrip("/")

    for section, obj, env_key, field_name in (
        ("bilibili", cfg.bilibili, "BILIBILI_COOKIE", "cookie"),
        ("asr", cfg.asr, "STEPFUN_API_KEY", "api_key"),
    ):
        sub = merged.get(section)
        if isinstance(sub, dict):
            val = sub.pop(field_name, None)
            if val:
                setattr(obj, field_name, str(val).strip())
                _write_env_secret(cfg.root / ".env", env_key, getattr(obj, field_name))
                changes.append(f"{section}.{field_name}")

    search_patch = merged.get("search")
    if isinstance(search_patch, dict):
        new_key = search_patch.pop("api_key", None)
        if new_key:
            cfg.search.api_key = str(new_key).strip()
            _write_env_secret(cfg.root / ".env", "SEARCH_API_KEY", cfg.search.api_key)
            changes.append("search.api_key")

    ocr_patch = merged.get("paddleocr")
    if isinstance(ocr_patch, dict):
        new_token = ocr_patch.pop("token", None)
        if new_token:
            cfg.paddleocr.token = str(new_token).strip()
            _write_env_secret(cfg.root / ".env", "PADDLEOCR_TOKEN", cfg.paddleocr.token)
            changes.append("paddleocr.token")

    for section, obj in (
        ("llm", cfg.llm),
        ("behavior", cfg.behavior),
        ("web", cfg.web),
        ("sticker", cfg.sticker),
        ("media", cfg.media),
        ("evolution", cfg.evolution),
        ("debug", cfg.debug),
        ("paddleocr", cfg.paddleocr),
        ("search", cfg.search),
        ("bilibili", cfg.bilibili),
        ("asr", cfg.asr),
    ):
        sub = merged.get(section)
        if not isinstance(sub, dict):
            continue
        for name in _apply_section(obj, sub):
            changes.append(f"{section}.{name}")

    if changes:
        _persist_overlay(cfg, merged, changes)

    return settings_snapshot(cfg), changes


def _persist_overlay(cfg: Config, merged: dict[str, dict[str, Any]], changes: list[str]) -> None:
    """把改动写进 data/settings.json 覆盖层（不含密钥）。"""
    path = cfg.settings_path
    data = _read_settings_file(path)

    touched = {c.split(".", 1)[0] for c in changes}
    for section in touched:
        if section not in SECTIONS:
            continue
        # 该节当前的全部非敏感值，保证覆盖层自洽
        current = strip_secrets(section, serialize_section(getattr(cfg, section)))
        data[section] = current

    # 面板改的 behavior 别名也要落到 behavior 节（上面已整体写入）

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)