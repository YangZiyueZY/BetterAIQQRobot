"""持久化层。

按领域拆成几个 repo，共享同一个 Database 连接：

    Repo           消息 / 群 / 成员 / 额度 / 决策 / meta
    MemoryRepo     带类型的长期记忆
    StickerRepo    表情库、使用记录、使用语境观察
    MediaRepo      图片/语音的多模态理解缓存
    ProfileRepo    群统计画像
    PolicyRepo     群策略参数（自进化产物）
    FeedbackRepo   奖励信号
"""

from .db import Database
from .evolution import (
    SIGNAL_WEIGHTS,
    BOUNDS,
    COLD_START_SAMPLES,
    FeedbackRepo,
    GroupPolicy,
    GroupProfile,
    PolicyRepo,
    ProfileRepo,
)
from .media import MediaRepo
from .ocr import OcrJobRepo
from .onboard import OnboardRepo
from .personas import PersonaRepo
from .prompts import PromptRepo
from .survival import SurvivalRepo
from .memories import TTL_BY_TYPE, MemoryRepo
from .repo import Repo
from .stickers import StickerRepo

__all__ = [
    "Database",
    "Repo",
    "MemoryRepo",
    "TTL_BY_TYPE",
    "StickerRepo",
    "MediaRepo",
    "OcrJobRepo",
    "OnboardRepo",
    "PersonaRepo",
    "PromptRepo",
    "SurvivalRepo",
    "ProfileRepo",
    "PolicyRepo",
    "FeedbackRepo",
    "GroupProfile",
    "GroupPolicy",
    "SIGNAL_WEIGHTS",
    "BOUNDS",
    "COLD_START_SAMPLES",
]
