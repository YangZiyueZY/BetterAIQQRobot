"""QQ 内置表情目录。

数据来自 NapCat 上游的 `face_config.json`（`packages/napcat-core/external/`），
包含两张表：

  sysface  329 条  QSid（数字 id） -> QDes（中文名）   ← OneBot `face` 段用的就是它
  emoji    165 条  Unicode 字符   -> QDes（QQ 给的中文名）

设计要点：**给 LLM 的是中文名，不是数字 id**。
模型完全不懂 `{"type":"face","id":178}`，但它非常懂"斜眼笑"在中文互联网的语感。
所以 prompt 里列出可用表情名，模型输出 `[face:斜眼笑]`，发送层再翻译回 id。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "face_config.json"

#: 最常用于群聊口语的一批。prompt 里默认只列这些，避免几百个名字把上下文吃光。
#: 挑选标准：低 id 的经典表情 + 中文互联网高频梗（doge/笑哭/裂开/尊嘟假嘟…）。
CORE_NAMES: tuple[str, ...] = (
    # 经典小黄脸
    "微笑", "呲牙", "偷笑", "大笑", "笑哭", "坏笑", "憨笑", "调皮",
    "流泪", "大哭", "快哭了", "委屈", "可怜", "难过", "衰", "尴尬",
    "害羞", "可爱", "卖萌", "亲亲", "飞吻", "么么哒", "比心", "抱抱",
    "白眼", "傲慢", "鄙视", "嫌弃", "抠鼻", "阴险", "坏笑", "呵呵哒",
    "发怒", "咒骂", "抓狂", "生气", "哼", "无语", "面无表情", "冷漠",
    "惊讶", "惊恐", "害怕", "吓", "疑问", "问号脸", "咦", "不是吧",
    "流汗", "冷汗", "擦汗", "晕", "裂开", "宕机", "emm", "呃",
    "困", "睡", "晚安", "失眠", "头秃", "脑阔疼", "沧桑", "emo",
    "得意", "酷", "耶", "开心", "快乐", "庆祝", "鼓掌", "喝彩",
    "赞", "点赞", "超级赞", "你真棒棒", "牛啊", "666", "真棒", "强",
    "踩", "差劲", "拒绝", "tui", "略略略", "略",
    "吃瓜", "暗中观察", "让我康康", "摸鱼", "发呆", "无聊", "托腮",
    "斜眼笑", "doge", "狗头", "汪汪", "喵喵", "羊驼", "扔狗",
    "捂脸", "辣眼睛", "我酸了", "酸Q", "我哭了", "呜呜呜",
    "求求", "拜托", "求放过", "贴贴", "尊嘟假嘟", "你懂的",
    "对的对的", "不对不对", "收到", "OK", "NO", "比爱心",
    "干饭", "喝茶", "咖啡", "西瓜", "蛋糕", "玫瑰", "送你花花",
    "打招呼", "嗨", "再见", "挥手", "抱拳", "敬礼", "崇拜",
    "心动", "爱心", "心碎", "坚强", "硬撑", "优雅",
    "开心", "出去玩", "c位出道", "闪亮登场", "打call", "好闪",
)

#: 明显不适合机器人用的（营销/节日/需要上下文），从核心集里剔除
_EXCLUDE = {"送你花花", "c位出道", "闪亮登场"}


@dataclass(frozen=True, slots=True)
class Face:
    id: int
    name: str

    @property
    def token(self) -> str:
        """prompt 里给模型看的写法。"""
        return f"[face:{self.name}]"


@dataclass(frozen=True, slots=True)
class Emoji:
    char: str
    codepoint: int
    name: str


class FaceCatalog:
    """QQ 表情目录。加载一次，全局复用。"""

    def __init__(self, faces: dict[int, Face], emojis: dict[str, Emoji]) -> None:
        self.faces = faces
        self.emojis = emojis
        self.by_name: dict[str, Face] = {f.name: f for f in faces.values()}
        # 同名不同 id 时保留 id 最小的那个（经典表情）
        for fid in sorted(faces):
            face = faces[fid]
            self.by_name.setdefault(face.name, face)

    # ---- 加载 ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "FaceCatalog":
        path = path or DEFAULT_PATH
        if not path.is_file():
            log.warning("表情目录不存在: %s，QQ 内置表情功能将不可用", path)
            return cls({}, {})

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error("表情目录解析失败: %s", exc)
            return cls({}, {})

        faces: dict[int, Face] = {}
        for item in raw.get("sysface") or []:
            qsid = str(item.get("QSid") or "")
            desc = str(item.get("QDes") or "").lstrip("/").strip()
            if not qsid.isdigit() or not desc:
                continue
            fid = int(qsid)
            # 同一 id 只留第一个（上游有重复项）
            faces.setdefault(fid, Face(id=fid, name=desc))

        emojis: dict[str, Emoji] = {}
        for item in raw.get("emoji") or []:
            char = str(item.get("QSid") or "")
            desc = str(item.get("QDes") or "").lstrip("/").strip()
            if not char or not desc:
                continue
            try:
                codepoint = int(item.get("QCid") or 0)
            except (TypeError, ValueError):
                codepoint = ord(char[0])
            emojis.setdefault(char, Emoji(char=char, codepoint=codepoint, name=desc))

        log.info("表情目录已加载: %d 个系统表情 / %d 个 emoji", len(faces), len(emojis))
        return cls(faces, emojis)

    # ---- 查询 ---------------------------------------------------------

    def get(self, face_id: int) -> Face | None:
        return self.faces.get(face_id)

    def by_token(self, token: str) -> Face | None:
        """解析模型输出：支持 "斜眼笑" / "/斜眼笑" / "178" / "[face:斜眼笑]"。"""
        text = (token or "").strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if text.lower().startswith("face:"):
            text = text[5:]
        text = text.lstrip("/").strip()

        if text.isdigit():
            return self.faces.get(int(text))
        return self.by_name.get(text)

    def core(self) -> list[Face]:
        """给 prompt 用的核心表情集：按 CORE_NAMES 顺序，去重、剔除黑名单。"""
        out: list[Face] = []
        seen: set[int] = set()
        for name in CORE_NAMES:
            if name in _EXCLUDE:
                continue
            face = self.by_name.get(name)
            if face is None or face.id in seen:
                continue
            seen.add(face.id)
            out.append(face)
        return out

    def prompt_block(self, limit: int = 90) -> str:
        """渲染成 prompt 里的一行清单。"""
        names = [f.name for f in self.core()[:limit]]
        if not names:
            return ""
        return "、".join(names)

    def stats(self) -> dict:
        return {
            "sysface": len(self.faces),
            "emoji": len(self.emojis),
            "core": len(self.core()),
        }

    def all_faces(self) -> list[Face]:
        return [self.faces[i] for i in sorted(self.faces)]


_catalog: FaceCatalog | None = None


def catalog() -> FaceCatalog:
    """进程内单例。"""
    global _catalog
    if _catalog is None:
        _catalog = FaceCatalog.load()
    return _catalog
