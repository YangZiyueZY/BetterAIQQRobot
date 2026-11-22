"""作息状态机与发言额度。

拟人的一半在于"有生活"。这一层负责：

  - **作息规律**：真人每天的上线时间是有规律的 —— 一个高中生工作日只可能
    中午和晚上出现，周末和节假日时间充裕。所以按"日期类型"给不同的时段表。
  - **随机性**：同一天的起止按日期确定性抖动（同一天内稳定、不同天不一样），
    否则每天分秒不差地上线，比固定时间还假。
  - **突发性**：真人也会有意外 —— 某天生病请假，工作日大白天就在线。
    用 `off_schedule_prob` 表达，一天最多掷一次。
  - **熬夜**：偶尔玩到凌晨，用 `late_night_prob` 把当晚的窗口往后延。
  - **随机外出**：按泊松速率随机进入 30~90 分钟的 AWAY，期间完全不说话。
  - **忙碌**：较短（5~20 分钟）的 BUSY，仍然会回，但语气应该更敷衍。
  - **三层限流**：单群每小时、单群每天、全局每天。额度是安全阀，不是优化项 ——
    LLM 一旦进入循环，没有硬限流就是刷屏灾难。

关于配置的兼容性
----------------
`weekday_windows` / `weekend_windows` / `holiday_windows` **默认为空**，
空表示回落到旧的单窗口 `active_from`~`active_to`。这样不改变既有行为，
想用"工作日中午+晚上、周末全天"这种作息时再显式配上即可。

格式：`"11:30-13:30,17:30-23:30"`，逗号分隔多个窗口，支持跨午夜（`22:00-02:00`）。
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..config import BehaviorConfig
from ..store.repo import Repo

log = logging.getLogger(__name__)

DAY = 86400


class Presence(str, Enum):
    ONLINE = "online"
    BUSY = "busy"
    AWAY = "away"
    SLEEPING = "sleeping"

    @property
    def label(self) -> str:
        return {
            "online": "在线",
            "busy": "忙碌",
            "away": "外出",
            "sleeping": "睡觉",
        }[self.value]


class DayKind(str, Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

    @property
    def label(self) -> str:
        return {
            "weekday": "工作日",
            "weekend": "周末",
            "holiday": "节假日",
        }[self.value]


@dataclass
class DayPlan:
    """某一天的作息计划。**同一天内结果稳定**（用日期做种子）。"""

    day_key: str
    kind: DayKind
    windows: list[tuple[float, float]] = field(default_factory=list)
    """epoch 秒的 [start, end) 列表，可能多个（如中午 + 晚上）。"""
    off_schedule: bool = False
    """今天作息被打乱（生病请假之类）—— 会在本不该出现的时候出现。"""
    late_night: bool = False
    """今天熬夜，最后一个窗口延到了凌晨。"""
    note: str = ""

    def covers(self, ts: float) -> bool:
        return any(start <= ts < end for start, end in self.windows)

    def current_window(self, ts: float) -> tuple[float, float] | None:
        for start, end in self.windows:
            if start <= ts < end:
                return start, end
        return None

    def next_window(self, ts: float) -> tuple[float, float] | None:
        """下一个还没开始的窗口。"""
        for start, end in self.windows:
            if ts < start:
                return start, end
        return None


def _parse_hhmm(text: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        hh, _, mm = str(text).partition(":")
        return int(hh) % 24, int(mm) % 60
    except (ValueError, AttributeError):
        return default


def parse_windows(spec: str) -> list[tuple[float, float]]:
    """把 `"11:30-13:30,17:30-23:30"` 解析成小时浮点对。

    支持跨午夜（`22:00-02:00` → end 加 24）。解析失败的片段直接跳过 ——
    作息配错了不该让机器人起不来。
    """
    out: list[tuple[float, float]] = []
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        left, _, right = chunk.partition("-")
        try:
            sh, sm = _parse_hhmm(left.strip(), (0, 0))
            eh, em = _parse_hhmm(right.strip(), (0, 0))
        except Exception:  # noqa: BLE001
            continue
        start = sh + sm / 60.0
        end = eh + em / 60.0
        if end <= start:
            end += 24.0          # 跨午夜
        out.append((start, end))
    return sorted(out)


class Scheduler:
    def __init__(self, cfg: BehaviorConfig, repo: Repo, debug=None) -> None:
        self.cfg = cfg
        self.repo = repo
        self.debug = debug
        """传入 DebugConfig 后，测试模式下作息与额度全部失效。"""
        self._away_until = 0.0
        self._busy_until = 0.0
        self._last_rate_check = time.time()
        self._cached_day = ""
        self._plan: DayPlan | None = None

    # ---- 作息计划 -----------------------------------------------------

    def _is_holiday(self, dt: datetime) -> bool:
        """`holidays` 里可以写 `MM-DD` 或 `YYYY-MM-DD`，两种都认。"""
        keys = {f"{dt.month:02d}-{dt.day:02d}", dt.strftime("%Y-%m-%d")}
        return any(str(h).strip() in keys for h in (self.cfg.holidays or []))

    def _windows_for(self, kind: DayKind) -> str:
        """取该日期类型的窗口配置；没配就回落到旧的单窗口。"""
        spec = {
            DayKind.WEEKDAY: self.cfg.weekday_windows,
            DayKind.WEEKEND: self.cfg.weekend_windows,
            DayKind.HOLIDAY: self.cfg.holiday_windows,
        }[kind]
        if str(spec or "").strip():
            return str(spec)
        # 回落：单窗口 active_from ~ active_to（老配置继续可用）
        return f"{self.cfg.active_from}-{self.cfg.active_to}"

    def plan_for(self, now_ts: float) -> DayPlan:
        """返回当前自然日的作息计划。

        **用日期字符串做随机种子** —— 同一天内反复调用结果一致，
        否则每次判断都在抖，会出现"刚上线又下线"。
        """
        dt = datetime.fromtimestamp(now_ts)
        day_key = dt.strftime("%Y-%m-%d")
        if day_key == self._cached_day and self._plan is not None:
            return self._plan

        rng = random.Random(f"qqbot-schedule-{day_key}")

        if self._is_holiday(dt):
            kind = DayKind.HOLIDAY
        elif dt.weekday() >= 5:
            kind = DayKind.WEEKEND
        else:
            kind = DayKind.WEEKDAY

        spec = self._windows_for(kind)
        raw = parse_windows(spec)
        if not raw:
            raw = parse_windows(f"{self.cfg.active_from}-{self.cfg.active_to}")
        if not raw:
            raw = [(9.0, 22.0)]        # 兜底：配置全坏了也别不出声

        note_parts = [kind.label]
        off_schedule = False
        # 突发：工作日也可能白天在线（生病请假 / 调休）。
        # 只在工作日掷，周末本来就自由，没有"突发"可言。
        prob = max(0.0, float(getattr(self.cfg, "off_schedule_prob", 0.0) or 0.0))
        if kind is DayKind.WEEKDAY and prob > 0 and rng.random() < prob:
            off_schedule = True
            free = parse_windows(self.cfg.weekend_windows or "") or [(9.0, 22.0)]
            raw = sorted(set(raw + free))
            note_parts.append("今天不太舒服/请假，作息打乱了")

        # 熬夜：把最后一个窗口往后延 1~2.5 小时
        late_night = False
        ln_prob = max(0.0, float(getattr(self.cfg, "late_night_prob", 0.0) or 0.0))
        if ln_prob > 0 and rng.random() < ln_prob:
            late_night = True
            start, end = raw[-1]
            raw = raw[:-1] + [(start, end + rng.uniform(1.0, 2.5))]
            note_parts.append("今天熬夜了")

        # 起止抖动：同一天一个整体偏移，模拟"今天起得早/晚"
        shift_min = rng.randint(-abs(self.cfg.jitter_minutes), abs(self.cfg.jitter_minutes))
        anchor = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        shift = timedelta(minutes=shift_min)
        windows = [
            (
                (anchor + timedelta(hours=s)).timestamp() + shift.total_seconds(),
                (anchor + timedelta(hours=e)).timestamp() + shift.total_seconds(),
            )
            for s, e in raw
        ]

        self._cached_day = day_key
        self._plan = DayPlan(
            day_key=day_key, kind=kind, windows=windows,
            off_schedule=off_schedule, late_night=late_night,
            note="；".join(note_parts),
        )
        log.debug("作息计划 %s: %s %s", day_key, self._plan.note,
                  [(int(a), int(b)) for a, b in windows])
        return self._plan

    # 兼容旧接口 ------------------------------------------------------

    def window_for(self, now_ts: float) -> tuple[float, float]:
        """返回 [start, end)：当前所在窗口，否则下一个，再否则当天最后一个。

        保留这个签名是因为面板和旧代码都在用。多窗口的完整信息看 `plan_for()`。
        """
        plan = self.plan_for(now_ts)
        cur = plan.current_window(now_ts)
        if cur:
            return cur
        nxt = plan.next_window(now_ts)
        if nxt:
            return nxt
        return plan.windows[-1]

    def in_window(self, now_ts: float) -> bool:
        return self.plan_for(now_ts).covers(now_ts)

    # ---- 状态 ---------------------------------------------------------

    @property
    def forced(self) -> bool:
        return bool(self.debug is not None and self.debug.enabled)

    def presence(self, now_ts: float | None = None) -> Presence:
        now = now_ts if now_ts is not None else time.time()

        # 测试模式：永远在线，也不掷外出/忙碌的骰子
        if self.forced:
            return Presence.ONLINE

        if not self.in_window(now):
            return Presence.SLEEPING

        if now < self._away_until:
            return Presence.AWAY
        if now < self._busy_until:
            return Presence.BUSY

        self._maybe_roll(now)
        if now < self._away_until:
            return Presence.AWAY
        if now < self._busy_until:
            return Presence.BUSY
        return Presence.ONLINE

    def _maybe_roll(self, now: float) -> None:
        """按经过的时间掷骰子，决定是否进入外出/忙碌。"""
        dt = max(0.0, now - self._last_rate_check)
        self._last_rate_check = now
        if dt <= 0:
            return

        away_rate = max(0.0, self.cfg.away_prob_per_hour) / 3600.0
        if away_rate and random.random() < 1 - math.exp(-away_rate * dt):
            minutes = random.uniform(30, 90)
            self._away_until = now + minutes * 60
            log.info("进入外出状态，%.0f 分钟后回来", minutes)
            return

        busy_rate = away_rate * 0.8
        if busy_rate and random.random() < 1 - math.exp(-busy_rate * dt):
            minutes = random.uniform(5, 20)
            self._busy_until = now + minutes * 60
            log.info("进入忙碌状态，%.0f 分钟后恢复", minutes)

    def force_state(self, presence: Presence, minutes: float = 30.0) -> None:
        """面板手动干预用。"""
        now = time.time()
        if presence is Presence.AWAY:
            self._away_until = now + minutes * 60
        elif presence is Presence.BUSY:
            self._busy_until = now + minutes * 60
        elif presence is Presence.ONLINE:
            self._away_until = 0.0
            self._busy_until = 0.0

    # ---- 额度 ---------------------------------------------------------

    async def check_budget(self, group_id: int, now_ts: float | None = None) -> tuple[bool, str]:
        if self.forced:
            return True, ""
        now = int(now_ts if now_ts is not None else time.time())
        cfg = self.cfg

        state = await self.repo.budget_state(group_id, now)
        if cfg.group_hour_budget > 0 and state["hour_used"] >= cfg.group_hour_budget:
            return False, f"本群本小时额度用完({state['hour_used']}/{cfg.group_hour_budget})"
        if cfg.group_day_budget > 0 and state["day_used"] >= cfg.group_day_budget:
            return False, f"本群今日额度用完({state['day_used']}/{cfg.group_day_budget})"

        global_used = await self.repo.global_day_used(now)
        if cfg.global_day_budget > 0 and global_used >= cfg.global_day_budget:
            return False, f"全局今日额度用完({global_used}/{cfg.global_day_budget})"

        return True, ""

    async def note_spoke(self, group_id: int, now_ts: float | None = None) -> None:
        """成功发言后扣额度。"""
        now = int(now_ts if now_ts is not None else time.time())
        await self.repo.bump_budget(group_id, now)
        await self.repo.bump_budget(0, now)  # 全局伪 group

    async def budget_snapshot(self, group_id: int) -> dict:
        now = int(time.time())
        state = await self.repo.budget_state(group_id, now)
        return {
            "hour_used": state["hour_used"],
            "hour_limit": self.cfg.group_hour_budget,
            "day_used": state["day_used"],
            "day_limit": self.cfg.group_day_budget,
        }

    def describe(self) -> dict:
        now = time.time()
        presence = self.presence(now)
        plan = self.plan_for(now)
        cur = plan.current_window(now)
        nxt = plan.next_window(now)
        return {
            "presence": presence.value,
            "presence_label": presence.label,
            "window_start": int(cur[0]) if cur else 0,
            "window_end": int(cur[1]) if cur else 0,
            "away_until": int(self._away_until) if self._away_until > now else 0,
            "busy_until": int(self._busy_until) if self._busy_until > now else 0,
            # 作息全景：面板要能让用户看懂"它今天打算什么时候在线"
            "day_kind": plan.kind.value,
            "day_kind_label": plan.kind.label,
            "day_note": plan.note,
            "off_schedule": plan.off_schedule,
            "late_night": plan.late_night,
            "windows": [
                {"start": int(a), "end": int(b)} for a, b in plan.windows
            ],
            "next_online": int(nxt[0]) if nxt else 0,
        }
