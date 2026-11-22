"""B 站扫码登录。

**这里有个必须遵守的坑（文档实测结论）**：二维码是**一次性消费**的。
`code=0` 那一次的响应才下发 `Set-Cookie`，如果当时没接住，
之后再 poll 就直接变 `86038 已失效`，Cookie **永久拿不回来**，只能重新生成重扫。

所以 poll 成功时必须**当场**从响应的 Set-Cookie 里把凭据抠出来落盘，
不能等下一轮，也不能靠 cookie jar 慢慢攒。

状态码：
    86101  未扫码
    86090  已扫码，待手机确认
    0      登录成功（此响应下发 Set-Cookie）
    86038  二维码已过期
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field

import aiohttp

log = logging.getLogger(__name__)

PASSPORT = "https://passport.bilibili.com"
API = "https://api.bilibili.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

#: 要从 Set-Cookie 里保留的字段。SESSDATA 是必需的，其余带上更稳。
WANTED_COOKIES = ("SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5",
                  "sid", "buvid3", "buvid4")

#: 二维码有效期（官方约 3 分钟，取保守值）
QR_TTL = 170.0

STATE_WAITING = "waiting"      # 未扫码
STATE_SCANNED = "scanned"      # 已扫码待确认
STATE_CONFIRMED = "confirmed"  # 成功
STATE_EXPIRED = "expired"
STATE_ERROR = "error"

CODE_TO_STATE = {
    86101: STATE_WAITING,
    86090: STATE_SCANNED,
    0: STATE_CONFIRMED,
    86038: STATE_EXPIRED,
}

STATE_TEXT = {
    STATE_WAITING: "等待扫码",
    STATE_SCANNED: "已扫码，请在手机上确认",
    STATE_CONFIRMED: "登录成功",
    STATE_EXPIRED: "二维码已过期，请重新获取",
    STATE_ERROR: "出错了",
}


@dataclass
class QrSession:
    qrcode_key: str = ""
    url: str = ""
    png_data_uri: str = ""
    created_at: float = field(default_factory=time.time)
    state: str = STATE_WAITING
    message: str = ""
    cookie: str = ""
    uname: str = ""
    mid: int = 0

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > QR_TTL

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "state_text": STATE_TEXT.get(self.state, self.state),
            "message": self.message,
            "qr": self.png_data_uri,
            "expires_in": max(0, int(QR_TTL - (time.time() - self.created_at))),
            "uname": self.uname,
            "mid": self.mid,
        }


def render_qr(data: str) -> str:
    """把内容渲染成 PNG 的 data URI，前端直接塞进 <img src>。"""
    try:
        import qrcode
    except ImportError:
        log.warning("没装 qrcode，无法渲染二维码（pip install qrcode）")
        return ""

    qr = qrcode.QRCode(box_size=6, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_cookie_header(cookies: dict[str, str]) -> str:
    """按固定顺序拼 Cookie 头。SESSDATA 放最前面。"""
    parts = []
    for name in WANTED_COOKIES:
        value = cookies.get(name)
        if value:
            parts.append(f"{name}={value}")
    # 其余没列进 WANTED 的也带上（有些字段名会变）
    for name, value in cookies.items():
        if name not in WANTED_COOKIES and value:
            parts.append(f"{name}={value}")
    return "; ".join(parts)


class BiliLogin:
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self.current: QrSession | None = None
        self.starts = 0
        self.successes = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # 带 cookie jar：B 站会在 generate 时下发 buvid 之类的指纹
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                cookie_jar=aiohttp.CookieJar(unsafe=True),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    @property
    def _headers(self) -> dict:
        return {"User-Agent": UA, "Referer": "https://www.bilibili.com/"}

    # ---- 生成二维码 ---------------------------------------------------

    async def start(self) -> QrSession:
        session = await self._get_session()
        url = (f"{PASSPORT}/x/passport-login/web/qrcode/generate"
               "?source=main-fe-header")
        session_state = QrSession()

        try:
            async with session.get(url, headers=self._headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    session_state.state = STATE_ERROR
                    session_state.message = f"HTTP {resp.status}"
                    self.current = session_state
                    return session_state
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            session_state.state = STATE_ERROR
            session_state.message = f"请求失败: {exc}"
            self.current = session_state
            return session_state

        import json
        try:
            data = json.loads(body).get("data") or {}
        except json.JSONDecodeError:
            session_state.state = STATE_ERROR
            session_state.message = "响应不是 JSON"
            self.current = session_state
            return session_state

        session_state.qrcode_key = str(data.get("qrcode_key") or "")
        session_state.url = str(data.get("url") or "")
        session_state.png_data_uri = render_qr(session_state.url)
        if not session_state.png_data_uri:
            session_state.state = STATE_ERROR
            session_state.message = "二维码渲染失败（缺 qrcode 库）"

        self.current = session_state
        self.starts += 1
        log.info("B站扫码登录：已生成二维码")
        return session_state

    # ---- 轮询 ---------------------------------------------------------

    async def poll(self) -> QrSession:
        """轮询一次。成功时**当场**接住 Set-Cookie 并返回。"""
        state = self.current
        if state is None or not state.qrcode_key:
            return QrSession(state=STATE_ERROR, message="还没生成二维码")

        if state.state == STATE_CONFIRMED:
            return state                       # 已经成功了，别把 Cookie 消费掉
        if state.expired and state.state != STATE_CONFIRMED:
            state.state = STATE_EXPIRED
            return state

        session = await self._get_session()
        url = (f"{PASSPORT}/x/passport-login/web/qrcode/poll"
               f"?qrcode_key={state.qrcode_key}&source=main-fe-header")

        try:
            async with session.get(url, headers=self._headers) as resp:
                body = await resp.text()
                # ★ 关键：Set-Cookie 只在 code=0 这一下发一次，过期就没了。
                #   aiohttp 会把它解析进 resp.cookies，立刻抠出来。
                raw_cookies = {
                    name: morsel.value
                    for name, morsel in (resp.cookies or {}).items()
                }
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            state.state = STATE_ERROR
            state.message = f"轮询失败: {exc}"
            return state

        import json
        try:
            payload = json.loads(body)
            data = payload.get("data") or {}
        except json.JSONDecodeError:
            state.message = "轮询响应不是 JSON"
            return state

        code = data.get("code")
        mapped = CODE_TO_STATE.get(code if isinstance(code, int) else -1)

        if mapped == STATE_CONFIRMED:
            if not raw_cookies.get("SESSDATA"):
                # 极少见：成功了但没给 SESSDATA。明说，别让人以为登录了
                state.state = STATE_ERROR
                state.message = ("登录成功但没收到 SESSDATA —— 请重新生成二维码再扫一次"
                                 "（二维码是一次性的，这一轮已经消费掉了）")
                return state

            state.state = STATE_CONFIRMED
            state.cookie = build_cookie_header(raw_cookies)
            state.message = data.get("message") or "登录成功"
            self.successes += 1
            log.info("B站扫码登录成功，已获取 new cookie（%d 个字段）", len(raw_cookies))

            info = await self.nav_info(state.cookie)
            state.uname = info.get("uname", "")
            state.mid = int(info.get("mid") or 0)
            return state

        if mapped == STATE_EXPIRED:
            state.state = STATE_EXPIRED
            state.message = data.get("message") or "二维码已过期"
            return state

        if mapped == STATE_SCANNED:
            state.state = STATE_SCANNED
            state.message = data.get("message") or "已扫码，请在手机上确认"
            return state

        if mapped == STATE_WAITING:
            state.state = STATE_WAITING
            state.message = data.get("message") or "等待扫码"
            return state

        state.message = f"未知状态 code={code}"
        return state

    # ---- 校验 ---------------------------------------------------------

    async def nav_info(self, cookie: str) -> dict:
        """用 nav 接口校验登录态。data.isLogin 才是权威信号。"""
        if not cookie:
            return {"is_login": False, "uname": "", "mid": 0}
        session = await self._get_session()
        headers = dict(self._headers)
        headers["Cookie"] = cookie
        try:
            async with session.get(f"{API}/x/web-interface/nav",
                                   headers=headers) as resp:
                body = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {"is_login": None, "uname": "", "mid": 0, "error": "网络失败"}

        import json
        try:
            data = json.loads(body).get("data") or {}
        except json.JSONDecodeError:
            return {"is_login": None, "uname": "", "mid": 0, "error": "响应异常"}

        return {
            "is_login": bool(data.get("isLogin")),
            "uname": str(data.get("uname") or ""),
            "mid": int(data.get("mid") or 0),
            "vip": bool((data.get("vipStatus") or 0)),
        }

    async def verify_current(self, cookie: str) -> dict:
        self.login = None
        return await self.nav_info(cookie)

    def stats(self) -> dict:
        return {"bili_login_starts": self.starts, "bili_login_successes": self.successes}
