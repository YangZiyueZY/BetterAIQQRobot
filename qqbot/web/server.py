"""管理面板的 HTTP 服务。

直接复用 aiohttp.web，和 gateway 跑在**同一个事件循环、同一个进程**里 ——
面板要读的实时状态（连接状态、队列长度、额度、决策留痕）本来就在内存里，
再起一个进程就得做 IPC，得不偿失。

认证：登录后下发一个随机 session id 放在 HttpOnly Cookie 里，令牌本身
不进 Cookie。web.token 为空时不做认证（仅建议在 127.0.0.1 上这么用）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from pathlib import Path

from aiohttp import web

from ..config import VERSION, Config, settings_snapshot, update_settings
from ..gateway.client import Gateway
from ..models import Draft
from ..generate.prompt import build_system, build_user
from ..persona import PersonaError, parse as parse_persona
from ..pipeline import Pipeline
from ..search.client import PROVIDER_HINTS, host_warning
from ..store.memories import TTL_BY_TYPE
from .. import prompts
from ..store.repo import Repo
from .logs import RingLogHandler

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
COOKIE_NAME = "qqbot_session"
SESSION_TTL = 7 * 86400

#: 登录失败节流。面板一旦绑到 0.0.0.0 就是局域网可达的，
#: 光靠一个令牌不够 —— 必须挡住在线爆破。
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW = 300.0
"""失败计数窗口（秒）。"""
LOGIN_LOCKOUT = 60.0
"""达到上限后的锁定时长（秒）。"""

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def json_response(data, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def error(message: str, status: int = 400) -> web.Response:
    return json_response({"error": message}, status=status)


class WebServer:
    def __init__(
        self,
        cfg: Config,
        repo: Repo,
        pipeline: Pipeline,
        gateway: Gateway,
        log_handler: RingLogHandler,
    ) -> None:
        self.cfg = cfg
        self.repo = repo
        self.pipeline = pipeline
        self.gateway = gateway
        self.logs = log_handler
        self.nickname = ""
        self._sessions: dict[str, float] = {}
        self._login_failures: dict[str, list[float]] = {}
        self._login_locked_until: dict[str, float] = {}
        self._runner: web.AppRunner | None = None
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # 应用装配
    # ------------------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application(middlewares=[self._auth_middleware])
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/favicon.ico", self.handle_favicon)
        app.router.add_get("/static/{name:.*}", self.handle_static)

        app.router.add_post("/api/login", self.handle_login)
        app.router.add_post("/api/logout", self.handle_logout)
        app.router.add_get("/api/me", self.handle_me)

        app.router.add_get("/api/status", self.handle_status)
        app.router.add_get("/api/groups", self.handle_groups)
        app.router.add_patch("/api/groups/{group_id}", self.handle_group_patch)
        app.router.add_get("/api/groups/{group_id}/messages", self.handle_group_messages)
        app.router.add_get("/api/decisions", self.handle_decisions)

        app.router.add_get("/api/persona", self.handle_persona_get)
        app.router.add_put("/api/persona", self.handle_persona_put)
        app.router.add_post("/api/persona/validate", self.handle_persona_validate)
        app.router.add_post("/api/persona/reset", self.handle_persona_reset)
        app.router.add_get("/api/prompts", self.handle_prompts_get)
        app.router.add_put("/api/prompts", self.handle_prompts_put)
        app.router.add_post("/api/prompts/reset", self.handle_prompts_reset)
        app.router.add_get("/api/prompts/preview", self.handle_prompts_preview)
        app.router.add_get("/api/onboard", self.handle_onboard_get)
        app.router.add_post("/api/onboard/restart", self.handle_onboard_restart)

        app.router.add_get("/api/settings", self.handle_settings_get)
        app.router.add_put("/api/settings", self.handle_settings_put)
        app.router.add_post("/api/settings/llm/test", self.handle_llm_test)
        app.router.add_post("/api/settings/search/test", self.handle_search_test)
        app.router.add_post("/api/bilibili/login/qrcode", self.handle_bili_qr_start)
        app.router.add_get("/api/bilibili/login/status", self.handle_bili_qr_poll)
        app.router.add_get("/api/bilibili/status", self.handle_bili_status)
        app.router.add_get("/api/models/profiles", self.handle_model_profiles)
        app.router.add_post("/api/models/apply", self.handle_model_apply)

        app.router.add_post("/api/control/pause", self.handle_pause)
        app.router.add_post("/api/control/resume", self.handle_resume)
        app.router.add_post("/api/control/debug", self.handle_debug)
        app.router.add_post("/api/control/say", self.handle_say)
        app.router.add_post("/api/control/test", self.handle_dry_run)

        app.router.add_get("/api/logs/recent", self.handle_logs_recent)
        app.router.add_get("/api/logs/stream", self.handle_logs_stream)
        app.router.add_get("/api/stats/timeseries", self.handle_timeseries)

        app.router.add_get("/api/memories", self.handle_memories)
        app.router.add_delete("/api/memories/{memory_id}", self.handle_memory_delete)

        # --- 表情包 ---
        app.router.add_get("/api/stickers", self.handle_stickers)
        app.router.add_get("/api/stickers/stats", self.handle_sticker_stats)
        app.router.add_get("/api/faces", self.handle_faces)
        app.router.add_post("/api/stickers/{sticker_id}/block", self.handle_sticker_block)
        app.router.add_get("/api/stickers/{sticker_id}/file", self.handle_sticker_file)
        app.router.add_delete("/api/stickers/{sticker_id}", self.handle_sticker_delete)

        # --- 群档案与自进化 ---
        app.router.add_get("/api/groups/{group_id}/profile", self.handle_group_profile)
        app.router.add_get("/api/groups/{group_id}/policy", self.handle_group_policy)
        app.router.add_post("/api/groups/{group_id}/policy/reset", self.handle_policy_reset)
        app.router.add_get("/api/feedback", self.handle_feedback)
        app.router.add_get("/api/feedback/timeseries", self.handle_feedback_series)

        return app

    # ------------------------------------------------------------------
    # 认证
    # ------------------------------------------------------------------

    def _is_authed(self, request: web.Request) -> bool:
        if not self.cfg.web.token:
            return True
        session = request.cookies.get(COOKIE_NAME)
        if session and session in self._sessions:
            if time.time() - self._sessions[session] < SESSION_TTL:
                self._sessions[session] = time.time()
                return True
            self._sessions.pop(session, None)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and secrets.compare_digest(
            auth[7:].strip(), self.cfg.web.token
        ):
            return True
        return False

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        path = request.path
        if path in ("/", "/api/login", "/favicon.ico") or path.startswith("/static/"):
            return await handler(request)
        if not self._is_authed(request):
            return error("未认证", status=401)
        return await handler(request)

    # ------------------------------------------------------------------
    # 静态资源
    # ------------------------------------------------------------------

    async def handle_index(self, request: web.Request) -> web.Response:
        return await self._serve_file(STATIC_DIR / "index.html")

    async def handle_favicon(self, request: web.Request) -> web.Response:
        """浏览器总会来要 /favicon.ico，直接回同一个 SVG，省一次 404。"""
        icon = STATIC_DIR / "icon.svg"
        if icon.is_file():
            return web.Response(body=icon.read_bytes(),
                                content_type="image/svg+xml",
                                headers={"Cache-Control": "max-age=3600"})
        return web.Response(status=204)

    async def handle_static(self, request: web.Request) -> web.Response:
        name = request.match_info.get("name", "")
        target = (STATIC_DIR / name).resolve()
        # 目录穿越防护
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            return error("非法路径", status=403)
        return await self._serve_file(target)

    async def _serve_file(self, path: Path) -> web.Response:
        if not path.is_file():
            return error("文件不存在", status=404)
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        return web.Response(
            body=path.read_bytes(),
            content_type=content_type.split(";")[0],
            charset="utf-8" if "charset" in content_type else None,
            headers={"Cache-Control": "no-cache"},
        )

    # ------------------------------------------------------------------
    # 认证接口
    # ------------------------------------------------------------------

    async def handle_login(self, request: web.Request) -> web.Response:
        # 用 socket 对端地址，不信任 X-Forwarded-For —— 否则节流可被伪造绕过
        peer = request.remote or "unknown"
        now = time.time()

        locked_until = self._login_locked_until.get(peer, 0.0)
        if locked_until > now:
            log.warning("面板登录被节流（锁定中），来源 %s", peer)
            return error(
                f"尝试次数过多，请 {int(locked_until - now) + 1} 秒后再试",
                status=429,
            )

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")

        token = str((payload or {}).get("token") or "")
        expected = self.cfg.web.token

        if not expected:
            return error("服务端未设置 WEB_TOKEN，无需登录", status=400)
        if not secrets.compare_digest(token, expected):
            self._record_login_failure(peer, now)
            # 不回显任何关于令牌长度/前缀的信息
            log.warning("面板登录失败，来源 %s", peer)
            return error("访问令牌不正确", status=401)

        self._login_failures.pop(peer, None)
        self._login_locked_until.pop(peer, None)

        session = secrets.token_urlsafe(32)
        self._sessions[session] = time.time()
        response = web.Response(status=204)
        response.set_cookie(
            COOKIE_NAME, session,
            max_age=SESSION_TTL, httponly=True, samesite="Lax", path="/",
        )
        log.info("面板登录成功，来源 %s", peer)
        return response

    def _record_login_failure(self, peer: str, now: float) -> None:
        attempts = [t for t in self._login_failures.get(peer, []) if now - t < LOGIN_WINDOW]
        attempts.append(now)
        self._login_failures[peer] = attempts

        if len(attempts) >= LOGIN_MAX_FAILURES:
            self._login_locked_until[peer] = now + LOGIN_LOCKOUT
            self._login_failures[peer] = []
            log.warning(
                "面板登录连续失败 %d 次，锁定来源 %s 约 %.0f 秒",
                LOGIN_MAX_FAILURES, peer, LOGIN_LOCKOUT,
            )

        # 顺手清理过期记录，避免长期运行后字典无界增长
        if len(self._login_failures) > 500:
            cutoff = now - LOGIN_WINDOW
            self._login_failures = {
                ip: [t for t in times if t >= cutoff]
                for ip, times in self._login_failures.items()
            }
            self._login_failures = {ip: t for ip, t in self._login_failures.items() if t}
        if len(self._login_locked_until) > 500:
            self._login_locked_until = {
                ip: until for ip, until in self._login_locked_until.items() if until > now
            }

    async def handle_logout(self, request: web.Request) -> web.Response:
        session = request.cookies.get(COOKIE_NAME)
        if session:
            self._sessions.pop(session, None)
        response = web.Response(status=204)
        response.del_cookie(COOKIE_NAME, path="/")
        return response

    async def handle_me(self, request: web.Request) -> web.Response:
        if not self._is_authed(request):
            return error("未认证", status=401)
        return json_response({"authenticated": True, "version": VERSION})

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    async def handle_status(self, request: web.Request) -> web.Response:
        snapshot = await self.pipeline.control_snapshot()
        message_stats = await self.repo.message_stats()
        decision_stats = await self.repo.decision_stats()
        groups = await self.repo.list_groups()
        memory_stats = await self.pipeline.memory.stats()
        sticker_stats = await self.pipeline.stickers.stats()
        media_stats = await self.pipeline.media_repo.stats()
        group_id = None
        group_raw = request.query.get("group_id")
        if group_raw and group_raw.lstrip("-").isdigit():
            group_id = int(group_raw)
        feedback_stats = await self.pipeline.feedbacks.summary(group_id)

        stats = {
            **snapshot["stats"],
            **message_stats,
            **decision_stats,
            **memory_stats,
            **sticker_stats,
            **media_stats,
            "signals": feedback_stats["signals"],
            "positive_rate": round(feedback_stats["positive_rate"], 3),
            "queue_size": snapshot["queue_size"],
        }
        stats["spoke_rate"] = decision_stats["spoke_rate"]

        return json_response({
            "version": VERSION,
            "uptime_seconds": snapshot["uptime_seconds"],
            "started_at": snapshot["started_at"],
            "connected": self.gateway.connected.is_set(),
            "self_id": snapshot["self_id"],
            "nickname": self.pipeline.nickname,
            "paused": snapshot["paused"],
            "presence": snapshot["presence"]["presence"],
            "presence_detail": snapshot["presence"],
            "mood": snapshot["mood"],
            "debug": snapshot["debug"],
            "stats": stats,
            "groups_enabled": sum(1 for g in groups if g["enabled"]),
            "groups_total": len(groups),
            "llm_configured": self.cfg.llm.configured,
        })

    # ------------------------------------------------------------------
    # 群
    # ------------------------------------------------------------------

    async def handle_groups(self, request: web.Request) -> web.Response:
        return json_response(await self.repo.list_groups())

    async def handle_group_patch(self, request: web.Request) -> web.Response:
        try:
            group_id = int(request.match_info["group_id"])
        except (KeyError, ValueError):
            return error("群号非法")

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")

        if "enabled" not in (payload or {}):
            return error("缺少 enabled 字段")

        enabled = bool(payload["enabled"])
        await self.repo.set_group_enabled(group_id, enabled)
        # 刚打开发言权限 -> 先冷启动：读一遍群历史再决定怎么融进去。
        # 真人进新群也是先潜水看的，不能一开权限就开口。
        onboard = {}
        if enabled:
            try:
                onboard = await self.pipeline.on_group_enabled(group_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("启动冷启动失败 group=%s: %s", group_id, exc)
        group = await self.repo.get_group(group_id)
        if group is None:
            return error("群不存在", status=404)
        log.info("群 %s 已%s", group_id, "启用" if enabled else "关闭")
        body = dict(group)
        if onboard:
            body["onboard"] = onboard
        return json_response(body)

    async def handle_group_messages(self, request: web.Request) -> web.Response:
        try:
            group_id = int(request.match_info["group_id"])
        except (KeyError, ValueError):
            return error("群号非法")

        limit = min(200, max(1, int(request.query.get("limit", 50))))
        before_raw = request.query.get("before_id")
        before_id = int(before_raw) if before_raw and before_raw.isdigit() else None

        messages = await self.repo.paged_messages(group_id, limit, before_id)
        return json_response({
            "messages": [
                {
                    "message_id": m.message_id,
                    "user_id": m.user_id,
                    "nickname": m.nickname,
                    "text": m.text,
                    "ts": m.ts,
                    "is_self": m.is_self,
                    "mentions_self": m.mentions_self,
                }
                for m in messages
            ]
        })

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    async def handle_decisions(self, request: web.Request) -> web.Response:
        limit = min(200, max(1, int(request.query.get("limit", 50))))
        group_raw = request.query.get("group_id")
        group_id = int(group_raw) if group_raw and group_raw.lstrip("-").isdigit() else None

        decisions = await self.repo.list_decisions(limit, group_id)
        return json_response([
            {
                "id": d.id,
                "group_id": d.group_id,
                "ts": d.ts,
                "should_reply": d.should_reply,
                "priority": round(d.priority, 3),
                "reason": d.reason,
                "text": d.text,
                "draft": d.draft,
                "features": d.features,
            }
            for d in decisions
        ])

    # ------------------------------------------------------------------
    # 人设
    # ------------------------------------------------------------------

    async def _persona_context(self, group_id: int | None) -> dict:
        """当前人设 + 逐群全景，供面板画选择器和状态标记。"""
        default_path = self.cfg.persona_path
        default_raw = default_path.read_text(encoding="utf-8") if default_path.is_file() else ""

        custom_ids = await self.pipeline.personas.custom_ids()
        groups = []
        for g in await self.repo.list_groups():
            gid = int(g["group_id"])
            groups.append({
                "group_id": gid,
                "name": g["name"],
                "enabled": bool(g["enabled"]),
                "custom": gid in custom_ids,
            })
        # 有自定义人设但群列表里没有的（已退群）也列出来，别让人改不了
        known = {g["group_id"] for g in groups}
        for gid in custom_ids:
            if gid not in known:
                groups.append({"group_id": gid, "name": f"群 {gid}",
                               "enabled": False, "custom": True})

        if group_id is None:
            raw = default_raw
            scope = "default"
            has_custom = False
        else:
            raw = await self.pipeline.personas.get(group_id)
            has_custom = bool(raw.strip())
            if not has_custom:
                raw = default_raw
            scope = "group" if has_custom else "default"

        try:
            persona = parse_persona(raw)
            valid, err, summary = True, None, persona.summary()
        except PersonaError as exc:
            valid, err, summary = False, str(exc), None

        return {
            "raw": raw, "valid": valid, "error": err, "summary": summary,
            "scope": scope, "has_custom": has_custom,
            "group_id": group_id, "groups": groups,
            "custom_count": len(custom_ids),
            "default_raw": default_raw,
        }

    async def handle_persona_get(self, request: web.Request) -> web.Response:
        """取人设。带 group_id 时取该群生效的那份（没有自定义就返回默认）。"""
        raw_gid = request.query.get("group_id")
        group_id = int(raw_gid) if raw_gid and raw_gid.lstrip("-").isdigit() else None
        return json_response(await self._persona_context(group_id))

    async def handle_persona_validate(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")
        raw = str((payload or {}).get("raw") or "")
        try:
            persona = parse_persona(raw)
            return json_response({"valid": True, "error": None, "summary": persona.summary()})
        except PersonaError as exc:
            return json_response({"valid": False, "error": str(exc), "summary": None})

    async def handle_persona_put(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")
        payload = payload or {}
        raw = str(payload.get("raw") or "")
        gid_raw = payload.get("group_id")
        group_id = int(gid_raw) if isinstance(gid_raw, (int, str)) and str(gid_raw).lstrip("-").isdigit() else None

        try:
            persona = parse_persona(raw)
        except PersonaError as exc:
            return json_response({"valid": False, "error": str(exc), "summary": None},
                                 status=400)

        if group_id is None:
            # 全局默认：写文件，先备份
            path = self.cfg.persona_path
            if path.is_file():
                backup = path.with_suffix(path.suffix + ".bak")
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            path.write_text(raw, encoding="utf-8")
            try:
                self.pipeline.reload_persona()
            except PersonaError as exc:
                return error(f"人设已写入但重载失败: {exc}", status=500)
            log.info("全局人设已更新（%s）", persona.name)
        else:
            # 逐群覆盖：存库。**不碰文件**，这样默认人设始终是干净的一份。
            await self.pipeline.personas.set(group_id, raw)
            self.pipeline.invalidate_persona(group_id)
            self.pipeline.mark_prompt_stale(group_id)
            log.info("群 %s 的自定义人设已更新（%s）", group_id, persona.name)
        # 回完整上下文，面板不用再拉一次
        ctx = await self._persona_context(group_id)
        ctx["valid"] = True
        ctx["error"] = None
        return json_response(ctx)

    async def handle_persona_reset(self, request: web.Request) -> web.Response:
        """删掉某个群的自定义人设，回到跟随默认。"""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        gid = (payload or {}).get("group_id")
        if not str(gid).lstrip("-").isdigit():
            return error("缺少 group_id")
        group_id = int(gid)

        removed = await self.pipeline.personas.reset(group_id)
        self.pipeline.invalidate_persona(group_id)
        self.pipeline.mark_prompt_stale(group_id)
        log.info("群 %s 恢复默认人设（删除自定义=%s）", group_id, removed)
        return json_response(await self._persona_context(group_id))

    # ------------------------------------------------------------------
    # 提示词
    # ------------------------------------------------------------------

    async def _prompt_context(self, group_id: int) -> dict:
        """某群的提示词全景：生效值来自哪一层、两层各改了什么。"""
        repo = self.pipeline.prompt_repo
        effective = await repo.all(group_id)
        global_own = await repo.own(0)
        group_own = await repo.own(group_id) if group_id else {}

        parts = prompts.all_parts(effective)
        # 标出每一层改过哪些 —— 面板要能区分"全局改的"和"这个群改的"
        for item in parts:
            item["custom_global"] = item["key"] in global_own
            item["custom_group"] = item["key"] in group_own
            item["custom"] = bool(item["custom_global"] or item["custom_group"])

        groups = [{"group_id": int(g["group_id"]), "name": g["name"]}
                  for g in await self.repo.list_groups()]
        return {
            "parts": parts,
            "group_id": group_id,
            "custom_count": sum(1 for i in parts if i["custom"]),
            "custom_global": len(global_own),
            "custom_group": len(group_own),
            "groups": groups,
        }

    async def handle_prompts_get(self, request: web.Request) -> web.Response:
        """列出全部提示词条目：默认值、当前生效值、被哪一层改过。"""
        raw = request.query.get("group_id")
        group_id = int(raw) if raw and raw.lstrip("-").isdigit() else 0
        return json_response(await self._prompt_context(group_id))

    async def handle_prompts_put(self, request: web.Request) -> web.Response:
        """保存单条提示词。传空字符串等于恢复默认。"""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")
        payload = payload or {}
        key = str(payload.get("key") or "")
        text = str(payload.get("text") or "")
        gid_raw = payload.get("group_id")
        group_id = int(gid_raw) if str(gid_raw).lstrip("-").isdigit() else 0

        part = prompts.BY_KEY.get(key)
        if part is None:
            return error(f"未知的提示词条目: {key}", status=400)

        # 占位符写错要拦住 —— 否则模型会收到字面量 {max_chars}，
        # 而这种错误在面板上看不出来，只有模型表现变怪时才会发现。
        used = set(prompts.placeholder_names(text))
        unknown = used - set(part.names())
        if unknown:
            return error(
                "用了未声明的占位符 " + str(sorted(unknown))
                + "。这一条只支持 " + str(sorted(part.names()) or "没有占位符"),
                status=400,
            )

        if text.strip():
            await self.pipeline.prompt_repo.set(key, text, group_id)
        else:
            await self.pipeline.prompt_repo.reset(key, group_id)

        await self.pipeline.reload_prompts()
        where = f"群 {group_id}" if group_id else "全局"
        log.info("提示词 %s（%s）已更新（%d 字）", key, where, len(text))
        return json_response(await self._prompt_context(group_id))

    async def handle_prompts_reset(self, request: web.Request) -> web.Response:
        """恢复默认。传 key 恢复一条，传 all=true 恢复全部。"""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        payload = payload or {}

        gid_raw = payload.get("group_id")
        group_id = int(gid_raw) if str(gid_raw).lstrip("-").isdigit() else 0

        if payload.get("all"):
            if payload.get("everything"):
                n = await self.pipeline.prompt_repo.reset_all(None)
                log.info("全部提示词覆盖已清空（两层，共 %d 条）", n)
            else:
                n = await self.pipeline.prompt_repo.reset_all(group_id)
                log.info("提示词覆盖已恢复默认（层 %s，清了 %d 条）", group_id, n)
        else:
            key = str(payload.get("key") or "")
            if key not in prompts.BY_KEY:
                return error(f"未知的提示词条目: {key}", status=400)
            await self.pipeline.prompt_repo.reset(key, group_id)
            log.info("提示词 %s 已恢复默认（层 %s）", key, group_id)

        await self.pipeline.reload_prompts()
        return json_response(await self._prompt_context(group_id))

    async def handle_prompts_preview(self, request: web.Request) -> web.Response:
        """渲染某个群实际会收到的 system / user 提示词。

        这是这个页面最有用的部分：不用猜，直接看到模型收到什么。
        """
        raw_gid = request.query.get("group_id")
        group_id = int(raw_gid) if raw_gid and raw_gid.lstrip("-").isdigit() else 0
        if not group_id:
            rows = await self.repo.list_groups()
            group_id = int(rows[0]["group_id"]) if rows else 0
        if not group_id:
            return json_response({"system": "", "user": "",
                                  "error": "还没有群，先去群管理里启用一个"})

        persona = await self.pipeline.load_persona_for(group_id)
        moods = await self.pipeline.selector.mood_labels()
        system = build_system(persona, self.cfg.behavior,
                              self.pipeline.faces, sticker_moods=moods,
                              group_id=group_id)

        from ..models import Conversation, Message
        import time as _t
        msg = Message(group_id=group_id, message_id=0, user_id=0,
                      nickname="张三", text="小七在吗", ts=int(_t.time()))
        conv = Conversation(group_id=group_id, messages=[msg], trigger=msg)
        user = build_user(
            conv, persona,
            memories=["这个群主要聊原神和作业"],
            presence="在线", mood_text="普通", now_text="21:30",
            is_mention=True,
            profile_note="一个技术交流群，聊编程和游戏",
            sticker_moods=moods,
            group_id=group_id,
        )
        custom = bool(await self.pipeline.personas.get(group_id))
        return json_response({
            "group_id": group_id,
            "persona_name": persona.name,
            "persona_scope": "group" if custom else "default",
            "system": system, "user": user,
            "system_chars": len(system), "user_chars": len(user),
        })

    # ------------------------------------------------------------------
    # 新群冷启动
    # ------------------------------------------------------------------

    async def handle_onboard_get(self, request: web.Request) -> web.Response:
        """冷启动状态 + 存活账本。群档案页用。"""
        raw = request.query.get("group_id")
        if not raw or not raw.lstrip("-").isdigit():
            return error("缺少 group_id")
        group_id = int(raw)
        out = {"group_id": group_id}
        try:
            out["onboard"] = await self.pipeline.onboard_snapshot(group_id)
        except Exception as exc:  # noqa: BLE001
            out["onboard"] = {"status": "failed", "note": str(exc)[:120]}
        try:
            out["survival"] = await self.pipeline.survival.snapshot(group_id)
        except Exception as exc:  # noqa: BLE001
            out["survival"] = {"enabled": False, "note": str(exc)[:120]}
        return json_response(out)

    async def handle_onboard_restart(self, request: web.Request) -> web.Response:
        """重新跑一次冷启动分析（群变化了就重来）。"""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        gid = (payload or {}).get("group_id")
        if not str(gid).lstrip("-").isdigit():
            return error("缺少 group_id")
        group_id = int(gid)
        result = await self.pipeline.on_group_enabled(group_id, force=True)
        log.info("群 %s 重新冷启动: %s", group_id, result)
        return json_response({"group_id": group_id, **result})

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------

    async def handle_settings_get(self, request: web.Request) -> web.Response:
        return json_response(settings_snapshot(self.cfg))

    async def handle_settings_put(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")
        try:
            snapshot, changes = update_settings(self.cfg, payload or {})
        except ValueError as exc:
            return error(str(exc))

        if any(c.endswith(("max_chars_per_bubble", "max_bubbles")) for c in changes):
            self.pipeline.mark_prompt_stale()
        if any(c.startswith("behavior.") for c in changes):
            self.pipeline.mark_prompt_stale()

        if changes:
            log.info("设置已更新: %s", ", ".join(changes))
        return json_response(snapshot)

    async def handle_model_profiles(self, request: web.Request) -> web.Response:
        """模型画像清单 —— 面板用来显示"这个模型有什么脾气"。"""
        from ..generate.model_profiles import advise, all_profiles, profile_for

        cfg = self.cfg.llm
        reply = cfg.reply_model or cfg.model
        analysis = cfg.analysis_model or cfg.model
        return json_response({
            "profiles": all_profiles(),
            "current": {
                "model": cfg.model,
                "reply_model": reply,
                "analysis_model": analysis,
                "reply_profile": profile_for(reply).describe(),
                "analysis_profile": profile_for(analysis).describe(),
                "warnings": advise(reply, analysis),
            },
        })

    async def handle_model_apply(self, request: web.Request) -> web.Response:
        """按画像把推荐预算应用到当前模型。"""
        from ..generate.model_profiles import profile_for, recommended_overrides

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        payload = payload or {}
        target = str(payload.get("target") or "reply")   # reply / analysis / both

        cfg = self.cfg.llm
        reply = cfg.reply_model or cfg.model
        analysis = cfg.analysis_model or cfg.model
        patch: dict = {}

        if target in ("reply", "both"):
            rp = profile_for(reply)
            patch["reply_max_tokens"] = rp.reply_max_tokens
        if target in ("analysis", "both"):
            ap = profile_for(analysis)
            patch["analysis_max_tokens"] = ap.analysis_max_tokens
        if target == "both":
            # reasoning_effort 是全局单值，但两个档的模型未必都认它。
            # 谁认就以谁为准 —— **优先分析档**，因为深度思考对分析更有价值，
            # 而认这个参数的回复档模型本来就少。
            rp = profile_for(reply)
            ap = profile_for(analysis)
            if ap.effort_effective:
                patch["reasoning_effort"] = ap.reasoning_effort
            elif rp.effort_effective:
                patch["reasoning_effort"] = rp.reasoning_effort
            else:
                # 两个都不认，填什么都是摆设，填 low 最省事
                patch["reasoning_effort"] = "low"

        # update_settings 是同步函数，返回 (cfg, changes) —— 别 await
        _, changed = update_settings(self.cfg, {"llm": patch})
        return json_response({"applied": patch, "changed": changed,
                              "settings": settings_snapshot(self.cfg)})

    async def handle_bili_qr_start(self, request: web.Request) -> web.Response:
        """生成扫码登录二维码。"""
        snapshot = await self.pipeline.bili_qr_start()
        return json_response(snapshot)

    async def handle_bili_qr_poll(self, request: web.Request) -> web.Response:
        """轮询扫码状态。成功时服务端会当场把新 Cookie 装上并写进 .env。"""
        snapshot = await self.pipeline.bili_qr_poll()
        if snapshot.get("state") == "confirmed":
            snapshot["settings"] = settings_snapshot(self.cfg)
        return json_response(snapshot)

    async def handle_bili_status(self, request: web.Request) -> web.Response:
        """当前 B 站登录态 —— 面板据此决定要不要提示扫码。"""
        return json_response(await self.pipeline.bili_auth_state())

    async def handle_search_test(self, request: web.Request) -> web.Response:
        """面板的"测试搜索"按钮。走真实请求，好让用户确认 key 和字段对不对。"""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        payload = payload or {}

        query = str(payload.get("query") or "").strip() or "今天的新闻"
        cfg = self.cfg.search

        if not cfg.enabled:
            return json_response({"ok": False, "query": query, "results": [],
                                  "error": "搜索功能未启用"})
        if not cfg.api_key:
            return json_response({"ok": False, "query": query, "results": [],
                                  "error": "未配置 SEARCH_API_KEY"})

        # 配置层面的已知陷阱，先拦下来再说 —— 省得对着 404 干瞪眼
        warn = host_warning(cfg.provider, cfg.base_url)
        if warn:
            return json_response({"ok": False, "query": query, "results": [],
                                  "provider": cfg.provider, "error": warn})

        started = time.monotonic()
        results = await self.pipeline.search_client.search(query)
        elapsed = int((time.monotonic() - started) * 1000)

        if not results:
            hint = PROVIDER_HINTS.get(cfg.provider, "")
            base = ("没拿到结果 —— 检查 API Key 是否正确、base_url 是否可达。"
                    "具体 HTTP 状态见「日志」页。")
            return json_response({
                "ok": False, "query": query, "results": [], "latency_ms": elapsed,
                "provider": cfg.provider,
                "error": f"{base}\n\n{hint}" if hint else base,
            })
        return json_response({
            "ok": True, "query": query, "latency_ms": elapsed, "provider": cfg.provider,
            "hint": PROVIDER_HINTS.get(cfg.provider, ""),
            "results": [{"title": r.title, "url": r.url, "snippet": r.snippet[:200]}
                        for r in results],
            "error": None,
        })

    async def _llm_models_hint(self) -> str:
        """看看代理到底提供哪些模型，配置的模型在不在里面。

        起因：上游把 DeepSeek-V4.1-Flash 下线了，但代理配置里还写着，
        结果每次生成都是 503。测试按钮如果只说"503"，用户得翻代理配置才知道。
        """
        cfg = self.cfg.llm
        try:
            import aiohttp
            headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
            base = cfg.base_url.rstrip("/")
            url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            ) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return ""
                    data = json.loads(await resp.text())
        except Exception:  # noqa: BLE001
            return ""

        ids = [str(m.get("id")) for m in (data.get("data") or []) if isinstance(m, dict)]
        if not ids:
            return ""
        if cfg.model in ids:
            return ""
        # 给出最接近的几个，省得用户一个个试
        close = [i for i in ids if any(t in i.lower() for t in
                                       (cfg.model.lower()[:6], "flash", "chat"))][:6]
        return (f"代理里没有模型 {cfg.model!r}（上游可能已下线）。"
                f"当前可用 {len(ids)} 个"
                + (f"，相近的：{'、'.join(close)}" if close else "")
                + f"。去「设置 → LLM」换一个。")

    async def handle_llm_test(self, request: web.Request) -> web.Response:
        result = await self.pipeline.llm.test()
        # 失败时补一条"代理里有没有这个模型"的线索 —— 光看 503 很难定位
        if isinstance(result, dict) and not result.get("ok"):
            hint = await self._llm_models_hint()
            if hint:
                # 上游的 503 原文很长（几百字），会把有用的提示淹掉。
                # 有明确线索时只留原文开头，让人一眼看到"该换模型"。
                raw = str(result.get("error") or "")
                head = raw.split("{", 1)[0].strip() or "请求失败"
                result["error"] = f"{head} —— {hint}"
                result["raw_error"] = raw
                result["models_hint"] = hint
        return json_response(result)

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------

    async def handle_pause(self, request: web.Request) -> web.Response:
        self.pipeline.paused = True
        log.info("面板：已暂停发言")
        return json_response({"paused": True})

    async def handle_resume(self, request: web.Request) -> web.Response:
        self.pipeline.paused = False
        log.info("面板：已恢复发言")
        return json_response({"paused": False})

    async def handle_debug(self, request: web.Request) -> web.Response:
        """测试模式开关。忽略作息/额度/冷却，方便随时验证。"""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}

        payload = payload or {}
        patch: dict = {}
        if "enabled" in payload:
            patch["enabled"] = bool(payload["enabled"])
        if "respond_to_all" in payload:
            patch["respond_to_all"] = bool(payload["respond_to_all"])
        if not patch:
            return error("需要 enabled 或 respond_to_all")

        try:
            update_settings(self.cfg, {"debug": patch})
        except ValueError as exc:
            return error(str(exc))

        enabled = bool(self.cfg.debug.enabled)

        if enabled:
            log.warning(
                "测试模式已开启：作息/额度/群冷却/随机放弃全部失效，发言门槛降到 %.2f。"
                "测完记得关掉。", self.cfg.debug.relax_threshold,
            )
        else:
            log.info("测试模式已关闭，恢复正常作息与限流")
        if self.cfg.debug.respond_to_all and enabled:
            log.warning("测试模式：响应所有消息已开启 —— 它会接每一句话，注意会刷屏")
        return json_response({
            "enabled": enabled,
            "respond_to_all": bool(self.cfg.debug.respond_to_all),
            "relax_threshold": self.cfg.debug.relax_threshold,
        })

    async def handle_say(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")

        payload = payload or {}
        try:
            group_id = int(payload["group_id"])
        except (KeyError, TypeError, ValueError):
            return error("group_id 非法")

        text = str(payload.get("text") or "").strip()
        if not text:
            return error("内容不能为空")

        bubbles = self.pipeline.sender.split(text) or [text]
        sent = await self.pipeline.sender.send(group_id, Draft(messages=bubbles))
        if not sent:
            return error("发送失败，详见日志", status=502)
        return json_response({"ok": True, "message_id": sent[0], "count": len(sent)})

    async def handle_dry_run(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error("请求体不是 JSON")

        payload = payload or {}
        try:
            group_id = int(payload.get("group_id") or 0)
        except (TypeError, ValueError):
            return error("group_id 非法")

        text = str(payload.get("text") or "").strip()
        if not text:
            return error("测试内容不能为空")

        if not self.cfg.llm.configured:
            return json_response({
                "should_reply": False, "priority": 0.0,
                "reason": "LLM 未配置", "draft": [],
                "error": "请先在设置里填写 LLM base_url / model / API Key",
            })

        force = bool((payload or {}).get("force"))
        return json_response(await self.pipeline.dry_run(group_id, text, force=force))

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    async def handle_logs_recent(self, request: web.Request) -> web.Response:
        limit = min(1000, max(1, int(request.query.get("limit", 200))))
        return json_response({"logs": self.logs.recent(limit)})

    async def handle_logs_stream(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        queue = self.logs.subscribe()
        try:
            await response.write(b": connected\n\n")
            while True:
                entry = await queue.get()
                data = json.dumps(entry, ensure_ascii=False)
                await response.write(f"event: log\ndata: {data}\n\n".encode("utf-8"))
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception:  # noqa: BLE001
            log.debug("日志流中断", exc_info=True)
        finally:
            self.logs.unsubscribe(queue)

        return response

    # ------------------------------------------------------------------
    # 统计 / 记忆
    # ------------------------------------------------------------------

    async def handle_timeseries(self, request: web.Request) -> web.Response:
        hours = min(168, max(1, int(request.query.get("hours", 24))))
        return json_response({"points": await self.repo.timeseries(hours)})

    async def handle_memories(self, request: web.Request) -> web.Response:
        scope = request.query.get("scope") or None
        scope_raw = request.query.get("scope_id")
        scope_id = int(scope_raw) if scope_raw and scope_raw.lstrip("-").isdigit() else None
        limit = min(500, max(1, int(request.query.get("limit", 200))))

        include_invalid = request.query.get("include_invalid", "1") != "0"
        type_raw = request.query.get("type")

        if scope and scope_id is not None:
            rows = await self.pipeline.memories.all_for(scope, scope_id, limit)
        else:
            rows = await self.pipeline.memories.list_all(scope, limit)

        if not include_invalid:
            rows = [r for r in rows if not r.get("invalid_at")]
        if type_raw:
            rows = [r for r in rows if r.get("type") == type_raw]

        # 群列表供筛选器用。
        # **以"全部已知群"为准，而不是"有记忆的群"** —— 否则没记忆的群会从
        # 下拉框里消失，用户就看不到"这个群还没记住任何东西"这个信息。
        counts = {
            int(r["scope_id"]): r
            for r in await self.pipeline.memories.counts_by_scope("group")
        }
        groups: list[dict] = []
        for g in await self.repo.list_groups():
            gid = int(g["group_id"])
            c = counts.pop(gid, None)
            groups.append({
                "group_id": gid,
                "name": g["name"],
                "enabled": bool(g["enabled"]),
                "msg_count": int(g["msg_count"] or 0),
                "total": int((c or {}).get("total") or 0),
                "active": int((c or {}).get("active") or 0),
                "invalidated": int((c or {}).get("invalidated") or 0),
                "last_ts": int((c or {}).get("last_ts") or 0),
            })
        # 记忆里有、但群列表里没有的（群被退出/删除），也补上，别丢数据
        for gid, c in counts.items():
            groups.append({
                "group_id": gid, "name": f"群 {gid}", "enabled": False,
                "msg_count": 0,
                "total": int(c["total"] or 0), "active": int(c["active"] or 0),
                "invalidated": int(c["invalidated"] or 0),
                "last_ts": int(c["last_ts"] or 0),
            })
        # 启用中的、记忆多的、消息多的排前面
        groups.sort(key=lambda g: (not g["enabled"], -g["active"], -g["msg_count"]))

        return json_response({
            "memories": rows,
            "summary": {
                "total": await self.pipeline.memories.count(),
                "by_type": await self.pipeline.memories.type_counts(scope, scope_id),
                "invalidated": (await self.pipeline.memories.stats())["memories_invalidated"],
                "groups": groups,
                "capacity_per_group": self.cfg.evolution.memory_capacity_per_group,
                "ttl_by_type": TTL_BY_TYPE,
            },
        })

    async def handle_memory_delete(self, request: web.Request) -> web.Response:
        try:
            memory_id = int(request.match_info["memory_id"])
        except (KeyError, ValueError):
            return error("id 非法")
        ok = await self.pipeline.memories.delete(memory_id)
        if not ok:
            return error("记忆不存在", status=404)
        return web.Response(status=204)

    # ------------------------------------------------------------------
    # 表情包
    # ------------------------------------------------------------------

    async def handle_stickers(self, request: web.Request) -> web.Response:
        limit = min(500, max(1, int(request.query.get("limit", 200))))
        include_blocked = request.query.get("include_blocked") == "1"
        collected_only = request.query.get("collected_only", "1") != "0"
        rows = await self.pipeline.stickers.list_all(
            limit, include_blocked=include_blocked, collected_only=collected_only
        )
        return json_response({
            "stickers": [
                {
                    "id": r["id"],
                    "hash": r["hash"],
                    "ext": r["ext"],
                    "size": r["size"],
                    "collected": bool(r["collected"]),
                    "reason": r.get("collect_reason") or "",
                    "mood_tags": r["mood_tags"],
                    "content_tags": r["content_tags"],
                    "desc": r["desc"],
                    "seen_count": r["seen_count"],
                    "use_count": r["use_count"],
                    "win_count": r["win_count"],
                    "source_group": r["source_group"],
                    "collected_ts": r["collected_ts"],
                    "blocked": bool(r["blocked"]),
                    "url": f"/api/stickers/{r['id']}/file",
                }
                for r in rows
            ]
        })

    async def handle_sticker_stats(self, request: web.Request) -> web.Response:
        stats = await self.pipeline.stickers.stats()
        stats.update(await self.pipeline.media_repo.stats())
        stats.update(self.pipeline.collector.stats())
        stats.update(self.pipeline.selector.stats())
        stats["sticker_moods"] = await self.pipeline.selector.mood_labels()
        return json_response(stats)

    async def handle_faces(self, request: web.Request) -> web.Response:
        cat = self.pipeline.faces
        stats = cat.stats()
        return json_response({
            "stats": stats,
            "core": [{"id": f.id, "name": f.name} for f in cat.core()],
            "all": [{"id": f.id, "name": f.name} for f in cat.all_faces()],
        })

    async def handle_sticker_file(self, request: web.Request) -> web.Response:
        try:
            sticker_id = int(request.match_info["sticker_id"])
        except (KeyError, ValueError):
            return error("id 非法")
        row = await self.pipeline.stickers.get(sticker_id)
        if not row or not row.get("path"):
            return error("表情不存在", status=404)
        path = Path(row["path"])
        if not path.is_file():
            return error("表情文件已丢失", status=404)
        suffix = path.suffix.lower()
        ctype = {".gif": "image/gif", ".png": "image/png",
                 ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".webp": "image/webp"}.get(suffix, "application/octet-stream")
        return web.Response(body=path.read_bytes(), content_type=ctype,
                            headers={"Cache-Control": "max-age=3600"})

    async def handle_sticker_block(self, request: web.Request) -> web.Response:
        try:
            sticker_id = int(request.match_info["sticker_id"])
        except (KeyError, ValueError):
            return error("id 非法")
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        blocked = bool((payload or {}).get("blocked", True))
        await self.pipeline.stickers.set_blocked(sticker_id, blocked)
        log.info("表情 #%s 已%s", sticker_id, "封禁" if blocked else "解禁")
        return json_response({"id": sticker_id, "blocked": blocked})

    async def handle_sticker_delete(self, request: web.Request) -> web.Response:
        try:
            sticker_id = int(request.match_info["sticker_id"])
        except (KeyError, ValueError):
            return error("id 非法")
        if not await self.pipeline.stickers.delete(sticker_id):
            return error("表情不存在", status=404)
        return web.Response(status=204)

    # ------------------------------------------------------------------
    # 群档案与自进化
    # ------------------------------------------------------------------

    async def handle_group_profile(self, request: web.Request) -> web.Response:
        try:
            group_id = int(request.match_info["group_id"])
        except (KeyError, ValueError):
            return error("群号非法")
        profile = await self.pipeline.profiles.get(group_id)
        data = profile.summary()
        data["hourly"] = profile.hourly
        data["sample_size"] = profile.sample_size
        data["updated_ts"] = profile.updated_ts
        return json_response(data)

    async def handle_group_policy(self, request: web.Request) -> web.Response:
        try:
            group_id = int(request.match_info["group_id"])
        except (KeyError, ValueError):
            return error("群号非法")
        policy = await self.pipeline.policies.get(group_id)
        return json_response(policy.snapshot())

    async def handle_policy_reset(self, request: web.Request) -> web.Response:
        try:
            group_id = int(request.match_info["group_id"])
        except (KeyError, ValueError):
            return error("群号非法")
        policy = await self.pipeline.policies.get(group_id)
        policy.reset_learned()
        await self.pipeline.policies.save(policy)
        log.info("群 %s 的策略已重置", group_id)
        return json_response(policy.snapshot())

    async def handle_feedback(self, request: web.Request) -> web.Response:
        group_raw = request.query.get("group_id")
        group_id = int(group_raw) if group_raw and group_raw.lstrip("-").isdigit() else None
        limit = min(300, max(1, int(request.query.get("limit", 100))))
        summary = await self.pipeline.feedbacks.summary(group_id)
        events = await self.pipeline.feedbacks.recent(group_id, limit)
        return json_response({
            "summary": summary,
            "events": [
                {
                    "id": e["id"],
                    "group_id": e["group_id"],
                    "ts": e["sent_ts"],
                    "kind": e["kind"],
                    "weight": e["weight"],
                    "detail": e["detail"],
                }
                for e in events
            ],
        })

    async def handle_feedback_series(self, request: web.Request) -> web.Response:
        group_raw = request.query.get("group_id")
        group_id = int(group_raw) if group_raw and group_raw.lstrip("-").isdigit() else None
        days = min(60, max(1, int(request.query.get("days", 14))))
        return json_response({"points": await self.pipeline.feedbacks.timeseries(group_id, days)})

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self.cfg.web.enabled:
            log.info("管理面板已禁用（config.toml 的 [web].enabled）")
            return

        self._runner = web.AppRunner(self.app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.cfg.web.host, self.cfg.web.port)
        await site.start()

        self.logs.bind_loop(asyncio.get_running_loop())

        exposed = self.cfg.web.host not in ("127.0.0.1", "localhost", "::1")

        if not self.cfg.web.token:
            log.warning(
                "管理面板未设置 WEB_TOKEN，任何人访问 %s:%s 都能操作！"
                "请在 .env 里设置 WEB_TOKEN",
                self.cfg.web.host, self.cfg.web.port,
            )
        elif exposed:
            log.warning(
                "管理面板对外开放（%s:%s），仅靠 WEB_TOKEN 认证，"
                "且是明文 HTTP —— 令牌在局域网内可被嗅探。"
                "不需要时请把 [web].host 改回 127.0.0.1",
                self.cfg.web.host, self.cfg.web.port,
            )

        log.info("管理面板: http://%s:%s", self.cfg.web.host, self.cfg.web.port)
        if exposed:
            for url in self._lan_urls():
                log.info("  局域网访问: %s", url)

    def _lan_urls(self) -> list[str]:
        """列出可通过局域网访问的地址，省得用户自己去查 IP。"""
        import socket

        found: list[str] = []

        # 主机名解析（可能只拿到 127.0.1.1，所以不够）
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                found.append(info[4][0])
        except OSError:
            pass

        # 主网卡出口 IP：UDP connect 不会真的发包，但内核会选好路由和源地址
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("8.8.8.8", 80))
                found.append(probe.getsockname()[0])
            finally:
                probe.close()
        except OSError:
            pass

        urls: list[str] = []
        for ip in found:
            if ip.startswith("127."):
                continue
            url = f"http://{ip}:{self.cfg.web.port}"
            if url not in urls:
                urls.append(url)
        return urls

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def set_nickname(self, nickname: str) -> None:
        self.nickname = nickname
