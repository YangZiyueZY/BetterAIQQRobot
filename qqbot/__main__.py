"""入口。

  python -m qqbot          常驻运行：网关 + 全部管线 + 管理面板
  python -m qqbot check    连一次 NapCat，打印账号和群列表后退出
  python -m qqbot stats    打印数据库统计
  python -m qqbot web      只起管理面板（不连 QQ，离线调前端用）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from .config import VERSION, Config, load_config
from .gateway import Gateway, OneBotAPI
from .instance import AlreadyRunning, ensure_single_instance
from .log import setup_logging
from .pipeline import Pipeline
from .store import Database, Repo
from .web import RingLogHandler, WebServer

log = logging.getLogger("qqbot")

TICK_INTERVAL = 60.0


async def _build(cfg: Config) -> tuple[Database, Repo, Pipeline, Gateway, RingLogHandler, WebServer]:
    """装配所有组件。run / web 两个子命令共用。"""
    ring = RingLogHandler()
    setup_logging(cfg.log_level, ring)

    db = await Database.open(cfg.storage.db_path)
    api = OneBotAPI(cfg.onebot.request_timeout)
    repo = Repo(db)
    pipeline = Pipeline(cfg, repo, api)
    gateway = Gateway(cfg.onebot, api, pipeline.handle_message, pipeline.on_ready)
    web_server = WebServer(cfg, repo, pipeline, gateway, ring)
    return db, repo, pipeline, gateway, ring, web_server


async def _ticker(pipeline: Pipeline) -> None:
    while True:
        await asyncio.sleep(TICK_INTERVAL)
        try:
            await pipeline.on_tick()
        except Exception:  # noqa: BLE001
            log.exception("定时任务失败")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


async def cmd_run() -> int:
    cfg = load_config()

    # 单实例保护：两个进程连同一个 NapCat 端点 = 群里一句话回两遍。
    # 这里真出过事 —— 三个历次会话遗留的实例同时在跑，其中两个共用真实端点和真实库。
    try:
        ensure_single_instance(cfg.onebot.ws_url, cfg.root)
    except AlreadyRunning as exc:
        # 只打一处：既 log.error 又 print 会让同一条拒绝信息出现两遍
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    db, repo, pipeline, gateway, ring, web_server = await _build(cfg)

    # 把 DB 里的提示词覆盖层载入进程缓存。
    # 不在这里做的话，只有 on_ready（NapCat 连上）之后才生效 ——
    # 而面板离线调 prompt 的场景恰恰不需要连 QQ。
    await pipeline.reload_prompts()

    if not cfg.llm.configured:
        log.warning("LLM 未配置（.env 里的 LLM_API_KEY），将只记录消息不发言")
    if not cfg.onebot.token:
        log.warning("未配置 ONEBOT_TOKEN，若 NapCat 侧设了 token 将连接失败")

    await web_server.start()

    groups = await repo.list_groups()
    enabled = [g["group_id"] for g in groups if g["enabled"]]
    log.info(
        "启动完成 v%s｜已启用群: %s｜在管理面板里配置人设与开关",
        VERSION, enabled or "无（面板或 config.toml 里启用）",
    )

    tick = asyncio.create_task(_ticker(pipeline))
    exit_code = 0
    try:
        await gateway.run()
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001
        log.exception("主循环异常退出")
        exit_code = 1
    finally:
        tick.cancel()
        await web_server.stop()
        await pipeline.pool.close()
        await pipeline.llm.close()
        purged = await repo.purge_old_messages(cfg.storage.retention_days)
        if purged:
            log.info("清理过期消息 %d 条", purged)
        await db.close()
        log.info(
            "统计: 收 %d / 发 %d / LLM %d 次",
            pipeline.stats["messages_received"],
            pipeline.sender.sent_count,
            pipeline.llm.stats.calls,
        )
    return exit_code


# --------------------------------------------------------------------------
# web（离线调面板）
# --------------------------------------------------------------------------


async def cmd_web() -> int:
    cfg = load_config()
    db, repo, pipeline, gateway, ring, web_server = await _build(cfg)

    # 把 DB 里的提示词覆盖层载入进程缓存。
    # 不在这里做的话，只有 on_ready（NapCat 连上）之后才生效 ——
    # 而面板离线调 prompt 的场景恰恰不需要连 QQ。
    await pipeline.reload_prompts()

    # 造一点假数据，方便在没有 QQ 连接时调前端。
    # 一律虚构 —— 这里的数据会出现在面板截图里。
    await repo.upsert_group(100001, "示例技术交流群", default_enabled=True)
    await repo.upsert_group(100002, "示例摸鱼群")
    await repo.upsert_member(100001, 10001, "张三")
    await repo.upsert_member(100001, 10002, "李四")
    from .models import Message
    import time as _t

    now = int(_t.time())
    for i in range(30):
        await repo.save_message(Message(
            group_id=100001,
            message_id=900000 + i,
            user_id=10001 + (i % 2),
            nickname="张三" if i % 2 == 0 else "李四",
            text=f"离线演示消息 {i + 1}",
            ts=now - (30 - i) * 90,
            is_self=(i % 7 == 0),
        ))

    await web_server.start()
    log.info("离线面板模式：未连接 QQ，数据为演示数据。Ctrl-C 退出")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await web_server.stop()
        await pipeline.llm.close()
        await db.close()
    return 0


# --------------------------------------------------------------------------
# check / stats
# --------------------------------------------------------------------------


async def cmd_check() -> int:
    cfg = load_config()
    setup_logging("DEBUG")
    api = OneBotAPI(cfg.onebot.request_timeout)

    async def on_message(_msg):
        return

    gateway = Gateway(cfg.onebot, api, on_message)
    task = asyncio.create_task(gateway.run())

    try:
        try:
            await asyncio.wait_for(_wait_connected(gateway), timeout=10)
        except asyncio.TimeoutError:
            log.error(
                "无法连接 %s —— 检查 NapCat websocketServers 是否开启、token 是否一致",
                cfg.onebot.ws_url,
            )
            return 1

        log.info("WebSocket 已连通: %s", cfg.onebot.ws_url)

        try:
            info = await api.get_login_info()
            log.info("登录账号: %s (%s)", info.get("nickname"), info.get("user_id"))
        except Exception as exc:  # noqa: BLE001
            log.error("get_login_info 失败: %s", exc)
            return 1

        try:
            groups = await api.get_group_list()
        except Exception as exc:  # noqa: BLE001
            log.error("get_group_list 失败: %s", exc)
            return 1

        log.info("加入的群共 %d 个:", len(groups))
        for g in groups:
            log.info("  %-12s %s", g.get("group_id"), g.get("group_name"))

        await asyncio.sleep(3)
        log.info("自检通过。")
        return 0
    finally:
        await gateway.close()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


async def _wait_connected(gateway: Gateway) -> None:
    while not gateway.connected.is_set():
        await asyncio.sleep(0.1)


async def cmd_stats() -> int:
    cfg = load_config()
    setup_logging("WARNING")
    db = await Database.open(cfg.storage.db_path)
    repo = Repo(db)
    try:
        message_stats = await repo.message_stats()
        decision_stats = await repo.decision_stats()
        groups = await repo.list_groups()

        print(f"QQAIRobot v{VERSION}")
        print(f"  消息总数   {message_stats['messages_total']}")
        print(f"  收到       {message_stats['messages_received']}")
        print(f"  发出       {message_stats['messages_sent']}")
        print(f"  决策次数   {decision_stats['decisions']}")
        print(f"  发言率     {decision_stats['spoke_rate'] * 100:.1f}%")
        print(f"  记忆条数   {await repo.memory_count()}")
        print("  群:")
        for g in groups:
            flag = "启用" if g["enabled"] else "关闭"
            print(f"    [{flag}] {g['group_id']:<12} {g['name']:<20} {g['msg_count']} 条")
    finally:
        await db.close()
    return 0


def _install_signal_valve() -> None:
    """第一次信号走优雅退出，第二次立刻强退。

    实测踩到：进程事件循环卡死时，SIGINT 的 handler 根本轮不到执行 ——
    优雅退出那条路依赖循环还活着，循环一死，`kill -INT` 和 systemd 的
    `KillSignal=SIGINT` 全部失效，只能等 `TimeoutStopSec` 到点再 SIGKILL。
    留个逃生口：再按一次 Ctrl-C 就直接 `os._exit`，不用等那 20 秒。
    """
    state = {"n": 0}

    def _handle(signum, frame) -> None:  # noqa: ARG001
        state["n"] += 1
        if state["n"] >= 2:
            try:
                # 注意：bytes 字面量里不能写中文，必须显式 encode
                os.write(sys.stderr.fileno(), "\n第二次信号：强制退出\n".encode("utf-8"))
            except OSError:
                pass
            os._exit(130)
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):
            pass        # 非主线程/不支持时别把启动搞崩


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qqbot", description="QQ 拟人角色机器人")
    parser.add_argument(
        "cmd", nargs="?", default="run",
        choices=["run", "check", "stats", "web"],
        help="run=常驻 | check=连通性自检 | stats=数据统计 | web=离线面板",
    )
    parser.add_argument("--version", action="version", version=f"QQAIRobot {VERSION}")
    args = parser.parse_args(argv)

    handler = {
        "run": cmd_run,
        "check": cmd_check,
        "stats": cmd_stats,
        "web": cmd_web,
    }[args.cmd]

    _install_signal_valve()

    try:
        return asyncio.run(handler())
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
