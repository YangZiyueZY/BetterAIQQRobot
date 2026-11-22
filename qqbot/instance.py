"""单实例锁 —— 防的是"同一个 NapCat 端点被两个进程同时连"。

为什么需要：这个项目跑过 3 个实例（都是历次 AI 会话用后台任务拉起来后没人收），
其中两个连的是**同一个真实 NapCat + 同一个真实库**。当时没出事故纯属运气 ——
卡死的那个不处理消息，所以没双份回复；可它一旦随 NapCat 重启而重连，
群里就会开始一句话回两遍。代码里原本**没有任何单实例保护**。

设计取舍：

* **锁按 OneBot WS 端点哈希，不按机器、也不按 DB。**
  双发的根因是"两个连接抢同一份事件流"，所以锁的粒度就该落在端点上。
  演示实例按约定把 `ONEBOT_WS_URL` 指到不可达地址（见 AGENTS.md），
  端点不同 → 锁不同 → 照样能并行跑，不会被误伤。
* **用 flock 而不是 pidfile。** flock 由内核在 fd 关闭/进程死亡时自动释放，
  所以崩溃、`kill -9`、断电都不会留下需要人工清理的陈旧锁；
  pidfile 做不到（PID 会被复用，还会留脏文件）。
* 锁文件里写 PID 只为了**报错时能说清是谁占着**，不参与判定。
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import os
from pathlib import Path

#: 进程生命周期内必须一直持有 fd，否则锁立刻失效 —— 所以存着不放
_held: list[tuple[Path, int]] = []


class AlreadyRunning(RuntimeError):
    """同一个 OneBot 端点已经有别的实例在跑。"""


def lock_path_for(ws_url: str, root: Path | str = ".") -> Path:
    """按端点算锁文件路径。

    **必须转成绝对路径**：root 传成空串或相对路径时，两个从不同 CWD 启动的进程
    会各自算出不同的锁文件，互斥就静默失效了 —— 锁最怕的不是抢不到，是"以为锁住了"。
    """
    digest = hashlib.sha1((ws_url or "").strip().lower().encode("utf-8")).hexdigest()[:12]
    base = Path(str(root).strip() or ".")
    d = (base / "data" / "instances").expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.lock"


def held_locks() -> list[Path]:
    """当前进程持有的锁文件（给测试和面板用）。"""
    return [p for p, _ in _held]


def ensure_single_instance(ws_url: str, root: Path | str = ".") -> Path:
    """抢这把端点锁；抢不到就抛 `AlreadyRunning`。

    成功时返回锁文件路径，并把 fd 一直留在 `_held` 里 ——
    故意不关，关了锁就没了。
    """
    path = lock_path_for(ws_url, root)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        holder = ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            holder = lines[0] if lines else ""
        except OSError:
            holder = ""
        raise AlreadyRunning(
            f"已有一个实例连着 {ws_url}（PID {holder or '?'}）在跑，"
            "同时开两个会让群里收到双份回复。"
            f"先停掉它，或把本实例的 ONEBOT_WS_URL 指到别处。锁文件：{path}"
        ) from exc

    # 拿到锁了：写入自己的 PID 供报错时展示
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n{ws_url}\n".encode("utf-8"))
        os.fsync(fd)
    except OSError:
        pass          # 写不进去也不影响锁本身
    _held.append((path, fd))
    return path


def release_all() -> None:
    """显式释放（正常退出用；不释放也会被内核回收）。"""
    while _held:
        _, fd = _held.pop()
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
