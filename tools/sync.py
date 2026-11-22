"""安全同步：扫描凭据 -> 检查敏感标识 -> 跑测试 -> 提交 -> 推送。

**为什么要脚本而不是手敲 git**：这个项目已经因为"顺手 git add ."泄漏过一次
真实群名和 QQ 号，最后只能删库重建才清干净。靠自觉不可靠，靠流程才可靠。

用法：
    ./.venv/bin/python -m tools.sync -m "fix: 修了 xxx"
    ./.venv/bin/python -m tools.sync -m "docs: ..." --no-test
    ./.venv/bin/python -m tools.sync -m "release 1.2.0" --tag v1.2.0

任何一步发现问题就中止，不带病推送。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: 必须被 .gitignore 排除的路径（**精确匹配**，别用前缀 —— `.env` 会误伤 `.env.example`）
FORBIDDEN_TRACKED = (".env", ".venv/", ".backups/", ".git/")

#: 允许被跟踪的例外
ALLOWED_TRACKED = ("data/.gitkeep", ".env.example")

#: 这些环境变量**不是**凭据，扫描时跳过（地址、端口、路径这类）
NON_SECRET_KEYS = (
    "ONEBOT_WS_URL", "QQBOT_DB_PATH", "QQBOT_WEB_PORT", "LOG_LEVEL", "WEB_HOST",
)

#: 已知敏感标识的清单 —— **存在被 gitignore 的本地文件里**。
#:
#: 刻意不写进这个脚本：这份清单本身就是真实群号/昵称的集合，
#: 放进仓库等于把要防的东西又泄露了一遍。
#: 首次使用时从 .sensitive_patterns.example 复制并填入自己的。
SENSITIVE_FILE = ROOT / ".sensitive_patterns"

#: 凭据值太短的不值得搜（会满屏误报）
MIN_SECRET_LEN = 8

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def sh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          check=False if not check else True)


def out(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def say(icon: str, msg: str) -> None:
    print(f"  {icon} {msg}", flush=True)


def tracked_files() -> list[str]:
    return [f for f in out("git", "ls-files").splitlines() if f]


def load_env_secrets() -> dict[str, str]:
    """从 .env 读所有凭据值。只管长度够的，短的会满屏误报。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    secrets: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key in NON_SECRET_KEYS:      # 地址/端口/路径不算凭据
            continue
        if len(val) >= MIN_SECRET_LEN:
            secrets[key] = val
    # Cookie 里每个字段单独当凭据查（整串太长，切片查更准）
    cookie = secrets.get("BILIBILI_COOKIE", "")
    for part in cookie.split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            if len(v.strip()) >= MIN_SECRET_LEN:
                secrets[f"cookie:{k.strip()}"] = v.strip()
    return secrets


# ---- 各项检查 ---------------------------------------------------------------


def check_ignored() -> list[str]:
    """敏感路径必须没被跟踪。"""
    problems = []
    files = tracked_files()
    for bad in FORBIDDEN_TRACKED:
        hits = [f for f in files
                if (f == bad or f.startswith(bad)) and f not in ALLOWED_TRACKED]
        if hits:
            problems.append(f"{bad} 被跟踪了：{hits[:3]}")
    # data/ 只允许 .gitkeep
    stray = [f for f in files if f.startswith("data/") and f not in ALLOWED_TRACKED]
    if stray:
        problems.append(f"data/ 下有非 .gitkeep 文件被跟踪：{stray[:3]}")
    return problems


def load_sensitive_patterns() -> tuple[str, ...]:
    """读本地敏感标识清单。没有就返回空（会在输出里提示）。"""
    if not SENSITIVE_FILE.exists():
        return ()
    out_ = []
    for line in SENSITIVE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out_.append(line)
    return tuple(out_)


def check_secrets_in_files() -> list[str]:
    """拿 .env 里的真实凭据值，逐个在**跟踪文件**里搜。"""
    secrets = load_env_secrets()
    if not secrets:
        return ["读不到 .env（或里面没有足够长的凭据值），无法完成扫描"]

    files = [ROOT / f for f in tracked_files()]
    problems = []
    for key, val in secrets.items():
        for path in files:
            if not path.is_file():
                continue
            try:
                if val in path.read_text(encoding="utf-8", errors="ignore"):
                    rel = path.relative_to(ROOT)
                    problems.append(f"{key} 的值出现在 {rel}")
            except OSError:
                continue
    say("✓" if not problems else "✗",
        f"扫描 {len(secrets)} 个凭据值 × {len(files)} 个跟踪文件")
    return problems


def check_sensitive_patterns() -> list[str]:
    """已知真实标识的回归护栏（清单来自本地文件）。"""
    patterns = load_sensitive_patterns()
    if not patterns:
        say("·", f"没有 {SENSITIVE_FILE.name}，跳过该检查"
                 f"（可从 .sensitive_patterns.example 复制）")
        return []
    files = tracked_files()
    problems = []
    for pat in patterns:
        for f in files:
            path = ROOT / f
            if not path.is_file():
                continue
            try:
                if pat in path.read_text(encoding="utf-8", errors="ignore"):
                    problems.append(f"敏感标识 {pat!r} 出现在 {f}")
            except OSError:
                continue
    say("✓" if not problems else "✗", f"扫描 {len(patterns)} 个敏感标识")
    return problems


def check_demo_data() -> list[str]:
    """演示库里不能有真实数据 —— 截图的数据源就是它。"""
    patterns = load_sensitive_patterns()
    if not patterns:
        return []
    problems = []
    for py in ("tools/seed_demo.py", "qqbot/__main__.py"):
        path = ROOT / py
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            if pat in text:
                problems.append(f"{py} 里有敏感标识 {pat!r}")
    say("✓" if not problems else "✗", "演示数据源检查")
    return problems


def check_screenshots_reviewed(staged: list[str], assume_yes: bool) -> bool:
    """截图变了就必须人工过一眼 —— 它是渲染结果，脚本查不出画面里有什么。"""
    shots = [f for f in staged if f.startswith("docs/screenshots/")]
    if not shots:
        return True
    print()
    print(f"  {YELLOW}⚠ 本次改动了 {len(shots)} 张截图：{RESET}")
    for s in shots:
        kb = (ROOT / s).stat().st_size // 1024 if (ROOT / s).exists() else 0
        print(f"      {s}  ({kb}KB)")
    print()
    print(f"  {YELLOW}截图是渲染结果，脚本查不出画面里有没有真实信息。{RESET}")
    print(f"  {YELLOW}请先逐张看过：群名 / 昵称 / QQ 号 / 机器人昵称 / 消息内容。{RESET}")
    print()
    if assume_yes:
        say("!", "已跳过确认（--yes）")
        return True
    try:
        ans = input("  确认全部核对过、没有真实信息？(输入 yes 继续): ").strip().lower()
    except EOFError:
        return False
    return ans in ("yes", "y")


def run_tests() -> list[str]:
    problems = []
    for mod in ("tests.test_qqbot", "tests.test_v11", "tests.test_ingest"):
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT,
                           capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()
        last = tail[-1].strip() if tail else ""
        if r.returncode != 0 or "全部通过" not in (r.stdout or ""):
            problems.append(f"{mod} 未通过：{last[:80]}")
            say("✗", f"{mod} —— {last[:60]}")
        else:
            say("✓", f"{mod} —— {last}")
    return problems


# ---- 推送 -------------------------------------------------------------------


def push(remote: str, branch: str, tag: str | None, dry: bool) -> None:
    token = os.environ.get("CNB_TOKEN") or _env_value("CNB_TOKEN")
    if not token:
        print(f"  {YELLOW}没找到 CNB_TOKEN（应放在 .env 里），推送可能会要求交互式输入{RESET}")

    env = dict(os.environ)
    if token:
        env["CNB_TOKEN"] = token
    helper = '!f(){ echo username=cnb; echo "password=$CNB_TOKEN"; };f'

    cmd = ["git", "-c", f"credential.helper={helper}",
           "-c", "http.version=HTTP/1.1", "push", remote, branch]
    if dry:
        say("·", f"[dry-run] {' '.join(cmd)}")
        return
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"  {RED}推送失败：{r.stderr.strip()[:300]}{RESET}")
        sys.exit(1)
    say("✓", f"已推送到 {remote}/{branch}")

    if tag:
        cmd = ["git", "-c", f"credential.helper={helper}",
               "-c", "http.version=HTTP/1.1", "push", remote, tag]
        if dry:
            say("·", f"[dry-run] {' '.join(cmd)}")
            return
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"  {RED}标签推送失败：{r.stderr.strip()[:200]}{RESET}")
            sys.exit(1)
        say("✓", f"已推送标签 {tag}")


def _env_value(key: str) -> str:
    path = ROOT / ".env"
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.partition("=")[2].strip()
    return ""


# ---- 主流程 -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="安全同步到远端仓库")
    ap.add_argument("-m", "--message", help="提交信息")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--tag", help="顺便打并推送这个标签，如 v1.2.0")
    ap.add_argument("--no-test", action="store_true", help="跳过测试（不推荐）")
    ap.add_argument("--yes", action="store_true", help="跳过截图确认（已亲自看过时用）")
    ap.add_argument("--dry-run", action="store_true", help="只检查，不提交不推送")
    args = ap.parse_args()

    print(f"\n{BOLD}安全同步检查{RESET}  {ROOT}\n")

    # --- 1. 暂存改动 ---
    status = out("git", "status", "--porcelain")
    if not status.strip():
        say("·", "工作区干净，没有要提交的改动")
        return 0
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    staged = [f for f in out("git", "diff", "--cached", "--name-only").splitlines() if f]
    say("·", f"暂存 {len(staged)} 个文件")

    # --- 2. 检查 ---
    problems: list[str] = []
    problems += check_ignored()
    problems += check_secrets_in_files()
    problems += check_sensitive_patterns()
    problems += check_demo_data()

    if problems:
        print(f"\n{RED}{BOLD}✗ 检查未通过，已中止（没有提交、没有推送）{RESET}")
        for p in problems:
            print(f"    {RED}· {p}{RESET}")
        print(f"\n  {YELLOW}恢复暂存：git reset{RESET}")
        return 1

    # --- 3. 截图确认 ---
    if not check_screenshots_reviewed(staged, args.yes):
        print(f"\n{YELLOW}已中止。核对完截图后重跑，或加 --yes。{RESET}")
        subprocess.run(["git", "reset"], cwd=ROOT, check=False)
        return 1

    # --- 4. 测试 ---
    if not args.no_test:
        print()
        problems = run_tests()
        if problems:
            print(f"\n{RED}{BOLD}✗ 测试未通过，已中止{RESET}")
            for p in problems:
                print(f"    {RED}· {p}{RESET}")
            subprocess.run(["git", "reset"], cwd=ROOT, check=False)
            return 1

    # --- 5. 提交 ---
    message = args.message
    if not message:
        if args.dry_run:
            message = "(dry-run)"
        else:
            print(f"\n  {YELLOW}缺少 -m 提交信息{RESET}")
            print(f"    例：-m \"fix: 修了 xxx\"")
            subprocess.run(["git", "reset"], cwd=ROOT, check=False)
            return 1
    if not args.dry_run:
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, check=True)
        sha = out("git", "rev-parse", "--short", "HEAD").strip()
        say("✓", f"已提交 {sha}")

    # --- 6. 标签 ---
    if args.tag:
        if args.dry_run:
            say("·", f"[dry-run] git tag -a {args.tag}")
        else:
            exists = out("git", "tag", "-l", args.tag).strip()
            if exists:
                say("·", f"标签 {args.tag} 已存在，跳过创建")
            else:
                subprocess.run(["git", "tag", "-a", args.tag,
                                "-m", message or args.tag], cwd=ROOT, check=True)
                say("✓", f"已打标签 {args.tag}")

    # --- 7. 推送 ---
    print()
    push(args.remote, args.branch, args.tag, args.dry_run)

    print(f"\n{GREEN}{BOLD}✓ 同步完成{RESET}")
    print(f"    你配置的远端仓库\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())