# AGENTS.md

给 AI 助手的工作约定。**改动这个项目前先读这里。**

---

## 铁律：每次更新都要同步远端，但绝不能泄密

远端仓库由使用者自行配置（`origin`）。公开发布版本不含任何私有仓库地址。

### 同步方式

**不要手工 `git add . && git commit && git push`** —— 用带安全检查的脚本：

```bash
./.venv/bin/python -m tools.sync -m "fix: 修了 xxx"
```

它会依次做：扫描凭据 → 检查敏感标识 → 跑测试 → 提交 → 推送。
**任何一步发现问题就中止，绝不带着密钥推上去。**

### 每次同步前后都要确认

1. **凭据不在仓库里**：`.env`、`data/`、`.venv`、`.backups/` 必须被 `.gitignore` 排除
2. **凭据值没混进代码**：脚本会自动从 `.env` 读出所有值，逐个在跟踪文件里搜
3. **截图里没有真实信息**：截图是**渲染结果，`.gitignore` 拦不住** ——
   看到截图必须逐张核对画面内容，不能只看文件名
4. **演示数据必须虚构**：`tools/seed_demo.py` 和 `qqbot/__main__.py` 的
   离线演示数据**只能用假群名/假群号/假昵称**

---

## 这个项目踩过的坑（别重复）

### 1. 真实信息泄漏（最严重）

`tools/seed_demo.py` 曾用**真实群名、群号、群成员昵称**，12 张截图全从那个演示库拍，
还有截图暴露了机器人昵称和真实 QQ 号。推上去之后只能**删库重建**才清干净——
Git 的强制推送**不删服务端对象**，旧提交按 SHA 仍可取回。

所以：
- 演示数据一律虚构（用 `示例xx群` / `100001` / 中性昵称）
- **截图前把演示实例的 WS 指向不可达地址**（`ONEBOT_WS_URL=ws://127.0.0.1:9`），
  否则它会连上真实 NapCat，把真实昵称、QQ 号、群列表同步进演示库再显示在画面上
- 拍完**逐张看图**。曾有一张（日志页）因为 SSE 流让浏览器不 settle 而**静默失败**，
  脚本只读到旧文件的大小就以为成功了

### 2. 配置文件被整个覆盖

一次补丁脚本写错了目标变量（`p.write_text(t)` 把 A 文件的内容写进了 B 文件），
`qqbot/config.py` 被 `llm.py` 的内容整个覆盖，而当时**还没有 git**，只能翻会话日志重放补丁救回来。

所以：**改文件的脚本必须断言**（`assert old in s`），别让 `replace()` 静默不匹配。

### 3. 密钥混进 settings.json

`_persist_overlay` 曾硬编码 `pop("api_key")` / `pop("token")`，
新增 `bilibili.cookie` 时没人记得加，443 字符的 B 站登录凭据被写进了 `data/settings.json`，
还导致 `load_config` 报 `got multiple values for keyword argument 'cookie'` 直接崩。

所以：敏感字段集中在 `SECRET_FIELDS`（`qqbot/config.py`）声明一次，处处引用。

### 4. 把"接口 404"当成了"这个能力没有"

StepFun 的 `step_plan` 套餐，我只试了 OpenAI 兼容的 multipart
`/step_plan/v1/audio/transcriptions`（404）就写下结论"免费套餐不含 ASR"，
还在代码里加了个**发请求之前就拦下**的 `base_url_warning()`。
结果 step_plan 一直有 ASR，只是走**完全不同的形状**：
`POST /step_plan/v1/audio/asr/sse`（JSON + base64，SSE 返回）。
那个护栏把本来能用的配置判死了两个星期。

所以：
- **判定"某套餐/某服务不支不支持 X"，必须找到它的专属文档并实测成功一次**，
  不能靠"我认识的接口形状都试过了"
- 同理，**"替用户提前失败"的护栏要慎加** —— 它出错的方式是把正确用法变成 bug
- 网关直出的 `404 page not found`（纯文本、无 `x-trace-id`、`X-Upstream-Latency: 0`）
  和业务层 JSON 404（有 trace id）是两回事，见 `docs/NOTES.md`

### 5. AI 会话留下的野实例（就是你自己）

排查时发现 **3 个 `python -m qqbot` 同时在跑**，cgroup 全是 `dsh-subprocess-*.scope` ——
每次会话用后台任务拉一个、从来不回收。其中两个连同一个真实 NapCat 端点 + 同一个真实库，
差一点就是群里双份回复；那个卡死的实例还顺带把 `qqbot.db-wal` 顶到 4.1MB 不释放。

所以：
- **拉实例前先 `ps aux | grep "[p]ython -m qqbot"`，收工前把你起的杀掉**
- 现在 `python -m qqbot` 有 flock 单实例锁（按 OneBot 端点），抢不到会直接退出 ——
  这是保护，不是 bug。要并行调演示实例就把 `ONEBOT_WS_URL` 指到不可达地址
- 正式实例交给 `systemctl --user enable --now qqbot`，别再手工拉
- **`kill -INT` 对后台拉起的进程无效**（继承 SIG_IGN + CPython 不再装 handler），
  别以为发个 SIGINT 就算优雅退出了。详见 `docs/NOTES.md`「进程与部署」

---

## 项目速览

```
qqbot/          机器人本体（gateway 接入 → buffer 蓄水池 → decide 决定层
                → generate 生成 → safety 拦截 → sender 发送）
qqbot/web/      管理面板（aiohttp.web，同进程同事件循环）
tools/          探针、基准、演示数据、同步脚本
tests/          327 项测试（test_qqbot 46 + test_v11 243 + test_ingest 38）
docs/NOTES.md   实现笔记与踩坑记录（各服务商实测坑）
```

**测试必须全绿才能推**：

```bash
./.venv/bin/python -m tests.test_qqbot    # 46
./.venv/bin/python -m tests.test_v11      # 243
./.venv/bin/python -m tests.test_ingest   # 38
```

## 代码约定

- **写注释解释"为什么"**，不是"做了什么" —— 尤其是那些看起来奇怪的地方，
  都是踩过坑才那么写的
- **不确定的行为要实测**，别照抄文档。`docs/NOTES.md` 里每条坑都对应一次真实失败
- **改 prompt 或阈值前先看决策记录页**，别猜

---

## 常用命令

```bash
./.venv/bin/python -m tools.sync -m "..."      # 安全同步（推荐）
./.venv/bin/python -m qqbot check              # 测 NapCat 连通性
./.venv/bin/python -m tests.test_v11           # 跑 v1.1 测试
./.venv/bin/python -m tools.profile_models     # 模型画像探针
./.venv/bin/python -m tools.bench_reply        # 生产链路延迟基准
```