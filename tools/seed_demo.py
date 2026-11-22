"""造一份演示数据，用来预览管理面板（不连 QQ、不调 LLM）。

    QQBOT_DB_PATH=/tmp/demo.db ./.venv/bin/python -m tools.seed_demo
    QQBOT_DB_PATH=/tmp/demo.db WEB_TOKEN= ./.venv/bin/python -m qqbot web

第二条命令会起一个不需要登录的面板，直接看效果。

⚠️ **这个文件里的数据必须全是虚构的。**
演示库会被截图放进 README 和文档，而 git 历史删不干净 ——
一旦写进真实的群名、群号、昵称，就等于公开了群友信息。
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qqbot.models import Decision, Message  # noqa: E402
from qqbot.store import (  # noqa: E402
    Database, FeedbackRepo, PolicyRepo, ProfileRepo, Repo, StickerRepo,
)

#: 演示群：群号用明显的占位数字，名字用虚构的
GROUPS = [
    (100001, "示例技术交流群", True),
    (100002, "示例摸鱼群", True),
    (100003, "示例学习打卡群", False),
    (100004, "示例游戏开黑群", False),
]

#: 演示成员：全部虚构，不要改成任何真实昵称
MEMBERS = ["夜航船", "咸鱼不翻身", "一只猫饼", "半糖去冰", "码农小王", "沉默的螺旋"]

CHAT = [
    "今天作业好多啊", "数学卷子最后一道谁会", "我也不会",
    "刚吃完饭", "有人打游戏吗", "来啊来啊", "等下我先写完作业",
    "这题我会", "发我看看", "草", "笑死", "确实",
    "明天要交吗", "应该是周五", "那还行", "谁有英语笔记",
    "我拍给你", "谢了", "不客气", "外面下雨了",
    "带伞了吗", "没带，惨", "啊这", "6", "在吗",
    "咋了", "问个事", "说", "计算机二级什么时候报名",
]

DRAFTS = ["确实", "我也不会啊", "发我看看", "笑死", "明天要交", "啊这"]
REASONS = ["被@", "提到名字", "话题命中(编程)", "有人在提问", "群在刷屏", "随机放弃", "分数不足(0.30<0.55)"]


async def main() -> int:
    import os

    db_path = os.environ.get("QQBOT_DB_PATH")
    if not db_path:
        print("请通过 QQBOT_DB_PATH 指定一个演示库路径，例如：")
        print("  QQBOT_DB_PATH=/tmp/demo.db ./.venv/bin/python -m tools.seed_demo")
        return 2

    path = Path(db_path)
    if path.exists():
        print(f"演示库已存在，先删除再重建: {path}")
        path.unlink()

    db = await Database.open(path)
    repo = Repo(db)
    rng = random.Random(20261007)
    now = int(time.time())

    print(f"写入演示数据 -> {path}")

    for group_id, name, enabled in GROUPS:
        await repo.upsert_group(group_id, name)
        if enabled:
            await repo.set_group_enabled(group_id, True)
        for index, member in enumerate(MEMBERS):
            for _ in range(rng.randint(1, 12)):
                await repo.upsert_member(group_id, 10000 + index, member)

    message_id = 700000
    for group_id, _name, enabled in GROUPS:
        count = 180 if enabled else 20
        # 按时间正序写入，让面板上的时间线读起来是连贯的
        start = now - count * 120
        for i in range(count):
            ts = start + i * 120 + rng.randint(-40, 40)
            is_self = rng.random() < 0.12
            member_index = rng.randrange(len(MEMBERS))
            message_id += 1
            await repo.save_message(Message(
                group_id=group_id,
                message_id=message_id,
                user_id=99901 if is_self else 10000 + member_index,
                nickname="我" if is_self else MEMBERS[member_index],
                text=rng.choice(DRAFTS if is_self else CHAT),
                ts=ts,
                is_self=is_self,
                mentions_self=(not is_self) and rng.random() < 0.15,
            ))

    # 决策留痕（同样按时间正序）
    decision_start = now - 120 * 150
    for i in range(120):
        ts = decision_start + i * 150 + rng.randint(-30, 30)
        spoke = rng.random() < 0.18
        reason = rng.choice(REASONS) if not spoke else rng.choice(REASONS[:5])
        if spoke:
            reason = rng.choice(["被@", "提到名字", "话题命中(编程)", "有人在提问"])
        await repo.save_decision(
            rng.choice([g[0] for g in GROUPS]),
            Decision(
                should_reply=spoke,
                priority=round(rng.uniform(0.15, 1.0) if spoke else rng.uniform(0.05, 0.62), 3),
                reason=reason,
                ts=ts,
                text=rng.choice(CHAT),
                draft="\n".join(rng.sample(DRAFTS, rng.randint(1, 2))) if spoke else "",
                features={"mentioned": spoke and reason == "被@", "question": rng.random() < 0.3},
            ),
        )

    # 长期记忆
    for content in [
        "夜航船喜欢打游戏，常约人一起",
        "咸鱼不翻身是群里的活跃分子，爱发表情包",
        "半糖去冰在准备计算机二级考试",
        "群里常聊作业和考试",
        "一只猫饼的数学很好",
    ]:
        await repo.add_memory("group", GROUPS[0][0], content,
                              now=now - rng.randint(0, 20) * 86400)

    await _seed_v11(db, rng, now)

    stats = await repo.message_stats()
    print(f"  消息 {stats['messages_total']} 条（收到 {stats['messages_received']} / 发出 {stats['messages_sent']}）")
    print(f"  决策 {len(await repo.list_decisions(999))} 条")
    print(f"  记忆 {await repo.memory_count()} 条")
    print("\n启动预览：")
    print(f"  QQBOT_DB_PATH={path} WEB_TOKEN= ./.venv/bin/python -m qqbot web")
    await db.close()
    return 0


async def _seed_v11(db, rng, now: int) -> None:
    """表情库 / 群画像 / 群策略 / 反馈曲线 —— 让 v1.1 的新页面也有东西可看。"""
    stickers = StickerRepo(db)
    profiles = ProfileRepo(db)
    policies = PolicyRepo(db)
    feedback = FeedbackRepo(db)

    tagsets = [["搞笑", "大笑"], ["无语", "嫌弃"], ["赞同", "牛"],
               ["哭泣", "委屈"], ["开心"], ["惊讶"], ["累", "摸鱼"]]
    group_id = GROUPS[0][0]
    for i, tags in enumerate(tagsets):
        sid, _ = await stickers.add(
            hash=f"demo{i}", ext="gif", size=4096 + i * 800,
            source_group=group_id, source_user=10001 + i,
        )
        await stickers.set_tags(sid, tags, ["猫"] if i % 2 else [], f"演示表情 {i + 1}")
        await stickers.mark_collected(sid, f"通用梗(见到{i + 2}次/{i + 2}人)")
        for _ in range(i + 1):
            await stickers.bump_use(sid, win=(i % 2 == 0))
        await stickers.record_usage(sid, group_id, None, "演示")

    # 见过但没收的台账 —— 面板上的"见过"会比"已收藏"多
    for i in range(5):
        sid, _ = await stickers.add(hash=f"seen{i}", ext="png",
                                    source_group=group_id, source_user=10005)
        await stickers.observe(sid, group_id, 10005 + i, "哈哈", "笑死")

    profile = await profiles.get(group_id)
    profile.topics = {"原神": 0.42, "作业": 0.21, "动漫": 0.19, "技术": 0.18}
    profile.avg_msg_len = 8.4
    profile.emoji_rate = 0.18
    profile.sticker_rate = 0.12
    profile.question_rate = 0.09
    profile.atmosphere = 0.47
    profile.sample_size = 320
    profile.hourly = [0.5, 0.2, 0, 0, 0, 0, 1, 3, 6, 4, 5, 7,
                      9, 6, 5, 8, 12, 18, 26, 34, 41, 38, 22, 8]
    profile.slang = {"上班": "打游戏", "打螺丝": "刷副本", "开黑": "组队打游戏"}
    profile.core_members = [
        {"user_id": 10000, "nickname": MEMBERS[0], "count": 88},
        {"user_id": 10001, "nickname": MEMBERS[1], "count": 61},
        {"user_id": 10002, "nickname": MEMBERS[2], "count": 44},
        {"user_id": 10003, "nickname": MEMBERS[3], "count": 23},
    ]
    await profiles.save(profile)

    policy = await policies.get(group_id)
    policy.samples = 248
    policy.threshold_delta = -0.042
    policy.initiative = 0.355
    policy.sticker_rate = 0.168
    policy.emoji_rate = 0.214
    policy.quote_alpha, policy.quote_beta = 7.0, 4.0
    policy.at_alpha, policy.at_beta = 3.0, 9.0
    await policies.save(policy)

    kinds = [("emoji_like", 1.0), ("reply", 0.8), ("topic_continue", 0.4),
             ("ignored", -0.3), ("cold", -0.5)]
    for day in range(14):
        for j in range(6):
            kind, _ = kinds[(day + j) % len(kinds)]
            eid = await feedback.open_event(
                group_id, [1], now=now - (13 - day) * 86400
            )
            await feedback.record(eid, kind, "")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
