"""非文字消息段解析（SegmentIngestor）的测试。

    ./.venv/bin/python -m tests.test_ingest

全部离线跑：api / asr / store 都是本文件里的小 stub，下载函数被替换掉，
不会碰真实 NapCat，也不会发任何网络请求。测试数据一律虚构。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qqbot.config import MediaConfig
from qqbot.media import ingest as ingest_mod
from qqbot.media.ingest import (
    SegmentIngestor, card_note, download_to_file, file_note, is_http,
    is_sticker_image, render_segments, size_human, sticker_note,
)
from qqbot.models import Message

TESTS: list = []


def test(fn):
    TESTS.append(fn)
    return fn


def eq(actual, expected, label: str = "") -> None:
    if actual != expected:
        raise AssertionError(f"{label} 期望 {expected!r}，实际 {actual!r}")


def ok(cond, label: str = "") -> None:
    if not cond:
        raise AssertionError(f"{label} 断言失败")


# ==========================================================================
# 假的依赖 + 工具
# ==========================================================================


class FakeAPI:
    def __init__(self, ptt: str = "", forward=None, record=None, *,
                 forward_error: bool = False, ptt_error: bool = False) -> None:
        self.ptt = ptt
        self.forward = forward
        self.record = record or {}
        self.forward_error = forward_error
        self.ptt_error = ptt_error
        self.ptt_calls = 0
        self.forward_calls: list[str] = []
        self.record_calls: list[str] = []

    async def fetch_ptt_text(self, message_id):
        self.ptt_calls += 1
        if self.ptt_error:
            raise RuntimeError("模拟 ptt 失败")
        return self.ptt

    async def get_forward_msg(self, forward_id):
        self.forward_calls.append(forward_id)
        if self.forward_error:
            raise RuntimeError("模拟取转发失败")
        return [] if self.forward is None else self.forward

    async def get_record(self, file, out_format="mp3"):
        self.record_calls.append(file)
        return self.record


class FakeASR:
    def __init__(self, text: str = "识别出来的话") -> None:
        self.text = text
        self.calls: list[str] = []

    async def transcribe_file(self, audio, hotwords=None):
        self.calls.append(str(audio))
        return SimpleNamespace(ok=bool(self.text), text=self.text)


class FakeStore:
    """ImageStore 的形状。表情包段**不该**触发它（那是看图的钱）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, url):
        self.calls.append(url)
        return None


def _msg(segments=None, *, mid: int = 1001, text: str = "") -> Message:
    return Message(group_id=100, message_id=mid, user_id=7, nickname="测试员",
                   text=text, segments=list(segments or []))


def _seg(stype: str, **data) -> dict:
    return {"type": stype, "data": dict(data)}


def _fake_downloader(suffix: str = ".wav", calls: list | None = None):
    """替换 ingest_mod.download_to_file：只在本地落一个假文件。"""
    async def _dl(url, dest_dir, **kwargs):
        if calls is not None:
            calls.append(url)
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"fake-{abs(hash(url))}{suffix}"
        path.write_bytes(b"fake-audio-bytes")
        return path
    return _dl


@contextmanager
def _patch(**attrs):
    old = {}
    for name, value in attrs.items():
        old[name] = getattr(ingest_mod, name)
        setattr(ingest_mod, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(ingest_mod, name, value)


# ==========================================================================
# 语音
# ==========================================================================


@test
async def t_voice_uses_ptt_text() -> None:
    """最快那条路：NapCat 自带的语音转文字直接命中。"""
    api = FakeAPI(ptt="明天下午三点集合")
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([
        _seg("record", file="a" * 32 + ".amr", url="", file_size="11367"),
    ]))
    eq(len(notes), 1)
    ok("转文字" in notes[0], f"要说明是转文字：{notes[0]!r}")
    ok("明天下午三点集合" in notes[0], notes[0])
    eq(api.ptt_calls, 1, "应该只问一次")
    eq(ing.stats()["voice_text"], 1)


@test
async def t_voice_ptt_empty_no_url_degrades() -> None:
    """fetch_ptt_text 空 + 没有 url → 占位符，绝不抛异常。

    实测 4 条里 3 条会走这条路，所以这是常态而不是异常。
    """
    api = FakeAPI(ptt="")
    ing = SegmentIngestor(MediaConfig(), api, asr=FakeASR(""))
    notes = await ing.ingest(_msg([
        _seg("record", file="b" * 32 + ".amr",
             path="/nowhere/nt_data/Ptt/b.amr", file_size="11367"),
    ]))
    eq(notes, ["[语音]（没听清）"])
    eq(api.record_calls, ["b" * 32 + ".amr"], "没有 url 应该让 NapCat 代下")
    eq(ing.stats()["voice_failed"], 1)


@test
async def t_voice_downloads_url_and_transcribes() -> None:
    """有 http url + 有 ASR：下载后转写。"""
    api = FakeAPI(ptt="")
    asr = FakeASR("大家好呀")
    calls: list[str] = []
    with _patch(download_to_file=_fake_downloader(".wav", calls)):
        ing = SegmentIngestor(MediaConfig(), api, asr=asr)
        notes = await ing.ingest(_msg([
            _seg("record", file="c" * 32 + ".amr",
                 url="https://example.invalid/download?rkey=fake-key"),
        ]))
    eq(len(notes), 1)
    ok("大家好呀" in notes[0], notes[0])
    eq(len(calls), 1, "应该下载一次")
    eq(len(asr.calls), 1, "应该转写一次")


@test
async def t_voice_no_asr_degrades() -> None:
    """没配 ASR：连下载都不该做（url 带 rkey、会过期，下了也没人用）。"""
    api = FakeAPI(ptt="")
    calls: list[str] = []
    with _patch(download_to_file=_fake_downloader(".wav", calls)):
        ing = SegmentIngestor(MediaConfig(), api, asr=None)
        notes = await ing.ingest(_msg([
            _seg("record", url="https://example.invalid/download?rkey=x"),
        ]))
    eq(notes, ["[语音]（没听清）"])
    eq(calls, [], "没有 ASR 时不该白下载")


@test
async def t_voice_get_record_fallback_used() -> None:
    """packet 后端不可用时 url 键直接不存在 —— 交给 get_record 代下。"""
    api = FakeAPI(ptt="", record={
        "file": "d" * 32 + ".mp3",
        "url": "https://example.invalid/record/d.mp3",
        "file_size": "9000", "file_name": "d.mp3",
    })
    calls: list[str] = []
    with _patch(download_to_file=_fake_downloader(".mp3", calls)):
        ing = SegmentIngestor(MediaConfig(), api, asr=FakeASR("代下成功"))
        notes = await ing.ingest(_msg([_seg("record", file="d" * 32 + ".amr")]))
    eq(len(notes), 1)
    ok("代下成功" in notes[0], notes[0])
    eq(len(calls), 1)


# ==========================================================================
# 视频
# ==========================================================================


@test
async def t_video_local_path_not_downloaded() -> None:
    """video.url 大概率是 NapCat 本机绝对路径 → 不能当 URL 去请求。"""
    calls: list[str] = []
    with _patch(download_to_file=_fake_downloader(".wav", calls)):
        ing = SegmentIngestor(MediaConfig(), FakeAPI(), asr=FakeASR("x"))
        notes = await ing.ingest(_msg([
            _seg("video", file="v.mp4",
                 url="/home/example/nt_data/Video/v.mp4", file_size="3005688"),
        ]))
    eq(notes, ["[视频]（本机文件，读不到内容）"])
    eq(calls, [], "本机路径绝不能触发下载")


@test
async def t_video_http_downloads_and_transcribes() -> None:
    """http url + ffmpeg + ASR → 抽音轨转写。"""
    calls: list[str] = []
    with _patch(download_to_file=_fake_downloader(".wav", calls),
                ffmpeg_available=lambda: True):
        ing = SegmentIngestor(MediaConfig(), FakeAPI(), asr=FakeASR("视频里说的话"))
        notes = await ing.ingest(_msg([
            _seg("video", file="v.mp4", url="https://example.invalid/v.mp4",
                 file_size="3005688"),
        ]))
    eq(len(notes), 1)
    ok(notes[0].startswith("[视频]"), notes[0])
    ok("视频里说的话" in notes[0], notes[0])
    eq(len(calls), 1)
    eq(ing.stats()["video_text"], 1)


@test
async def t_video_http_without_asr() -> None:
    calls: list[str] = []
    with _patch(download_to_file=_fake_downloader(".wav", calls),
                ffmpeg_available=lambda: True):
        ing = SegmentIngestor(MediaConfig(), FakeAPI(), asr=None)
        notes = await ing.ingest(_msg([
            _seg("video", url="https://example.invalid/v.mp4"),
        ]))
    eq(notes, ["[视频]（没配置语音识别）"])
    eq(calls, [], "没有 ASR 时不该白下载")


@test
async def t_video_http_without_ffmpeg() -> None:
    """ASR 不认视频容器，没 ffmpeg 就抽不出音轨 —— 要明说。"""
    with _patch(download_to_file=_fake_downloader(".mp4"),
                ffmpeg_available=lambda: False):
        ing = SegmentIngestor(MediaConfig(), FakeAPI(), asr=FakeASR("x"))
        notes = await ing.ingest(_msg([
            _seg("video", url="https://example.invalid/v.mp4"),
        ]))
    eq(notes, ["[视频]（没有 ffmpeg，提取不了声音）"])


# ==========================================================================
# 卡片
# ==========================================================================


@test
async def t_card_location_share() -> None:
    """真实形状：meta 的第一个 key 是 Location.Search，不是 news。"""
    payload = {
        "app": "com.tencent.map",
        "meta": {"Location.Search": {
            "address": "示例市示例路1号", "lat": "1.0", "lng": "2.0",
            "name": "示例咖啡店",
        }},
        "prompt": "[位置]", "ver": "0.0.0.1", "view": "LocationShare",
    }
    note = card_note({"data": json.dumps(payload, ensure_ascii=False)})
    ok(note.startswith("[位置]"), f"要标成位置：{note!r}")
    ok("示例咖啡店" in note, note)
    ok("示例市示例路1号" in note, note)


@test
async def t_card_structmsg_news() -> None:
    payload = {
        "app": "com.tencent.structmsg",
        "meta": {"news": {"title": "示例标题", "desc": "示例简介",
                          "jumpUrl": "https://example.invalid/news/1"}},
        "view": "news",
    }
    note = card_note({"data": json.dumps(payload, ensure_ascii=False)})
    ok(note.startswith("[分享]"), note)
    ok("标题「示例标题」" in note, note)
    ok("简介「示例简介」" in note, note)
    ok("https://example.invalid/news/1" in note, note)


@test
async def t_card_miniprogram_qqdocurl() -> None:
    """小程序字段（本机没观测到）——有就取，没有不算。"""
    payload = {"app": "com.tencent.miniapp", "meta": {"detail_1": {
        "title": "示例小程序", "qqdocurl": "https://example.invalid/doc/1",
    }}, "view": "miniapp"}
    note = card_note({"data": json.dumps(payload, ensure_ascii=False)})
    ok("示例小程序" in note, note)
    ok("https://example.invalid/doc/1" in note, note)


@test
async def t_card_invalid_json_degrades() -> None:
    """data.data 不是合法 JSON → [卡片]，不抛异常。"""
    eq(card_note({"data": "{这不是合法JSON"}), "[卡片]")
    api = FakeAPI()
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([_seg("json", data="{坏JSON")]))
    eq(notes, ["[卡片]"])
    eq(ing.stats()["card"], 1)


@test
async def t_card_prompt_fallback() -> None:
    """meta 里什么都没有，退到 prompt。"""
    note = card_note({"data": "{}", "prompt": "点这里看详情"})
    eq(note, "[卡片] 点这里看详情")


@test
async def t_card_never_raises() -> None:
    """极端输入全都要返回字符串。"""
    cases = [
        None, {}, {"data": None}, {"data": 123}, {"data": []}, {"data": "[]"},
        {"data": '{"meta": "不是字典"}'},
        {"data": '{"meta": null, "app": null, "view": null}'},
        {"data": '{"meta": {"a": {"title": 123, "desc": ["x"]}}}'},
        {"data": '{"meta": {"k": [{"title": "标题"}, None, 7]}}'},
        {"data": ["不是", "字符串"]},
    ]
    for case in cases:
        note = card_note(case)
        ok(isinstance(note, str) and note, f"{case!r} -> {note!r}")


# ==========================================================================
# 合并转发
# ==========================================================================


def _nodes() -> list[dict]:
    return [
        {"sender": {"nickname": "甲"}, "message": [
            {"type": "text", "data": {"text": "第一句"}}]},
        {"sender": {"card": "乙"}, "message": "第二句"},
    ]


@test
async def t_forward_expands_nodes() -> None:
    api = FakeAPI(forward=_nodes())
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([_seg("forward", id="1000000000000000001")]))
    eq(len(notes), 1)
    ok("共 2 条" in notes[0], notes[0])
    ok("甲: 第一句" in notes[0], notes[0])
    ok("乙: 第二句" in notes[0], notes[0])
    eq(api.forward_calls, ["1000000000000000001"])
    eq(ing.stats()["forward_nodes"], 2)


@test
async def t_forward_empty_degrades() -> None:
    api = FakeAPI(forward=[])
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([_seg("forward", id="1000000000000000002")]))
    eq(notes, ["[合并转发]（取不到内容）"])


@test
async def t_forward_api_error_degrades() -> None:
    """api 抛异常 → 占位符，不往外冒。"""
    api = FakeAPI(forward_error=True)
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([_seg("forward", id="1000000000000000003")]))
    eq(notes, ["[合并转发]（取不到内容）"])


@test
async def t_forward_respects_max_nodes() -> None:
    nodes = [{"sender": {"nickname": f"人{i}"},
              "message": [{"type": "text", "data": {"text": f"第{i}条"}}]}
             for i in range(5)]
    api = FakeAPI(forward=nodes)
    ing = SegmentIngestor(MediaConfig(forward_max_nodes=2), api)
    notes = await ing.ingest(_msg([_seg("forward", id="1000000000000000004")]))
    ok("共 5 条" in notes[0], notes[0])
    ok("（还有 3 条省略）" in notes[0], notes[0])
    ok("第2条" not in notes[0], f"超出上限的不该展开：{notes[0]!r}")


@test
async def t_forward_node_non_text_uses_placeholder() -> None:
    api = FakeAPI(forward=[{"sender": {"nickname": "丙"}, "message": [
        {"type": "text", "data": {"text": "看图"}},
        {"type": "image", "data": {"file": "x.jpg", "sub_type": 0}},
    ]}])
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([_seg("forward", id="1000000000000000005")]))
    ok("丙: 看图 [图片]" in notes[0], notes[0])


@test
async def t_forward_no_id_degrades() -> None:
    api = FakeAPI(forward=_nodes())
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([_seg("forward")]))
    eq(notes, ["[合并转发]（取不到内容）"])
    eq(api.forward_calls, [], "没有 id 不该去调 api")


# ==========================================================================
# 文件
# ==========================================================================


@test
async def t_file_uses_file_and_file_size() -> None:
    """字段是 file / file_size，**没有 name / size**。"""
    note = file_note({"file": "示例文档.pdf", "file_id": "/" + "A" * 37,
                      "file_size": "1137156",
                      "url": "https://example.invalid/f/1"})
    eq(note, "[文件] 示例文档.pdf（1.1MB）")


@test
async def t_file_segment_through_ingest() -> None:
    ing = SegmentIngestor(MediaConfig(), FakeAPI())
    notes = await ing.ingest(_msg([
        _seg("file", file="示例文档.pdf", file_id="/" + "A" * 37,
             file_size="1137156", url="https://example.invalid/f/1"),
    ]))
    eq(notes, ["[文件] 示例文档.pdf（1.1MB）"])
    eq(ing.stats()["file"], 1)


@test
async def t_file_missing_size() -> None:
    note = file_note({"file": "示例文档.txt"})
    eq(note, "[文件] 示例文档.txt（大小未知）")
    eq(file_note(None), "[文件] 未知文件（大小未知）")


@test
async def t_size_human_edges() -> None:
    eq(size_human(0), "0B")
    eq(size_human(512), "512B")
    eq(size_human(1024), "1.0KB")
    eq(size_human(1137156), "1.1MB")
    eq(size_human(5 * 1024 ** 3), "5.0GB")
    for bad in (None, "", "abc", -1, [], {}, float("nan")):
        eq(size_human(bad), "大小未知", f"{bad!r} 应该给出兜底文案")


# ==========================================================================
# 表情包 / 图片
# ==========================================================================


@test
async def t_emoji_id_image_is_sticker_not_photo() -> None:
    """商城表情入站是带 emoji_id 的 image 段 —— 必须识别成表情包，
    不能当普通图送去视觉理解（那是白花钱）。"""
    data = {"file": "x.jpg", "sub_type": 0, "emoji_id": "128512",
            "summary": "呲牙"}
    ok(is_sticker_image(data), "带 emoji_id 要算表情包")
    store = FakeStore()
    ing = SegmentIngestor(MediaConfig(), FakeAPI(), store=store)
    notes = await ing.ingest(_msg([_seg("image", **data)]))
    eq(len(notes), 1)
    ok(notes[0].startswith("[表情包]"), notes[0])
    ok("呲牙" in notes[0], notes[0])
    eq(store.calls, [], "表情包不该走图片下载")


@test
async def t_subtype_one_is_sticker() -> None:
    """sub_type 是数字 1（也兼容字符串写法）。"""
    ok(is_sticker_image({"sub_type": 1}), "数字 1 要算表情包")
    ok(is_sticker_image({"sub_type": "1"}), "字符串 1 也兼容")
    ing = SegmentIngestor(MediaConfig(), FakeAPI())
    notes = await ing.ingest(_msg([_seg("image", file="x.gif", sub_type=1)]))
    eq(notes, ["[表情包] x.gif"])


@test
async def t_unknown_subtype_not_sticker() -> None:
    """7 / 11 含义未确认，**不要**当表情包。"""
    for sub in (7, 11, "7", 0, "0", None, True):
        ok(not is_sticker_image({"sub_type": sub}), f"sub_type={sub!r} 不该算表情包")
    ing = SegmentIngestor(MediaConfig(), FakeAPI())
    notes = await ing.ingest(_msg([_seg("image", file="x.jpg", sub_type=7)]))
    eq(notes, [], "普通图片交给看图模块，这里不重复出备注")


@test
async def t_legacy_mface_segment() -> None:
    ing = SegmentIngestor(MediaConfig(), FakeAPI())
    notes = await ing.ingest(_msg([_seg("mface", summary="笑死")]))
    eq(notes, ["[表情包] 笑死"])


@test
async def t_sticker_note_never_raises() -> None:
    for case in (None, {}, {"summary": None}, {"emoji_id": 123}, {"file": ""},
                 {"summary": ["x"]}):
        note = sticker_note(case)
        ok(isinstance(note, str) and note, f"{case!r} -> {note!r}")


# ==========================================================================
# 多段 / 边界
# ==========================================================================


@test
async def t_multiple_segments_all_returned() -> None:
    api = FakeAPI(ptt="语音内容")
    ing = SegmentIngestor(MediaConfig(), api)
    notes = await ing.ingest(_msg([
        {"type": "text", "data": {"text": "看这个"}},
        _seg("record", file="e" * 32 + ".amr"),
        _seg("json", data=json.dumps({"app": "x",
                                      "meta": {"news": {"title": "示例标题"}}})),
        _seg("file", file="示例文档.pdf", file_size="2048"),
        _seg("face", id="175"),
    ]))
    eq(len(notes), 3, f"应只出 3 条备注（text/face 不算）：{notes!r}")
    ok(notes[0].startswith("[语音]"), notes[0])
    ok(notes[1].startswith("[分享]"), notes[1])
    ok(notes[2].startswith("[文件]"), notes[2])


@test
async def t_no_parseable_segments_returns_empty() -> None:
    ing = SegmentIngestor(MediaConfig(), FakeAPI())
    eq(await ing.ingest(_msg([])), [])
    eq(await ing.ingest(_msg([{"type": "text", "data": {"text": "纯文字"}}])), [])
    eq(await ing.ingest(_msg([{"type": "poke", "data": {"qq": "7"}}])), [])
    # segments 不是列表（模型层异常）也不能炸
    eq(await ing.ingest(SimpleNamespace(segments=None, message_id=1)), [])
    eq(await ing.ingest(SimpleNamespace(segments="oops", message_id=1)), [])


@test
async def t_extreme_segments_never_raise() -> None:
    """空 data / None / 类型不对 —— 一律不许抛。"""
    ing = SegmentIngestor(MediaConfig(), FakeAPI())
    notes = await ing.ingest(_msg([
        None, 123, "不是字典", {}, {"type": None, "data": None},
        {"type": "record", "data": None},
        {"type": "record", "data": "不是字典"},
        {"type": "record", "data": []},
        {"type": "json", "data": {"data": None}},
        {"type": "json", "data": "不是字典"},
        {"type": "forward", "data": None},
        {"type": "forward", "data": {"id": None}},
        {"type": "file", "data": []},
        {"type": "file", "data": None},
        {"type": "image", "data": "不是字典"},
        {"type": "image", "data": {"emoji_id": None, "sub_type": {}}},
        {"type": "video", "data": None},
        {"type": "video", "data": {"url": 12345}},
        {"type": "mface", "data": {"summary": 42}},
        {"type": "unknown_type", "data": {"x": 1}},
    ]))
    ok(isinstance(notes, list), f"必须返回列表：{notes!r}")
    for note in notes:
        ok(isinstance(note, str) and note, f"备注必须是非空字符串：{note!r}")


@test
async def t_segment_timeout_degrades() -> None:
    """卡住的段要被 ingest_timeout 掐掉，降级成占位符。"""
    class SlowAPI(FakeAPI):
        async def fetch_ptt_text(self, message_id):
            await asyncio.sleep(5)
            return "永远等不到"

    ing = SegmentIngestor(MediaConfig(ingest_timeout=0.05), SlowAPI())
    notes = await ing.ingest(_msg([
        _seg("record", file="f" * 32 + ".amr"),
        _seg("file", file="示例文档.pdf", file_size="1024"),
    ]))
    eq(notes[0], "[语音]（没听清）")
    ok(notes[1].startswith("[文件]"), "后面的段不受影响")
    ok(ing.stats()["timeouts"] >= 1, "超时要计数")


@test
async def t_disabled_switches_skip() -> None:
    cfg = MediaConfig(voice_enabled=False, video_enabled=False,
                      card_enabled=False, forward_enabled=False,
                      file_enabled=False)
    ing = SegmentIngestor(cfg, FakeAPI(ptt="不该被问到"))
    notes = await ing.ingest(_msg([
        _seg("record", file="g" * 32 + ".amr"),
        _seg("video", url="https://example.invalid/v.mp4"),
        _seg("json", data="{}"),
        _seg("forward", id="1000000000000000006"),
        _seg("file", file="示例文档.pdf", file_size="1024"),
    ]))
    eq(notes, [], "关掉的类型不该产出备注")


@test
async def t_stats_shape_and_counts() -> None:
    api = FakeAPI(ptt="语音内容", forward=_nodes())
    ing = SegmentIngestor(MediaConfig(), api)
    await ing.ingest(_msg([
        _seg("record", file="h" * 32 + ".amr"),
        _seg("json", data="{}"),
        _seg("forward", id="1000000000000000007"),
        _seg("file", file="示例文档.pdf", file_size="2048"),
        _seg("image", file="x.gif", sub_type=1),
    ]))
    st = ing.stats()
    eq(st["voice"], 1)
    eq(st["voice_text"], 1)
    eq(st["card"], 1)
    eq(st["forward"], 1)
    eq(st["forward_nodes"], 2)
    eq(st["file"], 1)
    eq(st["sticker"], 1)
    ok("timeouts" in st and "errors" in st, f"要有兜底计数：{st!r}")
    # stats 是快照，外部改不动内部
    st["voice"] = 999
    eq(ing.stats()["voice"], 1)


@test
async def t_helpers_never_raise() -> None:
    ok(is_http("http://example.invalid/a"))
    ok(is_http("https://example.invalid/a"))
    ok(not is_http("/home/example/nt_data/Video/v.mp4"), "本机路径不是 http")
    ok(not is_http(""))
    ok(not is_http(None))
    ok(not is_http(123))
    eq(render_segments(None), "")
    eq(render_segments("不是数组"), "")
    eq(render_segments([None, 7, {"type": "text", "data": {"text": "hi"}}]), "hi")
    eq(render_segments([{"type": "at", "data": {"qq": "all"}}]), "@全体成员")
    eq(render_segments([{"type": "record", "data": {}}]), "[语音]")
    eq(render_segments([{"type": "weird", "data": {}}]), "[weird]")
    for bad in (None, 7, {"sub_type": True}, {"emoji_id": 0}):
        ok(isinstance(is_sticker_image(bad), bool))


@test
async def t_download_rejects_non_http_without_network() -> None:
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        eq(await download_to_file("", dest), None)
        eq(await download_to_file("not a url", dest), None)
        eq(await download_to_file("/home/example/nt_data/Ptt/x.amr", dest), None)
        eq(list(dest.iterdir()), [], "非 http 不该产生任何文件")


# ==========================================================================
# 运行器
# ==========================================================================


async def run_all() -> int:
    print(f"\n运行 {len(TESTS)} 个 ingest 测试…\n")
    passed = 0
    failures: list[tuple[str, str]] = []

    for fn in TESTS:
        name = fn.__name__
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            print(f"  \033[32m✓\033[0m {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  \033[31m✗\033[0m {name}: {exc}")
            failures.append((name, traceback.format_exc()))

    print(f"\n{'=' * 62}")
    print(f"通过 {passed}/{len(TESTS)}")
    if failures:
        print("\n失败详情:")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all()))
