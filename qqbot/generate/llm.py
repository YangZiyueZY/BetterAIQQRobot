"""LLM 调用层。

OpenAI 兼容的 /chat/completions。DeepSeek / 豆包 / Qwen / 本地 vLLM / 自建代理
都走同一套协议，换供应商只改配置，不动代码。

## 保留思维链的代价和对策

推理模型先输出 `reasoning_content` 再输出 `content`。思考能显著提升回复质量，
但**思考量不可控** —— 实测同一个模型：

    普通群聊回复    思考 199~330 token    2.7~4.4s
    记忆抽取等分析  思考 539~728 token    6.6~8.5s
    真推理难题      思考 1839+ token      16.5s（4000 预算都不够，要 8000）

而且 `reasoning_effort` 不是所有供应商都认 —— 实测某些自建代理会**静默忽略**它，
各档位的 token 用量完全一样。所以真正兜底的不是那个参数，而是：

  1. **分档预算**：回复 1200 / 分析 4000。给紧了思考会把正文挤没，
     给太大又会让它在群聊里越想越久。
  2. **分档超时**：回复 30s / 分析 150s。想太久就放弃这一轮 ——
     真人也不会盯着屏幕憋 20 秒才回一个"在"。
  3. **截断自愈**：`finish_reason=length` 且正文为空时预算翻倍重试一次。

`reasoning_effort` 仍然会发出去（换到认这个参数的供应商就有用），被拒时自动摘掉。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from ..config import LLMConfig
from ..models import Bubble, Draft
from ..segments import extract_sticker_token, split_face_tokens

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

#: 单次请求最多尝试几次（每次都因为摘掉某个可选字段而重试）
MAX_ATTEMPTS = 3

#: 被截断时的自动扩容倍数
TRUNCATION_BOOST = 2.0

#: 调用画像 -> (预算字段, 超时字段)。
#: reply 用于群聊回复（要快），analysis 用于记忆抽取/群复盘/表情打标（要深）。
PROFILES: dict[str, tuple[str, str]] = {
    "reply": ("reply_max_tokens", "reply_timeout"),
    "analysis": ("analysis_max_tokens", "analysis_timeout"),
    "default": ("max_tokens", "timeout"),
}


class LLMError(RuntimeError):
    pass


class LLMTruncatedError(LLMError):
    """token 预算用尽，content 为空。"""


class LLMTimeoutError(LLMError):
    """思考超时。群聊里这轮直接不说，比硬等更自然。"""


@dataclass
class LLMStats:
    calls: int = 0
    errors: int = 0
    truncated: int = 0
    timeouts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    last_latency_ms: int = 0
    last_error: str = ""
    last_finish_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "llm_errors": self.errors,
            "llm_truncated": self.truncated,
            "llm_timeouts": self.timeouts,
            "llm_tokens_in": self.tokens_in,
            "llm_tokens_out": self.tokens_out,
            "llm_reasoning_tokens": self.reasoning_tokens,
        }


def extract_json(text: str) -> dict:
    """从模型输出里抠出第一个完整的 JSON 对象。

    容忍 ```json 围栏、前后废话、以及嵌套花括号。
    """
    text = (text or "").strip()
    if not text:
        raise LLMError("模型返回空内容")

    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # 扫描第一个花括号平衡块
        start = candidate.find("{")
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for idx in range(start, len(candidate)):
                ch = candidate[idx]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blob = candidate[start: idx + 1]
                        try:
                            obj = json.loads(blob)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
            start = candidate.find("{", start + 1)

    raise LLMError(f"无法从模型输出解析出 JSON: {text[:200]}")


def to_draft(payload: dict) -> Draft:
    """把模型返回的 JSON 转成 Draft。

    容忍三种写法，因为模型不总是严格听话：
        "messages": ["纯文本", ...]
        "messages": [{"text": "...", "face": "笑哭"}, ...]
        正文里内联的 [face:笑哭] / [sticker:搞笑] 标记
    """
    raw = payload.get("messages")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    if not raw and payload.get("reply"):
        raw = [str(payload["reply"])]

    bubbles: list[Bubble] = []
    for item in raw:
        if isinstance(item, str):
            bubbles.append(_bubble_from_text(item))
        elif isinstance(item, dict):
            text = str(item.get("text") or "")
            face = item.get("face")
            sticker = item.get("sticker")
            text, inline_faces = split_face_tokens(text)
            text, inline_sticker = extract_sticker_token(text)
            bubbles.append(
                Bubble(
                    text=text.strip(),
                    face=str(face).strip() if face else (inline_faces[0] if inline_faces else None),
                    sticker=str(sticker).strip() if sticker else inline_sticker,
                )
            )

    bubbles = [b for b in bubbles if b]

    emoji = payload.get("emoji_like")
    return Draft(
        bubbles=bubbles,
        thought=str(payload.get("thought") or ""),
        quote=bool(payload.get("reply") or payload.get("quote")),
        at=bool(payload.get("at")),
        emoji_like=str(emoji) if emoji else None,
    )


def _bubble_from_text(item: str) -> Bubble:
    text, faces = split_face_tokens(item)
    text, sticker = extract_sticker_token(text)
    return Bubble(
        text=text.strip(),
        face=faces[0] if faces else None,
        sticker=sticker,
    )


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.stats = LLMStats()
        self._sem = asyncio.Semaphore(max(1, cfg.max_concurrency))
        self._session: aiohttp.ClientSession | None = None
        # 乐观尝试，遇到 400 再永久摘掉对应字段
        self._json_mode_ok = True
        self._effort_param_ok = True
        self._enable_thinking_param_ok = True
        # 最近一次调用的用量，用于"预算被吃光"时给出可诊断的报错
        self._last_completion_tokens = 0
        self._last_reasoning_tokens = 0
        self._last_reasoning_estimated = False

    # ---- 生命周期 -----------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # 超时按请求传，因为 reply / analysis 两档差很多
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ---- 画像 -> 预算/超时 --------------------------------------------

    def model_for(self, profile: str) -> str:
        """按调用画像选模型。

        reply 走快模型（群聊等不起），analysis 走强模型（不在回复链路上）。
        """
        if profile == "reply" and self.cfg.reply_model:
            return self.cfg.reply_model
        if profile == "analysis" and self.cfg.analysis_model:
            return self.cfg.analysis_model
        return self.cfg.model

    def budget_for(self, profile: str, override: int | None = None) -> int:
        if override is not None:
            return int(override)
        token_field, _ = PROFILES.get(profile, PROFILES["default"])
        return int(getattr(self.cfg, token_field, self.cfg.max_tokens))

    def timeout_for(self, profile: str) -> float:
        _, timeout_field = PROFILES.get(profile, PROFILES["default"])
        return float(getattr(self.cfg, timeout_field, self.cfg.timeout))

    # ---- 请求体 -------------------------------------------------------

    def _build_body(
        self,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        json_mode: bool,
        profile: str = "default",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_for(profile),
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode and self._json_mode_ok:
            body["response_format"] = {"type": "json_object"}

        if self.cfg.keep_thinking:
            # 保留思维链，并按配置要强度。有的供应商静默忽略这个字段，
            # 有的直接 400 —— 后者会被 _degrade_option 摘掉。
            if self._effort_param_ok and self.cfg.reasoning_effort:
                body["reasoning_effort"] = self.cfg.reasoning_effort
        elif self._enable_thinking_param_ok:
            body["enable_thinking"] = False
        return body

    def _degrade_option(self, raw_error: str) -> bool:
        """400 时判断是哪个可选字段惹的祸，摘掉它并返回 True。"""
        low = raw_error.lower()

        def mentions(field: str) -> bool:
            return field in low

        if self._json_mode_ok and mentions("response_format"):
            self._json_mode_ok = False
            log.warning("供应商不支持 response_format，已降级为纯 prompt 约束")
            return True
        if self._effort_param_ok and mentions("reasoning_effort"):
            self._effort_param_ok = False
            log.warning("供应商不认识 reasoning_effort，已去掉该字段")
            return True
        if self._enable_thinking_param_ok and mentions("enable_thinking"):
            self._enable_thinking_param_ok = False
            log.warning("供应商不认识 enable_thinking，已去掉该字段")
            return True
        # 有些实现统一报 UNKNOWN_FIELD 而不点名，逐个摘掉重试
        if "unknown_field" in low or "未知请求字段" in raw_error:
            if self._effort_param_ok and self.cfg.keep_thinking:
                self._effort_param_ok = False
                log.warning("供应商拒绝了某个扩展字段，先去掉 reasoning_effort 重试")
                return True
            if self._enable_thinking_param_ok and not self.cfg.keep_thinking:
                self._enable_thinking_param_ok = False
                return True
        return False

    # ---- 原始调用 -----------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        profile: str = "default",
    ) -> str:
        """发一次补全请求。

        profile 决定预算和超时：`reply` 要快，`analysis` 要深。
        思考量不可控，所以这两个档位是实际约束它的唯一手段。
        """
        if not self.cfg.configured:
            raise LLMError("LLM 未配置：请在设置里填写 base_url / model / API Key")

        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        session = await self._get_session()
        req_timeout = aiohttp.ClientTimeout(total=self.timeout_for(profile))

        budget = self.budget_for(profile, max_tokens)
        raw = ""
        started = time.monotonic()

        # 预算可能因为"被截断"而翻倍重试，所以分两层循环
        for boost_round in range(2):
            body = self._build_body(messages, temperature, int(budget), json_mode,
                                    profile)

            for _ in range(MAX_ATTEMPTS):
                async with self._sem:
                    try:
                        async with session.post(
                            url, headers=headers, json=body, timeout=req_timeout
                        ) as resp:
                            raw = await resp.text()
                            status = resp.status
                    except (asyncio.TimeoutError, TimeoutError):
                        self.stats.timeouts += 1
                        self.stats.errors += 1
                        self.stats.last_error = (
                            f"思考超时（{self.timeout_for(profile):.0f}s，预算 {budget}）"
                        )
                        raise LLMTimeoutError(self.stats.last_error) from None
                    except aiohttp.ClientError as exc:
                        self.stats.errors += 1
                        self.stats.last_error = f"网络错误: {exc}"
                        raise LLMError(self.stats.last_error) from exc

                if status == 200:
                    break
                if status == 400 and self._degrade_option(raw):
                    body = self._build_body(messages, temperature, int(budget), json_mode,
                                    profile)
                    continue
                self.stats.errors += 1
                self.stats.last_error = f"HTTP {status}: {raw[:200]}"
                raise LLMError(self.stats.last_error)
            else:
                raise LLMError("多次重试仍失败，请检查模型名与 base_url")

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                self.stats.errors += 1
                # 代理出错时也会返回 200 + 错误页，把内容带上便于排查
                raise LLMError(f"响应不是 JSON: {raw[:200]}") from exc

            choices = payload.get("choices") or []
            if not choices:
                self.stats.errors += 1
                raise LLMError(f"响应没有 choices: {raw[:200]}")

            choice = choices[0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            finish = str(choice.get("finish_reason") or "")

            self._record_stats(payload, started, finish, message)

            if content:
                return content

            # 正文为空 —— 最典型的原因就是思考把预算吃光了
            if finish == "length" and boost_round == 0:
                self.stats.truncated += 1
                budget = int(budget * TRUNCATION_BOOST)
                log.warning(
                    "生成被截断（%s：%d token 用尽，其中思考约 %d%s），自动扩到 %d 再试一次",
                    self.model_for(profile),
                    self._last_completion_tokens, self._last_reasoning_tokens,
                    "，估算值" if self._last_reasoning_estimated else "",
                    budget,
                )
                # 注意是 continue 不是 break：此处已在 attempt 循环之外，
                # break 会直接跳出 boost 循环，等于完全没重试
                continue

            self.stats.errors += 1
            if finish == "length":
                raise LLMTruncatedError(
                    f"token 预算被用尽（{self._last_completion_tokens} 个，其中思考 "
                    f"{self._last_reasoning_tokens} 个）—— 它还想再想一会儿。"
                    "调高对应档位的预算，或降低思维链强度"
                )
            if message.get("reasoning_content"):
                raise LLMError("模型只输出了思考内容，没有正文")
            raise LLMError("模型返回空文本")

        raise LLMTruncatedError("扩预算重试后仍被截断，请提高该档位的 token 预算")

    def _record_stats(self, payload: dict, started: float, finish: str,
                      message: dict | None = None) -> None:
        usage = payload.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)

        # 有些代理（实测 step-5-preview）根本不填 reasoning_tokens，
        # 于是日志显示"思考 0"，而实际预算全被思考吃光了 —— 会误导排查方向。
        # 保守推断：content 为空 = 这一轮基本都花在思考上。
        self._last_reasoning_estimated = False
        if reasoning == 0 and message:
            rtext = message.get("reasoning_content") or message.get("reasoning") or ""
            if rtext:
                content = message.get("content") or ""
                reasoning = completion if not content else max(1, len(rtext) // 2)
                self._last_reasoning_estimated = True

        self.stats.calls += 1
        self.stats.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.stats.tokens_out += int(usage.get("completion_tokens") or 0)
        self.stats.reasoning_tokens += reasoning
        self.stats.last_latency_ms = int((time.monotonic() - started) * 1000)
        self.stats.last_finish_reason = finish
        self._last_completion_tokens = completion
        self._last_reasoning_tokens = reasoning

    # ---- 多模态 -------------------------------------------------------

    async def chat_vision(
        self,
        system: str,
        prompt: str,
        data_uri: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """带图提问。data_uri 形如 `data:image/jpeg;base64,...`。

        **走 reply profile，不走 analysis**。

        看图是**感知任务**，不是推理任务。原来用 analysis，实测踩了两个坑：
          · 重思考模型（deepseek-flash）把 token 全烧在 reasoning 上，
            content 返回 0 字 —— 完全拿不到描述
          · 另一张图直接 504（网关超时）
        换成 reply 那个快模型后，同一张图 7.6 秒返回 274 字，
        准确认出了"录取通知书 + 从 450 提到 639 的成绩单"。
        """
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]
        # 看图不用 JSON 约束，很多视觉接口对它支持不一致
        return await self.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
            json_mode=False, profile="reply",
        )

    # ---- 结构化草稿 ---------------------------------------------------

    async def draft(
        self,
        system: str,
        user: str,
        *,
        retries: int = 1,
        profile: str = "reply",
    ) -> Draft:
        """要求模型输出 JSON 草稿。解析失败会带着错误信息重试一次。

        默认走 `reply` 画像：群聊要快。分析类用途传 profile="analysis"。
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                text = await self.chat(messages, json_mode=True, profile=profile)
                return to_draft(extract_json(text))
            except (LLMTruncatedError, LLMTimeoutError):
                # 预算/时间问题，补一句"格式不对"的提示也没用，直接放弃
                raise
            except LLMError as exc:
                last_error = exc
                log.warning("草稿解析失败（第 %d 次）: %s", attempt + 1, exc)
                if attempt < retries:
                    messages = messages + [
                        {"role": "assistant", "content": "（上次输出格式不合法）"},
                        {
                            "role": "user",
                            "content": "上一条回复不是合法 JSON。请只输出一个 JSON 对象，"
                            "不要任何解释、不要 markdown 围栏。",
                        },
                    ]

        raise LLMError(f"生成失败: {last_error}")

    # ---- 自检 ---------------------------------------------------------

    async def test(self) -> dict:
        """面板的"测试连接"按钮。"""
        started = time.monotonic()
        before_reason = self.stats.reasoning_tokens

        base = {
            "thinking": bool(self.cfg.keep_thinking),
            "reasoning_effort": (
                self.cfg.reasoning_effort
                if (self.cfg.keep_thinking and self._effort_param_ok) else ""
            ),
            "finish_reason": "",
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }

        try:
            reply = await self.chat(
                [
                    {"role": "system", "content": "你是一个测试助手，只回一句话。"},
                    {"role": "user", "content": "回复「连接正常」四个字，不要标点。"},
                ],
                max_tokens=max(self.cfg.reply_max_tokens, 1000),
                temperature=0.1,
                profile="reply",
            )
            return {
                **base,
                "ok": True,
                "reply": reply.strip(),
                "latency_ms": int((time.monotonic() - started) * 1000),
                "finish_reason": self.stats.last_finish_reason,
                "completion_tokens": self._last_completion_tokens,
                "reasoning_tokens": self.stats.reasoning_tokens - before_reason,
                "error": None,
            }
        except LLMError as exc:
            return {
                **base,
                "ok": False,
                "reply": "",
                "latency_ms": int((time.monotonic() - started) * 1000),
                "finish_reason": self.stats.last_finish_reason,
                "completion_tokens": self._last_completion_tokens,
                "reasoning_tokens": self.stats.reasoning_tokens - before_reason,
                "error": str(exc),
            }
