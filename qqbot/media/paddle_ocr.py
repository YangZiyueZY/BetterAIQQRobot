"""PaddleOCR-VL 云端 OCR 客户端。

和 NapCat 自带的 `ocr_image` 是两类东西：

    ocr_image    同步、免费、秒回，只有纯文字
    PaddleOCR    任务制（提交 → 轮询 → 取 JSONL）、走云端、**很慢**，
                 但输出 markdown：公式变 LaTeX、表格保留结构

实测同一个接口：**快的时候 6 秒，慢的时候 94 秒**。所以这里有两条路径：

    recognize()  —— 同步等结果。只在"愿意为这张图等"的场合用
    submit()     —— 只提交，拿 jobId 就走。配合后台轮询使用（推荐）

因为任务是异步的，jobId 会落库（`ocr_jobs` 表），进程重启后还能接着收结果。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from ..config import PaddleOCRConfig

log = logging.getLogger(__name__)

#: PaddleOCR 会在正文前面加一行 `## 0000000000` 当页码标记，不是内容
PAGE_MARKER = re.compile(r"^#{1,6}\s*\d{6,}\s*$", re.MULTILINE)

#: 状态机
PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"
TERMINAL = (DONE, FAILED)


@dataclass(slots=True)
class JobState:
    job_id: str
    state: str
    jsonl_url: str | None = None
    error: str = ""
    total_pages: int = 0
    extracted_pages: int = 0


class PaddleOCRError(RuntimeError):
    pass


class PaddleOCRClient:
    def __init__(self, cfg: PaddleOCRConfig) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.timed_out = 0
        self.cache_hits = 0

    # ---- 生命周期 -----------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    @property
    def _headers(self) -> dict[str, str]:
        # 注意是小写 bearer —— 这个接口就是这么写的
        return {"Authorization": f"bearer {self.cfg.token}"}

    @staticmethod
    def _mime(path: Path) -> str:
        return {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }.get(path.suffix.lower(), "application/octet-stream")

    def _optional_payload(self) -> str:
        return json.dumps({
            "useDocOrientationClassify": bool(self.cfg.doc_orientation),
            "useDocUnwarping": bool(self.cfg.doc_unwarping),
            "useChartRecognition": bool(self.cfg.chart_recognition),
        })

    # ---- 提交 ---------------------------------------------------------

    async def submit(self, image: Path | str) -> str | None:
        """提交一张本地图，返回 jobId。失败返回 None（不抛，交给调用方决定）。"""
        if not self.cfg.configured:
            log.debug("PaddleOCR 未配置（缺 PADDLEOCR_TOKEN）")
            return None

        path = Path(image)
        if not path.is_file():
            log.warning("PaddleOCR 找不到文件: %s", path)
            return None

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("model", self.cfg.model)
        form.add_field("optionalPayload", self._optional_payload())
        try:
            form.add_field(
                "file", path.read_bytes(),
                filename=path.name, content_type=self._mime(path),
            )
        except OSError as exc:
            log.warning("PaddleOCR 读文件失败: %s", exc)
            return None

        try:
            async with session.post(
                self.cfg.base_url, data=form, headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self.cfg.submit_timeout),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    log.warning("PaddleOCR 提交失败 HTTP %s: %s", resp.status, body[:200])
                    self.failed += 1
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("PaddleOCR 提交异常: %s", exc)
            self.failed += 1
            return None

        try:
            job_id = json.loads(body)["data"]["jobId"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("PaddleOCR 响应结构异常: %s / %s", exc, body[:200])
            self.failed += 1
            return None

        self.submitted += 1
        log.info("PaddleOCR 任务已提交: %s (%s)", job_id, path.name)
        return str(job_id)

    # ---- 轮询 ---------------------------------------------------------

    async def poll(self, job_id: str) -> JobState:
        """查一次状态。不抛异常，失败也返回一个 JobState。"""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.cfg.base_url}/{job_id}", headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self.cfg.submit_timeout),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    return JobState(job_id, FAILED, error=f"HTTP {resp.status}: {body[:120]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            # 网络抖动不算任务失败，保持 pending 下次再查
            return JobState(job_id, RUNNING, error=f"轮询异常: {exc}")

        try:
            data = json.loads(body).get("data") or {}
        except json.JSONDecodeError:
            return JobState(job_id, RUNNING, error="响应不是 JSON")

        state = str(data.get("state") or RUNNING)
        progress = data.get("extractProgress") or {}
        out = JobState(
            job_id=job_id,
            state=state,
            error=str(data.get("errorMsg") or ""),
            total_pages=int(progress.get("totalPages") or 0),
            extracted_pages=int(progress.get("extractedPages") or 0),
        )
        if state == DONE:
            out.jsonl_url = ((data.get("resultUrl") or {}).get("jsonUrl")) or None
        return out

    async def wait(self, job_id: str, *, timeout: float | None = None) -> str | None:
        """一直轮询到完成，返回 markdown。超时/失败返回 None。"""
        timeout = timeout if timeout is not None else self.cfg.job_timeout
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            await asyncio.sleep(self.cfg.poll_interval)
            st = await self.poll(job_id)
            if st.state == DONE:
                if not st.jsonl_url:
                    log.warning("PaddleOCR 完成但没给 resultUrl")
                    self.failed += 1
                    return None
                markdown = await self.fetch_markdown(st.jsonl_url)
                if markdown:
                    self.completed += 1
                else:
                    self.failed += 1
                return markdown
            if st.state == FAILED:
                log.warning("PaddleOCR 任务失败: %s", st.error)
                self.failed += 1
                return None

        self.timed_out += 1
        log.warning("PaddleOCR 任务超时（%.0fs）: %s", timeout, job_id)
        return None

    # ---- 取结果 -------------------------------------------------------

    async def fetch_markdown(self, jsonl_url: str) -> str:
        """下载 JSONL 并拼出 markdown 文本。

        刻意**不下载图片** —— 群聊里只要文字，图片纯属浪费流量。
        """
        session = await self._get_session()
        try:
            async with session.get(
                jsonl_url, timeout=aiohttp.ClientTimeout(total=self.cfg.submit_timeout)
            ) as resp:
                if resp.status != 200:
                    log.warning("PaddleOCR 结果下载失败 HTTP %s", resp.status)
                    return ""
                text = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("PaddleOCR 结果下载异常: %s", exc)
            return ""

        chunks: list[str] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line).get("result") or {}
            except json.JSONDecodeError:
                continue
            for page in result.get("layoutParsingResults") or []:
                md = ((page.get("markdown") or {}).get("text")) or ""
                if md.strip():
                    chunks.append(md.strip())

        joined = "\n\n".join(chunks)
        joined = PAGE_MARKER.sub("", joined).strip()
        if len(joined) > self.cfg.max_chars:
            joined = joined[: self.cfg.max_chars] + "…"
        return joined

    # ---- 一步到位（愿意等的时候用） -----------------------------------

    async def recognize(self, image: Path | str, *,
                        timeout: float | None = None) -> str | None:
        job_id = await self.submit(image)
        if not job_id:
            return None
        return await self.wait(job_id, timeout=timeout)

    def stats(self) -> dict:
        return {
            "paddle_submitted": self.submitted,
            "paddle_completed": self.completed,
            "paddle_failed": self.failed,
            "paddle_timed_out": self.timed_out,
        }
