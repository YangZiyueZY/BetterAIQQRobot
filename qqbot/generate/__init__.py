"""生成层：prompt 组装 + LLM 调用 + 结构化输出 + 去 AI 味后处理。"""

from .llm import LLMClient, LLMError, LLMStats, extract_json, to_draft
from .postprocess import PostResult, postprocess, postprocess_bubbles
from .prompt import build_memory_prompt, build_system, build_user

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMStats",
    "extract_json",
    "to_draft",
    "postprocess",
    "postprocess_bubbles",
    "PostResult",
    "build_system",
    "build_user",
    "build_memory_prompt",
]
