"""安全层：发送前硬拦截 + 输入侧注入防护。"""

from .filter import SafetyFilter, SafetyResult
from .injection import looks_like_injection, scrub

__all__ = ["SafetyFilter", "SafetyResult", "scrub", "looks_like_injection"]
