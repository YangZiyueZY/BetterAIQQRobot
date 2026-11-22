"""发送层：分条发送 + 模拟打字延迟（Phase 2/3）。

注意：包名故意不叫 typing，避免遮蔽标准库 typing。
"""

from .typing_sim import TypingSender

__all__ = ["TypingSender"]
