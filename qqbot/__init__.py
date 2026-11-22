"""QQ 拟人角色机器人。

分层：gateway(收) -> pipeline(编排) -> buffer/decide -> generate -> safety -> sender(发)
管理面板：web/（aiohttp.web + 纯静态前端）
"""

from .config import VERSION

__version__ = VERSION
