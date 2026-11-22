"""OneBot v11 接入层：连接 NapCat、归一化事件、封装 API。"""

from .api import OneBotAPI, OneBotError
from .client import Gateway
from .events import parse_group_message

__all__ = ["Gateway", "OneBotAPI", "OneBotError", "parse_group_message"]
