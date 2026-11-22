"""表情包系统：库 / 收藏规则 / 语义标注 / 选用。"""

from .collector import CollectVerdict, StickerCollector
from .ingest import StickerIngest
from .library import FetchedImage, StickerLibrary
from .selector import StickerSelector
from .tagger import StickerTagger

__all__ = [
    "StickerLibrary",
    "FetchedImage",
    "StickerCollector",
    "CollectVerdict",
    "StickerTagger",
    "StickerSelector",
    "StickerIngest",
]
