"""决定层：该不该说话。

两个组件：
  · `Decider`      —— 手调线性打分器，免费瞬时，负责一票否决与先验分数
  · `HybridDecider` —— 在打分器的"灰区"分数上追加 LLM 判定

管线只跟 `HybridDecider` 打交道。
"""

from .llm_decider import HybridDecider, LLMDecider, LLMVerdict
from .scorer import Decider

__all__ = ["Decider", "HybridDecider", "LLMDecider", "LLMVerdict"]
