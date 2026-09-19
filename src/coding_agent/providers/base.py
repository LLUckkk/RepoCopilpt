from collections.abc import Sequence
from typing import Protocol

from coding_agent.domain import ConversationItem, ModelTurn, ToolSpec

"""
定义协议，所有model provider需要实现这个签名的generate方法
"""


class ModelProviderError(RuntimeError):
    """模型provider调用或者响应解析失败"""


class ModelProvider(Protocol):
    async def generate(
        self,
        history: Sequence[ConversationItem],
        tools: Sequence[ToolSpec],
    ) -> ModelTurn:
        """根据对话历史和工具列表生成下一轮的响应"""
        ...
