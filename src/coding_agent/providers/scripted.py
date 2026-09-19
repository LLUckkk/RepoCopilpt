from collections import deque
from collections.abc import Iterable, Sequence

from coding_agent.domain import ConversationItem, ModelTurn, ToolSpec

"""
modelprovider的具体实现
"""


class ScriptedModelProvider:
    """按照预先给定的顺序返回模型响应"""

    def __init__(self, turns: Iterable[ModelTurn]) -> None:
        self._turns = deque(turns)  # 双端序列
        self.received_histories: list[tuple[ConversationItem, ...]] = []
        self.received_tools: list[tuple[ToolSpec, ...]] = []

    async def generate(
        self,
        history: Sequence[
            ConversationItem
        ],  # Sequence是一个抽象基类，代表有序、可索引迭代有长度的序列，例如list和tuple
        # 依赖倒置，只声明最小的必要能力，而不规定底层的具体实现
        tools: Sequence[ToolSpec],
    ) -> ModelTurn:
        self.received_histories.append(tuple(history))
        self.received_tools.append(tuple(tools))

        if not self._turns:
            raise RuntimeError("ScriptedModelProvider has no turns!")

        # TODO:这里暂时写一个假的，还没有连上大模型

        return self._turns.popleft()
