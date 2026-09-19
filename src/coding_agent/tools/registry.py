from collections.abc import Iterable

from coding_agent.domain import ToolCall, ToolResult, ToolSpec
from coding_agent.tools.base import Tool, ToolContext, ToolExecutionError


class ToolRegistry:
    """保存可用工具，且负责分发模型生成的工具调用"""

    def __init__(self, tools: Iterable[Tool]) -> None:
        tool_map: dict[str, Tool] = {}
        specs: list[ToolSpec] = []

        for tool in tools:
            spec = tool.spec

            if spec.name in tool_map:
                raise ValueError(f"duplicate tool name: {spec.name}")

            tool_map[spec.name] = tool
            specs.append(spec)

        self._tools = tool_map
        self._specs = tuple(specs)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """返回需要提供给模型的工具定义"""
        return self._specs

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext,
    ) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=f"Unknown tool: {call.name}",
                is_error=True,
            )

        try:
            output = await tool.execute(call.arguments, context)

            if not isinstance(output, str):
                raise TypeError("tool output must be a string")

        except ToolExecutionError as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=str(exc),
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Registry 是隔离任意工具异常的边界，必须防止单个工具击穿 AgentLoop。
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=f"Internal tool error: {type(exc).__name__}",
                is_error=True,
            )

        return ToolResult(
            call_id=call.call_id, name=call.name, output=output, is_error=False
        )
