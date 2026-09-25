from dataclasses import dataclass
from typing import Protocol

from coding_agent.context import ContextUsageEstimate
from coding_agent.domain import TokenUsage, ToolCall, ToolResult


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    step: int
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolExecutionFinished:
    step: int
    result: ToolResult
    elapsed_seconds: float


type AgentEvent = (
    ToolExecutionStarted
    | ToolExecutionFinished
    | ModelRequestStarted
    | ModelRequestFinished
)


class AgentEventHandler(Protocol):
    async def handle(self, event: AgentEvent) -> None:
        """Handle an event emitted by AgentLoop."""
        ...


@dataclass(frozen=True, slots=True)
class ModelRequestStarted:
    step: int
    context_usage: ContextUsageEstimate


@dataclass(frozen=True, slots=True)
class ModelRequestFinished:
    step: int
    usage: TokenUsage | None
    elapsed_seconds: float
