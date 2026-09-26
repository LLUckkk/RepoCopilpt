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


@dataclass(frozen=True, slots=True)
class ModelRequestStarted:
    step: int
    context_usage: ContextUsageEstimate


@dataclass(frozen=True, slots=True)
class ModelRequestFinished:
    step: int
    usage: TokenUsage | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ContextBudgetWarning:
    step: int
    context_usage: ContextUsageEstimate
    max_context_tokens: int
    warning_ratio: float


@dataclass(frozen=True, slots=True)
class ContextCompacted:
    step: int
    before_usage: ContextUsageEstimate
    after_usage: ContextUsageEstimate
    removed_blocks: int
    retained_blocks: int
    target_reached: bool


type AgentEvent = (
    ToolExecutionStarted
    | ToolExecutionFinished
    | ModelRequestStarted
    | ModelRequestFinished
    | ContextBudgetWarning
    | ContextCompacted
)


class AgentEventHandler(Protocol):
    async def handle(self, event: AgentEvent) -> None:
        """Handle an event emitted by AgentLoop."""
        ...
