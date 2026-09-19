from dataclasses import dataclass
from typing import Protocol

from coding_agent.domain import ToolCall, ToolResult


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    step: int
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolExecutionFinished:
    step: int
    result: ToolResult
    elapsed_seconds: float


type AgentEvent = ToolExecutionStarted | ToolExecutionFinished


class AgentEventHandler(Protocol):
    async def handle(self, event: AgentEvent) -> None:
        """Handle an event emitted by AgentLoop."""
        ...
