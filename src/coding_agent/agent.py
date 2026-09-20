from dataclasses import dataclass
from time import perf_counter

from coding_agent.domain import ConversationItem, Message, MessageRole
from coding_agent.events import (
    AgentEvent,
    AgentEventHandler,
    ToolExecutionFinished,
    ToolExecutionStarted,
)
from coding_agent.providers import ModelProvider
from coding_agent.tools import ToolContext, ToolRegistry

DEFAULT_SYSTEM_PROMPT = """
You are a coding agent operating inside a restricted local workspace.

Understand the user's task and the intended behavior before acting.
Use only the tools provided to you.
Inspect relevant files before drawing conclusions.
Never invent file contents or tool results.
Do not attempt to access paths outside the workspace.

When fixing a bug:
- Identify and fix the root cause, not merely the visible symptom.
- Preserve the program's intended behavior unless the user requests otherwise.
- Do not delete or disable failing examples, assertions, tests, logging,
  or entry-point code merely to make an error disappear.
- Prefer the smallest change that correctly addresses the root cause.
- If the expected behavior is ambiguous, explain the ambiguity and choose
  the most conservative behavior instead of removing functionality.

Before modifying a file, read its current contents.
Use replace_text only when the exact old text occurs only once.
After modifying a file, read it again and verify the resulting contents.
Never retry the same write operation after the user denies approval.

When the task is complete, return a concise final answer.
""".strip()


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    final_text: str
    steps: int
    history: tuple[ConversationItem, ...]  # tuple表示只读不可变


class AgentStepLimitError(RuntimeError):
    def __init__(
        self,
        max_steps: int,
        history: tuple[ConversationItem, ...],
    ) -> None:
        super().__init__(f"agent did not finish within {max_steps} model steps")
        self.max_steps = max_steps
        self.history = history


class AgentLoop:
    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
        event_handler: AgentEventHandler | None = None,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be blank")

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self._provider = provider
        self._registry = registry
        self._context = context
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._event_handler = event_handler

    async def run(self, task: str) -> AgentRunResult:
        if not task.strip():
            raise ValueError("task must not be blank")

        history: list[ConversationItem] = [
            Message(
                role=MessageRole.SYSTEM,
                content=self._system_prompt,
            ),
            Message(
                role=MessageRole.USER,
                content=task,
            ),
        ]

        for step in range(1, self._max_steps + 1):
            turn = await self._provider.generate(
                history=tuple(history), tools=self._registry.specs
            )
            history.append(turn)

            if turn.final_text is not None:
                return AgentRunResult(
                    final_text=turn.final_text, steps=step, history=tuple(history)
                )

            for tool_call in turn.tool_calls:
                await self._emit(
                    ToolExecutionStarted(
                        step=step,
                        call=tool_call,
                    )
                )
                started_at = perf_counter()

                tool_result = await self._registry.execute(
                    call=tool_call,
                    context=self._context,
                )

                elapsed_seconds = perf_counter() - started_at
                history.append(tool_result)

                await self._emit(
                    ToolExecutionFinished(
                        step=step, result=tool_result, elapsed_seconds=elapsed_seconds
                    )
                )

        raise AgentStepLimitError(
            max_steps=self._max_steps,
            history=tuple(history),
        )

    async def _emit(self, event: AgentEvent) -> None:
        if self._event_handler is not None:
            await self._event_handler.handle(event)
