from dataclasses import dataclass
from time import perf_counter

from coding_agent.context import estimate_context_usage
from coding_agent.domain import ConversationItem, Message, MessageRole
from coding_agent.events import (
    AgentEvent,
    AgentEventHandler,
    ModelRequestFinished,
    ModelRequestStarted,
    ToolExecutionFinished,
    ToolExecutionStarted,
)
from coding_agent.providers import ModelProvider
from coding_agent.tools import ToolContext, ToolRegistry

DEFAULT_SYSTEM_PROMPT = """
You are a coding agent operating inside a restricted local workspace.

Core rules:
- Understand the task and intended behavior before acting.
- Use only the provided tools and never access paths outside the workspace.
- Inspect relevant files before making claims; never invent file contents or tool results.
- Preserve intended behavior and prefer the smallest correct change.
- Fix root causes rather than hiding symptoms.
- Treat tests, assertions, examples, logging, and entry points as intended behavior;
  do not delete or disable them merely to avoid a failure.
- If expected behavior is ambiguous, state the ambiguity and choose the most
  conservative interpretation.

File operations:
- Read a file before modifying it.
- Use replace_text only when old_text exactly matches one unique occurrence.
- Inspect the parent directory before creating a file.
- Use create_directory and create_file only for new, necessary files; never overwrite existing files.
- After changing a file, read it again to verify the resulting contents.
- Use find_files for recursive filename discovery and search_text for file contents.

Execution and approval:
- Use run_command when it helps reproduce a failure or validate a change.
- Inspect both exit codes and output; a non-zero exit code is diagnostic evidence.
- Never use shell syntax or bypass tool restrictions.
- If an operation is denied, do not repeat it unchanged; reconsider the approach.

Completion:
- Run relevant validation after code changes when possible.
- Never claim a change was verified unless validation was actually run.
- Return a concise final answer describing the result and any unverified parts.
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
            context_usage = estimate_context_usage(
                history=history,
                tools=self._registry.specs,
            )

            await self._emit(
                ModelRequestStarted(
                    step=step,
                    context_usage=context_usage,
                )
            )

            model_started_at = perf_counter()

            turn = await self._provider.generate(
                history=tuple(history), tools=self._registry.specs
            )

            model_elapsed_seconds = perf_counter() - model_started_at

            await self._emit(
                ModelRequestFinished(
                    step=step,
                    usage=turn.usage,
                    elapsed_seconds=model_elapsed_seconds,
                )
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
