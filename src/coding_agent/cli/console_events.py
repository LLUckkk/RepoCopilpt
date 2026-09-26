import json

import typer

from coding_agent.events import (
    AgentEvent,
    ContextBudgetWarning,
    ContextCompacted,
    ContextSummaryFailed,
    ContextSummaryFinished,
    ContextSummaryStarted,
    ModelRequestFinished,
    ModelRequestStarted,
    ToolExecutionFinished,
    ToolExecutionStarted,
)

MAX_ARGUMENT_DISPLAY_CHARS = 400
MAX_OUTPUT_DISPLAY_CHARS = 2_000


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...[truncated]"


class ConsoleEventHandler:
    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose

    async def handle(self, event: AgentEvent) -> None:
        if isinstance(event, ContextBudgetWarning):
            estimated_tokens = event.context_usage.estimated_tokens
            utilization = estimated_tokens / event.max_context_tokens
            remaining_tokens = event.max_context_tokens - estimated_tokens
            if remaining_tokens >= 0:
                remaining_text = f"remaining={remaining_tokens}"
                color = typer.colors.YELLOW
            else:
                remaining_text = f"over_by={abs(remaining_tokens)}"
                color = typer.colors.RED
            typer.secho(
                (
                    f"[context:warning] step={event.step} "
                    f"estimated={estimated_tokens} "
                    f"budget={event.max_context_tokens} "
                    f"usage={utilization:.1%} "
                    f"{remaining_text}"
                ),
                fg=color,
                bold=True,
                err=True,
            )
            return
        if isinstance(event, ModelRequestStarted):
            if self._verbose:
                usage = event.context_usage

                typer.secho(
                    (
                        f"[context] step={event.step} "
                        f"estimated={usage.estimated_tokens} "
                        f"history={usage.history_tokens} "
                        f"tools={usage.tool_tokens} "
                        f"items={usage.history_items}"
                    ),
                    fg=typer.colors.MAGENTA,
                    err=True,
                )
            return

        if isinstance(event, ModelRequestFinished):
            if self._verbose:
                if event.usage is None:
                    usage_text = "usage=unavailable"
                else:
                    usage_text = (
                        f"prompt={event.usage.prompt_tokens} "
                        f"completion={event.usage.completion_tokens} "
                        f"total={event.usage.total_tokens}"
                    )
                typer.secho(
                    (
                        f"[model] step={event.step} "
                        f"{usage_text} "
                        f"({event.elapsed_seconds:.3f}s)"
                    ),
                    fg=typer.colors.BLUE,
                    err=True,
                )
            return

        if isinstance(event, ToolExecutionStarted):
            arguments = json.dumps(
                event.call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            arguments = _truncate(
                arguments,
                MAX_ARGUMENT_DISPLAY_CHARS,
            )

            typer.secho(
                f"[tool:start] step={event.step} {event.call.name} {arguments}",
                fg=typer.colors.CYAN,
                err=True,
            )
            return

        if isinstance(event, ToolExecutionFinished):
            result = event.result

            if result.is_error:
                typer.secho(
                    f"[tool:error] {result.name} ({event.elapsed_seconds:.3f}s)",
                    fg=typer.colors.RED,
                    err=True,
                )
            else:
                typer.secho(
                    f"[tool:ok] {result.name} ({event.elapsed_seconds:.3f}s)",
                    fg=typer.colors.GREEN,
                    err=True,
                )

            if result.is_error or self._verbose:
                output = _truncate(
                    result.output,
                    MAX_OUTPUT_DISPLAY_CHARS,
                )

                for line in output.splitlines():
                    typer.echo(f"    {line}", err=True)
            return

        if isinstance(event, ContextCompacted):
            before = event.before_usage.estimated_tokens
            after = event.after_usage.estimated_tokens
            saved = before - after

            status = "target_reached" if event.target_reached else "best-effort"

            typer.secho(
                (
                    f"[context:compact] step={event.step} "
                    f"blocks_removed={event.removed_blocks} "
                    f"blocks_retained={event.retained_blocks} "
                    f"tokens={before}->{after} "
                    f"saved={saved} "
                    f"status={status}"
                ),
                fg=typer.colors.MAGENTA,
                bold=True,
                err=True,
            )
            return

        if isinstance(event, ContextSummaryStarted):
            typer.secho(
                (
                    f"[context:summary:start] step={event.step} "
                    f"new_blocks={event.new_blocks}"
                ),
                fg=typer.colors.BLUE,
                err=True,
            )
            return

        if isinstance(event, ContextSummaryFinished):
            if event.usage is None:
                usage_text = "usage=unavailable"
            else:
                usage_text = f"tokens={event.usage.total_tokens}"

            typer.secho(
                (
                    f"[context:summary:ok] step={event.step} "
                    f"total_blocks={event.total_summarized_blocks} "
                    f"memory_chars={event.memory_chars} "
                    f"{usage_text} "
                    f"({event.elapsed_seconds:.3f}s)"
                ),
                fg=typer.colors.GREEN,
                err=True,
            )
            return

        if isinstance(event, ContextSummaryFailed):
            typer.secho(
                (
                    f"[context:summary:error] step={event.step} "
                    f"attempted_blocks={event.attempted_blocks} "
                    f"error={event.error}"
                ),
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            return
