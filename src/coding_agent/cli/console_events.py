import json

import typer

from coding_agent.events import (
    AgentEvent,
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
                (f"[tool:start] step={event.step} {event.call.name} {arguments}"),
                fg=typer.colors.CYAN,
                err=True,
            )
            return

        if isinstance(event, ToolExecutionFinished):
            result = event.result

            if result.is_error:
                typer.secho(
                    (f"[tool:error] {result.name} ({event.elapsed_seconds:.3f}s)"),
                    fg=typer.colors.RED,
                    err=True,
                )
            else:
                typer.secho(
                    (f"[tool:ok] {result.name} ({event.elapsed_seconds:.3f}s)"),
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
