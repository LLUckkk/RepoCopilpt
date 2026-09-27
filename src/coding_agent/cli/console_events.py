import json

from rich.padding import Padding
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from coding_agent.cli.ui import ui_console
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
    marker = "...[truncated]"
    return text[: max_chars - len(marker)] + marker


class ConsoleEventHandler:
    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self._status: Status | None = None

    def close(self) -> None:
        """stop any active terminal animation"""
        self._stop_status()

    def _start_status(self, label: str) -> None:
        self._stop_status()

        self._status = Status(
            Text(label, style="agent"),
            console=ui_console,
            spinner="dots",
        )

        self._status.start()

    def _stop_status(self) -> None:
        if self._status is None:
            return

        self._status.stop()
        self._status = None

    def _print_tool_output(
        self,
        *,
        output: str,
        is_error: bool,
    ) -> None:
        rendered_output = Text(
            _truncate(output, MAX_OUTPUT_DISPLAY_CHARS),
            style="error" if is_error else "value",
        )
        ui_console.print(
            Padding(
                Panel(
                    rendered_output,
                    border_style="red" if is_error else "bright_black",
                    padding=(0, 1),
                ),
                (0, 0, 0, 4),
            )
        )

    async def handle(self, event: AgentEvent) -> None:
        if isinstance(event, ContextBudgetWarning):
            estimated = event.context_usage.estimated_tokens
            utilization = estimated / event.max_context_tokens

            message = Text()
            message.append("Context usage warning", style="warning")
            message.append(
                f"\n{estimated} / {event.max_context_tokens} estimated tokens "
                f"({utilization:.1%})",
                style="value",
            )
            ui_console.print(
                Panel.fit(
                    message,
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
            return
        if isinstance(event, ModelRequestStarted):
            self._start_status(f"Thinking · step {event.step}")
            return

        if isinstance(event, ModelRequestFinished):
            self._stop_status()
            if self._verbose:
                if event.usage is None:
                    usage_text = "usage=unavailable"
                else:
                    usage_text = (
                        f"{event.usage.total_tokens} tokens"
                        f"({event.usage.prompt_tokens} prompt + "
                        f"{event.usage.completion_tokens} completion)"
                    )
                line = Text("  ◦ ", style="muted")
                line.append("Model response", style="agent")
                line.append(f" · {usage_text}", style="muted")
                line.append(
                    f" · {event.elapsed_seconds:.2f}s",
                    style="duration",
                )
                ui_console.print(line)
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

            line = Text("  ├─ ", style="muted")
            line.append(event.call.name, style="tool.name")
            line.append(" ")
            line.append(arguments, style="tool.args")
            ui_console.print(line)
            return

        if isinstance(event, ToolExecutionFinished):
            result = event.result

            line = Text("  ╰─ ", style="muted")

            if result.is_error:
                line.append("failed", style="error")
            else:
                line.append("completed", style="success")

            line.append(
                f" · {event.elapsed_seconds:.2f}s",
                style="duration",
            )
            ui_console.print(line)

            if result.is_error or self._verbose:
                self._print_tool_output(
                    output=result.output,
                    is_error=result.is_error,
                )
            return

        if isinstance(event, ContextCompacted):
            before = event.before_usage.estimated_tokens
            after = event.after_usage.estimated_tokens
            saved = before - after

            line = Text("  ◦ ", style="muted")
            line.append("Context compacted", style="context")
            line.append(
                f" · {before} → {after} tokens",
                style="muted",
            )
            line.append(
                f" · saved {saved}",
                style="muted",
            )

            if not event.target_reached:
                line.append(" · best effort", style="warning")

            ui_console.print(line)
            return

        if isinstance(event, ContextSummaryStarted):
            self._start_status(f"Compressing context · {event.new_blocks} new block(s)")
            return

        if isinstance(event, ContextSummaryFinished):
            self._stop_status()

            line = Text("  ◦ ", style="muted")
            line.append("Context summarized", style="context")
            line.append(
                f" · {event.total_summarized_blocks} blocks",
                style="muted",
            )
            line.append(
                f" · {event.memory_chars} chars",
                style="muted",
            )
            if event.usage is not None:
                line.append(
                    f" · {event.usage.total_tokens} tokens",
                    style="muted",
                )
            line.append(
                f" · {event.elapsed_seconds:.2f}s",
                style="duration",
            )

            ui_console.print(line)
            return

        if isinstance(event, ContextSummaryFailed):
            self._stop_status()

            message = Text("Context summary failed", style="error")
            message.append(
                f" · {event.error}",
                style="value",
            )
            ui_console.print(message)
            return
