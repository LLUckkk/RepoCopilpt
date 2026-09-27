from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

CODING_AGENT_THEME = Theme(
    {
        "brand": "bold bright_cyan",
        "label": "bold bright_black",
        "value": "white",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "muted": "bright_black",
        "agent": "bold bright_cyan",
        "tool.name": "bold bright_blue",
        "tool.args": "bright_black",
        "context": "magenta",
        "duration": "bright_black",
    }
)

ui_console = Console(
    stderr=True,
    theme=CODING_AGENT_THEME,
    highlight=False,
)

result_console = Console(
    theme=CODING_AGENT_THEME,
    highlight=False,
)


def render_header(*, workspace: Path, model: str) -> None:
    information = Table.grid(
        padding=(0, 2),
    )
    information.add_column(style="label", justify="right")
    information.add_column(style="value")

    information.add_row("workspace", str(workspace))
    information.add_row("model", model)

    ui_console.print()
    ui_console.print(
        Panel(
            information,
            title="[brand]Coding Agent[/brand]",
            subtitle="[muted]restricted local workspace[/muted]",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    ui_console.print()


def render_final_answer(*, text: str, steps: int) -> None:
    result_console.print()
    result_console.print(
        Panel(
            Markdown(text),
            title="[success]✓ Completed[/success]",
            subtitle=f"[muted]{steps} model step(s)[/muted]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
