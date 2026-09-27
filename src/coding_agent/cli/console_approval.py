from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.text import Text

from coding_agent.approval import ApprovalDecision, ApprovalRequest
from coding_agent.cli.ui import ui_console

DIFF_OPERATIONS = frozenset(
    {
        "replace_text",
        "create_file",
    }
)


def _render_details(request: ApprovalRequest) -> RenderableType:
    if request.operation in DIFF_OPERATIONS:
        return Syntax(
            request.details,
            "diff",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
    return Text(
        request.details,
        style="value",
        overflow="fold",
    )


class ConsoleApprovalHandler:
    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        operation = Text()
        operation.append("operation ", style="label")
        operation.append(request.operation, style="tool.name")

        summary = Text(request.summary, style="bold white")

        content: list[RenderableType] = [
            operation,
            Text(),
            summary,
        ]

        if request.details:
            content.extend(
                [
                    Text(),
                    _render_details(request),
                ]
            )

        ui_console.print()
        ui_console.print(
            Panel(
                Group(*content),
                title="[warning]Approval required[/warning]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        approved = Confirm.ask(
            "Approve this operation?",
            default=False,
            console=ui_console,
        )

        if approved:
            ui_console.print("  ✓ Approved", style="success")
            return ApprovalDecision.APPROVED

        ui_console.print("  × Denied", style="error")
        return ApprovalDecision.DENIED
