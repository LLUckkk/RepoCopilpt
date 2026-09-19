import typer

from coding_agent.approval import ApprovalDecision, ApprovalRequest


class ConsoleApprovalHandler:
    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        typer.secho(
            f"\nApproval required: {request.summary}",
            fg=typer.colors.YELLOW,
            bold=True,
            err=True,
        )

        if request.details:
            typer.echo(request.details, err=True)

        approved = typer.confirm(
            "Approve this operation?",
            default=False,
        )

        if approved:
            return ApprovalDecision.APPROVED

        return ApprovalDecision.DENIED
