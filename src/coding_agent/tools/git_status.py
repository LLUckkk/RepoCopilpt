from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from coding_agent.approval import ApprovalDecision, ApprovalRequest
from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError
from coding_agent.tools.git_common import ensure_supported_git_repository, run_git


class GitStatusArguments(StrictModel):
    pass


GIT_STATUS_SPEC = ToolSpec(
    name="git_status",
    description=(
        "Show the current Git branch and working-tree status. "
        "This is read-only but still requires user approval. "
        "Use it before modifying files to identify existing user changes."
    ),
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
)


class GitStatusTool:
    @property
    def spec(self) -> ToolSpec:
        return GIT_STATUS_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            GitStatusArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for git_status: {exc}"
            ) from exc

        ensure_supported_git_repository(context)

        decision = await context.request_approval(
            ApprovalRequest(
                operation="git_status",
                summary="Inspect Git working-tree status",
                details=(
                    "Repository: workspace root\n"
                    "Command: git status --porcelain=v1 "  # 特殊输出模式，专门用来给脚本或程序解析
                    "--branch --untracked-files=all"
                ),
            )
        )

        if decision is not ApprovalDecision.APPROVED:
            raise ToolExecutionError("git status was denied by user")

        output = await run_git(
            arguments=(
                "status",
                "--porcelain=v1",
                "--branch",
                "--untracked-files=all",
            ),
            context=context,
            operation_name="git status",
        )

        if not output:
            return "Git working tree is clean."

        return output
