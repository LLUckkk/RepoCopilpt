import subprocess
from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from coding_agent.approval import ApprovalDecision, ApprovalRequest
from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError
from coding_agent.tools.git_common import ensure_supported_git_repository, run_git


class GitDiffArguments(StrictModel):
    path: str | None = Field(default=None, min_length=1, max_length=1024, pattern=r"\S")
    staged: bool = False


GIT_DIFF_SPEC = ToolSpec(
    name="git_diff",
    description=(
        "Show Git changes as a unified diff. By default it shows unstaged "
        "changes; set staged=true to show staged changes. An optional "
        "workspace-relative path can restrict the result. This tool does not "
        "show untracked file contents; use git_status and read_file for those. "
        "The operation is read-only but requires user approval."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional literal workspace-relative file or directory path",
            },
            "staged": {
                "type": "boolean",
                "description": "If ture, show staged changes. If false, show unstaged changes.",
                "default": False,
            },
        },
        "additionalProperties": False,
    },
)


class GitDiffTool:
    @property
    def spec(self) -> ToolSpec:
        return GIT_DIFF_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            parsed = GitDiffArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(f"Invalid arguments for git_diff: {exc}") from exc

        ensure_supported_git_repository(context)

        relative_path: str | None = None
        if parsed.path is not None:
            target = context.resolve_workspace_path(parsed.path, allow_symlinks=False)
            relative_path = target.relative_to(context.workspace_root).as_posix()

        git_arguments: list[str] = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--color=never",
            "--ignore-submodules=all",
            "--unified=3",
        ]

        if parsed.staged:
            git_arguments.append("--cached")

        git_arguments.append("--")
        git_arguments.append(relative_path or ".")

        command_text = subprocess.list2cmdline(
            [
                "git",
                "-c",
                "core.quotepath=false",  # 关闭 Git 对非 ASCII 字符（比如中文）路径的转义输出，让它们能直接正常显示。
                *git_arguments,
            ]
        )

        scope = relative_path or "entire workspace"
        diff_kind = "staged" if parsed.staged else "unstaged"

        decision = await context.request_approval(
            ApprovalRequest(
                operation="git_diff",
                summary=f"Inspect {diff_kind} Git changes",
                details=(f"Scope: {scope}\nCommand: {command_text}"),
            )
        )

        if decision is not ApprovalDecision.APPROVED:
            raise ToolExecutionError("git_diff was denied by the user")

        output = await run_git(
            arguments=git_arguments, context=context, operation_name="git diff"
        )

        if not output:
            return f"No {diff_kind} Git changes found for {scope}"

        return output
