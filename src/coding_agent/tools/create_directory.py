from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from coding_agent.approval import ApprovalDecision, ApprovalRequest
from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError


class CreateDirectoryArguments(StrictModel):
    path: str = Field(
        min_length=1,
        max_length=1_024,
        pattern=r"\S",
    )


CREATE_DIRECTORY_SPEC = ToolSpec(
    name="create_directory",
    description=(
        "Create one new directory inside the workspace. "
        "The parent directory must already exist. "
        "The user must approve the operation before creation."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "workspace-relative path of the new directory. "
                    "The parent directory must already exist. "
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


class CreateDirectoryTool:
    @property
    def spec(self) -> ToolSpec:
        return CREATE_DIRECTORY_SPEC

    async def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> str:
        try:
            parsed = CreateDirectoryArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for create_directory: {exc}"
            ) from exc

        target = context.resolve_workspace_path(parsed.path, allow_symlinks=False)

        if target == context.workspace_root:
            raise ToolExecutionError("cannot create the workspace root")
        if target.exists():
            raise ToolExecutionError(f"path already exists: {parsed.path}")
        if not target.parent.exists():
            raise ToolExecutionError(
                f"parent directory does not exist: {target.parent.name}"
            )
        if not target.parent.is_dir():
            raise ToolExecutionError(
                f"parent path is not a directory: {target.parent.name}"
            )

        relative_path = target.relative_to(context.workspace_root).as_posix()

        decision = await context.request_approval(
            ApprovalRequest(
                operation="create_directory",
                summary=f"Create directory {relative_path}",
                details=(
                    f"Directory: {relative_path}\n"
                    f"Parent: "
                    f"{target.parent.relative_to(context.workspace_root).as_posix() or '.'}"
                ),
            )
        )

        if decision is not ApprovalDecision.APPROVED:
            raise ToolExecutionError("create_file is denied by the user.")

        current_target = context.resolve_workspace_path(
            parsed.path, allow_symlinks=False
        )

        if current_target != target:
            raise ToolExecutionError("directory path changed while awaiting approval")
        if not target.parent.is_dir():
            raise ToolExecutionError("parent directory changed while awaiting approval")

        try:
            target.mkdir()
        except FileExistsError as exc:
            raise ToolExecutionError(
                f"path was created while awaiting approval: {relative_path}"
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"failed to create directory: {relative_path}"
            ) from exc

        return f"Created directory: {relative_path}"
