import os
from collections.abc import Mapping
from contextlib import suppress
from difflib import unified_diff
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from coding_agent.approval import ApprovalDecision, ApprovalRequest
from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError

MAX_CONTENT_CHARS = 100_000
MAX_FILE_BYTES = 1_000_000


class CreateFileArguments(StrictModel):
    path: str = Field(
        min_length=1,
        max_length=1_024,
        pattern=r"\S",
    )
    content: str = Field(
        max_length=MAX_CONTENT_CHARS,
    )


CREATE_FILE_SPEC = ToolSpec(
    name="create_file",
    description=(
        "Create one new UTF-8 text file inside the workspace. "
        "The file must not already exist and its parent directory must exist. "
        "The user must approve the full diff before the file is created."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative path of the new file. "
                    "The parent directory must already exist."
                ),
            },
            "content": {
                "type": "string",
                "description": "Complete initial UTF-8 contents of the new file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)


def _create_diff(*, relative_path: str, content: str) -> str:
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")

    diff_lines = unified_diff(
        [],
        normalized_content.splitlines(),
        fromfile="/dev/null",
        tofile=f"b/{relative_path}",
        lineterm="",
    )

    diff = "\n".join(diff_lines)

    if not diff:
        return f"Create empty file: {relative_path}"

    return diff


def _write_new_file_exclusively(
    *,
    target: Path,
    content: bytes,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL  # 文件打开标志

    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        file_descriptor = os.open(target, flags, 0o666)
    except FileExistsError as exc:
        raise ToolExecutionError(f"path already exists: {target.name}") from exc
    except OSError:
        raise ToolExecutionError(f"failed to create file: {target.name}")

    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except OSError as exc:
        with suppress(OSError):
            target.unlink()

        raise ToolExecutionError(f"failed to write new file: {target.name}") from exc


class CreateFileTool:
    @property
    def spec(self) -> ToolSpec:
        return CREATE_FILE_SPEC

    async def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> str:
        try:
            parsed = CreateFileArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for create_file: {exc}"
            ) from exc

        target = context.resolve_workspace_path(
            parsed.path,
            allow_symlinks=False,
        )

        if target.exists():
            raise ToolExecutionError(f"path already exists: {parsed.path}")
        if not target.parent.is_dir():
            raise ToolExecutionError(
                f"parent path is not a directory: {target.parent.name}"
            )

        encoded_content = parsed.content.encode("utf-8")

        if len(encoded_content) > MAX_FILE_BYTES:
            raise ToolExecutionError(f"new file would exceed {MAX_FILE_BYTES} bytes")

        relative_path = target.relative_to(context.workspace_root).as_posix()

        diff = _create_diff(
            relative_path=relative_path,
            content=parsed.content,
        )

        decision = await context.request_approval(
            ApprovalRequest(
                operation="create_file",
                summary=f"Create {relative_path}",
                details=diff,
            )
        )

        if decision is not ApprovalDecision.APPROVED:
            raise ToolExecutionError("create_file was denied by user")

        _write_new_file_exclusively(
            target=target,
            content=encoded_content,
        )

        return f"Created {relative_path} with {len(encoded_content)} bytes"
