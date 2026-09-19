from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import (
    DEFAULT_IGNORED_DIRECTORY_NAMES,
    ToolContext,
    ToolExecutionError,
)

MAX_DIRECTORY_ENTRIES = 200


class ListDirectoryArguments(StrictModel):
    path: str = Field(default=".", min_length=1, pattern=r"\S")


LIST_DIRECTORY_SPEC = ToolSpec(
    name="list_directory",
    description=(
        "List the immediate files and directories inside a workspace directory"
        "Common dependency and cache directories are omitted"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative directory path. Default to the workspace root"
                ),
                "default": ".",
            },
        },
        "additionalProperties": False,
    },
)


class ListDirectoryTool:
    @property
    def spec(self) -> ToolSpec:
        return LIST_DIRECTORY_SPEC

    async def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> str:
        try:
            parsed = ListDirectoryArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for list_directory: {exc}"
            ) from exc

        target = context.resolve_workspace_path(parsed.path)

        if not target.exists():
            raise ToolExecutionError(f"directory does not exist: {parsed.path}")

        if not target.is_dir():
            raise ToolExecutionError(f"path is not a directory: {parsed.path}")

        if target.name.casefold() in DEFAULT_IGNORED_DIRECTORY_NAMES:
            raise ToolExecutionError(f"directory is excluded: {parsed.path}")

        try:
            children = sorted(
                target.iterdir(),
                key=lambda child: child.name.casefold(),
            )
        except OSError as exc:
            raise ToolExecutionError(
                f"failed to list directory: {parsed.path}"
            ) from exc

        entries: list[str] = []
        truncated = False

        for child in children:
            if child.name.casefold() in DEFAULT_IGNORED_DIRECTORY_NAMES:
                continue

            relative_path = child.relative_to(context.workspace_root).as_posix()

            try:
                safe_path = context.resolve_workspace_path(relative_path)
            except ToolExecutionError:
                continue

            if len(entries) >= MAX_DIRECTORY_ENTRIES:
                truncated = True
                break

            if child.is_symlink():
                entries.append(f"symlink    {relative_path}")
            elif safe_path.is_dir():
                entries.append(f"directory  {relative_path}/")
            elif safe_path.is_file():
                entries.append(f"file       {relative_path}")

        display_path = (
            "."
            if target == context.workspace_root
            else target.relative_to(context.workspace_root).as_posix()
        )

        if not entries:
            return f"{display_path} / (empty directory)"

        output = [
            f"{display_path}/ ({len(entries)} entries)",
            *entries,
        ]

        if truncated:
            output.append(f"... output truncated after {MAX_DIRECTORY_ENTRIES} entries")

        return "\n".join(output)
