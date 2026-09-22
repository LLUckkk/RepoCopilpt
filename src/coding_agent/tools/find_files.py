from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, ValidationError, field_validator

from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import (
    DEFAULT_IGNORED_DIRECTORY_NAMES,
    ToolContext,
    ToolExecutionError,
)

MAX_FILES_SCANNED = 10_000
MAX_RESULTS = 100
MAX_OUTPUT_CHARS = 20_000


class FindFilesArguments(StrictModel):
    pattern: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"\S",
    )
    path: str = Field(
        default=".",
        min_length=1,
        max_length=1_024,
        pattern=r"\S",
    )
    case_sensitive: bool = False

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        normalized = value.replace("\\", "/")

        if any(ord(character) < 32 for character in normalized):
            raise ValueError("pattern must not contain control characters")

        pattern_path = Path(normalized)

        if pattern_path.is_absolute() or pattern_path.drive:
            raise ValueError("pattern must be relative to the search directory")

        if ".." in PurePosixPath(normalized).parts:
            raise ValueError("pattern must not contain parent path components")

        return normalized


FIND_FILES_SPEC = ToolSpec(
    name="find_files",
    description=(
        "Recursively find files by filename or workspace-relative glob pattern. "
        "Use '*.py' to find Python files at any depth, or a path pattern such "
        "as 'src/**/*.py' to restrict matches. "
        "Ignored directories and symbolic links are excluded."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Relative glob pattern, such as '*.py', "
                    "'test_*.py', or 'src/**/*.py'."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative directory in which to search. "
                    "Defaults to the workspace root."
                ),
                "default": ".",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": (
                    "Whether filename matching is case-sensitive. Defaults to false."
                ),
                "default": False,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
)


def _matches_pattern(
    *,
    relative_path: str,
    pattern: str,
    case_sensitive: bool,
) -> bool:
    candidate = relative_path
    effective_pattern = pattern

    if not case_sensitive:
        candidate = candidate.casefold()
        effective_pattern = effective_pattern.casefold()

    pure_path = PurePosixPath(candidate)

    # 没有目录分隔符时，只匹配文件名。
    # 因为遍历本身是递归的，所以 *.py 会匹配任意深度。
    if "/" not in effective_pattern:
        return fnmatchcase(
            pure_path.name,
            effective_pattern,
        )

    return pure_path.match(effective_pattern)


class FindFilesTool:
    @property
    def spec(self) -> ToolSpec:
        return FIND_FILES_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            parsed = FindFilesArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for find_files: {exc}"
            ) from exc

        target = context.resolve_workspace_path(
            parsed.path,
            allow_symlinks=False,
        )

        if not target.exists():
            raise ToolExecutionError(f"search directory does not exist: {parsed.path}")

        if not target.is_dir():
            raise ToolExecutionError(f"search path is not a directory: {parsed.path}")

        if target.name.casefold() in DEFAULT_IGNORED_DIRECTORY_NAMES:
            raise ToolExecutionError(f"directory is excluded: {parsed.path}")

        matches: list[str] = []
        scanned_files = 0
        output_chars = 0
        scan_limit_reached = False
        result_limit_reached = False
        output_limit_reached = False

        for (
            current_directory,
            directory_names,
            file_names,
        ) in target.walk(
            top_down=True,
            follow_symlinks=False,
        ):
            directory_names[:] = sorted(
                (
                    name
                    for name in directory_names
                    if (
                        name.casefold() not in DEFAULT_IGNORED_DIRECTORY_NAMES
                        and not (current_directory / name).is_symlink()
                    )
                ),
                key=str.casefold,
            )

            for file_name in sorted(
                file_names,
                key=str.casefold,
            ):
                if scanned_files >= MAX_FILES_SCANNED:
                    scan_limit_reached = True
                    break

                candidate = current_directory / file_name

                if candidate.is_symlink():
                    continue

                scanned_files += 1

                relative_path = candidate.relative_to(context.workspace_root).as_posix()

                try:
                    safe_path = context.resolve_workspace_path(
                        relative_path,
                        allow_symlinks=False,
                    )
                except ToolExecutionError:
                    # .env、凭据文件和内部目录等不会出现在结果中。
                    continue

                if not safe_path.is_file():
                    continue

                relative_to_search_root = safe_path.relative_to(target).as_posix()

                if not _matches_pattern(
                    relative_path=relative_to_search_root,
                    pattern=parsed.pattern,
                    case_sensitive=parsed.case_sensitive,
                ):
                    continue

                if len(matches) >= MAX_RESULTS:
                    result_limit_reached = True
                    break

                additional_chars = len(relative_path) + 1

                if output_chars + additional_chars > MAX_OUTPUT_CHARS:
                    output_limit_reached = True
                    break

                matches.append(relative_path)
                output_chars += additional_chars

            if scan_limit_reached or result_limit_reached or output_limit_reached:
                break

        display_path = (
            "."
            if target == context.workspace_root
            else target.relative_to(context.workspace_root).as_posix()
        )

        if matches:
            output = [
                (f"Files matching {parsed.pattern!r} under {display_path}:"),
                *matches,
                f"Scanned {scanned_files} file entries.",
            ]
        else:
            output = [
                (f"No files matching {parsed.pattern!r} under {display_path}."),
                f"Scanned {scanned_files} file entries.",
            ]

        if scan_limit_reached:
            output.append(f"... scan stopped after {MAX_FILES_SCANNED} files")

        if result_limit_reached:
            output.append(f"... results truncated after {MAX_RESULTS} matches")

        if output_limit_reached:
            output.append(
                f"... results truncated after {MAX_OUTPUT_CHARS} output characters"
            )

        return "\n".join(output)
