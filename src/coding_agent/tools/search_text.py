from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import (
    DEFAULT_IGNORED_DIRECTORY_NAMES,
    ToolContext,
    ToolExecutionError,
)

MAX_FILES_SCANNED = 5_000
MAX_FILE_BYTES = 1_000_000
MAX_RESULTS = 50
MAX_SNIPPET_CHARS = 300


class SearchTextArguments(StrictModel):
    query: str = Field(min_length=1, max_length=200, pattern=r"\S")
    path: str = Field(default=".", min_length=1, pattern=r"\S")
    case_sensitive: bool = False

    @field_validator("query")  # 表示此方法用于校验query字段
    @classmethod  # 普通方法变为类方法
    def validate_query(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("query must be a single line")

        return value


SEARCH_TEXT_SPEC = ToolSpec(
    name="search_text",
    description=(
        "Recursively search UTF-8 text files in the workspace for a literal "
        "text query. Returns matching file paths, line numbers, and snippets"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Literal single-line to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative file or directory to search. "
                    "Defaults to the workspace root."
                ),
                "default": ".",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "whether matching should be case-sensitive.",
                "default": False,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


def iter_candidate_files(target: Path) -> Iterator[Path]:
    if target.is_file():
        yield target
        return

    # walk方法每次产出一个三元组：当前目录、子目录名、文件名
    for current_directory, directory_names, file_names in target.walk(
        top_down=True,
        follow_symlinks=False,
    ):
        directory_names[:] = sorted(
            (
                name
                for name in directory_names
                if name.casefold() not in DEFAULT_IGNORED_DIRECTORY_NAMES
                and not (current_directory / name).is_symlink()
            ),
            key=lambda s: s.casefold(),
        )

        for file_name in sorted(file_names, key=lambda s: s.casefold()):
            candidate = current_directory / file_name

            if not candidate.is_symlink():
                yield candidate


class SearchTextTool:
    @property
    def spec(self) -> ToolSpec:
        return SEARCH_TEXT_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            parsed = SearchTextArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for search_text: {exc}"
            ) from exc

        target = context.resolve_workspace_path(parsed.path)

        if not target.exists():
            raise ToolExecutionError(f"search path does not exist: {parsed.path}")

        if not target.is_file() and not target.is_dir():
            raise ToolExecutionError(
                f"search path is not a file or a directory: {parsed.path}"
            )

        if (
            target.is_dir()
            and target.name.casefold() in DEFAULT_IGNORED_DIRECTORY_NAMES
        ):
            raise ToolExecutionError(f"directory is excluded: {parsed.path}")

        query = parsed.query if parsed.case_sensitive else parsed.query.casefold()

        matches: list[str] = []
        scanned_files = 0
        file_limit_reached = False
        result_limit_reached = False

        for candidate in iter_candidate_files(target):
            if scanned_files >= MAX_FILES_SCANNED:
                file_limit_reached = True
                break

            relative_path = candidate.relative_to(context.workspace_root).as_posix()

            try:
                safe_path = context.resolve_workspace_path(relative_path)
            except ToolExecutionError:
                continue

            scanned_files += 1

            try:
                if safe_path.stat().st_size > MAX_FILE_BYTES:
                    continue

                raw_content = safe_path.read_bytes()
            except OSError:
                continue

            if b"\x00" in raw_content:
                continue

            try:
                content = raw_content.decode("utf-8-sig")
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):
                searchable_line = line if parsed.case_sensitive else line.casefold()

                if query not in searchable_line:
                    continue

                snippet = line.strip()

                if len(snippet) > MAX_SNIPPET_CHARS:
                    snippet = snippet[:MAX_SNIPPET_CHARS] + "..."

                matches.append(f"{relative_path}:{line_number}: {snippet}")

                if len(matches) >= MAX_RESULTS:
                    result_limit_reached = True
                    break

            if result_limit_reached:
                break

        display_path = (
            "."
            if target == context.workspace_root
            else target.relative_to(context.workspace_root).as_posix()
        )

        if not matches:
            output = [
                f"No matches found for {parsed.query!r} under {display_path}.",
                f"Scanned {scanned_files} text-file candidates.",
            ]
        else:
            output = [
                f"Search results for {parsed.query!r} under {display_path}:",
                *matches,
            ]

        if result_limit_reached:
            output.append(f"...results truncated after {MAX_RESULTS} matches")

        if file_limit_reached:
            output.append(
                f"... search stopped after scanning {MAX_FILES_SCANNED} files"
            )

        return "\n".join(output)
