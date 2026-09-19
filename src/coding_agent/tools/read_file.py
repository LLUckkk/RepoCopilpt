from collections.abc import Mapping
from typing import Any, Self

from pydantic import Field, ValidationError, model_validator

from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError

MAX_FILE_BYTES = 1_000_000
MAX_READ_LINES = 400


class ReadFileArguments(StrictModel):
    path: str = Field(min_length=1, pattern=r"\S")
    start_line: int = Field(default=1, ge=1)  # ge是>=
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line is None:
            # 开放区间
            return self
        if self.end_line < self.start_line:
            raise ValueError("end_line must not be smaller than start_line")

        requested_lines = self.end_line - self.start_line + 1
        if requested_lines > MAX_READ_LINES:
            raise ValueError(f"read_file can return at most {MAX_READ_LINES} lines")

        return self


READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description=(
        "Read a UTF-8 text file inside the workspace and return numbered lines"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "workspace-relative file path.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "First line to return. Default to 1",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Last line to return, inclusive",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


class ReadFileTool:
    @property
    def spec(self) -> ToolSpec:
        return READ_FILE_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            parsed = ReadFileArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(f"Invalid arguments for read_file: {exc}") from exc

        target = context.resolve_workspace_path(parsed.path)

        if not target.exists():
            raise ToolExecutionError(f"file does not exist: {parsed.path}")

        if not target.is_file():
            raise ToolExecutionError(f"path is not a file: {parsed.path}")

        try:
            file_size = target.stat().st_size

            if file_size > MAX_FILE_BYTES:
                raise ToolExecutionError(f"file is larger than {MAX_FILE_BYTES} bytes")

            raw_content = target.read_bytes()
        except ToolExecutionError:
            raise
        except OSError as exc:
            raise ToolExecutionError(f"failed to read file: {parsed.path}") from exc

        if b"\x00" in raw_content:
            raise ToolExecutionError("binary files are not supported")

        try:
            content = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ToolExecutionError("file is not valid UTF-8 text")

        relative_name = target.relative_to(context.workspace_root).as_posix()
        lines = content.splitlines()

        if not lines:
            return f"{relative_name} (empty file)"

        if parsed.start_line > len(lines):
            raise ToolExecutionError(
                f"start_line exceeds file length: {len(lines)} lines"
            )

        requested_end = parsed.end_line or (parsed.start_line + MAX_READ_LINES - 1)
        actual_end = min(requested_end, len(lines))

        selected_lines = lines[parsed.start_line - 1 : actual_end]
        number_width = max(4, len(str(actual_end)))

        numbered_content = "\n".join(
            f"{line_number:>{number_width}} | {line}"
            for line_number, line in enumerate(
                selected_lines,
                start=parsed.start_line,  # enumerate的计数从start-line开始
            )
        )

        return (
            f"{relative_name} "
            f"(lines {parsed.start_line}-{actual_end} of {len(lines)})\n"
            f"{numbered_content}"
        )
