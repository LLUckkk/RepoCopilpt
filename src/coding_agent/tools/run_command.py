import asyncio
import locale
import os
import subprocess
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from coding_agent.approval import ApprovalDecision, ApprovalRequest
from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError

ALLOWED_COMMANDS = frozenset({"python", "pytest"})

MAX_ARGUMENTS = 32
MAX_ARGUMENT_LENGTH = 2_000
MAX_STREAM_CHARS = 10_000
MAX_TIMEOUT_SECONDS = 120

SENSITIVE_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class RunCommandArguments(StrictModel):
    command: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    args: list[str] = Field(default_factory=list, max_length=MAX_ARGUMENTS)
    cwd: str = Field(
        default=".",
        min_length=1,
        max_length=1_024,
        pattern=r"\S",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=MAX_TIMEOUT_SECONDS,
    )

    @field_validator("args")
    @classmethod
    def validate_args(cls, args: list[str]) -> list[str]:
        for argument in args:
            if len(argument) > MAX_ARGUMENT_LENGTH:
                raise ValueError(
                    f"each argument must contain at most {MAX_ARGUMENT_LENGTH} characters"
                )

            if any(ord(character) < 32 for character in argument):
                raise ValueError(
                    "command arguments must not contain control characters"
                )

        return args


RUN_COMMAND_SPEC = ToolSpec(
    name="run_command",
    description=(
        "Run an approved Python program or pytest command inside the workspace. "
        "The command is executed directly without a shell. "
        "Use it to  reproduce errors and validate fixes."
        "A non-zero exit code is a command result and should be inspected."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["python", "pytest"],
                "description": "Allowed executable name",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Argument list. Each argument must be a separate item. "
                    "Do not use shell syntax such as &&, |, >, or <."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Workspace relative working directory. Defaults to '.'.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_SECONDS,
                "description": "Maximum command runtime in seconds.",
            },
        },
        "required": ["command", "args"],
        "additionalProperties": False,
    },
)


# 复制环境变量，且去除敏感的环境变量
def _sanitized_environment() -> dict[str, str]:
    environment = os.environ.copy()

    for name in list(environment):
        upper_name = name.upper()

        if any(marker in upper_name for marker in SENSITIVE_ENV_MARKERS):
            environment.pop(name, None)

    environment["PYTHONUNBUFFERED"] = "1"  # stdout/stderr尽量不要缓冲，及时输出
    return environment


def _decode_output(content: bytes) -> str:
    if not content:
        return ""

    encodings = ("utf-8", locale.getpreferredencoding(False))

    for encoding in dict.fromkeys(encodings):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    return content.decode("utf-8", errors="replace")


def _truncate_output(content: str) -> str:
    if len(content) <= MAX_STREAM_CHARS:
        return content

    marker = "\n...[output truncated]...\n"
    remaining = MAX_STREAM_CHARS - len(marker)
    start_length = remaining // 2
    end_length = remaining - start_length

    return content[:start_length] + marker + content[-end_length:]


def _format_result(
    *,
    exit_code: int | None,
    stdout: bytes,
    stderr: bytes,
) -> str:
    stdout_text = _truncate_output(_decode_output(stdout))
    stderr_text = _truncate_output(_decode_output(stderr))

    return "\n".join(
        [
            f"Exit code: {exit_code}",
            "stdout:",
            stdout_text or "(empty)",
            "stderr:",
            stderr_text or "(empty)",
        ]
    )


def _validate_python_command(
    *,
    args: list[str],
    working_directory: Path,
    context: ToolContext,
) -> None:
    if not args:
        raise ToolExecutionError(
            "python requires a workspace-relative script or '-m pytest"
        )
    if args[0] == "-m":
        if len(args) < 2 or args[1] != "pytest":
            raise ToolExecutionError(
                "only 'python -m pytest is allowed with the -m option'"
            )
        return
    if args[0].startswith("-"):
        raise ToolExecutionError(
            "python options such as -c and reading code from stdin are not allowed"
        )
    script_argument = Path(args[0])

    if script_argument.is_absolute() or script_argument.drive:
        raise ToolExecutionError("Python script paths must be workspace-relative")

    working_directory_relative = working_directory.relative_to(context.workspace_root)
    script_relative = working_directory_relative / script_argument

    script_path = context.resolve_workspace_path(
        script_relative.as_posix(),
        allow_symlinks=False,
    )

    if script_path.suffix.casefold() != ".py":
        raise ToolExecutionError("python may only execute .py files")
    if not script_path.is_file():
        raise ToolExecutionError(f"python script does not exist: {args[0]}")


class RunCommandTool:
    @property
    def spec(self) -> ToolSpec:
        return RUN_COMMAND_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            parsed = RunCommandArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for run_command: {exc}"
            ) from exc

        if parsed.command not in ALLOWED_COMMANDS:
            raise ToolExecutionError(f"command is not allowed: {parsed.command}")

        working_directory = context.resolve_workspace_path(
            parsed.cwd,
            allow_symlinks=False,
        )

        if not working_directory.is_dir():
            raise ToolExecutionError(f"working directory does not exist: {parsed.cwd}")

        if parsed.command == "python":
            _validate_python_command(
                args=parsed.args, working_directory=working_directory, context=context
            )

        argv = [parsed.command, *parsed.args]
        command_text = subprocess.list2cmdline(argv)
        relative_cwd = working_directory.relative_to(context.workspace_root).as_posix()

        decision = await context.request_approval(
            ApprovalRequest(
                operation="run_command",
                summary=f"Run command: {command_text}",
                details=(
                    f"Working directory: {relative_cwd or '.'} \n"
                    f"Timeout: {parsed.timeout_seconds} seconds\n"
                    f"Command: {command_text}"
                ),
            )
        )

        if decision is not ApprovalDecision.APPROVED:
            raise ToolExecutionError("run_command was denied by the user")

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=working_directory,
                env=_sanitized_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                f"command was not found: {parsed.command}"
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"failed to start command: {parsed.command}"
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=parsed.timeout_seconds,
            )
        except TimeoutError as exc:
            with suppress(ProcessLookupError):
                process.kill()

            stdout, stderr = await process.communicate()
            partial_result = _format_result(
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )

            raise ToolExecutionError(
                f"command timed out after "
                f"{parsed.timeout_seconds} seconds\n{partial_result}"
            ) from exc

        return _format_result(
            exit_code=process.returncode,
            stderr=stderr,
            stdout=stdout,
        )
