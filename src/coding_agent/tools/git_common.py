import asyncio
import locale
import os
from collections.abc import Sequence
from contextlib import suppress

from coding_agent.tools.base import ToolContext, ToolExecutionError

GIT_TIMEOUT_SECONDS = 10
MAX_GIT_OUTPUT_CHARS = 20_000

SENSITIVE_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def ensure_supported_git_repository(
    context: ToolContext,
) -> None:
    git_directory = context.workspace_root / ".git"

    if git_directory.is_symlink():
        raise ToolExecutionError("symbolic Git directories are not supported")
    if not git_directory.is_dir():
        raise ToolExecutionError("workspace root is not a supported Git repository")


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()

    for name in list(environment):
        upper_name = name.upper()

        if any(marker in upper_name for marker in SENSITIVE_ENV_MARKERS):
            environment.pop(name, None)

    environment["GIT_OPTIONAL_LOCKS"] = "0"  # 禁止试图获取锁文件
    environment["GIT_TERMINAL_PROMPT"] = "0"  # 禁止在终端上弹出交互提示
    environment["GIT_LITERAL_PATHSPECS"] = "1"  # 将路径参数当做字面路径，忽略通配符
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
    if len(content) <= MAX_GIT_OUTPUT_CHARS:
        return content

    marker = "\n...[git output truncated]...\n"
    remaining = MAX_GIT_OUTPUT_CHARS - len(marker)

    return content[:remaining] + marker


async def run_git(
    *,
    arguments: Sequence[str],
    context: ToolContext,
    operation_name: str,
) -> str:
    ensure_supported_git_repository(context)

    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.quotepath=false",
            *arguments,
            cwd=context.workspace_root,
            env=_git_environment(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError("git executable was not found") from exc
    except OSError as exc:
        raise ToolExecutionError(f"failed to start {operation_name}") from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        with suppress(ProcessLookupError):
            process.kill()

        await process.communicate()

        raise ToolExecutionError(f"{operation_name} timed out") from exc

    stdout_text = _truncate_output(_decode_output(stdout).strip())
    stderr_text = _truncate_output(_decode_output(stderr).strip())

    if process.returncode != 0:
        detail = stderr_text or stdout_text or "unknown error"
        raise ToolExecutionError(
            f"{operation_name} failed with exit code {process.returncode}: {detail}"
        )

    return stdout_text
