import os
import shutil
import tempfile
from codecs import BOM_UTF8
from collections.abc import Mapping
from contextlib import suppress
from difflib import unified_diff
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from coding_agent.approval import (
    ApprovalDecision,
    ApprovalRequest,
)
from coding_agent.domain import StrictModel, ToolSpec
from coding_agent.tools.base import ToolContext, ToolExecutionError

MAX_FILE_BYTES = 1_000_000
MAX_EDIT_TEXT_CHARS = 50_000


class ReplaceTextArguments(StrictModel):
    path: str = Field(
        min_length=1,
        max_length=1_024,
        pattern=r"\S",
    )
    old_text: str = Field(
        min_length=1,
        max_length=MAX_EDIT_TEXT_CHARS,
    )
    new_text: str = Field(
        max_length=MAX_EDIT_TEXT_CHARS,
    )


REPLACE_TEXT_SPEC = ToolSpec(
    name="replace_text",
    description=(
        "Replace exactly one occurrence of text in an existing UTF-8 file. "
        "The user must approve the generated diff before the change is written. "
        "Read the file first and include enough surrounding text to make old_text unique."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path of the existing file.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text currently present in the file. It must occurs exactly only once.",
            },
            "new_text": {
                "type": "string",
                "description": "Text that should replace old_text.",
            },
        },
        "required": [
            "path",
            "old_text",
            "new_text",
        ],
        "additionalProperties": False,
    },
)


def _detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _convert_newlines(text: str, newline: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", newline)


def _normalize_for_diff(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _create_diff(
    *,
    relative_path: str,
    original_text: str,
    updated_text: str,
) -> str:
    original_lines = _normalize_for_diff(original_text).splitlines()
    updated_lines = _normalize_for_diff(updated_text).splitlines()

    diff_lines = unified_diff(
        original_lines,
        updated_lines,
        fromfile=f"a/{relative_path}",
        tofile=f"b/{relative_path}",
        lineterm="",
    )

    return "\n".join(diff_lines)


# 有的文件携带有BOM特殊标记，如果直接保存就会丢失，要做一个标记
def _encode_text(text: str, *, has_utf8_bom: bool) -> bytes:
    encoded = text.encode("utf-8")

    if has_utf8_bom:
        return BOM_UTF8 + encoded

    return encoded


def _atomic_replace(
    *,
    target: Path,
    content: bytes,
) -> None:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}",
            suffix=".coding-agent.tmp",
            delete=False,
        ) as temporary_file:
            (temporary_file.write(content),)
            temporary_file.flush()  # 缓冲数据交给操作系统
            os.fsync(temporary_file.fileno())  # 操作系统把相关数据同步到存储设备
            temporary_path = Path(temporary_file.name)
        shutil.copymode(target, temporary_path)  # 复制原来文件的权限模式
        os.replace(temporary_path, target)
        temporary_path = None

    except OSError as exc:
        raise ToolExecutionError(
            f"failed to write file atomically: {target.name}"
        ) from exc

    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()  # 无论失败还是成功，清理临时文件


class ReplaceTextTool:
    @property
    def spec(self) -> ToolSpec:
        return REPLACE_TEXT_SPEC

    async def execute(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> str:
        try:
            parsed = ReplaceTextArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for replace_text: {exc}"
            ) from exc

        target = context.resolve_workspace_path(
            parsed.path,
            allow_symlinks=False,
        )

        if not target.exists():
            raise ToolExecutionError(f"file does not exist: {parsed.path}")

        try:
            original_bytes = target.read_bytes()
        except OSError as exc:
            raise ToolExecutionError(f"failed to read file: {parsed.path}") from exc

        if len(original_bytes) > MAX_FILE_BYTES:
            raise ToolExecutionError(f"file is larger than {MAX_FILE_BYTES} bytes")

        if b"\x00" in original_bytes:
            raise ToolExecutionError("binary file are not supported")

        has_utf8_bom = original_bytes.startswith(BOM_UTF8)

        try:
            original_text = original_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError("file is not valid UTF-8 text") from exc

        newline = _detect_newline(original_text)
        old_text = _convert_newlines(parsed.old_text, newline)
        new_text = _convert_newlines(parsed.new_text, newline)

        if old_text == new_text:
            raise ToolExecutionError("old_text and new_text are identical")

        occurrence_count = original_text.count(old_text)

        if occurrence_count == 0:
            raise ToolExecutionError(
                "old_text was not found; read the file again "
                "and provide its exact current contents"
            )

        if occurrence_count > 1:
            raise ToolExecutionError(
                "old_text occurs more than once; include more "
                "surrounding context to make it unique"
            )

        updated_text = original_text.replace(
            old_text,
            new_text,
            1,
        )

        if not updated_text.strip():
            raise ToolExecutionError("replace_text cannot empty the entire file")

        updated_bytes = _encode_text(
            updated_text,
            has_utf8_bom=has_utf8_bom,
        )

        if len(updated_bytes) > MAX_FILE_BYTES:
            raise ToolExecutionError(
                f"updated file would exceed {MAX_FILE_BYTES} bytes"
            )

        relative_path = target.relative_to(context.workspace_root).as_posix()

        diff = _create_diff(
            relative_path=relative_path,
            original_text=original_text,
            updated_text=updated_text,
        )

        decision = await context.request_approval(
            ApprovalRequest(
                operation="replace_text",
                summary=f"Modify {relative_path}",
                details=diff,
            )
        )

        if decision is not ApprovalDecision.APPROVED:
            raise ToolExecutionError("replace_text was denied by the user")

        # 用户查看diff的时候可能同时在编辑器中修改文件，再检查一次
        try:
            current_bytes = target.read_bytes()
        except OSError as exc:
            raise ToolExecutionError(f"failed to re-read file: {parsed.path}") from exc

        if current_bytes != original_bytes:
            raise ToolExecutionError(
                "file changed while awaiting approval; read it again before retrying."
            )

        _atomic_replace(
            target=target,
            content=updated_bytes,
        )

        return f"Replaced one exact occurrence in {relative_path}"
