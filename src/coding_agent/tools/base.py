from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from coding_agent.domain import ToolSpec

DEFAULT_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


class ToolExecutionError(Exception):  # 表示继承Exception
    """工具无法安全或者正确执行。出现异常"""


@dataclass(frozen=True, slots=True)
class ToolContext:  # 工具上下文
    workspace_root: Path

    def __post_init__(
        self,
    ) -> None:  # dataclass的init是自动生成的，只能在pos_init里面添加逻辑
        root = Path(self.workspace_root).resolve(
            strict=True
        )  # resolve把路径解析为规范化的绝对路径

        if not root.is_dir():
            raise ValueError("workspace_root must be a directory!")

        object.__setattr__(self, "workspace_root", root)

    def resolve_workspace_path(self, raw_path: str) -> Path:
        """将工作区的相对路径解析为安全的绝对路径"""
        if not raw_path.strip():
            raise ToolExecutionError("path must not be blank")

        relative_path = Path(raw_path)

        if relative_path.is_absolute() or relative_path.drive:
            raise ToolExecutionError("absolute paths are not allowed")

        if ".." in relative_path.parts:
            raise ToolExecutionError("parent path components are not allowed")

        candidate = (self.workspace_root / relative_path).resolve(
            strict=False
        )  # 路径必须真实存在否则报错

        if not candidate.is_relative_to(self.workspace_root):
            raise ToolExecutionError("path escapes the workspace")

        relative_parts = candidate.relative_to(self.workspace_root).parts

        if any(part.casefold() == ".git" for part in relative_parts):
            raise ToolExecutionError("access to .git is not allowed")

        filename = candidate.name.casefold()
        blocked_names = {
            "credentials.json",
            "id_ed25519",
            "id_rsa",
            "service-account.json",
            ".npmrc",
            ".pypirc",
        }

        if filename == ".env" or filename.startswith(".env."):
            raise ToolExecutionError("access to environment files is not allowed")

        if filename in blocked_names:
            raise ToolExecutionError("access to credential files is not allowed")

        return candidate


class Tool(Protocol):
    @property  # 把一个方法变成属性来访问，在调用的时候不用写括号
    def spec(self) -> ToolSpec:
        """返回提供给模型的工具定义"""
        ...

    async def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> str:
        """执行工具并且返回文本结果"""
        ...
