from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    operation: str  # 机器可识别的操作类型
    summary: str  # 操作的简短描述
    details: str  # 完整diff或者命令内容


class ApprovalHandler(Protocol):
    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        """Ask the user whether a sensitive operation may continue."""
        ...
