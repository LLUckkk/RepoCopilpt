from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):  # 继承basemodel，复用数据验证能力
    model_config = ConfigDict(extra="forbid")  # 禁止出现任何未定义的字段


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(StrictModel):  # 在定义message的时候不能出现任何未定义的变量
    kind: Literal["message"] = "message"  # kind这个字段只能有唯一值message
    role: MessageRole
    content: str = Field(min_length=1, pattern=r"\S")  # field限制不能为空字符串


class ToolCall(StrictModel):  # 模型提出的工具调用请求
    kind: Literal["tool_call"] = "tool_call"
    call_id: str = Field(min_length=1, pattern=r"\S")
    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):  # Runtime执行工具之后的结果
    kind: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1, pattern=r"\S")
    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    output: str
    is_error: bool = False


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelTurn(StrictModel):  # 模型一次响应的完整结果
    final_text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def validate_outcome(
        self,
    ) -> Self:  # 这里的Self是一个类型注解，说明返回值和self是同一个类型
        if self.final_text is not None and not self.final_text.strip():
            raise ValueError("final_text must contain non-whitespace characters")

        has_final_text = self.final_text is not None
        has_tool_calls = bool(self.tool_calls)

        if has_tool_calls == has_final_text:
            raise ValueError(
                "ModelTurn must contain exactly one of final_text or tool_calls"
            )

        call_ids = [tool_call.call_id for tool_call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique within a model turn")

        return self  # pydantic校验器强制要求的写法，将校验后的对象返回


class ToolSpec(StrictModel):
    """提供给模型的工具说明"""

    kind: Literal["tool_spec"] = "tool_spec"
    name: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    description: str = Field(min_length=1, pattern=r"\S")
    input_schema: dict[str, Any]


type ConversationItem = Message | ModelTurn | ToolResult
