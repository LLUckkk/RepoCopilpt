import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from coding_agent.domain import (
    ConversationItem,
    ModelTurn,
    ToolSpec,
)

BASE_REQUEST_OVERHEAD_TOKENS = 16
HISTORY_ITEM_OVERHEAD_TOKENS = 6
TOOL_SPEC_OVERHEAD_TOKENS = 12


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    estimated_tokens: int
    history_tokens: int
    tool_tokens: int
    history_items: int
    tool_count: int


def _estimate_text_tokens(text: str) -> int:
    ascii_characters = 0
    non_ascii_units = 0

    for character in text:
        if character.isascii():
            ascii_characters += 1
        else:
            non_ascii_units += max(1, len(character.encode("utf-8")) // 2)

    return ceil(ascii_characters / 4) + non_ascii_units


def _serialize_conversation_item(item: ConversationItem) -> str:
    if isinstance(item, ModelTurn):
        payload = item.model_dump(
            mode="json",
            exclude={"usage"},
            exclude_none=True,
        )
    else:
        payload = item.model_dump(
            mode="json",
            exclude_none=True,
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _serialize_tool_spec(spec: ToolSpec) -> str:
    return json.dumps(
        spec.model_dump(
            mode="json",
            exclude_none=True,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def estimate_context_usage(
    *,
    history: Sequence[ConversationItem],
    tools: Sequence[ToolSpec],
) -> ContextUsageEstimate:
    history_tokens = sum(
        _estimate_text_tokens(_serialize_conversation_item(item))
        + HISTORY_ITEM_OVERHEAD_TOKENS
        for item in history
    )

    tool_tokens = sum(
        _estimate_text_tokens(_serialize_tool_spec(spec)) + TOOL_SPEC_OVERHEAD_TOKENS
        for spec in tools
    )

    estimate_tokens = BASE_REQUEST_OVERHEAD_TOKENS + history_tokens + tool_tokens

    return ContextUsageEstimate(
        estimated_tokens=estimate_tokens,
        history_tokens=history_tokens,
        tool_tokens=tool_tokens,
        history_items=len(history),
        tool_count=len(tools),
    )
