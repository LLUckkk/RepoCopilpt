import json
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from coding_agent.domain import (
    ConversationItem,
    Message,
    ModelTurn,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from coding_agent.providers.base import ModelProviderError


def _history_to_chat_message(
    history: Sequence[ConversationItem],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    for item in history:
        if isinstance(item, Message):
            messages.append({"role": item.role.value, "content": item.content})
            continue
        if isinstance(item, ModelTurn):
            if item.final_text is not None:
                messages.append({"role": "assistant", "content": item.final_text})
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        for call in item.tool_calls
                    ],
                }
            )
            continue
        if isinstance(item, ToolResult):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )
            continue

        raise ModelProviderError(
            f"unsupported conversation item: {type(item).__name__}"
        )

    return messages


def _tool_specs_to_chat_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }
        for spec in tools
    ]


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if base_url is not None and not base_url.strip():
            raise ValueError("base_url must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    async def generate(
        self,
        history: Sequence[ConversationItem],
        tools: Sequence[ToolSpec],
    ) -> ModelTurn:
        messages = _history_to_chat_message(history)
        chat_tools = _tool_specs_to_chat_tools(tools)

        try:
            if chat_tools:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=chat_tools,
                    tool_choice="auto",
                )
            else:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                )
        except OpenAIError as exc:
            raise ModelProviderError(
                f"model request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not response.choices:
            raise ModelProviderError("model response did not contain any choices")

        token_usage: TokenUsage | None = None

        if response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            total_tokens = response.usage.total_tokens or 0

            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens

            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        message = response.choices[0].message

        if message.tool_calls:
            tool_calls: list[ToolCall] = []

            try:
                for remote_call in message.tool_calls:
                    if remote_call.type != "function":
                        raise ModelProviderError(
                            "only function tool calls are supported."
                        )

                    arguments = json.loads(remote_call.function.arguments)

                    if not isinstance(arguments, dict):
                        raise ModelProviderError("tool arguments must be a JSON object")

                    tool_calls.append(
                        ToolCall(
                            call_id=remote_call.id,
                            name=remote_call.function.name,
                            arguments=arguments,
                        )
                    )

                return ModelTurn(tool_calls=tool_calls, usage=token_usage)
            except json.JSONDecodeError as exc:
                raise ModelProviderError(
                    "model returned invalid JSON tool arguments"
                ) from exc
            except ValidationError as exc:
                raise ModelProviderError("model returned an invalid tool call") from exc

        content = message.content

        if not isinstance(content, str) or not content.strip():
            refusal = getattr(message, "refusal", None)  # 模型拒绝回答

            if isinstance(refusal, str) and refusal.strip():
                content = refusal
            else:
                raise ModelProviderError("model returned neither text or tool calls")

        try:
            return ModelTurn(final_text=content, usage=token_usage)
        except ValidationError as exc:
            raise ModelProviderError("model returned invalid final text") from exc

    async def close(self) -> None:
        await self._client.close()
