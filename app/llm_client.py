from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import acompletion

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_METADATA: dict[str, Any] = {
    "max_tokens": 8192,
    "max_input_tokens": 1048576,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "litellm_provider": "openai",
    "mode": "chat",
}


def _register_model(model: str) -> None:
    """Register a model with LiteLLM if it isn't already known."""
    try:
        litellm.get_model_info(model)
        return
    except Exception:
        pass

    litellm.register_model({model: _DEFAULT_MODEL_METADATA})
    logger.debug("Registered unknown model %s", model)


@dataclass(slots=True)
class Message:
    role: str
    content: str | None = None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class LlmRequest:
    messages: list[Message | ToolCall | ToolResult]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    tool_choice: str | None = None


@dataclass(slots=True)
class LlmResponse:
    content: list[Message | ToolCall | ToolResult]
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None


class LlmClient:
    """Asynchronous LiteLLM client supporting local Ollama models."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 2,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "timeout must be a finite number greater than zero"
            )

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        _register_model(model)

        self.model = model
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("XAI_1_API_KEY")
        )
        self.api_base = (
            api_base
            or os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_API_BASE")
        )
        self.temperature = temperature
        self.timeout = timeout
        self.tools = tools or []
        self.max_retries = max_retries

    async def complete(self, request: LlmRequest) -> LlmResponse:
        """Send a conversation to the model and normalize its response."""

        if not request.messages:
            raise ValueError("request.messages cannot be empty")

        temperature = (
            request.temperature
            if request.temperature is not None
            else self.temperature
        )

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        messages = self._build_messages(request.messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
        }

        if self.api_key:
            payload["api_key"] = self.api_key

        if self.api_base:
            payload["api_base"] = self.api_base

        tool_definitions = (
            self.tools if request.tools is None else request.tools
        )

        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = (
                request.tool_choice
                if request.tool_choice is not None
                else "auto"
            )

        # Diagnostic logging: do not log message contents or credentials.
        message_chars = sum(
            len(str(message.get("content") or ""))
            for message in messages
            if isinstance(message, dict)
        )
        tool_count = len(tool_definitions or [])

        logger.warning(
            "LLM request diagnostics: model=%s, messages=%d, "
            "message_chars=%d, tools=%d, timeout=%s",
            self.model,
            len(messages),
            message_chars,
            tool_count,
            self.timeout,
        )

        try:
            response = await acompletion(**payload)
        except Exception:
            logger.exception(
                "LLM request failed: model=%s, messages=%d, "
                "message_chars=%d, tools=%d",
                self.model,
                len(messages),
                message_chars,
                tool_count,
            )
            raise

        return self._parse_response(response)

    @staticmethod
    def _serialize_tool_call(item: ToolCall) -> dict[str, Any]:
        return {
            "id": item.id,
            "type": "function",
            "function": {
                "name": item.name,
                "arguments": json.dumps(item.arguments),
            },
        }

    @staticmethod
    def _build_messages(
        items: list[Message | ToolCall | ToolResult],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        open_assistant: dict[str, Any] | None = None

        for item in items:
            if isinstance(item, Message):
                if item.role == "tool":
                    raise ValueError(
                        "Tool results must use the ToolResult class"
                    )

                api_message: dict[str, Any] = {
                    "role": item.role,
                    "content": item.content,
                }
                messages.append(api_message)

                if item.role == "assistant":
                    open_assistant = api_message
                else:
                    open_assistant = None

            elif isinstance(item, ToolCall):
                serialized_call = LlmClient._serialize_tool_call(item)

                if (
                    open_assistant is not None
                    and open_assistant.get("tool_calls") is None
                ):
                    open_assistant["tool_calls"] = [serialized_call]
                elif (
                    open_assistant is not None
                    and open_assistant.get("tool_calls") is not None
                ):
                    open_assistant["tool_calls"].append(serialized_call)
                else:
                    new_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [serialized_call],
                    }
                    messages.append(new_message)
                    open_assistant = new_message

            elif isinstance(item, ToolResult):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.tool_call_id,
                        "name": item.name,
                        "content": (
                            f"Tool error: {item.content}"
                            if item.is_error
                            else item.content
                        ),
                    }
                )
                open_assistant = None

            else:
                raise TypeError(
                    f"Unsupported message item: {type(item).__name__}"
                )

        return messages

    @staticmethod
    def _parse_response(response: Any) -> LlmResponse:
        choices = (
            response.get("choices", [])
            if isinstance(response, dict)
            else getattr(response, "choices", [])
        )

        if not choices:
            raise ValueError("The model returned no choices")

        choice = choices[0]

        finish_reason = (
            choice.get("finish_reason")
            if isinstance(choice, dict)
            else getattr(choice, "finish_reason", None)
        )
        message = (
            choice.get("message", {})
            if isinstance(choice, dict)
            else getattr(choice, "message", None)
        )

        if message is None:
            raise ValueError("The model response has no assistant message")

        if isinstance(message, dict):
            text_content = message.get("content")
            raw_tool_calls = message.get("tool_calls") or []
        else:
            text_content = getattr(message, "content", None)
            raw_tool_calls = getattr(message, "tool_calls", None) or []

        content: list[Message | ToolCall | ToolResult] = []

        if text_content:
            content.append(
                Message(role="assistant", content=text_content)
            )

        for raw_call in raw_tool_calls:
            if isinstance(raw_call, dict):
                call_id = raw_call.get("id", "")
                function = raw_call.get("function", {})
            else:
                call_id = getattr(raw_call, "id", "")
                function = getattr(raw_call, "function", None)

            if isinstance(function, dict):
                name = function.get("name", "")
                raw_arguments = function.get("arguments", "{}")
            else:
                name = getattr(function, "name", "")
                raw_arguments = getattr(function, "arguments", "{}")

            if not call_id:
                call_id = f"call_{uuid.uuid4().hex}"
                logger.warning(
                    "Tool call %r had no id; generated fallback id",
                    name,
                )

            parse_error: str | None = None

            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    logger.warning(
                        "Tool %r returned invalid JSON arguments",
                        name,
                    )
                    parsed_arguments = None
                    parse_error = "Tool arguments were not valid JSON"

                if parse_error is None and not isinstance(
                    parsed_arguments, dict
                ):
                    logger.warning(
                        "Tool %r arguments were not a JSON object",
                        name,
                    )
                    parse_error = "Tool arguments must be a JSON object"
                    parsed_arguments = None

                arguments = (
                    parsed_arguments
                    if isinstance(parsed_arguments, dict)
                    else {}
                )
            else:
                logger.warning(
                    "Tool %r returned unsupported argument type: %s",
                    name,
                    type(raw_arguments).__name__,
                )
                arguments = {}
                parse_error = "Tool arguments have an unsupported type"

            content.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )

        if not content:
            content.append(Message(role="assistant", content=""))

        usage = (
            response.get("usage", {})
            if isinstance(response, dict)
            else getattr(response, "usage", None)
        )

        if usage is None:
            input_tokens = 0
            output_tokens = 0
        elif isinstance(usage, dict):
            input_tokens = (
                usage.get("prompt_tokens", usage.get("input_tokens", 0))
                or 0
            )
            output_tokens = (
                usage.get(
                    "completion_tokens",
                    usage.get("output_tokens", 0),
                )
                or 0
            )
        else:
            input_tokens = (
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "input_tokens", 0)
                or 0
            )
            output_tokens = (
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "output_tokens", 0)
                or 0
            )

        return LlmResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )