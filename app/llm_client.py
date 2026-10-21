from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import acompletion

logger = logging.getLogger(__name__)

litellm.register_model(
    {
        "openai/gemini-2.5-flash": {
            "max_tokens": 8192,
            "max_input_tokens": 1048576,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
)


@dataclass(slots=True)
class Message:
    role: str
    content: str | None = None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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


@dataclass(slots=True)
class LlmResponse:
    content: list[Message | ToolCall | ToolResult]
    input_tokens: int = 0
    output_tokens: int = 0


class LlmClient:
    def __init__(
        self,
        model: str = "openai/gemini-2.5-flash",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 2,              # ← NEW
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        if max_retries < 0:                 # ← NEW
            raise ValueError("max_retries cannot be negative")

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
        self.max_retries = max_retries      # ← NEW

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

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(request.messages),
            "temperature": temperature,
            "timeout": self.timeout,
            "num_retries": self.max_retries,    # ← NEW
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
            payload["tool_choice"] = "auto"

        logger.debug("Sending request to model %s", self.model)

        response = await acompletion(**payload)

        return self._parse_response(response)

    @staticmethod
    def _serialize_tool_call(
        item: ToolCall,
    ) -> dict[str, Any]:
        """Convert a normalized tool call to API format."""

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
        """Convert normalized conversation items to API messages.

        Invariant
        ---------
        A ``ToolCall`` is only valid inside an assistant message that
        belongs to the *current* assistant turn. A turn ends as soon as
        either:

        * a ``ToolResult`` for that turn is appended, or
        * a non-assistant message (user/system) is appended.

        Any ``ToolCall`` that arrives after the current turn has ended
        starts a brand new assistant message with ``content=None``.
        Consecutive tool calls within the same turn are grouped onto
        the same assistant message.
        """

        messages: list[dict[str, Any]] = []

        # The assistant message that is still accepting tool calls,
        # or ``None`` if the current turn has been closed.
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
        """Convert a LiteLLM response to normalized objects."""

        choices = (
            response.get("choices", [])
            if isinstance(response, dict)
            else getattr(response, "choices", [])
        )

        if not choices:
            raise ValueError("The model returned no choices")

        choice = choices[0]
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

            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON arguments for tool {name!r}"
                    ) from exc

                if not isinstance(parsed_arguments, dict):
                    raise ValueError(
                        f"Arguments for tool {name!r} must be a JSON object"
                    )

                arguments = parsed_arguments
            else:
                raise ValueError(
                    f"Invalid arguments for tool {name!r}"
                )

            content.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
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
        )