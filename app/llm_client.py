from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Message:
    """A message in the conversation."""

    role: str
    content: str | None = None


@dataclass(slots=True)
class ToolCall:
    """A tool call requested by the language model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """The result of executing a tool."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class LlmRequest:
    """Input data for an LLM request."""

    messages: list[Message | ToolCall | ToolResult]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None


@dataclass(slots=True)
class LlmResponse:
    """Normalized response from the language model."""

    content: list[Message | ToolCall]


class LlmClient:
    """Asynchronous LLM client powered by LiteLLM."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = (
            model
            or os.getenv("LLM_MODEL")
            or "openai/gemini-2.5-flash"
        )

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

        if not self.model.strip():
            raise ValueError("model cannot be empty")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        self.tools = list(tools or [])
        self.temperature = temperature
        self.timeout = timeout

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
        }

        if self.api_key:
            payload["api_key"] = self.api_key

        if self.api_base:
            payload["api_base"] = self.api_base

        # None means use the client's default tools.
        # An empty list explicitly disables tools for this request.
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
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        """Serialize a tool call into the OpenAI-compatible format."""

        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(
                    tool_call.arguments,
                    ensure_ascii=False,
                ),
            },
        }

    @staticmethod
    def _build_messages(
        items: list[Message | ToolCall | ToolResult],
    ) -> list[dict[str, Any]]:
        """Convert normalized conversation items to API messages."""

        messages: list[dict[str, Any]] = []

        for item in items:
            if isinstance(item, Message):
                if item.role == "tool":
                    raise ValueError(
                        "Tool results must use the ToolResult class"
                    )

                messages.append(
                    {
                        "role": item.role,
                        "content": item.content,
                    }
                )

            elif isinstance(item, ToolCall):
                serialized_call = LlmClient._serialize_tool_call(item)

                # Merge consecutive tool calls into the same assistant
                # message when possible.
                if (
                    messages
                    and messages[-1].get("role") == "assistant"
                    and messages[-1].get("tool_calls") is not None
                ):
                    messages[-1]["tool_calls"].append(serialized_call)
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [serialized_call],
                        }
                    )

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

            else:
                raise TypeError(
                    "Unsupported conversation item: "
                    f"{type(item).__name__}"
                )

        return messages

    @staticmethod
    def _parse_response(response: Any) -> LlmResponse:
        """Parse a LiteLLM response into Message and ToolCall objects."""

        choices = getattr(response, "choices", None)

        if not choices:
            raise ValueError("LiteLLM returned no choices")

        message = choices[0].message
        content: list[Message | ToolCall] = []

        text_content = getattr(message, "content", None)

        if text_content:
            content.append(
                Message(
                    role="assistant",
                    content=text_content,
                )
            )

        for raw_call in getattr(message, "tool_calls", None) or []:
            function = raw_call.function
            raw_arguments = getattr(function, "arguments", None) or "{}"

            try:
                if isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments)
                else:
                    arguments = raw_arguments
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Invalid JSON arguments for tool "
                    f"{function.name!r}: {raw_arguments!r}"
                ) from exc

            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Arguments for tool {function.name!r} "
                    "must be a JSON object"
                )

            content.append(
                ToolCall(
                    id=raw_call.id,
                    name=function.name,
                    arguments=arguments,
                )
            )

        if not content:
            content.append(
                Message(
                    role="assistant",
                    content="",
                )
            )

        return LlmResponse(content=content)