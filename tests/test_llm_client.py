
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.llm_client import (
    LlmClient,
    LlmRequest,
    LlmResponse,
    Message,
    ToolCall,
    ToolResult,
)


def make_response(
    content=None,
    tool_calls=None,
    usage=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
        usage=usage,
    )


def make_tool_call(
    call_id="call-1",
    name="search_web",
    arguments='{"query": "MCP"}',
):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


class TestLlmClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = LlmClient(
            model="test-model",
            api_key="test-key",
            api_base="https://example.com/v1",
            temperature=0.2,
            timeout=15,
        )

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_returns_assistant_message(self, mock_completion):
        mock_completion.return_value = make_response(
            content="This is a simulated response."
        )

        request = LlmRequest(
            messages=[
                Message(role="user", content="Explain MCP briefly.")
            ]
        )

        response = await self.client.complete(request)

        self.assertIsInstance(response, LlmResponse)
        self.assertEqual(len(response.content), 1)
        self.assertIsInstance(response.content[0], Message)
        self.assertEqual(response.content[0].role, "assistant")
        self.assertEqual(
            response.content[0].content,
            "This is a simulated response.",
        )

        mock_completion.assert_awaited_once()
        kwargs = mock_completion.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["api_base"], "https://example.com/v1")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["timeout"], 15)

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_passes_user_messages(self, mock_completion):
        mock_completion.return_value = make_response(content="Done.")

        request = LlmRequest(
            messages=[
                Message(role="user", content="Hello")
            ]
        )

        await self.client.complete(request)

        messages = mock_completion.await_args.kwargs["messages"]
        self.assertEqual(
            messages,
            [{"role": "user", "content": "Hello"}],
        )

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_converts_tool_call(self, mock_completion):
        mock_completion.return_value = make_response(
            content=None,
            tool_calls=[make_tool_call()],
        )

        request = LlmRequest(
            messages=[
                Message(role="user", content="Search for MCP")
            ]
        )

        response = await self.client.complete(request)

        self.assertEqual(len(response.content), 1)
        tool_call = response.content[0]
        self.assertIsInstance(tool_call, ToolCall)
        self.assertEqual(tool_call.id, "call-1")
        self.assertEqual(tool_call.name, "search_web")
        self.assertEqual(tool_call.arguments, {"query": "MCP"})

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_handles_empty_tool_arguments(self, mock_completion):
        mock_completion.return_value = make_response(
            content=None,
            tool_calls=[
                make_tool_call(arguments="")
            ],
        )

        response = await self.client.complete(
            LlmRequest(messages=[Message(role="user", content="Run tool")])
        )

        self.assertEqual(len(response.content), 1)
        self.assertIsInstance(response.content[0], ToolCall)
        self.assertEqual(response.content[0].arguments, {})

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_includes_tool_definitions(self, mock_completion):
        mock_completion.return_value = make_response(content="Done.")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        await self.client.complete(
            LlmRequest(
                messages=[Message(role="user", content="Search")],
                tools=tools,
            )
        )

        kwargs = mock_completion.await_args.kwargs
        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(kwargs["tool_choice"], "auto")

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_does_not_send_tools_when_disabled(self, mock_completion):
        mock_completion.return_value = make_response(content="Done.")

        await self.client.complete(
            LlmRequest(
                messages=[Message(role="user", content="Hello")],
                tools=[],
            )
        )

        kwargs = mock_completion.await_args.kwargs
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_serializes_tool_result(self, mock_completion):
        mock_completion.return_value = make_response(content="Finished.")

        request = LlmRequest(
            messages=[
                Message(role="user", content="Search"),
                ToolCall(
                    id="call-1",
                    name="search_web",
                    arguments={"query": "MCP"},
                ),
                ToolResult(
                    tool_call_id="call-1",
                    name="search_web",
                    content="Found MCP information",
                ),
            ]
        )

        await self.client.complete(request)

        messages = mock_completion.await_args.kwargs["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")
        self.assertEqual(messages[2]["name"], "search_web")
        self.assertEqual(messages[2]["content"], "Found MCP information")

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_marks_tool_error(self, mock_completion):
        mock_completion.return_value = make_response(content="Recovered.")

        request = LlmRequest(
            messages=[
                Message(role="user", content="Search"),
                ToolResult(
                    tool_call_id="call-1",
                    name="search_web",
                    content="Search failed",
                    is_error=True,
                ),
            ]
        )

        await self.client.complete(request)

        messages = mock_completion.await_args.kwargs["messages"]
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertIn("Tool error:", messages[-1]["content"])
        self.assertIn("Search failed", messages[-1]["content"])

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_handles_empty_model_response(self, mock_completion):
        mock_completion.return_value = make_response(content=None)

        response = await self.client.complete(
            LlmRequest(
                messages=[Message(role="user", content="Hello")]
            )
        )

        self.assertEqual(len(response.content), 1)
        self.assertIsInstance(response.content[0], Message)
        self.assertEqual(response.content[0].role, "assistant")
        self.assertEqual(response.content[0].content, "")

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_propagates_api_error(self, mock_completion):
        mock_completion.side_effect = RuntimeError("Simulated API failure")

        with self.assertRaisesRegex(RuntimeError, "Simulated API failure"):
            await self.client.complete(
                LlmRequest(
                    messages=[Message(role="user", content="Hello")]
                )
            )

    async def test_complete_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            await self.client.complete(LlmRequest(messages=[]))

    async def test_constructor_rejects_invalid_timeout(self):
        with self.assertRaises(ValueError):
            LlmClient(model="test-model", timeout=0)

    async def test_constructor_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            LlmClient(model="test-model", temperature=3)

    async def test_request_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            await self.client.complete(
                LlmRequest(
                    messages=[Message(role="user", content="Hello")],
                    temperature=3,
                )
            )


if __name__ == "__main__":
    unittest.main()
