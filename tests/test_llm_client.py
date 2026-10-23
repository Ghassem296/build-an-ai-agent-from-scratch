import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.llm_client import (
    LlmClient,
    LlmRequest,
    Message,
    ToolCall,
    ToolResult,
)


def make_response(content=None, tool_calls=None, usage=None):
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
        type="function",
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

        response = await self.client.complete(
            LlmRequest(
                messages=[
                    Message(role="user", content="Explain MCP briefly.")
                ]
            )
        )

        self.assertEqual(len(response.content), 1)
        self.assertIsInstance(response.content[0], Message)
        self.assertEqual(response.content[0].content, "This is a simulated response.")

        kwargs = mock_completion.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["api_base"], "https://example.com/v1")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["timeout"], 15)

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_passes_user_messages(self, mock_completion):
        mock_completion.return_value = make_response(content="Done.")

        await self.client.complete(
            LlmRequest(messages=[Message(role="user", content="Hello")])
        )

        self.assertEqual(
            mock_completion.await_args.kwargs["messages"],
            [{"role": "user", "content": "Hello"}],
        )

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_converts_tool_call(self, mock_completion):
        mock_completion.return_value = make_response(
            tool_calls=[make_tool_call()]
        )

        response = await self.client.complete(
            LlmRequest(messages=[Message(role="user", content="Search for MCP")])
        )

        self.assertEqual(len(response.content), 1)
        tool_call = response.content[0]
        self.assertIsInstance(tool_call, ToolCall)
        self.assertEqual(tool_call.id, "call-1")
        self.assertEqual(tool_call.name, "search_web")
        self.assertEqual(tool_call.arguments, {"query": "MCP"})

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_preserves_text_with_tool_calls(self, mock_completion):
        mock_completion.return_value = make_response(
            content="I will search for MCP information.",
            tool_calls=[make_tool_call()],
        )

        response = await self.client.complete(
            LlmRequest(messages=[Message(role="user", content="Search for MCP")])
        )

        self.assertEqual(len(response.content), 2)
        self.assertIsInstance(response.content[0], Message)
        self.assertEqual(
            response.content[0].content,
            "I will search for MCP information.",
        )
        self.assertIsInstance(response.content[1], ToolCall)

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_build_messages_keeps_text_and_tool_calls_together(
        self,
        mock_completion,
    ):
        mock_completion.return_value = make_response(content="Done.")

        items = [
            Message(role="user", content="Search for MCP"),
            Message(
                role="assistant",
                content="I will search for MCP information.",
            ),
            ToolCall(
                id="call-1",
                name="search_web",
                arguments={"query": "MCP"},
            ),
        ]

        await self.client.complete(LlmRequest(messages=items))

        messages = mock_completion.await_args.kwargs["messages"]

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(
            messages[1]["content"],
            "I will search for MCP information.",
        )
        self.assertEqual(len(messages[1]["tool_calls"]), 1)
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_handles_empty_tool_arguments(self, mock_completion):
        mock_completion.return_value = make_response(
            tool_calls=[make_tool_call(arguments="")]
        )

        response = await self.client.complete(
            LlmRequest(messages=[Message(role="user", content="Run tool")])
        )

        self.assertIsInstance(response.content[0], ToolCall)
        self.assertEqual(response.content[0].arguments, {})

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_malformed_tool_arguments_are_flagged(self, mock_completion):
        """Invalid tool arguments must not silently become valid empty args."""
        for raw_arguments, expected_error in (
            ("{bad json", "not valid JSON"),
            ("[]", "must be a JSON object"),
            (None, "unsupported type"),
        ):
            with self.subTest(raw_arguments=raw_arguments):
                mock_completion.return_value = make_response(
                    tool_calls=[make_tool_call(arguments=raw_arguments)]
                )
                response = await self.client.complete(
                    LlmRequest(
                        messages=[Message(role="user", content="Run tool")]
                    )
                )
                tool_call = response.content[0]
                self.assertIsInstance(tool_call, ToolCall)
                self.assertEqual(tool_call.arguments, {})
                self.assertIn(expected_error, tool_call.parse_error)

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
                        "properties": {"query": {"type": "string"}},
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

        await self.client.complete(
            LlmRequest(
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
        )

        message = mock_completion.await_args.kwargs["messages"][-1]
        self.assertEqual(message["role"], "tool")
        self.assertIn("Tool error:", message["content"])
        self.assertIn("Search failed", message["content"])

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_handles_empty_model_response(self, mock_completion):
        mock_completion.return_value = make_response(content=None)

        response = await self.client.complete(
            LlmRequest(messages=[Message(role="user", content="Hello")])
        )

        self.assertEqual(len(response.content), 1)
        self.assertIsInstance(response.content[0], Message)
        self.assertEqual(response.content[0].content, "")

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_propagates_api_error(self, mock_completion):
        mock_completion.side_effect = RuntimeError("Simulated API failure")

        with self.assertRaisesRegex(RuntimeError, "Simulated API failure"):
            await self.client.complete(
                LlmRequest(messages=[Message(role="user", content="Hello")])
            )

    async def test_complete_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            await self.client.complete(LlmRequest(messages=[]))

    async def test_constructor_rejects_invalid_timeout(self):
        with self.assertRaises(ValueError):
            LlmClient(model="test-model", timeout=0)

    async def test_constructor_rejects_non_finite_timeout(self):
        for timeout in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "finite"):
                    LlmClient(model="test-model", timeout=timeout)

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

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_build_messages_groups_consecutive_tool_calls(
        self,
        mock_completion,
    ):
        """Two ToolCalls without any intermediate message must end up
        on the same assistant message.
        """

        mock_completion.return_value = make_response(content="Done.")

        items = [
            Message(role="user", content="Search twice"),
            ToolCall(
                id="call-1",
                name="search_web",
                arguments={"query": "one"},
            ),
            ToolCall(
                id="call-2",
                name="search_web",
                arguments={"query": "two"},
            ),
        ]

        await self.client.complete(LlmRequest(messages=items))

        messages = mock_completion.await_args.kwargs["messages"]

        # user + one assistant message with both tool calls
        self.assertEqual(len(messages), 2)

        self.assertEqual(messages[0]["role"], "user")

        assistant = messages[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertIsNone(assistant["content"])
        self.assertEqual(
            [c["id"] for c in assistant["tool_calls"]],
            ["call-1", "call-2"],
        )
    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_build_messages_starts_new_assistant_after_tool_result(
        self,
        mock_completion,
    ):
        """A ToolCall that follows a ToolResult must start a new
        assistant message instead of being merged into the previous
        assistant turn.
        """

        mock_completion.return_value = make_response(content="Done.")

        items = [
            Message(role="user", content="Search for MCP"),
            Message(role="assistant", content="Looking up MCP."),
            ToolCall(
                id="call-1",
                name="search_web",
                arguments={"query": "MCP"},
            ),
            ToolResult(
                tool_call_id="call-1",
                name="search_web",
                content="MCP info",
            ),
            # New turn: another tool call after the previous result.
            ToolCall(
                id="call-2",
                name="search_web",
                arguments={"query": "MCP spec"},
            ),
        ]

        await self.client.complete(LlmRequest(messages=items))

        messages = mock_completion.await_args.kwargs["messages"]

        # user, assistant(call-1), tool, assistant(call-2)
        self.assertEqual(len(messages), 4)

        self.assertEqual(messages[0]["role"], "user")

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(
            messages[1]["content"],
            "Looking up MCP.",
        )
        self.assertEqual(
            [c["id"] for c in messages[1]["tool_calls"]],
            ["call-1"],
        )

        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")

        # The second tool call must live on its OWN assistant message,
        # NOT be appended to the first assistant message.
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertIsNone(messages[3]["content"])
        self.assertEqual(
            [c["id"] for c in messages[3]["tool_calls"]],
            ["call-2"],
        )
    async def test_constructor_rejects_negative_max_retries(self):
        with self.assertRaises(ValueError):
            LlmClient(model="test-model", max_retries=-1)

    @patch("app.llm_client.acompletion", new_callable=AsyncMock)
    async def test_complete_passes_max_retries(self, mock_completion):
        mock_completion.return_value = make_response(content="Done.")

        client = LlmClient(
            model="test-model",
            max_retries=5,
        )

        await client.complete(
            LlmRequest(messages=[Message(role="user", content="Hello")])
        )

        kwargs = mock_completion.await_args.kwargs
        self.assertEqual(kwargs["num_retries"], 5)        

if __name__ == "__main__":
    unittest.main()