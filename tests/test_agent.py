import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent import Agent
from app.execution_context import ExecutionContext
from app.llm_client import (
    LlmRequest,
    LlmResponse,
    Message,
    ToolCall,
    ToolResult,
)


def make_tool_response():
    return SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="search_web",
                description="Search the web",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            )
        ]
    )


def make_mcp_result(text="Search result"):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        isError=False,
    )


class TestAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm_client = SimpleNamespace(complete=AsyncMock())
        self.mcp_session = SimpleNamespace(
            list_tools=AsyncMock(return_value=make_tool_response()),
            call_tool=AsyncMock(return_value=make_mcp_result()),
        )
        self.agent = Agent(
            llm_client=self.llm_client,
            mcp_session=self.mcp_session,
            tool_timeout=1.0,
        )

    async def test_run_returns_final_answer(self):
        self.llm_client.complete.return_value = LlmResponse(
            content=[
                Message(
                    role="assistant",
                    content="MCP lets applications use tools.",
                )
            ]
        )

        answer = await self.agent.run("Explain MCP")

        self.assertEqual(answer, "MCP lets applications use tools.")
        self.llm_client.complete.assert_awaited_once()
        self.mcp_session.list_tools.assert_awaited_once()
        self.mcp_session.call_tool.assert_not_awaited()

        self.assertIsInstance(self.agent.context, ExecutionContext)
        self.assertEqual(self.agent.context.events[0].type, "system")
        self.assertEqual(self.agent.context.events[1].type, "user")
        self.assertEqual(self.agent.context.events[2].type, "assistant")

    async def test_run_executes_tool_and_returns_follow_up_answer(self):
        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "Model Context Protocol"},
                    )
                ]
            ),
            LlmResponse(
                content=[
                    Message(
                        role="assistant",
                        content="The search returned useful information.",
                    )
                ]
            ),
        ]

        answer = await self.agent.run("Search for MCP")

        self.assertEqual(
            answer,
            "The search returned useful information.",
        )
        self.assertEqual(self.llm_client.complete.await_count, 2)
        self.mcp_session.call_tool.assert_awaited_once_with(
            "search_web",
            {"query": "Model Context Protocol"},
        )

        events = self.agent.context.get_events()
        self.assertEqual(
            [event.type for event in events],
            ["system", "user", "assistant", "tool", "assistant"],
        )

        tool_results = events[3].content
        self.assertEqual(len(tool_results), 1)
        self.assertIsInstance(tool_results[0], ToolResult)
        self.assertEqual(tool_results[0].tool_call_id, "call-1")
        self.assertEqual(tool_results[0].content, "Search result")
        self.assertFalse(tool_results[0].is_error)

        second_request = self.llm_client.complete.await_args_list[1].args[0]
        self.assertIsInstance(second_request, LlmRequest)
        self.assertTrue(
            any(
                isinstance(message, ToolResult)
                for message in second_request.messages
            )
        )

    async def test_run_passes_mcp_tools_to_llm(self):
        self.llm_client.complete.return_value = LlmResponse(
            content=[Message(role="assistant", content="Done")]
        )

        await self.agent.run("Hello")

        request = self.llm_client.complete.await_args.args[0]

        self.assertEqual(len(request.tools), 1)
        self.assertEqual(request.tools[0]["type"], "function")
        self.assertEqual(
            request.tools[0]["function"]["name"],
            "search_web",
        )
        self.assertEqual(
            request.tools[0]["function"]["parameters"]["required"],
            ["query"],
        )

    async def test_run_handles_multiple_tool_calls(self):
        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "first query"},
                    ),
                    ToolCall(
                        id="call-2",
                        name="search_web",
                        arguments={"query": "second query"},
                    ),
                ]
            ),
            LlmResponse(
                content=[
                    Message(role="assistant", content="Both searches completed.")
                ]
            ),
        ]

        answer = await self.agent.run("Search twice")

        self.assertEqual(answer, "Both searches completed.")
        self.assertEqual(self.mcp_session.call_tool.await_count, 2)

        calls = self.mcp_session.call_tool.await_args_list
        self.assertEqual(
            calls[0].args,
            ("search_web", {"query": "first query"}),
        )
        self.assertEqual(
            calls[1].args,
            ("search_web", {"query": "second query"}),
        )

        tool_event = self.agent.context.events[3]
        self.assertEqual(len(tool_event.content), 2)

    async def test_run_requests_final_answer_after_step_limit(self):
        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "MCP"},
                    )
                ]
            ),
            LlmResponse(
                content=[
                    Message(
                        role="assistant",
                        content="Here is the answer based on the search.",
                    )
                ]
            ),
        ]

        answer = await self.agent.run(
            "Search for MCP",
            max_steps=1,
        )

        self.assertEqual(
            answer,
            "Here is the answer based on the search.",
        )
        self.assertEqual(self.llm_client.complete.await_count, 2)

        final_request = self.llm_client.complete.await_args_list[1].args[0]
        self.assertEqual(final_request.tools, [])

        self.assertIsInstance(final_request.messages[0], Message)
        self.assertEqual(final_request.messages[0].role, "system")
        self.assertIn(
            "step limit has been reached",
            final_request.messages[0].content,
        )

    async def test_run_uses_fallback_if_final_answer_is_empty(self):
        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "MCP"},
                    )
                ]
            ),
            LlmResponse(content=[]),
        ]

        answer = await self.agent.run(
            "Search for MCP",
            max_steps=1,
        )

        self.assertIn("maximum number of tool-use steps was reached", answer)
        self.assertIn("did not return a textual final answer", answer)

    async def test_run_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            await self.agent.run("   ")

        self.mcp_session.list_tools.assert_not_awaited()
        self.llm_client.complete.assert_not_awaited()

    async def test_run_rejects_invalid_max_steps(self):
        with self.assertRaises(ValueError):
            await self.agent.run("Hello", max_steps=0)

        self.mcp_session.list_tools.assert_not_awaited()
        self.llm_client.complete.assert_not_awaited()

    async def test_constructor_rejects_invalid_tool_timeout(self):
        with self.assertRaises(ValueError):
            Agent(
                llm_client=self.llm_client,
                mcp_session=self.mcp_session,
                tool_timeout=0,
            )

    async def test_constructor_rejects_invalid_max_history_events(self):
        with self.assertRaises(ValueError):
            Agent(
                llm_client=self.llm_client,
                mcp_session=self.mcp_session,
                tool_timeout=1.0,
                max_history_events=0,
            )

    async def test_tool_error_is_returned_to_llm(self):
        self.mcp_session.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="Search service failed")],
            isError=True,
        )
        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "MCP"},
                    )
                ]
            ),
            LlmResponse(
                content=[
                    Message(
                        role="assistant",
                        content="The search tool reported an error.",
                    )
                ]
            ),
        ]

        answer = await self.agent.run("Search for MCP")

        self.assertEqual(answer, "The search tool reported an error.")

        tool_event = self.agent.context.events[3]
        result = tool_event.content[0]

        self.assertTrue(result.is_error)
        self.assertEqual(result.content, "Search service failed")

        second_request = self.llm_client.complete.await_args_list[1].args[0]
        returned_results = [
            item
            for item in second_request.messages
            if isinstance(item, ToolResult)
        ]
        self.assertEqual(len(returned_results), 1)
        self.assertTrue(returned_results[0].is_error)

    async def test_tool_timeout_becomes_tool_error(self):
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.1)
            return make_mcp_result()

        self.mcp_session.call_tool.side_effect = slow_tool
        self.agent.tool_timeout = 0.01

        result = await self.agent._execute_tool(
            ToolCall(
                id="call-timeout",
                name="search_web",
                arguments={"query": "MCP"},
            )
        )

        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)
        self.mcp_session.call_tool.assert_awaited_once()

    async def test_tool_exception_becomes_tool_error(self):
        self.mcp_session.call_tool.side_effect = RuntimeError(
            "Simulated MCP failure"
        )

        result = await self.agent._execute_tool(
            ToolCall(
                id="call-error",
                name="search_web",
                arguments={"query": "MCP"},
            )
        )

        self.assertTrue(result.is_error)
        # The raw exception text must NOT leak to the model.
        self.assertNotIn("RuntimeError", result.content)
        self.assertNotIn("Simulated MCP failure", result.content)
        # A generic user-friendly message is returned instead.
        self.assertIn("Tool execution failed", result.content)

    async def test_tool_result_normalization(self):
        result = SimpleNamespace(
            content=[
                SimpleNamespace(text="First line"),
                SimpleNamespace(text="Second line"),
            ],
            isError=False,
        )

        self.assertEqual(
            self.agent._normalize_tool_result(result),
            "First line\nSecond line",
        )

    async def test_tool_definitions_with_missing_schema(self):
        self.mcp_session.list_tools.return_value = SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="plain_tool",
                    description=None,
                )
            ]
        )

        definitions = await self.agent._get_tool_definitions()

        self.assertEqual(len(definitions), 1)
        self.assertEqual(
            definitions[0]["function"]["name"],
            "plain_tool",
        )
        self.assertEqual(
            definitions[0]["function"]["description"],
            "",
        )
        self.assertEqual(
            definitions[0]["function"]["parameters"],
            {
                "type": "object",
                "properties": {},
            },
        )

    async def test_tool_definition_failure_propagates(self):
        self.mcp_session.list_tools.side_effect = RuntimeError(
            "Simulated MCP list failure"
        )

        with self.assertRaisesRegex(RuntimeError, "Simulated MCP list failure"):
            await self.agent.run("Hello")

        self.llm_client.complete.assert_not_awaited()

    async def test_agent_truncates_long_history(self):
        agent = Agent(
            llm_client=self.llm_client,
            mcp_session=self.mcp_session,
            tool_timeout=1.0,
            max_history_events=3,
        )

        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "first"},
                    )
                ]
            ),
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-2",
                        name="search_web",
                        arguments={"query": "second"},
                    )
                ]
            ),
            LlmResponse(
                content=[
                    Message(
                        role="assistant",
                        content="Final answer.",
                    )
                ]
            ),
        ]

        answer = await agent.run(
            "Search twice",
            max_steps=2,
        )

        self.assertEqual(answer, "Final answer.")

        final_request = self.llm_client.complete.await_args_list[-1].args[0]
        self.assertGreaterEqual(len(final_request.messages), 1)

        system_msgs = [
            m
            for m in final_request.messages
            if isinstance(m, Message) and m.role == "system"
        ]
        self.assertTrue(system_msgs)

    async def test_final_answer_llm_error_is_hidden_from_user(self):
        self.llm_client.complete.side_effect = [
            LlmResponse(
                content=[
                    ToolCall(
                        id="call-1",
                        name="search_web",
                        arguments={"query": "MCP"},
                    )
                ]
            ),
            RuntimeError("Simulated final API failure"),
        ]

        answer = await self.agent.run(
            "Search for MCP",
            max_steps=1,
        )

        self.assertIn("a final answer could not be generated", answer)
        self.assertIn("Please try again later", answer)
        self.assertNotIn("Simulated final API failure", answer)
        self.assertEqual(self.mcp_session.call_tool.await_count, 1)


if __name__ == "__main__":
    unittest.main()