"""Offline tests for section 4.7 (Structured Output)."""
import json
import unittest
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from app.agent import Agent
from app.execution_context import ExecutionContext
from app.llm_client import LlmResponse, Message, ToolCall, ToolResult


# ----- Test schema -----
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float


# ----- Helpers (mock builders) -----
def make_tool_call(name: str, arguments: dict, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def make_llm_response(content: list[Any], finish_reason: str = "stop") -> LlmResponse:
    # NOTE: adjust if your LlmResponse signature differs
    return LlmResponse(content=content, finish_reason=finish_reason)


def make_mcp_session(tools: list[dict] | None = None) -> MagicMock:
    session = MagicMock()

    response = MagicMock()
    response.tools = []
    for t in (tools or []):
        tool = MagicMock()
        tool.name = t["name"]
        tool.description = t.get("description", "")
        tool.inputSchema = t.get("input_schema", {"type": "object", "properties": {}})
        response.tools.append(tool)

    session.list_tools = AsyncMock(return_value=response)
    session.call_tool = AsyncMock()
    return session


# ----- Tests -----
class StructuredOutputTests(unittest.IsolatedAsyncioTestCase):

    async def test_setup_creates_final_answer_tool(self):
        """Final_answer tool must be registered when output_type is set."""
        agent = Agent(MagicMock(), make_mcp_session(), output_type=Sentiment)
        await agent._ensure_tool_definitions()

        self.assertEqual(agent.output_tool_name, "final_answer")
        self.assertIn("final_answer", agent._allowed_tools)
        self.assertIsNotNone(agent._final_answer_tool)

        props = agent._final_answer_tool["function"]["parameters"]["properties"]
        self.assertIn("sentiment", props)
        self.assertIn("confidence", props)

    async def test_no_output_type_leaves_tool_name_none(self):
        """Backward compat: without output_type, no final_answer tool."""
        agent = Agent(MagicMock(), make_mcp_session())
        await agent._ensure_tool_definitions()

        self.assertIsNone(agent.output_tool_name)
        self.assertNotIn("final_answer", agent._allowed_tools)

    async def test_prepare_request_sets_required_tool_choice(self):
        """tool_choice must be 'required' when structured output is active."""
        agent = Agent(MagicMock(), make_mcp_session(), output_type=Sentiment)
        await agent._ensure_tool_definitions()

        ctx = ExecutionContext(execution_id="t1")
        agent.seed_system_prompt(ctx)
        ctx.create_event("user", [Message(role="user", content="hi")])

        request = agent._prepare_llm_request(
            ctx, tools=agent._tool_definitions
        )
        self.assertEqual(request.tool_choice, "required")

    async def test_prepare_request_uses_auto_without_output_type(self):
        """Without structured output, tool_choice should be 'auto'."""
        agent = Agent(MagicMock(), make_mcp_session(tools=[
            {"name": "search_web"}
        ]))
        await agent._ensure_tool_definitions()

        ctx = ExecutionContext(execution_id="t2")
        ctx.create_event("user", [Message(role="user", content="hi")])
        request = agent._prepare_llm_request(
            ctx, tools=agent._tool_definitions
        )
        self.assertEqual(request.tool_choice, "auto")

    async def test_execute_tool_intercepts_final_answer(self):
        """final_answer must NOT be forwarded to MCP."""
        session = make_mcp_session()
        agent = Agent(MagicMock(), session, output_type=Sentiment)
        await agent._ensure_tool_definitions()

        tc = make_tool_call(
            "final_answer", {"sentiment": "positive", "confidence": 0.9}
        )
        result = await agent._execute_tool(tc)

        self.assertEqual(result.name, "final_answer")
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content)["sentiment"], "positive")

        # Crucial: MCP must NOT be called for final_answer
        session.call_tool.assert_not_called()

    async def test_extract_structured_result_parses_json(self):
        """_extract_structured_result must build a Pydantic object."""
        agent = Agent(MagicMock(), make_mcp_session(), output_type=Sentiment)
        await agent._ensure_tool_definitions()

        ctx = ExecutionContext(execution_id="t3")
        ctx.create_event("tool", [ToolResult(
            tool_call_id="call_1",
            name="final_answer",
            content=json.dumps({"sentiment": "positive", "confidence": 0.9}),
            is_error=False,
        )])

        result = agent._extract_structured_result(ctx)
        self.assertIsInstance(result, Sentiment)
        self.assertEqual(result.sentiment, "positive")
        self.assertAlmostEqual(result.confidence, 0.9)

    async def test_extract_ignores_error_results(self):
        """Failed final_answer should not produce a structured result."""
        agent = Agent(MagicMock(), make_mcp_session(), output_type=Sentiment)
        await agent._ensure_tool_definitions()

        ctx = ExecutionContext(execution_id="t4")
        ctx.create_event("tool", [ToolResult(
            tool_call_id="c1",
            name="final_answer",
            content="bad json",
            is_error=True,
        )])

        self.assertIsNone(agent._extract_structured_result(ctx))

    async def test_extract_returns_none_on_invalid_json(self):
        """Malformed JSON must not crash; returns None."""
        agent = Agent(MagicMock(), make_mcp_session(), output_type=Sentiment)
        await agent._ensure_tool_definitions()

        ctx = ExecutionContext(execution_id="t5")
        ctx.create_event("tool", [ToolResult(
            tool_call_id="c1",
            name="final_answer",
            content="not-json",
            is_error=False,
        )])

        self.assertIsNone(agent._extract_structured_result(ctx))

    async def test_run_returns_pydantic_object(self):
        """End-to-end: run() with output_type returns a Pydantic instance."""
        session = make_mcp_session()

        llm = MagicMock()
        llm.complete = AsyncMock(return_value=make_llm_response([
            ToolCall(
                id="call_1",
                name="final_answer",
                arguments={"sentiment": "positive", "confidence": 0.85},
            )
        ]))

        agent = Agent(llm, session, output_type=Sentiment)
        result = await agent.run("I love this!")

        self.assertIsInstance(result, Sentiment)
        self.assertEqual(result.sentiment, "positive")
        self.assertAlmostEqual(result.confidence, 0.85)

    async def test_run_without_output_type_returns_string(self):
        """Without output_type, run() must return a string (backward compat)."""
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=make_llm_response([
            Message(role="assistant", content="plain answer")
        ]))

        agent = Agent(llm, make_mcp_session())
        result = await agent.run("hello?")

        self.assertEqual(result, "plain answer")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()