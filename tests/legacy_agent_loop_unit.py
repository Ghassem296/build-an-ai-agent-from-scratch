
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_agent_loop import simple_agent_loop


def make_tool_response(call_id):
    """Create a simulated LLM response requesting a tool."""

    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="search_web",
            arguments='{"query": "MCP"}',
        ),
    )

    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="tool_calls",
            )
        ]
    )


def make_final_response():
    """Create a simulated LLM response with a final answer."""

    message = SimpleNamespace(
        content="Simulated final answer.",
        tool_calls=None,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="stop",
            )
        ]
    )


class TestAgentLoop(unittest.TestCase):

    @patch("test_agent_loop.tool_execution")
    @patch("test_agent_loop.completion")
    def test_agent_stops_repeated_tool_calls(
        self,
        mock_completion,
        mock_tool_execution,
    ):
        mock_completion.side_effect = [
            make_tool_response("call-1"),
            make_tool_response("call-2"),
            make_final_response(),
        ]

        mock_tool_execution.return_value = {
            "results": []
        }

        with patch.dict(
            os.environ,
            {"XAI_1_API_KEY": "test-key"},
        ):
            result = simple_agent_loop(
                system_prompt="You are a test assistant.",
                question="Search for MCP.",
                max_steps=2,
            )

        self.assertEqual(
            result,
            "Simulated final answer.",
        )

        self.assertEqual(
            mock_completion.call_count,
            3,
        )

        self.assertEqual(
            mock_tool_execution.call_count,
            2,
        )

        final_call_kwargs = (
            mock_completion.call_args_list[-1].kwargs
        )

        self.assertEqual(
            final_call_kwargs.get("tool_choice"),
            "none",
        )



    @patch("test_agent_loop.tool_execution")
    @patch("test_agent_loop.completion")
    def test_forced_final_response_does_not_execute_more_tools(
        self,
        mock_completion,
        mock_tool_execution,
    ):
        mock_completion.side_effect = [
            make_tool_response("call-1"),
            make_tool_response("call-2"),
            make_final_response(),
        ]

        mock_tool_execution.return_value = {
            "results": []
        }

        with patch.dict(
            os.environ,
            {"XAI_1_API_KEY": "test-key"},
        ):
            result = simple_agent_loop(
                system_prompt="You are a test assistant.",
                question="Search for MCP.",
                max_steps=2,
            )

        self.assertEqual(
            result,
            "Simulated final answer.",
        )

        self.assertEqual(
            mock_tool_execution.call_count,
            2,
        )

        final_call_kwargs = (
            mock_completion.call_args_list[-1].kwargs
        )

        self.assertEqual(
            final_call_kwargs["tool_choice"],
            "none",
        )


    @patch("test_agent_loop.tool_execution")
    @patch("test_agent_loop.completion")
    def test_agent_handles_final_api_error(
        self,
        mock_completion,
        mock_tool_execution,
    ):
        mock_completion.side_effect = [
            make_tool_response("call-1"),
            make_tool_response("call-2"),
            RuntimeError("Simulated final API failure"),
        ]

        mock_tool_execution.return_value = {
            "results": []
        }

        with patch.dict(
            os.environ,
            {"XAI_1_API_KEY": "test-key"},
        ):
            result = simple_agent_loop(
                system_prompt="You are a test assistant.",
                question="Search for MCP.",
                max_steps=2,
            )

        self.assertIn(
            "final response could not be generated",
            result,
        )

        self.assertIn(
            "Simulated final API failure",
            result,
        )

        self.assertEqual(
            mock_tool_execution.call_count,
            2,
        )

        self.assertEqual(
            mock_completion.call_count,
            3,
        )



if __name__ == "__main__":
    unittest.main()
