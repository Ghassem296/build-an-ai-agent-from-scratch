import unittest
from datetime import datetime, timezone

from app.execution_context import Event, ExecutionContext
from app.llm_client import Message, ToolCall, ToolResult


class TestEvent(unittest.TestCase):
    def test_event_creation(self):
        context = ExecutionContext()

        event = Event(
            type="user",
            content=[
                Message(
                    role="user",
                    content="Find recent AI news",
                )
            ],
            execution_id=context.execution_id,
        )

        self.assertEqual(event.type, "user")
        self.assertEqual(event.execution_id, context.execution_id)
        self.assertIsInstance(event.created_at, datetime)
        self.assertEqual(event.created_at.tzinfo, timezone.utc)
        self.assertEqual(len(event.content), 1)
        self.assertEqual(event.content[0].content, "Find recent AI news")

    def test_event_has_utc_timestamp_by_default(self):
        event = Event(
            type="assistant",
            content="Hello",
            execution_id="execution-1",
        )

        self.assertIsNotNone(event.created_at.tzinfo)
        self.assertEqual(event.created_at.utcoffset().total_seconds(), 0)


class TestExecutionContext(unittest.TestCase):
    def test_context_initialization(self):
        context = ExecutionContext()

        self.assertTrue(context.execution_id)
        self.assertEqual(context.events, [])
        self.assertEqual(len(context), 0)

    def test_each_context_has_unique_id(self):
        first = ExecutionContext()
        second = ExecutionContext()

        self.assertNotEqual(first.execution_id, second.execution_id)

    def test_custom_execution_id(self):
        context = ExecutionContext(execution_id="execution-123")

        self.assertEqual(context.execution_id, "execution-123")

    def test_create_event(self):
        context = ExecutionContext()

        event = context.create_event(
            "user",
            [Message(role="user", content="Hello")],
        )

        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, "user")
        self.assertEqual(event.execution_id, context.execution_id)
        self.assertEqual(len(context.events), 1)
        self.assertIs(context.events[0], event)

    def test_add_event_rejects_non_event(self):
        context = ExecutionContext()

        with self.assertRaises(TypeError):
            context.add_event("not an event")

    def test_add_event_rejects_different_execution_id(self):
        context = ExecutionContext()

        event = Event(
            type="assistant",
            content="Hello",
            execution_id="another-execution",
        )

        with self.assertRaises(ValueError):
            context.add_event(event)

        self.assertEqual(len(context.events), 0)

    def test_get_events_returns_copy(self):
        context = ExecutionContext()
        context.create_event("user", "Hello")

        events = context.get_events()
        events.clear()

        self.assertEqual(len(context.events), 1)

    def test_clear_removes_all_events(self):
        context = ExecutionContext()
        context.create_event("user", "Hello")
        context.create_event("assistant", "Hi")

        context.clear()

        self.assertEqual(context.events, [])
        self.assertEqual(len(context), 0)

    def test_len_returns_event_count(self):
        context = ExecutionContext()

        self.assertEqual(len(context), 0)

        context.create_event("user", "Hello")
        self.assertEqual(len(context), 1)

        context.create_event("assistant", "Hi")
        self.assertEqual(len(context), 2)

    # ============================================================
    # truncate_history tests (turn-based behavior)
    # ============================================================

    def test_truncate_history_noop_when_under_limit(self):
        context = ExecutionContext()
        context.create_event("user", "hello")

        context.truncate_history(max_events=10)

        self.assertEqual(len(context.events), 1)

    def test_truncate_history_rejects_invalid_limit(self):
        context = ExecutionContext()
        with self.assertRaises(ValueError):
            context.truncate_history(max_events=0)

    def test_truncate_history_keeps_system_and_latest_turn(self):
        """The system prompt and the most recent turn (even if partial)
        are preserved.
        """
        context = ExecutionContext()
        context.create_event("system", "You are a helper.")
        context.create_event("user", [Message(role="user", content="first")])
        context.create_event(
            "assistant",
            [Message(role="assistant", content="a1")],
        )
        context.create_event("user", [Message(role="user", content="second")])
        context.create_event(
            "assistant",
            [Message(role="assistant", content="a2")],
        )
        context.create_event("user", [Message(role="user", content="third")])

        context.truncate_history(max_events=3, keep_system=True)

        # The system prompt is preserved.
        # The most recent partial turn ("third") is also preserved so
        # the user's current question is never lost.
        # Turn 2 does not fit in the remaining budget (1 + 2 > 2).
        self.assertEqual(len(context.events), 2)
        self.assertEqual(context.events[0].type, "system")
        self.assertEqual(context.events[0].content, "You are a helper.")
        self.assertEqual(context.events[1].content[0].content, "third")

    def test_truncate_history_preserves_tool_call_pairs(self):
        """Truncation must not split a tool_call from its tool_result."""
        context = ExecutionContext()
        context.create_event("system", "sys")

        # Turn 1: user + assistant(tool_call) + tool(result) + assistant(text)
        context.create_event("user", [Message(role="user", content="q1")])
        context.create_event(
            "assistant",
            [
                ToolCall(
                    id="call-1",
                    name="search_web",
                    arguments={"query": "x"},
                )
            ],
        )
        context.create_event(
            "tool",
            [
                ToolResult(
                    tool_call_id="call-1",
                    name="search_web",
                    content="r1",
                )
            ],
        )
        context.create_event(
            "assistant",
            [Message(role="assistant", content="a1")],
        )

        # Turn 2: user + assistant(text)
        context.create_event("user", [Message(role="user", content="q2")])
        context.create_event(
            "assistant",
            [Message(role="assistant", content="a2")],
        )

        context.truncate_history(max_events=3, keep_system=True)

        # We should keep: system + turn 2 (user q2, assistant a2).
        # Turn 1 must be dropped entirely (not partially).
        types = [e.type for e in context.events]
        self.assertEqual(types, ["system", "user", "assistant"])
        self.assertEqual(context.events[1].content[0].content, "q2")
        self.assertEqual(context.events[2].content[0].content, "a2")

    def test_truncate_history_keeps_latest_turn_even_if_too_big(self):
        """The most recent turn must always be kept, even if it exceeds
        the budget. Otherwise the user's current question is lost.
        """
        context = ExecutionContext()
        context.create_event("system", "sys")

        # One large turn: user + assistant(tool) + tool + assistant
        context.create_event("user", [Message(role="user", content="q1")])
        context.create_event(
            "assistant",
            [ToolCall(id="c1", name="search_web", arguments={"query": "x"})],
        )
        context.create_event(
            "tool",
            [ToolResult(tool_call_id="c1", name="search_web", content="r1")],
        )
        context.create_event(
            "assistant",
            [Message(role="assistant", content="a1")],
        )

        # Budget of 5, but the turn alone has 4 events.
        # The turn should still be kept.
        context.truncate_history(max_events=5, keep_system=True)

        types = [e.type for e in context.events]
        self.assertEqual(
            types,
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(context.events[1].content[0].content, "q1")


    def test_new_user_message_starts_new_turn_after_tool_result(self):
        """A later user turn must not be grouped with an unfinished old turn."""
        context = ExecutionContext()
        context.create_event("system", "sys")
        context.create_event("user", [Message(role="user", content="old question")])
        context.create_event(
            "assistant",
            [ToolCall(id="c1", name="search_web", arguments={"query": "x"})],
        )
        context.create_event(
            "tool",
            [ToolResult(tool_call_id="c1", name="search_web", content="result")],
        )
        context.create_event("user", [Message(role="user", content="new question")])
        context.create_event(
            "assistant",
            [Message(role="assistant", content="new answer")],
        )

        context.truncate_history(max_events=3, keep_system=True)

        self.assertEqual(
            [event.type for event in context.events],
            ["system", "user", "assistant"],
        )
        self.assertEqual(context.events[1].content[0].content, "new question")
        self.assertEqual(context.events[2].content[0].content, "new answer")

    def test_total_chars_and_drop_oldest_turn(self):
        """The char budget helper must drop only the oldest turn."""
        context = ExecutionContext()
        context.create_event(
            "system", [Message(role="system", content="sys")]
        )
        context.create_event(
            "user", [Message(role="user", content="a" * 10)]
        )
        context.create_event(
            "assistant", [Message(role="assistant", content="b" * 10)]
        )
        context.create_event(
            "user", [Message(role="user", content="c" * 10)]
        )
        context.create_event(
            "assistant", [Message(role="assistant", content="d" * 10)]
        )

        self.assertEqual(context.total_chars(), 43)
        self.assertTrue(context.drop_oldest_turn())
        self.assertEqual(len(context), 3)
        self.assertFalse(context.drop_oldest_turn())

if __name__ == "__main__":
    unittest.main()
