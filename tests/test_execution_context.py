
import unittest
from datetime import datetime, timezone

from app.execution_context import Event, ExecutionContext
from app.llm_client import Message


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

    def test_truncate_history_keeps_system_and_latest(self):
        context = ExecutionContext()
        context.create_event("system", "You are a helper.")
        context.create_event("user", "first")
        context.create_event("assistant", "a1")
        context.create_event("user", "second")
        context.create_event("assistant", "a2")
        context.create_event("user", "third")

        context.truncate_history(max_events=3, keep_system=True)

        # system + 2 latest non-system events
        self.assertEqual(len(context.events), 3)
        self.assertEqual(context.events[0].type, "system")
        self.assertEqual(context.events[0].content, "You are a helper.")
        self.assertEqual(context.events[1].content, "a2")
        self.assertEqual(context.events[2].content, "third")

    def test_truncate_history_noop_when_under_limit(self):
        context = ExecutionContext()
        context.create_event("user", "hello")

        context.truncate_history(max_events=10)

        self.assertEqual(len(context.events), 1)

    def test_truncate_history_rejects_invalid_limit(self):
        context = ExecutionContext()
        with self.assertRaises(ValueError):
            context.truncate_history(max_events=0)


if __name__ == "__main__":
    unittest.main()
