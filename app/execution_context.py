from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Event:
    """An event produced during an agent execution."""

    type: str
    content: Any
    execution_id: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ExecutionContext:
    """Stores the event history for one agent execution."""

    def __init__(self, execution_id: str | None = None) -> None:
        self.execution_id = execution_id or str(uuid4())
        self.events: list[Event] = []

    def add_event(self, event: Event) -> None:
        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance")

        if event.execution_id != self.execution_id:
            raise ValueError(
                "Event execution_id does not match this context"
            )

        self.events.append(event)

    def create_event(self, event_type: str, content: Any) -> Event:
        event = Event(
            type=event_type,
            content=content,
            execution_id=self.execution_id,
        )
        self.add_event(event)
        return event

    def get_events(self) -> list[Event]:
        """Return a copy so callers cannot modify the internal list."""
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()

    def truncate_history(
        self,
        max_events: int,
        keep_system: bool = True,
    ) -> None:
        """Trim the oldest events while preserving turn boundaries.

        A "turn" is a coherent group of events that the LLM API needs
        together:

            user
            assistant (with or without tool_calls)
            tool (only if the assistant requested tools)
            assistant (final answer of the turn)

        We cut only at turn boundaries so the truncated history is
        still valid for the API.

        The most recent turn is ALWAYS kept, even if it exceeds the
        budget. This guarantees the user's current question is never
        lost.
        """

        if max_events < 1:
            raise ValueError("max_events must be at least 1")

        if len(self.events) <= max_events:
            return

        # 1. Separate system events (always kept).
        system_events = (
            [e for e in self.events if e.type == "system"]
            if keep_system
            else []
        )
        non_system = [
            e for e in self.events
            if not (keep_system and e.type == "system")
        ]

        # 2. Group non-system events into turns.
        turns: list[list[Event]] = []
        current_turn: list[Event] = []

        for event in non_system:
            current_turn.append(event)

            if event.type == "assistant" and not _contains_tool_call(
                event.content
            ):
                turns.append(current_turn)
                current_turn = []

        if current_turn:
            turns.append(current_turn)

        # 3. Keep the newest turns that fit in the budget.
        #    The most recent turn is always kept.
        remaining = max(1, max_events - len(system_events))
        kept_turns: list[list[Event]] = []
        used = 0

        for turn in reversed(turns):
            if not kept_turns:
                # Always keep the most recent turn.
                kept_turns.append(turn)
                used += len(turn)
                continue

            if used + len(turn) > remaining:
                break

            kept_turns.append(turn)
            used += len(turn)

        kept_turns.reverse()

        # 4. Rebuild the event list preserving original order.
        kept_ids = {id(e) for e in system_events}
        for turn in kept_turns:
            for e in turn:
                kept_ids.add(id(e))

        self.events = [
            e for e in self.events if id(e) in kept_ids
        ]

    def __len__(self) -> int:
        return len(self.events)


def _contains_tool_call(content: Any) -> bool:
    """Return True if the content contains at least one ToolCall-like item."""
    items = content if isinstance(content, list) else [content]

    for item in items:
        if hasattr(item, "name") and hasattr(item, "arguments"):
            return True

    return False