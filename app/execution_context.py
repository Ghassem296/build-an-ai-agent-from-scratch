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
        """Trim the oldest events while preserving the system prompt.

        Keeps the most recent ``max_events`` events. If ``keep_system``
        is True, the leading system events (if any) are retained even
        when they fall outside the window.

        This prevents the conversation from exceeding the model's
        context window during long multi-step tool-use sessions.
        """

        if max_events < 1:
            raise ValueError("max_events must be at least 1")

        if len(self.events) <= max_events:
            return

        # Preserve every system event (they hold the agent instructions).
        if keep_system:
            system_events = [
                event
                for event in self.events
                if event.type == "system"
            ]
        else:
            system_events = []

        # The rest of the budget goes to the newest non-system events.
        non_system_events = [
            event
            for event in self.events
            if not (keep_system and event.type == "system")
        ]

        remaining = max(1, max_events - len(system_events))
        kept_non_system = non_system_events[-remaining:]

        # Rebuild the list preserving the original order.
        kept_ids = {id(e) for e in system_events} | {
            id(e) for e in kept_non_system
        }

        self.events = [
            event for event in self.events if id(event) in kept_ids
        ]

    def __len__(self) -> int:
        return len(self.events)