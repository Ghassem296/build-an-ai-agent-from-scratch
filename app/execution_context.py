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

    def __len__(self) -> int:
        return len(self.events)