"""TraceEvent + TraceEmitter — pub/sub for live agent step events.

Subscribers attach via `emitter.subscribe()` and receive an asyncio.Queue they can
drain. The ReAct loop calls `emit()` at every step. SSE handlers + OTel exporters
both consume the same stream.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """One event in the live agent trace."""

    kind: str  # loop.start | llm.response | tool.start | tool.end | final.answer | loop.halt | parse.retry
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class TraceEmitter:
    """Fan-out async pub/sub. Each subscriber gets its own bounded queue.

    Slow subscribers drop events (no backpressure on the loop). This is fine
    for the live trace UI use case — if the user navigates away, we don't
    want the agent run to stall.
    """

    def __init__(self, queue_maxsize: int = 256) -> None:
        self._subscribers: list[asyncio.Queue[TraceEvent]] = []
        self._queue_maxsize = queue_maxsize
        # History buffer so late subscribers can catch up to a finished run
        self._history: list[TraceEvent] = []
        self._history_max = 512

    def subscribe(self) -> asyncio.Queue[TraceEvent]:
        q: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[TraceEvent]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def history(self) -> list[TraceEvent]:
        return list(self._history)

    async def emit(self, event: TraceEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max :]
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the event for this subscriber; they're falling behind.
                pass
