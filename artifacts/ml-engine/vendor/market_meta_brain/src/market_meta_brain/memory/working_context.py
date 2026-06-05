from __future__ import annotations

from collections import deque


class WorkingContext:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._events: deque[dict] = deque(maxlen=capacity)

    def push(self, item: dict) -> None:
        self._events.append(item)

    def recent(self, n: int = 10) -> list[dict]:
        return list(self._events)[-n:]

    def trend(self, key: str, n: int = 20) -> float:
        values = [float(event.get(key, 0.0)) for event in self.recent(n)]
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "events": list(self._events)}
