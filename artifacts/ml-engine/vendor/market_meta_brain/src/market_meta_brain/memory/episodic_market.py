from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from market_meta_brain.domain.types import GovernanceEpisode, GovernanceOutcome


class EpisodicMarketMemory:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._buffer: deque[GovernanceEpisode] = deque(maxlen=capacity)

    def push(self, episode: GovernanceEpisode) -> None:
        self._buffer.append(episode)

    def recent(self, n: int = 20) -> list[GovernanceEpisode]:
        return list(self._buffer)[-n:]

    def __len__(self) -> int:
        return len(self._buffer)

    def summarize_rewards(self, n: int = 50) -> dict[str, float]:
        items = self.recent(n)
        if not items:
            return {"count": 0.0, "avg_reward": 0.0}
        avg_reward = sum(x.reward for x in items) / len(items)
        return {"count": float(len(items)), "avg_reward": float(avg_reward)}

    def state_dict(self) -> dict:
        """JSON-serialisable snapshot of every episode in the buffer.

        Used by the api-server checkpoint path and by the Task #467
        replay so episodic memory survives an ml-engine restart instead
        of evaporating into a 50-row reward summary. Episodes are
        emitted in chronological order (oldest first) so a reload
        reconstructs the deque exactly.
        """
        return {
            "capacity": self.capacity,
            "episodes": [asdict(episode) for episode in self._buffer],
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> int:
        """Hydrate the buffer from a `state_dict()` payload.

        Returns the number of episodes restored. Silently skips
        malformed entries so a partially-corrupted snapshot still
        reloads as much state as possible — losing one episode is
        always preferable to losing the whole buffer.
        """
        if not isinstance(snapshot, dict):
            return 0
        capacity = snapshot.get("capacity")
        if isinstance(capacity, int) and capacity > 0:
            self.capacity = capacity
            self._buffer = deque(self._buffer, maxlen=capacity)
        episodes = snapshot.get("episodes")
        if not isinstance(episodes, list):
            return 0
        restored = 0
        for raw in episodes:
            if not isinstance(raw, dict):
                continue
            outcome_raw = raw.get("outcome")
            if not isinstance(outcome_raw, dict):
                continue
            try:
                outcome = GovernanceOutcome(
                    realized_pnl=float(outcome_raw.get("realized_pnl", 0.0)),
                    realized_drawdown=float(outcome_raw.get("realized_drawdown", 0.0)),
                    realized_stability=float(outcome_raw.get("realized_stability", 0.0)),
                    turnover_cost=float(outcome_raw.get("turnover_cost", 0.0)),
                    action_churn=float(outcome_raw.get("action_churn", 0.0)),
                    correct_defense=_opt_float(outcome_raw.get("correct_defense")),
                    correct_suppression=_opt_float(outcome_raw.get("correct_suppression")),
                    missed_edge_cost=_opt_float(outcome_raw.get("missed_edge_cost")),
                )
                episode = GovernanceEpisode(
                    timestamp=raw.get("timestamp"),
                    meta_state_vector=[float(x) for x in raw.get("meta_state_vector", [])],
                    dominant_regime=str(raw.get("dominant_regime", "unknown")),
                    family_snapshot={
                        str(k): float(v) for k, v in (raw.get("family_snapshot") or {}).items()
                    },
                    action_summary={
                        str(k): float(v) for k, v in (raw.get("action_summary") or {}).items()
                    },
                    reward=float(raw.get("reward", 0.0)),
                    outcome=outcome,
                )
            except (TypeError, ValueError):
                continue
            self._buffer.append(episode)
            restored += 1
        return restored


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
