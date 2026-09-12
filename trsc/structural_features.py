"""Strictly causal structural evidence used by the TRSC decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor


FEATURE_NAMES = (
    "present_now",
    "frequency_w3",
    "frequency_w5",
    "recency",
    "consecutive_length",
    "source_out_degree",
    "target_in_degree",
    "preferential_attachment",
    "successor_jaccard",
    "predecessor_jaccard",
    "directed_two_hop_support",
    "historical_activity_support",
)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = len(left | right)
    return float(len(left & right) / union) if union else 0.0


@dataclass
class _SnapshotState:
    edges: set[tuple[int, int]]
    outgoing: list[set[int]]
    incoming: list[set[int]]
    active: set[int]


class CausalStructuralFeatures:
    """Build structural evidence using observations no later than query time."""

    def __init__(
        self,
        adjacency: Sequence[Tensor],
        history_window: int = 3,
    ) -> None:
        if not adjacency:
            raise ValueError("adjacency must contain at least one snapshot")
        self.adjacency = [snapshot.coalesce() for snapshot in adjacency]
        self.node_count = self.adjacency[0].size(0)
        self.history_window = history_window
        self.states = [self._state(snapshot) for snapshot in self.adjacency]

    def _state(self, snapshot: Tensor) -> _SnapshotState:
        indices = snapshot.indices().detach().cpu()
        values = snapshot.values().detach().cpu()
        outgoing = [set() for _ in range(self.node_count)]
        incoming = [set() for _ in range(self.node_count)]
        edges: set[tuple[int, int]] = set()
        active: set[int] = set()
        for position, (source, target) in enumerate(indices.t().tolist()):
            if source == target or values[position].item() == 0:
                continue
            pair = (int(source), int(target))
            edges.add(pair)
            outgoing[source].add(target)
            incoming[target].add(source)
            active.add(source)
            active.add(target)
        return _SnapshotState(edges, outgoing, incoming, active)

    def _frequency(
        self, pair: tuple[int, int], time_index: int, window: int
    ) -> float:
        start = max(0, time_index - window + 1)
        observations = time_index - start + 1
        count = sum(
            pair in self.states[t].edges for t in range(start, time_index + 1)
        )
        return float(count / max(observations, 1))

    def _recency(self, pair: tuple[int, int], time_index: int) -> float:
        start = max(0, time_index - 12)
        for previous in range(time_index, start - 1, -1):
            if pair in self.states[previous].edges:
                lag = time_index - previous
                return math.exp(-lag / max(float(self.history_window), 1.0))
        return 0.0

    def _consecutive(self, pair: tuple[int, int], time_index: int) -> float:
        length = 0
        for previous in range(time_index, max(-1, time_index - 5), -1):
            if pair not in self.states[previous].edges:
                break
            length += 1
        return float(length / 5.0)

    def _historical_activity(
        self, source: int, target: int, time_index: int
    ) -> float:
        if time_index == 0:
            return 1.0
        start = max(0, time_index - self.history_window)
        observations = time_index - start
        source_activity = sum(
            source in self.states[t].active for t in range(start, time_index)
        ) / max(observations, 1)
        target_activity = sum(
            target in self.states[t].active for t in range(start, time_index)
        ) / max(observations, 1)
        return math.sqrt(source_activity * target_activity)

    def _one(self, source: int, target: int, time_index: int) -> list[float]:
        state = self.states[time_index]
        pair = (source, target)
        source_out = state.outgoing[source]
        target_out = state.outgoing[target]
        source_in = state.incoming[source]
        target_in = state.incoming[target]
        source_degree = len(source_out)
        target_degree = len(target_in)
        degree_denominator = max(math.log1p(self.node_count), 1.0)
        preferential = math.log1p(source_degree * target_degree) / (
            2.0 * degree_denominator
        )
        two_hop_denominator = math.sqrt(
            max(len(source_out), 1) * max(len(target_in), 1)
        )
        two_hop = len(source_out & target_in) / two_hop_denominator
        return [
            float(pair in state.edges),
            self._frequency(pair, time_index, 3),
            self._frequency(pair, time_index, 5),
            self._recency(pair, time_index),
            self._consecutive(pair, time_index),
            math.log1p(source_degree) / degree_denominator,
            math.log1p(target_degree) / degree_denominator,
            min(preferential, 1.0),
            _jaccard(source_out, target_out),
            _jaccard(source_in, target_in),
            min(float(two_hop), 1.0),
            self._historical_activity(source, target, time_index),
        ]

    def query(
        self,
        edge_nodes: Tuple[Tensor, Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        sources, targets = edge_nodes
        sources = sources.detach().cpu().long()
        targets = targets.detach().cpu().long()
        if sources.ndim != 1 or targets.ndim != 1:
            raise ValueError("source and target query tensors must be one-dimensional")
        if sources.numel() != targets.numel():
            raise ValueError("source and target query tensors must have equal length")
        rows = []
        for source_index, target_index in zip(
            sources.tolist(), targets.tolist()
        ):
            time_index = source_index // self.node_count
            target_time_index = target_index // self.node_count
            source = source_index % self.node_count
            target = target_index % self.node_count
            if time_index < 0 or time_index >= len(self.states):
                raise IndexError(
                    f"query time {time_index} is outside adjacency history"
                )
            if target_time_index != time_index:
                raise ValueError(
                    "the source and target of a query must use the same time index"
                )
            rows.append(self._one(source, target, time_index))
        if not rows:
            return torch.empty(
                (0, len(FEATURE_NAMES)), device=device, dtype=dtype
            )
        return torch.tensor(rows, device=device, dtype=dtype)


def structural_reliability_prior(features: Tensor) -> Tensor:
    """Return the fixed causal reliability prior used by the decoder."""

    raw = (
        0.28 * features[:, 1]
        + 0.22 * features[:, 2]
        + 0.18 * features[:, 3]
        + 0.10 * features[:, 4]
        + 0.08 * features[:, 8]
        + 0.08 * features[:, 10]
        + 0.06 * features[:, 11]
    )
    return (0.10 + 0.85 * raw).clamp(0.05, 0.95)


__all__ = [
    "CausalStructuralFeatures",
    "FEATURE_NAMES",
    "structural_reliability_prior",
]
