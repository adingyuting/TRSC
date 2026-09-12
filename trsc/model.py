"""Standalone TRSC model for reliable dynamic-graph link prediction."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import (
    HistoricalActivityGate,
    StructuralEvidenceDecoder,
    TensorGraphConvolution,
)
from .structural_features import (
    CausalStructuralFeatures,
    FEATURE_NAMES,
    structural_reliability_prior,
)


def _temporal_mixing(
    time_slices: int, bandwidth: int, choice: int
) -> Tensor:
    if choice not in (1, 2):
        raise ValueError("mixing_choice must be 1 or 2")
    mixing = torch.zeros(time_slices, time_slices, dtype=torch.float32)
    for current in range(time_slices):
        start = max(0, current - bandwidth + 1)
        for history in range(start, current + 1):
            lag = current - history
            mixing[current, history] = 1.0 if choice == 1 else 1.0 / (lag + 1)
    return mixing / mixing.sum(dim=1, keepdim=True).clamp_min(1e-12)


class TRSC(nn.Module):
    """Tensor backbone with two evidence-guided reliability stages.

    The first stage softly assimilates observed edges using strictly past
    endpoint activity. The second jointly decodes links from tensor embeddings
    and causal, multi-scale structural evidence.

    ``edge_nodes`` contains flattened node indices. A node ``v`` at time ``t``
    is represented by ``t * N + v``.
    """

    def __init__(
        self,
        time_slices: int,
        N: int,
        hidden_features: Sequence[int],
        num_feature: int,
        out_features: int,
        bandwidth: int,
        tgc_dropout: float = 0.75,
        activity_window: int = 3,
        min_edge_weight: float = 0.25,
        decoder_hidden_dim: int = 32,
        decoder_dropout: float = 0.1,
        mixing_choice: int = 2,
    ) -> None:
        super().__init__()
        if not hidden_features:
            raise ValueError("hidden_features must contain at least one layer")
        if out_features != 1:
            raise ValueError("TRSC currently supports binary link prediction only")
        self.time_slices = time_slices
        self.N = N
        self.X = nn.Parameter(torch.empty(time_slices, N, num_feature))
        dimensions = [num_feature, *hidden_features]
        self.tensor_convolutions = nn.ModuleList(
            [
                TensorGraphConvolution(
                    time_slices,
                    dimensions[index],
                    dimensions[index + 1],
                    bandwidth,
                    tgc_dropout,
                )
                for index in range(len(hidden_features))
            ]
        )
        self.activation = nn.ReLU()
        self.activity_gate = HistoricalActivityGate(
            history_window=activity_window,
            min_edge_weight=min_edge_weight,
            preserve_row_mass=True,
        )
        self.decoder = StructuralEvidenceDecoder(
            node_features=hidden_features[-1],
            structural_features=len(FEATURE_NAMES),
            hidden_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
        )
        self.register_buffer(
            "default_mixing",
            _temporal_mixing(time_slices, bandwidth, mixing_choice),
        )
        self._activity_cache: dict[tuple, tuple[list[Tensor], Tensor]] = {}
        self._builder_cache: dict[tuple, CausalStructuralFeatures] = {}
        self._query_cache: dict[tuple, Tensor] = {}
        self.last_diagnostics: dict[str, Tensor] = {}
        nn.init.xavier_normal_(self.X)

    @property
    def n_f(self) -> nn.Parameter:
        """Compatibility alias for regularizing the learned node features."""

        return self.X

    @staticmethod
    def _adjacency_key(adjacency: Sequence[Tensor]) -> tuple:
        return tuple(
            (
                snapshot.indices().data_ptr(),
                snapshot.values().data_ptr(),
                snapshot._nnz(),
                tuple(snapshot.size()),
            )
            for snapshot in adjacency
        )

    @staticmethod
    def _query_key(
        adjacency_key: tuple, edge_nodes: Tuple[Tensor, Tensor]
    ) -> tuple:
        source, target = edge_nodes
        return (
            adjacency_key,
            source.data_ptr(),
            target.data_ptr(),
            source.numel(),
            target.numel(),
        )

    def clear_runtime_cache(self) -> None:
        """Release cached adjacency preprocessing and structural evidence."""

        self._activity_cache.clear()
        self._builder_cache.clear()
        self._query_cache.clear()

    def _prepare_adjacency(self, adjacency: Sequence[Tensor]) -> list[Tensor]:
        if len(adjacency) != self.time_slices:
            raise ValueError(
                f"expected {self.time_slices} snapshots, got {len(adjacency)}"
            )
        snapshots = [
            snapshot.to(device=self.X.device, dtype=self.X.dtype).coalesce()
            for snapshot in adjacency
        ]
        for snapshot in snapshots:
            if snapshot.layout != torch.sparse_coo:
                raise TypeError("every adjacency snapshot must be sparse COO")
            if snapshot.shape != (self.N, self.N):
                raise ValueError(
                    f"expected adjacency shape ({self.N}, {self.N}), "
                    f"got {tuple(snapshot.shape)}"
                )
        return snapshots

    def _causal_evidence(
        self,
        adjacency: Sequence[Tensor],
        edge_nodes: Tuple[Tensor, Tensor],
    ) -> Tensor:
        adjacency_key = self._adjacency_key(adjacency)
        if adjacency_key not in self._builder_cache:
            self._builder_cache[adjacency_key] = CausalStructuralFeatures(
                adjacency,
                history_window=self.activity_gate.history_window,
            )
        query_key = self._query_key(adjacency_key, edge_nodes)
        if query_key not in self._query_cache:
            self._query_cache[query_key] = self._builder_cache[
                adjacency_key
            ].query(edge_nodes, self.X.device, self.X.dtype)
        return self._query_cache[query_key]

    def forward(
        self,
        adjacency: Sequence[Tensor],
        edge_nodes: Tuple[Tensor, Tensor],
        mixing: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        snapshots = self._prepare_adjacency(adjacency)
        adjacency_key = self._adjacency_key(snapshots)
        if adjacency_key not in self._activity_cache:
            self._activity_cache[adjacency_key] = self.activity_gate(snapshots)
        filtered, mean_activity_gates = self._activity_cache[adjacency_key]

        hidden = self.X
        temporal_mixing = self.default_mixing if mixing is None else mixing
        for layer_index, convolution in enumerate(self.tensor_convolutions):
            hidden = convolution(filtered, hidden, temporal_mixing)
            if layer_index < len(self.tensor_convolutions) - 1:
                hidden = self.activation(hidden)

        evidence = self._causal_evidence(snapshots, edge_nodes)
        reliability_prior = structural_reliability_prior(evidence)
        output = self.decoder(
            hidden,
            edge_nodes,
            evidence,
            reliability_prior,
        )
        self.last_diagnostics = {
            "mean_activity_gate": mean_activity_gates.mean().detach(),
            "mean_structural_prior": reliability_prior.mean().detach(),
            "std_structural_prior": reliability_prior.std(
                unbiased=False
            ).detach(),
        }
        return output, hidden


# Backward-compatible names used by the source research project.
ERGTCN = TRSC
ReliableTensorDynamicGraph = TRSC

__all__ = ["ERGTCN", "ReliableTensorDynamicGraph", "TRSC"]

