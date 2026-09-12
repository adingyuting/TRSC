"""Neural layers used by TRSC."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TensorGraphConvolution(nn.Module):
    """Causal tensor graph convolution over a bounded temporal window."""

    def __init__(
        self,
        time_slices: int,
        in_features: int,
        out_features: int,
        bandwidth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if time_slices < 1:
            raise ValueError("time_slices must be positive")
        if bandwidth < 1:
            raise ValueError("bandwidth must be positive")
        self.time_slices = time_slices
        self.bandwidth = min(bandwidth, time_slices)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)

    def _causal_mixing_matrix(self, mixing: Tensor) -> Tensor:
        if mixing.shape != (self.time_slices, self.time_slices):
            raise ValueError(
                "mixing must have shape "
                f"({self.time_slices}, {self.time_slices}), got {tuple(mixing.shape)}"
            )
        time = torch.arange(self.time_slices, device=mixing.device)
        row = time[:, None]
        column = time[None, :]
        causal_band = (column <= row) & (
            column >= row - self.bandwidth + 1
        )
        return mixing * causal_band.to(dtype=mixing.dtype)

    def _mix_sparse_adjacency(
        self, adjacency: Sequence[Tensor], causal_mixing: Tensor
    ) -> list[Tensor]:
        node_count = adjacency[0].size(0)
        size = torch.Size((node_count, node_count))
        mixed: list[Tensor] = []
        for current_time in range(len(adjacency)):
            start = max(0, current_time - self.bandwidth + 1)
            result = torch.sparse_coo_tensor(
                size,
                device=adjacency[0].device,
                dtype=adjacency[0].dtype,
            )
            for history_time in range(start, current_time + 1):
                coefficient = causal_mixing[current_time, history_time]
                snapshot = adjacency[history_time].coalesce()
                weighted = torch.sparse_coo_tensor(
                    snapshot.indices(),
                    snapshot.values() * coefficient,
                    size,
                    device=snapshot.device,
                    dtype=snapshot.dtype,
                )
                result = result + weighted
            mixed.append(result.coalesce())
        return mixed

    def forward(
        self,
        adjacency: Sequence[Tensor],
        features: Tensor,
        mixing: Tensor,
    ) -> Tensor:
        if len(adjacency) != self.time_slices:
            raise ValueError(
                f"expected {self.time_slices} snapshots, got {len(adjacency)}"
            )
        mixing = mixing.to(device=features.device, dtype=features.dtype)
        causal_mixing = self._causal_mixing_matrix(mixing)
        mixed_adjacency = self._mix_sparse_adjacency(
            adjacency, causal_mixing
        )
        mixed_features = torch.matmul(
            causal_mixing, features.reshape(self.time_slices, -1)
        ).reshape_as(features)
        propagated = torch.stack(
            [
                torch.sparse.mm(snapshot, mixed_features[time_index])
                for time_index, snapshot in enumerate(mixed_adjacency)
            ],
            dim=0,
        )
        return torch.matmul(self.dropout(propagated), self.weight)


class HistoricalActivityGate(nn.Module):
    """Softly assimilate edges according to strictly past endpoint activity."""

    def __init__(
        self,
        history_window: int = 3,
        min_edge_weight: float = 0.25,
        preserve_row_mass: bool = True,
    ) -> None:
        super().__init__()
        if history_window < 1:
            raise ValueError("history_window must be positive")
        if not 0.0 <= min_edge_weight <= 1.0:
            raise ValueError("min_edge_weight must be in [0, 1]")
        self.history_window = history_window
        self.min_edge_weight = min_edge_weight
        self.preserve_row_mass = preserve_row_mass

    @staticmethod
    def _node_activity(snapshot: Tensor) -> Tensor:
        snapshot = snapshot.coalesce()
        indices = snapshot.indices()
        values = snapshot.values()
        non_self = (indices[0] != indices[1]) & (values != 0)
        activity = torch.zeros(
            snapshot.size(0),
            device=snapshot.device,
            dtype=snapshot.dtype,
        )
        if non_self.any():
            active_nodes = torch.cat(
                [indices[0, non_self], indices[1, non_self]], dim=0
            )
            activity[active_nodes.unique()] = 1.0
        return activity

    @staticmethod
    def _preserve_source_mass(
        indices: Tensor,
        original: Tensor,
        gated: Tensor,
        node_count: int,
    ) -> Tensor:
        original_mass = torch.zeros(
            node_count, device=original.device, dtype=original.dtype
        ).index_add(0, indices[0], original.abs())
        gated_mass = torch.zeros_like(original_mass).index_add(
            0, indices[0], gated.abs()
        )
        scale = torch.ones_like(original_mass)
        valid = gated_mass > 0
        scale[valid] = original_mass[valid] / gated_mass[valid]
        return gated * scale[indices[0]]

    def forward(
        self, adjacency: Sequence[Tensor]
    ) -> Tuple[list[Tensor], Tensor]:
        if not adjacency:
            raise ValueError("adjacency must contain at least one snapshot")
        snapshots = [snapshot.coalesce() for snapshot in adjacency]
        activity_history: list[Tensor] = []
        filtered: list[Tensor] = []
        mean_gates: list[Tensor] = []
        for time_index, snapshot in enumerate(snapshots):
            indices = snapshot.indices()
            values = snapshot.values()
            non_self = (indices[0] != indices[1]) & (values != 0)
            edge_gate = torch.ones_like(values)
            if time_index > 0 and non_self.any():
                start = max(0, time_index - self.history_window)
                past_activity = torch.stack(
                    activity_history[start:time_index], dim=0
                ).mean(dim=0)
                source_support = past_activity[indices[0, non_self]]
                target_support = past_activity[indices[1, non_self]]
                joint_support = torch.sqrt(source_support * target_support)
                edge_gate[non_self] = self.min_edge_weight + (
                    1.0 - self.min_edge_weight
                ) * joint_support
            gated_values = values * edge_gate
            if self.preserve_row_mass:
                gated_values = self._preserve_source_mass(
                    indices,
                    values,
                    gated_values,
                    snapshot.size(0),
                )
            filtered.append(
                torch.sparse_coo_tensor(
                    indices,
                    gated_values,
                    snapshot.size(),
                    device=snapshot.device,
                    dtype=snapshot.dtype,
                ).coalesce()
            )
            mean_gates.append(
                edge_gate[non_self].mean()
                if non_self.any()
                else values.new_ones(())
            )
            activity_history.append(self._node_activity(snapshot))
        return filtered, torch.stack(mean_gates)


class StructuralEvidenceDecoder(nn.Module):
    """Decode links from node interactions and causal structural evidence."""

    def __init__(
        self,
        node_features: int,
        structural_features: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = 4 * node_features + structural_features + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: Tensor,
        edge_nodes: Tuple[Tensor, Tensor],
        structural_features: Tensor,
        reliability_prior: Tensor,
    ) -> Tensor:
        source_indices, target_indices = edge_nodes
        flat = embeddings.reshape(-1, embeddings.size(-1))
        source = flat[source_indices.to(device=flat.device, dtype=torch.long)]
        target = flat[target_indices.to(device=flat.device, dtype=torch.long)]
        interactions = torch.cat(
            [
                source,
                target,
                source * target,
                source - target,
                structural_features,
                reliability_prior.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.network(interactions)).squeeze(-1)


__all__ = [
    "HistoricalActivityGate",
    "StructuralEvidenceDecoder",
    "TensorGraphConvolution",
]

