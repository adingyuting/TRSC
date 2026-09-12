"""Data loading and preprocessing utilities from the original TRSC project.

The research data are stored as sparse tensors in MATLAB ``.mat`` files. This
module loads that representation, creates temporal adjacency windows, samples
negative links, and converts edge triples into the flattened node indices
expected by :class:`trsc.TRSC`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch import Tensor


@dataclass(frozen=True)
class DatasetSpec:
    """Location and expected number of time slices for one dataset."""

    directory: str
    filename: str
    time_slices: int


DATASETS: Mapping[str, DatasetSpec] = {
    "wiki_gl": DatasetSpec("wiki_gl", "wiki_gl.mat", 60),
    "wiki_eo": DatasetSpec("wiki_eo", "wiki_eo.mat", 60),
    "digg": DatasetSpec("digg", "digg.mat", 50),
    "bitcoin_alpha": DatasetSpec("bitcoin_alpha", "bitcoin_alpha.mat", 60),
    "bitcoin_otc": DatasetSpec("bitcoin_otc", "bitcoin_otc.mat", 60),
    "dblp": DatasetSpec("dblp", "dblp.mat", 45),
    "last_fm": DatasetSpec("last_fm", "last_fm.mat", 53),
}


@dataclass
class LoadedDynamicGraph:
    """All tensors required by the original training protocol."""

    time_slices: int
    labels: Tensor
    train_adjacency: list[Tensor]
    validation_adjacency: list[Tensor]
    test_adjacency: list[Tensor]
    num_nodes: int

    def as_legacy_tuple(
        self,
    ) -> tuple[int, Tensor, list[Tensor], list[Tensor], list[Tensor], int]:
        """Return the tuple used by the original ``DataLoader.load_data``."""

        return (
            self.time_slices,
            self.labels,
            self.train_adjacency,
            self.validation_adjacency,
            self.test_adjacency,
            self.num_nodes,
        )


def _indices(array: np.ndarray, name: str) -> np.ndarray:
    """Normalize MATLAB sparse indices to shape ``[rank, entries]``."""

    result = np.asarray(array, dtype=np.int64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if result.shape[0] != 3 and result.shape[1] == 3:
        result = result.T
    if result.shape[0] != 3:
        raise ValueError(f"{name} must have shape [3, E] or [E, 3]")
    return result


def _values(array: np.ndarray, entries: int, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32).reshape(-1)
    if result.size != entries:
        raise ValueError(
            f"{name} contains {result.size} values for {entries} indices"
        )
    return result


def _sparse_tensor(
    payload: Mapping[str, np.ndarray],
    name: str,
    size: torch.Size,
    device: torch.device,
) -> Tensor:
    index_name = f"{name}_idx"
    value_name = f"{name}_vals"
    if index_name not in payload or value_name not in payload:
        raise KeyError(f"MAT file is missing {index_name!r} or {value_name!r}")
    indices = _indices(payload[index_name], index_name)
    values = _values(payload[value_name], indices.shape[1], value_name)
    if indices.size and (
        np.any(indices < 0)
        or any(np.any(indices[axis] >= size[axis]) for axis in range(3))
    ):
        raise ValueError(f"{index_name} contains an index outside {tuple(size)}")
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device=device),
        torch.from_numpy(values).to(device=device),
        size,
        dtype=torch.float32,
        device=device,
    ).coalesce()


def _snapshots(tensor: Tensor, count: int, num_nodes: int) -> list[Tensor]:
    tensor = tensor.coalesce()
    indices = tensor.indices()
    values = tensor.values()
    result: list[Tensor] = []
    for time_index in range(count):
        mask = indices[0] == time_index
        result.append(
            torch.sparse_coo_tensor(
                indices[1:, mask],
                values[mask],
                (num_nodes, num_nodes),
                dtype=values.dtype,
                device=values.device,
            ).coalesce()
        )
    return result


def _negative_edges(
    num_nodes: int,
    edge_index: Tensor,
    num_samples: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    existing = {
        (int(source), int(target))
        for source, target in edge_index.detach().cpu().t().tolist()
    }
    forbidden = existing | {(target, source) for source, target in existing}
    forbidden_non_self = {
        pair for pair in forbidden if pair[0] != pair[1]
    }
    maximum = num_nodes * (num_nodes - 1) - len(forbidden_non_self)
    if num_samples > maximum:
        raise ValueError(
            f"cannot sample {num_samples} non-edges from a graph with "
            f"{num_nodes} nodes and {len(existing)} stored edges"
        )

    samples: set[tuple[int, int]] = set()
    while len(samples) < num_samples:
        source = rng.randrange(num_nodes)
        target = rng.randrange(num_nodes)
        pair = (source, target)
        if source == target or pair in forbidden or pair in samples:
            continue
        samples.add(pair)
    return list(samples)


def negative_sample(
    adjacency: Tensor,
    ratio: float = 1.0,
    seed: int = 2024,
) -> Tensor:
    """Add explicit zero-valued non-edges to a sparse temporal tensor."""

    if ratio < 0:
        raise ValueError("negative-sampling ratio must be non-negative")
    adjacency = adjacency.coalesce()
    time_slices, num_nodes, _ = adjacency.shape
    rng = random.Random(seed)
    all_indices: list[Tensor] = []
    all_values: list[Tensor] = []

    for time_index in range(time_slices):
        mask = adjacency.indices()[0] == time_index
        positive_index = adjacency.indices()[1:, mask]
        positive_values = adjacency.values()[mask]
        positive_count = positive_index.size(1)
        negative_count = int(positive_count * ratio)

        time_row = torch.full(
            (1, positive_count),
            time_index,
            dtype=torch.long,
            device=adjacency.device,
        )
        all_indices.append(torch.cat([time_row, positive_index], dim=0))
        all_values.append(positive_values)

        if negative_count:
            sampled = _negative_edges(
                num_nodes, positive_index, negative_count, rng
            )
            negative_index = torch.tensor(
                sampled,
                dtype=torch.long,
                device=adjacency.device,
            ).t()
            negative_time = torch.full(
                (1, negative_count),
                time_index,
                dtype=torch.long,
                device=adjacency.device,
            )
            all_indices.append(
                torch.cat([negative_time, negative_index], dim=0)
            )
            all_values.append(
                torch.zeros(
                    negative_count,
                    dtype=adjacency.dtype,
                    device=adjacency.device,
                )
            )

    if not all_indices:
        return torch.sparse_coo_tensor(
            adjacency.shape, dtype=adjacency.dtype, device=adjacency.device
        ).coalesce()
    return torch.sparse_coo_tensor(
        torch.cat(all_indices, dim=1),
        torch.cat(all_values),
        adjacency.shape,
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def load_mat_dataset(
    dataset_name: str,
    data_dir: str | Path = "data",
    device: torch.device | str = "cpu",
    *,
    negative_ratio: float = 1.0,
    seed: int = 2024,
    val_rate: float = 0.1,
    test_rate: float = 0.2,
    check_time_slices: bool = True,
) -> LoadedDynamicGraph:
    """Load one original preprocessed ``.mat`` dynamic-graph dataset."""

    normalized = dataset_name.lower().replace("-", "_")
    if normalized not in DATASETS:
        choices = ", ".join(DATASETS)
        raise ValueError(f"unknown dataset {dataset_name!r}; choose from {choices}")
    if val_rate < 0 or test_rate < 0 or val_rate + test_rate >= 1:
        raise ValueError("val_rate and test_rate must be non-negative and sum to < 1")

    spec = DATASETS[normalized]
    path = Path(data_dir) / spec.directory / spec.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"preprocessed dataset not found: {path}. See data/README.md"
        )
    payload = sio.loadmat(path)
    required = {"tensor_idx", "A_idx", "A_vals", "train_idx", "train_vals",
                "val_idx", "val_vals", "test_idx", "test_vals"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"{path} is missing MAT keys: {', '.join(missing)}")

    tensor_indices = _indices(payload["tensor_idx"], "tensor_idx")
    if tensor_indices.shape[1] == 0:
        raise ValueError("tensor_idx must contain at least one edge")
    inferred_time_slices = int(tensor_indices[0].max()) + 1
    inferred_num_nodes = int(
        max(tensor_indices[1].max(), tensor_indices[2].max())
    ) + 1
    time_slices = int(
        np.asarray(payload.get("time_slices", inferred_time_slices)).reshape(-1)[0]
    )
    num_nodes = int(
        np.asarray(payload.get("num_nodes", inferred_num_nodes)).reshape(-1)[0]
    )
    if check_time_slices and time_slices != spec.time_slices:
        raise ValueError(
            f"{normalized} expects {spec.time_slices} time slices, "
            f"but tensor_idx contains {time_slices}"
        )

    validation_steps = int(time_slices * val_rate)
    test_steps = int(time_slices * test_rate)
    train_steps = time_slices - validation_steps - test_steps
    if train_steps < 2:
        raise ValueError("the temporal split must leave at least two training snapshots")

    resolved_device = torch.device(device)
    full_size = torch.Size((time_slices, num_nodes, num_nodes))
    window_size = torch.Size((train_steps, num_nodes, num_nodes))
    adjacency = _sparse_tensor(payload, "A", full_size, resolved_device)
    labels = negative_sample(adjacency, negative_ratio, seed)
    train = _sparse_tensor(payload, "train", window_size, resolved_device)
    validation = _sparse_tensor(payload, "val", window_size, resolved_device)
    test = _sparse_tensor(payload, "test", window_size, resolved_device)

    return LoadedDynamicGraph(
        time_slices=time_slices,
        labels=labels,
        train_adjacency=_snapshots(train, train_steps, num_nodes),
        validation_adjacency=_snapshots(validation, train_steps, num_nodes),
        test_adjacency=_snapshots(test, train_steps, num_nodes),
        num_nodes=num_nodes,
    )


def load_data(args, device: torch.device | str):
    """Compatibility wrapper accepting the original argparse namespace."""

    if not hasattr(args, "dataset_name"):
        raise AttributeError("args must define dataset_name")
    loaded = load_mat_dataset(
        args.dataset_name,
        getattr(args, "data_dir", "data"),
        device,
        negative_ratio=getattr(args, "negative_ratio", 1.0),
        seed=getattr(args, "seed", 2024),
        val_rate=getattr(args, "val_rate", 0.1),
        test_rate=getattr(args, "test_rate", 0.2),
        check_time_slices=getattr(args, "check_time_slices", True),
    )
    return loaded.as_legacy_tuple()


def get_edge_nodes(edges: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Convert ``[time, source, target]`` triples to flattened model indices."""

    if edges.ndim != 2 or edges.size(0) != 3:
        raise ValueError("edges must have shape [3, E]")
    source = edges[0].long() * num_nodes + edges[1].long()
    target = edges[0].long() * num_nodes + edges[2].long()
    return source, target


def get_all_edge_nodes(
    train_edges: Tensor,
    validation_edges: Tensor,
    test_edges: Tensor,
    num_nodes: int,
) -> tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    return (
        get_edge_nodes(train_edges, num_nodes),
        get_edge_nodes(validation_edges, num_nodes),
        get_edge_nodes(test_edges, num_nodes),
    )


def split_data(
    labels: Tensor,
    time_slices: int,
    val_rate: float = 0.1,
    test_rate: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor, int]:
    """Reproduce the original rolling-window train/validation/test split."""

    labels = labels.coalesce()
    edges = labels.indices().clone()
    values = labels.values()
    validation_steps = int(time_slices * val_rate)
    test_steps = int(time_slices * test_rate)
    train_steps = time_slices - validation_steps - test_steps

    train_mask = edges[0] < train_steps
    train_edges = edges[:, train_mask]
    train_targets = values[train_mask]
    keep = train_edges[0] != 0
    train_targets = train_targets[keep]
    train_edges = train_edges[:, keep]
    train_edges[0] -= 1

    validation_mask = (edges[0] >= validation_steps) & (
        edges[0] < train_steps + validation_steps
    )
    validation_edges = edges[:, validation_mask]
    validation_edges[0] -= validation_steps
    validation_targets = values[validation_mask]
    validation_tail = int(
        (validation_edges[0] > train_steps - validation_steps - 1).sum().item()
    )
    keep = validation_edges[0] != 0
    validation_targets = validation_targets[keep]
    validation_edges = validation_edges[:, keep]
    validation_edges[0] -= 1

    test_mask = edges[0] >= test_steps + validation_steps
    test_edges = edges[:, test_mask]
    test_edges[0] -= test_steps + validation_steps
    test_targets = values[test_mask]
    test_tail = int(
        (test_edges[0] > train_steps - test_steps - 1).sum().item()
    )
    keep = test_edges[0] != 0
    test_targets = test_targets[keep]
    test_edges = test_edges[:, keep]
    test_edges[0] -= 1

    return (
        train_edges,
        train_targets,
        validation_edges,
        validation_targets,
        validation_tail,
        test_edges,
        test_targets,
        test_tail,
    )


def compatibility_args(dataset_name: str, **kwargs) -> SimpleNamespace:
    """Build a small namespace for code that uses ``load_data(args, device)``."""

    return SimpleNamespace(dataset_name=dataset_name, **kwargs)


__all__ = [
    "DATASETS",
    "DatasetSpec",
    "LoadedDynamicGraph",
    "compatibility_args",
    "get_all_edge_nodes",
    "get_edge_nodes",
    "load_data",
    "load_mat_dataset",
    "negative_sample",
    "split_data",
]
