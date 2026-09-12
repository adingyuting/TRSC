"""Convert temporal edge CSV/TSV files into the MATLAB format used by TRSC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from .data import DATASETS


def _column(frame: pd.DataFrame, requested: str | int, role: str):
    if isinstance(requested, str) and requested.isdigit():
        requested = int(requested)
    if requested in frame.columns:
        return requested
    lowered = {str(name).lower(): name for name in frame.columns}
    aliases = {
        "source": ("from", "source", "src", "source_id"),
        "target": ("to", "target", "dst", "target_id"),
        "value": ("value", "weight", "rating", "label"),
        "timestamp": ("timestamp", "time", "date", "year"),
    }
    for candidate in (str(requested).lower(), *aliases[role]):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"cannot find the {role} column {requested!r}; "
        f"available columns: {list(frame.columns)}"
    )


def _timestamps(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float64)
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    if dates.isna().any():
        bad = series[dates.isna()].iloc[0]
        raise ValueError(f"cannot parse timestamp value {bad!r}")
    return dates.astype("int64").to_numpy(dtype=np.float64) / 1e9


def _time_bins(
    timestamps: np.ndarray,
    time_slices: int,
    strategy: str,
) -> np.ndarray:
    if time_slices < 3:
        raise ValueError("time_slices must be at least 3")
    if strategy == "auto":
        unique = np.unique(timestamps)
        consecutive = (
            unique.size == time_slices
            and np.array_equal(unique, np.arange(time_slices))
        )
        strategy = "discrete" if consecutive else "equal-width"
    if strategy == "discrete":
        unique = np.unique(timestamps)
        if unique.size != time_slices:
            raise ValueError(
                f"discrete timestamps contain {unique.size} unique values, "
                f"but {time_slices} time slices were requested"
            )
        lookup = {value: index for index, value in enumerate(unique.tolist())}
        return np.fromiter(
            (lookup[value] for value in timestamps.tolist()),
            dtype=np.int64,
            count=timestamps.size,
        )
    if strategy == "equal-count":
        order = np.argsort(timestamps, kind="stable")
        bins = np.empty(timestamps.size, dtype=np.int64)
        bins[order] = np.minimum(
            np.arange(timestamps.size) * time_slices // timestamps.size,
            time_slices - 1,
        )
        return bins
    if strategy != "equal-width":
        raise ValueError("binning must be auto, discrete, equal-width, or equal-count")
    minimum = float(timestamps.min())
    maximum = float(timestamps.max())
    if minimum == maximum:
        raise ValueError("equal-width binning requires more than one timestamp")
    scaled = (timestamps - minimum) / (maximum - minimum)
    return np.minimum((scaled * time_slices).astype(np.int64), time_slices - 1)


def _coalesced_edges(
    time_index: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    triples = np.column_stack((time_index, source, target)).astype(np.int64)
    triples = triples[triples[:, 1] != triples[:, 2]]
    if not triples.size:
        raise ValueError("the input contains no non-self edges")
    return np.unique(triples, axis=0)


def _normalized_snapshots(
    triples: np.ndarray,
    time_slices: int,
    num_nodes: int,
    make_symmetric: bool,
    edge_life: int,
) -> list[sp.coo_matrix]:
    raw: list[sp.csr_matrix] = []
    for time_index in range(time_slices):
        rows = triples[triples[:, 0] == time_index]
        matrix = sp.coo_matrix(
            (
                np.ones(rows.shape[0], dtype=np.float64),
                (rows[:, 1], rows[:, 2]),
            ),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        matrix.data[:] = 1.0
        raw.append(matrix)

    result: list[sp.coo_matrix] = []
    for time_index in range(time_slices):
        start = max(0, time_index - edge_life + 1) if edge_life else time_index
        matrix = sum(raw[start : time_index + 1], sp.csr_matrix((num_nodes, num_nodes)))
        matrix.data[:] = 1.0
        if make_symmetric:
            matrix = (matrix + matrix.T) * 0.5
        matrix = matrix + sp.eye(num_nodes, dtype=np.float64, format="csr")
        degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
        inverse_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        inverse_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
        scale = sp.diags(inverse_sqrt)
        result.append((scale @ matrix @ scale).tocoo())
    return result


def _window_arrays(
    snapshots: list[sp.coo_matrix],
    start: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for local_time, snapshot in enumerate(snapshots[start : start + length]):
        coo = snapshot.tocoo()
        indices.append(
            np.vstack(
                (
                    np.full(coo.nnz, local_time, dtype=np.int64),
                    coo.row.astype(np.int64),
                    coo.col.astype(np.int64),
                )
            )
        )
        values.append(coo.data.astype(np.float32))
    return np.concatenate(indices, axis=1), np.concatenate(values)


def convert_csv_to_mat(
    input_path: str | Path,
    dataset_name: str,
    output_path: str | Path | None = None,
    *,
    time_slices: int | None = None,
    source_column: str | int = "From",
    target_column: str | int = "To",
    value_column: str | int | None = "Value",
    timestamp_column: str | int = "TimeStamp",
    delimiter: str = ",",
    has_header: bool = True,
    binning: str = "auto",
    make_symmetric: bool = False,
    edge_life: int = 0,
    val_rate: float = 0.1,
    test_rate: float = 0.2,
) -> Path:
    """Convert a four-column temporal edge table to a TRSC ``.mat`` file.

    Node identifiers may be numeric or strings and are remapped to contiguous,
    zero-based indices. Edge values are retained as ``edge_values`` metadata;
    the link-prediction labels in ``A`` are binary edge-presence indicators.
    """

    normalized_name = dataset_name.lower().replace("-", "_")
    aliases = {"alpha": "bitcoin_alpha", "otc": "bitcoin_otc", "lastfm": "last_fm"}
    normalized_name = aliases.get(normalized_name, normalized_name)
    if normalized_name not in DATASETS:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    spec = DATASETS[normalized_name]
    time_slices = spec.time_slices if time_slices is None else time_slices
    if val_rate < 0 or test_rate < 0 or val_rate + test_rate >= 1:
        raise ValueError("val_rate and test_rate must sum to less than one")

    read_delimiter = "\t" if delimiter in {"tab", "\\t"} else delimiter
    header = 0 if has_header else None
    frame = pd.read_csv(input_path, sep=read_delimiter, header=header, comment="#")
    source_name = _column(frame, source_column, "source")
    target_name = _column(frame, target_column, "target")
    timestamp_name = _column(frame, timestamp_column, "timestamp")
    if value_column is None:
        edge_values = np.ones(len(frame), dtype=np.float32)
    else:
        value_name = _column(frame, value_column, "value")
        edge_values = pd.to_numeric(frame[value_name], errors="raise").to_numpy(
            dtype=np.float32
        )

    source_raw = frame[source_name].astype(str)
    target_raw = frame[target_name].astype(str)
    all_nodes = pd.concat([source_raw, target_raw], ignore_index=True)
    node_codes, node_names = pd.factorize(all_nodes, sort=True)
    source = node_codes[: len(frame)].astype(np.int64)
    target = node_codes[len(frame) :].astype(np.int64)
    timestamps = _timestamps(frame[timestamp_name])
    time_index = _time_bins(timestamps, time_slices, binning)
    triples = _coalesced_edges(time_index, source, target)
    num_nodes = len(node_names)

    validation_steps = int(time_slices * val_rate)
    test_steps = int(time_slices * test_rate)
    train_steps = time_slices - validation_steps - test_steps
    if train_steps < 2:
        raise ValueError("the split must leave at least two training snapshots")

    snapshots = _normalized_snapshots(
        triples, time_slices, num_nodes, make_symmetric, edge_life
    )
    train_idx, train_vals = _window_arrays(snapshots, 0, train_steps)
    val_idx, val_vals = _window_arrays(
        snapshots, validation_steps, train_steps
    )
    test_idx, test_vals = _window_arrays(
        snapshots, validation_steps + test_steps, train_steps
    )

    if output_path is None:
        output = Path("data") / spec.directory / spec.filename
    else:
        output = Path(output_path)
        if output.suffix.lower() != ".mat":
            output = output / spec.directory / spec.filename
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = output.with_name(f"{output.stem}_node_mapping.csv")
    pd.DataFrame(
        {"node_index": np.arange(num_nodes), "original_id": node_names}
    ).to_csv(mapping_path, index=False)

    sio.savemat(
        output,
        {
            "tensor_idx": triples,
            "A_idx": triples.T,
            "A_vals": np.ones(triples.shape[0], dtype=np.float32),
            "edge_values": edge_values,
            "train_idx": train_idx,
            "train_vals": train_vals,
            "val_idx": val_idx,
            "val_vals": val_vals,
            "test_idx": test_idx,
            "test_vals": test_vals,
            "num_nodes": np.asarray([[num_nodes]], dtype=np.int64),
            "time_slices": np.asarray([[time_slices]], dtype=np.int64),
        },
        do_compression=True,
    )
    return output


__all__ = ["convert_csv_to_mat"]

