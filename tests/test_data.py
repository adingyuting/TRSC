"""End-to-end checks for the original MATLAB data protocol."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import torch

from trsc import (
    TRSC,
    convert_csv_to_mat,
    get_all_edge_nodes,
    load_mat_dataset,
    split_data,
)
from train import train_once


def tensor_arrays(edges_by_time: list[list[tuple[int, int]]]):
    triples = []
    for time_index, edges in enumerate(edges_by_time):
        triples.extend((time_index, source, target) for source, target in edges)
    indices = np.asarray(triples, dtype=np.int64).T
    values = np.ones(indices.shape[1], dtype=np.float32)
    return indices, values


class DataPipelineTests(unittest.TestCase):
    def test_csv_to_mat_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "alpha.csv"
            rows = ["From,To,Value,TimeStamp"]
            rows.extend(
                f"{time % 5},{(time + 1) % 5},1,{time}"
                for time in range(60)
            )
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            mat_path = convert_csv_to_mat(
                csv_path,
                "bitcoin_alpha",
                root / "data",
                binning="discrete",
            )
            self.assertTrue(mat_path.is_file())
            self.assertTrue(
                mat_path.with_name("bitcoin_alpha_node_mapping.csv").is_file()
            )

            args = SimpleNamespace(
                dataset="bitcoin_alpha",
                data_dir=root / "data",
                output_dir=root / "outputs",
                negative_ratio=1.0,
                data_seed=2024,
                seed=2024,
                cuda=False,
                hidden_features=[4],
                num_feature=3,
                bandwidth=2,
                tgc_dropout=0.0,
                activity_window=3,
                min_edge_weight=0.25,
                decoder_hidden_dim=8,
                decoder_dropout=0.0,
                mixing_choice=2,
                lr=0.005,
                weight_decay=5e-4,
                lam=1e-5,
                epochs=1,
                patience=1,
            )
            metrics = train_once(args, run=0)

            self.assertIn("test_ap", metrics)
            self.assertIn("test_roc_auc", metrics)
            self.assertTrue((root / "outputs" / "bitcoin_alpha").is_dir())

    def test_mat_to_model_forward_backward(self) -> None:
        # Ten snapshots imply train=7, validation=1, test=2. The three
        # adjacency tensors follow the original rolling-window offsets.
        snapshots = [
            [(node, node) for node in range(5)] + [(t % 5, (t + 1) % 5)]
            for t in range(10)
        ]
        full_idx, full_vals = tensor_arrays(snapshots)
        train_idx, train_vals = tensor_arrays(snapshots[0:7])
        val_idx, val_vals = tensor_arrays(snapshots[1:8])
        test_idx, test_vals = tensor_arrays(snapshots[3:10])

        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "bitcoin_alpha"
            dataset_dir.mkdir()
            sio.savemat(
                dataset_dir / "bitcoin_alpha.mat",
                {
                    "tensor_idx": full_idx.T,
                    "A_idx": full_idx,
                    "A_vals": full_vals,
                    "train_idx": train_idx,
                    "train_vals": train_vals,
                    "val_idx": val_idx,
                    "val_vals": val_vals,
                    "test_idx": test_idx,
                    "test_vals": test_vals,
                },
            )
            data = load_mat_dataset(
                "bitcoin_alpha",
                temporary,
                negative_ratio=0.25,
                check_time_slices=False,
            )

        self.assertEqual(data.time_slices, 10)
        self.assertEqual(data.num_nodes, 5)
        self.assertEqual(len(data.train_adjacency), 7)
        self.assertEqual(len(data.validation_adjacency), 7)
        self.assertEqual(len(data.test_adjacency), 7)

        (
            train_edges,
            train_targets,
            validation_edges,
            validation_targets,
            validation_tail,
            test_edges,
            test_targets,
            test_tail,
        ) = split_data(data.labels, data.time_slices)
        edge_nodes = get_all_edge_nodes(
            train_edges, validation_edges, test_edges, data.num_nodes
        )
        model = TRSC(
            time_slices=6,
            N=5,
            hidden_features=[4],
            num_feature=3,
            out_features=1,
            bandwidth=2,
            tgc_dropout=0.0,
            decoder_dropout=0.0,
        )
        total_loss = torch.zeros(())
        for adjacency, nodes, targets in (
            (data.train_adjacency, edge_nodes[0], train_targets),
            (data.validation_adjacency, edge_nodes[1], validation_targets),
            (data.test_adjacency, edge_nodes[2], test_targets),
        ):
            output, _ = model(adjacency[:-1], nodes)
            self.assertEqual(output.shape, targets.shape)
            total_loss = total_loss + torch.nn.functional.binary_cross_entropy(
                output, targets
            )
        total_loss.backward()

        self.assertGreater(validation_tail, 0)
        self.assertGreater(test_tail, 0)
        self.assertTrue(torch.isfinite(total_loss))

    def test_missing_file_message(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                load_mat_dataset("wiki_gl", temporary)


if __name__ == "__main__":
    unittest.main()
