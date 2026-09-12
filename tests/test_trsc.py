"""Dependency-light tests for the standalone TRSC package."""

import unittest

import torch

from trsc import CausalStructuralFeatures, ERGTCN, TRSC


def snapshot(node_count: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    all_edges = [(node, node) for node in range(node_count)] + edges
    indices = torch.tensor(all_edges, dtype=torch.long).t()
    values = torch.ones(len(all_edges), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices, values, (node_count, node_count)
    ).coalesce()


class TRSCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adjacency = [
            snapshot(4, [(0, 1)]),
            snapshot(4, [(1, 2)]),
            snapshot(4, [(0, 2), (2, 3)]),
            snapshot(4, [(1, 3)]),
        ]
        self.edge_nodes = (
            torch.tensor([0, 4, 8, 12], dtype=torch.long),
            torch.tensor([2, 6, 11, 15], dtype=torch.long),
        )

    def make_model(self) -> TRSC:
        return TRSC(
            time_slices=4,
            N=4,
            hidden_features=[4],
            num_feature=3,
            out_features=1,
            bandwidth=2,
            tgc_dropout=0.0,
            decoder_dropout=0.0,
        )

    def test_legacy_alias(self) -> None:
        self.assertIs(ERGTCN, TRSC)

    def test_structural_evidence_is_causal(self) -> None:
        original = CausalStructuralFeatures(self.adjacency).query(
            self.edge_nodes, torch.device("cpu"), torch.float32
        )
        changed = list(self.adjacency)
        changed[-1] = snapshot(4, [(0, 3), (3, 2), (2, 0)])
        modified = CausalStructuralFeatures(changed).query(
            self.edge_nodes, torch.device("cpu"), torch.float32
        )
        torch.testing.assert_close(original[:3], modified[:3])

    def test_forward_backward_is_finite(self) -> None:
        torch.manual_seed(2024)
        model = self.make_model()
        output, hidden = model(self.adjacency, self.edge_nodes)
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = torch.nn.functional.binary_cross_entropy(output, target)
        loss.backward()

        self.assertEqual(tuple(output.shape), (4,))
        self.assertEqual(tuple(hidden.shape), (4, 4, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((0.0 <= output) & (output <= 1.0)).all())
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_invalid_snapshot_shape_is_rejected(self) -> None:
        bad = list(self.adjacency)
        bad[-1] = snapshot(3, [(0, 1)])
        with self.assertRaises(ValueError):
            self.make_model()(bad, self.edge_nodes)

    def test_cross_time_query_is_rejected(self) -> None:
        cross_time = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            self.make_model()(self.adjacency, cross_time)


if __name__ == "__main__":
    unittest.main()
