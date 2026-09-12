"""Load an original preprocessed dataset and run one TRSC forward pass."""

import argparse

import torch

from trsc import TRSC, get_all_edge_nodes, load_mat_dataset, split_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bitcoin_alpha")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_mat_dataset(args.dataset, args.data_dir, args.device)
    (
        train_edges,
        train_targets,
        validation_edges,
        _,
        _,
        test_edges,
        _,
        _,
    ) = split_data(data.labels, data.time_slices)
    train_nodes, _, _ = get_all_edge_nodes(
        train_edges, validation_edges, test_edges, data.num_nodes
    )

    model = TRSC(
        time_slices=len(data.train_adjacency) - 1,
        N=data.num_nodes,
        hidden_features=[16],
        num_feature=8,
        out_features=1,
        bandwidth=20,
    ).to(args.device)
    probabilities, embeddings = model(
        data.train_adjacency[:-1], train_nodes
    )
    loss = torch.nn.functional.binary_cross_entropy(
        probabilities, train_targets
    )
    loss.backward()

    print(f"dataset: {args.dataset}")
    print(f"snapshots: {data.time_slices}")
    print(f"nodes: {data.num_nodes}")
    print(f"training queries: {train_targets.numel()}")
    print(f"embedding shape: {tuple(embeddings.shape)}")
    print(f"loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

