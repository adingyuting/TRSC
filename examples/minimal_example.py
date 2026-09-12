"""Run one forward/backward optimization step on a tiny dynamic graph."""

import torch

from trsc import TRSC


def snapshot(node_count: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Create a sparse COO adjacency matrix with explicit self-loops."""

    all_edges = [(node, node) for node in range(node_count)] + edges
    indices = torch.tensor(all_edges, dtype=torch.long).t()
    values = torch.ones(len(all_edges), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices, values, (node_count, node_count)
    ).coalesce()


def main() -> None:
    torch.manual_seed(2024)
    node_count = 4
    adjacency = [
        snapshot(node_count, [(0, 1)]),
        snapshot(node_count, [(1, 2)]),
        snapshot(node_count, [(0, 2), (2, 3)]),
        snapshot(node_count, [(1, 3)]),
    ]

    # Query (0, 2) at t=0, (0, 2) at t=1, and so on. The flattened
    # representation for node v at time t is t * node_count + v.
    source = torch.tensor([0, 4, 8, 12], dtype=torch.long)
    target = torch.tensor([2, 6, 11, 15], dtype=torch.long)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    model = TRSC(
        time_slices=len(adjacency),
        N=node_count,
        hidden_features=[8],
        num_feature=4,
        out_features=1,
        bandwidth=2,
        tgc_dropout=0.0,
        decoder_dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    probabilities, embeddings = model(adjacency, (source, target))
    loss = torch.nn.functional.binary_cross_entropy(probabilities, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"probabilities: {probabilities.detach().tolist()}")
    print(f"embedding shape: {tuple(embeddings.shape)}")
    print(f"loss: {loss.item():.6f}")
    print(
        "diagnostics:",
        {key: float(value) for key, value in model.last_diagnostics.items()},
    )


if __name__ == "__main__":
    main()

