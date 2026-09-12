"""Train and evaluate TRSC on a preprocessed dynamic-graph dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from trsc import TRSC, get_all_edge_nodes, load_mat_dataset, split_data


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_by_time(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    edges: torch.Tensor,
) -> dict[str, float]:
    """Average AP and ROC-AUC over time slices, as in the source project."""

    ap_values: list[float] = []
    auc_values: list[float] = []
    for time_index in edges[0].unique(sorted=True):
        mask = edges[0] == time_index
        truth = targets[mask].detach().cpu().numpy()
        scores = predictions[mask].detach().cpu().numpy()
        if np.unique(truth).size < 2:
            continue
        ap_values.append(float(average_precision_score(truth, scores)))
        auc_values.append(float(roc_auc_score(truth, scores)))
    if not ap_values:
        raise ValueError("no time slice contains both positive and negative labels")
    return {
        "AP": float(np.mean(ap_values)),
        "ROC_AUC": float(np.mean(auc_values)),
    }


def _loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    node_features: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    result = torch.nn.functional.binary_cross_entropy(probabilities, targets)
    if regularization:
        result = result + regularization * torch.norm(node_features, p=2)
    return result


def _load_state(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train_once(args: argparse.Namespace, run: int) -> dict[str, float]:
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    data = load_mat_dataset(
        args.dataset,
        args.data_dir,
        device,
        negative_ratio=args.negative_ratio,
        seed=args.data_seed,
    )
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
    train_nodes, validation_nodes, test_nodes = get_all_edge_nodes(
        train_edges, validation_edges, test_edges, data.num_nodes
    )
    if validation_tail <= 0 or test_tail <= 0:
        raise ValueError("validation and test splits must each contain labels")

    model_seed = args.seed + run
    set_seed(model_seed)
    model = TRSC(
        time_slices=len(data.train_adjacency) - 1,
        N=data.num_nodes,
        hidden_features=args.hidden_features,
        num_feature=args.num_feature,
        out_features=1,
        bandwidth=args.bandwidth,
        tgc_dropout=args.tgc_dropout,
        activity_window=args.activity_window,
        min_edge_weight=args.min_edge_weight,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
        mixing_choice=args.mixing_choice,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    run_dir = Path(args.output_dir) / args.dataset / f"seed_{model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "AP": run_dir / "best_ap.pt",
        "ROC_AUC": run_dir / "best_roc_auc.pt",
    }
    best = {"AP": float("-inf"), "ROC_AUC": float("-inf")}
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_output, _ = model(data.train_adjacency[:-1], train_nodes)
        train_loss = _loss(train_output, train_targets, model.n_f, args.lam)
        train_loss.backward()
        optimizer.step()
        train_metrics = metrics_by_time(train_output, train_targets, train_edges)

        model.eval()
        with torch.no_grad():
            validation_output, _ = model(
                data.validation_adjacency[:-1], validation_nodes
            )
            validation_output = validation_output[-validation_tail:]
            validation_target_tail = validation_targets[-validation_tail:]
            validation_edge_tail = validation_edges[:, -validation_tail:]
            validation_loss = _loss(
                validation_output,
                validation_target_tail,
                model.n_f,
                args.lam,
            )
            validation_metrics = metrics_by_time(
                validation_output,
                validation_target_tail,
                validation_edge_tail,
            )

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach()),
            "train_ap": train_metrics["AP"],
            "train_roc_auc": train_metrics["ROC_AUC"],
            "validation_loss": float(validation_loss.detach()),
            "validation_ap": validation_metrics["AP"],
            "validation_roc_auc": validation_metrics["ROC_AUC"],
        }
        history.append(record)
        print(json.dumps(record))

        improved = False
        for metric_name, value in validation_metrics.items():
            if value > best[metric_name]:
                best[metric_name] = value
                torch.save(model.state_dict(), checkpoint_paths[metric_name])
                improved = True
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if epochs_without_improvement >= args.patience:
            break

    results = {
        "seed": model_seed,
        "epochs": len(history),
        "best_validation_ap": best["AP"],
        "best_validation_roc_auc": best["ROC_AUC"],
    }
    for metric_name, checkpoint_path in checkpoint_paths.items():
        model.load_state_dict(_load_state(checkpoint_path, device))
        model.eval()
        with torch.no_grad():
            test_output, _ = model(data.test_adjacency[:-1], test_nodes)
            test_metrics = metrics_by_time(
                test_output[-test_tail:],
                test_targets[-test_tail:],
                test_edges[:, -test_tail:],
            )
        results[f"test_{metric_name.lower()}"] = test_metrics[metric_name]

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate TRSC")
    parser.add_argument("--dataset", default="bitcoin_alpha")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lam", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--data-seed", type=int, default=2024)
    parser.add_argument("--cuda", type=parse_bool, default=True)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--num-feature", type=int, default=8)
    parser.add_argument("--hidden-features", type=int, nargs="+", default=[16])
    parser.add_argument("--bandwidth", type=int, default=20)
    parser.add_argument("--mixing-choice", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tgc-dropout", type=float, default=0.75)
    parser.add_argument("--activity-window", type=int, default=3)
    parser.add_argument("--min-edge-weight", type=float, default=0.25)
    parser.add_argument("--decoder-hidden-dim", type=int, default=32)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_results = [train_once(args, run) for run in range(args.num_runs)]
    summary = {
        "dataset": args.dataset,
        "runs": all_results,
        "mean_test_ap": float(np.mean([item["test_ap"] for item in all_results])),
        "std_test_ap": float(np.std([item["test_ap"] for item in all_results])),
        "mean_test_roc_auc": float(
            np.mean([item["test_roc_auc"] for item in all_results])
        ),
        "std_test_roc_auc": float(
            np.std([item["test_roc_auc"] for item in all_results])
        ),
    }
    output = Path(args.output_dir) / args.dataset / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

