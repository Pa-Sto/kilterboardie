from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from graph_transformer_data import KilterGraphSequenceDataset, collate_graph_sequences
from graph_transformer_model import HierarchicalGraphTransformer, hierarchical_graph_transformer_loss


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hierarchical graph-conditioned route Transformer.")
    parser.add_argument("--data-dir", default="ImageData/50Degree/ExportClean")
    parser.add_argument("--holds-path", default="ImageData/References/holds.json")
    parser.add_argument("--grade-min", type=int, default=3)
    parser.add_argument("--grade-max", type=int, default=13)
    parser.add_argument("--band-rows", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=48)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--edge-radius", type=float, default=4.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--pointer-weight", type=float, default=1.0)
    parser.add_argument("--role-weight", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for smoke tests; 0 uses all data.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out-dir", default="runs/graph_transformer")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: HierarchicalGraphTransformer,
    loader: DataLoader,
    graph,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            action_logits, pointer_logits, hidden, node_context = model(
                graph=graph,
                input_actions=batch["input_actions"],
                input_nodes=batch["input_nodes"],
                input_roles=batch["input_roles"],
                grade=batch["grade"],
            )
            loss, metrics = hierarchical_graph_transformer_loss(
                model=model,
                action_logits=action_logits,
                pointer_logits=pointer_logits,
                hidden=hidden,
                node_context=node_context,
                batch=batch,
                action_weight=args.action_weight,
                pointer_weight=args.pointer_weight,
                role_weight=args.role_weight,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = int(batch["grade"].shape[0])
            sample_count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def split_dataset(
    dataset: KilterGraphSequenceDataset,
    val_split: float,
    seed: int,
    max_samples: int,
) -> Tuple[Subset, Subset]:
    if max_samples > 0 and max_samples < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        working_dataset = Subset(dataset, chosen)
    else:
        working_dataset = dataset

    val_size = max(int(len(working_dataset) * val_split), 1)
    train_size = len(working_dataset) - val_size
    if train_size < 1:
        raise RuntimeError("Dataset is too small for the requested validation split.")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(working_dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = KilterGraphSequenceDataset(
        data_dir=args.data_dir,
        holds_path=args.holds_path,
        grade_min=args.grade_min,
        grade_max=args.grade_max,
        band_rows=args.band_rows,
        max_sequence_length=args.max_sequence_length,
        k_neighbors=args.k_neighbors,
        edge_radius=args.edge_radius,
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed, args.max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_graph_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graph_sequences,
    )

    graph = dataset.board_graph.to(device)
    model = HierarchicalGraphTransformer(
        num_grades=dataset.num_grades,
        node_feature_dim=graph.node_feature_dim,
        edge_feature_dim=graph.edge_feature_dim,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_sequence_length=args.max_sequence_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "num_grades": dataset.num_grades,
            "node_feature_dim": graph.node_feature_dim,
            "edge_feature_dim": graph.edge_feature_dim,
            "num_nodes": graph.num_nodes,
            "coordinate_spacing": graph.coordinate_spacing,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best_val = float("inf")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training on {device} | {len(train_dataset)} train / {len(val_dataset)} val | "
        f"{graph.num_nodes} nodes / {graph.edge_index.shape[1]} directed edges | "
        f"{parameter_count:,} parameters",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, graph, device, optimizer, args)
        val_metrics = run_epoch(model, val_loader, graph, device, None, args)
        record = {"epoch": epoch}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({f"val_{key}": value for key, value in val_metrics.items()})
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Epoch {epoch:03d} | "
            f"train {train_metrics['loss']:.4f} "
            f"(action {train_metrics['action_accuracy']:.3f}, node {train_metrics['node_accuracy']:.3f}, "
            f"role {train_metrics['role_accuracy']:.3f}) | "
            f"val {val_metrics['loss']:.4f} "
            f"(action {val_metrics['action_accuracy']:.3f}, node {val_metrics['node_accuracy']:.3f}, "
            f"role {val_metrics['role_accuracy']:.3f})",
            flush=True,
        )

        checkpoint = {"model_state": model.state_dict(), "config": config, "epoch": epoch}
        torch.save(checkpoint, os.path.join(run_dir, "last.pt"))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, os.path.join(run_dir, "best.pt"))

    print(f"Training complete. Artifacts saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
