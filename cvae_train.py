import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from cvae_data import KilterRouteDataset, compute_pos_weight
from cvae_model import KilterCVAE, cvae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional VAE on Kilterboard routes.")
    parser.add_argument("--data-dir", type=str, default="ImageData/50degree/export")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--count-weight", type=float, default=0.01)
    parser.add_argument("--count-min", type=int, default=1)
    parser.add_argument("--count-max", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--path-weight", type=float, default=0.01)
    parser.add_argument("--path-reach", type=int, default=10)
    parser.add_argument("--path-steps", type=int, default=4)
    parser.add_argument("--upward-weight", type=float, default=0.005)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/cvae")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: KilterRouteDataset, val_split: float, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = KilterRouteDataset(args.data_dir)
    train_ds, val_ds = split_dataset(dataset, args.val_split, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_paths = [dataset.npy_paths()[i] for i in train_ds.indices]
    pos_weight = compute_pos_weight(train_paths).to(args.device)

    model = KilterCVAE(
        num_grades=dataset.num_grades,
        emb_dim=args.emb_dim,
        latent_dim=args.latent_dim,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    config = vars(args)
    config.update({"num_grades": dataset.num_grades, "grade_min": dataset.grade_min, "grade_max": dataset.grade_max})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_count_loss = 0.0
        total_path_loss = 0.0
        total_upward_loss = 0.0
        count = 0

        for route, static, grade in train_loader:
            route = route.to(args.device)
            static = static.to(args.device)
            grade = grade.to(args.device)
            hold_mask = static[:, 0:1]

            logits, mu, logvar = model(route, static, grade)
            loss, recon, kl, count_loss, path_loss, upward_loss = cvae_loss(
                logits,
                route,
                hold_mask,
                mu,
                logvar,
                beta=args.beta,
                pos_weight=pos_weight,
                count_weight=args.count_weight,
                count_min=args.count_min,
                count_max=args.count_max,
                focal_gamma=args.focal_gamma,
                path_weight=args.path_weight,
                path_reach=args.path_reach,
                path_steps=args.path_steps,
                upward_weight=args.upward_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = route.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kl += kl.item() * batch_size
            total_count_loss += count_loss.item() * batch_size
            total_path_loss += path_loss.item() * batch_size
            total_upward_loss += upward_loss.item() * batch_size
            count += batch_size

        train_loss = total_loss / count
        train_recon = total_recon / count
        train_kl = total_kl / count
        train_count = total_count_loss / count
        train_path = total_path_loss / count
        train_upward = total_upward_loss / count

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_count_loss_total = 0.0
        val_path_loss_total = 0.0
        val_upward_loss_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for route, static, grade in val_loader:
                route = route.to(args.device)
                static = static.to(args.device)
                grade = grade.to(args.device)
                hold_mask = static[:, 0:1]

                logits, mu, logvar = model(route, static, grade)
                loss, recon, kl, count_loss, path_loss, upward_loss = cvae_loss(
                    logits,
                    route,
                    hold_mask,
                    mu,
                    logvar,
                    beta=args.beta,
                    pos_weight=pos_weight,
                    count_weight=args.count_weight,
                    count_min=args.count_min,
                    count_max=args.count_max,
                    focal_gamma=args.focal_gamma,
                    path_weight=args.path_weight,
                    path_reach=args.path_reach,
                    path_steps=args.path_steps,
                    upward_weight=args.upward_weight,
                )

                batch_size = route.size(0)
                val_loss += loss.item() * batch_size
                val_recon += recon.item() * batch_size
                val_kl += kl.item() * batch_size
                val_count_loss_total += count_loss.item() * batch_size
                val_path_loss_total += path_loss.item() * batch_size
                val_upward_loss_total += upward_loss.item() * batch_size
                val_samples += batch_size

        val_loss /= max(val_samples, 1)
        val_recon /= max(val_samples, 1)
        val_kl /= max(val_samples, 1)
        val_count_loss = val_count_loss_total / max(val_samples, 1)
        val_path_loss = val_path_loss_total / max(val_samples, 1)
        val_upward_loss = val_upward_loss_total / max(val_samples, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}, "
            f"count {train_count:.4f}, path {train_path:.4f}, up {train_upward:.4f}) | "
            f"val loss {val_loss:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f}, "
            f"count {val_count_loss:.4f}, path {val_path_loss:.4f}, up {val_upward_loss:.4f})"
        , flush=True)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "train_count": train_count,
                "train_path": train_path,
                "train_upward": train_upward,
                "val_loss": val_loss,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "val_count": val_count_loss,
                "val_path": val_path_loss,
                "val_upward": val_upward_loss,
            }) + "\n")

        ckpt = {
            "model_state": model.state_dict(),
            "pos_weight": pos_weight.detach().cpu(),
            "config": config,
        }
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))


if __name__ == "__main__":
    main()
