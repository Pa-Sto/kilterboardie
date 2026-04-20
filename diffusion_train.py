import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from cvae_data import KilterRouteDataset, compute_pos_weight
from diffusion_model import (
    GaussianDiffusion,
    KilterDiffusionUNet,
    masked_weighted_bce_prob,
    masked_weighted_mse,
    structure_losses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model on Kilterboard routes.")
    parser.add_argument("--data-dir", type=str, default="ImageData/50degree/export")
    parser.add_argument("--grade-min", type=int, default=3)
    parser.add_argument("--grade-max", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--grade-emb-dim", type=int, default=32)
    parser.add_argument("--time-emb-dim", type=int, default=128)
    parser.add_argument("--eps-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=0.25)
    parser.add_argument("--count-weight", type=float, default=0.01)
    parser.add_argument("--count-min", type=int, default=1)
    parser.add_argument("--count-max", type=int, default=2)
    parser.add_argument("--path-weight", type=float, default=0.01)
    parser.add_argument("--path-reach", type=int, default=10)
    parser.add_argument("--path-steps", type=int, default=4)
    parser.add_argument("--upward-weight", type=float, default=0.005)
    parser.add_argument("--hand-density-weight", type=float, default=0.03)
    parser.add_argument("--foot-density-weight", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/diffusion")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: KilterRouteDataset, val_split: float, seed: int
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = int(len(dataset) * val_split)
    if val_size <= 0:
        val_size = 1
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Dataset too small for the requested validation split.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _forward_losses(
    model: KilterDiffusionUNet,
    diffusion: GaussianDiffusion,
    route: torch.Tensor,
    static: torch.Tensor,
    grade: torch.Tensor,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hold_mask = static[:, 0:1]
    x0 = route * 2.0 - 1.0
    noise = torch.randn_like(x0)
    t = torch.randint(0, diffusion.timesteps, (route.shape[0],), device=route.device, dtype=torch.long)
    x_t = diffusion.q_sample(x0, t, noise=noise)
    eps_pred = model(x_t, static, grade, t)

    eps_loss = masked_weighted_mse(
        pred=eps_pred,
        target=noise,
        hold_mask=hold_mask,
        pos_weight=pos_weight,
        targets_binary=route,
    )
    x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred).clamp(-1.0, 1.0)
    probs = ((x0_pred + 1.0) * 0.5).clamp(0.0, 1.0)
    probs = probs * hold_mask

    recon_loss = masked_weighted_bce_prob(
        probs=probs,
        targets=route,
        hold_mask=hold_mask,
        pos_weight=pos_weight,
    )
    count_loss, path_loss, upward_loss, hand_density_loss, foot_density_loss = structure_losses(
        probs=probs,
        targets=route,
        hold_mask=hold_mask,
        count_min=args.count_min,
        count_max=args.count_max,
        path_reach=args.path_reach,
        path_steps=args.path_steps,
    )

    loss = (
        args.eps_weight * eps_loss
        + args.recon_weight * recon_loss
        + args.count_weight * count_loss
        + args.path_weight * path_loss
        + args.upward_weight * upward_loss
        + args.hand_density_weight * hand_density_loss
        + args.foot_density_weight * foot_density_loss
    )
    return loss, eps_loss, recon_loss, count_loss, path_loss, upward_loss, hand_density_loss, foot_density_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = KilterRouteDataset(args.data_dir, grade_min=args.grade_min, grade_max=args.grade_max)
    train_ds, val_ds = split_dataset(dataset, args.val_split, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_paths = [dataset.npy_paths()[i] for i in train_ds.indices]
    pos_weight = compute_pos_weight(train_paths).to(args.device)

    model = KilterDiffusionUNet(
        num_grades=dataset.num_grades,
        static_channels=dataset.static_channels,
        route_channels=4,
        base_channels=args.base_channels,
        grade_emb_dim=args.grade_emb_dim,
        time_emb_dim=args.time_emb_dim,
    ).to(args.device)
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    config = vars(args)
    config.update(
        {
            "num_grades": dataset.num_grades,
            "grade_min": dataset.grade_min,
            "grade_max": dataset.grade_max,
            "static_channels": dataset.static_channels,
            "route_channels": 4,
        }
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_eps = 0.0
        total_recon = 0.0
        total_count = 0.0
        total_path = 0.0
        total_up = 0.0
        total_hand_density = 0.0
        total_foot_density = 0.0
        total_samples = 0

        for route, static, grade in train_loader:
            route = route.to(args.device)
            static = static.to(args.device)
            grade = grade.to(args.device)

            loss, eps_loss, recon_loss, count_loss, path_loss, upward_loss, hand_density_loss, foot_density_loss = _forward_losses(
                model=model,
                diffusion=diffusion,
                route=route,
                static=static,
                grade=grade,
                pos_weight=pos_weight,
                args=args,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = route.shape[0]
            total_loss += loss.item() * bs
            total_eps += eps_loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_count += count_loss.item() * bs
            total_path += path_loss.item() * bs
            total_up += upward_loss.item() * bs
            total_hand_density += hand_density_loss.item() * bs
            total_foot_density += foot_density_loss.item() * bs
            total_samples += bs

        train_loss = total_loss / max(total_samples, 1)
        train_eps = total_eps / max(total_samples, 1)
        train_recon = total_recon / max(total_samples, 1)
        train_count = total_count / max(total_samples, 1)
        train_path = total_path / max(total_samples, 1)
        train_up = total_up / max(total_samples, 1)
        train_hand_density = total_hand_density / max(total_samples, 1)
        train_foot_density = total_foot_density / max(total_samples, 1)

        model.eval()
        val_loss = 0.0
        val_eps = 0.0
        val_recon = 0.0
        val_count = 0.0
        val_path = 0.0
        val_up = 0.0
        val_hand_density = 0.0
        val_foot_density = 0.0
        val_samples = 0

        with torch.no_grad():
            for route, static, grade in val_loader:
                route = route.to(args.device)
                static = static.to(args.device)
                grade = grade.to(args.device)

                loss, eps_loss, recon_loss, count_loss, path_loss, upward_loss, hand_density_loss, foot_density_loss = _forward_losses(
                    model=model,
                    diffusion=diffusion,
                    route=route,
                    static=static,
                    grade=grade,
                    pos_weight=pos_weight,
                    args=args,
                )

                bs = route.shape[0]
                val_loss += loss.item() * bs
                val_eps += eps_loss.item() * bs
                val_recon += recon_loss.item() * bs
                val_count += count_loss.item() * bs
                val_path += path_loss.item() * bs
                val_up += upward_loss.item() * bs
                val_hand_density += hand_density_loss.item() * bs
                val_foot_density += foot_density_loss.item() * bs
                val_samples += bs

        val_loss /= max(val_samples, 1)
        val_eps /= max(val_samples, 1)
        val_recon /= max(val_samples, 1)
        val_count /= max(val_samples, 1)
        val_path /= max(val_samples, 1)
        val_up /= max(val_samples, 1)
        val_hand_density /= max(val_samples, 1)
        val_foot_density /= max(val_samples, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} (eps {train_eps:.4f}, recon {train_recon:.4f}, "
            f"count {train_count:.4f}, path {train_path:.4f}, up {train_up:.4f}, "
            f"hand_den {train_hand_density:.4f}, foot_den {train_foot_density:.4f}) | "
            f"val loss {val_loss:.4f} (eps {val_eps:.4f}, recon {val_recon:.4f}, "
            f"count {val_count:.4f}, path {val_path:.4f}, up {val_up:.4f}, "
            f"hand_den {val_hand_density:.4f}, foot_den {val_foot_density:.4f})",
            flush=True,
        )

        with open(metrics_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_eps": train_eps,
                        "train_recon": train_recon,
                        "train_count": train_count,
                        "train_path": train_path,
                        "train_upward": train_up,
                        "train_hand_density": train_hand_density,
                        "train_foot_density": train_foot_density,
                        "val_loss": val_loss,
                        "val_eps": val_eps,
                        "val_recon": val_recon,
                        "val_count": val_count,
                        "val_path": val_path,
                        "val_upward": val_up,
                        "val_hand_density": val_hand_density,
                        "val_foot_density": val_foot_density,
                    }
                )
                + "\n"
            )

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
