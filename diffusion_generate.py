import argparse
import json
import os

import numpy as np
import torch

from cvae_data import KilterRouteDataset
from cvae_generate import _grade_count_histograms, decode_route_with_priors
from diffusion_model import GaussianDiffusion, KilterDiffusionUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kilterboard routes with a trained diffusion model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="ImageData/50degree/export")
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--start-min", type=int, default=1)
    parser.add_argument("--start-max", type=int, default=2)
    parser.add_argument("--finish-min", type=int, default=1)
    parser.add_argument("--finish-max", type=int, default=2)
    parser.add_argument("--start-max-dist", type=float, default=8.0)
    parser.add_argument("--finish-max-dist", type=float, default=8.0)
    parser.add_argument("--foot-count-mode", type=str, choices=["sample", "median", "trimmed_sample"], default="median")
    parser.add_argument("--foot-count-quantile", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="generated_route.npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = ckpt.get("config", {})

    dataset = KilterRouteDataset(
        args.data_dir,
        grade_min=config.get("grade_min", 3),
        grade_max=config.get("grade_max", 13),
    )
    grade_min = config.get("grade_min", dataset.grade_min)
    grade_max = config.get("grade_max", dataset.grade_max)

    if args.grade < grade_min or args.grade > grade_max:
        raise ValueError(f"grade must be in [{grade_min}, {grade_max}]")

    model = KilterDiffusionUNet(
        num_grades=config.get("num_grades", dataset.num_grades),
        static_channels=dataset.static_channels,
        route_channels=config.get("route_channels", 4),
        base_channels=config.get("base_channels", 64),
        grade_emb_dim=config.get("grade_emb_dim", 32),
        time_emb_dim=config.get("time_emb_dim", 128),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=config.get("timesteps", 200),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
    )

    static = dataset.get_static().unsqueeze(0).to(args.device)
    static = static.expand(args.n, -1, -1, -1)
    hold_mask = static[:, 0:1]

    grade_idx = torch.tensor([args.grade - grade_min], dtype=torch.int64, device=args.device)
    grade_idx = grade_idx.expand(args.n)
    grade_values = torch.tensor([args.grade] * args.n, dtype=torch.int64, device=args.device)
    count_histograms = _grade_count_histograms(args.data_dir, int(grade_min), int(grade_max))

    with torch.no_grad():
        x0_pred = diffusion.sample(
            model=model,
            shape=(args.n, 4, static.shape[2], static.shape[3]),
            static=static,
            grade=grade_idx,
            device=torch.device(args.device),
            hold_mask=hold_mask,
        )
        probs = ((x0_pred + 1.0) * 0.5).clamp(0.0, 1.0)
        probs = probs * hold_mask
        route = decode_route_with_priors(
            probs,
            hold_mask=hold_mask,
            grade_values=grade_values,
            count_histograms=count_histograms,
            seed=args.seed,
            threshold=args.threshold,
            start_min=args.start_min,
            start_max=args.start_max,
            finish_min=args.finish_min,
            finish_max=args.finish_max,
            start_max_dist=args.start_max_dist,
            finish_max_dist=args.finish_max_dist,
            foot_count_mode=args.foot_count_mode,
            foot_count_quantile=args.foot_count_quantile,
        )
        route = route * hold_mask

    route_np = route.cpu().numpy()
    static_np = static.cpu().numpy()
    full = np.concatenate([route_np, static_np], axis=1)
    full = np.transpose(full, (0, 2, 3, 1))

    if args.n == 1:
        np.save(args.out, full[0])
    else:
        base, ext = os.path.splitext(args.out)
        for i in range(args.n):
            np.save(f"{base}_{i:02d}{ext}", full[i])

    meta = {
        "model": "masked_ddpm",
        "grade_v": args.grade,
        "n": args.n,
        "threshold": args.threshold,
        "timesteps": int(config.get("timesteps", 200)),
        "start_min": args.start_min,
        "start_max": args.start_max,
        "finish_min": args.finish_min,
        "finish_max": args.finish_max,
        "start_max_dist": args.start_max_dist,
        "finish_max_dist": args.finish_max_dist,
        "foot_count_mode": args.foot_count_mode,
        "foot_count_quantile": args.foot_count_quantile,
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
