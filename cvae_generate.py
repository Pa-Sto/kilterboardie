import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch

from cvae_data import KilterRouteDataset
from cvae_model import KilterCVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kilterboard routes with a trained CVAE.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="generated_route.npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _pick_topk_from_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    flat = scores.view(-1)
    if k <= 0:
        return torch.zeros_like(flat)
    k = min(k, flat.numel())
    vals, idx = torch.topk(flat, k)
    out = torch.zeros_like(flat)
    out[idx] = 1.0
    return out.view_as(scores)


def _pick_best_pair_within_distance(scores: torch.Tensor, mask: torch.Tensor, max_dist: float) -> torch.Tensor:
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0 or coords.shape[0] < 2:
        return torch.zeros_like(scores)

    coords_np = coords.cpu().numpy()
    scores_np = scores[mask].cpu().numpy()
    max_dist2 = float(max_dist) ** 2

    best_sum = None
    best_pair = None
    n = coords_np.shape[0]
    for i in range(n):
        yi, xi = coords_np[i]
        si = scores_np[i]
        for j in range(i + 1, n):
            yj, xj = coords_np[j]
            dy = float(yi - yj)
            dx = float(xi - xj)
            if dy * dy + dx * dx > max_dist2:
                continue
            s = si + scores_np[j]
            if best_sum is None or s > best_sum:
                best_sum = s
                best_pair = (i, j)

    if best_pair is None:
        return torch.zeros_like(scores)

    out = torch.zeros_like(scores)
    i, j = best_pair
    yi, xi = coords_np[i]
    yj, xj = coords_np[j]
    out[int(yi), int(xi)] = 1.0
    out[int(yj), int(xj)] = 1.0
    return out


def _enforce_start_finish_counts(
    probs: torch.Tensor,
    hold_mask: torch.Tensor,
    threshold: float,
    start_min: int,
    start_max: int,
    finish_min: int,
    finish_max: int,
    start_max_dist: float = None,
    finish_max_dist: float = None,
) -> torch.Tensor:
    """
    Enforce min/max counts for start (ch=0) and finish (ch=1).
    Uses threshold to decide whether to pick 1 or 2 when max > min.
    """
    out = probs.clone()
    for i in range(out.shape[0]):
        mask = hold_mask[i, 0] > 0
        for ch, cmin, cmax in [(0, start_min, start_max), (1, finish_min, finish_max)]:
            scores = out[i, ch] * mask
            count = int((scores >= threshold).sum().item())
            k = max(cmin, min(cmax, count))
            if k == 0:
                k = cmin
            if k == 2:
                max_dist = start_max_dist if ch == 0 else finish_max_dist
                if max_dist is not None:
                    pair_mask = mask & (scores >= threshold)
                    pair = _pick_best_pair_within_distance(scores, pair_mask, max_dist)
                    if pair.sum() > 0:
                        out[i, ch] = pair
                        continue
            out[i, ch] = _pick_topk_from_scores(scores, k)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = ckpt.get("config", {})

    dataset = KilterRouteDataset(args.data_dir)
    grade_min = config.get("grade_min", dataset.grade_min)
    grade_max = config.get("grade_max", dataset.grade_max)

    if args.grade < grade_min or args.grade > grade_max:
        raise ValueError(f"grade must be in [{grade_min}, {grade_max}]")

    model = KilterCVAE(
        num_grades=config.get("num_grades", dataset.num_grades),
        emb_dim=config.get("emb_dim", 16),
        latent_dim=config.get("latent_dim", 64),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    static = dataset.get_static().unsqueeze(0).to(args.device)
    static = static.expand(args.n, -1, -1, -1)

    grade_idx = torch.tensor([args.grade - grade_min], dtype=torch.int64, device=args.device)
    grade_idx = grade_idx.expand(args.n)

    with torch.no_grad():
        logits = model.sample(grade_idx, static, n=args.n)
        probs = torch.sigmoid(logits)
        # Enforce start/finish count constraints
        probs = _enforce_start_finish_counts(
            probs,
            hold_mask=static[:, 0:1],
            threshold=args.threshold,
            start_min=args.start_min,
            start_max=args.start_max,
            finish_min=args.finish_min,
            finish_max=args.finish_max,
            start_max_dist=args.start_max_dist,
            finish_max_dist=args.finish_max_dist,
        )

        route = (probs >= args.threshold).float()

    # Build full 6-channel matrix: route(4) + static(2)
    route_np = route.cpu().numpy()  # N x 4 x H x W
    static_np = static.cpu().numpy()  # N x 2 x H x W
    full = np.concatenate([route_np, static_np], axis=1)  # N x 6 x H x W
    full = np.transpose(full, (0, 2, 3, 1))  # N x H x W x 6

    if args.n == 1:
        np.save(args.out, full[0])
    else:
        base, ext = os.path.splitext(args.out)
        for i in range(args.n):
            np.save(f"{base}_{i:02d}{ext}", full[i])

    meta = {
        "grade_v": args.grade,
        "threshold": args.threshold,
        "n": args.n,
        "start_min": args.start_min,
        "start_max": args.start_max,
        "finish_min": args.finish_min,
        "finish_max": args.finish_max,
        "start_max_dist": args.start_max_dist,
        "finish_max_dist": args.finish_max_dist,
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
