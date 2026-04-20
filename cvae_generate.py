import argparse
import json
import os
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Tuple

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
    parser.add_argument("--foot-count-mode", type=str, choices=["sample", "median", "trimmed_sample"], default="median")
    parser.add_argument("--foot-count-quantile", type=float, default=0.5)
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


@lru_cache(maxsize=8)
def _grade_count_histograms(data_dir: str, grade_min: int, grade_max: int) -> Dict[int, Dict[str, Dict[int, int]]]:
    dataset = KilterRouteDataset(data_dir, grade_min=grade_min, grade_max=grade_max)
    hist = defaultdict(lambda: {name: Counter() for name in ("start", "finish", "hand", "foot")})

    for sample in dataset.samples:
        arr = np.load(sample.npy_path)
        grade_v = int(sample.grade_v)
        hist[grade_v]["start"][int((arr[..., 0] > 0).sum())] += 1
        hist[grade_v]["finish"][int((arr[..., 1] > 0).sum())] += 1
        hist[grade_v]["hand"][int((arr[..., 2] > 0).sum())] += 1
        hist[grade_v]["foot"][int((arr[..., 3] > 0).sum())] += 1

    return {grade: {name: dict(counter) for name, counter in counts.items()} for grade, counts in hist.items()}


def _sample_count_from_hist(hist: Dict[int, int], rng: np.random.Generator, fallback: int) -> int:
    if not hist:
        return int(fallback)
    counts = np.array(sorted(hist.keys()), dtype=np.int64)
    weights = np.array([hist[int(c)] for c in counts], dtype=np.float64)
    probs = weights / weights.sum()
    return int(rng.choice(counts, p=probs))


def _quantile_count_from_hist(hist: Dict[int, int], quantile: float, fallback: int) -> int:
    if not hist:
        return int(fallback)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    counts = np.array(sorted(hist.keys()), dtype=np.int64)
    weights = np.array([hist[int(c)] for c in counts], dtype=np.float64)
    cdf = np.cumsum(weights) / weights.sum()
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(idx, len(counts) - 1)
    return int(counts[idx])


def decode_route_with_priors(
    probs: torch.Tensor,
    hold_mask: torch.Tensor,
    grade_values: torch.Tensor,
    count_histograms: Dict[int, Dict[str, Dict[int, int]]],
    seed: int,
    threshold: float,
    start_min: int,
    start_max: int,
    finish_min: int,
    finish_max: int,
    start_max_dist: float = None,
    finish_max_dist: float = None,
    foot_count_mode: str = "median",
    foot_count_quantile: float = 0.5,
) -> torch.Tensor:
    """
    Decode route channels using empirical per-grade count priors.
    - start/finish counts are sampled from the dataset distribution and then
      constrained by min/max plus proximity for pairs
    - hand/foot counts are sampled from the dataset distribution and decoded by top-k
    """
    out = torch.zeros_like(probs)
    for i in range(out.shape[0]):
        rng = np.random.default_rng(seed + i)
        mask = hold_mask[i, 0] > 0
        grade_v = int(grade_values[i].item())
        grade_hist = count_histograms.get(grade_v, {})

        start_k = _sample_count_from_hist(grade_hist.get("start", {}), rng, fallback=start_max)
        finish_k = _sample_count_from_hist(grade_hist.get("finish", {}), rng, fallback=finish_min)
        hand_k = _sample_count_from_hist(grade_hist.get("hand", {}), rng, fallback=5)
        foot_hist = grade_hist.get("foot", {})
        foot_cap = _quantile_count_from_hist(foot_hist, foot_count_quantile, fallback=4)
        if foot_count_mode == "sample":
            foot_k = _sample_count_from_hist(foot_hist, rng, fallback=4)
        elif foot_count_mode == "trimmed_sample":
            trimmed_hist = {count: weight for count, weight in foot_hist.items() if int(count) <= foot_cap}
            foot_k = _sample_count_from_hist(trimmed_hist, rng, fallback=foot_cap)
        else:
            foot_k = foot_cap

        start_k = max(start_min, min(start_max, start_k))
        finish_k = max(finish_min, min(finish_max, finish_k))
        hand_k = max(1, hand_k)
        foot_k = max(0, foot_k)

        for ch, k, max_dist in [
            (0, start_k, start_max_dist),
            (1, finish_k, finish_max_dist),
        ]:
            if k == 2:
                if max_dist is not None:
                    pair = _pick_best_pair_within_distance(probs[i, ch] * mask, mask, max_dist)
                    if pair.sum() > 0:
                        out[i, ch] = pair
                        continue
                    k = 1
            out[i, ch] = _pick_topk_from_scores(probs[i, ch] * mask, k)

        out[i, 2] = _pick_topk_from_scores(probs[i, 2] * mask, hand_k)
        out[i, 3] = _pick_topk_from_scores(probs[i, 3] * mask, foot_k)
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
        static_channels=dataset.static_channels,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    static = dataset.get_static().unsqueeze(0).to(args.device)
    static = static.expand(args.n, -1, -1, -1)

    grade_idx = torch.tensor([args.grade - grade_min], dtype=torch.int64, device=args.device)
    grade_idx = grade_idx.expand(args.n)
    grade_values = torch.tensor([args.grade] * args.n, dtype=torch.int64, device=args.device)
    count_histograms = _grade_count_histograms(args.data_dir, int(grade_min), int(grade_max))

    with torch.no_grad():
        logits = model.sample(grade_idx, static, n=args.n)
        probs = torch.sigmoid(logits)
        route = decode_route_with_priors(
            probs,
            hold_mask=static[:, 0:1],
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
        "foot_count_mode": args.foot_count_mode,
        "foot_count_quantile": args.foot_count_quantile,
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
