import json
import numpy as np
import torch

from cvae_model import KilterCVAE


def load_bundle(bundle_dir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with open(f"{bundle_dir}/config.json", "r") as f:
        config = json.load(f)

    static = np.load(f"{bundle_dir}/static.npy")  # H x W x 6
    static = torch.from_numpy(static).float().permute(2, 0, 1).unsqueeze(0).to(device)

    ckpt = torch.load(f"{bundle_dir}/best.pt", map_location=device)
    model = KilterCVAE(
        num_grades=config["grade_max"] - config["grade_min"] + 1,
        emb_dim=16,
        latent_dim=64,
        static_channels=config["static_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, static, config, device


def enforce_counts(probs, hold_mask, threshold, start_min, start_max, finish_min, finish_max, start_max_dist, finish_max_dist):
    # reuse the logic from cvae_generate.py in a minimal form
    def pick_topk(scores, k):
        flat = scores.view(-1)
        if k <= 0:
            return torch.zeros_like(flat).view_as(scores)
        k = min(k, flat.numel())
        vals, idx = torch.topk(flat, k)
        out = torch.zeros_like(flat)
        out[idx] = 1.0
        return out.view_as(scores)

    def pick_best_pair(scores, mask, max_dist):
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
                pair_mask = mask & (scores >= threshold)
                pair = pick_best_pair(scores, pair_mask, max_dist)
                if pair.sum() > 0:
                    out[i, ch] = pair
                    continue
            out[i, ch] = pick_topk(scores, k)
    return out


def predict(bundle_dir, grade_v, n=1, threshold=0.5, seed=None):
    model, static, config, device = load_bundle(bundle_dir)
    if seed is not None:
        torch.manual_seed(seed)
    if grade_v < config["grade_min"] or grade_v > config["grade_max"]:
        raise ValueError(f"grade_v must be in [{config['grade_min']}, {config['grade_max']}]")

    if n > 1:
        static = static.expand(n, -1, -1, -1)
    grade_idx = torch.tensor([grade_v - config["grade_min"]], dtype=torch.int64, device=device).expand(n)

    with torch.no_grad():
        logits = model.sample(grade_idx, static, n=n)
        probs = torch.sigmoid(logits)
        probs = enforce_counts(
            probs,
            hold_mask=static[:, 0:1],
            threshold=threshold,
            start_min=config["start_min"],
            start_max=config["start_max"],
            finish_min=config["finish_min"],
            finish_max=config["finish_max"],
            start_max_dist=config["start_max_dist"],
            finish_max_dist=config["finish_max_dist"],
        )
        route = (probs >= threshold).float()

    route_np = route.cpu().numpy()  # N x 4 x H x W
    static_np = static.cpu().numpy()  # N x 6 x H x W
    full = np.concatenate([route_np, static_np], axis=1)  # N x 10 x H x W
    full = np.transpose(full, (0, 2, 3, 1))  # N x H x W x 10

    if n == 1:
        return full[0]
    return full
