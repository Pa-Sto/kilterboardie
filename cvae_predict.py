import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from cvae_data import KilterRouteDataset
from cvae_model import KilterCVAE
from cvae_generate import _enforce_start_finish_counts


@dataclass
class ModelBundle:
    model: KilterCVAE
    static: torch.Tensor  # 1 x 2 x H x W
    grade_min: int
    grade_max: int
    device: torch.device


def load_model_bundle(checkpoint: str, data_dir: str, device: Optional[str] = None) -> ModelBundle:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=device_t)
    config = ckpt.get("config", {})

    dataset = KilterRouteDataset(data_dir)
    grade_min = config.get("grade_min", dataset.grade_min)
    grade_max = config.get("grade_max", dataset.grade_max)

    model = KilterCVAE(
        num_grades=config.get("num_grades", dataset.num_grades),
        emb_dim=config.get("emb_dim", 16),
        latent_dim=config.get("latent_dim", 64),
        static_channels=dataset.static_channels,
    ).to(device_t)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    static = dataset.get_static().unsqueeze(0).to(device_t)

    return ModelBundle(
        model=model,
        static=static,
        grade_min=grade_min,
        grade_max=grade_max,
        device=device_t,
    )


def predict_routes(
    bundle: ModelBundle,
    grade_v: int,
    n: int = 1,
    threshold: float = 0.5,
    enforce_constraints: bool = True,
    start_min: int = 1,
    start_max: int = 2,
    finish_min: int = 1,
    finish_max: int = 2,
    start_max_dist: float = 8.0,
    finish_max_dist: float = 8.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    if grade_v < bundle.grade_min or grade_v > bundle.grade_max:
        raise ValueError(f"grade_v must be in [{bundle.grade_min}, {bundle.grade_max}]")

    if seed is not None:
        torch.manual_seed(seed)

    static = bundle.static
    if n > 1:
        static = static.expand(n, -1, -1, -1)

    grade_idx = torch.tensor([grade_v - bundle.grade_min], dtype=torch.int64, device=bundle.device)
    grade_idx = grade_idx.expand(n)

    with torch.no_grad():
        logits = bundle.model.sample(grade_idx, static, n=n)
        probs = torch.sigmoid(logits)

        if enforce_constraints:
            probs = _enforce_start_finish_counts(
                probs,
                hold_mask=static[:, 0:1],
                threshold=threshold,
                start_min=start_min,
                start_max=start_max,
                finish_min=finish_min,
                finish_max=finish_max,
                start_max_dist=start_max_dist,
                finish_max_dist=finish_max_dist,
            )

        route = (probs >= threshold).float()

    route_np = route.cpu().numpy()  # N x 4 x H x W
    static_np = static.cpu().numpy()  # N x 2 x H x W
    full = np.concatenate([route_np, static_np], axis=1)  # N x 6 x H x W
    full = np.transpose(full, (0, 2, 3, 1))  # N x H x W x 6

    if n == 1:
        return full[0]
    return full
