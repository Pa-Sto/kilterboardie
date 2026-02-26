import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvae_model import KilterCVAE
from cvae_generate import _enforce_start_finish_counts


COLORS = {
    "start": (0, 185, 90, 230),
    "finish": (206, 0, 145, 230),
    "hand": (0, 172, 199, 200),
    "foot": (255, 126, 30, 200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a route and render it on the board image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--holds-json", required=True)
    parser.add_argument("--board-image", required=True)
    parser.add_argument("--out-image", required=True)
    parser.add_argument("--out-matrix", required=True)
    parser.add_argument("--out-meta", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--start-min", type=int, default=1)
    parser.add_argument("--start-max", type=int, default=2)
    parser.add_argument("--finish-min", type=int, default=1)
    parser.add_argument("--finish-max", type=int, default=2)
    parser.add_argument("--start-max-dist", type=float, default=8.0)
    parser.add_argument("--finish-max-dist", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_hold_map(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def build_static_from_holds(hold_map: Dict) -> torch.Tensor:
    rows = int(hold_map["rows"])
    cols = int(hold_map["cols"])
    holds = hold_map["holds"]
    presence = np.zeros((rows, cols), dtype=np.float32)
    size = np.zeros((rows, cols), dtype=np.float32)
    max_area = max(float(h.get("area_shape", 1.0)) for h in holds)
    for hold in holds:
        r = int(hold["row"])
        c = int(hold["col"])
        presence[r, c] = 1.0
        size[r, c] = float(hold.get("area_shape", 0.0)) / max_area
    stacked = np.stack([presence, size], axis=0)
    return torch.from_numpy(stacked)


def sample_route(
    checkpoint: Path,
    grade_v: int,
    static: torch.Tensor,
    threshold: float,
    start_min: int,
    start_max: int,
    finish_min: int,
    finish_max: int,
    start_max_dist: float,
    finish_max_dist: float,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict]:
    torch.manual_seed(seed)
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config", {})

    grade_min = int(config.get("grade_min", 3))
    grade_max = int(config.get("grade_max", 13))
    if grade_v < grade_min or grade_v > grade_max:
        raise ValueError(f"grade must be in [{grade_min}, {grade_max}]")

    model = KilterCVAE(
        num_grades=int(config.get("num_grades", grade_max - grade_min + 1)),
        emb_dim=int(config.get("emb_dim", 16)),
        latent_dim=int(config.get("latent_dim", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    static = static.unsqueeze(0).to(device)
    grade_idx = torch.tensor([grade_v - grade_min], dtype=torch.int64, device=device)

    with torch.no_grad():
        logits = model.sample(grade_idx, static, n=1)
        probs = torch.sigmoid(logits)
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

    route_np = route.cpu().numpy()[0]  # 4 x H x W
    static_np = static.cpu().numpy()[0]  # 2 x H x W
    full = np.concatenate([route_np, static_np], axis=0)
    full = np.transpose(full, (1, 2, 0))  # H x W x 6

    meta = {
        "grade_v": grade_v,
        "threshold": threshold,
        "start_min": start_min,
        "start_max": start_max,
        "finish_min": finish_min,
        "finish_max": finish_max,
        "start_max_dist": start_max_dist,
        "finish_max_dist": finish_max_dist,
        "seed": seed,
    }

    return full, meta


def render_overlay(board_image: Path, hold_map: Dict, route: np.ndarray, out_path: Path) -> None:
    img = Image.open(board_image).convert("RGBA")
    draw = ImageDraw.Draw(img)

    holds = hold_map["holds"]
    channels = {
        "start": 0,
        "finish": 1,
        "hand": 2,
        "foot": 3,
    }

    # Precompute active flags for speed.
    active = {name: route[:, :, idx] > 0 for name, idx in channels.items()}

    for hold in holds:
        r = int(hold["row"])
        c = int(hold["col"])
        x = float(hold["x"])
        y = float(hold["y"])

        radius = 12
        for name, color in COLORS.items():
            if not active[name][r, c]:
                continue
            offset = 2 if name in ("start", "finish") else 0
            r_inner = radius + offset
            bbox = (x - r_inner, y - r_inner, x + r_inner, y + r_inner)
            draw.ellipse(bbox, outline=color, width=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    hold_map = load_hold_map(Path(args.holds_json))
    static = build_static_from_holds(hold_map)

    matrix, meta = sample_route(
        checkpoint=Path(args.checkpoint),
        grade_v=args.grade,
        static=static,
        threshold=args.threshold,
        start_min=args.start_min,
        start_max=args.start_max,
        finish_min=args.finish_min,
        finish_max=args.finish_max,
        start_max_dist=args.start_max_dist,
        finish_max_dist=args.finish_max_dist,
        seed=args.seed,
        device=device,
    )

    request_id = args.request_id or f"run-{int(datetime.utcnow().timestamp())}"
    meta.update({
        "request_id": request_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "matrix_path": str(args.out_matrix),
        "image_path": str(args.out_image),
    })

    np.save(args.out_matrix, matrix)
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    render_overlay(Path(args.board_image), hold_map, matrix, Path(args.out_image))


if __name__ == "__main__":
    main()
