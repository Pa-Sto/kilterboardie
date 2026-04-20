import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from cvae_data import KilterRouteDataset
from cvae_generate import _grade_count_histograms, decode_route_with_priors
from diffusion_model import GaussianDiffusion, KilterDiffusionUNet


CHANNEL_NAMES = ("start", "finish", "hand", "foot")
CHANNEL_COLORS = {
    "start": (0, 185, 90, 230),
    "finish": (206, 0, 145, 230),
    "hand": (0, 172, 199, 200),
    "foot": (255, 126, 30, 200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize diffusion training curves and generated samples.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing best.pt and metrics.jsonl")
    parser.add_argument("--data-dir", type=str, default="ImageData/50degree/export")
    parser.add_argument("--holds-json", type=str, default="ImageData/References/holds.json")
    parser.add_argument("--board-image", type=str, default="ImageData/References/empty_board.png")
    parser.add_argument("--grades", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]


def _draw_line_plot(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    title: str,
    epochs: Sequence[int],
    train_vals: Sequence[float],
    val_vals: Sequence[float],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = bounds
    draw.rounded_rectangle(bounds, radius=12, outline=(190, 185, 175), width=2, fill=(255, 255, 255))
    draw.text((x0 + 12, y0 + 10), title, fill=(30, 30, 30), font=font)

    plot_left = x0 + 36
    plot_top = y0 + 36
    plot_right = x1 - 16
    plot_bottom = y1 - 28
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(120, 120, 120), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(120, 120, 120), width=1)

    all_vals = list(train_vals) + list(val_vals)
    vmin = min(all_vals)
    vmax = max(all_vals)
    if vmax <= vmin:
        vmax = vmin + 1.0

    colors = {"train": (25, 92, 180), "val": (200, 75, 45)}
    for split, vals in (("train", train_vals), ("val", val_vals)):
        points = []
        for i, value in enumerate(vals):
            if len(vals) == 1:
                x = (plot_left + plot_right) / 2
            else:
                x = plot_left + i * (plot_right - plot_left) / max(len(vals) - 1, 1)
            y = plot_bottom - (value - vmin) * (plot_bottom - plot_top) / (vmax - vmin)
            points.append((x, y))

        if len(points) >= 2:
            draw.line(points, fill=colors[split], width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=colors[split])

        label_y = plot_top if split == "train" else plot_top + 18
        draw.text((plot_right - 90, label_y), f"{split} {vals[-1]:.3f}", fill=colors[split], font=font)

    for i, epoch in enumerate(epochs):
        if len(epochs) == 1:
            x = (plot_left + plot_right) / 2
        else:
            x = plot_left + i * (plot_right - plot_left) / max(len(epochs) - 1, 1)
        draw.text((x - 4, plot_bottom + 6), str(epoch), fill=(100, 100, 100), font=font)


def render_training_curves(records: List[Dict[str, float]], out_path: Path) -> None:
    font = ImageFont.load_default()
    width, height = 1500, 1200
    image = Image.new("RGB", (width, height), (248, 246, 240))
    draw = ImageDraw.Draw(image)

    plots = [
        ("loss", "Loss"),
        ("eps", "Epsilon"),
        ("recon", "Reconstruction"),
        ("count", "Count"),
        ("path", "Path"),
        ("upward", "Upward"),
        ("hand_density", "Hand Density"),
        ("foot_density", "Foot Density"),
    ]
    margin = 40
    cell_w = (width - margin * 4) // 3
    cell_h = (height - margin * 4) // 3
    epochs = [int(r["epoch"]) for r in records]

    for idx, (suffix, title) in enumerate(plots):
        row = idx // 3
        col = idx % 3
        x0 = margin + col * (cell_w + margin)
        y0 = margin + row * (cell_h + margin)
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        _draw_line_plot(
            draw=draw,
            bounds=(x0, y0, x1, y1),
            title=title,
            epochs=epochs,
            train_vals=[float(r.get(f"train_{suffix}", 0.0)) for r in records],
            val_vals=[float(r.get(f"val_{suffix}", 0.0)) for r in records],
            font=font,
        )

    image.save(out_path)


def _load_model_bundle(run_dir: Path, data_dir: str, device: torch.device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    config = ckpt["config"]

    dataset = KilterRouteDataset(
        data_dir,
        grade_min=int(config["grade_min"]),
        grade_max=int(config["grade_max"]),
    )
    model = KilterDiffusionUNet(
        num_grades=int(config["num_grades"]),
        static_channels=int(dataset.static_channels),
        route_channels=int(config["route_channels"]),
        base_channels=int(config["base_channels"]),
        grade_emb_dim=int(config["grade_emb_dim"]),
        time_emb_dim=int(config["time_emb_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=int(config["timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )

    return config, dataset, model, diffusion


def _sample_route(
    model: KilterDiffusionUNet,
    diffusion: GaussianDiffusion,
    static: torch.Tensor,
    grade_idx: int,
    grade_v: int,
    count_histograms: Dict[int, Dict[str, Dict[int, int]]],
    seed: int,
    threshold: float,
    start_min: int,
    start_max: int,
    finish_min: int,
    finish_max: int,
    start_max_dist: float,
    finish_max_dist: float,
    foot_count_mode: str,
    foot_count_quantile: float,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    hold_mask = static[:, 0:1]
    grade = torch.tensor([grade_idx], dtype=torch.int64, device=device)
    grade_values = torch.tensor([grade_v], dtype=torch.int64, device=device)
    with torch.no_grad():
        x0_pred = diffusion.sample(
            model=model,
            shape=(1, 4, static.shape[2], static.shape[3]),
            static=static,
            grade=grade,
            device=device,
            hold_mask=hold_mask,
        )
        probs = ((x0_pred + 1.0) * 0.5).clamp(0.0, 1.0)
        probs = probs * hold_mask
        route = decode_route_with_priors(
            probs,
            hold_mask=hold_mask,
            grade_values=grade_values,
            count_histograms=count_histograms,
            seed=seed,
            threshold=threshold,
            start_min=start_min,
            start_max=start_max,
            finish_min=finish_min,
            finish_max=finish_max,
            start_max_dist=start_max_dist,
            finish_max_dist=finish_max_dist,
            foot_count_mode=foot_count_mode,
            foot_count_quantile=foot_count_quantile,
        )
        route = route[0]
    return route.permute(1, 2, 0).cpu().numpy()


def _render_route(board: Image.Image, hold_map: Dict, route: np.ndarray) -> Image.Image:
    image = board.copy()
    draw = ImageDraw.Draw(image)
    active = {name: route[:, :, idx] > 0 for idx, name in enumerate(CHANNEL_NAMES)}

    for hold in hold_map["holds"]:
        row = int(hold["row"])
        col = int(hold["col"])
        x = float(hold["x"])
        y = float(hold["y"])

        for name, color in CHANNEL_COLORS.items():
            if not active[name][row, col]:
                continue
            offset = 3 if name in ("start", "finish") else 0
            half = 22 + offset
            draw.rectangle((x - half, y - half, x + half, y + half), outline=color, width=4)

    return image


def render_sample_grid(
    run_dir: Path,
    data_dir: str,
    holds_json: Path,
    board_image: Path,
    grades: Sequence[int],
    threshold: float,
    start_min: int,
    start_max: int,
    finish_min: int,
    finish_max: int,
    start_max_dist: float,
    finish_max_dist: float,
    foot_count_mode: str,
    foot_count_quantile: float,
    seed: int,
    device: torch.device,
) -> None:
    with holds_json.open("r") as f:
        hold_map = json.load(f)

    board = Image.open(board_image).convert("RGBA")
    font = ImageFont.load_default()
    config, dataset, model, diffusion = _load_model_bundle(run_dir, data_dir, device)
    static = dataset.get_static().unsqueeze(0).to(device)
    count_histograms = _grade_count_histograms(data_dir, int(config["grade_min"]), int(config["grade_max"]))

    tiles = []
    for idx, grade_v in enumerate(grades):
        grade_idx = grade_v - int(config["grade_min"])
        route = _sample_route(
            model=model,
            diffusion=diffusion,
            static=static,
            grade_idx=grade_idx,
            grade_v=grade_v,
            count_histograms=count_histograms,
            seed=seed + idx,
            threshold=threshold,
            start_min=start_min,
            start_max=start_max,
            finish_min=finish_min,
            finish_max=finish_max,
            start_max_dist=start_max_dist,
            finish_max_dist=finish_max_dist,
            foot_count_mode=foot_count_mode,
            foot_count_quantile=foot_count_quantile,
            device=device,
        )
        counts = {name: int(route[:, :, channel].sum()) for channel, name in enumerate(CHANNEL_NAMES)}
        tiles.append((grade_v, counts, _render_route(board, hold_map, route)))

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    pad = 24
    label_h = 54
    thumb_w, thumb_h = tiles[0][2].size
    canvas = Image.new(
        "RGBA",
        (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + label_h) + (rows + 1) * pad),
        (247, 245, 240, 255),
    )

    for idx, (grade_v, counts, tile) in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        canvas.alpha_composite(tile, (x, y + label_h))

        draw = ImageDraw.Draw(canvas)
        draw.text((x, y), f"Grade {grade_v}", fill=(20, 20, 20, 255), font=font)
        draw.text(
            (x, y + 18),
            f"S {counts['start']}  F {counts['finish']}  H {counts['hand']}  T {counts['foot']}",
            fill=(90, 90, 90, 255),
            font=font,
        )

    canvas.save(run_dir / "sample_grid.png")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    records = _load_metrics(run_dir / "metrics.jsonl")
    render_training_curves(records, run_dir / "training_curves.png")
    render_sample_grid(
        run_dir=run_dir,
        data_dir=args.data_dir,
        holds_json=Path(args.holds_json),
        board_image=Path(args.board_image),
        grades=args.grades,
        threshold=args.threshold,
        start_min=args.start_min,
        start_max=args.start_max,
        finish_min=args.finish_min,
        finish_max=args.finish_max,
        start_max_dist=args.start_max_dist,
        finish_max_dist=args.finish_max_dist,
        foot_count_mode=args.foot_count_mode,
        foot_count_quantile=args.foot_count_quantile,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(f"Wrote visuals to {run_dir}")


if __name__ == "__main__":
    main()
