from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROLE_NAMES = ("start", "finish", "hand", "foot")
ROLE_COLORS = {
    "start": (0, 190, 100, 240),
    "finish": (220, 25, 150, 240),
    "hand": (0, 180, 210, 235),
    "foot": (255, 132, 35, 235),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize graph Transformer metrics and generated routes.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--holds-path", default="ImageData/References/holds.json")
    parser.add_argument("--board-image", default="ImageData/References/empty_board.png")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _plot_series(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    title: str,
    records: Sequence[Dict[str, float]],
    keys: Sequence[Tuple[str, str, Tuple[int, int, int]]],
) -> None:
    x0, y0, x1, y1 = bounds
    draw.rounded_rectangle(bounds, radius=12, fill=(255, 255, 255), outline=(185, 181, 170), width=2)
    draw.text((x0 + 16, y0 + 12), title, fill=(28, 31, 30), font=load_font(18, bold=True))
    left, top, right, bottom = x0 + 48, y0 + 48, x1 - 20, y1 - 36
    values = [float(record[key]) for record in records for key, _, _ in keys]
    value_min, value_max = min(values), max(values)
    if value_max <= value_min:
        value_max = value_min + 1.0
    draw.line((left, bottom, right, bottom), fill=(130, 130, 125), width=1)
    draw.line((left, top, left, bottom), fill=(130, 130, 125), width=1)
    for key, label, color in keys:
        points = []
        for index, record in enumerate(records):
            x = left + index * (right - left) / max(len(records) - 1, 1)
            value = float(record[key])
            y = bottom - (value - value_min) * (bottom - top) / (value_max - value_min)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        label_x = right - 150
        label_y = top + 20 * list(keys).index((key, label, color))
        draw.text((label_x, label_y), f"{label}: {records[-1][key]:.3f}", fill=color, font=load_font(13))
    draw.text((left, bottom + 8), "1", fill=(100, 100, 95), font=load_font(12))
    draw.text((right - 20, bottom + 8), str(records[-1]["epoch"]), fill=(100, 100, 95), font=load_font(12))


def render_metrics(run_dir: Path, output_path: Path) -> None:
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
    canvas = Image.new("RGB", (1500, 520), (244, 241, 232))
    draw = ImageDraw.Draw(canvas)
    _plot_series(
        draw,
        (30, 30, 730, 490),
        "Total loss",
        records,
        (("train_loss", "train", (20, 100, 180)), ("val_loss", "validation", (205, 75, 45))),
    )
    _plot_series(
        draw,
        (770, 30, 1470, 490),
        "Validation accuracy",
        records,
        (
            ("val_action_accuracy", "action", (30, 120, 175)),
            ("val_node_accuracy", "exact hold", (225, 120, 20)),
            ("val_role_accuracy", "role", (15, 150, 95)),
        ),
    )
    canvas.save(output_path)


def render_route(board: Image.Image, hold_map: Dict, route: np.ndarray) -> Image.Image:
    image = board.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    for hold in hold_map["holds"]:
        row, col = int(hold["row"]), int(hold["col"])
        x, y = float(hold["x"]), float(hold["y"])
        for channel, role in enumerate(ROLE_NAMES):
            if route[row, col, channel] <= 0.5:
                continue
            radius = 27 if role in ("start", "finish") else 23
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ROLE_COLORS[role], width=6)
    return image.convert("RGB")


def collect_samples(samples_dir: Path) -> List[Tuple[int, Path]]:
    samples = []
    for metadata_path in sorted(samples_dir.glob("V*.json")):
        metadata = json.loads(metadata_path.read_text())
        grade = int(metadata["grade_v"])
        for route in metadata["routes"]:
            samples.append((grade, Path(route["path"])))
    return samples


def render_sample_grid(
    samples: Sequence[Tuple[int, Path]],
    hold_map: Dict,
    board: Image.Image,
    output_path: Path,
    columns: int,
    tile_width: int,
) -> None:
    columns = max(columns, 1)
    board_ratio = board.height / board.width
    image_height = int(tile_width * board_ratio)
    header_height = 54
    rows = math.ceil(len(samples) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * (image_height + header_height)), (238, 235, 225))
    draw = ImageDraw.Draw(canvas)
    for index, (grade, route_path) in enumerate(samples):
        route = np.load(route_path)
        counts = [int((route[..., channel] > 0.5).sum()) for channel in range(4)]
        rendered = render_route(board, hold_map, route)
        rendered.thumbnail((tile_width, image_height), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + header_height)
        draw.rectangle((x, y, x + tile_width, y + header_height), fill=(249, 247, 240))
        draw.text((x + 10, y + 7), f"V{grade}", fill=(25, 28, 27), font=load_font(17, bold=True))
        draw.text(
            (x + 10, y + 29),
            f"start {counts[0]}  finish {counts[1]}  hand {counts[2]}  foot {counts[3]}",
            fill=(80, 82, 78),
            font=load_font(12),
        )
        canvas.paste(rendered, (x, y + header_height))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    hold_map = json.loads(Path(args.holds_path).read_text())
    board = Image.open(args.board_image)
    samples = collect_samples(samples_dir)
    if not samples:
        raise RuntimeError(f"No V*.json sample metadata found in {samples_dir}.")
    render_metrics(run_dir, out_dir / "training_metrics.png")
    render_sample_grid(samples, hold_map, board, out_dir / "generated_routes.png", args.columns, args.tile_width)
    print(f"Rendered metrics and {len(samples)} routes to {out_dir}")


if __name__ == "__main__":
    main()
