from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from graph_transformer_data import (
    ACTION_GROUP_END,
    ACTION_SELECT,
    ROLE_FINISH,
    ROLE_FOOT,
    ROLE_HAND,
    ROLE_START,
    KilterGraphSequenceDataset,
)


ROLE_NAMES = {
    ROLE_START: "start",
    ROLE_FINISH: "finish",
    ROLE_HAND: "hand",
    ROLE_FOOT: "foot",
}
ROLE_COLORS = {
    ROLE_START: (0, 190, 100, 235),
    ROLE_FINISH: (220, 25, 150, 235),
    ROLE_HAND: (0, 180, 210, 225),
    ROLE_FOOT: (255, 132, 35, 225),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize grouped pseudo-sequences on the real board image.")
    parser.add_argument("--data-dir", default="ImageData/50Degree/ExportClean")
    parser.add_argument("--holds-path", default="ImageData/References/holds.json")
    parser.add_argument("--board-image", default="ImageData/References/empty_board.png")
    parser.add_argument("--grades", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
    parser.add_argument("--per-grade", type=int, default=4)
    parser.add_argument("--band-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=390)
    parser.add_argument("--out-dir", default="runs/graph_sequence_preview")
    parser.add_argument("--save-individuals", action="store_true")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def split_groups(events: Sequence[Tuple[int, int, int]]) -> List[List[Tuple[int, int]]]:
    groups: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    for action, node, role in events:
        if action == ACTION_SELECT:
            current.append((node, role))
        elif action == ACTION_GROUP_END and current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def group_centroid(group: Sequence[Tuple[int, int]], raw_xy) -> Tuple[float, float]:
    path_nodes = [node for node, role in group if role != ROLE_FOOT]
    if not path_nodes:
        path_nodes = [node for node, _ in group]
    points = raw_xy[path_nodes]
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def maximum_group_gap(groups: Sequence[Sequence[Tuple[int, int]]], raw_xy, spacing: float) -> float:
    maximum = 0.0
    for previous, current in zip(groups, groups[1:]):
        previous_nodes = [node for node, role in previous if role != ROLE_FOOT] or [node for node, _ in previous]
        current_nodes = [node for node, role in current if role != ROLE_FOOT] or [node for node, _ in current]
        previous_xy = raw_xy[previous_nodes].numpy()
        current_xy = raw_xy[current_nodes].numpy()
        delta = previous_xy[:, None, :] - current_xy[None, :, :]
        closest = float(np.sqrt(np.square(delta).sum(axis=2)).min()) / spacing
        maximum = max(maximum, closest)
    return maximum


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    draw.line((start, end), fill=(250, 250, 246, 235), width=11)
    draw.line((start, end), fill=(35, 43, 48, 220), width=5)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = (end[0] - ux * 28, end[1] - uy * 28)
    left = (tip[0] - ux * 17 + px * 13, tip[1] - uy * 17 + py * 13)
    right = (tip[0] - ux * 17 - px * 13, tip[1] - uy * 17 - py * 13)
    draw.polygon((tip, left, right), fill=(35, 43, 48, 230))


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((center[0] - width / 2, center[1] - height / 2 - box[1]), text, font=font, fill=fill)


def render_route(
    board: Image.Image,
    groups: Sequence[Sequence[Tuple[int, int]]],
    graph,
) -> Image.Image:
    image = board.copy().convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    group_font = load_font(25, bold=True)

    centroids = [group_centroid(group, graph.raw_xy) for group in groups]
    for start, end in zip(centroids, centroids[1:]):
        draw_arrow(draw, start, end)

    for group_index, group in enumerate(groups, start=1):
        for node, role in group:
            x = float(graph.raw_xy[node, 0])
            y = float(graph.raw_xy[node, 1])
            radius = 25 if role in (ROLE_START, ROLE_FINISH) else 21
            color = ROLE_COLORS[role]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255, 245), width=4)

        center = centroids[group_index - 1]
        badge_center = (center[0] + 29, center[1] - 29)
        draw.ellipse(
            (badge_center[0] - 18, badge_center[1] - 18, badge_center[0] + 18, badge_center[1] + 18),
            fill=(24, 29, 32, 245),
            outline=(255, 255, 255, 245),
            width=3,
        )
        draw_centered_text(draw, badge_center, str(group_index), group_font, (255, 255, 255, 255))

    return image


def sample_indices(dataset: KilterGraphSequenceDataset, grades: Sequence[int], per_grade: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    selected: List[int] = []
    by_grade: Dict[int, Dict[str, int]] = {grade: {} for grade in grades}
    for index, sample in enumerate(dataset.samples):
        if sample.grade_v in by_grade:
            route = np.load(sample.npy_path, mmap_mode="r")
            signature = hashlib.sha1(np.asarray(route[..., :4]).tobytes()).hexdigest()
            by_grade[sample.grade_v].setdefault(signature, index)
    for grade in grades:
        candidates = list(by_grade[grade].values())
        if not candidates:
            continue
        selected.extend(rng.sample(candidates, min(per_grade, len(candidates))))
    return selected


def route_title(npy_path: str) -> str:
    json_path = os.path.splitext(npy_path)[0] + ".json"
    with open(json_path, "r") as f:
        metadata = json.load(f)
    name = str(metadata.get("name") or Path(npy_path).stem)
    return name if len(name) <= 28 else name[:25] + "..."


def make_tile(
    overlay: Image.Image,
    grade: int,
    title: str,
    groups: Sequence[Sequence[Tuple[int, int]]],
    gap: float,
    width: int,
) -> Image.Image:
    header_height = 66
    board_height = round(width * overlay.height / overlay.width)
    tile = Image.new("RGBA", (width, header_height + board_height), (245, 242, 234, 255))
    draw = ImageDraw.Draw(tile)
    title_font = load_font(18, bold=True)
    detail_font = load_font(14)
    draw.text((10, 8), f"V{grade}  {title}", font=title_font, fill=(25, 29, 31, 255))
    hold_count = sum(len(group) for group in groups)
    detail_color = (185, 55, 45, 255) if gap > 10.0 else (82, 88, 88, 255)
    draw.text(
        (10, 36),
        f"{len(groups)} groups  |  {hold_count} holds  |  max gap {gap:.1f}u",
        font=detail_font,
        fill=detail_color,
    )
    resized = overlay.resize((width, board_height), Image.Resampling.LANCZOS)
    tile.alpha_composite(resized, (0, header_height))
    return tile


def render_legend(width: int) -> Image.Image:
    height = 76
    image = Image.new("RGBA", (width, height), (34, 39, 41, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(21, bold=True)
    font = load_font(15)
    draw.text((18, 10), "Pseudo-sequence inspection: numbered groups progress from start to finish", font=title_font, fill="white")
    x = 20
    for role in (ROLE_START, ROLE_HAND, ROLE_FOOT, ROLE_FINISH):
        y = 50
        draw.ellipse((x, y - 8, x + 16, y + 8), fill=ROLE_COLORS[role])
        draw.text((x + 23, y - 9), ROLE_NAMES[role], font=font, fill=(225, 228, 225, 255))
        x += 105
    draw.text((x + 8, 41), "gap unit = median nearest-hold spacing", font=font, fill=(180, 185, 182, 255))
    return image


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = KilterGraphSequenceDataset(
        data_dir=args.data_dir,
        holds_path=args.holds_path,
        band_rows=args.band_rows,
    )
    graph = dataset.board_graph
    board = Image.open(args.board_image).convert("RGBA")
    indices = sample_indices(dataset, args.grades, args.per_grade, args.seed)
    if not indices:
        raise RuntimeError("No routes matched the requested grades.")

    tiles: List[Image.Image] = []
    manifest = []
    individual_dir = out_dir / "routes"
    if args.save_individuals:
        individual_dir.mkdir(parents=True, exist_ok=True)

    for display_index, dataset_index in enumerate(indices):
        sample = dataset.samples[dataset_index]
        route = np.load(sample.npy_path)
        events = dataset.route_to_events(route, sample_index=dataset_index)
        groups = split_groups(events)
        gap = maximum_group_gap(groups, graph.raw_xy, graph.coordinate_spacing)
        overlay = render_route(board, groups, graph)
        title = route_title(sample.npy_path)
        tiles.append(make_tile(overlay, sample.grade_v, title, groups, gap, args.tile_width))

        individual_path = None
        if args.save_individuals:
            individual_path = individual_dir / f"{display_index:02d}_V{sample.grade_v}_{Path(sample.npy_path).stem}.png"
            overlay.save(individual_path)
        manifest.append(
            {
                "dataset_index": dataset_index,
                "npy_path": sample.npy_path,
                "grade_v": sample.grade_v,
                "name": title,
                "groups": len(groups),
                "holds": sum(len(group) for group in groups),
                "max_group_gap_units": gap,
                "individual_image": str(individual_path) if individual_path else None,
            }
        )

    columns = max(1, min(args.columns, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    padding = 14
    legend = render_legend(columns * args.tile_width + (columns - 1) * padding)
    tile_height = max(tile.height for tile in tiles)
    canvas_width = columns * args.tile_width + (columns - 1) * padding
    canvas_height = legend.height + padding + rows * tile_height + (rows - 1) * padding
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (225, 222, 214, 255))
    canvas.alpha_composite(legend, (0, 0))

    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        x = col * (args.tile_width + padding)
        y = legend.height + padding + row * (tile_height + padding)
        canvas.alpha_composite(tile, (x, y))

    grid_path = out_dir / "pseudo_sequence_grid.png"
    canvas.convert("RGB").save(grid_path, quality=95)
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(
            {
                "coordinate_source": "calibrated_image_hold_centers",
                "coordinate_spacing_pixels": graph.coordinate_spacing,
                "band_rows": args.band_rows,
                "routes": manifest,
            },
            f,
            indent=2,
        )
    print(f"Rendered {len(tiles)} pseudo-sequences to {grid_path}")


if __name__ == "__main__":
    main()
