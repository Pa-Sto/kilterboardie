from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


ACTION_SELECT = 0
ACTION_GROUP_END = 1
ACTION_EOS = 2
ACTION_BOS = 3
ACTION_PAD = 4
NUM_INPUT_ACTIONS = 5
NUM_OUTPUT_ACTIONS = 3

ROLE_START = 0
ROLE_FINISH = 1
ROLE_HAND = 2
ROLE_FOOT = 3
NUM_ROLES = 4

IGNORE_INDEX = -100


@dataclass(frozen=True)
class GraphSequenceSample:
    npy_path: str
    grade_v: int


@dataclass
class BoardGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_rows: torch.Tensor
    node_cols: torch.Tensor
    raw_xy: torch.Tensor
    coordinate_spacing: float
    rows: int
    cols: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def edge_feature_dim(self) -> int:
        return int(self.edge_features.shape[1])

    def to(self, device: torch.device | str) -> "BoardGraph":
        return BoardGraph(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_rows=self.node_rows.to(device),
            node_cols=self.node_cols.to(device),
            raw_xy=self.raw_xy.to(device),
            coordinate_spacing=self.coordinate_spacing,
            rows=self.rows,
            cols=self.cols,
        )


def _orientation_features(orientations: Sequence[float]) -> List[float]:
    features: List[float] = []
    for index in range(2):
        if index < len(orientations):
            theta = float(orientations[index])
            features.extend([math.sin(theta), math.cos(theta), 1.0])
        else:
            features.extend([0.0, 0.0, 0.0])
    return features


def load_board_graph(
    holds_path: str,
    k_neighbors: int = 12,
    edge_radius: float = 4.5,
) -> BoardGraph:
    """
    Build a fixed graph from calibrated hold centers.

    Distances use image-space hold centers from holds.json, not matrix row/column
    distances. Values are normalized by the median nearest-neighbour spacing so
    the radius is expressed in approximate hold-spacing units.
    """
    with open(holds_path, "r") as f:
        hold_map = json.load(f)

    holds = sorted(hold_map["holds"], key=lambda hold: int(hold["id"]))
    raw_xy = torch.tensor([[float(h["x"]), float(h["y"])] for h in holds], dtype=torch.float32)
    node_rows = torch.tensor([int(h["row"]) for h in holds], dtype=torch.long)
    node_cols = torch.tensor([int(h["col"]) for h in holds], dtype=torch.long)

    pair_distance = torch.cdist(raw_xy, raw_xy)
    non_self_distance = pair_distance.clone()
    non_self_distance.fill_diagonal_(float("inf"))
    nearest_distance = non_self_distance.min(dim=1).values
    coordinate_spacing = float(nearest_distance.median().item())
    if coordinate_spacing <= 0.0:
        raise RuntimeError("Could not determine a positive hold-coordinate spacing.")

    x_min, y_min = raw_xy.min(dim=0).values
    x_max, y_max = raw_xy.max(dim=0).values
    board_span = (x_max - x_min).clamp_min(1.0), (y_max - y_min).clamp_min(1.0)
    area = torch.tensor([float(h.get("area_shape", 0.0)) for h in holds], dtype=torch.float32)
    log_area = torch.log1p(area)
    log_area = (log_area - log_area.mean()) / log_area.std().clamp_min(1e-6)

    node_feature_rows: List[List[float]] = []
    for index, hold in enumerate(holds):
        x_norm = float((raw_xy[index, 0] - x_min) / board_span[0])
        # Image y increases downward; invert it so larger values mean higher holds.
        y_up_norm = float((y_max - raw_xy[index, 1]) / board_span[1])
        node_feature_rows.append(
            [x_norm, y_up_norm, float(log_area[index])] + _orientation_features(hold.get("orientations", []))
        )
    node_features = torch.tensor(node_feature_rows, dtype=torch.float32)

    distance_units = pair_distance / coordinate_spacing
    neighbour_mask = distance_units <= float(edge_radius)
    neighbour_mask.fill_diagonal_(False)

    k = min(max(int(k_neighbors), 1), max(len(holds) - 1, 1))
    nearest_indices = non_self_distance.topk(k, largest=False, dim=1).indices
    neighbour_mask.scatter_(1, nearest_indices, True)

    source, target = neighbour_mask.nonzero(as_tuple=True)
    edge_index = torch.stack([source, target], dim=0)

    delta = raw_xy[target] - raw_xy[source]
    dx_units = delta[:, 0] / coordinate_spacing
    dy_up_units = -delta[:, 1] / coordinate_spacing
    dist_units = torch.sqrt(dx_units.square() + dy_up_units.square()).clamp_min(1e-6)
    radius_scale = max(float(edge_radius), 1.0)
    edge_features = torch.stack(
        [
            dx_units / radius_scale,
            dy_up_units / radius_scale,
            dist_units / radius_scale,
            dx_units / dist_units,
            dy_up_units / dist_units,
        ],
        dim=1,
    ).float()

    return BoardGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_rows=node_rows,
        node_cols=node_cols,
        raw_xy=raw_xy,
        coordinate_spacing=coordinate_spacing,
        rows=int(hold_map["rows"]),
        cols=int(hold_map["cols"]),
    )


class KilterGraphSequenceDataset(Dataset):
    """Convert final route matrices into bottom-to-top grouped event sequences."""

    def __init__(
        self,
        data_dir: str,
        holds_path: str = "ImageData/References/holds.json",
        grade_min: int = 3,
        grade_max: int = 13,
        band_rows: int = 3,
        max_sequence_length: int = 48,
        k_neighbors: int = 12,
        edge_radius: float = 4.5,
    ) -> None:
        self.data_dir = data_dir
        self.grade_min = int(grade_min)
        self.grade_max = int(grade_max)
        self.band_rows = max(int(band_rows), 1)
        self.max_sequence_length = int(max_sequence_length)
        self.board_graph = load_board_graph(holds_path, k_neighbors=k_neighbors, edge_radius=edge_radius)
        self.grid_to_node: Dict[Tuple[int, int], int] = {
            (int(row), int(col)): index
            for index, (row, col) in enumerate(zip(self.board_graph.node_rows, self.board_graph.node_cols))
        }
        self.samples: List[GraphSequenceSample] = []
        self._index_samples()

    @property
    def num_grades(self) -> int:
        return self.grade_max - self.grade_min + 1

    def _index_samples(self) -> None:
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith(".json"):
                continue
            stem = filename[:-5]
            npy_path = os.path.join(self.data_dir, stem + ".npy")
            if not os.path.exists(npy_path):
                continue
            with open(os.path.join(self.data_dir, filename), "r") as f:
                metadata = json.load(f)
            grade_v = metadata.get("grade_v")
            if grade_v is None or not self.grade_min <= int(grade_v) <= self.grade_max:
                continue
            self.samples.append(GraphSequenceSample(npy_path=npy_path, grade_v=int(grade_v)))

        if not self.samples:
            raise RuntimeError(
                f"No route samples found in {self.data_dir} for grades V{self.grade_min}-V{self.grade_max}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _nodes_for_channel(self, route: np.ndarray, channel: int) -> List[int]:
        nodes: List[int] = []
        for row, col in np.argwhere(route[..., channel] > 0.5):
            node = self.grid_to_node.get((int(row), int(col)))
            if node is not None:
                nodes.append(node)
        return nodes

    def _sort_group(self, nodes_and_roles: List[Tuple[int, int]], reverse: bool = False) -> List[Tuple[int, int]]:
        return sorted(
            nodes_and_roles,
            key=lambda item: (
                float(self.board_graph.raw_xy[item[0], 0]),
                int(item[1]),
            ),
            reverse=reverse,
        )

    def _append_group(
        self,
        events: List[Tuple[int, int, int]],
        nodes_and_roles: List[Tuple[int, int]],
        reverse: bool = False,
    ) -> None:
        if not nodes_and_roles:
            return
        for node, role in self._sort_group(nodes_and_roles, reverse=reverse):
            events.append((ACTION_SELECT, node, role))
        events.append((ACTION_GROUP_END, IGNORE_INDEX, IGNORE_INDEX))

    def route_to_events(self, route: np.ndarray, sample_index: int = 0) -> List[Tuple[int, int, int]]:
        # The image export occasionally labels one hold in multiple channels.
        # The autoregressive representation assigns one LED role per hold, using
        # the precedence expected by the board UI.
        start_nodes = set(self._nodes_for_channel(route, ROLE_START))
        finish_nodes = set(self._nodes_for_channel(route, ROLE_FINISH)) - start_nodes
        hand_nodes = set(self._nodes_for_channel(route, ROLE_HAND)) - start_nodes - finish_nodes
        foot_nodes = set(self._nodes_for_channel(route, ROLE_FOOT)) - start_nodes - finish_nodes - hand_nodes

        starts = [(node, ROLE_START) for node in start_nodes]
        finishes = [(node, ROLE_FINISH) for node in finish_nodes]
        hands = [(node, ROLE_HAND) for node in hand_nodes]
        feet = [(node, ROLE_FOOT) for node in foot_nodes]

        events: List[Tuple[int, int, int]] = []
        self._append_group(events, starts, reverse=bool(sample_index % 2))

        bands: Dict[int, List[Tuple[int, int]]] = {}
        for node, role in hands:
            row = int(self.board_graph.node_rows[node])
            band = row // self.band_rows
            bands.setdefault(band, []).append((node, role))

        # Feet support hand movements but do not define the climbing path. Attach
        # every foot to the physically closest hand-led group. Routes without an
        # intermediate hand fall back to foot-led vertical bands.
        if bands:
            for foot in feet:
                foot_node = foot[0]
                closest_band = min(
                    bands,
                    key=lambda band: min(
                        float(torch.linalg.vector_norm(self.board_graph.raw_xy[foot_node] - self.board_graph.raw_xy[node]))
                        for node, _ in bands[band]
                    ),
                )
                bands[closest_band].append(foot)
        else:
            for node, role in feet:
                row = int(self.board_graph.node_rows[node])
                band = row // self.band_rows
                bands.setdefault(band, []).append((node, role))

        # Larger image rows are lower on the wall, so descending bands are bottom-to-top.
        for group_index, band in enumerate(sorted(bands, reverse=True)):
            self._append_group(
                events,
                bands[band],
                reverse=bool((sample_index + group_index) % 2),
            )

        self._append_group(events, finishes, reverse=bool(sample_index % 2))
        events.append((ACTION_EOS, IGNORE_INDEX, IGNORE_INDEX))

        if len(events) > self.max_sequence_length:
            raise RuntimeError(
                f"Route sequence has {len(events)} events, exceeding max_sequence_length={self.max_sequence_length}."
            )
        return events

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        route = np.load(sample.npy_path)
        events = self.route_to_events(route, sample_index=index)

        target_actions = torch.tensor([event[0] for event in events], dtype=torch.long)
        target_nodes = torch.tensor([event[1] for event in events], dtype=torch.long)
        target_roles = torch.tensor([event[2] for event in events], dtype=torch.long)

        input_actions = torch.cat([torch.tensor([ACTION_BOS]), target_actions[:-1]])
        input_nodes = torch.cat([torch.tensor([IGNORE_INDEX]), target_nodes[:-1]])
        input_roles = torch.cat([torch.tensor([IGNORE_INDEX]), target_roles[:-1]])

        return {
            "input_actions": input_actions,
            "input_nodes": input_nodes,
            "input_roles": input_roles,
            "target_actions": target_actions,
            "target_nodes": target_nodes,
            "target_roles": target_roles,
            "grade": torch.tensor(sample.grade_v - self.grade_min, dtype=torch.long),
        }


def collate_graph_sequences(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    def padded(key: str, value: int) -> torch.Tensor:
        return pad_sequence([sample[key] for sample in samples], batch_first=True, padding_value=value)

    return {
        "input_actions": padded("input_actions", ACTION_PAD),
        "input_nodes": padded("input_nodes", IGNORE_INDEX),
        "input_roles": padded("input_roles", IGNORE_INDEX),
        "target_actions": padded("target_actions", IGNORE_INDEX),
        "target_nodes": padded("target_nodes", IGNORE_INDEX),
        "target_roles": padded("target_roles", IGNORE_INDEX),
        "grade": torch.stack([sample["grade"] for sample in samples]),
    }


def events_to_route_matrix(
    events: Sequence[Tuple[int, int, int]],
    board_graph: BoardGraph,
    static_matrix: np.ndarray,
) -> np.ndarray:
    output = np.zeros((board_graph.rows, board_graph.cols, 4 + static_matrix.shape[2]), dtype=np.float32)
    output[..., 4:] = static_matrix
    for action, node, role in events:
        if action != ACTION_SELECT or node < 0 or role < 0:
            continue
        row = int(board_graph.node_rows[node])
        col = int(board_graph.node_cols[node])
        output[row, col, role] = 1.0
    return output
