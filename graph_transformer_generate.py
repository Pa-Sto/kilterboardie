from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from graph_transformer_data import (
    ACTION_BOS,
    ACTION_EOS,
    ACTION_GROUP_END,
    ACTION_SELECT,
    IGNORE_INDEX,
    ROLE_FINISH,
    ROLE_FOOT,
    ROLE_HAND,
    ROLE_START,
    KilterGraphSequenceDataset,
    events_to_route_matrix,
)
from graph_transformer_model import HierarchicalGraphTransformer


ROLE_NAMES = {
    ROLE_START: "start",
    ROLE_FINISH: "finish",
    ROLE_HAND: "hand",
    ROLE_FOOT: "foot",
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate routes with a hierarchical graph Transformer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="ImageData/50Degree/ExportClean")
    parser.add_argument("--holds-path", default="ImageData/References/holds.json")
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--start-min", type=int, default=1)
    parser.add_argument("--start-max", type=int, default=2)
    parser.add_argument("--finish-min", type=int, default=1)
    parser.add_argument("--finish-max", type=int, default=2)
    parser.add_argument("--min-body-holds", type=int, default=4)
    parser.add_argument("--max-body-holds", type=int, default=20)
    parser.add_argument("--max-group-size", type=int, default=4)
    parser.add_argument("--pair-max-distance", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="generated_graph_route.npy")
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def choose(logits: torch.Tensor, temperature: float, top_k: int, greedy: bool) -> int:
    if greedy:
        return int(logits.argmax().item())
    scaled = logits / max(float(temperature), 1e-4)
    if 0 < top_k < scaled.numel():
        threshold = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


def constrain_logits(logits: torch.Tensor, allowed: Sequence[int]) -> torch.Tensor:
    constrained = torch.full_like(logits, float("-inf"))
    constrained[list(allowed)] = logits[list(allowed)]
    return constrained


def pair_distance_mask(
    graph,
    candidates: torch.Tensor,
    anchors: Sequence[int],
    maximum_distance: float,
) -> torch.Tensor:
    if not anchors or maximum_distance <= 0.0:
        return candidates
    anchor_xy = graph.raw_xy[torch.tensor(anchors, dtype=torch.long, device=graph.raw_xy.device)]
    distance = torch.cdist(graph.raw_xy, anchor_xy).min(dim=1).values / graph.coordinate_spacing
    return candidates & distance.le(maximum_distance)


def generate_events(
    model: HierarchicalGraphTransformer,
    graph,
    grade_index: int,
    args: argparse.Namespace,
) -> List[Tuple[int, int, int]]:
    device = graph.node_features.device
    input_actions = [ACTION_BOS]
    input_nodes = [IGNORE_INDEX]
    input_roles = [IGNORE_INDEX]
    events: List[Tuple[int, int, int]] = []
    used_nodes: set[int] = set()
    start_nodes: List[int] = []
    finish_nodes: List[int] = []

    phase = "start"
    group_count = 0
    body_count = 0
    grade = torch.tensor([grade_index], dtype=torch.long, device=device)
    node_context = model.graph_encoder(graph)

    for _ in range(model.max_sequence_length - 1):
        action_tensor = torch.tensor([input_actions], dtype=torch.long, device=device)
        node_tensor = torch.tensor([input_nodes], dtype=torch.long, device=device)
        role_tensor = torch.tensor([input_roles], dtype=torch.long, device=device)
        action_logits, pointer_logits, hidden = model.decode(
            node_context=node_context,
            input_actions=action_tensor,
            input_nodes=node_tensor,
            input_roles=role_tensor,
            grade=grade,
        )
        next_action_logits = action_logits[0, -1]

        if phase == "start":
            allowed_actions = [ACTION_SELECT] if group_count < args.start_min else [ACTION_SELECT, ACTION_GROUP_END]
            if group_count >= args.start_max:
                allowed_actions = [ACTION_GROUP_END]
        elif phase == "body":
            allowed_actions = [ACTION_SELECT] if group_count == 0 else [ACTION_SELECT, ACTION_GROUP_END]
            if group_count >= args.max_group_size or (body_count >= args.max_body_holds and group_count > 0):
                allowed_actions = [ACTION_GROUP_END]
        elif phase == "finish":
            allowed_actions = [ACTION_SELECT] if group_count < args.finish_min else [ACTION_SELECT, ACTION_GROUP_END]
            if group_count >= args.finish_max:
                allowed_actions = [ACTION_GROUP_END]
        else:
            allowed_actions = [ACTION_EOS]

        action = choose(
            constrain_logits(next_action_logits, allowed_actions),
            temperature=args.temperature,
            top_k=len(allowed_actions),
            greedy=args.greedy,
        )

        node = IGNORE_INDEX
        role = IGNORE_INDEX
        if action == ACTION_SELECT:
            candidate_mask = torch.ones(graph.num_nodes, dtype=torch.bool, device=device)
            if used_nodes:
                candidate_mask[torch.tensor(sorted(used_nodes), dtype=torch.long, device=device)] = False
            if phase == "start" and start_nodes:
                close_mask = pair_distance_mask(graph, candidate_mask, start_nodes, args.pair_max_distance)
                if close_mask.any():
                    candidate_mask = close_mask
            if phase == "finish" and finish_nodes:
                close_mask = pair_distance_mask(graph, candidate_mask, finish_nodes, args.pair_max_distance)
                if close_mask.any():
                    candidate_mask = close_mask

            node_logits = pointer_logits[0, -1].masked_fill(~candidate_mask, float("-inf"))
            node = choose(node_logits, args.temperature, args.top_k, args.greedy)
            selected_hidden = hidden[0, -1].unsqueeze(0)
            selected_node = torch.tensor([node], dtype=torch.long, device=device)
            role_logits = model.role_logits(selected_hidden, selected_node, node_context)[0]

            if phase == "start":
                allowed_roles = [ROLE_START]
            elif phase == "finish":
                allowed_roles = [ROLE_FINISH]
            else:
                force_finish = body_count >= args.max_body_holds and group_count == 0
                if force_finish:
                    allowed_roles = [ROLE_FINISH]
                else:
                    allowed_roles = [ROLE_HAND, ROLE_FOOT]
                    if body_count >= args.min_body_holds and group_count == 0:
                        allowed_roles.append(ROLE_FINISH)
            role = choose(
                constrain_logits(role_logits, allowed_roles),
                temperature=args.temperature,
                top_k=len(allowed_roles),
                greedy=args.greedy,
            )
            used_nodes.add(node)

            if role == ROLE_START:
                start_nodes.append(node)
                group_count += 1
            elif role == ROLE_FINISH:
                if phase == "body":
                    phase = "finish"
                    group_count = 0
                finish_nodes.append(node)
                group_count += 1
            else:
                body_count += 1
                group_count += 1

        elif action == ACTION_GROUP_END:
            group_count = 0
            if phase == "start":
                phase = "body"
            elif phase == "finish":
                phase = "after_finish"
        elif action == ACTION_EOS:
            events.append((action, node, role))
            break

        events.append((action, node, role))
        input_actions.append(action)
        input_nodes.append(node)
        input_roles.append(role)
    else:
        events.append((ACTION_EOS, IGNORE_INDEX, IGNORE_INDEX))

    return events


def event_metadata(events: Sequence[Tuple[int, int, int]], graph) -> List[Dict[str, int | str]]:
    result: List[Dict[str, int | str]] = []
    group = 0
    for action, node, role in events:
        if action == ACTION_GROUP_END:
            group += 1
        elif action == ACTION_SELECT:
            result.append(
                {
                    "group": group,
                    "node": node,
                    "row": int(graph.node_rows[node]),
                    "col": int(graph.node_cols[node]),
                    "role": ROLE_NAMES[role],
                }
            )
    return result


def output_path(base_path: str, index: int, count: int) -> str:
    if count == 1:
        return base_path
    stem, extension = os.path.splitext(base_path)
    return f"{stem}_{index:02d}{extension}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    grade_min = int(config.get("grade_min", 3))
    grade_max = int(config.get("grade_max", 13))
    if not grade_min <= args.grade <= grade_max:
        raise ValueError(f"grade must be in [{grade_min}, {grade_max}]")

    dataset = KilterGraphSequenceDataset(
        data_dir=args.data_dir,
        holds_path=args.holds_path,
        grade_min=grade_min,
        grade_max=grade_max,
        band_rows=int(config.get("band_rows", 3)),
        max_sequence_length=int(config.get("max_sequence_length", 48)),
        k_neighbors=int(config.get("k_neighbors", 12)),
        edge_radius=float(config.get("edge_radius", 4.5)),
    )
    graph = dataset.board_graph.to(device)
    model = HierarchicalGraphTransformer(
        num_grades=int(config["num_grades"]),
        node_feature_dim=graph.node_feature_dim,
        edge_feature_dim=graph.edge_feature_dim,
        hidden_dim=int(config.get("hidden_dim", 128)),
        graph_layers=int(config.get("graph_layers", 3)),
        transformer_layers=int(config.get("transformer_layers", 3)),
        attention_heads=int(config.get("attention_heads", 4)),
        feedforward_dim=int(config.get("feedforward_dim", 384)),
        dropout=float(config.get("dropout", 0.1)),
        max_sequence_length=int(config.get("max_sequence_length", 48)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    reference = np.load(dataset.samples[0].npy_path)
    static_matrix = reference[..., 4:].astype(np.float32)
    all_metadata = []
    with torch.no_grad():
        for index in range(args.n):
            torch.manual_seed(args.seed + index)
            events = generate_events(model, graph, args.grade - grade_min, args)
            route_matrix = events_to_route_matrix(events, graph, static_matrix)
            path = output_path(args.out, index, args.n)
            output_directory = os.path.dirname(path)
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
            np.save(path, route_matrix)
            all_metadata.append({"path": path, "events": event_metadata(events, graph)})

    metadata = {
        "model": "hierarchical_graph_transformer",
        "checkpoint": args.checkpoint,
        "grade_v": args.grade,
        "coordinate_source": "calibrated_image_hold_centers",
        "coordinate_spacing_pixels": graph.coordinate_spacing,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "routes": all_metadata,
    }
    metadata_path = os.path.splitext(args.out)[0] + ".json"
    metadata_directory = os.path.dirname(metadata_path)
    if metadata_directory:
        os.makedirs(metadata_directory, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {args.n} route(s) from V{args.grade} to {args.out}")


if __name__ == "__main__":
    main()
