from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_transformer_data import (
    ACTION_PAD,
    ACTION_SELECT,
    IGNORE_INDEX,
    NUM_INPUT_ACTIONS,
    NUM_OUTPUT_ACTIONS,
    NUM_ROLES,
    BoardGraph,
)


class GatedGraphLayer(nn.Module):
    """Shared edge-message network followed by a residual node update."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        source_nodes = nodes[source]
        target_nodes = nodes[target]
        messages = self.message(torch.cat([source_nodes, edge_features], dim=-1))
        weights = self.gate(torch.cat([source_nodes, target_nodes, edge_features], dim=-1))

        weighted_messages = messages * weights
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, target, weighted_messages)

        weight_sum = torch.zeros(nodes.shape[0], 1, dtype=nodes.dtype, device=nodes.device)
        weight_sum.index_add_(0, target, weights)
        aggregate = aggregate / weight_sum.clamp_min(1e-6)

        update = self.update(torch.cat([nodes, aggregate], dim=-1))
        return self.norm(nodes + update)


class BoardGraphEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            GatedGraphLayer(hidden_dim, edge_feature_dim, dropout) for _ in range(num_layers)
        )

    def forward(self, graph: BoardGraph) -> torch.Tensor:
        nodes = self.input_projection(graph.node_features)
        for layer in self.layers:
            nodes = layer(nodes, graph.edge_index, graph.edge_features)
        return nodes


class HierarchicalGraphTransformer(nn.Module):
    """
    Static graph encoder plus a causal route decoder.

    The decoder predicts an action (select/group-end/end), points to a real
    board hold for select actions, and assigns that hold a route role.
    """

    def __init__(
        self,
        num_grades: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        transformer_layers: int = 3,
        attention_heads: int = 4,
        feedforward_dim: int = 384,
        dropout: float = 0.1,
        max_sequence_length: int = 48,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.num_grades = int(num_grades)
        self.hidden_dim = int(hidden_dim)
        self.max_sequence_length = int(max_sequence_length)

        self.graph_encoder = BoardGraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=graph_layers,
            dropout=dropout,
        )
        self.action_embedding = nn.Embedding(NUM_INPUT_ACTIONS, hidden_dim)
        self.role_embedding = nn.Embedding(NUM_ROLES, hidden_dim)
        self.grade_embedding = nn.Embedding(num_grades, hidden_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=transformer_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.action_head = nn.Linear(hidden_dim, NUM_OUTPUT_ACTIONS)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_bias = nn.Linear(hidden_dim, 1)
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_ROLES),
        )

    def _input_embeddings(
        self,
        input_actions: torch.Tensor,
        input_nodes: torch.Tensor,
        input_roles: torch.Tensor,
        grade: torch.Tensor,
        node_context: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = input_actions.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds configured maximum {self.max_sequence_length}."
            )

        positions = torch.arange(sequence_length, device=input_actions.device)
        embeddings = self.action_embedding(input_actions)
        embeddings = embeddings + self.position_embedding(positions)[None]
        embeddings = embeddings + self.grade_embedding(grade)[:, None]

        select_mask = input_actions.eq(ACTION_SELECT)
        safe_nodes = input_nodes.clamp_min(0)
        safe_roles = input_roles.clamp_min(0)
        selected_embeddings = node_context[safe_nodes] + self.role_embedding(safe_roles)
        embeddings = embeddings + selected_embeddings * select_mask[..., None]
        return self.input_dropout(self.input_norm(embeddings))

    def forward(
        self,
        graph: BoardGraph,
        input_actions: torch.Tensor,
        input_nodes: torch.Tensor,
        input_roles: torch.Tensor,
        grade: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_context = self.graph_encoder(graph)
        action_logits, pointer_logits, hidden = self.decode(
            node_context=node_context,
            input_actions=input_actions,
            input_nodes=input_nodes,
            input_roles=input_roles,
            grade=grade,
        )
        return action_logits, pointer_logits, hidden, node_context

    def decode(
        self,
        node_context: torch.Tensor,
        input_actions: torch.Tensor,
        input_nodes: torch.Tensor,
        input_roles: torch.Tensor,
        grade: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a route using precomputed board embeddings."""
        embeddings = self._input_embeddings(input_actions, input_nodes, input_roles, grade, node_context)

        sequence_length = input_actions.shape[1]
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=input_actions.device),
            diagonal=1,
        )
        padding_mask = input_actions.eq(ACTION_PAD)
        hidden = self.decoder(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.decoder_norm(hidden)

        action_logits = self.action_head(hidden)
        queries = self.pointer_query(hidden)
        keys = self.pointer_key(node_context)
        pointer_logits = torch.einsum("btd,nd->btn", queries, keys) / math.sqrt(self.hidden_dim)
        pointer_logits = pointer_logits + self.pointer_bias(node_context).view(1, 1, -1)
        return action_logits, pointer_logits, hidden

    def role_logits(
        self,
        hidden: torch.Tensor,
        node_indices: torch.Tensor,
        node_context: torch.Tensor,
    ) -> torch.Tensor:
        return self.role_head(torch.cat([hidden, node_context[node_indices]], dim=-1))


def mask_previously_selected_nodes(
    pointer_logits: torch.Tensor,
    input_actions: torch.Tensor,
    input_nodes: torch.Tensor,
) -> torch.Tensor:
    num_nodes = pointer_logits.shape[-1]
    safe_nodes = input_nodes.clamp(min=0, max=num_nodes - 1)
    selected = F.one_hot(safe_nodes, num_classes=num_nodes).bool()
    selected = selected & input_actions.eq(ACTION_SELECT)[..., None]
    previously_used = selected.cumsum(dim=1).gt(0)
    return pointer_logits.masked_fill(previously_used, torch.finfo(pointer_logits.dtype).min)


def hierarchical_graph_transformer_loss(
    model: HierarchicalGraphTransformer,
    action_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    hidden: torch.Tensor,
    node_context: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    action_weight: float = 1.0,
    pointer_weight: float = 1.0,
    role_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    action_class_weights = action_logits.new_tensor([1.0, 1.5, 2.0])
    role_class_weights = action_logits.new_tensor([2.0, 2.0, 1.0, 1.0])

    action_loss = F.cross_entropy(
        action_logits.reshape(-1, NUM_OUTPUT_ACTIONS),
        batch["target_actions"].reshape(-1),
        weight=action_class_weights,
        ignore_index=IGNORE_INDEX,
    )

    pointer_logits = mask_previously_selected_nodes(
        pointer_logits,
        batch["input_actions"],
        batch["input_nodes"],
    )
    select_mask = batch["target_actions"].eq(ACTION_SELECT)
    if select_mask.any():
        pointer_loss = F.cross_entropy(pointer_logits[select_mask], batch["target_nodes"][select_mask])
        selected_role_logits = model.role_logits(
            hidden[select_mask],
            batch["target_nodes"][select_mask],
            node_context,
        )
        role_loss = F.cross_entropy(
            selected_role_logits,
            batch["target_roles"][select_mask],
            weight=role_class_weights,
        )
        node_accuracy = pointer_logits[select_mask].argmax(dim=-1).eq(batch["target_nodes"][select_mask]).float().mean()
        role_accuracy = selected_role_logits.argmax(dim=-1).eq(batch["target_roles"][select_mask]).float().mean()
    else:
        pointer_loss = action_loss.new_zeros(())
        role_loss = action_loss.new_zeros(())
        node_accuracy = action_loss.new_zeros(())
        role_accuracy = action_loss.new_zeros(())

    valid_actions = batch["target_actions"].ne(IGNORE_INDEX)
    action_accuracy = action_logits.argmax(dim=-1)[valid_actions].eq(batch["target_actions"][valid_actions]).float().mean()
    total = action_weight * action_loss + pointer_weight * pointer_loss + role_weight * role_loss
    metrics = {
        "loss": total.detach(),
        "action_loss": action_loss.detach(),
        "pointer_loss": pointer_loss.detach(),
        "role_loss": role_loss.detach(),
        "action_accuracy": action_accuracy.detach(),
        "node_accuracy": node_accuracy.detach(),
        "role_accuracy": role_accuracy.detach(),
    }
    return total, metrics
