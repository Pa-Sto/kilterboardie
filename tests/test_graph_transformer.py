import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from graph_transformer_data import (
    ACTION_EOS,
    ACTION_GROUP_END,
    ACTION_SELECT,
    ROLE_FINISH,
    ROLE_START,
    KilterGraphSequenceDataset,
    collate_graph_sequences,
)
from graph_transformer_model import HierarchicalGraphTransformer, hierarchical_graph_transformer_loss


class GraphTransformerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = KilterGraphSequenceDataset(
            "ImageData/50Degree/ExportClean",
            max_sequence_length=48,
        )

    def test_graph_uses_real_hold_centers(self) -> None:
        graph = self.dataset.board_graph
        self.assertEqual(graph.num_nodes, 476)
        self.assertEqual(tuple(graph.raw_xy.shape), (476, 2))
        self.assertGreater(graph.coordinate_spacing, 0.0)
        self.assertGreater(graph.edge_index.shape[1], graph.num_nodes)

    def test_pseudo_sequence_has_start_and_finish_groups(self) -> None:
        route = np.load(self.dataset.samples[0].npy_path).copy()
        start_row, start_col = np.argwhere(route[..., ROLE_START] > 0.5)[0]
        route[start_row, start_col, 3] = 1.0
        events = self.dataset.route_to_events(route)
        selected_roles = [role for action, _, role in events if action == ACTION_SELECT]
        selected_nodes = [node for action, node, _ in events if action == ACTION_SELECT]
        self.assertEqual(selected_roles[0], ROLE_START)
        self.assertEqual(selected_roles[-1], ROLE_FINISH)
        self.assertEqual(len(selected_nodes), len(set(selected_nodes)))
        self.assertIn(ACTION_GROUP_END, [action for action, _, _ in events])
        self.assertEqual(events[-1][0], ACTION_EOS)

    def test_forward_and_backward(self) -> None:
        loader = DataLoader(
            Subset(self.dataset, [0, 1]),
            batch_size=2,
            collate_fn=collate_graph_sequences,
        )
        batch = next(iter(loader))
        graph = self.dataset.board_graph
        model = HierarchicalGraphTransformer(
            num_grades=self.dataset.num_grades,
            node_feature_dim=graph.node_feature_dim,
            edge_feature_dim=graph.edge_feature_dim,
            hidden_dim=32,
            graph_layers=1,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=64,
            max_sequence_length=48,
        )
        outputs = model(
            graph,
            batch["input_actions"],
            batch["input_nodes"],
            batch["input_roles"],
            batch["grade"],
        )
        loss, _ = hierarchical_graph_transformer_loss(model, *outputs, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
