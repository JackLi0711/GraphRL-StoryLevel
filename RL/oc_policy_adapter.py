import torch
import torch.nn as nn
from typing import Optional

from RL import model as rl_model


class GraphStateAdapter(nn.Module):
    """
    Adapter that converts graph observations into a single fixed-length state vector
    for Option-Critic using StateGNN as the backbone.

    It aggregates per-action embeddings S ∈ R^{A×(2H)} into z ∈ R^{3H} using:
      - story_mean = mean(S[:, :H])
      - story_max  = max(S[:, :H])
      - graph_feat = mean(S[:, H:])
      - z = concat(story_mean, story_max, graph_feat)
    """

    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        self.gnn = rl_model.StateGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            member_state_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

    @torch.no_grad()
    def forward(self,
                graph_x: torch.Tensor,
                graph_edge_index: torch.Tensor,
                graph_edge_attr: torch.Tensor,
                story_batch: torch.Tensor,
                structure_story_ptr: Optional[list] = None) -> torch.Tensor:
        """
        Compute aggregated state vector z from graph inputs.

        Returns:
            z: Tensor of shape [3 * hidden_dim]
        """
        graph_x = graph_x.to(self.device)
        graph_edge_index = graph_edge_index.to(self.device)
        graph_edge_attr = graph_edge_attr.to(self.device)
        story_batch = story_batch.to(self.device)

        S = self.gnn.forward(
            x=graph_x,
            edge_index=graph_edge_index,
            edge_attr=graph_edge_attr,
            batch=None,
            story_batch=story_batch,
            structure_story_ptr=structure_story_ptr,
        )  # shape: [A, 2H]

        H = self.hidden_dim
        story_part = S[:, :H]
        graph_part = S[:, H:]

        story_mean = torch.mean(story_part, dim=0)
        story_max = torch.amax(story_part, dim=0)
        graph_feat = torch.mean(graph_part, dim=0)

        z = torch.cat([story_mean, story_max, graph_feat], dim=0)  # [3H]
        return z

    @staticmethod
    def num_actions(structure) -> int:
        return len(structure.story_level_actions)


