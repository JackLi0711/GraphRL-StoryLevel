import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import global_mean_pool, global_add_pool


class StateGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, member_state_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # node embedding
        self.encoder_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU()
        )

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(tgnn.GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_feature_dim))

        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # edge embedding
        edge_embedding_input_dim = hidden_dim * 2 + edge_feature_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_embedding_input_dim, member_state_dim),
            nn.ReLU(),
            nn.BatchNorm1d(member_state_dim)
        )


    def _state_global_aggregation(self, story_embedding, graph_embedding, structure_story_ptr):
        # shapes
        story_num = story_embedding.shape[0]
        hidden_dim = story_embedding.shape[1]
        bs = len(structure_story_ptr) - 1

        # combine story_embedding and graph_embedding by batch
        state = torch.zeros(story_num, hidden_dim*2).to(story_embedding.device)
        state[:, :hidden_dim] = story_embedding
        for b in range(bs):
            state[structure_story_ptr[b]:structure_story_ptr[b+1], hidden_dim:hidden_dim*2] = graph_embedding[b]
        return state


    def forward(self, x, edge_index, edge_attr, batch, story_batch, structure_story_ptr):
        # node embedding
        x = self.encoder_mlp(x)
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = F.relu(x)
        node_embedding = self.decoder_mlp(x)

        # edge embedding
        # concat node embedding in two ends and go through an MLP
        index_i, index_j = edge_index
        node_embedding_i, node_embedding_j = node_embedding[index_i], node_embedding[index_j]
        edge_input = torch.cat([node_embedding_i[::2], node_embedding_j[::2], edge_attr[::2]], dim=1)
        edge_embedding = self.edge_mlp(edge_input)

        # graph embedding
        # use edge embedding to create story-level and graph-level embedding
        batch = torch.zeros(edge_embedding.shape[0]).to(x.device).to(torch.int64) if batch is None else batch
        graph_embedding = global_add_pool(edge_embedding, batch)
        story_embedding = global_add_pool(edge_embedding, story_batch)

        # state embedding
        # state for story k = story_embedding(k) + graph_embedding
        structure_story_ptr = [0, story_embedding.shape[0]] if structure_story_ptr is None else structure_story_ptr
        state = self._state_global_aggregation(story_embedding, graph_embedding, structure_story_ptr)

        return state




class Q_Network(nn.Module):
    def __init__(self, member_state_dim, hidden_dim, q_value_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(member_state_dim)
        self.q_network = nn.Sequential(
            nn.Linear(member_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, q_value_dim),
        )

    def forward(self, edge_state):
        edge_state = self.batch_norm(edge_state)
        return self.q_network(edge_state)




def synchronize_q_networks(target_q_network: nn.Module, online_q_network: nn.Module):
    """In place, synchronization of target_q_network and online_q_network."""
    _ = target_q_network.load_state_dict(online_q_network.state_dict())


def soft_update_q_network_parameters(target_q_network: nn.Module, online_q_network: nn.Module, soft_update_alpha: float):
    """In-place, soft-update of target_q_network parameters with parameters from online_q_network."""
    for p1, p2 in zip(target_q_network.parameters(), online_q_network.parameters()):
        p1.data.copy_(soft_update_alpha * p2.data + (1 - soft_update_alpha) * p1.data)
