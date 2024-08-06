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

        # combine story_embedding and graph_embedding by batch
        state = torch.zeros(story_num, hidden_dim*2).to(story_embedding.device)
        state[:, :hidden_dim] = story_embedding
        for b in range(len(structure_story_ptr) - 1):
            state[structure_story_ptr[b]:structure_story_ptr[b+1], hidden_dim:hidden_dim*2] = graph_embedding[b]
        
        return state


    def forward(self, x, edge_index, edge_attr, batch, story_batch, structure_story_ptr):
        # node embedding
        x = self.encoder_mlp(x)
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = F.relu(x)
        node_embedding = self.decoder_mlp(x)  # shape: [total node_num, hidden_dim]

        # edge embedding --> concat node embedding in two ends and go through an MLP
        index_i, index_j = edge_index
        node_embedding_i, node_embedding_j = node_embedding[index_i], node_embedding[index_j]
        edge_input = torch.cat([node_embedding_i[::2], node_embedding_j[::2], edge_attr[::2]], dim=1)
        edge_embedding = self.edge_mlp(edge_input)  # shape: [total edge_num, hidden_dim]

        # graph embedding --> use edge embedding to create story-level and graph-level embedding
        batch = torch.zeros(edge_embedding.shape[0]).to(x.device).to(torch.int64) if batch is None else batch
        graph_embedding = global_add_pool(edge_embedding, batch)        # shape: [graph_num, hidden_dim]
        story_embedding = global_add_pool(edge_embedding, story_batch)  # shape: [total story_member_num, hidden_dim], story_member_num = story_num * 4 (x-beam, z-beam, out-col, in-col)

        # state embedding --> state for story k = story_embedding(k) + graph_embedding
        structure_story_ptr = [0, story_embedding.shape[0]] if structure_story_ptr is None else structure_story_ptr
        state = self._state_global_aggregation(story_embedding, graph_embedding, structure_story_ptr)  # shape: [total story_member_num, hidden_dim*2]

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
        edge_state = self.batch_norm(edge_state)  # shape: [total story_member_num, member_state_dim]
        q_value = self.q_network(edge_state)    # shape: [total story_member_num, q_value_dim]

        return q_value


def synchronize_q_networks(target_q_network: nn.Module, online_q_network: nn.Module):
    """In place, synchronization of target_q_network and online_q_network."""
    _ = target_q_network.load_state_dict(online_q_network.state_dict())


def soft_update_q_network_parameters(target_q_network: nn.Module, online_q_network: nn.Module, soft_update_alpha: float):
    """In-place, soft-update of target_q_network parameters with parameters from online_q_network."""
    for p1, p2 in zip(target_q_network.parameters(), online_q_network.parameters()):
        p1.data.copy_(soft_update_alpha * p2.data + (1 - soft_update_alpha) * p1.data)




### Japan's model ###
import numpy as np


INIT_MEAN = 0.0
INIT_STD = 0.05
USE_BIAS = False
# BATCH_SIZE = 32


class GraphEmbedding(nn.Module):
    def __init__(self, n_node_inputs, n_edge_inputs, n_feature_outputs, n_action_types, device):
        super(GraphEmbedding, self).__init__()
        # GNN
        self.l1_1 = torch.nn.Linear(n_edge_inputs, n_feature_outputs, bias=USE_BIAS)
        self.l1_2 = torch.nn.Linear(n_node_inputs, n_feature_outputs, bias=USE_BIAS)
        self.l1_3 = torch.nn.Linear(n_feature_outputs, n_feature_outputs, bias=USE_BIAS)
        #self.l1_4 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
        #self.l1_5 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)

        # Q-network
        self.l2_1 = torch.nn.Linear(n_feature_outputs*2, n_action_types, bias=USE_BIAS)
        #self.l2_2 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
        #self.l2_3 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)

        self.ActivationF = torch.nn.LeakyReLU(0.2)

        self.Initialize_weight()

        self.n_feature_outputs = n_feature_outputs
        self.device = device
        self.to(self.device)

        # if use_gpu:
        #     self.to('cuda')
        #     self.device = torch.device('cuda')
        # else:
        #     self.to('cpu')
        #     self.device = torch.device('cpu')
    
    def Initialize_weight(self):
        for m in self._modules.values():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)

    def Connectivity(self, connectivity, n_nodes):
        n_edges = connectivity.shape[0]  # shape: [n_edges, 2], 2: node1_index, node2_index
        adjacency = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=self.device, requires_grad=False)
        adjacency[connectivity[:,0], connectivity[:,1]] = 1
        adjacency[connectivity[:,1], connectivity[:,0]] = 1

        order = np.arange(n_edges)
        incidence = torch.zeros(n_nodes, n_edges, dtype=torch.float32, device=self.device, requires_grad=False)
        incidence[connectivity[:,0], order] = -1  # (i, j): member j leaves node i
        incidence[connectivity[:,1], order] = 1   # (i, j): member j enters node i

        incidence_A = torch.abs(incidence)#.to_sparse()
        incidence_1 = (incidence == -1).type(torch.float32)
        incidence_2 = (incidence == 1).type(torch.float32)

        return incidence_A, incidence_1, incidence_2, adjacency

    def mu(self, v, mu, w, incidence_A, incidence_1, incidence_2, adjacency, mu_iter):
        '''
        - v [n_nodes, n_node_features]
        - mu [n_edges, n_edge_out_features]
        - w [n_edges, n_edge_in_features]
        '''
        h1 = self.ActivationF(self.l1_1.forward(w))
        h2_0 = self.ActivationF(self.l1_2.forward(v))
        h2 = torch.mm(incidence_A.T, h2_0)  # 矩陣相乘
        if mu_iter == 0:
            mu = h1 + h2
        else:
            h3_0 = torch.mm(incidence_A, mu)
            n_connect_edges_1 = torch.sum(torch.mm(adjacency,incidence_1),axis=0).repeat(self.n_feature_outputs,1).T
            n_connect_edges_2 = torch.sum(torch.mm(adjacency,incidence_2),axis=0).repeat(self.n_feature_outputs,1).T
            h3_1 = self.ActivationF(self.l1_3.forward((torch.mm(incidence_1.T,h3_0)-mu)/n_connect_edges_1))
            h3_2 = self.ActivationF(self.l1_3.forward((torch.mm(incidence_2.T,h3_0)-mu)/n_connect_edges_2))
            mu = h1 + h2 + h3_1 + h3_2  # shape: [n_edges, n_edge_out_features]

        return mu
        
    # def Q(self, mu, n_edges):
    #     if type(n_edges) is int: # normal operation
    #         mu_sum = torch.sum(mu, axis=0)
    #         mu_sum = mu_sum.repeat(n_edges, 1)
    #     else: # for mini-batch training
    #         mu_sum = torch.zeros((n_edges[-1],self.n_feature_outputs), dtype=torch.float32, device=self.device)
    #         for i in range(BATCH_SIZE):
    #             mu_sum[n_edges[i]:n_edges[i+1],:] = torch.sum(mu[n_edges[i]:n_edges[i+1],:], axis=0)

    #     Q = self.l2_1(torch.cat((mu_sum,mu),1))
    #     return Q

    def forward(self, x, edge_index, edge_attr, batch, story_batch, structure_story_ptr):
        '''
        - graph.x [node_num, node_feature_num] --> v [n_nodes, n_node_in_features]
        - graph.edge_attr [edge_num * 2, edge_attr_num] --> w [n_edges, n_edge_in_features]
        - graph.edge_index [2, edge_num * 2] --> connectivity [n_edges, 2]
        - member_batch [edge_num]: not used
        - story_batch [edge_num]: used
        - structure_story_ptr [graph_num + 1], e.g., [0, story_member_num1, story_member_num1+story_member_num2, story_member_num1+story_member_num2+story_member_num3, ..., total story_member_num]
        '''
        v = x
        w = edge_attr[::2, :]
        #edge_ptr = [0, w.shape[0]] if edge_ptr is None else edge_ptr
        #connectivity = torch.cat([edge_index[0, edge_ptr[b]*2:edge_ptr[b+1]*2].reshape(-1, 2) for b in range(len(edge_ptr)-1)], dim=0)
        connectivity = edge_index[0].reshape(-1, 2)  # the result is the same as the above line

        IA, I1, I2, D = self.Connectivity(connectivity, v.shape[0])

        if type(v) is np.ndarray: 
            v = torch.tensor(v, dtype=torch.float32, device=self.device, requires_grad=False)
        if type(w) is np.ndarray:
            w = torch.tensor(w, dtype=torch.float32, device=self.device, requires_grad=False)
        mu = torch.zeros((connectivity.shape[0],self.n_feature_outputs), device=self.device)  # shape: [total n_edges, n_edge_out_features]
        
        n_mu_iter = 3
        for i in range(n_mu_iter):
            mu = self.mu(v, mu, w, IA, I1, I2, D, mu_iter=i)  # shape: [total n_edges, n_edge_out_features]
            # print("iter {0}: {1}".format(i,mu.norm(p=2)))

        story_mu = global_add_pool(mu, story_batch)  # shape: [total n_story_members, n_edge_out_features]

        # if nm_batch is None:
        #     Q = self.Q(mu, w.shape[0])
        # else:
        #     Q = self.Q(mu, nm_batch)

        # mu_sum = torch.zeros((w.shape[0],self.n_feature_outputs), dtype=torch.float32, device=self.device)
        # for b in range(len(edge_ptr)-1):
        #     mu_sum[edge_ptr[b]:edge_ptr[b+1], :] = torch.sum(mu[edge_ptr[b]:edge_ptr[b+1], :], axis=0)
        # state = torch.cat((mu_sum,mu), 1)  # shape: [total n_edges, n_edge_out_features*2]

        structure_story_ptr = [0, story_mu.shape[0]] if structure_story_ptr is None else structure_story_ptr

        story_mu_sum = torch.zeros((story_mu.shape[0],self.n_feature_outputs), dtype=torch.float32, device=self.device)
        for b in range(len(structure_story_ptr)-1):
            story_mu_sum[structure_story_ptr[b]:structure_story_ptr[b+1], :] = torch.sum(story_mu[structure_story_ptr[b]:structure_story_ptr[b+1], :], axis=0)
        state = torch.cat((story_mu_sum,story_mu), 1)  # shape: [total n_story_members, n_edge_out_features*2]

        return state
    
    def get_Q(self, edge_state) -> torch.Tensor:
        q_value = self.l2_1(edge_state)  # shape: [total n_story_members, n_action_types=2]
        return q_value[:, 0]  # action_type = 0: dec, 1: inc