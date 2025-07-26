import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import global_mean_pool, global_add_pool


# MuZero Value Transform Functions
EPS = 0.001

def scalar_to_support(x):
    """
    MuZero 的 scalar to support transform
    將任意大小的 scalar 值轉換為較穩定的支撐表示
    
    Args:
        x: tensor of scalars (reward/value)
    Returns:
        transformed tensor
    """
    s = torch.sign(x)
    y = torch.sqrt(torch.abs(x) + 1) - 1
    return s * (y + EPS * x)


def support_to_scalar(z):
    """
    MuZero 的 support to scalar transform (inverse transform)
    將支撐表示轉換回原始 scalar 值
    
    Args:
        z: tensor of transformed values
    Returns:
        original scalar tensor
    """
    s = torch.sign(z)
    return s * (((torch.sqrt(1 + 4 * EPS * (torch.abs(z) + 1 + EPS)) - 1) / (2 * EPS))**2 - 1)


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
            #nn.BatchNorm1d(hidden_dim)
        )

        # edge embedding
        edge_embedding_input_dim = hidden_dim * 2 + edge_feature_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_embedding_input_dim, member_state_dim),
            nn.ReLU(),
            #nn.BatchNorm1d(member_state_dim)
        )

        # self._initialize_weight()
    
    def _initialize_weight(self):
        for name, m in self.encoder_mlp.named_children():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)
        for name, m in self.decoder_mlp.named_children():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)
        for name, m in self.edge_mlp.named_children():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)

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

    def forward(self, x, edge_index, edge_attr, batch, story_batch, structure_story_ptr) -> torch.Tensor:
        t_start = time.time()
        # 確保所有輸入都在同一個設備上
        device = x.device
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        if batch is not None:
            batch = batch.to(device)
        if story_batch is not None:
            story_batch = story_batch.to(device)
        if structure_story_ptr is not None:
            structure_story_ptr = structure_story_ptr.to(device)
        
        # node embedding
        x = self.encoder_mlp(x)
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = F.relu(x)
        node_embedding = self.decoder_mlp(x)  # shape: [total node_num, hidden_dim]

        # edge embedding --> concat node embedding in two ends and go through an MLP
        index_i, index_j = edge_index
        node_embedding_i, node_embedding_j = node_embedding[index_i], node_embedding[index_j]
        
        # 確保維度匹配
        edge_attr_half = edge_attr[::2]  # 只取一半的邊特徵
        node_embedding_i_half = node_embedding_i[::2]  # 只取一半的節點嵌入
        node_embedding_j_half = node_embedding_j[::2]  # 只取一半的節點嵌入
        
        # 確保所有張量都有相同的批次大小
        min_size = min(node_embedding_i_half.size(0), node_embedding_j_half.size(0), edge_attr_half.size(0))
        node_embedding_i_half = node_embedding_i_half[:min_size]
        node_embedding_j_half = node_embedding_j_half[:min_size]
        edge_attr_half = edge_attr_half[:min_size]
        
        edge_input = torch.cat([node_embedding_i_half, node_embedding_j_half, edge_attr_half], dim=1)  # shape: [total edge_num, hidden_dim] + [total edge_num, hidden_dim] + [total edge_num, edge_feature_dim] = [total edge_num, hidden_dim*2 + edge_feature_dim]
        edge_embedding = self.edge_mlp(edge_input)  # shape: [total edge_num, member_state_dim]

        # graph embedding --> use edge embedding to create story-level and graph-level embedding
        batch = torch.zeros(edge_embedding.shape[0], device=device, dtype=torch.int64) if batch is None else batch
        graph_embedding = global_add_pool(edge_embedding, batch)        # shape: [graph_num, member_state_dim]
        story_embedding = global_add_pool(edge_embedding, story_batch)  # shape: [total story_member_num, member_state_dim], story_member_num = story_num * 4 (x-beam, z-beam, out-col, in-col)

        # state embedding --> state for story k = story_embedding(k) + graph_embedding
        structure_story_ptr = [0, story_embedding.shape[0]] if structure_story_ptr is None else structure_story_ptr
        state = self._state_global_aggregation(story_embedding, graph_embedding, structure_story_ptr)  # shape: [total story_member_num, member_state_dim*2]
        t_end = time.time()
        print(f"\tused time StateGNN.forward(): {t_end - t_start:.3f} sec")
        return state


class Q_Network(nn.Module):
    def __init__(self, member_state_dim, hidden_dim, q_value_dim):
        super().__init__()
        # self.batch_norm = nn.BatchNorm1d(member_state_dim)
        # self.q_network = nn.Sequential(
        #     nn.Linear(member_state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, q_value_dim),
        # )
        self.l2_1 = nn.Linear(member_state_dim, q_value_dim, bias=False)
        # self._initialize_weight()

    def _initialize_weight(self):
        for name, m in self._modules.items():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)

    def forward(self, edge_state) -> torch.Tensor:
        # edge_state = self.batch_norm(edge_state)  # shape: [total story_member_num, member_state_dim]
        # q_value = self.q_network(edge_state)  # shape: [total story_member_num, q_value_dim]

        q_value = self.l2_1(edge_state)  # shape: [total story_member_num, q_value_dim]

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

        self._initialize_weight()

        self.n_feature_outputs = n_feature_outputs
        self.device = device
        self.to(self.device)

        # if use_gpu:
        #     self.to('cuda')
        #     self.device = torch.device('cuda')
        # else:
        #     self.to('cpu')
        #     self.device = torch.device('cpu')
    

    def _initialize_weight(self):
        for m in self._modules.values():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)


    def _connectivity(self, connectivity, n_nodes):
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


    def _mu(self, v, mu, w, incidence_A, incidence_1, incidence_2, adjacency, mu_iter):
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


    def forward(self, x, edge_index, edge_attr, batch, story_batch, structure_story_ptr) -> torch.Tensor:
        '''
        - graph.x [node_num, node_feature_num] --> v [n_nodes, n_node_in_features]
        - graph.edge_attr [edge_num * 2, edge_attr_num] --> w [n_edges, n_edge_in_features]
        - graph.edge_index [2, edge_num * 2] --> connectivity [n_edges, 2]
        - member_batch [edge_num]: not used
        - story_batch [edge_num]: used
        - structure_story_ptr [graph_num + 1], e.g., [0, story_member_num1, story_member_num1+story_member_num2, story_member_num1+story_member_num2+story_member_num3, ..., total story_member_num]
        '''
        # 確保所有輸入都在同一個設備上
        device = self.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        if story_batch is not None:
            story_batch = story_batch.to(device)
        if structure_story_ptr is not None:
            structure_story_ptr = structure_story_ptr.to(device)
        
        v = x
        w = edge_attr[::2, :]
        #edge_ptr = [0, w.shape[0]] if edge_ptr is None else edge_ptr
        #connectivity = torch.cat([edge_index[0, edge_ptr[b]*2:edge_ptr[b+1]*2].reshape(-1, 2) for b in range(len(edge_ptr)-1)], dim=0)
        connectivity = edge_index[0].reshape(-1, 2)  # the result is the same as the above line

        IA, I1, I2, D = self._connectivity(connectivity, v.shape[0])

        if type(v) is np.ndarray: 
            v = torch.tensor(v, dtype=torch.float32, device=self.device, requires_grad=False)
        if type(w) is np.ndarray:
            w = torch.tensor(w, dtype=torch.float32, device=self.device, requires_grad=False)
        mu = torch.zeros((connectivity.shape[0],self.n_feature_outputs), device=self.device)  # shape: [total n_edges, n_edge_out_features]
        
        n_mu_iter = 3
        for i in range(n_mu_iter):
            mu = self._mu(v, mu, w, IA, I1, I2, D, mu_iter=i)  # shape: [total n_edges, n_edge_out_features]
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
        
        # 確保兩個張量都在同一個設備上
        story_mu_sum = story_mu_sum.to(device)
        story_mu = story_mu.to(device)
        
        state = torch.cat((story_mu_sum,story_mu), 1)  # shape: [total n_story_members, n_edge_out_features*2]

        return state


    def get_Q(self, edge_state) -> torch.Tensor:

        q_value = self.l2_1(edge_state)  # shape: [total n_story_members, n_action_types=2]
        
        return q_value[:, 0]  # action_type = 0: dec, 1: inc


### MuZero Implementation ###

class MuZeroNetwork(nn.Module):
    """
    MuZero 的核心神經網路，包含三個子網路：
    - Representation Network (h): 將觀察編碼成隱藏狀態
    - Dynamics Network (g): 在隱藏狀態空間中進行推演
    - Prediction Network (f): 預測策略和價值
    支援動態動作空間
    """
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, max_num_actions=32, 
                 num_layers=3, representation_network_type="Taiwan"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_num_actions = max_num_actions  # 最大動作數量
        self.num_layers = num_layers
        
        # Representation Network (h) - 使用與現有 DQN 相同的架構
        if representation_network_type == "Taiwan":
            # 使用 StateGNN 作為 representation network
            self.representation_network = StateGNN(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                member_state_dim=hidden_dim,  # 讓輸出維度等於 hidden_dim
                num_layers=num_layers
            )
        elif representation_network_type == "Japan":
            # 使用 GraphEmbedding 作為 representation network
            self.representation_network = GraphEmbedding(
                n_node_inputs=node_feature_dim,
                n_edge_inputs=edge_feature_dim,
                n_feature_outputs=hidden_dim // 2,  # GraphEmbedding 輸出 hidden_dim
                n_action_types=max_num_actions,  # 使用最大動作數量
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            # 簡單的 MLP 作為備用
            self.representation_network = nn.Sequential(
                nn.Linear(node_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        self.representation_network_type = representation_network_type
        
        # Prediction Network (f) - 從隱藏狀態預測策略和價值
        self.prediction_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim // 4, max_num_actions)  # 使用最大動作數量
        self.value_head = nn.Linear(hidden_dim // 4, 1)
        
        # Dynamics Network (g) - 在隱藏狀態空間中推演
        # 輸入：hidden_state + action_one_hot
        self.dynamics_network = nn.Sequential(
            nn.Linear(hidden_dim + max_num_actions, hidden_dim // 2),  # 使用最大動作數量
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        self.reward_head = nn.Linear(hidden_dim // 2, 1)
        self.next_state_head = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # 正規化層
        self.state_norm = nn.LayerNorm(hidden_dim)
        
    def represent(self, observation):
        """ 
        Representation Network: h(o) -> s_0 
        將環境觀察編碼成隱藏狀態
        """
        if self.representation_network_type == "Taiwan":
            # StateGNN 需要完整的圖數據
            if hasattr(observation, 'x'):  # GraphData object
                hidden_state = self.representation_network(
                    observation.x, 
                    observation.edge_index, 
                    observation.edge_attr, 
                    observation.batch, 
                    observation.story_batch, 
                    observation.structure_story_ptr
                )
                # StateGNN 輸出的是 [story_num, hidden_dim*2]，我們取前 hidden_dim
                if hidden_state.dim() > 1 and hidden_state.size(-1) > self.hidden_dim:
                    hidden_state = hidden_state[:, :self.hidden_dim]
            else:
                raise ValueError("Taiwan representation network requires GraphData object")
                
        elif self.representation_network_type == "Japan":
            # GraphEmbedding 的處理方式
            if hasattr(observation, 'x'):
                hidden_state = self.representation_network(
                    observation.x, 
                    observation.edge_index, 
                    observation.edge_attr, 
                    observation.batch, 
                    observation.story_batch, 
                    observation.structure_story_ptr
                )
                # GraphEmbedding 輸出的是 [story_num, hidden_dim]
            else:
                raise ValueError("Japan representation network requires GraphData object")
        else:
            # 簡單的處理方式
            if hasattr(observation, 'x'):
                hidden_state = self.representation_network(observation.x.mean(dim=0, keepdim=True))
            else:
                hidden_state = self.representation_network(observation)
        
        # 確保輸出維度正確
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
            
        # 取第一個樣本（batch中的第一個）作為代表
        if hidden_state.size(0) > 1:
            hidden_state = hidden_state[0:1]  # [1, hidden_dim]
            
        # 正規化隱藏狀態
        hidden_state = self.state_norm(hidden_state)
        return hidden_state
    
    def predict(self, hidden_state, num_actions=None):
        """ 
        Prediction Network: f(s) -> p, v 
        從隱藏狀態預測策略和價值
        支援動態動作空間
        """
        x = self.prediction_network(hidden_state)
        policy_logits = self.policy_head(x)
        
        # 如果提供了當前動作數量，只取前 num_actions 個輸出
        if num_actions is not None and num_actions < self.max_num_actions:
            policy_logits = policy_logits[:, :num_actions]
        
        # 網路輸出原始value，然後應用MuZero transform
        raw_value = self.value_head(x)
        value = scalar_to_support(raw_value)
        return policy_logits, value
    
    def dynamics(self, hidden_state, action, num_actions=None):
        """ 
        Dynamics Network: g(s, a) -> r, s' 
        在隱藏狀態空間中推演下一步
        支援動態動作空間
        """
        # 確保 action 在有效範圍內
        if num_actions is not None:
            action = torch.clamp(action, 0, num_actions - 1)
        
        # 將 action 轉為 one-hot vector
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 2 and action.size(0) == 1 and action.size(1) == 1:
            action = action.squeeze()
        
        # 確保 action 在正確的設備上
        device = hidden_state.device
        action = action.to(device)
        
        # 確保 action 是標量或單個值
        if action.dim() == 0:
            action = action.unsqueeze(0)  # 變成 [1]
        
        # 如果提供了當前動作數量，使用當前大小的 one-hot
        if num_actions is not None and num_actions < self.max_num_actions:
            action_one_hot = F.one_hot(action, num_classes=num_actions).float()
            # 填充到最大大小
            padded_one_hot = torch.zeros(action_one_hot.size(0), self.max_num_actions, device=device)
            padded_one_hot[:, :num_actions] = action_one_hot
            action_one_hot = padded_one_hot
        else:
            action_one_hot = F.one_hot(action, num_classes=self.max_num_actions).float()
        
        # 確保 action_one_hot 在正確的設備上
        action_one_hot = action_one_hot.to(device)
        
        # 確保 action_one_hot 的批次維度與 hidden_state 匹配
        if action_one_hot.size(0) != hidden_state.size(0):
            # 如果批次大小不匹配，取第一個樣本
            action_one_hot = action_one_hot[0:1]  # 只取第一個樣本
            
        # 拼接 hidden_state 和 action_one_hot
        stacked_input = torch.cat([hidden_state, action_one_hot], dim=-1)
        
        x = self.dynamics_network(stacked_input)
        # 網路輸出原始reward，然後應用MuZero transform
        raw_reward = self.reward_head(x)
        reward = scalar_to_support(raw_reward)
        next_hidden_state = self.next_state_head(x)
        
        # 正規化下一個隱藏狀態
        next_hidden_state = self.state_norm(next_hidden_state)
        
        return reward, next_hidden_state
