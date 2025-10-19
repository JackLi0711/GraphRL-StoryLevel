"""
Base Policy Gradient Agent

Author: Claude Code
Date: 2025-10-14
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import PolicyNetwork, ValueNetwork
from .buffer import ExperienceBuffer


class BasePGAgent:
    """
    Policy Gradient 方法的基礎類別

    包含共用功能：
    - Feature extraction (StateGNN) - 一起訓練
    - Action selection with masking
    - Advantage computation
    - Entropy annealing
    """
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 gamma: float,
                 entropy_coef_initial: float,
                 entropy_decay: float,
                 accumulate_episodes: int,
                 device: str,
                 logger = None):

        self.device = device
        self.logger = logger

        # StateGNN for feature extraction (需要訓練)
        from RL.model import StateGNN
        self.state_gnn = StateGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            member_state_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)

        # Policy and Value networks
        state_dim = hidden_dim * 2
        self.policy_net = PolicyNetwork(state_dim, hidden_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        # Training parameters
        self.gamma = gamma
        self.entropy_coef_initial = entropy_coef_initial
        self.entropy_decay = entropy_decay
        self.accumulate_episodes = accumulate_episodes

        # Experience buffer
        self.buffer = ExperienceBuffer()

        # Episode counter
        self._number_episodes = 0

    def get_entropy_coef(self):
        """計算當前的 entropy coefficient (exponential annealing)"""
        return self.entropy_coef_initial * (self.entropy_decay ** self._number_episodes)

    def get_features(self, graph, structure):
        """
        從 graph 提取特徵 (no grad for action selection)

        輸入:
            - graph: structure.graph (PyG Data object)
            - structure: Structure object

        輸出:
            - story_features: [N, hidden_dim*2] N個story member的特徵
            - global_features: [1, hidden_dim*2] 全局特徵 (mean pooling)
        """
        with torch.no_grad():  # Action selection 時不需要梯度
            graph = graph.to(self.device)
            story_features = self.state_gnn(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                None,  # batch
                structure.aux["story_batch"].to(self.device),
                None   # structure_story_ptr
            )  # [N, hidden_dim*2]

            # Global feature: mean pooling
            global_features = story_features.mean(dim=0, keepdim=True)  # [1, hidden_dim*2]

        return story_features, global_features

    def choose_action(self, story_features, global_features, structure, greedy=False):
        """
        選擇 action (處理動態 action size 和 masking)

        輸入:
            - story_features: [N, state_dim] N個story member的特徵
            - global_features: [1, state_dim] 全局特徵
            - structure: Structure object (用於獲取 invalid actions)
            - greedy: bool, 是否貪婪選擇

        輸出:
            - action: int, 選擇的 action index
            - log_prob: Tensor, log probability
            - value: Tensor, state value V(s)
            - entropy: Tensor, policy entropy
        """
        N = story_features.shape[0]

        # Get action scores from policy network
        action_scores = self.policy_net(story_features)  # [N]

        # Get state value
        value = self.value_net(global_features)  # [1, 1]

        # Create valid actions mask
        valid_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        invalid_indices = structure.already_minimum_section_story_indexes
        if len(invalid_indices) > 0:
            valid_mask[invalid_indices] = False

        # Mask invalid actions
        masked_scores = action_scores.masked_fill(~valid_mask, -1e9)

        # Compute probabilities
        probs = F.softmax(masked_scores, dim=0)  # [N]
        dist = Categorical(probs)

        # Sample or greedy
        if greedy:
            action = probs.argmax()
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Entropy for exploration bonus
        entropy = dist.entropy()

        return action.item(), log_prob, value, entropy

    def compute_returns(self, rewards, gamma):
        """
        計算 Monte Carlo returns (with normalization)

        輸入:
            - rewards: List[float], length T
            - gamma: discount factor

        輸出:
            - returns: Tensor [T], normalized
        """
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        R = 0

        # Reverse iterate
        for t in reversed(range(T)):
            R = rewards[t] + gamma * R
            returns[t] = R

        # Normalize returns to stabilize value learning
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def compute_advantages(self, returns, values):
        """
        計算 advantages (normalized)

        輸入:
            - returns: Tensor [T]
            - values: List[Tensor], length T

        輸出:
            - advantages: Tensor [T]
        """
        values_tensor = torch.cat(values).squeeze()  # [T]
        advantages = returns - values_tensor

        # Normalize for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
