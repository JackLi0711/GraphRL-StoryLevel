"""
Policy and Value Networks for Actor-Critic methods

Author: Claude Code
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Story-level policy network (Actor)

    輸入: story_features [N, state_dim]
        - N: 當前建築的 story member 數量 (動態: 16, 20, 24, 28...)
        - state_dim: StateGNN 輸出維度 (hidden_dim * 2)

    輸出: action_scores [N]
        - 每個 story member 的分數
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 單一分數輸出
        )

    def forward(self, story_features):
        """
        Args:
            story_features: [N, state_dim]

        Returns:
            action_scores: [N]
        """
        scores = self.policy_head(story_features)  # [N, 1]
        return scores.squeeze(-1)  # [N]


class ValueNetwork(nn.Module):
    """
    State value network (Critic) - V(s)

    輸入: global_features [1, state_dim]
        - 整個建築的全局特徵 (mean pooling from story_features)
        - state_dim: StateGNN 輸出後面一半 (hidden_dim * 1)

    輸出: value [1]
        - 當前狀態的價值估計 V(s)
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_features):
        """
        Args:
            global_features: [1, state_dim] or [batch, state_dim]

        Returns:
            value: [1, 1] or [batch, 1]
        """
        return self.value_head(global_features)  # [1, 1] or [batch, 1]
