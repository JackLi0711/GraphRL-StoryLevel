"""
Experience Buffer for Policy Gradient methods

Author: Claude Code
Date: 2025-10-14
"""

import torch
from typing import List, Dict


class ExperienceBuffer:
    """
    儲存 episode(s) 的經驗

    用於 A2C (單 episode) 和 PPO (多 episodes) 的 trajectory 收集
    """
    def __init__(self):
        # Episode-level data (list of episodes)
        self.episodes = []

        # Current episode buffer
        self.states = []           # List of [N_i, state_dim] tensors
        self.global_states = []    # List of [1, state_dim] tensors (for value)
        self.actions = []          # List of integers
        self.rewards = []          # List of floats
        self.log_probs = []        # List of tensors
        self.values = []           # List of tensors
        self.entropies = []        # List of tensors
        self.valid_masks = []      # List of [N_i] bool tensors

    def add_step(self, state, global_state, action, reward, log_prob, value, entropy, valid_mask):
        """
        添加一個 step 到當前 episode

        Args:
            state: [N, state_dim] tensor
            global_state: [1, state_dim] tensor
            action: int
            reward: float
            log_prob: tensor
            value: tensor
            entropy: tensor
            valid_mask: [N] bool tensor
        """
        self.states.append(state)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.valid_masks.append(valid_mask)

    def finish_episode(self):
        """完成當前 episode，儲存到 episodes 列表"""
        episode_data = {
            'states': self.states,
            'global_states': self.global_states,
            'actions': self.actions,
            'rewards': self.rewards,
            'log_probs': self.log_probs,
            'values': self.values,
            'entropies': self.entropies,
            'valid_masks': self.valid_masks
        }
        self.episodes.append(episode_data)

        # Reset current episode buffer
        self.states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.valid_masks = []

    def get_all_episodes(self):
        """獲取所有收集的 episodes"""
        return self.episodes

    def clear(self):
        """清空所有 buffer"""
        self.episodes = []
        self.states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.valid_masks = []

    def __len__(self):
        """返回收集的 episodes 數量"""
        return len(self.episodes)
