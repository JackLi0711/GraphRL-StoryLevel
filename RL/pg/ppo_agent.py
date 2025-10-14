"""
PPO (Proximal Policy Optimization) Agent

Author: Claude Code
Date: 2025-10-14
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import BasePGAgent


class PPOAgent(BasePGAgent):
    """
    Proximal Policy Optimization (PPO)

    特點:
    - Clipped surrogate objective
    - 收集多個 episodes 後更新 (accumulate_episodes=5)
    - Multiple epochs per update
    - Entropy bonus with exponential annealing
    - StateGNN + Policy + Value 一起訓練
    """
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef_initial: float = 0.01,
                 entropy_decay: float = 0.99,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 accumulate_episodes: int = 5,
                 device: str = "cuda",
                 logger = None):

        super().__init__(
            node_feature_dim, edge_feature_dim, hidden_dim, num_layers,
            gamma, entropy_coef_initial, entropy_decay,
            accumulate_episodes, device, logger
        )

        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # Single optimizer for all networks (一起訓練)
        self.optimizer = optim.Adam(
            list(self.state_gnn.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

    def update(self):
        """
        PPO update

        當收集到 accumulate_episodes 個 episodes 後調用
        進行 ppo_epochs 次更新

        Returns:
            - loss_dict: Dict with loss components
        """
        if len(self.buffer) < self.accumulate_episodes:
            return {}  # 尚未收集足夠的 episodes

        episodes = self.buffer.get_all_episodes()

        # Flatten all episodes
        all_states = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_valid_masks = []

        for episode in episodes:
            # Compute returns and advantages
            returns = self.compute_returns(episode['rewards'], self.gamma)
            advantages = self.compute_advantages(returns, episode['values'])

            # Store
            all_states.extend(episode['states'])
            all_global_states.extend(episode['global_states'])
            all_actions.extend(episode['actions'])
            all_old_log_probs.extend(episode['log_probs'])
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_valid_masks.extend(episode['valid_masks'])

        # Convert to tensors
        old_log_probs = torch.stack(all_old_log_probs).detach()
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)

        # Get current entropy coefficient
        entropy_coef = self.get_entropy_coef()

        # PPO epochs
        policy_losses = []
        value_losses = []
        entropies_list = []

        for epoch in range(self.ppo_epochs):
            # Re-compute with current policy (有梯度)
            new_log_probs = []
            new_values = []
            new_entropies = []

            for t in range(len(all_states)):
                state = all_states[t]
                global_state = all_global_states[t]
                action = all_actions[t]
                valid_mask = all_valid_masks[t]

                # Forward pass (有梯度)
                action_scores = self.policy_net(state)
                masked_scores = action_scores.masked_fill(~valid_mask, -1e9)
                probs = F.softmax(masked_scores, dim=0)
                dist = Categorical(probs)

                # New log prob and entropy
                new_log_prob = dist.log_prob(torch.tensor(action, device=self.device))
                new_log_probs.append(new_log_prob)
                new_entropies.append(dist.entropy())

                # New value
                new_value = self.value_net(global_state)
                new_values.append(new_value)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.cat(new_values).squeeze()
            new_entropies = torch.stack(new_entropies)

            # PPO clipped loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns)

            # Entropy loss
            entropy_loss = -new_entropies.mean()

            # Total loss
            total_loss = (policy_loss +
                         self.value_loss_coef * value_loss +
                         entropy_coef * entropy_loss)

            # Optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.state_gnn.parameters()) +
                list(self.policy_net.parameters()) +
                list(self.value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            # Record
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies_list.append(-entropy_loss.item())

        # Log
        if self.logger:
            self.logger.info(
                f"PPO Update (Episode {self._number_episodes}) - "
                f"Policy Loss: {np.mean(policy_losses):.4f}, "
                f"Value Loss: {np.mean(value_losses):.4f}, "
                f"Entropy: {np.mean(entropies_list):.4f}, "
                f"Entropy Coef: {entropy_coef:.6f}"
            )

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies_list),
            'entropy_coef': entropy_coef,
            'total_loss': np.mean(policy_losses) + np.mean(value_losses)
        }
