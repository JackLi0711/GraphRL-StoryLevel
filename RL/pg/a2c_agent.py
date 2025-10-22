"""
A2C (Advantage Actor-Critic) Agent

Author: Claude Code
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BasePGAgent


class A2CAgent(BasePGAgent):
    """
    Advantage Actor-Critic (A2C)

    特點:
    - On-policy learning
    - 每個 episode 更新一次 (accumulate_episodes=1)
    - Entropy bonus with exponential annealing
    - StateGNN + Policy + Value 一起訓練
    """
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 value_loss_coef: float = 0.5,
                 entropy_coef_initial: float = 0.01,
                 entropy_decay: float = 0.99,
                 max_grad_norm: float = 0.5,
                 accumulate_episodes: int = 1,
                 state_gnn_lr_multiplier: float = 100.0,
                 device: str = "cuda",
                 logger = None):

        super().__init__(
            node_feature_dim, edge_feature_dim, hidden_dim, num_layers,
            gamma, entropy_coef_initial, entropy_decay,
            accumulate_episodes, device, logger
        )

        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Separate optimizers with different learning rates
        # StateGNN gets higher LR to compensate for smaller gradients
        self.state_gnn_optimizer = optim.Adam(
            self.state_gnn.parameters(),
            lr=lr * state_gnn_lr_multiplier
        )
        self.policy_value_optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

        if self.logger:
            self.logger.info(f"[OPTIMIZER] StateGNN LR: {lr * state_gnn_lr_multiplier:.6f}, Policy/Value LR: {lr:.6f}")

    def update(self):
        """
        A2C update

        當收集到 accumulate_episodes 個 episodes 後調用

        Returns:
            - loss_dict: Dict with loss components
        """
        if len(self.buffer) < self.accumulate_episodes:
            return {}  # 尚未收集足夠的 episodes

        episodes = self.buffer.get_all_episodes()

        all_log_probs = []
        all_entropies = []
        all_values = []
        all_returns = []
        all_advantages = []

        # Process each episode
        for episode in episodes:
            # Compute returns
            returns = self.compute_returns(episode['rewards'], self.gamma)

            # Compute advantages
            advantages = self.compute_advantages(returns, episode['values'])

            # Collect
            all_log_probs.extend(episode['log_probs'])
            all_entropies.extend(episode['entropies'])
            all_values.extend(episode['values'])
            all_returns.append(returns)
            all_advantages.append(advantages)

        # Convert to tensors
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        values = torch.cat(all_values).squeeze()
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)

        # Get current entropy coefficient
        entropy_coef = self.get_entropy_coef()

        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        # Total loss
        total_loss = (policy_loss +
                     self.value_loss_coef * value_loss +
                     entropy_coef * entropy_loss)

        # Optimization step
        self.state_gnn_optimizer.zero_grad()
        self.policy_value_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients separately for each optimizer
        nn.utils.clip_grad_norm_(
            self.state_gnn.parameters(),
            self.max_grad_norm
        )
        nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            self.max_grad_norm
        )

        # Step both optimizers
        self.state_gnn_optimizer.step()
        self.policy_value_optimizer.step()

        # Log
        if self.logger:
            self.logger.info(
                f"A2C Update (Episode {self._number_episodes}) - "
                f"Policy Loss: {policy_loss.item():.4f}, "
                f"Value Loss: {value_loss.item():.4f}, "
                f"Entropy: {-entropy_loss.item():.4f}, "
                f"Entropy Coef: {entropy_coef:.6f}"
            )

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'entropy_coef': entropy_coef,
            'total_loss': total_loss.item()
        }
