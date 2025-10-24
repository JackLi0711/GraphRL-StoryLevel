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
from torch_geometric.data import Batch

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
                 state_gnn_lr_multiplier: float = 100.0,
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

    def _batch_all_graphs(self, all_graphs, all_structures):
        """
        Batch所有graphs成一個大batch並準備相關數據

        Args:
            all_graphs: List of graph objects
            all_structures: List of structure data dicts

        Returns:
            batched_graph: Batched PyG graph
            batched_story_batch: Batched story_batch tensor
            structure_story_ptr: Pointer array for each structure's story range
            num_stories_per_graph: List of story counts for each graph
        """
        # 1. 使用PyG的Batch合併所有graphs
        data_list = [g.to(self.device) for g in all_graphs]
        batched_graph = Batch.from_data_list(data_list)

        # 2. 準備batched story_batch（需要調整offset）
        story_batch_list = [s['story_batch'] for s in all_structures]
        story_batch_offset = 0
        all_story_batches = []
        num_stories_per_graph = []

        for sb in story_batch_list:
            # 計算當前graph的story數量
            num_stories = sb.max().item() + 1
            num_stories_per_graph.append(num_stories)

            # 調整offset
            adjusted_sb = sb + story_batch_offset
            all_story_batches.append(adjusted_sb)

            # 更新offset
            story_batch_offset += num_stories

        batched_story_batch = torch.cat(all_story_batches).to(self.device)

        # 3. 創建structure_story_ptr
        structure_story_ptr = [0]
        for num_stories in num_stories_per_graph:
            structure_story_ptr.append(structure_story_ptr[-1] + num_stories)

        return batched_graph, batched_story_batch, structure_story_ptr, num_stories_per_graph

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

        # === DIAGNOSTIC: Save parameters before update ===
        state_gnn_params_before = [p.clone().detach() for p in self.state_gnn.parameters()]
        policy_params_before = [p.clone().detach() for p in self.policy_net.parameters()]
        value_params_before = [p.clone().detach() for p in self.value_net.parameters()]

        episodes = self.buffer.get_all_episodes()

        # Flatten all episodes
        all_states = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_valid_masks = []
        all_graphs = []           # Store graphs for recomputing features
        all_structures = []       # Store structures for recomputing features

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
            all_graphs.extend(episode['graphs'])
            all_structures.extend(episode['structures'])

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
        state_gnn_grads = []
        policy_grads = []
        value_grads = []

        for epoch in range(self.ppo_epochs):
            # === BATCH OPTIMIZATION: 一次forward所有graphs ===

            # 1. Batch所有graphs
            batched_graph, batched_story_batch, structure_story_ptr, num_stories_per_graph = \
                self._batch_all_graphs(all_graphs, all_structures)

            if self.logger and epoch == 0:
                self.logger.info(f"  [BATCH] Processing {len(all_graphs)} graphs in one forward pass")
                self.logger.info(f"  [BATCH] Total nodes: {batched_graph.x.shape[0]}, "
                               f"Total edges: {batched_graph.edge_index.shape[1]//2}")

            # 2. Create edge-level batch index from node-level batch index
            # batched_graph.batch is [num_nodes], we need [num_edges]
            # For each edge, get the batch index from its source node
            edge_batch = batched_graph.batch[batched_graph.edge_index[0]]  # [num_edges*2]
            # StateGNN uses edge_index[::2], so we also take edge_batch[::2]
            edge_batch_half = edge_batch[::2]  # [num_edges]

            # 3. 一次forward StateGNN處理所有graphs（關鍵：只forward一次！）
            all_story_features_batched = self.state_gnn(
                batched_graph.x,
                batched_graph.edge_index,
                batched_graph.edge_attr,
                edge_batch_half,  # Edge-level batch index
                batched_story_batch,
                structure_story_ptr
            )  # [total_story_members_all_timesteps, hidden_dim*2]

            # 4. 分割batched features回individual timesteps
            story_features_list = []
            global_features_list = []
            start_idx = 0

            for num_stories in num_stories_per_graph:
                end_idx = start_idx + num_stories

                # 提取當前timestep的features
                story_feat = all_story_features_batched[start_idx:end_idx]
                global_feat = story_feat.mean(dim=0, keepdim=True)

                story_features_list.append(story_feat)
                global_features_list.append(global_feat)

                start_idx = end_idx

            # 5. 使用cached features計算loss（這些features有梯度連接到StateGNN）
            new_log_probs = []
            new_values = []
            new_entropies = []

            for t in range(len(all_states)):
                story_features = story_features_list[t]  # 重用已計算的features（有梯度）
                global_features = global_features_list[t]
                action = all_actions[t]
                valid_mask = all_valid_masks[t]

                # Forward pass through policy and value networks
                action_scores = self.policy_net(story_features)
                masked_scores = action_scores.masked_fill(~valid_mask, -1e9)
                probs = F.softmax(masked_scores, dim=0)
                dist = Categorical(probs)

                # New log prob and entropy
                new_log_prob = dist.log_prob(torch.tensor(action, device=self.device))
                new_log_probs.append(new_log_prob)
                new_entropies.append(dist.entropy())

                # New value
                new_value = self.value_net(global_features)
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
            self.state_gnn_optimizer.zero_grad()
            self.policy_value_optimizer.zero_grad()
            total_loss.backward()

            # === DIAGNOSTIC: Check gradients ===
            if epoch == 0:  # Only check first epoch to avoid spam
                grad_norms = {}
                for name, param in [
                    ('state_gnn', list(self.state_gnn.parameters())[0]),
                    ('policy_net', list(self.policy_net.parameters())[0]),
                    ('value_net', list(self.value_net.parameters())[0])
                ]:
                    if param.grad is not None:
                        grad_norms[name] = param.grad.norm().item()
                    else:
                        grad_norms[name] = 0.0

                if self.logger:
                    self.logger.info(f"  [DIAGNOSTIC] Gradient norms - StateGNN: {grad_norms['state_gnn']:.6f}, "
                                   f"Policy: {grad_norms['policy_net']:.6f}, Value: {grad_norms['value_net']:.6f}")

                    # 額外診斷：計算梯度提升倍數
                    if grad_norms['state_gnn'] > 0:
                        ratio_to_policy = grad_norms['state_gnn'] / (grad_norms['policy_net'] + 1e-8)
                        self.logger.info(f"  [DIAGNOSTIC] StateGNN/Policy gradient ratio: {ratio_to_policy:.4f} "
                                       f"(should be ~0.3-1.0 after batching)")

            # Record gradient norms BEFORE clipping
            state_gnn_grad_norm = 0.0
            policy_grad_norm = 0.0
            value_grad_norm = 0.0

            for p in self.state_gnn.parameters():
                if p.grad is not None:
                    state_gnn_grad_norm += p.grad.norm().item() ** 2
            state_gnn_grad_norm = state_gnn_grad_norm ** 0.5

            for p in self.policy_net.parameters():
                if p.grad is not None:
                    policy_grad_norm += p.grad.norm().item() ** 2
            policy_grad_norm = policy_grad_norm ** 0.5

            for p in self.value_net.parameters():
                if p.grad is not None:
                    value_grad_norm += p.grad.norm().item() ** 2
            value_grad_norm = value_grad_norm ** 0.5

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

            # Record
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies_list.append(-entropy_loss.item())
            state_gnn_grads.append(state_gnn_grad_norm)
            policy_grads.append(policy_grad_norm)
            value_grads.append(value_grad_norm)

        # === DIAGNOSTIC: Check parameter changes ===
        state_gnn_params_after = [p.clone().detach() for p in self.state_gnn.parameters()]
        policy_params_after = [p.clone().detach() for p in self.policy_net.parameters()]
        value_params_after = [p.clone().detach() for p in self.value_net.parameters()]

        state_gnn_change = torch.norm(state_gnn_params_after[0] - state_gnn_params_before[0]).item()
        policy_change = torch.norm(policy_params_after[0] - policy_params_before[0]).item()
        value_change = torch.norm(value_params_after[0] - value_params_before[0]).item()

        # Log
        if self.logger:
            self.logger.info(
                f"PPO Update (Episode {self._number_episodes}) - "
                f"Policy Loss: {np.mean(policy_losses):.4f}, "
                f"Value Loss: {np.mean(value_losses):.4f}, "
                f"Entropy: {np.mean(entropies_list):.4f}, "
                f"Entropy Coef: {entropy_coef:.6f}"
            )
            self.logger.info(
                f"  [DIAGNOSTIC] Parameter changes - StateGNN: {state_gnn_change:.6f}, "
                f"Policy: {policy_change:.6f}, Value: {value_change:.6f}"
            )

            # 更詳細的診斷
            if state_gnn_change > 0:
                ratio = state_gnn_change / (policy_change + 1e-8)
                self.logger.info(f"  [DIAGNOSTIC] ✓ StateGNN is updating! Change ratio (StateGNN/Policy): {ratio:.4f}")
                if ratio > 0.1:
                    self.logger.info("  [DIAGNOSTIC] ✓✓ Batch optimization working well!")
                else:
                    self.logger.warning("  [DIAGNOSTIC] ⚠ StateGNN update still weak, may need higher learning rate")
            else:
                self.logger.warning("  [DIAGNOSTIC] ✗ WARNING: StateGNN still not updating!")

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies_list),
            'entropy_coef': entropy_coef,
            'total_loss': np.mean(policy_losses) + np.mean(value_losses),
            'state_gnn_param_change': state_gnn_change,
            'policy_param_change': policy_change,
            'value_param_change': value_change,
            'state_gnn_grad_norm': np.mean(state_gnn_grads),
            'policy_grad_norm': np.mean(policy_grads),
            'value_grad_norm': np.mean(value_grads)
        }
