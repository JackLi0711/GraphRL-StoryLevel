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
                 gae_lambda: float = 0.95,
                 value_clip_epsilon: float = 0.2,
                 minibatch_size: int = 64,
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
        self.gae_lambda = gae_lambda
        self.value_clip_epsilon = value_clip_epsilon
        self.minibatch_size = minibatch_size

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

        # Flatten all episodes with GAE(λ) and normalized rewards
        all_states = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_valid_masks = []
        all_graphs = []           # Store graphs for recomputing features
        all_structures = []       # Store structures for recomputing features
        all_old_values = []       # For value clipping
        all_advantages = []
        all_returns = []

        for idx, episode in enumerate(episodes):
            rewards = torch.tensor(episode['rewards'], dtype=torch.float32, device=self.device)
            old_values = torch.cat(episode['values']).squeeze().detach()  # [T]
            initial_material = episode['initial_material_usage']
            rewards_norm = rewards / (initial_material + 1e-8)

            # GAE(λ)
            T = len(rewards_norm)
            advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
            gae = 0.0
            for t in reversed(range(T)):
                v_t = old_values[t]
                v_tp1 = old_values[t+1] if t+1 < T else torch.tensor(0.0, device=self.device)
                delta = rewards_norm[t] + self.gamma * v_tp1 - v_t
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
            returns = advantages + old_values

            if self.logger and idx == 0:
                self.logger.info(f"  [GAE] Episode initial_material_usage: {initial_material:.2f}")
                self.logger.info(f"  [GAE] rewards_norm: mean={rewards_norm.mean():.4f}, std={rewards_norm.std():.4f}")
                self.logger.info(f"  [GAE] values_old: mean={old_values.mean():.4f}, std={old_values.std():.4f}")

            # Collect
            all_states.extend(episode['states'])
            all_global_states.extend(episode['global_states'])
            all_actions.extend(episode['actions'])
            all_old_log_probs.extend(episode['log_probs'])
            all_valid_masks.extend(episode['valid_masks'])
            all_graphs.extend(episode['graphs'])
            all_structures.extend(episode['structures'])
            all_old_values.append(old_values)
            all_advantages.append(advantages)
            all_returns.append(returns)

        old_log_probs = torch.stack(all_old_log_probs).detach()
        old_values_flat = torch.cat(all_old_values).detach()
        advantages = torch.cat(all_advantages)
        returns = torch.cat(all_returns)

        # Normalize advantages across episodes
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
            # === 單次 StateGNN 前向（整個 epoch 保持一致圖），epoch 結尾再 backward/step 一次 ===
            batched_graph, batched_story_batch, structure_story_ptr, num_stories_per_graph = \
                self._batch_all_graphs(all_graphs, all_structures)

            if self.logger and epoch == 0:
                self.logger.info(f"  [BATCH] Processing {len(all_graphs)} graphs in one forward pass")
                self.logger.info(f"  [BATCH] Total nodes: {batched_graph.x.shape[0]}, "
                               f"Total edges: {batched_graph.edge_index.shape[1]//2}")

            edge_batch = batched_graph.batch[batched_graph.edge_index[0]]
            edge_batch_half = edge_batch[::2]

            all_story_features_batched = self.state_gnn(
                batched_graph.x,
                batched_graph.edge_index,
                batched_graph.edge_attr,
                edge_batch_half,
                batched_story_batch,
                structure_story_ptr
            )

            # 分割回 timesteps
            story_features_list = []
            global_features_list = []
            start_idx = 0
            for num_stories in num_stories_per_graph:
                end_idx = start_idx + num_stories
                story_feat = all_story_features_batched[start_idx:end_idx]
                global_feat = story_feat.mean(dim=0, keepdim=True)
                story_features_list.append(story_feat)
                global_features_list.append(global_feat)
                start_idx = end_idx

            # === Minibatch（累加 loss，epoch 結尾一次 backward） ===
            num_steps = len(all_states)
            indices = np.arange(num_steps)
            np.random.shuffle(indices)
            mb_size = min(self.minibatch_size, num_steps)

            if self.logger and epoch == 0:
                self.logger.info(f"  [MINIBATCH] total_steps={num_steps}, minibatch_size={mb_size}, num_batches={(num_steps+mb_size-1)//mb_size}")

            epoch_policy_loss_sum = 0.0
            epoch_value_loss_sum = 0.0
            epoch_entropy_sum = 0.0

            # 清零梯度（整個 epoch 聚合梯度）
            self.state_gnn_optimizer.zero_grad()
            self.policy_value_optimizer.zero_grad()

            for start in range(0, num_steps, mb_size):
                mb_idx = indices[start:start+mb_size]

                new_log_probs_mb = []
                entropies_mb = []
                new_values_mb = []

                for t in mb_idx:
                    story_features = story_features_list[t]
                    global_features = global_features_list[t]
                    valid_mask = all_valid_masks[t]
                    action = all_actions[t]

                    # 防呆：至少一個可行動作
                    assert valid_mask.any().item(), "valid_mask is all False at a timestep"

                    action_scores = self.policy_net(story_features)
                    masked_scores = action_scores.masked_fill(~valid_mask, -1e9)
                    probs = F.softmax(masked_scores, dim=0)
                    dist = Categorical(probs)

                    new_log_probs_mb.append(dist.log_prob(torch.tensor(action, device=self.device)))
                    entropies_mb.append(dist.entropy())
                    new_values_mb.append(self.value_net(global_features))

                new_log_probs_mb = torch.stack(new_log_probs_mb)
                entropies_mb = torch.stack(entropies_mb)
                new_values_mb = torch.cat(new_values_mb).squeeze()

                old_log_probs_mb = old_log_probs[mb_idx]
                returns_mb = returns[mb_idx]
                advantages_mb = advantages[mb_idx]
                old_values_mb = old_values_flat[mb_idx]

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs_mb - old_log_probs_mb)
                surr1 = ratio * advantages_mb.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_mb.detach()
                policy_loss_mb = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_pred = new_values_mb
                v_old = old_values_mb
                v_clipped = v_old + torch.clamp(v_pred - v_old, -self.value_clip_epsilon, self.value_clip_epsilon)
                value_loss_unclipped = (v_pred - returns_mb).pow(2)
                value_loss_clipped = (v_clipped - returns_mb).pow(2)
                value_loss_mb = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss_mb = -entropies_mb.mean()

                # 累加（取和，再除以批次數得到平均）
                epoch_policy_loss_sum = epoch_policy_loss_sum + policy_loss_mb
                epoch_value_loss_sum = epoch_value_loss_sum + value_loss_mb
                epoch_entropy_sum = epoch_entropy_sum + entropy_loss_mb

            num_batches = max(1, (num_steps + mb_size - 1) // mb_size)
            policy_loss = epoch_policy_loss_sum / num_batches
            value_loss = epoch_value_loss_sum / num_batches
            entropy_loss = epoch_entropy_sum / num_batches
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.get_entropy_coef() * entropy_loss

            # 反傳與最佳化（單次）
            total_loss.backward()

            # 記錄梯度範圍（pre-clipping）
            state_gnn_grad_norm = 0.0
            for p in self.state_gnn.parameters():
                if p.grad is not None:
                    state_gnn_grad_norm += p.grad.norm().item() ** 2
            state_gnn_grad_norm = state_gnn_grad_norm ** 0.5

            policy_grad_norm = 0.0
            for p in self.policy_net.parameters():
                if p.grad is not None:
                    policy_grad_norm += p.grad.norm().item() ** 2
            policy_grad_norm = policy_grad_norm ** 0.5

            value_grad_norm = 0.0
            for p in self.value_net.parameters():
                if p.grad is not None:
                    value_grad_norm += p.grad.norm().item() ** 2
            value_grad_norm = value_grad_norm ** 0.5

            nn.utils.clip_grad_norm_(self.state_gnn.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + list(self.value_net.parameters()), self.max_grad_norm)

            self.state_gnn_optimizer.step()
            self.policy_value_optimizer.step()

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

        def _param_change(before_list, after_list):
            return torch.sqrt(sum((a - b).pow(2).sum() for b, a in zip(before_list, after_list))).item()

        state_gnn_change = _param_change(state_gnn_params_before, state_gnn_params_after)
        policy_change = _param_change(policy_params_before, policy_params_after)
        value_change = _param_change(value_params_before, value_params_after)

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
