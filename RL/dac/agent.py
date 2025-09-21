"""
DAC Agent Module with PPO Optimization

This module implements the main DAC agent logic with PPO optimization,
based on the ASquaredC_PPO_agent implementation and adapted for GNN
and structural design optimization.

Key components:
- Dual MDP management (high-level and low-level)
- PPO loss computation for both levels
- Experience collection and storage
- Option termination handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

from .config import DACConfig
from .gnn import DACDoubleActorCritic
from .buffer import DACStorage, DACExperienceBuffer, random_sample
from .environment import DACEnvironmentWrapper


class DACAgent:
    """
    DAC (Double Actor-Critic) Agent with PPO optimization.

    Based on ASquaredC_PPO_agent but adapted for:
    - GNN-based state representation
    - Dual MDP structure (high-level and low-level)
    - Option length bonus
    - Structural design optimization
    """

    def __init__(self,
                 config: DACConfig,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize DAC agent.

        Args:
            config: DAC configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize network
        network_config = config.get_network_config()
        network_config['logger'] = self.logger  # Pass logger to network
        self.network = DACDoubleActorCritic(**network_config)

        # Initialize optimizers (separate for high and low level)
        self.high_level_optimizer = torch.optim.Adam(
            self._get_high_level_parameters(),
            lr=config.high_level_lr
        )
        self.low_level_optimizer = torch.optim.Adam(
            self._get_low_level_parameters(),
            lr=config.low_level_lr
        )

        # Experience storage
        self.storage = DACStorage(config.rollout_length, config.device)
        self.experience_buffer = DACExperienceBuffer(
            buffer_size=config.buffer_size,
            batch_size=config.mini_batch_size,
            length_bonus_weight=config.length_bonus_weight,
            device=config.device
        )

        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.learning_count = 0

        # Option tracking (like ASquaredC_PPO_agent)
        self.prev_options = torch.zeros(config.num_workers, dtype=torch.long, device=config.device)
        self.is_initial_states = torch.ones(config.num_workers, dtype=torch.bool, device=config.device)

        # New: Option step tracking for consistent reward calculation
        self.current_option_step = torch.zeros(config.num_workers, dtype=torch.long, device=config.device)
        self.option_reward_accumulator = torch.zeros(config.num_workers, dtype=torch.float, device=config.device)

        # Statistics
        self.episode_rewards = []
        self.option_lengths = []
        self.high_level_losses = []
        self.low_level_losses = []

        self.logger.info(f"DAC Agent initialized with {sum(p.numel() for p in self.network.parameters())} parameters")

    def _get_high_level_parameters(self) -> List[nn.Parameter]:
        """Get parameters for high-level networks."""
        params = []
        params.extend(self.network.inter_option_policy.parameters())
        params.extend(self.network.option_termination.parameters())
        params.extend(self.network.high_level_value.parameters())
        params.extend(self.network.option_value.parameters())
        return params

    def _get_low_level_parameters(self) -> List[nn.Parameter]:
        """Get parameters for low-level networks."""
        params = []
        params.append(self.network.intra_option_mean)
        params.append(self.network.intra_option_std)
        params.extend(self.network.low_level_value.parameters())
        return params

    def select_option_and_action(self,
                                graph_data: Dict[str, torch.Tensor],
                                structure_obj=None,
                                deterministic: bool = False) -> Tuple[int, torch.Tensor, Dict[str, Any]]:
        """
        Select option and action for current state with action restrictions.

        Args:
            graph_data: Graph data for current state
            structure_obj: Structure object for action validation
            deterministic: Whether to select deterministically

        Returns:
            Tuple of (option, action, info)
        """
        with torch.no_grad():
            # Extract features
            story_features, global_features = self.network.get_features(graph_data)

            # Select option (high-level)
            option, option_log_prob = self.network.select_option(
                global_features,
                self.prev_options[0] if len(self.prev_options) > 0 else None,
                self.is_initial_states[0] if len(self.is_initial_states) > 0 else None,
                deterministic=deterministic
            )

            # Create valid actions mask (like Option-Critic)
            valid_mask = None
            self.logger.debug(f"structure_obj is None  : {structure_obj is None}")
            if structure_obj is not None:
                valid_mask = self._create_valid_actions_mask(structure_obj, self.config.device)
                self.logger.debug(f"Valid actions mask: {valid_mask.sum().item()}/{len(valid_mask)} actions available")

            # Select action (low-level) with action restrictions
            action, action_log_prob = self.network.select_action(
                story_features,
                option,
                valid_mask=valid_mask,
                deterministic=deterministic
            )

            self.logger.debug(f"Selected action in agent : {action}")
            self.logger.debug(f"action_log_prob in agent : {action_log_prob}")
            # Get values
            high_value = self.network.get_high_value(global_features)
            low_value = self.network.get_low_value(story_features, option)

            # Update state tracking
            old_option = self.prev_options[0]
            self.prev_options[0] = option
            self.is_initial_states[0] = False

            # Update option step tracking
            if old_option != option or self.is_initial_states[0]:
                # New option started, reset step counter
                self.current_option_step[0] = 1
                self.option_reward_accumulator[0] = 0.0
            else:
                # Same option continues, increment step counter
                self.current_option_step[0] += 1

            info = {
                'option': option.item(),
                'option_log_prob': option_log_prob.item(),
                'action_log_prob': action_log_prob.mean().item(),  # Take mean for multi-story case
                'high_value': high_value.item(),
                'low_value': low_value.item(),
                'story_features': story_features,
                'global_features': global_features,
                'valid_mask': valid_mask,
                'current_option_step': self.current_option_step[0].item()  # Add option step info
            }

        return option.item(), action, info

    def store_experience(self,
                        graph_data: Dict[str, torch.Tensor],
                        option: int,
                        action: torch.Tensor,
                        base_reward: float,
                        next_graph_data: Dict[str, torch.Tensor],
                        done: bool,
                        option_terminated: bool,
                        fail_name: Optional[str] = None,
                        info: Dict[str, Any] = None):
        """
        Store experience in storage for PPO training with unified reward calculation.

        Args:
            graph_data: Current state graph data
            option: Selected option
            action: Selected action
            base_reward: Base environment reward
            next_graph_data: Next state graph data
            done: Episode done flag
            option_terminated: Option terminated flag
            fail_name: Name of failure type (if any), triggers penalty
            info: Additional information
        """
        # Get network outputs for storage
        with torch.no_grad():
            prediction = self.network(graph_data)

        # Calculate unified reward with option length bonus and failure penalty
        current_step = info.get('current_option_step', 1) if info else 1

        # Low-level reward: base_reward + length_bonus_weight * (step - 1)
        low_level_reward = base_reward + self.config.length_bonus_weight * (current_step - 1)

        # Apply failure penalty if needed
        if fail_name is not None:
            low_level_reward -= self.config.failure_penalty
            self.logger.debug(f"Applied failure penalty for {fail_name}: -{self.config.failure_penalty}")

        # Accumulate low-level reward for high-level calculation
        self.option_reward_accumulator[0] += low_level_reward

        # High-level reward is the accumulated low-level rewards when option terminates
        if option_terminated:
            high_level_reward = self.option_reward_accumulator[0].item()
            # Reset accumulator for next option
            self.option_reward_accumulator[0] = 0.0
        else:
            # Option continues, high-level reward is 0 for intermediate steps
            high_level_reward = 0.0

        self.logger.debug(f"Reward calculation: base={base_reward:.3f}, step={current_step}, "
                         f"low_level={low_level_reward:.3f}, high_level={high_level_reward:.3f}, "
                         f"option_terminated={option_terminated}")

        # Store data in format compatible with ASquaredC_PPO_agent, but using unified reward
        storage_data = {
            's': graph_data,  # States
            'a': action,      # Actions
            'o': torch.tensor([option], device=self.config.device),  # Options
            'r': torch.tensor([low_level_reward], device=self.config.device),  # Unified low-level reward
            'r_hat': torch.tensor([high_level_reward], device=self.config.device),  # Accumulated high-level reward
            'm': torch.tensor([1 - done], device=self.config.device),  # Mask (1 - done)
            'prev_o': torch.tensor([self.prev_options[0]], device=self.config.device),  # Previous options
            'init': torch.tensor([self.is_initial_states[0]], device=self.config.device),  # Initial state flag
            'pi_hat': prediction['inter_pi'],  # Inter-option policy
            'beta': prediction['beta'],        # Termination probabilities
            'mean': prediction['mean_all'],    # Action means
            'std': prediction['std_all'],      # Action stds
            'v_hat': prediction['high_value'], # High-level value
            'v_bar': prediction['option_value'][:, option:option+1],  # Low-level value
            'log_pi_hat': torch.log(prediction['inter_pi'][0, option] + 1e-8).unsqueeze(0),
            'log_pi_bar': info.get('action_log_prob', 0.0) if info else 0.0
        }

        self.storage.add(storage_data)

    def compute_adv(self, storage: DACStorage, mdp: str):
        """
        Compute advantages using GAE.
        Based on ASquaredC_PPO_agent.compute_adv()

        Args:
            storage: Storage object
            mdp: MDP type ('hat' or 'bar')
        """
        config = self.config

        # Both MDPs now use the same underlying reward structure
        # but with different aggregation strategies
        if mdp == 'hat':
            v_key = 'v_hat'
            r_key = 'r_hat'  # Accumulated rewards at option termination
        else:  # mdp == 'bar'
            v_key = 'v_bar'
            r_key = 'r'      # Step-wise rewards with length bonus

        v = storage.data[v_key]
        adv_key = f'adv_{mdp}'
        ret_key = f'ret_{mdp}'

        ret = v[-1].detach() if v else torch.zeros(1, device=self.config.device)
        advantages = torch.zeros_like(ret)

        for i in reversed(range(len(storage.data[r_key]))):
            reward = storage.data[r_key][i]
            mask = storage.data['m'][i]
            value = v[i] if i < len(v) else torch.zeros_like(ret)

            ret = reward + config.discount * mask * ret

            if not config.use_gae:
                advantages = ret - value.detach()
            else:
                next_value = v[i + 1] if i + 1 < len(v) else torch.zeros_like(value)
                td_error = reward + config.discount * mask * next_value - value
                advantages = advantages * config.gae_tau * config.discount * mask + td_error

            storage.data[adv_key][i] = advantages.detach()
            storage.data[ret_key][i] = ret.detach()

    def compute_log_pi_a(self,
                        options: torch.Tensor,
                        pi_hat: torch.Tensor,
                        action: torch.Tensor,
                        mean: torch.Tensor,
                        std: torch.Tensor,
                        mdp: str) -> torch.Tensor:
        """
        Compute log probability of action.
        Based on ASquaredC_PPO_agent.compute_log_pi_a()
        """
        if mdp == 'hat':
            # Ensure options has the right shape for gather operation
            if options.dim() == 1:
                options = options.unsqueeze(1)  # Convert [batch] to [batch, 1]
            return pi_hat.add(1e-5).log().gather(1, options)
        elif mdp == 'bar':
            # Compute log probability for continuous actions
            pi_bar = self.compute_pi_bar(options, action, mean, std)
            return pi_bar.add(1e-5).log()
        else:
            raise NotImplementedError(f"Unknown MDP type: {mdp}")

    def compute_pi_bar(self,
                      options: torch.Tensor,
                      action: torch.Tensor,
                      mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:
        """
        Compute low-level policy probability.
        Based on ASquaredC_PPO_agent.compute_pi_bar()
        """
        # Get option-specific parameters
        # From debug: mean shape is [4, 256, 14] = [num_options, hidden_dim, action_dim]
        # options shape is [1] = [batch_size]
        # We need to extract the option-specific mean and std

        if mean.dim() == 3:  # [num_options, hidden_dim, action_dim]
            # Select the specific option for each batch element
            batch_size = options.size(0)
            option_indices = options.view(-1)  # [batch_size]

            # Extract option-specific parameters
            option_mean = mean[option_indices]  # [batch_size, hidden_dim, action_dim]
            option_std = std[option_indices]    # [batch_size, hidden_dim, action_dim]

            # For simplicity, use the mean across hidden dimensions
            # This is a simplification and may need adjustment based on the actual model
            option_mean = option_mean.mean(dim=1)  # [batch_size, action_dim]
            option_std = option_std.mean(dim=1)    # [batch_size, action_dim]
        else:
            # Handle 2D tensors (fallback)
            if options.dim() == 1:
                options = options.unsqueeze(-1)  # [batch_size, 1]
            options_expanded = options.expand(-1, mean.size(-1))
            option_mean = mean.gather(1, options_expanded).squeeze(1)
            option_std = std.gather(1, options_expanded).squeeze(1)

        # Compute log probability
        dist = Normal(option_mean, option_std)
        log_prob = dist.log_prob(action).sum(-1)
        pi_bar = log_prob.exp().unsqueeze(-1)

        return pi_bar

    def learn(self, storage: DACStorage, mdp: str, freeze_v: bool = False):
        """
        Learn from collected experiences using PPO.
        Based on ASquaredC_PPO_agent.learn()

        Args:
            storage: Storage containing experiences
            mdp: MDP type ('hat' or 'bar')
            freeze_v: Whether to freeze value function
        """
        config = self.config

        # Get data from storage
        data_keys = ['s', 'a', 'o', f'log_pi_{mdp}', f'ret_{mdp}', f'adv_{mdp}',
                    'prev_o', 'init', 'pi_hat', 'mean', 'std']
        data = storage.cat(data_keys)

        # Check if we have sufficient data
        if len(data) != len(data_keys):
            self.logger.warning(f"Insufficient data for {mdp} learning")
            return

        # Check if any tensor data is empty (skip lists like graph data)
        for i, d in enumerate(data):
            if hasattr(d, 'numel') and d.numel() == 0:
                self.logger.warning(f"Empty tensor data for {mdp} learning at index {i}")
                return
            elif isinstance(d, list) and len(d) == 0:
                self.logger.warning(f"Empty list data for {mdp} learning at index {i}")
                return

        (states, actions, options, log_probs_old, returns, advantages,
         prev_options, inits, pi_hat, mean, std) = data

        # Detach old data
        log_probs_old = log_probs_old.detach()
        pi_hat = pi_hat.detach()
        mean = mean.detach()
        std = std.detach()

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO optimization
        for epoch in range(config.optimization_epochs):
            # Sample mini-batches
            # Get the number of samples (handle both tensor and list states)
            num_samples = len(states) if isinstance(states, list) else states.size(0)
            sampler = random_sample(np.arange(num_samples), config.mini_batch_size)

            for batch_indices in sampler:
                batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=config.device)

                # Sample batch data
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_options = options[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_prev_o = prev_options[batch_indices]
                batch_init = inits[batch_indices]

                # Forward pass
                loss = self._compute_ppo_loss(
                    batch_states, batch_actions, batch_options,
                    batch_log_probs_old, batch_returns, batch_advantages,
                    batch_prev_o, batch_init, mdp, freeze_v
                )

                # Backward pass
                optimizer = self.high_level_optimizer if mdp == 'hat' else self.low_level_optimizer
                optimizer.zero_grad()
                loss['total_loss'].backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                optimizer.step()

                # Log losses
                if mdp == 'hat':
                    self.high_level_losses.append(loss['total_loss'].item())
                else:
                    self.low_level_losses.append(loss['total_loss'].item())

    def _compute_ppo_loss(self,
                         states: List[Dict[str, torch.Tensor]],
                         actions: torch.Tensor,
                         options: torch.Tensor,
                         log_probs_old: torch.Tensor,
                         returns: torch.Tensor,
                         advantages: torch.Tensor,
                         prev_options: torch.Tensor,
                         inits: torch.Tensor,
                         mdp: str,
                         freeze_v: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss for given MDP level.

        Args:
            states: Batch of states
            actions: Batch of actions
            options: Batch of options
            log_probs_old: Old log probabilities
            returns: Target returns
            advantages: Computed advantages
            prev_options: Previous options
            inits: Initial state flags
            mdp: MDP type ('hat' or 'bar')
            freeze_v: Whether to freeze value function

        Returns:
            Dictionary containing loss components
        """
        config = self.config

        # Process batch of states (assuming they're graph data)
        if len(states) > 0 and isinstance(states[0], dict):
            # Combine graph data from batch
            batch_graph_data = self._combine_graph_batch(states)
            prediction = self.network(batch_graph_data)
        else:
            raise ValueError("Invalid state format")

        # CURRENT LIMITATION: Due to graph batching complexity, we only process
        # one sample at a time. This limits the effective mini-batch size to 1,
        # reducing training efficiency but maintaining correctness.
        effective_batch_size = 1

        # Take only first element of each tensor to match single graph prediction
        # This ensures dimensional consistency with the single graph output
        actions = actions[:effective_batch_size]
        options = options[:effective_batch_size]
        log_probs_old = log_probs_old[:effective_batch_size]
        returns = returns[:effective_batch_size]
        advantages = advantages[:effective_batch_size]
        prev_options = prev_options[:effective_batch_size]
        inits = inits[:effective_batch_size]

        # Compute current policy
        if mdp == 'hat':
            # High-level policy
            pi_hat = self.network.compute_pi_hat(
                prediction['global_features'],
                prev_options.view(-1),
                inits.view(-1)
            )
            entropy = -(pi_hat * pi_hat.add(1e-5).log()).sum(-1).mean()
            log_pi_a = self.compute_log_pi_a(
                options, pi_hat, actions, prediction['mean_all'], prediction['std_all'], mdp
            )
            beta_loss = prediction['beta'].mean()
            value = (prediction['option_value'] * pi_hat).sum(-1).unsqueeze(-1)

        else:  # mdp == 'bar'
            # Low-level policy
            log_pi_a = self.compute_log_pi_a(
                options, prediction['inter_pi'], actions,
                prediction['mean_all'], prediction['std_all'], mdp
            )
            entropy = torch.tensor(0.0, device=config.device)
            beta_loss = torch.tensor(0.0, device=config.device)
            # Ensure options has the right shape for gather operation
            if options.dim() == 1:
                options_for_gather = options.unsqueeze(1)  # Convert [batch] to [batch, 1]
            else:
                options_for_gather = options
            value = prediction['option_value'].gather(1, options_for_gather)

        # PPO loss computation
        ratio = (log_pi_a - log_probs_old).exp()
        obj = ratio * advantages
        obj_clipped = ratio.clamp(1.0 - config.ppo_ratio_clip, 1.0 + config.ppo_ratio_clip) * advantages
        policy_loss = -torch.min(obj, obj_clipped).mean()

        # Add regularization
        policy_loss = policy_loss - config.entropy_weight * entropy + config.beta_weight * beta_loss

        # Value loss
        value_loss = 0.5 * (returns - value).pow(2).mean()
        if freeze_v:
            value_loss = torch.tensor(0.0, device=config.device)

        # Total loss
        total_loss = policy_loss + config.value_loss_coef * value_loss

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'beta_loss': beta_loss
        }

    def _combine_graph_batch(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Combine a batch of graph data.

        CURRENT LIMITATION: This implementation only uses the first sample from the batch
        due to the complexity of graph batching. Each graph represents a building structure
        with variable number of nodes and edges, making naive tensor concatenation impossible.

        Proper graph batching would require:
        1. PyTorch Geometric's Batch.from_data_list()
        2. Careful handling of batch indices for story_batch and structure_story_ptr
        3. Proper unbatching for action selection per structure

        For now, training effectively uses batch_size=1.
        """
        if not states:
            raise ValueError("Empty states list")

        # Use only the first state - this limits effective batch size to 1
        return states[0]

    def should_update(self) -> bool:
        """Check if agent should perform an update."""
        return len(self.storage.data.get('s', [])) >= self.config.rollout_length

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update.
        Returns loss statistics.
        """
        if not self.should_update():
            return {}

        # Compute advantages for both MDPs
        self.compute_adv(self.storage, 'hat')
        self.compute_adv(self.storage, 'bar')

        # Learn according to schedule
        if self.config.learning_schedule == 'all':
            # Learn both levels (randomly shuffled)
            mdps = ['hat', 'bar']
            np.random.shuffle(mdps)
            self.learn(self.storage, mdps[0])
            self.learn(self.storage, mdps[1])

        elif self.config.learning_schedule == 'alt':
            # Alternating learning
            if self.learning_count % 2:
                self.learn(self.storage, 'hat')
            else:
                self.learn(self.storage, 'bar')

        elif self.config.learning_schedule == 'sequential':
            # Sequential learning (hat then bar)
            self.learn(self.storage, 'hat')
            self.learn(self.storage, 'bar')

        # Reset storage
        self.storage.reset()
        self.learning_count += 1

        # Return loss statistics
        losses = {}
        if self.high_level_losses:
            losses['high_level_loss'] = np.mean(self.high_level_losses[-10:])
        if self.low_level_losses:
            losses['low_level_loss'] = np.mean(self.low_level_losses[-10:])

        return losses

    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'high_level_optimizer_state_dict': self.high_level_optimizer.state_dict(),
            'low_level_optimizer_state_dict': self.low_level_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.high_level_optimizer.load_state_dict(checkpoint['high_level_optimizer_state_dict'])
        self.low_level_optimizer.load_state_dict(checkpoint['low_level_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']
        self.logger.info(f"Checkpoint loaded from {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_option_length': np.mean(self.option_lengths[-100:]) if self.option_lengths else 0,
            'avg_high_level_loss': np.mean(self.high_level_losses[-100:]) if self.high_level_losses else 0,
            'avg_low_level_loss': np.mean(self.low_level_losses[-100:]) if self.low_level_losses else 0,
        }

    def _create_valid_actions_mask(self, structure_obj, device: torch.device) -> torch.Tensor:
        """
        Create valid actions mask based on structure constraints.
        Based on Option-Critic rollout_option logic (lines 80-115).

        Args:
            structure_obj: Structure object
            device: torch device

        Returns:
            Boolean tensor mask where True = valid action
        """
        # First, create dynamic action mask based on current structure's story_num
        current_story_num = getattr(structure_obj, 'story_num', 4)  # Default to 4 if not found
        dynamic_mask = self.config.get_action_mask(current_story_num)

        self.logger.debug(f"Dynamic mask based on {current_story_num} stories: {dynamic_mask.sum().item()}/{len(dynamic_mask)} actions")

        # Get all restricted actions based on structure constraints
        already_minimum = set(getattr(structure_obj, 'already_minimum_section_story_indexes', []) or [])
        restricted_actions = set()

        if hasattr(structure_obj, 'restrict_action_space'):
            restricted = structure_obj.restrict_action_space()
            if restricted is not None:
                restricted_actions.update(restricted)

        # Combine all invalid actions
        invalid_actions = already_minimum | restricted_actions

        self.logger.debug(f"Already minimum: {already_minimum}, Restricted: {restricted_actions}")
        self.logger.debug(f"Total invalid actions: {invalid_actions}")

        # Start with dynamic mask, then apply structure-specific restrictions
        valid_mask = dynamic_mask.clone()

        # Apply structure-specific restrictions (but only to valid actions within current structure size)
        for invalid_action in invalid_actions:
            if 0 <= invalid_action < current_story_num * 4:  # Only mask if within current structure's action space
                valid_mask[invalid_action] = False

        self.logger.debug(f"Final mask after structure constraints: {valid_mask.sum().item()}/{len(valid_mask)} actions")
        return valid_mask

    def _check_all_indexes_zero(self, structure_obj) -> bool:
        """
        Check if all structure indexes are at minimum (all zeros).
        This indicates that the episode should be terminated.

        Args:
            structure_obj: Structure object

        Returns:
            True if all indexes are zero (episode should terminate)
        """
        # Check if structure has already_minimum_section_story_indexes
        already_minimum = getattr(structure_obj, 'already_minimum_section_story_indexes', [])

        # Get total number of story level actions
        total_actions = len(getattr(structure_obj, 'story_level_actions', []))

        # If all story-level actions are at minimum, episode should terminate
        if total_actions > 0 and len(already_minimum) >= total_actions:
            self.logger.info(f"All structure indexes are at minimum: {len(already_minimum)}/{total_actions} actions at minimum")
            return True

        return False

    def check_termination_conditions(self, structure_obj) -> Tuple[bool, str]:
        """
        Check various termination conditions for episode and option.

        Args:
            structure_obj: Structure object

        Returns:
            Tuple of (should_terminate, termination_reason)
        """
        # Check if all indexes are zero (most important condition)
        if self._check_all_indexes_zero(structure_obj):
            return True, "all_indexes_zero"

        # Add other termination conditions here if needed

        return False, None