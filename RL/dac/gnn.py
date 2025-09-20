"""
DAC Double Actor-Critic GNN Network Module

This module implements the Double Actor-Critic architecture from the DAC paper
with PPO optimization, adapted for Graph Neural Networks and structural design.

Based on:
- "DAC: The Double Actor-Critic Architecture for Learning Options" (NeurIPS 2019)
- ASquaredC_PPO_agent implementation with 'hat' (high-MDP) and 'bar' (low-MDP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Optional, Dict, Any
import numpy as np

from ..model import StateGNN


class DACDoubleActorCritic(nn.Module):
    """
    DAC Double Actor-Critic network with GNN backbone.

    Implements the dual MDP architecture:
    - High-level MDP (hat): Option selection using global features
    - Low-level MDP (bar): Action selection using story-level features

    Based on ASquaredC_PPO_agent design patterns.
    """

    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 member_state_dim: int,
                 num_layers: int,
                 num_actions: int,
                 num_options: int,
                 device: str = 'cpu'):
        """
        Initialize DAC Double Actor-Critic network.

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN
            member_state_dim: Member state dimension
            num_layers: Number of GNN layers
            num_actions: Number of primitive actions
            num_options: Number of options
            device: Device to use
        """
        super(DACDoubleActorCritic, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.member_state_dim = member_state_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = torch.device(device)

        # StateGNN for feature extraction
        self.state_gnn = StateGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            member_state_dim=member_state_dim,
            num_layers=num_layers
        )

        # Feature dimensions from StateGNN
        # StateGNN outputs: [total_story_member_num, member_state_dim * 2]
        # - story features: member_state_dim * 2 (for low-level)
        # - global features: member_state_dim (for high-level)
        story_feature_dim = member_state_dim * 2
        global_feature_dim = member_state_dim

        # ============ High-level Actor-Critic (Option Selection) ============
        # Inter-option policy (policy over options) - pi_Omega in DAC paper
        self.inter_option_policy = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )

        # Option termination function - beta in DAC paper
        self.option_termination = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )

        # High-level value function - V^Omega in DAC paper
        self.high_level_value = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Option-value function - Q^Omega in DAC paper
        self.option_value = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )

        # ============ Low-level Actor-Critic (Action Selection) ============
        # Intra-option policies - pi_omega in DAC paper
        # Using parameter matrices like ASquaredC_PPO_agent
        self.intra_option_mean = nn.Parameter(
            torch.zeros(num_options, story_feature_dim, num_actions)
        )
        self.intra_option_std = nn.Parameter(
            torch.ones(num_options, story_feature_dim, num_actions)
        )

        # Low-level value function - V^omega in DAC paper
        self.low_level_value = nn.Sequential(
            nn.Linear(story_feature_dim + 1, hidden_dim),  # +1 for option encoding
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ============ Auxiliary Networks ============
        # Option encoding for low-level networks (1-dimensional option encoding)
        self.option_embedding = nn.Embedding(num_options, 1)

        # Initialize parameters
        self._initialize_parameters()

        # Move to device
        self.to(self.device)

    def _initialize_parameters(self):
        """Initialize network parameters."""
        # Initialize option policy parameters (like ASquaredC_PPO_agent)
        nn.init.normal_(self.intra_option_mean, mean=0.0, std=0.01)
        nn.init.constant_(self.intra_option_std, 0.1)

        # Initialize other networks
        for module in [self.inter_option_policy, self.option_termination,
                      self.high_level_value, self.option_value, self.low_level_value]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        # Initialize termination with negative bias (start with low termination)
        with torch.no_grad():
            self.option_termination[-1].bias.fill_(-2.2)  # sigmoid(-2.2) ~= 0.1

    def get_features(self, graph_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract story-level and global features using StateGNN.

        Args:
            graph_data: Dictionary containing graph data
                - x: Node features
                - edge_index: Edge indices
                - edge_attr: Edge attributes
                - story_batch: Story batch indices
                - structure_story_ptr: Structure story pointers

        Returns:
            Tuple of (story_features, global_features)
        """
        # Extract graph components
        x = graph_data['x'].to(self.device)
        edge_index = graph_data['edge_index'].to(self.device)
        edge_attr = graph_data['edge_attr'].to(self.device)
        story_batch = graph_data['story_batch'].to(self.device)
        structure_story_ptr = graph_data.get('structure_story_ptr', None)

        # Get features from StateGNN
        # Output: [total_story_member_num, member_state_dim * 2]
        features = self.state_gnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=None,
            story_batch=story_batch,
            structure_story_ptr=structure_story_ptr
        )

        # Split features for dual MDP
        story_features = features  # Full features for story-level (low-level)
        global_features = features[:, self.member_state_dim:]  # Global part for high-level

        # Aggregate global features (mean pooling over stories)
        if len(global_features.shape) > 1:
            global_features = torch.mean(global_features, dim=0, keepdim=True)

        return story_features, global_features

    def compute_pi_hat(self,
                       global_features: torch.Tensor,
                       prev_option: torch.Tensor,
                       is_initial_state: torch.Tensor) -> torch.Tensor:
        """
        Compute high-level policy pi_hat (option selection policy).
        Based on ASquaredC_PPO_agent.compute_pi_hat()

        Args:
            global_features: Global state features
            prev_option: Previous option
            is_initial_state: Whether in initial state

        Returns:
            Option selection probabilities
        """
        # Get inter-option policy (policy over options)
        inter_pi_logits = self.inter_option_policy(global_features)
        inter_pi = F.softmax(inter_pi_logits, dim=-1)

        # Get termination probabilities
        beta_logits = self.option_termination(global_features)
        beta = torch.sigmoid(beta_logits)

        # Compute pi_hat according to DAC/Option-Critic formulation
        batch_size = global_features.shape[0]
        device = global_features.device

        # Create mask for previous option
        mask = torch.zeros_like(inter_pi, device=device)
        if prev_option.dim() == 0:
            prev_option = prev_option.unsqueeze(0)
        mask.scatter_(1, prev_option.unsqueeze(-1), 1.0)

        # pi_hat = beta * pi_Omega + (1 - beta) * mask_prev_option
        pi_hat = beta * inter_pi + (1 - beta) * mask

        # For initial states, use inter-option policy directly
        is_initial_expanded = is_initial_state.view(-1, 1).expand_as(inter_pi)
        pi_hat = torch.where(is_initial_expanded, inter_pi, pi_hat)

        return pi_hat

    def compute_pi_bar(self,
                       story_features: torch.Tensor,
                       options: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute low-level policy pi_bar (intra-option policy).
        Based on ASquaredC_PPO_agent.compute_pi_bar()

        Args:
            story_features: Story-level features [batch_size, feature_dim]
            options: Selected options [batch_size] or scalar

        Returns:
            Tuple of (mean, std) for action distribution
        """
        # Ensure options is a tensor with proper shape
        if options.dim() == 0:
            options = options.unsqueeze(0)

        batch_size = story_features.shape[0]
        feature_dim = story_features.shape[1]

        # Expand options to match batch_size if needed
        # In DAC, all story members use the same option for an episode
        if options.shape[0] == 1 and batch_size > 1:
            options = options.expand(batch_size)

        # Get option-specific weight matrices
        # intra_option_mean: [num_options, feature_dim, num_actions]
        # intra_option_std: [num_options, feature_dim, num_actions]

        # Select parameters for the chosen options
        option_mean_weights = self.intra_option_mean[options]  # [batch_size, feature_dim, num_actions]
        option_std_weights = self.intra_option_std[options]    # [batch_size, feature_dim, num_actions]

        # Compute action means and stds using matrix multiplication
        # story_features: [batch_size, feature_dim]
        # option_weights: [batch_size, feature_dim, num_actions]
        mean = torch.bmm(story_features.unsqueeze(1), option_mean_weights).squeeze(1)  # [batch_size, num_actions]
        std = torch.bmm(story_features.unsqueeze(1), option_std_weights).squeeze(1)    # [batch_size, num_actions]

        # Ensure positive std
        std = F.softplus(std) + 1e-5

        return mean, std

    def select_option(self,
                      global_features: torch.Tensor,
                      prev_option: Optional[torch.Tensor] = None,
                      is_initial_state: Optional[torch.Tensor] = None,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select option using high-level policy.

        Args:
            global_features: Global state features
            prev_option: Previous option
            is_initial_state: Whether in initial state
            deterministic: Whether to select deterministically

        Returns:
            Tuple of (option, log_prob)
        """
        if prev_option is None:
            prev_option = torch.zeros(global_features.shape[0], dtype=torch.long, device=self.device)
        if is_initial_state is None:
            is_initial_state = torch.ones(global_features.shape[0], dtype=torch.bool, device=self.device)

        # Get option selection probabilities
        pi_hat = self.compute_pi_hat(global_features, prev_option, is_initial_state)

        # Sample or select greedily
        if deterministic:
            option = torch.argmax(pi_hat, dim=-1)
        else:
            dist = Categorical(pi_hat)
            option = dist.sample()

        # Compute log probability
        log_prob = torch.log(pi_hat.gather(1, option.unsqueeze(-1)) + 1e-8).squeeze(-1)

        return option, log_prob

    def select_action(self,
                      story_features: torch.Tensor,
                      option: torch.Tensor,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using low-level policy.

        Args:
            story_features: Story-level features
            option: Selected option
            deterministic: Whether to select deterministically

        Returns:
            Tuple of (action, log_prob)
        """
        # Get action distribution parameters
        mean, std = self.compute_pi_bar(story_features, option)

        # Sample or select greedily
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        # Compute log probability
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def get_high_value(self, global_features: torch.Tensor) -> torch.Tensor:
        """Get high-level value function."""
        return self.high_level_value(global_features)

    def get_low_value(self, story_features: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
        """Get low-level value function."""
        # Embed option
        option_embed = self.option_embedding(option)

        # Combine story features with option embedding
        if story_features.dim() == 2 and option_embed.dim() == 2:
            # Average story features if multiple stories
            story_avg = torch.mean(story_features, dim=0, keepdim=True)
            combined = torch.cat([story_avg, option_embed], dim=-1)
        else:
            combined = torch.cat([story_features, option_embed], dim=-1)

        return self.low_level_value(combined)

    def get_option_value(self, global_features: torch.Tensor) -> torch.Tensor:
        """Get option-value function Q^Omega."""
        return self.option_value(global_features)

    def get_termination_prob(self, global_features: torch.Tensor) -> torch.Tensor:
        """Get option termination probabilities."""
        beta_logits = self.option_termination(global_features)
        return torch.sigmoid(beta_logits)

    def forward(self, graph_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all network outputs.

        Args:
            graph_data: Graph data dictionary

        Returns:
            Dictionary containing all network outputs
        """
        # Extract features
        story_features, global_features = self.get_features(graph_data)

        # High-level outputs
        inter_pi_logits = self.inter_option_policy(global_features)
        inter_pi = F.softmax(inter_pi_logits, dim=-1)
        beta_logits = self.option_termination(global_features)
        beta = torch.sigmoid(beta_logits)
        high_value = self.high_level_value(global_features)
        option_value = self.option_value(global_features)

        # Low-level outputs (need option for full computation)
        # Return parameters for all options
        mean_all = self.intra_option_mean  # [num_options, story_feature_dim, num_actions]
        std_all = F.softplus(self.intra_option_std) + 1e-5

        return {
            'story_features': story_features,
            'global_features': global_features,
            'inter_pi': inter_pi,
            'beta': beta,
            'high_value': high_value,
            'option_value': option_value,
            'mean_all': mean_all,
            'std_all': std_all
        }