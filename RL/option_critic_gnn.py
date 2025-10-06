import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
import torch.nn.functional as F
from math import exp
import numpy as np
from typing import Tuple, Optional, List

from .model import StateGNN
from .utils import to_tensor, ensure_tensor_on_device, calculate_epsilon


class OptionCriticGNN(nn.Module):
    """
    Option-Critic network with Graph Neural Network backbone.

    **NEW ARCHITECTURE (v2)**: Supports dynamic action sizes!

    Key changes from v1:
    - Intra-option policies use per-story-member scoring (MLPs)
    - No longer uses fixed-size action logits (options_W, options_b)
    - Can handle buildings with varying numbers of floors
    - Fully aligned with Option-Critic paper (Bacon et al., 2016)

    Architecture components:
    1. StateGNN: Extracts features from graph-structured observations
       - Story-level features: [num_story_members, gnn_output_dim]
       - Global features: [1, member_state_dim]

    2. Q network: Option-value function Q_Ω(s,ω) [global state → num_options]
       - Learns value of each option in given state
       - Used for policy-over-options

    3. Termination networks: β(s,ω) [global state → num_options]
       - Learns when to terminate each option
       - Sigmoid output gives termination probability

    4. Intra-option policies: π(m|s,ω) [story-level state → score per member]
       - Each option has independent MLP
       - Scores each story member for selection
       - Dynamic action space based on building size

    The action space is dynamic:
    - 4-story building: 16 story members → 16 possible actions
    - 7-story building: 28 story members → 28 possible actions
    - 10-story building: 40 story members → 40 possible actions
    - Action = which story member to reduce section

    Theoretical alignment with Option-Critic paper:
    - Policy gradient (Theorem 1): ∇_θ J = E[∇log π(a|s,ω) * Q_U(s,ω,a)]
    - Termination gradient (Theorem 2): ∇_ϑ J = E[β(s',ω) * A_Ω(s',ω)]
    - Q_U estimation (Page 4): Q_U(s,ω,a) = r + γ*[(1-β)Q_Ω(s',ω) + β*V_Ω(s')]
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 member_state_dim: int,
                 num_layers: int,
                 num_actions: int,
                 num_options: int,
                 temperature: float = 1.0,
                 eps_start: float = 1.0,
                 eps_min: float = 0.1,
                 eps_decay: int = int(1e6),
                 eps_test: float = 0.05,
                 device: str = 'cpu',
                 testing: bool = False,
                 debug_logging: bool = False):
        """
        Initialize OptionCriticGNN with support for dynamic action sizes.

        Key architectural change (v2):
        - Instead of fixed-size action logits, we use per-story-member scoring
        - This allows handling varying numbers of story members across episodes

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN
            member_state_dim: Member state dimension
            num_layers: Number of GNN layers
            num_actions: [DEPRECATED] Not used in new architecture, kept for compatibility
            num_options: Number of options
            temperature: Temperature for action selection
            eps_start: Starting epsilon for exploration
            eps_min: Minimum epsilon
            eps_decay: Epsilon decay rate
            eps_test: Epsilon for testing
            device: Device to use
            testing: Whether in testing mode
            debug_logging: Enable debug logging
        """
        super(OptionCriticGNN, self).__init__()

        # Store parameters
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.member_state_dim = member_state_dim
        self.num_layers = num_layers
        self.num_actions = num_actions  # Kept for compatibility, not used
        self.num_options = num_options
        self.device = torch.device(device)
        self.testing = testing
        self.debug_logging = debug_logging

        # Exploration parameters
        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        # StateGNN for feature extraction
        self.state_gnn = StateGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            member_state_dim=member_state_dim,
            num_layers=num_layers
        )

        # GNN output dimension
        gnn_output_dim = member_state_dim * 2

        # =========================================================================
        # Option-Critic Components
        # =========================================================================

        # 1. Policy-Over-Options (uses global state)
        self.Q = nn.Linear(member_state_dim, num_options)

        # 2. Termination functions (uses global state)
        self.terminations = nn.Linear(member_state_dim, num_options)

        # 3. Intra-Option Policies (NEW ARCHITECTURE v2)
        # ✅ NEW: Per-story-member scoring networks
        # Each option has its own MLP that scores story members
        self.intra_option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gnn_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)  # Output: score for each story member
            )
            for _ in range(num_options)
        ])

        # Initialize parameters
        self._initialize_parameters()

        # Move to device
        self.to(self.device)
        self.train(not testing)
    
    def _initialize_parameters(self):
        """Initialize network parameters."""

        # ✅ NEW: Initialize intra-option policy networks
        for option_idx, option_policy in enumerate(self.intra_option_policies):
            for layer in option_policy:
                if isinstance(layer, nn.Linear):
                    # Xavier initialization for better gradient flow
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize Q network (unchanged)
        if hasattr(self.Q, 'weight'):
            nn.init.xavier_uniform_(self.Q.weight)
            if hasattr(self.Q, 'bias') and self.Q.bias is not None:
                nn.init.zeros_(self.Q.bias)

        # Initialize termination network (unchanged)
        # Start with low termination probabilities (~0.1)
        nn.init.xavier_uniform_(self.terminations.weight)
        nn.init.constant_(self.terminations.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1
    
    def get_state(self, 
                  graph_x: torch.Tensor,
                  graph_edge_index: torch.Tensor, 
                  graph_edge_attr: torch.Tensor,
                  story_batch: torch.Tensor,
                  structure_story_ptr: Optional[List] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract state features using StateGNN.
        
        Args:
            graph_x: Node features
            graph_edge_index: Edge indices
            graph_edge_attr: Edge attributes
            story_batch: Story batch indices
            structure_story_ptr: Structure story pointers
            
        Returns:
            Tuple of (story_level_state, global_state):
            - story_level_state: [total_story_member_num, member_state_dim*2] for intra-option policies
            - global_state: [1, member_state_dim*2] for policy-over-options and termination
        """
        # Ensure inputs are on correct device
        graph_x = ensure_tensor_on_device(graph_x, self.device)
        graph_edge_index = ensure_tensor_on_device(graph_edge_index, self.device)
        graph_edge_attr = ensure_tensor_on_device(graph_edge_attr, self.device)
        story_batch = ensure_tensor_on_device(story_batch, self.device)
        
        # Extract features using StateGNN
        features = self.state_gnn(
            x=graph_x,
            edge_index=graph_edge_index,
            edge_attr=graph_edge_attr,
            batch=None,
            story_batch=story_batch,
            structure_story_ptr=structure_story_ptr
        )

        story_level_features = features
        graph_level_features_all = features[:, self.hidden_dim:]

        # StateGNN returns [total_story_member_num, member_state_dim*2]
        # For Option-Critic, we need both story-level AND global states:
        # - Story-level for intra-option policies (like DQN)
        # - Global for policy-over-options and termination functions

        # Extract global state as the first element to ensure consistent shape [1, member_state_dim]
        # This ensures Q network always receives the same input shape regardless of building size
        global_state = graph_level_features_all[0:1]  # shape: [1, member_state_dim]
        return story_level_features, global_state
    
    def get_Q(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all options using global state.

        Args:
            global_state: Global state features [1, feature_dim]

        Returns:
            Q-values for all options
        """
        return self.Q(global_state)

    def compute_Q_U(self,
                    global_state: torch.Tensor,
                    option: int,
                    action: int,
                    reward: float,
                    next_global_state: torch.Tensor,
                    gamma: float = 0.99) -> torch.Tensor:
        """
        Compute action-value Q_U(s,ω,a) for a specific (state, option, action) tuple.

        Following Option-Critic paper (Page 4):
        Q_U(s,ω,a) = r(s,a) + γ * U(ω,s')
        U(ω,s') = (1 - β(s')) * Q_Ω(s',ω) + β(s') * V_Ω(s')

        NOTE: 'action' here refers to the story member index selected.
        The action value is independent of the specific action index,
        as Q_U only depends on the reward and next state value.

        Args:
            global_state: Current global state [1, member_state_dim]
            option: Current option index
            action: Executed action (story member index) - not directly used in computation
            reward: Immediate reward r(s,a)
            next_global_state: Next global state [1, member_state_dim]
            gamma: Discount factor

        Returns:
            Q_U value (scalar tensor)
        """
        with torch.no_grad():
            # Compute termination probability β(s') for the current option
            next_beta = self.get_terminations(next_global_state)  # [1, num_options]
            if next_beta.dim() > 1:
                next_beta_omega = next_beta[0, option]  # Scalar
            else:
                next_beta_omega = next_beta[option]

            # Compute Q_Ω(s',ω) and V_Ω(s')
            next_Q = self.get_Q(next_global_state)  # [1, num_options]
            if next_Q.dim() > 1:
                next_Q_omega = next_Q[0, option]  # Q_Ω(s',ω)
                next_V = next_Q[0].max()  # V_Ω(s') = max_ω Q_Ω(s',ω) (greedy)
            else:
                next_Q_omega = next_Q[option]
                next_V = next_Q.max()

            # Compute U(ω,s') - value upon arrival
            U_omega = (1 - next_beta_omega) * next_Q_omega + next_beta_omega * next_V

            # Compute Q_U(s,ω,a) - action value
            Q_U = reward + gamma * U_omega

        return Q_U

    def compute_Q_U_batch(self,
                          global_states: List[torch.Tensor],
                          options: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_global_states: List[torch.Tensor],
                          gamma: float = 0.99) -> torch.Tensor:
        """
        Batch version of compute_Q_U for efficient processing.

        Args:
            global_states: List of global state tensors [batch_size]
            options: Option indices [batch_size]
            actions: Action indices [batch_size]
            rewards: Rewards [batch_size]
            next_global_states: List of next global state tensors [batch_size]
            gamma: Discount factor

        Returns:
            Q_U values [batch_size]
        """
        batch_size = len(options)
        Q_U_values = []

        for i in range(batch_size):
            Q_U_i = self.compute_Q_U(
                global_states[i],
                options[i].item(),
                actions[i].item(),
                rewards[i].item(),
                next_global_states[i],
                gamma
            )
            Q_U_values.append(Q_U_i)

        return torch.stack(Q_U_values)

    def get_terminations(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Get termination probabilities for all options using global state.
        
        Args:
            global_state: Global state features [1, feature_dim]
            
        Returns:
            Termination probabilities (after sigmoid)
        """
        return self.terminations(global_state).sigmoid()
    
    def predict_option_termination(self, 
                                 global_state: torch.Tensor, 
                                 current_option: int) -> Tuple[bool, int]:
        """
        Predict whether current option should terminate and select next option.
        
        Args:
            global_state: Global state features [1, feature_dim]
            current_option: Current option index
            
        Returns:
            (should_terminate, next_option_index)
        """
        # Get termination probability for current option
        termination_probs = self.get_terminations(global_state)
        
        if global_state.dim() == 2:
            # Batch of states - take mean or first sample
            termination_prob = termination_probs[0, current_option] if len(termination_probs) > 1 else termination_probs.mean(dim=0)[current_option]
        else:
            # Single state
            termination_prob = termination_probs[current_option]
        
        # Always use probabilistic sampling for consistency between training and inference
        option_termination = Bernoulli(termination_prob).sample()
        
        # Select next option greedily based on Q-values
        Q = self.get_Q(global_state)
        if global_state.dim() == 2:
            next_option = Q[0].argmax(dim=-1) if len(Q) > 1 else Q.mean(dim=0).argmax(dim=-1)
        else:
            next_option = Q.argmax(dim=-1)
        
        return bool(option_termination.item()), int(next_option.item())
    
    def get_action(self,
                   story_level_state: torch.Tensor,
                   structure_story_ptr: Optional[List[int]],
                   option: int,
                   valid_actions_mask: torch.Tensor = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action by scoring story members (NEW ARCHITECTURE v2).

        Key change: Instead of computing logits for fixed action space,
        we score each story member and select which one to modify.

        Args:
            story_level_state: Story-level state features
                              Shape: [total_story_member_num, gnn_output_dim]
                              - For 4-story building: [16, gnn_output_dim]
                              - For 7-story building: [28, gnn_output_dim]
                              - For 10-story building: [40, gnn_output_dim]
            structure_story_ptr: NOT USED in new architecture (kept for compatibility)
            option: Current option index
            valid_actions_mask: Boolean tensor indicating which story members can be modified
                              Shape: [total_story_member_num]
                              True = can modify this member, False = cannot

        Returns:
            (action_index, log_prob, entropy)
            action_index: Which story member to reduce (0 to total_story_member_num-1)
        """
        num_story_members = story_level_state.shape[0]

        if self.debug_logging:
            print(f"DEBUG get_action: num_story_members={num_story_members}, option={option}")
            if valid_actions_mask is not None:
                print(f"DEBUG get_action: valid_count={valid_actions_mask.sum().item()}/{num_story_members}")

        # =========================================================================
        # Step 1: Compute score for each story member using option-specific MLP
        # =========================================================================
        # story_level_state: [N, gnn_output_dim] where N = num_story_members
        # Output: [N, 1]
        scores = self.intra_option_policies[option](story_level_state)  # [N, 1]
        scores = scores.squeeze(-1)  # [N]

        if self.debug_logging:
            print(f"DEBUG get_action: scores range=[{scores.min().item():.3f}, {scores.max().item():.3f}]")

        # =========================================================================
        # Step 2: Apply valid actions mask
        # =========================================================================
        if valid_actions_mask is not None:
            # Ensure mask is on correct device
            if valid_actions_mask.device != scores.device:
                valid_actions_mask = valid_actions_mask.to(scores.device)

            # Ensure mask has correct shape [num_story_members]
            if valid_actions_mask.shape[0] != num_story_members:
                raise ValueError(
                    f"Mask shape mismatch: expected [{num_story_members}], "
                    f"got {valid_actions_mask.shape}"
                )

            # Mask out invalid actions with very negative scores
            scores = scores.masked_fill(~valid_actions_mask, -1e8)

            if self.debug_logging:
                print(f"DEBUG get_action: After masking, valid_scores={scores[valid_actions_mask].shape[0]}")

        # =========================================================================
        # Step 3: Convert scores to probabilities
        # =========================================================================
        action_probs = (scores / self.temperature).softmax(dim=0)  # [N]

        # Additional safety: ensure no invalid actions have probability
        if valid_actions_mask is not None:
            # Zero out probabilities for invalid actions
            action_probs = action_probs * valid_actions_mask.float()

            # Renormalize
            prob_sum = action_probs.sum()
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                # Emergency: all actions masked out
                if valid_actions_mask.sum() > 0:
                    # Uniform over valid actions
                    action_probs = valid_actions_mask.float() / valid_actions_mask.sum().float()
                else:
                    raise ValueError("No valid actions available!")

        # =========================================================================
        # Step 4: Sample action
        # =========================================================================
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # =========================================================================
        # Step 5: Validation
        # =========================================================================
        action_idx = action.item()

        # Check if action is valid
        if valid_actions_mask is not None:
            if not valid_actions_mask[action_idx].item():
                if self.debug_logging:
                    print(f"ERROR: Selected invalid action {action_idx}!")
                # Emergency fallback
                valid_indices = torch.where(valid_actions_mask)[0]
                if len(valid_indices) > 0:
                    action_idx = valid_indices[0].item()
                    logp = torch.log(action_probs[action_idx] + 1e-8)
                else:
                    raise ValueError("No valid actions and fallback failed!")

        if self.debug_logging:
            print(f"DEBUG get_action: Selected action={action_idx}, logp={logp.item():.4f}, entropy={entropy.item():.4f}")

        return action_idx, logp, entropy
    
    def greedy_option(self, global_state: torch.Tensor) -> int:
        """
        Select option greedily based on Q-values.
        
        Args:
            global_state: Global state features [1, feature_dim]
            
        Returns:
            Option index
        """
        Q = self.get_Q(global_state)
        if global_state.dim() == 2:
            return Q[0].argmax(dim=-1).item() if len(Q) > 1 else Q.mean(dim=0).argmax(dim=-1).item()
        else:
            return Q.argmax(dim=-1).item()
    
    @property
    def epsilon(self) -> float:
        """
        Get current epsilon value for exploration.
        
        Returns:
            Current epsilon value
        """
        if not self.testing:
            eps = calculate_epsilon(self.num_steps, self.eps_start, self.eps_min, self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss(model: OptionCriticGNN,
                model_prime: OptionCriticGNN,
                data_batch: Tuple,
                gamma: float = 0.99) -> torch.Tensor:
    """
    Compute critic loss for Option-Critic.

    Learns Q_Ω(s,ω) - the value of taking option ω in state s.
    Following Option-Critic paper (Page 4):

    Target: r + γ * [(1 - β(s')) * Q_Ω(s',ω) + β(s') * max_ω' Q_Ω(s',ω')]
    Loss: MSE between Q_Ω(s,ω) and target

    ✅ VERIFIED: This implementation is correct and compatible with dynamic action size.
    The critic only depends on global state, not action space size.

    Args:
        model: Current model
        model_prime: Target model
        data_batch: Batch of transitions (obs, options, rewards, next_obs, dones)
        gamma: Discount factor

    Returns:
        Critic loss (TD error)
    """
    obs, options, rewards, next_obs, dones = data_batch
    batch_size = len(options)
    batch_idx = torch.arange(batch_size).long()
    
    # Convert to tensors and move to device
    options = torch.LongTensor(options).to(model.device)
    rewards = torch.FloatTensor(rewards).to(model.device)
    masks = 1 - torch.FloatTensor(dones).to(model.device)
    
    # For graph observations, we need to handle them differently
    # Assuming obs contains graph data: (graph_x, graph_edge_index, graph_edge_attr, story_batch)
    if isinstance(obs, (tuple, list)) and len(obs) >= 4:
        graph_x, graph_edge_index, graph_edge_attr, story_batch = obs[:4]
        next_graph_x, next_graph_edge_index, next_graph_edge_attr, next_story_batch = next_obs[:4]

        # Move to model device if needed
        # Note: graph_x and other components are now lists (for dynamic graph sizes)
        if isinstance(graph_x, list):
            graph_x = [x.to(model.device) for x in graph_x]
        else:
            graph_x = graph_x.to(model.device)

        if graph_edge_index is not None:
            if isinstance(graph_edge_index, list):
                graph_edge_index = [ei.to(model.device) for ei in graph_edge_index]
            else:
                graph_edge_index = graph_edge_index.to(model.device)

        if graph_edge_attr is not None:
            if isinstance(graph_edge_attr, list):
                graph_edge_attr = [ea.to(model.device) for ea in graph_edge_attr]
            else:
                graph_edge_attr = graph_edge_attr.to(model.device)

        if story_batch is not None:
            if isinstance(story_batch, list):
                story_batch = [sb.to(model.device) for sb in story_batch]
            else:
                story_batch = story_batch.to(model.device)

        if isinstance(next_graph_x, list):
            next_graph_x = [x.to(model.device) for x in next_graph_x]
        else:
            next_graph_x = next_graph_x.to(model.device)

        if next_graph_edge_index is not None:
            if isinstance(next_graph_edge_index, list):
                next_graph_edge_index = [ei.to(model.device) for ei in next_graph_edge_index]
            else:
                next_graph_edge_index = next_graph_edge_index.to(model.device)

        if next_graph_edge_attr is not None:
            if isinstance(next_graph_edge_attr, list):
                next_graph_edge_attr = [ea.to(model.device) for ea in next_graph_edge_attr]
            else:
                next_graph_edge_attr = next_graph_edge_attr.to(model.device)

        if next_story_batch is not None:
            if isinstance(next_story_batch, list):
                next_story_batch = [sb.to(model.device) for sb in next_story_batch]
            else:
                next_story_batch = next_story_batch.to(model.device)
    else:
        # Handle tensor observations
        if isinstance(obs, (tuple, list)):
            graph_x = obs[0] if len(obs) == 1 else torch.stack(list(obs))
        else:
            graph_x = obs
        if isinstance(next_obs, (tuple, list)):
            next_graph_x = next_obs[0] if len(next_obs) == 1 else torch.stack(list(next_obs))
        else:
            next_graph_x = next_obs
        
        # Move to device
        graph_x = graph_x.to(model.device)
        next_graph_x = next_graph_x.to(model.device)
        
        graph_edge_index = graph_edge_attr = story_batch = None
        next_graph_edge_index = next_graph_edge_attr = next_story_batch = None
    
    # Process each sample individually due to graph structure
    Q_list = []
    next_Q_prime_list = []
    next_termination_probs_list = []
    
    for i in range(batch_size):
        # Current state processing
        if graph_edge_index is not None:
            # Extract single graph from batch
            # Handle list format (for dynamic graph sizes)
            if isinstance(graph_x, list):
                single_graph_x = graph_x[i]
                single_edge_index = graph_edge_index[i] if isinstance(graph_edge_index, list) else graph_edge_index
                single_edge_attr = graph_edge_attr[i] if isinstance(graph_edge_attr, list) else graph_edge_attr
                single_story_batch = story_batch[i] if isinstance(story_batch, list) else story_batch
            else:
                single_graph_x = graph_x[i:i+1]
                single_edge_index = graph_edge_index[i:i+1] if graph_edge_index.dim() > 2 else graph_edge_index
                single_edge_attr = graph_edge_attr[i:i+1] if graph_edge_attr.dim() > 2 else graph_edge_attr
                single_story_batch = story_batch[i:i+1] if story_batch.dim() > 1 else story_batch

            # Use global state for critic loss computation
            if isinstance(graph_x, list):
                _, state = model.get_state(single_graph_x, single_edge_index,
                                          single_edge_attr, single_story_batch, None)
            else:
                _, state = model.get_state(single_graph_x.squeeze(0), single_edge_index.squeeze(0),
                                          single_edge_attr.squeeze(0), single_story_batch.squeeze(0), None)
        else:
            state = model.feature_processor(graph_x[i:i+1])
        
        Q_single = model.get_Q(state)
        # Ensure Q_single is 1D with num_options elements
        if Q_single.dim() == 0:
            Q_single = Q_single.unsqueeze(0)
        if Q_single.dim() > 1:
            Q_single = Q_single.view(-1)
        Q_list.append(Q_single)
        
        # Next state processing
        if next_graph_edge_index is not None:
            # Handle list format (for dynamic graph sizes)
            if isinstance(next_graph_x, list):
                next_single_graph_x = next_graph_x[i]
                next_single_edge_index = next_graph_edge_index[i] if isinstance(next_graph_edge_index, list) else next_graph_edge_index
                next_single_edge_attr = next_graph_edge_attr[i] if isinstance(next_graph_edge_attr, list) else next_graph_edge_attr
                next_single_story_batch = next_story_batch[i] if isinstance(next_story_batch, list) else next_story_batch
            else:
                next_single_graph_x = next_graph_x[i:i+1]
                next_single_edge_index = next_graph_edge_index[i:i+1] if next_graph_edge_index.dim() > 2 else next_graph_edge_index
                next_single_edge_attr = next_graph_edge_attr[i:i+1] if next_graph_edge_attr.dim() > 2 else next_graph_edge_attr
                next_single_story_batch = next_story_batch[i:i+1] if next_story_batch.dim() > 1 else next_story_batch

            # Use global state for critic loss computation
            if isinstance(next_graph_x, list):
                _, next_state_prime = model_prime.get_state(next_single_graph_x, next_single_edge_index,
                                                           next_single_edge_attr, next_single_story_batch, None)
                _, next_state = model.get_state(next_single_graph_x, next_single_edge_index,
                                               next_single_edge_attr, next_single_story_batch, None)
            else:
                _, next_state_prime = model_prime.get_state(next_single_graph_x.squeeze(0), next_single_edge_index.squeeze(0),
                                                           next_single_edge_attr.squeeze(0), next_single_story_batch.squeeze(0), None)
                _, next_state = model.get_state(next_single_graph_x.squeeze(0), next_single_edge_index.squeeze(0),
                                               next_single_edge_attr.squeeze(0), next_single_story_batch.squeeze(0), None)
        else:
            next_state_prime = model_prime.feature_processor(next_graph_x[i:i+1])
            next_state = model.feature_processor(next_graph_x[i:i+1])
        
        next_Q_prime_single = model_prime.get_Q(next_state_prime)
        next_termination_single = model.get_terminations(next_state)
        
        # Ensure consistent dimensions
        if next_Q_prime_single.dim() == 0:
            next_Q_prime_single = next_Q_prime_single.unsqueeze(0)
        if next_Q_prime_single.dim() > 1:
            next_Q_prime_single = next_Q_prime_single.view(-1)
            
        if next_termination_single.dim() == 0:
            next_termination_single = next_termination_single.unsqueeze(0)
        if next_termination_single.dim() > 1:
            next_termination_single = next_termination_single.view(-1)
        
        next_Q_prime_list.append(next_Q_prime_single)
        next_termination_probs_list.append(next_termination_single)
    
    # Stack results
    Q = torch.stack(Q_list, dim=0)  # Shape: [batch_size, num_options]
    next_Q_prime = torch.stack(next_Q_prime_list, dim=0)  # Shape: [batch_size, num_options]  
    next_termination_probs = torch.stack(next_termination_probs_list, dim=0).detach()  # Shape: [batch_size, num_options]
    
    # Debug: print shapes (reduced frequency for step-level updates)
    # Note: These debug prints are controlled by the model's debug_logging flag
    # but we don't have access to the model instance here, so we'll make it conditional on batch size
    if batch_size <= 5:  # Only print for small batches to reduce noise
        # Optional debug prints - these can be commented out in production
        pass  # Replaced with logger-based logging in calling code
    
    # Ensure all have correct batch dimension
    if Q.dim() == 1:
        Q = Q.unsqueeze(0).expand(batch_size, -1)
    if next_Q_prime.dim() == 1:
        next_Q_prime = next_Q_prime.unsqueeze(0).expand(batch_size, -1)
    if next_termination_probs.dim() == 1:
        next_termination_probs = next_termination_probs.unsqueeze(0).expand(batch_size, -1)
    
    # Select Q-values for chosen options
    next_options_term_prob = next_termination_probs[batch_idx, options]
    next_Q_option = next_Q_prime[batch_idx, options]
    next_Q_max = next_Q_prime.max(dim=-1)[0]
    Q_option = Q[batch_idx, options]
    
    # Compute target - ensure all tensors have same shape
    next_options_term_prob = next_options_term_prob.view(-1)
    next_Q_option = next_Q_option.view(-1)  
    next_Q_max = next_Q_max.view(-1)
    Q_option = Q_option.view(-1)
    rewards = rewards.view(-1)
    masks = masks.view(-1)
    
    gt = rewards + masks * gamma * ((1 - next_options_term_prob) * next_Q_option + next_options_term_prob * next_Q_max)
    
    td_err = (Q_option - gt.detach()).pow(2).mul(0.5).mean()
    return td_err


def actor_loss(obs, option: int, action: int, logp: torch.Tensor, entropy: torch.Tensor,
               reward: float, done: bool, next_obs,
               model: OptionCriticGNN, model_prime: OptionCriticGNN,
               gamma: float = 0.99, termination_reg: float = 0.01,
               entropy_reg: float = 0.01) -> torch.Tensor:
    """
    Compute actor loss for Option-Critic following original paper.

    Combines:
    1. Intra-option policy gradient (Theorem 1): -log π(a|s,ω) * Q_U(s,ω,a)
    2. Termination gradient (Theorem 2): β(s',ω) * [A_Ω(s',ω) + ε]

    Args:
        obs: Current observation (graph data tuple)
        option: Current option index
        action: Executed action (story member index) - NEWLY ADDED
        logp: Log probability of selected action
        entropy: Entropy of action distribution
        reward: Reward received
        done: Whether episode terminated
        next_obs: Next observation (graph data tuple)
        model: Current model
        model_prime: Target model
        gamma: Discount factor
        termination_reg: Termination regularization weight (ε in paper)
        entropy_reg: Entropy regularization weight

    Returns:
        Actor loss (combines policy gradient and termination gradient)
    """
    # =========================================================================
    # Extract global states from graph observations
    # =========================================================================
    if isinstance(obs, (tuple, list)) and len(obs) >= 4:
        graph_x, graph_edge_index, graph_edge_attr, story_batch = obs[:4]
        next_graph_x, next_graph_edge_index, next_graph_edge_attr, next_story_batch = next_obs[:4]

        # Use global state for Q_U and termination computation
        _, state = model.get_state(graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
        _, next_state = model.get_state(next_graph_x, next_graph_edge_index,
                                       next_graph_edge_attr, next_story_batch, None)
        _, next_state_prime = model_prime.get_state(next_graph_x, next_graph_edge_index,
                                                   next_graph_edge_attr, next_story_batch, None)
    else:
        # Fallback for non-graph observations
        if isinstance(obs, (tuple, list)):
            obs = obs[0] if len(obs) == 1 else torch.stack(list(obs))
        if isinstance(next_obs, (tuple, list)):
            next_obs = next_obs[0] if len(next_obs) == 1 else torch.stack(list(next_obs))

        state = model.feature_processor(obs) if hasattr(model, 'feature_processor') else obs
        next_state = model.feature_processor(next_obs) if hasattr(model, 'feature_processor') else next_obs
        next_state_prime = model_prime.feature_processor(next_obs) if hasattr(model_prime, 'feature_processor') else next_obs

    # =========================================================================
    # Part 1: Intra-Option Policy Gradient (Theorem 1)
    # =========================================================================
    # Compute Q_U(s,ω,a) using the executed action
    Q_U = model.compute_Q_U(
        global_state=state,
        option=option,
        action=action,
        reward=reward,
        next_global_state=next_state_prime,
        gamma=gamma
    )

    # Policy gradient: -log π(a|s,ω) * Q_U(s,ω,a)
    # (Negative because we're minimizing loss, equivalent to maximizing objective)
    policy_loss = -logp * Q_U.detach() - entropy_reg * entropy

    # =========================================================================
    # Part 2: Termination Gradient (Theorem 2)
    # =========================================================================
    # Get termination probability for current option at next state
    next_termination_probs = model.get_terminations(next_state)  # [1, num_options] or [num_options]

    # Handle dimensions
    if next_termination_probs.dim() > 1:
        next_beta_omega = next_termination_probs[0, option]
    else:
        next_beta_omega = next_termination_probs[option]

    # Compute advantage A_Ω(s',ω) = Q_Ω(s',ω) - V_Ω(s')
    next_Q = model.get_Q(next_state).detach()  # [1, num_options] or [num_options]

    if next_Q.dim() > 1:
        next_Q_omega = next_Q[0, option]  # Q_Ω(s',ω)
        # V_Ω(s') = max_ω Q_Ω(s',ω) for greedy policy-over-options
        next_V = next_Q[0].max()
    else:
        next_Q_omega = next_Q[option]
        next_V = next_Q.max()

    advantage_omega = next_Q_omega - next_V

    # Termination gradient: β(s',ω) * [A_Ω(s',ω) + ε]
    # Only applies if episode hasn't terminated
    termination_loss = next_beta_omega * (advantage_omega + termination_reg) * (1 - done)

    # =========================================================================
    # Combine losses
    # =========================================================================
    # Ensure all components are scalars
    if policy_loss.dim() > 0:
        policy_loss = policy_loss.mean()
    if termination_loss.dim() > 0:
        termination_loss = termination_loss.mean()

    total_actor_loss = policy_loss + termination_loss

    # Optional debug logging
    if hasattr(model, 'debug_logging') and model.debug_logging:
        print(f"[ActorLoss] Policy: {policy_loss.item():.6f}, "
              f"Termination: {termination_loss.item():.6f}, "
              f"Total: {total_actor_loss.item():.6f}")
        print(f"[ActorLoss] Q_U: {Q_U.item():.4f}, "
              f"Advantage: {advantage_omega.item():.4f}, "
              f"β: {next_beta_omega.item():.4f}")

    return total_actor_loss