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
    Integrates StateGNN for processing graph-structured observations.
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
                 testing: bool = False):
        """
        Initialize OptionCriticGNN.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN
            member_state_dim: Member state dimension
            num_layers: Number of GNN layers
            num_actions: Number of primitive actions
            num_options: Number of options
            temperature: Temperature for action selection
            eps_start: Starting epsilon for exploration
            eps_min: Minimum epsilon
            eps_decay: Epsilon decay rate
            eps_test: Epsilon for testing
            device: Device to use
            testing: Whether in testing mode
        """
        super(OptionCriticGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.member_state_dim = member_state_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = torch.device(device)
        self.testing = testing
        
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
        
        # Feature processing layers
        # StateGNN outputs shape: [total story_member_num, member_state_dim*2]
        gnn_output_dim = member_state_dim * 2
        feature_dim = 512
        
        self.feature_processor = nn.Sequential(
            nn.Linear(gnn_output_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        
        # Option-Critic components
        self.Q = nn.Linear(feature_dim, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(feature_dim, num_options)  # Option-Termination
        
        # Intra-option policies (one for each option)
        self.options_W = nn.Parameter(torch.zeros(num_options, feature_dim, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Move to device
        self.to(self.device)
        self.train(not testing)
    
    def _initialize_parameters(self):
        """Initialize network parameters."""
        # Initialize option policy parameters
        nn.init.normal_(self.options_W, mean=0.0, std=0.01)
        nn.init.zeros_(self.options_b)
        
        # Initialize other layers
        for module in [self.feature_processor, self.Q, self.terminations]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            elif hasattr(module, 'children'):
                for child in module.children():
                    if hasattr(child, 'weight'):
                        nn.init.xavier_uniform_(child.weight)
    
    def get_state(self, 
                  graph_x: torch.Tensor,
                  graph_edge_index: torch.Tensor, 
                  graph_edge_attr: torch.Tensor,
                  story_batch: torch.Tensor,
                  structure_story_ptr: Optional[List] = None) -> torch.Tensor:
        """
        Extract state features using StateGNN.
        
        Args:
            graph_x: Node features
            graph_edge_index: Edge indices
            graph_edge_attr: Edge attributes
            story_batch: Story batch indices
            structure_story_ptr: Structure story pointers
            
        Returns:
            Processed state features
        """
        # Ensure inputs are on correct device
        graph_x = ensure_tensor_on_device(graph_x, self.device)
        graph_edge_index = ensure_tensor_on_device(graph_edge_index, self.device)
        graph_edge_attr = ensure_tensor_on_device(graph_edge_attr, self.device)
        story_batch = ensure_tensor_on_device(story_batch, self.device)
        
        # Extract features using StateGNN
        gnn_features = self.state_gnn(
            x=graph_x,
            edge_index=graph_edge_index,
            edge_attr=graph_edge_attr,
            batch=None,
            story_batch=story_batch,
            structure_story_ptr=structure_story_ptr
        )
        
        # Process features for Option-Critic
        state = self.feature_processor(gnn_features)
        
        return state
    
    def get_Q(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all options.
        
        Args:
            state: State features
            
        Returns:
            Q-values for all options
        """
        return self.Q(state)
    
    def get_terminations(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get termination probabilities for all options.
        
        Args:
            state: State features
            
        Returns:
            Termination probabilities (after sigmoid)
        """
        return self.terminations(state).sigmoid()
    
    def predict_option_termination(self, 
                                 state: torch.Tensor, 
                                 current_option: int) -> Tuple[bool, int]:
        """
        Predict whether current option should terminate and select next option.
        
        Args:
            state: State features
            current_option: Current option index
            
        Returns:
            (should_terminate, next_option_index)
        """
        # Get termination probability for current option
        termination_probs = self.get_terminations(state)
        
        if state.dim() == 2:
            # Batch of states - take mean or first sample
            termination_prob = termination_probs[0, current_option] if len(termination_probs) > 1 else termination_probs.mean(dim=0)[current_option]
        else:
            # Single state
            termination_prob = termination_probs[current_option]
        
        # Sample termination decision
        option_termination = Bernoulli(termination_prob).sample()
        
        # Select next option greedily based on Q-values
        Q = self.get_Q(state)
        if state.dim() == 2:
            next_option = Q[0].argmax(dim=-1) if len(Q) > 1 else Q.mean(dim=0).argmax(dim=-1)
        else:
            next_option = Q.argmax(dim=-1)
        
        return bool(option_termination.item()), int(next_option.item())
    
    def get_action(self, 
                   state: torch.Tensor, 
                   option: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action within an option using intra-option policy.
        
        Args:
            state: State features
            option: Current option index
            
        Returns:
            (action, log_probability, entropy)
        """
        # Handle batch dimension - ensure we get a single sample
        if state.dim() == 2 and state.shape[0] > 1:
            # Take the mean across batch dimension for action selection
            state_for_action = state.mean(dim=0)
        elif state.dim() == 2:
            # Single batch item
            state_for_action = state[0]
        else:
            # Already single state
            state_for_action = state
        
        # Ensure state_for_action is 1D
        if state_for_action.dim() > 1:
            state_for_action = state_for_action.flatten()
        
        # Compute action logits for the given option
        logits = state_for_action @ self.options_W[option] + self.options_b[option]
        
        # Apply temperature and get action distribution
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)
        
        # Sample action
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        # Ensure logp and entropy are scalars
        if logp.dim() > 0:
            logp = logp.mean()
        if entropy.dim() > 0:
            entropy = entropy.mean()
        
        return action.item(), logp, entropy
    
    def greedy_option(self, state: torch.Tensor) -> int:
        """
        Select option greedily based on Q-values.
        
        Args:
            state: State features
            
        Returns:
            Option index
        """
        Q = self.get_Q(state)
        if state.dim() == 2:
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
    
    Args:
        model: Current model
        model_prime: Target model
        data_batch: Batch of transitions
        gamma: Discount factor
        
    Returns:
        Critic loss
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
        graph_edge_index = graph_edge_attr = story_batch = None
        next_graph_edge_index = next_graph_edge_attr = next_story_batch = None
    
    # Get current state and Q-values
    if graph_edge_index is not None:
        states = model.get_state(graph_x, graph_edge_index, graph_edge_attr, story_batch)
    else:
        states = model.feature_processor(graph_x)
    Q = model.get_Q(states)
    
    # Get next state Q-values using target network
    if next_graph_edge_index is not None:
        next_states_prime = model_prime.get_state(next_graph_x, next_graph_edge_index, 
                                                next_graph_edge_attr, next_story_batch)
        next_states = model.get_state(next_graph_x, next_graph_edge_index, 
                                    next_graph_edge_attr, next_story_batch)
    else:
        next_states_prime = model_prime.feature_processor(next_graph_x)
        next_states = model.feature_processor(next_graph_x)
    
    next_Q_prime = model_prime.get_Q(next_states_prime)
    next_termination_probs = model.get_terminations(next_states).detach()
    
    # Handle batch dimension for termination probabilities
    if next_termination_probs.dim() > 1 and len(options) == next_termination_probs.shape[0]:
        next_options_term_prob = next_termination_probs[batch_idx, options]
    else:
        # Average over batch dimension if needed
        next_options_term_prob = next_termination_probs.mean(dim=0)[options] if next_termination_probs.dim() > 1 else next_termination_probs[options]
    
    # Compute target
    if next_Q_prime.dim() > 1 and len(options) == next_Q_prime.shape[0]:
        next_Q_option = next_Q_prime[batch_idx, options]
        next_Q_max = next_Q_prime.max(dim=-1)[0]
    else:
        next_Q_option = next_Q_prime.mean(dim=0)[options] if next_Q_prime.dim() > 1 else next_Q_prime[options]
        next_Q_max = next_Q_prime.max(dim=-1)[0] if next_Q_prime.dim() > 1 else next_Q_prime.max()
    
    gt = rewards + masks * gamma * ((1 - next_options_term_prob) * next_Q_option + next_options_term_prob * next_Q_max)
    
    # Compute TD error
    if Q.dim() > 1 and len(options) == Q.shape[0]:
        Q_option = Q[batch_idx, options]
    else:
        Q_option = Q.mean(dim=0)[options] if Q.dim() > 1 else Q[options]
    
    td_err = (Q_option - gt.detach()).pow(2).mul(0.5).mean()
    return td_err


def actor_loss(obs, option: int, logp: torch.Tensor, entropy: torch.Tensor, 
               reward: float, done: bool, next_obs, 
               model: OptionCriticGNN, model_prime: OptionCriticGNN,
               gamma: float = 0.99, termination_reg: float = 0.01, 
               entropy_reg: float = 0.01) -> torch.Tensor:
    """
    Compute actor loss for Option-Critic.
    
    Args:
        obs: Current observation
        option: Current option
        logp: Log probability of selected action
        entropy: Entropy of action distribution
        reward: Reward received
        done: Whether episode terminated
        next_obs: Next observation
        model: Current model
        model_prime: Target model
        gamma: Discount factor
        termination_reg: Termination regularization weight
        entropy_reg: Entropy regularization weight
        
    Returns:
        Actor loss
    """
    # Handle graph observations
    if isinstance(obs, (tuple, list)) and len(obs) >= 4:
        graph_x, graph_edge_index, graph_edge_attr, story_batch = obs[:4]
        next_graph_x, next_graph_edge_index, next_graph_edge_attr, next_story_batch = next_obs[:4]
        
        state = model.get_state(graph_x, graph_edge_index, graph_edge_attr, story_batch)
        next_state = model.get_state(next_graph_x, next_graph_edge_index, 
                                   next_graph_edge_attr, next_story_batch)
        next_state_prime = model_prime.get_state(next_graph_x, next_graph_edge_index, 
                                                next_graph_edge_attr, next_story_batch)
    else:
        # For tensor observations, convert to tensor if needed
        if isinstance(obs, (tuple, list)):
            obs = obs[0] if len(obs) == 1 else torch.stack(list(obs))
        if isinstance(next_obs, (tuple, list)):
            next_obs = next_obs[0] if len(next_obs) == 1 else torch.stack(list(next_obs))
            
        state = model.feature_processor(obs)
        next_state = model.feature_processor(next_obs)
        next_state_prime = model_prime.feature_processor(next_obs)
    
    # Get termination probabilities
    option_term_prob = model.get_terminations(state)
    next_option_term_prob = model.get_terminations(next_state).detach()
    
    # Handle dimensions
    if option_term_prob.dim() > 1:
        option_term_prob = option_term_prob.mean(dim=0)
    if next_option_term_prob.dim() > 1:
        next_option_term_prob = next_option_term_prob.mean(dim=0)
    
    option_term_prob = option_term_prob[option]
    next_option_term_prob = next_option_term_prob[option]
    
    # Get Q-values
    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()
    
    # Handle dimensions
    if Q.dim() == 0:
        Q = Q.unsqueeze(0)
    if next_Q_prime.dim() == 0:
        next_Q_prime = next_Q_prime.unsqueeze(0)
    
    # Compute target
    gt = reward + (1 - done) * gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + 
         next_option_term_prob * next_Q_prime.max())
    
    # Termination loss
    termination_loss = option_term_prob * (Q[option] - Q.max() + termination_reg) * (1 - done)
    
    # Policy gradient loss with entropy regularization
    advantage = gt.detach() - Q[option]
    policy_loss = -logp * advantage - entropy_reg * entropy
    
    # Ensure all components are scalars
    if termination_loss.dim() > 0:
        termination_loss = termination_loss.mean()
    if policy_loss.dim() > 0:
        policy_loss = policy_loss.mean()
    
    actor_loss = termination_loss + policy_loss
    return actor_loss