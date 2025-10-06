import numpy as np
import random
import torch
from collections import deque
from typing import Tuple, List


class ReplayBuffer(object):
    """
    Experience replay buffer for Option-Critic with graph observations.
    Handles graph-structured state representations.
    """
    
    def __init__(self, capacity: int, seed: int = 42):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option: int, reward: float, next_obs, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Current observation (graph data tuple or tensor)
            option: Selected option index
            reward: Reward received
            next_obs: Next observation (graph data tuple or tensor)
            done: Whether episode terminated
        """
        # Handle graph data tuples
        def move_to_cpu(data):
            if isinstance(data, (tuple, list)):
                return tuple(t.detach().cpu() if torch.is_tensor(t) else t for t in data)
            elif torch.is_tensor(data):
                return data.detach().cpu()
            else:
                return data
        
        obs = move_to_cpu(obs)
        next_obs = move_to_cpu(next_obs)
        
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (observations, options, rewards, next_observations, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        batch = self.rng.sample(self.buffer, batch_size)
        obs, option, reward, next_obs, done = zip(*batch)

        # Handle graph data tuples or single tensors
        def stack_observations(obs_list):
            if isinstance(obs_list[0], (tuple, list)):
                # Graph data tuples - for dynamic graph sizes (random structure_shape),
                # we keep graph components as lists instead of stacking them.
                # This allows graphs with different numbers of nodes/edges to coexist in a batch.
                stacked = []
                for i in range(len(obs_list[0])):
                    components = [obs[i] for obs in obs_list]
                    # Keep graph tensors as list (for graph_x, edge_index, edge_attr, story_batch)
                    # The loss functions (critic_loss, actor_loss) will process them individually
                    if components[0] is not None:
                        # Convert to tensors if needed, but keep as list
                        tensor_components = [torch.tensor(c) if not torch.is_tensor(c) else c
                                           for c in components]
                        stacked.append(tensor_components)
                    else:
                        stacked.append([None] * len(components))
                return tuple(stacked)
            else:
                # Single tensor observations (non-graph case)
                if torch.is_tensor(obs_list[0]):
                    return torch.stack(obs_list)
                else:
                    return np.stack(obs_list)

        obs_stacked = stack_observations(obs)
        next_obs_stacked = stack_observations(next_obs)

        return obs_stacked, option, reward, next_obs_stacked, done

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size