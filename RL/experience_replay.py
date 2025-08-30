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
            obs: Current observation (graph state tensor)
            option: Selected option index
            reward: Reward received
            next_obs: Next observation (graph state tensor)
            done: Whether episode terminated
        """
        # Store tensors on CPU to save GPU memory
        if torch.is_tensor(obs):
            obs = obs.detach().cpu()
        if torch.is_tensor(next_obs):
            next_obs = next_obs.detach().cpu()
            
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
        
        # Stack graph observations
        if torch.is_tensor(obs[0]):
            obs = torch.stack(obs)
            next_obs = torch.stack(next_obs)
        else:
            obs = np.stack(obs)
            next_obs = np.stack(next_obs)
        
        return obs, option, reward, next_obs, done

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size