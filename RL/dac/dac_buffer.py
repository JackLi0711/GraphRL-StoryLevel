"""
DAC Experience Buffer Module for PPO

This module implements experience replay buffers for the Double Actor-Critic
architecture with PPO optimization. It handles both high-level (option) and
low-level (action) experiences separately.

Based on the ASquaredC_PPO_agent storage system and adapted for DAC.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, deque


@dataclass
class HighLevelExperience:
    """High-level experience for option selection (hat MDP)."""
    global_state: torch.Tensor
    option: int
    option_reward: float  # Includes length bonus
    next_global_state: torch.Tensor
    done: bool
    log_prob: float
    value: float
    advantage: Optional[float] = None
    return_value: Optional[float] = None


@dataclass
class LowLevelExperience:
    """Low-level experience for action selection (bar MDP)."""
    story_state: torch.Tensor
    option: int
    action: torch.Tensor
    reward: float
    next_story_state: torch.Tensor
    done: bool
    log_prob: float
    value: float
    advantage: Optional[float] = None
    return_value: Optional[float] = None


class DACStorage:
    """
    Storage for DAC experiences during rollout collection.
    Similar to ASquaredC_PPO_agent Storage class but adapted for DAC.
    """

    def __init__(self, rollout_length: int, device: str = 'cpu'):
        """
        Initialize DAC storage.

        Args:
            rollout_length: Number of steps to store
            device: Device to store tensors on
        """
        self.rollout_length = rollout_length
        self.device = torch.device(device)
        self.step = 0

        # Storage dictionaries
        self.data = defaultdict(list)

        # Additional storage for advantages and returns
        self.additional_keys = ['adv_hat', 'adv_bar', 'ret_hat', 'ret_bar']
        for key in self.additional_keys:
            self.data[key] = [None] * rollout_length

    def add(self, data_dict: Dict[str, Any]):
        """Add a step of data to storage."""
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
            self.data[key].append(value)

    def placeholder(self):
        """Add placeholder for final step (bootstrap value)."""
        self.step += 1

    def cat(self, keys: List[str]) -> List[Union[torch.Tensor, List]]:
        """
        Concatenate stored data for given keys.

        Args:
            keys: List of keys to concatenate

        Returns:
            List of concatenated tensors or lists (for graph data)
        """
        result = []
        for key in keys:
            if key in self.data:
                data_list = self.data[key]
                if data_list and data_list[0] is not None:
                    if isinstance(data_list[0], torch.Tensor):
                        result.append(torch.cat(data_list, dim=0))
                    elif isinstance(data_list[0], dict):
                        # Handle graph data dictionaries - return the list as is
                        # The agent will handle batching of graph data separately
                        result.append(data_list)
                    else:
                        result.append(torch.tensor(data_list, device=self.device))
                else:
                    result.append(torch.empty(0, device=self.device))
            else:
                result.append(torch.empty(0, device=self.device))
        return result

    def get_advantages_and_returns(self, mdp: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get advantages and returns for specified MDP."""
        adv_key = f'adv_{mdp}'
        ret_key = f'ret_{mdp}'

        advantages = []
        returns = []

        for i in range(len(self.data[adv_key])):
            if self.data[adv_key][i] is not None:
                advantages.append(self.data[adv_key][i])
                returns.append(self.data[ret_key][i])

        if advantages:
            return torch.cat(advantages), torch.cat(returns)
        else:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)

    def reset(self):
        """Reset storage for new rollout."""
        self.data.clear()
        self.step = 0
        for key in self.additional_keys:
            self.data[key] = [None] * self.rollout_length


class DACExperienceBuffer:
    """
    DAC Experience Buffer for PPO training.

    Manages separate buffers for high-level and low-level experiences,
    with support for option length bonus computation.
    """

    def __init__(self,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 length_bonus_weight: float = 0.1,
                 device: str = 'cpu'):
        """
        Initialize DAC experience buffer.

        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Batch size for sampling
            length_bonus_weight: Weight for option length bonus
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.length_bonus_weight = length_bonus_weight
        self.device = torch.device(device)

        # Separate buffers for high and low level
        self.high_level_buffer: List[HighLevelExperience] = []
        self.low_level_buffer: List[LowLevelExperience] = []

        # Option tracking for length bonus
        self.current_option_length = defaultdict(int)
        self.option_start_step = defaultdict(int)

        # Statistics
        self.total_high_experiences = 0
        self.total_low_experiences = 0

    def add_high_level_experience(self,
                                global_state: torch.Tensor,
                                option: int,
                                option_reward: float,
                                next_global_state: torch.Tensor,
                                done: bool,
                                log_prob: float,
                                value: float):
        """Add high-level experience to buffer."""
        experience = HighLevelExperience(
            global_state=global_state.to(self.device),
            option=option,
            option_reward=option_reward,
            next_global_state=next_global_state.to(self.device),
            done=done,
            log_prob=log_prob,
            value=value
        )

        self.high_level_buffer.append(experience)
        self.total_high_experiences += 1

        # Maintain buffer size
        if len(self.high_level_buffer) > self.buffer_size:
            self.high_level_buffer.pop(0)

    def add_low_level_experience(self,
                               story_state: torch.Tensor,
                               option: int,
                               action: torch.Tensor,
                               reward: float,
                               next_story_state: torch.Tensor,
                               done: bool,
                               log_prob: float,
                               value: float):
        """Add low-level experience to buffer."""
        experience = LowLevelExperience(
            story_state=story_state.to(self.device),
            option=option,
            action=action.to(self.device),
            reward=reward,
            next_story_state=next_story_state.to(self.device),
            done=done,
            log_prob=log_prob,
            value=value
        )

        self.low_level_buffer.append(experience)
        self.total_low_experiences += 1

        # Maintain buffer size
        if len(self.low_level_buffer) > self.buffer_size:
            self.low_level_buffer.pop(0)

    def compute_option_length_bonus(self, option: int, step: int) -> float:
        """
        Compute option length bonus as described in DAC paper.

        Args:
            option: Option index
            step: Current step

        Returns:
            Length bonus for the option
        """
        if option not in self.current_option_length:
            self.current_option_length[option] = 0
            self.option_start_step[option] = step

        self.current_option_length[option] += 1
        length_bonus = self.length_bonus_weight * self.current_option_length[option]

        return length_bonus

    def terminate_option(self, option: int, final_reward: float, step: int) -> float:
        """
        Terminate an option and return final reward with length bonus.

        Args:
            option: Option that terminated
            final_reward: Base reward for the option
            step: Current step

        Returns:
            Final reward including length bonus
        """
        length_bonus = self.compute_option_length_bonus(option, step)
        total_reward = final_reward + length_bonus

        # Reset option tracking
        if option in self.current_option_length:
            del self.current_option_length[option]
        if option in self.option_start_step:
            del self.option_start_step[option]

        return total_reward

    def compute_gae_advantages(self,
                             experiences: List,
                             gamma: float = 0.99,
                             gae_lambda: float = 0.95) -> List:
        """
        Compute Generalized Advantage Estimation for experiences.

        Args:
            experiences: List of experiences (either high or low level)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            List of experiences with computed advantages and returns
        """
        if not experiences:
            return experiences

        # Compute advantages using GAE
        advantages = []
        returns = []
        gae = 0

        # Work backwards through experiences
        for i in reversed(range(len(experiences))):
            exp = experiences[i]

            if i == len(experiences) - 1:
                # Last experience
                next_value = 0 if exp.done else exp.value  # Bootstrap value
            else:
                next_value = experiences[i + 1].value

            # Temporal difference error
            delta = exp.reward + gamma * next_value * (not exp.done) - exp.value

            # GAE computation
            gae = delta + gamma * gae_lambda * (not exp.done) * gae
            advantages.insert(0, gae)

            # Return computation
            return_val = gae + exp.value
            returns.insert(0, return_val)

        # Update experiences with advantages and returns
        for i, exp in enumerate(experiences):
            exp.advantage = advantages[i]
            exp.return_value = returns[i]

        return experiences

    def sample_high_level_batch(self) -> Optional[List[HighLevelExperience]]:
        """Sample batch from high-level buffer."""
        if len(self.high_level_buffer) < self.batch_size:
            return None

        indices = np.random.choice(len(self.high_level_buffer), self.batch_size, replace=False)
        return [self.high_level_buffer[i] for i in indices]

    def sample_low_level_batch(self) -> Optional[List[LowLevelExperience]]:
        """Sample batch from low-level buffer."""
        if len(self.low_level_buffer) < self.batch_size:
            return None

        indices = np.random.choice(len(self.low_level_buffer), self.batch_size, replace=False)
        return [self.low_level_buffer[i] for i in indices]

    def get_all_high_level_experiences(self) -> List[HighLevelExperience]:
        """Get all high-level experiences."""
        return self.high_level_buffer.copy()

    def get_all_low_level_experiences(self) -> List[LowLevelExperience]:
        """Get all low-level experiences."""
        return self.low_level_buffer.copy()

    def clear(self):
        """Clear all buffers."""
        self.high_level_buffer.clear()
        self.low_level_buffer.clear()
        self.current_option_length.clear()
        self.option_start_step.clear()

    def clear_high_level(self):
        """Clear only high-level buffer."""
        self.high_level_buffer.clear()

    def clear_low_level(self):
        """Clear only low-level buffer."""
        self.low_level_buffer.clear()

    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'high_level_size': len(self.high_level_buffer),
            'low_level_size': len(self.low_level_buffer),
            'total_high_experiences': self.total_high_experiences,
            'total_low_experiences': self.total_low_experiences,
            'active_options': len(self.current_option_length),
            'current_option_lengths': dict(self.current_option_length)
        }


def random_sample(indices: np.ndarray, batch_size: int):
    """
    Utility function for random sampling.
    Compatible with ASquaredC_PPO_agent sampling.
    """
    indices = np.asarray(indices)
    for _ in range(len(indices) // batch_size):
        batch_indices = np.random.choice(indices, batch_size, replace=False)
        yield batch_indices