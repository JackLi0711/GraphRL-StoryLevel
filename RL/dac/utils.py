"""
DAC Utilities Module

This module provides utility functions for DAC training and evaluation.
"""

import torch
import numpy as np
import random
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor(x, device: str = 'cpu'):
    """Convert input to tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def to_numpy(x):
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ensure_tensor_on_device(x, device):
    """Ensure input is tensor on specified device."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return x.to(device)


class DACUtils:
    """Utility class for DAC operations."""

    @staticmethod
    def save_config(config, filepath: Path):
        """Save configuration to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    @staticmethod
    def load_config(filepath: Path):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    @staticmethod
    def count_parameters(model):
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_device():
        """Get available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @staticmethod
    def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
        """Normalize advantages."""
        if advantages.numel() <= 1:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    @staticmethod
    def clip_gradients(model, max_norm: float):
        """Clip gradients by norm."""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    @staticmethod
    def linear_schedule(initial_value: float, final_value: float,
                       current_step: int, total_steps: int) -> float:
        """Linear learning rate schedule."""
        if current_step >= total_steps:
            return final_value

        alpha = current_step / total_steps
        return initial_value * (1 - alpha) + final_value * alpha

    @staticmethod
    def exponential_schedule(initial_value: float, decay_rate: float,
                           current_step: int) -> float:
        """Exponential decay schedule."""
        return initial_value * (decay_rate ** current_step)

    @staticmethod
    def cosine_schedule(initial_value: float, final_value: float,
                       current_step: int, total_steps: int) -> float:
        """Cosine annealing schedule."""
        if current_step >= total_steps:
            return final_value

        alpha = current_step / total_steps
        cosine_factor = 0.5 * (1 + np.cos(np.pi * alpha))
        return final_value + (initial_value - final_value) * cosine_factor


class RunningMeanStd:
    """Running mean and standard deviation tracker."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class StateNormalizer:
    """State normalization utility."""

    def __init__(self, shape, clip_range=10.0):
        self.rms = RunningMeanStd(shape)
        self.clip_range = clip_range
        self.read_only = False

    def __call__(self, x):
        if not self.read_only:
            self.rms.update(x)

        normalized = (x - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False


class RewardNormalizer:
    """Reward normalization utility."""

    def __init__(self, clip_range=10.0, gamma=0.99):
        self.clip_range = clip_range
        self.gamma = gamma
        self.returns = 0
        self.ret_rms = RunningMeanStd(shape=())

    def __call__(self, rewards):
        self.returns = self.returns * self.gamma + rewards
        self.ret_rms.update(np.array([self.returns]))

        normalized = rewards / np.sqrt(self.ret_rms.var + 1e-8)
        return np.clip(normalized, -self.clip_range, self.clip_range)


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def restore_checkpoint(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class MovingAverage:
    """Moving average tracker."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_average(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []


def create_logger(name: str, log_file: Optional[Path] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_gae(rewards: List[float], values: List[float],
                 dones: List[bool], gamma: float = 0.99,
                 gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Calculate Generalized Advantage Estimation.

    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    returns = []
    gae = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]

        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return advantages, returns


def save_training_curves(metrics: Dict[str, List[float]],
                        save_dir: Path, filename: str = "training_curves.png"):
    """Save training curves plot."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (metric_name, values) in enumerate(metrics.items()):
        if i >= len(axes):
            break

        axes[i].plot(values)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel('Episode/Update')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)

    plt.tight_layout()
    save_path = save_dir / filename
    plt.savefig(save_path)
    plt.close()
        