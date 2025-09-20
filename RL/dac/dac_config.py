"""
DAC Configuration Management Module

This module provides configuration classes for the Double Actor-Critic (DAC)
implementation with PPO optimization, specifically designed for structural
design optimization with GNN.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import datetime


@dataclass
class DACConfig:
    """
    Configuration class for DAC (Double Actor-Critic) with PPO optimization.

    Based on the ASquaredC_PPO_agent implementation and adapted for GNN-based
    structural design optimization.
    """

    # ========== Network Architecture ==========
    node_feature_dim: int = 32
    edge_feature_dim: int = 16
    hidden_dim: int = 256
    member_state_dim: int = 128
    num_gnn_layers: int = 3
    num_actions: int = 14  # Based on structural design action space
    num_options: int = 4   # Number of options for hierarchical RL

    # ========== PPO Hyperparameters ==========
    # Learning rates
    learning_rate: float = 3e-4
    high_level_lr: float = 3e-4    # High-level (option selection) learning rate
    low_level_lr: float = 3e-4     # Low-level (action selection) learning rate

    # PPO specific parameters
    ppo_ratio_clip: float = 0.2    # PPO clipping parameter
    optimization_epochs: int = 4   # Number of optimization epochs per update
    mini_batch_size: int = 64      # Mini-batch size for SGD
    rollout_length: int = 128      # Steps to collect before update

    # Loss weights
    entropy_weight: float = 0.01   # Entropy regularization weight
    beta_weight: float = 0.01      # Option termination regularization weight
    value_loss_coef: float = 0.5   # Value function loss coefficient

    # Advantage estimation
    use_gae: bool = True           # Use Generalized Advantage Estimation
    gae_tau: float = 0.95          # GAE lambda parameter
    discount: float = 0.99         # Discount factor (gamma)

    # Gradient clipping
    gradient_clip: float = 0.5     # Gradient clipping threshold

    # ========== DAC Specific Parameters ==========
    # Option length bonus (from DAC paper)
    length_bonus_weight: float = 0.1    # Weight for option length bonus

    # Termination parameters
    termination_reg: float = 0.01       # Termination regularization

    # Learning schedule
    learning_schedule: str = "all"      # "all", "alt" (alternating), "sequential"
    freeze_value_function: bool = False # Whether to freeze value function occasionally

    # ========== Environment Parameters ==========
    num_workers: int = 1                # Number of parallel environments
    structure_shape: str = "fixed"     # "fixed", "small_random", "random"
    add_structure_geometry: bool = True
    add_response_features: bool = True
    reward_type: str = "material_usage"
    scwb_driven_design: bool = False
    do_nonlinear_dynamic_analysis: bool = False
    check_acceleration: bool = False
    check_displacement: bool = True

    # ========== Training Parameters ==========
    max_episodes: int = 10              # Maximum training episodes
    max_steps_per_episode: int = 500    # Maximum steps per episode
    save_interval: int = 100            # Model save interval (episodes)
    eval_interval: int = 50             # Evaluation interval (episodes)
    log_interval: int = 10              # Logging interval (episodes)

    # Evaluation parameters (like option_critic)
    eval_frequency: int = 10            # Periodic inference frequency (episodes)
    eval_episodes: int = 5              # Number of episodes per evaluation

    # Early stopping
    early_stopping_patience: int = 200
    early_stopping_threshold: float = 1e-4

    # ========== Device and Paths ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Directory paths
    checkpoint_dir: Optional[Path] = None

    # ========== Normalization ==========
    state_normalization: bool = True
    reward_normalization: bool = True
    observation_norm_clip: float = 10.0
    reward_norm_clip: float = 10.0

    # ========== Debugging and Logging ==========
    debug_mode: bool = False
    verbose_logging: bool = False
    log_gradients: bool = False
    log_activations: bool = False

    # ========== Experience Buffer ==========
    buffer_size: int = 10000           # Total buffer size
    high_level_buffer_size: int = 2000 # High-level experience buffer size
    low_level_buffer_size: int = 8000  # Low-level experience buffer size

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate parameters
        assert self.num_options > 0, "Number of options must be positive"
        assert self.num_actions > 0, "Number of actions must be positive"
        assert 0 < self.ppo_ratio_clip < 1, "PPO clip ratio must be in (0, 1)"
        assert self.discount > 0 and self.discount <= 1, "Discount must be in (0, 1]"
        assert self.gae_tau > 0 and self.gae_tau <= 1, "GAE tau must be in (0, 1]"

        # Setup device
        self.device = torch.device(self.device)

        # Setup directories with timestamped structure (like option_critic)
        if self.checkpoint_dir is None:
            # Create timestamped checkpoint directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "dac" / timestamp

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Created timestamped checkpoint directory: {self.checkpoint_dir}")

        # Validate learning schedule
        assert self.learning_schedule in ["all", "alt", "sequential"], \
            "Learning schedule must be 'all', 'alt', or 'sequential'"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DACConfig':
        """Create config from dictionary."""
        # Convert path strings back to Path objects
        if 'checkpoint_dir' in config_dict and config_dict['checkpoint_dir'] is not None:
            config_dict['checkpoint_dir'] = Path(config_dict['checkpoint_dir'])

        return cls(**config_dict)

    def get_network_config(self) -> Dict[str, Any]:
        """Get network-specific configuration."""
        return {
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'hidden_dim': self.hidden_dim,
            'member_state_dim': self.member_state_dim,
            'num_layers': self.num_gnn_layers,
            'num_actions': self.num_actions,
            'num_options': self.num_options,
            'device': self.device
        }

    def get_ppo_config(self) -> Dict[str, Any]:
        """Get PPO-specific configuration."""
        return {
            'learning_rate': self.learning_rate,
            'high_level_lr': self.high_level_lr,
            'low_level_lr': self.low_level_lr,
            'ppo_ratio_clip': self.ppo_ratio_clip,
            'optimization_epochs': self.optimization_epochs,
            'mini_batch_size': self.mini_batch_size,
            'rollout_length': self.rollout_length,
            'entropy_weight': self.entropy_weight,
            'beta_weight': self.beta_weight,
            'value_loss_coef': self.value_loss_coef,
            'use_gae': self.use_gae,
            'gae_tau': self.gae_tau,
            'discount': self.discount,
            'gradient_clip': self.gradient_clip
        }

    def get_dac_config(self) -> Dict[str, Any]:
        """Get DAC-specific configuration."""
        return {
            'length_bonus_weight': self.length_bonus_weight,
            'termination_reg': self.termination_reg,
            'learning_schedule': self.learning_schedule,
            'freeze_value_function': self.freeze_value_function,
            'num_options': self.num_options
        }

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        return {
            'structure_shape': self.structure_shape,
            'add_structure_geometry': self.add_structure_geometry,
            'add_response_features': self.add_response_features,
            'reward_type': self.reward_type,
            'scwb_driven_design': self.scwb_driven_design,
            'do_nonlinear_dynamic_analysis': self.do_nonlinear_dynamic_analysis,
            'check_acceleration': self.check_acceleration,
            'check_displacement': self.check_displacement,
            'checkpoint_dir': self.checkpoint_dir
        }


# Predefined configurations for different use cases
def get_default_config() -> DACConfig:
    """Get default DAC configuration."""
    return DACConfig()


def get_debug_config() -> DACConfig:
    """Get configuration optimized for debugging."""
    return DACConfig(
        max_episodes=10,
        rollout_length=32,
        mini_batch_size=16,
        optimization_epochs=2,
        debug_mode=True,
        verbose_logging=True,
        save_interval=5
    )


def get_fast_training_config() -> DACConfig:
    """Get configuration for fast training (smaller models)."""
    return DACConfig(
        hidden_dim=128,
        member_state_dim=64,
        num_gnn_layers=2,
        rollout_length=64,
        mini_batch_size=32,
        optimization_epochs=2
    )


def get_performance_config() -> DACConfig:
    """Get configuration optimized for best performance."""
    return DACConfig(
        hidden_dim=512,
        member_state_dim=256,
        num_gnn_layers=4,
        rollout_length=256,
        mini_batch_size=128,
        optimization_epochs=8,
        learning_rate=1e-4,
        entropy_weight=0.005
    )