"""
DAC (Double Actor-Critic) Module for Structural Design Optimization

This module implements the Double Actor-Critic architecture from:
"DAC: The Double Actor-Critic Architecture for Learning Options" (NeurIPS 2019)

The module provides:
- DACConfig: Configuration management
- DACDoubleActorCritic: Dual Actor-Critic network with GNN backbone
- DACExperienceBuffer: Dual-layer experience replay buffer
- DACEnvironmentWrapper: Environment wrapper with response feature integration
- DACAgent: Main agent logic
- DACTrainer: Training loop coordinator
"""

from .config import DACConfig, get_default_config, get_debug_config, get_performance_config
from .gnn import DACDoubleActorCritic
from .buffer import DACExperienceBuffer
from .environment import DACEnvironmentWrapper
from .agent import DACAgent
from .trainer import DACTrainer
from .utils import DACUtils
from .logger import DACLogger

__version__ = "1.0.0"
__all__ = [
    "DACConfig",
    "get_default_config",
    "get_debug_config",
    "get_performance_config",
    "DACDoubleActorCritic",
    "DACExperienceBuffer",
    "DACEnvironmentWrapper",
    "DACAgent",
    "DACTrainer",
    "DACUtils",
    "DACLogger"
]