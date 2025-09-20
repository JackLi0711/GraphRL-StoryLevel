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

from .dac_config import DACConfig, get_default_config, get_debug_config, get_performance_config
from .dac_gnn import DACDoubleActorCritic
from .dac_buffer import DACExperienceBuffer
from .dac_environment import DACEnvironmentWrapper
from .dac_agent import DACAgent
from .dac_trainer import DACTrainer
from .dac_utils import DACUtils
from .dac_logger import DACLogger

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