#!/usr/bin/env python3
"""
Option-Critic Training Script for Structural Design

This is the main training script for the Option-Critic algorithm applied to
structural design problems. This file has been refactored for better modularity
while maintaining backward compatibility.

Original implementation refactored into separate modules:
- RL.option_critic.trainer: Main training logic
- RL.option_critic.rollout: Option rollout functionality
- RL.option_critic.evaluation: Model evaluation
- RL.option_critic.utils: Utility functions
- RL.option_critic.logger: Logging functionality
- RL.option_critic.config: Configuration and argument parsing

For backward compatibility, key functions are re-exported below.
"""

import argparse
import datetime
import logging
from pathlib import Path

# Import the refactored components
from RL.option_critic.trainer import OptionCriticTrainer
from RL.option_critic.config import parse_args

# Backward compatibility exports for external modules
# (e.g., inference_best_model.py imports these functions)
from RL.option_critic.utils import get_graph_data, num_actions, apply_primitive_action, check_constraints_without_update
from RL.option_critic.evaluation import evaluate_model
from RL.option_critic.rollout import rollout_option
from RL.option_critic.logger import TerminationProbabilityLogger, termination_logger

# Legacy load_best_model function for compatibility
def load_best_model(checkpoint_dir, model_args=None):
    """
    Load the best model from checkpoint directory.

    This function is kept for backward compatibility with inference scripts.
    """
    import json
    import torch
    from RL.option_critic_gnn import OptionCriticGNN

    ckpt_dir = Path(checkpoint_dir)
    best_model_path = ckpt_dir / "best_model.pt"
    best_model_info_path = ckpt_dir / "best_model_info.json"

    if not best_model_path.exists() or not best_model_info_path.exists():
        print(f"Best model files not found in {checkpoint_dir}")
        return None, None

    # Load model info
    with open(best_model_info_path, "r", encoding="utf-8") as f:
        best_model_info = json.load(f)

    print(f"Loading best model from episode {best_model_info['episode']}")
    print(f"  Score: {best_model_info['score']:.2f}")
    print(f"  Success rate: {best_model_info['success_rate']:.1f}%")
    print(f"  Saved at: {best_model_info['timestamp']}")

    return best_model_path, best_model_info


def main(args=None):
    """
    Main training function for Option-Critic algorithm.

    Args:
        args: Parsed command line arguments. If None, will parse from command line.
    """
    if args is None:
        args = parse_args()

    print("=" * 60)
    print("Option-Critic Training for Structural Design")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Episodes: {args.epochs}")
    print(f"  Options: {args.num_options}")
    print(f"  Max option length: {args.max_option_len}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rates: actor={args.actor_lr}, critic={args.critic_lr}")
    print("=" * 60)

    # Create and run trainer
    trainer = OptionCriticTrainer(args)
    trainer.train()

    print("=" * 60)
    print("Training completed successfully!")
    print(f"Results saved to: {trainer.ckpt_dir}")
    print(f"Best model score: {trainer.best_model_score:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ============================================================================
# REMOVED DUPLICATE IMPLEMENTATIONS
#
# The following duplicate implementations have been removed during refactoring:
# 1. Second set of argument parsing (parse_arguments function)
# 2. Second main function with OptionEnvironmentWrapper/OptionCriticAgent
# 3. setup_environment, setup_agent, train_episode functions
# 4. Duplicate plotting and logging functionality
#
# Only the first implementation (OptionCriticGNN-based) is preserved and
# refactored into the RL.option_critic module.
# ============================================================================