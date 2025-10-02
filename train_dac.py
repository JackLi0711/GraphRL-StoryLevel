#!/usr/bin/env python3
"""
DAC Training Script

Main script for training DAC (Double Actor-Critic) with PPO optimization
for structural design optimization using Graph Neural Networks.

Usage:
    python train_dac.py [--config CONFIG_FILE] [--resume CHECKPOINT_PATH]

Examples:
    # Basic training
    python train_dac.py

    # Training with custom config
    python train_dac.py --config configs/dac_config.json

    # Resume from checkpoint
    python train_dac.py --resume checkpoints_dac/checkpoint_episode_500.pt

    # Debug mode
    python train_dac.py --debug

    # Performance mode
    python train_dac.py --performance
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from RL.dac import (
    DACConfig, DACTrainer, DACLogger,
    get_default_config, get_debug_config, get_performance_config
)
from RL.environment import Environment


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DAC (Double Actor-Critic) for Structural Design Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON format)'
    )

    # Checkpointing
    parser.add_argument(
        '--resume', '-r',
        type=str,
        help='Path to checkpoint file to resume training from'
    )

    # Predefined configurations
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Use debug configuration (fast, small model)'
    )

    parser.add_argument(
        '--performance',
        action='store_true',
        help='Use performance configuration (best quality, slower)'
    )

    # Experiment settings
    parser.add_argument(
        '--experiment-name', '-n',
        type=str,
        default='dac_training',
        help='Name of the experiment'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        help='Directory for logging (overrides config)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory for checkpoints (overrides config)'
    )

    # Training options
    parser.add_argument(
        '--max-episodes',
        type=int,
        help='Maximum number of episodes (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config)'
    )

    # Environment options
    parser.add_argument(
        '--structure-shape',
        type=str,
        choices=['fixed', 'small_random', 'random'],
        help='Structure shape type (overrides config)'
    )

    parser.add_argument(
        '--no-response-features',
        action='store_true',
        help='Disable response feature analysis'
    )

    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce logging output'
    )

    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )

    return parser.parse_args()


def load_config(args) -> DACConfig:
    """
    Load configuration from arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        DAC configuration
    """
    # Start with predefined config
    if args.debug:
        config = get_debug_config()
    elif args.performance:
        config = get_performance_config()
    else:
        config = get_default_config()

    # Load from file if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        config = DACConfig.from_dict(config_dict)


    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)

    if args.max_episodes:
        config.max_episodes = args.max_episodes

    if args.seed:
        config.seed = args.seed

    if args.structure_shape:
        config.structure_shape = args.structure_shape
        # Recalculate num_actions based on new structure_shape
        config.num_actions = config._get_max_action_size()

    if args.no_response_features:
        config.add_response_features = False

    # Device handling
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device

    # Logging configuration
    if args.verbose:
        config.verbose_logging = True
        config.debug_mode = True

    if args.quiet:
        config.verbose_logging = False

    return config


def create_environment(config: DACConfig) -> Environment:
    """
    Create and configure environment.

    Args:
        config: DAC configuration

    Returns:
        Configured environment
    """
    env_config = config.get_environment_config()

    # Placeholder values for required environment parameters
    # These should be set based on your specific setup
    environment = Environment(
        structure_shape=env_config['structure_shape'],
        add_structure_geometry=env_config['add_structure_geometry'],
        add_response_features=env_config['add_response_features'],
        reward_type=env_config['reward_type'],
        scwb_driven_design=env_config['scwb_driven_design'],
        do_nonlinear_dynamic_analysis=env_config['do_nonlinear_dynamic_analysis'],
        check_acceleration=env_config['check_acceleration'],
        check_displacement=env_config['check_displacement'],
        nda_simulator=None,  # Set to your NDA simulator
        nda_norm_dict={},    # Set to your normalization dictionary
        DBE_ground_motion_set=[],  # Set to your ground motion data
        MCE_ground_motion_set=[],  # Set to your ground motion data
        checkpoint_dir=env_config['checkpoint_dir'],
        logger=logging.getLogger('environment'),
        device=torch.device(config.device)
    )

    return environment


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        config = load_config(args)

        # Create logger
        logger = DACLogger(
            log_dir=config.checkpoint_dir,
            experiment_name=args.experiment_name,
            console_level=logging.DEBUG if args.verbose else logging.INFO,
            use_tensorboard=not args.no_tensorboard
        )

        logger.logger.info("Starting DAC training script")
        logger.logger.info(f"Configuration: {config.to_dict()}")

        # Create environment
        environment = create_environment(config)
        logger.logger.info("Environment created successfully")

        # Dynamically infer feature dimensions from environment (like Option-Critic)
        logger.logger.info("Inferring feature dimensions from environment...")
        structure = environment.reset()
        graph = structure.graph
        actual_node_feature_dim = graph.x.shape[1]
        actual_edge_feature_dim = graph.edge_attr.shape[1]

        logger.logger.info(f"Inferred node_feature_dim: {actual_node_feature_dim}")
        logger.logger.info(f"Inferred edge_feature_dim: {actual_edge_feature_dim}")
        logger.logger.info(f"Config node_feature_dim: {config.node_feature_dim}")
        logger.logger.info(f"Config edge_feature_dim: {config.edge_feature_dim}")

        # Override config with actual dimensions
        if actual_node_feature_dim != config.node_feature_dim:
            logger.logger.warning(f"Overriding node_feature_dim: {config.node_feature_dim} -> {actual_node_feature_dim}")
            config.node_feature_dim = actual_node_feature_dim

        if actual_edge_feature_dim != config.edge_feature_dim:
            logger.logger.warning(f"Overriding edge_feature_dim: {config.edge_feature_dim} -> {actual_edge_feature_dim}")
            config.edge_feature_dim = actual_edge_feature_dim

        # Create trainer
        trainer = DACTrainer(
            config=config,
            base_environment=environment,
            logger=logger
        )

        logger.logger.info("Trainer initialized successfully")

        # Training
        if args.resume:
            logger.logger.info(f"Resuming training from checkpoint: {args.resume}")
            results = trainer.resume_training(args.resume)
        else:
            logger.logger.info("Starting fresh training")
            results = trainer.train()

        # Log final results
        logger.logger.info("Training completed successfully!")
        logger.logger.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.logger.info(f"  {key}: {value}")

        # Save final configuration
        final_config_path = config.checkpoint_dir / "final_config.json"
        with open(final_config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.logger.info(f"Final configuration saved to: {final_config_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_default_config_file():
    """Create a default configuration file for reference."""
    config = get_default_config()
    config_path = Path("configs/dac_default.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"Default configuration saved to: {config_path}")


if __name__ == "__main__":
    # Create configs directory with default config if it doesn't exist
    if not Path("configs").exists():
        create_default_config_file()

    # Run main training
    main()