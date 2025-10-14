"""
Configuration and argument parsing for Option-Critic algorithm.

This module contains the command-line argument parsing functionality
for the Option-Critic training script.
"""

import argparse


def parse_args():
    """Parse command line arguments for Option-Critic training."""
    parser = argparse.ArgumentParser()

    # Environment arguments
    parser.add_argument("--structure_shape", type=str, default="random")
    parser.add_argument("--add_structure_geometry", action="store_true", default=True)
    parser.add_argument("--add_response_features", action="store_true", default=True)
    parser.add_argument("--reward_type", type=str, default="material")
    parser.add_argument("--scwb_driven_design", action="store_true", default=False)
    parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
    parser.add_argument("--check_acceleration", action="store_true", default=False)
    parser.add_argument("--check_displacement", action="store_true", default=True)

    # Option-Critic arguments
    parser.add_argument("--num_options", type=int, default=8)
    parser.add_argument("--max_option_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=int, default=int(3e4))
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_frequency", type=int, default=10)
    parser.add_argument("--freeze_interval", type=int, default=512) # 512
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4) # 1e-4
    parser.add_argument("--actor_lr", type=float, default=3e-4) # 1e-4
    parser.add_argument("--critic_lr", type=float, default=3e-4) # 1e-4
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128) # 128
    parser.add_argument("--num_layers", type=int, default=3)  # 3
    parser.add_argument("--termination_reg", type=float, default=0.01)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--option_length_bonus", type=float, default=0.01, help="Bonus reward for longer options: reward += (step-1) * bonus")
    parser.add_argument("--failure_penalty", type=float, default=1.0, help="Penalty applied when option fails constraint check")
    parser.add_argument("--termination_lr_ratio", type=float, default=0.01, help="Termination learning rate as ratio of actor_lr (termination_lr = actor_lr * ratio)")
    parser.add_argument("--eval_frequency", type=int, default=10, help="Evaluate model every N training episodes")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")

    # Option preview visualization settings
    parser.add_argument("--enable_option_preview", action="store_true", default=True, help="Enable option preview visualization")
    parser.add_argument("--option_preview_frequency", type=int, default=1, help="Generate option preview every N evaluation rounds")

    return parser.parse_args()