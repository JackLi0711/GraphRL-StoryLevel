"""
DAC Model evaluation functionality.

This module contains the evaluate_dac_model function for testing
the performance of trained DAC models and collecting behavior data.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional


def evaluate_dac_model(agent, env_wrapper, num_episodes: int, logger=None, seed: int = 42) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Evaluate the DAC model performance over multiple test episodes and collect action/option histories.

    Args:
        agent: DAC agent instance
        env_wrapper: DAC environment wrapper
        num_episodes: Number of evaluation episodes
        logger: Logger instance
        seed: Random seed for reproducibility

    Returns:
        avg_score: average testing score
        avg_episodes_length: average number of actions per episode
        success_rate: percentage of episodes that reached minimum section
        eval_history: dict containing action and option histories for visualization
    """
    if logger:
        logger.info(f"Starting DAC evaluation for {num_episodes} episodes with seed {seed}")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    episode_scores = []
    episode_total_rewards = []
    episode_lengths = []
    episode_actions = []  # Collect action sequences
    episode_options = []  # Collect option sequences
    episode_option_instances = []  # Collect option instance data
    episode_actions_SCWB = []  # For compatibility
    successful_episodes = 0

    # Store original agent state - DACAgent doesn't have training attribute
    # Instead, we'll set the network to eval mode
    original_training_mode = agent.network.training

    # Set network to evaluation mode
    agent.network.eval()

    for ep in range(num_episodes):
        if logger:
            logger.info(f"DAC evaluation episode {ep+1}/{num_episodes}")

        # Reset environment for evaluation
        dual_states, info = env_wrapper.reset(testing=True)

        episode_score = 0.0
        episode_total_reward = 0.0
        episode_steps = 0
        current_option = None
        option_start_step = 0

        # Track action and option sequences for this episode
        episode_action_sequence = []
        episode_option_sequence = []
        current_episode_option_instances = []
        current_option_instance_id = 0

        # Evaluation episode loop
        for step in range(agent.config.max_steps_per_episode):
            episode_steps += 1

            # Select option and action deterministically
            option, action, _ = agent.select_option_and_action(
                dual_states['graph_data'],
                deterministic=True
            )

            # Check for option change
            option_terminated = (current_option is not None and current_option != option)
            if current_option != option:
                if current_option is not None:
                    # Option instance ended
                    current_option_instance_id += 1

                current_option = option
                option_start_step = step

            # Execute action
            next_dual_states, rewards, done, env_info = env_wrapper.step(
                action, option, option_terminated
            )

            # Collect actions and options
            episode_action_sequence.append(action)
            episode_option_sequence.append(option)
            current_episode_option_instances.append({
                "option_index": option,
                "instance_id": current_option_instance_id,
                "action_step": len(episode_action_sequence) - 1
            })

            # Update rewards and scores
            episode_total_reward += rewards['low_level_reward']

            # For DAC, we accumulate score based on environment feedback
            if 'material_saved' in env_info:
                episode_score += float(env_info.get('material_saved', 0))

            # Update for next step
            dual_states = next_dual_states

            # Check termination
            if done:
                # Check if episode completed successfully
                env_stats = env_wrapper.get_episode_statistics()
                if env_stats.get('success', False):
                    successful_episodes += 1
                    if logger:
                        logger.info(f"Episode {ep+1} completed successfully")
                break

        # Store episode data
        episode_scores.append(episode_score)
        episode_total_rewards.append(episode_total_reward)
        episode_lengths.append(episode_steps)
        episode_actions.append(episode_action_sequence)
        episode_options.append(episode_option_sequence)
        episode_option_instances.append(current_episode_option_instances)
        episode_actions_SCWB.append([])  # Empty for compatibility

        if logger:
            logger.info(f"Episode {ep+1} completed: score={episode_score:.2f}, "
                       f"length={episode_steps}, actions={len(episode_action_sequence)}")

    # Restore original agent state
    if original_training_mode:
        agent.network.train()

    # Calculate metrics
    avg_score = float(np.mean(episode_scores)) if episode_scores else 0.0
    avg_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    success_rate = (successful_episodes / num_episodes) * 100.0 if num_episodes > 0 else 0.0

    if logger:
        logger.info(f"DAC evaluation completed:")
        logger.info(f"  Average score: {avg_score:.2f}")
        logger.info(f"  Average episode length: {avg_episode_length:.2f}")
        logger.info(f"  Success rate: {success_rate:.1f}% ({successful_episodes}/{num_episodes})")

    # Prepare evaluation history for visualization
    eval_history = {
        "scores": episode_scores,
        "actions": episode_actions,
        "options": episode_options,
        "option_instances": episode_option_instances,
        "actions_SCWB": episode_actions_SCWB,
        "avg_score": avg_score,
        "success_rate": success_rate
    }

    return avg_score, avg_episode_length, success_rate, eval_history