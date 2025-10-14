"""
Model evaluation functionality for Option-Critic algorithm.

This module contains the evaluate_model function for testing
the performance of trained Option-Critic models.
"""

import numpy as np
import torch
from .utils import get_graph_data
from .rollout import rollout_option
from .logger import termination_logger


def evaluate_model(base_env, oc_model, device, num_episodes, max_option_len, logger=None, seed=42, option_length_bonus=0.0, failure_penalty=1.0):
    """
    Evaluate the model performance over multiple test episodes and collect action/option histories.

    Args:
        base_env: Base environment
        oc_model: Option-Critic model
        device: torch device
        num_episodes: Number of evaluation episodes
        max_option_len: Maximum option length
        logger: Logger instance
        seed: Random seed for reproducibility
        option_length_bonus: Bonus reward for longer options (default 0.0 for backward compatibility)

    Returns:
        avg_score: average testing score
        avg_episodes_length: average number of options per episode
        success_rate: percentage of episodes that reached minimum section
        eval_history: dict containing action and option histories for visualization
    """
    logger.info(f"Starting evaluation for {num_episodes} episodes with seed {seed}")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    episode_scores = []
    episode_total_rewards = []
    episode_lengths = []
    episode_actions = []  # Collect action sequences
    episode_options = []  # Collect option sequences
    episode_option_instances = []  # Collect option instance data with termination info
    episode_actions_SCWB = []  # For compatibility
    successful_episodes = 0

    # Store original testing mode and set to testing mode
    original_testing = oc_model.testing
    oc_model.testing = True

    for ep in range(num_episodes):
        logger.info(f"Evaluation episode {ep+1}/{num_episodes}")

        structure = base_env.reset(testing=True)
        done = False
        option_termination = True
        curr_option = 0
        greedy_option = 0
        episode_score = 0.0
        episode_total_reward = 0.0
        episode_option_count = 0

        # Track action and option sequences for this episode
        episode_action_sequence = []
        episode_option_sequence = []
        current_episode_option_instances = []  # Track option instances with termination info for this episode
        current_option_instance_id = 0  # Unique ID for each option instance

        while not done :
            if option_termination:
                # Use epsilon-greedy option selection for consistency with training
                epsilon = oc_model.epsilon
                if np.random.rand() < epsilon:
                    curr_option = np.random.choice(oc_model.num_options)
                else:
                    curr_option = greedy_option

            structure, next_state, option_done, episode_done, o_stats, step_transitions, termination_reason = rollout_option(
                structure, base_env, device, max_option_len, curr_option, oc_model, None, logger, option_length_bonus, failure_penalty
            )

            # Collect actions and options from step transitions
            for transition in step_transitions:
                episode_action_sequence.append(transition["action"])
                episode_option_sequence.append(curr_option)
                # Record option instance info for this step
                current_episode_option_instances.append({
                    "option_index": curr_option,
                    "instance_id": current_option_instance_id,
                    "action_step": len(episode_action_sequence) - 1
                })

            # Accumulate score only from successful options (using original reward without length bonus)
            if bool(o_stats.get("passed", False)):
                option_total_reward = sum(tr["original_reward"] for tr in step_transitions)
                episode_score += float(option_total_reward)

            # Accumulate total reward from ALL options (including failed ones with penalties)
            option_total_real_reward = sum(tr["reward"] for tr in step_transitions)
            episode_total_reward += float(option_total_real_reward)

            episode_option_count += 1

            # Check if episode completed successfully
            if termination_reason == "minimum_section":
                successful_episodes += 1
                logger.info(f"Episode {ep+1} reached minimum section successfully")

            done = episode_done

            # If the option terminated (for any reason), increment instance ID for next option
            if option_done:
                current_option_instance_id += 1

            # Update option termination for next iteration
            if not done:
                current_graph_data = get_graph_data(structure, device)
                _, global_state = oc_model.get_state(*current_graph_data)

                # Get termination probabilities for logging
                termination_probs = oc_model.get_terminations(global_state)
                option_termination, greedy_option = oc_model.predict_option_termination(global_state, curr_option)

                # Log termination probability prediction
                termination_logger.log_termination_prediction(
                    option=curr_option,
                    termination_probs=termination_probs,
                    termination_decision=option_termination,
                    episode=ep,
                    step=episode_option_count,
                    context="evaluation_loop"
                )

        episode_scores.append(episode_score)
        episode_total_rewards.append(episode_total_reward)
        episode_lengths.append(episode_option_count)
        episode_actions.append(episode_action_sequence)
        episode_options.append(episode_option_sequence)
        episode_option_instances.append(current_episode_option_instances)  # Store option instance data for this episode
        episode_actions_SCWB.append([])  # Empty for compatibility
        logger.info(f"Episode {ep+1} completed: score={episode_score:.2f}, length={episode_option_count}, actions={len(episode_action_sequence)}")

    # Restore original testing mode
    oc_model.testing = original_testing

    # Calculate metrics
    avg_score = float(np.mean(episode_scores)) if episode_scores else 0.0
    avg_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    success_rate = (successful_episodes / num_episodes) * 100.0 if num_episodes > 0 else 0.0

    logger.info(f"Evaluation completed:")
    logger.info(f"  Average score: {avg_score:.2f}")
    logger.info(f"  Average episode length: {avg_episode_length:.2f}")
    logger.info(f"  Success rate: {success_rate:.1f}% ({successful_episodes}/{num_episodes})")

    # Prepare evaluation history for visualization
    eval_history = {
        "scores": episode_scores,
        "total_rewards": episode_total_rewards,
        "actions": episode_actions,
        "options": episode_options,
        "option_instances": episode_option_instances,  # Include option instance data
        "actions_SCWB": episode_actions_SCWB,
        "avg_score": avg_score,
        "success_rate": success_rate
    }

    # Note: Test behavior visualization will be generated externally after collecting evaluation histories

    return avg_score, avg_episode_length, success_rate, eval_history