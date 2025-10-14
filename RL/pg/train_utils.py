"""
Training utility functions for A2C and PPO

Author: Claude Code
Date: 2025-10-14
"""

import torch
import numpy as np
from copy import deepcopy
from pathlib import Path


def train_episode(agent, env, rec, logger):
    """
    訓練一個 episode

    Args:
        agent: A2CAgent or PPOAgent
        env: Environment
        rec: Record
        logger: Logger

    Returns:
        score: float
        loss_dict: dict
    """
    structure = env.reset()
    rec.record_in_beginning(structure, testing=False)

    graph = structure.graph.clone()
    score = 0
    done = False

    while not done:
        original_structure = deepcopy(structure)

        # Get features (no grad)
        story_features, global_features = agent.get_features(graph, structure)

        # Choose action
        action, log_prob, value, entropy = agent.choose_action(
            story_features,
            global_features,
            structure,
            greedy=False
        )

        # Step environment
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # Store experience
        valid_mask = torch.ones(story_features.shape[0], dtype=torch.bool, device=agent.device)
        if len(structure.already_minimum_section_story_indexes) > 0:
            valid_mask[structure.already_minimum_section_story_indexes] = False

        agent.buffer.add_step(
            state=story_features,
            global_state=global_features,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            valid_mask=valid_mask
        )

        # Update
        graph = structure.graph.clone()
        score += reward

        if logger:
            logger.info(f"Episode {agent._number_episodes+1}, Step {len(agent.buffer.rewards)}, "
                       f"Action: {action}, Reward: {reward:.4f}, Score: {score:.4f}")

    # Finish episode
    agent.buffer.finish_episode()
    agent._number_episodes += 1

    # Update if enough episodes collected
    loss_dict = {}
    if len(agent.buffer) >= agent.accumulate_episodes:
        loss_dict = agent.update()

    # Record
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    rec.record_in_end(final_structure, env, testing=False)

    return score, loss_dict


def test_episode(agent, env, rec, logger):
    """
    測試一個 episode (greedy policy)

    Args:
        agent: A2CAgent or PPOAgent
        env: Environment
        rec: Record
        logger: Logger

    Returns:
        score: float
        design_process: list of dicts
    """
    structure = env.reset(testing=True)
    rec.record_in_beginning(structure, testing=True)

    graph = structure.graph.clone()
    score = 0
    done = False
    design_process = []  # 記錄設計過程

    while not done:
        original_structure = deepcopy(structure)

        # Get features
        story_features, global_features = agent.get_features(graph, structure)

        # Choose action (greedy)
        action, _, _, _ = agent.choose_action(
            story_features,
            global_features,
            structure,
            greedy=True
        )

        # Record design step (only store serializable data)
        design_process.append({
            'action': action,
            'story_level_sections': structure.story_level_sections.copy()
        })

        # Step
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        graph = structure.graph.clone()
        score += reward

    # Final structure
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    rec.record_in_end(final_structure, env, testing=True)

    return score, design_process


def plot_training_testing_curves(rec, ckpt_dir, test_frequency):
    """
    繪製訓練和測試曲線

    Training: 每個 episode 的 score
    Testing: 每 test_frequency episodes 的 mean ± std
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training curve
    episodes = range(1, len(rec.training_record["score"]) + 1)
    ax.plot(episodes, rec.training_record["score"],
            label='Training Score', alpha=0.6, color='blue')

    # Testing curve (mean ± std)
    if "score_mean" in rec.testing_record and len(rec.testing_record["score_mean"]) > 0:
        test_episodes = range(test_frequency,
                             len(rec.training_record["score"]) + 1,
                             test_frequency)
        test_means = rec.testing_record["score_mean"]
        test_stds = rec.testing_record["score_std"]

        ax.plot(test_episodes, test_means,
                label='Testing Score (mean)', color='red', linewidth=2)
        ax.fill_between(test_episodes,
                         np.array(test_means) - np.array(test_stds),
                         np.array(test_means) + np.array(test_stds),
                         alpha=0.3, color='red', label='Testing Score (±std)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training and Testing Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ckpt_dir / 'training_testing_curves.png', dpi=150)
    plt.close()


def save_model(agent, ckpt_dir, episode):
    """保存模型"""
    save_path = ckpt_dir / "models" / f"model_episode_{episode+1}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'state_gnn': agent.state_gnn.state_dict(),
        'policy_net': agent.policy_net.state_dict(),
        'value_net': agent.value_net.state_dict(),
        'episode': episode,
        'entropy_coef': agent.get_entropy_coef()
    }, save_path)

    if agent.logger:
        agent.logger.info(f"Model saved to {save_path}")
