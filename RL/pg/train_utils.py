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
            valid_mask=valid_mask,
            graph=graph.clone(),  # Store graph for recomputing features
            structure=structure   # Store structure for recomputing features
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
    # Set models to eval mode
    agent.state_gnn.eval()
    agent.policy_net.eval()
    agent.value_net.eval()

    structure = env.reset(testing=True)

    # === DIAGNOSTIC: Log initial structure ===
    if logger:
        logger.info(f"  [DIAGNOSTIC TEST] Initial structure story_level_sections: {structure.story_level_sections}")
        logger.info(f"  [DIAGNOSTIC TEST] Structure shape: x_span={structure.x_span_num}, z_span={structure.z_span_num}, stories={structure.story_num}")

    rec.record_in_beginning(structure, testing=True)

    graph = structure.graph.clone()
    score = 0
    done = False
    design_process = []  # 記錄設計過程
    step_count = 0

    while not done:
        original_structure = deepcopy(structure)

        # Get features
        story_features, global_features = agent.get_features(graph, structure)

        # === DIAGNOSTIC: Get action probabilities ===
        with torch.no_grad():
            action_scores = agent.policy_net(story_features)
            valid_mask = torch.ones(story_features.shape[0], dtype=torch.bool, device=agent.device)
            if len(structure.already_minimum_section_story_indexes) > 0:
                valid_mask[structure.already_minimum_section_story_indexes] = False
            masked_scores = action_scores.masked_fill(~valid_mask, -1e9)
            probs = torch.nn.functional.softmax(masked_scores, dim=0)

        # Choose action (greedy)
        action, _, _, _ = agent.choose_action(
            story_features,
            global_features,
            structure,
            greedy=True
        )

        # === DIAGNOSTIC: Log action details ===
        if logger and step_count < 3:  # Only log first 3 steps to avoid spam
            top_3_probs, top_3_indices = torch.topk(probs, min(3, len(probs)))
            logger.info(f"  [DIAGNOSTIC TEST] Step {step_count}: Action={action}, "
                       f"Action_prob={probs[action].item():.4f}")
            logger.info(f"    Top 3 actions: {top_3_indices.cpu().tolist()}, "
                       f"probs: {[f'{p:.4f}' for p in top_3_probs.cpu().tolist()]}")

        # Record design step (only store serializable data)
        design_process.append({
            'action': action,
            'story_level_sections': structure.story_level_sections.copy()
        })

        # Step
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        graph = structure.graph.clone()
        score += reward
        step_count += 1

    # Final structure
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    rec.record_in_end(final_structure, env, testing=True)

    # Set models back to train mode
    agent.state_gnn.train()
    agent.policy_net.train()
    agent.value_net.train()

    return score, design_process


def plot_training_testing_curves(rec, ckpt_dir, test_frequency):
    """
    繪製訓練和測試曲線

    Training: 每個 episode 的 score
    Testing: 每 test_frequency episodes 的 mean ± std
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Testing curve (mean ± std)
    if "score_mean" in rec.testing_record and len(rec.testing_record["score_mean"]) > 0:
        # Use actual number of test points
        num_tests = len(rec.testing_record["score_mean"])
        test_episodes = [(i+1) * test_frequency for i in range(num_tests)]
        test_means = rec.testing_record["score_mean"]
        test_stds = rec.testing_record["score_std"]

        # Only plot training data up to the last test episode
        last_test_episode = test_episodes[-1]
        training_scores = rec.training_record["score"][:last_test_episode]
        episodes = range(1, len(training_scores) + 1)

        # Training curve
        ax.plot(episodes, training_scores,
                label='Training Score', alpha=0.6, color='blue')

        ax.plot(test_episodes, test_means,
                label='Testing Score (mean)', color='red', linewidth=2)
        ax.fill_between(test_episodes,
                         np.array(test_means) - np.array(test_stds),
                         np.array(test_means) + np.array(test_stds),
                         alpha=0.3, color='red', label='Testing Score (±std)')
    else:
        # If no testing data, plot all training data
        episodes = range(1, len(rec.training_record["score"]) + 1)
        ax.plot(episodes, rec.training_record["score"],
                label='Training Score', alpha=0.6, color='blue')

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


def plot_loss_curves(rec, ckpt_dir):
    """
    繪製所有loss在同一張圖上

    Args:
        rec: Record object with loss_record
        ckpt_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt

    if len(rec.loss_record['episodes']) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = rec.loss_record['episodes']

    # Plot different losses
    ax.plot(episodes, rec.loss_record['policy_loss'],
            label='Policy Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(episodes, rec.loss_record['value_loss'],
            label='Value Loss', linewidth=2, marker='s', markersize=4)
    ax.plot(episodes, rec.loss_record['entropy_loss'],
            label='Entropy Loss', linewidth=2, marker='^', markersize=4)
    ax.plot(episodes, rec.loss_record['total_loss'],
            label='Total Loss', linewidth=2, marker='d', markersize=4, linestyle='--')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(ckpt_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_gradient_norms(rec, ckpt_dir):
    """
    繪製所有gradient norms在同一張圖上 (使用log scale)

    Args:
        rec: Record object with gradient_record
        ckpt_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt

    if len(rec.gradient_record['episodes']) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = rec.gradient_record['episodes']

    # Plot gradient norms
    ax.plot(episodes, rec.gradient_record['state_gnn_grad'],
            label='StateGNN Gradient', linewidth=2, marker='o', markersize=4)
    ax.plot(episodes, rec.gradient_record['policy_grad'],
            label='Policy Network Gradient', linewidth=2, marker='s', markersize=4)
    ax.plot(episodes, rec.gradient_record['value_grad'],
            label='Value Network Gradient', linewidth=2, marker='^', markersize=4)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('Gradient Norms During Training', fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Use log scale for better visualization
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    plt.tight_layout()
    plt.savefig(ckpt_dir / 'gradient_norms.png', dpi=150, bbox_inches='tight')
    plt.close()
