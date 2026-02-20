import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

from RL.record import Record
from RL.environment import Environment


def plot_reward(train_scores: List[float], test_scores: List[float], checkpoint_dir: Path) -> None:
    """Plot train scores during every episode and test score every few episode."""
    train_episodes = np.arange(1, len(train_scores)+1)
    episode_per_test = len(train_scores) / len(test_scores)
    test_episodes = np.arange(episode_per_test, len(train_scores)+1, episode_per_test)
    plt.figure(figsize=(12, 6))
    plt.plot(train_episodes, train_scores, label="training", color='black', linestyle='--', linewidth=1)
    plt.plot(test_episodes, test_scores, label="testing", color='red', linestyle='-', linewidth=2)
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel("trained episodes", fontsize=16)
    plt.ylabel("cumulative reward", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "reward.png")
    plt.close()


def plot_loss(learn_losses: List[List[float]], checkpoint_dir: Path) -> None:
    ave_losses = [np.nanmean(losses) for losses in learn_losses if len(losses) > 0]
    plt.figure(figsize=(12, 6))
    plt.plot(ave_losses, color='black', linewidth=1)
    plt.grid()
    plt.yscale("log")
    plt.xlabel("trained episodes")
    plt.ylabel("average batch loss")
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "loss.png")
    plt.close()


def plot_Qvalues(Q_values: List[List[float]], checkpoint_dir: Path) -> None:
    train_q_values, test_q_values = Q_values
    train_episodes = np.arange(1, len(train_q_values)+1, 1)
    test_episodes = np.arange(1, len(test_q_values)+1, 1) * int(len(train_q_values) / len(test_q_values))

    plt.figure(figsize=(12, 6))
    plt.plot(train_episodes, train_q_values, color='black', linewidth=1, label="train")
    plt.plot(test_episodes, test_q_values, color='red', linewidth=2, label="test")
    plt.grid()
    plt.title("Q value for first timestep in each episode")
    plt.xlabel("trained episodes")
    plt.ylabel("Q value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "q_vals.png")
    plt.close()


def plot_fail_names(train_fail_names: List[str], test_fail_names: List[str], checkpoint_dir: Path) -> None:
    train_names = {}
    test_names = {}
    for name in train_fail_names:
        if name == None: name = "none"
        if name in train_names:
            train_names[name] += 1
        else:
            train_names[name] = 1
    for name in test_fail_names:
        if name == None: name = "none"
        if name in test_names:
            test_names[name] += 1
        else:
            test_names[name] = 1
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(train_names.keys(), train_names.values())
    axs[0].set_title("training fail names")
    axs[1].bar(test_names.keys(), test_names.values())
    axs[1].set_title("testing fail names")
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "fail_names.png")
    plt.close()


def plot_fail_reasons(train_fail_reasons: List[str], test_fail_reasons: List[str], checkpoint_dir: Path) -> None:
    # full string is too long, only keep first half
    train_fail_reasons = [reason.split('_')[0] for reason in train_fail_reasons]
    test_fail_reasons = [reason.split('_')[0] for reason in test_fail_reasons]
    
    train_reasons = {}
    test_reasons = {}
    for reason in train_fail_reasons:
        if reason in train_reasons:
            train_reasons[reason] += 1
        else:
            train_reasons[reason] = 1
    for reason in test_fail_reasons:
        if reason in test_reasons:
            test_reasons[reason] += 1
        else:
            test_reasons[reason] = 1
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(train_reasons.keys(), train_reasons.values())
    axs[0].set_title("training fail reasons")
    axs[1].bar(test_reasons.keys(), test_reasons.values())
    axs[1].set_title("testing fail reasons")
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "fail_reasons.png")
    plt.close()


def plot_test_behaviors(rec: Record, env: Environment, checkpoint_dir: Path) -> None:
    train_scores = rec.training_record["score"]
    test_scores = rec.testing_record["score"]
    test_actions, test_actions_SCWB = rec.testing_record["action"], rec.testing_record["action_SCWB"]
    story_num = env._testing_structure.story_num

    action_types = []
    for action in test_actions:
        types = []
        for a in action:
            if a < story_num: type = "xdir-beam"
            elif a < story_num*2: type = "zdir-beam"
            elif a < story_num*3: type = "out-col"
            else: type = "in-col"
            types.append(type)
        action_types.append(types)
        
    test_episodes = np.arange(1, len(test_scores)+1, 1) * int(len(train_scores) / len(test_scores))
    color_mapping = {'xdir-beam': 'dodgerblue', 'zdir-beam': 'yellowgreen', 'out-col': 'orange', 'in-col': 'red'}

    fig, ax1 = plt.subplots(figsize=(10, 8))

    for i, types in enumerate(action_types):
        for j, type in enumerate(types):
            color = color_mapping[type]
            count = 1
            ax1.bar(test_episodes[i], count, color=color, width=3, bottom=j, zorder=1)

    ax1.set_xlabel('Trained Episode', fontsize=16)
    ax1.set_ylabel('Iteration', fontsize=16)
    ax1.tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(test_episodes, test_scores, color='black', linestyle='-', linewidth=1, label='test score', zorder=2)
    ax2.set_ylabel('Test score', fontsize=16)
    ax2.tick_params(labelsize=14)
    ax2.grid(zorder=0)

    legend_labels = ['xdir-beam', 'zdir-beam', 'out-col', 'in-col']
    legend_colors = ['dodgerblue', 'yellowgreen', 'orange', 'red']

    ax1.legend(labels=legend_labels, loc='upper left', fontsize=14, handles=[plt.Line2D([0], [0], color=color, linewidth=4) for color in legend_colors])
    ax2.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "testing_behaviors.png", dpi=1000)
    plt.close()




if __name__ == "__main__":
    train_fail_reasons = ["minimum_section", "minimum_section", "drift_ratio", "beam_moment", "minimum_section", "column_tension", "drift_ratio", "minimum_section"]
    test_fail_reasons = ["strong_column_weak_beam", "column_compression", "soft_story", "drift_ratio", "beam_moment", "minimum_section", "column_tension", "drift_ratio", "minimum_section"]
    plot_fail_reasons(train_fail_reasons, test_fail_reasons, None)

