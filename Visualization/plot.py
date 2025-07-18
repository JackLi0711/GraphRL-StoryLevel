import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

from RL.record import Record
from RL.environment import Environment


def plot_reward(train_scores: List[float], test_scores: List[float], checkpoint_dir: Path, is_mcts: bool = False) -> None:
    """Plot train scores during every episode and test score every few episode."""
    plt.figure(figsize=(12, 6))
    
    if is_mcts:
        eval_episodes = np.arange(1, len(test_scores) + 1)
        plt.plot(eval_episodes, test_scores, label="evaluation", color='blue', linestyle='-', linewidth=2)
        plt.xlabel("Evaluation Episodes", fontsize=16)
        plt.title("MCTS Evaluation Scores", fontsize=18)
    else:
        train_episodes = np.arange(1, len(train_scores)+1)
        episode_per_test = len(train_scores) / len(test_scores) if len(test_scores) > 0 else 0
        test_episodes = np.arange(episode_per_test, len(train_scores)+1, episode_per_test)
        plt.plot(train_episodes, train_scores, label="training", color='black', linestyle='--', linewidth=1)
        plt.plot(test_episodes, test_scores, label="testing", color='red', linestyle='-', linewidth=2)
        plt.xlabel("Trained Episodes", fontsize=16)
        plt.title("Training and Testing Scores", fontsize=18)

    plt.legend(fontsize=14)
    plt.grid()
    plt.ylabel("Cumulative Reward", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
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
    plt.savefig(checkpoint_dir / "q_vals.png")
    plt.close()


def plot_fail_names(train_fail_names: List[str], test_fail_names: List[str], checkpoint_dir: Path, is_mcts: bool = False) -> None:
    if is_mcts:
        test_names = {}
        for name in test_fail_names:
            if name is None: name = "pass"
            test_names[name] = test_names.get(name, 0) + 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(test_names.keys(), test_names.values())
        plt.title("MCTS Evaluation Fail Names", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

    else:
        train_names = {}
        test_names = {}
        for name in train_fail_names:
            if name == None: name = "pass"
            train_names[name] = train_names.get(name, 0) + 1
        
        for name in test_fail_names:
            if name == None: name = "pass"
            test_names[name] = test_names.get(name, 0) + 1
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].bar(train_names.keys(), train_names.values())
        axs[0].set_title("Training Fail Names")
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].bar(test_names.keys(), test_names.values())
        axs[1].set_title("Testing Fail Names")
        axs[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()

    plt.savefig(checkpoint_dir / "fail_names.png")
    plt.close()


def plot_fail_reasons(train_fail_reasons: List[str], test_fail_reasons: List[str], checkpoint_dir: Path, is_mcts: bool = False) -> None:
    if is_mcts:
        test_fail_reasons = [r.split('_')[0] if r else "pass" for r in test_fail_reasons]
        test_reasons = {}
        for reason in test_fail_reasons:
            test_reasons[reason] = test_reasons.get(reason, 0) + 1
            
        plt.figure(figsize=(10, 6))
        plt.bar(test_reasons.keys(), test_reasons.values())
        plt.title("MCTS Evaluation Fail Reasons", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

    else:
        # full string is too long, only keep first half
        train_fail_reasons = [reason.split('_')[0] if reason else "pass" for reason in train_fail_reasons]
        test_fail_reasons = [reason.split('_')[0] if reason else "pass" for reason in test_fail_reasons]
        
        train_reasons = {}
        test_reasons = {}
        for reason in train_fail_reasons:
            train_reasons[reason] = train_reasons.get(reason, 0) + 1
        
        for reason in test_fail_reasons:
            test_reasons[reason] = test_reasons.get(reason, 0) + 1
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].bar(train_reasons.keys(), train_reasons.values())
        axs[0].set_title("Training Fail Reasons")
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].bar(test_reasons.keys(), test_reasons.values())
        axs[1].set_title("Testing Fail Reasons")
        axs[1].tick_params(axis='x', rotation=45)
        
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

