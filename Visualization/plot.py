import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Sequence
from pathlib import Path
from collections import Counter

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

# ===========================================
# ============= MuZero 相關 =================
# ===========================================

def _moving_average(arr: Sequence[float], window: int = None) -> np.ndarray:
    """1‑D moving average. If *window* is None, use len(arr)//10 (min 1)."""
    if len(arr) == 0:
        return np.asarray(arr)
    if window is None:
        window = max(1, len(arr) // 10)
    weights = np.ones(window) / window
    return np.convolve(arr, weights, mode="valid")


def _plot_history(ax, data: Sequence[float], title: str, ylabel: str, color: str = "C0"):
    episodes = np.arange(1, len(data) + 1)
    ax.plot(episodes, data, color=color, linewidth=1, alpha=0.8, label="raw")
    if len(data) > 10:
        ma = _moving_average(data)
        ax.plot(episodes[len(episodes) - len(ma):], ma, linestyle="--", color="C1", label="moving avg")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def _plot_distribution(ax, values: Sequence[float], title: str, xlabel: str):
    if len(values) == 0:
        return
    ax.hist(values, bins=min(10, len(values)), alpha=0.7, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.axvline(np.mean(values), color="red", linestyle="--", label=f"mean={np.mean(values):.2f}")
    ax.legend()


def _plot_action_sequence_history(score_hist: List[float], story_num:int, actions_list_hist: List[Sequence[int]], out_path: Path):
    action_types = []
    for action in actions_list_hist:
        types = []
        for a in action:
            if a < story_num: type = "xdir-beam"
            elif a < story_num*2: type = "zdir-beam"
            elif a < story_num*3: type = "out-col"
            else: type = "in-col"
            types.append(type)
        action_types.append(types)
        
    test_episodes = np.arange(1, len(actions_list_hist)+1, 1) * int(len(score_hist) / len(actions_list_hist))
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
    ax2.plot(test_episodes, score_hist, color='black', linestyle='-', linewidth=1, label='test score', zorder=2)
    ax2.set_ylabel('Test score', fontsize=16)
    ax2.tick_params(labelsize=14)
    ax2.grid(zorder=0)

    legend_labels = ['xdir-beam', 'zdir-beam', 'out-col', 'in-col']
    legend_colors = ['dodgerblue', 'yellowgreen', 'orange', 'red']

    ax1.legend(labels=legend_labels, loc='upper left', fontsize=14, handles=[plt.Line2D([0], [0], color=color, linewidth=4) for color in legend_colors])
    ax2.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path / "testing_behaviors.png", dpi=1000)
    plt.close()


def _plot_fail_reason_bar(ax, fail_reasons: Sequence[str], title: str):
    counter = Counter(fail_reasons)
    if not counter:
        return
    ax.bar(counter.keys(), counter.values())
    ax.set_title(title)
    ax.set_ylabel("Count")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

###############################################################################
# High‑level figure helpers
###############################################################################

def plot_MuZero_inference_figure(
    out_path: Path,
    reward_history: List[float],
    length_history: List[int],
    best_actions_history: List[Sequence[int]],
    rewards_this_round: Sequence[float],
    fail_reasons_this_round: Sequence[str],
    story_num: int,
):
    """Produces and saves the *inference* plot (5 subplots)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # a. reward history
    _plot_history(axes[0, 0], reward_history, "Total reward history", "Reward")
    # b. episode length history
    _plot_history(axes[0, 1], length_history, "Episode length history", "Length", color="C2")
    # c. reward distribution (current round)
    _plot_distribution(axes[1, 0], rewards_this_round, "Reward distribution (this round)", "Reward")
    # d. fail reason distribution (current round)
    _plot_fail_reason_bar(axes[1, 1], fail_reasons_this_round, "Fail reason distribution (this round)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)

    # e. best action sequence history
    out_path = out_path.parent 
    _plot_action_sequence_history(reward_history, story_num, best_actions_history, out_path)
    



def plot_MuZero_selfplay_figure(
    out_path: Path,
    reward_history: List[float],
    length_history: List[int],
    best_actions_history: List[Sequence[int]],
    rewards_this_round: Sequence[float],
    fail_reasons_this_round: Sequence[str],
    story_num: int,
):
    """Same signature as inference; kept separate for clarity (could reuse)."""
    plot_MuZero_inference_figure(
        out_path, reward_history, length_history, best_actions_history, rewards_this_round, fail_reasons_this_round, story_num
    )


def plot_MuZero_combined_figure(
    out_path: Path,
    selfplay_reward_hist: List[float],
    selfplay_length_hist: List[int],
    inference_reward_hist: List[float],
    inference_length_hist: List[int],
    frequency: int,
):
    """Two‑subplot figure comparing self‑play vs inference history."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # a. total reward history (two curves)
    _plot_history(axes[0], selfplay_reward_hist, "Total reward history", "Reward")
    axes[0].plot(np.arange(1, len(inference_reward_hist) + 1)*int(len(selfplay_reward_hist) / len(inference_reward_hist)), inference_reward_hist, color="C3", alpha=0.8, label="inference")
    axes[0].legend()
    # b. episode length history
    _plot_history(axes[1], selfplay_length_hist, "Episode length history", "Length", color="C2")
    axes[1].plot(np.arange(1, len(inference_length_hist) + 1)*int(len(selfplay_length_hist) / len(inference_length_hist)), inference_length_hist, color="C4", alpha=0.8, label="inference")
    axes[1].legend()

    # c. bar chart of current round reward comparsion
    if len(selfplay_reward_hist) >= frequency:
        recent_self_play_rewards = selfplay_reward_hist[-frequency:]
        avg_self_play = np.mean(recent_self_play_rewards)
        std_self_play = np.std(recent_self_play_rewards)
        
        # Inference 的獎勵
        inference_rewards = inference_reward_hist[-frequency:]
        avg_inference = np.mean(inference_rewards)
        std_inference = np.std(inference_rewards)
        
        # 繪製比較圖
        categories = [f'Self-play (Recent {frequency})', 'Inference']
        means = [avg_self_play, avg_inference]
        stds = [std_self_play, std_inference]
        
        bars = axes[2].bar(categories, means, yerr=stds, capsize=5, alpha=0.7, 
                              color=['blue', 'orange'])
        axes[2].set_title('Reward Comparison: Self-play vs Inference')
        axes[2].set_ylabel('Average Reward')
        axes[2].grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    

    plt.tight_layout()
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def plot_muzero_losses(loss_record: List[Dict], save_dir: Path, save_json: bool = True):
    """
    繪製並儲存 MuZero 訓練過程中的各項 loss 曲線。

    Args:
        loss_record: List of dicts, 每個 dict 包含 'episode', 'value_loss', 'policy_loss', 'reward_loss', 'total_loss'
        save_dir: 儲存圖片和 JSON 的目錄路徑
        save_json: 是否同時儲存 JSON 檔案
    """
    if not loss_record:  # 確保有數據再畫
        return
    
    # 1. 儲存 JSON
    if save_json:
        with open(save_dir / 'loss_history.json', 'w') as f:
            json.dump(loss_record, f)
    
    # 2. 準備數據
    episodes = [d['episode'] for d in loss_record]
    v_losses = [d['value_loss'] for d in loss_record]
    p_losses = [d['policy_loss'] for d in loss_record]
    r_losses = [d['reward_loss'] for d in loss_record]
    t_losses = [d['total_loss'] for d in loss_record]
    
    # 3. 畫圖
    plt.figure(figsize=(12, 8))
    
    # 主圖：所有 loss
    plt.subplot(2, 1, 1)
    plt.plot(episodes, v_losses, label='Value Loss', marker='o', markersize=3)
    plt.plot(episodes, p_losses, label='Policy Loss', marker='s', markersize=3)
    plt.plot(episodes, r_losses, label='Reward Loss', marker='^', markersize=3)
    plt.plot(episodes, t_losses, label='Total Loss', marker='*', markersize=4, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Loss Value')
    plt.title('MuZero Training Losses')
    plt.legend()
    plt.grid(True)
    
    # 子圖：三個主要 loss（不含 total）的細節
    plt.subplot(2, 1, 2)
    plt.plot(episodes, v_losses, label='Value Loss', marker='o', markersize=3)
    plt.plot(episodes, p_losses, label='Policy Loss', marker='s', markersize=3)
    plt.plot(episodes, r_losses, label='Reward Loss', marker='^', markersize=3)
    plt.xlabel('Episode')
    plt.ylabel('Loss Value')
    plt.title('Individual Losses (Detail View)')
    plt.legend()
    plt.grid(True)
    
    # 調整子圖間距
    plt.tight_layout()
    
    # 4. 儲存圖片
    plt.savefig(save_dir / 'loss_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_muzero_metrics(metrics: Dict[str, List], save_dir: Path):
    """
    繪製 MuZero 訓練過程中的其他指標（如 reward、episode length 等）。

    Args:
        metrics: Dict of lists, 包含各種指標的歷史記錄
        save_dir: 儲存圖片的目錄路徑
    """
    # TODO: 實作其他指標的繪圖邏輯
    pass


if __name__ == "__main__":
    train_fail_reasons = ["minimum_section", "minimum_section", "drift_ratio", "beam_moment", "minimum_section", "column_tension", "drift_ratio", "minimum_section"]
    test_fail_reasons = ["strong_column_weak_beam", "column_compression", "soft_story", "drift_ratio", "beam_moment", "minimum_section", "column_tension", "drift_ratio", "minimum_section"]
    plot_fail_reasons(train_fail_reasons, test_fail_reasons, None)

