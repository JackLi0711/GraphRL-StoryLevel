import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from RL.record import Record


def plot_testing_behaviors(rec: Record, story_num: int, save_dir: Path, save=True) -> None:
    train_scores = rec.training_record["score"]
    test_scores = rec.testing_record["score"]
    test_actions = rec.testing_record["action"]
    test_option_indices = rec.option_indices["test"]
    test_option_rollout_lengths = rec.option_rollout_lengths["test"]

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
        
    fig, ax1 = plt.subplots(figsize=(10, 8))
    test_episodes = np.arange(1, len(test_scores)+1, 1) * int(len(train_scores) / len(test_scores))
    color_mapping = {'xdir-beam': 'dodgerblue', 'zdir-beam': 'yellowgreen', 'out-col': 'orange', 'in-col': 'red'}
    bar_width = 3
    for i, (types, option_indices, option_rollout_lengths) in enumerate(zip(action_types, test_option_indices, test_option_rollout_lengths)):
        x = test_episodes[i]
        for j, type in enumerate(types):
            color = color_mapping[type]
            ax1.bar(x, 1, color=color, width=bar_width, bottom=j, zorder=1)
        
        if len(option_rollout_lengths) == 0: 
            # flat testing: one option index --> one env.expensive_step()
            groups = []
            # group consecutive equal option indices
            for idx, opt in enumerate(option_indices):
                if not groups or groups[-1][0] != opt:
                    groups.append([opt, idx, idx+1])
                else:
                    groups[-1][2] = idx+1
        else: 
            # hierarchical testing: one option index --> multiple env.expensive_step()
            groups = []
            start = 0
            for idx, (opt, roll_len) in enumerate(zip(option_indices, option_rollout_lengths)):
                groups.append([opt, start, start+roll_len])
                start += roll_len

        # for opt, start, end in groups:
        #     height = end - start
        #     if height != 0:
        #         rect = Rectangle((x-bar_width/2, start), bar_width, height, fill=False, edgecolor='black', linewidth=1.2, zorder=3)
        #         ax1.add_patch(rect)
        #         ax1.text(x, start+height/2, str(opt), ha='center', va='center', color='black', fontsize=10, zorder=4)
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
    if save:
        plt.savefig(save_dir/"testing_behaviors.png", dpi=1000, bbox_inches='tight')
    plt.close()


def plot_advantage(advantage_record: dict, name: str, save_dir: Path, save=True) -> None:
    # train_record: (episode_num, optimization_epoch, timestep_num)
    train_record = advantage_record["train"]
    train_episode_num = len(train_record)
    train_means = np.zeros(train_episode_num)
    train_stds = np.zeros(train_episode_num)
    for i in range(train_episode_num):
        # advantages at different optimization epochs are all the same, so just take the first one
        advantages = np.array(train_record[i][0])
        train_means[i] = advantages.mean()
        train_stds[i] = advantages.std()
    
    # test_record: (episode_num, timestep_num)
    test_record = advantage_record["test"]
    test_episode_num = len(test_record)
    test_means = np.zeros(test_episode_num)
    test_stds = np.zeros(test_episode_num)
    for i in range(test_episode_num):
        advantages = np.array(test_record[i])
        test_means[i] = advantages.mean()
        test_stds[i] = advantages.std()

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = int(train_episode_num / test_episode_num)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean ± std", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3)
    plt.plot(test_episodes, test_means, label="testing: mean ± std", color="red", linewidth=1)
    plt.fill_between(test_episodes, test_means-test_stds, test_means+test_stds, color="red", alpha=0.3)
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("advantages", fontsize=14)
    plt.title("Advantages over Episodes", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"{name}.png", bbox_inches='tight')
    plt.close()

def plot_entropy(entropy_record: dict, name: str, save_dir: Path, save=True):
    # train_record: (episode_num, timestep_num)
    train_record = entropy_record["train"]
    train_episode_num = len(train_record)
    train_means = np.zeros(train_episode_num)
    train_stds = np.zeros(train_episode_num)
    for i in range(train_episode_num):
        entropies = np.array(train_record[i])
        train_means[i] = entropies.mean()
        train_stds[i] = entropies.std()
    
    # test_record: (episode_num, timestep_num)
    test_record = entropy_record["test"]
    test_episode_num = len(test_record)
    test_means = np.zeros(test_episode_num)
    test_stds = np.zeros(test_episode_num)
    for i in range(test_episode_num):
        entropies = np.array(test_record[i])
        test_means[i] = entropies.mean()
        test_stds[i] = entropies.std()

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = int(train_episode_num / test_episode_num)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean ± std", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3)
    plt.plot(test_episodes, test_means, label="testing: mean ± std", color="red", linewidth=1)
    plt.fill_between(test_episodes, test_means-test_stds, test_means+test_stds, color="red", alpha=0.3)
    plt.xlabel("trained episodes", fontsize=14)
    
    plt.ylabel("entropy", fontsize=14)
    plt.title("Entropy over Episodes", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"{name}.png", bbox_inches='tight')
    plt.close()


def plot_gjsd(gjsd_record: dict, num_options: int, save_dir: Path, save=True): 
    """Plot the Generalized Jensen-Shannon Divergence (GJSD) recorded during training and testing."""
    # train_record: (episode_num, timestep_num)
    train_record = gjsd_record["train"]
    train_episode_num = len(train_record)
    train_means = np.zeros(train_episode_num)
    train_stds = np.zeros(train_episode_num)
    for i in range(train_episode_num):
        train_means[i] = np.mean(train_record[i])
        train_stds[i] = np.std(train_record[i])
    
    # test_record: (episode_num, timestep_num)
    test_record = gjsd_record["test"]
    test_episode_num = len(test_record)
    test_means = np.zeros(test_episode_num)
    test_stds = np.zeros(test_episode_num)
    for i in range(test_episode_num):
        test_means[i] = np.mean(test_record[i])
        test_stds[i] = np.std(test_record[i])

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = int(train_episode_num / test_episode_num)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    
    plt.plot(train_episodes, train_means, label="training: mean ± std", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3)
    plt.plot(test_episodes, test_means, label="testing: mean ± std", color="red", linewidth=1)
    plt.fill_between(test_episodes, test_means-test_stds, test_means+test_stds, color="red", alpha=0.3)
    
    plt.axhline(np.log(num_options), color='red', linestyle='--', linewidth=1)
    plt.axhline(np.log(num_options)*0.7, color='red', linestyle='--', linewidth=1)
    plt.axhline(np.log(num_options)*0.3, color='red', linestyle='--', linewidth=1)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("GJSD", fontsize=14)
    plt.title("Generalized Jensen-Shannon Divergence (GJSD) over Episodes", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"gjsd.png", bbox_inches='tight')
    plt.close()

def plot_gjsd_return(test_gjsds: list[list[float]], test_scores: list[float], test_frequency: int, num_options: int, save_dir: Path, save=True):
    # gjsd_record: (episode_num, timestep_num)
    test_episode_num = len(test_gjsds)
    gjsd_means = np.zeros(test_episode_num)
    gjsd_stds = np.zeros(test_episode_num)
    for i in range(test_episode_num):
        gjsd_means[i] = np.mean(test_gjsds[i])
        gjsd_stds[i] = np.std(test_gjsds[i])

    # score_record: (episode_num,)
    assert len(test_scores) == test_episode_num

    test_episodes = np.arange(1, test_episode_num+1) * test_frequency
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(test_episodes, gjsd_means, label="testing GJSD: mean ± std", color="red", linewidth=1)
    ax1.fill_between(test_episodes, gjsd_means-gjsd_stds, gjsd_means+gjsd_stds, color="red", alpha=0.3)
    ax1.axhline(np.log(num_options), color='red', linestyle='--', linewidth=1)
    ax1.axhline(np.log(num_options)*0.7, color='red', linestyle='--', linewidth=1)
    ax1.axhline(np.log(num_options)*0.3, color='red', linestyle='--', linewidth=1)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_ylabel("GJSD", fontsize=14, color='red')
    ax1.legend(loc='upper left', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(test_episodes, test_scores, label="testing score", color="black", linewidth=1)
    ax2.set_ylabel("score", fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid()
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.title("GJSD & Return over Trained Episodes", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"testing_gjsd_return.png", bbox_inches='tight')
    plt.close()

def plot_option_usage(option_indices: list[list[int]], option_rollout_lengths: list[list[float]], num_options: int, name: str, save_dir: Path, save=True) -> None:
    episode_num = len(option_indices)
    assert len(option_rollout_lengths) == episode_num
    counts = np.zeros((episode_num, num_options), dtype=float)

    for i, (indices, rollout_lengths) in enumerate(zip(option_indices, option_rollout_lengths)):
        if len(rollout_lengths) == 0:
            lengths = np.ones(len(indices), dtype=float)
        else:
            lengths = np.array(rollout_lengths, dtype=float)
        for opt_idx, length in zip(indices, lengths):
            counts[i, opt_idx] += length

    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    pct = counts / totals * 100.0

    episode_x = np.arange(1, episode_num+1)
    bottom = np.zeros(episode_num, dtype=float)

    # cmap = plt.get_cmap("tab20")
    # colors = [cmap(i % cmap.N) for i in range(num_options)]
    colors = ["blue", "orange", "green", "red", "brown", "purple", "gray", "black"][:num_options]

    fig, ax = plt.subplots(figsize=(12, 6))
    # fig_width_inches = 12
    # bar_width = (fig_width_inches * 0.9) / episode_num
    # bar_width = min(bar_width, 0.8)
    
    for opt in range(num_options):
        # bar chart
        heights = pct[:, opt]
        ax.bar(episode_x, heights, bottom=bottom, color=colors[opt],
               width=0.8, edgecolor="black", linewidth=0, label=f"option {opt}")
        for x, b, h, p in zip(episode_x, bottom, heights, pct[:, opt]):
            break
            if h >= 5:  # only label segments large enough to read
                ax.text(x, b+h/2, f"{p:.0f}%", ha="center", va="center", 
                        fontsize=8, color="white" if sum(colors[opt][:3]) < 1.5 else "black")
        bottom += heights

        # line chart
        # ax.plot(episode_x, pct[:, opt], color=colors[opt], linewidth=1, label=f"option {opt}")

    ax.set_xlabel("episode", fontsize=14)
    ax.set_ylabel("option usage (%)", fontsize=14)
    ax.set_title(f"{name}: Option usage per episode", fontsize=16)
    # ax.set_xticks(episode_x)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=14, ncol=1)
    ax.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"option_usage_{name}.png", bbox_inches="tight")        
    plt.close()

