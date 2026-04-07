import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from RL.record import Record
from RL.environment import Environment


def plot_return(train_scores: list[float], test_scores: list[float], num_options: int, save_dir: Path, save=True) -> None:
    train_episode_num = len(train_scores)
    test_total_num = len(test_scores)
    episode_per_test = int(train_episode_num / (test_total_num / num_options))

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    # plt.plot(train_episodes, train_scores, label="training", color='black', linestyle='--', linewidth=1)
    for option_idx in range(num_options):
        option_test_scores = test_scores[option_idx::num_options]
        best_episode = np.argmax(option_test_scores)
        best_score = option_test_scores[best_episode]
        plt.plot(test_episodes, option_test_scores, label=f"testing (option {option_idx}), best score: {best_score:.2f} (episode {(best_episode+1)*episode_per_test})", linewidth=1)
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("cumulative reward", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Return over Trained Episodes", fontsize=16)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"return.png", bbox_inches='tight')


def plot_fail_name(train_fail_names: list[str], test_fail_names: list[str], train_option_indices: list[list[int]], test_option_indices: list[list[int]], num_options: int, save_dir: Path, save=True) -> None:
    train_names = {}
    test_names = {}
    
    # Count training failures by name and option
    for i, name in enumerate(train_fail_names):
        if name == None: name = "none"
        if name not in train_names:
            train_names[name] = [0] * num_options
        train_option_idx = train_option_indices[i][0]
        assert np.all(np.array(train_option_indices[i]) == train_option_idx), "option in the same episode should be the same"
        train_names[name][train_option_idx] += 1
    
    # Count testing failures by name and option
    for i, name in enumerate(test_fail_names):
        if name == None: name = "none"
        if name not in test_names:
            test_names[name] = [0] * num_options
        test_option_idx = test_option_indices[i][0]
        assert np.all(np.array(test_option_indices[i]) == test_option_idx), "option in the same episode should be the same"
        test_names[name][test_option_idx] += 1

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training failures
    names = list(train_names.keys())
    x = np.arange(len(names))
    colors = plt.cm.tab10(np.linspace(0, 1, num_options))
    for option_idx in range(num_options):
        values = [train_names[name][option_idx] for name in names]
        axs[0].bar(x, values, label=f"option {option_idx}: {sum(values)}", color=colors[option_idx], bottom=[sum([train_names[name][j] for j in range(option_idx)]) for name in names])
        for i, (name, v) in enumerate(zip(names, values)):
            if v > 0:
                axs[0].text(i, sum([train_names[name][j] for j in range(option_idx)]) + v/2, str(v), ha='center', va='center', fontsize=10)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(names)
    axs[0].set_title("training fail names")
    axs[0].legend()
    
    # Plot testing failures
    names = list(test_names.keys())
    x = np.arange(len(names))
    for option_idx in range(num_options):
        values = [test_names[name][option_idx] for name in names]
        axs[1].bar(x, values, label=f"option {option_idx}: {sum(values)}", color=colors[option_idx], bottom=[sum([test_names[name][j] for j in range(option_idx)]) for name in names])
        for i, (name, v) in enumerate(zip(names, values)):
            if v > 0:
                axs[1].text(i, sum([test_names[name][j] for j in range(option_idx)]) + v/2, str(v), ha='center', va='center', fontsize=10)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names)
    axs[1].set_title("testing fail names")
    axs[1].legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"fail_names.png", bbox_inches='tight')
    plt.close()


def plot_fail_reason(train_fail_reasons: list[str], test_fail_reasons: list[str], train_option_indices: list[list[int]], test_option_indices: list[list[int]], num_options: int, save_dir: Path, save=True) -> None:
    # full string is too long, only keep first half
    train_fail_reasons = [reason.split('_')[0] for reason in train_fail_reasons]
    test_fail_reasons = [reason.split('_')[0] for reason in test_fail_reasons]
    
    train_reasons = {}
    test_reasons = {}
    
    # Count training failures by reason and option
    for i, reason in enumerate(train_fail_reasons):
        if reason not in train_reasons:
            train_reasons[reason] = [0] * num_options
        train_option_idx = train_option_indices[i][0]
        assert np.all(np.array(train_option_indices[i]) == train_option_idx), "option in the same episode should be the same"
        train_reasons[reason][train_option_idx] += 1
    
    # Count testing failures by reason and option
    for i, reason in enumerate(test_fail_reasons):
        if reason not in test_reasons:
            test_reasons[reason] = [0] * num_options
        test_option_idx = test_option_indices[i][0]
        assert np.all(np.array(test_option_indices[i]) == test_option_idx), "option in the same episode should be the same"
        test_reasons[reason][test_option_idx] += 1

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training failure reasons
    reasons = list(train_reasons.keys())
    x = np.arange(len(reasons))
    colors = plt.cm.tab10(np.linspace(0, 1, num_options))
    for option_idx in range(num_options):
        values = [train_reasons[reason][option_idx] for reason in reasons]
        axs[0].bar(x, values, label=f"option {option_idx}: {sum(values)}", color=colors[option_idx], bottom=[sum([train_reasons[reason][j] for j in range(option_idx)]) for reason in reasons])
        for i, (reason, v) in enumerate(zip(reasons, values)):
            if v > 0:
                axs[0].text(i, sum([train_reasons[reason][j] for j in range(option_idx)]) + v/2, str(v), ha='center', va='center', fontsize=10)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(reasons)
    axs[0].set_title("training fail reasons")
    axs[0].legend()
    
    # Plot testing failure reasons
    reasons = list(test_reasons.keys())
    x = np.arange(len(reasons))
    for option_idx in range(num_options):
        values = [test_reasons[reason][option_idx] for reason in reasons]
        axs[1].bar(x, values, label=f"option {option_idx}: {sum(values)}", color=colors[option_idx], bottom=[sum([test_reasons[reason][j] for j in range(option_idx)]) for reason in reasons])
        for i, (reason, v) in enumerate(zip(reasons, values)):
            if v > 0:
                axs[1].text(i, sum([test_reasons[reason][j] for j in range(option_idx)]) + v/2, str(v), ha='center', va='center', fontsize=10)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(reasons)
    axs[1].set_title("testing fail reasons")
    axs[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"fail_reasons.png", bbox_inches='tight')
    plt.close()


def plot_testing_behavior(rec: Record, env: Environment, num_options: int, save_dir: Path, save=True) -> None:
    train_scores = rec.training_record["score"]
    test_scores = rec.testing_record["score"]
    test_actions = rec.testing_record["action"]
    story_num = env._testing_structure.story_num

    test_scores = np.array(test_scores).reshape(-1, num_options)  # (episode_num, option_num)
    best_option_idx = np.max(test_scores, axis=0).argmax()
    assert best_option_idx in np.arange(num_options).tolist(), "best_option_idx should be a valid option index"

    action_types = []
    for action in test_actions[best_option_idx::num_options]:  # select actions corresponding to the best option
        types = []
        for a in action:
            if a < story_num: type = "xdir-beam"
            elif a < story_num*2: type = "zdir-beam"
            elif a < story_num*3: type = "out-col"
            else: type = "in-col"
            types.append(type)
        action_types.append(types)

    train_episode_num = len(train_scores)
    test_episode_num = test_scores.shape[0]
    episode_per_test = train_episode_num / test_episode_num
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    color_mapping = {'xdir-beam': 'dodgerblue', 'zdir-beam': 'yellowgreen', 'out-col': 'orange', 'in-col': 'red'}

    fig, ax1 = plt.subplots(figsize=(10, 8))
    for i, types in enumerate(action_types):
        for j, type in enumerate(types):
            color = color_mapping[type]
            count = 1
            ax1.bar(test_episodes[i], count, color=color, width=3, bottom=j, zorder=1)
    ax1.set_xlabel('trained episodes', fontsize=16)
    ax1.set_ylabel('iteration', fontsize=16)
    ax1.tick_params(labelsize=14)

    ax2 = ax1.twinx()
    # for option_idx in range(num_options):
    #     option_test_scores = test_scores[:, option_idx]
    #     ax2.plot(test_episodes, option_test_scores, label=f"testing (option {option_idx})", linewidth=1, zorder=2)
    ax2.plot(test_episodes, test_scores.max(axis=1), label="testing (max over options)", color="black", linewidth=1, zorder=2)
    test_means = test_scores.mean(axis=1)
    test_stds = test_scores.std(axis=1)
    ax2.plot(test_episodes, test_means, label="testing (mean ± std over options)", color="gray", linewidth=1, zorder=2)
    ax2.fill_between(test_episodes, test_means-test_stds, test_means+test_stds, color="gray", alpha=0.3, zorder=2)
    ax2.set_ylabel('test score', fontsize=16)
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




def plot_advantage(advantage_record: dict, num_options: int, save_dir: Path, save=True) -> None:
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
    # test_record: (episode_num*option_num, timestep_num)
    test_record = advantage_record["test"]
    test_total_num = len(test_record)
    test_means = np.zeros(test_total_num)
    test_stds = np.zeros(test_total_num)
    for i in range(test_total_num):
        advantages = np.array(test_record[i])
        test_means[i] = advantages.mean()
        test_stds[i] = advantages.std()
    test_means = test_means.reshape(-1, num_options)  # (episode_num, option_num)
    test_stds = test_stds.reshape(-1, num_options)  # (episode_num, option_num)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = train_episode_num / (test_total_num / num_options)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3, label="training: mean ± std")
    for option_idx in range(num_options):
        means = test_means[:, option_idx]
        stds = test_stds[:, option_idx]
        plt.plot(test_episodes, means, label=f"testing (option {option_idx}): mean ± std", linewidth=1)
        plt.fill_between(test_episodes, means-stds, means+stds, alpha=0.3)
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("advantages", fontsize=14)
    plt.title("Advantages over Episodes", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"advantages.png", bbox_inches='tight')


def plot_entropy(entropy_record: dict, num_options: int, save_dir=None, save=True):
    # train_record: (episode_num, timestep_num)
    train_record = entropy_record["train"]
    train_episode_num = len(train_record)
    train_means = np.zeros(train_episode_num)
    train_stds = np.zeros(train_episode_num)
    for i in range(train_episode_num):
        entropies = np.array(train_record[i])
        train_means[i] = entropies.mean()
        train_stds[i] = entropies.std()
    # test_record: (episode_num*option_num, timestep_num)
    test_record = entropy_record["test"]
    test_total_num = len(test_record)
    test_means = np.zeros(test_total_num)
    test_stds = np.zeros(test_total_num)
    for i in range(test_total_num):
        entropies = np.array(test_record[i])
        test_means[i] = entropies.mean()
        test_stds[i] = entropies.std()
    test_means = test_means.reshape(-1, num_options)  # (episode_num, option_num)
    test_stds = test_stds.reshape(-1, num_options)  # (episode_num, option_num)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = train_episode_num / (test_total_num / num_options)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3, label="training: mean ± std")
    for option_idx in range(num_options):
        means = test_means[:, option_idx]
        stds = test_stds[:, option_idx]
        plt.plot(test_episodes, means, label=f"testing (option {option_idx}): mean ± std", linewidth=1)
        plt.fill_between(test_episodes, means-stds, means+stds, alpha=0.3)
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("entropy", fontsize=14)
    plt.title("Entropy over Episodes", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"entropy.png", bbox_inches='tight')




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
    # test_record: (episode_num*option_num, timestep_num)
    test_record = gjsd_record["test"]
    test_total_num = len(test_record)
    test_means = np.zeros(test_total_num)
    test_stds = np.zeros(test_total_num)
    for i in range(test_total_num):
        test_means[i] = np.mean(test_record[i])
        test_stds[i] = np.std(test_record[i])
    test_means = test_means.reshape(-1, num_options)  # (episode_num, option_num)
    test_stds = test_stds.reshape(-1, num_options)  # (episode_num, option_num)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = train_episode_num / (test_total_num / num_options)
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3, label="training: mean ± std")
    for option_idx in range(num_options):
        means = test_means[:, option_idx]
        stds = test_stds[:, option_idx]
        plt.plot(test_episodes, means, label=f"testing (option {option_idx}): mean ± std", linewidth=1)
        plt.fill_between(test_episodes, means-stds, means+stds, alpha=0.3)
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("GJSD", fontsize=14)
    plt.title("Generalized Jensen-Shannon Divergence (GJSD) over Episodes", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"gjsd.png", bbox_inches='tight')


# for testing, plot GJSD (max, mean ± std over options) and return (max, mean ± std over options) curves together
def plot_gjsd_return(test_gjsds: list[list[float]], test_scores: list[float], num_options: int, test_frequency: int, save_dir: Path, save=True):
    # test_gjsds: (episode_num*option_num, timestep_num)
    test_total_num = len(test_gjsds)
    gjsd_means = np.zeros(test_total_num)
    for i in range(test_total_num):
        gjsd_means[i] = np.mean(test_gjsds[i])
    gjsd_means = gjsd_means.reshape(-1, num_options)  # (episode_num, option_num)
    gjsd_means_option_maxs = gjsd_means.max(axis=1)  # (episode_num,)
    gjsd_means_option_means = gjsd_means.mean(axis=1)  # (episode_num,)
    gjsd_means_option_stds = gjsd_means.std(axis=1)  # (episode_num,)

    # test_scores: (episode_num*option_num,)
    test_scores = np.array(test_scores).reshape(-1, num_options)  # (episode_num, option_num)
    score_option_maxs = test_scores.max(axis=1)  # (episode_num,)
    score_option_means = test_scores.mean(axis=1)  # (episode_num,)
    score_option_stds = test_scores.std(axis=1)  # (episode_num,)

    test_episode_num = test_total_num // num_options
    test_episodes = np.arange(1, test_episode_num+1) * test_frequency
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(test_episodes, gjsd_means_option_maxs, label="testing GJSD (max over options)", color="red", linewidth=1)
    ax1.plot(test_episodes, gjsd_means_option_means, label="testing GJSD (mean ± std over options)", color="orange", linewidth=1)
    ax1.fill_between(test_episodes, gjsd_means_option_means-gjsd_means_option_stds, gjsd_means_option_means+gjsd_means_option_stds, color="orange", alpha=0.3)
    ax1.axhline(np.log(num_options), color='red', linestyle='--', linewidth=1)
    ax1.axhline(np.log(num_options)*0.7, color='red', linestyle='--', linewidth=1)
    ax1.axhline(np.log(num_options)*0.3, color='red', linestyle='--', linewidth=1)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel("trained episodes", fontsize=14)
    ax1.set_ylabel("GJSD", fontsize=14)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid()
    
    ax2 = ax1.twinx()
    ax2.plot(test_episodes, score_option_maxs, label="testing score (max over options)", color="black", linewidth=1)
    ax2.plot(test_episodes, score_option_means, label="testing score (mean ± std over options)", color="gray", linewidth=1)
    ax2.fill_between(test_episodes, score_option_means-score_option_stds, score_option_means+score_option_stds, color="gray", alpha=0.3)
    ax2.set_ylabel("score", fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.title("GJSD & Return over Trained Episodes", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"testing_gjsd_return.png", bbox_inches='tight')
    
