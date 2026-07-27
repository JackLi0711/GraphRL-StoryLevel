import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_option_duration(option_indices: list[list[int]], num_options: int, test_frequency: int, name: str, save_dir: Path, save=True, show=False):
    # option_indices: (episode_num, timestep_num)
    max_duration_ratios = np.zeros((len(option_indices), num_options))
    for i, opt_idxs in enumerate(option_indices): 
        option_durations = {i: [0] for i in range(num_options)}
        current_option = opt_idxs[0]
        current_duration = 1
        for j in range(1, len(opt_idxs)):
            if opt_idxs[j] == current_option:
                current_duration += 1
            else:
                option_durations[current_option].append(current_duration)
                current_option = opt_idxs[j]
                current_duration = 1
        option_durations[current_option].append(current_duration)
        
        max_durations = np.array([max(durations) for durations in option_durations.values()])
        max_duration_ratios[i] = max_durations / len(opt_idxs) * 100
    
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(option_indices)+1) * test_frequency
    for o in range(num_options):
        plt.plot(episodes, max_duration_ratios[:, o], label=f"option {o}")
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("option duration ratio (%)", fontsize=14)
    plt.title("Longest Option Duration Ratios over Trained Episodes", fontsize=16)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.grid()
    if save: plt.savefig(save_dir/f"option_duration_{name}.png", bbox_inches='tight')
    if show: plt.show()
    plt.close()


def plot_option_beta(option_betas: list[list[list[float]]], option_indices: list[list[int]], num_options: int, test_frequency: int, name: str, save_dir: Path, save=True, show=False):
    # option_betas: (episode_num, timestep_num, option_num)
    # option_indices: (episode_num, timestep_num)
    means = np.zeros((len(option_betas), num_options))
    stds = np.zeros_like(means)
    selected_means = np.zeros(len(option_indices))
    selected_stds = np.zeros_like(selected_means)
    for i, (betas, indices) in enumerate(zip(option_betas, option_indices)):
        betas = np.array(betas)  # (timestep_num, option_num)
        indices = np.array(indices)  # (timestep_num,)
        selected_betas = betas[np.arange(betas.shape[0]), indices]  # (timestep_num,)
        means[i] = betas.mean(axis=0) * 100
        stds[i] = betas.std(axis=0) * 100
        selected_means[i] = selected_betas.mean() * 100
        selected_stds[i] = selected_betas.std() * 100
        np.random.randint(0, 10, size=(5,4))

    # cmap = plt.get_cmap("tab20")
    # colors = [cmap(i % cmap.N) for i in range(num_options)]
    colors = ["blue", "orange", "green", "red", "brown", "purple", "gray", "black"][:num_options]

    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(option_betas)+1) * test_frequency
    for o in range(num_options):
        plt.plot(episodes, means[:, o], label=f"option {o}: mean ± std", color=colors[o])
        plt.fill_between(episodes, means[:, o]-stds[:, o], means[:, o]+stds[:, o], color=colors[o], alpha=0.3)
    plt.plot(episodes, selected_means, label=f"used option: mean ± std", color="black", linestyle="--")
    plt.fill_between(episodes, selected_means-selected_stds, selected_means+selected_stds, color="gray", alpha=0.3)
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("probability (%)", fontsize=14)
    plt.title("Option Termination Probabilities over Trained Episodes", fontsize=16)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.grid()
    if save: plt.savefig(save_dir/f"option_beta_{name}.png", bbox_inches='tight')
    if show: plt.show()
    plt.close()


def plot_option_margin(option_values: list[list[list[float]]], test_frequency: int, name: str, save_dir: Path, save=True, show=False):
    # option_values: (episode_num, timestep_num, option_num)
    means = np.zeros(len(option_values))
    stds = np.zeros_like(means)
    for i, values in enumerate(option_values):
        values = np.array(values)  # (timestep_num, option_num)
        second_max = np.partition(values, -2, axis=1)[:, -2]
        margins = values.max(axis=1) - second_max  # (timestep_num,)  
        means[i] = margins.mean()
        stds[i] = margins.std()     
    
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(option_values)+1) * test_frequency
    plt.plot(episodes, means, label=f"mean ± std", color="black")
    plt.fill_between(episodes, means-stds, means+stds, color="gray", alpha=0.3)
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("margin", fontsize=14)
    plt.title("Option Margin over Trained Episodes", fontsize=16)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.grid()
    if save: plt.savefig(save_dir/f"option_margin_{name}.png", bbox_inches='tight')
    if show: plt.show()
    plt.close()


