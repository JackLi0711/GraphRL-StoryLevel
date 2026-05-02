import numpy as np
import matplotlib.pyplot as plt


def plot_advantage(advantage_record: dict, save_dir=None, save=True):
    # advantage_record shape: (episode_num, optimization_epoch, timestep_num)
    train_record = advantage_record["train"]
    train_episode_num = len(train_record)
    train_means = np.zeros(train_episode_num)
    train_stds = np.zeros(train_episode_num)
    for i in range(train_episode_num):
        # advantages at different optimization epochs are all the same, so just take the first one
        advantages = np.array(train_record[i][0])
        train_means[i] = advantages.mean()
        train_stds[i] = advantages.std()
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
    episode_per_test = train_episode_num / test_episode_num
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3, label="training: mean ± std")
    plt.plot(test_episodes, test_means, label="testing: mean", color="red", linestyle='-', linewidth=2)
    plt.fill_between(test_episodes, test_means-test_stds, test_means+test_stds, color="pink", alpha=0.3, label="testing: mean ± std")
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("advantage", fontsize=14)
    plt.title("Advantage over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"advantage.png", bbox_inches='tight')


def plot_entropy(entropy_record: dict, save_dir=None, save=True):
    # entropy_record shape: (episode_num, timestep_num)
    train_record = entropy_record["train"]
    train_episode_num = len(train_record)
    train_means = np.zeros(train_episode_num)
    train_stds = np.zeros(train_episode_num)
    for i in range(train_episode_num):
        train_means[i] = np.array(train_record[i]).mean()
        train_stds[i] = np.array(train_record[i]).std()
    test_record = entropy_record["test"]
    test_episode_num = len(test_record)
    test_means = np.zeros(test_episode_num)
    test_stds = np.zeros(test_episode_num)
    for i in range(test_episode_num):
        test_means[i] = np.array(test_record[i]).mean()
        test_stds[i] = np.array(test_record[i]).std()

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, train_episode_num+1)
    episode_per_test = train_episode_num / test_episode_num
    test_episodes = np.arange(episode_per_test, train_episode_num+1, episode_per_test)
    plt.plot(train_episodes, train_means, label="training: mean", color="black", linestyle='--', linewidth=1)
    plt.fill_between(train_episodes, train_means-train_stds, train_means+train_stds, color="gray", alpha=0.3, label="training: mean ± std")
    plt.plot(test_episodes, test_means, label="testing: mean", color="red", linestyle='-', linewidth=2)
    plt.fill_between(test_episodes, test_means-test_stds, test_means+test_stds, color="pink", alpha=0.3, label="testing: mean ± std")
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("entropy", fontsize=14)
    plt.title("Entropy over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"entropy.png", bbox_inches='tight')


def plot_explained_variance(return_record, value_record, save_dir=None, save=True):
    # return_record shape: (episode_num, optimization_epoch, timestep_num)
    # value_record shape: (episode_num, optimization_epoch, timestep_num)
    episode_num = len(return_record)
    optimization_epoch_num = len(return_record[0])
    assert (episode_num == len(value_record)) and (optimization_epoch_num == len(value_record[0])), "Return and value records must have the same shape."

    explained_variances = np.zeros((episode_num, optimization_epoch_num))
    for i in range(episode_num):
        for j in range(optimization_epoch_num):
            returns = np.array(return_record[i][j])
            values = np.array(value_record[i][j])
            explained_variances[i][j] = 1 - (np.var(returns - values) / np.var(returns))
    explained_variance_means = explained_variances.mean(axis=1)
    explained_variance_stds = explained_variances.std(axis=1)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, episode_num+1)
    plt.plot(train_episodes, explained_variance_means, label="mean over optimization epochs", color="black", linestyle='-', linewidth=1)
    plt.fill_between(train_episodes, explained_variance_means-explained_variance_stds, explained_variance_means+explained_variance_stds, color="gray", alpha=0.3, label="mean ± std")
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("explained variance", fontsize=14)
    plt.title("Explained Variance over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/"explained_variance.png", bbox_inches='tight')


def plot_kl_divergence(kl_record, target_kl=0.02, name="kl_divergence", save_dir=None, save=True):
    # kl_record shape: (episode_num, optimization_epoch)
    kl_array = np.array(kl_record)
    kl_means = kl_array.mean(axis=1)
    kl_stds = kl_array.std(axis=1)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, len(kl_array)+1)
    label = f"mean over optimization epochs\nmax: {kl_means.max():.4f} (at episode {kl_means.argmax()+1})\nmin: {kl_means.min():.4f} (at episode {kl_means.argmin()+1})"
    plt.plot(train_episodes, kl_means, label=label, color="black", linestyle='-', linewidth=1)
    plt.fill_between(train_episodes, kl_means-kl_stds, kl_means+kl_stds, color="gray", alpha=0.3, label="mean ± std")
    exceed_ratio = (kl_means > 1.5*target_kl).sum() / len(kl_means)
    label = f'update threshold: 1.5 * {target_kl} = {1.5*target_kl}\nexceed ratio: {exceed_ratio:.2%}'
    plt.hlines(y=1.5*target_kl, xmin=0, xmax=len(kl_array), colors='red', linestyles='--', label=label)
    # plt.ylim(kl_means.mean()-3*kl_means.std(), kl_means.mean()+3*kl_means.std())
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("KL divergence", fontsize=14)
    plt.title("KL Divergence over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"{name}.png", bbox_inches='tight')


def plot_clip_fraction(ratio_record, clip_eps=0.2, name="clip_fraction", save_dir=None, save=True):
    # ratio_record shape: (episode_num, optimization_epoch, timestep_num)
    episode_num = len(ratio_record)
    optimization_epoch_num = len(ratio_record[0])
    clip_fractions = np.zeros((episode_num, optimization_epoch_num))
    for i in range(episode_num):
        for j in range(optimization_epoch_num):
            ratios = np.array(ratio_record[i][j])
            clip_fractions[i][j] = (np.abs(ratios - 1.0) > clip_eps).mean()
    clip_fraction_means = clip_fractions.mean(axis=1)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, episode_num+1)
    plt.plot(train_episodes, clip_fraction_means, label="mean over optimization epochs", color="black", linestyle='-', linewidth=1)
    exceed_ratio = (clip_fraction_means > clip_eps).sum() / len(clip_fraction_means)
    label = f'clip threshold: {clip_eps}\nexceed ratio: {exceed_ratio:.2%}'
    plt.hlines(y=clip_eps, xmin=0, xmax=episode_num, colors='red', linestyles='--', label=label)
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("clip fraction", fontsize=14)
    plt.title("Clip Fraction over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"{name}.png", bbox_inches='tight')


def plot_loss(loss_record, loss_type: str, save_dir=None, save=True):
    # loss_record shape: (episode_num, optimization_epoch)
    loss_array = np.array(loss_record)
    mean_loss = loss_array.mean(axis=1)

    plt.figure(figsize=(12, 6))
    train_episodes = np.arange(1, len(loss_array)+1)
    label = f"mean over optimization epochs\nmax: {mean_loss.max():.4f} (at episode {mean_loss.argmax()+1})\nmean: {mean_loss.mean():.4f}\nmin: {mean_loss.min():.4f} (at episode {mean_loss.argmin()+1})"
    plt.plot(train_episodes, mean_loss, label=label, color="black", linestyle='-', linewidth=1)
    # plt.ylim(mean_loss.mean()-3*mean_loss.std(), mean_loss.mean()+3*mean_loss.std())
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.title(f"{loss_type.capitalize()} Losses over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"loss_{loss_type}.png", bbox_inches='tight')


def plot_grad_norm(grad_norm_record: dict, save_dir=None, save=True):
    # grad_norm_record shape: (episode_num, optimization_epoch)

    plt.figure(figsize=(12, 6))
    for model_part, norms in grad_norm_record.items():
        norm_array = np.array(norms)
        train_episodes = np.arange(1, len(norm_array)+1)
        mean_norm = norm_array.mean(axis=1)
        label = f"{model_part}\nmax: {mean_norm.max():.3f} (at episode {mean_norm.argmax()+1})\nmean: {mean_norm.mean():.3f}\nmin: {mean_norm.min():.3f} (at episode {mean_norm.argmin()+1})"
        plt.plot(train_episodes, mean_norm, label=label, linestyle='-', linewidth=1)
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("gradient norm (L2 norm)", fontsize=14)
    plt.title(f"Gradient Norms over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"grad_norm.png", bbox_inches='tight')


def plot_param_change(param_change_record: dict, save_dir=None, save=True):
    # param_change_record shape: (episode_num)
    
    plt.figure(figsize=(12, 6))
    for model_part, changes in param_change_record.items():
        change_array = np.array(changes)
        train_episodes = np.arange(1, len(change_array)+1)
        label = f"{model_part}\nmax: {change_array.max():.6f} (at episode {change_array.argmax()+1})\nmean: {change_array.mean():.6f}\nmin: {change_array.min():.6f} (at episode {change_array.argmin()+1})"
        plt.plot(train_episodes, change_array, label=label, linestyle='-', linewidth=1)
    
    plt.xlabel("trained episodes", fontsize=14)
    plt.ylabel("parameter change (L2 norm)", fontsize=14)
    plt.title(f"Parameter Changes over Trained Episodes", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(save_dir/f"param_change.png", bbox_inches='tight')
