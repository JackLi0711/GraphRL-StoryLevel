import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

    plt.savefig(checkpoint_dir / "fail_reasons.png")
    plt.close()


def plot_test_behaviors(rec: Record, env: Environment, checkpoint_dir: Path) -> None:
    train_scores = rec.training_record["score"]
    test_scores = rec.testing_record["score"]
    test_actions, test_actions_SCWB = rec.testing_record["action"], rec.testing_record["action_SCWB"]
    test_options = rec.testing_record.get("option", None)  # Get option sequences (optional for hierarchical RL)
    test_option_instances = rec.testing_record.get("option_instances", None)  # Get option instance data
    story_num = env._testing_structure.story_num
    

    # If there is no test data, skip plotting
    if len(test_scores) == 0 or len(test_actions) == 0:
        return

    action_types = []
    for action in test_actions:
        types = []
        for a in action:
            # Handle different tensor types and extract action index
            if hasattr(a, 'item'):
                # Single element tensor
                if a.numel() == 1:
                    a_val = a.item()
                else:
                    # Multi-element tensor - get the argmax as action index
                    a_val = a.argmax().item()
            else:
                a_val = a

            if a_val < story_num: action_type = "xdir-beam"
            elif a_val < story_num*2: action_type = "zdir-beam"
            elif a_val < story_num*3: action_type = "out-col"
            else: action_type = "in-col"
            types.append(action_type)
        action_types.append(types)
        
    # robust episode mapping
    if len(train_scores) > 0:
        test_episodes = np.linspace(1, len(train_scores), num=len(test_scores))
    else:
        test_episodes = np.arange(1, len(test_scores)+1, 1)
    color_mapping = {'xdir-beam': 'dodgerblue', 'zdir-beam': 'yellowgreen', 'out-col': 'orange', 'in-col': 'red'}

    fig, ax1 = plt.subplots(figsize=(10, 8))

    for i, types in enumerate(action_types):
        for j, action_type in enumerate(types):
            color = color_mapping[action_type]
            count = 1
            ax1.bar(test_episodes[i], count, color=color, width=3, bottom=j, zorder=1)
    
    # Add option boundaries with thick black rectangles
    # Use option instances data if available, otherwise fall back to old method
    use_new_method = False
    if test_option_instances is not None and len(test_option_instances) > 0:
        use_new_method = True
        # New method using option instance data
        for i, (actions, instances) in enumerate(zip(test_actions, test_option_instances)):
            if not instances or len(instances) == 0:
                continue
            
            # Validate that instances is a list and has the expected structure
            if not isinstance(instances, list) or not isinstance(instances[0], dict):
                print(f"Warning: Invalid instances format for episode {i}, falling back to old method")
                use_new_method = False
                break
                
            if "instance_id" not in instances[0] or "option_index" not in instances[0]:
                print(f"Warning: Missing keys in instances for episode {i}, falling back to old method")
                use_new_method = False
                break
            
            # Group consecutive steps by option instance ID
            current_instance_id = instances[0]["instance_id"]
            current_option_index = instances[0]["option_index"]
            option_start = 0
            
            for j in range(1, len(instances)):
                # Check if option instance changed or if we're at the end
                if instances[j]["instance_id"] != current_instance_id or j == len(instances) - 1:
                    # Option instance boundary detected or end of episode
                    option_end = j if instances[j]["instance_id"] != current_instance_id else j + 1
                    
                    # Draw thick black rectangle around this option instance
                    rect_x = test_episodes[i] - 1.5  # Adjust for bar width
                    rect_width = 3  # Match bar width
                    rect_y = option_start
                    rect_height = option_end - option_start
                    
                    rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                                   linewidth=3, edgecolor='black', facecolor='none', 
                                   zorder=3, linestyle='-')
                    ax1.add_patch(rect)
                    
                    # Add option number label with instance info
                    ax1.text(rect_x + rect_width/2, rect_y + rect_height/2, 
                            f'O{current_option_index}', ha='center', va='center', 
                            fontsize=8, fontweight='bold', color='black', zorder=4,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                    
                    # Update for next option instance
                    if j < len(instances):
                        current_instance_id = instances[j]["instance_id"]
                        current_option_index = instances[j]["option_index"]
                    option_start = j
    
    # Fall back to old method if new method was not used or failed
    if not use_new_method and test_options is not None:
        # Fall back to old method using option index changes
        for i, (actions, options) in enumerate(zip(test_actions, test_options)):
            if len(options) == 0:
                continue
            
            # Find option boundaries
            current_option = options[0]
            option_start = 0
            
            for j in range(1, len(options)):
                if options[j] != current_option or j == len(options) - 1:
                    # Option boundary detected or end of episode
                    option_end = j if options[j] != current_option else j + 1
                    
                    # Draw thick black rectangle around this option
                    rect_x = test_episodes[i] - 1.5  # Adjust for bar width
                    rect_width = 3  # Match bar width
                    rect_y = option_start
                    rect_height = option_end - option_start
                    
                    rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                                   linewidth=3, edgecolor='black', facecolor='none', 
                                   zorder=3, linestyle='-')
                    ax1.add_patch(rect)
                    
                    # Add option number label
                    ax1.text(rect_x + rect_width/2, rect_y + rect_height/2, 
                            f'O{current_option}', ha='center', va='center', 
                            fontsize=8, fontweight='bold', color='black', zorder=4,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                    
                    # Update for next option
                    current_option = options[j] if j < len(options) else current_option
                    option_start = j

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




def plot_eval_score_mean_std(eval_round_episode_scores: List[List[float]], round_numbers: List[int], checkpoint_dir: Path) -> None:
    """Plot mean test eval score per round with ±1 std shaded region.

    Args:
        eval_round_episode_scores: List of lists, each inner list holds per-episode scores for a round
        round_numbers: Training episode indices corresponding to each evaluation round
        checkpoint_dir: Directory to save the plot
    """
    if eval_round_episode_scores is None or len(eval_round_episode_scores) == 0:
        return

    means = []
    stds = []
    for scores in eval_round_episode_scores:
        arr = np.array(scores, dtype=float) if scores is not None and len(scores) > 0 else np.array([np.nan])
        means.append(float(np.nanmean(arr)))
        stds.append(float(np.nanstd(arr)))

    # X-axis: use round_numbers if valid, else simple 1..N
    if isinstance(round_numbers, (list, tuple)) and len(round_numbers) == len(means) and len(round_numbers) > 0:
        x = np.array(round_numbers, dtype=float)
        x_label = "Trained Episode (evaluation round)"
    else:
        x = np.arange(1, len(means) + 1, dtype=float)
        x_label = "Evaluation Round"

    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(x, means, color='tab:blue', linewidth=2, label='mean test eval score')
    plt.fill_between(x, means - stds, means + stds, color='tab:blue', alpha=0.2, label='±1 std')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Test eval score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "test_eval_scores_mean_std.png", dpi=300)
    plt.close()


def plot_eval_reward_mean_std(eval_round_episode_rewards: List[List[float]], round_numbers: List[int], checkpoint_dir: Path) -> None:
    """Plot mean test eval total reward per round with ±1 std shaded region.

    Args:
        eval_round_episode_rewards: List of lists, each inner list holds per-episode total rewards for a round
        round_numbers: Training episode indices corresponding to each evaluation round
        checkpoint_dir: Directory to save the plot
    """
    if eval_round_episode_rewards is None or len(eval_round_episode_rewards) == 0:
        return

    means = []
    stds = []
    for rewards in eval_round_episode_rewards:
        arr = np.array(rewards, dtype=float) if rewards is not None and len(rewards) > 0 else np.array([np.nan])
        means.append(float(np.nanmean(arr)))
        stds.append(float(np.nanstd(arr)))

    # X-axis: use round_numbers if valid, else simple 1..N
    if isinstance(round_numbers, (list, tuple)) and len(round_numbers) == len(means) and len(round_numbers) > 0:
        x = np.array(round_numbers, dtype=float)
        x_label = "Trained Episode (evaluation round)"
    else:
        x = np.arange(1, len(means) + 1, dtype=float)
        x_label = "Evaluation Round"

    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(x, means, color='tab:green', linewidth=2, label='mean test eval total reward')
    plt.fill_between(x, means - stds, means + stds, color='tab:green', alpha=0.2, label='±1 std')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Test eval total reward', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "test_eval_rewards_mean_std.png", dpi=300)
    plt.close()


def plot_training_losses(actor_losses: List[float], critic_losses: List[float], checkpoint_dir: Path) -> None:
    """Plot training losses in separate subplots.

    Args:
        actor_losses: List of actor loss values
        critic_losses: List of critic loss values
        checkpoint_dir: Directory to save the plot
    """
    if (actor_losses is None or len(actor_losses) == 0) and (critic_losses is None or len(critic_losses) == 0):
        return

    # Convert to numpy arrays and take absolute values
    actor_arr = np.array(actor_losses, dtype=float) if actor_losses and len(actor_losses) > 0 else np.array([])
    actor_arr = np.abs(actor_arr)  # Take absolute value

    critic_arr = np.array(critic_losses, dtype=float) if critic_losses and len(critic_losses) > 0 else np.array([])
    critic_arr = np.abs(critic_arr)  # Take absolute value

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Subplot 1: Actor Loss
    if len(actor_arr) > 0:
        update_steps_actor = np.arange(1, len(actor_arr) + 1)
        axes[0].plot(update_steps_actor, actor_arr, color='tab:red', linewidth=1, alpha=0.6, label='Actor Loss')

        # Add moving average smoothing if enough data points
        if len(actor_arr) > 50:
            window_size = min(50, len(actor_arr) // 10)
            actor_smooth = np.convolve(actor_arr, np.ones(window_size)/window_size, mode='valid')
            smooth_steps = np.arange(window_size, len(actor_arr) + 1)
            axes[0].plot(smooth_steps, actor_smooth, color='darkred', linewidth=2, label=f'Moving Avg (window={window_size})')

        axes[0].grid(True, linestyle='--', alpha=0.4)
        axes[0].set_xlabel('Update Steps', fontsize=12)
        axes[0].set_ylabel('Actor Loss', fontsize=12)
        axes[0].set_title('Actor Loss vs Update Steps', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].tick_params(labelsize=10)
    else:
        axes[0].text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', fontsize=14)
        axes[0].set_title('Actor Loss vs Update Steps', fontsize=14, fontweight='bold')

    # Subplot 2: Critic Loss
    if len(critic_arr) > 0:
        update_steps_critic = np.arange(1, len(critic_arr) + 1)
        axes[1].plot(update_steps_critic, critic_arr, color='tab:blue', linewidth=1, alpha=0.6, label='Critic Loss')

        # Add moving average smoothing if enough data points
        if len(critic_arr) > 50:
            window_size = min(50, len(critic_arr) // 10)
            critic_smooth = np.convolve(critic_arr, np.ones(window_size)/window_size, mode='valid')
            smooth_steps = np.arange(window_size, len(critic_arr) + 1)
            axes[1].plot(smooth_steps, critic_smooth, color='darkblue', linewidth=2, label=f'Moving Avg (window={window_size})')

        axes[1].grid(True, linestyle='--', alpha=0.4)
        axes[1].set_xlabel('Update Steps', fontsize=12)
        axes[1].set_ylabel('Critic Loss', fontsize=12)
        axes[1].set_title('Critic Loss vs Update Steps', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].tick_params(labelsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', fontsize=14)
        axes[1].set_title('Critic Loss vs Update Steps', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_losses.png", dpi=300)
    plt.close()


def plot_training_episode_rewards(episode_rewards: List[float], checkpoint_dir: Path) -> None:
    """Plot training episode total rewards.

    Args:
        episode_rewards: List of total reward per episode
        checkpoint_dir: Directory to save the plot
    """
    if episode_rewards is None or len(episode_rewards) == 0:
        return

    rewards_arr = np.array(episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards_arr) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards_arr, color='tab:purple', linewidth=1, alpha=0.6, label='Episode Total Reward')

    # Add moving average smoothing if enough data points
    if len(rewards_arr) > 10:
        window_size = min(20, len(rewards_arr) // 5)
        rewards_smooth = np.convolve(rewards_arr, np.ones(window_size)/window_size, mode='valid')
        smooth_episodes = np.arange(window_size, len(rewards_arr) + 1)
        plt.plot(smooth_episodes, rewards_smooth, color='darkviolet', linewidth=2, label=f'Moving Avg (window={window_size})')

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel('Training Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title('Training Episode Total Rewards', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_episode_rewards.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    train_fail_reasons = ["minimum_section", "minimum_section", "drift_ratio", "beam_moment", "minimum_section", "column_tension", "drift_ratio", "minimum_section"]
    test_fail_reasons = ["strong_column_weak_beam", "column_compression", "soft_story", "drift_ratio", "beam_moment", "minimum_section", "column_tension", "drift_ratio", "minimum_section"]
    plot_fail_reasons(train_fail_reasons, test_fail_reasons, None)

