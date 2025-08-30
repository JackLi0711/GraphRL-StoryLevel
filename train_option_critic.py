import argparse
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from RL.environment import Environment
from RL.option_critic_gnn import OptionCriticGNN, critic_loss, actor_loss
from RL.experience_replay import ReplayBuffer
from Validation import normalization as nda_norm
from Structure import check, check_nda


def get_graph_data(structure, device):
    """
    Extract graph data for OptionCriticGNN.
    Returns tuple of graph inputs for the neural network.
    """
    graph = structure.graph
    story_batch = structure.aux["story_batch"].to(device)
    return (
        graph.x.to(device),
        graph.edge_index.to(device),
        graph.edge_attr.to(device),
        story_batch,
        None  # structure_story_ptr
    )


def num_actions(structure) -> int:
    return len(structure.story_level_actions)


def apply_primitive_action(base_env, structure, action: int):
    """
    Apply a primitive action using the base environment.

    Returns:
        step_reward: float, reward from base env for this step
        step_pass: bool, whether constraints pass after this step
        is_minimum_section: bool, whether minimum section is reached
        fail_reason: str, fail reason string from base env
    """
    structure, step_reward, done, fail_name, fail_reason = base_env.step(structure, action)

    # Derive per-step pass/minimum-section states from base env outputs
    is_minimum_section = bool(done and (fail_reason == "minimum_section"))
    step_pass = not (done and (fail_reason != "minimum_section"))
    return structure, float(step_reward), step_pass, is_minimum_section, fail_reason


def check_constraints_without_update(structure, base_env):
    """
    Run code checks on current structure WITHOUT applying any action.
    Returns:
        whether_pass (bool), fail_reason (str)
    """
    # static analysis
    load_cases, static_responses = check.get_response(structure, base_env.code_analysis_dir)
    static_constraint_condition, static_response_features, _ = check.process_response(structure, load_cases, static_responses)
    whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, base_env.check_displacement)

    # dynamic analysis if enabled and statically passed
    dynamic_response_features = None
    if base_env.do_nonlinear_dynamic_analysis and whether_pass:
        structure.update_graph_GraphLSTM()
        dynamic_responses = check_nda.get_response(structure, base_env.nda_simulator, base_env.MCE_ground_motion_set, base_env.device)
        dynamic_constraint_condition, dynamic_response_features, _ = check_nda.process_response(structure, dynamic_responses, base_env.nda_norm_dict)
        whether_pass, fail_name, fail_reason = check_nda.check_pass(dynamic_constraint_condition, base_env.check_displacement)

    # update graph features for consistency
    structure.update_graph_GraphRL(static_response_features, dynamic_response_features)

    return whether_pass, fail_reason


def rollout_option(structure, base_env, device, max_option_len, current_option: int, oc_model, epsilon: float = None):
    """
    Execute a single option composed of a sequence of primitive actions.

    Args:
        structure: current structure state
        base_env: base Environment instance 
        device: torch device
        max_option_len: maximum option length
        current_option: selected option index
        oc_model: OptionCriticGNN model with get_state, get_action, predict_option_termination
        epsilon: optional exploration indicator (for logging)

    Returns:
        next_obs_z: aggregated state vector after option terminates (for next option)
        option_reward: accumulated reward within this option (or -1000 on failure)
        option_done: whether option terminated
        episode_done: whether episode terminated as a result of this option
        stats: dict with diagnostics (length, termination_reason, entropies, epsilon)
        step_transitions: list of step-level transitions
    """
    option_reward_sum: float = 0.0
    entropies = []
    length = 0
    termination_reason = None
    episode_done = False

    # Record pre-option material usage to compute saved amount for this option
    try:
        pre_option_material_usage = float(structure.calculate_material_usage())
    except Exception:
        pre_option_material_usage = None

    # initial state for intra-option policy
    graph_data = get_graph_data(structure, device)
    state = oc_model.get_state(*graph_data)

    step_transitions = []  # list of dicts: {obs_z, action, logp, entropy, reward, done, next_obs_z}

    while length < max_option_len:
        # intra-option action
        action, logp, entropy = oc_model.get_action(state, current_option)
        entropies.append(float(entropy.detach().cpu().numpy()))

        # Pre-check: if this action targets a minimum section, force terminate option & episode
        infeasible = set(getattr(structure, 'already_minimum_section_story_indexes', []) or [])
        if hasattr(structure, 'restrict_action_space'):
            try:
                ra = structure.restrict_action_space()
                infeasible |= set(ra if ra is not None else [])
            except Exception:
                pass
        if action in infeasible:
            termination_reason = "minimum_section"
            whether_pass, fail_reason = check_constraints_without_update(structure, base_env)
            # finalize option according to rules
            option_done = True
            episode_done = True
            # reward policy handled after loop using option_reward_sum & whether_pass
            step_pass = whether_pass
            break

        structure, step_reward, step_pass, is_min_section, fail_reason = apply_primitive_action(base_env, structure, action)
        option_reward_sum += step_reward
        length += 1

        # minimum section: force terminate option and episode
        if is_min_section:
            termination_reason = "minimum_section"
            episode_done = True
            next_graph_data = get_graph_data(structure, device)
            step_transitions.append({
                "graph_data": graph_data,
                "action": action,
                "logp": logp.detach().clone(),
                "entropy": entropy.detach().clone(),
                "reward": float(step_reward),
                "done": True,
                "next_graph_data": next_graph_data,
            })
            break

        # compute next state for termination prediction
        next_graph_data = get_graph_data(structure, device)
        next_state = oc_model.get_state(*next_graph_data)

        # record step transition
        step_transitions.append({
            "graph_data": graph_data,
            "action": action,
            "logp": logp.detach().clone(),
            "entropy": entropy.detach().clone(),
            "reward": float(step_reward),
            "done": False,
            "next_graph_data": next_graph_data,
        })

        # option termination by beta
        option_termination, _ = oc_model.predict_option_termination(next_state, current_option)
        if option_termination:
            termination_reason = "beta"
            state = next_state
            graph_data = next_graph_data
            break

        # continue the option
        state = next_state
        graph_data = next_graph_data

    # if not terminated by beta/minimum_section, it hits max length
    option_done = True
    if termination_reason is None:
        termination_reason = "max_len"

    # finalize reward per rules
    # Pass if not failed at last step (i.e., not an immediate constraint failure)
    # Note: step_pass refers to the last evaluated step
    passed = step_pass if length > 0 else True
    if not passed:
        option_reward = -1000.0
        episode_done = True
        # ensure the last step gets the terminal penalty and marks episode done
        if len(step_transitions) > 0:
            step_transitions[-1]["reward"] = -1000.0
            step_transitions[-1]["done"] = True
    else:
        option_reward = float(option_reward_sum)

    # Compute option-level saved material (before vs after this option)
    try:
        post_option_material_usage = float(structure.calculate_material_usage())
        option_saved_material = float(max(0.0, (pre_option_material_usage - post_option_material_usage))) if pre_option_material_usage is not None else float("nan")
    except Exception:
        option_saved_material = float("nan")

    stats = {
        "option_length": length,
        "termination_reason": termination_reason,
        "entropy_mean": float(torch.tensor(entropies).mean().item()) if len(entropies) > 0 else float("nan"),
        "entropy_last": entropies[-1] if len(entropies) > 0 else float("nan"),
        "epsilon": float(epsilon) if epsilon is not None else None,
        "passed": bool(passed),
        "option_saved_material": option_saved_material,
        "story_level_sections": list(getattr(structure, 'story_level_sections', [])),
    }

    next_state = state  # latest state
    return structure, next_state, option_reward, option_done, episode_done, stats, step_transitions, termination_reason


def parse_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--structure_shape", type=str, default="fixed")
    parser.add_argument("--add_structure_geometry", action="store_true", default=True)
    parser.add_argument("--add_response_features", action="store_true", default=True)
    parser.add_argument("--reward_type", type=str, default="material")
    parser.add_argument("--scwb_driven_design", action="store_true", default=False)
    parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
    parser.add_argument("--check_acceleration", action="store_true", default=False)
    parser.add_argument("--check_displacement", action="store_true", default=True)
    # OC
    parser.add_argument("--num_options", type=int, default=4)
    parser.add_argument("--max_option_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=int, default=int(1e6))
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_frequency", type=int, default=4)
    parser.add_argument("--freeze_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--termination_reg", type=float, default=0.01)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    logger = logging.getLogger("OC_Train")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    # Build base env (NDA/SCWB disabled per spec)
    nda_simulator = None
    nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints" / "option_oc"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_env = Environment(
        structure_shape=args.structure_shape,
        add_structure_geometry=args.add_structure_geometry,
        add_response_features=args.add_response_features,
        reward_type=args.reward_type,
        scwb_driven_design=False,
        do_nonlinear_dynamic_analysis=False,
        check_acceleration=args.check_acceleration,
        check_displacement=args.check_displacement,
        nda_simulator=nda_simulator,
        nda_norm_dict=nda_norm_dict,
        DBE_ground_motion_set=[],
        MCE_ground_motion_set=[],
        checkpoint_dir=ckpt_dir,
        logger=logger,
        device=device,
    )

    # Reset once to infer feature sizes
    structure = base_env.reset()
    graph = structure.graph
    node_feature_dim = graph.x.shape[1]
    edge_feature_dim = graph.edge_attr.shape[1]
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    # No need for separate adapter - OptionCriticGNN includes StateGNN directly

    # Create OptionCriticGNN model
    A = num_actions(structure)
    oc = OptionCriticGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        member_state_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=A,
        num_options=args.num_options,
        temperature=args.temperature,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        eps_test=args.eps_test,
        device=device,
        testing=False,
    )
    oc_prime = OptionCriticGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        member_state_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=A,
        num_options=args.num_options,
        temperature=args.temperature,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        eps_test=args.eps_test,
        device=device,
        testing=True,
    )
    oc_prime.load_state_dict(oc.state_dict())

    # Separate parameters for different components
    actor_params = [oc.options_W, oc.options_b]
    critic_params = [p for n, p in oc.named_parameters() 
                    if (n.startswith('Q') or n.startswith('terminations'))]
    state_gnn_params = [p for n, p in oc.named_parameters() 
                       if n.startswith('state_gnn')]
    feature_params = [p for n, p in oc.named_parameters() 
                     if n.startswith('feature_processor')]

    # Use different learning rates for different components
    gnn_lr = args.actor_lr * 0.1  # Lower learning rate for GNN
    
    actor_optimizer = optim.Adam([
        {"params": actor_params, "lr": args.actor_lr},
        {"params": state_gnn_params, "lr": gnn_lr},
        {"params": feature_params, "lr": args.actor_lr},
    ])
    critic_optimizer = optim.Adam([
        {"params": critic_params, "lr": args.critic_lr},
        {"params": state_gnn_params, "lr": gnn_lr},
        {"params": feature_params, "lr": args.critic_lr},
    ])

    # Replay buffer
    buffer = ReplayBuffer(capacity=100000)

    # Stats
    stats_path = ckpt_dir / "oc_stats.json"
    all_stats = {
        "option_lengths": [],
        "termination_reasons": {"beta": 0, "max_len": 0, "minimum_section": 0},
        "entropy_mean": [],
        "entropy_last": [],
        "epsilon": [],
        "episode_rewards": [],
        "episode_option_lengths_mean": [],
        "episode_termination_counts": [],
        "episode_entropy_mean": [],
        "episode_score": [],
        "last_pass_sections": [],
        "last_pass_saved_material": [],
    }

    for ep in range(args.epochs):
        structure = base_env.reset(testing=False)
        done = False
        option_termination = True
        curr_option = 0
        greedy_option = 0
        steps = 0

        episode_opt_lengths = []
        episode_termination_counter = {"beta": 0, "max_len": 0, "minimum_section": 0}
        episode_entropies = []
        episode_score = 0.0
        last_pass_sections = None
        last_pass_saved_material = None

        while not done:
            epsilon = oc.epsilon
            if option_termination:
                curr_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option

            structure, next_state, option_reward, option_done, episode_done, o_stats, step_transitions = rollout_option(structure, base_env, device, args.max_option_len, curr_option, oc, epsilon)

            # logging stats
            all_stats["option_lengths"].append(o_stats["option_length"])
            all_stats["termination_reasons"][o_stats["termination_reason"]] = all_stats["termination_reasons"].get(o_stats["termination_reason"], 0) + 1
            all_stats["entropy_mean"].append(o_stats["entropy_mean"])
            all_stats["entropy_last"].append(o_stats["entropy_last"])
            all_stats["epsilon"].append(o_stats["epsilon"] if o_stats["epsilon"] is not None else float(epsilon))

            # per-episode accumulators
            episode_opt_lengths.append(o_stats["option_length"])
            episode_termination_counter[o_stats["termination_reason"]] = episode_termination_counter.get(o_stats["termination_reason"], 0) + 1
            if not np.isnan(o_stats["entropy_mean"]):
                episode_entropies.append(o_stats["entropy_mean"])

            # score definition: only add reward if this option passed constraints
            if bool(o_stats.get("passed", False)):
                episode_score += float(option_reward)
                last_pass_sections = o_stats.get("story_level_sections", last_pass_sections)
                last_pass_saved_material = o_stats.get("option_saved_material", last_pass_saved_material)

            # push option transition to buffer (store current state for replay)
            current_graph_data = get_graph_data(structure, device)
            buffer.push(current_graph_data, curr_option, option_reward, current_graph_data, episode_done)  # Note: using current state for both obs and next_obs in option-level buffer

            # Per-step updates (mean-style as in original): for each intra-option step do one actor update
            if len(step_transitions) > 0:
                for tr in step_transitions:
                    a_loss = actor_loss(
                        tr["graph_data"], curr_option, tr["logp"], tr["entropy"], tr["reward"], tr["done"], tr["next_graph_data"], 
                        oc, oc_prime, args.gamma, args.termination_reg, args.entropy_reg
                    )
                    actor_optimizer.zero_grad()
                    a_loss.backward()
                    if args.grad_clip is not None and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=args.grad_clip)
                    actor_optimizer.step()

            # Critic updates on schedule (option-level replay)
            if len(buffer) > args.batch_size and (steps % args.update_frequency == 0):
                data_batch = buffer.sample(args.batch_size)
                c_loss = critic_loss(oc, oc_prime, data_batch, args.gamma)
                critic_optimizer.zero_grad()
                c_loss.backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=args.grad_clip)
                critic_optimizer.step()

                if steps % args.freeze_interval == 0:
                    oc_prime.load_state_dict(oc.state_dict())

            current_graph_data = get_graph_data(structure, device)
            state = oc.get_state(*current_graph_data)
            option_termination, greedy_option = oc.predict_option_termination(state, curr_option)
            done = episode_done
            steps += 1

        # episode-level aggregates
        all_stats["episode_rewards"].append(float(np.sum([t[2] if isinstance(t, (list, tuple)) and len(t) > 2 else 0.0 for t in []])))  # placeholder, kept for compatibility
        # use the last option reward as proxy; if需要更精準可在 roll 過程累加
        all_stats["episode_rewards"][-1] = float(option_reward) if len(all_stats["episode_rewards"]) > 0 else float(option_reward)
        all_stats["episode_score"].append(float(episode_score))
        all_stats["last_pass_sections"].append(last_pass_sections)
        all_stats["last_pass_saved_material"].append(float(last_pass_saved_material) if last_pass_saved_material is not None else None)
        all_stats["episode_option_lengths_mean"].append(float(np.mean(episode_opt_lengths)) if len(episode_opt_lengths) > 0 else 0.0)
        all_stats["episode_termination_counts"].append(episode_termination_counter)
        all_stats["episode_entropy_mean"].append(float(np.mean(episode_entropies)) if len(episode_entropies) > 0 else float("nan"))

        # persist stats every episode
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed saving stats: {e}")


    # Plot episode-level histories: episode_score and last_pass_saved_material
    try:
        def _safe_nanify(arr):
            return [float('nan') if (v is None) else float(v) for v in arr]

        # Episode Score curve
        scores = all_stats.get("episode_score", [])
        if len(scores) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(scores)+1), scores, label='Episode Score', color='#1f77b4')
            if len(scores) >= 10:
                import numpy as _np
                w = min(20, max(3, len(scores)//10))
                mv = _np.convolve(scores, _np.ones(w)/w, mode='valid')
                plt.plot(range(w, len(scores)+1), mv, label=f'Moving Avg ({w})', color='#ff7f0e')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.title('Episode Score History')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(ckpt_dir / 'episode_score.png', dpi=200)
            plt.close()

        # Last passing design saved material per episode
        last_saved = _safe_nanify(all_stats.get("last_pass_saved_material", []))
        if len(last_saved) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(last_saved)+1), last_saved, label='Last Passed Option Saved Material', color='#2ca02c')
            plt.xlabel('Episode')
            plt.ylabel('Saved Material (m^3)')
            plt.title('Last Passing Design Saved Material History')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(ckpt_dir / 'last_pass_saved_material.png', dpi=200)
            plt.close()
    except Exception as e:
        logger.warning(f"Failed plotting episode histories: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
Training script for Option-Critic on Graph-based Structural Design
Integrates StateGNN, OptionEnvironmentWrapper, and OptionCriticAgent
"""

"""
import os
import sys
import argparse
import datetime
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Add paths for imports (matching original train.py)
sys.path.append("RL/")
sys.path.append("Structure/")
sys.path.append("Visualization")
sys.path.append("NonlinearDynamicAnalysisSimulator/")

from RL.environment import Environment
from RL.option_environment import OptionEnvironmentWrapper
from RL.option_critic_agent import OptionCriticAgent


def parse_arguments():
    Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Option-Critic for Structural Design')
    
    # Environment parameters (matching original train.py)
    parser.add_argument('--structure_shape', type=str, default='random', 
                        help='Structure shape: fixed, small_random, random')
    parser.add_argument('--add_structure_geometry', action='store_true', default=True,
                        help='Add structure geometry to observation')
    parser.add_argument('--add_response_features', action='store_true', default=False,
                        help='Add response features to observation')
    parser.add_argument('--reward_type', type=str, default='material',
                        help='Reward type: material, acceleration, displacement, normalized, total, combined')
    parser.add_argument('--do_nonlinear_dynamic_analysis', action='store_true', default=False,
                        help='Perform nonlinear dynamic analysis')
    parser.add_argument('--check_acceleration', action='store_true', default=False,
                        help='Check acceleration constraints')
    parser.add_argument('--scwb_driven_design', action='store_true', default=False,
                        help='Enable strong column weak beam driven design')
    parser.add_argument('--check_displacement', action='store_true', default=True,
                        help='Check displacement constraints')
    
    # NDA simulator parameters
    parser.add_argument('--graph_lstm_dir', type=str, default=None,
                        help='Path to GraphLSTM model directory')
    parser.add_argument('--ground_motion_dir', type=str, default=None,
                        help='Path to ground motion data directory')
    parser.add_argument('--ground_motion_number', type=int, default=11,
                        help='Number of ground motions to use')
    
    # Checkpoint directory
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/option_critic',
                        help='Checkpoint directory for analysis files')
    
    # Option-Critic parameters
    parser.add_argument('--num_options', type=int, default=4,
                        help='Number of options to learn')
    parser.add_argument('--max_option_length', type=int, default=16,
                        help='Maximum primitive actions per option')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for action selection')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=8,
                        help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.1,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=20000,
                        help='Epsilon decay steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--buffer_capacity', type=int, default=10000,
                        help='Experience buffer capacity')
    parser.add_argument('--target_update_freq', type=int, default=200,
                        help='Target network update frequency')
    parser.add_argument('--update_freq', type=int, default=4,
                        help='Training frequency')
    
    # Logging parameters
    parser.add_argument('--save_freq', type=int, default=500,
                        help='Model save frequency (episodes)')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Logging frequency (episodes)')
    parser.add_argument('--save_dir', type=str, default='models/option_critic',
                        help='Directory to save models')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


def setup_environment(args, logger, device) -> OptionEnvironmentWrapper:
    Setup the training environment.
    import os
    import torch
    import logging
    from pathlib import Path
    from NonlinearDynamicAnalysisSimulator import load_simulator
    
    # Setup NDA simulator if needed
    nda_simulator = None
    nda_norm_dict = None
    DBE_ground_motion_set = None
    MCE_ground_motion_set = None
    
    if args.do_nonlinear_dynamic_analysis:
        if args.graph_lstm_dir and args.ground_motion_dir:
            nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(
                args.graph_lstm_dir, device
            )
            DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(
                args.ground_motion_dir, args.ground_motion_number, nda_norm_dict
            )
        else:
            logger.warning("NDA enabled but graph_lstm_dir or ground_motion_dir not provided")
            args.do_nonlinear_dynamic_analysis = False
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment kwargs matching original train.py
    env_kwargs = {
        "structure_shape": args.structure_shape,
        "add_structure_geometry": args.add_structure_geometry,
        "add_response_features": args.add_response_features,
        "reward_type": args.reward_type,
        "scwb_driven_design": args.scwb_driven_design,
        "do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
        "check_acceleration": args.check_acceleration,
        "check_displacement": args.check_displacement,
        "nda_simulator": nda_simulator,
        "nda_norm_dict": nda_norm_dict,
        "DBE_ground_motion_set": DBE_ground_motion_set,
        "MCE_ground_motion_set": MCE_ground_motion_set,
        "checkpoint_dir": checkpoint_dir,
        "logger": logger,
        "device": device,
    }
    
    # Create base environment
    base_env = Environment(**env_kwargs)
    
    # Wrap with option environment
    option_env = OptionEnvironmentWrapper(
        base_env=base_env,
        max_option_length=args.max_option_length
    )
    
    return option_env


def setup_agent(args, num_actions: int, device) -> OptionCriticAgent:
    Setup the Option-Critic agent.
    # Determine feature dimensions based on args (matching original train.py)
    node_feature_dim = 8 if args.add_structure_geometry else 5
    edge_feature_dim = 13 if args.add_response_features else 11
    
    # Create agent
    agent = OptionCriticAgent(
        num_options=args.num_options,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        update_freq=args.update_freq,
        temperature=args.temperature,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=100,  # Default from original
        member_state_dim=100,  # Default from original
        num_layers=3,  # Default from original
        device=device
    )
    
    # Initialize action space
    agent.initialize_action_space(num_actions)
    
    print(f"Agent configured with:")
    print(f"  Node features: {node_feature_dim}, Edge features: {edge_feature_dim}")
    print(f"  Hidden dim: 100, Member state dim: 100, Layers: 3")
    print(f"  Options: {args.num_options}, Max option length: {args.max_option_length}")
    
    return agent


def train_episode(env: OptionEnvironmentWrapper, agent: OptionCriticAgent) -> Dict:
    Train for one episode.
    obs = env.reset()
    agent.reset_episode()
    
    episode_reward = 0.0
    episode_steps = 0
    options_used = []
    option_lengths = []
    current_option_length = 0
    
    done = False
    
    while not done:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            # No valid actions available
            print("Warning: No valid actions available")
            break
        
        # Agent selects action and decides option termination
        action, option_terminated = agent.act(obs, valid_actions)
        
        # Track current option
        if agent.current_option is not None:
            if len(options_used) == 0 or options_used[-1] != agent.current_option:
                options_used.append(agent.current_option)
        
        # Execute action in environment
        next_obs, reward, done, info = env.step(action, option_terminated)
        
        # Let agent observe the transition
        agent.observe(obs, action, reward, next_obs, done, option_terminated, info)
        
        # Update episode statistics
        episode_reward += reward
        episode_steps += 1
        current_option_length += 1
        
        # Track option completion
        if option_terminated or done:
            if current_option_length > 0:
                option_lengths.append(current_option_length)
                current_option_length = 0
        
        # Update observation
        obs = next_obs
        
        # Safety check - prevent infinite episodes
        if episode_steps > 1000:
            print(f"Warning: Episode exceeded 1000 steps, terminating")
            break
    
    # Episode statistics
    episode_info = {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'num_options_used': len(set(options_used)),
        'unique_options': list(set(options_used)),
        'avg_option_length': np.mean(option_lengths) if option_lengths else 0,
        'num_option_switches': len(option_lengths),
        'final_info': info if 'info' in locals() else {}
    }
    
    # Update agent statistics
    agent.episode_rewards.append(episode_reward)
    if option_lengths:
        agent.option_lengths.extend(option_lengths)
    
    return episode_info


def save_model_and_logs(agent: OptionCriticAgent, args, episode: int, episode_stats: List[Dict]):
    Save model and training logs.
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"option_critic_{args.num_options}opt_{timestamp}"
    else:
        exp_name = args.experiment_name
    
    # Save model
    model_path = os.path.join(args.save_dir, f"{exp_name}_episode_{episode}.pt")
    agent.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training configuration
    config_path = os.path.join(args.save_dir, f"{exp_name}_config.json")
    config = vars(args)
    config.update(agent.get_statistics())
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save episode statistics
    stats_path = os.path.join(args.save_dir, f"{exp_name}_episode_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(episode_stats, f, indent=2)
    
    # Plot training curves
    plot_training_curves(agent, episode_stats, args.save_dir, exp_name)


def plot_training_curves(agent: OptionCriticAgent, episode_stats: List[Dict], save_dir: str, exp_name: str):
    Plot and save training curves.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Option-Critic Training Progress - {exp_name}')
    
    episodes = range(len(episode_stats))
    
    # Episode rewards
    rewards = [stat['episode_reward'] for stat in episode_stats]
    axes[0, 0].plot(episodes, rewards, alpha=0.6, label='Episode Reward')
    if len(rewards) > 50:
        # Moving average
        window = min(50, len(rewards) // 4)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode steps
    steps = [stat['episode_steps'] for stat in episode_stats]
    axes[0, 1].plot(episodes, steps, alpha=0.6, label='Episode Steps')
    if len(steps) > 50:
        window = min(50, len(steps) // 4)
        moving_avg = np.convolve(steps, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Option lengths
    option_lengths = [stat['avg_option_length'] for stat in episode_stats if stat['avg_option_length'] > 0]
    if option_lengths:
        axes[1, 0].plot(range(len(option_lengths)), option_lengths, alpha=0.6, label='Avg Option Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Option Length')
        axes[1, 0].set_title('Option Lengths')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Loss history
    if agent.loss_history:
        axes[1, 1].plot(agent.loss_history, alpha=0.6, label='Training Loss')
        if len(agent.loss_history) > 50:
            window = min(50, len(agent.loss_history) // 4)
            moving_avg = np.convolve(agent.loss_history, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(agent.loss_history)), moving_avg, 'r-', linewidth=2, 
                          label=f'Moving Avg ({window})')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f"{exp_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")


def get_logger(checkpoint_dir):
    # Setup logging.
    import logging
    from pathlib import Path
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name='Option-Critic-Graph-RL')
    logger.setLevel(level=logging.INFO)
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler
    file_handler = logging.FileHandler(Path(checkpoint_dir) / "record.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def main():
    Main training function.
    args = parse_arguments()
    
    print("=" * 60)
    print("Option-Critic Training for Graph-based Structural Design")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f"Using device: {device_name}")
    
    # Setup logger
    logger = get_logger(args.checkpoint_dir)
    logger.critical(f"Starting Option-Critic training")
    logger.critical(f"Device: {device_name}")
    logger.critical(args)
    
    # Setup environment
    print("Setting up environment...")
    env = setup_environment(args, logger, device)
    
    # Get action space size from environment structure
    test_structure = env.reset()
    num_actions = len(test_structure.story_level_actions)
    print(f"Action space size: {num_actions}")
    
    # Setup agent
    print("Setting up agent...")
    agent = setup_agent(args, num_actions, device)
    
    # Training loop
    print(f"Starting training for {args.num_episodes} episodes...")
    episode_stats = []
    
    for episode in range(args.num_episodes):
        try:
            # Train one episode
            episode_info = train_episode(env, agent)
            episode_stats.append(episode_info)
            
            # Logging
            if episode % args.log_freq == 0 or episode == args.num_episodes - 1:
                agent_stats = agent.get_statistics()
                print(f"\nEpisode {episode + 1}/{args.num_episodes}")
                print(f"  Episode Reward: {episode_info['episode_reward']:.2f}")
                print(f"  Episode Steps: {episode_info['episode_steps']}")
                print(f"  Options Used: {episode_info['num_options_used']}")
                print(f"  Avg Option Length: {episode_info['avg_option_length']:.2f}")
                print(f"  Agent Epsilon: {agent_stats['epsilon']:.4f}")
                print(f"  Buffer Size: {agent_stats['buffer_size']}")
                print(f"  Avg Recent Reward: {agent_stats['avg_episode_reward']:.2f}")
                
                if episode_info['final_info']:
                    print(f"  Final Info: {episode_info['final_info']}")
            
            # Save model
            if (episode + 1) % args.save_freq == 0 or episode == args.num_episodes - 1:
                save_model_and_logs(agent, args, episode + 1, episode_stats)
                
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nTraining completed!")
    print(f"Final model saved to: {args.save_dir}")
    
    # Final statistics
    agent_stats = agent.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Steps: {agent_stats['num_steps']}")
    print(f"  Total Updates: {agent_stats['num_updates']}")
    print(f"  Final Epsilon: {agent_stats['epsilon']:.4f}")
    print(f"  Average Episode Reward (last 100): {agent_stats['avg_episode_reward']:.2f}")
    print(f"  Average Option Length (last 100): {agent_stats['avg_option_length']:.2f}")


"""