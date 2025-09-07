import argparse
import datetime
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.optim as optim

from RL.environment import Environment
from RL.option_critic_gnn import OptionCriticGNN, critic_loss, actor_loss
from RL.experience_replay import ReplayBuffer
from RL.record import Record
from Validation import normalization as nda_norm
from Structure import check, check_nda
from Visualization.plot import plot_test_behaviors


class TerminationProbabilityLogger:
    """
    Logger for Option-Critic termination probabilities.
    Collects and saves termination statistics to JSON files.
    """
    
    def __init__(self, save_path="oc_stats.json"):
        self.save_path = save_path
        self.stats = {
            "termination_probabilities": [],
            "episode_stats": [],
            "global_stats": {
                "total_terminations": 0,
                "total_predictions": 0,
                "avg_termination_prob": 0.0
            }
        }
        self.episode_terminations = []
        self.current_episode = 0
    
    def log_termination_prediction(self, option, termination_probs, termination_decision, episode=None, step=None, context=""):
        """
        Log a termination probability prediction.
        
        Args:
            option: current option index
            termination_probs: tensor of termination probabilities for all options
            termination_decision: boolean decision for current option
            episode: episode number (optional)
            step: step number (optional)
            context: additional context string
        """
        # Convert tensor to list if needed
        if isinstance(termination_probs, torch.Tensor):
            if termination_probs.dim() > 1:
                # Handle multi-dimensional tensor (e.g., [batch_size, num_options])
                # Take the mean across batch dimension or first sample
                if termination_probs.shape[0] > 1:
                    termination_probs = termination_probs.mean(dim=0)  # Average across batch
                else:
                    termination_probs = termination_probs[0]  # Take first sample
            termination_probs = termination_probs.detach().cpu().numpy().tolist()
        
        entry = {
            "episode": episode if episode is not None else self.current_episode,
            "step": step,
            "option": int(option),
            "termination_probs": termination_probs,
            "termination_decision": bool(termination_decision),
            "context": context,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.stats["termination_probabilities"].append(entry)
        
        # Track for episode stats
        if termination_decision:
            try:
                # termination_probs should now be a list after conversion above
                if isinstance(termination_probs, (list, np.ndarray)) and len(termination_probs) > option:
                    term_prob = float(termination_probs[option])
                else:
                    term_prob = 0.0
                    print(f"DEBUG: Cannot extract termination prob for option {option} from {termination_probs}")
            except (ValueError, TypeError, IndexError) as e:
                print(f"WARNING: Error extracting episode termination probability: {e}, using 0.0")
                term_prob = 0.0
            
            self.episode_terminations.append({
                "option": int(option),
                "termination_prob": term_prob,
                "step": step
            })
        
        # Update global stats
        self.stats["global_stats"]["total_predictions"] += 1
        if termination_decision:
            self.stats["global_stats"]["total_terminations"] += 1
        
        # Update average termination probability
        try:
            # termination_probs should now be a list after conversion above
            if isinstance(termination_probs, (list, np.ndarray)) and len(termination_probs) > option:
                current_prob = float(termination_probs[option])
            else:
                current_prob = 0.0
                print(f"DEBUG: Cannot extract termination prob for option {option} from {termination_probs}")
        except (ValueError, TypeError, IndexError) as e:
            print(f"WARNING: Error extracting termination probability: {e}, using 0.0")
            current_prob = 0.0
        
        total_preds = self.stats["global_stats"]["total_predictions"]
        prev_avg = self.stats["global_stats"]["avg_termination_prob"]
        self.stats["global_stats"]["avg_termination_prob"] = (prev_avg * (total_preds - 1) + current_prob) / total_preds
    
    def end_episode(self):
        """Mark the end of an episode and save episode statistics."""
        episode_stats = {
            "episode": self.current_episode,
            "num_terminations": len(self.episode_terminations),
            "terminations": self.episode_terminations.copy(),
            "avg_termination_prob": np.mean([t["termination_prob"] for t in self.episode_terminations]) if self.episode_terminations else 0.0
        }
        
        self.stats["episode_stats"].append(episode_stats)
        self.episode_terminations.clear()
        self.current_episode += 1
    
    def save_stats(self):
        """Save statistics to JSON file."""
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            print(f"Termination probability stats saved to {self.save_path}")
        except Exception as e:
            print(f"Error saving termination probability stats: {e}")
    
    def print_summary(self):
        """Print a summary of collected statistics."""
        global_stats = self.stats["global_stats"]
        print(f"\n=== Termination Probability Summary ===")
        print(f"Total predictions: {global_stats['total_predictions']}")
        print(f"Total terminations: {global_stats['total_terminations']}")
        print(f"Termination rate: {global_stats['total_terminations'] / max(1, global_stats['total_predictions']) * 100:.2f}%")
        print(f"Average termination probability: {global_stats['avg_termination_prob']:.4f}")
        print(f"Episodes recorded: {len(self.stats['episode_stats'])}")


# Global logger instance
termination_logger = TerminationProbabilityLogger()


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


def rollout_option(structure, base_env, device, max_option_len, current_option: int, oc_model, epsilon: float = None, logger=None, option_length_bonus: float = 0.0):
    """
    Execute a single option composed of a sequence of primitive actions.
    Now returns step-level transitions for step-based critic updates.

    Args:
        structure: current structure state
        base_env: base Environment instance 
        device: torch device
        max_option_len: maximum option length
        current_option: selected option index
        oc_model: OptionCriticGNN model with get_state, get_action, predict_option_termination
        epsilon: optional exploration indicator (for logging)

    Returns:
        structure: updated structure after option execution
        next_state: final state after option terminates
        option_done: whether option terminated
        episode_done: whether episode terminated as a result of this option
        stats: dict with diagnostics (length, termination_reason, entropies, epsilon)
        step_transitions: list of step-level transitions for buffer storage
        termination_reason: reason for option termination
    """
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

    step_transitions = []  # list of dicts: {obs, action, logp, entropy, reward, done, next_obs, option}

    if logger:
        logger.debug(f"Starting rollout_option for option {current_option}, max_len={max_option_len}")
    else:
        print(f"DEBUG: Starting rollout_option for option {current_option}, max_len={max_option_len}")
        
    safety_counter = 0
    max_safety_iterations = max_option_len + 10  # Extra safety margin
    while length < max_option_len:
        safety_counter += 1
        if logger:
            logger.debug(f"rollout_option loop iteration {length+1}/{max_option_len}, safety_counter={safety_counter}")
        else:
            print(f"DEBUG: rollout_option loop iteration {length+1}/{max_option_len}, safety_counter={safety_counter}")
        
        # Safety check to prevent infinite loops
        if safety_counter > max_safety_iterations:
            if logger:
                logger.error(f"rollout_option exceeded safety counter ({max_safety_iterations}), forcing termination")
            else:
                print(f"ERROR: rollout_option exceeded safety counter ({max_safety_iterations}), forcing termination")
            termination_reason = "safety_timeout"
            break
        
        # Create valid actions mask
        try:
            # Get all restricted actions
            already_minimum = set(getattr(structure, 'already_minimum_section_story_indexes', []) or [])
            restricted_actions = set()
            if hasattr(structure, 'restrict_action_space'):
                restricted = structure.restrict_action_space()
                if restricted is not None:
                    restricted_actions.update(restricted)
            
            # Combine all invalid actions
            invalid_actions = already_minimum | restricted_actions
            log_msg = f"Already minimum: {already_minimum}, Restricted: {restricted_actions}"
            if logger:
                logger.debug(log_msg)
            else:
                print(f"DEBUG: {log_msg}")
                
            log_msg = f"Total invalid actions: {invalid_actions}"
            if logger:
                logger.debug(log_msg)
            else:
                print(f"DEBUG: {log_msg}")
            
            # Create mask tensor (True = valid action)
            num_actions = len(structure.story_level_actions)
            valid_mask = torch.ones(num_actions, dtype=torch.bool, device=device)
            for invalid_action in invalid_actions:
                if 0 <= invalid_action < num_actions:
                    valid_mask[invalid_action] = False
            
            log_msg = f"Valid actions mask: {valid_mask.sum().item()}/{num_actions} actions available"
            if logger:
                logger.debug(log_msg)
            else:
                print(f"DEBUG: {log_msg}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to create valid actions mask: {e}")
            else:
                print(f"ERROR: Failed to create valid actions mask: {e}")
            valid_mask = None

        # intra-option action with valid actions mask
        try:
            log_msg = f"Getting action for option {current_option}"
            if logger:
                logger.debug(log_msg)
            else:
                print(f"DEBUG: {log_msg}")
                
            action, logp, entropy = oc_model.get_action(state, current_option, valid_mask)
            entropies.append(float(entropy.detach().cpu().numpy()))
            
            log_msg = f"Got action {action}, entropy: {entropy.item()}"
            if logger:
                logger.debug(log_msg)
            else:
                print(f"DEBUG: {log_msg}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to get action: {e}")
            else:
                print(f"ERROR: Failed to get action: {e}")
            termination_reason = "action_error"
            break

        log_msg = f"Applying primitive action {action}"
        if logger:
            logger.debug(log_msg)
        else:
            print(f"DEBUG: {log_msg}")
            
        try:
            structure, step_reward, step_pass, is_min_section, fail_reason = apply_primitive_action(base_env, structure, action)
            
            # Apply option length bonus: reward += (step_number - 1) * bonus
            # length is 0-indexed, so length equals (step_number - 1)
            length_bonus = length * option_length_bonus
            original_reward = step_reward  # Store original reward before adding bonus
            step_reward += length_bonus
            
            log_msg = f"Action applied - original_reward: {original_reward}, length_bonus: {length_bonus}, final_reward: {step_reward}, pass: {step_pass}, min_section: {is_min_section}, fail_reason: {fail_reason}"
            if logger:
                logger.debug(log_msg)
            else:
                print(f"DEBUG: {log_msg}")
        except Exception as e:
            print(f"ERROR: Failed to apply primitive action: {e}")
            termination_reason = "action_apply_error"
            step_reward = -1 # -1000
            step_pass = False
            is_min_section = False
            fail_reason = "action_apply_error"
        # Don't accumulate rewards at option level anymore
        length += 1
        print(f"DEBUG: Updated length to {length}, step_reward: {step_reward}")

        # minimum section: force terminate option and episode
        if is_min_section:
            print(f"DEBUG: Minimum section reached, terminating option and episode")
            termination_reason = "minimum_section"
            episode_done = True
            try:
                next_graph_data = get_graph_data(structure, device)
                step_transitions.append({
                    "obs": graph_data,
                    "action": action,
                    "logp": logp.detach().clone(),
                    "entropy": entropy.detach().clone(),
                    "reward": float(step_reward),
                    "original_reward": float(original_reward),
                    "done": True,
                    "next_obs": next_graph_data,
                    "option": current_option,  # Add option to each step transition
                })
                print(f"DEBUG: Added final step transition, breaking from loop")
            except Exception as e:
                print(f"ERROR: Failed to create step transition: {e}")
            break

        # compute next state for termination prediction
        print(f"DEBUG: Computing next state for termination prediction")
        try:
            next_graph_data = get_graph_data(structure, device)
            next_state = oc_model.get_state(*next_graph_data)
            print(f"DEBUG: Successfully computed next state")
        except Exception as e:
            print(f"ERROR: Failed to compute next state: {e}")
            termination_reason = "state_error"
            break

        # record step transition
        print(f"DEBUG: Recording step transition")
        try:
            step_transitions.append({
                "obs": graph_data,
                "action": action,
                "logp": logp.detach().clone(),
                "entropy": entropy.detach().clone(),
                "reward": float(step_reward),
                "original_reward": float(original_reward),
                "done": False,
                "next_obs": next_graph_data,
                "option": current_option,  # Add option to each step transition
            })
            print(f"DEBUG: Step transition recorded, total transitions: {len(step_transitions)}")
        except Exception as e:
            print(f"ERROR: Failed to record step transition: {e}")

        # option termination by beta
        print(f"DEBUG: Checking option termination by beta")
        try:
            # Get termination probabilities for logging
            termination_probs = oc_model.get_terminations(next_state)
            option_termination, _ = oc_model.predict_option_termination(next_state, current_option)
            
            # Print termination probabilities for monitoring
            if hasattr(termination_probs, 'shape'):
                if termination_probs.dim() > 1:
                    term_probs_display = termination_probs.mean(dim=0).detach().cpu().numpy()
                else:
                    term_probs_display = termination_probs.detach().cpu().numpy()
                
                # Print each option's termination probability clearly
                term_probs_str = ", ".join([f"β{i}: {prob:.4f}" for i, prob in enumerate(term_probs_display)])
                print(f"TERMINATION_PROBS Step {length+1}: [{term_probs_str}] | Current Option {current_option}: β{current_option}={term_probs_display[current_option]:.4f} → {'TERMINATE' if option_termination else 'CONTINUE'}")
            
            # Log termination probability prediction
            try:
                termination_logger.log_termination_prediction(
                    option=current_option,
                    termination_probs=termination_probs,
                    termination_decision=option_termination,
                    step=length,
                    context="rollout_option"
                )
            except Exception as log_e:
                print(f"WARNING: Failed to log termination probability: {log_e}")
            
            if option_termination:
                print(f"==> Option {current_option} TERMINATED by β={term_probs_display[current_option]:.4f} at step {length}")
                termination_reason = "beta"
                state = next_state
                graph_data = next_graph_data
                break
        except Exception as e:
            print(f"ERROR: Failed option termination prediction: {e}")
            termination_reason = "beta_error"
            break

        # continue the option
        print(f"DEBUG: Continuing option, updating state")
        state = next_state
        graph_data = next_graph_data
        print(f"DEBUG: State updated, continuing to next iteration")

    # if not terminated by beta/minimum_section, it hits max length
    print(f"DEBUG: Exited rollout loop with length={length}, max_len={max_option_len}, termination_reason={termination_reason}")
    option_done = True
    if termination_reason is None:
        termination_reason = "max_len"
        print(f"DEBUG: Set termination reason to max_len")

    # Check constraint compliance and apply penalty to last step if failed
    # Note: step_pass refers to the last evaluated step
    passed = step_pass if length > 0 else True
    print(f"DEBUG: Final step_pass check: step_pass={step_pass}, length={length}, passed={passed}")
    
    if not passed:
        print(f"DEBUG: Option failed, applying penalty to last step")
        episode_done = True
        # Apply penalty reward to the last step that caused the failure
        if len(step_transitions) > 0:
            step_transitions[-1]["reward"] = -1 # -1000.0
            step_transitions[-1]["original_reward"] = -1 # -1000.0
            step_transitions[-1]["done"] = True
            print(f"DEBUG: Updated last step transition with penalty reward")
    
    print(f"DEBUG: Option execution completed, passed={passed}")

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
    print(f"DEBUG: rollout_option returning - option_done: {option_done}, episode_done: {episode_done}, termination_reason: {termination_reason}, num_transitions: {len(step_transitions)}")
    return structure, next_state, option_done, episode_done, stats, step_transitions, termination_reason



def evaluate_model(base_env, oc_model, device, num_episodes, max_option_len, logger=None, seed=42):
    """
    Evaluate the model performance over multiple test episodes and collect action/option histories.
    
    Args:
        base_env: Base environment
        oc_model: Option-Critic model
        device: torch device
        num_episodes: Number of evaluation episodes
        max_option_len: Maximum option length
        logger: Logger instance
        seed: Random seed for reproducibility
    
    Returns:
        avg_score: average testing score
        avg_episodes_length: average number of options per episode
        success_rate: percentage of episodes that reached minimum section
        eval_history: dict containing action and option histories for visualization
    """
    logger.info(f"Starting evaluation for {num_episodes} episodes with seed {seed}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    episode_scores = []
    episode_lengths = []
    episode_actions = []  # Collect action sequences
    episode_options = []  # Collect option sequences
    episode_option_instances = []  # Collect option instance data with termination info
    episode_actions_SCWB = []  # For compatibility
    successful_episodes = 0
    
    # Store original testing mode and set to testing mode
    original_testing = oc_model.testing
    oc_model.testing = True
    
    try:
        for ep in range(num_episodes):
            logger.info(f"Evaluation episode {ep+1}/{num_episodes}")
            
            try:
                structure = base_env.reset(testing=True)
                done = False
                option_termination = True
                curr_option = 0
                greedy_option = 0
                episode_score = 0.0
                episode_option_count = 0
                
                # Track action and option sequences for this episode
                episode_action_sequence = []
                episode_option_sequence = []
                current_episode_option_instances = []  # Track option instances with termination info for this episode
                current_option_instance_id = 0  # Unique ID for each option instance
                
                while not done :
                    if option_termination:
                        # Always use greedy option selection during evaluation
                        curr_option = greedy_option
                    
                    try:
                        structure, next_state, option_done, episode_done, o_stats, step_transitions, termination_reason = rollout_option(
                            structure, base_env, device, max_option_len, curr_option, oc_model, None, logger, 0.0
                        )
                        
                        # Collect actions and options from step transitions
                        for transition in step_transitions:
                            episode_action_sequence.append(transition["action"])
                            episode_option_sequence.append(curr_option)
                            # Record option instance info for this step
                            current_episode_option_instances.append({
                                "option_index": curr_option,
                                "instance_id": current_option_instance_id,
                                "action_step": len(episode_action_sequence) - 1
                            })
                        
                        # Accumulate score only from successful options (using original reward without length bonus)
                        if bool(o_stats.get("passed", False)):
                            option_total_reward = sum(tr["original_reward"] for tr in step_transitions)
                            episode_score += float(option_total_reward)
                        
                        episode_option_count += 1
                        
                        # Check if episode completed successfully
                        if termination_reason == "minimum_section":
                            successful_episodes += 1
                            logger.info(f"Episode {ep+1} reached minimum section successfully")
                        
                        done = episode_done
                        
                        # If the option terminated (for any reason), increment instance ID for next option
                        if option_done:
                            current_option_instance_id += 1
                        
                        # Update option termination for next iteration
                        if not done:
                            try:
                                current_graph_data = get_graph_data(structure, device)
                                state = oc_model.get_state(*current_graph_data)
                                
                                # Get termination probabilities for logging
                                termination_probs = oc_model.get_terminations(state)
                                option_termination, greedy_option = oc_model.predict_option_termination(state, curr_option)
                                
                                # Log termination probability prediction
                                try:
                                    termination_logger.log_termination_prediction(
                                        option=curr_option,
                                        termination_probs=termination_probs,
                                        termination_decision=option_termination,
                                        episode=ep,
                                        step=episode_option_count,
                                        context="evaluation_loop"
                                    )
                                except Exception as log_e:
                                    print(f"WARNING: Failed to log termination probability in evaluation: {log_e}")
                                    
                            except Exception as e:
                                logger.error(f"Error in option termination prediction during eval: {e}")
                                option_termination = True
                                greedy_option = 0
                        
                    except Exception as e:
                        logger.error(f"Error in rollout_option during evaluation: {e}")
                        break
                
                episode_scores.append(episode_score)
                episode_lengths.append(episode_option_count)
                episode_actions.append(episode_action_sequence)
                episode_options.append(episode_option_sequence)
                episode_option_instances.append(current_episode_option_instances)  # Store option instance data for this episode
                episode_actions_SCWB.append([])  # Empty for compatibility
                logger.info(f"Episode {ep+1} completed: score={episode_score:.2f}, length={episode_option_count}, actions={len(episode_action_sequence)}")
                
            except Exception as e:
                logger.error(f"Error in evaluation episode {ep+1}: {e}")
                continue
    
    finally:
        # Restore original testing mode
        oc_model.testing = original_testing
    
    # Calculate metrics
    avg_score = float(np.mean(episode_scores)) if episode_scores else 0.0
    avg_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    success_rate = (successful_episodes / num_episodes) * 100.0 if num_episodes > 0 else 0.0
    
    logger.info(f"Evaluation completed:")
    logger.info(f"  Average score: {avg_score:.2f}")
    logger.info(f"  Average episode length: {avg_episode_length:.2f}")
    logger.info(f"  Success rate: {success_rate:.1f}% ({successful_episodes}/{num_episodes})")
    
    # Prepare evaluation history for visualization
    eval_history = {
        "scores": episode_scores,
        "actions": episode_actions,
        "options": episode_options,
        "option_instances": episode_option_instances,  # Include option instance data
        "actions_SCWB": episode_actions_SCWB,
        "avg_score": avg_score,
        "success_rate": success_rate
    }
    
    return avg_score, avg_episode_length, success_rate, eval_history


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
    parser.add_argument("--num_options", type=int, default=8)
    parser.add_argument("--max_option_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=int, default=int(1e3))
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_frequency", type=int, default=4)
    parser.add_argument("--freeze_interval", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64) # 128
    parser.add_argument("--num_layers", type=int, default=3)  # 3 
    parser.add_argument("--termination_reg", type=float, default=0.01)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--option_length_bonus", type=float, default=0.1, help="Bonus reward for longer options: reward += (step-1) * bonus")
    parser.add_argument("--eval_frequency", type=int, default=5, help="Evaluate model every N training episodes")
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes for evaluation")
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    
    # Create timestamped checkpoint directory to avoid overwriting previous results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints" / "option_oc" / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created timestamped checkpoint directory: {ckpt_dir}")
    
    logger = logging.getLogger("OC_Train")
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler - save to timestamped checkpoint directory
        log_file = ckpt_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Set logger level
        logger.setLevel(logging.DEBUG)
        logger.info(f"Debug logging enabled - logs saved to {log_file}")
        logger.info(f"Timestamped checkpoint directory: {ckpt_dir}")
        logger.info("Use INFO level to reduce verbosity")

    # Build base env (NDA/SCWB disabled per spec)
    nda_simulator = None
    nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}
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
    termination_params = [p for n, p in oc.named_parameters() 
                         if n.startswith('terminations')]
    actor_params = [oc.options_W, oc.options_b] + termination_params
    critic_params = [p for n, p in oc.named_parameters() 
                    if n.startswith('Q')]
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
        "eval_scores": [],
        "eval_success_rates": [],
        "eval_episode_lengths": [],
        "actor_losses": [],
        "critic_losses": [],
    }

    # Best model tracking
    best_model_score = float('-inf')
    best_model_path = ckpt_dir / "best_model.pt"
    best_model_info_path = ckpt_dir / "best_model_info.json"

    # Evaluation history tracking for visualization
    evaluation_histories = {
        "all_scores": [],      # Flattened list of all scores from all evaluation episodes
        "all_actions": [],     # Flattened list of all action sequences
        "all_options": [],     # Flattened list of all option sequences
        "all_option_instances": [], # Flattened list of all option instances with termination info
        "all_actions_SCWB": [] # For compatibility
    }

    for ep in range(args.epochs):
        logger.info(f"Starting episode {ep+1}/{args.epochs}")
        try:
            structure = base_env.reset(testing=False)
            logger.debug(f"Environment reset successful for episode {ep+1}")
            
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
            
        except Exception as e:
            logger.error(f"Failed to initialize episode {ep+1}: {e}")
            logger.error(f"Exception details:", exc_info=True)
            continue

        logger.info(f"Starting episode {ep+1} main loop")
        loop_iteration = 0
        max_loop_iterations = 1000  # Safety check to prevent infinite loops
        while not done:
            loop_iteration += 1
            logger.debug(f"Episode {ep+1}, Loop iteration {loop_iteration}, done={done}, option_termination={option_termination}")
            
            # Safety check for infinite loops
            if loop_iteration > max_loop_iterations:
                logger.error(f"Episode {ep+1} exceeded maximum loop iterations ({max_loop_iterations}), forcing termination")
                done = True
                break
            
            epsilon = oc.epsilon
            logger.debug(f"Current epsilon: {epsilon}")
            
            if option_termination:
                logger.debug(f"Option terminated, selecting new option")
                curr_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                logger.debug(f"Selected option: {curr_option} (greedy_option: {greedy_option})")

            logger.debug(f"Calling rollout_option with curr_option={curr_option}")
            try:
                structure, next_state, option_done, episode_done, o_stats, step_transitions, termination_reason = rollout_option(structure, base_env, device, args.max_option_len, curr_option, oc, epsilon, logger, args.option_length_bonus)
                logger.debug(f"rollout_option returned: option_done={option_done}, episode_done={episode_done}, termination_reason={termination_reason}")
            except Exception as e:
                logger.error(f"Error in rollout_option: {e}")
                logger.error(f"Exception details:", exc_info=True)
                break

            # logging stats
            all_stats["option_lengths"].append([o_stats["option_length"]])
            all_stats["termination_reasons"][o_stats["termination_reason"]] = all_stats["termination_reasons"].get(o_stats["termination_reason"], 0) + 1
            all_stats["entropy_mean"].append(o_stats["entropy_mean"])
            all_stats["entropy_last"].append(o_stats["entropy_last"])
            all_stats["epsilon"].append(o_stats["epsilon"] if o_stats["epsilon"] is not None else float(epsilon))

            # per-episode accumulators
            episode_opt_lengths.append(o_stats["option_length"])
            episode_termination_counter[o_stats["termination_reason"]] = episode_termination_counter.get(o_stats["termination_reason"], 0) + 1
            if not np.isnan(o_stats["entropy_mean"]):
                episode_entropies.append(o_stats["entropy_mean"])

            # Calculate episode score from step rewards (only if option passed, using original reward without length bonus)
            if bool(o_stats.get("passed", False)):
                option_total_reward = sum(tr["original_reward"] for tr in step_transitions)
                episode_score += float(option_total_reward)
                last_pass_sections = o_stats.get("story_level_sections", last_pass_sections)
                last_pass_saved_material = o_stats.get("option_saved_material", last_pass_saved_material)

            # Push step-level transitions to buffer (like option-critic-pytorch)
            logger.debug(f"Pushing {len(step_transitions)} step transitions to buffer, buffer size before: {len(buffer)}")
            try:
                for tr in step_transitions:
                    buffer.push(tr["obs"], tr["option"], tr["reward"], tr["next_obs"], tr["done"])
                logger.debug(f"Buffer size after push: {len(buffer)}")
            except Exception as e:
                logger.error(f"Error pushing step transitions to buffer: {e}")

            # Per-step actor updates: for each intra-option step do one actor update
            logger.debug(f"Number of step transitions: {len(step_transitions)}")
            if len(step_transitions) > 0:
                try:
                    for i, tr in enumerate(step_transitions):
                        logger.debug(f"Processing step transition {i+1}/{len(step_transitions)}")
                        a_loss = actor_loss(
                            tr["obs"], tr["option"], tr["logp"], tr["entropy"], tr["reward"], tr["done"], tr["next_obs"], 
                            oc, oc_prime, args.gamma, args.termination_reg, args.entropy_reg
                        )
                        actor_optimizer.zero_grad()
                        a_loss.backward()
                        if args.grad_clip is not None and args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=args.grad_clip)
                        actor_optimizer.step()
                        
                        # Record actor loss for plotting
                        all_stats["actor_losses"].append(float(a_loss.item()))
                        logger.debug(f"Actor update {i+1} completed, loss: {a_loss.item()}")
                except Exception as e:
                    logger.error(f"Error in actor updates: {e}")

            # Critic updates on schedule (step-level replay like option-critic-pytorch)
            should_update_critic = len(buffer) > args.batch_size and (steps % args.update_frequency == 0)
            logger.debug(f"Critic update check: buffer_size={len(buffer)}, batch_size={args.batch_size}, steps={steps}, should_update={should_update_critic}")
            if should_update_critic:
                try:
                    logger.debug(f"Performing step-level critic update")
                    data_batch = buffer.sample(args.batch_size)  # Now sampling step-level transitions
                    logger.debug(f"Sampled batch with {args.batch_size} step-level transitions")
                    c_loss = critic_loss(oc, oc_prime, data_batch, args.gamma)
                    critic_optimizer.zero_grad()
                    c_loss.backward()
                    if args.grad_clip is not None and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=args.grad_clip)
                    critic_optimizer.step()
                    
                    # Record critic loss for plotting
                    all_stats["critic_losses"].append(float(c_loss.item()))
                    logger.debug(f"Critic update completed, loss: {c_loss.item()}")

                    if steps % args.freeze_interval == 0:
                        logger.debug(f"Updating target network at step {steps}")
                        oc_prime.load_state_dict(oc.state_dict())
                except Exception as e:
                    logger.error(f"Error in critic update: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

            logger.debug(f"Getting next state and option termination prediction")
            try:
                current_graph_data = get_graph_data(structure, device)
                state = oc.get_state(*current_graph_data)
                
                # Get termination probabilities for logging
                termination_probs = oc.get_terminations(state)
                option_termination, greedy_option = oc.predict_option_termination(state, curr_option)
                
                # Print termination probabilities for monitoring in main loop
                if hasattr(termination_probs, 'shape'):
                    if termination_probs.dim() > 1:
                        term_probs_display = termination_probs.mean(dim=0).detach().cpu().numpy()
                    else:
                        term_probs_display = termination_probs.detach().cpu().numpy()
                    
                    # Print each option's termination probability clearly
                    term_probs_str = ", ".join([f"β{i}: {prob:.4f}" for i, prob in enumerate(term_probs_display)])
                    print(f"MAIN_LOOP Ep{ep+1}-Loop{loop_iteration}: [{term_probs_str}] | Current Option {curr_option}: β{curr_option}={term_probs_display[curr_option]:.4f} → {'TERMINATE' if option_termination else 'CONTINUE'} → Next Option: {greedy_option}")
                
                # Log termination probability prediction
                try:
                    termination_logger.log_termination_prediction(
                        option=curr_option,
                        termination_probs=termination_probs,
                        termination_decision=option_termination,
                        episode=ep,
                        step=loop_iteration,
                        context="main_training_loop"
                    )
                except Exception as log_e:
                    logger.warning(f"Failed to log termination probability in main loop: {log_e}")
                
                logger.debug(f"Option termination prediction: {option_termination}, greedy_option: {greedy_option}")
            except Exception as e:
                logger.error(f"Error in option termination prediction: {e}")
                option_termination = True  # Force termination on error
                greedy_option = 0
            
            done = episode_done
            steps += 1
            logger.debug(f"End of loop iteration {loop_iteration}, done={done}, steps={steps}")

        logger.info(f"Episode {ep+1} completed with {loop_iteration} iterations, {steps} steps")
        
        # Print episode termination probability summary
        try:
            if len(termination_logger.episode_terminations) > 0:
                avg_term_prob = np.mean([t["termination_prob"] for t in termination_logger.episode_terminations])
                num_terminations = len(termination_logger.episode_terminations)
                print(f"EPISODE_SUMMARY: Episode {ep+1}, Terminations: {num_terminations}, Avg β: {avg_term_prob:.4f}")
            else:
                print(f"EPISODE_SUMMARY: Episode {ep+1}, No terminations recorded")
        except Exception as e:
            logger.warning(f"Failed to compute episode termination summary: {e}")
        
        # Mark episode end for termination probability logging
        try:
            termination_logger.end_episode()
        except Exception as e:
            logger.warning(f"Failed to end episode in termination logger: {e}")
        
        # episode-level aggregates
        try:
            all_stats["episode_rewards"].append(float(np.sum([t[2] if isinstance(t, (list, tuple)) and len(t) > 2 else 0.0 for t in []])))  # placeholder, kept for compatibility
            # Calculate episode reward from step rewards
            episode_total_reward = sum(tr["reward"] for tr in step_transitions)
            all_stats["episode_rewards"][-1] = float(episode_total_reward) if len(all_stats["episode_rewards"]) > 0 else float(episode_total_reward)
            all_stats["episode_score"].append(float(episode_score))
            all_stats["last_pass_sections"].append(last_pass_sections)
            all_stats["last_pass_saved_material"].append(float(last_pass_saved_material) if last_pass_saved_material is not None else None)
            all_stats["episode_option_lengths_mean"].append(float(np.mean(episode_opt_lengths)) if len(episode_opt_lengths) > 0 else 0.0)
            all_stats["episode_termination_counts"].append(episode_termination_counter)
            all_stats["episode_entropy_mean"].append(float(np.mean(episode_entropies)) if len(episode_entropies) > 0 else float("nan"))
            logger.debug(f"Episode {ep+1} stats updated successfully")
        except Exception as e:
            logger.error(f"Failed to update episode {ep+1} stats: {e}")

        # Periodic evaluation
        if (ep + 1) % args.eval_frequency == 0:
            logger.info(f"Starting evaluation after episode {ep+1}")
            try:
                avg_score, avg_episode_length, success_rate, eval_history = evaluate_model(
                    base_env, oc, device, args.eval_episodes, args.max_option_len, logger, seed=42
                )
                
                # Store evaluation results
                all_stats["eval_scores"].append(float(avg_score))
                all_stats["eval_success_rates"].append(float(success_rate))
                all_stats["eval_episode_lengths"].append(float(avg_episode_length))
                
                # Collect evaluation histories for visualization
                evaluation_histories["all_scores"].extend(eval_history["scores"])
                evaluation_histories["all_actions"].extend(eval_history["actions"])
                evaluation_histories["all_options"].extend(eval_history["options"])
                evaluation_histories["all_option_instances"].extend(eval_history["option_instances"])
                evaluation_histories["all_actions_SCWB"].extend(eval_history["actions_SCWB"])
                
                logger.info(f"Episode {ep+1} evaluation: score={avg_score:.2f}, success_rate={success_rate:.1f}%")
                logger.info(f"Total evaluation episodes collected so far: {len(evaluation_histories['all_scores'])}")
                
                # Check if this is the best model so far
                if avg_score > best_model_score:
                    best_model_score = avg_score
                    logger.info(f"New best model found! Score: {avg_score:.2f} (previous best: {best_model_score:.2f})")
                    
                    # Save best model
                    try:
                        torch.save(oc.state_dict(), best_model_path)
                        
                        # Save best model info
                        best_model_info = {
                            "episode": ep + 1,
                            "score": float(avg_score),
                            "success_rate": float(success_rate),
                            "avg_episode_length": float(avg_episode_length),
                            "timestamp": str(datetime.datetime.now()),
                            "hyperparameters": vars(args)
                        }
                        
                        with open(best_model_info_path, "w", encoding="utf-8") as f:
                            json.dump(best_model_info, f, ensure_ascii=False, indent=2)
                            
                        logger.info(f"Best model saved to {best_model_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save best model: {e}")
                
            except Exception as e:
                logger.error(f"Evaluation failed at episode {ep+1}: {e}")

        # persist stats every episode
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2)
            logger.debug(f"Stats saved to {stats_path}")
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

        # Evaluation metrics
        eval_scores = all_stats.get("eval_scores", [])
        if len(eval_scores) > 0:
            eval_episodes = [i * args.eval_frequency for i in range(1, len(eval_scores) + 1)]
            
            plt.figure(figsize=(15, 5))
            
            # Evaluation scores
            plt.subplot(1, 3, 1)
            plt.plot(eval_episodes, eval_scores, 'o-', label='Evaluation Score', color='#d62728')
            plt.xlabel('Training Episode')
            plt.ylabel('Average Score')
            plt.title('Evaluation Score History')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Success rates
            eval_success_rates = all_stats.get("eval_success_rates", [])
            if len(eval_success_rates) > 0:
                plt.subplot(1, 3, 2)
                plt.plot(eval_episodes, eval_success_rates, 'o-', label='Success Rate', color='#ff7f0e')
                plt.xlabel('Training Episode')
                plt.ylabel('Success Rate (%)')
                plt.title('Evaluation Success Rate History')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            # Episode lengths
            eval_episode_lengths = all_stats.get("eval_episode_lengths", [])
            if len(eval_episode_lengths) > 0:
                plt.subplot(1, 3, 3)
                plt.plot(eval_episodes, eval_episode_lengths, 'o-', label='Avg Episode Length', color='#9467bd')
                plt.xlabel('Training Episode')
                plt.ylabel('Average Episode Length')
                plt.title('Evaluation Episode Length History')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(ckpt_dir / 'evaluation_metrics.png', dpi=200)
            plt.close()

        # Loss histories
        actor_losses = all_stats.get("actor_losses", [])
        critic_losses = all_stats.get("critic_losses", [])
        if len(actor_losses) > 0 or len(critic_losses) > 0:
            plt.figure(figsize=(15, 5))
            
            # Actor loss
            if len(actor_losses) > 0:
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(actor_losses)+1), actor_losses, alpha=0.6, label='Actor Loss', color='#e377c2')
                if len(actor_losses) >= 20:
                    w = min(50, max(10, len(actor_losses)//20))
                    mv = np.convolve(actor_losses, np.ones(w)/w, mode='valid')
                    plt.plot(range(w, len(actor_losses)+1), mv, label=f'Moving Avg ({w})', color='#8c564b', linewidth=2)
                plt.xlabel('Update Step')
                plt.ylabel('Loss')
                plt.title('Actor Loss History')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            # Critic loss
            if len(critic_losses) > 0:
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(critic_losses)+1), critic_losses, alpha=0.6, label='Critic Loss', color='#17becf')
                if len(critic_losses) >= 20:
                    w = min(50, max(10, len(critic_losses)//20))
                    mv = np.convolve(critic_losses, np.ones(w)/w, mode='valid')
                    plt.plot(range(w, len(critic_losses)+1), mv, label=f'Moving Avg ({w})', color='#bcbd22', linewidth=2)
                plt.xlabel('Update Step')
                plt.ylabel('Loss')
                plt.title('Critic Loss History')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(ckpt_dir / 'loss_history.png', dpi=200)
            plt.close()

    except Exception as e:
        logger.warning(f"Failed plotting episode histories: {e}")

    # Generate training behavior visualization using evaluation histories
    try:
        logger.info("Generating training behavior visualization using evaluation histories...")
        
        if len(evaluation_histories["all_scores"]) > 0:
            # Create a Record-like object for compatibility with plot_test_behaviors
            class EvaluationRecord:
                def __init__(self, training_scores, evaluation_histories):
                    self.training_record = {"score": training_scores}
                    self.testing_record = {
                        "score": evaluation_histories["all_scores"],
                        "action": evaluation_histories["all_actions"],
                        "action_SCWB": evaluation_histories["all_actions_SCWB"],
                        "option": evaluation_histories["all_options"],
                        "option_instances": evaluation_histories["all_option_instances"]
                    }
            
            # Create record with training scores from all_stats
            training_scores = all_stats.get("episode_score", [])
            eval_record = EvaluationRecord(training_scores, evaluation_histories)
            
            # Generate the behavior visualization
            logger.info("Generating evaluation behavior visualization...")
            plot_test_behaviors(eval_record, base_env, ckpt_dir)
            logger.info(f"Evaluation behavior plot saved to: {ckpt_dir / 'testing_behaviors.png'}")
            logger.info(f"Total evaluation episodes plotted: {len(evaluation_histories['all_scores'])}")
        else:
            logger.warning("No evaluation histories available for visualization. Make sure eval_frequency is set properly.")
        
    except Exception as e:
        logger.error(f"Error in evaluation behavior visualization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    # Final summary with best model information
    try:
        logger.info("Training completed!")
        
        # Save and summarize termination probability statistics
        try:
            termination_logger.save_stats()
            termination_logger.print_summary()
            logger.info(f"Termination probability statistics saved to: {termination_logger.save_path}")
        except Exception as e:
            logger.warning(f"Failed to save termination probability statistics: {e}")
        
        if best_model_path.exists():
            logger.info(f"Best model saved at: {best_model_path}")
            logger.info(f"Best model score: {best_model_score:.2f}")
            logger.info(f"Best model info saved at: {best_model_info_path}")
            logger.info(f"To run inference with the best model, use: python inference_best_model.py --checkpoint_dir {ckpt_dir}")
        else:
            logger.info("No best model was saved (no evaluations performed)")
            
        if len(evaluation_histories["all_scores"]) > 0:
            logger.info(f"Evaluation behavior visualization saved at: {ckpt_dir / 'testing_behaviors.png'}")
            logger.info(f"Visualization includes {len(evaluation_histories['all_scores'])} evaluation episodes from training")
        
        # Loss history summary
        actor_losses = all_stats.get("actor_losses", [])
        critic_losses = all_stats.get("critic_losses", [])
        if len(actor_losses) > 0 or len(critic_losses) > 0:
            logger.info(f"Loss history visualization saved at: {ckpt_dir / 'loss_history.png'}")
            if len(actor_losses) > 0:
                logger.info(f"Total actor updates: {len(actor_losses)}, final actor loss: {actor_losses[-1]:.4f}")
            if len(critic_losses) > 0:
                logger.info(f"Total critic updates: {len(critic_losses)}, final critic loss: {critic_losses[-1]:.4f}")
    except Exception as e:
        logger.error(f"Error in final summary: {e}")


def load_best_model(checkpoint_dir, model_args=None):
    """
    Load the best performing model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        model_args: Optional arguments for model initialization
        
    Returns:
        tuple: (model, best_model_info) or (None, None) if not found
    """
    ckpt_dir = Path(checkpoint_dir)
    best_model_path = ckpt_dir / "best_model.pt"
    best_model_info_path = ckpt_dir / "best_model_info.json"
    
    if not best_model_path.exists() or not best_model_info_path.exists():
        print(f"Best model not found in {checkpoint_dir}")
        return None, None
    
    try:
        # Load model info
        with open(best_model_info_path, "r", encoding="utf-8") as f:
            best_model_info = json.load(f)
        
        print(f"Loading best model from episode {best_model_info['episode']}")
        print(f"  Score: {best_model_info['score']:.2f}")
        print(f"  Success Rate: {best_model_info['success_rate']:.1f}%")
        print(f"  Saved at: {best_model_info['timestamp']}")
        
        # Use provided args or get from saved hyperparameters
        if model_args is None:
            model_args = argparse.Namespace(**best_model_info['hyperparameters'])
        
        # Create model with same architecture
        device = torch.device(model_args.device if hasattr(model_args, 'device') else 'cuda')
        
        # You would need to reconstruct the model here with proper parameters
        # This is a template - you'd need to adapt based on your model creation code
        # For now, returning info only as model reconstruction needs environment setup
        
        return best_model_path, best_model_info
        
    except Exception as e:
        print(f"Error loading best model: {e}")
        return None, None

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