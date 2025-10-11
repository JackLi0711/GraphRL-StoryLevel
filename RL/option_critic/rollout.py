"""
Option rollout functionality for Option-Critic algorithm.

This module contains the rollout_option function which executes
a single option composed of a sequence of primitive actions.
"""

import torch
from .utils import get_graph_data, apply_primitive_action
from .logger import termination_logger


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

    # DEBUG: Track termination_reason changes
    def set_termination_reason(reason, location):
        nonlocal termination_reason
        if logger:
            logger.debug(f"Setting termination_reason='{reason}' at location: {location}")
        termination_reason = reason

    # Initialize variables that are used later
    step_pass = True  # Default to True - will be updated if step fails
    step_transitions = []  # list of dicts: {obs, action, logp, entropy, reward, done, next_obs, option}

    # Record pre-option material usage to compute saved amount for this option
    pre_option_material_usage = float(structure.calculate_material_usage())

    # initial state for intra-option policy
    graph_data = get_graph_data(structure, device)
    story_level_state, global_state = oc_model.get_state(*graph_data)
    # Extract structure_story_ptr for action selection
    structure_story_ptr = graph_data[4]  # 5th element from get_graph_data

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

        # Safety check to prevent infinite loops
        if safety_counter > max_safety_iterations:
            if logger:
                logger.error(f"rollout_option exceeded safety counter ({max_safety_iterations}), forcing termination")
            set_termination_reason("safety_timeout", "safety_counter_exceeded")
            break

        # Create valid actions mask
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

        # Check if there are any valid actions left
        if valid_mask.sum() == 0:
            # No valid actions - check current structure constraints
            if logger:
                logger.debug("No valid actions available - checking current structure constraints")

            from .utils import check_constraints_without_update
            whether_pass, check_fail_reason = check_constraints_without_update(structure, base_env)

            if whether_pass:
                # Structure passes all constraints - successful termination
                if logger:
                    logger.debug("Structure passes constraints - terminating as minimum_section")
                set_termination_reason("minimum_section", "no_valid_actions_passed")
            else:
                # Structure fails constraints - mark as failed
                if logger:
                    logger.debug(f"Structure fails constraints - fail_reason: {check_fail_reason}")
                step_pass = False
                # termination_reason will be set to "max_len" after loop

            episode_done = True
            break

        # intra-option action with valid actions mask
        log_msg = f"Getting action for option {current_option}"
        if logger:
            logger.debug(log_msg)
        else:
            print(f"DEBUG: {log_msg}")

        action, logp, entropy = oc_model.get_action(story_level_state, structure_story_ptr, current_option, valid_mask)
        entropies.append(float(entropy.detach().cpu().numpy()))

        log_msg = f"Got action {action}, entropy: {entropy.item()}"
        if logger:
            logger.debug(log_msg)
        else:
            print(f"DEBUG: {log_msg}")

        # DEBUG: Action validation checks
        if logger:
            logger.debug(f"Action validation checks:")
            logger.debug(f"  action = {action}, type = {type(action)}")
            logger.debug(f"  action is integer: {isinstance(action, int)}")

            # Validate action bounds and restriction status
            num_story_members = len(structure.aux.get('story_xdir_beam_member', [])) + len(structure.aux.get('story_zdir_beam_member', [])) + len(structure.aux.get('story_outer_column_member', [])) + len(structure.aux.get('story_inner_column_member', []))
            logger.debug(f"  structure has {num_story_members} story members")
            logger.debug(f"  structure has {len(structure.story_level_actions)} story-level actions")
            logger.debug(f"  action bounds check: 0 <= {action} < {len(structure.story_level_actions)} = {0 <= action < len(structure.story_level_actions)}")

            # Check if the selected action is in the valid mask
            if valid_mask is not None:
                action_is_valid = valid_mask[action].item() if action < len(valid_mask) else False
                logger.debug(f"  action in valid_mask: {action_is_valid}")

                if not action_is_valid:
                    logger.error(f"CRITICAL: Selected action {action} is INVALID according to valid_mask!")
                    logger.error(f"  valid_mask[{action}] = {valid_mask[action].item() if action < len(valid_mask) else 'out_of_bounds'}")

                    # Find valid actions
                    valid_action_indices = torch.where(valid_mask)[0].tolist()
                    logger.error(f"  Available valid actions: {valid_action_indices}")

            # Check restriction reasons
            already_minimum = set(getattr(structure, 'already_minimum_section_story_indexes', []) or [])
            restricted_actions = set()
            if hasattr(structure, 'restrict_action_space'):
                restricted = structure.restrict_action_space()
                if restricted is not None:
                    restricted_actions.update(restricted)

            if action in already_minimum:
                logger.error(f"  Action {action} is in already_minimum: {already_minimum}")
            if action in restricted_actions:
                logger.error(f"  Action {action} is in restricted_actions: {restricted_actions}")

            if hasattr(structure, 'get_valid_actions'):
                valid_actions = structure.get_valid_actions()
                logger.debug(f"  structure valid actions: {valid_actions}")
                logger.debug(f"  action in valid actions: {action in valid_actions}")

        log_msg = f"Applying primitive action {action}"
        if logger:
            logger.debug(log_msg)
        else:
            print(f"DEBUG: {log_msg}")

        if logger:
            logger.debug(f"Calling apply_primitive_action with action={action}, structure type={type(structure)}")

            # Enhanced pre-call validation
            logger.debug(f"Pre-call validation:")
            logger.debug(f"  - action type: {type(action)}")
            logger.debug(f"  - action value: {action}")
            logger.debug(f"  - action is integer: {isinstance(action, int)}")
            logger.debug(f"  - base_env type: {type(base_env)}")
            logger.debug(f"  - base_env has step method: {hasattr(base_env, 'step')}")

        # Validate structure
        if logger:
            logger.debug(f"  - structure story_level_actions length: {len(structure.story_level_actions)}")
            logger.debug(f"  - action bounds valid: {0 <= action < len(structure.story_level_actions)}")

        # Check if action is within valid bounds
        if not (0 <= action < len(structure.story_level_actions)):
            error_msg = f"Action {action} is out of bounds [0, {len(structure.story_level_actions)})"
            if logger:
                logger.error(error_msg)
            raise ValueError(f"Invalid action index: {action}")

        # Ensure action is integer
        if not isinstance(action, int):
            if logger:
                logger.warning(f"Converting action from {type(action)} to int")
            action = int(action)
            if logger:
                logger.debug(f"  - converted action: {action}")

        # =====================================================================
        # 🔍 DEBUG LOGGING: Capture state BEFORE action (Hypothesis 1 & 2)
        # =====================================================================
        if logger:
            sections_before = list(getattr(structure, 'story_level_sections', []))
            material_before = float(structure.calculate_material_usage())
            logger.info(f"[REWARD_DEBUG] BEFORE action {action}:")
            logger.info(f"  sections (all): {sections_before[:]}")
            logger.info(f"  material_usage: {material_before:.6f} kg")

        # Call with enhanced error capture
        if logger:
            logger.debug(f"Calling base_env.step(structure, {action})")
        structure, step_reward, step_pass, is_min_section, fail_reason = apply_primitive_action(base_env, structure, action, logger)

        # =====================================================================
        # 🔍 DEBUG LOGGING: Capture state AFTER action (Hypothesis 1 & 2)
        # =====================================================================
        if logger:
            sections_after = list(getattr(structure, 'story_level_sections', []))
            material_after = float(structure.calculate_material_usage())
            material_saved = material_before - material_after
            sections_changed = (sections_before != sections_after)

            # Count how many sections changed
            num_sections_changed = sum(1 for i in range(min(len(sections_before), len(sections_after)))
                                      if sections_before[i] != sections_after[i])

            logger.info(f"[REWARD_DEBUG] AFTER action {action}:")
            logger.info(f"  sections (first 10): {sections_after[:10]}")
            logger.info(f"  material_usage: {material_after:.6f} kg")
            logger.info(f"  material_saved: {material_saved:.6f} kg")
            logger.info(f"  step_reward (from env): {step_reward:.6f}")
            logger.info(f"  sections_changed: {sections_changed}")
            logger.info(f"  num_sections_changed: {num_sections_changed}")

            # 🚨 Alert if no change detected
            if not sections_changed:
                logger.warning(f"[REWARD_DEBUG] ⚠️ WARNING: Sections did NOT change after action {action}!")
            if abs(material_saved) < 1e-6:
                logger.warning(f"[REWARD_DEBUG] ⚠️ WARNING: Material usage did NOT change (saved={material_saved:.9f})!")
            if abs(step_reward) < 1e-6 and step_pass:
                logger.warning(f"[REWARD_DEBUG] ⚠️ WARNING: step_reward is 0 but step_pass=True!")

        # Apply option length bonus: reward += (step_number - 1) * bonus
        # length is 0-indexed, so length equals (step_number - 1)
        length_bonus = length * option_length_bonus
        original_reward = step_reward  # Store original reward before adding bonus
        step_reward += length_bonus

        log_msg = f"Action applied - original_reward: {original_reward}, length_bonus: {length_bonus}, final_reward: {step_reward}, pass: {step_pass}, min_section: {is_min_section}, fail_reason: {fail_reason}"
        if logger:
            logger.debug(log_msg)
            logger.debug("Action application completed successfully")
        # Don't accumulate rewards at option level anymore
        length += 1
        if logger:
            logger.debug(f"Updated length to {length}, step_reward: {step_reward}")

        # minimum section: force terminate option and episode
        if is_min_section:
            if logger:
                logger.debug("Minimum section reached, terminating option and episode")
            set_termination_reason("minimum_section", "is_min_section_true")
            episode_done = True
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
            if logger:
                logger.debug("Added final step transition, breaking from loop")
            break

        # compute next state for termination prediction
        if logger:
            logger.debug("Computing next state for termination prediction")
        next_graph_data = get_graph_data(structure, device)
        next_story_level_state, next_global_state = oc_model.get_state(*next_graph_data)
        next_state = next_global_state  # Use global state for termination prediction
        if logger:
            logger.debug("Successfully computed next state")

        # record step transition
        if logger:
            logger.debug("Recording step transition")
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
        if logger:
            logger.debug(f"Step transition recorded, total transitions: {len(step_transitions)}")

        # option termination by beta
        if logger:
            logger.debug("Checking option termination by beta")
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
            if logger:
                logger.debug(f"TERMINATION_PROBS Step {length+1}: [{term_probs_str}] | Current Option {current_option}: β{current_option}={term_probs_display[current_option]:.4f} → {'TERMINATE' if option_termination else 'CONTINUE'}")

        # Log termination probability prediction
        termination_logger.log_termination_prediction(
            option=current_option,
            termination_probs=termination_probs,
            termination_decision=option_termination,
            step=length,
            context="rollout_option"
        )

        if option_termination:
            if logger:
                logger.debug(f"==> Option {current_option} TERMINATED by β={term_probs_display[current_option]:.4f} at step {length}")
            set_termination_reason("beta", "option_termination_true")
            # Update state for potential next iteration (though loop will break)
            story_level_state, global_state = next_story_level_state, next_global_state
            graph_data = next_graph_data
            break

        # continue the option
        if logger:
            logger.debug("Continuing option, updating state")
        story_level_state, global_state = next_story_level_state, next_global_state
        graph_data = next_graph_data
        if logger:
            logger.debug("State updated, continuing to next iteration")

    # if not terminated by beta/minimum_section, it hits max length
    if logger:
        logger.debug(f"Exited rollout loop with length={length}, max_len={max_option_len}, termination_reason={termination_reason}")
    option_done = True
    if termination_reason is None:
        set_termination_reason("max_len", "loop_completed_naturally")
        if logger:
            logger.debug("Set termination reason to max_len")

    # Check constraint compliance and apply penalty to last step if failed
    # Note: step_pass refers to the last evaluated step
    passed = step_pass if length > 0 else True
    if logger:
        logger.debug(f"Final step_pass check: step_pass={step_pass}, length={length}, passed={passed}")

    if not passed:
        if logger:
            logger.debug("Option failed, applying penalty to last step")
        episode_done = True
        # Apply penalty reward to the last step that caused the failure
        if len(step_transitions) > 0:
            step_transitions[-1]["reward"] = -1 # -1000.0
            step_transitions[-1]["original_reward"] = -1 # -1000.0
            step_transitions[-1]["done"] = True
            if logger:
                logger.debug("Updated last step transition with penalty reward")

    if logger:
        logger.debug(f"Option execution completed, passed={passed}")

    # Compute option-level saved material (before vs after this option)
    post_option_material_usage = float(structure.calculate_material_usage())
    option_saved_material = float(max(0.0, (pre_option_material_usage - post_option_material_usage)))

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

    next_state = global_state  # latest global state for return
    if logger:
        logger.debug(f"rollout_option returning - option_done: {option_done}, episode_done: {episode_done}, termination_reason: {termination_reason}, num_transitions: {len(step_transitions)}")
    return structure, next_state, option_done, episode_done, stats, step_transitions, termination_reason