"""
Utility functions for Option-Critic algorithm.

This module contains helper functions for graph data extraction,
action space utilities, and environment interaction.
"""

from Structure import check, check_nda


def get_graph_data(structure, device):
    """
    Extract graph data for OptionCriticGNN.
    Returns tuple of graph inputs for the neural network.
    """
    graph = structure.graph
    story_batch = structure.aux["story_batch"].to(device)

    # Calculate structure_story_ptr for proper story-level processing
    # This matches the DQN pattern for story member indexing
    structure_story_ptr = None
    if hasattr(structure.aux, 'story_xdir_beam_member'):
        # Count story members: x-beam, z-beam, outer-column, inner-column
        story_members_lists = [
            structure.aux.get("story_xdir_beam_member", []),
            structure.aux.get("story_zdir_beam_member", []),
            structure.aux.get("story_outer_column_member", []),
            structure.aux.get("story_inner_column_member", [])
        ]

        structure_story_ptr = [0]
        story_count = 0
        for story_members_list in story_members_lists:
            for story_members in story_members_list:
                story_count += 1
        structure_story_ptr.append(story_count)

    return (
        graph.x.to(device),
        graph.edge_index.to(device),
        graph.edge_attr.to(device),
        story_batch,
        structure_story_ptr
    )


def num_actions(structure) -> int:
    return len(structure.story_level_actions)


def apply_primitive_action(base_env, structure, action: int, logger=None):
    """
    Apply a primitive action using the base environment.

    Returns:
        step_reward: float, reward from base env for this step
        step_pass: bool, whether constraints pass after this step
        is_minimum_section: bool, whether minimum section is reached
        fail_reason: str, fail reason string from base env
    """
    if logger:
        logger.debug(f"apply_primitive_action: Called with action={action}, type={type(action)}")
        logger.debug(f"apply_primitive_action: base_env type={type(base_env)}")
        logger.debug(f"apply_primitive_action: structure type={type(structure)}")

    # Additional validation
    if hasattr(structure, 'story_level_actions'):
        if logger:
            logger.debug(f"apply_primitive_action: structure has {len(structure.story_level_actions)} story_level_actions")

    # =========================================================================
    # 🔍 DEBUG LOGGING: Material tracking INSIDE apply_primitive_action
    # =========================================================================
    if logger:
        material_before_env_step = float(structure.calculate_material_usage())
        sections_before_env_step = list(getattr(structure, 'story_level_sections', []))
        logger.info(f"[APPLY_ACTION_DEBUG] BEFORE base_env.step(action={action}):")
        logger.info(f"  material_usage: {material_before_env_step:.6f} kg")
        logger.info(f"  sections (all): {sections_before_env_step[:]}")

    # Call base environment step
    if logger:
        logger.debug(f"apply_primitive_action: Calling base_env.step(structure, {action})")
    result = base_env.step(structure, action)
    if logger:
        logger.debug(f"apply_primitive_action: base_env.step returned {len(result)} items")

    structure, step_reward, done, fail_name, fail_reason = result
    if logger:
        logger.debug(f"apply_primitive_action: Unpacked result - step_reward={step_reward}, done={done}, fail_reason={fail_reason}")

    # =========================================================================
    # 🔍 DEBUG LOGGING: Material tracking AFTER base_env.step
    # =========================================================================
    if logger:
        material_after_env_step = float(structure.calculate_material_usage())
        sections_after_env_step = list(getattr(structure, 'story_level_sections', []))
        material_change = material_before_env_step - material_after_env_step

        logger.info(f"[APPLY_ACTION_DEBUG] AFTER base_env.step(action={action}):")
        logger.info(f"  material_usage: {material_after_env_step:.6f} kg")
        logger.info(f"  sections (all): {sections_after_env_step[:]}")
        logger.info(f"  material_change: {material_change:.6f} kg")
        logger.info(f"  step_reward (from env.step): {step_reward:.6f}")
        logger.info(f"  sections_changed: {sections_before_env_step != sections_after_env_step}")

        # 🚨 Alert if reward doesn't match material change
        reward_mismatch = abs(float(step_reward) - material_change) > 1e-3
        if reward_mismatch:
            logger.warning(f"[APPLY_ACTION_DEBUG] ⚠️ WARNING: Reward mismatch!")
            logger.warning(f"  Expected reward (material_change): {material_change:.6f}")
            logger.warning(f"  Actual reward (step_reward): {step_reward:.6f}")
            logger.warning(f"  Difference: {abs(float(step_reward) - material_change):.6f}")

    # Derive per-step pass/minimum-section states from base env outputs
    is_minimum_section = bool(done and (fail_reason == "minimum_section"))
    step_pass = not (done and (fail_reason != "minimum_section"))

    if logger:
        logger.debug(f"apply_primitive_action: Returning - step_reward={float(step_reward)}, step_pass={step_pass}, is_minimum_section={is_minimum_section}")
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