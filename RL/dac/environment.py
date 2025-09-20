"""
DAC Environment Wrapper Module

This module provides environment wrapper for DAC that integrates:
- Response feature analysis
- Option-level checking mechanisms
- Dual MDP state space handling (global vs story-level)
- Option length bonus computation

Based on the existing Environment class and adapted for DAC requirements.
"""

import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
from copy import deepcopy

from ..environment import Environment
from Structure import structure, check


class DACEnvironmentWrapper:
    """
    Environment wrapper for DAC with response feature integration.

    Provides dual MDP interface:
    - High-level MDP: Global state for option selection
    - Low-level MDP: Story-level state for action selection

    Integrates option-level checking and length bonus computation.
    """

    def __init__(self,
                 base_env: Environment,
                 response_feature: bool = True,
                 length_bonus_weight: float = 0.1,
                 option_timeout: int = 50,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize DAC environment wrapper.

        Args:
            base_env: Base environment instance
            response_feature: Whether to enable response feature analysis
            length_bonus_weight: Weight for option length bonus
            option_timeout: Maximum steps per option
            logger: Logger instance
        """
        self.base_env = base_env
        self.response_feature = response_feature
        self.length_bonus_weight = length_bonus_weight
        self.option_timeout = option_timeout
        self.logger = logger or logging.getLogger(__name__)

        # Option tracking
        self.current_option = None
        self.option_start_step = 0
        self.option_length = 0
        self.total_steps = 0

        # Accumulated rewards and states
        self.option_accumulated_reward = 0.0
        self.option_states = []
        self.option_actions = []

        # Material usage tracking for option-level evaluation
        self.initial_material_usage = None
        self.current_material_usage = None

        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0

    def reset(self, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Reset environment and return initial states for both MDPs.

        Returns:
            Tuple of (dual_states, info) where dual_states contains:
            - story_state: Story-level state for low-level MDP
            - global_state: Global state for high-level MDP
            - graph_data: Full graph data for feature extraction
        """
        self.logger.info(f"=== Environment Reset Called with kwargs: {kwargs} ===")

        # Reset base environment
        structure_obj = self.base_env.reset(**kwargs)
        self.logger.info(f"Base environment reset completed, structure: {type(structure_obj)}")

        # Validate structure object
        if structure_obj is None:
            raise ValueError("Base environment reset returned None structure")

        # Store current structure for easy access
        self._current_structure = structure_obj

        # Verify structure has required attributes
        required_attrs = ['already_minimum_section_story_indexes', 'story_level_actions']
        missing_attrs = [attr for attr in required_attrs if not hasattr(structure_obj, attr)]
        if missing_attrs:
            self.logger.warning(f"Structure missing attributes: {missing_attrs}")
            # Initialize missing attributes with empty defaults
            for attr in missing_attrs:
                setattr(structure_obj, attr, [])

        # Reset DAC-specific tracking
        self.current_option = None
        self.option_start_step = 0
        self.option_length = 0
        self.total_steps = 0
        self.option_accumulated_reward = 0.0
        self.option_states = []
        self.option_actions = []
        self.episode_step = 0
        self.episode_reward = 0.0

        # Get initial material usage
        self.initial_material_usage = structure_obj.calculate_material_usage()
        self.current_material_usage = self.initial_material_usage
        self.logger.info(f"Initial material usage: {self.initial_material_usage}")

        # Check initial structure state
        already_minimum = set(getattr(structure_obj, 'already_minimum_section_story_indexes', []) or [])
        total_actions = len(getattr(structure_obj, 'story_level_actions', []))
        self.logger.info(f"Initial structure state - already_minimum: {len(already_minimum)}/{total_actions} actions")

        # Extract dual state representation
        dual_states = self._extract_dual_states(structure_obj)

        info = {
            'structure': structure_obj,
            'material_usage': self.initial_material_usage,
            'episode_step': self.episode_step,
            'option_length': self.option_length
        }

        self.logger.info(f"=== Environment Reset Completed Successfully ===")
        return dual_states, info

    def step(self,
             action: torch.Tensor,
             option: int,
             option_terminated: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute action in environment with DAC-specific handling.

        Args:
            action: Action to execute
            option: Current option
            option_terminated: Whether option just terminated

        Returns:
            Tuple of (next_dual_states, rewards, done, info)
        """
        # Update option tracking
        if self.current_option != option or option_terminated:
            # Option changed or terminated
            if self.current_option is not None:
                self.logger.debug(f"Option {self.current_option} terminated after {self.option_length} steps")

            self.current_option = option
            self.option_start_step = self.total_steps
            self.option_length = 0
            self.option_accumulated_reward = 0.0
            self.option_states = []
            self.option_actions = []

        # Get current structure
        current_structure = self._get_current_structure()
        self.logger.debug(f"Step {self.episode_step}: action={action}, option={option}, option_terminated={option_terminated}")

        # Execute action in base environment
        next_structure, base_reward, done, fail_name, fail_reason, is_min_section = self._execute_base_action(
            current_structure, action
        )

        # Update tracking
        self.total_steps += 1
        self.episode_step += 1
        self.option_length += 1
        self.option_accumulated_reward += base_reward

        # Store option experience
        self.option_states.append(self._extract_dual_states(current_structure))
        self.option_actions.append(action)

        # Update current structure and material usage
        self._current_structure = next_structure
        self.current_material_usage = next_structure.calculate_material_usage()

        # Check structure state before termination
        already_minimum = set(getattr(next_structure, 'already_minimum_section_story_indexes', []) or [])
        total_actions = len(getattr(next_structure, 'story_level_actions', []))
        all_indexes_zero = self._check_all_indexes_zero(next_structure)

        self.logger.debug(f"Step {self.episode_step} status - is_min_section: {is_min_section}, "
                         f"all_indexes_zero: {all_indexes_zero}, already_minimum: {len(already_minimum)}/{total_actions}, "
                         f"base_done: {done}, fail_name: {fail_name}")

        # Check if all indexes are zero (structure at minimum) - force terminate episode and option
        if is_min_section or all_indexes_zero:
            self.logger.info(f"TERMINATION TRIGGERED - Episode {self.episode_step}: is_min_section={is_min_section}, "
                           f"all_indexes_zero={all_indexes_zero}, already_minimum={len(already_minimum)}/{total_actions}")
            done = True
            option_terminated = True
            # Apply penalty for minimum section (like Option-Critic)
            base_reward = -1.0

        # Compute dual rewards
        rewards = self._compute_dual_rewards(
            base_reward, done, option_terminated, fail_name, fail_reason
        )

        # Extract next dual states
        next_dual_states = self._extract_dual_states(next_structure)

        # Check for option timeout
        if self.option_length >= self.option_timeout and not done:
            option_terminated = True
            self.logger.debug(f"Option {option} timed out after {self.option_length} steps")

        # Prepare info
        info = {
            'structure': next_structure,
            'base_reward': base_reward,
            'material_usage': self.current_material_usage,
            'fail_name': fail_name,
            'fail_reason': fail_reason,
            'option_terminated': option_terminated,
            'option_length': self.option_length,
            'episode_step': self.episode_step,
            'option_accumulated_reward': self.option_accumulated_reward
        }

        # Option-level checking if option terminated
        if option_terminated and not done:
            option_check_result = self._perform_option_level_check(next_structure)
            info.update(option_check_result)

        self.episode_reward += base_reward

        return next_dual_states, rewards, done, info

    def _extract_dual_states(self, structure_obj: structure.Structure) -> Dict[str, torch.Tensor]:
        """
        Extract dual state representation for both MDPs.

        Args:
            structure_obj: Structure object

        Returns:
            Dictionary containing dual states
        """
        # Get graph data from structure
        graph_data = self._structure_to_graph_data(structure_obj)

        # Story-level state (for low-level MDP)
        # This should be the full detailed state for action selection
        story_state = graph_data

        # Global state (for high-level MDP)
        # This should be a more abstract representation for option selection
        global_state = self._compute_global_state(graph_data, structure_obj)

        return {
            'story_state': story_state,
            'global_state': global_state,
            'graph_data': graph_data
        }

    def _structure_to_graph_data(self, structure_obj: structure.Structure) -> Dict[str, torch.Tensor]:
        """Convert structure object to graph data format."""
        # Initialize graph if it doesn't exist
        if not hasattr(structure_obj, 'graph') or structure_obj.graph is None:
            # Calculate static response features like the base environment does
            try:
                if hasattr(self.base_env, 'code_analysis_dir'):
                    load_cases, static_responses = check.get_response(structure_obj, self.base_env.code_analysis_dir)
                    _, static_response_features, _ = check.process_response(structure_obj, load_cases, static_responses)
                    structure_obj.init_graph_GraphRL(static_response_features, None)
                else:
                    # Fallback if code_analysis_dir is not available
                    structure_obj.init_graph_GraphRL()
            except Exception as e:
                self.logger.warning(f"Failed to initialize graph for structure: {e}")
                # Create a minimal fallback graph structure
                structure_obj.init_graph_GraphRL()

        # Extract graph representation from structure.graph (same as option_critic)
        graph = structure_obj.graph
        story_batch = structure_obj.aux["story_batch"]

        # Calculate structure_story_ptr for proper story-level processing
        # This matches the option_critic pattern for story member indexing
        structure_story_ptr = None
        if hasattr(structure_obj.aux, 'story_xdir_beam_member'):
            # Count story members: x-beam, z-beam, outer-column, inner-column
            story_members_lists = [
                structure_obj.aux.get("story_xdir_beam_member", []),
                structure_obj.aux.get("story_zdir_beam_member", []),
                structure_obj.aux.get("story_outer_column_member", []),
                structure_obj.aux.get("story_inner_column_member", [])
            ]

            structure_story_ptr = [0]
            story_count = 0
            for story_members_list in story_members_lists:
                for story_members in story_members_list:
                    story_count += 1
            structure_story_ptr.append(story_count)

        graph_data = {
            'x': graph.x.clone().detach(),
            'edge_index': graph.edge_index.clone().detach(),
            'edge_attr': graph.edge_attr.clone().detach(),
            'story_batch': story_batch.clone().detach(),
            'structure_story_ptr': structure_story_ptr
        }

        return graph_data

    def _compute_global_state(self, graph_data: Dict[str, torch.Tensor], structure_obj: structure.Structure) -> torch.Tensor:
        """
        Compute global state representation for high-level MDP.

        Args:
            graph_data: Graph data
            structure_obj: Structure object

        Returns:
            Global state tensor
        """
        # Extract global features for option selection
        # This could include:
        # - Aggregated structural properties
        # - Material usage information
        # - Constraint satisfaction status
        # - Progress indicators

        features = []

        # Material usage features
        material_usage = structure_obj.calculate_material_usage()
        features.append(material_usage / 1000.0)  # Normalize

        # Structural dimensions
        features.extend([
            structure_obj.x_span_num / 10.0,
            structure_obj.z_span_num / 10.0,
            structure_obj.story_num / 20.0,
            structure_obj.story_height / 5000.0
        ])

        # Response features are handled via structure.graph edge/node features
        # No need to access separate response_features attribute

        # Episode progress
        features.extend([
            self.episode_step / 500.0,  # Normalize by max episode length
            self.option_length / self.option_timeout  # Option progress
        ])

        return torch.tensor(features, dtype=torch.float32)

    def _execute_base_action(self, structure_obj: structure.Structure, action: torch.Tensor) -> Tuple:
        """Execute action in base environment and return is_min_section flag."""
        # Convert action to int if needed (like Option-Critic rollout)
        if isinstance(action, torch.Tensor):
            if action.numel() == 1:
                action_int = int(action.item())
            else:
                # If action is a multi-element tensor (e.g., probabilities), take argmax
                action_int = int(action.argmax().item())
        else:
            action_int = int(action)

        # Validate action bounds
        num_actions = len(structure_obj.story_level_actions)
        self.logger.debug(f"Action validation: action_int={action_int}, num_actions={num_actions}, action_shape={action.shape if isinstance(action, torch.Tensor) else 'N/A'}")

        if not (0 <= action_int < num_actions):
            # If action is out of bounds, clip it to valid range
            action_int = max(0, min(action_int, num_actions - 1))
            self.logger.warning(f"Action was out of bounds, clipped to {action_int} (valid range: [0, {num_actions}))")

        if not (0 <= action_int < num_actions):
            error_msg = f"Action {action_int} is out of bounds [0, {num_actions})"
            self.logger.error(error_msg)
            raise ValueError(f"Invalid action index: {action_int}")

        try:
            # Ensure structure has graph initialized before calling step
            if not hasattr(structure_obj, 'graph') or structure_obj.graph is None:
                self.logger.debug("Initializing graph for structure before step execution")
                self._structure_to_graph_data(structure_obj)  # This will initialize the graph

            # Call the actual base environment step method (like apply_primitive_action in Option-Critic)
            # This should be similar to Option-Critic's apply_primitive_action call
            if hasattr(self.base_env, 'step'):
                next_structure, reward, step_done, is_min_section, fail_reason = self.base_env.step(
                    structure_obj, action_int
                )
                fail_name = "constraint_violation" if step_done else None
                self.logger.debug(f"Base environment step returned: step_done={step_done}, fail_name={fail_name}, fail_reason={fail_reason}")
                
                done = step_done  # Episode done if constraints violated

                self.logger.debug(f"Base environment step returned: next_structure={next_structure}, reward={reward}, step_done={step_done}, is_min_section={is_min_section}, fail_reason={fail_reason}")
                self.logger.debug(f"done:{done}")
            else:
                # Fallback: simulate the step
                self.logger.warning("Base environment doesn't have step method, using fallback")
                next_structure = deepcopy(structure_obj)
                reward = -1.0
                done = False
                fail_name = None
                fail_reason = None
                is_min_section = False

            return next_structure, reward, done, fail_name, fail_reason, is_min_section

        except Exception as e:
            self.logger.error(f"Error executing action {action_int}: {e}")
            # Return safe fallback values
            raise ValueError(f"Error executing action {action_int}: {e}")
            # return structure_obj, -10.0, True, "execution_error", str(e), False

    def _compute_dual_rewards(self,
                            base_reward: float,
                            done: bool,
                            option_terminated: bool,
                            fail_name: Optional[str],
                            fail_reason: Optional[str]) -> Dict[str, float]:
        """
        Compute rewards for both MDPs.

        Args:
            base_reward: Base environment reward
            done: Whether episode is done
            option_terminated: Whether option terminated
            fail_name: Failure name if any
            fail_reason: Failure reason if any

        Returns:
            Dictionary with rewards for both MDPs
        """
        # Low-level reward (bar MDP) - immediate action reward
        low_level_reward = base_reward

        # High-level reward (hat MDP) - option-level reward with length bonus
        if option_terminated or done:
            # Compute option reward with length bonus
            material_saved = self.initial_material_usage - self.current_material_usage
            material_reward = material_saved / 1000.0  # Normalize

            # Length bonus as in DAC paper
            length_bonus = self.length_bonus_weight * self.option_length

            # Penalty for failure
            failure_penalty = -10.0 if fail_name else 0.0

            high_level_reward = material_reward + length_bonus + failure_penalty
        else:
            # No high-level reward until option terminates
            high_level_reward = 0.0

        return {
            'high_level_reward': high_level_reward,
            'low_level_reward': low_level_reward
        }

    def _perform_option_level_check(self, structure_obj: structure.Structure) -> Dict[str, Any]:
        """
        Perform option-level structural checking.

        Args:
            structure_obj: Structure object

        Returns:
            Dictionary with check results
        """
        check_results = {
            'whether_pass': True,
            'constraint_violations': [],
            'material_efficiency': 0.0
        }

        try:
            # Perform structural analysis if response features enabled
            if self.response_feature:
                # This would call the actual checking functions
                # For now, placeholder implementation

                # Calculate material efficiency
                material_saved = self.initial_material_usage - self.current_material_usage
                check_results['material_efficiency'] = material_saved / self.initial_material_usage

                # Check structural constraints
                # This would involve calling Structure/check.py functions

                self.logger.debug(f"Option-level check completed. Material efficiency: {check_results['material_efficiency']:.3f}")

        except Exception as e:
            raise e
            # self.logger.error(f"Error in option-level check: {e}")
            # check_results['whether_pass'] = False
            # check_results['constraint_violations'].append(f"Check error: {str(e)}")

        return check_results

    def _get_current_structure(self) -> structure.Structure:
        """Get current structure from base environment."""
        # Check if we have a current structure stored
        if hasattr(self, '_current_structure') and self._current_structure is not None:
            return self._current_structure

        # Fallback to base environment's testing structure
        if hasattr(self.base_env, '_testing_structure') and self.base_env._testing_structure is not None:
            return self.base_env._testing_structure

        # If nothing is available, raise an error with helpful message
        raise ValueError("No current structure available. Make sure environment has been reset properly.")

    def get_option_statistics(self) -> Dict[str, Any]:
        """Get statistics about current option."""
        return {
            'current_option': self.current_option,
            'option_length': self.option_length,
            'option_accumulated_reward': self.option_accumulated_reward,
            'option_start_step': self.option_start_step,
            'material_usage_change': self.initial_material_usage - self.current_material_usage if self.current_material_usage else 0
        }

    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics about current episode."""
        return {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'total_steps': self.total_steps,
            'current_material_usage': self.current_material_usage,
            'material_saved': self.initial_material_usage - self.current_material_usage if self.current_material_usage else 0
        }

    def _check_all_indexes_zero(self, structure_obj: structure.Structure) -> bool:
        """
        Check if all structure indexes are at minimum (all zeros).
        This indicates that the episode should be terminated.
        Based on Option-Critic rollout logic.

        Args:
            structure_obj: Structure object

        Returns:
            True if all indexes are zero (episode should terminate)
        """
        # Check if structure has already_minimum_section_story_indexes
        already_minimum = set(getattr(structure_obj, 'already_minimum_section_story_indexes', []) or [])

        # Get total number of story level actions
        total_actions = len(getattr(structure_obj, 'story_level_actions', []))

        # Log detailed structure state for debugging
        self.logger.debug(f"Structure state check - already_minimum: {already_minimum}, "
                         f"total_actions: {total_actions}, "
                         f"will_terminate: {total_actions > 0 and len(already_minimum) >= total_actions}")

        # If all story-level actions are at minimum, episode should terminate
        if total_actions > 0 and len(already_minimum) >= total_actions:
            self.logger.warning(f"CRITICAL: All structure indexes are at minimum: {len(already_minimum)}/{total_actions} actions at minimum")
            return True

        # Additional check: if there are no available actions
        if total_actions == 0:
            self.logger.warning(f"CRITICAL: No story level actions available, total_actions=0")
            return True

        return False