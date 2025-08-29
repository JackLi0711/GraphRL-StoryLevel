import typing
from typing import Dict, Tuple

import torch
from Structure import check, check_nda

from RL.environment import Environment
from RL.oc_policy_adapter import GraphStateAdapter


class OptionDesignEnv:
    """
    A wrapper that elevates step-level structural design environment to option-level interaction
    while preserving the original environment and option-critic implementation untouched.

    Rules implemented:
    - Only at option termination do we emit the option reward and decide to stop/continue the episode.
    - If an option ends and fails constraints: episode terminates with reward -1000 for that option.
    - If minimum section is reached during an option: forcibly terminate the option, check constraints,
      and terminate the entire episode.
    - With add_response_features=True, each primitive action invokes static analysis via base env.
    """

    def __init__(self,
                 base_env: Environment,
                 adapter: GraphStateAdapter,
                 max_option_len: int = 16,
                 device: torch.device = torch.device("cpu")) -> None:
        self.base_env = base_env
        self.adapter = adapter
        self.max_option_len = max_option_len
        self.device = device

        self.structure = None  # set by reset()

    def reset(self, testing: bool = False):
        self.structure = self.base_env.reset(testing=testing)
        return self.structure

    @torch.no_grad()
    def get_obs(self) -> torch.Tensor:
        """
        Aggregate graph observation into a fixed-length vector z ∈ R^{3H} for Option-Critic.
        """
        graph = self.structure.graph
        story_batch = self.structure.aux["story_batch"].to(self.device)
        z = self.adapter(
            graph_x=graph.x,
            graph_edge_index=graph.edge_index,
            graph_edge_attr=graph.edge_attr,
            story_batch=story_batch,
            structure_story_ptr=None,
        )
        return z

    @staticmethod
    def num_actions(structure) -> int:
        return GraphStateAdapter.num_actions(structure)

    def _apply_primitive_action(self, action: int) -> Tuple[float, bool, bool, str]:
        """
        Apply a primitive action using the base environment.

        Returns:
            step_reward: float, reward from base env for this step
            step_pass: bool, whether constraints pass after this step
            is_minimum_section: bool, whether minimum section is reached
            fail_reason: str, fail reason string from base env
        """
        self.structure, step_reward, done, fail_name, fail_reason = self.base_env.step(self.structure, action)

        # Derive per-step pass/minimum-section states from base env outputs
        is_minimum_section = bool(done and (fail_reason == "minimum_section"))
        step_pass = not (done and (fail_reason != "minimum_section"))
        return float(step_reward), step_pass, is_minimum_section, fail_reason

    def _check_constraints_without_update(self) -> Tuple[bool, str]:
        """
        Run code checks on current structure WITHOUT applying any action.
        Returns:
            whether_pass (bool), fail_reason (str)
        """
        # static analysis
        load_cases, static_responses = check.get_response(self.structure, self.base_env.code_analysis_dir)
        static_constraint_condition, static_response_features, _ = check.process_response(self.structure, load_cases, static_responses)
        whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, self.base_env.check_displacement)

        # dynamic analysis if enabled and statically passed
        dynamic_response_features = None
        if self.base_env.do_nonlinear_dynamic_analysis and whether_pass:
            self.structure.update_graph_GraphLSTM()
            dynamic_responses = check_nda.get_response(self.structure, self.base_env.nda_simulator, self.base_env.MCE_ground_motion_set, self.base_env.device)
            dynamic_constraint_condition, dynamic_response_features, _ = check_nda.process_response(self.structure, dynamic_responses, self.base_env.nda_norm_dict)
            whether_pass, fail_name, fail_reason = check_nda.check_pass(dynamic_constraint_condition, self.base_env.check_displacement)

        # update graph features for consistency
        self.structure.update_graph_GraphRL(static_response_features, dynamic_response_features)

        return whether_pass, fail_reason

    @torch.no_grad()
    def rollout_option(self,
                       current_option: int,
                       oc_model,
                       epsilon: float = None) -> Tuple[torch.Tensor, float, bool, bool, Dict, list]:
        """
        Execute a single option composed of a sequence of primitive actions.

        Args:
            current_option: selected option index
            oc_model: OptionCriticFeatures-like model with get_state, get_action, predict_option_termination
            epsilon: optional exploration indicator (for logging)

        Returns:
            next_obs_z: aggregated state vector after option terminates (for next option)
            option_reward: accumulated reward within this option (or -1000 on failure)
            option_done: whether option terminated
            episode_done: whether episode terminated as a result of this option
            stats: dict with diagnostics (length, termination_reason, entropies, epsilon)
        """
        option_reward_sum: float = 0.0
        entropies = []
        length = 0
        termination_reason = None
        episode_done = False

        # Record pre-option material usage to compute saved amount for this option
        try:
            pre_option_material_usage = float(self.structure.calculate_material_usage())
        except Exception:
            pre_option_material_usage = None

        # initial state for intra-option policy
        obs_z = self.get_obs()
        state = oc_model.get_state(obs_z)

        step_transitions = []  # list of dicts: {obs_z, action, logp, entropy, reward, done, next_obs_z}

        while length < self.max_option_len:
            # intra-option action
            action, logp, entropy = oc_model.get_action(state, current_option)
            entropies.append(float(entropy.detach().cpu().numpy()))

            # Pre-check: if this action targets a minimum section, force terminate option & episode
            infeasible = set(getattr(self.structure, 'already_minimum_section_story_indexes', []) or [])
            if hasattr(self.structure, 'restrict_action_space'):
                try:
                    ra = self.structure.restrict_action_space()
                    infeasible |= set(ra if ra is not None else [])
                except Exception:
                    pass
            if action in infeasible:
                termination_reason = "minimum_section"
                whether_pass, fail_reason = self._check_constraints_without_update()
                # finalize option according to rules
                option_done = True
                episode_done = True
                # reward policy handled after loop using option_reward_sum & whether_pass
                step_pass = whether_pass
                break

            step_reward, step_pass, is_min_section, fail_reason = self._apply_primitive_action(action)
            option_reward_sum += step_reward
            length += 1

            # minimum section: force terminate option and episode
            if is_min_section:
                termination_reason = "minimum_section"
                episode_done = True
                next_obs_z = self.get_obs()
                step_transitions.append({
                    "obs_z": obs_z.clone(),
                    "action": action,
                    "logp": logp.detach().clone(),
                    "entropy": entropy.detach().clone(),
                    "reward": float(step_reward),
                    "done": True,
                    "next_obs_z": next_obs_z.clone(),
                })
                break

            # compute next state for termination prediction
            next_obs_z = self.get_obs()
            next_state = oc_model.get_state(next_obs_z)

            # record step transition
            step_transitions.append({
                "obs_z": obs_z.clone(),
                "action": action,
                "logp": logp.detach().clone(),
                "entropy": entropy.detach().clone(),
                "reward": float(step_reward),
                "done": False,
                "next_obs_z": next_obs_z.clone(),
            })

            # option termination by beta
            option_termination, _ = oc_model.predict_option_termination(next_state, current_option)
            if option_termination:
                termination_reason = "beta"
                state = next_state
                obs_z = next_obs_z
                break

            # continue the option
            state = next_state
            obs_z = next_obs_z

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
            post_option_material_usage = float(self.structure.calculate_material_usage())
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
            "story_level_sections": list(getattr(self.structure, 'story_level_sections', [])),
        }

        next_obs_z = obs_z  # latest aggregated observation
        return next_obs_z, option_reward, option_done, episode_done, stats, step_transitions

"""
Option Environment Wrapper for GraphRL structural design
Provides temporal abstraction over primitive actions by leveraging existing Environment functionality
"""

import copy
import torch
import numpy as np
from typing import Tuple, Dict, Any, List


class OptionEnvironmentWrapper:
    """
    Environment wrapper that provides temporal abstraction through options.
    
    Key design principles:
    - Reuse existing Environment.step() method for all structural analysis
    - Only handle option-level logic: action buffering, termination, reward aggregation
    - Maintain compatibility with add_response_features=True/False modes
    """
    
    def __init__(self, base_env, max_option_length: int = 8):
        """
        Initialize the option environment wrapper.
        
        Args:
            base_env: The base GraphRL Environment instance
            max_option_length: Maximum number of primitive actions per option
        """
        self.base_env = base_env
        self.max_option_length = max_option_length
        
        # Option state tracking
        self.action_buffer = []  # Actions taken in current option
        self.reward_buffer = []  # Rewards from each primitive action
        self.option_step_count = 0
        self.current_structure = None
        
        # For debugging and monitoring
        self.option_count = 0
        self.total_primitive_actions = 0
        
    def reset(self, testing: bool = False, taller: bool = False, initial_design=None):
        """Reset the environment and option state."""
        self._reset_option_state()
        self.option_count = 0
        self.total_primitive_actions = 0
        
        # Use base environment's reset method
        self.current_structure = self.base_env.reset(testing=testing, taller=taller, initial_design=initial_design)
        return self.current_structure
    
    def step(self, primitive_action: int, option_terminated_by_policy: bool) -> Tuple[Any, float, bool, Dict]:
        """
        Execute a primitive action within the current option.
        
        Key logic:
        1. Check if action is valid
        2. If option should continue: execute action but don't check constraints
        3. If option should terminate: execute action sequence and check constraints
        
        Args:
            primitive_action: The primitive action to execute
            option_terminated_by_policy: Whether the policy decides to terminate the option
            
        Returns:
            Tuple of (next_observation, reward, episode_done, info)
        """
        self.total_primitive_actions += 1
        
        # 1. Check if action is valid (not already at minimum section)
        if not self._is_action_valid(primitive_action):
            current_obs = self.current_structure
            return current_obs, -10, False, {
                "invalid_action": True, 
                "reason": "minimum_section",
                "option_completed": False
            }
        
        # 2. Add action to buffer
        self.action_buffer.append(primitive_action)
        self.option_step_count += 1
        
        # 3. Check forced termination conditions
        option_terminated = self._should_terminate_option(option_terminated_by_policy)
        
        if option_terminated:
            # 4. Execute complete option sequence and get final result
            final_structure, total_reward, episode_done, info = self._execute_option_sequence()
            self._reset_option_state()
            self.option_count += 1
            
            # Add option completion info
            info.update({
                "option_completed": True,
                "option_length": len(self.action_buffer) if self.action_buffer else self.option_step_count,
                "option_count": self.option_count
            })
            
            return final_structure, total_reward, episode_done, info
            
        else:
            # 5. Option continues - just update structure without full analysis
            # For intermediate steps, we only update structure to get new observation
            temp_structure = copy.deepcopy(self.current_structure)
            material_saved = temp_structure.update_action(primitive_action)
            
            # Store intermediate reward for later aggregation
            self.reward_buffer.append(material_saved * 0.1)  # Small intermediate reward
            
            # Update current structure for next step
            self.current_structure = temp_structure
            
            info = {
                "option_completed": False,
                "option_step": self.option_step_count,
                "material_saved": material_saved,
                "intermediate_reward": material_saved * 0.1,
                "actions_buffered": len(self.action_buffer)
            }
            
            return self.current_structure, material_saved * 0.1, False, info
    
    def _is_action_valid(self, action: int) -> bool:
        """Check if the action is valid using existing structure logic."""
        if action >= len(self.current_structure.story_level_sections):
            return False
        return self.current_structure.story_level_sections[action] > 0
    
    def _should_terminate_option(self, policy_termination: bool) -> bool:
        """
        Decide if option should terminate based on:
        1. Policy decision
        2. Maximum option length reached
        3. Minimum section reached (forced termination)
        """
        # Check if minimum section reached (forced termination)
        minimum_reached = sum(self.current_structure.story_level_sections) == 0
        
        # Check if maximum length reached
        max_length_reached = self.option_step_count >= self.max_option_length
        
        return policy_termination or minimum_reached or max_length_reached
    
    def _execute_option_sequence(self) -> Tuple[Any, float, bool, Dict]:
        """
        Execute the complete option sequence using base environment's step method.
        This is where we actually perform structural analysis and constraint checking.
        """
        if not self.action_buffer:
            # Empty option - shouldn't happen but handle gracefully
            return self.current_structure, 0.0, False, {"reason": "empty_option"}
        
        # Start with a fresh copy of the structure from before this option
        working_structure = copy.deepcopy(self.current_structure)
        
        # Execute each action in the buffer using base environment's step method
        total_reward = 0.0
        episode_done = False
        final_info = {}
        
        for i, action in enumerate(self.action_buffer):
            try:
                # Use base environment's full step method (includes all analysis)
                working_structure, step_reward, step_done, fail_name, fail_reason = \
                    self.base_env.step(working_structure, action)
                
                total_reward += step_reward
                
                if step_done:
                    # Episode terminated due to constraint failure or minimum section
                    episode_done = True
                    
                    if fail_reason == "minimum_section":
                        # Successfully reached minimum section
                        final_info = {
                            "reason": "minimum_section_reached",
                            "success": True,
                            "constraint_passed": True,
                            "actions_executed": i + 1,
                            "total_actions_planned": len(self.action_buffer)
                        }
                        # Add completion bonus
                        total_reward += 100
                        
                    else:
                        # Constraint violation
                        final_info = {
                            "reason": "constraint_violation", 
                            "fail_name": fail_name,
                            "fail_reason": fail_reason,
                            "constraint_passed": False,
                            "actions_executed": i + 1,
                            "total_actions_planned": len(self.action_buffer)
                        }
                        # Apply penalty as specified
                        total_reward = -1000
                    
                    break
                    
            except Exception as e:
                # Handle analysis errors
                print(f"Error executing action {action}: {e}")
                episode_done = True
                total_reward = -1000
                final_info = {
                    "reason": "execution_error",
                    "error": str(e),
                    "constraint_passed": False,
                    "actions_executed": i + 1,
                    "total_actions_planned": len(self.action_buffer)
                }
                break
        
        if not episode_done:
            # Option completed successfully without episode termination
            final_info = {
                "reason": "option_completed",
                "success": True,
                "constraint_passed": True,
                "actions_executed": len(self.action_buffer),
                "total_actions_planned": len(self.action_buffer)
            }
            # Small bonus for successful option completion
            total_reward += 10
        
        # Update current structure
        self.current_structure = working_structure
        
        return working_structure, total_reward, episode_done, final_info
    
    def _reset_option_state(self):
        """Reset option-specific state variables."""
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.option_step_count = 0
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions using existing structure logic."""
        if self.current_structure is None:
            return []
            
        valid_actions = []
        for i, section_level in enumerate(self.current_structure.story_level_sections):
            if section_level > 0:
                valid_actions.append(i)
        return valid_actions
    
    def get_option_info(self) -> Dict:
        """Get information about current option state."""
        return {
            "option_step": self.option_step_count,
            "max_option_length": self.max_option_length,
            "actions_in_buffer": len(self.action_buffer),
            "option_count": self.option_count,
            "total_primitive_actions": self.total_primitive_actions,
            "valid_actions": self.get_valid_actions(),
            "minimum_reached": sum(self.current_structure.story_level_sections) == 0 if self.current_structure else False,
            "current_sections": self.current_structure.story_level_sections.copy() if self.current_structure else []
        }
    
    def calculate_reward(self, whether_pass: bool) -> float:
        """Delegate to base environment's reward calculation."""
        return self.base_env.calculate_reward(whether_pass)
    
    def init_records(self, structure):
        """Delegate to base environment's record initialization."""
        return self.base_env.init_records(structure)
    
    # Expose base environment properties that might be needed
    @property
    def add_response_features(self):
        return self.base_env.add_response_features
    
    @property
    def structure(self):
        return self.current_structure
    
    @property 
    def reward_type(self):
        return self.base_env.reward_type
    
    @property
    def scwb_driven_design(self):
        return self.base_env.scwb_driven_design
    
    @property
    def do_nonlinear_dynamic_analysis(self):
        return self.base_env.do_nonlinear_dynamic_analysis
    
    @property 
    def check_displacement(self):
        return self.base_env.check_displacement
