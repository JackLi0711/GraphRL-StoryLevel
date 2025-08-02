import os
import torch
import typing
import logging
import numpy as np

from pathlib import Path
from copy import deepcopy

from RL import new_strategy
from Structure import structure, check, check_nda


class Environment:
    def __init__(self, 
                 structure_shape: str,
                 restrict_action: bool,
                 add_structure_geometry: bool,
                 add_response_features: bool,
                 reward_type: str,
                 scwb_driven_design: bool,
                 do_nonlinear_dynamic_analysis: bool,
                 check_acceleration: bool,
                 check_displacement: bool,
                 nda_simulator: torch.nn.Module,
                 nda_norm_dict: dict,
                 DBE_ground_motion_set: list[torch.Tensor],
                 MCE_ground_motion_set: list[torch.Tensor],
                 checkpoint_dir: Path,
                 logger: logging.Logger,
                 device: torch.device) -> None:
        
        self.structure_shape = structure_shape
        self.restrict_action = restrict_action
        self.add_structure_geometry = add_structure_geometry
        self.add_response_features = add_response_features
        self.reward_type = reward_type
        self.scwb_driven_design = scwb_driven_design
        self.do_nonlinear_dynamic_analysis = do_nonlinear_dynamic_analysis

        self.nda_simulator = nda_simulator
        self.nda_norm_dict = nda_norm_dict
        self.DBE_ground_motion_set = DBE_ground_motion_set
        self.MCE_ground_motion_set = MCE_ground_motion_set
        self.logger = logger
        self.device = device
        
        # checkpoint path
        self.checkpoint_dir = checkpoint_dir
        self.code_analysis_dir = checkpoint_dir / "Code_Analysis"
        self.modal_analysis_dir = checkpoint_dir / "Modal_Analysis"
        self.model_dir = checkpoint_dir / "models"
        self.code_analysis_dir.mkdir(parents=True, exist_ok=True)        
        self.modal_analysis_dir.mkdir(parents=True, exist_ok=True)   
        self.model_dir.mkdir(parents=True, exist_ok=True)     

        # the prescribed, test generalization ability
        self._testing_structure = None
        
        # MuZero 動態動作空間支援
        self.current_num_actions = None
        self.muzero_agent = None  # 將在需要時設置
    
        # initiaization
        self._init_testing_structure()
        self.init_check_setting(check_acceleration, check_displacement)
        
    
    def _init_testing_structure(self, initial_design=None):
        """Initialize the prescribed structure."""
        if self.structure_shape in ["fixed"]:
            # x_span_num = 4 # 4, 6
            # z_span_num = 4 # 4, 6
            # x_span_len = 7000
            # z_span_len = 7000
            # x_span_lens = [x_span_len for i in range(x_span_num)]
            # z_span_lens = [z_span_len for i in range(z_span_num)]
            # story_num = 4 # 4, 7 
            # story_height = 3200

            x_span_num = 5 # 4, 6
            z_span_num = 5 # 4, 6
            x_span_len = 7000
            z_span_len = 7000
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]
            story_num = 5 # 4, 7 
            story_height = 3200
        
        elif self.structure_shape == "small_random":
            x_span_num = 3 # 4, 6
            z_span_num = 3 # 4, 6
            x_span_len = 7000
            z_span_len = 7000
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]
            story_num = 3 # 4, 7 
            story_height = 3200

        elif self.structure_shape == "random":
            # x_span_num = 4  # original: 3
            # z_span_num = 4  # original: 3
            # x_span_len = 6000
            # z_span_len = 8000
            # x_span_lens = [x_span_len for i in range(x_span_num)]
            # z_span_lens = [z_span_len for i in range(z_span_num)]
            # story_num = 6  # original: 5
            # story_height = 3200
            x_span_num = 4 # 4, 6
            z_span_num = 4 # 4, 6
            x_span_len = 7000
            z_span_len = 7000
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]
            story_num = 4 # 4, 7 
            story_height = 3200

        story_level_sections = initial_design if initial_design is not None else new_strategy.sample_initial_story_sections(x_span_num, x_span_len, z_span_num, z_span_len, story_num, thickest_prob=1.0)
        self._testing_structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                                          "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                                          "story_num": story_num, "story_height": story_height,
                                          "story_level_sections": story_level_sections, 
                                          "add_structure_geometry": self.add_structure_geometry, 
                                          "add_response_features": self.add_response_features, 
                                          "do_nonlinear_dynamic_analysis": self.do_nonlinear_dynamic_analysis,
                                          "nda_norm_dict": self.nda_norm_dict,
                                          "analysis_dir": self.modal_analysis_dir}
        self._testing_structure = structure.Structure(**self._testing_structure_kwargs)
        
        # update beam sections based on strong-column-weak-beam principle
        self.logger.info(f"before_SCWB_update, testing_story_level_sections: {self._testing_structure.story_level_sections}")
        if self.scwb_driven_design:
            new_strategy.strong_column_weak_beam_driven_update(self._testing_structure, self.code_analysis_dir)
        self.logger.info(f"after_SCWB_update, testing_story_level_sections: {self._testing_structure.story_level_sections}\n")
        
        # 設置動作空間
        self._update_action_space(self._testing_structure)


    def init_check_setting(self, check_acc: bool, check_disp: bool):
        """Reset whether to check the constraints according to the reward type."""
        self.check_acceleration = check_acc
        self.check_displacement = check_disp
        if 'acceleration' in self.reward_type:
            self.check_acceleration = False
        if 'displacement' in self.reward_type:
            self.check_displacement = False


    def init_records(self, structure: structure.Structure):
        """Initialize various records corresponding to different reward types."""
        # material usage
        self.saved_material_record = []
        self.saved_material_record_SCWB = []
        self.material_usage_record = [structure.calculate_material_usage()]

        # action
        self.update_actions_record = []
        self.update_actions_record_SCWB = []

        # static response: max stress ratio, min stress ratio, max drift ratio, min SCWB ratio (all normalized by limit)
        load_cases, static_responses = check.get_response(structure, self.code_analysis_dir)
        _, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
        self.static_response_record = [list(static_response_rewards.values())]

        # dynamic response: max drift ratio, max plastic hinge occurrence (all normalized by limit)
        dynamic_response_features = None
        if self.do_nonlinear_dynamic_analysis:
            structure.init_graph_GraphLSTM()
            dynamic_responses = check_nda.get_response(structure, self.nda_simulator, self.MCE_ground_motion_set, self.device)
            _, dynamic_response_features, dynamic_response_rewards = check_nda.process_response(structure, dynamic_responses, self.nda_norm_dict)
            self.dynamic_response_record = [list(dynamic_response_rewards.values())]
            
            if "acceleration" in self.reward_type:
                self.acc_record = {'X-dir': [], 'Z-dir': []}
                self.acc_record = check_nda.record_acc(structure, 
                                                       self.nda_simulator, 
                                                       self.MCE_ground_motion_set, 
                                                       self.nda_norm_dict, 
                                                       self.device,
                                                       self.acc_record,
                                                       self.logger)
            if "displacement" in self.reward_type:
                self.disp_record = {'X-dir': [], 'Z-dir': []}

        structure.init_graph_GraphRL(static_response_features, dynamic_response_features)

        # reward 
        self.reward_record = []


    def reset(self, testing=False, taller=False, initial_design=None) -> structure.Structure:
        """Return a random generated structure."""
        if testing:
            # increase the variety of initial design for testing structure to prove model's capability
            self._init_testing_structure(initial_design)
            self.init_records(self._testing_structure)
            return deepcopy(self._testing_structure)
        elif taller:
            x_span_num = 3
            z_span_num = 3
            x_span_len = 6000
            z_span_len = 8000
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]
            story_num = 8
            story_height = 3200
        else:
            if self.structure_shape == "fixed":
                # x_span_num = 4 # 4, 6
                # z_span_num = 4 # 4, 6
                # x_span_len = 7000
                # z_span_len = 7000
                # x_span_lens = [x_span_len for i in range(x_span_num)]
                # z_span_lens = [z_span_len for i in range(z_span_num)]
                # story_num = 4 # 4, 7 
                # story_height = 3200

                x_span_num = 5 # 4, 6
                z_span_num = 5 # 4, 6
                x_span_len = 7000
                z_span_len = 7000
                x_span_lens = [x_span_len for i in range(x_span_num)]
                z_span_lens = [z_span_len for i in range(z_span_num)]
                story_num = 5 # 4, 7 
                story_height = 3200


            elif self.structure_shape == "small_random":
                x_span_num = np.random.randint(2, 5)
                z_span_num = np.random.randint(2, 5)
                x_span_lens = [np.random.randint(5, 15) * 1000 for i in range(x_span_num)]
                z_span_lens = [np.random.randint(5, 15) * 1000 for i in range(z_span_num)]
                x_span_len = sum(x_span_lens) / len(x_span_lens)
                z_span_len = sum(z_span_lens) / len(z_span_lens)
                story_num = np.random.randint(2, 5)
                story_height = np.random.randint(0, 11) * 100 + 3000

            elif self.structure_shape == "random":
                x_span_num = np.random.randint(2, 7)
                z_span_num = np.random.randint(2, 7)
                x_span_len = np.random.randint(6, 9) * 1000
                z_span_len = np.random.randint(6, 9) * 1000
                x_span_lens = [x_span_len for i in range(x_span_num)]
                z_span_lens = [z_span_len for i in range(z_span_num)]
                story_num = np.random.randint(4, 8)
                story_height = 3200

        story_level_sections = initial_design if initial_design is not None else new_strategy.sample_initial_story_sections(x_span_num, x_span_len, z_span_num, z_span_len, story_num, thickest_prob=1.0)            
        structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                            "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                            "story_num": story_num, "story_height": story_height,
                            "story_level_sections": story_level_sections, 
                            "add_structure_geometry": self.add_structure_geometry,
                            "add_response_features": self.add_response_features,
                            "do_nonlinear_dynamic_analysis": self.do_nonlinear_dynamic_analysis, 
                            "nda_norm_dict": self.nda_norm_dict,
                            "analysis_dir": self.modal_analysis_dir}
        random_structure = structure.Structure(**structure_kwargs)

        # if list(structure_kwargs.values())[:6] == list(self._testing_structure_kwargs.values())[:6]:
        #     random_structure = self.reset(testing=testing, taller=taller)
        # else:    
        self.logger.info(random_structure)
        # update beam sections based on strong-column-weak-beam principle
        self.logger.info(f"before_SCWB_update, story_level_sections: {random_structure.story_level_sections}")
        if self.scwb_driven_design:
            new_strategy.strong_column_weak_beam_driven_update(random_structure, self.code_analysis_dir)
        self.logger.info(f"after_SCWB_update, story_level_sections: {random_structure.story_level_sections}")
        self.init_records(random_structure)
        
        # 設置動作空間
        self._update_action_space(random_structure)

        return random_structure
    
    
    def calculate_reward(self, whether_pass: bool) -> float:
        """Combine various target into total reward"""
        if whether_pass == True:
            volume_saved = self.saved_material_record[-1]
            volume_saved_SCWB = self.saved_material_record_SCWB[-1]
            if "material" in self.reward_type:
                # print(f"saved_material_record len: {len(self.saved_material_record)}")
                # print(f"{volume_saved = :.3f} m3")
                # print(f"{volume_saved_SCWB = :.3f} m3")
                # print(f"material usage difference: {(self.material_usage_record[-2] - self.material_usage_record[-1]):.3f} m3")
                reward = volume_saved
                if "total" in self.reward_type: reward += volume_saved_SCWB
                if "normalized" in self.reward_type: reward /= self.material_usage_record[0]

            if "combined" in self.reward_type:
                delta_v = volume_saved + volume_saved_SCWB
                stress_ratio_range = self.static_response_record[-1][0] - self.static_response_record[-1][1]  # max_stress_ratio - min_stress_ratio
                max_stress_ratio_reward = np.clip(self.static_response_record[-2][0]/self.static_response_record[-1][0], 0.0, 0.99)  # max_stress_ratio_before / max_stress_ratio
                max_drift_ratio_reward = np.clip(self.static_response_record[-2][2]/self.static_response_record[-1][2], 0.0, 0.99)  # max_drift_ratio_before / max_drift_ratio
                min_scwb_ratio_reward = np.clip(self.static_response_record[-2][3]/self.static_response_record[-1][3], 0.0, 0.99)  # min_scwb_ratio_before / min_scwb_ratio
                reward = 0.1 * delta_v**0.5 / stress_ratio_range * -(np.log(1-max_stress_ratio_reward) + np.log(1-max_drift_ratio_reward))
                
            if "acceleration" in self.reward_type:
                acc_record_x = np.array(self.acc_record['X-dir'])
                acc_record_z = np.array(self.acc_record['Z-dir'])
                print(f"acc_record shape: {acc_record_x.shape}")

                # normalized reward: decrement / initial amount
                acc_decrement_x = np.sum(acc_record_x[-2, :] - acc_record_x[-1, :])
                acc_decrement_z = np.sum(acc_record_z[-2, :] - acc_record_z[-1, :])
                if "normalized" in self.reward_type:
                    reward = (acc_decrement_x / np.sum(acc_record_x[0, :]) + acc_decrement_z / np.sum(acc_record_z[0, :]))
                else:
                    reward = (acc_decrement_x + acc_decrement_z)
        else: 
            reward = 0.0

        self.reward_record.append(reward)

        return reward


    def step(self, structure: structure.Structure, action: int) -> typing.Tuple[structure.Structure, float, bool, str, str]:
        """
        1. Based on the member action, update the structure & graph.
        2. Based on the analysis result, see whether meets the code.
        3. Return [updated structure, reward, whether meet terminal state, fail load name, fail reason].
        """
        if action is None:
            # This case is for MCTS simulation to get the final reward of a terminal state.
            # No action is taken, just calculate the reward for the current state.
            passed, fail_name, fail_reason = self._check_design_feasibility(structure), "Terminal", "Terminal"
            reward = self.calculate_reward(passed)
            return deepcopy(structure), reward, True, fail_name, fail_reason

        # 1-1. 將固定action space的action映射到結構實際的action index
        mapped_action = self.map_action_to_structure_index(action, structure)
        # update structure, graph and get saved material amount(m^3) (ORIGINAL)
        material_saved = structure.update_action(mapped_action)
        before_SCWB_structure = deepcopy(structure)
        # 1-2. update structure, graph and get saved material amount(m^3) (STRONG-COLUMN-WEAK-BEAM)
        if self.scwb_driven_design:
            material_saved_SCWB, update_actions_SCWB, auxiliary_values, load_cases, static_responses = new_strategy.strong_column_weak_beam_driven_update(structure, self.code_analysis_dir, self.logger)
            if material_saved_SCWB != 0:
                print(f"before_SCWB_update, story_level_sections: {before_SCWB_structure.story_level_sections}")
                print(f"after_SCWB_update,  story_level_sections: {structure.story_level_sections}")
        else:
            material_saved_SCWB = 0
            update_actions_SCWB = []
            load_cases, static_responses = check.get_response(structure, self.code_analysis_dir)
        
        # 2-1. linear static analysis: check if response pass constraints
        static_constraint_condition, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
        whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, self.check_displacement)
        # 2-2. nonlinear dynamic analysis: check if response pass constraints
        dynamic_response_features = None
        dynamic_response_rewards = None
        if self.do_nonlinear_dynamic_analysis and whether_pass == True:
            structure.update_graph_GraphLSTM()
            dynamic_responses = check_nda.get_response(structure, self.nda_simulator, self.MCE_ground_motion_set, self.device)
            dynamic_constraint_condition, dynamic_response_features, dynamic_response_rewards = check_nda.process_response(structure, dynamic_responses, self.nda_norm_dict)
            whether_pass, fail_name, fail_reason = check_nda.check_pass(dynamic_constraint_condition, self.check_displacement)

        structure.update_graph_GraphRL(static_response_features, dynamic_response_features)

        # 3. record all information after updating and checking
        # material usage
        self.saved_material_record.append(material_saved)
        self.saved_material_record_SCWB.append(material_saved_SCWB)
        self.material_usage_record.append(structure.calculate_material_usage())
        # action
        self.update_actions_record.append(action)
        self.update_actions_record_SCWB.append(update_actions_SCWB)
        # static response
        self.static_response_record.append(list(static_response_rewards.values()))
        # dynamic response
        if self.do_nonlinear_dynamic_analysis:
            self.dynamic_response_record.append(list(dynamic_response_rewards.values())) if dynamic_response_rewards is not None else None
            if "acceleration" in self.reward_type:
                self.acc_record = check_nda.record_acc(structure, 
                                                       self.nda_simulator, 
                                                       self.MCE_ground_motion_set, 
                                                       self.nda_norm_dict, 
                                                       self.device,
                                                       self.acc_record,
                                                       self.logger)
        
        # 4. calculate reward based on recorded information
        reward = self.calculate_reward(whether_pass)

        # 5. make proper adjustments if the structure meets terminal state
        if whether_pass == False:
            # fail constraints
            done = True
            self.saved_material_record.pop(-1)
            self.saved_material_record_SCWB.pop(-1)
            self.material_usage_record.pop(-1)
            self.update_actions_record.pop(-1)
            self.update_actions_record_SCWB.pop(-1)
            self.static_response_record.pop(-1)
            self.dynamic_response_record.pop(-1) if self.do_nonlinear_dynamic_analysis else None
        elif whether_pass == True and sum(structure.story_level_sections) == 0:  
            # pass all constraints & already has minimum sections
            done = True
            fail_name = None
            fail_reason = "minimum_section"
        else:
            # pass all constraints & still has sections to reduce
            done = False
        
        if done:
            self.fail_name = fail_name
            self.fail_reason = fail_reason

        return structure, reward, done, fail_name, fail_reason
	

    def clone(self):
        """Creates a deep copy of the environment for MCTS simulation."""
        # The logger can't be deep-copied, so we handle it manually.
        logger = self.logger
        self.logger = None
        cloned = deepcopy(self)
        cloned.logger = logger
        self.logger = logger
        return cloned

    
    def get_legal_actions(self, structure_obj) -> typing.List[int]:
        """
        Returns a list of all valid actions from the given structure state,
        aligned with the DeepQAgent's action space and restriction rules.

        An action is an integer representing a story-level member group to modify
        (e.g., 1F outer columns, 3F x-direction beams).

        An action is illegal if:
        1. The member group is already at its minimum section size.
        2. The action would violate structural hierarchy (e.g., making a lower
           column weaker than an upper column).
        3. The action corresponds to a story that doesn't exist in the current structure.
        """
        # 步驟 1: 使用固定的最大 Action Space (設定最大樓層數)
        max_story_num = self._get_max_story_num()
        max_num_actions = max_story_num * 4  # 4種類型：xdir_beam, zdir_beam, outer_column, inner_column
        all_actions = list(range(max_num_actions))

        # 步驟 2: 計算所有不合法的 actions
        illegal_actions = set()
        
        # 規則一: 最小斷面規則 (需要映射到固定action space)
        illegal_min_section_structure = structure_obj.already_minimum_section_story_indexes
        illegal_min_section = [self.map_structure_index_to_action(idx, structure_obj) 
                              for idx in illegal_min_section_structure]
        illegal_actions.update(illegal_min_section)
        
        # 規則二: 結構層級規則 (需要映射到固定action space)
        illegal_hierarchy_structure = structure_obj.restrict_action_space() if self.restrict_action else []
        illegal_hierarchy = [self.map_structure_index_to_action(idx, structure_obj) 
                           for idx in illegal_hierarchy_structure]
        illegal_actions.update(illegal_hierarchy)
        
        # 規則三: 樓層不存在的規則 (新增)
        current_story_num = structure_obj.story_num
        illegal_story_actions = self._get_illegal_story_actions(current_story_num, max_story_num)
        illegal_actions.update(illegal_story_actions)
        
        # 步驟 3: 從所有 actions 中排除不合法的，得到最終的合法 actions
        legal_actions = [a for a in all_actions if a not in illegal_actions]
        
        return legal_actions
    
    def _get_max_story_num(self) -> int:
        """
        根據structure_shape決定最大樓層數
        """
        if self.structure_shape == "fixed":
            return 5  # 根據_init_testing_structure中的設定
        elif self.structure_shape == "small_random":
            return 4  # 根據reset方法中的np.random.randint(2, 5)，最大是4
        elif self.structure_shape == "random":
            return 7  # 根據reset方法中的np.random.randint(4, 8)，最大是7
        else:
            return 8  # 默認最大值，包含taller情況
    
    def _get_illegal_story_actions(self, current_story_num: int, max_story_num: int) -> typing.List[int]:
        """
        計算不屬於當前樓層數的action indices
        
        Args:
            current_story_num: 當前結構的樓層數
            max_story_num: 最大可能的樓層數
        
        Returns:
            illegal_actions: 不屬於當前樓層數的action indices列表
        """
        illegal_actions = []
        
        # 計算每種類型的action數量
        actions_per_type = max_story_num  # 每種類型最多有max_story_num個action
        
        # 對於每種類型，將超出當前樓層數的action標記為illegal
        for action_type in range(4):  # 0: xdir_beam, 1: zdir_beam, 2: outer_column, 3: inner_column
            start_idx = action_type * actions_per_type
            # 將超出當前樓層數的action加入illegal list
            for story in range(current_story_num, max_story_num):
                illegal_action_idx = start_idx + story
                illegal_actions.append(illegal_action_idx)
        
        return illegal_actions
    
    def map_action_to_structure_index(self, action_idx: int, structure_obj) -> int:
        """
        將固定action space中的action index映射到結構的實際action index
        
        Args:
            action_idx: 固定action space中的action index
            structure_obj: 結構物件
            
        Returns:
            mapped_idx: 結構中實際的action index
        """
        max_story_num = self._get_max_story_num()
        current_story_num = structure_obj.story_num
        
        # 確定action類型和樓層
        action_type = action_idx // max_story_num
        story_in_type = action_idx % max_story_num
        
        # 檢查是否為有效的樓層
        if story_in_type >= current_story_num:
            raise ValueError(f"Action {action_idx} refers to story {story_in_type + 1} but structure only has {current_story_num} stories")
        
        # 映射到結構的實際index
        if action_type == 0:  # xdir_beam
            mapped_idx = story_in_type
        elif action_type == 1:  # zdir_beam  
            mapped_idx = current_story_num + story_in_type
        elif action_type == 2:  # outer_column
            mapped_idx = 2 * current_story_num + story_in_type
        elif action_type == 3:  # inner_column
            mapped_idx = 3 * current_story_num + story_in_type
        else:
            raise ValueError(f"Invalid action type {action_type}")
            
        return mapped_idx
    
    def get_action_info(self, action_idx: int, structure_obj) -> typing.Tuple[str, int]:
        """
        根據固定action space中的action index獲取action信息
        
        Args:
            action_idx: 固定action space中的action index
            structure_obj: 結構物件
            
        Returns:
            tuple: (member_category, story_number)
        """
        max_story_num = self._get_max_story_num()
        
        # 確定action類型和樓層
        action_type = action_idx // max_story_num
        story_in_type = action_idx % max_story_num
        
        # 獲取類型名稱
        type_names = ['xdir_beam', 'zdir_beam', 'outer_column', 'inner_column']
        member_category = type_names[action_type]
        
        # 樓層編號（從1開始）
        story_number = story_in_type + 1
        
        return member_category, story_number
    
    def map_structure_index_to_action(self, structure_idx: int, structure_obj) -> int:
        """
        將結構的實際action index映射到固定action space中的action index
        
        Args:
            structure_idx: 結構中實際的action index
            structure_obj: 結構物件
            
        Returns:
            mapped_idx: 固定action space中的action index
        """
        current_story_num = structure_obj.story_num
        max_story_num = self._get_max_story_num()
        
        # 確定在結構中的類型和樓層
        if structure_idx < current_story_num:
            # xdir_beam
            action_type = 0
            story_in_type = structure_idx
        elif structure_idx < 2 * current_story_num:
            # zdir_beam
            action_type = 1
            story_in_type = structure_idx - current_story_num
        elif structure_idx < 3 * current_story_num:
            # outer_column
            action_type = 2
            story_in_type = structure_idx - 2 * current_story_num
        elif structure_idx < 4 * current_story_num:
            # inner_column
            action_type = 3
            story_in_type = structure_idx - 3 * current_story_num
        else:
            raise ValueError(f"Invalid structure action index {structure_idx}")
        
        # 映射到固定action space
        mapped_idx = action_type * max_story_num + story_in_type
        
        return mapped_idx
    
    def _update_action_space(self, structure_obj):
        """更新動作空間資訊並通知 MuZero Agent"""
        # 使用固定的最大action數量，而不是依賴結構的實際action數量
        max_story_num = self._get_max_story_num()
        self.current_num_actions = max_story_num * 4  # 固定action space大小
        
        # 通知 MuZero Agent 動作空間變化
        if self.muzero_agent is not None:
            self.muzero_agent.set_action_space(self.current_num_actions)
        
        self.logger.info(f"Action space updated: {self.current_num_actions} actions (max_story_num={max_story_num}) for structure_shape={self.structure_shape}, current_story_num={structure_obj.story_num}")
    
    def set_muzero_agent(self, agent):
        """設置 MuZero Agent 的引用"""
        self.muzero_agent = agent
        if self.current_num_actions is not None:
            agent.set_action_space(self.current_num_actions)
    
    def get_action_space_info(self):
        """返回當前動作空間資訊"""
        return {
            'num_actions': self.current_num_actions,
            'structure_shape': self.structure_shape
        }

    def _check_design_feasibility(self, structure_obj):
        """
        Helper function to run the full design check process.
        Returns True if the design passes, False otherwise.
        """
        from Structure import check, load
        try:
            load_cases, responses = check.get_response(structure_obj, self.code_analysis_dir)
            constraint_condition, _, _ = check.process_response(structure_obj, load_cases, responses)
            passed, _, _ = check.check_pass(load_cases, constraint_condition, self.check_displacement)
            return passed
        except Exception as e:
            # If any error occurs during opensees analysis, consider it a failure.
            self.logger.error(f"Error during feasibility check: {e}")
            return False

    def is_terminal(self, structure_obj):
        """
        Checks if a given structure state is terminal.
        A state is terminal if the design is infeasible (fails checks).
        """
        passed = self._check_design_feasibility(structure_obj)
        return not passed
        

