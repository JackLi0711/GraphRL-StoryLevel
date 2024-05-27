import os
import torch
import typing
import logging
import numpy as np

from typing import List, Dict
from pathlib import Path
from copy import deepcopy

from Structure import check
from Structure import check_nda
from Structure import structure
from RL import new_strategy


class Environment:
    def __init__(self, 
                 structure_shape: str,
                 add_structure_geometry: bool,
                 reward_type: str,
                 scwb_driven_design: bool,
                 do_nonlinear_dynamic_analysis: bool,
                 check_acceleration: bool,
                 check_displacement: bool,
                 nda_simulator: torch.nn.Module,
                 nda_norm_dict: dict,
                 DBE_ground_motion_set: List[torch.Tensor],
                 MCE_ground_motion_set: List[torch.Tensor],
                 checkpoint_dir: Path,
                 logger: logging.Logger,
                 device = torch.device) -> None:
        
        self.structure_shape = structure_shape
        self.add_structure_geometry = add_structure_geometry
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
    
        # initiaization
        self._init_testing_structure()
        self.init_check_setting(check_acceleration, check_displacement)
        
    
    def _init_testing_structure(self, initial_design=None):
        """Initialize the prescribed structure."""
        if self.structure_shape in ["fixed", "small_random"]:
            x_span_num = 3
            z_span_num = 3
            x_span_lens = [7000, 11000, 14000]
            z_span_lens = [12000, 8000, 10000]
            story_num = 3
            story_height = 3500
        elif self.structure_shape == "random":
            x_span_num = 4  # original: 3
            z_span_num = 4  # original: 3
            x_span_len = 6000
            z_span_len = 8000
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]
            story_num = 6  # original: 5
            story_height = 3200

        story_level_sections = initial_design if initial_design is not None else new_strategy.sample_initial_story_sections(x_span_num, x_span_len, z_span_num, z_span_len, story_num, thickest_prob=1.0)
        self._testing_structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                                          "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                                          "story_num": story_num, "story_height": story_height,
                                          "story_level_sections": story_level_sections, 
                                          "add_structure_geometry": self.add_structure_geometry, 
                                          "do_nonlinear_dynamic_analysis": self.do_nonlinear_dynamic_analysis,
                                          "nda_norm_dict": self.nda_norm_dict,
                                          "analysis_dir": self.modal_analysis_dir}
        self._testing_structure = structure.Structure(**self._testing_structure_kwargs)
        
        # update beam sections based on strong-column-weak-beam principle
        self.logger.info(f"before_SCWB_update, testing_story_level_sections: {self._testing_structure.story_level_sections}")
        if self.scwb_driven_design:
            new_strategy.strong_column_weak_beam_driven_update(self._testing_structure, self.code_analysis_dir)
        self.logger.info(f"after_SCWB_update, testing_story_level_sections: {self._testing_structure.story_level_sections}\n")


    def init_check_setting(self, check_acc: bool, check_disp: bool):
        """Reset whether to check the constraints according to the reward type"""
        self.check_acceleration = check_acc
        self.check_displacement = check_disp
        if 'acceleration' in self.reward_type:
            self.check_acceleration = False
        if 'displacement' in self.reward_type:
            self.check_displacement = False


    def init_records(self, structure: structure.Structure):
        """Initialize various records corresponding to different reward types"""
        # material usage
        self.saved_material_record = []
        self.saved_material_record_SCWB = []
        self.material_usage_record = [structure.calculate_material_usage()]

        # action
        self.update_actions_record = []
        self.update_actions_record_SCWB = []

        # static response: max stress ratio, min stress ratio, max drift ratio, min SCWB ratio (all normalized by limit)
        _, load_cases, responses = check.get_response(structure, self.code_analysis_dir)
        _, static_response = check.check_response(structure, load_cases, responses)
        self.static_response_record = [static_response.tolist()]

        # dynamic response: acc, disp
        if self.do_nonlinear_dynamic_analysis and "acceleration" in self.reward_type:
            self.acc_record = {'X-dir': [], 'Z-dir': []}
            self.acc_record = check_nda.record_acc(structure, 
                                                   self.nda_simulator, 
                                                   self.MCE_ground_motion_set, 
                                                   self.nda_norm_dict, 
                                                   self.device,
                                                   self.acc_record,
                                                   self.logger)
        self.disp_record = {'X-dir': [], 'Z-dir': []}

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
                x_span_num = 3
                z_span_num = 3
                x_span_lens = [7000, 11000, 14000]
                z_span_lens = [12000, 8000, 10000]
                story_num = 3
                story_height = np.random.randint(0, 11) * 100 + 3000

            elif self.structure_shape == "small_random":
                x_span_num = np.random.randint(2, 5)
                z_span_num = np.random.randint(2, 5)
                x_span_lens = [np.random.randint(5, 15) * 1000 for i in range(x_span_num)]
                z_span_lens = [np.random.randint(5, 15) * 1000 for i in range(z_span_num)]
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
                            "do_nonlinear_dynamic_analysis": self.do_nonlinear_dynamic_analysis, 
                            "nda_norm_dict": self.nda_norm_dict,
                            "analysis_dir": self.modal_analysis_dir}
        random_structure = structure.Structure(**structure_kwargs)

        if list(structure_kwargs.values())[:6] == list(self._testing_structure_kwargs.values())[:6]:
            random_structure = self.reset(testing=testing, taller=taller)
        else:    
            self.logger.info(random_structure)
            # update beam sections based on strong-column-weak-beam principle
            self.logger.info(f"before_SCWB_update, story_level_sections: {random_structure.story_level_sections}")
            if self.scwb_driven_design:
                new_strategy.strong_column_weak_beam_driven_update(random_structure, self.code_analysis_dir)
            self.logger.info(f"after_SCWB_update, story_level_sections: {random_structure.story_level_sections}")
            self.init_records(random_structure)

        return random_structure
    
    
    def calculate_reward(self) -> float:
        """Combine various target into total reward"""
        reward = 0
        volume_saved = self.saved_material_record[-1]
        volume_saved_SCWB = self.saved_material_record_SCWB[-1]
        if "material" in self.reward_type:
            print(f"saved_material_record len: {len(self.saved_material_record)}")
            print(f"{volume_saved = :.3f} m3")
            print(f"{volume_saved_SCWB = :.3f} m3")
            print(f"material usage difference: {(self.material_usage_record[-2] - self.material_usage_record[-1]):.3f} m3")
            
            reward += volume_saved
            if "total" in self.reward_type: reward += volume_saved_SCWB
            if "normalized" in self.reward_type: reward /= self.material_usage_record[0]

        if "combined" in self.reward_type:
            delta_v = volume_saved + volume_saved_SCWB
            stress_ratio_range = self.static_response_record[-1][0] - self.static_response_record[-1][1]  # max_stress_ratio - min_stress_ratio
            max_stress_ratio_reward = np.clip(self.static_response_record[-2][0]/self.static_response_record[-1][0], 0.0, 0.99)  # max_stress_ratio_before / max_stress_ratio
            max_drift_ratio_reward = np.clip(self.static_response_record[-2][2]/self.static_response_record[-1][2], 0.0, 0.99)  # max_drift_ratio_before / max_drift_ratio
            min_scwb_ratio_reward = np.clip(self.static_response_record[-2][3]/self.static_response_record[-1][3], 0.0, 0.99)  # min_scwb_ratio_before / min_scwb_ratio

            reward += 0.1 * delta_v**0.5 / stress_ratio_range * -(np.log(1-max_stress_ratio_reward) + np.log(1-max_drift_ratio_reward))
            
        if "acceleration" in self.reward_type:
            acc_record_x = np.array(self.acc_record['X-dir'])
            acc_record_z = np.array(self.acc_record['Z-dir'])
            print(f"acc_record shape: {acc_record_x.shape}")

            # normalized reward: decrement / initial amount
            acc_decrement_x = np.sum(acc_record_x[-2, :] - acc_record_x[-1, :])
            acc_decrement_z = np.sum(acc_record_z[-2, :] - acc_record_z[-1, :])
            if "normalized" in self.reward_type:
                reward += (acc_decrement_x / np.sum(acc_record_x[0, :]) + acc_decrement_z / np.sum(acc_record_z[0, :]))
            else:
                reward += (acc_decrement_x + acc_decrement_z)

        self.reward_record.append(reward)

        return reward


    def step(self, structure: structure.Structure, action: int) -> typing.Tuple[structure.Structure, float, bool, str, str]:
        """
        1. Based on the member action, update the structure & graph.
        2. Based on the analysis result, see whether meets the code.
        3. Return [updated structure, reward, whether meet terminal state, fail load name, fail reason].
        """

        # 1-1. update structure, graph and get saved material amount(m^3) (ORIGINAL)
        material_saved = structure.update_action(action)
        before_SCWB_structure = deepcopy(structure)

        # 1-2. update structure, graph and get saved material amount(m^3) (STRONG-COLUMN-WEAK-BEAM)
        if self.scwb_driven_design:
            material_saved_SCWB, update_actions_SCWB, auxiliary_values, load_cases, responses = new_strategy.strong_column_weak_beam_driven_update(structure, self.code_analysis_dir, self.logger)
            if material_saved_SCWB != 0:
                print(f"before_SCWB_update, story_level_sections: {before_SCWB_structure.story_level_sections}")
                print(f"after_SCWB_update,  story_level_sections: {structure.story_level_sections}")
        else:
            material_saved_SCWB = 0
            update_actions_SCWB = []
            auxiliary_values, load_cases, responses = check.get_response(structure, self.code_analysis_dir)
        
        # 3. linear static analysis: check if response pass constraints
        constraint_condition, static_response = check.check_response(structure, load_cases, responses)
        whether_pass, fail_name, fail_reason = check.check_pass(load_cases, constraint_condition, self.check_displacement)

        # 4. nonlinear dynamic analysis: check if response pass constraints
        if self.do_nonlinear_dynamic_analysis and whether_pass == True:
            whether_pass, fail_reason = check_nda.check(structure, 
                                                        self.nda_simulator, 
                                                        self.DBE_ground_motion_set,
                                                        self.MCE_ground_motion_set, 
                                                        self.check_acceleration, 
                                                        self.check_displacement,
                                                        self.nda_norm_dict, 
                                                        auxiliary_values,
                                                        self.device,
                                                        self.logger)
            
        # 5. record all information after updating and checking
        # material usage
        self.saved_material_record.append(material_saved)
        self.saved_material_record_SCWB.append(material_saved_SCWB)
        self.material_usage_record.append(structure.calculate_material_usage())
        # action
        self.update_actions_record.append(action)
        self.update_actions_record_SCWB.append(update_actions_SCWB)
        # static response
        self.static_response_record.append(static_response.tolist())
        # dynamic response
        if self.do_nonlinear_dynamic_analysis and "acceleration" in self.reward_type:
            self.acc_record = check_nda.record_acc(structure, 
                                                   self.nda_simulator, 
                                                   self.MCE_ground_motion_set, 
                                                   self.nda_norm_dict, 
                                                   self.device,
                                                   self.acc_record,
                                                   self.logger)
        
        # 6. calculate reward based on recorded information
        reward = self.calculate_reward()

        # 7. make proper adjustments if structure meets terminal state
        if whether_pass == False:
            # fail constraints
            done = True
            self.saved_material_record.pop(-1)
            self.saved_material_record_SCWB.pop(-1)
            self.material_usage_record.pop(-1)
            self.update_actions_record.pop(-1)
            self.update_actions_record_SCWB.pop(-1)
            self.static_response_record.pop(-1)
            self.reward_record.pop(-1)
        elif whether_pass == True and sum(structure.story_level_sections) == 0:  
            # pass all constraints & already has minimum sections
            done = True
            fail_name = None
            fail_reason = "minimum_section"
        else:
            # pass all constraints & still has sections to reduce
            done = False
        
        if done:
            reward = -1
            self.reward_record.append(reward)
            self.fail_name = fail_name
            self.fail_reason = fail_reason

        return structure, reward, done, fail_name, fail_reason
        
        
        

