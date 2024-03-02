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


class Environment:
    def __init__(self, 
                 structure_shape: str,
                 add_structure_geometry: bool,
                 reward_type: str,
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
        self.code_analysis_dir.mkdir(parents=True, exist_ok=True)        
        self.modal_analysis_dir.mkdir(parents=True, exist_ok=True)        

        # the prescribed, test ge
        self._testing_structure = None
        self.saved_material_record = None
        self.material_usage_record = None
        self.acc_record, self.disp_record = None, None
    
        # initiaization
        self._init_testing_structure()
        self.init_check_setting(check_acceleration, check_displacement)
        
    
    def _init_testing_structure(self):
        """Initialize the prescribed structure."""
        if self.structure_shape in ["fixed", "small_random"]:
            x_span_num = 3
            z_span_num = 3
            x_span_lens = [7000, 11000, 14000]
            z_span_lens = [12000, 8000, 10000]
            story_num = 3
            story_height = 3500
        elif self.structure_shape == "random":
            x_span_num = 3
            z_span_num = 3
            x_span_len = 6000
            z_span_len = 8000
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]
            story_num = 5
            story_height = 3200
        self._testing_structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                            "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                            "story_num": story_num, "story_height": story_height,
                            "add_structure_geometry": self.add_structure_geometry, 
                            "do_nonlinear_dynamic_analysis": self.do_nonlinear_dynamic_analysis,
                            "nda_norm_dict": self.nda_norm_dict,
                            "analysis_dir": self.modal_analysis_dir}
        self._testing_structure = structure.Structure(**self._testing_structure_kwargs)


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
        self.saved_material_record = []
        self.material_usage_record = [structure.calculate_material_usage()]

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




    def reset(self, testing=False, taller=False) -> structure.Structure:
        """Return a random generated structure."""
        if testing:
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
            
        structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                            "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                            "story_num": story_num, "story_height": story_height,
                            "add_structure_geometry": self.add_structure_geometry,
                            "do_nonlinear_dynamic_analysis": self.do_nonlinear_dynamic_analysis, 
                            "nda_norm_dict": self.nda_norm_dict,
                            "analysis_dir": self.modal_analysis_dir}
        random_structure = structure.Structure(**structure_kwargs)

        if structure_kwargs == self._testing_structure_kwargs:
            random_structure = self.reset(testing=testing, taller=taller)
        else:    
            self.logger.info(random_structure)
            self.init_records(random_structure)

        return random_structure
    
    
    def calculate_reward(self, material_saved: float) -> float:
        """Combine various target into total reward"""
        reward = 0.0
        if "material" in self.reward_type:
            print(f"saved_material_record len: {len(self.saved_material_record)}")
            #print(f"material saved: {material_saved} m3")
            #print(f"material usage difference: {self.material_usage_record[-2] - self.material_usage_record[-1]} m3")
            
            # normalized reward: decrement / initial amount
            if "normalized" in self.reward_type:
                reward += (material_saved / self.material_usage_record[0])
            else:
                reward += material_saved
            
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

        return reward


    def step(self, structure: structure.Structure, action: int) -> typing.Tuple[structure.Structure, float, bool]:
        """
        1. Based on the member action, update the structure & graph.
        2. Based on the analysis result, see whether meets the code.
        3. Return [updated structure, reward, whether meet terminal state, fail load name, fail reason].
        """

        # 1. update structure and graph and get saved material amount (unit: m^3)
        material_saved = structure.update_action(action)
        material_usage = structure.calculate_material_usage()

        # 2. record the information after updating structure and graph
        self.saved_material_record.append(material_saved)
        self.material_usage_record.append(material_usage)
        
        if self.do_nonlinear_dynamic_analysis and "acceleration" in self.reward_type:
            self.acc_record = check_nda.record_acc(structure, 
                                                   self.nda_simulator, 
                                                   self.MCE_ground_motion_set, 
                                                   self.nda_norm_dict, 
                                                   self.device,
                                                   self.acc_record,
                                                   self.logger)

        # 3. calculate the total reward
        reward = self.calculate_reward(material_saved)
        
        # 4. check if linear static analysis response pass regulation
        whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, 
                                                                             self.check_displacement,
                                                                             self.code_analysis_dir,
                                                                             self.logger)

        # 5. check if nonlinear dynamic analysis response pass regulation if needed
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

        done = True if whether_pass == False else False

        return structure, reward, done, fail_name, fail_reason
        
        
        

