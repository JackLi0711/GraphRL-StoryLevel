import json
import numpy as np
from pathlib import Path

from Structure.structure import Structure
from RL.environment import Environment


class Record:
    def __init__(self):
        self.training_record = {
            "geometry": [],      
            "initial_design": [],  
            "initial_volume": [], 

            "score": [],
            "final_design": [], 
            "final_volume": [],  
            "action": [],
            "action_SCWB": [],
            "saved_material": [],      
            "saved_material_SCWB": [],
            "fail_name": [], 
            "fail_reason": []
        }

        self.testing_record = {
            "geometry": [],     
            "initial_design": [],
            "initial_volume": [],

            "score": [],  
            "final_design": [],
            "final_volume": [],
            "action": [], 
            "action_SCWB": [], 
            "saved_material": [],      
            "saved_material_SCWB": [],
            "fail_name": [],   
            "fail_reason": [] 
        }

        self.learn_losses = [[]]
        self.Q_values = [[], []]  # Q_values[0] for training, Q_values[1] for testing


    def record_in_beginning(self, structure: Structure, testing: bool=False):
        """
        Record the initial state of the structure.
        * geometry (span_num, story_num, span, story_height)
        * initial design
        * initial volume
        """
        record = self.testing_record if testing else self.training_record
        record["geometry"].append([structure.x_span_num, structure.z_span_num, structure.story_num, structure.x_span_lens, structure.z_span_lens, structure.story_height])
        record["initial_design"].append([i for i in structure.story_level_sections])
        record["initial_volume"].append(structure.calculate_material_usage())

    def record_in_end(self, structure: Structure, env: Environment, testing: bool=False):
        """
        Record the final state of the structure and design process.
        * score (cumulative reward)
        * final design, final volume
        * action, action_SCWB
        * saved_material, saved_material_SCWB
        * fail_name, fail_reason
        """
        record = self.testing_record if testing else self.training_record
        record["score"].append(sum(env.reward_record))

        record["final_design"].append([i for i in structure.story_level_sections])
        record["final_volume"].append(structure.calculate_material_usage())

        record["action"].append(env.update_actions_record)
        record["action_SCWB"].append(env.update_actions_record_SCWB)

        record["saved_material"].append(sum(env.saved_material_record))
        record["saved_material_SCWB"].append(sum(env.saved_material_record_SCWB))

        record["fail_name"].append(env.fail_name)
        record["fail_reason"].append(env.fail_reason)
    
    def output(self, ckpt_dir: Path):
        with open(ckpt_dir / "training_record.txt", "w") as f: json.dump(self.training_record, f)
        with open(ckpt_dir / "testing_record.txt", "w") as f: json.dump(self.testing_record, f)
        with open(ckpt_dir / "learn_losses.txt", "w") as f: json.dump(self.learn_losses, f)
        with open(ckpt_dir / "Q_values.txt", "w") as f: json.dump(self.Q_values, f)


