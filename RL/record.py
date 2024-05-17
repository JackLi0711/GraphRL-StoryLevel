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

            "final_design": [], 
            "final_volume": [],  
            "action": [],
            "action_SCWB": [],
            "score": [],      
            "score_SCWB": [],
            "fail_name": [], 
            "fail_reason": []
        }

        self.testing_record = {
            "geometry": [],     
            "initial_design": [],
            "initial_volume": [],

            "final_design": [],
            "final_volume": [],
            "action": [], 
            "action_SCWB": [], 
            "score": [],    
            "score_SCWB": [],  
            "fail_name": [],   
            "fail_reason": [] 
        }

        self.learn_losses = []
        self.Q_values = []


    def record_in_beginning(self, structure: Structure, testing: bool=False):
        """
        Record the initial state of the structure
        * geometry (span_num, story_num, span, story_height)
        * initial design (story_level_sections)
        * initial volume
        """
        record = self.testing_record if testing else self.training_record
        record["geometry"].append([structure.x_span_num, structure.z_span_num, structure.story_num, structure.x_span_lens[0], structure.z_span_lens[0], structure.story_height])
        initial_design = [i for i in structure.story_level_sections]
        record["initial_design"].append(initial_design)
        record["initial_volume"].append(structure.calculate_material_usage())

    def record_in_end(self, structure: Structure, env: Environment, testing: bool=False):
        """
        Record the final state of the structure and design process
        * final design (story_level_sections)
        * final volume
        * action, action_SCWB
        * score, score_SCWB
        * fail_name
        * fail_reason
        """
        record = self.testing_record if testing else self.training_record
        final_design = [i for i in structure.story_level_sections]
        record["final_design"].append(final_design)
        record["final_volume"].append(structure.calculate_material_usage())

        record["action"].append(env.update_actions_record)
        record["action_SCWB"].append(env.update_actions_record_SCWB)

        record["score"].append(sum(env.saved_material_record))
        record["score_SCWB"].append(sum(env.saved_material_record_SCWB))

        record["fail_name"].append(env.fail_name)
        record["fail_reason"].append(env.fail_reason)
    
    def output(self, ckpt_dir: Path):
        score_info = {
            "train_score": self.training_record["score"],
            "train_score_SCWB": self.training_record["score_SCWB"],
            "test_score": self.testing_record["score"],
            "test_score_SCWB": self.testing_record["score_SCWB"]
        }
        action_info = {
            "train_action": self.training_record["action"],
            "train_action_SCWB": self.training_record["action_SCWB"],
            "test_action": self.testing_record["action"],
            "test_action_SCWB": self.testing_record["action_SCWB"]
        }
        design_info = {
            "train_initial_design": self.training_record["initial_design"],
            "train_final_design": self.training_record["final_design"],
            "test_initial_design": self.testing_record["initial_design"],
            "test_final_design": self.testing_record["final_design"]
        }
        volume_info = {
            "train_initial_volume": self.training_record["initial_volume"],
            "train_final_volume": self.training_record["final_volume"],
            "test_initial_volume": self.testing_record["initial_volume"],
            "test_final_volume": self.testing_record["final_volume"]
        }
        fail_info = {
            "train_fail_name": self.training_record["fail_name"],
            "train_fail_reason": self.training_record["fail_reason"],
            "test_fail_name": self.testing_record["fail_name"],
            "test_fail_reason": self.testing_record["fail_reason"]
        }
        
        # with open(ckpt_dir / "score_info.txt", "w") as f: json.dump(score_info, f)
        # with open(ckpt_dir / "action_info.txt", "w") as f: json.dump(action_info, f)
        # with open(ckpt_dir / "design_info.txt", "w") as f: json.dump(design_info, f)
        # with open(ckpt_dir / "volume_info.txt", "w") as f: json.dump(volume_info, f)
        # with open(ckpt_dir / "fail_info.txt", "w") as f: json.dump(fail_info, f)

        with open(ckpt_dir / "training_record.txt", "w") as f: json.dump(self.training_record, f)
        with open(ckpt_dir / "testing_record.txt", "w") as f: json.dump(self.testing_record, f)