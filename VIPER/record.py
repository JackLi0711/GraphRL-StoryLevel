import json
import numpy as np
from pathlib import Path

from Structure.structure import Structure
from VIPER.environment import Environment
from sklearn.tree import DecisionTreeClassifier


class Record:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.training_record = {
            # Structure and design process information
            "geometry": [[]],     
            "initial_design": [[]],
            "initial_volume": [[]],

            "score": [[]],  
            "final_design": [[]],
            "final_volume": [[]],
            "action": [[]],
            "saved_material": [[]],
            "fail_name": [[]],
            "fail_reason": [[]]
        }
        self.evaluation_record = {
            # Structure and design process information
            "geometry": [[]],     
            "initial_design": [[]],
            "initial_volume": [[]],

            "score": [[]],
            "final_design": [[]],
            "final_volume": [[]],
            "action": [[]],
            "saved_material": [[]],
            "fail_name": [[]],
            "fail_reason": [[]],

            # DTC information
            "depth": [],
            "num_node": [],
            "num_leave": [],
            "important_feature": []
        }


    def record_in_beginning(self, structure: Structure, eval: bool = False):
        """
        Record the initial state of the structure.
        * geometry (span_num, story_num, span, story_height)
        * initial design
        * initial volume
        """
        record = self.evaluation_record if eval else self.training_record
        record["geometry"][-1].append([structure.x_span_num, structure.z_span_num, structure.story_num, structure.x_span_lens, structure.z_span_lens, structure.story_height])
        record["initial_design"][-1].append([i for i in structure.story_level_sections])
        record["initial_volume"][-1].append(structure.calculate_material_usage())

    def record_in_end(self, structure: Structure, env: Environment, eval: bool = False):
        """
        Record the final state of the structure and design process.
        * score (cumulative reward)
        * final design, final volume
        * action
        * saved_material
        * fail_name, fail_reason
        """
        record = self.evaluation_record if eval else self.training_record
        record["score"][-1].append(sum(env.reward_record))
        record["final_design"][-1].append([i for i in structure.story_level_sections])
        record["final_volume"][-1].append(structure.calculate_material_usage())
        record["action"][-1].append(env.update_actions_record)
        record["saved_material"][-1].append(sum(env.saved_material_record))
        record["fail_name"][-1].append(env.fail_name)
        record["fail_reason"][-1].append(env.fail_reason)

    def record_next_iteration(self): 
        """
        Prepare for the next iteration by resetting the current trajectory data.
        """
        for key in self.training_record.keys():
            self.training_record[key].append([])
            self.evaluation_record[key].append([])


    def record_dtc_information(self, dtc: DecisionTreeClassifier) -> None:
        """
        Record the evaluation metrics for a single run.
        """
        self.evaluation_record["depth"].append(int(dtc.get_depth()))
        self.evaluation_record["num_node"].append(dtc.tree_.node_count)
        self.evaluation_record["num_leave"].append(int(dtc.get_n_leaves()))

        important_features = []
        importances = dtc.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(top_indices):
            important_features.append((self.feature_names[idx], importances[idx]))
        self.evaluation_record["important_feature"].append(important_features)


    def output(self, ckpt_dir: Path) -> None:
        """
        Save the evaluation record to a txt file.
        """
        with open(ckpt_dir / "training_record.txt", "w") as f: json.dump(self.training_record, f)
        with open(ckpt_dir / "evaluation_record.txt", "w") as f: json.dump(self.evaluation_record, f)

