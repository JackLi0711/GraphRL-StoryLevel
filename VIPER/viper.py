import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union
from copy import deepcopy
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import sys
sys.path.append('../')
from Structure.structure import Structure
from RL.agent import DeepQAgent
import VIPER.plot as plot
from VIPER.environment import Environment
from VIPER.record import Record
from VIPER.utils import Feature, extract_features, get_size
from VIPER.memory_monitor import MemoryMonitor, monitor_graph_operations


class VIPER:
    """
    Encapsulated VIPER policy extraction algorithm.

    Usage:
        viper = VIPER(env, oracle_agent, M=5, N=10, batch_size=2048, dtc_args={...})
        best_tree = viper.run()
    """

    def __init__(self,
                 env: Environment,
                 oracle_policy: DeepQAgent,
                 M: int,
                 N: int,
                 dataset_size: int,
                 batch_size: int,
                 epsilon_schedule: callable,
                 dtc_kwargs: dict,
                 feature_types: list[str],
                 logger: logging.Logger,
                 checkpoint_dir: Path, 
                 enable_memory_monitoring: bool = False,) -> None:
        self.env = env
        self.oracle_policy = oracle_policy
        self.M = M  # number of trajectories per iteration
        self.N = N  # number of VIPER iterations
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epsilon_schedule = epsilon_schedule
        self.dtc_kwargs = dtc_kwargs
        self.feature_types = feature_types
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.monitor = MemoryMonitor() if enable_memory_monitoring else None

        # aggregated dataset container
        self.dataset = {"features": [], "oracle_action": [], "weight": []}
        self.feature_names: list[str] = None
        self.class_names: list[str] = None
        self.extracted_policies: list[DecisionTreeClassifier] = []

    def select_features(self, feature: Feature) -> tuple[list[str], list[float]]:
        """
        Flatten and optionally filter raw extracted features.         
        """
        feature_dict = {}
        for grid_num, direction in zip(feature.grid_num, ['x', 'z', 'y']):
            feature_dict[f"grid_num_{direction}"] = grid_num.item()
        for mode_period, mode in zip(feature.mode_period, ['1st', '2nd', '3rd']):
            feature_dict[f"mode_period_{mode}"] = mode_period.item()

        story_num, member_category_num = feature.member_Ag.shape
        member_categories = ['xbeam', 'zbeam', 'outcol', 'incol']
        for story_index in range(story_num):
            story = f'{story_index+1}F'
            for category_index in range(member_category_num):
                category = member_categories[category_index]
                feature_dict[f"member_Ag_{story}_{category}"] = feature.member_Ag[story_index, category_index].item()
                feature_dict[f"member_Iy_{story}_{category}"] = feature.member_Iy[story_index, category_index].item()
                feature_dict[f"member_Iz_{story}_{category}"] = feature.member_Iz[story_index, category_index].item()
                feature_dict[f"member_Zz_{story}_{category}"] = feature.member_Zz[story_index, category_index].item()

        story_num, location_num, direction_num = feature.min_scwb_ratio.shape
        locations = ['out', 'in']
        directions = ['x', 'z']
        for story_index in range(story_num):
            story = f'{story_index+1}F'
            for location_index in range(location_num):
                location = locations[location_index]
                for direction_index in range(direction_num):
                    direction = directions[direction_index]
                    feature_dict[f"min_scwb_ratio_{story}_{location}_{direction}"] = feature.min_scwb_ratio[story_index, location_index, direction_index].item()
                    feature_dict[f"max_drift_ratio_{story}_{location}_{direction}"] = feature.max_drift_ratio[story_index, location_index, direction_index].item()

        story_num, member_category_num = feature.max_stress_ratio.shape
        member_categories = ['xbeam', 'zbeam', 'outcol', 'incol']
        for story_index in range(story_num):
            story = f'{story_index+1}F'
            for category_index in range(member_category_num):
                category = member_categories[category_index]
                feature_dict[f"max_stress_ratio_{story}_{category}"] = feature.max_stress_ratio[story_index, category_index].item()

        used_feature_names, used_feature_values = [], []
        for type in self.feature_types:  # e.g., ["grid_num", "member_Ag", "min_scwb_ratio"]
            for key, value in feature_dict.items():  # e.g., ["grid_num_x", "member_Ag_1F_xbeam", "min_scwb_ratio_1F_out_x"]
                if type in key:
                    used_feature_names.append(key)
                    used_feature_values.append(value)
        
        return used_feature_names, used_feature_values

    def get_oracle_output(self, structure: Structure, oracle_state: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int, float]:
        """
        Get the oracle output for a given structure and oracle state.
        """
        dont_select_story_member_indexes = structure.restrict_action_space() if self.oracle_policy.restrict_action else []
        infeasible_actions = list(set(structure.already_minimum_section_story_indexes + dont_select_story_member_indexes))
        mask = np.array([True if i in infeasible_actions else False for i in range(len(structure.story_level_sections))], dtype=bool)
        
        with torch.no_grad():
            q_values = self.oracle_policy.online_q_network.forward(oracle_state).detach().cpu().numpy().squeeze()
        q_values_feasible = np.ma.masked_where(mask, q_values)
        action = int(np.argmax(q_values_feasible))
        weight = np.max(q_values_feasible) - np.min(q_values_feasible)

        return q_values, mask, action, weight


    def select_valid_action(self,
                            structure: Structure, 
                            policy: DecisionTreeClassifier,
                            feature_values: list[float]) -> tuple[int, str]:
        """
        Select the highest-probability valid action from the decision tree.
        Falls back to oracle_action if every predicted candidate is invalid.
        """
        invalid_actions = set(structure.already_minimum_section_story_indexes)
        if self.oracle_policy.restrict_action:
            invalid_actions.update(structure.restrict_action_space())
        
        # Predict class probabilities (aligned with policy.classes_)
        probs = policy.predict_proba([feature_values])[0]
        print(f"\t{probs = }, {np.sum(probs) = }, \n\t{np.min(probs) = }, {np.mean(probs) = }, {np.max(probs) = }, {np.std(probs) = }")
        sorted_idx = np.argsort(probs)[::-1]  # descending probability
        
        for rank, idx in enumerate(sorted_idx, start=1):
            candidate = int(policy.classes_[idx])
            if candidate not in invalid_actions:
                return candidate, f"policy_action_rank{rank}"

    def sample_trajectories(self, iteration_count: int, policy: Union[DecisionTreeClassifier, DeepQAgent], record: Record) -> None:
        """
        Use the given policy to rollout and collect data.
        """
        for trajectory_count in range(self.M):
            self.monitor.log_memory("start_of_trajectory") if self.monitor else None
            self.logger.info(f"Start trajectory {trajectory_count+1}/{self.M}")
            
            self.monitor.log_memory("env.reset") if self.monitor else None
            structure = self.env.reset() if np.random.rand() > self.env.testing_prob else self.env.reset(testing=True)
            
            self.monitor.log_memory("record_in_beginning") if self.monitor else None
            record.record_in_beginning(structure)

            self.monitor.log_memory("graph_clone") if self.monitor else None
            graph = structure.graph.clone()
            done = False
            timestep = 0
            score = 0
            while not done:
                # Explicitly delete any lingering references before deepcopy
                # if 'original_structure' in locals():
                #     del original_structure
                # if 'feature' in locals():
                #     del feature
                # if 'used_feature_names' in locals():
                #     del used_feature_names
                # if 'used_feature_values' in locals():
                #     del used_feature_values
                # if 'oracle_state' in locals():
                #     del oracle_state
                # if 'q_values' in locals():
                #     del q_values
                # if 'mask' in locals():
                #     del mask
                if self.monitor:
                    self.monitor.cleanup_memory()  # Force cleanup before deepcopy

                self.monitor.log_memory("deepcopy") if self.monitor else None
                original_structure = deepcopy(structure)

                self.monitor.log_memory("extract_features") if self.monitor else None
                feature = extract_features(structure)

                self.monitor.log_memory("select_features") if self.monitor else None
                used_feature_names, used_feature_values = self.select_features(feature)
                assert self.feature_names == used_feature_names, "Feature names mismatch across trajectories"
                assert len(self.class_names) == len(structure.story_level_sections), "Class number mismatch across trajectories"
                
                with monitor_graph_operations(self.monitor) if self.monitor else torch.no_grad():
                    with torch.no_grad():
                        graph = graph.to(self.oracle_policy.device)
                        oracle_state = self.oracle_policy.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(self.oracle_policy.device), None)
                self.monitor.log_memory("get_oracle_output") if self.monitor else None
                q_values, mask, oracle_action, weight = self.get_oracle_output(structure, oracle_state)
                
                # update dataset
                self.monitor.log_memory("update_dataset") if self.monitor else None
                if len(self.dataset["features"]) >= self.dataset_size:
                    for key in self.dataset.keys():
                        self.dataset[key].pop(0)
                self.dataset["features"].append(used_feature_values)
                self.dataset["oracle_action"].append(oracle_action)
                self.dataset["weight"].append(weight)
                print(f"\t{q_values = }, {oracle_action = }")

                self.monitor.log_memory("select_valid_action") if self.monitor else None
                if isinstance(policy, DecisionTreeClassifier):
                    pre_action, pre_action_type = self.select_valid_action(structure, policy, used_feature_values)
                else:  # 1st iteration only
                    pre_action, pre_action_type = oracle_action, "oracle_action"

                # epsilon-greedy exploration
                epsilon = self.epsilon_schedule(iteration_count)
                print(f"{epsilon = }")
                if np.random.rand() < epsilon:  # exploration: Decision Tree Model
                    action, action_type = pre_action, pre_action_type
                else:                           # exploitation: Deep Q-Network
                    action, action_type = oracle_action, "oracle_action"

                member_category = structure.story_level_categories[action]
                update_story = (action % structure.story_num) + 1
                print(f"-----trajectory: {trajectory_count+1:4d}, timestep: {timestep+1:3d}, story_level_sections: {structure.story_level_sections}, {action_type}: {action:3d} [{update_story}F {member_category}]")

                self.monitor.log_memory("env.step") if self.monitor else None
                structure, reward, done, fail_name, fail_reason = self.env.step(structure, action)
                score += reward
                self.logger.info(f"trajectory: {trajectory_count+1:4d}, timestep: {timestep+1:3d}, {action_type}: {action:3d}, reward: {reward:4f},  accumulated_score: {score:.4f}, accumulated_saved_material: {sum(self.env.saved_material_record):.4f}")
                print()

                self.monitor.log_memory("end_of_timestep") if self.monitor else None
                timestep += 1
                if not done:
                    del original_structure, graph  # free memory
                    self.monitor.cleanup_memory() if self.monitor else None
                    graph = structure.graph.clone()
                    self.monitor.log_memory("del_graph_and_clone") if self.monitor else None
                    print()
                if self.monitor and timestep % 10 == 0:
                    self.monitor.cleanup_memory()
                    self.monitor.log_memory("cleanup_memory") if self.monitor else None
                    print()
            
            self.logger.info(f"Complete trajectory {trajectory_count+1}/{self.M}")
            self.logger.info(f"Dataset size: {len(self.dataset['features'])}, memory usage: {get_size(self.dataset)/1024**2:.3f} MB")

            final_structure = structure if fail_reason == "minimum_section" else original_structure
            final_story_level_sections = final_structure.story_level_sections
            self.logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
            self.logger.info(f"{final_story_level_sections = }")
            self.logger.info(f"Trajectory: {trajectory_count+1:4d}, fail name: {fail_name}, fail reason: {fail_reason}\n\n")
    
            self.monitor.log_memory("record_in_end") if self.monitor else None
            record.record_in_end(final_structure, self.env)
            del structure, graph  # free memory
            self.monitor.cleanup_memory() if self.monitor else None
            self.monitor.log_memory("end_of_trajectory") if self.monitor else None


    def resample_dataset(self) -> tuple[np.ndarray, np.ndarray]: 
        """
        Resample the dataset based on weights calculated from the Q-values
        """
        weights = np.array(self.dataset["weight"])
        probs = weights / np.sum(weights)
        self.logger.info(f"Resampling: {probs.shape = }, {np.min(probs) = }, {np.mean(probs) = }, {np.max(probs) = }, {np.std(probs) = }")

        features = np.array(self.dataset["features"])
        actions = np.array(self.dataset["oracle_action"])
        dataset_size = len(features)
        size = min(self.batch_size, dataset_size)
        resampled_indices = np.random.choice(dataset_size, size=size, p=probs, replace=True)
        
        return features[resampled_indices], actions[resampled_indices]

    def train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
        print(f"{X_train.shape = }, {y_train.shape = }")
        dtc = DecisionTreeClassifier(**self.dtc_kwargs)
        dtc.fit(X_train, y_train)
        
        cv_scores = cross_val_score(dtc, X_train, y_train, cv=5, scoring='accuracy', verbose=1)
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean cross-validation score: {np.mean(cv_scores)}")

        return dtc


    def save_model(self, policy: DecisionTreeClassifier, suffix: str) -> None:
        # Save the decision tree model
        model_filename = self.env.model_dir / f"dtc_{suffix}.joblib"
        joblib.dump(policy, model_filename)
        self.logger.info(f"Saved DecisionTreeClassifier to {model_filename}")
                
        # Visualize the decision tree model
        try:
            # plt.figure(figsize=(32, 20))
            # plt.title(f"Decision Tree ({suffix})")
            # plot_tree(policy, feature_names=self.feature_names, class_names=self.class_names, filled=True)
            # plt.savefig(self.env.model_dir / f"dtc_{suffix}.png", dpi=300, bbox_inches='tight')
            # plt.close()
            
            rules = export_text(policy, feature_names=self.feature_names, show_weights=True)
            with open(self.env.model_dir / f"dtc_{suffix}.txt", "w") as f:
                f.write(rules)
        
        except Exception as e:
            self.logger.warning(f"Failed to visualize / export tree: {e}")

    def evaluate_policy(self, policy: Union[DecisionTreeClassifier, DeepQAgent], record: Record) -> float:
        scores = []
        fail_names, fail_reasons = [], []
        structure = self.env.reset(testing=True)
        record.record_in_beginning(structure, eval=True)
        graph = structure.graph.clone()
        done = False
        timestep = 0
        score = 0
        while not done:
            original_structure = deepcopy(structure)
            if isinstance(policy, DecisionTreeClassifier):
                feature = extract_features(structure)
                _, feature_values = self.select_features(feature)
                action, action_type = self.select_valid_action(structure, policy, feature_values)
            else:
                with torch.no_grad():
                    graph = graph.to(self.oracle_policy.device)
                    state = self.oracle_policy.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(self.oracle_policy.device), None)
                    q_values, mask, action, weight = self.get_oracle_output(structure, state)
            member_category = structure.story_level_categories[action]
            update_story = (action % structure.story_num) + 1
            print(f"\n*****Evaluation, timestep: {timestep+1:3d}, story_level_sections: {structure.story_level_sections}, action: {action:3d} [{update_story}F {member_category}]")

            if action not in structure.already_minimum_section_story_indexes:
                structure, reward, done, fail_name, fail_reason = self.env.step(structure, action)
                score += reward
                self.logger.info(f"*****Evaluation, timestep: {timestep+1:3d}, action: {action:3d}, reward: {reward:4f},  accumulated_score: {score:.4f}, accumulated_saved_material: {sum(self.env.saved_material_record):.4f}")
            else: 
                done = True
                fail_name = None
                fail_reason = "invalid_action"
                self.logger.warning(f"Action {action} is already at minimum section, stopping evaluation.")

            timestep += 1
            if not done:
                del original_structure, graph  # free memory
                graph = structure.graph.clone()
        
        final_structure = structure if fail_reason == "minimum_section" else original_structure
        final_story_level_sections = final_structure.story_level_sections
        self.logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
        self.logger.info(f"{final_story_level_sections = }")
        self.logger.info(f"Evaluation, fail name: {fail_name}, fail reason: {fail_reason}\n\n")

        scores.append(score)
        
        record.record_in_end(final_structure, self.env, eval=True)
        del structure, graph  # free memory
        if isinstance(policy, DecisionTreeClassifier):
            record.record_dtc_information(policy)
        
        return float(np.mean(scores))


    def run(self) -> DecisionTreeClassifier:
        """
        Run the VIPER algorithm to extract a decision tree policy.
        
        This method performs the following steps:
        1. Sample M trajectories using the oracle policy or extracted policy.
        2. Resample the dataset to train the decision tree classifier
        3. Evaluate all the extracted policies.
        """
        if self.monitor:
            self.monitor.start_monitoring()

        structure = self.env._testing_structure
        feature = extract_features(structure)
        used_feature_names, used_feature_values = self.select_features(feature)
        self.feature_names = used_feature_names
        self.logger.debug(f"Feature names: {self.feature_names}\n")
        rec = Record(self.feature_names)
        
        class_names = []
        for i in range(len(structure.story_level_sections)):
            story_index = i % structure.story_num
            class_names.append(f"{story_index+1}F_{structure.story_level_categories[i]}")
        self.class_names = class_names
        self.logger.debug(f"Class names: {self.class_names}\n")

        current_policy: Union[DecisionTreeClassifier, DeepQAgent] = self.oracle_policy
        best_score = -np.inf
        best_idx, best_policy = None, None
        for i in range(self.N):
            # Training
            self.logger.info(f"START ITERATION {i+1}/{self.N}")
            
            self.sample_trajectories(i, current_policy, rec)
            X_resampled, y_resampled = self.resample_dataset()
            new_policy = self.train_decision_tree(X_resampled, y_resampled)
            self.extracted_policies.append(new_policy)
            self.save_model(new_policy, f"Iteration{i+1}")
            current_policy = new_policy

            self.logger.info(f"COMPLETED ITERATION {i+1}/{self.N}\n\n")

            # Evaluation
            self.logger.info(f"Evaluating policy {i+1}/{self.N}")
            score = self.evaluate_policy(current_policy, rec)
            self.logger.info(f"Policy {i+1} score: {score:.4f}")
            if (score > best_score) or (score == best_score and current_policy.get_depth() < best_policy.get_depth()):
                best_score = score
                best_idx = i
                best_policy = current_policy

            rec.output(self.env.checkpoint_dir)
            if i != self.N-1:
                rec.record_next_iteration()
        
        if best_policy is not None:
            self.logger.info(f"\n\n\nBest policy is from iteration {best_idx+1} with score {best_score:.4f}")
            self.logger.info(f"Best policy structure: {best_policy.tree_}")  # structure of the tree
            self.logger.info(f"Max depth: {best_policy.get_depth()}")  # maximum depth of the tree
            self.logger.info(f"Number of nodes: {best_policy.tree_.node_count}")  # number of nodes
            self.logger.info(f"Number of leaf nodes: {best_policy.get_n_leaves()}")  # number of leaf nodes
            
            # Find top 10 feature importances and their names
            importances = best_policy.feature_importances_
            top_indices = np.argsort(importances)[::-1][:10]
            self.logger.info("Top 10 feature importances:")
            for i, idx in enumerate(top_indices):
                self.logger.info(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

            # params = best_policy.get_params()
            # for key, value in params.items():
            #     self.logger.info(f"Best policy parameter '{key}': {value}")
            self.save_model(best_policy, "best")

        rec.output(self.env.checkpoint_dir)

        plot.plot_policy_scores(rec.evaluation_record["score"], self.env.checkpoint_dir)
        plot.plot_fail_names(rec.evaluation_record["fail_name"], self.env.checkpoint_dir)
        plot.plot_fail_reasons(rec.evaluation_record["fail_reason"], self.env.checkpoint_dir)
        
        return best_policy


def train(env: Environment,
          oracle_policy: DeepQAgent,
          M: int, N: int, batch_size: int,
          dtc_kwargs: dict,
          feature_types: list[str],
          logger: logging.Logger, 
          checkpoint_dir: Path,
          enable_memory_monitoring: bool = True,) -> DecisionTreeClassifier:
    
    extractor = VIPER(env, oracle_policy, M, N, batch_size, dtc_kwargs, feature_types, logger, checkpoint_dir, enable_memory_monitoring)
    
    return extractor.run()
