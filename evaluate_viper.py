import json
import torch
import shutil
import joblib
import random
import logging
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from pyparsing import Union
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from VIPER import viper, environment, record, utils
from Structure.sections import beam_sections, column_sections
from Structure.structure import Structure
from Structure import pisa, check
from RL import agent
from NonlinearDynamicAnalysisSimulator import load_simulator


def set_random_seed(SEED: int):
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
	torch.autograd.set_detect_anomaly(True)


def get_loggings(ckpt_dir: Path):
	logger = logging.getLogger(name="VIPER")
	logger.setLevel(level=logging.DEBUG)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(ckpt_dir / "inference_record.log")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def inference(structure: Structure, 
              policy: Union[DecisionTreeClassifier, agent.DeepQAgent], 
              env: environment.Environment,
              extractor: viper.VIPER, 
              rec: record.Record,
              logger: logging.Logger) -> None:
    graph = structure.graph.clone()
    done = None
    timestep = 0
    accumulated_reward = 0
    while not done:
        original_structure = deepcopy(structure)
        
        if isinstance(policy, DecisionTreeClassifier):
            feature = utils.extract_features(structure)
            _, feature_values = extractor.select_features(feature)
            action, action_type = extractor.select_valid_action(structure, policy, feature_values)
            logger.info(f"greedy policy's selection: {action}, {action_type}")
        else:
            with torch.no_grad():
                graph = graph.to(extractor.oracle_policy.device)
                state = extractor.oracle_policy.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(extractor.oracle_policy.device), None)
            q_values, mask, action, weight = extractor.get_oracle_output(structure, state)            
            logger.info(f"greedy policy's selection: {action}, Q-value: {q_values[action]}")
        
        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"story_level_sections: {structure.story_level_sections}, action: {action:3d} [{update_story}F {member_category}]")
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # get next state
        if not done:
            del original_structure, graph
            graph = structure.graph.clone()
        timestep += 1
        accumulated_reward += reward
        logger.info(f"timestep: {timestep}, accumulated_reward: {accumulated_reward}\n")

    total_saved_material = np.sum(env.saved_material_record)
    total_saved_percentage = accumulated_reward / env.material_usage_record[0]

    logger.info(f"final_timestep: {timestep}, fail_name: {fail_name}, fail_reason: {fail_reason}")
    logger.info(f"-----x_span_num: {structure.x_span_num}, z_span_num: {structure.z_span_num}, story_num: {structure.story_num}, reduced_percentage: {total_saved_percentage:.4f}, reduced_material: {total_saved_material:.4f} m3\n")
    
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    rec.record_in_end(final_structure, env, eval=True)
    del structure, graph

    if isinstance(policy, DecisionTreeClassifier):
        output_path = env.checkpoint_dir / "inference_record_dtc.txt"
    else:
        output_path = env.checkpoint_dir / "inference_record_dqn.txt"
    with open(output_path, 'w') as f:
        json.dump(rec.evaluation_record, f)
    rec.record_next_iteration()


def main():
    ckpt_dir = Path("./VIPER/results/DDQN_2025_06_05__VIPER_2025_10_30__16_25_41__FixShape_StoryNum6_TestProb1e-1_ConEps_MaxDepthNone_MinSplit2_MinLeaf10_Trajec100_Iter10_Dataset1e5_Batch5e4")
    with open(ckpt_dir / "training_args.json", 'r') as f:
        training_args = json.load(f)
    train_args = Namespace(**training_args)

    # Set random seed
    set_random_seed(train_args.random_seed)

    # Set logger
    logger = get_loggings(ckpt_dir)
    logger.info(ckpt_dir)
    logger.info(train_args)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
    logger.info(f"Device: {device_name}")

	# Setup nonlinear dynamic analysis simulator
    nda_simulator = None
    nda_norm_dict = None
    DBE_gm_set = None
    MCE_gm_set = None
    if train_args.do_nda:
        nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(train_args.graph_lstm_dir, device)
        DBE_gm_set, MCE_gm_set = load_simulator.load_ground_motions(train_args.gm_dir, train_args.gm_num, nda_norm_dict)

    # Extracted decision tree model
    tree_model_path = ckpt_dir / "models" / "dtc_best.joblib"
    tree_model = joblib.load(tree_model_path)
    logger.info(f"Decision Tree Classifier loaded from {tree_model_path}")
    logger.info(f"Max depth: {tree_model.get_depth()}")  # maximum depth of the tree
    logger.info(f"Number of nodes: {tree_model.tree_.node_count}")  # number of nodes
    logger.info(f"Number of leaf nodes: {tree_model.get_n_leaves()}")  # number of leaf nodes

    # Double DQN agent
    oracle_model_path = ckpt_dir / "trained_ddqn_model.pt"
    node_feature_dim = 8 if train_args.add_geometry_feature else 5
    edge_feature_dim = 13 if train_args.add_response_feature else 11
    _agent_kwargs = {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "hidden_dim": train_args.ddqn_hidden_dim,
        "num_layers": train_args.ddqn_layer_num,
        "batch_size": None,
        "lr": train_args.lr,
        "buffer_size": None,
        "epsilon_decay_schedule": None,
        "synchronize_steps": None,
        "soft_update_alpha": None,
        "gamma": None,
        "update_frequency": None,
        "add_experience_frequency": None,
        "test_frequency": None,
        "restrict_action": train_args.restrict_action,
        "seed": train_args.random_seed,
        "logger": logger,
        "pretrained_ckpt_dir": oracle_model_path,
        "device": device,
    }
    if train_args.ddqn_model_type == "Taiwan":
        double_dqn_agent = agent.DeepQAgent(**_agent_kwargs)
    elif train_args.ddqn_model_type == "Japan":
        double_dqn_agent = agent.JapanDeepQAgent(**_agent_kwargs)

	# Environment
    _env_kwargs = {
        "story_num": train_args.story_num, 
        "testing_prob": train_args.testing_prob,
        "structure_shape": train_args.structure_shape,
        "add_structure_geometry": train_args.add_geometry_feature,
        "add_response_features": train_args.add_response_feature,
        "reward_type": train_args.reward_type,
        "scwb_driven_design": train_args.scwb_driven_design,
        "do_nonlinear_dynamic_analysis": train_args.do_nda,
        "check_acceleration": train_args.check_acc,
        "check_displacement": train_args.check_disp,
        "nda_simulator": nda_simulator,
        "nda_norm_dict": nda_norm_dict,
        "DBE_ground_motion_set": DBE_gm_set,
        "MCE_ground_motion_set": MCE_gm_set,
        "checkpoint_dir": ckpt_dir,
        "logger": logger,
        "device": device,
    }
    env = environment.Environment(**_env_kwargs)

    # VIPER
    _dtc_kwargs = {
        "criterion": train_args.criterion,
        "max_depth": train_args.max_depth,
        "min_samples_split": train_args.min_samples_split,
        "min_samples_leaf": train_args.min_samples_leaf,
        "max_features": train_args.max_features,
        "max_leaf_nodes": train_args.max_leaf_nodes,
    }
    _viper_kwargs = {
        "env": env,
        "oracle_policy": double_dqn_agent, 
        "M": train_args.trajectory_num, 
        "N": train_args.iteration_num, 
        "dataset_size": train_args.dataset_size,
        "batch_size": train_args.batch_size,
        "epsilon_schedule": None,
        "dtc_kwargs": _dtc_kwargs, 
        "feature_types": train_args.feature_types, 
        "logger": logger, 
        "checkpoint_dir": train_args.ckpt_dir,
        "enable_memory_monitoring": False,
    }
    extractor = viper.VIPER(**_viper_kwargs)

    feature = utils.extract_features(env._testing_structure)
    feature_names, feature_values = extractor.select_features(feature)
    logger.debug(f"Feature names: {feature_names}\n")
    rec_dqn = record.Record(feature_names)
    rec_dtc = record.Record(feature_names)

    structure_shape = env.structure_shape
    if structure_shape == "specific":
        x_span_num_list = [4]
        z_span_num_list = [4]
        x_span_len_list = [7000]
        z_span_len_list = [7000]
    elif structure_shape == "fixed":
        x_span_num_list = [4]
        z_span_num_list = [4]
        x_span_len_list = [6000, 7000, 8000]
        z_span_len_list = [6000, 7000, 8000]
    elif structure_shape == "small_random":
        x_span_num_list = [3, 4, 5]
        z_span_num_list = [3, 4, 5]
        x_span_len_list = [7000]
        z_span_len_list = [7000]
    elif structure_shape == "random":
        x_span_num_list = [2, 3, 4, 5, 6]
        z_span_num_list = [2, 3, 4, 5, 6]
        x_span_len_list = [7000]
        z_span_len_list = [7000]

    for x_span_num, z_span_num, x_span_len, z_span_len in itertools.product(x_span_num_list, z_span_num_list, x_span_len_list, z_span_len_list):
        x_span_lens = [x_span_len for i in range(x_span_num)]
        z_span_lens = [z_span_len for i in range(z_span_num)]
        story_num = train_args.story_num
        story_height = 3200        
        story_level_sections = [len(beam_sections)-1] * (story_num*2) + [len(column_sections)-1] * (story_num*2)
        structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                            "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                            "story_num": story_num, "story_height": story_height,
                            "story_level_sections": story_level_sections,
                            "add_structure_geometry": train_args.add_geometry_feature,
                            "add_response_features": train_args.add_response_feature,
                            "do_nonlinear_dynamic_analysis": train_args.do_nda, 
                            "nda_norm_dict": nda_norm_dict,
                            "analysis_dir": ckpt_dir / "Modal_Analysis"}
        
        structure_kwargs_dqn = deepcopy(structure_kwargs)
        structure_dqn = Structure(**structure_kwargs_dqn)
        logger.info(structure_dqn)
        env.init_records(structure_dqn)
        rec_dqn.record_in_beginning(structure_dqn, eval=True)
        inference(structure_dqn, double_dqn_agent, env, extractor, rec_dqn, logger)
        del structure_kwargs_dqn, structure_dqn
        
        structure_kwargs_dtc = deepcopy(structure_kwargs)
        structure_dtc = Structure(**structure_kwargs_dtc)
        logger.info(structure_dtc)
        env.init_records(structure_dtc)
        rec_dtc.record_in_beginning(structure_dtc, eval=True)
        inference(structure_dtc, tree_model, env, extractor, rec_dtc, logger)
        del structure_kwargs_dtc, structure_dtc




if __name__ == "__main__":
    main()