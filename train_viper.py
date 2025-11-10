import json
import torch
import shutil
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from copy import deepcopy

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from VIPER import viper, environment
from RL import agent
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train VIPER model")

    # Comment
    parser.add_argument("--comment", type=str, default="")

    # Checkpoint directory
    parser.add_argument("--ckpt_dir", type=Path, 
        default="./VIPER/results", 
        help="Path to checkpoint directory"
    )
    parser.add_argument("--ckpt_name", type=str, 
        default="FixShape_StoryNum6_TestProb1e-1_LinEps_MaxDepthNone_MinSplit2_MinLeaf10_Trajec50_Iter10_Dataset50000_Batch20000", 
        help="Name of the checkpoint file"
    )

    # Trained Double DQN model
    parser.add_argument("--ddqn_ckpt_dir", type=Path, 
        default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA"
    )
    parser.add_argument("--ddqn_model_folder", type=str, 
        default="2025_06_05__21_45_28__TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000"
    )
    parser.add_argument("--ddqn_model_type", type=str, default="Taiwan", help="Taiwan, Japan")
    parser.add_argument("--ddqn_hidden_dim", type=int, default=100)
    parser.add_argument("--ddqn_layer_num", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)

    # Structure
    parser.add_argument("--story_num", type=int, default=6, help="Number of stories of the structure")
    parser.add_argument("--testing_prob", type=float, default=0.1, help="Probability of using testing structure")
    parser.add_argument("--structure_shape", type=str, default="fixed", help="specific, fixed, small_random, random")
    parser.add_argument("--add_geometry_feature", default=True, help="Whether to add structure geometry in graph features")
    parser.add_argument("--add_response_feature", default=True, help="Whether to add structure response in graph features")
    parser.add_argument("--reward_type", type=str, default="material", help="material, acceleration, displacement, normalized, total, combined")
    parser.add_argument("--restrict_action", action="store_true", default=False)
    parser.add_argument("--scwb_driven_design", action="store_true", default=False)
	
    # Nonlinear dynamic analysis simulator
    parser.add_argument("--do_nda", action="store_true", default=False)
    parser.add_argument("--check_acc", action="store_true", default=False)
    parser.add_argument("--check_disp", action="store_true", default=True)
    parser.add_argument("--graph_lstm_dir", type=Path, 
        default=None  # "./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2025_05_19__22_59_28/"
    )  
    parser.add_argument("--gm_dir", type=Path, 
        default=None  # "./NonlinearDynamicAnalysisSimulator/ground_motions/selected_ground_motions_World_processed_one_scaling_MCE/"
    )
    parser.add_argument("--gm_num", type=int, default=11, help="ASCE says 11 is better")

    # Running VIPER
    parser.add_argument("--trajectory_num", type=int, default=50, help="Number of trajectories to sample per iteration.")
    parser.add_argument("--iteration_num", type=int, default=10, help="Number of iterations (and number of trees to train).")
    parser.add_argument("--dataset_size", type=int, default=int(5e4), help="Size of the dataset to resample state-action pairs from the collected trajectories.")
    parser.add_argument("--batch_size", type=int, default=int(2e4), help="Batch size for resampling and training the decision trees.")
    parser.add_argument("--feature_types", type=list[str], 
        default=["grid_num", "mode_period", "member_Ag", "member_Iy", "member_Iz", "member_Zz", "min_scwb_ratio", "max_drift_ratio", "max_stress_ratio"], 
        help="List of used feature names."
    )
    parser.add_argument("--random_seed", type=int, default=731, help="Fixed random seed")

    # Decision tree classifier
    parser.add_argument("--criterion", type=str, default="gini", help="Criterion for splitting: 'gini' or 'entropy'.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the decision tree.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=10, help="Minimum number of samples required to be at a leaf node.")
    parser.add_argument("--max_features", type=int, default=None, help="Number of features to consider when looking for the best split.")
    parser.add_argument("--max_leaf_nodes", type=int, default=None, help="Grow a tree with max_leaf_nodes in best-first fashion.")

    args = parser.parse_args()
    return args


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
	file_handler = logging.FileHandler(ckpt_dir / "record.log")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger




def main(args):
    # Set random seed
    set_random_seed(args.random_seed)

	# Set checkpoint directory
    ddqn_model_date = args.ddqn_model_folder.split("__")[0]
    current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    args.ckpt_dir = args.ckpt_dir / f"DDQN_{ddqn_model_date}__VIPER_{current_time}__{args.ckpt_name}"
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Set logger
    logger = get_loggings(args.ckpt_dir)
    logger.info(args.ckpt_dir)
    logger.info(args)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
    logger.info(f"Device: {device_name}")

	# Setup nonlinear dynamic analysis simulator
    nda_simulator = None
    nda_norm_dict = None
    DBE_gm_set = None
    MCE_gm_set = None
    if args.do_nda:
        nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
        DBE_gm_set, MCE_gm_set = load_simulator.load_ground_motions(args.gm_dir, args.gm_num, nda_norm_dict)

	# Constant epsilon schedule
    def constant_epsilon_schedule(iteration_num: int, epsilon: float=0.5) -> float:
        return epsilon
    fixed_epsilon_schedule = lambda n: constant_epsilon_schedule(n, 1.0)

    # Linear epsilon schedule
    def linear_increase_schedule(iteration_num: int, max_iterations: int, max_epsilon: float=1.0) -> float:
        return min(iteration_num/(max_iterations-1), max_epsilon)
    linear_epsilon_schedule = lambda n: linear_increase_schedule(n, args.iteration_num, 1.0)

    # Save the training arguments
    args_temp = deepcopy(args)
    for key, value in vars(args_temp).items():
        if isinstance(value, Path):
            setattr(args_temp, key, str(value))
    with open(args.ckpt_dir / "training_args.json", "w") as f:
        json.dump(vars(args_temp), f, indent=4)
    del args_temp

    # Double DQN agent
    node_feature_dim = 8 if args.add_geometry_feature else 5
    edge_feature_dim = 13 if args.add_response_feature else 11
    trained_ddqn_path = args.ddqn_ckpt_dir / args.ddqn_model_folder / "models" / "model_HighestScore.pt"
    shutil.copy(trained_ddqn_path, args.ckpt_dir / "trained_ddqn_model.pt")
    _agent_kwargs = {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "hidden_dim": args.ddqn_hidden_dim,
        "num_layers": args.ddqn_layer_num,
        "batch_size": None,
        "lr": args.lr,
        "buffer_size": None,
        "epsilon_decay_schedule": None,
        "synchronize_steps": None,
        "soft_update_alpha": None,
        "gamma": None,
        "update_frequency": None,
        "add_experience_frequency": None,
        "test_frequency": None,
        "restrict_action": args.restrict_action,
        "seed": args.random_seed,
        "logger": logger,
        "pretrained_ckpt_dir": trained_ddqn_path,
        "device": device,
    }
    if args.ddqn_model_type == "Taiwan":
        double_dqn_agent = agent.DeepQAgent(**_agent_kwargs)
    elif args.ddqn_model_type == "Japan":
        double_dqn_agent = agent.JapanDeepQAgent(**_agent_kwargs)


	# Environment
    _env_kwargs = {
        "story_num": args.story_num, 
        "testing_prob": args.testing_prob,
        "structure_shape": args.structure_shape,
        "add_structure_geometry": args.add_geometry_feature,
        "add_response_features": args.add_response_feature,
        "reward_type": args.reward_type,
        "scwb_driven_design": args.scwb_driven_design,
        "do_nonlinear_dynamic_analysis": args.do_nda,
        "check_acceleration": args.check_acc,
        "check_displacement": args.check_disp,
        "nda_simulator": nda_simulator,
        "nda_norm_dict": nda_norm_dict,
        "DBE_ground_motion_set": DBE_gm_set,
        "MCE_ground_motion_set": MCE_gm_set,
        "checkpoint_dir": args.ckpt_dir,
        "logger": logger,
        "device": device,
    }
    env = environment.Environment(**_env_kwargs)


    # VIPER
    _dtc_kwargs = {
        "criterion": args.criterion,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "max_leaf_nodes": args.max_leaf_nodes,
    }
    _viper_kwargs = {
        "env": env,
        "oracle_policy": double_dqn_agent, 
        "M": args.trajectory_num, 
        "N": args.iteration_num, 
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "epsilon_schedule": linear_epsilon_schedule,
        "dtc_kwargs": _dtc_kwargs, 
        "feature_types": args.feature_types, 
        "logger": logger, 
        "checkpoint_dir": args.ckpt_dir,
        "enable_memory_monitoring": True,
    }
    extractor = viper.VIPER(**_viper_kwargs)
    best_dtc_model = extractor.run()




if __name__ == "__main__":
	args = parse_args()
	main(args)
