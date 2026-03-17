import json
import torch
import random
import logging
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("RL/")
sys.path.append("Visualization/")
sys.path.append("NonlinearDynamicAnalysisSimulator/")

from RL import agent_DQN, agent_PPO, environment
from Visualization import plot, visualize
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
	parser = ArgumentParser()
 
	# trained model path
	parser.add_argument("--trained_model_path", type=Path, 
		default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2026_03_14__20_06_42__PPO_MatReward_doNDA_UseGAE095_LR5e-4_ActLossCoef10_CriLossCoef001_EntroWei01to001_OptimEpoch5_Episode1000/models/model_HighestScore.pt"
	)
	# checkpoint directory
	parser.add_argument("--ckpt_dir", type=Path, 
		default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2026_03_14__20_06_42__PPO_MatReward_doNDA_UseGAE095_LR5e-4_ActLossCoef10_CriLossCoef001_EntroWei01to001_OptimEpoch5_Episode1000"
	)
	# chances
	parser.add_argument("--chances", type=int, default=0)

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


def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='GraphRL')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	# file_handler = logging.FileHandler(ckpt_dir / "record.log")
	# file_handler.setFormatter(formatter)
	# logger.addHandler(file_handler)
	return logger




def main(args):
    # Load training arguments
    train_args_path = args.ckpt_dir / "train_args.json"
    with open(train_args_path, 'r') as f:
        train_args = json.load(f)
        for key, value in train_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)

	# Set random seed
    set_random_seed(args.random_seed)

	# Set logger
    logger = get_loggings(args.ckpt_dir)

	# Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

	# setup nonliear dynamic analysis simulator
    nda_simulator = None
    nda_norm_dict = None
    DBE_ground_motion_set = None
    MCE_ground_motion_set = None
    if args.do_nda:
        nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(Path(args.graph_lstm_dir), device)
        DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(Path(args.gm_dir), args.gm_num, nda_norm_dict)

	# Agent
    node_feature_dim = 8 if args.add_geometry_feature else 5
    edge_feature_dim = 13 if args.add_response_feature else 11
    _agent_kwargs = {
		"node_feature_dim": node_feature_dim,
		"edge_feature_dim": edge_feature_dim,
		"hidden_dim": args.hidden_dim,
		"num_layers": args.layer_num,
		"lr": args.lr,
		"use_lr_scheduler": args.use_lr_scheduler,
		"gamma": args.gamma,
		"use_gae": args.use_gae,
		"gae_tau": args.gae_tau,
		"target_kl": args.target_kl,
		"clip_eps": args.clip_eps,
		"entropy_weight_schedule": None,
		"actor_loss_coef": args.actor_loss_coef,
		"critic_loss_coef": args.critic_loss_coef,
		"max_grad_norm": args.max_grad_norm,
		"optimization_epoch": args.optimization_epoch,
		"seed": args.random_seed,
		"logger": logger,
		"pretrained_model_path": None,
		"device": device,
	}
    agent_model = agent_PPO.PPOAgent(**_agent_kwargs)

	# Environment
    _env_kwargs = {
		"structure_shape": args.structure_shape,
		"add_structure_geometry": args.add_geometry_feature,
		"add_response_features": args.add_response_feature,
		"reward_type": args.reward_type,
		"scwb_driven_design": False,
		"do_nonlinear_dynamic_analysis": args.do_nda,
		"check_acceleration": args.check_acc,
		"check_displacement": args.check_disp,
		"nda_simulator": nda_simulator,
		"nda_norm_dict": nda_norm_dict,
		"DBE_ground_motion_set": DBE_ground_motion_set,
		"MCE_ground_motion_set": MCE_ground_motion_set,
		"checkpoint_dir": args.ckpt_dir,
		"logger": logger,
		"device": device,
	}
    env = environment.Environment(**_env_kwargs)

    # load best-validation model
    checkpoint = torch.load(args.trained_model_path, map_location=torch.device(agent_model.device))
    agent_model.gnn.load_state_dict(checkpoint['gnn'])
    agent_model.actor_critic_network.load_state_dict(checkpoint['actor_critic'])
    logger.info(f"model are loaded from {args.trained_model_path}")

    visualize.visualize_design_process(agent_model, env, logger, testing_structure=True, initial_design=None, chances=args.chances)




if __name__ == "__main__":
	args = parse_args()
	main(args)