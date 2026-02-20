import json
import torch
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("RL/")
sys.path.append("Structure/")
sys.path.append("Visualization")
sys.path.append("NonlinearDynamicAnalysisSimulator/")

from RL import agent_PPO, environment, record
from Visualization import plot, plot_PPO
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
	parser = ArgumentParser()
 
	# comment
	parser.add_argument("--comment", type=list[str], default=[
		"reward 0 for the fail action",
		"entropy weight linear decay (1e-2 to 1e-3)",
		"testing with argmax",
		"mask logits first before softmax",
		"no consider target KL for early stopping", 
	])
 
	# pretrained model
	parser.add_argument("--pretrained_ckpt_dir", type=Path, default=None)

	# checkpoint
	parser.add_argument("--ckpt_dir", type=Path, 
		default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA"
	)
	parser.add_argument("--suffix", type=str, 
		default="PPO_MatReward_UseGAE095_LR5e-4_ActLossCoef10_CriLossCoef001_EntroWeiLinDecay_OptimEpoch5_Episode2000"
	)
	
    # nonlinear dynamic analysis simulator
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
	
    # structure
	parser.add_argument("--structure_shape", type=str, default="random", help="fixed, small_random, random")
	parser.add_argument("--add_geometry_feature", action="store_true", default=True)
	parser.add_argument("--add_response_feature", action="store_true", default=False)
	parser.add_argument("--reward_type", type=str, default="material", choices=["material", "acceleration", "displacement", "normalized", "total", "combined"])

	# model
	parser.add_argument("--hidden_dim", type=int, default=100)
	parser.add_argument("--layer_num", type=int, default=3)
	
    # training
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--use_lr_scheduler", action="store_true", default=False)
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor, 1.0, 0.99, 0.9")
	parser.add_argument("--use_gae", action="store_true", default=True, help="whether to use generalized advantage estimation")
	parser.add_argument("--gae_tau", type=float, default=0.95, help="0.9 ~ 0.97 is recommended")
	parser.add_argument("--target_kl", type=float, default=0.02)
	parser.add_argument("--clip_eps", type=float, default=0.2)
	parser.add_argument("--entropy_weight", type=float, default=0.01)
	parser.add_argument("--actor_loss_coef", type=float, default=10.0)
	parser.add_argument("--critic_loss_coef", type=float, default=0.01)
	parser.add_argument("--max_grad_norm", type=float, default=None)
	parser.add_argument("--optimization_epoch", type=int, default=5)
	parser.add_argument("--test_frequency", type=int, default=10)
	parser.add_argument("--train_episode_num", type=int, default=2000)
	parser.add_argument("--test_episode_num", type=int, default=1)
	parser.add_argument("--random_seed", type=int, default=731)
	
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
	logger = logging.getLogger(name='Graph-RL')
	logger.setLevel(level=logging.INFO)
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
	if len(args.suffix) > 2:
		args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}__{args.suffix}'
	else:
		args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)

	# Save training arguments
	with open(args.ckpt_dir / "train_args.json", "w") as f:
		args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
		json.dump(args_dict, f, indent=2)
	
	# Set logger
	logger = get_loggings(args.ckpt_dir)
	logger.critical(args.ckpt_dir)

	# Set device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
	logger.critical(f"Device: {device_name}")

	# Set up nonlinear dynamic analysis simulator
	nda_simulator = None
	nda_norm_dict = None
	DBE_ground_motion_set = None
	MCE_ground_motion_set = None
	if args.do_nda:
		nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
		DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(args.gm_dir, args.gm_num, nda_norm_dict)

	# Entropy weight decay
	linear_decay_schedule = list(np.linspace(args.entropy_weight, 1e-3, args.train_episode_num))
	constant_schedule = list(np.ones(args.train_episode_num) * args.entropy_weight)
	entropy_weight_schedule = lambda episode: linear_decay_schedule[episode]

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
		"entropy_weight_schedule": entropy_weight_schedule,
		"actor_loss_coef": args.actor_loss_coef,
		"critic_loss_coef": args.critic_loss_coef,
		"max_grad_norm": args.max_grad_norm,
		"optimization_epoch": args.optimization_epoch,
		"seed": args.random_seed,
		"logger": logger,
		"pretrained_model_path": None,
		"device": device
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

	# Record
	rec = record.Record(additional_info={
		"returns": {"train": [], "test": []},  # shape: (episode_num, optimization_epoch, timestep_num)
		"advantages": {"train": [], "test": []},  # shape: (episode_num, optimization_epoch, timestep_num)
		"entropies": {"train": [], "test": []},  # shape: (episode_num, timestep_num)
		"pred_values": [],  # shape: (episode_num, optimization_epoch, timestep_num)
		"kl_divergences": [],  # shape: (episode_num, optimization_epoch)
		"ratios": [],  # shape: (episode_num, optimization_epoch, timestep_num)
		"losses": {"actor": [], "entropy": [], "critic": []},  # shape: (episode_num, optimization_epoch)
		"grad_norms": {"gnn": [], "actor": [], "critic": []},  # shape: (episode_num, optimization_epoch)
		"param_changes": {"gnn": [], "actor": [], "critic": []},  # shape: (episode_num)
	})
	
	# Training
	_train_kwargs = {
		"agent": agent_model,
		"env": env,
		"rec": rec,
		"train_episode_num": args.train_episode_num,
		"test_frequency": args.test_frequency,
		"test_episode_num": args.test_episode_num,
		"logger": logger,
	}
	agent_PPO.train(**_train_kwargs)

	logger.critical(f"Minimum Material Usage: {np.min(rec.testing_record['final_volume']):.3f} m3, Story Level Sections: {rec.testing_record['final_design'][np.argmin(rec.testing_record['final_volume'])]}")
	logger.critical(f"Highest Score: {np.max(rec.testing_record['score']):.3f}, Story Level Sections: {rec.testing_record['final_design'][np.argmax(rec.testing_record['score'])]}")

	# Visualization
	plot.plot_reward(rec.training_record["score"], rec.testing_record["score"], args.ckpt_dir)
	plot.plot_fail_names(rec.training_record["fail_name"], rec.testing_record["fail_name"], args.ckpt_dir)
	plot.plot_fail_reasons(rec.training_record["fail_reason"], rec.testing_record["fail_reason"], args.ckpt_dir)	
	plot.plot_test_behaviors(rec, env, args.ckpt_dir)

	plot_PPO.plot_advantage(rec.advantages, args.ckpt_dir)
	plot_PPO.plot_entropy(rec.entropies, args.ckpt_dir)
	plot_PPO.plot_explained_variance(rec.returns["train"], rec.pred_values, args.ckpt_dir)
	plot_PPO.plot_kl_divergence(rec.kl_divergences, args.target_kl, args.ckpt_dir)
	plot_PPO.plot_clip_fraction(rec.ratios, args.clip_eps, args.ckpt_dir)
	plot_PPO.plot_loss(rec.losses["actor"], "actor", args.ckpt_dir)
	plot_PPO.plot_loss(rec.losses["entropy"], "entropy", args.ckpt_dir)
	plot_PPO.plot_loss(rec.losses["critic"], "critic", args.ckpt_dir)
	plot_PPO.plot_grad_norm(rec.grad_norms, args.ckpt_dir)
	plot_PPO.plot_param_change(rec.param_changes, args.ckpt_dir)




if __name__ == "__main__":
	args = parse_args()
	main(args)