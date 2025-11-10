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

from RL import agent, environment, record
from Visualization import plot, visualize
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
	parser = ArgumentParser()
 
	# comment
	parser.add_argument("--comment", type=str, default="reward 0 for the fail action, proportional prioritized buffer (alpha=1.0, beta-annealing for rate = 5e-3), target Q-network is detached")
 
	# pretrained model
	parser.add_argument("--pretrained_ckpt_dir", type=Path, default=None)

	# checkpoint
	parser.add_argument("--ckpt_dir", type=Path, default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA")
	parser.add_argument("--suffix", type=str, default="TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000")

	# nonlinear dynamic analysis simulator
	parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
	parser.add_argument("--check_acceleration", action="store_true", default=False)
	parser.add_argument("--check_displacement", action="store_true", default=True)
	parser.add_argument("--graph_lstm_dir", type=Path, default=None)  # "./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2025_05_19__22_59_28/"
	parser.add_argument("--ground_motion_dir", type=Path, default=None)  # "./NonlinearDynamicAnalysisSimulator/ground_motions/selected_ground_motions_World_processed_one_scaling_MCE/"
	parser.add_argument("--ground_motion_number", type=int, default=11, help="ASCE says 11 is better")

	# structure
	parser.add_argument("--structure_shape", type=str, default="random", help="fixed, small_random, random")
	parser.add_argument("--add_structure_geometry", action="store_true", default=True)
	parser.add_argument("--add_response_features", action="store_true", default=True)
	parser.add_argument("--reward_type", type=str, default="material", help="material, acceleration, displacement, normalized, total, combined")
	parser.add_argument("--restrict_action", action="store_true", default=False)
	parser.add_argument("--scwb_driven_design", action="store_true", default=False)

	# model
	parser.add_argument("--model_type", type=str, default="Taiwan", help="Taiwan, Japan")
	parser.add_argument("--hidden_dim", type=int, default=100)
	parser.add_argument("--num_layers", type=int, default=3)

	# buffer
	parser.add_argument("--buffer_size", type=int, default=10000)
	parser.add_argument("--update_frequency", type=int, default=1)
	parser.add_argument("--add_experience_frequency", type=int, default=1)

	# training
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor, 1.0, 0.99, 0.9")
	parser.add_argument("--epsilon", type=float, default=0.99, help="epsilon decay factor")
	parser.add_argument("--synchronize_steps", type=int, default=None)
	parser.add_argument("--soft_update_alpha", type=float, default=1e-3)
	parser.add_argument("--test_frequency", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=256)  # original: 256
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--num_epoch", type=int, default=1000, help="epoch == episode")
	parser.add_argument("--random_seed", type=int, default=731, help="fixed random seed")

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
	# set random seed
	set_random_seed(args.random_seed)

	# set checkpoint directory
	if len(args.suffix) > 2:
		args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}__{args.suffix}'
	else:
		args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)

	# set logger
	logger = get_loggings(args.ckpt_dir)
	logger.critical(args.ckpt_dir)
	logger.critical(args)

	# set device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
	logger.critical(f"Device: {device_name}")

	# setupt nonliear dynamic analysis simulator
	nda_simulator = None
	nda_norm_dict = None
	DBE_ground_motion_set = None
	MCE_ground_motion_set = None
	if args.do_nonlinear_dynamic_analysis:
		nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
		DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(args.ground_motion_dir, args.ground_motion_number, nda_norm_dict)


	# Beta-annealing schedule
	def exponential_annealing_schedule(n, rate=0.02):
		return 1 - np.exp(-rate * n)
	beta_annealing_schedule = lambda n: exponential_annealing_schedule(n, 0.02)

	# Power decay schedule
	def power_decay_schedule(episode_number: int, decay_factor: float, minimum_epsilon: float=1e-2) -> float:
		return max(decay_factor ** episode_number, minimum_epsilon)
	epsilon_decay_schedule = lambda n: power_decay_schedule(n, args.epsilon, 1e-2)

	# Linear decay schedule
	def linear_decay_schedule(episode_number: int, total_episode: int, minimum_epsilon: float=1e-1) -> float:
		return max(1.0 - episode_number/total_episode, minimum_epsilon)
	straight_decay_schedule = lambda n: linear_decay_schedule(n, args.num_epoch, 1e-1)

	# Cosine decay schedule
	def cosine_decay_schedule(episode_number: int, total_episode: int, minimum_epsilon: float=1e-1) -> float:
		linear_decay = 1.0 - episode_number / total_episode
		cosine_decay =  0.75 * linear_decay + 0.25 * linear_decay * np.cos(np.pi / 100 * episode_number)
		return max(cosine_decay, minimum_epsilon)
	periodic_decay_schedule = lambda n: cosine_decay_schedule(n, args.num_epoch, 1e-1)

	# Constant epsilon schedule (Japan: RL for 2D frame)
	def constant_epsilon_schedule(episode_number: int, constant_epsilon: float=1e-1) -> float:
		return constant_epsilon
	fixed_epsilon_schedule = lambda n: constant_epsilon_schedule(n, 1e-1)


	# Agent
	node_feature_dim = 8 if args.add_structure_geometry else 5
	edge_feature_dim = 13 if args.add_response_features else 11
	_agent_kwargs = {
		"node_feature_dim": node_feature_dim,
		"edge_feature_dim": edge_feature_dim,
		"hidden_dim": args.hidden_dim,
		"num_layers": args.num_layers,
		"batch_size": args.batch_size,
		"lr": args.lr,
		"buffer_size": args.buffer_size,
		"epsilon_decay_schedule": straight_decay_schedule,
		"synchronize_steps": args.synchronize_steps,
		"soft_update_alpha": args.soft_update_alpha,
		"gamma": args.gamma,
		"update_frequency": args.update_frequency,
		"add_experience_frequency": args.add_experience_frequency,
		"test_frequency": args.test_frequency,
		"restrict_action": args.restrict_action,
		"seed": args.random_seed,
		"logger": logger,
		"pretrained_ckpt_dir": args.pretrained_ckpt_dir,
		"device": device,
	}
	if args.model_type == "Taiwan":
		double_dqn_agent = agent.DeepQAgent(**_agent_kwargs)
	elif args.model_type == "Japan":
		double_dqn_agent = agent.JapanDeepQAgent(**_agent_kwargs)

	# Environment
	_env_kwargs = {
		"structure_shape": args.structure_shape,
		"add_structure_geometry": args.add_structure_geometry,
		"add_response_features": args.add_response_features,
		"reward_type": args.reward_type,
		"scwb_driven_design": args.scwb_driven_design,
		"do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
		"check_acceleration": args.check_acceleration,
		"check_displacement": args.check_displacement,
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
	rec = record.Record()
	
	# Training the DeepQAgent using Double DQN
	_train_kwargs = {
		"agent": double_dqn_agent,
		"env": env,
		"rec": rec,
		"number_episodes": args.num_epoch,
		"logger": logger,
	}


	agent.train(**_train_kwargs)

	logger.critical(f"Minimum Material Usage: {np.min(rec.testing_record['final_volume']):.3f} m3, Story Level Sections: {rec.testing_record['final_design'][np.argmin(rec.testing_record['final_volume'])]}")
	logger.critical(f"Highest Score: {np.max(rec.testing_record['score']):.3f}, Story Level Sections: {rec.testing_record['final_design'][np.argmax(rec.testing_record['score'])]}")

	plot.plot_reward(rec.training_record["score"], rec.testing_record["score"], args.ckpt_dir)
	plot.plot_loss(rec.learn_losses, args.ckpt_dir)
	plot.plot_Qvalues(rec.Q_values, args.ckpt_dir)
	plot.plot_fail_names(rec.training_record["fail_name"], rec.testing_record["fail_name"], args.ckpt_dir)
	plot.plot_fail_reasons(rec.training_record["fail_reason"], rec.testing_record["fail_reason"], args.ckpt_dir)	
	plot.plot_test_behaviors(rec, env, args.ckpt_dir)

	# inference
	#visualize.visualize_design_process(double_dqn_agent, env, logger, args.ckpt_dir, testing_structure=True)
	#visualize.visualize_design_process(double_dqn_agent, env, logger, args.ckpt_dir, taller_structure=True)
	#visualize.visualize_edge_embedding(double_dqn_agent, env, logger, args.ckpt_dir)




if __name__ == "__main__":
	args = parse_args()
	main(args)

