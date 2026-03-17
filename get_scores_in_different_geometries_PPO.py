import json
import torch
import random
import logging
import numpy as np

from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser, Namespace

from Structure.structure import Structure
from Structure.sections import beam_sections, column_sections
from RL import agent_PPO, environment, record
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


def get_loggings(ckpt_dir, chances):
	logger = logging.getLogger(name='GraphRL')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(ckpt_dir / f"inferencing_record_{chances}chance.log")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
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
	logger = get_loggings(args.ckpt_dir, args.chances)
	logger.critical(args.ckpt_dir)

	# Set device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
	logger.critical(f"Device: {device_name}")

	# Setup nonliear dynamic analysis simulator
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
		"pretrained_model_path": args.trained_model_path,
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

	# Record
	rec = record.Record()

	# Start inferencing on different geometries
	for x_span_num in range(2, 7):
		for z_span_num in range(2, 7):
			for story_num in range(4, 8):
				x_span_len = 7000
				z_span_len = 7000
				x_span_lens = [x_span_len for i in range(x_span_num)]
				z_span_lens = [z_span_len for i in range(z_span_num)]
				story_height = 3200
				story_level_sections = [len(beam_sections)-1] * (story_num*2) + [len(column_sections)-1] * (story_num*2)
				structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
									"z_span_num": z_span_num, "z_span_lens": z_span_lens, 
									"story_num": story_num, "story_height": story_height,
									"story_level_sections": story_level_sections, 
									"add_structure_geometry": args.add_geometry_feature,
									"add_response_features": args.add_response_feature,
									"do_nonlinear_dynamic_analysis": args.do_nda, 
									"nda_norm_dict": nda_norm_dict,
									"analysis_dir": args.ckpt_dir / "Modal_Analysis"}
				structure = Structure(**structure_kwargs)
				logger.info(structure)
				env.init_records(structure)
				rec.record_in_beginning(structure, testing=False)
				graph = structure.graph.clone()

				chances = args.chances
				done = None
				timestep = 0
				accumulated_reward = 0
				while not done:
					original_structure = deepcopy(structure)
					# go through gnn and get embedding before actor-critic network
					with torch.no_grad():
						graph = graph.to(device)
						state = agent_model.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(device), None)

					dont_select_story_member_indexes = structure.restrict_action_space() if agent_model.restrict_action else []
					infeasible_actions = list(set(structure.already_minimum_section_story_indexes + dont_select_story_member_indexes))
					logits, value, entropy, action, log_prob = agent_model.choose_action(state, infeasible_actions, greedy=True)
					member_category = structure.story_level_categories[action]
					update_story = (action % structure.story_num) + 1
					print(f"story_level_sections: {structure.story_level_sections}, action: {action:3d} [{update_story}F {member_category}]")
					structure, reward, done, fail_name, fail_reason = env.step(structure, action)

					dont_select_during_cahnce_loop = []
					while done and chances > 0:
						print(f"----- Fails, current failed action: {action:3d}, current chances: {chances:3d}")

						# remove the record of current design which didn't pass the constraints
						env.saved_material_record.pop(-1)
						env.material_usage_record.pop(-1)
						if env.do_nonlinear_dynamic_analysis and "acceleration" in env.reward_type:
							env.acc_record['X-dir'].pop(-1)
							env.acc_record['Z-dir'].pop(-1)

						# if greedy will fail, choose the subgreedy
						structure = deepcopy(original_structure)

						dont_select_during_cahnce_loop.append(action)
						print(f"dont select during chance loop: {dont_select_during_cahnce_loop}")
						print(f"original minimum: {structure.already_minimum_section_story_indexes}")

						print(f"restrict actions: {structure.restrict_action_space()}")
						print(f"xdir beam: {structure.story_xdir_beam_section}")
						print(f"zdir beam: {structure.story_zdir_beam_section}")
						print(f"out col: {structure.story_outer_column_section}")
						print(f"in col: {structure.story_inner_column_section}")

						if agent_model.restrict_action:
							dont_select = list(set(dont_select_during_cahnce_loop + structure.already_minimum_section_story_indexes + structure.restrict_action_space()))
						else: 
							dont_select = list(set(dont_select_during_cahnce_loop + structure.already_minimum_section_story_indexes))
						if len(dont_select) >= len(structure.story_level_actions):
							done = True
							break
							
						logits, value, entropy, action, log_prob = agent_model.choose_action(state, dont_select, greedy=True)
						structure, reward, done, fail_name, fail_reason = env.step(structure, action)
						chances -= 1
						
					# get next state
					graph = structure.graph.clone()
					timestep += 1
					accumulated_reward += reward
					logger.info(f"timestep: {timestep:3d}, action: {action:3d}, reward: {reward:6.3f}, accumulated_reward: {accumulated_reward:6.3f}")

					# if can't select anymore, then stop
					if agent_model.restrict_action:
						dont_select = list(set(structure.already_minimum_section_story_indexes + structure.restrict_action_space()))
					else: 
						dont_select = list(set(structure.already_minimum_section_story_indexes))
					if len(dont_select) >= len(structure.story_level_actions):
						done = True
				
				total_saved_material = np.sum(env.saved_material_record)
				total_saved_percentage = accumulated_reward / env.material_usage_record[0]
				
				if env.do_nonlinear_dynamic_analysis and "acceleration" in env.reward_type:
					acc_record_x = np.array(env.acc_record['X-dir'])
					acc_record_z = np.array(env.acc_record['Z-dir'])
					total_acc_decrement = np.sum(acc_record_x[0,:] - acc_record_x[-1,:]) + np.sum(acc_record_z[0,:] - acc_record_z[-1,:])
				else:
					total_acc_decrement = 0

				logger.info(f"final_timestep: {timestep}, fail_name: {fail_name}, fail_reason: {fail_reason}")
				logger.info(f"-----x_span_num: {x_span_num}, z_span_num: {z_span_num}, story_num: {story_num}, reduced_percentage: {total_saved_percentage:.4f}, reduced_material: {total_saved_material:.4f} m3, reduced_acceleration: {total_acc_decrement:.4f} g\n")
				
				final_structure = structure if fail_reason == "minimum_section" else original_structure
				rec.record_in_end(final_structure, env, testing=False)
	
				with open(args.ckpt_dir / f"inferencing_record_{chances}chance.txt", 'w') as f: 
					json.dump(rec.training_record, f)




if __name__ == "__main__":
	args = parse_args()	
	main(args)