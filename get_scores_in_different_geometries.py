import json
import torch
import random
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser, Namespace

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("RL/")
sys.path.append("Structure/")
sys.path.append("NonlinearDynamicAnalysisSimulator/")

from RL import agent, environment, new_strategy, record
from Structure.sections import beam_sections, column_sections
from Structure.structure import Structure
from Structure import pisa, check
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
	parser = ArgumentParser()
 
	# trained model path
	# without doNDA: "./Results/AdjustedSections/2023_05_06__21_36_03__3d_storyLevel_random_shape_addYfeature_epoch_1000_buffer_10000_batch_size_256_gamma_099/model.pt"
	# with doNDA: "./Results/AdjustedSections/2023_05_07__11_39_51__3d_storyLevel_random_shape_doNDA_addYfeature_epoch_1000_buffer_10000_batch_size_256_gamma_099/model.pt"
	parser.add_argument("--trained_model_path", type=Path, default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2025_06_05__21_45_28__TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000/models/model_HighestScore.pt")
	# checkpoint directory
	parser.add_argument("--ckpt_dir", type=Path, default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2025_06_05__21_45_28__TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000")

	# chances
	parser.add_argument("--chances", type=int, default=0)

	# nonlinear dynamic analysis simulator
	parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
	parser.add_argument("--check_acceleration", action="store_true", default=False)
	parser.add_argument("--check_displacement", action="store_true", default=True)
	# RelAcc: "./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2023_07_20__15_43_32/"
	# AbsAcc: "./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2023_07_20__22_46_09/"
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
	parser.add_argument("--batch_size", type=int, default=256)
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


def get_loggings(ckpt_dir, chances):
	logger = logging.getLogger(name='Graph-RL')
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
	# set random seed
	set_random_seed(args.random_seed)

	# set logger
	logger = get_loggings(args.ckpt_dir, args.chances)
	logger.critical(args.ckpt_dir)
	logger.critical(args)

	# set device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
	logger.critical(f"Device: {device_name}")

	# setup nonliear dynamic analysis simulator
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

	# Power-decay schedule
	def power_decay_schedule(episode_number: int, decay_factor: float, minimum_epsilon: float=1e-2) -> float:
		"""Power decay schedule found in other practical applications."""
		return max(decay_factor ** episode_number, minimum_epsilon)
	epsilon_decay_schedule = lambda n: power_decay_schedule(n, args.epsilon, 1e-2)

	# Linear-decay schedule
	def linear_decay_schedule(episode_number: int, total_episode: int, minimum_epsilon: float=1e-1) -> float:
		return max(1.0 - episode_number/total_episode, minimum_epsilon)
	straight_decay_schedule = lambda n: linear_decay_schedule(n, args.num_epoch, 1e-1)

	# Cosine-decay schedule
	def cosine_decay_schedule(episode_number: int, total_episode: int, minimum_epsilon: float=1e-1) -> float:
		linear_decay = 1.0 - episode_number / total_episode
		cosine_decay =  0.75 * linear_decay + 0.25 * linear_decay * np.cos(np.pi / 100 * episode_number)
		return max(cosine_decay, minimum_epsilon)
	periodic_decay_schedule = lambda n: cosine_decay_schedule(n, args.num_epoch, 1e-1)

	# Constant-epsilon schedule (Japan: RL for 2D frame)
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
		"pretrained_ckpt_dir": args.trained_model_path,
		"device":device,
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


	# start inferencing on different geometries
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
									"add_structure_geometry": args.add_structure_geometry,
									"add_response_features": args.add_response_features,
									"do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis, 
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
					# go through gnn and get embedding before q-network
					with torch.no_grad():
						graph = graph.to(device)
						state = double_dqn_agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(device), None)

					# select action and update structure
					# print(f"original minimum: {structure.already_minimum_section_story_indexes}")
					# print(f"restrict actions: {structure.restrict_action_space()}")
					# print(f"xdir beam: {structure.story_xdir_beam_section}")
					# print(f"zdir beam: {structure.story_zdir_beam_section}")
					# print(f"out col: {structure.story_outer_column_section}")
					# print(f"in col: {structure.story_inner_column_section}")

					dont_select_story_member_indexes = structure.restrict_action_space() if double_dqn_agent.restrict_action else []
					action, _ = double_dqn_agent.choose_action(state, 
															   structure.already_minimum_section_story_indexes, 
															   dont_select_story_member_indexes,
															   greedy=True)
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

						#structure.already_minimum_section_story_indexes.append(action)
						dont_select_during_cahnce_loop.append(action)
						print(f"dont select during chance loop: {dont_select_during_cahnce_loop}")
						print(f"original minimum: {structure.already_minimum_section_story_indexes}")

						print(f"restrict actions: {structure.restrict_action_space()}")
						print(f"xdir beam: {structure.story_xdir_beam_section}")
						print(f"zdir beam: {structure.story_zdir_beam_section}")
						print(f"out col: {structure.story_outer_column_section}")
						print(f"in col: {structure.story_inner_column_section}")

						if double_dqn_agent.restrict_action:
							dont_select = list(set(dont_select_during_cahnce_loop + structure.already_minimum_section_story_indexes + structure.restrict_action_space()))
						else: 
							dont_select = list(set(dont_select_during_cahnce_loop + structure.already_minimum_section_story_indexes))

						if len(dont_select) >= len(structure.story_level_actions):
							done = True
							break
							
						dont_select_story_member_indexes = structure.restrict_action_space() if double_dqn_agent.restrict_action else []
						action, _ = double_dqn_agent.choose_action(state, 
												 				   dont_select, 
																   dont_select_story_member_indexes,
																   greedy=True)
						structure, reward, done, fail_name, fail_reason = env.step(structure, action)
						chances -= 1
						
					# get next state
					graph = structure.graph.clone()
					timestep += 1
					accumulated_reward += reward
					logger.info(f"timestep: {timestep}, accumulated_reward: {accumulated_reward}\n")

					# if can't select anymore, then stop
					if double_dqn_agent.restrict_action:
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


def inference_record_to_pisa_ipt(args):
	ckpt_dir = Path("./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2025_06_05__21_45_28__TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000")
	chances = 0
	inference_record_path = ckpt_dir / f"inferencing_record_{chances}chance.txt"
	with open(inference_record_path, 'r') as f:
		inference_record = json.load(f)

	x_span_num = 6  # 2 - 6
	z_span_num = 6  # 2 - 6
	story_num = 7   # 4 - 7
	for index, geo_info in enumerate(inference_record["geometry"]):
		if geo_info[:3] == [x_span_num, z_span_num, story_num]:
			x_span_lens = geo_info[3]
			z_span_lens = geo_info[4]
			story_height = geo_info[5]
			break

	geo_name = f"x{x_span_num}z{z_span_num}y{story_num}"
	final_design = inference_record["final_design"][index]
	structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
						"z_span_num": z_span_num, "z_span_lens": z_span_lens, 
						"story_num": story_num, "story_height": story_height,
						"story_level_sections": None, 
						"add_structure_geometry": False,
						"add_response_features": False,
						"do_nonlinear_dynamic_analysis": False, 
						"nda_norm_dict": None,
						"analysis_dir": ckpt_dir/"Modal_Analysis"}
	structure = Structure(**structure_kwargs)
	print(structure)
	print(structure.calculate_material_usage(), inference_record["initial_volume"][index])
	print(structure.story_level_sections, inference_record["initial_design"][index])

	reward = 0
	for action in inference_record["action"][index]:
		material_saved = structure.update_action(action)
		reward += material_saved
	print(reward, inference_record["saved_material"][index])
	print(structure.calculate_material_usage(), inference_record["final_volume"][index])
	print(structure.story_level_sections, final_design)
	assert structure.story_level_sections == final_design, "The final design does not match the inference record."

	load_cases, static_responses = check.get_response(structure, ckpt_dir/"Code_Analysis")    
	static_constraint_condition, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
	whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, check_displacement=True)

	save_ipt_path = ckpt_dir / f"final_design_{geo_name}_{chances}chance.ipt"
	pisa._generate_analysis_ipt(structure, save_ipt_path, analysis="modal")




if __name__ == "__main__":
	args = parse_args()
	
	# main(args)

	inference_record_to_pisa_ipt(args)
