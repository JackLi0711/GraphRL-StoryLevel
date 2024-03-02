import os
import torch
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

from Validation import design_space, check_design, analyze
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
	parser = ArgumentParser()

	# trained model
	#parser.add_argument("--ckpt_dir", type=Path, default="./Results/3d_random/2023_05_06__21_36_03__3d_storyLevel_random_shape_addYfeature_epoch_1000_buffer_10000_batch_size_256_gamma_099/")
	#parser.add_argument("--ckpt_dir", type=Path, default="./Results/3d_random/2023_05_07__11_39_51__3d_storyLevel_random_shape_doNDA_addYfeature_epoch_1000_buffer_10000_batch_size_256_gamma_099/")
	
	parser.add_argument("--ckpt_dir", type=Path, default="./Results/MaterialReward/")  # material
	#parser.add_argument("--ckpt_dir", type=Path, default="./Results/AccelerationReward/")  # acceleration

	# chances
	parser.add_argument("--chances", type=int, default=2)

	# nonlinear dynamic analysis simulator
	parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
	parser.add_argument("--check_acceleration", action="store_true", default=False)
	parser.add_argument("--check_displacement", action="store_true", default=True)
	
	#parser.add_argument("--graph_lstm_dir", type=Path, default="./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2023_07_20__15_43_32/")  # RelAcc
	parser.add_argument("--graph_lstm_dir", type=Path, default="./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2023_07_20__22_46_09/")  # AbsAcc
	parser.add_argument("--ground_motion_dir", type=Path, default="./NonlinearDynamicAnalysisSimulator/ground_motions/selected_ground_motions_MCE/")
	parser.add_argument("--ground_motion_number", type=int, default=11, help='ASCE says 11 is better')

	# structure
	parser.add_argument("--structure_shape", type=str, default='random', help='fixed, small_random, random')
	parser.add_argument("--add_structure_geometry", action="store_true", default=True)
	parser.add_argument("--reward_type", type=str, default='material', help="material, acceleration, displacement, normalized")
	parser.add_argument("--restrict_action", action="store_true", default=True)
	
	# sample number
	parser.add_argument("--sample_num", type=int, default=200)  # paper: 1000

	args = parser.parse_args()
	return args




def main(args):
	# set device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# setupt nonliear dynamic analysis simulator
	nda_simulator = None
	nda_norm_dict = None
	DBE_ground_motion_set = None
	MCE_ground_motion_set = None
	if args.do_nonlinear_dynamic_analysis:
		nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
		DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(args.ground_motion_dir, args.ground_motion_number, nda_norm_dict)


	'''
	# evaluate the testing structure
	# get sampled structures from design space
	analysis_dir = args.ckpt_dir / "Validate_Analysis"
	analysis_dir.mkdir(parents=True, exist_ok=True)
	sample_kwargs = {"sample_num": args.sample_num, "do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
					 "nda_norm_dict": nda_norm_dict, "analysis_dir": analysis_dir}
	print("---Generating structures:", args.sample_num)
	structures, rewards = design_space.sample_structures_from_design_space(**sample_kwargs)


	# check each sampled structure
	save_result_root = args.ckpt_dir / "Validate_Results"
	save_result_root.mkdir(parents=True, exist_ok=True)
	check_kwargs = {"structures": structures, "rewards": rewards, "code_analysis_dir": analysis_dir,
					"do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
					"nda_simulator": nda_simulator, "ground_motion_set": ground_motion_set,
					"nda_norm_dict": nda_norm_dict, "device": device, "save_root": save_result_root}
	print("---Checking generated structures:")
	check_design.check_designs_from_design_space(**check_kwargs)

	# analyze the design space scores
	save_result_root = args.ckpt_dir / "Validate_Results"
	analyze.analyze_design(save_result_root)
	'''


	# '''
	# read the ai scores from file
	ai_record_path = args.ckpt_dir / f"testing_record_{args.chances}chance.log"
	ai_scores = {}
	for line in open(ai_record_path, 'r').readlines():
		if "-----" in line:
			content = line.split()
			# print(content)
			x_span_num = int(content[8].replace(",", ""))
			z_span_num = int(content[10].replace(",", ""))
			story_num = int(content[12].replace(",", ""))

			reduced_material = float(content[-5])
			reduced_acceleration = float(content[-2])
			if "material" in args.reward_type: score = reduced_material
			elif "acceleration" in args.reward_type: score = reduced_acceleration
			
			geo_name = f"x{x_span_num}_z{z_span_num}_y{story_num}"
			ai_scores[geo_name] = score
	print("AI scores:", ai_scores)


	# testing in the training space
	analysis_dir = args.ckpt_dir / "Test_TrainingSpace_Analysis"
	analysis_dir.mkdir(parents=True, exist_ok=True)
	ranks = {}	# key: geo_name, value: rank percentage

	for story_num in range(4, 8):
		for x_span_num in range(2, 7):
			for z_span_num in range(2, 7):

				geo_name = f"x{x_span_num}_z{z_span_num}_y{story_num}"
				print("geo_name:", geo_name)
				save_result_root = args.ckpt_dir / "Validate_TrainingSpace_Results" / geo_name
				save_result_root.mkdir(parents=True, exist_ok=True)
			
				x_span_len = 7000
				z_span_len = 7000
				x_span_lens = [x_span_len for i in range(x_span_num)]
				z_span_lens = [z_span_len for i in range(z_span_num)]

				sample_kwargs = {"sample_num": args.sample_num, "do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
								"nda_norm_dict": nda_norm_dict, "analysis_dir": analysis_dir,
								"x_span_num": x_span_num, "z_span_num": z_span_num, "story_num": story_num,
								"x_span_len": x_span_len, "z_span_len": z_span_len,
								"x_span_lens": x_span_lens, "z_span_lens": z_span_lens}
				
				# '''
				print("---Generating structures:", args.sample_num)
				structures, rewards = design_space.sample_structures_from_design_space(**sample_kwargs)


				# check each sampled structure				
				check_kwargs = {"structures": structures, "rewards": rewards, 
		    					"code_analysis_dir": analysis_dir,
								"do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis, 
								"nda_simulator": nda_simulator, 
								"DBE_ground_motion_set": DBE_ground_motion_set, 
								"MCE_ground_motion_set": MCE_ground_motion_set,
								"check_acceleration": args.check_acceleration,
								"check_displacement": args.check_displacement,
								"nda_norm_dict": nda_norm_dict, "device": device, 
								"save_root": save_result_root}
				print("---Checking generated structures:")
				check_design.check_designs_from_design_space(**check_kwargs)
				# '''


				# analyze rank
				ai_rank_percentage = analyze.analyze_design_rank(ai_scores[geo_name], save_result_root)
				print(f"---GeoName: {geo_name}, rank_percentage: {ai_rank_percentage}")
				ranks[geo_name] = ai_rank_percentage
	
	print()
	print("final ranks:")
	print(ranks)
	save_result_root = args.ckpt_dir / "Validate_TrainingSpace_Results"
	analyze.analyze_ai_ranks(save_result_root, ranks, args.chances)
	# '''


	

if __name__ == "__main__":
	args = parse_args()
	main(args)

