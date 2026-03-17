import json
import torch
import random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace

from Validation import design_space, analyze
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
	parser = ArgumentParser()

	# checkpoint directory
	parser.add_argument("--ckpt_dir", type=Path, 
		default="./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/"
	)

	# chances
	parser.add_argument("--chances", type=int, default=0)

	# nonlinear dynamic analysis simulator
	parser.add_argument("--do_nda", action="store_true", default=False)
	parser.add_argument("--check_acc", action="store_true", default=False)
	parser.add_argument("--check_disp", action="store_true", default=True)
	parser.add_argument("--graph_lstm_dir", type=Path, 
		# "./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2025_04_23__13_18_19/"
		default=None
	)
	parser.add_argument("--gm_dir", type=Path, 
		# "./NonlinearDynamicAnalysisSimulator/ground_motions/selected_ground_motions_World_processed_one_scaling_MCE/"
		default=None
	)
	parser.add_argument("--gm_num", type=int, default=11, help="ASCE says 11 is better")

	# sampling
	parser.add_argument("--random_seed", type=int, default=731)
	parser.add_argument("--sample_num", type=int, default=100)  # paper: 1000

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


def main(args):
	# set random seed
	set_random_seed(args.random_seed)

	# set device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# setup nonliear dynamic analysis simulator
	nda_simulator = None
	nda_norm_dict = None
	DBE_ground_motion_set = None
	MCE_ground_motion_set = None
	if args.do_nda:
		nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
		DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(args.gm_dir, args.gm_num, nda_norm_dict)

	# read the AI scores from file
	inference_record_path = args.ckpt_dir / f"inferencing_record_{args.chances}chance.txt"
	with open(inference_record_path, 'r') as f:
		inference_record = json.load(f)
	ai_scores = {}
	for geometry, saved_material in zip(inference_record["geometry"], inference_record["saved_material"]):
		x_span_num, z_span_num, story_num = geometry[0:3]
		geo_name = f"x{x_span_num}_z{z_span_num}_y{story_num}"
		ai_scores[geo_name] = saved_material
	print("AI scores:", ai_scores)

	# sampling and testing in the training space
	ranks_with_fail_case = {}  # key: geo_name, value: rank percentage
	ranks_without_fail_case = {}  # key: geo_name, value: rank percentage
	for x_span_num in range(2, 7):
		for z_span_num in range(2, 7):
			for story_num in range(4, 8):
				# if x_span_num != 2 or z_span_num != 2 or story_num != 4: continue
				geo_name = f"x{x_span_num}_z{z_span_num}_y{story_num}"
				print("geo_name:", geo_name)
				save_result_root = args.ckpt_dir / "sampling_results" / geo_name
				save_result_root.mkdir(parents=True, exist_ok=True)

				if len(list(save_result_root.iterdir())) > 2: continue
			
				x_span_lens = [7000 for i in range(x_span_num)]
				z_span_lens = [7000 for i in range(z_span_num)]
				sample_kwargs = {"sample_num": args.sample_num, 
								 "x_span_num": x_span_num, "x_span_lens": x_span_lens, 
								 "z_span_num": z_span_num, "z_span_lens": z_span_lens,
								 "story_num": story_num, "story_height": 3200,
								 "do_nonlinear_dynamic_analysis": args.do_nda,
								 "nda_norm_dict": nda_norm_dict, 
								 "analysis_dir": args.ckpt_dir / "Modal_Analysis"}
				check_kwargs = {"code_analysis_dir": args.ckpt_dir / "Code_Analysis",
								"do_nonlinear_dynamic_analysis": args.do_nda, 
								"nda_simulator": nda_simulator, 
								"DBE_ground_motion_set": DBE_ground_motion_set, 
								"MCE_ground_motion_set": MCE_ground_motion_set,
								"check_acceleration": args.check_acc,
								"check_displacement": args.check_disp,
								"nda_norm_dict": nda_norm_dict, "device": device, 
								"save_root": save_result_root}
				# all_kwargs = {**sample_kwargs, "check_kwargs": check_kwargs}
				# lightest_structure, heaviest_structure, structures, rewards = design_space.sample_pass_structures_from_design_space(**all_kwargs)
				# design_space.check_sample_diversity(lightest_structure, heaviest_structure, structures, rewards, save_result_root)

				# analyze the PR of the AI score 
				ai_pr_with_fail_case, ai_pr_without_fail_case = analyze.analyze_score_rank(ai_scores[geo_name], save_result_root)
				print(f"---GeoName: {geo_name}, PR (with fail case): {ai_pr_with_fail_case}, PR (without fail case): {ai_pr_without_fail_case}")
				ranks_with_fail_case[geo_name] = ai_pr_with_fail_case
				ranks_without_fail_case[geo_name] = ai_pr_without_fail_case
	
	print(f"\nfinal ranks (with fail case): \n{ranks_with_fail_case}")
	print(f"\nfinal ranks (without fail case): \n{ranks_without_fail_case}")
	save_result_root = args.ckpt_dir / "sampling_results"
	analyze.analyze_ai_ranks(save_result_root, ranks_with_fail_case, suffix=f"{args.chances}chance_WithFailCase")
	analyze.analyze_ai_ranks(save_result_root, ranks_without_fail_case, suffix=f"{args.chances}chance_NoFailCase")




if __name__ == "__main__":
	args = parse_args()
	main(args)

