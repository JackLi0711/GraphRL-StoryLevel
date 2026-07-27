import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch_geometric.loader import DataLoader

import sys
sys.path.append("../")
from pathlib import Path
from Structure import pisa, earthquake, check, check_nda
from Structure import structure as struct
from Structure.structure import Structure
from NonlinearDynamicAnalysisSimulator import load_simulator
from Validation import accuracy, normalization, opensees


result_dir = Path("./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA")
static_model_folder  = "2026_03_10__13_19_18__PPO_MatReward_UseGAE095_LR5e-4_ActLossCoef10_CriLossCoef001_EntroWei01to001_OptimEpoch5_Episode1000"
dynamic_model_folder = "2026_03_17__17_12_00__PPO_MatReward_doNDA_UseGAE095_LR5e-4_ActLossCoef10_CriLossCoef001_EntroWei01to001_OptimEpoch5_Episode1000"
model_date           = [static_model_folder.split("__")[0], static_model_folder.split("__")[1], dynamic_model_folder.split("__")[0], dynamic_model_folder.split("__")[1]]
model_setting        = "__".join(model_date)

strucutre_type       = "testing"  # testing, taller, random, x2z2y4, x6z6y7
gm_level             = "World_processed_one_scaling_MCE"  # DBE, MCE, MCEx2, World_processed_one_scaling_MCE, TAP3_two_scaling_MCE
chance               = 0  # 0, 1, 2

working_dir = Path("./Validation/Final_Design_Comparison") / model_setting / strucutre_type / gm_level
graph_lstm_dir = Path("./NonlinearDynamicAnalysisSimulator/trained_GraphLSTM") / "2025_05_19__22_59_28"
ground_motion_dir = Path("./NonlinearDynamicAnalysisSimulator/ground_motions") / "selected_ground_motions_World_processed_one_scaling_MCE"
ground_motion_number = 11
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_structure(geo_name: str):
    if geo_name == "testing":
        x_span_num = 4
        z_span_num = 4
        x_span_len = 6000
        z_span_len = 8000
        x_span_lens = [x_span_len for i in range(x_span_num)]
        z_span_lens = [z_span_len for i in range(z_span_num)]
        story_num = 6
        story_height = 3200
    else:
        geometry = [int(char) for char in geo_name if char.isdigit()]
        x_span_num = geometry[0]
        z_span_num = geometry[1]
        story_num = geometry[2]
        x_span_len = 7000
        z_span_len = 7000
        x_span_lens = [x_span_len for i in range(x_span_num)]
        z_span_lens = [z_span_len for i in range(z_span_num)]
        story_height = 3200

    analysis_dir = Path("./check/Analysis/")
    structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens,
                        "z_span_num": z_span_num, "z_span_lens": z_span_lens,
                        "story_num": story_num, "story_height": story_height,
                        "story_level_sections": None,
                        "add_structure_geometry": False,
                        "add_response_features": False,
                        "do_nonlinear_dynamic_analysis": True,
                        "nda_norm_dict": None,
                        "analysis_dir": analysis_dir}
    structure = struct.Structure(**structure_kwargs)

    return structure


# plastic hinge
Myield_yn_index = 28
Myield_yp_index = 30
output_My_yn_index, output_My_yp_index = 8, 9
output_Mz_yn_index, output_Mz_yp_index = 14, 15
yield_factor = 0.90
def get_plastic_hinge_occurrence(structure: Structure, feature: np.ndarray, response: np.ndarray):
    plastic_hinge_occurrence = np.zeros((structure.node_number, 4))  # 4: My_yn, My_yp, Mz_yn, Mz_yp
    hinge_My_yn = (np.max(np.abs(response[structure.hinge_node_yn, :, output_My_yn_index]), axis=1) >= yield_factor * feature[structure.hinge_node_yn, Myield_yn_index])
    hinge_My_yp = (np.max(np.abs(response[structure.hinge_node_yp, :, output_My_yp_index]), axis=1) >= yield_factor * feature[structure.hinge_node_yp, Myield_yp_index])
    hinge_Mz_yn = (np.max(np.abs(response[structure.hinge_node_yn, :, output_Mz_yn_index]), axis=1) >= yield_factor * feature[structure.hinge_node_yn, Myield_yn_index])
    hinge_Mz_yp = (np.max(np.abs(response[structure.hinge_node_yp, :, output_Mz_yp_index]), axis=1) >= yield_factor * feature[structure.hinge_node_yp, Myield_yp_index])

    plastic_hinge_occurrence[structure.hinge_node_yn, 0] = hinge_My_yn
    plastic_hinge_occurrence[structure.hinge_node_yp, 1] = hinge_My_yp
    plastic_hinge_occurrence[structure.hinge_node_yn, 2] = hinge_Mz_yn
    plastic_hinge_occurrence[structure.hinge_node_yp, 3] = hinge_Mz_yp

    return (np.sum(plastic_hinge_occurrence, axis=1) > 0).astype(int)  # shape: (node_num,)

def get_story_level_plastic_hinge_occurrence(structure: Structure, scenario="static"):
    target_folder = working_dir / scenario
    gm_num = len(os.listdir(target_folder))
    story_plastic_hinge_occurrence = np.zeros((gm_num, len(structure.y_grid)))

    for i, gm_name in enumerate(os.listdir(target_folder)[:gm_num]):
        graph_path = target_folder / gm_name / "structure_graph_NodeAsNode.pt"
        graph = torch.load(graph_path)
        feature, response = graph.x.numpy(), graph.y.numpy()
        plastic_hinge_occurrence = get_plastic_hinge_occurrence(structure, feature, response)
        for story, y in enumerate(structure.y_grid):
            if story == 0: continue
            story_plastic_hinge_occurrence[i, story] = (np.sum(plastic_hinge_occurrence[structure.story_nodes[f"{story}F"]]) > 0).astype(int)
        
    return story_plastic_hinge_occurrence

def plot_story_level_plastic_hinge_occurrence(plastic_hinge_occurrence_static, plastic_hinge_occurrence_dynamic):
    storys = np.arange(len(plastic_hinge_occurrence_static)) + 1
    limit = 11 / 2
    
    fig = plt.figure(figsize=(12, 6), facecolor="w")
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(plastic_hinge_occurrence_static, storys, color="black", label="RSA", linewidth=3)
    ax.axvline(x=limit, color="red", label=f"{limit = }", linestyle="dashed")
    ax.set_title("plastic hinge occurrence (RSA)", fontsize=18)
    ax.set_xlabel("Number of ground motion", fontsize=16)
    ax.set_ylabel("Floor", fontsize=16)
    ax.grid(axis='x')
    ax.tick_params(labelsize=14)
    ax.legend(loc="best", fontsize=16)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(plastic_hinge_occurrence_dynamic, storys, color="blue", label="RSA+NRHA", linewidth=3)
    ax.axvline(x=limit, color="red", label=f"{limit = }", linestyle="dashed")
    ax.set_title("plastic hinge occurrence (RSA+NRHA)", fontsize=18)
    ax.set_xlabel("Number of ground motion", fontsize=16)
    ax.set_ylabel("Floor", fontsize=16)
    ax.grid(axis='x')
    ax.tick_params(labelsize=14)
    ax.legend(loc="best", fontsize=16)

    plt.tight_layout()
    plt.savefig(working_dir / "plastic_hinge_occurrence.png")  # , dpi=300, bbox_inches='tight'

def compare_plastic_hinge_occurrence():
    structure = generate_structure(strucutre_type)
    print(structure, structure.story_level_sections)

    story_plastic_hinge_occurrence_static = get_story_level_plastic_hinge_occurrence(structure, scenario="static")  # shape: (gm_num, len(structure.y_grid))
    story_plastic_hinge_occurrence_dynamic = get_story_level_plastic_hinge_occurrence(structure, scenario="dynamic")  # shape: (gm_num, len(structure.y_grid))
    gm_sum_story_plastic_hinge_occurrence_static = story_plastic_hinge_occurrence_static.sum(axis=0)  # shape: (len(structure.y_grid),)
    gm_sum_story_plastic_hinge_occurrence_dynamic = story_plastic_hinge_occurrence_dynamic.sum(axis=0)  # shape: (len(structure.y_grid),)

    plot_story_level_plastic_hinge_occurrence(gm_sum_story_plastic_hinge_occurrence_static, gm_sum_story_plastic_hinge_occurrence_dynamic)


# drift ratio (member-wise)
def get_drift_ratio(structure: Structure, response: np.ndarray):
    node_disp_x = response[:, :, 4]
    node_disp_z = response[:, :, 5]

    member_lengths = np.array(list(structure.member_length_dict.values())) * 1e+3  # unit conversion: m --> mm
    member_lengths = member_lengths[:, np.newaxis]  # shape: (member_num, 1)
    timeseries_length = node_disp_x.shape[-1]
    member_lengths = np.tile(member_lengths, (1, timeseries_length))  # shape: (member_num, timesteps)

    member_end_nodes = np.array(list(structure.member_to_nodeIndex_dict.values()))[:, 0:2]  # shape: (member_num, 2), 2: node1_index, node2_index
    timeseries_drift_ratio_x = np.abs((node_disp_x[member_end_nodes[:, 1], :] - node_disp_x[member_end_nodes[:, 0], :]) / member_lengths)  # shape: (member_num, timesteps)
    timeseries_drift_ratio_z = np.abs((node_disp_z[member_end_nodes[:, 1], :] - node_disp_z[member_end_nodes[:, 0], :]) / member_lengths)  # shape: (member_num, timesteps)

    return timeseries_drift_ratio_x, timeseries_drift_ratio_z

def get_story_level_drift_ratio(structure: Structure, scenario="static"):
    target_folder = working_dir / scenario
    gm_num = len(os.listdir(target_folder))
    story_mean_drift_ratio = np.zeros((gm_num, structure.story_num, 2))  # 2: X-dir., Z-dir.
    story_peak_drift_ratio = np.zeros((gm_num, structure.story_num, 2))  # 2: X-dir., Z-dir.

    for i, gm_name in enumerate(os.listdir(target_folder)[:gm_num]):
        graph_path = target_folder / gm_name / "structure_graph_NodeAsNode.pt"
        graph = torch.load(graph_path)
        feature, response = graph.x.numpy(), graph.y.numpy()
        timeseries_drift_ratio_x, timeseries_drift_ratio_z = get_drift_ratio(structure, response)  # shape: (member_num, timesteps)
        for j in range(structure.story_num):
            story_drift_ratio_x = np.mean(timeseries_drift_ratio_x[structure.story_column_member[j], :], axis=0)  # shape: (timesteps,)
            story_drift_ratio_z = np.mean(timeseries_drift_ratio_z[structure.story_column_member[j], :], axis=0)  # shape: (timesteps,)
            story_mean_drift_ratio[i, j, 0] = np.max(story_drift_ratio_x)
            story_mean_drift_ratio[i, j, 1] = np.max(story_drift_ratio_z)

            story_drift_ratio_x = np.max(timeseries_drift_ratio_x[structure.story_column_member[j], :], axis=1)  # shape: (member_num,)
            story_drift_ratio_z = np.max(timeseries_drift_ratio_z[structure.story_column_member[j], :], axis=1)  # shape: (member_num,)
            story_peak_drift_ratio[i, j, 0] = np.max(story_drift_ratio_x)
            story_peak_drift_ratio[i, j, 1] = np.max(story_drift_ratio_z)

    return story_mean_drift_ratio, story_peak_drift_ratio
    
def plot_story_level_drift_ratio(drift_ratio_static, drift_ratio_dynamic=None, type="mean"):
    storys = np.arange(1, len(drift_ratio_static)+1) + 1
    if type == "mean":
        limit = check_nda.NDA_MEAN_DRIFT_RATIO_LIMIT
    elif type == "peak":
        limit = check_nda.NDA_PEAK_DRIFT_RATIO_LIMIT
    
    fig = plt.figure(figsize=(12, 6), facecolor="w")
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(drift_ratio_static[:, 0], storys, color="black", label="RSA", linewidth=3)
    if drift_ratio_dynamic is not None:
        ax.plot(drift_ratio_dynamic[:, 0], storys, color="blue", label="RSA+NRHA", linewidth=3)
    ax.axvline(x=limit, color="red", label=f"{limit = }", linestyle="dashed")
    ax.set_title(f"X-dir. gm-{type} story-peak drift ratio", fontsize=18)
    ax.set_xlabel("Drift ratio", fontsize=16)
    ax.set_ylabel("Floor", fontsize=16)
    ax.grid(axis='y')
    ax.set_xlim(0, limit+0.01)
    ax.tick_params(labelsize=14)
    ax.legend(loc="best", fontsize=16)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(drift_ratio_static[:, 1], storys, color="black", label="RSA", linewidth=3)
    if drift_ratio_dynamic is not None:
        ax.plot(drift_ratio_dynamic[:, 1], storys, color="blue", label="RSA+NRHA", linewidth=3)
    ax.axvline(x=limit, color="red", label=f"{limit = }", linestyle="dashed")
    ax.set_title(f"Z-dir. gm-{type} story-peak drift ratio", fontsize=18)
    ax.set_xlabel("Drift ratio", fontsize=16)
    ax.set_ylabel("Floor", fontsize=16)
    ax.grid(axis='y')
    ax.set_xlim(0, limit+0.01)
    ax.tick_params(labelsize=14)
    ax.legend(loc="best", fontsize=16)

    plt.tight_layout()
    plt.savefig(working_dir / f"gm_{type}_story_peak_drift_ratio.png")  # , dpi=300, bbox_inches='tight'

def compare_drift_ratio():
    structure = generate_structure(strucutre_type)
    print(structure, structure.story_level_sections)

    story_mean_drift_ratio_static, story_peak_drift_ratio_static = get_story_level_drift_ratio(structure, scenario="static")  # shape: (gm_num, story_num, 2)
    # gm_mean_story_mean_drift_ratio_static = story_mean_drift_ratio_static.mean(axis=0)  # shape: (story_num, 2)
    # gm_peak_story_mean_drift_ratio_static = story_mean_drift_ratio_static.max(axis=0)  # shape: (story_num, 2)
    gm_mean_story_peak_drift_ratio_static = story_peak_drift_ratio_static.mean(axis=0)  # shape: (story_num, 2)
    gm_peak_story_peak_drift_ratio_static = story_peak_drift_ratio_static.max(axis=0)  # shape: (story_num, 2)

    story_mean_drift_ratio_dynamic, story_peak_drift_ratio_dynamic = get_story_level_drift_ratio(structure, scenario="dynamic")  # shape: (gm_num, story_num, 2)
    # gm_mean_story_mean_drift_ratio_dynamic = story_mean_drift_ratio_dynamic.mean(axis=0)  # shape: (story_num, 2)
    # gm_peak_story_mean_drift_ratio_dynamic = story_mean_drift_ratio_dynamic.max(axis=0)  # shape: (story_num, 2)
    gm_mean_story_peak_drift_ratio_dynamic = story_peak_drift_ratio_dynamic.mean(axis=0)  # shape: (story_num, 2)
    gm_peak_story_peak_drift_ratio_dynamic = story_peak_drift_ratio_dynamic.max(axis=0)  # shape: (story_num, 2)

    plot_story_level_drift_ratio(gm_mean_story_peak_drift_ratio_static, gm_mean_story_peak_drift_ratio_dynamic, type="mean")
    plot_story_level_drift_ratio(gm_peak_story_peak_drift_ratio_static, gm_peak_story_peak_drift_ratio_dynamic, type="peak")


def evaluate_design_response(model_path, nda_simulator, nda_norm_dict, MCE_ground_motion_set):
    if strucutre_type == "testing":
        inference_record_name = f"testing_record.txt"
    else:
        inference_record_name = f"inferencing_record_{chance}chance.txt"

    with open(model_path / inference_record_name, 'r') as f: inference_record_static = json.load(f)
    best_index = np.argmax(inference_record_static["saved_material"])
    best_action = inference_record_static["action"][best_index]

    # Access the design
    structure = generate_structure(strucutre_type)
    score = 0
    for action in best_action:
        volume_saved = structure.update_action(action)
        score += volume_saved
    print(structure, "\nstory level sections:", structure.story_level_sections)
    print("Score: ", score, inference_record_static["saved_material"][best_index], inference_record_static["score"][best_index])

    # Static Analysis
    load_cases, static_responses = check.get_response(structure, structure.analysis_dir)        
    static_constraint_condition, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
    whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, True)
    print(whether_pass, fail_name, fail_reason)

    # Dynamic Analysis
    structure.nda_norm_dict = nda_norm_dict
    structure.init_graph_GraphLSTM()
    dynamic_responses = check_nda.get_response(structure, nda_simulator, MCE_ground_motion_set, device)
    dynamic_constraint_condition, dynamic_response_features, dynamic_response_rewards = check_nda.process_response(structure, dynamic_responses, nda_norm_dict)
    whether_pass, fail_name, fail_reason = check_nda.check_pass(dynamic_constraint_condition, True)
    print(whether_pass, fail_name, fail_reason)

    print("\n")

def compare_design_response():
    model_path_static = result_dir / static_model_folder
    model_path_dynamic = result_dir / dynamic_model_folder

    nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(graph_lstm_dir, device)
    DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(ground_motion_dir, ground_motion_number, nda_norm_dict)
    print(np.array(MCE_ground_motion_set).shape)

    evaluate_design_response(model_path_static, nda_simulator, nda_norm_dict, MCE_ground_motion_set)
    evaluate_design_response(model_path_dynamic, nda_simulator, nda_norm_dict, MCE_ground_motion_set)


def get_target_accuracy_new(output, y, neglect_beam_My_Sz=True):
    if neglect_beam_My_Sz:
        target_dict = {"Acc": [0, 2], "Vel": [2, 4], "Disp": [4, 6], "My": [8, 10], "Mz": [12, 18], "Sy": [18, 24], "Sz": [26, 28]}
    else: 
        target_dict = {"Acc": [0, 2], "Vel": [2, 4], "Disp": [4, 6], "My": [6, 12], "Mz": [12, 18], "Sy": [18, 24], "Sz": [24, 30]}

    overall_accuracy_dict = {target: 0 for target in target_dict.keys()}
    peak_accuracy_dict = {target: 0 for target in target_dict.keys()}
    
    for target, (y_start, y_end) in target_dict.items():
        overall_acc = accuracy.R2_score(output[:, :, y_start:y_end], y[:, :, y_start:y_end])
        peak_acc = accuracy.peak_R2_score(output[:, :, y_start:y_end], y[:, :, y_start:y_end])
        overall_accuracy_dict[target] = overall_acc.item()
        peak_accuracy_dict[target] = peak_acc.item()
    
    return overall_accuracy_dict, peak_accuracy_dict

def inference_for_one_batch(structure, batch, device, model, neglect_beam_My_Sz=True):
    batch = batch.to(device)
    sampled_index_batch = [None]
    output, _ = model.forward(batch.x, batch.edge_index, batch.edge_attr, 
                              batch.batch, batch.ptr, sampled_index_batch, 
                              batch.ground_motions, sample_node=False)
    x, y = batch.x, batch.y

    mask = torch.ones(y.shape, dtype=bool)
    if neglect_beam_My_Sz:
        mask[:, :, (6,7,10,11)] = False    # neglect beam's MomentY
        mask[:, :, (24,25,28,29)] = False  # neglect beam's ShearZ
    mask_y = y[mask].reshape(y.shape[0], y.shape[1], -1)
    mask_output = output[mask].reshape(output.shape[0], output.shape[1], -1)

    R2_acc = accuracy.R2_score(mask_output, mask_y)
    peak_acc = accuracy.peak_R2_score(mask_output, mask_y)
    target_overall_acc, target_peak_acc = get_target_accuracy_new(output, y, neglect_beam_My_Sz)

    plastic_hinge_acc = {}
    mask_plastic_hinge = torch.zeros(y.shape, dtype=bool)
    mask_plastic_hinge[structure.hinge_node_yn, :, output_My_yn_index] = True  # My_yn
    mask_plastic_hinge[structure.hinge_node_yp, :, output_My_yp_index] = True  # My_yp
    mask_y = y[mask_plastic_hinge].reshape(-1, y.shape[1])
    mask_output = output[mask_plastic_hinge].reshape(-1, output.shape[1])
    plastic_hinge_acc["My"] = accuracy.peak_R2_score(mask_output, mask_y).item()
    
    mask_plastic_hinge = torch.zeros(y.shape, dtype=bool)
    mask_plastic_hinge[structure.hinge_node_yn, :, output_Mz_yn_index] = True  # Mz_yn
    mask_plastic_hinge[structure.hinge_node_yp, :, output_Mz_yp_index] = True  # Mz_yp
    mask_y = y[mask_plastic_hinge].reshape(-1, y.shape[1])
    mask_output = output[mask_plastic_hinge].reshape(-1, output.shape[1])
    plastic_hinge_acc["Mz"] = accuracy.peak_R2_score(mask_output, mask_y).item()

    return R2_acc.item(), peak_acc.item(), target_overall_acc, target_peak_acc, plastic_hinge_acc

def get_inference_accuracy(structure, nda_simulator, nda_norm_dict, scenario="static"):
    inference_overall_accuracy = {}
    inference_peak_accuracy = {}
    plastic_hinge_accuracy = {}
    target_folder = working_dir / scenario
    for gm_index, gm_name in enumerate(os.listdir(target_folder)):
        print(f"\nScenario: {scenario}, Ground motion: {gm_name}")
        graph_path = target_folder / gm_name / "structure_graph_NodeAsNode.pt"
        graph = torch.load(graph_path)
        
        gm_X_name = graph.gm_X_name
        gm_Z_name = gm_X_name.replace("FN", "FP")
        max_steps = 2000
        batch = 10
        timesteps = 1400
        ground_motion_1 = torch.zeros((max_steps, batch))
        ground_motion_2 = torch.zeros((max_steps, batch))
        with open(gm_X_name, "r") as f:
            for index, line in enumerate(f.readlines()):
                i, j = index//10, index%10
                ground_motion_1[i, j] = float(line.split()[1])
        with open(gm_Z_name, "r") as f:
            for index, line in enumerate(f.readlines()):
                i, j = index//10, index%10
                ground_motion_2[i, j] = float(line.split()[1])
        graph.ground_motion_1 = ground_motion_1[:timesteps, :]
        graph.ground_motion_2 = ground_motion_2[:timesteps, :]

        # AbsAcc = RelAcc + GroundMotionAcc (PISA outputs response every 10 steps: 0, 10, 20,...)
        new_y = torch.zeros(graph.y.shape)
        new_y[:, :, 0] = graph.y[:, :, 0] + graph.ground_motion_1[:, 0]  # torch.mean(graph.ground_motion_1, dim=1)
        new_y[:, :, 1] = graph.y[:, :, 1] + graph.ground_motion_2[:, 0]  # torch.mean(graph.ground_motion_2, dim=1)
        new_y[:, :, 2:] = graph.y[:, :, 2:]
        graph.y = new_y
        
        dataset_norm, norm_dict = normalization.normalize_dataset([graph], nda_norm_dict, False)
        graph_loader = DataLoader(dataset_norm, batch_size=1, shuffle=False)

        with torch.no_grad():
            for graph_index, graph in enumerate(graph_loader):
                # print("[PISA]")
                # periods_pisa = graph.x[0, 11:14].numpy() * (norm_dict['period'][1] - norm_dict['period'][0]) + norm_dict['period'][0]
                # print(f"periods: {periods_pisa}")
                # betas_pisa = graph.x[:, 9:11].numpy()
                # print(f"betas: \n{betas_pisa}")

                # R2_acc, peak_acc, target_overall_acc, target_peak_acc, plastic_hinge_acc = inference_for_one_batch(structure, graph, device, nda_simulator, neglect_beam_My_Sz=True)
                # print(f"OVERALL - R2_acc: {R2_acc:.3f}, peak_acc: {peak_acc:.3f}")
                # content = ["TARGET(ALL) - "]
                # content += [f"{target}: {acc:.3f}, " for target, acc in target_overall_acc.items()]
                # print("".join(content))
                # content = ["TARGET(PEAK) -"]
                # content += [f"{target}: {acc:.3f}, " for target, acc in target_peak_acc.items()]
                # print("".join(content))

                print("[OPENSEES]")
                ipt_path = working_dir / "static" / gm_name / "structure.ipt"
                beta_x_opensees, beta_z_opensees, mode_periods_opensees, mode_shapes_opensees = opensees.pisa_file_to_opensees_model(ipt_path)
                periods_opensees = mode_periods_opensees[:3]
                print(f"periods: {periods_opensees}")
                betas_opensees = np.array([beta_x_opensees, beta_z_opensees]).T
                # print(f"betas: \n{betas_opensees}")

                graph.x[:, 9:11] = torch.tensor(betas_opensees)
                normed_periods = (periods_opensees - norm_dict['period'][0]) / (norm_dict['period'][1] - norm_dict['period'][0])
                graph.x[:, 11:14] = torch.tensor(normed_periods)
                normed_shapes = (mode_shapes_opensees - norm_dict['modal_shape'][0]) / (norm_dict['modal_shape'][1] - norm_dict['modal_shape'][0])
                graph.x[:, 14:23] = torch.tensor(normed_shapes)

                R2_acc, peak_acc, target_overall_acc, target_peak_acc, plastic_hinge_acc = inference_for_one_batch(structure, graph, device, nda_simulator, neglect_beam_My_Sz=True)
                # print(f"OVERALL - R2_acc: {R2_acc:.3f}, peak_acc: {peak_acc:.3f}")
                # content = ["TARGET(ALL) - "]
                # content += [f"{target}: {acc:.3f}, " for target, acc in target_overall_acc.items()]
                # print("".join(content))
                # content = ["TARGET(PEAK) -"]
                # content += [f"{target}: {acc:.3f}, " for target, acc in target_peak_acc.items()]
                # print("".join(content))

        inference_overall_accuracy[gm_name] = target_overall_acc
        inference_peak_accuracy[gm_name] = target_peak_acc
        plastic_hinge_accuracy[gm_name] = plastic_hinge_acc
    
    return inference_overall_accuracy, inference_peak_accuracy, plastic_hinge_accuracy

def plot_inference_accuracy(target_overall_accuracy, target_peak_accuracy, scenario="static"):
    gm_names = list(target_overall_accuracy.keys())
    targets = ["Disp", "My", "Mz"]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for i, target in enumerate(targets):
        x = np.arange(len(gm_names))
        width = 0.35
        overall_accs = [target_overall_accuracy[gm][target] for gm in gm_names]
        peak_accs = [target_peak_accuracy[gm][target] for gm in gm_names]
        axes[i].bar(x-width/2, overall_accs, width, label=f'overall_acc (mean: {np.mean(overall_accs):.4f})', alpha=0.8)
        axes[i].bar(x+width/2, peak_accs, width, label=f'peak_acc (mean: {np.mean(peak_accs):.4f})', alpha=0.8)
        
        axes[i].set_title(target, fontsize=18)
        axes[i].set_xlabel("Ground Motion", fontsize=16)
        axes[i].set_ylabel("Accuracy", fontsize=16)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(gm_names, rotation=45, ha='right', fontsize=14)
        axes[i].tick_params(axis='y', labelsize=14)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].legend(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(working_dir / f"inference_accuracy_{scenario}.png")

def plot_plastic_hinge_accuracy(plastice_hinge_accuracy, scenario="static"):
    gm_names = list(plastice_hinge_accuracy.keys())
    x = np.arange(len(gm_names))
    width = 0.35
    My_accs = [plastice_hinge_accuracy[gm]["My"] for gm in gm_names]
    Mz_accs = [plastice_hinge_accuracy[gm]["Mz"] for gm in gm_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(x-width/2, My_accs, width, label=f'My (mean: {np.mean(My_accs):.4f})', alpha=0.8)
    plt.bar(x+width/2, Mz_accs, width, label=f'Mz (mean: {np.mean(Mz_accs):.4f})', alpha=0.8)
    
    plt.title("Plastic Hinge Accuracy", fontsize=18)
    plt.xlabel("Ground Motion", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xticks(x, gm_names, rotation=45, ha='right', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(working_dir / f"plastic_hinge_accuracy_{scenario}.png")

def compare_GraphLSTM_inference():
    structure = generate_structure(strucutre_type)
    print(structure, structure.story_level_sections)

    nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(graph_lstm_dir, device)
    # DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(ground_motion_dir, ground_motion_number, nda_norm_dict)

    inference_overall_accuracy_static, inference_peak_accuracy_static, plastic_hinge_accuracy_static = get_inference_accuracy(structure, nda_simulator, nda_norm_dict, scenario="static")
    inference_overall_accuracy_dynamic, inference_peak_accuracy_dynamic, plastic_hinge_accuracy_dynamic = get_inference_accuracy(structure, nda_simulator, nda_norm_dict, scenario="dynamic")
    plot_inference_accuracy(inference_overall_accuracy_static, inference_peak_accuracy_static, scenario="static")
    plot_inference_accuracy(inference_overall_accuracy_dynamic, inference_peak_accuracy_dynamic, scenario="dynamic")
    plot_plastic_hinge_accuracy(plastic_hinge_accuracy_static, scenario="static")
    plot_plastic_hinge_accuracy(plastic_hinge_accuracy_dynamic, scenario="dynamic")
    

if __name__ == "__main__":

    # compare_plastic_hinge_occurrence()

    compare_drift_ratio()
    
    # compare_design_response()

    # compare_GraphLSTM_inference()
