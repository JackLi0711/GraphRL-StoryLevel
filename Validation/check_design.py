import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from Structure import check, check_nda
from Structure.sections import beam_sections, column_sections


def check_one_design(structure,
                     code_analysis_dir,
                     do_nonlinear_dynamic_analysis,
                     nda_simulator,
                     DBE_ground_motion_set,
                     MCE_ground_motion_set,
                     check_acceleration,
                     check_displacement,
                     nda_norm_dict,
                     device) -> tuple[bool, str, str]:
    # 1. check if linear static analysis response pass regulation
    # original code 
    # whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, check_displacement, code_analysis_dir)
    
    # modified code
    auxiliary_values, load_cases, static_responses = check.get_response(structure, code_analysis_dir)
    static_constraint_condition, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
    whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, check_displacement)

    # 2. check if nonlinear dynamic analysis response pass regulation if needed
    if do_nonlinear_dynamic_analysis and whether_pass == True:
        # original code
        # whether_pass, fail_reason = check_nda.check(structure, 
        #                                             nda_simulator, 
        #                                             DBE_ground_motion_set,  
        #                                             MCE_ground_motion_set,
        #                                             check_acceleration,
        #                                             check_displacement,
        #                                             nda_norm_dict, 
        #                                             auxiliary_values,
        #                                             device)
        
        # modified code
        dynamic_responses = check_nda.get_response(structure, nda_simulator, MCE_ground_motion_set, device)
        dynamic_constraint_condition, dynamic_response_features, dynamic_response_rewards = check_nda.process_response(structure, dynamic_responses, nda_norm_dict)
        whether_pass, fail_name, fail_reason = check_nda.check_pass(dynamic_constraint_condition, check_displacement)

    return whether_pass, fail_name, fail_reason


def visualize_one_design(structure, reward, whether_pass, fail_reason, save_root, i):
    # plot 3d
    fig = plt.figure(figsize=(7, 7), facecolor="w")
    ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor="w")
    ax.set_axis_off()

    # plot nodes
    x_coord, y_coord, z_coord = [], [], []
    for node_index in range(len(structure.node_coord_dict.keys())):
        node_name = f"N{node_index+1}"
        x, y, z = structure.node_coord_dict[node_name]
        x_coord.append(x)
        y_coord.append(y)
        z_coord.append(z)
    ax.scatter(x_coord, z_coord, y_coord, color="black", s=50, alpha=0.75, edgecolors="black")
    ax.set_box_aspect((np.ptp(x_coord), np.ptp(z_coord), np.ptp(y_coord)))

    # plot edges
    edges_coord = []
    edges_color = []
    for member_name in structure.member_to_nodeIndex_dict.keys():
        # member coord
        elems = structure.member_to_nodeIndex_dict[member_name]
        node1_index, node2_index = elems[:2]
        node1_name, node2_name = f"N{node1_index+1}", f"N{node2_index+1}"
        x1, y1, z1 = structure.node_coord_dict[node1_name]
        x2, y2, z2 = structure.node_coord_dict[node2_name]
        edges_coord.append([x1, y1, z1, x2, y2, z2])
        # member section
        member_category = structure.member_category_dict[member_name]
        member_section = structure.member_section_dict[member_name]
        section = column_sections[member_section] if member_category == 'y' else beam_sections[member_section]
        color = section["color"]
        color = (color[0]/255, color[1]/255, color[2]/255)
        edges_color.append(color)

    for edge_i in range(len(edges_coord)):
        x1, y1, z1, x2, y2, z2 = edges_coord[edge_i]
        xx = [x1, x2]
        yy = [y1, y2]
        zz = [z1, z2]
        color = edges_color[edge_i]
        ax.plot(xx, zz, yy, c=(color), linewidth=3)

    save_dir = save_root / "pass" if whether_pass else save_root / "fail"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_path = save_dir / f"{reward:5.3f}_structure{i}.png"

    ax.set_title(f"reward: {reward:5.3f} fail_reason: {fail_reason}", fontsize=15)
    fig.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()


def check_designs_from_design_space(structures,
                                    rewards,
                                    code_analysis_dir,
                                    do_nonlinear_dynamic_analysis,
                                    nda_simulator,
                                    DBE_ground_motion_set,
                                    MCE_ground_motion_set,
                                    check_acceleration,
                                    check_displacement,
                                    nda_norm_dict,
                                    device,
                                    save_root) -> None:
    # pass_dir = save_root / "pass"
    # fail_dir = save_root / "fail"
    # pass_dir.mkdir(parents=True, exist_ok=True)
    # fail_dir.mkdir(parents=True, exist_ok=True)

    sampling_record = {"geometry": [], "final_design": [], 
                       "final_volume": [], "saved_material": [], 
                       "whether_pass": [], "fail_name": [], "fail_reason": []}
    for i, (structure, reward) in tqdm(enumerate(zip(structures, rewards))):
        # check whether pass
        whether_pass, fail_name, fail_reason = check_one_design(structure,
                                                                code_analysis_dir,
                                                                do_nonlinear_dynamic_analysis,
                                                                nda_simulator,
                                                                DBE_ground_motion_set,
                                                                MCE_ground_motion_set,
                                                                check_acceleration,
                                                                check_displacement,
                                                                nda_norm_dict,
                                                                device)
        # record the result
        sampling_record["geometry"].append([structure.x_span_num, structure.z_span_num, structure.story_num, structure.x_span_lens, structure.z_span_lens, structure.story_height])
        sampling_record["final_design"].append([i for i in structure.story_level_sections])
        sampling_record["final_volume"].append(structure.calculate_material_usage())
        sampling_record["saved_material"].append(reward)
        print("heaviest weight: ", structure.calculate_material_usage() + reward)
        sampling_record["whether_pass"].append(whether_pass)
        sampling_record["fail_name"].append(fail_name)
        sampling_record["fail_reason"].append(fail_reason)
        # visualize the result --> the saved figures take too much space
        # visualize_one_design(structure, reward, whether_pass, fail_reason, save_root, i)
    
    with open(save_root / "sampling_record.txt", 'w') as f:
        json.dump(sampling_record, f)
