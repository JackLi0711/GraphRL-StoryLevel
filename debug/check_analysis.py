import sys
sys.path.append("../")
from Structure import structure, pisa, earthquake, load
from Structure.sections import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from NonlinearDynamicAnalysisSimulator import load_simulator


from Structure.check import _check_drift_ratio_pass, _check_soft_story_pass, _check_column_compression_strength_pass, \
                            _check_column_tension_strength_pass, _check_beam_strength_pass, \
                            _check_beam_axial_moment_pass, _check_strong_column_weak_beam_pass

from Structure.check_nda import _check_nda_drift_ratio_pass, _check_nda_plastic_hinge_location_pass, \
                                _generate_nda_response



def env_setup():
    graph_lstm_dir = Path("../NonlinearDynamicAnalysisSimulator/trained_GraphLSTM/2023_03_26__10_35_37/")
    ground_motion_dir = Path("../NonlinearDynamicAnalysisSimulator/ground_motions/selected_ground_motions/")
    device = "cuda"
    norm_dict = {"ground_motion": 9442.7060546875, "grid_num": 8.0, "coord": 7.0, "period": 1.580899953842163, "modal_shape": 7.78000020980835, "elem_length": 8.0, "acc": 34430.0, "vel": 2680.0, "disp": 773.5999755859375, "moment": 2552083.0, "shear": 1119.0}
    nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(graph_lstm_dir, device)
    ground_motion_set = load_simulator.load_ground_motions(ground_motion_dir, nda_norm_dict)
    return nda_simulator, ground_motion_set, device


def generate_structure(index):
    x_span_num = 3
    z_span_num = 3
    x_span_len = 6000
    z_span_len = 8000
    x_span_lens = [x_span_len for i in range(x_span_num)]
    z_span_lens = [z_span_len for i in range(z_span_num)]
    story_num = 5
    story_height = 3200

    analysis_dir = Path("./analysis") / str(index)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    norm_dict = {"ground_motion": 9442.7060546875, "grid_num": 8.0, "coord": 7.0, "period": 1.580899953842163, "modal_shape": 7.78000020980835, "elem_length": 8.0, "acc": 34430.0, "vel": 2680.0, "disp": 773.5999755859375, "moment": 2552083.0, "shear": 1119.0}

    structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, "z_span_num": z_span_num,
                            "z_span_lens": z_span_lens, "story_num": story_num, "story_height": story_height,
                            "add_structure_geometry": False, 
                            "do_nonlinear_dynamic_analysis": True,
                            "nda_norm_dict": norm_dict,
                            "analysis_dir": analysis_dir}
    s = structure.Structure(**structure_kwargs)
    return s



def check_static(structure):
    first_mode_period, _ = pisa.dynamic_analysis_period(structure, structure.analysis_dir)
    earthquake_forces, Fu = earthquake.design_earthquake_force(structure, first_mode_period)
    load_cases = load.get_load_cases(structure, earthquake_forces, Fu)
    for load_case in load_cases:
        response = pisa.run_load_case(structure, load_case, structure.analysis_dir)
        whether_pass, fail_reason = _check_code(structure, response, load_case)
        if whether_pass is False:
            print("fail at load case:", load_case.load_name)
            return False, fail_reason
    return True, None


def check_dynamic(structure,
                  nda_simulator,
                  ground_motion_set,
                  device) -> Tuple[bool, str]:
    
    # inference on model to get the response
    nda_responses = _generate_nda_response(structure, nda_simulator, ground_motion_set, device)

    # check drift ratio
    if not _check_nda_drift_ratio_pass(structure, nda_responses): return False, "nda_drift"

    # check plastic hinge location
    if not _check_nda_plastic_hinge_location_pass(structure, nda_responses): return False, "nda_hinge"

    return True, None





def _check_code(structure, response: pisa.Response, load_case: load.NodalLoad) -> bool:
    # checking for specific load combinations
    if load_case.E > 0:
        if not _check_soft_story_pass(structure, response): return False, "soft_story"
    if load_case.drift_case:
        if not _check_drift_ratio_pass(structure, response): return False, "drift_ratio"
    if load_case.col_strength_case:
        if not _check_column_compression_strength_pass(structure, response): return False, "column_compression"
        if not _check_column_tension_strength_pass(structure, response): return False, "column_tension"

    # checking for common constraint
    if not _check_beam_strength_pass(structure, response): return False, "beam_strength"
    if not _check_beam_axial_moment_pass(structure, response): return False, "beam_moment"
    if not _check_strong_column_weak_beam_pass(structure, response): return False, "strong_column_weak_beam"

    return True, None













def visualize_structure(structure):
    # plot 3d
    fig = plt.figure(figsize=(3, 3), facecolor="w")
    ax = fig.add_subplot(111, projection="3d", facecolor="w")
    ax.set_axis_off()

    # plot nodes
    x_coord, y_coord, z_coord = [], [], []
    for node_index in range(len(structure.node_coord_dict.keys())):
        node_name = f"N{node_index+1}"
        x, y, z = structure.node_coord_dict[node_name]
        x_coord.append(x)
        y_coord.append(y)
        z_coord.append(z)
    ax.scatter(x_coord, z_coord, y_coord, color="black", s=80, alpha=0.75, edgecolors="black")
    ax.set_box_aspect((np.ptp(x_coord), np.ptp(z_coord), np.ptp(y_coord)))

    # plot edges
    # print("section:", structure.member_section_dict.values())
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

    fig.tight_layout()
    plt.show()