import torch
import numpy as np
import collections

import sys
sys.path.append('../')
from Structure.structure import Structure
from Structure import sections, opensees, check


_field_names = [
    # overall representation
    "grid_num",  # [x, z, y]
    "mode_period",  # [1st, 2nd, 3rd]

    # member information
    "member_Ag",  # shape: (story, member_category)
    "member_Iy",  # shape: (story, member_category)
    "member_Iz",  # shape: (story, member_category)
    "member_Zz",  # shape: (story, member_category)

    # local response
    "min_scwb_ratio",  # shape: (story, location, direction)
    "max_drift_ratio",  # shape: (story, location, direction)
    "max_stress_ratio",  # shape: (story, member_category)
]

Feature = collections.namedtuple("Feature", field_names=_field_names)

def get_min_scwb_ratio(structure: Structure) -> np.ndarray:
    responses = structure.static_responses if structure.static_responses is not None else check.get_response(structure, structure.analysis_dir)[1]
    scwb_ratios_xdir = np.zeros((structure.node_number, len(responses)))
    scwb_ratios_zdir = np.zeros((structure.node_number, len(responses)))
    for i, response in enumerate(responses): 
        Puc = response.node_neighbor_Puc_matrix
        Puc[Puc > 0] = 0       # only consider compression case
        Ag = structure.node_neighbor_Ag_matrix
        Ag[Ag < 1e-5] = 1e+10  # in case area = 0 for faces not connected with members
        reduced_stress = torch.abs(Puc / Ag)
        
        Zz = structure.node_neighbor_Zz_matrix
        ZcFyc =  (Zz * (sections.YIELDING_STRESS - reduced_stress)) @ np.array([0, 0, 1, 1, 0, 0])  # (node_number, 6) @ (6, 1) = (node_number, 1)
        ZbFyb_x = (sections.YIELDING_STRESS * Zz) @ np.array([1, 1, 0, 0, 0, 0])                    # (node_number, 6) @ (6, 1) = (node_number, 1)
        ZbFyb_z = (sections.YIELDING_STRESS * Zz) @ np.array([0, 0, 0, 0, 1, 1])                    # (node_number, 6) @ (6, 1) = (node_number, 1)
        scwb_ratio_x = (ZcFyc / (ZbFyb_x + 1e-6)).numpy()  # shape: (node_number,)
        scwb_ratio_z = (ZcFyc / (ZbFyb_z + 1e-6)).numpy()  # shape: (node_number,)
        scwb_ratios_xdir[:, i] = scwb_ratio_x
        scwb_ratios_zdir[:, i] = scwb_ratio_z

    min_scwb_ratio = np.zeros((structure.story_num, 2, 2))  # 2: outer_node, inner_node; 2: X-dir, Z-dir
    for story, y in enumerate(structure.y_grid):
        if story == 0: continue
        story_name = f'{story}F'
        story_node_indexes = structure.story_nodes[story_name]
        outer_node_indexes = []
        inner_node_indexes = []
        for node_index in story_node_indexes:
            node_name = f'N{node_index+1}'
            x, y, z = structure.node_coord_dict[node_name]
            if x == structure.x_grid[0] or x == structure.x_grid[-1] or z == structure.z_grid[0] or z == structure.z_grid[-1]:  # outer column
                outer_node_indexes.append(node_index)
            else:  # inner column
                inner_node_indexes.append(node_index)
        
        min_scwb_ratio[story-1, 0, 0] = np.min(scwb_ratios_xdir[outer_node_indexes, :])  # outer_node, X-dir
        min_scwb_ratio[story-1, 0, 1] = np.min(scwb_ratios_zdir[outer_node_indexes, :])  # outer_node, Z-dir
        min_scwb_ratio[story-1, 1, 0] = np.min(scwb_ratios_xdir[inner_node_indexes, :])  # inner_node, X-dir
        min_scwb_ratio[story-1, 1, 1] = np.min(scwb_ratios_zdir[inner_node_indexes, :])  # inner_node, Z-dir

    return min_scwb_ratio

def get_max_drift_ratio(structure: Structure) -> np.ndarray:
    load_cases, responses = structure.load_cases, structure.static_responses if (structure.load_cases is not None and structure.static_responses is not None) else check.get_response(structure, structure.analysis_dir)
    drift_ratios_xdir = np.zeros((structure.member_number, len(responses)))
    drift_ratios_zdir = np.zeros((structure.member_number, len(responses)))
    for i, response in enumerate(responses):
        node_disp_x = np.array(list(response.node_response["dispX"].values()))
        node_disp_z = np.array(list(response.node_response["dispZ"].values()))

        member_end_nodes = np.array(list(structure.member_to_nodeIndex_dict.values()))[:, 0:2]  # node1_index, node2_index
        member_lengths = np.array(list(structure.member_length_dict.values())) * 1e+3  # unit conversion: m --> mm
        drift_ratio_x = np.abs((node_disp_x[member_end_nodes[:, 1]] - node_disp_x[member_end_nodes[:, 0]]) / member_lengths)
        drift_ratio_z = np.abs((node_disp_z[member_end_nodes[:, 1]] - node_disp_z[member_end_nodes[:, 0]]) / member_lengths)
        drift_ratios_xdir[:, i] = drift_ratio_x
        drift_ratios_zdir[:, i] = drift_ratio_z

    max_drift_ratio = np.zeros((structure.story_num, 2, 2))  # shape: (story, 2, 2), 2: outer_column, inner_column; 2: X-dir, Z-dir
    considered_load_cases = [i for i, load_case in enumerate(load_cases) if load_case.drift_case]
    for story, y in enumerate(structure.y_grid): 
        if story == 0: continue
        outer_column_member_indexes = structure.story_outer_column_member[story-1]
        inner_column_member_indexes = structure.story_inner_column_member[story-1]
        max_drift_ratio[story-1, 0, 0] = np.max(drift_ratios_xdir[outer_column_member_indexes, :][:, considered_load_cases])  # outer_column, X-dir
        max_drift_ratio[story-1, 0, 1] = np.max(drift_ratios_zdir[outer_column_member_indexes, :][:, considered_load_cases])  # outer_column, Z-dir
        max_drift_ratio[story-1, 1, 0] = np.max(drift_ratios_xdir[inner_column_member_indexes, :][:, considered_load_cases])  # inner_column, X-dir
        max_drift_ratio[story-1, 1, 1] = np.max(drift_ratios_zdir[inner_column_member_indexes, :][:, considered_load_cases])  # inner_column, Z-dir

    return max_drift_ratio

def get_max_stress_ratio(structure: Structure) -> np.ndarray:
    load_cases, responses = structure.load_cases, structure.static_responses if (structure.load_cases is not None and structure.static_responses is not None) else check.get_response(structure, structure.analysis_dir)
    stress_ratios = np.zeros((structure.member_number, len(responses)))
    for i, response in enumerate(responses):
        beam_compression_ratios = check.get_ratio_beam_compression_strength(structure, response)
        beam_tension_ratios = check.get_ratio_beam_tension_strength(structure, response)
        column_compression_ratios = check.get_ratio_column_compression_strength(structure, response)
        column_tension_ratios = check.get_ratio_column_tension_strength(structure, response)
        stress_ratios[structure.member_beam_index_list, i] = beam_compression_ratios + beam_tension_ratios
        stress_ratios[structure.member_column_index_list, i] = column_compression_ratios + column_tension_ratios

    max_stress_ratio = np.zeros((structure.story_num, 4))  # 4: xdir_beam, zdir_beam, outer_column, inner_column
    considered_load_cases = [i for i, load_case in enumerate(load_cases) if load_case.col_strength_case]
    for story, y in enumerate(structure.y_grid): 
        if story == 0: continue
        xdir_beam_member_indexes = structure.story_xdir_beam_member[story-1]
        zdir_beam_member_indexes = structure.story_zdir_beam_member[story-1]
        outer_column_member_indexes = structure.story_outer_column_member[story-1]
        inner_column_member_indexes = structure.story_inner_column_member[story-1]
        max_stress_ratio[story-1, 0] = np.max(stress_ratios[xdir_beam_member_indexes, :])  # xdir_beam
        max_stress_ratio[story-1, 1] = np.max(stress_ratios[zdir_beam_member_indexes, :])  # zdir_beam
        max_stress_ratio[story-1, 2] = np.max(stress_ratios[outer_column_member_indexes, :][:, considered_load_cases])  # outer_column
        max_stress_ratio[story-1, 3] = np.max(stress_ratios[inner_column_member_indexes, :][:, considered_load_cases])  # inner_column

    return max_stress_ratio

def extract_features(structure: Structure) -> Feature:
    """
    Extracts features from the given structure for training the decision tree.

    Args:
        structure (Structure): The current structure object.

    Returns:
        Feature: A named tuple containing the extracted features.
    """
    # overall representation
    grid_num = np.array([structure.x_span_num, structure.z_span_num, structure.story_num])
    mode_period = np.array([structure.first_mode_period, structure.second_mode_period, structure.third_mode_period]) if structure.first_mode_period is not None else opensees.run_modal_analysis(structure)[0][:3]

    # member information
    member_Ag = np.zeros((structure.story_num, 4))  # 4: xdir_beam, zdir_beam, outer_column, inner_column
    member_Iy = np.zeros((structure.story_num, 4))  # 4: xdir_beam, zdir_beam, outer_column, inner_column
    member_Iz = np.zeros((structure.story_num, 4))  # 4: xdir_beam, zdir_beam, outer_column, inner_column
    member_Zz = np.zeros((structure.story_num, 4))  # 4: xdir_beam, zdir_beam, outer_column, inner_column
    for i, section_index in enumerate(structure.story_level_sections): 
        category_index, story_index = divmod(i, structure.story_num)
        section_list = sections.beam_sections if category_index < 2 else sections.column_sections
        member_Ag[story_index, category_index] = section_list[section_index]['A(cm2)']
        member_Iy[story_index, category_index] = section_list[section_index]['I_y(cm4)']
        member_Iz[story_index, category_index] = section_list[section_index]['I_z(cm4)']
        member_Zz[story_index, category_index] = section_list[section_index]['Z_z(cm3)']

    # local response
    min_scwb_ratio = get_min_scwb_ratio(structure)  # shape: (story_num, 2, 2), 2: outer_node, inner_node; 2: X-dir, Z-dir
    max_drift_ratio = get_max_drift_ratio(structure)  # shape: (story_num, 2, 2), 2: outer_column, inner_column; 2: X-dir, Z-dir
    max_stress_ratio = get_max_stress_ratio(structure)  # shape: (story_num, 4), 4: xdir_beam, zdir_beam, outer_column, inner_column

    feature = Feature(
        grid_num=grid_num,
        mode_period=mode_period,
        member_Ag=member_Ag,
        member_Iy=member_Iy,
        member_Iz=member_Iz,
        member_Zz=member_Zz,
        min_scwb_ratio=min_scwb_ratio,
        max_drift_ratio=max_drift_ratio,
        max_stress_ratio=max_stress_ratio,
    )
    return feature


def get_size(obj): 
    """Recursively finds size of objects in bytes."""
    if isinstance(obj, dict):
        return sum(get_size(v) for v in obj.values()) + sys.getsizeof(obj)
    elif isinstance(obj, list):
        return sum(get_size(item) for item in obj) + sys.getsizeof(obj)
    else:
        return sys.getsizeof(obj)  # unit: bytes
