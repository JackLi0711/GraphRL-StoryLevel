import torch
import random
import numpy as np

from pathlib import Path
from logging import Logger

from Structure import pisa, check
from Structure.sections import *
from Structure.structure import Structure


def sample_initial_story_sections(x_span_num: int, x_span_len: int, 
                                  z_span_num: int, z_span_len: int, 
                                  story_num: int) -> list[int]:
    min_geo_sum = (2+6) + (2+6) + 4
    max_geo_sum = (6+8) + (6+8) + 7
    geo_sum = (x_span_num + x_span_len/1000) + (z_span_num + z_span_len/1000) + story_num
    main_type = geo_sum - min_geo_sum
    
    section_pool = [i for i in range(len(column_sections))]
    distance = np.abs(np.array(section_pool) - main_type)
    exp_negative_distance = np.exp(-1 * distance * 0.25)
    sample_prob = exp_negative_distance / np.sum(exp_negative_distance)

    story_outer_column_section = sorted(random.choices(section_pool, weights=sample_prob, k=story_num), reverse=True)
    story_inner_column_section = sorted(random.choices(section_pool, weights=sample_prob, k=story_num), reverse=True)
    story_xdir_beam_section = [len(beam_sections)-1 for _ in range(story_num)]
    story_zdir_beam_section = [len(beam_sections)-1 for _ in range(story_num)]

    story_level_sections = story_xdir_beam_section + story_zdir_beam_section + story_outer_column_section + story_inner_column_section
    
    return story_level_sections




def check_strong_column_weak_beam(structure: Structure, responses: list[pisa.Response]) -> np.ndarray:
    fail_conditions = np.zeros((structure.node_number, 2))  # X-dir(0), Z-dir(1)
    for response in responses:
        # calculate strong-column-weak-beam ratio
        Zc = structure.node_neighbor_Zz_matrix
        Puc = response.node_neighbor_Puc_matrix
        Puc[Puc > 0] = 0       # only consider compression case
        Ag = structure.node_neighbor_Ag_matrix
        Ag[Ag < 1e-5] = 1e+10  # in case area = 0 for faces not connected with members
        
        reduced_stress = torch.abs(Puc / Ag)
        ZcFyc =  (Zc * (YIELDING_STRESS - reduced_stress)) @ np.array([0, 0, 1, 1, 0, 0])               # [node_number, 1]
        ZbFyb_x = (YIELDING_STRESS * structure.node_neighbor_Zz_matrix) @ np.array([1, 1, 0, 0, 0, 0])  # [node_number, 1]
        ZbFyb_z = (YIELDING_STRESS * structure.node_neighbor_Zz_matrix) @ np.array([0, 0, 0, 0, 1, 1])  # [node_number, 1]
        
        ratio_x = (ZcFyc / (ZbFyb_x + 1e-6)).squeeze()  # [node_number]
        ratio_z = (ZcFyc / (ZbFyb_z + 1e-6)).squeeze()  # [node_number]
        fail_conditions[(ratio_x < 1.25), 0] = 1
        fail_conditions[(ratio_z < 1.25), 1] = 1

    return fail_conditions


def strong_column_weak_beam_driven_action(structure: Structure, fail_conditions: np.ndarray) -> tuple[list[int], list[int]]:
    # don't consider nodes that are located at base or top 
    all_node_index = [i for i in range(structure.node_number)]
    node_no_need_strong_column_weak_beam_list = list(set(all_node_index) - set(structure.node_need_strong_column_weak_beam_list))
    fail_conditions[node_no_need_strong_column_weak_beam_list, :] = 0

    xdir_fail_node_index = list(np.where(fail_conditions[:, 0] == 1)[0])
    zdir_fail_node_index = list(np.where(fail_conditions[:, 1] == 1)[0])

    story_beam_update_conditions = np.zeros((structure.story_num, 2))   # X-dir(0), Z-dir(1)
    for node_index in xdir_fail_node_index:
        node_name = f"N{node_index+1}"
        x_index, story, z_index = structure.node_grid_coord_dict[node_name]
        story_beam_update_conditions[story-1, 0] = 1
    for node_index in zdir_fail_node_index:
        node_name = f"N{node_index+1}"
        x_index, story, z_index = structure.node_grid_coord_dict[node_name]
        story_beam_update_conditions[story-1, 1] = 1
    #print(f"{story_beam_update_conditions = }")

    xdir_beam_update_action = sorted(list(np.where(story_beam_update_conditions[:, 0] == 1)[0]), reverse=True)
    zdir_beam_update_action = sorted(list(np.where(story_beam_update_conditions[:, 1] == 1)[0] + structure.story_num), reverse=True)

    return xdir_beam_update_action, zdir_beam_update_action


def strong_column_weak_beam_driven_update(structure: Structure, analysis_dir: Path, logger: Logger) -> tuple[float, list[int], dict]:    
    auxiliary_values, load_cases, responses = check.get_response(structure, analysis_dir)
    fail_conditions = check_strong_column_weak_beam(structure, responses)
    xdir_beam_update_action, zdir_beam_update_action = strong_column_weak_beam_driven_action(structure, fail_conditions)
    update_actions = xdir_beam_update_action + zdir_beam_update_action
    
    material_saved = 0
    all_update_actions = []
    while len(update_actions) != 0:
        print(f"{xdir_beam_update_action = }")
        print(f"{zdir_beam_update_action = }")
        all_update_actions.extend(update_actions)

        initial_usage = structure.calculate_material_usage()
        score = 0
        for action in (update_actions):
            reward = structure.update_action(action)
            score += reward
        final_usage = structure.calculate_material_usage()
        print(f"{score:.3f}, {initial_usage - final_usage:.3f}")
        print(f"{structure.story_level_sections}\n")
        material_saved += score
    
        auxiliary_values, load_cases, responses = check.get_response(structure, analysis_dir)
        fail_conditions = check_strong_column_weak_beam(structure, responses)
        xdir_beam_update_action, zdir_beam_update_action = strong_column_weak_beam_driven_action(structure, fail_conditions)
        update_actions = xdir_beam_update_action + zdir_beam_update_action
    
    structural_behaviors = {
        "auxiliary_value": auxiliary_values,
        "load_case": load_cases,
        "response": responses
    }

    return material_saved, all_update_actions, structural_behaviors
