import torch
import logging
import numpy as np
from pathlib import Path
from Structure import load
from Structure import pisa
from Structure import earthquake
from Structure.sections import *
from Structure.structure import Structure


PHI_C = 0.85
BEAM_AXIAL_MOMENT_LIMIT = 1.0
SCWB_RATIO_LIMIT = 1.25
STORY_SHEAR_RATIO_LIMIT = 0.80
STORY_DRIFT_RATIO_LIMIT = 0.005


def get_response(structure: Structure, analysis_dir: Path) -> tuple[dict[str, float], list[load.NodalLoad], list[pisa.Response]]:
    '''Get load cases and responses of static analysis run by PISA3D.'''
    first_mode_period, second_mode_period = pisa.dynamic_analysis_period(structure, analysis_dir)[0:2]
    earthquake_forces, Fus = earthquake.design_earthquake_force(structure, first_mode_period, second_mode_period)
    auxiliary_values = {"first_mode_period": first_mode_period, 
                        "second_mode_period": second_mode_period, 
                        "Fu1": Fus[0],
                        "Fu2": Fus[1]}
    load_cases = load.get_load_cases(structure, earthquake_forces, Fus)
    responses = []
    for load_case in load_cases:
        response = pisa.run_load_case(structure, load_case, analysis_dir)
        responses.append(response)
    return auxiliary_values, load_cases, responses


def process_response(structure: Structure, 
                   load_cases: list[load.NodalLoad],
                   responses: list[pisa.Response]) -> tuple[np.ndarray, dict[str, torch.Tensor], dict[str, np.float64]]:
    '''Process structural responses and calculate constraint conditions.'''
    stress_ratios = np.zeros((structure.member_number, len(load_cases)))
    drift_ratios = np.zeros((structure.member_number, len(load_cases)))
    scwb_ratios_x = np.zeros((structure.node_number, len(load_cases)))
    scwb_ratios_z = np.zeros((structure.node_number, len(load_cases)))
    constraint_condition = np.zeros((len(load_cases), 8))
    for i, load_case in enumerate(load_cases):
        response = responses[i]
        beam_compression_ratios = get_ratio_beam_compression_strength(structure, response)
        beam_tension_ratios = get_ratio_beam_tension_strength(structure, response)
        column_compression_ratios = get_ratio_column_compression_strength(structure, response) #if load_case.col_strength_case else -1 * np.ones(len(structure.member_column_index_list))
        column_tension_ratios = get_ratio_column_tension_strength(structure, response) #if load_case.col_strength_case else -1 * np.ones(len(structure.member_column_index_list))
        stress_ratios[structure.member_beam_index_list, i] = beam_compression_ratios + beam_tension_ratios
        stress_ratios[structure.member_column_index_list, i] = column_compression_ratios + column_tension_ratios

        beam_axial_moment_ratios = get_ratio_beam_axial_moment(structure, response)
        node_scwb_ratios_x, node_scwb_ratios_z = get_strong_column_weak_beam_ratio(structure, response)
        scwb_ratios_x[:, i] = node_scwb_ratios_x
        scwb_ratios_z[:, i] = node_scwb_ratios_z

        story_shear_ratios = get_story_shear_ratio(structure, response) #if load_case.E > 0 else -1 * np.ones(structure.story_num - 1)
        member_drift_ratios = get_member_drift_ratio(structure, response) #if load_case.drift_case else -1 * np.ones(structure.member_number)
        drift_ratios[:, i] = member_drift_ratios

        constraint_condition[i, 0] = np.max(beam_compression_ratios)
        constraint_condition[i, 1] = np.max(beam_tension_ratios)
        constraint_condition[i, 2] = np.max(beam_axial_moment_ratios)
        constraint_condition[i, 3] = np.min(np.minimum(node_scwb_ratios_x, node_scwb_ratios_z)[structure.node_need_strong_column_weak_beam_list])

        constraint_condition[i, 4] = np.min(story_shear_ratios)
        constraint_condition[i, 5] = np.max(member_drift_ratios[structure.member_column_index_list])
        constraint_condition[i, 6] = np.max(column_compression_ratios)
        constraint_condition[i, 7] = np.max(column_tension_ratios)

    response_features = {
        # extreme values among all the load cases for each member/node
        "max_stress_ratio": torch.tensor(np.max(stress_ratios, axis=1) / PHI_C),
        "max_drift_ratio": torch.tensor(np.max(drift_ratios, axis=1) / STORY_DRIFT_RATIO_LIMIT),
        "min_SCWB_ratio_x": torch.tensor(np.min(scwb_ratios_x, axis=1) / SCWB_RATIO_LIMIT), 
        "min_SCWB_ratio_z": torch.tensor(np.min(scwb_ratios_z, axis=1) / SCWB_RATIO_LIMIT)
    }

    response_rewards = {
        # extreme values among all the load cases and all the members/nodes
        "max_stress_ratio": np.max(stress_ratios) / PHI_C,
        "min_stress_ratio": np.min(np.max(stress_ratios, axis=1)) / PHI_C,
        "max_drift_ratio": np.max(drift_ratios[structure.member_column_index_list]) / STORY_DRIFT_RATIO_LIMIT,
        "min_SCWB_ratio": np.min(np.minimum(scwb_ratios_x, scwb_ratios_z)[structure.node_need_strong_column_weak_beam_list]) / SCWB_RATIO_LIMIT
    }
    print(f"response_rewards: {response_rewards}")

    return constraint_condition, response_features, response_rewards


def check_pass(load_cases: list[load.NodalLoad], 
               constraint_condition: np.ndarray, 
               check_displacement: bool=True) -> tuple[bool, str, str]:
    '''Check structural responses whether pass constraints or not under various load cases.'''
    for i, load_case in enumerate(load_cases):
        fail_name = load_case.load_name
        if constraint_condition[i, 0] > PHI_C: 
            print(f"fail at {fail_name}, beam_compression: {constraint_condition[i, 0]} > {PHI_C}")
            return False, fail_name, "beam_compression"
        if constraint_condition[i, 1] > PHI_C: 
            print(f"fail at {fail_name}, beam_tension: {constraint_condition[i, 1]} > {PHI_C}")
            return False, fail_name, "beam_tension"
        if constraint_condition[i, 2] > BEAM_AXIAL_MOMENT_LIMIT: 
            print(f"fail at {fail_name}, beam_axial_moment: {constraint_condition[i, 2]} > {BEAM_AXIAL_MOMENT_LIMIT}")
            return False, fail_name, "beam_axial_moment"
        if constraint_condition[i, 3] < SCWB_RATIO_LIMIT: 
            print(f"fail at {fail_name}, strong_column_weak_beam: {constraint_condition[i, 3]} < {SCWB_RATIO_LIMIT}")
            return False, fail_name, "strong_column_weak_beam"

        if load_case.E > 0 and (constraint_condition[i, 4] < STORY_SHEAR_RATIO_LIMIT): 
            print(f"fail at {fail_name}, story_shear: {constraint_condition[i, 4]} < {STORY_SHEAR_RATIO_LIMIT}")
            return False, fail_name, "soft_story"
        if load_case.drift_case and check_displacement and (constraint_condition[i, 5] > STORY_DRIFT_RATIO_LIMIT): 
            print(f"fail at {fail_name}, story_drift: {constraint_condition[i, 5]} > {STORY_DRIFT_RATIO_LIMIT}")
            return False, fail_name, "drift_ratio"
        if load_case.col_strength_case and (constraint_condition[i, 6] > PHI_C): 
            print(f"fail at {fail_name}, column_compression: {constraint_condition[i, 6]} > {PHI_C}")
            return False, fail_name, "column_compression"
        if load_case.col_strength_case and (constraint_condition[i, 7] > PHI_C): 
            print(f"fail at {fail_name}, column_tension: {constraint_condition[i, 7]} > {PHI_C}")
            return False, fail_name, "column_tension"

    # for load_case, response in list(zip(load_cases, responses)):
    #     whether_pass, fail_reason = _check_code(structure, response, load_case, check_displacement)
    #     if whether_pass is False:
    #         return False, load_case.load_name, fail_reason, auxiliary_values
    
    # if _too_much_minimum_section(structure): return False, None, "minimum_section", auxiliary_values
        
    return True, None, None


def get_ratio_beam_compression_strength(structure: Structure, response: pisa.Response) -> np.ndarray:
    """Get beam-compression-strength ratio given a specific structure and response."""
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    beam_axial_force[beam_axial_force > 0] = 0  # only consider compression case
    
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_beam_index_list]  # kN/mm^2
    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]      # mm^2
    Puc = Fcr * A  # kN

    return np.abs(beam_axial_force) / Puc


def get_ratio_beam_tension_strength(structure: Structure, response: pisa.Response) -> np.ndarray:
    """Get beam-tension-strength ratio given a specific structure and response."""
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    beam_axial_force[beam_axial_force < 0] = 0  # only consider tension case
    
    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]  # mm^2
    Put = YIELDING_STRESS * A  # kN

    return np.abs(beam_axial_force) / Put


def get_ratio_beam_axial_moment(structure: Structure, response: pisa.Response) -> np.ndarray:
    """
    Get beam-axial-moment ratio given a specific structure and response.
    - [鋼構規範(LRFD) 8.2 對稱構材承受彎矩及軸力之作用](https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/3495-8.pdf)
    """
    PHI, PHI_b = 0.85, 0.90
    # demand
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    Pu = np.abs(beam_axial_force)
    Mux = np.abs(np.array(list(response.member_response["momentZ"].values()))[structure.member_beam_index_list])
    Muy = np.abs(np.array(list(response.member_response["momentY"].values()))[structure.member_beam_index_list])

    # capacity
    tension_condition = (beam_axial_force > 0)
    compression_condition = (beam_axial_force <= 0)

    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]  # mm2
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_beam_index_list]  # kN/mm2
    Puc = Fcr * A  # kN
    Put = YIELDING_STRESS * A  # kN

    Pn = np.zeros_like(Pu)
    Pn += Put * tension_condition
    Pn += Puc * compression_condition
    phi_Pn = PHI * Pn

    Mnx = np.array(list(structure.member_Mnx_dict.values()))[structure.member_beam_index_list]
    Mny = np.array(list(structure.member_Mny_dict.values()))[structure.member_beam_index_list]
    phi_Mnx = PHI_b * Mnx
    phi_Mny = PHI_b * Mny
    
    case_1 = ((Pu / phi_Pn) >= 0.2)
    ratio_1 = ((Pu / phi_Pn) + 8/9 * (Mux / phi_Mnx + Muy / phi_Mny)) * case_1
    case_2 = ((Pu / phi_Pn) < 0.2)
    ratio_2 = ((Pu / (2 * phi_Pn)) + (Mux / phi_Mnx + Muy / phi_Mny)) * case_2
    ratio = ratio_1 + ratio_2

    return ratio


def get_strong_column_weak_beam_ratio(structure: Structure, response: pisa.Response) -> tuple[np.ndarray, np.ndarray]:
    """
    Get strong-column-weak-beam ratio given a specific structure and response.
    - [鋼構規範(LRFD) 13.6.5 梁柱彎矩強度比](https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf)
    """
    Puc = response.node_neighbor_Puc_matrix
    Puc[Puc > 0] = 0       # only consider compression case
    Ag = structure.node_neighbor_Ag_matrix
    Ag[Ag < 1e-5] = 1e+10  # in case area = 0 for faces not connected with members
    #reduced_stress = np.minimum(Puc/Ag, 0.3*YIELDING_STRESS)  # 0.3我自己訂的，最少保留0.7 yielding stress，這個要再找資料怎麼訂
    reduced_stress = torch.abs(Puc / Ag)
    
    Zz = structure.node_neighbor_Zz_matrix
    ZcFyc =  (Zz * (YIELDING_STRESS - reduced_stress)) @ np.array([0, 0, 1, 1, 0, 0])  # [node_number, 1]
    ZbFyb_x = (YIELDING_STRESS * Zz) @ np.array([1, 1, 0, 0, 0, 0])                    # [node_number, 1]
    ZbFyb_z = (YIELDING_STRESS * Zz) @ np.array([0, 0, 0, 0, 1, 1])                    # [node_number, 1]

    """    
    A = torch.abs(Puc[:, 2:4])
    B = 0.3 * YIELDING_STRESS * Ag[:, 2:4]
    negelected_node_index_list = np.where(np.all(A < B, axis=1))[0].tolist()

    # 13.6.2.1 .梁柱腹板交會區剪力強度
    Vn = 0.6 * YIELDING_STRESS

    node_need_SCWB_list = structure.node_need_strong_column_weak_beam_list.copy()
    for node_index in negelected_node_index_list:
        if node_index in node_need_SCWB_list:
            node_need_SCWB_list.remove(node_index)
    """

    # min_ratio_x = torch.min((ZcFyc / (ZbFyb_x + 1e-6))[structure.node_need_strong_column_weak_beam_list])
    # min_ratio_z = torch.min((ZcFyc / (ZbFyb_z + 1e-6))[structure.node_need_strong_column_weak_beam_list])
    scwb_ratio_x = (ZcFyc / (ZbFyb_x + 1e-6)).numpy()
    scwb_ratio_z = (ZcFyc / (ZbFyb_z + 1e-6)).numpy()
    
    return scwb_ratio_x, scwb_ratio_z


def get_story_shear_ratio(structure: Structure, response: pisa.Response) -> np.ndarray:
    """
    Get story-shear ratio given a specific structure and response.
    - [耐震規範 2.17 極限層剪力強度之檢核](https://www.nlma.gov.tw/filesys/file/EMMA/c1130301-2.pdf)
    """
    shears_Y = np.array(list(response.member_response["shearY"].values()))
    shears_Z = np.array(list(response.member_response["shearZ"].values()))
    member_shears = np.maximum(np.abs(shears_Y), np.abs(shears_Z))

    shear_ratios = np.zeros(structure.story_num - 1)
    for lower_story in range(structure.story_num - 1):
        upper_story = lower_story + 1
        lower_story_column_idxs = structure.story_column_dict[f"{lower_story+1}F"]
        upper_story_column_idxs = structure.story_column_dict[f"{upper_story+1}F"]
        shear_ratios[lower_story] = np.sum(member_shears[lower_story_column_idxs]) / np.sum(member_shears[upper_story_column_idxs])

    return shear_ratios


def get_member_drift_ratio(structure: Structure, response: pisa.Response) -> np.ndarray:
    """
    Get member-drift ratio given a specific structure and response.
    - [耐震規範 2.16.1 容許層間相對側向位移角](https://www.nlma.gov.tw/filesys/file/EMMA/c1130301-2.pdf)
    """
    # node_disp_x = np.array(list(response.node_response["dispX"].values()))
    # node_disp_z = np.array(list(response.node_response["dispZ"].values()))
    # bottom_node_disp_x = node_disp_x[structure.bottom_node_index_list]
    # bottom_node_disp_z = node_disp_z[structure.bottom_node_index_list]

    # member_length = structure.bottom_member_length_array  # mm
    # drift_ratio_x = np.abs((node_disp_x - bottom_node_disp_x) / member_length)
    # drift_ratio_z = np.abs((node_disp_z - bottom_node_disp_z) / member_length)

    node_disp_x = np.array(list(response.node_response["dispX"].values()))
    node_disp_z = np.array(list(response.node_response["dispZ"].values()))

    member_end_nodes = np.array(list(structure.member_to_nodeIndex_dict.values()))[:, 0:2]  # node1_index, node2_index
    member_lengths = np.array(list(structure.member_length_dict.values())) * 1e+3  # unit conversion: m --> mm
    drift_ratio_x = np.abs((node_disp_x[member_end_nodes[:, 1]] - node_disp_x[member_end_nodes[:, 0]]) / member_lengths)
    drift_ratio_z = np.abs((node_disp_z[member_end_nodes[:, 1]] - node_disp_z[member_end_nodes[:, 0]]) / member_lengths)

    return np.maximum(drift_ratio_x, drift_ratio_z)


def get_ratio_column_compression_strength(structure: Structure, response: pisa.Response) -> np.ndarray:
    """
    Get column-compression-strength ratio given a specific structure and response.
     - [鋼構規範(LRFD) 13.4.1 柱強度要求](https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf)
    """
    column_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_column_index_list]
    column_axial_force[column_axial_force > 0] = 0  # only consider compression case

    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_column_index_list]  # kN/mm^2
    A = np.array(list(structure.member_A_dict.values()))[structure.member_column_index_list]      # mm^2
    Puc = Fcr * A  # kN

    return np.abs(column_axial_force) / Puc


def get_ratio_column_tension_strength(structure: Structure, response: pisa.Response) -> np.ndarray:
    """
    Get column-tension-strength ratio given a specific structure and response.
    - [鋼構規範(LRFD) 13.4.1 柱強度要求](https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf)
    """
    column_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_column_index_list]
    column_axial_force[column_axial_force < 0] = 0  # only consider tension case

    A = np.array(list(structure.member_A_dict.values()))[structure.member_column_index_list]  # mm^2
    Put = YIELDING_STRESS * A  # kN

    return np.abs(column_axial_force) / Put




def _too_much_minimum_section(structure: Structure):
    acceptable_rate = 0
    if sum(structure.story_level_sections) <= acceptable_rate * structure.full_section_sum:
        print(f"Already over {(1 - acceptable_rate) * 100}% of sections are minimum")
        return True
    return False


def _check_code(structure: Structure, response: pisa.Response, load_case: load.NodalLoad, check_displacement: bool) -> bool:
    # checking for specific load combinations
    if load_case.E > 0:
        if not _check_soft_story_pass(structure, response): return False, "soft_story"
    if load_case.drift_case and check_displacement:
        if not _check_drift_ratio_pass(structure, response): return False, "drift_ratio"
    if load_case.col_strength_case:
        if not _check_column_compression_strength_pass(structure, response): return False, "column_compression"
        if not _check_column_tension_strength_pass(structure, response): return False, "column_tension"

    # checking for common constraint
    if not _check_beam_strength_pass(structure, response): return False, "beam_strength"
    if not _check_beam_axial_moment_pass(structure, response): return False, "beam_moment"
    if not _check_strong_column_weak_beam_pass(structure, response): return False, "strong_column_weak_beam"

    return True, None




def _check_soft_story_pass(structure: Structure, response: pisa.Response) -> bool:
    """
    耐震規範 2.17 極限層剪力強度之檢核
    https://www.nlma.gov.tw/filesys/file/EMMA/c1130301-2.pdf
    """
    column_shears_Y = np.array(list(response.member_response["shearY"].values()))
    column_shears_Z = np.array(list(response.member_response["shearZ"].values()))
    column_shears = np.maximum(np.abs(column_shears_Y), np.abs(column_shears_Z))
    upper_story_shear = 0
    for story_name in list(structure.story_column_dict.keys())[::-1]:
        story_column_idxs = structure.story_column_dict[story_name]
        story_shear = np.sum(column_shears[story_column_idxs])
        if story_shear < STORY_SHEAR_RATIO_LIMIT * upper_story_shear:
            print("Soft story didn't pass")
            return False
        upper_story_shear = story_shear
    return True


def _check_drift_ratio_pass(structure: Structure, response: pisa.Response) -> bool:
    """
    耐震規範 2.16.1 容許層間相對側向位移角
    https://www.nlma.gov.tw/filesys/file/EMMA/c1130301-2.pdf
    """
    node_disp = np.array(list(response.node_response["disp"].values()))
    bottom_node_disp = node_disp[structure.bottom_node_index_list]
    member_length = structure.bottom_member_length_array  # mm
    drift_ratio = np.abs((node_disp - bottom_node_disp) / member_length)
    # print("drift ratio:", drift_ratio)
    if np.max(drift_ratio) > STORY_DRIFT_RATIO_LIMIT:
        print("Drift ratio didn't pass!")
        return False
    return True


def _check_column_compression_strength_pass(structure: Structure, response: pisa.Response) -> bool:
    """
    鋼構規範(LRFD) 13.4.1 柱強度要求
    https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf
    """
    # assume tension is positive value, compression is negative value
    column_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_column_index_list]
    compression_condition = (column_axial_force <= 0)
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_column_index_list]  # kN/mm2
    A = np.array(list(structure.member_A_dict.values()))[structure.member_column_index_list]      # mm2
    Puc = Fcr * A  # kN
    # print("column compress ratio:", (np.abs(column_axial_force) > PHI_C * Puc)[compression_condition])
    if np.any((np.abs(column_axial_force) > PHI_C * Puc)[compression_condition]):
        print("column compression strength didn't pass")
        return False
    return True


def _check_column_tension_strength_pass(structure: Structure, response: pisa.Response) -> bool:
    """
    鋼構規範(LRFD) 13.4.1 柱強度要求
    https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf
    """
    column_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_column_index_list]
    tension_condition = (column_axial_force > 0)
    A = np.array(list(structure.member_A_dict.values()))[structure.member_column_index_list]  # mm2
    Put = YIELDING_STRESS * A  # kN
    # print("column tension ratio:", np.abs((column_axial_force / (PHI_C * Put))[tension_condition]))
    if np.any((np.abs(column_axial_force) > PHI_C * Put)[tension_condition]):
        print("column tension strength didn't pass")
        return False
    return True


def _check_beam_strength_pass(structure: Structure, response: pisa.Response) -> bool:
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    tension_condition = (beam_axial_force > 0)
    compression_condition = (beam_axial_force <= 0)
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_beam_index_list]  # kN/mm2
    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]      # mm2
    Put = YIELDING_STRESS * A  # kN
    Puc = Fcr * A  # kN
    if np.any((np.abs(beam_axial_force) > PHI_C * Put)[tension_condition]):
        print("beam tension strength didn't pass")
        return False
    if np.any((np.abs(beam_axial_force) > PHI_C * Puc)[compression_condition]):
        print("beam compression strength didn't pass")
        return False
    return True


def _check_beam_axial_moment_pass(structure: Structure, response: pisa.Response) -> bool:
    """
    鋼構規範(LRFD) 8.2 對稱構材承受彎矩及軸力之作用
    https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/3495-8.pdf
    """
    PHI, PHI_b = 0.85, 0.90
    # demand
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    Pu = np.abs(beam_axial_force)
    Mux = np.abs(np.array(list(response.member_response["momentZ"].values()))[structure.member_beam_index_list])
    Muy = np.abs(np.array(list(response.member_response["momentY"].values()))[structure.member_beam_index_list])

    # capacity
    tension_condition = (beam_axial_force > 0)
    compression_condition = (beam_axial_force <= 0)

    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]  # mm2
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_beam_index_list]  # kN/mm2
    Puc = Fcr * A  # kN
    Put = YIELDING_STRESS * A  # kN

    Pn = np.zeros_like(Pu)
    Pn += Put * tension_condition
    Pn += Puc * compression_condition
    phi_Pn = PHI * Pn

    Mnx = np.array(list(structure.member_Mnx_dict.values()))[structure.member_beam_index_list]
    Mny = np.array(list(structure.member_Mny_dict.values()))[structure.member_beam_index_list]
    phi_Mnx = PHI_b * Mnx
    phi_Mny = PHI_b * Mny
    
    # case 1:  Pu/phi*Pn >= 0.2:
    case_1 = ((Pu / phi_Pn) >= 0.2)
    ratio_1 = ((Pu / phi_Pn) + 8/9 * (Mux / phi_Mnx + Muy / phi_Mny)) * case_1
    if np.max(ratio_1) > 1:
        print("beam force ratio didn't pass")
        return False

    # case 2:  Pu/phi*Pn < 0.2:
    case_2 = ((Pu / phi_Pn) < 0.2)
    ratio_2 = ((Pu / (2 * phi_Pn)) + (Mux / phi_Mnx + Muy / phi_Mny)) * case_2
    if np.max(ratio_2) > 1:
        print("beam force ratio didn't pass")
        return False
    return True
        

def _check_strong_column_weak_beam_pass(structure: Structure, response: pisa.Response) -> bool:
    """
    鋼構規範(LRFD) 13.6.5 梁柱彎矩強度比
    https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf
    """
    Zc = structure.node_neighbor_Zz_matrix
    Puc = response.node_neighbor_Puc_matrix
    Puc[Puc > 0] = 0       # only consider compression case
    Ag = structure.node_neighbor_Ag_matrix
    Ag[Ag < 1e-5] = 1e+10  # in case area = 0 for faces not connected with members
    
    #reduced_stress = np.minimum(Puc/Ag, 0.3*YIELDING_STRESS)  # 0.3我自己訂的，最少保留0.7 yielding stress，這個要再找資料怎麼訂
    reduced_stress = torch.abs(Puc / Ag)

    ZcFyc =  (Zc * (YIELDING_STRESS - reduced_stress)) @ np.array([0, 0, 1, 1, 0, 0])               # [node_number, 1]
    ZbFyb_x = (YIELDING_STRESS * structure.node_neighbor_Zz_matrix) @ np.array([1, 1, 0, 0, 0, 0])  # [node_number, 1]
    ZbFyb_z = (YIELDING_STRESS * structure.node_neighbor_Zz_matrix) @ np.array([0, 0, 0, 0, 1, 1])  # [node_number, 1]
    
    ratio_x = (ZcFyc / (ZbFyb_x + 1e-6))[structure.node_need_strong_column_weak_beam_list]
    ratio_z = (ZcFyc / (ZbFyb_z + 1e-6))[structure.node_need_strong_column_weak_beam_list]
    if torch.any(ratio_x < 1.25) or torch.any(ratio_z < 1.25):
        print("strong column weak beam didn't pass:", torch.min(ratio_x), torch.min(ratio_z))
        return False
    return True

