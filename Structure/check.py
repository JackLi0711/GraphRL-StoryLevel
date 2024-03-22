import torch
import logging
import numpy as np
from pathlib import Path
from Structure import load
from Structure import pisa
from Structure import earthquake
from Structure.sections import *
from Structure.structure import Structure


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


def check(structure: Structure, 
          analysis_dir: Path,
          auxiliary_values: dict[str, float]=None,
          load_cases: list[load.NodalLoad]=None,
          responses: list[pisa.Response]=None,
          check_displacement: bool=True) -> tuple[bool, str, str, dict[str, float]]:
    '''Check structural responses whether pass constraints or not under various load cases.'''
    if auxiliary_values is None:
        auxiliary_values, load_cases, responses = get_response(structure, analysis_dir)

    for load_case, response in list(zip(load_cases, responses)):
        whether_pass, fail_reason = _check_code(structure, response, load_case, check_displacement)
        if whether_pass is False:
            return False, load_case.load_name, fail_reason, auxiliary_values
    
    #if _too_much_minimum_section(structure): return False, None, "minimum_section", auxiliary_values
        
    return True, None, None, auxiliary_values


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




# 耐震規範 2.17 極限層剪力強度之檢核
UPPER_STORY_SHEAR_RATIO = 0.80
def _check_soft_story_pass(structure: Structure, response: pisa.Response) -> bool:
    column_shears_Y = np.array(list(response.member_response["shearY"].values()))
    column_shears_Z = np.array(list(response.member_response["shearZ"].values()))
    column_shears = np.maximum(np.abs(column_shears_Y), np.abs(column_shears_Z))
    upper_story_shear = 0
    for story_name in list(structure.story_column_dict.keys())[::-1]:
        story_column_idxs = structure.story_column_dict[story_name]
        story_shear = np.sum(column_shears[story_column_idxs])
        if story_shear < UPPER_STORY_SHEAR_RATIO * upper_story_shear:
            print("Soft story didn't pass")
            return False
        upper_story_shear = story_shear
    return True


# 耐震規範 2.16 層間相對側向位移與建築物之間隔
DRIFT_RAIO_LIMIT = 0.005
def _check_drift_ratio_pass(structure: Structure, response: pisa.Response) -> bool:
    node_disp = np.array(list(response.node_response["disp"].values()))
    bottom_node_disp = node_disp[structure.bottom_node_index_list]
    member_length = structure.bottom_member_length_array    # mm
    drift_ratio = np.abs((node_disp - bottom_node_disp) / member_length)
    # print("drift ratio:", drift_ratio)
    if np.max(drift_ratio) > DRIFT_RAIO_LIMIT:
        print("Drift ratio didn't pass!")
        return False
    return True


# 鋼構規範(LRFD) 13.4.1 柱強度要求
PHI_C = 0.85
def _check_column_compression_strength_pass(structure: Structure, response: pisa.Response) -> bool:
    # assume tension is positive value, compression is negative value
    column_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_column_index_list]
    compression_condition = (column_axial_force <= 0)
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_column_index_list]    # kN/mm2
    A = np.array(list(structure.member_A_dict.values()))[structure.member_column_index_list]        # mm2
    Puc = Fcr * A     # kN
    # print("column compress ratio:", (np.abs(column_axial_force) > PHI_C * Puc)[compression_condition])
    if np.any((np.abs(column_axial_force) > PHI_C * Puc)[compression_condition]):
        print("column compression strength didn't pass")
        return False
    return True


# 鋼構規範(LRFD) 13.4.1 柱強度要求
def _check_column_tension_strength_pass(structure: Structure, response: pisa.Response) -> bool:
    column_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_column_index_list]
    tension_condition = (column_axial_force > 0)
    A = np.array(list(structure.member_A_dict.values()))[structure.member_column_index_list]        # mm2
    Put = YIELDING_STRESS * A    # kN
    # print("column tension ratio:", np.abs((column_axial_force / (PHI_C * Put))[tension_condition]))
    if np.any((np.abs(column_axial_force) > PHI_C * Put)[tension_condition]):
        print("column tension strength didn't pass")
        return False
    return True


def _check_beam_strength_pass(structure: Structure, response: pisa.Response) -> bool:
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    tension_condition = (beam_axial_force > 0)
    compression_condition = (beam_axial_force <= 0)
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_beam_index_list]    # kN/mm2
    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]        # mm2
    Put = YIELDING_STRESS * A    # kN
    Puc = Fcr * A     # kN
    if np.any((np.abs(beam_axial_force) > PHI_C * Put)[tension_condition]):
        print("beam tension strength didn't pass")
        return False
    if np.any((np.abs(beam_axial_force) > PHI_C * Puc)[compression_condition]):
        print("beam compression strength didn't pass")
        return False
    return True


# 鋼構規範(LRFD) 8.2 構材承受彎矩及軸力之作用
PHI, PHI_b = 0.85, 0.90
def _check_beam_axial_moment_pass(structure: Structure, response: pisa.Response) -> bool:
    # demand
    beam_axial_force = np.array(list(response.member_response["axial"].values()))[structure.member_beam_index_list]
    Pu = np.abs(beam_axial_force)
    Mux = np.abs(np.array(list(response.member_response["momentZ"].values()))[structure.member_beam_index_list])
    Muy = np.abs(np.array(list(response.member_response["momentY"].values()))[structure.member_beam_index_list])

    # capacity
    tension_condition = (beam_axial_force > 0)
    compression_condition = (beam_axial_force <= 0)

    A = np.array(list(structure.member_A_dict.values()))[structure.member_beam_index_list]        # mm2
    Put = YIELDING_STRESS * A  # kN
    Fcr = np.array(list(structure.member_Fcr_dict.values()))[structure.member_beam_index_list]    # kN/mm2
    Puc = Fcr * A  # kN

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
        

# 鋼構規範(LRFD) 13.6.5 梁柱彎矩強度比
def _check_strong_column_weak_beam_pass(structure: Structure, response: pisa.Response) -> bool:
    Zc = structure.node_neighbor_Zz_matrix
    Puc = response.node_neighbor_Puc_matrix
    Puc[Puc > 0] = 0        # only consider compression case
    Ag = structure.node_neighbor_Ag_matrix
    Ag[Ag < 1e-5] = 1e+10   # in case area = 0 for faces not connected with members
    
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


