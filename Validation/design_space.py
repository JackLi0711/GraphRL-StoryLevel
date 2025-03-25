import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("D:/GraphRL_storyLevel")
from Structure import check, check_nda
from Structure.structure import Structure
from Structure.sections import beam_sections, column_sections


# Tony's implementation --> material usage distribution of sampled structures seems to be OK
def sample_reductions_from_design_space(sample_num: int, story_num: int) -> List[int]:
    """
    return a list containing the number of reduction for beams and another for columns
    for example:
        beam: [0, 1, 1, 2, 3] means first floor beam should't be reduced,
        and 5th floor beam should be reduced three times.
    """
    lightest = story_num * (2 * (len(beam_sections) - 1) + 2 * (len(column_sections) - 1))  # original: ... + 1 * (BEAM_OPTIONS - 1))
    heaviest = 0
    group_num = 6
    extra_group = 4
    sample_in_group = int(sample_num / group_num) + 1
    weight_interval = (lightest - heaviest) / (group_num + extra_group)
    member_reduction_groups = [[] for _ in range(group_num + extra_group)]

    total_num = 0
    iter_count = 0
    while total_num < sample_num:
        current_nums = [len(member_reduction_groups[i]) for i in range(group_num + extra_group)]
        print(f"iter: {iter_count}, current: {current_nums}")

        # generate initial reductions
        xdir_beam_reduction = list(np.random.randint(0, len(beam_sections), size=(story_num)))
        zdir_beam_reduction = list(np.random.randint(0, len(beam_sections), size=(story_num)))
        outer_column_reduction = list(np.random.randint(0, len(column_sections), size=(story_num)))
        inner_column_reduction = list(np.random.randint(0, len(column_sections), size=(story_num)))

        # sort them to make them monotonic increasing, so that lower members are stronger than upper ones
        xdir_beam_reduction.sort()
        zdir_beam_reduction.sort()
        outer_column_reduction.sort()
        inner_column_reduction.sort()

        reduction = xdir_beam_reduction + zdir_beam_reduction + outer_column_reduction + inner_column_reduction
        reduction_weight = sum(reduction)
        weight_index = min(int(reduction_weight / weight_interval), group_num + extra_group - 1)
        if len(member_reduction_groups[weight_index]) < sample_in_group:
            member_reduction_groups[weight_index].append(reduction)
            total_num += 1
        
        iter_count += 1

    member_reductions = []
    for reduction_group in member_reduction_groups:
        member_reductions.extend(reduction_group)
    assert len(member_reductions) == sample_num

    return member_reductions

def sample_structures_from_design_space(sample_num=1000, 
                                        x_span_num=3, x_span_lens=[6000]*3, 
                                        z_span_num=3, z_span_lens=[8000]*3, 
                                        story_num=5, story_height=3200, 
                                        do_nonlinear_dynamic_analysis=False,
                                        nda_norm_dict=None,
                                        analysis_dir=None) -> Tuple[Structure, Structure, List[Structure], List[float]]:
    structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                        "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                        "story_num": story_num, "story_height": story_height,
                        "story_level_sections": None,
                        "add_structure_geometry": False, 
                        "add_response_features": False,
                        "do_nonlinear_dynamic_analysis": do_nonlinear_dynamic_analysis,
                        "nda_norm_dict": nda_norm_dict,
                        "analysis_dir": analysis_dir}
    """Based on weight grouping to sample, which may **contain fail cases**"""
    # generate the lightest structure
    lightest_design = [0] * (story_num*2) + [0] * (story_num*2)
    lightest_kwargs = deepcopy(structure_kwargs)
    lightest_kwargs["story_level_sections"] = lightest_design
    lightest_structure = Structure(**lightest_kwargs)
    
    # generate the heaviest structure
    heaviest_design = [len(beam_sections)-1] * (story_num*2) + [len(column_sections)-1] * (story_num*2)
    heaviest_kwargs = deepcopy(structure_kwargs)
    heaviest_kwargs["story_level_sections"] = heaviest_design
    heaviest_structure = Structure(**heaviest_kwargs)
    
    # generate sampled structures
    member_reductions = sample_reductions_from_design_space(sample_num, story_num)
    structures = []
    rewards = []
    for member_reduction in tqdm(member_reductions):
        # generate sampled structure
        initial_design = [len(beam_sections)-1] * (story_num*2) + [len(column_sections)-1] * (story_num*2)
        sampled_kwargs = deepcopy(structure_kwargs)
        sampled_kwargs["story_level_sections"] = initial_design
        structure = Structure(**sampled_kwargs)

        # reduce the sections, starts from the upper beam columns
        initial_material_usage = structure.calculate_material_usage()
        reward = 0
        for update_story_index in range(len(member_reduction)-1, -1, -1):
            reduced_times = member_reduction[update_story_index]
            for _ in range(reduced_times):
                reward += structure.update_action(update_story_index)
        final_material_usage = structure.calculate_material_usage()
        material_usage_difference = initial_material_usage - final_material_usage
        print(f"accumulated_saved_material: {reward:5.3f}, material_usage_diff: {material_usage_difference:5.3f}")

        structures.append(structure)
        rewards.append(reward)

    return lightest_structure, heaviest_structure, structures, rewards

def sample_pass_structures_from_design_space(sample_num=1000,
                                             x_span_num=4, x_span_lens=[6000]*3,
                                             z_span_num=4, z_span_lens=[8000]*3,
                                             story_num=6, story_height=3200,
                                             do_nonlinear_dynamic_analysis=False,
                                             nda_norm_dict=None,
                                             analysis_dir=None, 
                                             check_kwargs=None) -> Tuple[Structure, Structure, List[Structure], List[float]]:
    structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens,
                        "z_span_num": z_span_num, "z_span_lens": z_span_lens,
                        "story_num": story_num, "story_height": story_height,
                        "story_level_sections": None,
                        "add_structure_geometry": False,
                        "add_response_features": False,
                        "do_nonlinear_dynamic_analysis": do_nonlinear_dynamic_analysis,
                        "nda_norm_dict": nda_norm_dict,
                        "analysis_dir": analysis_dir}
    # generate the lightest structure
    lightest_design = [0] * (story_num*2) + [0] * (story_num*2)
    lightest_kwargs = deepcopy(structure_kwargs)
    lightest_kwargs["story_level_sections"] = lightest_design
    lightest_structure = Structure(**lightest_kwargs)
    
    # generate the heaviest structure
    heaviest_design = [len(beam_sections)-1] * (story_num*2) + [len(column_sections)-1] * (story_num*2)
    heaviest_kwargs = deepcopy(structure_kwargs)
    heaviest_kwargs["story_level_sections"] = heaviest_design
    heaviest_structure = Structure(**heaviest_kwargs)

    # generate sampled structures
    structures = []
    rewards = []
    sampling_record = {"geometry": [], "final_design": [], 
                       "final_volume": [], "saved_material": [], 
                       "whether_pass": [], "fail_name": [], "fail_reason": []}
    while len(structures) < sample_num:
        story_xdir_beam_sections = sorted(np.random.randint(0, len(beam_sections), size=story_num).tolist(), reverse=True)
        story_zdir_beam_sections = sorted(np.random.randint(0, len(beam_sections), size=story_num).tolist(), reverse=True)
        story_outer_column_sections = sorted(np.random.randint(0, len(column_sections), size=story_num).tolist(), reverse=True)
        story_inner_column_sections = sorted(np.random.randint(0, len(column_sections), size=story_num).tolist(), reverse=True)
        initial_design = story_xdir_beam_sections + story_zdir_beam_sections + story_outer_column_sections + story_inner_column_sections
        sampled_kwargs = deepcopy(structure_kwargs)
        sampled_kwargs["story_level_sections"] = initial_design
        structure = Structure(**sampled_kwargs)
        print(structure.story_level_sections)

        # 1. check if linear static analysis response pass regulations
        auxiliary_values, load_cases, static_responses = check.get_response(structure, check_kwargs["code_analysis_dir"])
        static_constraint_condition, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
        whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, check_kwargs["check_displacement"])

        # 2. check if nonlinear dynamic analysis response pass regulations if needed
        if do_nonlinear_dynamic_analysis and whether_pass == True:
            dynamic_responses = check_nda.get_response(structure, check_kwargs["nda_simulator"], check_kwargs["MCE_ground_motion_set"], check_kwargs["device"])
            dynamic_constraint_condition, dynamic_response_features, dynamic_response_rewards = check_nda.process_response(structure, dynamic_responses, nda_norm_dict)
            whether_pass, fail_name, fail_reason = check_nda.check_pass(dynamic_constraint_condition, check_kwargs["check_displacement"])

        if whether_pass:
            reward = heaviest_structure.calculate_material_usage() - structure.calculate_material_usage()
            structures.append(structure)
            rewards.append(reward)
            print(f"Pass! accumulated_saved_material: {reward:5.3f}, current structure num: {len(structures)} \n")
            sampling_record["geometry"].append([structure.x_span_num, structure.z_span_num, structure.story_num, structure.x_span_lens, structure.z_span_lens, structure.story_height])
            sampling_record["final_design"].append([i for i in structure.story_level_sections])
            sampling_record["final_volume"].append(structure.calculate_material_usage())
            sampling_record["saved_material"].append(reward)
            sampling_record["whether_pass"].append(whether_pass)
            sampling_record["fail_name"].append(fail_name)
            sampling_record["fail_reason"].append(fail_reason)
    
    assert sampling_record["whether_pass"].count(True) == sample_num
    assert sampling_record["fail_name"].count(None) == sample_num
    assert sampling_record["fail_reason"].count(None) == sample_num
    with open(check_kwargs["save_root"] / "sampling_record_AllPass.txt", 'w') as f: json.dump(sampling_record, f)
    
    return lightest_structure, heaviest_structure, structures, rewards


def check_sample_diversity(lightest_structure: Structure, heaviest_structure: Structure, structures: List[Structure], rewards: List[float], save_root: Path) -> None:
    """Check the diversity of the sampled structures by plotting the material usage distribution."""
    lightest_weight = lightest_structure.calculate_material_usage()
    heaviest_weight = heaviest_structure.calculate_material_usage()
    weights = [structure.calculate_material_usage() for structure in structures]

    plt.hist(weights, bins=10, range=(lightest_weight, heaviest_weight), edgecolor="black")
    plt.vlines(lightest_weight, 0, len(structures)/5, colors='r', linestyles="dashed", label=f"lightest weight: {lightest_weight:.3f}")
    plt.vlines(heaviest_weight, 0, len(structures)/5, colors='r', linestyles="dashed", label=f"heaviest weight: {heaviest_weight:.3f}")
    plt.xlabel("material usage (m3)")
    plt.ylabel("Count")
    geo_name = save_root.name  # e.g., x2_z2_y4
    plt.title(f"Material Usage Distribution of {len(structures)} {geo_name} Samples")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig(save_root / "material_usage_distribution_AllPass.png")
    plt.close()


# Jack's implementation (based on actual weight grouping) --> spend too much time
def sample_designs_from_design_space(sample_num=1000, 
                                     x_span_num=4, x_span_lens=[6000]*4,
                                     z_span_num=4, z_span_lens=[8000]*4,
                                     story_num=6, story_height=3200,
                                     do_nonlinear_dynamic_analysis=False,
                                     nda_norm_dict=None,
                                     analysis_dir=None) -> Tuple[Structure, Structure, List[Structure], List[float]]:
    """
    Sample structures from the design space and calculate according saved material,
    based on the actual weight grouping to ensure the diversity of the samples.
    """
    structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                        "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                        "story_num": story_num, "story_height": story_height,
                        "story_level_sections": None, 
                        "add_structure_geometry": False,
                        "add_response_features": False,
                        "do_nonlinear_dynamic_analysis": do_nonlinear_dynamic_analysis, 
                        "nda_norm_dict": nda_norm_dict,
                        "analysis_dir": analysis_dir}
    
    lightest_design = [0] * (story_num*2) + [0] * (story_num*2)
    lightest_kwargs = deepcopy(structure_kwargs)
    lightest_kwargs["story_level_sections"] = lightest_design
    lightest_structure = Structure(**lightest_kwargs)
    lightest_weight = lightest_structure.calculate_material_usage()
    print(lightest_structure.story_level_sections, lightest_weight)

    heaviest_design = [len(beam_sections)-1] * (story_num*2) + [len(column_sections)-1] * (story_num*2)
    heaviest_kwargs = deepcopy(structure_kwargs)
    heaviest_kwargs["story_level_sections"] = heaviest_design
    heaviest_structure = Structure(**heaviest_kwargs)
    heaviest_weight = heaviest_structure.calculate_material_usage()
    print(heaviest_structure.story_level_sections, heaviest_weight)

    group_num = 6
    extra_group = 4
    sample_in_group = int(sample_num / group_num) + 1
    weight_interval = (heaviest_weight - lightest_weight) / (group_num + extra_group)
    structure_groups = [[] for _ in range(group_num + extra_group)]
    reward_groups = [[] for _ in range(group_num + extra_group)]

    total_num = 0
    iter_count = 0
    while total_num < sample_num:
        current_nums = [len(structure_groups[i]) for i in range(group_num + extra_group)]
        print(f"iter: {iter_count}, current: {current_nums}")

        # generate initial design
        story_xdir_beam_sections = list(np.random.randint(0, len(beam_sections), size=(story_num)))
        story_zdir_beam_sections = list(np.random.randint(0, len(beam_sections), size=(story_num)))
        story_outer_column_sections = list(np.random.randint(0, len(column_sections), size=(story_num)))
        story_inner_column_sections = list(np.random.randint(0, len(column_sections), size=(story_num)))
        
        # sort them to make them monotonic decreasing, so that lower members are stronger than upper ones
        story_xdir_beam_sections.sort(reverse=True)
        story_zdir_beam_sections.sort(reverse=True)
        story_outer_column_sections.sort(reverse=True)
        story_inner_column_sections.sort(reverse=True)
        sampled_design = story_xdir_beam_sections + story_zdir_beam_sections + story_outer_column_sections + story_inner_column_sections

        sampled_kwargs = deepcopy(structure_kwargs)
        sampled_kwargs["story_level_sections"] = sampled_design
        sampled_structure = Structure(**sampled_kwargs)
        sampled_weight = sampled_structure.calculate_material_usage()
        print(sampled_structure.story_level_sections, sampled_weight)

        interval_index = int((sampled_weight - lightest_weight) / weight_interval)  # round down to integer
        interval_index = min(interval_index, group_num+extra_group-1)
        if len(structure_groups[interval_index]) < sample_in_group:
            structure_groups[interval_index].append(sampled_structure)
            reward_groups[interval_index].append(heaviest_weight - sampled_weight)
            total_num += 1
        
        iter_count += 1
    
    sampled_structures = []
    sampled_rewards = []
    for i in range(group_num + extra_group):
        sampled_structures.extend(structure_groups[i])
        sampled_rewards.extend(reward_groups[i])
    print(sampled_rewards)
    assert (len(sampled_structures) == sample_num) and (len(sampled_rewards) == sample_num)

    return lightest_structure, heaviest_structure, sampled_structures, sampled_rewards




if __name__ == "__main__":
    # Tony's implementation
    # structures = sample_reductions_from_design_space(sample_num=1000, story_num=4)
    # for structure in structures:
    #     print(structure)

    lightest_structure, heaviest_structure, structures, rewards = sample_structures_from_design_space(sample_num=1000, story_num=5)
    check_sample_diversity(lightest_structure, heaviest_structure, structures, rewards)

    # Jack's implementation
    # lightest_structure, heaviest_structure, structures, rewards = sample_designs_from_design_space()
    # check_sample_diversity(lightest_structure, heaviest_structure, structures, rewards)
