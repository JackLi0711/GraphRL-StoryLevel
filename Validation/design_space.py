import sys
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

sys.path.append("D:/GraphRL_story_level")
from Structure.structure import Structure


BEAM_OPTIONS = 15    # original: 9
COLUMN_OPTIONS = 15  # original: 9


def sample_reductions_from_design_space(sample_num: int, story_num: int) -> List[int]:
    """
    return a list containing reduces time for beam and another for column
    for example:
        beam: [0, 1, 1, 2, 3] means first floor beam should't be reduced,
        and 5th floor beam should be reduced three time.
    """

    lightest = story_num * (2 * (COLUMN_OPTIONS - 1) + 2 * (BEAM_OPTIONS - 1))  # original: ... + 1 * (BEAM_OPTIONS - 1))
    heaviest = 0
    group_num = 6
    sample_in_group = int(sample_num / group_num) + 1
    extra_group = 4
    weight_interval = (lightest - heaviest) / (group_num + extra_group)
    member_reduction_groups = [[] for _ in range(group_num + extra_group)]

    total_num = 0
    iter_count = 0

    while total_num < sample_num:

        current_nums = [len(member_reduction_groups[i]) for i in range(group_num + extra_group)]
        print(f"iter: {iter_count}, current:", current_nums)

        # generate initial reductions
        xdir_beam_reduction = list(np.random.randint(0, BEAM_OPTIONS, size=(story_num)))
        zdir_beam_reduction = list(np.random.randint(0, BEAM_OPTIONS, size=(story_num)))
        outer_column_reduction = list(np.random.randint(0, COLUMN_OPTIONS, size=(story_num)))
        inner_column_reduction = list(np.random.randint(0, COLUMN_OPTIONS, size=(story_num)))

        # sort them to make them monotonic increasing, so that lower members are stronger than uppers
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

    return member_reductions




def sample_structures_from_design_space(sample_num=10000, 
                                        do_nonlinear_dynamic_analysis=None,
                                        nda_norm_dict=None,
                                        analysis_dir=None,
                                        x_span_num=3, z_span_num=3, story_num=5,
                                        x_span_len=6000, z_span_len=8000,
                                        x_span_lens=[6000]*3, 
                                        z_span_lens=[8000]*3) -> Tuple[List[Structure], List[float]]:

    # first get the reductions
    member_reductions = sample_reductions_from_design_space(sample_num, story_num)

    # generate structures
    structures = []
    rewards = []
    for member_reduction in tqdm(member_reductions):
        # generate testing structure
        story_height = 3200
        structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, "z_span_num": z_span_num,
                            "z_span_lens": z_span_lens, "story_num": story_num, "story_height": story_height,
                            "add_structure_geometry": True, 
                            "do_nonlinear_dynamic_analysis": do_nonlinear_dynamic_analysis,
                            "nda_norm_dict": nda_norm_dict,
                            "analysis_dir": analysis_dir}
        structure = Structure(**structure_kwargs)

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

    return structures, rewards




if __name__ == "__main__":
    structures = sample_reductions_from_design_space(sample_num=1000, story_num=4)
    for structure in structures:
        print(structure)


