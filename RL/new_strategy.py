import random
import numpy as np

from Structure import sections


def sample_initial_story_sections(x_span_num: int, x_span_len: int, 
                                  z_span_num: int, z_span_len: int, 
                                  story_num: int) -> list[int]:
    min_geo_sum = (2+6) + (2+6) + 4
    max_geo_sum = (6+8) + (6+8) + 7
    geo_sum = (x_span_num + x_span_len/1000) + (z_span_num + z_span_len/1000) + story_num
    main_type = geo_sum - min_geo_sum
    
    section_pool = np.array([i for i in range(len(sections.beam_sections))])
    distance = np.abs(section_pool - main_type)
    exp_negative_distance = np.exp(-1 * distance * 0.25)
    sample_prob = exp_negative_distance / np.sum(exp_negative_distance)

    story_outer_column_section = sorted(random.choices(section_pool, weights=sample_prob, k=story_num), reverse=True)
    story_inner_column_section = sorted(random.choices(section_pool, weights=sample_prob, k=story_num), reverse=True)
    mean_column_section = round(np.mean(story_outer_column_section + story_inner_column_section))
    story_xdir_beam_section = [mean_column_section for _ in range(story_num)]
    story_zdir_beam_section = [mean_column_section for _ in range(story_num)]

    story_level_sections = story_xdir_beam_section + story_zdir_beam_section + story_outer_column_section + story_inner_column_section
    
    return story_level_sections