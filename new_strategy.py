import torch
import random
import logging
import numpy as np
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys
sys.path.append("./RL/")
sys.path.append("./Structure/")
sys.path.append("./Visualization/")
sys.path.append("./NonlinearDynamicAnalysisSimulator/")

from Structure import structure as struc
from Structure import check

from RL import agent
from RL import environment
from RL.agent import train
from Visualization import plot
from Visualization import visualize
from NonlinearDynamicAnalysisSimulator import load_simulator


def get_loggings(ckpt_dir, name="record"):
	logger = logging.getLogger(name='Graph-RL')
	logger.setLevel(level=logging.INFO)
	
    # set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	
    # console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	
    # file handler
	file_handler = logging.FileHandler(ckpt_dir / f"{name}.log")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
     
	return logger


def test_section_pool_lower_bound(logger, analysis_dir):
    fail_geometry = []
    for x_span_num in range(2, 7):
        for z_span_num in range(2, 7):
            for story_num in range(4, 8):

                x_span_len, z_span_len, story_height = 7000, 7000, 3200
                x_span_lens = [x_span_len for i in range(x_span_num)]
                z_span_lens = [z_span_len for i in range(z_span_num)]

                geo_name = f"x{x_span_num}_y{story_num}_z{z_span_num}"
                logger.critical(f"geo_name: {geo_name}\n")

                action_set = [[story_num * i for i in range(4)]] * 14
                #print(action_set)

                structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                                    "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                                    "story_num": story_num, "story_height": story_height,
                                    "add_structure_geometry": True, 
                                    "do_nonlinear_dynamic_analysis": False,
                                    "nda_norm_dict": None,
                                    "analysis_dir": analysis_dir}
                
                structure = struc.Structure(**structure_kwargs)
                logger.info(f"material_usage: {structure.calculate_material_usage():.3f}")
                logger.info(f"story_level_section: {structure.story_level_sections}")

                whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, True, analysis_dir)
                logger.info(f"fail_name: {fail_name} , fail_reason: {fail_reason}\n")

                for actions in action_set:
                    for action in actions:
                        _ = structure.update_action(action)

                    logger.info(f"material_usage: {structure.calculate_material_usage():.3f}")
                    logger.info(f"story_level_section: {structure.story_level_sections}")

                    whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, True, analysis_dir)
                    logger.info(f"fail_name: {fail_name} , fail_reason: {fail_reason}\n")

                    if whether_pass is False:
                        fail_geometry.append(geo_name)
                        break

    logger.critical(f"fail_geometry_num: {len(fail_geometry)}")
    logger.critical(f"fail_geometry: \n{fail_geometry}")


def test_original_section_quick_failure(logger, analysis_dir):
    for x_span_num in range(2, 7):
        for z_span_num in range(2, 7):
            for story_num in range(4, 8):
                x_span_len, z_span_len, story_height = 7000, 7000, 3200
                x_span_lens = [x_span_len for i in range(x_span_num)]
                z_span_lens = [z_span_len for i in range(z_span_num)]

                geo_name = f"x{x_span_num}_y{story_num}_z{z_span_num}"
                logger.critical(f"geo_name: {geo_name}\n")

                structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                                    "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                                    "story_num": story_num, "story_height": story_height,
                                    "add_structure_geometry": True, 
                                    "do_nonlinear_dynamic_analysis": False,
                                    "nda_norm_dict": None,
                                    "analysis_dir": analysis_dir}
                
                initial_structure = struc.Structure(**structure_kwargs)
                
                quick_fail_actions = [story_num * i - 1 for i in range(1, 4+1)]
                
                # X-beam & Z-beam
                structure = deepcopy(initial_structure)
                _ = structure.update_action(quick_fail_actions[0])
                _ = structure.update_action(quick_fail_actions[1])
                logger.info(f"material_usage: {structure.calculate_material_usage():.3f}")
                logger.info(f"story_level_section: {structure.story_level_sections}")
                whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, True, analysis_dir)
                logger.info(f"fail_name: {fail_name} , fail_reason: {fail_reason}\n")

                # out-column
                structure = deepcopy(initial_structure)
                _ = structure.update_action(quick_fail_actions[2])
                logger.info(f"material_usage: {structure.calculate_material_usage():.3f}")
                logger.info(f"story_level_section: {structure.story_level_sections}")
                whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, True, analysis_dir)
                logger.info(f"fail_name: {fail_name} , fail_reason: {fail_reason}\n")

                # in-column
                structure = deepcopy(initial_structure)
                _ = structure.update_action(quick_fail_actions[3])
                logger.info(f"material_usage: {structure.calculate_material_usage():.3f}")
                logger.info(f"story_level_section: {structure.story_level_sections}")
                whether_pass, fail_name, fail_reason, auxiliary_values = check.check(structure, True, analysis_dir)
                logger.info(f"fail_name: {fail_name} , fail_reason: {fail_reason}\n")
                

if __name__ == "__main__":
    ckpt_dir = Path("./NewStrategy/")
    analysis_dir = ckpt_dir / "PISA_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger = get_loggings(ckpt_dir, name="quick_fail")
    #test_section_pool_lower_bound(logger, analysis_dir)
    test_original_section_quick_failure(logger, analysis_dir)