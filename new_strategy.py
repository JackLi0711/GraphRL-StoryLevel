import torch
import random
import logging
import numpy as np
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


ckpt_dir = Path("./NewStrategy/")
analysis_dir = ckpt_dir / "PISA_Analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='Graph-RL')
	logger.setLevel(level=logging.INFO)
	
    # set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	
    # console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	
    # file handler
	file_handler = logging.FileHandler(ckpt_dir / "record.log")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
     
	return logger


logger = get_loggings(ckpt_dir)

fail_geometry = []
for x_span_num in range(2, 7):
    for z_span_num in range(2, 7):
        for story_num in range(4, 8):

            x_span_len, z_span_len, story_height = 7000, 7000, 3200
            x_span_lens = [x_span_len for i in range(x_span_num)]
            z_span_lens = [z_span_len for i in range(z_span_num)]

            geo_name = f"x{x_span_num}_y{story_num}_z{z_span_num}"
            #print(geo_name)
            logger.critical(geo_name)

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
            logger.info(f"fail_name: {fail_name}, fail_reason: {fail_reason}")

            if whether_pass is False:
                 fail_geometry.append(geo_name)

logger.critical(f"fail_geometry_num: {len(fail_geometry)}")
logger.critical(f"fail_geometry: \n{fail_geometry}")
