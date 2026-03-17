import json
from pathlib import Path
from Structure import pisa, check
from Structure.structure import Structure


ckpt_dir = Path("./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2025_06_05__21_45_28__TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000")
chances = 0
inference_record_path = ckpt_dir / f"inferencing_record_{chances}chance.txt"
with open(inference_record_path, 'r') as f:
    inference_record = json.load(f)

x_span_num = 6  # 2 - 6
z_span_num = 6  # 2 - 6
story_num = 7   # 4 - 7
for index, geo_info in enumerate(inference_record["geometry"]):
    if geo_info[:3] == [x_span_num, z_span_num, story_num]:
        x_span_lens = geo_info[3]
        z_span_lens = geo_info[4]
        story_height = geo_info[5]
        break

geo_name = f"x{x_span_num}z{z_span_num}y{story_num}"
final_design = inference_record["final_design"][index]
structure_kwargs = {"x_span_num": x_span_num, "x_span_lens": x_span_lens, 
                    "z_span_num": z_span_num, "z_span_lens": z_span_lens, 
                    "story_num": story_num, "story_height": story_height,
                    "story_level_sections": None, 
                    "add_structure_geometry": False,
                    "add_response_features": False,
                    "do_nonlinear_dynamic_analysis": False, 
                    "nda_norm_dict": None,
                    "analysis_dir": ckpt_dir/"Modal_Analysis"}
structure = Structure(**structure_kwargs)
print(structure)
print(structure.calculate_material_usage(), inference_record["initial_volume"][index])
print(structure.story_level_sections, inference_record["initial_design"][index])

reward = 0
for action in inference_record["action"][index]:
    material_saved = structure.update_action(action)
    reward += material_saved
print(reward, inference_record["saved_material"][index])
print(structure.calculate_material_usage(), inference_record["final_volume"][index])
print(structure.story_level_sections, final_design)
assert structure.story_level_sections == final_design, "The final design does not match the inference record."

load_cases, static_responses = check.get_response(structure, ckpt_dir/"Code_Analysis")    
static_constraint_condition, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)
whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, check_displacement=True)

save_ipt_path = ckpt_dir / f"final_design_{geo_name}_{chances}chance.ipt"
pisa._generate_analysis_ipt(structure, save_ipt_path, analysis="modal")
