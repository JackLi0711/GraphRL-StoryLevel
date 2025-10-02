"""
The script is to compare the two final designs' seismic behavior under nonlinear dynamic analysis.
The time-history analysis done when desiging is predicted by AI,
so we would like to test their real response using PISA.
Displacement and plastic hinge behavior should be compared.
"""
import os
import time
import shutil
import threading
import subprocess

import sys
sys.path.append("Validation")

from pathlib import Path

from Validation import make_file
from Validation import pisa3d_finished_check
from Validation import generate_structural_graph


thread_quota           = 4
gm_level               = "World_processed_one_scaling_MCE"  # DBE, MCE, MCEx2, World_processed_one_scaling_MCE, TAP3_two_scaling_MCE
strucutre_type         = "x6z6y7"  # testing, taller, random, x2z2y4, x6z6y7
chance                 = 0          # 0, 1, 2

static_model_folder    = "2025_06_05__21_45_28__TaiModifiedModel_MatReward_StaResFeatures_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000"
dynamic_model_folder   = ""
model_date             = [static_model_folder.split("__")[0], static_model_folder.split("__")[1]]
model_setting          = "__".join(model_date)

working_dir            = f"./Validation/Final_Design_Comparison/{model_setting}/{strucutre_type}/{gm_level}"
static_checkpoint_dir  = f"./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/{static_model_folder}"
dynamic_checkpoint_dir = f"./Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/{dynamic_model_folder}"
pisa                   = "PISA3D_Batch_500nodes.exe"

ground_motion_num      = 11
ground_motion_root     = f"NonlinearDynamicAnalysisSimulator/ground_motions/selected_ground_motions_{gm_level}"




def check_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    '''
    print("Start checking path......")
    if os.path.exists(path):
        print('This dir is already exist.')
    else:
        os.mkdir(path)
    '''


def generate_seismic_ipt(working_dir, checkpoint_dir, scenario="static"):
    working_dir = Path(working_dir)
    checkpoint_dir = Path(checkpoint_dir)
    
    # first copy the ipt files from checkpoints to destination working directory
    ipt_path = checkpoint_dir / f"final_design_{strucutre_type}_{chance}chance.ipt"
    target_ipt_path = working_dir / f"{scenario}_design_{strucutre_type}_{chance}chance.ipt"
    shutil.copy(ipt_path, target_ipt_path)
    
    # create directory for the upcoming analysis
    target_analysis_dir = working_dir / scenario
    target_analysis_dir.mkdir(parents=True, exist_ok=True)    
    for gm_name_scale in os.listdir(ground_motion_root)[:ground_motion_num]:
        print("generating ipt file for:", target_analysis_dir, gm_name_scale)
        gm_name = gm_name_scale.split("_")[0]
        gm_analysis_dir = target_analysis_dir / gm_name
        gm_analysis_dir.mkdir(parents=True, exist_ok=True)
        gm_analysis_ipt = gm_analysis_dir / "structure.ipt"
        shutil.copy(target_ipt_path, gm_analysis_ipt)
        
        # modify the content of the ipt file
        structure_ipt_string = ""
        with open(gm_analysis_ipt, 'r') as f:
            structure_ipt_string = f.read()

        original_analysis = "Analysis  ModeShape  3  1  2  0.02  0.02"
        dynamic_analysis = "# Analysis  Dynamic  Newmark  XGndMot  1  none  0  ZGndMot  1  0.005  14000  alpha  beta  0"
        load_pattern1 = fr"# LoadPattern  GroundAccel  XGndMot  D:\GraphRL_StoryLevel_GitHub\{ground_motion_root}\{gm_name_scale}\{gm_name}_FN.txt  1  1 "
        load_pattern2 = fr"# LoadPattern  GroundAccel  ZGndMot  D:\GraphRL_StoryLevel_GitHub\{ground_motion_root}\{gm_name_scale}\{gm_name}_FP.txt  1  1 "
        new_analysis = f"{original_analysis}\n{dynamic_analysis}\n{load_pattern1}\n{load_pattern2}"

        structure_ipt_string = structure_ipt_string.replace(original_analysis, new_analysis)
        structure_ipt_string = structure_ipt_string.replace("Material  Elastic steel 200 0.3", "Material  Bilinear steel 200 0.00 0.35 -0.35 0.3")
        structure_ipt_string = structure_ipt_string.replace("GUI_Output  OutFlag  1  1  0  1  1  1  1", "GUI_Output  OutFlag  10  10  0  2  2  10  10")
        structure_ipt_string = structure_ipt_string.replace("Output  OutFlag  1  1  0  1  1  1  1", "Output  OutFlag  10  10  0  2  2  10  10")

        with open(gm_analysis_ipt, 'w') as f:
            f.write(structure_ipt_string)
        
        with open(gm_analysis_dir / "modal.ipt", 'w') as f:
            f.write(structure_ipt_string)


def run_pisa_all(target_dir, analysis="structure"):
    print("Start running pisa analysis......")
    target_dir = os.path.join(working_dir, target_dir)
    len_argv = len(sys.argv)
    semaphore = threading.Semaphore(thread_quota)
    case_counter_unfinished = 1
    
    while (case_counter_unfinished != 0):
        print('\n'*2)
        case_counter_unfinished = 0
        case_list = os.listdir(target_dir)
        case_list.sort()
        target_case_list = []
        t_pool           = []
        for case in case_list:
            case_dir = os.path.join(target_dir, case) + "\\"
            if analysis == "modal":
                state = pisa3d_finished_check.check_finish_eigen(case_dir)
            elif analysis == "structure":
                state = pisa3d_finished_check.check_finish(case_dir)
            else:
                raise Exception("Please either use modal analysis or dynamic analysis!")
            
            if state != "finished":
                case_counter_unfinished += 1
                target_case_list.append(case)

        if (case_counter_unfinished != 0):
            for case in target_case_list:
                case_dir = os.path.join(target_dir, case) + "\\"
                ipt      = case_dir + analysis
                t_pool.append(threading.Thread(target=run_single_analysis, args=(semaphore, case_dir, ipt, )))
            for t in t_pool:
                t.start()
            for t in t_pool:
                t.join()

    for case in os.listdir(target_dir):
        case_dir = os.path.join(target_dir, case) + "\\"
        make_sure_clean_again(case_dir)

 
def run_single_analysis(semaphore, case_dir, ipt):
    semaphore.acquire()
    os.system("@ECHO OFF")
    os.system(pisa + " " + ipt)

    delete_useless_files_in_dir(case_dir)

    try:    # In case it occur errors
        if("modal" in ipt):
            make_file.clean_Eigen(case_dir)
        elif("structure" in ipt):
            make_file.clean_ElemRecord(case_dir)
            make_file.clean_NodeAccRecord(case_dir)
            make_file.clean_NodeVelRecord(case_dir)
            make_file.clean_NodeDisRecord(case_dir)
    except:
        print("There are no such response files yet!!!")

    semaphore.release()


def make_sure_clean_again(case_dir):
    print(f"Checking again case {case_dir}......")
    if "STRUCTURE.VISA3D" in os.listdir(case_dir):
        print(f"Fuond uncleaned folder, now clean again folder {case_dir}......")
        delete_useless_files_in_dir(case_dir)
        make_file.clean_ElemRecord(case_dir)
        make_file.clean_NodeAccRecord(case_dir)
        make_file.clean_NodeVelRecord(case_dir)
        make_file.clean_NodeDisRecord(case_dir)


keeped_extensions = ['ipt', 'Eigen', 'Modal', 'ElemRecord', 'NodeDisRecord', 'NodeAccRecord', 'NodeVelRecord', 'pt', 'txt']
def delete_useless_files_in_dir(case_dir):
    print("Start deleting useless files......")
    for analysis_file in os.listdir(case_dir):
        file_name = os.path.join(case_dir, analysis_file)
        extension = analysis_file.split('.')[1]
        if extension not in keeped_extensions:
            os.remove(file_name)




if __name__ == '__main__':

    # 1. Check if the path exists, if not, create folder
    check_path(working_dir)
    
    # 2. Generate ipt files for each pair of ground motion
    generate_seismic_ipt(working_dir, static_checkpoint_dir, scenario="static")
    # generate_seismic_ipt(working_dir, dynamic_checkpoint_dir, scenario="dynamic")
    
    # 3. Get alpha, beta
    run_pisa_all(target_dir="static", analysis="modal")
    make_file.set_Rayleigh_coeff(root=working_dir, target_dir="static")
    # run_pisa_all(target_dir="dynamic", analysis="modal")
    # make_file.set_Rayleigh_coeff(root=working_dir, target_dir="dynamic")

    # 4. Run dynamic analysis
    run_pisa_all(target_dir="static", analysis="structure")
    # run_pisa_all(target_dir="dynamic", analysis="structure")

    # 5. Generate graph
    generate_structural_graph.generate_graph_NodeAsNode(os.path.join(working_dir, "static"))
    # generate_structural_graph.generate_graph_NodeAsNode(os.path.join(working_dir, "dynamic"))
    
    
