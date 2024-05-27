import os
import numpy as np
from typing import Tuple

from Structure.sections import *


DISP_UX_INDEX = 1
DISP_UZ_INDEX = 3
ELEM_MZI_INDEX = 9
ELEM_MZJ_INDEX = 10
ELEM_MYI_INDEX = 11
ELEM_MYJ_INDEX = 12
ELEM_SYI_INDEX = 13
ELEM_SYJ_INDEX = 14
ELEM_SZI_INDEX = 15
ELEM_SZJ_INDEX = 16
ELEM_AXIAL_INDEX = 18

PISA_EXE = "PISA3D_Batch_500nodes.exe"


class Response:        
    def __init__(self, analysis_name: str, structure):
        disp_file = open(analysis_name + ".NodeAbsDisp", 'r').readlines()
        elem_file = open(analysis_name + ".Element", 'r').readlines()
        self.structure = structure
        self.node_number = structure.node_number
        self.member_number = structure.member_number
        self.node_response = dict()
        self.node_response["dispX"] = dict()
        self.node_response["dispZ"] = dict()
        self.member_response = dict()
        self.member_response["shearY"] = dict()
        self.member_response["shearZ"] = dict()
        self.member_response["momentY"] = dict()
        self.member_response["momentZ"] = dict()
        self.member_response["axial"] = dict()
        self.node_neighbor_Puc_matrix = np.zeros((self.node_number, 6))
        self._load_disp(disp_file)
        self._load_elem(elem_file)

    def __str__(self):
        info = ""
        for node_name in self.node_response["disp"].keys():
            info += f"{node_name:^4s}, {self.node_response['disp'][node_name]:^8.2f}\n"
        return info

    def _load_disp(self, disp_file):
        for line in disp_file:
            line = line.strip()
            if len(line) < 1: continue
            if line[0] == "N" and line[1].isdigit():
                contents = line.split()
                node_name = contents[0]
                UX, UZ = float(contents[DISP_UX_INDEX]), float(contents[DISP_UZ_INDEX])
                #self.node_response["disp"][node_name] = UX if abs(UX) > abs(UZ) else UZ
                self.node_response["dispX"][node_name] = UX
                self.node_response["dispZ"][node_name] = UZ
        assert len(self.node_response["dispX"].keys()) == self.node_number
        assert len(self.node_response["dispZ"].keys()) == self.node_number

    def _load_elem(self, elem_file):
        for line in elem_file:
            line = line.strip()
            if len(line) < 1: continue
            if line[0] == "E" and line[1].isdigit():
                contents = line.split()
                member_name = contents[0]
                MZI, MZJ = float(contents[ELEM_MZI_INDEX]), float(contents[ELEM_MZJ_INDEX])
                MYI, MYJ = float(contents[ELEM_MYI_INDEX]), float(contents[ELEM_MYJ_INDEX])
                SYI, SYJ = float(contents[ELEM_SYI_INDEX]), float(contents[ELEM_SYJ_INDEX])
                SZI, SZJ = float(contents[ELEM_SZI_INDEX]), float(contents[ELEM_SZJ_INDEX])
                AXIAL = float(contents[ELEM_AXIAL_INDEX])
                self.member_response["momentZ"][member_name] = MZI if abs(MZI) > abs(MZJ) else MZJ
                self.member_response["momentY"][member_name] = MYI if abs(MYI) > abs(MYJ) else MYJ
                self.member_response["shearY"][member_name] = SYI if abs(SYI) > abs(SYJ) else SYJ
                self.member_response["shearZ"][member_name] = SZI if abs(SZI) > abs(SZJ) else SZJ
                self.member_response["axial"][member_name] = AXIAL

                node1_index, node2_index, face_number1, face_number2, _, _ = self.structure.member_to_nodeIndex_dict[member_name]
                self.node_neighbor_Puc_matrix[node1_index, face_number1] = AXIAL
                self.node_neighbor_Puc_matrix[node2_index, face_number2] = AXIAL     
                



# return period from modal analysis
def dynamic_analysis_period(structure, analysis_dir) -> Tuple[float, np.array]:
    modal_ipt_path = os.path.join(analysis_dir, "modal.ipt")
    # 1. generate modal analysis ipt file
    _generate_analysis_ipt(structure, modal_ipt_path, analysis="modal")
    # 2. run modal analysis, derive the period
    periods_and_shapes = _run_modal_analysis(structure.node_number, analysis_dir)
    return periods_and_shapes


# return structure's response under the given load case
def run_load_case(structure, load_case, analysis_dir) -> Response:
    # 1. generate load case's ipt file
    load_name = load_case.load_name
    analysis_ipt_path = os.path.join(analysis_dir, f"{load_name}.ipt")
    nodal_loads = load_case.calculate_nodal_load(structure)
    _generate_analysis_ipt(structure, analysis_ipt_path, analysis="static", nodal_loads=nodal_loads)
    # 2. run analysis, derive the response
    response = _run_static_analysis(analysis_dir, analysis_ipt_path, load_name, structure)
    return response




def _generate_analysis_ipt(structure, ipt_path: str, analysis="static", nodal_loads=None) -> None:
    """Generate modal.ipt""" 
    x_grid, y_grid, z_grid = structure.x_grid, structure.y_grid, structure.z_grid
    x_grid_string = '  ' + '  '.join([str(x) for x in x_grid])
    y_grid_string = '  ' + '  '.join([str(y) for y in y_grid])
    z_grid_string = '  ' + '  '.join([str(z) for z in z_grid])

    # nodes
    node_string = ''
    for node_name in structure.node_coord_dict.keys():
        x, y, z = structure.node_coord_dict[node_name]
        node_string = node_string + 'Node  ' + node_name + ' ' + ' '.join([str(x), str(y), str(z)]) + '\n'
    node_string += '\n'

    # master nodes
    master_node_list = []
    for story, y in enumerate(y_grid):
        if story > 0:
            master_name = 'M' + str(story) + 'F'
            master_node_list.append(master_name)
            node_string += 'Node  ' + master_name + ' ' + ' '.join([str(structure.master_x), str(y), str(structure.master_z)]) + '\n'

    # DOF
    dof_string = ''
    for node_name in structure.node_dof_dict.keys():
        if structure.node_dof_dict[node_name] == 1: 
           dof_string += 'DOF  ' + node_name + ' -1 -1 -1 -1 -1 -1' + '\n'
    
    # master node DOF
    for master_node in master_node_list:
        dof_string += 'DOF  ' + master_node + ' 0 -1 0 -1 0 -1' + '\n'
      
    # nodal mass, translational mass (Ux, Uy, Uz, Rx, Ry, Rz) | U: translational mass, R: moment of inertia
    mass_string = ''
    for node_name in structure.node_translational_mass_dict.keys():
        trans_mass = structure.node_translational_mass_dict[node_name]  # kN / mm/s^2
        Rx, Ry, Rz = structure.node_inertia_dict[node_name]             # kN / (mm/s2) * mm2
        mass_string += '#NodeMass  Mass  ' + node_name + ' ' + f"{trans_mass:.5f}" + ' ' + f"{trans_mass:.5f}" + ' ' + f"{trans_mass:.5f}" + ' ' + str(int(Rx)) + ' ' + str(int(Ry)) + ' ' + str(int(Rz)) + '\n'
    
    # master node's translational mass (Ux, Uy, Uz, Rx, Ry, Rz)
    for master_name in master_node_list:
        trans_mass, Ry = 0, 0
        for slave_name in structure.slave_node_dict[master_name]:
            trans_mass += structure.node_translational_mass_dict[slave_name]  # kN / mm/s^2
            Ry += structure.node_inertia_dict[slave_name][1]                  # kN / (mm/s2) * mm2
        mass_string += 'Mass  ' + master_name + ' ' + f"{trans_mass:.7f}" + ' ' + f"{0}" + ' ' + f"{trans_mass:.7f}" + ' ' + str(0) + ' ' + str(int(Ry)) + ' ' + str(0) + '\n'

    # loadings (Fx, Fy, Fz, Mx, My, Mz) | F: external force, M: external moment
    load_string = ''
    for node_name in structure.node_self_weight_dict.keys():
        node_index = int(node_name[1:]) - 1
        if analysis == 'modal':
            load_x = 0
            load_z = 0
            load_y = -1 * structure.node_self_weight_dict[node_name]  # kN
        elif analysis == 'static':
            load_y, load_x, load_z = nodal_loads[0][node_index], nodal_loads[1][node_index], nodal_loads[2][node_index]  # kN
            load_y *= -1
        load_string += 'LoadPattern  NodalLoad  DL ' + node_name + ' ' + f"{load_x:.2f}" + ' ' + f"{load_y:.2f}" + ' ' + f"{load_z:.2f}" + ' 0 0 0' + '\n'
        load_string += 'GUI_LoadPattern  NodalLoad  DL ' + node_name + ' ' + f"{load_x:.2f}" + ' ' + f"{load_y:.2f}" + ' ' + f"{load_z:.2f}" + ' 0 0 0' + '\n'

    # constraint diaphragm
    diaphragm_string = ''
    for diaphragm_index, master_name in enumerate(master_node_list):
        slaves_string = ''
        for slave_name in structure.slave_node_dict[master_name]:
            slaves_string += slave_name + '  '
        diaphragm_string += "Constraint  Diaphragm  " + f"D{diaphragm_index+1}  " + f"{master_name}  " + slaves_string + "\n"

    # beam & column
    beam_column_string = ''
    for member_name in structure.member_section_dict.keys():
        section_index = structure.member_section_dict[member_name]
        section_category = structure.member_category_dict[member_name]
        section_name = column_sections[section_index]['name'] if section_category == 'y' else beam_sections[section_index]['name']
        node1_index, node2_index = structure.member_to_nodeIndex_dict[member_name][:2]
        end1_name, end2_name = 'N'+str(node1_index+1), 'N'+str(node2_index+1)
        beam_column_string += 'Element  BeamColumn   ' + member_name + ' ' + end1_name + ' ' + end2_name + ' ' + section_name + '  0 1 1 0 0 0 0 0' + '\n'        
            
    # start writing ipt file
    f = open(ipt_path, 'w')

    f.write('PISA3D\n')
    f.write(f'{ipt_path}\n')
    f.write('kN\n')
    f.write('mm\n')
    f.write('ControlData    GeometricNL  1\n')

    if analysis == "modal":
        f.write('Analysis  ModeShape  3  1  2  0.02  0.02\n')
    elif analysis == "static":
        f.write('GUI_Analysis  Gravity  gravity  DL  1  ON\n')
        f.write('Analysis  Gravity  DL  1\n')

    f.write('GUI_GRID  XDIR' + x_grid_string + '\n')
    f.write('GUI_GRID  YDIR' + y_grid_string + '\n')
    f.write('GUI_GRID  ZDIR' + z_grid_string + '\n')
    f.write('\n'*2)

    f.write(node_string)
    f.write('\n'*2)

    f.write(dof_string)
    f.write('\n'*2)
    
    f.write(mass_string)
    f.write('\n'*2)

    f.write(load_string)
    f.write('\n'*2)

    f.write(diaphragm_string)
    f.write('\n'*2)

    f.write(beam_column_string)
    f.write('\n'*2)

    f.write('% MATERIAL DATA %\n')
    f.write('Material  Elastic steel 200 0.3\n')
    f.write('\n'*2)

    f.write('% SECTION DATA %\n')

    for section in beam_sections:
        R, G, B = np.array(section['color'])/255
        f.write(f"GUI_Section I_SHAPE_SECTION {section['name']} steel steel steel steel steel steel steel steel steel 0 {section['H(mm)']} {section['B(mm)']} {section['t_f(mm)']} {section['t_w(mm)']} {section['B(mm)']} {section['t_f(mm)']} \n")
        f.write(f"GUI_SECTION_PROPERTY_FACTOR {section['name']} 1 1 1 1 1 1 1 1 \n")
        f.write(f"GUI_SECTION_DISPLAY_COLOR {section['name']} {R}  {G}  {B}  \n")
        f.write(f"Section  BCSection03 {section['name']} steel steel steel steel steel steel steel steel steel 0 {section['A(cm2)']*100} {section['I_z(cm4)']*10000} {section['I_y(cm4)']*10000} {section['J(cm4)']*10000} {section['S_z(cm3)']*1000} {section['S_y(cm3)']*1000} {section['Av_y(cm2)']*100} {section['Av_z(cm2)']*100} \n")
        f.write("\n")                                   
    
    for section in column_sections:
        R, G, B = np.array(section['color'])/255
        f.write(f"GUI_Section BOX_SECTION {section['name']} steel steel steel steel steel steel steel steel steel 0 {section['H(mm)']} {section['B(mm)']} {section['t_f(mm)']} {section['t_w(mm)']} \n")
        f.write(f"GUI_SECTION_PROPERTY_FACTOR {section['name']} 1 1 1 1 1 1 1 1 \n")          
        f.write(f"GUI_SECTION_DISPLAY_COLOR {section['name']} {R}  {G}  {B}  \n")
        f.write(f"Section  BCSection03 {section['name']} steel steel steel steel steel steel steel steel steel 0 {section['A(cm2)']*100} {section['I_z(cm4)']*10000} {section['I_y(cm4)']*10000} {section['J(cm4)']*10000} {section['S_z(cm3)']*1000} {section['S_y(cm3)']*1000} {section['Av_y(cm2)']*100} {section['Av_z(cm2)']*100} \n")
        f.write('\n')

    f.write('\n'*2)

    f.write('GUI_LoadCase  GUI_AREA_LOAD_DL  DL\n')
    f.write('GUI_AREA_LOAD_ASSIGNED_TYPE  BY_BEAN_SPAN_LOAD\n')
    f.write('GUI_Output  OutFlag  1  1  0  1  1  1  1\n')
    f.write('Output  OutFlag  1  1  0  1  1  1  1\n')
    f.write('% RESPONSE HISTORY %\n')
    f.write('STOP\n')

    f.close()        


def _check_modal_analysis(eigen_file_path) -> bool:
    if os.path.exists(eigen_file_path):
        return True
    return False


def _run_modal_analysis(node_number: int, analysis_dir: str) -> float:
    """Run modal analysis with PISA3D"""
    
    modal_path = os.path.join(analysis_dir, 'modal.ipt')
    modal_name = modal_path.replace(".ipt", "")

    # delete the last .Eigen file, ensure pisa really run it
    eigen_file_path = os.path.join(analysis_dir, 'MODAL.Eigen')
    if os.path.exists(eigen_file_path):
        os.remove(eigen_file_path)

    # run and load result (here put them together since I found that eigen file may be empty, maybe the analysis didn't run successfully)
    finished = False
    first_mode_period = None
    second_mode_period = None
    third_mode_period = None
    
    while not finished or first_mode_period is None:
        # print("in pisa while loop")
        os.system(PISA_EXE + " " + modal_name + " " + f">{os.path.join(analysis_dir, 'null')} 2>&1")
        finished = _check_modal_analysis(eigen_file_path)
        if finished == False:
            continue

        is_mode_1, is_mode_2, is_mode_3 = False, False, False
        
        node_first_mode_shape = np.zeros((node_number, 3))
        node_second_mode_shape = np.zeros((node_number, 3))
        node_third_mode_shape = np.zeros((node_number, 3))

        # get first mode shape from modal result
        with open(eigen_file_path, 'r') as f:
            for line in f.readlines():

                # in case the line is not finished, it will occur error, like the number beocome -9.616E
                # and cannot be converted to float.
                try:    
                    
                    if "Period of Mode 1" in line:
                        contents = line.strip().split()
                        first_mode_period = float(contents[5])
                    elif "Period of Mode 2" in line:
                        contents = line.strip().split()
                        second_mode_period = float(contents[5])
                    elif "Period of Mode 3" in line:
                        contents = line.strip().split()
                        third_mode_period = float(contents[5])

                    elif "Mode 1, Period =" in line:
                        is_mode_1 = True
                    elif "----------" in line:
                        is_mode_1 = False
                    elif is_mode_1 == True:
                        if "Node" in line or "N" not in line:   continue
                        contents = line.strip().split()
                        node_index = int(contents[0][1:]) - 1
                        node_first_mode_shape[node_index, 0] = float(contents[1])
                        node_first_mode_shape[node_index, 1] = float(contents[3])
                        node_first_mode_shape[node_index, 2] = float(contents[5])

                    elif "Mode 2, Period =" in line:
                        is_mode_2 = True
                    elif "-----------" in line and is_mode_2:
                        is_mode_2 = False
                    elif is_mode_2 == True:
                        if "Node" in line or "N" not in line: continue
                        contents = line.strip().split()
                        node_index = int(contents[0][1:]) - 1
                        node_second_mode_shape[node_index, 0] = float(contents[1])
                        node_second_mode_shape[node_index, 1] = float(contents[3])
                        node_second_mode_shape[node_index, 2] = float(contents[5])
                    
                    elif "Mode 3, Period =" in line:
                        is_mode_3 = True
                    elif "-----------" in line and is_mode_3:
                        is_mode_3 = False
                    elif is_mode_3 == True:
                        if "Node" in line or "N" not in line: continue
                        contents = line.strip().split()
                        node_index = int(contents[0][1:]) - 1
                        node_third_mode_shape[node_index, 0] = float(contents[1])
                        node_third_mode_shape[node_index, 1] = float(contents[3])
                        node_third_mode_shape[node_index, 2] = float(contents[5])
                
                except:
                    continue

        if first_mode_period is None or second_mode_period is None or third_mode_period is None:
            continue

    return first_mode_period, second_mode_period, third_mode_period, node_first_mode_shape, node_second_mode_shape, node_third_mode_shape




def _check_static_analysis(disp_file_path, elem_file_path) -> bool:
    if os.path.exists(disp_file_path) is True and \
       os.stat(disp_file_path).st_size > 1024 and \
       os.stat(elem_file_path).st_size > 8192:
        return True
    return False


def _run_static_analysis(analysis_dir, ipt_path, load_name, structure) -> Response:    
    """Run static analysis with PISA3D"""
    analysis_name = ipt_path.replace(".ipt", "")

    # delete the last .NodeAbsDisp file, ensure pisa really run it
    disp_file_path = os.path.join(analysis_dir, load_name.upper() + ".NodeAbsDisp")
    elem_file_path = os.path.join(analysis_dir, load_name.upper() + ".Element")
    if os.path.exists(disp_file_path):
        os.remove(disp_file_path)

    # run and load result (here put them together since I found that eigen file may be empty, maybe the analysis didn't run successfully)
    finished = False
    while not finished:
        os.system(PISA_EXE + " " + analysis_name + " " + f">{os.path.join(analysis_dir, f'null_{load_name}')} 2>&1")
        finished = _check_static_analysis(disp_file_path, elem_file_path)
        if finished == False:
            continue

    # get response from the analysis result
    response = Response(analysis_name, structure)
    return response


