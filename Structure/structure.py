import torch
import numpy as np

from copy import deepcopy
from typing import Tuple, List, Dict
from torch_geometric.data import Data

from Structure import pisa
from Structure.sections import *
from Structure.sections import _Fcr, _Mn


# A table which correspond face to node feature's My face index
NODE_FEATURE_FACE_INDEX = {
    'x_n': 23,
    'x_p': 25,
    'y_n': 27,
    'y_p': 29,
    'z_n': 31,
    'z_p': 33,
}

FACE_INDEX = {
    'x_n': 0,
    'x_p': 1,
    'y_n': 2,
    'y_p': 3,
    'z_n': 4,
    'z_p': 5,
}


GRAVITY = 9.81              # m/s2
CONCRETE_DENSITY = 2400     # 2400 kg/m3
STEEL_DENSITY = 8050        # 8050 kg/m3
SLAB_THICKNESS = 0.15       # m

DL = 1.0    # kN/m2


GRAPH_NORM_DICT = {
    # node
    "grid_num": 7,
    "coord": 7,
    # edge
    "L": 8,
    "A": column_sections[-1]['A(cm2)'],     # this is from thickest column
    "Iz": column_sections[-1]['I_z(cm4)'],  # this is from thickest column
    "Iy": column_sections[-1]['I_y(cm4)'],  # this is from thickest column
    "Zz": column_sections[-1]['Z_z(cm3)'],  # this is from thickest column
}




class Structure:
    def __init__(self, 
                 x_span_num: int, x_span_lens: List[int], 
                 z_span_num: int, z_span_lens: List[int], 
                 story_num: int, story_height: float,
                 story_level_sections: List[int]=None,
                 add_structure_geometry=True, 
                 add_response_features=True,
                 do_nonlinear_dynamic_analysis=False, 
                 nda_norm_dict: Dict=None, 
                 analysis_dir: str=None):
        
        self.x_span_num = x_span_num
        self.x_span_lens = x_span_lens
        self.z_span_num = z_span_num
        self.z_span_lens = z_span_lens
        self.story_num = story_num
        self.story_height = story_height
        self.story_height_1F = story_height + 1000
        self.story_level_sections = story_level_sections
        self.add_structure_geometry = add_structure_geometry
        self.add_response_features = add_response_features
        self.do_nonlinear_dynamic_analysis = do_nonlinear_dynamic_analysis
        self.nda_norm_dict = nda_norm_dict
        self.analysis_dir = analysis_dir
        
        self._structure_initialization()
        # self._init_graph()
        if do_nonlinear_dynamic_analysis:
            self._init_nda_graph()

    def __str__(self):
        description = f"Structure: x_span_num: {self.x_span_num}, z_span_num: {self.z_span_num}, story_num: {self.story_num}, "
        description += f"x_span_lens: {self.x_span_lens}, z_span_lens: {self.z_span_lens}, story_height: {self.story_height}"
        return description

    def to(self, device):
        self.graph = self.graph.to(device)
        return self


    def _structure_initialization(self):
        # geometry settings
        x_grid = [sum(self.x_span_lens[:i]) for i in range(self.x_span_num+1)]  
        y_grid = [0, self.story_height_1F]
        y_grid += [self.story_height_1F + num * self.story_height for num in range(1, self.story_num)]
        z_grid = [sum(self.z_span_lens[:i]) for i in range(self.z_span_num+1)] 
        self.x_grid, self.y_grid, self.z_grid = x_grid, y_grid, z_grid
        self.height = y_grid[-1] / 1000     # m
        # print("x_grid:", x_grid)

        # master node settings
        # slave_node_dict --> key: master_name(M1F, M2F, ...), value: [N1, N3, N5, ...]
        self.master_x, self.master_z = int(x_grid[-1]/2), int(z_grid[-1]/2)
        slave_node_dict = dict()

        # node settings (part 1)
        node_number = len(x_grid) * len(y_grid) * len(z_grid)
        self.node_number = node_number

        # coord_to_nodeIndex_dict --> key: coord_str(ex: 6000_3200_6000), value: node_index
        # node_coord_dict --> key: node_name(N1, N2, ...), value: tuple(x, y, z)
        # node_grid_coord_dict --> key: node_name(N1, N2, ...), value: tuple(x_grid_index, y_grid_index, z_grid_index)
        # node_dof_dict --> key: node_name(N1, N2, ...), value: dof(0, 1)
        # bottom_node_index_list --> record bottom node's index. For drift ratio, if y == 0, index = node_index, else index = bottom_node_index
        # bottom_member_length_array --> record the member length under the node
        # node_need_strong_column_weak_beam_list --> [node_index1, node_index2, ...], record whether a node need to satisfy strong column weak beam constraint
        coord_to_nodeIndex_dict = {}
        node_coord_dict = dict()
        node_grid_coord_dict = dict()
        node_dof_dict = dict()
        bottom_node_index_list = []
        bottom_member_length_array = np.zeros((node_number))
        node_need_strong_column_weak_beam_list = []

        hinge_node_yn = []
        hinge_node_yp = []
        story_nodes = {}
        story_mass_center_nodes = {}

        # find the grid index of nodes which locates at or near the center of mass
        mass_center_node_x = [len(x_grid)//2] if len(x_grid) % 2 == 1 else [len(x_grid)//2 - 1, len(x_grid)//2]
        mass_center_node_z = [len(z_grid)//2] if len(z_grid) % 2 == 1 else [len(z_grid)//2 - 1, len(z_grid)//2]
        mass_center_node = []
        for x in mass_center_node_x:
            for z in mass_center_node_z:
                mass_center_node.append((x, z))
    
        node_index = 0
        for story, y in enumerate(y_grid):
            if story > 0:
                master_name = 'M' + str(story) + 'F'
                slave_node_dict[master_name] = []
                story_name = str(story) + 'F'
                story_nodes[story_name] = []
                story_mass_center_nodes[story_name] = []
            for x_index, x in enumerate(x_grid):
                for z_index, z in enumerate(z_grid):
                    # node name
                    node_name = f"N{node_index+1}"
                    # slave node
                    if story > 0:
                        slave_node_dict[master_name].append(node_name)
                    # dof
                    if story == 0:
                        node_dof_dict[node_name] = 1
                    else:
                        node_dof_dict[node_name] = 0
                    # coord
                    node_coord_dict[node_name] = (x, y, z)
                    node_grid_coord_dict[node_name] = (x_index, story, z_index)
                    # coord to node index
                    coord_to_nodeIndex_dict['_'.join([str(x), str(y), str(z)])] = node_index

                    # bottom node index, member length
                    if y != 0:  # if not base node, then record it's bottom node
                        bottom_node_index_list.append(coord_to_nodeIndex_dict['_'.join([str(x), str(y_grid[story-1]), str(z)])])
                        bottom_length = self.story_height_1F if story == 1 else self.story_height
                        bottom_member_length_array[node_index] = bottom_length
                    else:       # if base node
                        bottom_node_index_list.append(node_index)
                        bottom_member_length_array[node_index] = self.story_height  # this value doesn't matter, but set to non-zero for dividend
                    
                    # if need strong column weak beam constraint
                    if y != 0 and y != y_grid[-1]:
                        node_need_strong_column_weak_beam_list.append(node_index)

                    # record which node should check y_n and y_p plastic hinge
                    if y != 0:
                        hinge_node_yn.append(node_index)
                    if y != 0 and y != y_grid[-1]:
                        hinge_node_yp.append(node_index)
                    
                    # node which locates at the story and near center of mass location
                    if story > 0:
                        story_nodes[story_name].append(node_index)
                        if (x_index, z_index) in mass_center_node:
                            story_mass_center_nodes[story_name].append(node_index)

                    node_index += 1

        self.slave_node_dict = slave_node_dict
        self.node_coord_dict = node_coord_dict
        self.node_grid_coord_dict = node_grid_coord_dict
        self.node_dof_dict = node_dof_dict
        self.bottom_node_index_list = bottom_node_index_list
        self.bottom_member_length_array = bottom_member_length_array
        self.node_need_strong_column_weak_beam_list = node_need_strong_column_weak_beam_list
        self.hinge_node_yn = hinge_node_yn
        self.hinge_node_yp = hinge_node_yp
        self.story_nodes = story_nodes
        self.story_mass_center_nodes = story_mass_center_nodes

        # member
        member_number = self.story_num * (3 * self.x_span_num * self.z_span_num + 2 * self.x_span_num + 2 * self.z_span_num + 1)
        self.member_number = member_number

        # member_to_nodeIndex_dict --> key: member_name, value: [node1_index, node2_index, face1_number, face2_number, My_face1_index, My_face2_index], whrere face_index is the index in the node feature
        # member_section_dict --> key: member_name, value: section's index. Initial: 8, where 8 is the thickest among beam sections and column sections (each 9 sections).
        # member_category_dict --> key: member_name, value: x, y, or z. Record whether member is x_beam, z_beam or y_column.
        # member_XZ_dict --> key: member_name, value: xz
        # member_same_location_dict --> key: x_z, value: [member_name...]
        # member_Fcr_dict --> key: member_name, key: Fcr
        # member_A_dict --> key: member_name, key: Area
        # member_length_dict --> key: member_name, key: beam column length
        # story_column_dict --> key: story_name(1F, 2F, ...), value: [column_index1, column_index2, ...]
        # member_column_index_list --> record member index if it's column
        # member_beam_index_list --> record member index if it's beam
        # node_neighbor_Zz_matrix --> [node_number, 6], where 6 is for xn(0), xp(1), yn(2), yp(3), zn(4), zp(5)
        # node_neighbor_Ag_matrix --> [node_number, 6], where 6 is for xn(0), xp(1), yn(2), yp(3), zn(4), zp(5)
        # node_neighbor_member_dict --> key: node_name, value: [member_name1, member_name2, ....]
        member_to_nodeIndex_dict = {}
        member_section_dict = {}
        member_category_dict = {}
        member_XZ_dict = {}
        member_same_location_dict = {}
        member_Fcr_dict = {}
        member_A_dict = {}
        member_length_dict = {}
        member_Mnx_dict = {}
        member_Mny_dict = {}
        story_column_dict = {}
        member_column_index_list = []
        member_beam_index_list = []
        node_neighbor_Zz_matrix = torch.zeros((node_number, 6))
        node_neighbor_Ag_matrix = torch.zeros((node_number, 6))
        node_neighbor_member_dict = {}  
        for i in range(node_number):
            node_neighbor_member_dict[f"N{i+1}"] = []
        story_beam_member = []
        story_xdir_beam_member = []
        story_zdir_beam_member = []
        story_column_member = []
        story_inner_column_member = []
        story_outer_column_member = []

        # prepare story level member group sections for member_section_dict
        if self.story_level_sections is not None:
            story_xdir_beam_section, story_zdir_beam_section, story_outer_column_section, story_inner_column_section = np.array_split(self.story_level_sections, 4)
        
        member_index = 0
        for i, y in enumerate(y_grid):
            # for columns:
            if y != y_grid[-1]:
                story_column = []
                story_inner_column = []
                story_outer_column = []
                y_upper = y_grid[i+1]
                for x in x_grid:
                    for z in z_grid:
                        member_name = f"E{member_index+1}"
                        story_name = f"{i+1}F"

                        story_column.append(member_index)
                        if x == 0 or x == x_grid[-1] or z == 0 or z == z_grid[-1]:
                            story_outer_column.append(member_index)
                        else:
                            story_inner_column.append(member_index)

                        if story_name not in story_column_dict.keys():
                            story_column_dict[story_name] = []
                        story_column_dict[story_name].append(member_index)
                        coord1 = "_".join([str(x), str(y), str(z)])
                        coord2 = "_".join([str(x), str(y_upper), str(z)])
                        node1_index = coord_to_nodeIndex_dict[coord1]
                        node2_index = coord_to_nodeIndex_dict[coord2]
                        member_to_nodeIndex_dict[member_name] = [node1_index, node2_index, FACE_INDEX['y_p'], FACE_INDEX['y_n'], NODE_FEATURE_FACE_INDEX['y_p'], NODE_FEATURE_FACE_INDEX['y_n']]
                        member_column_index_list.append(member_index)

                        # assign member with same location (vertically)
                        column_XZ = "_".join([str(x), str(z)])
                        member_XZ_dict[member_name] = column_XZ
                        if column_XZ not in member_same_location_dict.keys():
                            member_same_location_dict[column_XZ] = []
                        member_same_location_dict[column_XZ].append(member_name)
            
                        if self.story_level_sections is None:
                            init_section = len(column_sections) - 1
                        else:
                            if x == 0 or x == x_grid[-1] or z == 0 or z == z_grid[-1]:
                                init_section = story_outer_column_section[y_grid.index(y)]
                            else:
                                init_section = story_inner_column_section[y_grid.index(y)]
                        member_section_dict[member_name] = int(init_section)
                        member_category_dict[member_name] = 'y'

                        # Fcr
                        I = column_sections[init_section]["I_z(cm4)"]
                        A = column_sections[init_section]["A(cm2)"]
                        L = (y_upper - y) / 10  # cm
                        member_Fcr_dict[member_name] = _Fcr(I, A, L)
                        member_A_dict[member_name] = A * 1e+2  # mm2
                        member_length_dict[member_name] = L / 100  # m

                        # Mnx, Mny
                        member_Mnx_dict[member_name] = None
                        member_Mny_dict[member_name] = None

                        member_index += 1
                
                story_column_member.append(story_column)
                story_outer_column_member.append(story_outer_column)
                story_inner_column_member.append(story_inner_column)

            
            # for x beams and z beams:
            if y != 0:
                story_beam = []
                story_xdir_beam = []
                story_zdir_beam = []
                for z in z_grid:
                    for j in range(len(x_grid)-1):
                        member_name = f"E{member_index+1}"
                        story_beam.append(member_index)
                        story_xdir_beam.append(member_index)

                        x = x_grid[j]
                        x_next = x_grid[j+1]
                        # x_next = x + self.x_span_len
                        coord1 = "_".join([str(x), str(y), str(z)])
                        coord2 = "_".join([str(x_next), str(y), str(z)])
                        node1_index = coord_to_nodeIndex_dict[coord1]
                        node2_index = coord_to_nodeIndex_dict[coord2]
                        member_to_nodeIndex_dict[member_name] = [node1_index, node2_index, FACE_INDEX['x_p'], FACE_INDEX['x_n'], NODE_FEATURE_FACE_INDEX['x_p'], NODE_FEATURE_FACE_INDEX['x_n']]
                        member_beam_index_list.append(member_index)
                        
                        # assign member with same location (vertically)
                        beam_XZ = "_".join([str((x + x_next) / 2), str(z)])
                        member_XZ_dict[member_name] = beam_XZ
                        if beam_XZ not in member_same_location_dict.keys():
                            member_same_location_dict[beam_XZ] = []
                        member_same_location_dict[beam_XZ].append(member_name)
                        
                        if self.story_level_sections is None:
                            init_section = len(beam_sections) - 1
                        else:
                            init_section = story_xdir_beam_section[y_grid.index(y)-1]
                        member_section_dict[member_name] = int(init_section)
                        member_category_dict[member_name] = 'x'

                        # Fcr
                        I = beam_sections[init_section]["I_z(cm4)"]
                        A = beam_sections[init_section]["A(cm2)"]
                        L = (x_next - x) / 10  # cm
                        member_Fcr_dict[member_name] = _Fcr(I, A, L)
                        member_A_dict[member_name] = A * 1e+2  # mm2
                        member_length_dict[member_name] = L / 100  # m

                        # Mnx, Mny
                        Mnx, Mny = _Mn(init_section, L/100)
                        member_Mnx_dict[member_name] = Mnx
                        member_Mny_dict[member_name] = Mny

                        member_index += 1
                
                
                for x in x_grid:
                    for j in range(len(z_grid)-1):
                        member_name = f"E{member_index+1}"
                        story_beam.append(member_index)
                        story_zdir_beam.append(member_index)

                        z = z_grid[j]
                        z_next = z_grid[j+1]
                        coord1 = "_".join([str(x), str(y), str(z)])
                        coord2 = "_".join([str(x), str(y), str(z_next)])
                        node1_index = coord_to_nodeIndex_dict[coord1]
                        node2_index = coord_to_nodeIndex_dict[coord2]
                        member_to_nodeIndex_dict[member_name] = [node1_index, node2_index, FACE_INDEX['z_p'], FACE_INDEX['z_n'], NODE_FEATURE_FACE_INDEX['z_p'], NODE_FEATURE_FACE_INDEX['z_n']]
                        member_beam_index_list.append(member_index)

                        # assign member with same location (vertically)
                        beam_XZ = "_".join([str(x), str((z + z_next) / 2)])
                        member_XZ_dict[member_name] = beam_XZ
                        if beam_XZ not in member_same_location_dict.keys():
                            member_same_location_dict[beam_XZ] = []
                        member_same_location_dict[beam_XZ].append(member_name)
                        
                        if self.story_level_sections is None:
                            init_section = len(beam_sections) - 1
                        else:
                            init_section = story_zdir_beam_section[y_grid.index(y)-1]
                        member_section_dict[member_name] = int(init_section)
                        member_category_dict[member_name] = 'z'

                        # Fcr
                        I = beam_sections[init_section]["I_z(cm4)"]
                        A = beam_sections[init_section]["A(cm2)"]
                        L = (z_next - z) / 10  # cm
                        member_Fcr_dict[member_name] = _Fcr(I, A, L)
                        member_A_dict[member_name] = A * 1e+2  # mm2
                        member_length_dict[member_name] = L / 100  # m

                        # Mnx, Mny
                        Mnx, Mny = _Mn(init_section, L/100)
                        member_Mnx_dict[member_name] = Mnx
                        member_Mny_dict[member_name] = Mny

                        member_index += 1

                story_beam_member.append(story_beam)
                story_xdir_beam_member.append(story_xdir_beam)
                story_zdir_beam_member.append(story_zdir_beam)

        
        self.member_XZ_dict = member_XZ_dict
        self.member_same_location_dict = member_same_location_dict
        self.member_section_dict = member_section_dict
        self.member_category_dict = member_category_dict
        self.story_column_dict = story_column_dict
        self.member_to_nodeIndex_dict = member_to_nodeIndex_dict

        self.member_Fcr_dict = member_Fcr_dict
        self.member_A_dict = member_A_dict
        self.member_length_dict = member_length_dict
        self.member_Mnx_dict = member_Mnx_dict
        self.member_Mny_dict = member_Mny_dict

        self.member_column_index_list = member_column_index_list
        self.member_beam_index_list = member_beam_index_list
        self.already_minimum_section_story_indexes = [] if self.story_level_sections is None else [i for i in range(len(self.story_level_sections)) if self.story_level_sections[i] == 0]

        self.story_beam_member = story_beam_member
        self.story_xdir_beam_member = story_xdir_beam_member
        self.story_zdir_beam_member = story_zdir_beam_member
        self.story_column_member = story_column_member
        self.story_outer_column_member = story_outer_column_member
        self.story_inner_column_member = story_inner_column_member

        if self.story_level_sections is None:
            self.story_xdir_beam_section = [len(beam_sections) - 1 for _ in range(len(self.story_xdir_beam_member))]
            self.story_zdir_beam_section = [len(beam_sections) - 1 for _ in range(len(self.story_zdir_beam_member))]
            self.story_outer_column_section = [len(column_sections) - 1 for _ in range(len(self.story_outer_column_member))]
            self.story_inner_column_section = [len(column_sections) - 1 for _ in range(len(self.story_inner_column_member))]
            self.story_level_sections = self.story_xdir_beam_section + self.story_zdir_beam_section + self.story_outer_column_section + self.story_inner_column_section
        else:
            xdir_beam, zdir_beam, outer_column, inner_column = np.array_split(self.story_level_sections, 4)
            self.story_xdir_beam_section = xdir_beam.tolist()
            self.story_zdir_beam_section = zdir_beam.tolist()
            self.story_outer_column_section = outer_column.tolist()
            self.story_inner_column_section = inner_column.tolist()

        self.story_level_actions = self.story_xdir_beam_member + self.story_zdir_beam_member + self.story_outer_column_member + self.story_inner_column_member
        self.story_level_categories = ['xdir_beam' for _ in range(len(self.story_xdir_beam_member))] + ['zdir_beam' for _ in range(len(self.story_zdir_beam_member))] + ['outer_column' for _ in range(len(self.story_outer_column_member))] + ['inner_column' for _ in range(len(self.story_inner_column_member))]
        self.full_section_sum = sum(self.story_level_sections)

        # update member feature into node feature and the node neighbor Mp list, and embedding node
        for member_index in range(0, member_number):
            member_name = f"E{member_index+1}"
            section_index = member_section_dict[member_name]
            category = member_category_dict[member_name]
            if category == 'y':
                member_Zz = column_sections[section_index]['Z_z(cm3)'] * 1e+3  # mm3
                member_Ag = column_sections[section_index]['A(cm2)'] * 1e+2    # mm2
            else:
                member_Zz = beam_sections[section_index]['Z_z(cm3)'] * 1e+3    # mm3
                member_Ag = beam_sections[section_index]['A(cm2)'] * 1e+2      # mm2
            
            node1_index, node2_index, face_number1, face_number2, _, _ = member_to_nodeIndex_dict[member_name]
            node_neighbor_Zz_matrix[node1_index, face_number1] = member_Zz      
            node_neighbor_Zz_matrix[node2_index, face_number2] = member_Zz 
            node_neighbor_Ag_matrix[node1_index, face_number1] = member_Ag      
            node_neighbor_Ag_matrix[node2_index, face_number2] = member_Ag 
            node_neighbor_member_dict[f"N{node1_index+1}"].append(member_name)   
            node_neighbor_member_dict[f"N{node2_index+1}"].append(member_name) 
        
        self.node_neighbor_Zz_matrix = node_neighbor_Zz_matrix
        self.node_neighbor_Ag_matrix = node_neighbor_Ag_matrix
        self.node_neighbor_member_dict = node_neighbor_member_dict

        # node settings (part 2)
        # node_area_dict --> key: node_name(N1, N2, ...), value: area distributed to node
        # node_self_weight_dict --> key: node_name(N1, N2, ...), value: mass distributed to node
        # node_dead_load_self_weight_dict --> key: node_name, value: nodal self weight + dead load
        # node_translational_mass_dict --> key: node_name, value: translational mass
        # node_inertia_dict --> key: node_name(N1, N2, ...), value: inertia distributed to node (Rx, Ry, Rz)
        # story_weight_distribution_ratio_dict --> key: story_name(1F, 2F, ...), value: story self weight * height from ground / total weight
        node_area_dict = dict()                   # m^2
        node_self_weight_dict = dict()            # kN
        node_dead_load_self_weight_dict = dict()  # kN
        node_translational_mass_dict = dict()     # kN / (mm/s^2)
        node_inertia_dict = dict()                # kN / (mm/s^2) * mm^2
        story_weight_distribution_ratio_dict = dict()

        node_index = 0
        for story, y in enumerate(y_grid):
            story_weight = 0
            for x in x_grid:
                for z in z_grid:
                    # node name
                    node_name = f"N{node_index+1}"
                    # node area: m^2
                    distributed_area = self._calculate_node_distributed_area(x, y, z)
                    node_area_dict[node_name] = distributed_area
                    # node mass: kN
                    node_self_weight = self._calculate_node_distributed_mass(node_name, distributed_area)
                    node_self_weight_dict[node_name] = node_self_weight
                    story_weight += node_self_weight
                    # self weight + dead load: kN
                    node_dead_load_self_weight_dict[node_name] = node_self_weight + distributed_area * DL
                    # translational mass: kN / (mm/s^2)
                    node_translational_mass_dict[node_name] = self._calculate_translational_mass(node_self_weight)
                    # node inertia: kN / (mm/s^2) * mm^2
                    node_inertia_dict[node_name] = self._calculate_moment_inertia(x, y, z)
                    node_index += 1

                story_weight_distribution_ratio_dict[f"{story}F"] = story_weight * y
        
        total_weight_height_multiplication = sum(story_weight_distribution_ratio_dict.values())
        for story_name in story_weight_distribution_ratio_dict.keys():
            story_weight_distribution_ratio_dict[story_name] /= total_weight_height_multiplication

        self.node_area_dict = node_area_dict
        self.node_self_weight_dict = node_self_weight_dict
        self.node_dead_load_self_weight_dict = node_dead_load_self_weight_dict
        self.node_translational_mass_dict = node_translational_mass_dict
        self.node_inertia_dict = node_inertia_dict

        # calculate the ratio each node recieves the earthquake force
        node_Exn_ratio_dict = dict()
        node_Exp_ratio_dict = dict()
        node_Ezn_ratio_dict = dict()
        node_Ezp_ratio_dict = dict()
        node_Ey_ratio_dict = dict()
        node_index = 0
        for story, y in enumerate(y_grid):
            story_name = f"{story}F"
            story_weight_ratio = story_weight_distribution_ratio_dict[story_name]
            for x in x_grid:
                for z in z_grid:
                    # node name
                    node_name = f"N{node_index+1}"
                    # Exp
                    if x == 0:
                        node_Exp_ratio_dict[node_name] = story_weight_ratio / len(z_grid)
                    else:
                        node_Exp_ratio_dict[node_name] = 0
                    # Exn
                    if x == x_grid[-1]:
                        node_Exn_ratio_dict[node_name] = story_weight_ratio / len(z_grid)
                    else:
                        node_Exn_ratio_dict[node_name] = 0
                    # Ezp
                    if z == 0:
                        node_Ezp_ratio_dict[node_name] = story_weight_ratio / len(x_grid)
                    else:
                        node_Ezp_ratio_dict[node_name] = 0
                    # Ezn
                    if z == z_grid[-1]:
                        node_Ezn_ratio_dict[node_name] = story_weight_ratio / len(x_grid)
                    else:
                        node_Ezn_ratio_dict[node_name] = 0
                    # Ey
                    node_Ey_ratio_dict[node_name] = 1 / node_number
                    node_index += 1
                    
        self.node_Exn_ratio_dict = node_Exn_ratio_dict
        self.node_Exp_ratio_dict = node_Exp_ratio_dict
        self.node_Ezn_ratio_dict = node_Ezn_ratio_dict
        self.node_Ezp_ratio_dict = node_Ezp_ratio_dict
        self.node_Ey_ratio_dict = node_Ey_ratio_dict


    def _calculate_node_distributed_area(self, x, y, z) -> float:
        # given node x, y, z, return the area distrubte to node
        # node distributed area depends on only neighboring slab, so doesn't need to update when reducing sections
        if y == 0: return 0

        x_grid_coord = self.x_grid.index(x)
        if x_grid_coord == 0:
            width_x = (self.x_grid[1] - self.x_grid[0]) / 2
        elif x_grid_coord == len(self.x_grid) - 1:
            width_x = (self.x_grid[-1] - self.x_grid[-2]) / 2
        else:
            width_x = (self.x_grid[x_grid_coord+1] - self.x_grid[x_grid_coord-1]) / 2
        
        z_grid_coord = self.z_grid.index(z)
        if z_grid_coord == 0:
            width_z = (self.z_grid[1] - self.z_grid[0]) / 2
        elif z_grid_coord == len(self.z_grid) - 1:
            width_z = (self.z_grid[-1] - self.z_grid[-2]) / 2
        else:
            width_z = (self.z_grid[z_grid_coord+1] - self.z_grid[z_grid_coord-1]) / 2

        # width_x = self.x_grid[1]/2 if x == min(self.x_grid) or x == max(self.x_grid) else self.x_grid[1]
        # width_z = self.z_grid[1]/2 if z == min(self.z_grid) or z == max(self.z_grid) else self.z_grid[1]
        area = width_x/1000 * width_z/1000  # m2

        return area


    def _calculate_node_distributed_mass(self, node_name, distributed_area) -> float:
        # 1. slab weight
        slab_weight = distributed_area * SLAB_THICKNESS * CONCRETE_DENSITY * GRAVITY / 1000  # kN

        # 2. beam-column weight
        beam_column_weight = 0
        for member_name in self.node_neighbor_member_dict[node_name]:
            member_length = self.member_length_dict[member_name]
            member_category = self.member_category_dict[member_name]
            section_index = self.member_section_dict[member_name]
            section_area = column_sections[section_index]['A(cm2)'] if member_category == 'y' else beam_sections[section_index]['A(cm2)']
            member_weight = section_area/100/100 * member_length/2 * STEEL_DENSITY * GRAVITY / 1000  # kN
            beam_column_weight += member_weight

        node_distributed_weight = slab_weight + beam_column_weight
        # print(f"node: {node_name}, slab_weight: {slab_weight}, beam_column_weight: {beam_column_weight}")
        return node_distributed_weight

    
    def _calculate_translational_mass(self, self_weight) -> float:
        """How much force (in kN) is needed to achieve an acceleration of 1 mm/s^2"""
        kN = self_weight
        kg = kN * 1000 / GRAVITY
        translational_mass = kg / 1e+06
        return translational_mass


    def _calculate_moment_inertia(self, x, y, z) -> Tuple[float]:
        # Inertia is mostly contributed by slab, so here only calculate slab's inertia. 
        # Since inertia is only contributed by slab, inertia values don't need updation during reducing sections
        # first find the surrounding slabs, define quarter_slab as 1/4 slab, 1/2 x_span_len * 1/2 z_span_len
        quarter_slab_number = 0
        if y == 0:
            quarter_slab_number += 0
        elif x == min(self.x_grid):
            if z == min(self.z_grid):
                quarter_slab_number += 1
            elif z == max(self.z_grid):
                quarter_slab_number += 1
            else:
                quarter_slab_number += 2

        elif x == max(self.x_grid):
            if z == min(self.z_grid):
                quarter_slab_number += 1
            elif z == max(self.z_grid):
                quarter_slab_number += 1
            else:
                quarter_slab_number += 2
                
        else:
            if z == min(self.z_grid):
                quarter_slab_number += 2
            elif z == max(self.z_grid):
                quarter_slab_number += 2
            else:
                quarter_slab_number += 4
        
        # calculate inertia for all slabs
        # For simplicity, use the average span len instead of the variant individual span lengths
        x_span_len = sum(self.x_span_lens) / len(self.x_span_lens)
        z_span_len = sum(self.z_span_lens) / len(self.z_span_lens)
        slab_mass = x_span_len/1000 * z_span_len/1000 * SLAB_THICKNESS * CONCRETE_DENSITY   # kg

        # 1 kN = 1e+6 kg * 1 mm/s2 --> 1 kg = 1e-6 kN / (mm/s2)
        # E.g., mass = 300 kg, it will be represented as 3e-4 (kN / (mm/s2)) in PISA
        slab_translational_mass = slab_mass / 1e+06  # kN / (mm/s2)
            
        slab_global_Ix = 1 / 12 * slab_translational_mass * ((SLAB_THICKNESS*1000) ** 2 + z_span_len ** 2) 
        slab_global_Iy = 1 / 12 * slab_translational_mass * (x_span_len ** 2 + z_span_len ** 2)
        slab_global_Iz = 1 / 12 * slab_translational_mass * ((SLAB_THICKNESS*1000) ** 2 + x_span_len ** 2)
        
        Rx = (quarter_slab_number / 4) * slab_global_Ix
        Ry = (quarter_slab_number / 4) * slab_global_Iy
        Rz = (quarter_slab_number / 4) * slab_global_Iz

        return Rx, Ry, Rz  # unit: kN / (mm/s2) * mm2
    

    def calculate_material_usage(self):
        """Calculate the total volume of members as material usage"""
        material_usage = 0.0
        for member_index in range(0, self.member_number):
            member_name = f"E{member_index+1}"
            section_index = self.member_section_dict[member_name]
            member_category = self.member_category_dict[member_name]
            
            section_area = column_sections[section_index]['A(cm2)'] if member_category == 'y' else beam_sections[section_index]['A(cm2)']
            member_length = self.member_length_dict[member_name]
           
            member_volume = section_area / 10000 * member_length  # m3
            material_usage += member_volume

        return material_usage




    def _strongColumn_weakBeam_beta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Set ratio beta to embedding node v_hat, and then check if all beta is greater than 1."""
        Mpc = self.node_neighbor_Zz_matrix.float() @ torch.tensor([0, 0, 1, 1, 0, 0]).unsqueeze(1).float()  # [node_number, 6] @ [6, 1] = [node_number, 1]
        Mpb_x = self.node_neighbor_Zz_matrix @ torch.tensor([1, 1, 0, 0, 0, 0]).unsqueeze(1).float()        # [node_number, 6] @ [6, 1] = [node_number, 1]
        Mpb_z = self.node_neighbor_Zz_matrix @ torch.tensor([0, 0, 0, 0, 1, 1]).unsqueeze(1).float()        # [node_number, 6] @ [6, 1] = [node_number, 1]
        beta_x = (Mpc / (Mpb_x + 1e-6)).squeeze()  # [node_number]
        beta_z = (Mpc / (Mpb_z + 1e-6)).squeeze()  # [node_number]
        beta_x = torch.tanh(1.0 / beta_x)
        beta_z = torch.tanh(1.0 / beta_z)
        return beta_x, beta_z


    def _init_graph(self, response_features: dict[str, torch.Tensor] = None):
        # new node feature: if fix, if top, if side, beta_x, beta_z
        node_feature_num = 8 if self.add_structure_geometry else 5
        node_feature = torch.zeros(self.node_number, node_feature_num)

        if self.add_response_features:
            beta_x = torch.tanh(1.0 / response_features["min_SCWB_ratio_x"]) 
            beta_z = torch.tanh(1.0 / response_features["min_SCWB_ratio_z"])
        else: 
            beta_x, beta_z = self._strongColumn_weakBeam_beta()
        node_feature[:, 3] = beta_x
        node_feature[:, 4] = beta_z

        node_index = 0
        for y in self.y_grid:
            for x in self.x_grid:
                for z in self.z_grid:
                    # node name
                    node_name = f"N{node_index+1}"

                    # if_fixed, if_top, if_side
                    if y == 0:
                        node_feature[node_index, 0] = 1
                    if y == self.y_grid[-1]:
                        node_feature[node_index, 1] = 1
                    if x == 0 or x == self.x_grid[-1] or z == 0 or z == self.z_grid[-1]:
                        node_feature[node_index, 2] = 1

                    if self.add_structure_geometry:
                        x_grid_coord, y_grid_coord, z_grid_coord = self.node_grid_coord_dict[node_name]
                        node_feature[node_index, 5] = self.story_num  # story number
                        node_feature[node_index, 6] = y_grid_coord  # current story
                        node_feature[node_index, 7] = y_grid_coord / self.story_num  # current height ratio

                    node_index += 1

                            
        # edge feature:     is_col, is_beam, L, (A, Iz, Iy, Zz), (A', Iz', Iy', Zz')
        # new edge feature: is_col, is_beam, L, (A, Iz, Iy, Zz), (A', Iz', Iy', Zz'), tanh(stress_ratio), tanh(drift_ratio)
        edge_feature_num = 13 if self.add_response_features else 11
        edge_feature = torch.zeros(self.member_number * 2, edge_feature_num)

        for original_member_index in range(self.member_number):
            member_name = f"E{original_member_index+1}"
            member_index = original_member_index * 2
            # is_column, is_beam
            if self.member_category_dict[member_name] == 'y':
                edge_feature[member_index, 0] = 1
            else:
                edge_feature[member_index, 1] = 1

            # length
            edge_feature[member_index, 2] = self.member_length_dict[member_name]
            # A, Iz, Iy, Zz
            member_category = self.member_category_dict[member_name]
            section_index = self.member_section_dict[member_name]
            section_info = column_sections[section_index] if member_category == 'y' else beam_sections[section_index]
            A = section_info["A(cm2)"]
            Iz = section_info["I_z(cm4)"]
            Iy = section_info["I_y(cm4)"]
            Zz = section_info["Z_z(cm3)"]
            edge_feature[member_index, 3] = A
            edge_feature[member_index, 4] = Iz
            edge_feature[member_index, 5] = Iy
            edge_feature[member_index, 6] = Zz
            # A', Iz', Iy', Zz'
            edge_feature[member_index, 7] = A
            edge_feature[member_index, 8] = Iz
            edge_feature[member_index, 9] = Iy
            edge_feature[member_index, 10] = Zz

            if self.add_response_features:
                edge_feature[member_index, 11] = torch.tanh(response_features["max_stress_ratio"][original_member_index])
                edge_feature[member_index, 12] = torch.tanh(response_features["max_drift_ratio"][original_member_index])

            # add another direction of edge feature back
            edge_feature[member_index+1, :] = edge_feature[member_index, :]

        # edge_index [2, edge_num(member_num * 2)]
        edge_i, edge_j = [], []
        for member_index in range(self.member_number):
            member_name = f"E{member_index+1}"
            node1_index, node2_index = self.member_to_nodeIndex_dict[member_name][:2]
            edge_i.append(node1_index)
            edge_j.append(node2_index)
            edge_i.append(node2_index)
            edge_j.append(node1_index)
        
        #story_beam_member = [torch.tensor(member) for member in self.story_beam_member]
        story_xdir_beam_member = [torch.tensor(member) for member in self.story_xdir_beam_member]
        story_zdir_beam_member = [torch.tensor(member) for member in self.story_zdir_beam_member]
        story_outer_column_member = [torch.tensor(member) for member in self.story_outer_column_member]
        story_inner_column_member = [torch.tensor(member) for member in self.story_inner_column_member]

        # story_batch
        story_batch = torch.zeros(self.member_number)
        story_count = 0
        for story_members in (story_xdir_beam_member + story_zdir_beam_member + story_outer_column_member + story_inner_column_member):
            story_batch[story_members] = story_count
            story_count += 1

        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)

        aux = {
            "story_xdir_beam_member": story_xdir_beam_member,
            "story_zdir_beam_member": story_zdir_beam_member,
            "story_outer_column_member": story_outer_column_member,
            "story_inner_column_member": story_inner_column_member,
            "story_batch": story_batch.to(torch.int64)
        }
        self.aux = aux

        self.graph = Data(x=node_feature, y=None, edge_index=edge_index, edge_attr=edge_feature)
                        #   story_beam_member=tuple(story_beam_member), 
                        #   story_outer_column_member=tuple(story_outer_column_member), 
                        #   story_inner_column_member=tuple(story_inner_column_member),
                        #   story_batch=story_batch.to(torch.int64))
        self._normalize()
        print("graph:", self.graph)

        
    def _init_nda_graph(self):

        # data.x: XYZ grid nums(3), node_grid(3), if_bottom(1), if_top(1), if_side(1), beta(2), period(3), mode_shape(3*3), section_info_per_face(2*6)
        node_feature_num = 35
        node_feature = torch.zeros(self.node_number, node_feature_num)

        # grid nums
        node_feature[:, 0] = len(self.x_grid)
        node_feature[:, 1] = len(self.y_grid)
        node_feature[:, 2] = len(self.z_grid)

        # beta_x, beta_z
        beta_x, beta_z = self._strongColumn_weakBeam_beta()
        node_feature[:, 9] = beta_x
        node_feature[:, 10] = beta_z

        # period, mode shape
        first_mode_period, second_mode_period, third_mode_period, node_first_mode_shape, node_second_mode_shape, node_third_mode_shape = pisa.dynamic_analysis_period(self, self.analysis_dir)
        node_feature[:, 11] = first_mode_period
        node_feature[:, 12] = second_mode_period
        node_feature[:, 13] = third_mode_period
        node_feature[:, 14:17] = torch.tensor(node_first_mode_shape)   # (Ux, Uz, Ry)
        node_feature[:, 17:20] = torch.tensor(node_second_mode_shape)  # (Ux, Uz, Ry)
        node_feature[:, 20:23] = torch.tensor(node_third_mode_shape)   # (Ux, Uz, Ry)

        # geometry
        node_index = 0
        for y in self.y_grid:
            for x in self.x_grid:
                for z in self.z_grid:
                    # node name
                    node_name = f"N{node_index+1}"

                    # coord
                    x_grid_coord, y_grid_coord, z_grid_coord = self.node_grid_coord_dict[node_name]
                    node_feature[node_index, 3] = x_grid_coord
                    node_feature[node_index, 4] = y_grid_coord
                    node_feature[node_index, 5] = z_grid_coord

                    # if_fixed, if_top, if_side
                    if y == 0:
                        node_feature[node_index, 6] = 1
                    if y == self.y_grid[-1]:
                        node_feature[node_index, 7] = 1
                    if x == 0 or x == self.x_grid[-1] or z == 0 or z == self.z_grid[-1]:
                        node_feature[node_index, 8] = 1
                        
                    node_index += 1

        # section information
        for member_index in range(self.member_number):
            member_name = f"E{member_index+1}"

            L = self.member_length_dict[member_name]
            member_category = self.member_category_dict[member_name]
            section_index = self.member_section_dict[member_name]
            if member_category == 'y':
                My = column_sections[section_index]['My_z(kN-mm)']
            else:
                My = beam_sections[section_index]['My_z(kN-mm)']

            node1_index, node2_index, _, _, node_feature_face_index1, node_feature_face_index2 = self.member_to_nodeIndex_dict[member_name]
            node_feature[node1_index, node_feature_face_index1] = L
            node_feature[node1_index, node_feature_face_index1+1] = My
            node_feature[node2_index, node_feature_face_index2] = L
            node_feature[node2_index, node_feature_face_index2+1] = My


        # edge_attr: member_length(1), is_beam(1), is_column(1), My(1)
        edge_feature = torch.zeros(self.member_number * 2, 4)

        for original_member_index in range(self.member_number):
            member_name = f"E{original_member_index+1}"
            member_index = original_member_index * 2

            # length
            edge_feature[member_index, 0] = self.member_length_dict[member_name]

            # is_column, is_beam, My
            section_index = self.member_section_dict[member_name]
            if self.member_category_dict[member_name] == 'y':
                edge_feature[member_index, 1] = 0
                edge_feature[member_index, 2] = 1
                edge_feature[member_index, 3] = column_sections[section_index]['My_z(kN-mm)']
            else:
                edge_feature[member_index, 1] = 1
                edge_feature[member_index, 2] = 0
                edge_feature[member_index, 3] = beam_sections[section_index]['My_z(kN-mm)']

            # add another direction of edge feature back
            edge_feature[member_index+1, :] = edge_feature[member_index, :]

        # edge_index [2, edge_num(member_num * 2)]
        edge_i, edge_j = [], []
        for member_index in range(self.member_number):
            member_name = f"E{member_index+1}"
            node1_index, node2_index = self.member_to_nodeIndex_dict[member_name][:2]
            edge_i.append(node1_index)
            edge_j.append(node2_index)
            edge_i.append(node2_index)
            edge_j.append(node1_index)

        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)

        self.nda_graph = Data(x=node_feature, y=None, edge_index=edge_index, edge_attr=edge_feature)
        self._nda_normalize()
        print("nda_graph", self.nda_graph)


    def _normalize(self):
        if self.add_structure_geometry:
            self.graph.x[:, 5] /= GRAPH_NORM_DICT["grid_num"]  # 7
            self.graph.x[:, 6] /= GRAPH_NORM_DICT["coord"]  # 7
        self.graph.edge_attr[:, 2] /= GRAPH_NORM_DICT["L"]  # 8
        self.graph.edge_attr[:, 3] /= GRAPH_NORM_DICT["A"]
        self.graph.edge_attr[:, 4] /= GRAPH_NORM_DICT["Iz"]
        self.graph.edge_attr[:, 5] /= GRAPH_NORM_DICT["Iy"]
        self.graph.edge_attr[:, 6] /= GRAPH_NORM_DICT["Zz"]
        self.graph.edge_attr[:, 7] /= GRAPH_NORM_DICT["A"]
        self.graph.edge_attr[:, 8] /= GRAPH_NORM_DICT["Iz"]
        self.graph.edge_attr[:, 9] /= GRAPH_NORM_DICT["Iy"]
        self.graph.edge_attr[:, 10] /= GRAPH_NORM_DICT["Zz"]

    
    def _nda_normalize(self):
        self.nda_graph.x[:, :3] /= self.nda_norm_dict["grid_num"]
        self.nda_graph.x[:, 3:6] /= self.nda_norm_dict["coord"]
        self.nda_graph.x[:, 11:14] /= self.nda_norm_dict["period"]
        self.nda_graph.x[:, 14:23] /= self.nda_norm_dict["modal_shape"]

        self.nda_graph.x[:, list(range(23, 35, 2))] /= self.nda_norm_dict["elem_length"]
        self.nda_graph.x[:, list(range(24, 35, 2))] /= self.nda_norm_dict["moment"]
        
        self.nda_graph.edge_attr[:, 0] /= self.nda_norm_dict["elem_length"]
        self.nda_graph.edge_attr[:, 3] /= self.nda_norm_dict["moment"]




    def restrict_action_space(self) -> List[int]:
        """Don't choose the member that change section will cause upper members are thicker than lower members"""
        dont_select_story_member_indexes = []
        
        # start_story_xdir_beam_index = 0
        # start_story_zdir_beam_index = start_story_xdir_beam_index + len(self.start_story_xdir_beam_index)
        # start_story_outer_column_index = start_story_zdir_beam_index + len(self.start_story_zdir_beam_index)
        # start_story_inner_column_index = start_story_outer_column_index + len(self.story_outer_column_section)

        start_story_member_index = 0
        for story_member_section in [self.story_xdir_beam_section, self.story_zdir_beam_section, self.story_outer_column_section, self.story_inner_column_section]:
            for story in range(len(story_member_section)-1):
                lower_story_member_section = story_member_section[story]
                upper_story_member_section = story_member_section[story+1]
                
                if lower_story_member_section - upper_story_member_section <= 0:
                    story_member_index = start_story_member_index + story
                    dont_select_story_member_indexes.append(story_member_index)
                    
            start_story_member_index += len(story_member_section)

        return dont_select_story_member_indexes            


    def update_action(self, update_story_index: int) -> float:

        if self.story_level_sections[update_story_index] == 0:
            print("update_story_index:", update_story_index)
            print("current story sections:", self.story_level_sections)
            print("aleady_minimun_section_story_indexes:", self.already_minimum_section_story_indexes)

        # update_story_members is the members in the story_xdir_beam, story_zdir_beam, story_inner_col, or story_outer_col
        update_story_members = self.story_level_actions[update_story_index]
        update_story_category = self.story_level_categories[update_story_index]  # ["xdir_beam", "zdir_beam", "outer_column", "inner_column"]
        update_member_names = [f"E{update_member_index+1}" for update_member_index in update_story_members]

        new_section_index = self.story_level_sections[update_story_index] - 1
        assert new_section_index >= 0

        new_section_info = column_sections[new_section_index] if update_story_category in ["outer_column", "inner_column"] else beam_sections[new_section_index]
        
        self.story_level_sections[update_story_index] -= 1
        if self.story_level_sections[update_story_index] == 0:
            self.already_minimum_section_story_indexes.append(update_story_index)

        # auto-correct if there are upper stories thicker than the current story
        correct_needed_members = update_member_names.copy()
        if update_story_category == "xdir_beam":
            end_story_index = len(self.story_xdir_beam_member)
        elif update_story_category == "zdir_beam":
            end_story_index = len(self.story_xdir_beam_member) + len(self.story_zdir_beam_member)
        elif update_story_category == "outer_column":
            end_story_index = len(self.story_xdir_beam_member) + len(self.story_zdir_beam_member) + len(self.story_outer_column_member)
        elif update_story_category == "inner_column":
            end_story_index = len(self.story_level_actions)  # xdir_beam + zdir_beam + outer_column + inner_column
        
        for story_action in range(update_story_index, end_story_index):
            if self.story_level_sections[story_action] > new_section_index:
                story_members = self.story_level_actions[story_action]
                member_names = [f"E{member_index+1}" for member_index in story_members]
                correct_needed_members += member_names
            
                # update story section and record if already minimum
                self.story_level_sections[story_action] -= 1
                if self.story_level_sections[story_action] == 0:
                    self.already_minimum_section_story_indexes.append(story_action)
        
        # update story member section from story_level_sections
        xdir_beam, zdir_beam, outer_column, inner_column = np.array_split(self.story_level_sections, 4)
        self.story_xdir_beam_section = xdir_beam.tolist()
        self.story_zdir_beam_section = zdir_beam.tolist()
        self.story_outer_column_section = outer_column.tolist()
        self.story_inner_column_section = inner_column.tolist()
        #print(f"Before & After auto-correct are same: {update_member_names == correct_needed_members}")
        
        # update structure and graph's variables
        # calculate saved volume
        volume_saved = 0
        I = new_section_info["I_z(cm4)"]
        A = new_section_info["A(cm2)"]
        Iz = new_section_info["I_z(cm4)"]
        Iy = new_section_info["I_y(cm4)"]
        Zz = new_section_info['Z_z(cm3)']
        My = new_section_info["My_z(kN-mm)"]
        for member_name in correct_needed_members:
            assert self.member_section_dict[member_name] > 0
            # volume saved
            old_section_index = self.member_section_dict[member_name]
            old_A = column_sections[old_section_index]["A(cm2)"] if update_story_category in ["outer_column", "inner_column"] else beam_sections[old_section_index]["A(cm2)"]
            area_difference = old_A - A
            assert area_difference >= 0
            volume_saved += area_difference / 1e+4 * self.member_length_dict[member_name]   # m3

            # member level update (edge_attr have number of twice member_number, need to update both)
            original_member_index = int(member_name[1:]) - 1
            member_index = original_member_index * 2
            
            self.member_section_dict[member_name] -= 1
            L = self.member_length_dict[member_name] * 100  # cm
            self.member_Fcr_dict[member_name] = _Fcr(I, A, L)
            self.member_A_dict[member_name] = A * 1e+2  # mm2

            if update_story_category == "beam":
                Mnx, Mny = _Mn(self.member_section_dict[member_name], L/100)
                self.member_Mnx_dict[member_name] = Mnx
                self.member_Mny_dict[member_name] = Mny
            
            # node level update
            node1_index, node2_index, face_number1, face_number2, node_feature_face_index1, node_feature_face_index2 = self.member_to_nodeIndex_dict[member_name]
            self.node_neighbor_Zz_matrix[node1_index, face_number1] = Zz * 1e+3  # mm3
            self.node_neighbor_Zz_matrix[node2_index, face_number2] = Zz * 1e+3  # mm3
            self.node_neighbor_Ag_matrix[node1_index, face_number1] = A * 1e+2  # mm2
            self.node_neighbor_Ag_matrix[node2_index, face_number2] = A * 1e+2  # mm2
            
            node1_name, node2_name = f"N{node1_index+1}", f"N{node2_index+1}"
            distributed_area1 = self.node_area_dict[node1_name]
            distributed_area2 = self.node_area_dict[node2_name]
            node1_self_weight = self._calculate_node_distributed_mass(node1_name, distributed_area1)
            node2_self_weight = self._calculate_node_distributed_mass(node2_name, distributed_area2)
            self.node_self_weight_dict[node1_name] = node1_self_weight
            self.node_self_weight_dict[node2_name] = node2_self_weight
            self.node_dead_load_self_weight_dict[node1_name] = node1_self_weight + distributed_area1 * DL
            self.node_dead_load_self_weight_dict[node2_name] = node2_self_weight + distributed_area2 * DL
            self.node_translational_mass_dict[node1_name] = self._calculate_translational_mass(node1_self_weight)
            self.node_translational_mass_dict[node2_name] = self._calculate_translational_mass(node2_self_weight)
            
            # graph update
            # edge feature
            # self.graph.edge_attr[member_index, 3:7] = self.graph.edge_attr[member_index, 7:11]
            # self.graph.edge_attr[member_index, 7] = A / GRAPH_NORM_DICT["A"]
            # self.graph.edge_attr[member_index, 8] = Iz / GRAPH_NORM_DICT["Iz"]
            # self.graph.edge_attr[member_index, 9] = Iy / GRAPH_NORM_DICT["Iy"]
            # self.graph.edge_attr[member_index, 10] = Zz / GRAPH_NORM_DICT["Zz"]
            # self.graph.edge_attr[member_index+1, :] = self.graph.edge_attr[member_index, :]

            # nda_graph update
            # if self.do_nonlinear_dynamic_analysis:
                # node's member feature
                # self.nda_graph.x[node1_index, node_feature_face_index1+1] = My / self.nda_norm_dict["moment"]
                # self.nda_graph.x[node2_index, node_feature_face_index2+1] = My / self.nda_norm_dict["moment"]
                # edge feature
                # self.nda_graph.edge_attr[member_index, 3] = My / self.nda_norm_dict["moment"]
                # self.nda_graph.edge_attr[member_index+1, :] = self.nda_graph.edge_attr[member_index, :]
            

        # graph beta, modal period, shape update
        # beta_x, beta_z = self._strongColumn_weakBeam_beta()
        # self.graph.x[:, 3] = beta_x
        # self.graph.x[:, 4] = beta_z

        # also update nda_graph
        # if self.do_nonlinear_dynamic_analysis:
        #     self.nda_graph.x[:, 9] = beta_x
        #     self.nda_graph.x[:, 10] = beta_z
        #     first_mode_period, second_mode_period, third_mode_period, node_first_mode_shape, node_second_mode_shape, node_third_mode_shape = pisa.dynamic_analysis_period(self, self.analysis_dir)
        #     self.nda_graph.x[:, 11] = first_mode_period / self.nda_norm_dict["period"]
        #     self.nda_graph.x[:, 12] = second_mode_period / self.nda_norm_dict["period"]
        #     self.nda_graph.x[:, 13] = third_mode_period / self.nda_norm_dict["period"]
        #     self.nda_graph.x[:, 14:17] = torch.tensor(node_first_mode_shape) / self.nda_norm_dict["modal_shape"]
        #     self.nda_graph.x[:, 17:20] = torch.tensor(node_second_mode_shape) / self.nda_norm_dict["modal_shape"]
        #     self.nda_graph.x[:, 20:23] = torch.tensor(node_third_mode_shape) / self.nda_norm_dict["modal_shape"]
        
        return volume_saved
        

    def update_garph(self, response_features: dict[str, torch.Tensor] = None):
        # update node features
        if self.add_response_features: 
            beta_x = torch.tanh(1.0 / response_features["min_SCWB_ratio_x"])
            beta_z = torch.tanh(1.0 / response_features["min_SCWB_ratio_z"])
        else: 
            beta_x, beta_z = self._strongColumn_weakBeam_beta()
        self.graph.x[:, 3] = beta_x
        self.graph.x[:, 4] = beta_z

        # update edge features
        for member_index in range(self.member_number):
            self.graph.edge_attr[member_index*2, 3:7] = self.graph.edge_attr[member_index*2, 7:11]

            member_name = f"E{member_index+1}"
            section_index = self.member_section_dict[member_name]  # structure is updated, so section_index is updated
            section_info = column_sections[section_index] if self.member_category_dict[member_name] == 'y' else beam_sections[section_index]
            A = section_info["A(cm2)"]
            Iz = section_info["I_z(cm4)"]
            Iy = section_info["I_y(cm4)"]
            Zz = section_info["Z_z(cm3)"]
            self.graph.edge_attr[member_index*2, 7] = A / GRAPH_NORM_DICT["A"]
            self.graph.edge_attr[member_index*2, 8] = Iz / GRAPH_NORM_DICT["Iz"]
            self.graph.edge_attr[member_index*2, 9] = Iy / GRAPH_NORM_DICT["Iy"]
            self.graph.edge_attr[member_index*2, 10] = Zz / GRAPH_NORM_DICT["Zz"]
            if self.add_response_features:
                self.graph.edge_attr[member_index*2, 11] = torch.tanh(response_features["max_stress_ratio"][member_index])
                self.graph.edge_attr[member_index*2, 12] = torch.tanh(response_features["max_drift_ratio"][member_index])

            self.graph.edge_attr[member_index*2+1, :] = self.graph.edge_attr[member_index*2, :]


    def update_nda_grpah(self, response_features: dict[str, torch.Tensor] = None):
        # update node features
        beta_x, beta_z = self._strongColumn_weakBeam_beta()
        self.nda_graph.x[:, 9] = beta_x
        self.nda_graph.x[:, 10] = beta_z
        first_mode_period, second_mode_period, third_mode_period, node_first_mode_shape, node_second_mode_shape, node_third_mode_shape = pisa.dynamic_analysis_period(self, self.analysis_dir)
        self.nda_graph.x[:, 11] = first_mode_period / self.nda_norm_dict["period"]
        self.nda_graph.x[:, 12] = second_mode_period / self.nda_norm_dict["period"]
        self.nda_graph.x[:, 13] = third_mode_period / self.nda_norm_dict["period"]
        self.nda_graph.x[:, 14:17] = torch.tensor(node_first_mode_shape) / self.nda_norm_dict["modal_shape"]
        self.nda_graph.x[:, 17:20] = torch.tensor(node_second_mode_shape) / self.nda_norm_dict["modal_shape"]
        self.nda_graph.x[:, 20:23] = torch.tensor(node_third_mode_shape) / self.nda_norm_dict["modal_shape"]

        # update edge features
        for member_index in range(self.member_number):
            member_name = f"E{member_index+1}"
            node1_index, node2_index, face_number1, face_number2, node_feature_face_index1, node_feature_face_index2 = self.member_to_nodeIndex_dict[member_name]
            
            section_index = self.member_section_dict[member_name]  # structure is updated, so section_index is updated
            section_info = column_sections[section_index] if self.member_category_dict[member_name] == 'y' else beam_sections[section_index]
            My = section_info["My_z(kN-mm)"]

            # node's member feature
            self.nda_graph.x[node1_index, node_feature_face_index1+1] = My / self.nda_norm_dict["moment"]
            self.nda_graph.x[node2_index, node_feature_face_index2+1] = My / self.nda_norm_dict["moment"]
            # edge feature
            self.nda_graph.edge_attr[member_index*2, 3] = My / self.nda_norm_dict["moment"]
            self.nda_graph.edge_attr[member_index*2+1, :] = self.nda_graph.edge_attr[member_index*2, :]




