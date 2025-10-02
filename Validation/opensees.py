import time
import torch
import numpy as np
import openseespy.opensees as ops
from pathlib import Path

import sys
sys.path.append("../")
from Structure.sections import beam_sections, column_sections
from Structure.structure import NODE_FEATURE_FACE_INDEX

mm = 1  # length
kN = 1  # force
m = 1e+3 * mm
N = 1e-3 * kN

Pa = N / m**2
GPa = 1e+9 * Pa
E = 200 * GPa  # Young's modulud, GPa
Nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + Nu))  # Shear modulus, GPa


def _generate_analysis_model(x_grid, y_grid, z_grid, 
                             node_coord_dict, 
                             node_dof_dict, 
                             node_translational_mass_dict, 
                             node_inertia_dict, 
                             slave_node_dict, 
                             member_to_nodeIndex_dict, 
                             member_category_dict, 
                             member_section_dict, 
                             member_A_dict) -> None:
    """Generate analysis model in **OpenSees**"""
    # 7.25. wipe command (https://openseespydoc.readthedocs.io/en/latest/src/wipe.html)
    ops.wipe()  # remove existing model

    # 4.1. model command (https://openseespydoc.readthedocs.io/en/latest/src/model.html)
    ops.model("basic", "-ndm",3, "-ndf",6)

    # 4.3. node command (https://openseespydoc.readthedocs.io/en/latest/src/node.html)
    for node_name, (x, y, z) in node_coord_dict.items():
        nodeTag = int(node_name[1:])
        crds = [x, z, y]
        # node_translational_mass_dict --> key: node_name(N1, N2, ...), value: translational mass, unit: kN / (mm/s^2)
        trans_mass = node_translational_mass_dict[node_name]
        # node_inertia_dict --> key: node_name(N1, N2, ...), value: inertia distributed to node (Rx, Ry, Rz), unit: kN / (mm/s^2) * mm^2
        Rx, Ry, Rz = node_inertia_dict[node_name]
        mass = [trans_mass, trans_mass, 0] + [0, 0, Ry]
        ops.node(nodeTag, *crds)  # , '-mass',*mass
    
    # 4.4. fix command (https://openseespydoc.readthedocs.io/en/latest/src/fix.html)
    # node_dof_dict --> key: node_name(N1, N2, ...), value: dof(free:0, fixed:1)
    for node_name in node_dof_dict.keys():
        nodeTag = int(node_name[1:])
        ops.fix(nodeTag, *[1,1,1, 1,1,1])

    node_masses = np.array([ops.nodeMass(nodeTag) for nodeTag in ops.getNodeTags()])
    total_mass = np.sum(node_masses, axis=0)
    # print(f"total mass of all DOFs: {total_mass}")
    
    # 4.18. geomTransf commands (https://openseespydoc.readthedocs.io/en/latest/src/geomTransf.html)
    # 14.7.3. 3D beam (https://openseespydoc.readthedocs.io/en/stable/src/ops_vis_ex_3d_3el_cantilever.html)
    ColumnTransfTag = 1    # Kyle: PDelta [1,0,0], Example 14.7.3: Linear [0, -1, 0]
    XdirBeamTransfTag = 2  # Kyle: Linear [0,0,1], Example 14.7.3: Linear [0, -1, 0]
    ZdirBeamTransfTag = 3  # Kyle: Linear [0,0,1], Example 14.7.3: Linear [1, 0, 0]
    ops.geomTransf('PDelta', ColumnTransfTag, *[1,0,0])
    ops.geomTransf('Linear', XdirBeamTransfTag, *[0,1,0])
    ops.geomTransf('Linear', ZdirBeamTransfTag, *[1,0,0])

    # 4.2. element command (https://openseespydoc.readthedocs.io/en/latest/src/element.html)
    # member_to_nodeIndex_dict --> key: member_name, value: [node1_index, node2_index, face1_number, face2_number, My_face1_index, My_face2_index], whrere face_index is the index in the node feature
    for member_name, member_info in member_to_nodeIndex_dict.items():
        eleTag = int(member_name[1:])
        node1_index, node2_index = member_info[:2]
        eleNodes = [node1_index+1, node2_index+1]  # nodeTag == node_index + 1 == int(node_name[1:])
        Area = member_A_dict[member_name]  # mm^2

        member_sections = column_sections if member_category_dict[member_name] == 'y' else beam_sections
        sections_index = member_section_dict[member_name]
        section = member_sections[sections_index]
        Jxx = section["J(cm4)"] * 1e+4  # torsional constant, mm^4
        Iy = section["I_y(cm4)"] * 1e+4  # moment of inertia, mm^4
        Iz = section["I_z(cm4)"] * 1e+4  # moment of inertia, mm^4
        Avy = section["Av_y(cm2)"] * 1e+2  # shear area, mm^2
        Avz = section["Av_z(cm2)"] * 1e+2  # shear area, mm^2

        if member_category_dict[member_name] == 'x':
            transfTag = XdirBeamTransfTag
        elif member_category_dict[member_name] == 'z':
            transfTag = ZdirBeamTransfTag
        else:
            transfTag = ColumnTransfTag

        # 4.2.3.1. Elastic Beam Column Element
        # ops.element('elasticBeamColumn', eleTag, *eleNodes, Area, E, G, Jxx, Iy, Iz, transfTag)
        # 4.2.3.3. Elastic Timoshenko Beam Column Element --> accounts for shear deformations
        ops.element('ElasticTimoshenkoBeam', eleTag, *eleNodes, E, G, Area, Jxx, Iy, Iz, Avy, Avz, transfTag)

    # master node
    node_number = len(node_coord_dict)
    master_x = x_grid[0] + (x_grid[-1] - x_grid[0]) / 2
    master_z = z_grid[0] + (z_grid[-1] - z_grid[0]) / 2
    total_trans_mass, total_moment_inertia = 0, 0
    for story, y in enumerate(y_grid):
        if story > 0:
            master_name = f"M{story}F"            
            trans_mass, moment_inertia = 0, 0
            for slave_name in slave_node_dict[master_name]:
                trans_mass += node_translational_mass_dict[slave_name]  # kN / (mm/s^2)
                Rx, Ry, Rz = node_inertia_dict[slave_name]  # kN / (mm/s^2) * mm^2
                moment_inertia += Ry
            rNodeTag = story + node_number
            crds = [master_x, master_z, y]
            mass = [trans_mass, trans_mass, 0] + [0, 0, moment_inertia]
            # print(f"Master Node {rNodeTag}: {crds}, {mass}")
            ops.node(rNodeTag, *crds, '-mass',*mass)
            ops.fix(rNodeTag, *[0,0,1, 1,1,0])

            total_trans_mass += trans_mass
            total_moment_inertia += moment_inertia

            # 4.5.1. equalDOF command (https://openseespydoc.readthedocs.io/en/latest/src/equalDOF.html)
            # for slave_name in structure.slave_node_dict[master_name]:
            #     slaveTag = int(slave_name[1:])
            #     dofs = [1,2,3,4,5,6]
            #     ops.equalDOF(nodeTag, slaveTag, *dofs)
            
            # 4.5.3. rigidDiaphragm command (https://openseespydoc.readthedocs.io/en/latest/src/rigidDiaphragm.html)
            perpDirn = 3  # 3 (Z) corresponds to the 1-2 (X-Y) plane
            cNodeTags = [int(slave_name[1:]) for slave_name in slave_node_dict[master_name]]
            ops.rigidDiaphragm(perpDirn, rNodeTag, *cNodeTags)

    # print(f"Total Mass, Translational: {total_trans_mass}, Rotational: {total_moment_inertia}\n")

    # omega1, omega2, omega3 = omegas
    # a = np.array([
    #     [1/omega1, omega1],
    #     [1/omega2, omega2]
    # ])
    # zeta1, zeta2 = 0.02, 0.02  # damping ratio
    # b = np.array([2*zeta1, 2*zeta2])
    # alphaM, betaK = np.linalg.solve(a, b)
    # print(f"alphaM: {alphaM:.4f}, betaK: {betaK:.4f}\n")

    # # 4.11. rayleigh command (https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html)
    # ops.rayleigh(alphaM, betaK, 0, 0)  # set damping matrix (for desired damping ratio)


def _run_single_modal_analysis(node_number) -> tuple[np.ndarray, np.ndarray]:
    """
    * Run **OpenSees** modal analysis
    * Return **mode periods** and **mode shapes**
    """
    # 5.8. eigen command (https://openseespydoc.readthedocs.io/en/latest/src/eigen.html#id0)
    numEigenvalues = 3
    eigenValues = ops.eigen('-fullGenLapack', numEigenvalues)  # use '-fullGenLapack' (non-default solver) to avoid error
    omegas = np.sqrt(eigenValues)
    periods = 2 * np.pi / omegas
    
    # 5.10. modalProperties Command (https://openseespydoc.readthedocs.io/en/latest/src/modalProperties.html#modalProperties)
    # reportFileName = str(analysis_dir / "modal_property.txt")
    # ops.modalProperties('-file', reportFileName)

    node_first_mode_shape = np.zeros((node_number, 3))  # Ux, Uy, Rz
    node_second_mode_shape = np.zeros((node_number, 3))
    node_third_mode_shape = np.zeros((node_number, 3))
    for nodeTag in range(1, node_number+1):
        first_mode_shape = np.array(ops.nodeEigenvector(nodeTag, 1))
        second_mode_shape = np.array(ops.nodeEigenvector(nodeTag, 2))
        third_mode_shape = np.array(ops.nodeEigenvector(nodeTag, 3))
        node_first_mode_shape[nodeTag-1] = first_mode_shape[[0, 1, 5]]  # Ux, Uy, Rz
        node_second_mode_shape[nodeTag-1] = second_mode_shape[[0, 1, 5]]
        node_third_mode_shape[nodeTag-1] = third_mode_shape[[0, 1, 5]]
    mode_shapes = np.hstack((node_first_mode_shape, node_second_mode_shape, node_third_mode_shape))  # shape: (node_number, 9)

    return periods, mode_shapes

def run_modal_analysis(x_grid, y_grid, z_grid, 
                       node_coord_dict, 
                       node_dof_dict, 
                       node_translational_mass_dict, 
                       node_inertia_dict, 
                       slave_node_dict, 
                       member_to_nodeIndex_dict, 
                       member_category_dict, 
                       member_section_dict, 
                       member_A_dict) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Generate analysis model in **OpenSees**
    2. Run modal analysis
    3. Return **mode periods** and **mode shapes**
    """
    t_start = time.time()

    _generate_analysis_model(x_grid, y_grid, z_grid, 
                             node_coord_dict, 
                             node_dof_dict, 
                             node_translational_mass_dict, 
                             node_inertia_dict, 
                             slave_node_dict, 
                             member_to_nodeIndex_dict, 
                             member_category_dict, 
                             member_section_dict, 
                             member_A_dict)
    mode_periods, mode_shapes = _run_single_modal_analysis()

    t_end = time.time()
    print(f"\t\tused time for opensees.run_modal_analysis: {t_end - t_start:.3f} sec")
    
    return mode_periods, mode_shapes


def _strongColumn_weakBeam_beta(node_neighbor_Zz_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Set ratio beta to embedding node v_hat, and then check if all beta is greater than 1."""
    Mpc = node_neighbor_Zz_matrix.float() @ torch.tensor([0, 0, 1, 1, 0, 0]).unsqueeze(1).float()  # [node_number, 6] @ [6, 1] = [node_number, 1]
    Mpb_x = node_neighbor_Zz_matrix @ torch.tensor([1, 1, 0, 0, 0, 0]).unsqueeze(1).float()        # [node_number, 6] @ [6, 1] = [node_number, 1]
    Mpb_z = node_neighbor_Zz_matrix @ torch.tensor([0, 0, 0, 0, 1, 1]).unsqueeze(1).float()        # [node_number, 6] @ [6, 1] = [node_number, 1]
    beta_x = (Mpc / (Mpb_x + 1e-6)).squeeze()  # [node_number]
    beta_z = (Mpc / (Mpb_z + 1e-6)).squeeze()  # [node_number]
    beta_x = torch.tanh(1.0 / beta_x)
    beta_z = torch.tanh(1.0 / beta_z)
    return beta_x, beta_z

def pisa_file_to_opensees_model(ipt_path: Path):
    ipt_file = open(ipt_path, 'r')
    lines = ipt_file.readlines()

    node_coord_dict = {}  # node_name: [x, y, z]
    node_dof_dict = {}  # node_name: dof (free:0, fixed:1)
    node_translational_mass_dict = {}  # node_name: translation_mass (unit: kN / (mm/s^2))
    node_inertia_dict = {}  # node_name: [Rx, Ry, Rz] (unit: kN / (mm/s^2) * mm^2)
    slave_node_dict = {}  # master_name: [slave_name1, slave_name2, ...]

    member_to_nodeIndex_dict = {}  # member_name: [node1_index, node2_index, face1_number, face2_number, My_face1_index, My_face2_index], whrere face_index is the index in the node feature
    member_category_dict = {}  # member_name: category ('x', 'y', 'z')
    member_section_dict = {}  # member_name: section_index (0, 1, 2, ..., 14)
    member_A_dict = {}  # member_name: area (unit: mm^2)
    for line in lines:
        contents = line.split()
        if len(contents) == 0: continue

        if contents[0] == "GUI_GRID":
            if contents[1] == "XDIR": x_grid = [int(x) for x in contents[2:]]
            if contents[1] == "YDIR": y_grid = [int(y) for y in contents[2:]]
            if contents[1] == "ZDIR": z_grid = [int(z) for z in contents[2:]]

        if contents[0] == "Node": 
            node_name = contents[1]
            if node_name[0] == "N":
                x = int(contents[2])
                y = int(contents[3])
                z = int(contents[4])
                node_coord_dict[node_name] = [x, y, z]

        if contents[0] == "DOF":
            node_name = contents[1]
            if node_name[0] == "N":
                node_dof_dict[node_name] = 1

        if contents[0] == "#Mass": 
            node_name = contents[1]
            mass = float(contents[2])
            node_translational_mass_dict[node_name] = mass
            Rx = float(contents[5])
            Ry = float(contents[6])
            Rz = float(contents[7])
            node_inertia_dict[node_name] = [Rx, Ry, Rz]

        if contents[0] == "Constraint":
            master_name = contents[3]
            slave_node_dict[master_name] = [slave_name for slave_name in contents[4:]]

        if contents[0] == "Element":
            member_name = contents[2]
            node1_name = contents[3]
            node2_name = contents[4]
            node_distance = np.array(node_coord_dict[node1_name]) - np.array(node_coord_dict[node2_name])
            node1_index = int(node1_name[1:]) - 1
            node2_index = int(node2_name[1:]) - 1
            if node_distance[0] != 0:
                member_to_nodeIndex_dict[member_name] = [node1_index, node2_index, 1, 0, NODE_FEATURE_FACE_INDEX['x_p'], NODE_FEATURE_FACE_INDEX['x_n']]
                member_category_dict[member_name] = 'x'
            elif node_distance[1] != 0:
                member_to_nodeIndex_dict[member_name] = [node1_index, node2_index, 3, 2, NODE_FEATURE_FACE_INDEX['y_p'], NODE_FEATURE_FACE_INDEX['y_n']]
                member_category_dict[member_name] = 'y'
            elif node_distance[2] != 0:
                member_to_nodeIndex_dict[member_name] = [node1_index, node2_index, 5, 4, NODE_FEATURE_FACE_INDEX['z_p'], NODE_FEATURE_FACE_INDEX['z_n']]
                member_category_dict[member_name] = 'z'
            else:
                raise ValueError(f"Invalid node distance: {node_distance}")
            
            section_name = contents[5]
            member_sections = column_sections if member_category_dict[member_name] == 'y' else beam_sections
            for section_index, section in enumerate(member_sections):
                if section["name"] == section_name:
                    member_section_dict[member_name] = section_index
                    member_A_dict[member_name] = section["A(cm2)"] * 1e+2
                    break

    # for member_name, section_index in member_section_dict.items():
    #     node1_index, node2_index = member_to_nodeIndex_dict[member_name][:2]
    #     category = member_category_dict[member_name]
    #     member_sections = sections_tw.column_sections if category == 'y' else sections_tw.beam_sections
    #     section_name = member_sections[section_index]["name"]
    #     print(f"{member_name} N{node1_index+1} N{node2_index+1} {section_name}")

    node_number = len(node_coord_dict)
    node_need_beta = torch.ones(node_number)
    for node_name, coord in node_coord_dict.items():
        node_index = int(node_name[1:])-1
        x, y, z = coord
        if y == 0 or y == y_grid[-1]:
            node_need_beta[node_index] = 0

    node_neighbor_Zz_matrix = torch.zeros(node_number, 6)  # 6 is for xn(0), xp(1), yn(2), yp(3), zn(4), zp(5)
    for member_name in member_to_nodeIndex_dict.keys():
        category = member_category_dict[member_name]
        section_index = member_section_dict[member_name]
        if category == 'y':
            member_Zz = column_sections[section_index]['Z_z(cm3)'] * 1e+3  # mm^3
        else: 
            member_Zz = beam_sections[section_index]['Z_z(cm3)'] * 1e+3  # mm^3

        node1_index, node2_index, face1_number, face2_number, My_face1_index, My_face2_index = member_to_nodeIndex_dict[member_name]
        node_neighbor_Zz_matrix[node1_index, face1_number] = member_Zz
        node_neighbor_Zz_matrix[node2_index, face2_number] = member_Zz

    beta_x, beta_z = _strongColumn_weakBeam_beta(node_neighbor_Zz_matrix)

    _generate_analysis_model(x_grid, y_grid, z_grid, 
                             node_coord_dict, node_dof_dict, 
                             node_translational_mass_dict, node_inertia_dict, slave_node_dict, 
                             member_to_nodeIndex_dict, member_category_dict,
                             member_section_dict, member_A_dict)
    mode_periods, mode_shapes = _run_single_modal_analysis(node_number)

    return beta_x, beta_z, mode_periods, mode_shapes

