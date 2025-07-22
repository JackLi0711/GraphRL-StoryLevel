import time
import numpy as np
import openseespy.opensees as ops
from pathlib import Path

from Structure import earthquake, load
from Structure.structure import Structure
from Structure.sections import beam_sections, column_sections


class Response:        
    def __init__(self, structure: Structure):
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
        self._load_disp()
        self._load_elem()

    def __str__(self):
        info = ""
        for node_name in self.node_response["dispX"].keys():
            info += f"{node_name:^4s}, {self.node_response['dispX'][node_name]:^8.2f}\n"
        return info

    def _load_disp(self):
        # 6.15. nodeDisp command (https://openseespydoc.readthedocs.io/en/latest/src/nodeDisp.html)
        for nodeTag in range(1, self.node_number+1):
            node_disp = np.array(ops.nodeDisp(nodeTag))
            node_name = f"N{nodeTag}"
            self.node_response["dispX"][node_name] = node_disp[0]
            self.node_response["dispZ"][node_name] = node_disp[1]

    def _load_elem(self):
        # 6.7. eleResponse command (https://openseespydoc.readthedocs.io/en/latest/src/eleResponse.html)
        for eleTag in range(1, self.member_number+1):
            member_name = f"E{eleTag}"
            category = self.structure.member_category_dict[member_name]
            ele_response = np.array(ops.eleResponse(eleTag, 'force')).reshape(-1, 6)
            FXI, FYI, FZI, MXI, MYI, MZI = ele_response[0]
            FXJ, FYJ, FZJ, MXJ, MYJ, MZJ = ele_response[1]
            if category == 'y':
                self.member_response["shearY"][member_name] = FXI if abs(FXI) > abs(FXJ) else FXJ
                self.member_response["shearZ"][member_name] = FYI if abs(FYI) > abs(FYJ) else FYJ
                self.member_response["momentY"][member_name] = MXI if abs(MXI) > abs(MXJ) else MXJ
                self.member_response["momentZ"][member_name] = MYI if abs(MYI) > abs(MYJ) else MYJ
                self.member_response["axial"][member_name] = FZI if abs(FZI) > abs(FZJ) else FZJ
            elif category == 'x':
                self.member_response["shearY"][member_name] = FZI if abs(FZI) > abs(FZJ) else FZJ
                self.member_response["shearZ"][member_name] = FYI if abs(FYI) > abs(FYJ) else FYJ
                self.member_response["momentY"][member_name] = MZI if abs(MZI) > abs(MZJ) else MZJ
                self.member_response["momentZ"][member_name] = MYI if abs(MYI) > abs(MYJ) else MYJ
                self.member_response["axial"][member_name] = FXI if abs(FXI) > abs(FXJ) else FXJ
            elif category == 'z':
                self.member_response["shearY"][member_name] = FZI if abs(FZI) > abs(FZJ) else FZJ
                self.member_response["shearZ"][member_name] = FXI if abs(FXI) > abs(FXJ) else FXJ
                self.member_response["momentY"][member_name] = MZI if abs(MZI) > abs(MZJ) else MZJ
                self.member_response["momentZ"][member_name] = MXI if abs(MXI) > abs(MXJ) else MXJ
                self.member_response["axial"][member_name] = FYI if abs(FYI) > abs(FYJ) else FYJ

            node1_index, node2_index, face_number1, face_number2, _, _ = self.structure.member_to_nodeIndex_dict[member_name]
            AXIAL = self.member_response["axial"][member_name]
            self.node_neighbor_Puc_matrix[node1_index, face_number1] = AXIAL
            self.node_neighbor_Puc_matrix[node2_index, face_number2] = AXIAL


class NewResponse:
    def __init__(self, structure: Structure):
        self.structure = structure
        self.node_number = structure.node_number
        self.member_number = structure.member_number
        self.all_node_response = np.array([ops.nodeDisp(nodeTag) for nodeTag in range(1, self.node_number+1)])  # shape: (node_number, 6)
        self.all_member_response = np.array([np.array(ops.eleResponse(eleTag, 'force')).reshape(-1, 6) for eleTag in range(1, self.member_number+1)])  # shape: (member_number, 2, 6)
        
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

    def to_response_dict(self):
        for nodeTag in range(1, self.node_number+1):
            node_disp = self.all_node_response[nodeTag-1]  # shape: (6,)
            node_name = f"N{nodeTag}"
            self.node_response["dispX"][node_name] = node_disp[0]
            self.node_response["dispZ"][node_name] = node_disp[1]
        for eleTag in range(1, self.member_number+1):
            member_name = f"E{eleTag}"
            category = self.structure.member_category_dict[member_name]
            member_response = self.all_member_response[eleTag-1]  # shape: (2, 6)
            FXI, FYI, FZI, MXI, MYI, MZI = member_response[0]
            FXJ, FYJ, FZJ, MXJ, MYJ, MZJ = member_response[1]
            if category == 'y':
                self.member_response["shearY"][member_name] = FXI if abs(FXI) > abs(FXJ) else FXJ
                self.member_response["shearZ"][member_name] = FYI if abs(FYI) > abs(FYJ) else FYJ
                self.member_response["momentY"][member_name] = MXI if abs(MXI) > abs(MXJ) else MXJ
                self.member_response["momentZ"][member_name] = MYI if abs(MYI) > abs(MYJ) else MYJ
                self.member_response["axial"][member_name] = FZI if abs(FZI) > abs(FZJ) else FZJ
            elif category == 'x':
                self.member_response["shearY"][member_name] = FZI if abs(FZI) > abs(FZJ) else FZJ
                self.member_response["shearZ"][member_name] = FYI if abs(FYI) > abs(FYJ) else FYJ
                self.member_response["momentY"][member_name] = MZI if abs(MZI) > abs(MZJ) else MZJ
                self.member_response["momentZ"][member_name] = MYI if abs(MYI) > abs(MYJ) else MYJ
                self.member_response["axial"][member_name] = FXI if abs(FXI) > abs(FXJ) else FXJ
            elif category == 'z':
                self.member_response["shearY"][member_name] = FZI if abs(FZI) > abs(FZJ) else FZJ
                self.member_response["shearZ"][member_name] = FXI if abs(FXI) > abs(FXJ) else FXJ
                self.member_response["momentY"][member_name] = MZI if abs(MZI) > abs(MZJ) else MZJ
                self.member_response["momentZ"][member_name] = MXI if abs(MXI) > abs(MXJ) else MXJ
                self.member_response["axial"][member_name] = FYI if abs(FYI) > abs(FYJ) else FYJ

            node1_index, node2_index, face_number1, face_number2, _, _ = self.structure.member_to_nodeIndex_dict[member_name]
            AXIAL = self.member_response["axial"][member_name]
            self.node_neighbor_Puc_matrix[node1_index, face_number1] = AXIAL
            self.node_neighbor_Puc_matrix[node2_index, face_number2] = AXIAL


def design_spectrum():
    """
    Design spectrum for Taipei Zone III
    * return: target_start, target_end, periods, design_spectrum_BSE1, design_spectrum_BSE2
    """
    T_start = 0.06  # 0.2 * the smallest 1st mode period, unit: sec
    T_end = 2.6     # 2 * the largest 1st mode period, unit: sec
    periods = np.linspace(0.001, 5, 100)
    target_start, target_end = None, None  # indexes for t = min_t ~ max_t
    for i, t in enumerate(periods):
        if target_start == None and t >= T_start:
            target_start = i-1
        if target_end == None and t > T_end:
            target_end = i
    # print("start:", target_start, "T:", periods[target_start]) --> start: 1, T: 0.05149494949494949
    # print("end:", target_end, "T:", periods[target_end]) --> end: 52, T: 2.6267373737373734

    S_DS = 0.6
    T0 = 1.05
    design_spectrum_BSE1 = np.zeros(100)
    design_spectrum_BSE2 = np.zeros(100)
    for i, t in enumerate(periods):
        if t < 0.2 * T0:
            design_spectrum_BSE1[i] = S_DS * (0.4 + 3 * t / T0)
        elif t >= 0.2 * T0 and t <= T0:
            design_spectrum_BSE1[i] = S_DS
        elif t > T0:
            design_spectrum_BSE1[i] = S_DS * T0 / t
        else:
            design_spectrum_BSE1[i] = None
    design_spectrum_BSE2 = design_spectrum_BSE1 * (4/3)

    return target_start, target_end, periods, design_spectrum_BSE1, design_spectrum_BSE2


def CQC(responses, lambdas, damping_ratios, scale_factors):
    """
    Complete Quadratic Combination (CQC) function
    * responses: np.array of modal responses (e.g., disp, vel, acc, force), shape: (n_modes, n_elements, ...)
    * lambdas: list of eigenvalues
    * damping_ratios: list of damping ratios
    * scale_factors: list of scaling factors
    """
    total_responses = 0
    for i in range(len(lambdas)):
        for j in range(len(lambdas)):
            zeta_i = damping_ratios[i]
            zeta_j = damping_ratios[j]
            r = np.sqrt(lambdas[j] / lambdas[i])
            rho = (8*np.sqrt(zeta_i*zeta_j)*(zeta_i+r*zeta_j)*(r**(3/2))) / ((1-r**2)**2 + 4*zeta_i*zeta_j*r*(1+r**2) + 4*(zeta_i**2+zeta_j**2)*r**2)
            total_responses += scale_factors[i]*responses[i, :] * scale_factors[j]*responses[j, :] * rho
    
    return np.sqrt(total_responses)  # shape: (n_elements, ...)


mm = 1  # length
kN = 1  # force
m = 1e+3 * mm
N = 1e-3 * kN

Pa = N / m**2
GPa = 1e+9 * Pa
E = 200 * GPa  # Young's modulud, GPa
Nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + Nu))  # Shear modulus, GPa


def _generate_analysis_model(structure: Structure) -> None:
    """Generate analysis model in **OpenSees**"""
    # 7.25. wipe command (https://openseespydoc.readthedocs.io/en/latest/src/wipe.html)
    ops.wipe()  # remove existing model

    # 4.1. model command (https://openseespydoc.readthedocs.io/en/latest/src/model.html)
    ops.model("basic", "-ndm",3, "-ndf",6)

    # 4.3. node command (https://openseespydoc.readthedocs.io/en/latest/src/node.html)
    for node_name, (x, y, z) in structure.node_coord_dict.items():
        nodeTag = int(node_name[1:])
        crds = [x, z, y]
        # node_translational_mass_dict --> key: node_name(N1, N2, ...), value: translational mass, unit: kN / (mm/s^2)
        trans_mass = structure.node_translational_mass_dict[node_name]
        # node_inertia_dict --> key: node_name(N1, N2, ...), value: inertia distributed to node (Rx, Ry, Rz), unit: kN / (mm/s^2) * mm^2
        Rx, Ry, Rz = structure.node_inertia_dict[node_name]
        mass = [trans_mass, trans_mass, 0] + [0, 0, Ry]
        ops.node(nodeTag, *crds)  # , '-mass',*mass
    
        # 4.4. fix command (https://openseespydoc.readthedocs.io/en/latest/src/fix.html)
        # node_dof_dict --> key: node_name(N1, N2, ...), value: dof(free:0, fixed:1)
        if structure.node_dof_dict[node_name] == 1:
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
    for member_name, member_info in structure.member_to_nodeIndex_dict.items():
        eleTag = int(member_name[1:])
        node1_index, node2_index = member_info[:2]
        eleNodes = [node1_index+1, node2_index+1]  # nodeTag == node_index + 1 == int(node_name[1:])
        Area = structure.member_A_dict[member_name]  # mm^2

        member_sections = column_sections if structure.member_category_dict[member_name] == 'y' else beam_sections
        sections_index = structure.member_section_dict[member_name]
        section = member_sections[sections_index]
        Jxx = section["J(cm4)"] * 1e+4  # torsional constant, mm^4
        Iy = section["I_y(cm4)"] * 1e+4  # moment of inertia, mm^4
        Iz = section["I_z(cm4)"] * 1e+4  # moment of inertia, mm^4
        Avy = section["Av_y(cm2)"] * 1e+2  # shear area, mm^2
        Avz = section["Av_z(cm2)"] * 1e+2  # shear area, mm^2

        if structure.member_category_dict[member_name] == 'x':
            transfTag = XdirBeamTransfTag
        elif structure.member_category_dict[member_name] == 'z':
            transfTag = ZdirBeamTransfTag
        else:
            transfTag = ColumnTransfTag

        # 4.2.3.1. Elastic Beam Column Element
        # ops.element('elasticBeamColumn', eleTag, *eleNodes, Area, E, G, Jxx, Iy, Iz, transfTag)
        # 4.2.3.3. Elastic Timoshenko Beam Column Element --> accounts for shear deformations
        ops.element('ElasticTimoshenkoBeam', eleTag, *eleNodes, E, G, Area, Jxx, Iy, Iz, Avy, Avz, transfTag)

    # master node
    total_trans_mass, total_moment_inertia = 0, 0
    for story, y in enumerate(structure.y_grid):
        if story > 0:
            master_name = f"M{story}F"            
            trans_mass, moment_inertia = 0, 0
            for slave_name in structure.slave_node_dict[master_name]:
                trans_mass += structure.node_translational_mass_dict[slave_name]  # kN / (mm/s^2)
                Rx, Ry, Rz = structure.node_inertia_dict[slave_name]  # kN / (mm/s^2) * mm^2
                moment_inertia += Ry
            rNodeTag = story + structure.node_number
            crds = [structure.master_x, structure.master_z, y]
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
            cNodeTags = [int(slave_name[1:]) for slave_name in structure.slave_node_dict[master_name]]
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

    filename = str(structure.analysis_dir / "model.json")
    ops.printModel('-JSON', '-file',filename)


def _run_single_modal_analysis(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    """
    * Run **OpenSees** modal analysis
    * Return **mode periods** and **mode shapes**
    """
    # 5.8. eigen command (https://openseespydoc.readthedocs.io/en/latest/src/eigen.html#id0)
    numEigenvalues = 10
    eigenValues = ops.eigen('-fullGenLapack', numEigenvalues)  # use '-fullGenLapack' (non-default solver) to avoid error
    omegas = np.sqrt(eigenValues)
    periods = 2 * np.pi / omegas
    
    # 5.10. modalProperties Command (https://openseespydoc.readthedocs.io/en/latest/src/modalProperties.html#modalProperties)
    reportFileName = str(structure.analysis_dir / "modal_property.txt")
    ops.modalProperties('-file', reportFileName)

    node_number = structure.node_number
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
    mode_shapes = np.hstack((node_first_mode_shape, node_second_mode_shape, node_third_mode_shape))

    return periods, mode_shapes

def run_modal_analysis(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Generate analysis model in **OpenSees**
    2. Run modal analysis
    3. Return **mode periods** and **mode shapes**
    """
    t_start = time.time()

    _generate_analysis_model(structure)
    mode_periods, mode_shapes = _run_single_modal_analysis(structure)

    t_end = time.time()
    print(f"\t\tused time for opensees.run_modal_analysis: {t_end - t_start:.3f} sec")
    
    return mode_periods, mode_shapes


def _run_single_static_analysis(structure: Structure, nodal_loads: list[np.ndarray], analysis_dir: Path, response_type="original"):
    """
    * Run **OpenSees** static analysis
    * Return response
    """
    t_start = time.time()
    # 7.27. wipeAnalysis command (https://openseespydoc.readthedocs.io/en/latest/src/wipeAnalysis.html)
    ops.wipeAnalysis()  # remove existing analysis

    # 4.7. timeSeries command (https://openseespydoc.readthedocs.io/en/latest/src/timeSeries.html)
    tsTag = 1
    ops.timeSeries('Linear', tsTag)  # 'Constant', 'Linear', 'Rectangular', 'Path'

    # 4.8. pattern command (https://openseespydoc.readthedocs.io/en/latest/src/pattern.html)
    patternTag = 1
    ops.pattern('Plain', patternTag, tsTag)

    # 4.8.1. load command (https://openseespydoc.readthedocs.io/en/latest/src/load.html)
    # nodal_loads = load_case.calculate_nodal_load(structure)
    for node_name, self_weight in structure.node_self_weight_dict.items():
        nodeTag = int(node_name[1:])
        node_index = nodeTag - 1
        load_y, load_x, load_z = nodal_loads[0][node_index], nodal_loads[1][node_index], nodal_loads[2][node_index]  # kN
        load_y *= -1
        loadValues = [load_x, load_z, load_y, 0, 0, 0]
        # print(f"Node {nodeTag}: {loadValues}")
        ops.load(nodeTag, *loadValues)

    # 5.3. system command (https://openseespydoc.readthedocs.io/en/latest/src/system.html)
    ops.system('BandGen')  # 'BandGen', 'BandSPD', 'ProfileSPD', 'UmfPack', 'FullGeneral'

    # 5.2. numberer command (https://openseespydoc.readthedocs.io/en/latest/src/numberer.html)
    ops.numberer('Plain')  # 'Plain', 'RCM', 'AMD'

    # 5.1. constraints command (https://openseespydoc.readthedocs.io/en/latest/src/constraints.html)
    ops.constraints('Transformation')  # 'Plain', 'Lagrange', 'Penalty', 'Transformation'

    # 5.6. integrator command (https://openseespydoc.readthedocs.io/en/latest/src/integrator.html)
    ops.integrator('LoadControl', 1.0)

    # 5.5. algorithm command (https://openseespydoc.readthedocs.io/en/latest/src/algorithm.html)
    ops.algorithm('Linear')  # 'Linear', 'Newton', 'NewtonLineSearch', 'ModifiedNewton'

    # 5.7. analysis command (https://openseespydoc.readthedocs.io/en/latest/src/analysis.html)
    ops.analysis('Static')  # Static', 'Transient'

    # # 6.31.1. node recorder command (https://openseespydoc.readthedocs.io/en/latest/src/nodeRecorder.html)
    # filename = str(analysis_dir / f"{load_case.load_name}_node_disp.txt")
    # nSD = 6
    # nodeTags = list(range(1, structure.node_number+1))
    # dofs = [1, 2, 3, 4, 5, 6]
    # respType = 'disp'
    # ops.recorder('Node', '-file',filename, '-precision',nSD, '-node',*nodeTags, '-dof',*dofs, respType)

    # # 6.31.3. element recorder command (https://openseespydoc.readthedocs.io/en/latest/src/elementRecorder.html)
    # filename = str(analysis_dir / f"{load_case.load_name}_member_response.txt")
    # nSD = 6
    # eleTags = ops.getEleTags()
    # args = ['force']
    # ops.recorder('Element', '-file',filename, '-precision',nSD, '-ele',*eleTags, *args)
    
    # 5.9. analyze command (https://openseespydoc.readthedocs.io/en/latest/src/analyze.html)
    ops.analyze(1)

    # 7.5. loadConst command (https://openseespydoc.readthedocs.io/en/latest/src/loadConst.html)
    ops.loadConst('-time', 0.0)

    # 7.8. remove command (https://openseespydoc.readthedocs.io/en/latest/src/remove.html)
    ops.remove('timeSeries', tsTag)
    ops.remove('loadPattern', patternTag)

    if response_type == "original":
        response = Response(structure)
    elif response_type == "new":
        response = NewResponse(structure)

    t_end = time.time()
    # print(f"\t\t\tused time for opensees._run_single_static_analysis: {t_end - t_start:.3f} sec")

    return response

def run_static_analysis(structure: Structure, load_cases: list[load.NodalLoad], analysis_dir: Path):
    """
    1. Generate analysis model in **OpenSees**
    2. Run static analysis
    3. Return a list of responses for each load case
    """
    t_start = time.time()

    _generate_analysis_model(structure)
    responses = []
    for load_case in load_cases:
        nodal_loads = load_case.calculate_nodal_load(structure)
        response = _run_single_static_analysis(structure, nodal_loads, analysis_dir)
        responses.append(response)
    
    t_end = time.time()
    print(f"\t\tused time for opensees.run_static_analysis: {t_end - t_start:.3f} sec")
    
    return responses


def _run_single_response_spectrum_analysis_v1(structure: Structure, lambdas: list[float], Tn: np.ndarray, Sa: np.ndarray, direction=1) -> Response:
    """
    * Run **OpenSees** response spectrum analysis
    * Return modal response
    """
    t_start = time.time()
    modal_dispX = np.zeros((len(lambdas), structure.node_number))
    modal_dispZ = np.zeros((len(lambdas), structure.node_number))
    modal_shearY = np.zeros((len(lambdas), structure.member_number))
    modal_shearZ = np.zeros((len(lambdas), structure.member_number))
    modal_momentY = np.zeros((len(lambdas), structure.member_number))
    modal_momentZ = np.zeros((len(lambdas), structure.member_number))
    modal_axial = np.zeros((len(lambdas), structure.member_number))

    for mode in range(1, len(lambdas)+1):
        ops.responseSpectrumAnalysis(direction, '-Tn',*Tn, '-Sa',*Sa, '-mode',mode)
        response = Response(structure)
        for nodeTag in range(1, structure.node_number+1):
            node_name = f"N{nodeTag}"
            modal_dispX[mode-1, nodeTag-1] = response.node_response["dispX"][node_name]
            modal_dispZ[mode-1, nodeTag-1] = response.node_response["dispZ"][node_name]
        for eleTag in range(1, structure.member_number+1):
            member_name = f"E{eleTag}"
            modal_shearY[mode-1, eleTag-1] = response.member_response["shearY"][member_name]
            modal_shearZ[mode-1, eleTag-1] = response.member_response["shearZ"][member_name]
            modal_momentY[mode-1, eleTag-1] = response.member_response["momentY"][member_name]
            modal_momentZ[mode-1, eleTag-1] = response.member_response["momentZ"][member_name]
            modal_axial[mode-1, eleTag-1] = response.member_response["axial"][member_name]

    damping_ratios = [0.05] * len(lambdas)  # same damping for each mode (5% as same as used one in response spectrum)
    scale_factors = [1.0] * len(lambdas)  # treat all modes equally
    superposition_modal_dispX = CQC(modal_dispX, lambdas, damping_ratios, scale_factors)  # shape: (structure.node_number, )
    superposition_modal_dispZ = CQC(modal_dispZ, lambdas, damping_ratios, scale_factors)  # shape: (structure.node_number, )
    superposition_modal_shearY = CQC(modal_shearY, lambdas, damping_ratios, scale_factors)  # shape: (structure.member_number, )
    superposition_modal_shearZ = CQC(modal_shearZ, lambdas, damping_ratios, scale_factors)  # shape: (structure.member_number, )
    superposition_modal_momentY = CQC(modal_momentY, lambdas, damping_ratios, scale_factors)  # shape: (structure.member_number, )
    superposition_modal_momentZ = CQC(modal_momentZ, lambdas, damping_ratios, scale_factors)  # shape: (structure.member_number, )
    superposition_modal_axial = CQC(modal_axial, lambdas, damping_ratios, scale_factors)  # shape: (structure.member_number, )
    
    superposition_modal_response = Response(structure)
    for nodeTag in range(1, structure.node_number+1):
        node_name = f"N{nodeTag}"
        superposition_modal_response.node_response["dispX"][node_name] = superposition_modal_dispX[nodeTag-1]
        superposition_modal_response.node_response["dispZ"][node_name] = superposition_modal_dispZ[nodeTag-1]
    for eleTag in range(1, structure.member_number+1):
        member_name = f"E{eleTag}"
        superposition_modal_response.member_response["shearY"][member_name] = superposition_modal_shearY[eleTag-1]
        superposition_modal_response.member_response["shearZ"][member_name] = superposition_modal_shearZ[eleTag-1]
        superposition_modal_response.member_response["momentY"][member_name] = superposition_modal_momentY[eleTag-1]
        superposition_modal_response.member_response["momentZ"][member_name] = superposition_modal_momentZ[eleTag-1]
        superposition_modal_response.member_response["axial"][member_name] = superposition_modal_axial[eleTag-1]

        node1_index, node2_index, face_number1, face_number2, _, _ = structure.member_to_nodeIndex_dict[member_name]
        AXIAL = superposition_modal_response.member_response["axial"][member_name]
        superposition_modal_response.node_neighbor_Puc_matrix[node1_index, face_number1] = AXIAL
        superposition_modal_response.node_neighbor_Puc_matrix[node2_index, face_number2] = AXIAL
    
    t_end = time.time()
    # print(f"\t\t\tused time for opensees._run_single_response_spectrum_analysis: {t_end - t_start:.3f} sec")

    return superposition_modal_response

def _run_single_response_spectrum_analysis_v2(structure: Structure, lambdas: list[float], Tn: np.ndarray, Sa: np.ndarray, direction=1) -> NewResponse:
    modal_all_node_response = np.zeros((len(lambdas), structure.node_number, 6))  # shape: (mode, node_number, 6)
    modal_all_member_response = np.zeros((len(lambdas), structure.member_number, 2, 6))  # shape: (mode, member_number, 2, 6)

    for mode in range(1, len(lambdas)+1):
        ops.responseSpectrumAnalysis(direction, '-Tn',*Tn, '-Sa',*Sa, '-mode',mode)
        response = NewResponse(structure) 
        modal_all_node_response[mode-1, :, :] = response.all_node_response  # shape: (node_number, 6)
        modal_all_member_response[mode-1, :, :, :] = response.all_member_response  # shape: (member_number, 2, 6)

    damping_ratios = [0.05] * len(lambdas)  # same damping for each mode (5% as same as used one in response spectrum)
    scale_factors = [1.0] * len(lambdas)  # treat all modes equally
    superposition_modal_node_response = CQC(modal_all_node_response, lambdas, damping_ratios, scale_factors)  # shape: (node_number, 6)
    superposition_modal_member_response = CQC(modal_all_member_response, lambdas, damping_ratios, scale_factors)  # shape: (member_number, 2, 6)

    superposition_modal_response = NewResponse(structure)
    superposition_modal_response.all_node_response = superposition_modal_node_response  # shape: (node_number, 6)
    superposition_modal_response.all_member_response = superposition_modal_member_response  # shape: (member_number, 2, 6)

    # For base shear in practice (1. sum all column reponse, 2. perform CQC)
    # modal_column_response = modal_all_member_response[:, structure.story_column_member[0], 0, :]
    # sum_modal_column_response = np.sum(modal_column_response, axis=1)  # shape: (mode, 6)
    # superposition_modal_column_response = CQC(sum_modal_column_response, lambdas, damping_ratios, scale_factors)  # shape: (6,)
    # if direction == 1:
    #     modal_base_shear = superposition_modal_column_response[0]  # shearY
    # elif direction == 2:
    #     modal_base_shear = superposition_modal_column_response[1]  # shearZ

    return superposition_modal_response


def run_response_spectrum_analysis(structure: Structure, load_cases: list[load.NodalLoad], analysis_dir: Path) -> list[NewResponse]:
    t_start = time.time()

    _, _, periods, design_spectrum_BSE1, design_spectrum_BSE2 = design_spectrum()
    Tn = periods
    Sa = design_spectrum_BSE1 * 9.81 * 1000  # unit: g --> mm/s^2

    mode_periods = np.array([structure.first_mode_period, structure.second_mode_period, structure.third_mode_period])
    eigenvalues = 4 * np.pi**2 / mode_periods**2

    static_analysis_counter = 0
    response_spectrum_analysis_counter = 0

    horizontal_earthquake_forces, scale_factors, Fus, vertical_earthquake_force, Fuv = earthquake.design_earthquake_force(structure)
    horizontal_earthquake_forces_xdir, horizontal_earthquake_forces_zdir = horizontal_earthquake_forces
    scale_factors_xdir, scale_factors_zdir = scale_factors
    horizontal_earthquake_force_xdir = horizontal_earthquake_forces_xdir["V"]
    horizontal_earthquake_force_zdir = horizontal_earthquake_forces_zdir["V"]
    scale_factor_xdir = scale_factors_xdir["V"]
    scale_factor_zdir = scale_factors_zdir["V"]

    ### Vertical Response ###
    nodal_dead_loads = load.get_nodal_dead_load(structure)
    nodal_live_loads = load.get_nodal_live_load(structure)
    nodal_self_weights = load.get_nodal_self_weight(structure)
    nodal_Ey_loads = load.get_nodal_vertical_earthquake_load(structure, vertical_earthquake_force)
    dead_load_response = _run_single_static_analysis(structure, nodal_dead_loads, analysis_dir, response_type="new")
    live_load_response = _run_single_static_analysis(structure, nodal_live_loads, analysis_dir, response_type="new")
    self_weight_response = _run_single_static_analysis(structure, nodal_self_weights, analysis_dir, response_type="new")
    Ey_response = _run_single_static_analysis(structure, nodal_Ey_loads, analysis_dir, response_type="new")
    static_analysis_counter += 4

    ### Horizontal Response ###
    scale_Sa_xdir = Sa * scale_factor_xdir
    scale_Sa_zdir = Sa * scale_factor_zdir
    modal_response_xdir = _run_single_response_spectrum_analysis_v2(structure, eigenvalues, Tn, scale_Sa_xdir, direction=1)
    modal_response_zdir = _run_single_response_spectrum_analysis_v2(structure, eigenvalues, Tn, scale_Sa_zdir, direction=2)
    modal_response_xdir.to_response_dict()
    modal_response_zdir.to_response_dict()
    modal_base_shear_xdir = np.sum([modal_response_xdir.member_response["shearY"][f"E{member_index+1}"] for member_index in structure.story_column_member[0]])
    modal_base_shear_zdir = np.sum([modal_response_zdir.member_response["shearZ"][f"E{member_index+1}"] for member_index in structure.story_column_member[0]])
    base_shear_ratio_xdir = horizontal_earthquake_force_xdir / modal_base_shear_xdir
    base_shear_ratio_zdir = horizontal_earthquake_force_zdir / modal_base_shear_zdir
    response_spectrum_analysis_counter += 2

    Exn_loads = load.get_nodal_horizontal_earthquake_load(structure, horizontal_earthquake_force_xdir, direction="x_n")
    Exp_loads = load.get_nodal_horizontal_earthquake_load(structure, horizontal_earthquake_force_xdir, direction="x_p")
    Ezn_loads = load.get_nodal_horizontal_earthquake_load(structure, horizontal_earthquake_force_zdir, direction="z_n")
    Ezp_loads = load.get_nodal_horizontal_earthquake_load(structure, horizontal_earthquake_force_zdir, direction="z_p")
    Exn_response = _run_single_static_analysis(structure, Exn_loads, analysis_dir, response_type="new")
    Exp_response = _run_single_static_analysis(structure, Exp_loads, analysis_dir, response_type="new")
    Ezn_response = _run_single_static_analysis(structure, Ezn_loads, analysis_dir, response_type="new")
    Ezp_response = _run_single_static_analysis(structure, Ezp_loads, analysis_dir, response_type="new")
    static_analysis_counter += 4

    all_node_response_Exn = base_shear_ratio_xdir * (modal_response_xdir.all_node_response * np.sign(Exn_response.all_node_response))
    all_node_response_Exp = base_shear_ratio_xdir * (modal_response_xdir.all_node_response * np.sign(Exp_response.all_node_response))
    all_node_response_Ezn = base_shear_ratio_zdir * (modal_response_zdir.all_node_response * np.sign(Ezn_response.all_node_response))
    all_node_response_Ezp = base_shear_ratio_zdir * (modal_response_zdir.all_node_response * np.sign(Ezp_response.all_node_response))
    all_member_response_Exn = base_shear_ratio_xdir * (modal_response_xdir.all_member_response * np.sign(Exn_response.all_member_response))
    all_member_response_Exp = base_shear_ratio_xdir * (modal_response_xdir.all_member_response * np.sign(Exp_response.all_member_response))
    all_member_response_Ezn = base_shear_ratio_zdir * (modal_response_zdir.all_member_response * np.sign(Ezn_response.all_member_response))
    all_member_response_Ezp = base_shear_ratio_zdir * (modal_response_zdir.all_member_response * np.sign(Ezp_response.all_member_response))

    responses = []
    for load_case in load_cases:
        ### Vertical Response ###
        all_node_response_vertical = load_case.D * (dead_load_response.all_node_response + self_weight_response.all_node_response) + load_case.L * live_load_response.all_node_response
        all_member_response_vertical = load_case.D * (dead_load_response.all_member_response + self_weight_response.all_member_response) + load_case.L * live_load_response.all_member_response
        if load_case.E:
            all_node_response_vertical += load_case.E_y * Ey_response.all_node_response
            all_member_response_vertical += load_case.E_y * Ey_response.all_member_response
        
        ### Horizontal Response ###
        if load_case.E_x_n:
            all_node_response_horizontal = load_case.E_x_n * all_node_response_Exn * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_xdir)
            all_member_response_horizontal = load_case.E_x_n * all_member_response_Exn * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_xdir)
        elif load_case.E_x_p:
            all_node_response_horizontal = load_case.E_x_p * all_node_response_Exp * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_xdir)
            all_member_response_horizontal = load_case.E_x_p * all_member_response_Exp * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_xdir)
        elif load_case.E_z_n:
            all_node_response_horizontal = load_case.E_z_n * all_node_response_Ezn * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_zdir)
            all_member_response_horizontal = load_case.E_z_n * all_member_response_Ezn * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_zdir)
        elif load_case.E_z_p:
            all_node_response_horizontal = load_case.E_z_p * all_node_response_Ezp * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_zdir)
            all_member_response_horizontal = load_case.E_z_p * all_member_response_Ezp * (load_case.horizontal_earthquake_force / horizontal_earthquake_force_zdir)
        
        ### Total Response ###
        total_response = NewResponse(structure)
        total_response.all_node_response = all_node_response_vertical
        total_response.all_member_response = all_member_response_vertical
        if load_case.E:
            total_response.all_node_response += all_node_response_horizontal
            total_response.all_member_response += all_member_response_horizontal
        total_response.to_response_dict()
        responses.append(total_response)

    t_end = time.time()
    print(f"\t\tused time for opensees.run_response_spectrum_analysis: {t_end - t_start:.3f} sec")
    # print(f"\t\tstatic analysis count: {static_analysis_counter}, response spectrum analysis count: {response_spectrum_analysis_counter}")

    return responses

