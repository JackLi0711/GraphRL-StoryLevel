import numpy as np
import openseespy.opensees as ops

from pathlib import Path
from logging import Logger

from Structure.sections import *
from Structure.structure import Structure
from Structure.load import NodalLoad


mm = 1  # length
kN = 1  # force
m = 1e+3 * mm
N = 1e-3 * kN
Pa = N / m**2

E = 200 * 1e+9 * Pa  # Young's modulud, GPa
Nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + Nu))  # Shear modulus, GPa


"""
For OpenSees, X-Y plane: horizontal plane, Z-dir: vertical direction
# order of dof values: Ux, Uy, Uz,  Rx, Ry, Rz
# nodal mass: self-wieght, self-wieght, 0,  0, 0, moment of inertia
# nodal load: load_x, load_z, load_y, 0, 0, 0
load_x, load_z: horizontal seismic load
load_y: self-weight + dead load + live load + vertical seismic load

* rigid diaphragm: ops.equalDOF() vs. ops.rigidDiaphragm() --> 兩個都可

modal analysis: eigen analysis --> 輸出 natural period 文字檔，再讀取去抓
"""


def run_static_analysis(structure: Structure, load_case: NodalLoad, analysis_dir: Path, logger: Logger=None):
    # remove existing model
    ops.wipe()

    # 4.1 model command (https://openseespydoc.readthedocs.io/en/latest/src/model.html)
    ops.model('basic', '-ndm',3, '-ndf',6)

    # 4.3 node command (https://openseespydoc.readthedocs.io/en/latest/src/node.html)
    for i in range(structure.node_number):
        nodeTag = i+1
        node_name = f"N{nodeTag}"

        # node_coord_dict --> key: node_name(N1, N2, ...), value: tuple(x, y, z), unit: mm
        x, y, z = structure.node_coord_dict[node_name]
        crds = [x, z, y]

        # node_translational_mass_dict --> key: node_name(N1, N2, ...), value: translational mass, unit: kN / (mm/s^2)
        #translational_mass = structure.node_translational_mass_dict[node_name]
        translational_mass = 0
        # node_inertia_dict --> key: node_name(N1, N2, ...), value: inertia distributed to node (Rx, Ry, Rz), unit: kN / (mm/s^2) * mm^2
        #moment_inertia = structure.node_inertia_dict[node_name]
        moment_inertia = 0
        mass = [translational_mass for _ in range(3)] + [moment_inertia for _ in range(3)]

        ops.node(nodeTag, *crds, '-mass',*mass)

    # 4.4 fix command (https://openseespydoc.readthedocs.io/en/latest/src/fix.html)
    # node_dof_dict --> key: node_name(N1, N2, ...), value: dof(free:0, fixed:1)
    for node_name, dof in structure.node_dof_dict.items():
        if dof == 1:
            nodeTag = int(node_name[1:])
            ops.fix(nodeTag, *[1,1,1,1,1,1])

    # 4.18 geomTransf commands (https://openseespydoc.readthedocs.io/en/latest/src/geomTransf.html)
    XdirBeamTransfTag = 1
    ZdirBeamTransfTag = 2
    ColumnTransfTag = 3
    ops.geomTransf('Linear', XdirBeamTransfTag, *[0,0,1])
    ops.geomTransf('Linear', ZdirBeamTransfTag, *[0,0,1])
    ops.geomTransf('PDelta', ColumnTransfTag, *[1,0,0])

    # 4.2 element command (https://openseespydoc.readthedocs.io/en/latest/src/element.html)
    for i in range(structure.member_number):
        eleTag = i+1
        member_name = f"E{eleTag}"
        node1_index, node2_index = structure.member_to_nodeIndex_dict[member_name][:2]
        nodeTag_1 = node1_index + 1
        nodeTag_2 = node2_index + 1

        Area = structure.member_A_dict[member_name]  # mm^2

        member_sections = column_sections if structure.member_category_dict[member_name] == 'y' else beam_sections
        sections_index = structure.member_section_dict[member_name]
        section = member_sections[sections_index]
        #J = section["J(cm4)"] * 1e+4  # torsional constant, mm^4
        Jxx = (section["I_y(cm4)"] + section["I_z(cm4)"]) * 1e+4  # torsional moment of inertia, mm^4
        Iy = section["I_y(cm4)"] * 1e+4  # moment of inertia, mm^4
        Iz = section["I_z(cm4)"] * 1e+4  # moment of inertia, mm^4

        if structure.member_category_dict[member_name] == 'x':
            transfTag = XdirBeamTransfTag
        elif structure.member_category_dict[member_name] == 'z':
            transfTag = ZdirBeamTransfTag
        else:
            transfTag = ColumnTransfTag
            
        ops.element('elasticBeamColumn', eleTag, nodeTag_1, nodeTag_2, Area, E, G, Jxx, Iy, Iz, transfTag)

    # master node
    for story, y in enumerate(structure.y_grid):
        if story > 0:
            nodeTag = story + structure.node_number
            master_name = f"M{story}F"

            crds = [structure.master_x, structure.master_z, y]
            
            translational_mass = 0
            moment_inertia = 0
            for slave_name in structure.slave_node_dict[master_name]:
                translational_mass += structure.node_translational_mass_dict[slave_name]  # kN / (mm/s^2)
                moment_inertia += structure.node_inertia_dict[slave_name]  # kN / (mm/s^2) * mm^2
            mass = [translational_mass, translational_mass, 0] + [0, 0, moment_inertia]
            ops.node(nodeTag, *crds, '-mass',*mass)
            ops.fix(nodeTag, *[0,0,1,1,1,0])

            for slave_name in structure.slave_node_dict[master_name]:
                slaveTag = int(slave_name[1:])
                # 4.5.1 equalDOF command (https://openseespydoc.readthedocs.io/en/latest/src/equalDOF.html)
                ops.equalDOF(nodeTag, slaveTag, *[1,2,3,4,5,6])
            
            # rigidDiaphragm(perpDirn, NodeTag, *cNodeTags)

    # 4.7 timeSeries command (https://openseespydoc.readthedocs.io/en/latest/src/timeSeries.html)
    tsTag = 1
    ops.timeSeries('Linear', tsTag)  # 'Constant', 'Linear', 'Rectangular', 'Path'

    # 4.8 pattern command (https://openseespydoc.readthedocs.io/en/latest/src/pattern.html)
    patternTag = 1
    ops.pattern('Plain', patternTag, tsTag)

    # 4.8.1 load command (https://openseespydoc.readthedocs.io/en/latest/src/load.html)
    nodal_loads = load_case.calculate_nodal_load(structure)
    for i in range(structure.node_number):
        nodeTag = i+1
        load_y, load_x, load_z = nodal_loads[0][i], nodal_loads[1][i], nodal_loads[2][i]  # kN
        load_y *= -1
        ops.load(nodeTag, load_x, load_z, load_y, 0, 0, 0)

    # 5.1 constraints command (https://openseespydoc.readthedocs.io/en/latest/src/constraints.html)
    ops.constraints('Plain') # plain constraint handler only enforce fix command and equalDOF command

    # 5.2 numberer command (https://openseespydoc.readthedocs.io/en/latest/src/numberer.html)
    ops.numberer('RCM')  # 'Plain', 'RCM', 'AMD'

    # 5.3 system command (https://openseespydoc.readthedocs.io/en/latest/src/system.html)
    ops.system('ProfileSPD')  # 'BandGen', 'BandSPD', 'ProfileSPD'

    # 5.6 integrator command (https://openseespydoc.readthedocs.io/en/latest/src/integrator.html)
    ops.integrator('LoadControl', 1.0)

    # 5.5 algorithm command (https://openseespydoc.readthedocs.io/en/latest/src/algorithm.html)
    ops.algorithm('Linear')  # 'Linear', 'Newton', 'NewtonLineSearch', 'ModifiedNewton'

    # 5.7 analysis command (https://openseespydoc.readthedocs.io/en/latest/src/analysis.html)
    ops.analysis('Static')

    # 5.9 analyze command (https://openseespydoc.readthedocs.io/en/latest/src/analyze.html)
    ops.analyze(1)

    # 6.15 nodeDisp command (https://openseespydoc.readthedocs.io/en/latest/src/nodeDisp.html)
    disp = np.array([ops.nodeDisp(i+1) for i in range(structure.node_number)])

    # 6.2 basicForce command (https://openseespydoc.readthedocs.io/en/latest/src/basicForce.html)
    force = np.array([ops.basicForce(i+1) for i in range(structure.member_number)])  # 0: axial force, 1: moment at i end, 2: moment at j end
    
    # 6.31.1 nodeRecorder command (https://openseespydoc.readthedocs.io/en/latest/src/nodeRecorder.html)
    file_name = analysis_dir / f"{load_case.name}.txt"
    ops.recorder('Node', '-file',file_name, '-time', '-node',*[i+1 for i in range(structure.node_number)] ,'-dof',1,2,3, 'disp')
    
    return disp, force