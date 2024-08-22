import numpy as np
from typing import List, Tuple, Dict
from Structure.structure import Structure


DL = 1.0    # kN/m2
LL = 2.4    # kN/m2
STEEL_DENSITY = 77  # kN/m3


class LoadCase:
    def __init__(self, D=0.0, L=0.0, E=0, E_xz='x', E_np='n'):
        self.D = D
        self.L = L
        self.E = E
        self.E_x_n = 0
        self.E_x_p = 0
        self.E_z_n = 0
        self.E_z_p = 0
        self.E_y = 0

        if E_xz == 'x':
            if E_np == 'n':
                self.E_x_n = E
            else:
                self.E_x_p = E
        else:
            if E_np == 'n':
                self.E_z_n = E
            else:
                self.E_z_p = E

        self.E_y = 0.3 * max(self.E_x_n, self.E_x_p, self.E_z_n, self.E_z_p)




class NodalLoad:
    def __init__(self, 
                 load_name, 
                 structure: Structure, 
                 load_case: LoadCase, 
                 earthquake_loads: List[Dict[str, float]], 
                 drift_case=False, 
                 col_strength_case=False, 
                 Fus: List[float] = None):
        
        self.load_name = load_name
        self.D = load_case.D
        self.L = load_case.L
        self.E = load_case.E
        self.E_x_n = load_case.E_x_n
        self.E_x_p = load_case.E_x_p
        self.E_z_n = load_case.E_z_n
        self.E_z_p = load_case.E_z_p
        self.E_y = load_case.E_y

        self.structure = structure
        self.drift_case = drift_case
        self.col_strength_case = col_strength_case

        if sum(self.structure.x_span_lens) <= sum(self.structure.z_span_lens):
            earthquake_load_xdir = earthquake_loads[0]  # 1st period --> parallel to shorter side
            earthquake_load_zdir = earthquake_loads[1]  # 2nd period --> parallel to longer side
        else:
            earthquake_load_zdir = earthquake_loads[0]  # 1st period --> parallel to shorter side
            earthquake_load_xdir = earthquake_loads[1]  # 2nd period --> parallel to longer side

        if (self.E_x_n or self.E_x_p):
            earthquake_load = earthquake_load_xdir
        else: 
            earthquake_load = earthquake_load_zdir
        
        if drift_case:
            self.earthquake_load = earthquake_load["V_star"]
        else:
            self.earthquake_load = max(earthquake_load.values())

        if col_strength_case:
            Fu = Fus[1] if (self.E_x_n or self.E_x_p) else Fus[0]
            self.update_E(Fu)

    def __str__(self):
        info = f"D: {self.load_case.D:.1f}, L: {self.load_case.L:.1f}, E_x_n: {self.load_case.E_x_n}, E_x_p: {self.load_case.E_x_p}, E_z_n: {self.load_case.E_z_n}, E_z_p: {self.load_case.E_z_p}, E_y: {self.load_case.E_y}, "
        info += f"if_drift_case: {self.drift_case}, earthquake_load: {self.earthquake_load:.2f}"
        return info

    def update_E(self, Fu):
        """
        鋼構規範(LRFD) 13.3 設計放大地震力
        https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf
        """
        Fu = min(Fu, 2.5)
        self.E_x_n *= (1.4 * Fu)
        self.E_x_p *= (1.4 * Fu)
        self.E_z_n *= (1.4 * Fu)
        self.E_z_p *= (1.4 * Fu)
        self.E_y *= (1.4 * Fu)

    def calculate_nodal_load(self, structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        * vertical load: self-weight, D, L, Ey
        * horizontal load: Ex or Ez
        
        return: nodal_vertical_load, nodal_horizontal_x_load, nodal_horizontal_z_load
        """
        nodal_vertical_load = self._distribute_vertical_load(structure)
        nodal_horizontal_x_load, nodal_horizontal_z_load = self._distribute_horizontal_load(structure)
        return nodal_vertical_load, nodal_horizontal_x_load, nodal_horizontal_z_load  # kN

    def _distribute_vertical_load(self, structure: Structure) -> np.ndarray:
        dead_load = self.D * DL * np.array(list(structure.node_area_dict.values()))
        live_load = self.L * LL * np.array(list(structure.node_area_dict.values()))
        self_weight = self.D * np.array(list(structure.node_self_weight_dict.values()))
        vertical_eathquake_force = self.E_y * self.earthquake_load * np.array(list(structure.node_Ey_ratio_dict.values()))
        vertical_load = dead_load + live_load + self_weight + vertical_eathquake_force
        # for i in range(len(dead_load)):
            # print(f"Node: {i}, vertical load, DL: {dead_load[i]}, LL: {live_load[i]}, weight: {self_weight[i]}, earthquake: {vertical_eathquake_force[i]}, total: {vertical_load[i]}")
        return vertical_load

    def _distribute_horizontal_load(self, structure: Structure) -> Tuple[np.ndarray, np.ndarray]:
        horizontal_x_load = np.zeros((structure.node_number))
        horizontal_z_load = np.zeros((structure.node_number))
        if self.E_x_n > 0:
            horizontal_x_load = -1 * self.E_x_n * self.earthquake_load * np.array(list(structure.node_Exn_ratio_dict.values()))
        elif self.E_x_p > 0:
            horizontal_x_load = +1 * self.E_x_p * self.earthquake_load * np.array(list(structure.node_Exp_ratio_dict.values()))
        elif self.E_z_n > 0:
            horizontal_z_load = -1 * self.E_z_n * self.earthquake_load * np.array(list(structure.node_Ezn_ratio_dict.values()))
        elif self.E_z_p > 0:
            horizontal_z_load = +1 * self.E_z_p * self.earthquake_load * np.array(list(structure.node_Ezp_ratio_dict.values()))
        return horizontal_x_load, horizontal_z_load




def get_load_cases(structure: Structure, earthquake_loads: List[Dict[str, float]], Fus: List[float]) -> List[NodalLoad]:
    load_case_1 = LoadCase(D=1.4, L=0.0, E=0)
    load_case_2 = LoadCase(D=1.2, L=1.6, E=0)
    load_case_3 = LoadCase(D=1.2, L=0.5, E=1, E_xz='x', E_np='n')
    load_case_4 = LoadCase(D=1.2, L=0.5, E=1, E_xz='x', E_np='p')
    load_case_5 = LoadCase(D=1.2, L=0.5, E=1, E_xz='z', E_np='n')
    load_case_6 = LoadCase(D=1.2, L=0.5, E=1, E_xz='z', E_np='p')

    load_cases = []
    load_cases.append(NodalLoad("load_1", structure, load_case_1, earthquake_loads))
    load_cases.append(NodalLoad("load_2", structure, load_case_2, earthquake_loads))

    load_cases.append(NodalLoad("load_3", structure, load_case_3, earthquake_loads))
    load_cases.append(NodalLoad("load_4", structure, load_case_4, earthquake_loads))
    load_cases.append(NodalLoad("load_5", structure, load_case_5, earthquake_loads))
    load_cases.append(NodalLoad("load_6", structure, load_case_6, earthquake_loads))

    load_cases.append(NodalLoad("drift_1", structure, load_case_3, earthquake_loads, drift_case=True))
    load_cases.append(NodalLoad("drift_2", structure, load_case_4, earthquake_loads, drift_case=True))
    load_cases.append(NodalLoad("drift_3", structure, load_case_5, earthquake_loads, drift_case=True))
    load_cases.append(NodalLoad("drift_4", structure, load_case_6, earthquake_loads, drift_case=True))
    
    load_cases.append(NodalLoad("col_strength_1", structure, load_case_3, earthquake_loads, col_strength_case=True, Fus=Fus))
    load_cases.append(NodalLoad("col_strength_2", structure, load_case_4, earthquake_loads, col_strength_case=True, Fus=Fus))
    load_cases.append(NodalLoad("col_strength_3", structure, load_case_5, earthquake_loads, col_strength_case=True, Fus=Fus))
    load_cases.append(NodalLoad("col_strength_4", structure, load_case_6, earthquake_loads, col_strength_case=True, Fus=Fus))
    
    return load_cases
