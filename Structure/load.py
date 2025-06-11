import numpy as np
from typing import List, Tuple, Dict
from Structure import earthquake
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
                 load_name: str, 
                 structure: Structure, 
                 load_case: LoadCase, 
                 horizontal_earthquake_forces: List[Dict[str, float]], 
                 horizontal_scale_factors: List[Dict[str, float]],
                 vertical_earthquake_force: float,
                 Fus: List[float], 
                 Fuv: float,
                 drift_case=False, 
                 col_strength_case=False):
        
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

        earthquake_force_xdir, earthquake_force_zdir = horizontal_earthquake_forces
        scale_factor_xdir, scale_factor_zdir = horizontal_scale_factors
        Fu_xdir, Fu_zdir = Fus
        if (self.E_x_n or self.E_x_p):
            horizontal_earthquake_force = earthquake_force_xdir
            horizontal_scale_factor = scale_factor_xdir
            Fu = Fu_xdir
            direction = 1
        else: 
            horizontal_earthquake_force = earthquake_force_zdir
            horizontal_scale_factor = scale_factor_zdir
            Fu = Fu_zdir
            direction = 2
        
        if drift_case:
            earthquake_force_type = "V_star"
        else:
            earthquake_force_index = np.array(list(horizontal_earthquake_force.values())).argmax()
            earthquake_force_type = list(horizontal_earthquake_force.keys())[earthquake_force_index]

        self.earthquake_force_type = earthquake_force_type
        self.horizontal_earthquake_force = horizontal_earthquake_force[self.earthquake_force_type]
        self.horizontal_scale_factor = horizontal_scale_factor[self.earthquake_force_type]
        self.vertical_earthquake_force = vertical_earthquake_force
        self.Fu = Fu
        self.Fuv = Fuv
        self.direction = direction

        if self.col_strength_case:
            self.update_E()

    def __str__(self):
        info = f"{self.load_name}, D: {self.D:.1f}, L: {self.L:.1f}, E:{self.E:.3f}, E_x_n: {self.E_x_n:.3f}, E_x_p: {self.E_x_p:.3f}, E_z_n: {self.E_z_n:.3f}, E_z_p: {self.E_z_p:.3f}, E_y: {self.E_y:.3f}, "
        info += f"if_drift_case: {self.drift_case}, if_col_strength_case: {self.col_strength_case}, "
        info += f"\n{self.earthquake_force_type}, horizontal_earthquake_force: {self.horizontal_earthquake_force:.3f}, vertical_earthquake_force: {self.vertical_earthquake_force:.3f}, "
        return info

    def update_E(self):
        """
        鋼構規範(LRFD) 13.3 設計放大地震力
        https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/0990807042-2.pdf
        """
        Fu = min(self.Fu, 2.5)
        self.E *= (1.4 * Fu)
        self.E_x_n *= (1.4 * Fu)
        self.E_x_p *= (1.4 * Fu)
        self.E_z_n *= (1.4 * Fu)
        self.E_z_p *= (1.4 * Fu)
        Fuv = min(self.Fuv, 2.5)
        self.E_y *= (1.4 * Fuv)

    def calculate_nodal_load(self, structure: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        * vertical load: self-weight + D + L + Ey
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
        # vertical_earthquake_force = self.E_y * self.horizontal_earthquake_force * np.array(list(structure.node_Ey_ratio_dict.values()))
        vertical_earthquake_force = self.E_y * self.vertical_earthquake_force * np.array(list(structure.node_Ey_ratio_dict.values()))
        
        vertical_load = dead_load + live_load + self_weight + vertical_earthquake_force
        # for i in range(len(dead_load)):
            # print(f"Node: {i}, vertical load, DL: {dead_load[i]}, LL: {live_load[i]}, weight: {self_weight[i]}, earthquake: {vertical_earthquake_force[i]}, total: {vertical_load[i]}")
        
        return vertical_load

    def _distribute_horizontal_load(self, structure: Structure) -> Tuple[np.ndarray, np.ndarray]:
        horizontal_x_load = np.zeros((structure.node_number))
        horizontal_z_load = np.zeros((structure.node_number))
        if self.E_x_n > 0:
            horizontal_x_load = -1 * self.E_x_n * self.horizontal_earthquake_force * np.array(list(structure.node_Exn_ratio_dict.values()))
        elif self.E_x_p > 0:
            horizontal_x_load = +1 * self.E_x_p * self.horizontal_earthquake_force * np.array(list(structure.node_Exp_ratio_dict.values()))
        elif self.E_z_n > 0:
            horizontal_z_load = -1 * self.E_z_n * self.horizontal_earthquake_force * np.array(list(structure.node_Ezn_ratio_dict.values()))
        elif self.E_z_p > 0:
            horizontal_z_load = +1 * self.E_z_p * self.horizontal_earthquake_force * np.array(list(structure.node_Ezp_ratio_dict.values()))
        return horizontal_x_load, horizontal_z_load
    

def get_nodal_dead_load(structure: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_x_load = np.zeros((structure.node_number))
    horizontal_z_load = np.zeros((structure.node_number))
    dead_load = DL * np.array(list(structure.node_area_dict.values()))
    return dead_load, horizontal_x_load, horizontal_z_load
    
def get_nodal_live_load(structure: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_x_load = np.zeros((structure.node_number))
    horizontal_z_load = np.zeros((structure.node_number))
    live_load = LL * np.array(list(structure.node_area_dict.values()))
    return live_load, horizontal_x_load, horizontal_z_load
    
def get_nodal_self_weight(structure: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_x_load = np.zeros((structure.node_number))
    horizontal_z_load = np.zeros((structure.node_number))
    self_weight = np.array(list(structure.node_self_weight_dict.values()))
    return self_weight, horizontal_x_load, horizontal_z_load

def get_nodal_vertical_earthquake_load(structure: Structure, vertical_earthquake_force: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_x_load = np.zeros((structure.node_number))
    horizontal_z_load = np.zeros((structure.node_number))
    vertical_earthquake_load = vertical_earthquake_force * np.array(list(structure.node_Ey_ratio_dict.values()))
    return vertical_earthquake_load, horizontal_x_load, horizontal_z_load

def get_nodal_horizontal_earthquake_load(structure: Structure, horizontal_earthquake_force: float, direction: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical_load = np.zeros((structure.node_number))
    horizontal_x_load = np.zeros((structure.node_number))
    horizontal_z_load = np.zeros((structure.node_number))
    if direction == "x_n":
        horizontal_x_load = -1 * horizontal_earthquake_force * np.array(list(structure.node_Exn_ratio_dict.values()))
    elif direction == "x_p":
        horizontal_x_load = +1 * horizontal_earthquake_force * np.array(list(structure.node_Exp_ratio_dict.values()))
    elif direction == "z_n":
        horizontal_z_load = -1 * horizontal_earthquake_force * np.array(list(structure.node_Ezn_ratio_dict.values()))
    elif direction == "z_p":
        horizontal_z_load = +1 * horizontal_earthquake_force * np.array(list(structure.node_Ezp_ratio_dict.values()))
    else:
        raise ValueError("Invalid direction for horizontal earthquake load.")

    return vertical_load, horizontal_x_load, horizontal_z_load


def get_load_cases(structure: Structure) -> List[NodalLoad]:
    horizontal_earthquake_loads, scale_factors, Fus, vertical_earthquake_load, Fuv = earthquake.design_earthquake_force(structure)

    load_case_1 = LoadCase(D=1.4, L=0.0, E=0)
    load_case_2 = LoadCase(D=1.2, L=1.6, E=0)
    load_case_3 = LoadCase(D=1.2, L=0.5, E=1, E_xz='x', E_np='n')
    load_case_4 = LoadCase(D=1.2, L=0.5, E=1, E_xz='x', E_np='p')
    load_case_5 = LoadCase(D=1.2, L=0.5, E=1, E_xz='z', E_np='n')
    load_case_6 = LoadCase(D=1.2, L=0.5, E=1, E_xz='z', E_np='p')

    load_cases = []
    load_cases.append(NodalLoad("load_1", structure, load_case_1, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv))
    load_cases.append(NodalLoad("load_2", structure, load_case_2, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv))

    load_cases.append(NodalLoad("load_3", structure, load_case_3, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv))
    load_cases.append(NodalLoad("load_4", structure, load_case_4, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv))
    load_cases.append(NodalLoad("load_5", structure, load_case_5, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv))
    load_cases.append(NodalLoad("load_6", structure, load_case_6, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv))

    load_cases.append(NodalLoad("drift_1", structure, load_case_3, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, drift_case=True))
    load_cases.append(NodalLoad("drift_2", structure, load_case_4, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, drift_case=True))
    load_cases.append(NodalLoad("drift_3", structure, load_case_5, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, drift_case=True))
    load_cases.append(NodalLoad("drift_4", structure, load_case_6, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, drift_case=True))
    
    load_cases.append(NodalLoad("col_strength_1", structure, load_case_3, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, col_strength_case=True))
    load_cases.append(NodalLoad("col_strength_2", structure, load_case_4, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, col_strength_case=True))
    load_cases.append(NodalLoad("col_strength_3", structure, load_case_5, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, col_strength_case=True))
    load_cases.append(NodalLoad("col_strength_4", structure, load_case_6, horizontal_earthquake_loads, scale_factors, vertical_earthquake_load, Fus, Fuv, col_strength_case=True))
    
    return load_cases
