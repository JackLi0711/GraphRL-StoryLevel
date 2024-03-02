from typing import Tuple, List, Dict
from Structure.structure import Structure


# 鋼結構
ay = 1.0  # 2.9 鋼構造或鋼骨鋼筋混凝土構造採極限設計法者，ay值可取與地震力之載重因子相同，即ay為1.0
I = 1.0   # 2.8 第四類建築物其他一般建築物，I = 1.0
R = 4.8   # 表1-3特殊抗彎矩構架：鋼造

# 台北三區
Ra = 1 + (R - 1) / 2.0  # 2.9 臺北盆地結構系統容許韌性容量Ra與韌性容量R值間之關係
T0D = 1.05              # 表2.6
T0M = 1.05              # 表2.6
SDS = 0.6               # 表2.6
SMS = 0.8               # 表2.6


# 2.6 鋼構造建築物之基本振動週期 T(sec) = 0.085 * hn ^ (3/4), hn為公尺
# 基本振動週期得用其他結構力學方法計算。但所得之 T 值不得大於前述經驗公式週期之 1.4 倍
def _T(height: float, mode_period: float) -> float:
    T_experience = 0.085 * (height ** (3/4))
    T = min(1.4 * T_experience, mode_period)
    return T

# 表2-7(a)
def _SaD(T: float) -> float:
    if T <= 0.2 * T0D:
        SaD = SDS * (0.4 + 3 * T / T0D)
    elif T > 0.2 * T0D and T <= T0D:
        SaD = SDS
    elif T > T0D and T <= 2.5 * T0D:
        SaD = SDS * T0D / T
    elif T > 2.5 * T0D:
        SaD = 0.4 * SDS
    return SaD

# 表2-7(b)
def _SaM(T: float) -> float:
    if T <= 0.2 * T0M:
        SaM = SMS * (0.4 + 3 * T / T0M)
    elif T > 0.2 * T0M and T <= T0M:
        SaM = SMS
    elif T > T0M and T <= 2.5 * T0M:
        SaM = SMS * T0M / T
    elif T > 2.5 * T0M:
        SaM = 0.4 * SMS
    return SaM

# 2.9 結構系統地震力折減係數
def _Fu(T: float) -> float:
    if T >= T0D:
        Fu = Ra
    elif T >= 0.6 * T0D and T <= T0D:
        Fu = (2 * Ra - 1) ** 0.5 + (Ra - (2 * Ra - 1) ** 0.5) * ((T - 0.6 * T0D) / (0.4 * T0D))
    elif T >= 0.2 * T0D and T <= 0.6 * T0D:
        Fu = (2 * Ra - 1) ** 0.5
    elif T <= 0.2 * T0D:
        Fu = (2 * Ra - 1) ** 0.5 + ((2 * Ra - 1) ** 0.5 - 1) * ((T - 0.2 * T0D) / 0.2 * T0D)
    return Fu

# 2.10.2 避免最大考量地震崩塌之設計地震力
def _FuM(T: float) -> float:
    if T >= T0D:
        FuM = R
    elif T >= 0.6 * T0D and T <= T0D:
        FuM = (2 * R - 1) ** 0.5 + (R - (2 * R - 1) ** 0.5) * ((T - 0.6 * T0D) / (0.4 * T0D))
    elif T >= 0.2 * T0D and T <= 0.6 * T0D:
        FuM = (2 * R - 1) ** 0.5
    elif T <= 0.2 * T0D:
        FuM = (2 * R - 1) ** 0.5 + ((2 * R - 1) ** 0.5 - 1) * ((T - 0.2 * T0D) / 0.2 * T0D)
    return FuM


# 2.2 最小設計水平總橫力
def _SaD_div_Fu_modified(SaD_div_Fu: float) -> float:
    if SaD_div_Fu <= 0.3:
        SaD_div_Fu_modified = SaD_div_Fu
    elif SaD_div_Fu > 0.3 and SaD_div_Fu < 0.8:
        SaD_div_Fu_modified = 0.52 * SaD_div_Fu + 0.144
    elif SaD_div_Fu >= 0.8:
        SaD_div_Fu_modified = 0.70 * SaD_div_Fu
    return SaD_div_Fu_modified

# 2.10.2 避免最大考量地震崩塌之設計地震力
def _SaM_div_FuM_modified(SaM_div_FuM: float) -> float:
    if SaM_div_FuM <= 0.3:
        SaM_div_FuM_modified = SaM_div_FuM
    elif SaM_div_FuM > 0.3 and SaM_div_FuM < 0.8:
        SaM_div_FuM_modified = 0.52 * SaM_div_FuM + 0.144
    elif SaM_div_FuM >= 0.8:
        SaM_div_FuM_modified = 0.70 * SaM_div_FuM
    return SaM_div_FuM_modified


# 2.2 最小設計水平總橫力
def minimum_design_horizontal_force(T: float, W: float) -> float:
    SaD = _SaD(T)
    Fu = _Fu(T)
    SaD_div_Fu = SaD / Fu
    SaD_div_Fu_modified = _SaD_div_Fu_modified(SaD_div_Fu)
    V = I / (1.4 * ay) * SaD_div_Fu_modified * W
    return V, Fu

# 2.10.1 避免中小度地震降伏之設計地震力
def minimum_design_force_avoid_yield_at_small_earthquake(T: float, W: float) -> float:
    SaD = _SaD(T)
    Fu = _Fu(T)
    SaD_div_Fu = SaD / Fu
    SaD_div_Fu_modified = _SaD_div_Fu_modified(SaD_div_Fu)
    V_star = I * Fu / (3.5 * ay) * SaD_div_Fu_modified * W
    return V_star

# 2.10.2 避免最大考量地震崩塌之設計地震力
def minimum_design_force_avoid_collapse_at_big_earthquake(T: float, W: float) -> float:
    SaM = _SaM(T)
    FuM = _FuM(T)
    SaM_div_FuM = SaM / FuM
    SaM_div_FuM_modified = _SaM_div_FuM_modified(SaM_div_FuM)
    V_M = I / (1.4 * ay) * SaM_div_FuM_modified * W
    return V_M


# select the highest among those 3 design earthquake force
def design_earthquake_force(structure: Structure, first_mode_period: float, second_mode_period: float) -> Tuple[List[Dict[str, float]], List[float]]:
    T1 = _T(structure.height, first_mode_period)
    T2 = _T(structure.height, second_mode_period)
    W = sum(structure.node_dead_load_self_weight_dict.values())  # kN

    V1, Fu1 = minimum_design_horizontal_force(T1, W)
    V_star1 = minimum_design_force_avoid_yield_at_small_earthquake(T1, W)
    V_M1 = minimum_design_force_avoid_collapse_at_big_earthquake(T1, W)
    earthquake_force1 = {"V": V1, "V_star": V_star1, "V_M": V_M1}

    V2, Fu2 = minimum_design_horizontal_force(T2, W)
    V_star2 = minimum_design_force_avoid_yield_at_small_earthquake(T2, W)
    V_M2 = minimum_design_force_avoid_collapse_at_big_earthquake(T2, W)
    earthquake_force2 = {"V": V2, "V_star": V_star2, "V_M": V_M2}

    return [earthquake_force1, earthquake_force2], [Fu1, Fu2]

