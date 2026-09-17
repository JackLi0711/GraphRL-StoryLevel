"""Canonical key map for train args.

訓練腳本的參數名稱在不同世代被改過。同一個概念會有多個 key 名稱，
若不合併，兩個設定相同的 Run 會在 diff 裡顯示成大量假差異。

這份對照表是刻意攤開來給人看的 —— 合併是判斷，不是事實，你可以否決任何一條。
每一條後面的數字是該別名/正名在 81 個 Run 中的出現次數：兩者互補到 81，
即「舊世代用 A、新世代用 B、沒有任何 Run 同時擁有兩者」，這是合併的依據。

要否決某一條合併：把它從 ALIASES 刪掉即可，索引會把它當成兩個獨立欄位。
"""

# alias (舊名) -> canonical (正名)
ALIASES = {
    # DQN 世代 -> PPO/OC 世代，各組皆 15 + 66 = 81
    "add_response_features":         "add_response_feature",
    "add_structure_geometry":        "add_geometry_feature",
    "check_acceleration":            "check_acc",
    "check_displacement":            "check_disp",
    "do_nonlinear_dynamic_analysis": "do_nda",
    "ground_motion_dir":             "gm_dir",
    "ground_motion_number":          "gm_num",
    "num_layers":                    "layer_num",

    # 訓練回合數：四個名稱同一個概念
    # num_epoch(15) + num_episode(13) = 28 = 全部 DQN
    # train_episode_num(52) + episode_num(1) = 53 = 全部 PPO+OC
    "num_epoch":    "train_episode_num",
    "num_episode":  "train_episode_num",
    "episode_num":  "train_episode_num",
}

# 刻意「不」合併的組合（避免日後有人以為是漏掉的）：
# - entropy_weight 與 entropy_weight_interval：前者是純量，後者是 [起始, 結束] 區間搭配
#   entropy_weight_schedule，是不同的參數化方式而非改名。PPO 17 + 6 = 23，雖然互補但語意不同。

# 這些 key 是紀錄用途，不是超參數；不參與 diff 比對。
NON_PARAMETER_KEYS = {
    "ckpt_dir",   # 訓練當下的路徑，多半已過時，見 ADR-0002
    "suffix",     # 由資料夾名稱衍生
    "comment",    # 自由文字，單獨呈現
}


def canonicalize(args: dict) -> dict:
    """Rename aliased keys to their canonical form."""
    out = {}
    for k, v in args.items():
        out[ALIASES.get(k, k)] = v
    return out
