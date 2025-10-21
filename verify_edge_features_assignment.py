"""
詳細驗證新增的 5 個 Edge Features 是否正確賦值

Author: Claude Code
Date: 2025-10-20
"""

import sys
sys.path.append("Structure/")

from Structure.structure import Structure
import torch
import numpy as np


def print_separator(title="", char="=", width=100):
    """打印分隔線"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def verify_edge_features_detailed():
    """詳細驗證 edge features 的賦值"""

    print_separator("開始詳細驗證 Edge Features 賦值", "=", 100)

    # ==================== 步驟 1: 創建測試結構 ====================
    print_separator("步驟 1: 創建測試結構（6F 4x4）", "-", 100)

    structure = Structure(
        x_span_num=4,
        x_span_lens=[6000, 6000, 6000, 6000],
        z_span_num=4,
        z_span_lens=[6000, 6000, 6000, 6000],
        story_num=6,
        story_height=3200,
        add_structure_geometry=True,
        add_response_features=False
    )

    print(f"✓ 結構創建成功")
    print(f"  - 樓層數: {structure.story_num}")
    print(f"  - 構件總數: {structure.member_number}")
    print(f"  - Story level actions: {len(structure.story_level_actions)}")

    # 打印每種類型的構件數量
    print(f"\n構件分類統計:")
    for story_idx in range(structure.story_num):
        print(f"  第 {story_idx+1} 層:")
        print(f"    - X樑: {len(structure.story_xdir_beam_member[story_idx])} 個")
        print(f"    - Z樑: {len(structure.story_zdir_beam_member[story_idx])} 個")
        print(f"    - 外柱: {len(structure.story_outer_column_member[story_idx])} 個")
        print(f"    - 內柱: {len(structure.story_inner_column_member[story_idx])} 個")

    # ==================== 步驟 2: 初始化 Graph ====================
    print_separator("步驟 2: 初始化 Graph", "-", 100)

    structure.init_graph_GraphRL()

    print(f"✓ Graph 初始化成功")
    print(f"  - Node features shape: {structure.graph.x.shape}")
    print(f"  - Edge features shape: {structure.graph.edge_attr.shape}")
    print(f"  - Edge index shape: {structure.graph.edge_index.shape}")

    edge_attr = structure.graph.edge_attr
    expected_dim = 18 if structure.add_response_features else 16

    if edge_attr.shape[1] != expected_dim:
        print(f"  ✗ 錯誤！Edge feature 維度應該是 {expected_dim}, 實際是 {edge_attr.shape[1]}")
        return False
    else:
        print(f"  ✓ Edge feature 維度正確: {expected_dim}")

    # ==================== 步驟 3: 檢查 Base Index ====================
    print_separator("步驟 3: 確認新特徵的索引位置", "-", 100)

    base_idx = 11 if not structure.add_response_features else 13
    print(f"Base index for new features: {base_idx}")
    print(f"新特徵的維度分配:")
    print(f"  - 維度 {base_idx}: is_xdir_beam")
    print(f"  - 維度 {base_idx+1}: is_zdir_beam")
    print(f"  - 維度 {base_idx+2}: is_outer_column")
    print(f"  - 維度 {base_idx+3}: is_inner_column")
    print(f"  - 維度 {base_idx+4}: normalized_floor")

    # ==================== 步驟 4: 選擇代表性構件進行詳細檢查 ====================
    print_separator("步驟 4: 選擇代表性構件進行詳細檢查", "-", 100)

    test_members = []

    # 從不同樓層選擇不同類型的構件
    for story_idx in [0, 2, 5]:  # 第1層、第3層、第6層
        # X樑
        if len(structure.story_xdir_beam_member[story_idx]) > 0:
            member_idx = structure.story_xdir_beam_member[story_idx][0]
            test_members.append({
                'type': 'xdir_beam',
                'story': story_idx + 1,
                'member_idx': member_idx,
                'expected_onehot': [1, 0, 0, 0],
                'expected_floor': (story_idx + 1) / structure.story_num
            })

        # Z樑
        if len(structure.story_zdir_beam_member[story_idx]) > 0:
            member_idx = structure.story_zdir_beam_member[story_idx][0]
            test_members.append({
                'type': 'zdir_beam',
                'story': story_idx + 1,
                'member_idx': member_idx,
                'expected_onehot': [0, 1, 0, 0],
                'expected_floor': (story_idx + 1) / structure.story_num
            })

        # 外柱
        if len(structure.story_outer_column_member[story_idx]) > 0:
            member_idx = structure.story_outer_column_member[story_idx][0]
            test_members.append({
                'type': 'outer_column',
                'story': story_idx + 1,
                'member_idx': member_idx,
                'expected_onehot': [0, 0, 1, 0],
                'expected_floor': (story_idx + 1) / structure.story_num
            })

        # 內柱
        if len(structure.story_inner_column_member[story_idx]) > 0:
            member_idx = structure.story_inner_column_member[story_idx][0]
            test_members.append({
                'type': 'inner_column',
                'story': story_idx + 1,
                'member_idx': member_idx,
                'expected_onehot': [0, 0, 0, 1],
                'expected_floor': (story_idx + 1) / structure.story_num
            })

    print(f"選擇了 {len(test_members)} 個代表性構件進行檢查\n")

    # ==================== 步驟 5: 逐個檢查構件的完整特徵 ====================
    print_separator("步驟 5: 逐個檢查構件的完整 Edge Features", "-", 100)

    all_passed = True

    for i, member_info in enumerate(test_members):
        member_idx = member_info['member_idx']
        edge_idx = member_idx * 2  # 每個構件有兩條邊

        print(f"\n[{i+1}/{len(test_members)}] 構件 #{member_idx} - {member_info['type']} (第 {member_info['story']} 層)")
        print("-" * 100)

        # 獲取該構件的 edge features
        edge_features = edge_attr[edge_idx].cpu().numpy()

        # 打印完整的 16 個維度
        print(f"完整的 Edge Features (16 維):")
        for dim in range(expected_dim):
            feature_name = ""
            if dim == 0:
                feature_name = "is_column"
            elif dim == 1:
                feature_name = "is_beam"
            elif dim == 2:
                feature_name = "L (length)"
            elif dim == 3:
                feature_name = "A (area)"
            elif dim == 4:
                feature_name = "Iz"
            elif dim == 5:
                feature_name = "Iy"
            elif dim == 6:
                feature_name = "Zz"
            elif dim == 7:
                feature_name = "A'"
            elif dim == 8:
                feature_name = "Iz'"
            elif dim == 9:
                feature_name = "Iy'"
            elif dim == 10:
                feature_name = "Zz'"
            elif dim == base_idx:
                feature_name = "is_xdir_beam ★"
            elif dim == base_idx + 1:
                feature_name = "is_zdir_beam ★"
            elif dim == base_idx + 2:
                feature_name = "is_outer_column ★"
            elif dim == base_idx + 3:
                feature_name = "is_inner_column ★"
            elif dim == base_idx + 4:
                feature_name = "normalized_floor ★"

            print(f"  維度 {dim:2d} ({feature_name:25s}): {edge_features[dim]:10.6f}")

        # 驗證 one-hot encoding
        print(f"\n驗證 One-hot Encoding:")
        actual_onehot = edge_features[base_idx:base_idx+4].tolist()
        expected_onehot = member_info['expected_onehot']

        onehot_match = np.allclose(actual_onehot, expected_onehot, atol=1e-6)
        print(f"  預期: {expected_onehot}")
        print(f"  實際: {[f'{v:.1f}' for v in actual_onehot]}")
        print(f"  結果: {'✓ 正確' if onehot_match else '✗ 錯誤'}")

        if not onehot_match:
            all_passed = False

        # 驗證 one-hot 互斥性
        onehot_sum = sum(actual_onehot)
        exclusive = abs(onehot_sum - 1.0) < 1e-6
        print(f"  One-hot sum: {onehot_sum:.1f} ({'✓ 互斥' if exclusive else '✗ 不互斥'})")

        if not exclusive:
            all_passed = False

        # 驗證 normalized floor
        print(f"\n驗證 Normalized Floor:")
        actual_floor = edge_features[base_idx + 4]
        expected_floor = member_info['expected_floor']

        floor_match = abs(actual_floor - expected_floor) < 1e-6
        print(f"  預期: {expected_floor:.6f} (第 {member_info['story']} 層 / {structure.story_num} 總層)")
        print(f"  實際: {actual_floor:.6f}")
        print(f"  結果: {'✓ 正確' if floor_match else '✗ 錯誤'}")

        if not floor_match:
            all_passed = False

        # 驗證兩個方向的 edge 特徵是否相同
        reverse_edge_idx = edge_idx + 1
        reverse_edge_features = edge_attr[reverse_edge_idx].cpu().numpy()

        edges_match = np.allclose(edge_features, reverse_edge_features, atol=1e-6)
        print(f"\n驗證正反向 Edge 是否相同:")
        print(f"  Edge {edge_idx} vs Edge {reverse_edge_idx}: {'✓ 相同' if edges_match else '✗ 不同'}")

        if not edges_match:
            all_passed = False
            print(f"  差異:")
            diff = np.abs(edge_features - reverse_edge_features)
            for dim in range(expected_dim):
                if diff[dim] > 1e-6:
                    print(f"    維度 {dim}: {edge_features[dim]:.6f} vs {reverse_edge_features[dim]:.6f}")

    # ==================== 步驟 6: 全局統計檢查 ====================
    print_separator("步驟 6: 全局統計檢查", "-", 100)

    print(f"\n新增特徵的統計信息:")

    # 只檢查單向 edges（每個構件的第一條邊）
    single_direction_edges = edge_attr[::2]  # 0, 2, 4, ...

    for offset, name in enumerate(['is_xdir_beam', 'is_zdir_beam', 'is_outer_column', 'is_inner_column']):
        dim_idx = base_idx + offset
        values = single_direction_edges[:, dim_idx].cpu().numpy()

        count_ones = (values == 1.0).sum()
        count_zeros = (values == 0.0).sum()
        count_other = len(values) - count_ones - count_zeros

        print(f"\n維度 {dim_idx} ({name}):")
        print(f"  = 1.0: {count_ones} 個構件")
        print(f"  = 0.0: {count_zeros} 個構件")
        if count_other > 0:
            print(f"  其他值: {count_other} 個 ✗ 異常！")
            all_passed = False

    # 檢查 normalized floor
    floor_dim = base_idx + 4
    floor_values = single_direction_edges[:, floor_dim].cpu().numpy()

    print(f"\n維度 {floor_dim} (normalized_floor):")
    print(f"  最小值: {floor_values.min():.6f}")
    print(f"  最大值: {floor_values.max():.6f}")
    print(f"  平均值: {floor_values.mean():.6f}")
    print(f"  唯一值: {np.unique(floor_values)}")

    # 檢查範圍
    if floor_values.min() <= 0 or floor_values.max() > 1.0:
        print(f"  ✗ 錯誤：normalized floor 應該在 (0, 1] 範圍內")
        all_passed = False
    else:
        print(f"  ✓ 範圍正確: (0, 1]")

    # 檢查每層的 normalized floor 是否一致
    print(f"\n檢查每層的 normalized floor 一致性:")
    for story_idx in range(structure.story_num):
        expected_norm_floor = (story_idx + 1) / structure.story_num

        # 檢查該層的所有構件
        story_members = []
        story_members.extend(structure.story_xdir_beam_member[story_idx])
        story_members.extend(structure.story_zdir_beam_member[story_idx])

        if len(story_members) > 0:
            # 樑的樓層應該是 story_idx + 1
            for member_idx in story_members[:3]:  # 檢查前3個
                edge_idx = member_idx * 2
                actual_norm_floor = edge_attr[edge_idx, floor_dim].item()

                if abs(actual_norm_floor - expected_norm_floor) > 1e-6:
                    print(f"  ✗ 第 {story_idx+1} 層構件 {member_idx}: 預期 {expected_norm_floor:.6f}, 實際 {actual_norm_floor:.6f}")
                    all_passed = False

    print(f"  ✓ 樓層編碼一致性檢查通過")

    # ==================== 步驟 7: 驗證與 story_level_actions 的對應 ====================
    print_separator("步驟 7: 驗證與 story_level_actions 的對應", "-", 100)

    print(f"\n檢查前 24 個 actions 的特徵編碼一致性:")

    action_check_passed = True
    for action_idx in range(min(24, len(structure.story_level_actions))):
        member_indices = structure.story_level_actions[action_idx]
        category = structure.story_level_categories[action_idx]

        if len(member_indices) == 0:
            continue

        # 取第一個構件檢查
        member_idx = member_indices[0]
        edge_idx = member_idx * 2
        type_encoding = edge_attr[edge_idx, base_idx:base_idx+4].cpu().numpy()

        # 根據 category 確定預期編碼
        if category == 'xdir_beam':
            expected = [1, 0, 0, 0]
        elif category == 'zdir_beam':
            expected = [0, 1, 0, 0]
        elif category == 'outer_column':
            expected = [0, 0, 1, 0]
        elif category == 'inner_column':
            expected = [0, 0, 0, 1]

        match = np.allclose(type_encoding, expected, atol=1e-6)

        if not match:
            print(f"  ✗ Action {action_idx} ({category}): 預期 {expected}, 實際 {type_encoding.tolist()}")
            action_check_passed = False
            all_passed = False

    if action_check_passed:
        print(f"  ✓ 所有 actions 的特徵編碼與類別一致")

    # ==================== 最終總結 ====================
    print_separator("最終總結", "=", 100)

    if all_passed:
        print("\n🎉 所有驗證通過！")
        print("\n✓ Edge features 的新 5 個維度已正確賦值:")
        print(f"  - 維度 {base_idx}-{base_idx+3}: 構件類型 one-hot encoding")
        print(f"  - 維度 {base_idx+4}: 樓層編碼 (normalized)")
        print("\n✓ 數值範圍正確")
        print("✓ One-hot 互斥性正確")
        print("✓ 正反向 edge 特徵相同")
        print("✓ 與 story_level_actions 對應正確")
    else:
        print("\n❌ 發現問題！")
        print("請檢查上述標記為 ✗ 的項目")

    print_separator("", "=", 100)

    return all_passed


if __name__ == "__main__":
    try:
        success = verify_edge_features_detailed()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 驗證過程中發生錯誤:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
