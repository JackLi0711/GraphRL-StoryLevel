"""
測試新添加的構件類型和樓層編碼特徵

Author: Claude Code
Date: 2025-10-19
"""

import sys
sys.path.append("Structure/")

from Structure.structure import Structure
import torch

def test_new_edge_features():
    """測試新的 edge features 是否正確編碼"""

    print("=" * 80)
    print("測試新的 Edge Features")
    print("=" * 80)

    # 創建一個測試結構 (6F 4x4)
    print("\n創建 6F 4x4 測試結構...")
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
    print(f"  - Story members: {len(structure.story_level_actions)}")

    # 初始化 graph
    print(f"\n初始化 Graph...")
    structure.init_graph_GraphRL()
    print(f"✓ Graph 初始化成功")

    # 檢查 graph 維度
    print(f"\n檢查 Graph 維度:")
    print(f"  - Node features shape: {structure.graph.x.shape}")
    print(f"  - Edge features shape: {structure.graph.edge_attr.shape}")
    print(f"  - 預期 edge feature 維度: 16 (不含 response features)")

    if structure.graph.edge_attr.shape[1] != 16:
        print(f"  ❌ 錯誤! Edge feature 維度應該是 16, 但得到 {structure.graph.edge_attr.shape[1]}")
        return False
    else:
        print(f"  ✓ Edge feature 維度正確: 16")

    # 檢查構件類型編碼
    print(f"\n檢查構件類型編碼 (one-hot):")
    print(f"  維度 11-14: [is_xdir_beam, is_zdir_beam, is_outer_column, is_inner_column]")

    # 檢查幾個構件
    test_members = []

    # 找一個X樑
    if len(structure.story_xdir_beam_member[0]) > 0:
        xbeam_idx = structure.story_xdir_beam_member[0][0]
        test_members.append(('X樑', xbeam_idx, [1, 0, 0, 0]))

    # 找一個Z樑
    if len(structure.story_zdir_beam_member[0]) > 0:
        zbeam_idx = structure.story_zdir_beam_member[0][0]
        test_members.append(('Z樑', zbeam_idx, [0, 1, 0, 0]))

    # 找一個外柱
    if len(structure.story_outer_column_member[0]) > 0:
        outer_col_idx = structure.story_outer_column_member[0][0]
        test_members.append(('外柱', outer_col_idx, [0, 0, 1, 0]))

    # 找一個內柱
    if len(structure.story_inner_column_member[0]) > 0:
        inner_col_idx = structure.story_inner_column_member[0][0]
        test_members.append(('內柱', inner_col_idx, [0, 0, 0, 1]))

    all_correct = True
    for member_type, member_idx, expected_encoding in test_members:
        edge_idx = member_idx * 2
        actual_encoding = structure.graph.edge_attr[edge_idx, 11:15].tolist()

        print(f"  - {member_type} (member {member_idx}): {actual_encoding}", end="")
        if actual_encoding == expected_encoding:
            print(" ✓")
        else:
            print(f" ❌ (預期: {expected_encoding})")
            all_correct = False

    # 檢查樓層編碼
    print(f"\n檢查樓層編碼 (normalized floor):")
    print(f"  維度 15: floor_number / story_num")

    # 檢查不同樓層的構件
    for story_idx in [0, structure.story_num-1]:  # 第1層和最高層
        if len(structure.story_outer_column_member[story_idx]) > 0:
            member_idx = structure.story_outer_column_member[story_idx][0]
            edge_idx = member_idx * 2
            normalized_floor = structure.graph.edge_attr[edge_idx, 15].item()
            expected_floor = (story_idx + 1) / structure.story_num

            print(f"  - 第 {story_idx+1} 層外柱 (member {member_idx}): {normalized_floor:.4f}", end="")
            if abs(normalized_floor - expected_floor) < 1e-5:
                print(f" ✓ (預期: {expected_floor:.4f})")
            else:
                print(f" ❌ (預期: {expected_floor:.4f})")
                all_correct = False

    print(f"\n" + "=" * 80)
    if all_correct:
        print("✓ 所有測試通過!")
        print("=" * 80)
        return True
    else:
        print("❌ 部分測試失敗")
        print("=" * 80)
        return False


def test_story_level_actions_mapping():
    """測試 story_level_actions 和新特徵的對應關係"""

    print("\n" + "=" * 80)
    print("測試 Story Level Actions 映射")
    print("=" * 80)

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

    print(f"\n結構配置:")
    print(f"  - 樓層: {structure.story_num}")
    print(f"  - Total story actions: {len(structure.story_level_actions)}")
    print(f"    • X樑: {len(structure.story_xdir_beam_member)} layers x ? members each")
    print(f"    • Z樑: {len(structure.story_zdir_beam_member)} layers x ? members each")
    print(f"    • 外柱: {len(structure.story_outer_column_member)} layers x ? members each")
    print(f"    • 內柱: {len(structure.story_inner_column_member)} layers x ? members each")

    # 初始化 graph
    print(f"\n初始化 Graph...")
    structure.init_graph_GraphRL()
    print(f"✓ Graph 初始化成功")

    # 驗證每個 action 的特徵編碼是否與其類別一致
    print(f"\n驗證前 24 個 actions 的特徵編碼:")

    all_correct = True
    for action_idx in range(min(24, len(structure.story_level_actions))):
        member_indices = structure.story_level_actions[action_idx]
        category = structure.story_level_categories[action_idx]

        # 取第一個構件檢查
        if len(member_indices) > 0:
            member_idx = member_indices[0]
            edge_idx = member_idx * 2
            type_encoding = structure.graph.edge_attr[edge_idx, 11:15].tolist()

            # 根據 category 確定預期編碼
            if category == 'xdir_beam':
                expected = [1, 0, 0, 0]
            elif category == 'zdir_beam':
                expected = [0, 1, 0, 0]
            elif category == 'outer_column':
                expected = [0, 0, 1, 0]
            elif category == 'inner_column':
                expected = [0, 0, 0, 1]

            match = type_encoding == expected
            symbol = "✓" if match else "❌"

            print(f"  Action {action_idx:2d} ({category:15s}): {type_encoding} {symbol}")

            if not match:
                all_correct = False

    print(f"\n" + "=" * 80)
    if all_correct:
        print("✓ 所有 actions 的特徵編碼與類別一致!")
    else:
        print("❌ 部分 actions 的特徵編碼不匹配")
    print("=" * 80)

    return all_correct


if __name__ == "__main__":
    print("\n開始測試新添加的 Edge Features...\n")

    try:
        # 測試 1: 基本特徵檢查
        test1_pass = test_new_edge_features()

        # 測試 2: Actions 映射檢查
        test2_pass = test_story_level_actions_mapping()

        print("\n" + "=" * 80)
        print("總結:")
        print("=" * 80)
        print(f"測試 1 (Edge Features): {'✓ 通過' if test1_pass else '❌ 失敗'}")
        print(f"測試 2 (Actions 映射): {'✓ 通過' if test2_pass else '❌ 失敗'}")

        if test1_pass and test2_pass:
            print(f"\n🎉 所有測試通過! 新特徵已成功添加。")
            print(f"\n接下來:")
            print(f"  1. 可以開始訓練 PPO/A2C")
            print(f"  2. 測試在 6F 4x4 結構上是否能正確選擇柱子")
            print(f"  3. DQN (train.py) 仍使用 11/13 維 edge features,不受影響")
        else:
            print(f"\n⚠️ 部分測試失敗,請檢查實現")

        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
