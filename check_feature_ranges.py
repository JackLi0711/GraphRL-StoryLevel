"""
檢查 Edge Features 的數值範圍，確認是否有 normalization

Author: Claude Code
Date: 2025-10-20
"""

import sys
sys.path.append("Structure/")

from Structure.structure import Structure
import torch

# 創建測試結構
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

structure.init_graph_GraphRL()

# 檢查 edge features 的維度 0-15 的數值範圍
edge_attr = structure.graph.edge_attr

print('=' * 80)
print('Edge Features 數值範圍檢查')
print('=' * 80)

feature_names = [
    'is_column', 'is_beam', 'L (length)', 'A', 'Iz', 'Iy', 'Zz',
    "A'", "Iz'", "Iy'", "Zz'",
    'is_xdir_beam', 'is_zdir_beam', 'is_outer_column', 'is_inner_column', 'normalized_floor'
]

print('\n只看單向 edges (每個構件的第一條邊):')
print('-' * 80)

for dim in range(16):
    values = edge_attr[::2, dim]  # 只看單向

    print(f'\n維度 {dim:2d} - {feature_names[dim]:20s}')
    print(f'  Min:    {values.min().item():15.6f}')
    print(f'  Max:    {values.max().item():15.6f}')
    print(f'  Mean:   {values.mean().item():15.6f}')
    print(f'  Std:    {values.std().item():15.6f}')

    # 檢查是否在 [0, 1] 範圍內
    if values.min() >= 0 and values.max() <= 1:
        print(f'  範圍:   [0, 1] ✓ (已 normalized)')
    elif values.min() >= -1 and values.max() <= 1:
        print(f'  範圍:   [-1, 1] ✓ (可能已 normalized)')
    else:
        print(f'  範圍:   非標準化 ✗ (原始數值)')

print('\n' + '=' * 80)
print('特徵尺度分析')
print('=' * 80)

print('\n維度 0-1 (is_column, is_beam): 二元特徵 [0, 1]')
print('維度 2 (L): 長度特徵')
print('維度 3-6 (A, Iz, Iy, Zz): 截面性質')
print('維度 7-10 (A\', Iz\', Iy\', Zz\'): 減小後的截面性質')
print('維度 11-14 (構件類型): One-hot encoding [0, 1]')
print('維度 15 (樓層): Normalized [0, 1]')

print('\n' + '=' * 80)
print('結論')
print('=' * 80)

# 檢查連續特徵的範圍
continuous_features = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # L, A, Iz, Iy, Zz, A', Iz', Iy', Zz'

normalized_continuous = []
unnormalized_continuous = []

for dim in continuous_features:
    values = edge_attr[::2, dim]
    if values.min() >= 0 and values.max() <= 1:
        normalized_continuous.append(feature_names[dim])
    else:
        unnormalized_continuous.append(feature_names[dim])

if len(normalized_continuous) > 0:
    print(f'\n✓ 已 normalized 的連續特徵 ({len(normalized_continuous)} 個):')
    for name in normalized_continuous:
        print(f'  - {name}')

if len(unnormalized_continuous) > 0:
    print(f'\n✗ 未 normalized 的連續特徵 ({len(unnormalized_continuous)} 個):')
    for name in unnormalized_continuous:
        print(f'  - {name}')

print('\n' + '=' * 80)
