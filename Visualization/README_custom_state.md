# GAT Attention Visualization for Custom Structure States

這個腳本允許您指定具體的 `story_level_sections`（例如 `[5, 0, 0, 1]`），然後可視化該結構狀態下的 attention values。

## 功能說明

與原始的 `visualize_attention.py` 相比，這個腳本的主要差異：
- ✅ 可以指定**自定義的 section indices**
- ✅ 可以比較**不同結構狀態**下的 attention 分佈
- ✅ 支援**兩種輸入格式**（空格分隔 或 JSON）
- ✅ 可選的**輸出檔案後綴**，方便管理多個狀態的結果

## Story Level Sections 格式

`story_level_sections` 是一個整數列表，表示每個 story member 的斷面索引：

```
[xbeam_s1, ..., xbeam_sN, zbeam_s1, ..., zbeam_sN, outcol_s1, ..., outcol_sN, incol_s1, ..., incol_sN]
```

### 範例（4層樓建築）
假設結構有 4 層樓，每層有 4 個 story members（x-beam, z-beam, outer-column, inner-column）：

```python
story_sections = [
    5, 0, 0, 1,  # X-beams for stories 1-4 (section indices)
    3, 2, 1, 0,  # Z-beams for stories 1-4
    4, 3, 2, 1,  # Outer-columns for stories 1-4
    2, 1, 0, 0   # Inner-columns for stories 1-4
]
# Total: 16 values for 4 stories × 4 member types
```

### Section Index 對應
- `0` = 最細的斷面
- `8` = 最粗的斷面
- 中間值 = 中等粗細

## 使用方法

### 方法 1: 空格分隔的 section indices

```bash
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/OB_002/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/OB_002 \
    --structure_shape random \
    --story_sections 5 0 0 1 3 2 1 0 4 3 2 1 2 1 0 0 \
    --device cuda
```

### 方法 2: JSON 格式

```bash
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/OB_002/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/OB_002 \
    --structure_shape random \
    --story_sections_json "[5,0,0,1,3,2,1,0,4,3,2,1,2,1,0,0]" \
    --device cuda
```

### 方法 3: 使用輸出後綴（方便比較多個狀態）

```bash
# 初始狀態 - 全部最粗
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/OB_002/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/OB_002 \
    --structure_shape random \
    --story_sections 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
    --output_suffix "initial_thick" \
    --device cuda

# 最終狀態 - 優化後
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/OB_002/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/OB_002 \
    --structure_shape random \
    --story_sections 2 0 0 0 1 1 0 0 2 1 1 0 1 0 0 0 \
    --output_suffix "final_optimized" \
    --device cuda
```

## 參數說明

### 必需參數
- `--model_path`: 訓練好的模型路徑 (`.pt` 檔)
- `--checkpoint_dir`: 保存結果的目錄
- `--structure_shape`: 結構形狀 (`4story`, `7story`, `random` 等)
- `--story_sections` 或 `--story_sections_json`: Section indices（二選一）

### 可選參數
- `--output_suffix`: 輸出檔案後綴（預設為空）
- `--device`: 運算設備 (`cuda` 或 `cpu`，預設 `cuda`）

## 輸出結果

### 檔案結構

#### 不使用 suffix:
```
checkpoints/option_oc/xxx/attention_custom_sections/
├── attention_head_0.jpg
├── attention_head_1.jpg
├── attention_head_2.jpg
├── attention_head_3.jpg
└── attention_stats.json
```

#### 使用 suffix (例如 `--output_suffix "state_A"`):
```
checkpoints/option_oc/xxx/attention_custom_state_A/
├── attention_head_0_state_A.jpg
├── attention_head_1_state_A.jpg
├── attention_head_2_state_A.jpg
├── attention_head_3_state_A.jpg
└── attention_stats_state_A.json
```

### attention_stats.json 範例

```json
{
  "model_path": "checkpoints/option_oc/OB_002/best_model.pt",
  "structure_shape": "random",
  "story_sections": [5, 0, 0, 1, 3, 2, 1, 0, 4, 3, 2, 1, 2, 1, 0, 0],
  "num_heads": 4,
  "num_members": 148,
  "heads": [
    {
      "head_index": 0,
      "num_members": 148,
      "attention_min": 0.1234,
      "attention_max": 0.8765,
      "attention_mean": 0.4567,
      "attention_std": 0.1234
    },
    ...
  ]
}
```

## 實際應用場景

### 1. 比較訓練前後的 Attention 變化

```bash
# 初始狀態（全粗）
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/xxx/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/xxx \
    --structure_shape random \
    --story_sections 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
    --output_suffix "init" \
    --device cuda

# 最終狀態（優化後）
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/xxx/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/xxx \
    --structure_shape random \
    --story_sections 2 0 0 0 1 1 0 0 2 1 1 0 1 0 0 0 \
    --output_suffix "final" \
    --device cuda
```

然後比較 `attention_custom_init/` 和 `attention_custom_final/` 中的圖片。

### 2. 分析特定失敗案例

如果某個 episode 失敗了，可以用該狀態的 sections 來分析：

```bash
# 從 training log 或 record 中找到失敗時的 story_level_sections
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/xxx/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/xxx \
    --structure_shape random \
    --story_sections 5 4 3 2 5 4 3 2 6 5 4 3 4 3 2 1 \
    --output_suffix "failed_state" \
    --device cuda
```

### 3. 驗證 Attention 是否關注正確的構件

```bash
# 創建一個不平衡的結構（一側粗、一側細）
python Visualization/visualize_attention_custom_state.py \
    --model_path checkpoints/option_oc/xxx/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/xxx \
    --structure_shape random \
    --story_sections 8 8 0 0 8 8 0 0 8 8 0 0 8 8 0 0 \
    --output_suffix "unbalanced" \
    --device cuda
```

看 attention 是否集中在細的構件上。

## 如何確定 story_sections 的長度？

不同的結構形狀需要不同長度的 `story_sections`：

### 4 層樓建築
```python
story_num = 4
num_member_types = 4  # x-beam, z-beam, outer-col, inner-col
total_length = story_num * num_member_types = 16
```

### 7 層樓建築
```python
story_num = 7
total_length = 7 * 4 = 28
```

### 自動獲取正確長度

如果不確定長度，可以先用原始腳本查看：

```bash
python Visualization/visualize_attention.py \
    --model_path checkpoints/option_oc/xxx/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/xxx \
    --structure_shape random \
    --device cuda
```

查看輸出中的 "Total members" 或檢查生成的 `attention_stats.json`。

## 錯誤處理

### 錯誤 1: Section 長度不匹配

```
ValueError: story_sections length mismatch! Expected 16, got 12
```

**解決**: 檢查您的結構有幾層樓，計算正確的長度（`story_num * 4`）。

### 錯誤 2: Section index 超出範圍

```
IndexError: list index out of range
```

**解決**: Section indices 應該在 0-8 之間（對應 9 個可用的斷面）。

### 錯誤 3: JSON 格式錯誤

```
Error: Invalid JSON format in --story_sections_json
```

**解決**: 確保 JSON 格式正確：
- ✅ 正確: `"[5,0,0,1]"`
- ❌ 錯誤: `[5,0,0,1]` （缺少引號）
- ❌ 錯誤: `"[5, 0, 0, 1]"` （JSON 中空格可能導致問題，建議不加空格）

## 進階用法

### Python 腳本中調用

```python
from Visualization.visualize_attention_custom_state import visualize_attention_custom_state

# 定義多個狀態
states = {
    "initial": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    "mid": [5, 4, 3, 2, 5, 4, 3, 2, 6, 5, 4, 3, 4, 3, 2, 1],
    "final": [2, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 1, 0, 0, 0]
}

for state_name, sections in states.items():
    visualize_attention_custom_state(
        model_path="checkpoints/option_oc/xxx/best_model.pt",
        checkpoint_dir="checkpoints/option_oc/xxx",
        structure_shape="random",
        story_sections=sections,
        device="cuda",
        output_suffix=state_name
    )
```

### 批量處理

```bash
#!/bin/bash
# batch_visualize.sh

MODEL="checkpoints/option_oc/OB_002/best_model.pt"
CKPT="checkpoints/option_oc/OB_002"
SHAPE="random"

# State 1: Initial
python Visualization/visualize_attention_custom_state.py \
    --model_path $MODEL --checkpoint_dir $CKPT --structure_shape $SHAPE \
    --story_sections 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
    --output_suffix "t0" --device cuda

# State 2: Mid
python Visualization/visualize_attention_custom_state.py \
    --model_path $MODEL --checkpoint_dir $CKPT --structure_shape $SHAPE \
    --story_sections 5 4 3 2 5 4 3 2 6 5 4 3 4 3 2 1 \
    --output_suffix "t50" --device cuda

# State 3: Final
python Visualization/visualize_attention_custom_state.py \
    --model_path $MODEL --checkpoint_dir $CKPT --structure_shape $SHAPE \
    --story_sections 2 0 0 0 1 1 0 0 2 1 1 0 1 0 0 0 \
    --output_suffix "t100" --device cuda
```

## 常見問題

### Q1: 如何知道我的結構需要多少個 section values？
首先運行原始腳本看結構資訊，或者查看訓練 log 中的 `story_level_sections`。

### Q2: Section index 的範圍是什麼？
通常是 0-8（共 9 個斷面），0 是最細，8 是最粗。

### Q3: 可以同時可視化多個狀態嗎？
可以！使用不同的 `--output_suffix` 多次運行即可。

### Q4: 輸出圖片與原始腳本有什麼不同？
完全相同的視覺化風格，只是結構的 sections 不同。

### Q5: 如何從 training record 中提取 story_sections？
查看訓練過程的 log 或使用 inference 腳本記錄的 action sequence。

## 與原始腳本的比較

| 特性 | visualize_attention.py | visualize_attention_custom_state.py |
|------|------------------------|-------------------------------------|
| 結構狀態 | 環境自動生成（隨機或固定） | 手動指定 sections |
| 輸入方式 | 只需指定 structure_shape | 需指定 sections 列表 |
| 使用場景 | 快速查看預設狀態的 attention | 分析特定狀態的 attention |
| 比較能力 | 單一狀態 | 可比較多個狀態 |

## 建議工作流程

1. **初步探索**: 使用 `visualize_attention.py` 快速查看
2. **深入分析**: 使用 `visualize_attention_custom_state.py` 分析特定狀態
3. **狀態比較**: 使用不同 suffix 生成多個狀態的可視化
4. **結果整理**: 比較不同狀態下的 attention 分佈差異

## 參考資料

- 原始腳本: `visualize_attention.py`
- 結構定義: `Structure/structure.py`
- Section 定義: `Structure/sections.py`


python Visualization/visualize_attention_custom_state.py --model_path checkpoints/option_oc/OB_002/best_model.pt --checkpoint_dir checkpoints/option_oc/OB_002  --structure_shape fixed --story_sections 5 4 4 1 0 0 5 3 1 0 0 0 3 3 1 1 0 0 1 0 0 0 0 0  --output_suffix "final_optimized"  --device cuda


python Visualization/visualize_attention_custom_state.py --model_path checkpoints/option_oc/OB_002/best_model.pt --checkpoint_dir checkpoints/option_oc/OB_002  --structure_shape fixed --story_sections 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  --output_suffix "test_min"  --device cuda

  5,
      4,
      4,
      1,
      0,
      0,
      5,
      3,
      1,
      0,
      0,
      0,
      3,
      3,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0