# GAT Attention Visualization

這個工具用於可視化 Option-Critic 模型中 Graph Attention Network (GAT) 的 attention weights。

## 功能說明

從訓練好的模型提取 GAT 最後一層的 attention weights，並生成 3D 結構可視化圖：
- **每個 attention head 一張圖**（預設 4 個 heads = 4 張圖）
- **灰階顏色映射**：黑色 = 高 attention，白色 = 低 attention
- **線寬變化**：反映 attention 強度
- **靜態 JPG 輸出**：固定視角（elevation=20°, azimuth=45°）

## 架構修改

### 1. `RL/model.py` - StateGNN
增加了 `return_attention_weights` 參數：

```python
def forward(self, x, edge_index, edge_attr, batch, story_batch, structure_story_ptr,
            return_attention_weights=False):
    # ...
    if return_attention_weights:
        return state, attention_weights
    return state
```

### 2. `Visualization/visualize_attention.py` - 新檔案
完整的 attention 可視化實現，包含：
- `build_edge_mapping()`: 建立 graph edge 到 structure member 的映射
- `extract_gat_attention()`: 從模型提取 attention weights
- `plot_3d_structure_attention()`: 3D 繪圖核心
- `visualize_model_attention()`: 主入口函數

## 使用方法

### 方法 1: 命令行運行

```bash
python Visualization/visualize_attention.py \
    --model_path checkpoints/option_oc/20250113_xxx/best_model.pt \
    --checkpoint_dir checkpoints/option_oc/20250113_xxx \
    --structure_shape 4story \
    --device cuda
```

**參數說明：**
- `--model_path`: 訓練好的模型檔案路徑 (`.pt` 檔)
- `--checkpoint_dir`: 保存結果的目錄（通常與 model_path 同一目錄）
- `--structure_shape`: 結構形狀 (例如 `4story`, `7story`)
- `--device`: 運算設備 (`cuda` 或 `cpu`)

### 方法 2: Python 代碼調用

```python
from Visualization.visualize_attention import visualize_model_attention

visualize_model_attention(
    model_path="checkpoints/option_oc/xxx/best_model.pt",
    checkpoint_dir="checkpoints/option_oc/xxx",
    structure_config={'structure_shape': '4story'},
    device='cuda'
)
```

### 方法 3: 使用測試腳本

1. 編輯 `test_attention_viz.py`，修改模型路徑：
```python
model_path = "checkpoints/option_oc/YOUR_TIMESTAMP/best_model.pt"
checkpoint_dir = "checkpoints/option_oc/YOUR_TIMESTAMP"
```

2. 運行測試：
```bash
python test_attention_viz.py
```

## 輸出結果

生成的檔案位於 `{checkpoint_dir}/attention_visualization/`：

```
attention_visualization/
├── attention_head_0.jpg          # Head 0 的 attention 可視化
├── attention_head_1.jpg          # Head 1 的 attention 可視化
├── attention_head_2.jpg          # Head 2 的 attention 可視化
├── attention_head_3.jpg          # Head 3 的 attention 可視化
└── attention_stats.json          # 統計資訊
```

### attention_stats.json 範例

```json
{
  "model_path": "checkpoints/option_oc/xxx/best_model.pt",
  "structure_shape": "4story",
  "num_heads": 4,
  "num_members": 208,
  "heads": [
    {
      "head_index": 0,
      "num_members": 208,
      "attention_min": 0.1234,
      "attention_max": 0.8765,
      "attention_mean": 0.4567,
      "attention_std": 0.1234
    },
    ...
  ]
}
```

## 可視化解讀

### 圖片元素
- **節點（nodes）**: 小灰點，表示結構的節點位置
- **構件（members）**: 連接節點的線條
  - **樑（beams）**: 水平方向的線條
  - **柱（columns）**: 垂直方向的線條
- **顏色**:
  - **黑色/深灰**: 高 attention（模型重點關注的構件）
  - **白色/淺灰**: 低 attention（模型較不關注的構件）
- **線寬**: 粗線表示高 attention，細線表示低 attention

### Attention 的意義
- **高 attention 的構件**：模型在做決策時重點關注這些構件之間的關係
- **不同 heads 的差異**：每個 attention head 學習到不同的結構模式
  - Head 0 可能關注樑的連接
  - Head 1 可能關注柱的連接
  - Head 2, 3 可能關注全局結構或特定樓層

## 技術細節

### Attention 提取流程
1. 從訓練好的模型載入權重
2. 使用 `model.state_gnn(..., return_attention_weights=True)` 提取最後一層 GAT 的 attention
3. 處理雙向邊（bidirectional edges）：取平均值
4. 映射 graph edge index 到 structure member names

### 顏色映射
- **Colormap**: `'gray_r'` (reversed gray)
- **歸一化**: 基於每個 head 的 min/max attention 值
- **公式**: `color = gray_r(normalize(attention))`

### 線寬計算
```python
linewidth = 1.0 + 3.0 * normalized_attention
# 範圍: [1.0, 4.0]
```

## 常見問題

### Q1: 如何找到訓練好的模型？
檢查 `checkpoints/option_oc/` 目錄下的時間戳子目錄，尋找 `best_model.pt` 檔案。

### Q2: 出現 CUDA out of memory 錯誤？
將 `--device` 改為 `cpu`：
```bash
python Visualization/visualize_attention.py ... --device cpu
```

### Q3: 如何調整視角？
編輯 `visualize_attention.py` 中的 `plot_3d_structure_attention()` 函數：
```python
ax.view_init(elev=20, azim=45)  # 修改這兩個參數
```

### Q4: 想要不同的 colormap？
編輯 `plot_3d_structure_attention()` 函數：
```python
cmap = plt.cm.get_cmap('gray_r')  # 改為 'viridis', 'hot', 'coolwarm' 等
```

### Q5: 如何只看特定的 head？
修改 `visualize_model_attention()` 中的循環：
```python
for head_idx in [0, 2]:  # 只繪製 head 0 和 2
    ...
```

## 擴展功能建議

如果需要以下功能，可以進一步擴展：

1. **互動式 3D 視圖**: 使用 plotly 替代 matplotlib
2. **動畫**: 顯示訓練過程中 attention 的演變
3. **Option 比較**: 比較不同 options 對相同結構的 attention 分佈
4. **熱力圖**: 將 attention 投影到 2D 平面（俯視圖、側視圖）
5. **統計分析**: attention 與構件類型（樑/柱）、樓層的關係

## 引用

如果這個可視化工具對您的研究有幫助，請引用原論文：
- Veličković et al. (2018). "Graph Attention Networks." ICLR.
- Bacon et al. (2017). "The Option-Critic Architecture." AAAI.

## 聯繫方式

如有問題或建議，請聯繫開發團隊。



python Visualization/visualize_attention.py --model_path checkpoints/option_oc/OB_002/best_model.pt --checkpoint_dir checkpoints/option_oc/OB_002 --structure_shape random --device cuda
