# Option-Critic for Structural Design

## Overview

本專案實現 **Option-Critic Architecture** (Bacon et al., 2016) 用於建築結構優化，使用 Graph Neural Networks 處理複雜的結構關係。

### 🆕 Version 2.0 - 動態 Action Space

**重大更新 (2025-10-02)**: 支持動態 action size，可處理任意樓層數的建築！

---

## Key Features

### 1. **動態 Action Space** 🆕

- 支持不同樓層數的建築 (4層、7層、10層等)
- 使用 **per-story-member scoring** 架構
- Action = 選擇哪個結構構件減小斷面

**示例：**
```python
# 4層樓建築
4 floors × 4 members/floor = 16 actions

# 7層樓建築
7 floors × 4 members/floor = 28 actions

# 10層樓建築
10 floors × 4 members/floor = 40 actions
```

### 2. **完整對齊原文理論**

實現了 Option-Critic 論文的所有關鍵組件：

- ✅ **Intra-Option Policy Gradient** (Theorem 1)
  ```
  ∇_θ J = E[∇log π(a|s,ω) * Q_U(s,ω,a)]
  ```

- ✅ **Termination Gradient** (Theorem 2)
  ```
  ∇_ϑ J = E[β(s',ω) * A_Ω(s',ω)]
  ```

- ✅ **Q_U 估計機制** (Page 4)
  ```
  Q_U(s,ω,a) = r(s,a) + γ * U(ω,s')
  U(ω,s') = (1-β(s'))*Q_Ω(s',ω) + β(s')*V_Ω(s')
  ```

### 3. **Graph Neural Network Backbone**

- 處理圖結構的建築模型
- 提取 story-level 和 global-level 特徵
- 支持複雜的節點和邊關係

---

## Architecture

### Dynamic Action Size Handling

#### 問題
不同樓層數 → 不同的 action 數量

#### 解決方案: Per-Member Scoring

```python
# 舊架構 (v1) - 固定 action size ❌
options_W = [num_options, gnn_dim, num_actions]  # 固定!
logits = state @ options_W[option] + options_b[option]
# logits shape: [N, num_actions] ← 固定大小

# 新架構 (v2) - 動態 action size ✅
intra_option_policies = ModuleList([
    MLP(gnn_output_dim → 1)  # 每個 story member 一個 score
    for _ in range(num_options)
])
scores = intra_option_policies[option](story_features)
# scores shape: [N] ← 動態大小！
```

**理論等價性：**
- 原文：`π(a|s,ω)` over fixed action set A
- 我們：`π(m|s,ω)` over dynamic member set M
- 其中 action a ↔ story member m

### Model Components

```
Input: Graph(nodes, edges, features)
  ↓
StateGNN
  ↓
├─ Story-level features [N, 128] ──→ Intra-option policies (per-member scoring)
└─ Global features [1, 64] ────────→ Q_Ω, β (option values & termination)
```

#### 1. StateGNN
- 處理圖結構輸入
- 輸出 story-level 和 global 特徵

#### 2. Policy-Over-Options (Q_Ω)
- `Q_Ω(s,ω)`: 在狀態 s 選擇 option ω 的價值
- 用於決定執行哪個 option

#### 3. Termination Functions (β)
- `β(s,ω)`: 在狀態 s 終止 option ω 的概率
- 控制 option 持續時間

#### 4. Intra-Option Policies (π)
- `π(m|s,ω)`: 在 option ω 下選擇 story member m 的概率
- 每個 option 有獨立的 MLP
- **動態支持任意數量的 story members**

---

## Installation

```bash
# Clone repository
git clone <repo_url>
cd GraphRL_story_level

# Install dependencies
pip install torch torch_geometric numpy

# Verify installation
python tests/test_dynamic_action_size.py
```

---

## Training

### Quick Start

```bash
python train_option_critic.py \
    --epochs 1000 \
    --num_options 4 \
    --max_option_len 10 \
    --batch_size 32 \
    --device cuda
```

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of training episodes | 1000 |
| `--num_options` | Number of options | 4 |
| `--max_option_len` | Maximum steps per option | 10 |
| `--batch_size` | Batch size for updates | 32 |
| `--actor_lr` | Actor learning rate | 0.0001 |
| `--critic_lr` | Critic learning rate | 0.001 |
| `--gamma` | Discount factor | 0.99 |
| `--device` | Device (cpu/cuda) | cuda |
| `--debug_logging` | Enable debug logs | False |

### Monitoring Training

訓練日誌會保存在 `logs/` 目錄：

```bash
# 查看訓練進度
tail -f logs/training.log

# 查看 Q_U 值監控 (如果啟用 debug_logging)
grep "Q_U Monitor" logs/training.log
```

---

## Testing

### 單元測試 - 動態 Action Size

```bash
python tests/test_dynamic_action_size.py
```

測試內容：
- ✅ 架構驗證 (MLP-based policies)
- ✅ 4/7/10 層樓的 action selection
- ✅ Q_U 計算獨立於 action index
- ✅ Actor loss 與不同 episode sizes

### 整合測試 - 完整流程

```bash
python tests/test_integration.py
```

測試內容：
- ✅ 完整 forward pass
- ✅ Loss 計算流程
- ✅ Gradient 反向傳播
- ✅ 參數數量驗證

### 快速驗證 - 短期訓練

```bash
python train_option_critic.py --epochs 10 --device cpu
```

確認：
- ✅ 無 action size 相關錯誤
- ✅ Loss 為有限數字
- ✅ 可處理不同樓層數建築

詳細驗證指南請參考：[TRAINING_VERIFICATION.md](TRAINING_VERIFICATION.md)

---

## Implementation Details

### 與原文對應

| 原文符號 | 實現位置 | 說明 |
|---------|---------|------|
| `Q_Ω(s,ω)` | `model.get_Q(state)` | Option-value function |
| `Q_U(s,ω,a)` | `model.compute_Q_U(...)` | Action-value function (estimated) |
| `β(s,ω)` | `model.get_terminations(state)` | Termination probability |
| `π(a\|s,ω)` | `model.get_action(...)` | Intra-option policy (per-member scoring) |
| `V_Ω(s)` | `max Q_Ω(s,ω)` | State value (greedy policy-over-options) |

### Gradient Updates

#### 1. Critic - 學習 Q_Ω(s,ω)

```python
δ = r + γ[(1-β)Q_Ω(s',ω) + β max_ω' Q_Ω(s',ω')] - Q_Ω(s,ω)
Loss = δ²
```

#### 2. Intra-Option Policy

```python
∇_θ J = ∇_θ log π(a|s,ω) * Q_U(s,ω,a)
```

#### 3. Termination

```python
∇_ϑ J = -β(s',ω) * [Q_Ω(s',ω) - V_Ω(s') + ε]
```

---

## Architecture Comparison

### v1 (舊版) - 固定 Action Size ❌

```python
# 固定參數矩陣
self.options_W = nn.Parameter(torch.zeros(num_options, gnn_dim, num_actions))
self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

# Action selection
logits = story_state @ self.options_W[option] + self.options_b[option]
# logits shape: [num_story_members, num_actions] ← 固定!
```

**問題：**
- ❌ `num_actions` 必須在初始化時確定
- ❌ 無法處理不同樓層數的建築
- ❌ 參數數量與 action space 大小成正比

### v2 (新版) - 動態 Action Size ✅

```python
# 每個 option 的 MLP
self.intra_option_policies = nn.ModuleList([
    nn.Sequential(
        nn.Linear(gnn_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)  # 每個 story member 一個 score
    )
    for _ in range(num_options)
])

# Action selection
scores = self.intra_option_policies[option](story_state)
# scores shape: [num_story_members] ← 動態!
```

**優點：**
- ✅ 支持任意數量的 story members
- ✅ 同一模型處理不同規模建築
- ✅ 參數數量固定，與 action space 無關

---

## Project Structure

```
GraphRL_story_level/
├── RL/
│   ├── option_critic_gnn.py       # 主模型實現 (v2)
│   ├── model.py                    # StateGNN
│   └── option_critic/
│       └── trainer.py              # 訓練器
├── tests/
│   ├── test_dynamic_action_size.py # 單元測試
│   └── test_integration.py         # 整合測試
├── OPTION_CRITIC_REFACTOR_PLAN.md  # 重構計畫
├── TRAINING_VERIFICATION.md        # 驗證指南
├── README.md                       # 本文件
└── train_option_critic.py          # 訓練腳本
```

---

## Expected Performance

### 訓練指標

- **Loss 趨勢**: Actor & Critic loss 應逐漸下降並收斂
- **Episode Reward**: 隨訓練增加
- **Success Rate**: 目標 > 80%
- **Option Length**: 應合理分佈，避免過短或過長

### 性能改進

相比 v1 架構：
- ✅ **通用性**: 單一模型處理所有建築規模
- ✅ **穩定性**: 正確的 Q_U 估計提升訓練穩定性
- ✅ **理論正確**: 完全對齊 Option-Critic 原文

---

## Troubleshooting

### 常見問題

1. **Shape Mismatch**
   - 確認 `valid_actions_mask` shape = `[num_story_members]`
   - 檢查 `story_level_state` shape = `[num_story_members, gnn_dim]`

2. **NaN Loss**
   - 檢查 learning rates 是否過大
   - 確認 rewards 在合理範圍
   - 啟用 `debug_logging` 查看 Q_U 值

3. **無效 Action**
   - 驗證 `valid_actions_mask` 正確傳遞
   - 檢查 masking 邏輯

詳細排查指南：[TRAINING_VERIFICATION.md](TRAINING_VERIFICATION.md)

---

## References

### Papers

1. **Bacon, P. L., Harb, J., & Precup, D. (2016)**
   The Option-Critic Architecture.
   arXiv preprint arXiv:1609.05140.

   **關鍵公式**:
   - Equation 1: `Q_Ω(s,ω) = Σ_a π(a|s,ω) * Q_U(s,ω,a)`
   - Equation 2: `Q_U(s,ω,a) = r(s,a) + γ * U(ω,s')`
   - Equation 3: `U(ω,s') = (1-β(s'))*Q_Ω(s',ω) + β(s')*V_Ω(s')`
   - Theorem 1: Intra-Option Policy Gradient
   - Theorem 2: Termination Gradient

### Documentation

- [重構計畫](OPTION_CRITIC_REFACTOR_PLAN.md) - 詳細的架構重構說明
- [訓練驗證指南](TRAINING_VERIFICATION.md) - 完整的測試和驗證流程

---

## Changelog

### Version 2.0 (2025-10-02)

**重大更新：**
- ✅ 支持動態 action size
- ✅ 重構 intra-option policies 為 MLP-based
- ✅ 添加 Q_U 估計機制
- ✅ 修正 policy gradient 使用正確的 Q_U
- ✅ 完整測試套件
- ✅ 訓練驗證指南

**向後兼容：**
- `num_actions` 參數保留但不再使用
- API 接口保持一致
- 可直接加載舊的 checkpoint（會自動忽略 options_W/b）

---

## License

MIT License

---

## Citation

如果您使用本專案，請引用：

```bibtex
@article{bacon2016option,
  title={The option-critic architecture},
  author={Bacon, Pierre-Luc and Harb, Jean and Precup, Doina},
  journal={arXiv preprint arXiv:1609.05140},
  year={2016}
}
```

---

## Contact

如有問題或建議，請提交 Issue 或 Pull Request。

**Happy Training! 🚀**
