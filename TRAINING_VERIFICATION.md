# Option-Critic 訓練驗證指南

## 概述

本文檔說明如何驗證 Option-Critic 架構重構（v2）是否正確實現。

---

## 快速驗證步驟

### 1. 運行單元測試

測試動態 action size 功能：

```bash
cd /mnt/d/kyle_MD_project/optionRL/GraphRL_story_level

# 如果有 conda 環境
conda activate optionRL
python tests/test_dynamic_action_size.py

# 或直接運行
python tests/test_dynamic_action_size.py
```

**預期輸出：**
```
✓ Architecture test passed
✓ 4-story building (16 members): action=X, logp=-X.XX, entropy=X.XX
✓ 7-story building (28 members): action=X, logp=-X.XX, entropy=X.XX
✓ 10-story building (40 members): action=X, logp=-X.XX, entropy=X.XX
✓ Q_U values independent of action index: X.XXXX
✓ Episode with 16 members (action=5): loss=X.XXXXXX
✓ Episode with 28 members (action=15): loss=X.XXXXXX
✓ Episode with 40 members (action=25): loss=X.XXXXXX
All tests passed! ✅
```

### 2. 運行整合測試

測試完整訓練流程：

```bash
python tests/test_integration.py
```

**預期輸出：**
```
Testing complete forward pass...
✓ 4-story building: Q_shape=torch.Size([1, 4]), action=X, logp=-X.XX, entropy=X.XX
✓ 7-story building: Q_shape=torch.Size([1, 4]), action=X, logp=-X.XX, entropy=X.XX
✓ 10-story building: Q_shape=torch.Size([1, 4]), action=X, logp=-X.XX, entropy=X.XX
✓ Complete forward pass test passed

Testing loss computation pipeline...
✓ Actor loss: X.XXXXXX
✓ Critic loss: X.XXXXXX
✓ Loss computation pipeline test passed

Testing gradient flow...
  ✓ Gradient exists for intra_option_policies.X.X.weight: mean=X.XXXXXX
  ...
✓ Gradient flow test passed

Testing parameter count...
  Policy parameters: XXX,XXX
  Q network parameters: XXX
  Termination parameters: XXX
  StateGNN parameters: XXX,XXX
  Total parameters: XXX,XXX
✓ Parameter count test passed

All integration tests passed! ✅
```

### 3. 短期訓練測試（推薦）

運行 10 個 episodes 來驗證沒有錯誤：

```bash
python train_option_critic.py \
    --epochs 10 \
    --num_options 2 \
    --eval_frequency 5 \
    --batch_size 16 \
    --device cpu \
    --debug_logging False
```

**檢查項目：**
- ✅ 沒有 action size 相關的錯誤
- ✅ 沒有 shape mismatch 錯誤
- ✅ Loss 值為有限數字（非 NaN 或 Inf）
- ✅ 模型能夠處理不同樓層數的建築（4層、7層、10層）

**預期日誌片段：**
```
Episode 1: score=XXX, reward=XXX, steps=XXX
Episode 2: score=XXX, reward=XXX, steps=XXX
...
Evaluation: avg_score=XXX, success_rate=XX%
```

---

## 完整訓練測試

### 標準訓練（1000 episodes）

```bash
python train_option_critic.py \
    --epochs 1000 \
    --num_options 4 \
    --max_option_len 10 \
    --batch_size 32 \
    --device cuda \
    --eval_frequency 50 \
    --save_frequency 100
```

### 監控關鍵指標

1. **Loss 趨勢**
   - Actor loss 應該逐漸下降
   - Critic loss 應該收斂到穩定值

2. **Episode Reward**
   - 應該隨訓練逐漸增加
   - 檢查 `logs/training.log` 中的 reward 曲線

3. **Success Rate**
   - Evaluation 時的成功率應該提升
   - 目標: > 80% success rate

4. **Q_U 值（如果啟用 debug_logging）**
   ```
   [Q_U Monitor] Sample 0: ω=X, a=X, r=X.XXXX, Q_U=X.XXXX
   ```
   - Q_U 值應該合理（通常在 [-10, 10] 範圍內）
   - 與 reward 有相關性

---

## 驗證清單

### ✅ 架構驗證
- [ ] `intra_option_policies` 是 ModuleList
- [ ] 每個 option 有獨立的 MLP
- [ ] 沒有 `options_W` 和 `options_b` 參數
- [ ] 模型總參數數量合理

### ✅ 功能驗證
- [ ] `get_action` 支持不同數量的 story members
- [ ] `compute_Q_U` 計算正確（獨立於 action index）
- [ ] `actor_loss` 使用正確的 Q_U
- [ ] `critic_loss` 與動態 action size 兼容

### ✅ 訓練驗證
- [ ] 能處理 4 層樓建築（16 actions）
- [ ] 能處理 7 層樓建築（28 actions）
- [ ] 能處理 10 層樓建築（40 actions）
- [ ] 不同 episode 間切換無錯誤
- [ ] Loss 不會出現 NaN 或 Inf
- [ ] Gradients 正確流動

### ✅ 性能驗證
- [ ] 訓練速度與之前相當
- [ ] Memory 使用合理
- [ ] 能夠收斂到好的解

---

## 常見問題排查

### 問題 1: Shape Mismatch 錯誤

**症狀：**
```
RuntimeError: shape mismatch...
```

**檢查：**
- 確認 `valid_actions_mask` shape 是 `[num_story_members]`
- 確認傳遞給 `get_action` 的 `story_level_state` shape 是 `[num_story_members, gnn_output_dim]`

### 問題 2: NaN Loss

**症狀：**
```
Actor loss: nan
```

**檢查：**
- 確認 Q_U 值不是 NaN（檢查 debug logs）
- 確認 rewards 在合理範圍內
- 檢查 learning rates 是否過大

### 問題 3: 無效 Action

**症狀：**
```
ERROR: Selected invalid action X!
```

**檢查：**
- 確認 `valid_actions_mask` 正確傳遞
- 檢查 masking 邏輯是否正確應用
- 查看 debug logs 確認 action selection 過程

---

## 與舊版本對比

### 舊架構（v1）- 固定 action size
```python
# 固定大小的參數
options_W = [num_options, gnn_dim, num_actions]  # num_actions 固定
options_b = [num_options, num_actions]

# Action selection
logits = state @ options_W[option] + options_b[option]  # [N, num_actions]
```

**限制：** 只能處理預定義數量的 actions

### 新架構（v2）- 動態 action size ✅
```python
# 每個 option 的 MLP
intra_option_policies = ModuleList([
    MLP(gnn_dim → 1)  # 每個 story member 一個 score
    for _ in range(num_options)
])

# Action selection
scores = intra_option_policies[option](state)  # [N] - 動態大小
```

**優點：** 支持任意數量的 story members

---

## 參考資料

- **Option-Critic 原文**: Bacon et al. (2016) - The Option-Critic Architecture
- **重構計畫**: `OPTION_CRITIC_REFACTOR_PLAN.md`
- **測試代碼**: `tests/test_dynamic_action_size.py`, `tests/test_integration.py`

---

## 總結

完成以上驗證步驟後，您應該能夠確認：

1. ✅ **動態 action size 支持正常工作**
2. ✅ **理論對齊 Option-Critic 原文**
3. ✅ **訓練穩定且能收斂**
4. ✅ **可處理不同規模的建築結構**

如有任何問題，請查看日誌文件或參考測試代碼進行調試。
