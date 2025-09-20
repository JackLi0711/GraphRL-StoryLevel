# DAC Action Mask 維度修正總結

## 問題分析

### 原始問題
從您的日誌可以看出：
```
story_features.shape=torch.Size([16, 256])
option=tensor([0]), option.shape=torch.Size([1])
valid_mask=torch.Size([16])
final mean.shape=torch.Size([16, 16]), final std.shape=torch.Size([16, 16])
```

**問題：** 代碼錯誤地將 `story_features.shape[0] = 16` 當作 `batch_size`，但實際上：
- 16 是 **action space 大小**（4層樓 × 4種action）
- 這是**單一結構的資料**，要從16個actions中選擇1個
- `valid_mask.shape=[16]` 是正確的

## 修正內容

### 1. 修正 `gnn.compute_pi_bar` 方法

**修正前：**
```python
batch_size = story_features.shape[0]  # 錯誤：把action space當batch size
if options.shape[0] == 1 and batch_size > 1:
    options = options.expand(batch_size)  # 錯誤：擴展option到16個
```

**修正後：**
```python
# story_features: [num_actions, feature_dim] - 每個action的特徵
num_actions, feature_dim = story_features.shape
# 正確理解：這是16個action的特徵，不是16個樣本
```

### 2. 修正 action selection 邏輯

**修正前：** 輸出 `[16, 16]` 維度的 mean/std
**修正後：** 輸出 `[16]` 維度，表示16個action的分佈參數

### 3. 修正 action mask 應用

**修正前：** 試圖處理 `[batch_size, action_size]` 的 mask
**修正後：** 正確處理 `[action_size]` 的 mask，用於單一資料的action selection

### 4. 改進離散action選擇

使用 softmax + categorical distribution 進行離散action選擇：
```python
# 轉換continuous values為discrete action probabilities
action_probs = F.softmax(masked_mean / 0.1, dim=-1)
action_dist = torch.distributions.Categorical(action_probs)
action_idx = action_dist.sample()
```

## 預期修正效果

### 修正後的日誌應該顯示：

```
compute_pi_bar: story_features.shape=torch.Size([16, 256])  # 16個action的特徵
compute_pi_bar: options=tensor([0]), options.shape=torch.Size([1])  # 單一option
compute_pi_bar: final mean.shape=torch.Size([16]), final std.shape=torch.Size([16])  # 每個action的分佈參數

select_action: mean.shape=torch.Size([16]), std.shape=torch.Size([16])  # 16個action的值
select_action: valid_mask=torch.Size([16])  # 16個action的mask
select_action: selected action_idx=5, action=tensor([5.])  # 選擇的單一action
```

## Batch Processing 限制

### 當前限制
由於graph batching的複雜性，目前實現：
- 每次只處理一個graph sample
- 有效 mini_batch_size = 1
- 降低訓練效率但保持正確性

### 未來改進方向
1. 使用 PyTorch Geometric 的 Batch.from_data_list()
2. 處理variable graph size的batching
3. 實現真正的multi-sample batch processing

## 驗證方法

運行修正後的代碼，檢查：
1. `compute_pi_bar` 輸出維度：`[num_actions]` 而不是 `[batch_size, num_actions]`
2. Action selection 從16個action中選擇1個
3. Action mask 正確限制invalid actions
4. 沒有維度不匹配的錯誤

## 重要提醒

這次修正解決了**概念性錯誤**：
- **修正前：** 誤認為處理16個sample的batch
- **修正後：** 正確理解為處理1個sample的16個action choices

這是action space vs batch size的根本性概念修正！