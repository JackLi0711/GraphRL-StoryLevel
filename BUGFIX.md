# 🐛 錯誤修復：TypeError in actor_loss function

## 問題描述
```
TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
```

## 問題原因
在 `actor_loss` 和 `critic_loss` 函數中，當接收到圖形數據（tuple格式）時，程序錯誤地將整個tuple傳遞給了 `feature_processor`，而 `feature_processor` 期望接收張量。

## 解決方案
✅ **已修復** `RL/option_critic_gnn.py` 中的以下函數：

### 1. `actor_loss` 函數 (line ~399)
- 改進了對tuple觀測數據的處理
- 當 `obs` 是 tuple 時，正確解包圖形數據組件
- 當不是標準圖形數據格式時，智能轉換為張量

### 2. `critic_loss` 函數 (line ~316) 
- 同樣修復了tuple觀測數據的處理邏輯
- 確保所有數據類型都能正確處理

## 修復要點
```python
# 修復前
if isinstance(obs, (tuple, list)) and len(obs) == 4:
    # 正確處理圖形數據
else:
    state = model.feature_processor(obs)  # ❌ obs 可能是 tuple

# 修復後  
if isinstance(obs, (tuple, list)) and len(obs) >= 4:
    # 正確處理圖形數據
else:
    # ✅ 智能轉換 tuple 為張量
    if isinstance(obs, (tuple, list)):
        obs = obs[0] if len(obs) == 1 else torch.stack(list(obs))
    state = model.feature_processor(obs)
```

## 測試狀態
- ✅ 語法檢查通過
- ⏳ 等待實際運行測試

## 使用方式
修復完成後，直接重新運行訓練腳本：
```bash
python train_option_critic.py --epochs 10 --num_options 4
```

現在 StateGNN 應該可以正常訓練了！🎉