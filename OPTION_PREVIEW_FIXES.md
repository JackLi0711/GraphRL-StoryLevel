# Option Preview Visualization 問題修復

## 修復的問題

### 1. 模塊導入錯誤修復

**問題**: `ModuleNotFoundError: No module named 'model'`

**原因**: `RL/q_algorithm.py` 中使用了錯誤的導入語句：
```python
from model import StateGNN, Q_Network, GraphEmbedding
```

**修復**: 改為正確的相對導入：
```python
from .model import StateGNN, Q_Network, GraphEmbedding
```

**文件**: `RL/q_algorithm.py:6`

### 2. sklearn 依賴問題修復

**問題**: `ModuleNotFoundError: No module named 'sklearn'`

**原因**: `Visualization/visualize.py` 在模塊級別導入了 sklearn.manifold.TSNE，但這個包只在 `visualize_edge_embedding` 函數中使用。

**修復**: 將 TSNE 導入移到函數內部：
```python
@torch.no_grad()
def visualize_edge_embedding(agent, env, logger, ckpt_dir):
    # Import TSNE only when needed to avoid import errors if sklearn is not available
    from sklearn.manifold import TSNE
    # ... rest of function
```

**文件**: `Visualization/visualize.py:373-374`

### 3. 移除 try-except 語句以便調試

**問題**: 用戶要求移除不必要的 try-except 語句，讓錯誤直接顯示以便調試。

**修復內容**:

1. **移除 trainer.py 中的主要 try-except**:
   - 文件: `RL/option_critic/trainer.py`
   - 移除了 `generate_option_preview_visualization` 方法中的整體 try-except 包裝
   - 現在錯誤會直接顯示，便於調試

2. **移除 visualize.py 中 Q-values 計算的 try-except**:
   - 文件: `Visualization/visualize.py:293-314`
   - 移除了 Q-values 計算的 try-except 包裝
   - 保留了基本的條件檢查

3. **保留必要的 try-except**:
   - 保留了 action 執行時的 try-except (visualize.py:334-340)
   - 這個是必要的，因為 action 執行可能失敗，需要記錄並繼續處理

## 驗證結果

### 導入測試
```bash
python -c "from Visualization.visualize import action_to_story_component; print('Import successful')"
# 結果: Import successful ✅

python -c "from Visualization.visualize import visualize_option_preview_process; print('Option preview imports successful')"
# 結果: Option preview imports successful ✅
```

### 功能測試
```bash
python test_option_preview_simple.py
# 結果: All tests passed! ✅
```

## 現在可以正常使用的功能

1. **Action 映射**: action index → (樓層, 構件類型)
2. **統計分析**: 按樓層和構件類型統計 option actions
3. **Option preview 可視化**: 生成預告圖和 GIF 動畫
4. **訓練集成**: 在 Option-Critic 訓練中自動調用

## 調試建議

現在錯誤會直接顯示，如果遇到問題：

1. **檢查完整的錯誤堆棧**: 不再被 try-except 隱藏
2. **驗證數據結構**: 確認 eval_history 包含正確的 option_instances 數據
3. **檢查模型狀態**: 確認 Option-Critic 模型正確加載
4. **查看日誌輸出**: 關注 INFO 和 WARNING 級別的日誌

## 使用方式

```bash
# 正常使用（每輪評估都生成，因為配置已改為 frequency=1）
python train_option_critic.py --enable_option_preview

# 自定義頻率
python train_option_critic.py --option_preview_frequency 5

# 禁用功能
python train_option_critic.py --no-enable_option_preview
```

---

**修復完成時間**: 2025-09-18
**修復狀態**: ✅ 完成並測試通過