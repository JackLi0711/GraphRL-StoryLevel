# MuZero 實作說明

本專案已成功集成 MuZero 演算法，完全獨立於原有的 DQN 實作。

## 新增的檔案和功能

### 核心實作檔案

1. **`RL/model.py`** - 新增 `MuZeroNetwork` 類別
   - 包含 Representation Network (h)、Dynamics Network (g)、Prediction Network (f)
   - 支援 Taiwan 和 Japan 兩種模型類型
   - 使用 one-hot 編碼表示動作（遵循 MuZero 論文）

2. **`RL/mcts.py`** - 新增 MuZero 專用 MCTS
   - `MuZeroNode` 類別：MCTS 節點
   - `run_muzero_mcts` 函式：執行基於模型的搜尋
   - 使用 PUCT 公式計算 UCB 分數

3. **`RL/buffer.py`** - 新增 MuZero 回放緩衝區
   - `MuZeroGame` 類別：儲存完整遊戲軌跡
   - `MuZeroReplayBuffer` 類別：管理遊戲軌跡的緩衝區

4. **`RL/agent.py`** - 新增 `MuZeroAgent` 類別
   - 使用 MCTS 進行決策
   - 支援溫度參數控制探索
   - 完全獨立於現有 DQN Agent

5. **`train_muzero.py`** - 獨立的 MuZero 訓練腳本
   - 包含自我對弈 (Self-Play) 和網路訓練
   - 支援 N-step bootstrapped return
   - 完整的訓練迴圈實作

### 修改的檔案

6. **`inference.py`** - 新增 MuZero 支援
   - 新增 `--use_muzero` 參數
   - 支援載入 MuZero 模型進行推論

## 使用方法

### 1. 訓練 MuZero 模型

```bash
python train_muzero.py --num_episodes 1000 --muzero_num_simulations 50 --batch_size 32
```

主要參數：
- `--muzero_num_simulations`: MCTS 模擬次數（預設 50）
- `--muzero_unroll_steps`: 訓練時展開步數（預設 3）
- `--muzero_temperature`: 動作選擇溫度參數（預設 1.0）
- `--muzero_discount`: 折扣因子（預設 0.99）
- `--model_type`: 模型類型，Taiwan 或 Japan（預設 Taiwan）

### 2. 使用 MuZero 進行推論

```bash
python inference.py --use_muzero --trained_model_path ./Results/MuZero/muzero_model_final.pt
```

### 3. 繼續使用原有的 DQN

```bash
python inference.py  # 不加 --use_muzero 參數
```

## MuZero 核心特色

### 1. 學習環境模型
- **Representation Network**: 將真實觀察編碼成隱藏狀態
- **Dynamics Network**: 在隱藏狀態空間中進行想像推演
- **Prediction Network**: 預測策略和價值

### 2. 基於模型的規劃
- 使用學到的模型進行 MCTS 搜尋
- 完全在「想像」中進行前瞻性規劃
- 不需要與真實環境互動進行規劃

### 3. 自我對弈訓練
- 使用 MCTS 產生高品質的訓練數據
- N-step bootstrapped return 計算價值目標
- 結合實際獎勵和網路預測

## 技術細節

### N-step Return 計算
```
R_t = r_{t+1} + γ * r_{t+2} + ... + γ^(n-1) * r_{t+n} + γ^n * v(s_{t+n})
```

### 損失函數
MuZero 使用三種損失的組合：
1. **價值損失**: 網路預測價值 vs N-step return
2. **策略損失**: 網路預測策略 vs MCTS 搜尋策略
3. **獎勵損失**: 網路預測獎勵 vs 真實獎勵

### 動作編碼
遵循 MuZero 論文，使用 one-hot 編碼表示動作。

## 動態動作空間支援

MuZero 實作完全支援動態動作空間，能自動適應不同的 `structure_shape` 設定：

- **fixed**: 16 個動作 (4層 × 4種構件類型)
- **small_random**: 8-16 個動作 (2-4層 × 4種構件類型)  
- **random**: 16-28 個動作 (4-7層 × 4種構件類型)

### 技術實作

1. **最大動作數量設計**: 使用固定的最大動作數量 (32) 覆蓋所有可能情況
2. **動作遮罩**: 自動調整策略輸出只考慮當前有效的動作數量
3. **環境-Agent 協調**: 每次 `reset()` 時自動通知 Agent 當前的動作空間大小

## 與原有程式碼的兼容性

- ✅ 完全不影響現有 DQN 實作
- ✅ 可以在 MuZero 和 DQN 之間切換
- ✅ 共享相同的環境和視覺化工具
- ✅ 支援相同的結構設計參數
- ✅ 完全支援動態動作空間

## 未來改進方向

1. **動作編碼優化**: 如果動作空間過大，可考慮使用 embedding 替代 one-hot
2. **模型壓縮**: 對於大型結構，可考慮模型壓縮技術
3. **分散式訓練**: 實作分散式自我對弈加速訓練
4. **遷移學習**: 從 DQN 模型初始化 MuZero 網路

## 論文參考

- Schrittwieser, J., et al. "Mastering Atari, Go, chess and shogi by planning with a learned model." Nature 588.7839 (2020): 604-609. 