# Option Preview Visualization 使用說明

## 功能概述

Option Preview Visualization 是為 Option-Critic 算法設計的可視化工具，用於展示每個 option 即將執行的 actions 統計信息。該功能會在每輪 inference 後，選擇分數最高的 episode，並為每個 option instance 生成預告圖片，最終合成 GIF 動畫。

## 功能特點

1. **Option 級別可視化**: 每個 option instance 生成一張預告圖
2. **3D 結構展示**: 左側顯示當前結構狀態和 Q-values
3. **Actions 統計**: 右側顯示該 option 即將執行的所有 actions 統計
4. **按樓層分類**: 統計各樓層各構件類型的修改次數
5. **自動 GIF 生成**: 將所有預告圖合成動畫

## 配置選項

在 `RL/option_critic/config.py` 中添加了以下配置：

```python
# Option preview visualization settings
parser.add_argument("--enable_option_preview", action="store_true", default=True,
                   help="Enable option preview visualization")
parser.add_argument("--option_preview_frequency", type=int, default=10,
                   help="Generate option preview every N evaluation rounds")
```

### 配置說明

- `--enable_option_preview`: 是否啟用 option preview 功能（默認啟用）
- `--option_preview_frequency`: 每 N 輪評估生成一次可視化（默認每 10 輪）

## 使用方法

### 1. 訓練時自動生成

使用 Option-Critic 訓練時會自動生成：

```bash
python train_option_critic.py \
    --epochs 100 \
    --enable_option_preview \
    --option_preview_frequency 10
```

### 2. 自定義頻率

如果希望更頻繁地生成可視化：

```bash
python train_option_critic.py \
    --epochs 100 \
    --option_preview_frequency 5  # 每 5 輪生成一次
```

### 3. 禁用功能

如果不需要可視化以節省計算資源：

```bash
python train_option_critic.py \
    --epochs 100 \
    --no-enable_option_preview  # 禁用功能
```

## 輸出結果

### 目錄結構

```
checkpoint_dir/
├── option_preview_process_round{round}_ep{episode}_score{score:.2f}/
│   ├── 0.png                           # Option instance 0 預告圖
│   ├── 1.png                           # Option instance 1 預告圖
│   ├── 2.png                           # Option instance 2 預告圖
│   ├── ...
│   ├── option_preview_round{round}_score{score:.2f}.gif  # 動畫
│   └── episode_summary.json            # Episode 摘要
└── ...
```

### 檔案說明

- **PNG 圖片**: 每個 option instance 的預告圖，包含結構狀態和統計信息
- **GIF 動畫**: 所有預告圖合成的動畫，每幀顯示時間 1.5 秒
- **JSON 摘要**: 包含 episode 信息和詳細統計數據

## 圖片內容解讀

### 左側面板：Q-values 3D 視圖
- 顯示當前結構的 Q-values 分佈
- 深色表示較高的 Q-values
- 節點為黑色球體，邊為結構構件

### 中間面板：結構 3D 視圖
- 顯示當前結構狀態
- 不同顏色表示不同的截面尺寸
- 顯示材料使用量和節省量統計

### 右側面板：Option 預告統計
- 標題顯示 Option 編號和 Instance ID
- 總計顯示該 option 將執行的 action 數量
- 按樓層分類統計各構件類型的修改次數

### 統計信息範例
```
Option 2 (Instance 1)
即將執行的動作

總計: 5 個動作

1F:
  • 內柱: 2 次
  • X向樑: 1 次

2F:
  • 外柱: 1 次
  • Z向樑: 1 次
```

## 技術實現

### 核心函數

1. **action_to_story_component**: 將 action index 映射到樓層和構件類型
2. **analyze_actions_by_story_component**: 分析 actions 的統計分佈
3. **prepare_episode_option_data**: 準備 episode 中的 option instance 數據
4. **visualize_option_preview_process**: 主要可視化函數

### 整合方式

- 在 `RL/option_critic/trainer.py` 的 `evaluate_and_save` 方法中調用
- 通過 mock agent 適配 Option-Critic 模型
- 自動處理模型權重加載和環境重置

## 測試

運行測試腳本驗證核心功能：

```bash
python test_option_preview_simple.py
```

測試包括：
- Action 映射邏輯正確性
- 統計分析功能準確性
- Option instance 數據處理

## 故障排除

### 常見問題

1. **ImportError**: 缺少依賴包
   - 解決方案: 安裝 sklearn, matplotlib, PIL 等依賴

2. **No option_instances data**: 評估數據中沒有 option instance 信息
   - 解決方案: 確認使用的是 Option-Critic 模型且評估正常運行

3. **Empty visualization**: 生成的圖片為空
   - 解決方案: 檢查模型權重是否正確加載，結構狀態是否有效

### 調試建議

1. 檢查日誌輸出中的 INFO 和 WARNING 信息
2. 確認 `option_preview_frequency` 設置是否合理
3. 驗證評估過程是否產生有效的 episode 數據

## 擴展功能

可以基於此框架擴展的功能：

1. **Option 行為分析**: 分析不同 option 的專業化程度
2. **策略演化追蹤**: 比較不同訓練階段的 option 行為變化
3. **實時預測**: 在線顯示當前 option 的預期行為
4. **互動式分析**: 支持用戶選擇特定 option 或樓層進行詳細分析

---

**注意**: 此功能專為 Option-Critic 算法設計，使用其他 RL 算法時需要相應的適配修改。