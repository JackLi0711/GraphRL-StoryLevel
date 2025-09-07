# Termination Function 差異化學習率實作總結

## 修改概述

實作了 termination function 與 policy parameters 的差異化學習率，其中 `termination_lr = 0.01 * actor_lr`，以提供更穩定的 termination function 學習過程。

## 修改內容

### 1. 添加命令行參數 (train_option_critic.py:725)

```python
parser.add_argument("--termination_lr_ratio", type=float, default=0.01, 
                    help="Termination learning rate as ratio of actor_lr (termination_lr = actor_lr * ratio)")
```

### 2. 分離參數組 (train_option_critic.py:832-834)

**修改前:**
```python
actor_params = [oc.options_W, oc.options_b] + termination_params
```

**修改後:**
```python
termination_params = [p for n, p in oc.named_parameters() if n.startswith('terminations')]
policy_params = [oc.options_W, oc.options_b]  # 純策略參數 (與 termination 分離)
```

### 3. 差異化優化器配置 (train_option_critic.py:844-851)

```python
gnn_lr = args.actor_lr * 0.1  # StateGNN 較低學習率
termination_lr = args.actor_lr * args.termination_lr_ratio  # Termination 差異化學習率

actor_optimizer = optim.Adam([
    {"params": policy_params, "lr": args.actor_lr},        # 策略參數標準學習率
    {"params": termination_params, "lr": termination_lr},   # 終止參數差異化學習率
    {"params": state_gnn_params, "lr": gnn_lr},
    {"params": feature_params, "lr": args.actor_lr},
])
```

### 4. 學習率配置日誌 (train_option_critic.py:858-863)

```python
logger.info(f"Learning rate configuration:")
logger.info(f"  Policy parameters: {args.actor_lr}")
logger.info(f"  Termination parameters: {termination_lr} (ratio: {args.termination_lr_ratio})")
logger.info(f"  Critic parameters: {args.critic_lr}")
logger.info(f"  StateGNN parameters: {gnn_lr}")
```

## 預期效果

### 1. 更穩定的 Termination Function 學習
- Termination function 使用較小的學習率 (默認為 actor_lr 的 1%)
- 減少 termination 概率的劇烈波動
- 提供更平滑的 option switching 行為

### 2. 靈活的超參數調整
- 可通過 `--termination_lr_ratio` 調整比例
- 保持與原始 Option-Critic 算法的理論一致性
- 支持不同實驗需求的學習率配置

### 3. 更好的訓練監控
- 日誌輸出詳細的學習率配置
- 便於分析各組件的學習進度
- 提供清晰的調試信息

## 使用方式

### 默認配置 (termination_lr = 0.01 * actor_lr)
```bash
python3 train_option_critic.py
```

### 自定義 termination learning rate 比例
```bash
# 更保守的學習率 (0.5%)
python3 train_option_critic.py --termination_lr_ratio 0.005

# 較激進的學習率 (2%)  
python3 train_option_critic.py --termination_lr_ratio 0.02
```

### 與其他參數結合使用
```bash
python3 train_option_critic.py --actor_lr 5e-5 --termination_lr_ratio 0.01
# 結果: policy_lr = 5e-5, termination_lr = 5e-7
```

## 理論依據

### Option-Critic 原文公式
- **Policy gradient**: `∇_θ J = E[∇_θ log π_θ(a|s,ω) A^Ω(s,a,ω)]`
- **Termination gradient**: `∇_η J = -E[∇_η β_ω(s;η) A^Ω(s,ω)]`

### 差異化學習率的合理性
1. **Termination function 更敏感**: 小的變化可能導致 option switching 行為劇變
2. **需要更穩定的學習**: 保守的學習率有助於 termination 概率的平滑收斂
3. **保持理論一致性**: 仍在統一的 actor-critic 框架下，只是調整學習步長

## 驗證結果

✅ 語法檢查通過  
✅ 邏輯驗證通過  
✅ 學習率計算正確  
✅ 參數組分配正確  
✅ 命令行參數功能正常  

## 後續建議

1. **實驗比較**: 對比不同 `termination_lr_ratio` 值的訓練效果
2. **監控指標**: 特別關注 termination 概率的變化趨勢和 option length 統計
3. **進一步調優**: 根據實際訓練結果調整最佳的學習率比例

## 修改文件

- `train_option_critic.py`: 主要修改文件
- `MODIFICATION_SUMMARY.md`: 本總結文檔 (新增)

---

**修改完成時間**: 2025-09-07  
**修改者**: Claude  
**驗證狀態**: ✅ 通過