# Option-Critic 網路架構與更新方法詳細文檔

## 概述

本文檔詳細介紹 GraphRL_story_level 專案中 Option-Critic 演算法的網路架構設計與訓練更新方法。Option-Critic 是一種階層強化學習方法，結合了圖神經網路（Graph Neural Network, GNN）來處理結構工程設計問題。

## 1. 整體架構

### 1.1 核心組件

Option-Critic 系統由以下核心組件組成：

- **StateGNN**: 圖神經網路特徵提取器
- **Policy-over-Options (Q-network)**: 選項層級的價值函數
- **Intra-Option Policies**: 選項內部的原始動作策略
- **Option Termination Networks**: 選項終止概率預測

### 1.2 數學符號定義

- $s_t$: 時刻 $t$ 的狀態
- $o_t$: 時刻 $t$ 的當前選項
- $a_t$: 時刻 $t$ 的原始動作
- $\pi(a|s,o)$: 給定狀態 $s$ 和選項 $o$ 下的動作策略
- $\beta(s,o)$: 選項 $o$ 在狀態 $s$ 下的終止概率
- $Q(s,o)$: 狀態-選項價值函數
- $\gamma$: 折扣因子

## 2. 網路架構詳細設計

### 2.1 OptionCriticGNN 主網路

```python
class OptionCriticGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, 
                 member_state_dim, num_layers, num_actions, num_options, 
                 temperature=1.0, device='cpu'):
```

#### 2.1.1 StateGNN 特徵提取器

StateGNN 負責從圖結構數據中提取狀態特徵：

**輸入處理**：
- 節點特徵: $X \in \mathbb{R}^{N \times d_{node}}$
- 邊特徵: $E \in \mathbb{R}^{M \times d_{edge}}$
- 邊索引: $edge\_index \in \mathbb{R}^{2 \times M}$

**圖卷積層**：
```python
self.conv_layers = nn.ModuleList()
for i in range(num_layers):
    self.conv_layers.append(
        tgnn.GATConv(hidden_dim, hidden_dim, heads=4, 
                    concat=False, edge_dim=edge_feature_dim)
    )
```

**特徵聚合過程**：
1. 節點嵌入: $h_v^{(0)} = \text{MLP}(x_v)$
2. 多層 GAT 更新: $h_v^{(l+1)} = \text{GAT}(h_v^{(l)}, \{h_u^{(l)} : u \in N(v)\}, e_{uv})$
3. 邊嵌入: $h_e = \text{MLP}([h_i, h_j, e_{ij}])$
4. 故事層級聚合: $s_{story} = \text{GlobalAddPool}(h_e, story\_batch)$
5. 圖層級聚合: $s_{graph} = \text{GlobalAddPool}(h_e, batch)$

**最終狀態表示**：
$$s = [s_{story}, s_{graph}] \in \mathbb{R}^{member\_state\_dim \times 2}$$

#### 2.1.2 特徵處理層

```python
self.feature_processor = nn.Sequential(
    nn.Linear(gnn_output_dim, feature_dim),
    nn.ReLU(),
    nn.Linear(feature_dim, feature_dim),
    nn.ReLU()
)
```

將 GNN 輸出轉換為 Option-Critic 所需的特徵向量。

### 2.2 Policy-over-Options (Q-Network)

**Q-網路結構**：
```python
self.Q = nn.Linear(feature_dim, num_options)
```

**價值函數計算**：
$$Q(s, o) = W_Q \cdot \phi(s) + b_Q$$

其中 $\phi(s)$ 是經過特徵處理的狀態表示。

### 2.3 Option Termination Network

**終止網路結構**：
```python
self.terminations = nn.Linear(feature_dim, num_options)
```

**終止概率計算**：
$$\beta(s, o) = \sigma(W_\beta \cdot \phi(s) + b_\beta)$$

**初始化策略**：
```python
nn.init.constant_(self.terminations.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1
```

初始偏置設置為 -2.2，使得初始終止概率約為 0.1。

### 2.4 Intra-Option Policies

**參數化策略**：
```python
self.options_W = nn.Parameter(torch.zeros(num_options, feature_dim, num_actions))
self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
```

**動作選擇策略**：
$$\pi(a|s,o) = \text{Categorical}(\text{softmax}(\frac{W_o \cdot \phi(s) + b_o}{\tau}))$$

其中：
- $W_o, b_o$ 是選項 $o$ 的策略參數
- $\tau$ 是溫度參數
- 支持動作遮罩以確保只選擇有效動作

## 3. 損失函數與更新演算法

### 3.1 Critic Loss (價值函數損失)

#### 3.1.1 TD 目標計算

對於每個狀態-選項對 $(s_t, o_t)$，TD 目標為：

$$G_t = r_t + \gamma \cdot \text{mask}_t \cdot [(1 - \beta(s_{t+1}, o_t)) \cdot Q'(s_{t+1}, o_t) + \beta(s_{t+1}, o_t) \cdot \max_{o'} Q'(s_{t+1}, o')]$$

其中：
- $r_t$ 是即時獎勵
- $\text{mask}_t = 1 - done_t$ 是終止遮罩
- $Q'$ 是目標網路的價值函數
- $\beta(s_{t+1}, o_t)$ 是選項終止概率

#### 3.1.2 損失函數

$$\mathcal{L}_{\text{critic}} = \frac{1}{2} \mathbb{E}[(Q(s_t, o_t) - G_t)^2]$$

```python
def critic_loss(model, model_prime, data_batch, gamma=0.99):
    # ... 批次數據處理 ...
    
    # 計算目標值
    gt = rewards + masks * gamma * ((1 - next_options_term_prob) * next_Q_option + 
                                   next_options_term_prob * next_Q_max)
    
    # TD 誤差
    td_err = (Q_option - gt.detach()).pow(2).mul(0.5).mean()
    return td_err
```

### 3.2 Actor Loss (策略損失)

Actor 損失由兩部分組成：策略梯度損失和終止損失。

#### 3.2.1 策略梯度損失

**優勢函數**：
$$A_t = G_t - Q(s_t, o_t)$$

**策略損失**：
$$\mathcal{L}_{\text{policy}} = -\mathbb{E}[\log \pi(a_t|s_t, o_t) \cdot A_t] - \lambda_H \cdot H[\pi(\cdot|s_t, o_t)]$$

其中 $H[\pi]$ 是策略熵，$\lambda_H$ 是熵正則化係數。

#### 3.2.2 終止損失

**終止損失**：
$$\mathcal{L}_{\text{termination}} = \mathbb{E}[\beta(s_t, o_t) \cdot (Q(s_t, o_t) - \max_{o'} Q(s_t, o') + \lambda_{\text{reg}})]$$

其中 $\lambda_{\text{reg}}$ 是終止正則化係數，用於防止過度終止。

#### 3.2.3 總 Actor 損失

$$\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{termination}}$$

```python
def actor_loss(obs, option, logp, entropy, reward, done, next_obs, 
               model, model_prime, gamma=0.99, termination_reg=0.01, entropy_reg=0.01):
    # ... 狀態處理 ...
    
    # 計算目標值
    gt = reward + (1 - done) * gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + 
         next_option_term_prob * next_Q_max)
    
    # 終止損失
    termination_loss = option_term_prob * (Q[option].detach() - Q_max.detach() + termination_reg) * (1 - done)
    
    # 策略梯度損失
    advantage = gt.detach() - Q[option]
    policy_loss = -logp * advantage - entropy_reg * entropy
    
    return termination_loss + policy_loss
```

## 4. 訓練流程

### 4.1 選項執行 (Option Rollout)

```python
def rollout_option(structure, base_env, device, max_option_len, current_option, oc_model):
    # 1. 初始化狀態
    state = oc_model.get_state(*graph_data)
    
    for step in range(max_option_len):
        # 2. 選擇原始動作
        action, logp, entropy = oc_model.get_action(state, current_option, valid_mask)
        
        # 3. 執行動作
        structure, reward, step_pass, is_min_section, fail_reason = apply_primitive_action(base_env, structure, action)
        
        # 4. 記錄轉換
        step_transitions.append({
            "obs": graph_data, "action": action, "logp": logp, 
            "entropy": entropy, "reward": reward, "done": done, 
            "next_obs": next_graph_data, "option": current_option
        })
        
        # 5. 檢查選項終止
        option_termination, _ = oc_model.predict_option_termination(next_state, current_option)
        if option_termination or is_min_section:
            break
            
        state = next_state
```

### 4.2 更新程序

#### 4.2.1 Actor 更新（每步更新）

```python
# 對每個步驟轉換進行 Actor 更新
for tr in step_transitions:
    a_loss = actor_loss(tr["obs"], tr["option"], tr["logp"], tr["entropy"], 
                       tr["reward"], tr["done"], tr["next_obs"], 
                       oc, oc_prime, gamma, termination_reg, entropy_reg)
    actor_optimizer.zero_grad()
    a_loss.backward()
    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=grad_clip)
    actor_optimizer.step()
```

#### 4.2.2 Critic 更新（定期更新）

```python
# 定期從經驗回放緩衝區採樣進行 Critic 更新
if len(buffer) > batch_size and (steps % update_frequency == 0):
    data_batch = buffer.sample(batch_size)
    c_loss = critic_loss(oc, oc_prime, data_batch, gamma)
    critic_optimizer.zero_grad()
    c_loss.backward()
    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=grad_clip)
    critic_optimizer.step()
```

#### 4.2.3 目標網路更新

```python
# 定期同步目標網路
if steps % freeze_interval == 0:
    oc_prime.load_state_dict(oc.state_dict())
```

## 5. 探索策略

### 5.1 選項選擇

**ε-貪婪選項選擇**：
$$o_t = \begin{cases} 
\text{random choice from } \{0, 1, ..., |\mathcal{O}|-1\} & \text{if } \xi < \epsilon \\
\arg\max_{o} Q(s_t, o) & \text{otherwise}
\end{cases}$$

### 5.2 動作選擇

**Categorical 分佈採樣**：
- 訓練時：從 $\pi(a|s_t, o_t)$ 採樣
- 測試時：選擇 $\arg\max_a \pi(a|s_t, o_t)$

### 5.3 探索參數調度

```python
@property
def epsilon(self):
    if not self.testing:
        eps = calculate_epsilon(self.num_steps, self.eps_start, self.eps_min, self.eps_decay)
        self.num_steps += 1
    else:
        eps = self.eps_test
    return eps
```

## 6. 網路初始化

### 6.1 選項策略初始化

```python
def _initialize_parameters(self):
    # 選項策略參數使用小隨機值初始化
    nn.init.normal_(self.options_W, mean=0.0, std=0.01)
    nn.init.zeros_(self.options_b)
    
    # Xavier 初始化其他層
    for module in [self.feature_processor, self.Q]:
        if hasattr(module, 'weight'):
            nn.init.xavier_uniform_(module.weight)
    
    # 終止網路特殊初始化（低初始終止概率）
    nn.init.xavier_uniform_(self.terminations.weight)
    nn.init.constant_(self.terminations.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1
```

### 6.2 學習率設置

```python
# 使用不同組件的差異化學習率
gnn_lr = actor_lr * 0.1  # GNN 使用較低學習率

actor_optimizer = optim.Adam([
    {"params": actor_params, "lr": actor_lr},
    {"params": state_gnn_params, "lr": gnn_lr},
    {"params": feature_params, "lr": actor_lr},
])

critic_optimizer = optim.Adam([
    {"params": critic_params, "lr": critic_lr},
    {"params": state_gnn_params, "lr": gnn_lr},  
    {"params": feature_params, "lr": critic_lr},
])
```

## 7. 關鍵特性與優化

### 7.1 動作遮罩機制

```python
def get_action(self, state, option, valid_actions_mask=None):
    # 計算動作機率
    logits = state @ self.options_W[option] + self.options_b[option]
    
    # 應用動作遮罩
    if valid_actions_mask is not None:
        logits[~valid_actions_mask] = -1e8
    
    # 採樣動作
    action_dist = Categorical((logits / self.temperature).softmax(dim=-1))
    return action_dist.sample(), action_dist.log_prob(action), action_dist.entropy()
```

### 7.2 梯度裁剪

防止梯度爆炸：
```python
if grad_clip is not None and grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=grad_clip)
```

### 7.3 經驗回放

使用經驗回放緩衝區儲存步驟級別的轉換：
```python
class ReplayBuffer:
    def push(self, obs, option, reward, next_obs, done):
        # 儲存轉換
        
    def sample(self, batch_size):
        # 採樣批次數據
```

## 8. 評估與監控

### 8.1 終止概率記錄

```python
class TerminationProbabilityLogger:
    def log_termination_prediction(self, option, termination_probs, termination_decision):
        # 記錄終止概率決策
        
    def end_episode(self):
        # 計算回合統計
```

### 8.2 性能指標

- **回合獎勵**: 累積獎勵
- **回合長度**: 選項數量
- **成功率**: 達到最小截面的比例
- **選項長度**: 平均原始動作數
- **終止概率**: 各選項的終止行為

## 9. 超參數配置

### 9.1 網路參數

- `num_options`: 4 (選項數量)
- `max_option_len`: 16 (最大選項長度)
- `hidden_dim`: 128 (隱藏層維度)
- `temperature`: 1.0 (動作選擇溫度)

### 9.2 訓練參數

- `lr`: 1e-5 (學習率)
- `gamma`: 0.99 (折扣因子)  
- `batch_size`: 32 (批次大小)
- `update_frequency`: 4 (更新頻率)
- `freeze_interval`: 2000 (目標網路更新間隔)

### 9.3 正則化參數

- `termination_reg`: 0.01 (終止正則化)
- `entropy_reg`: 0.01 (熵正則化)
- `grad_clip`: 10.0 (梯度裁剪)

## 10. 總結

這個 Option-Critic 實現將階層強化學習與圖神經網路相結合，專門用於結構工程設計優化問題。主要特點包括：

1. **圖結構處理**: 使用 StateGNN 處理複雜的結構圖數據
2. **階層決策**: 通過選項和原始動作的兩層決策結構
3. **終止學習**: 自動學習選項終止時機
4. **約束處理**: 支持動作遮罩處理工程約束
5. **穩定訓練**: 使用目標網路、經驗回放和梯度裁剪

該架構能夠有效學習長期策略並處理結構設計中的複雜約束條件。