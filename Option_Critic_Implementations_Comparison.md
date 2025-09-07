# Option-Critic 實作比較：GraphRL vs PyTorch 經典實作

## 概述

本文檔詳細比較了 GraphRL_story_level 專案中的 Option-Critic 實作與 option-critic-pytorch 經典實作之間的關鍵差異。兩個實作都基於相同的理論基礎，但針對不同的應用場景進行了特化設計。

## 目錄

1. [整體架構對比](#1-整體架構對比)
2. [特徵提取器差異](#2-特徵提取器差異)
3. [網路設計對比](#3-網路設計對比)
4. [數學公式差異](#4-數學公式差異)
5. [訓練更新方法對比](#5-訓練更新方法對比)
6. [損失函數實作差異](#6-損失函數實作差異)
7. [選項執行機制對比](#7-選項執行機制對比)
8. [約束處理能力](#8-約束處理能力)
9. [優化器與學習率策略](#9-優化器與學習率策略)
10. [經驗回放機制](#10-經驗回放機制)
11. [評估與監控](#11-評估與監控)
12. [應用場景差異](#12-應用場景差異)
13. [總結與建議](#13-總結與建議)

---

## 1. 整體架構對比

### 1.1 GraphRL Option-Critic 架構

```
結構圖輸入 → StateGNN (GAT) → 特徵處理層 → {Q網路, 終止網路, 選項策略}
     ↓            ↓                ↓                    ↓
圖神經網路    多頭注意力機制    全連接處理器        參數化策略矩陣
```

### 1.2 PyTorch 經典架構

```
觀測輸入 → CNN/FC特徵提取 → {Q網路, 終止網路, 選項策略}
   ↓           ↓                    ↓
標準張量   卷積/全連接        線性層輸出頭
```

### 1.3 關鍵差異

| 組件 | GraphRL 實作 | PyTorch 經典實作 |
|------|-------------|------------------|
| **輸入類型** | 圖結構數據 (節點+邊) | 標準向量/圖像 |
| **特徵提取器** | StateGNN (GAT-based) | CNN/全連接層 |
| **狀態表示** | 動態維度圖特徵 | 固定維度向量 |
| **批次處理** | 圖批次聚合 | 標準張量批次 |
| **約束處理** | 內建動作遮罩 | 不支持 |

---

## 2. 特徵提取器差異

### 2.1 GraphRL StateGNN 設計

#### 2.1.1 圖注意力機制

```python
# GraphRL: 使用多層 GAT 進行特徵提取
self.conv_layers = nn.ModuleList()
for i in range(num_layers):
    self.conv_layers.append(
        tgnn.GATConv(hidden_dim, hidden_dim, heads=4, 
                    concat=False, edge_dim=edge_feature_dim)
    )
```

**數學表示**：
$$h_v^{(l+1)} = \text{GAT}(h_v^{(l)}, \{h_u^{(l)} : u \in N(v)\}, e_{uv})$$

其中注意力權重為：
$$\alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T[W h_u \| W h_v \| W_e e_{uv}]))}{\sum_{k \in N(v)} \exp(\text{LeakyReLU}(a^T[W h_k \| W h_v \| W_e e_{kv}]))}$$

#### 2.1.2 邊嵌入與聚合

```python
# GraphRL: 邊嵌入計算
index_i, index_j = edge_index
node_embedding_i, node_embedding_j = node_embedding[index_i], node_embedding[index_j]
edge_input = torch.cat([node_embedding_i[::2], node_embedding_j[::2], edge_attr[::2]], dim=1)
edge_embedding = self.edge_mlp(edge_input)
```

**數學表示**：
$$e_{ij} = \text{MLP}([h_i, h_j, \text{edge\_attr}_{ij}])$$

#### 2.1.3 階層聚合

```python
# GraphRL: 多層次聚合
graph_embedding = global_add_pool(edge_embedding, batch)        # 圖層級
story_embedding = global_add_pool(edge_embedding, story_batch)  # 故事層級
state = self._state_global_aggregation(story_embedding, graph_embedding, structure_story_ptr)
```

**最終狀態表示**：
$$\phi(s) = [s_{\text{story}}, s_{\text{graph}}] \in \mathbb{R}^{\text{member\_state\_dim} \times 2}$$

### 2.2 PyTorch 經典特徵提取

#### 2.2.1 卷積特徵提取 (Atari)

```python
# PyTorch 經典: CNN 特徵提取
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(7*7*64, 512),
    nn.ReLU()
)
```

**特徵提取流程**：
$$\phi(s) = \text{Linear}(\text{Flatten}(\text{Conv3}(\text{Conv2}(\text{Conv1}(s)))))$$

#### 2.2.2 全連接特徵提取 (簡單環境)

```python
# PyTorch 經典: 全連接特徵提取
self.features = nn.Sequential(
    nn.Linear(in_features, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU()
)
```

**數學表示**：
$$\phi(s) = \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot s + b_1) + b_2)$$

### 2.3 特徵提取器比較總結

| 方面 | GraphRL StateGNN | PyTorch 經典 |
|------|------------------|-------------|
| **複雜度** | 高 (多層 GAT + 聚合) | 中等 (CNN/FC) |
| **參數量** | 動態 (取決於圖大小) | 固定 |
| **計算成本** | 高 (圖操作) | 低 (矩陣運算) |
| **表達能力** | 強 (結構關係) | 一般 (局部特徵) |
| **可解釋性** | 高 (注意力權重) | 低 |

---

## 3. 網路設計對比

### 3.1 輸出頭部網路

#### 3.1.1 GraphRL 設計

```python
# GraphRL: 複雜的特徵處理 + 輸出層
self.feature_processor = nn.Sequential(
    nn.Linear(gnn_output_dim, feature_dim),
    nn.ReLU(),
    nn.Linear(feature_dim, feature_dim),
    nn.ReLU()
)

self.Q = nn.Linear(feature_dim, num_options)                    # 512 → num_options
self.terminations = nn.Linear(feature_dim, num_options)        # 512 → num_options
self.options_W = nn.Parameter(torch.zeros(num_options, feature_dim, num_actions))  # [num_options, 512, num_actions]
self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
```

#### 3.1.2 PyTorch 經典設計

```python
# PyTorch 經典: 直接輸出層
self.Q = nn.Linear(512, num_options)                          # 512 → num_options
self.terminations = nn.Linear(512, num_options)              # 512 → num_options
self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))  # [num_options, 512, num_actions]
self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
```

### 3.2 參數初始化策略

#### 3.2.1 GraphRL 初始化

```python
def _initialize_parameters(self):
    # 選項策略參數 - 小隨機初始化
    nn.init.normal_(self.options_W, mean=0.0, std=0.01)
    nn.init.zeros_(self.options_b)
    
    # Xavier 初始化其他層
    for module in [self.feature_processor, self.Q]:
        if hasattr(module, 'weight'):
            nn.init.xavier_uniform_(module.weight)
    
    # 終止網路特殊初始化 - 低初始終止概率
    nn.init.xavier_uniform_(self.terminations.weight)
    nn.init.constant_(self.terminations.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1
```

#### 3.2.2 PyTorch 經典初始化

```python
# PyTorch 經典: 默認初始化 (無特殊策略)
# 使用 PyTorch 默認的 Xavier/Kaiming 初始化
```

### 3.3 初始化策略比較

| 組件 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **選項策略** | 小隨機初始化 (std=0.01) | 默認初始化 |
| **Q網路** | Xavier uniform | 默認 |
| **終止網路** | 特殊偏置 (-2.2) | 默認 |
| **特徵層** | Xavier uniform | 默認 |

---

## 4. 數學公式差異

### 4.1 狀態表示計算

#### 4.1.1 GraphRL 狀態計算

```python
def get_state(self, graph_x, graph_edge_index, graph_edge_attr, story_batch, structure_story_ptr):
    # 複雜的圖神經網路前向傳播
    gnn_features = self.state_gnn(x=graph_x, edge_index=graph_edge_index, 
                                 edge_attr=graph_edge_attr, batch=None, 
                                 story_batch=story_batch, structure_story_ptr=structure_story_ptr)
    processed_features = self.feature_processor(gnn_features)
    
    # 全局池化聚合
    if processed_features.dim() == 2 and processed_features.shape[0] > 1:
        state = processed_features.mean(dim=0, keepdim=True)
    else:
        state = processed_features
    return state
```

**數學表示**：
$$\phi(s) = \text{GlobalMeanPool}(\text{FeatureProcessor}(\text{StateGNN}(G)))$$

其中 $G = (V, E, X_v, X_e)$ 是帶屬性的圖。

#### 4.1.2 PyTorch 經典狀態計算

```python
def get_state(self, obs):
    if obs.ndim < 4:
        obs = obs.unsqueeze(0)
    obs = obs.to(self.device)
    state = self.features(obs)    # 直接特徵提取
    return state
```

**數學表示**：
$$\phi(s) = f_{\text{features}}(s)$$

### 4.2 動作選擇策略

#### 4.2.1 GraphRL 動作選擇 (支援動作遮罩)

```python
def get_action(self, state, option, valid_actions_mask=None):
    # 處理批次維度
    if state.dim() == 2 and state.shape[0] > 1:
        state_for_action = state.mean(dim=0)
    else:
        state_for_action = state[0] if state.dim() == 2 else state
    
    # 計算動作機率
    logits = state_for_action @ self.options_W[option] + self.options_b[option]
    
    # 應用動作遮罩
    if valid_actions_mask is not None:
        logits[~valid_actions_mask] = -1e8
    
    # 採樣動作
    action_dist = Categorical((logits / self.temperature).softmax(dim=-1))
    action = action_dist.sample()
    return action.item(), action_dist.log_prob(action), action_dist.entropy()
```

**數學表示 (含遮罩)**：
$$\pi_{\text{masked}}(a|s,o) = \text{Categorical}(\text{softmax}(\frac{\text{mask}(\text{logits})}{\tau}))$$

其中：
$$\text{mask}(\text{logits})_i = \begin{cases} 
\text{logits}_i & \text{if action } i \text{ is valid} \\
-10^8 & \text{otherwise}
\end{cases}$$

#### 4.2.2 PyTorch 經典動作選擇 (無遮罩)

```python
def get_action(self, state, option):
    logits = state.data @ self.options_W[option] + self.options_b[option]
    action_dist = Categorical((logits / self.temperature).softmax(dim=-1))
    
    action = action_dist.sample()
    logp = action_dist.log_prob(action)
    entropy = action_dist.entropy()
    
    return action.item(), logp, entropy
```

**數學表示**：
$$\pi(a|s,o) = \text{Categorical}(\text{softmax}(\frac{W_o \cdot \phi(s) + b_o}{\tau}))$$

### 4.3 選項終止決策

#### 4.3.1 GraphRL 終止決策

```python
def predict_option_termination(self, state, current_option):
    termination_probs = self.get_terminations(state)
    
    # 處理批次維度
    if state.dim() == 2:
        termination_prob = termination_probs[0, current_option] if len(termination_probs) > 1 else termination_probs.mean(dim=0)[current_option]
    else:
        termination_prob = termination_probs[current_option]
    
    # 確定性 vs 隨機決策
    if self.testing:
        option_termination = termination_prob > 0.5
    else:
        option_termination = Bernoulli(termination_prob).sample()
    
    # 選擇下個選項
    Q = self.get_Q(state)
    next_option = Q.argmax(dim=-1)
    
    return bool(option_termination.item()), int(next_option.item())
```

#### 4.3.2 PyTorch 經典終止決策

```python
def predict_option_termination(self, state, current_option):
    termination = self.terminations(state)[:, current_option].sigmoid()
    option_termination = Bernoulli(termination).sample()
    
    Q = self.get_Q(state)
    next_option = Q.argmax(dim=-1)
    return bool(option_termination.item()), next_option.item()
```

**數學表示**：
$$\text{terminate} \sim \text{Bernoulli}(\sigma(W_\beta \cdot \phi(s) + b_\beta)_{o_t})$$
$$o_{t+1} = \arg\max_{o'} Q(s_t, o')$$

---

## 5. 訓練更新方法對比

### 5.1 GraphRL 更新機制

#### 5.1.1 Actor 更新 (每步更新)

```python
# GraphRL: 對每個 step transition 進行 actor 更新
for i, tr in enumerate(step_transitions):
    a_loss = actor_loss(tr["obs"], tr["option"], tr["logp"], tr["entropy"], 
                       tr["reward"], tr["done"], tr["next_obs"], 
                       oc, oc_prime, gamma, termination_reg, entropy_reg)
    
    actor_optimizer.zero_grad()
    a_loss.backward()
    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=grad_clip)  # 梯度裁剪
    actor_optimizer.step()
```

#### 5.1.2 Critic 更新 (定期批次)

```python
# GraphRL: 定期從經驗回放採樣更新
if len(buffer) > batch_size and (steps % update_frequency == 0):
    data_batch = buffer.sample(batch_size)
    c_loss = critic_loss(oc, oc_prime, data_batch, gamma)
    
    critic_optimizer.zero_grad()
    c_loss.backward()
    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=grad_clip)
    critic_optimizer.step()
```

### 5.2 PyTorch 經典更新機制

#### 5.2.1 混合更新策略

```python
# PyTorch 經典: Actor + Critic 同時更新
if len(buffer) > batch_size:
    # Actor 損失 (每步)
    actor_loss = actor_loss_fn(obs, current_option, logp, entropy,
        reward, done, next_obs, option_critic, option_critic_prime, args)
    loss = actor_loss
    
    # Critic 損失 (定期)
    if steps % update_frequency == 0:
        data_batch = buffer.sample(batch_size)
        critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
        loss += critic_loss
    
    # 聯合優化
    optim.zero_grad()
    loss.backward()
    optim.step()
```

### 5.3 更新頻率對比

| 更新類型 | GraphRL | PyTorch 經典 |
|----------|---------|-------------|
| **Actor 更新** | 每步多次 (每個 transition) | 每步單次 |
| **Critic 更新** | 定期 (每 4 步) | 定期 (每 4 步) |
| **目標網路更新** | 定期同步 (每 2000 步) | 定期同步 (每 200 步) |
| **梯度裁剪** | 支援 (max_norm=10.0) | 不支援 |
| **優化器數量** | 2 個 (Actor, Critic 分離) | 1 個 (聯合優化) |

---

## 6. 損失函數實作差異

### 6.1 Critic Loss 對比

#### 6.1.1 GraphRL Critic Loss

```python
def critic_loss(model, model_prime, data_batch, gamma=0.99):
    obs, options, rewards, next_obs, dones = data_batch
    batch_size = len(options)
    
    # 複雜的圖數據處理
    for i in range(batch_size):
        # 處理單個圖樣本
        if graph_edge_index is not None:
            single_graph_x = graph_x[i:i+1]
            # ... 複雜的圖數據解包
            state = model.get_state(single_graph_x.squeeze(0), single_edge_index.squeeze(0), ...)
        
        Q_single = model.get_Q(state)
        Q_list.append(Q_single)
    
    # 堆疊結果並計算 TD 誤差
    Q = torch.stack(Q_list, dim=0)
    next_Q_prime = torch.stack(next_Q_prime_list, dim=0)
    next_termination_probs = torch.stack(next_termination_probs_list, dim=0).detach()
    
    # TD 目標計算
    gt = rewards + masks * gamma * ((1 - next_options_term_prob) * next_Q_option + 
                                   next_options_term_prob * next_Q_max)
    
    td_err = (Q_option - gt.detach()).pow(2).mul(0.5).mean()
    return td_err
```

#### 6.1.2 PyTorch 經典 Critic Loss

```python
def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    
    # 簡單的批次處理
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q = model.get_Q(states)
    
    # 目標網路計算
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime = model_prime.get_Q(next_states_prime)
    
    # 終止概率
    next_states = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]
    
    # TD 目標
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + 
         next_options_term_prob * next_Q_prime.max(dim=-1)[0])
    
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err
```

### 6.2 Actor Loss 對比

#### 6.2.1 GraphRL Actor Loss

```python
def actor_loss(obs, option, logp, entropy, reward, done, next_obs, 
               model, model_prime, gamma=0.99, termination_reg=0.01, entropy_reg=0.01):
    # 複雜的圖狀態處理
    if isinstance(obs, (tuple, list)) and len(obs) >= 4:
        graph_x, graph_edge_index, graph_edge_attr, story_batch = obs[:4]
        state = model.get_state(graph_x, graph_edge_index, graph_edge_attr, story_batch)
        # ... 類似的 next_state 處理
    
    # 終止概率計算
    option_term_prob = model.get_terminations(state)
    if option_term_prob.dim() > 1:
        option_term_prob = option_term_prob.mean(dim=0)
    option_term_prob = option_term_prob[option]
    
    # Q 值計算與維度處理
    Q = model.get_Q(state).detach().squeeze()
    if Q.dim() == 0:
        Q = Q.unsqueeze(0)
    
    # 計算目標與損失
    gt = reward + (1 - done) * gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + 
         next_option_term_prob * next_Q_max)
    
    # 終止損失
    termination_loss = option_term_prob * (Q[option].detach() - Q_max.detach() + termination_reg) * (1 - done)
    
    # 策略損失
    advantage = gt.detach() - Q[option]
    policy_loss = -logp * advantage - entropy_reg * entropy
    
    return termination_loss + policy_loss
```

#### 6.2.2 PyTorch 經典 Actor Loss

```python
def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    # 簡單的狀態處理
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))
    
    # 終止概率
    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()
    
    # Q 值
    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()
    
    # 目標計算
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + 
         next_option_term_prob * next_Q_prime.max(dim=-1)[0])
    
    # 損失計算
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    
    return termination_loss + policy_loss
```

### 6.3 損失函數比較

| 方面 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **批次處理** | 逐樣本循環處理 | 向量化批次處理 |
| **維度處理** | 複雜 (多層檢查) | 簡單 (固定維度) |
| **計算效率** | 低 (循環) | 高 (向量化) |
| **記憶體使用** | 高 (中間結果存儲) | 低 |
| **調試難度** | 高 (複雜邏輯) | 低 |

---

## 7. 選項執行機制對比

### 7.1 GraphRL 選項執行 (Option Rollout)

```python
def rollout_option(structure, base_env, device, max_option_len, current_option, oc_model, ...):
    """複雜的選項執行機制，支援多步原始動作序列"""
    
    step_transitions = []  # 存儲每步轉換
    
    # 安全計數器防止無限循環
    safety_counter = 0
    max_safety_iterations = max_option_len + 10
    
    for length in range(max_option_len):
        safety_counter += 1
        if safety_counter > max_safety_iterations:
            termination_reason = "safety_timeout"
            break
        
        # 創建有效動作遮罩
        try:
            invalid_actions = already_minimum | restricted_actions
            valid_mask = torch.ones(num_actions, dtype=torch.bool, device=device)
            for invalid_action in invalid_actions:
                if 0 <= invalid_action < num_actions:
                    valid_mask[invalid_action] = False
        except Exception as e:
            valid_mask = None
        
        # 選項內動作選擇
        action, logp, entropy = oc_model.get_action(state, current_option, valid_mask)
        
        # 執行動作並記錄轉換
        structure, step_reward, step_pass, is_min_section, fail_reason = apply_primitive_action(base_env, structure, action)
        
        step_transitions.append({
            "obs": graph_data, "action": action, "logp": logp, "entropy": entropy,
            "reward": step_reward, "done": False, "next_obs": next_graph_data, "option": current_option
        })
        
        # 選項終止檢查
        if is_min_section:
            termination_reason = "minimum_section"
            episode_done = True
            break
            
        # Beta 終止檢查
        termination_probs = oc_model.get_terminations(next_state)
        option_termination, _ = oc_model.predict_option_termination(next_state, current_option)
        
        if option_termination:
            termination_reason = "beta"
            break
    
    return structure, next_state, option_done, episode_done, stats, step_transitions, termination_reason
```

### 7.2 PyTorch 經典選項執行

```python
# PyTorch 經典: 簡單的單步執行
while not done and ep_steps < max_steps_ep:
    epsilon = option_critic.epsilon
    
    # 選項選擇
    if option_termination:
        current_option = np.random.choice(num_options) if np.random.rand() < epsilon else greedy_option
    
    # 動作選擇與執行
    action, logp, entropy = option_critic.get_action(state, current_option)
    next_obs, reward, done, _ = env.step(action)
    
    # 存儲經驗
    buffer.push(obs, current_option, reward, next_obs, done)
    
    # 更新網路
    if len(buffer) > batch_size:
        actor_loss = actor_loss_fn(obs, current_option, logp, entropy, reward, done, next_obs, ...)
        # ... 更新邏輯
    
    # 選項終止決策
    state = option_critic.get_state(to_tensor(next_obs))
    option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
```

### 7.3 選項執行對比

| 特性 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **執行模式** | 多步選項序列 | 單步動作執行 |
| **轉換記錄** | 詳細步驟級記錄 | 簡單經驗存儲 |
| **安全機制** | 安全計數器 | 回合步數限制 |
| **約束處理** | 動態動作遮罩 | 無約束處理 |
| **終止條件** | 多種 (β, 長度, 約束) | 單一 (β 終止) |
| **狀態管理** | 複雜 (圖數據) | 簡單 (向量) |

---

## 8. 約束處理能力

### 8.1 GraphRL 約束處理

#### 8.1.1 動作約束識別

```python
# GraphRL: 複雜的約束檢查
already_minimum = set(getattr(structure, 'already_minimum_section_story_indexes', []) or [])
restricted_actions = set()
if hasattr(structure, 'restrict_action_space'):
    restricted = structure.restrict_action_space()
    if restricted is not None:
        restricted_actions.update(restricted)

# 合併所有無效動作
invalid_actions = already_minimum | restricted_actions
```

#### 8.1.2 動作遮罩實作

```python
# GraphRL: 動態動作遮罩
def get_action(self, state, option, valid_actions_mask=None):
    logits = state_for_action @ self.options_W[option] + self.options_b[option]
    
    # 應用動作遮罩
    if valid_actions_mask is not None:
        if valid_actions_mask.device != logits.device:
            valid_actions_mask = valid_actions_mask.to(logits.device)
        logits = logits.clone()
        logits[~valid_actions_mask] = -1e8  # 設置無效動作為極小值
    
    action_dist = (logits / self.temperature).softmax(dim=-1)
    return action_dist.sample(), action_dist.log_prob(action), action_dist.entropy()
```

#### 8.1.3 約束違反處理

```python
# GraphRL: 約束失敗的懲罰機制
if not passed:
    episode_done = True
    # 對導致失敗的最後一步應用懲罰
    if len(step_transitions) > 0:
        step_transitions[-1]["reward"] = -1000.0  # 懲罰獎勵
        step_transitions[-1]["done"] = True
```

### 8.2 PyTorch 經典約束處理

```python
# PyTorch 經典: 無內建約束處理機制
def get_action(self, state, option):
    logits = state.data @ self.options_W[option] + self.options_b[option]
    action_dist = (logits / self.temperature).softmax(dim=-1)
    
    # 直接從所有動作中採樣，無約束檢查
    action = action_dist.sample()
    return action.item(), action_dist.log_prob(action), action_dist.entropy()
```

### 8.3 約束處理能力對比

| 方面 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **動作遮罩** | ✅ 完整支援 | ❌ 不支援 |
| **約束檢查** | ✅ 多層次檢查 | ❌ 無檢查 |
| **失敗處理** | ✅ 懲罰機制 | ❌ 無處理 |
| **動態約束** | ✅ 運行時更新 | ❌ 不支援 |
| **工程適用性** | ✅ 高 (結構約束) | ❌ 低 |

---

## 9. 優化器與學習率策略

### 9.1 GraphRL 優化策略

#### 9.1.1 分離式優化器

```python
# GraphRL: Actor 和 Critic 使用分離的優化器
termination_params = [p for n, p in oc.named_parameters() if n.startswith('terminations')]
actor_params = [oc.options_W, oc.options_b] + termination_params
critic_params = [p for n, p in oc.named_parameters() if n.startswith('Q')]
state_gnn_params = [p for n, p in oc.named_parameters() if n.startswith('state_gnn')]
feature_params = [p for n, p in oc.named_parameters() if n.startswith('feature_processor')]

# 差異化學習率
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

#### 9.1.2 梯度處理

```python
# GraphRL: 梯度裁剪
if grad_clip is not None and grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(oc.parameters(), max_norm=grad_clip)
```

### 9.2 PyTorch 經典優化策略

```python
# PyTorch 經典: 單一優化器
optim = torch.optim.RMSprop(option_critic.parameters(), lr=learning_rate)

# 簡單的聯合更新
optim.zero_grad()
loss.backward()
optim.step()
```

### 9.3 學習率策略對比

| 組件 | GraphRL 學習率 | PyTorch 經典學習率 |
|------|-------------|----------------|
| **Actor 參數** | 1e-5 | 0.0005 |
| **Critic 參數** | 1e-5 | 0.0005 |
| **StateGNN** | 1e-6 (0.1×) | 0.0005 |
| **特徵處理器** | 1e-5 | 0.0005 |
| **優化器類型** | Adam | RMSprop |
| **梯度裁剪** | ✅ (10.0) | ❌ |

---

## 10. 經驗回放機制

### 10.1 GraphRL 經驗回放

#### 10.1.1 步驟級經驗存儲

```python
# GraphRL: 存儲詳細的步驟轉換
class ReplayBuffer:
    def push(self, obs, option, reward, next_obs, done):
        """存儲步驟級轉換"""
        # obs: 複雜的圖數據結構
        # option: 選項索引
        # reward: 包含長度獎勵的調整獎勵
        # next_obs: 下一步圖數據
        # done: 步驟終止標誌

# 從選項執行中推送多個轉換
for tr in step_transitions:
    buffer.push(tr["obs"], tr["option"], tr["reward"], tr["next_obs"], tr["done"])
```

#### 10.1.2 複雜採樣邏輯

```python
# GraphRL: 批次採樣需要處理圖數據結構
def sample(self, batch_size):
    # 需要特殊處理圖批次數據
    # 每個樣本包含完整的圖結構信息
    return batch_observations, batch_options, batch_rewards, batch_next_observations, batch_dones
```

### 10.2 PyTorch 經典經驗回放

#### 10.2.1 簡單經驗存儲

```python
# PyTorch 經典: 簡單的經驗存儲
class ReplayBuffer:
    def push(self, obs, option, reward, next_obs, done):
        """存儲單一轉換"""
        self.buffer.append((obs, option, reward, next_obs, done))

# 每步存儲一個經驗
buffer.push(obs, current_option, reward, next_obs, done)
```

#### 10.2.2 高效採樣

```python
def sample(self, batch_size):
    """高效的向量化採樣"""
    obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
    return np.stack(obs), option, reward, np.stack(next_obs), done
```

### 10.3 經驗回放對比

| 特性 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **存儲粒度** | 步驟級 (多轉換/選項) | 選項級 (單轉換/選項) |
| **數據複雜度** | 高 (圖結構) | 低 (向量/矩陣) |
| **記憶體需求** | 高 | 低 |
| **採樣效率** | 低 (圖批次處理) | 高 (向量化) |
| **緩衝區容量** | 100000 | 10000 |

---

## 11. 評估與監控

### 11.1 GraphRL 評估機制

#### 11.1.1 終止概率記錄

```python
# GraphRL: 詳細的終止概率記錄系統
class TerminationProbabilityLogger:
    def log_termination_prediction(self, option, termination_probs, termination_decision, episode, step, context):
        """記錄每個終止決策的詳細信息"""
        entry = {
            "episode": episode,
            "step": step,
            "option": int(option),
            "termination_probs": termination_probs.detach().cpu().numpy().tolist(),
            "termination_decision": bool(termination_decision),
            "context": context,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.stats["termination_probabilities"].append(entry)

# 在多個地方記錄終止概率
termination_logger.log_termination_prediction(option=current_option, termination_probs=termination_probs, 
                                             termination_decision=option_termination, step=length, context="rollout_option")
```

#### 11.1.2 複合評估指標

```python
# GraphRL: 多維度評估
def evaluate_model(base_env, oc_model, device, num_episodes, max_option_len, logger):
    """綜合性模型評估"""
    episode_scores = []
    episode_lengths = []
    episode_actions = []          # 動作序列
    episode_options = []          # 選項序列
    episode_option_instances = []  # 選項實例數據
    successful_episodes = 0
    
    # 評估指標
    return avg_score, avg_episode_length, success_rate, eval_history
```

### 11.2 PyTorch 經典評估機制

```python
# PyTorch 經典: 基礎記錄系統
class Logger:
    def log_data(self, steps, actor_loss, critic_loss, entropy, epsilon):
        """記錄訓練數據"""
        
    def log_episode(self, steps, rewards, option_lengths, ep_steps, epsilon):
        """記錄回合數據"""

# 簡單的性能記錄
logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
```

### 11.3 評估系統對比

| 功能 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **終止概率追蹤** | ✅ 詳細記錄 | ❌ 無記錄 |
| **選項行為分析** | ✅ 實例級追蹤 | ✅ 長度統計 |
| **成功率監控** | ✅ 多維指標 | ❌ 僅獎勵 |
| **可視化支援** | ✅ 完整繪圖 | ✅ 基本繪圖 |
| **統計導出** | ✅ JSON 格式 | ❌ 內存記錄 |

---

## 12. 應用場景差異

### 12.1 GraphRL 應用特性

#### 12.1.1 結構工程優化

- **問題域**: 建築結構設計優化
- **狀態空間**: 複雜的結構圖 (節點=樓層, 邊=構件)
- **動作空間**: 結構參數調整 (截面尺寸變更)
- **約束條件**: 結構強度、變位、穩定性等工程約束

#### 12.1.2 特化設計

```python
# GraphRL: 工程特化功能
def apply_primitive_action(base_env, structure, action):
    """應用結構工程動作"""
    structure, step_reward, done, fail_name, fail_reason = base_env.step(structure, action)
    
    # 工程特化的狀態判定
    is_minimum_section = bool(done and (fail_reason == "minimum_section"))
    step_pass = not (done and (fail_reason != "minimum_section"))
    
    return structure, float(step_reward), step_pass, is_minimum_section, fail_reason

def check_constraints_without_update(structure, base_env):
    """結構約束檢查 (靜力+動力分析)"""
    # 靜力分析
    load_cases, static_responses = check.get_response(structure, base_env.code_analysis_dir)
    static_constraint_condition, static_response_features, _ = check.process_response(structure, load_cases, static_responses)
    whether_pass, fail_name, fail_reason = check.check_pass(load_cases, static_constraint_condition, base_env.check_displacement)
    
    # 動力分析 (如啟用)
    if base_env.do_nonlinear_dynamic_analysis and whether_pass:
        structure.update_graph_GraphLSTM()
        dynamic_responses = check_nda.get_response(structure, base_env.nda_simulator, base_env.MCE_ground_motion_set, base_env.device)
        # ...
```

### 12.2 PyTorch 經典應用特性

#### 12.2.1 通用強化學習

- **問題域**: 通用 RL 任務 (Atari, CartPole, FourRooms)
- **狀態空間**: 標準觀測 (圖像/向量)
- **動作空間**: 離散動作集合
- **約束條件**: 環境內建約束

#### 12.2.2 環境適配

```python
# PyTorch 經典: 通用環境適配
def make_env(env_name):
    if env_name == 'fourrooms':
        return Fourrooms(), False
    
    env = gym.make(env_name)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    
    # Atari 預處理
    if is_atari:
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
        env = TransformReward(env, lambda r: np.clip(r, -1, 1))
        env = FrameStack(env, 4)
    
    return env, is_atari
```

### 12.3 應用場景對比

| 特性 | GraphRL | PyTorch 經典 |
|------|---------|-------------|
| **領域特化** | 高 (結構工程) | 低 (通用 RL) |
| **問題複雜度** | 極高 (多物理約束) | 中等 (遊戲規則) |
| **實際應用價值** | 高 (工程設計) | 低 (學術研究) |
| **可轉移性** | 低 (特化設計) | 高 (通用框架) |
| **部署難度** | 高 (依賴工程軟體) | 低 (標準環境) |

---

## 13. 總結與建議

### 13.1 關鍵差異總結

#### 13.1.1 架構層面

| 方面 | GraphRL | PyTorch 經典 | 影響 |
|------|---------|-------------|------|
| **特徵提取** | StateGNN (GAT-based) | CNN/FC | GraphRL 更適合關係數據 |
| **狀態表示** | 動態維度圖特徵 | 固定維度向量 | GraphRL 更靈活但複雜 |
| **批次處理** | 圖批次聚合 | 標準張量批次 | 經典實作更高效 |

#### 13.1.2 算法層面

| 方面 | GraphRL | PyTorch 經典 | 影響 |
|------|---------|-------------|------|
| **更新頻率** | Actor 多步, Critic 定期 | Actor+Critic 同步 | GraphRL 更穩定但複雜 |
| **優化策略** | 分離優化器, 差異學習率 | 單一優化器 | GraphRL 更精細控制 |
| **約束處理** | 內建動作遮罩 | 無約束處理 | GraphRL 更適合實際應用 |

#### 13.1.3 實作層面

| 方面 | GraphRL | PyTorch 經典 | 影響 |
|------|---------|-------------|------|
| **代碼複雜度** | 極高 (>1400 行) | 低 (<250 行) | 維護成本差異巨大 |
| **計算效率** | 低 (圖操作+循環) | 高 (向量化) | 性能差異顯著 |
| **調試難度** | 高 (多層抽象) | 低 (直觀邏輯) | 開發效率差異 |

### 13.2 各自優勢

#### 13.2.1 GraphRL 優勢

1. **問題適配性**
   - 天然支援圖結構數據
   - 內建工程約束處理
   - 特化的結構優化功能

2. **算法穩定性**
   - 分離的 Actor-Critic 更新
   - 差異化學習率策略
   - 完整的梯度裁剪機制

3. **監控與分析**
   - 詳細的終止概率記錄
   - 多維度評估指標
   - 豐富的可視化支援

#### 13.2.2 PyTorch 經典優勢

1. **實作簡潔性**
   - 代碼邏輯清晰
   - 易於理解和修改
   - 低維護成本

2. **計算效率**
   - 高效的向量化操作
   - 低記憶體占用
   - 快速的批次處理

3. **通用性**
   - 支援多種環境類型
   - 易於擴展到新問題
   - 標準的 RL 框架設計

### 13.3 適用場景建議

#### 13.3.1 選擇 GraphRL 實作的情況

✅ **推薦使用 GraphRL 當**:
- 處理圖結構數據
- 需要複雜的工程約束
- 要求高精度的結構優化
- 有充足的計算資源
- 團隊具備圖神經網路經驗

#### 13.3.2 選擇 PyTorch 經典實作的情況

✅ **推薦使用 PyTorch 經典當**:
- 標準的 RL 學習任務
- 快速原型開發
- 資源受限的環境
- 需要快速部署
- 作為 Option-Critic 的學習基礎

### 13.4 改進建議

#### 13.4.1 GraphRL 改進方向

1. **性能優化**
   ```python
   # 建議: 使用 torch.jit 編譯關鍵函數
   @torch.jit.script
   def batch_graph_processing(graphs):
       # 批次化圖處理邏輯
   ```

2. **代碼簡化**
   ```python
   # 建議: 提取共用的圖處理邏輯
   class GraphBatchProcessor:
       def process_batch(self, batch_graphs):
           # 統一的批次圖處理
   ```

3. **模組化設計**
   ```python
   # 建議: 分離約束處理邏輯
   class ConstraintHandler:
       def apply_action_mask(self, logits, constraints):
           # 獨立的約束處理模組
   ```

#### 13.4.2 PyTorch 經典改進方向

1. **約束支援**
   ```python
   # 建議: 添加動作遮罩功能
   def get_action(self, state, option, valid_actions_mask=None):
       logits = self.compute_logits(state, option)
       if valid_actions_mask is not None:
           logits[~valid_actions_mask] = -1e8
       # ... 其餘邏輯
   ```

2. **監控增強**
   ```python
   # 建議: 添加終止概率記錄
   def log_termination_stats(self, termination_probs, decisions):
       # 記錄終止行為統計
   ```

### 13.5 最終建議

#### 13.5.1 學習路徑

1. **初學者**: 從 PyTorch 經典實作開始
   - 理解核心 Option-Critic 概念
   - 掌握基礎實作技巧
   - 在簡單環境中驗證理解

2. **進階開發**: 過渡到 GraphRL 實作
   - 學習圖神經網路基礎
   - 理解工程約束處理
   - 掌握複雜系統設計

#### 13.5.2 實際應用

1. **研究目的**: 使用 PyTorch 經典實作
   - 快速驗證算法想法
   - 比較不同方法性能
   - 發表學術成果

2. **工程應用**: 選擇 GraphRL 實作
   - 解決實際結構設計問題
   - 處理複雜工程約束
   - 提供產業解決方案

這兩個實作代表了 Option-Critic 演算法在不同應用場景下的特化發展，各有其適用的領域和優勢，選擇時應根據具體需求和資源情況進行權衡。