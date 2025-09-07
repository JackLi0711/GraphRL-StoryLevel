# Option-Critic PyTorch 實作架構與更新方法詳細分析

## 概述

本文檔詳細分析 option-critic-pytorch 專案中的經典 Option-Critic 演算法實作。此為原始 Option-Critic 論文的標準 PyTorch 實現，支援 Atari 遊戲和簡單環境，採用階層強化學習方法學習選項 (options) 和原始動作策略。

## 1. 整體架構概要

Option-Critic 系統包含以下核心組件：

- **特徵提取器**: CNN (Atari) 或全連接層 (簡單環境)
- **Policy-over-Options (Q)**: 選項層級的價值函數
- **Option Termination (β)**: 選項終止概率預測
- **Intra-Option Policies (π)**: 選項內部的動作策略

### 1.1 數學符號定義

- $s_t$: 時刻 $t$ 的狀態
- $o_t$: 時刻 $t$ 的選項
- $a_t$: 時刻 $t$ 的動作
- $Q(s,o)$: 狀態-選項價值函數
- $\beta(s,o)$: 選項終止概率
- $\pi(a|s,o)$: 給定狀態和選項的動作策略
- $\phi(s)$: 狀態特徵表示

## 2. 網路架構詳細設計

### 2.1 OptionCriticConv (Atari 專用)

適用於高維視覺輸入的卷積神經網路架構：

#### 2.1.1 特徵提取器

```python
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 第一層卷積
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 第二層卷積
    nn.ReLU(), 
    nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 第三層卷積
    nn.ReLU(),
    nn.Flatten(),                                         # 展平
    nn.Linear(7*7*64, 512),                               # 全連接層
    nn.ReLU()
)
```

**架構詳解**：
- **輸入**: $(C, H, W)$ 其中 $C$ 是通道數，$H, W$ 是高度和寬度
- **卷積層 1**: $32$ 個 $8 \times 8$ 濾波器，步長 $4$
- **卷積層 2**: $64$ 個 $4 \times 4$ 濾波器，步長 $2$  
- **卷積層 3**: $64$ 個 $3 \times 3$ 濾波器，步長 $1$
- **輸出**: $\phi(s) \in \mathbb{R}^{512}$

#### 2.1.2 輸出頭部網路

```python
self.Q            = nn.Linear(512, num_options)                      # Policy-over-Options
self.terminations = nn.Linear(512, num_options)                     # Option Termination  
self.options_W    = nn.Parameter(torch.zeros(num_options, 512, num_actions))  # Option Policies
self.options_b    = nn.Parameter(torch.zeros(num_options, num_actions))
```

### 2.2 OptionCriticFeatures (簡單環境專用)

適用於低維特徵輸入的全連接架構：

#### 2.2.1 特徵提取器

```python
self.features = nn.Sequential(
    nn.Linear(in_features, 32),    # 第一層全連接
    nn.ReLU(),
    nn.Linear(32, 64),             # 第二層全連接
    nn.ReLU()
)
```

**架構詳解**：
- **輸入**: $s \in \mathbb{R}^{d}$ 其中 $d$ 是狀態維度
- **隱藏層 1**: $32$ 個神經元
- **隱藏層 2**: $64$ 個神經元
- **輸出**: $\phi(s) \in \mathbb{R}^{64}$

#### 2.2.2 輸出頭部網路

```python
self.Q            = nn.Linear(64, num_options)                       # Policy-over-Options
self.terminations = nn.Linear(64, num_options)                      # Option Termination
self.options_W    = nn.Parameter(torch.zeros(num_options, 64, num_actions))   # Option Policies  
self.options_b    = nn.Parameter(torch.zeros(num_options, num_actions))
```

## 3. 前向傳播計算

### 3.1 狀態表示計算

```python
def get_state(self, obs):
    if obs.ndim < 4:
        obs = obs.unsqueeze(0)
    obs = obs.to(self.device)
    state = self.features(obs)      # φ(s)
    return state
```

$$\phi(s) = f_{\text{features}}(s)$$

### 3.2 選項價值函數

```python
def get_Q(self, state):
    return self.Q(state)
```

$$Q(s, o) = W_Q \cdot \phi(s) + b_Q$$

### 3.3 選項終止概率

```python
def get_terminations(self, state):
    return self.terminations(state).sigmoid()
```

$$\beta(s, o) = \sigma(W_\beta \cdot \phi(s) + b_\beta)$$

其中 $\sigma$ 是 sigmoid 函數。

### 3.4 選項內動作策略

```python
def get_action(self, state, option):
    logits = state.data @ self.options_W[option] + self.options_b[option]
    action_dist = (logits / self.temperature).softmax(dim=-1)
    action_dist = Categorical(action_dist)
    
    action = action_dist.sample()
    logp = action_dist.log_prob(action)
    entropy = action_dist.entropy()
    
    return action.item(), logp, entropy
```

$$\pi(a|s,o) = \text{Categorical}(\text{softmax}(\frac{W_o \cdot \phi(s) + b_o}{\tau}))$$

其中：
- $W_o, b_o$ 是選項 $o$ 的策略參數
- $\tau$ 是溫度參數，控制探索程度

### 3.5 選項終止決策

```python
def predict_option_termination(self, state, current_option):
    termination = self.terminations(state)[:, current_option].sigmoid()
    option_termination = Bernoulli(termination).sample()
    
    Q = self.get_Q(state)  
    next_option = Q.argmax(dim=-1)
    return bool(option_termination.item()), next_option.item()
```

**終止決策流程**：
1. 計算當前選項的終止概率: $\beta(s_t, o_t)$
2. 從伯努利分布採樣終止決策: $\text{terminate} \sim \text{Bernoulli}(\beta(s_t, o_t))$
3. 若終止，選擇下一個選項: $o_{t+1} = \arg\max_{o'} Q(s_t, o')$

## 4. 損失函數與更新演算法

### 4.1 Critic Loss (價值函數損失)

#### 4.1.1 TD 目標計算

```python
def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    
    # 計算當前 Q 值
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q = model.get_Q(states)
    
    # 使用目標網路計算下個狀態的 Q 值
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime = model_prime.get_Q(next_states_prime)
    
    # 計算下個狀態的終止概率
    next_states = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]
```

**TD 目標公式**：
$$G_t = r_t + \gamma \cdot \text{mask}_t \cdot [(1 - \beta(s_{t+1}, o_t)) \cdot Q'(s_{t+1}, o_t) + \beta(s_{t+1}, o_t) \cdot \max_{o'} Q'(s_{t+1}, o')]$$

其中：
- $r_t$ 是即時獎勵
- $\gamma$ 是折扣因子
- $\text{mask}_t = 1 - \text{done}_t$ 處理回合終止
- $Q'$ 是目標網路的價值函數

```python
    # 計算 TD 目標
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + 
         next_options_term_prob * next_Q_prime.max(dim=-1)[0])
```

#### 4.1.2 損失函數

```python
    # TD 誤差
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err
```

$$\mathcal{L}_{\text{critic}} = \frac{1}{2} \mathbb{E}[(Q(s_t, o_t) - G_t)^2]$$

### 4.2 Actor Loss (策略損失)

Actor 損失包含策略梯度損失和選項終止損失兩部分：

#### 4.2.1 狀態處理

```python
def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))
```

#### 4.2.2 終止概率計算

```python
    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()
```

#### 4.2.3 價值函數計算

```python
    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()
```

#### 4.2.4 目標值計算

```python
    # TD 目標
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + 
         next_option_term_prob * next_Q_prime.max(dim=-1)[0])
```

#### 4.2.5 終止損失

```python
    # 終止損失：鼓勵有利的終止決策
    termination_loss = option_term_prob * \
        (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
```

**終止損失公式**：
$$\mathcal{L}_{\text{termination}} = \beta(s_t, o_t) \cdot (Q(s_t, o_t) - \max_{o'} Q(s_t, o') + \lambda_{\text{reg}}) \cdot (1 - \text{done}_t)$$

其中：
- $\lambda_{\text{reg}}$ 是終止正則化參數，防止過度終止
- 只在非終端狀態計算損失

#### 4.2.6 策略梯度損失

```python
    # 策略梯度損失與熵正則化
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
```

**策略損失公式**：
$$\mathcal{L}_{\text{policy}} = -\log \pi(a_t|s_t, o_t) \cdot A_t - \lambda_H \cdot H[\pi(\cdot|s_t, o_t)]$$

其中：
- $A_t = G_t - Q(s_t, o_t)$ 是優勢函數
- $H[\pi]$ 是策略熵
- $\lambda_H$ 是熵正則化係數

#### 4.2.7 總 Actor 損失

```python
    actor_loss = termination_loss + policy_loss
    return actor_loss
```

$$\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{termination}}$$

## 5. 訓練流程

### 5.1 主要訓練循環

```python
def run(args):
    # 初始化環境和模型
    env, is_atari = make_env(args.env)
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    option_critic_prime = deepcopy(option_critic)  # 目標網路
    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)
    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
```

### 5.2 單回合執行流程

#### 5.2.1 回合初始化

```python
    while steps < args.max_steps_total:
        obs = env.reset()
        state = option_critic.get_state(to_tensor(obs))
        greedy_option = option_critic.greedy_option(state)
        current_option = 0
        done = False
        option_termination = True
```

#### 5.2.2 選項選擇

```python
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon
            
            if option_termination:
                # ε-貪婪選項選擇
                current_option = np.random.choice(args.num_options) if \
                    np.random.rand() < epsilon else greedy_option
```

**選項選擇策略**：
$$o_t = \begin{cases} 
\text{random choice} & \text{if } \xi < \epsilon \\
\arg\max_{o} Q(s_t, o) & \text{otherwise}
\end{cases}$$

#### 5.2.3 動作執行

```python
            action, logp, entropy = option_critic.get_action(state, current_option)
            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
```

#### 5.2.4 網路更新

```python
            if len(buffer) > args.batch_size:
                # Actor 更新 (每步)
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy,
                    reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss
                
                # Critic 更新 (定期)
                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss
                
                # 反向傳播
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                # 目標網路更新
                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())
```

#### 5.2.5 選項終止決策

```python
            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
```

### 5.3 更新頻率與時機

- **Actor 更新**: 每個時間步驟 (在線學習)
- **Critic 更新**: 每 `update_frequency` 步 (預設每 4 步)
- **目標網路更新**: 每 `freeze_interval` 步 (預設每 200 步)
- **經驗回放**: 從緩衝區採樣 `batch_size` 個轉換

## 6. 探索策略

### 6.1 ε-衰減排程

```python
@property
def epsilon(self):
    if not self.testing:
        eps = self.eps_min + (self.eps_start - self.eps_min) * \
              exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1
    else:
        eps = self.eps_test
    return eps
```

**探索參數衰減**：
$$\epsilon_t = \epsilon_{\text{min}} + (\epsilon_{\text{start}} - \epsilon_{\text{min}}) \cdot e^{-t/\tau}$$

其中：
- $\epsilon_{\text{start}} = 1.0$ (初始探索率)
- $\epsilon_{\text{min}} = 0.1$ (最小探索率)
- $\tau$ = `eps_decay` (衰減時間常數)

### 6.2 動作選擇策略

- **訓練時**: 從 Categorical 分布採樣
- **測試時**: 使用確定性策略 (通過 testing 標誌控制)

## 7. 經驗回放機制

### 7.1 ReplayBuffer 實作

```python
class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))
    
    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        return np.stack(obs), option, reward, np.stack(next_obs), done
```

**存儲格式**：每個經驗包含 $(s_t, o_t, r_t, s_{t+1}, \text{done}_t)$

## 8. 關鍵特性與設計選擇

### 8.1 目標網路穩定性

使用獨立的目標網路 `option_critic_prime` 提供穩定的 TD 目標：
- 定期從主網路複製參數
- 避免自舉問題 (bootstrapping issues)
- 提升訓練穩定性

### 8.2 溫度參數化策略

動作選擇使用溫度縮放的 softmax：
- 高溫度 ($\tau > 1$): 更隨機的動作選擇
- 低溫度 ($\tau < 1$): 更確定性的動作選擇
- 預設 $\tau = 1.0$

### 8.3 正則化機制

#### 8.3.1 終止正則化

```python
termination_reg = 0.01  # 防止過度終止
```

鼓勵選項持續更長時間，避免頻繁切換。

#### 8.3.2 熵正則化

```python  
entropy_reg = 0.01  # 鼓勵探索
```

維持策略多樣性，防止過早收斂。

## 9. 超參數配置

### 9.1 網路參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `num_options` | 2 | 選項數量 |
| `temperature` | 1.0 | 動作選擇溫度 |
| `learning_rate` | 0.0005 | 學習率 |
| `gamma` | 0.99 | 折扣因子 |

### 9.2 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 32 | 批次大小 |
| `update_frequency` | 4 | Critic 更新頻率 |
| `freeze_interval` | 200 | 目標網路更新間隔 |
| `max_history` | 10000 | 經驗回放容量 |

### 9.3 探索參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_min` | 0.1 | 最小探索率 |
| `epsilon_decay` | 20000 | 探索衰減步數 |
| `optimal_eps` | 0.05 | 測試時探索率 |

### 9.4 正則化參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `termination_reg` | 0.01 | 終止正則化強度 |
| `entropy_reg` | 0.01 | 熵正則化強度 |

## 10. 與 GraphRL 實作的比較

### 10.1 架構差異

| 組件 | Option-Critic PyTorch | GraphRL Option-Critic |
|------|----------------------|----------------------|
| **特徵提取** | CNN/FC | StateGNN (GAT) |
| **狀態表示** | 固定維度向量 | 圖結構動態維度 |
| **批次處理** | 標準張量操作 | 圖批次聚合 |
| **動作遮罩** | 不支持 | 支持約束處理 |

### 10.2 訓練差異

| 方面 | Option-Critic PyTorch | GraphRL Option-Critic |
|------|----------------------|----------------------|
| **Actor 更新** | 每步單樣本 | 每步多轉換 |
| **Critic 更新** | 批次經驗回放 | 定期批次處理 |
| **優化器** | RMSprop | Adam (差異化學習率) |
| **目標網路** | 完整複製 | load_state_dict |

### 10.3 應用場景

- **Option-Critic PyTorch**: 通用強化學習問題 (Atari, CartPole, FourRooms)
- **GraphRL Option-Critic**: 結構工程設計優化，圖結構數據處理

## 11. 總結

Option-Critic PyTorch 實作是經典階層強化學習演算法的標準實現，具有以下特點：

### 11.1 優勢

1. **清晰的架構設計**: 分離的特徵提取器和輸出頭部
2. **穩定的訓練機制**: 目標網路、經驗回放、適當的正則化
3. **靈活的環境適應**: 支援視覺和向量輸入
4. **完整的探索策略**: ε-衰減和熵正則化

### 11.2 核心創新

1. **選項終止學習**: 自動學習何時切換選項
2. **階層決策結構**: 選項選擇 + 動作執行的兩層架構
3. **統一的損失函數**: 結合策略梯度和選項終止的 Actor 損失

### 11.3 應用價值

Option-Critic 方法特別適合需要長期規劃和階層決策的複雜任務，通過學習有意義的選項，能夠提高樣本效率並發現可解釋的策略結構。GraphRL 專案將此概念擴展到圖結構數據，為結構工程等領域提供了強大的優化工具。