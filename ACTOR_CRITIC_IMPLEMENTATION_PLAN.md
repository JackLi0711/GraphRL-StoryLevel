# Actor-Critic Implementation Plan for Dynamic Action Spaces

## 📋 目錄
1. [整體架構設計](#整體架構設計)
2. [核心問題與解決方案](#核心問題與解決方案)
3. [設計決策總結](#設計決策總結)
4. [模組化設計](#模組化設計)
5. [詳細檔案結構](#詳細檔案結構)
6. [訓練與測試流程](#訓練與測試流程)
7. [實作細節](#實作細節)
8. [維度總整理](#維度總整理)

---

## 🎯 整體架構設計

### 設計理念
採用**模組化分層設計**，將 Policy Gradient 方法拆分為可重用組件：
```
Base Components (可重用)
├─ StateGNN (特徵提取 - 已存在，需一起訓練)
├─ PolicyNetwork (Actor - 新建)
├─ ValueNetwork (Critic - 新建)
└─ ExperienceBuffer (經驗儲存 - 新建)

PG Algorithms (可擴展)
├─ A2CAgent (新建)
└─ PPOAgent (新建)
```

---

## 🔧 核心問題與解決方案

### 問題 1: 動態 Action Size
**問題描述**：
- 4層建築: 4 stories × 4 member types = **16 actions**
- 7層建築: 7 stories × 4 member types = **28 actions**
- 固定大小的輸出層無法處理

**解決方案：Story-Level Scoring**
```python
# ❌ 傳統方式 (固定大小)
logits = linear(state)  # [batch, fixed_action_size]

# ✅ 我們的方式 (動態大小)
scores = policy_mlp(story_features)  # [N_stories×4, 1] where N varies
# 每個 story member 獨立計算分數，N 可以是 16, 20, 24, 28...
```

### 問題 2: Invalid Actions (已最小斷面)
**解決方案：Masking**
```python
# 取得 valid actions mask
valid_mask = ~torch.tensor(structure.already_minimum_section_story_indexes)

# 在計算機率前 mask
masked_logits = logits.masked_fill(~valid_mask, -1e9)
probs = F.softmax(masked_logits, dim=0)
```

### 問題 3: Exploration
**方案：Entropy Bonus + Exponential Annealing**
```python
# Exponential annealing
entropy_coef = initial_entropy_coef * (decay_rate ** episode)
# 例如: 0.01 * (0.99 ** episode)

# Loss function
loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

**A2C 與 PPO 共用此機制**

---

## ✅ 設計決策總結

基於討論，以下設計已確認：

### 1. Value Network 類型
- **✅ 採用 State Value Function V(s)**
- 輸入：`global_features` [1, hidden_dim*2]
- 輸出：單一 value scalar
- 不需要 action information

### 2. Entropy Annealing
- **✅ 採用 Exponential Annealing (方案 B)**
```python
entropy_coef_t = entropy_coef_initial * (entropy_decay ** episode)
# 預設: entropy_coef_initial = 0.01, entropy_decay = 0.99
```

### 3. 更新頻率
- **A2C**: 每個 episode 結束後立即更新 (accumulate_episodes = 1)
- **PPO**: 收集 K 個 episodes 後更新 (accumulate_episodes = 5)
- **✅ 使用 config 參數控制**: `--accumulate_episodes`

### 4. Testing/Inference 配置
- **n**: 每 n 個 training episodes 後進行一次 testing
  - Config: `--test_frequency` (預設 5，可調整)
- **m**: 每次 testing 進行 m 次 inference
  - Config: `--test_runs` (預設 10，可調整)

**繪圖需求**：
1. Training score curve (每個 training episode)
2. Testing score curve (mean ± std of m runs，每 n episodes)
3. Best design process visualization (m 次中最好的那次)

### 5. StateGNN 訓練方式
- **✅ StateGNN 與 Policy/Value Networks 一起訓練**
- 所有參數加入同一個 optimizer
- Action selection 時：`with torch.no_grad()` (節省記憶體)
- Update 時：重新 forward，計算梯度

### 6. Global Features 計算
- **✅ 使用 Mean Pooling**
```python
global_features = story_features.mean(dim=0, keepdim=True)  # [1, hidden_dim*2]
```

### 7. Advantage 計算
- **✅ 使用 Monte Carlo Returns**
```python
returns = compute_returns(rewards, gamma)
advantages = returns - values
# Normalize for stability
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 8. 其他配置
- **Learning Rate**: 固定值 (A2C: 1e-4, PPO: 3e-4)
- **不需要與其他模型比較**: 專注於 A2C 和 PPO
- **測試泛化性**: 按照現有 environment 設定

---

## 📦 模組化設計

### Layer 1: Base Networks (可重用)

#### 1.1 PolicyNetwork
```python
class PolicyNetwork(nn.Module):
    """
    Story-level policy network (Actor)

    輸入: story_features [N, state_dim]
        - N: 當前建築的 story member 數量 (動態)
        - state_dim: StateGNN 輸出維度 (hidden_dim * 2)

    輸出: action_scores [N]
        - 每個 story member 的分數
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 單一分數輸出
        )

    def forward(self, story_features):
        # story_features: [N, state_dim]
        scores = self.policy_head(story_features)  # [N, 1]
        return scores.squeeze(-1)  # [N]
```

#### 1.2 ValueNetwork
```python
class ValueNetwork(nn.Module):
    """
    State value network (Critic) - V(s)

    輸入: global_features [1, state_dim]
        - 整個建築的全局特徵 (mean pooling from story_features)
        - state_dim: StateGNN 輸出維度 (hidden_dim * 2)

    輸出: value [1]
        - 當前狀態的價值估計 V(s)
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_features):
        # global_features: [1, state_dim]
        return self.value_head(global_features)  # [1]
```

#### 1.3 ExperienceBuffer
```python
class ExperienceBuffer:
    """
    儲存 episode(s) 的經驗

    用於 A2C (單 episode) 和 PPO (多 episodes) 的 trajectory 收集
    """
    def __init__(self):
        # Episode-level data (list of episodes)
        self.episodes = []

        # Current episode buffer
        self.states = []           # List of [N_i, state_dim] tensors
        self.global_states = []    # List of [1, state_dim] tensors (for value)
        self.actions = []          # List of integers
        self.rewards = []          # List of floats
        self.log_probs = []        # List of tensors
        self.values = []           # List of tensors
        self.entropies = []        # List of tensors
        self.valid_masks = []      # List of [N_i] bool tensors

    def add_step(self, state, global_state, action, reward, log_prob, value, entropy, valid_mask):
        """添加一個 step 到當前 episode"""
        self.states.append(state)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.valid_masks.append(valid_mask)

    def finish_episode(self):
        """完成當前 episode，儲存到 episodes 列表"""
        episode_data = {
            'states': self.states,
            'global_states': self.global_states,
            'actions': self.actions,
            'rewards': self.rewards,
            'log_probs': self.log_probs,
            'values': self.values,
            'entropies': self.entropies,
            'valid_masks': self.valid_masks
        }
        self.episodes.append(episode_data)

        # Reset current episode buffer
        self.states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.valid_masks = []

    def get_all_episodes(self):
        """獲取所有收集的 episodes"""
        return self.episodes

    def clear(self):
        """清空所有 buffer"""
        self.episodes = []
        self.states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.valid_masks = []

    def __len__(self):
        """返回收集的 episodes 數量"""
        return len(self.episodes)
```

---

### Layer 2: Base Agent (可重用)

```python
class BasePGAgent:
    """
    Policy Gradient 方法的基礎類別

    包含共用功能：
    - Feature extraction (StateGNN) - 一起訓練
    - Action selection with masking
    - Advantage computation
    - Entropy annealing
    """
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 gamma: float,
                 entropy_coef_initial: float,
                 entropy_decay: float,
                 accumulate_episodes: int,
                 device: str,
                 logger = None):

        self.device = device
        self.logger = logger

        # StateGNN for feature extraction (需要訓練)
        from RL.model import StateGNN
        self.state_gnn = StateGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            member_state_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)

        # Policy and Value networks
        state_dim = hidden_dim * 2
        self.policy_net = PolicyNetwork(state_dim, hidden_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        # Training parameters
        self.gamma = gamma
        self.entropy_coef_initial = entropy_coef_initial
        self.entropy_decay = entropy_decay
        self.accumulate_episodes = accumulate_episodes

        # Experience buffer
        self.buffer = ExperienceBuffer()

        # Episode counter
        self._number_episodes = 0

    def get_entropy_coef(self):
        """計算當前的 entropy coefficient (exponential annealing)"""
        return self.entropy_coef_initial * (self.entropy_decay ** self._number_episodes)

    def get_features(self, graph, structure):
        """
        從 graph 提取特徵 (no grad for action selection)

        輸入:
            - graph: structure.graph (PyG Data object)
            - structure: Structure object

        輸出:
            - story_features: [N, hidden_dim*2] N個story member的特徵
            - global_features: [1, hidden_dim*2] 全局特徵 (mean pooling)
        """
        with torch.no_grad():  # Action selection 時不需要梯度
            graph = graph.to(self.device)
            story_features = self.state_gnn(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                None,  # batch
                structure.aux["story_batch"].to(self.device),
                None   # structure_story_ptr
            )  # [N, hidden_dim*2]

            # Global feature: mean pooling
            global_features = story_features.mean(dim=0, keepdim=True)  # [1, hidden_dim*2]

        return story_features, global_features

    def choose_action(self, story_features, global_features, structure, greedy=False):
        """
        選擇 action (處理動態 action size 和 masking)

        輸入:
            - story_features: [N, state_dim] N個story member的特徵
            - global_features: [1, state_dim] 全局特徵
            - structure: Structure object (用於獲取 invalid actions)
            - greedy: bool, 是否貪婪選擇

        輸出:
            - action: int, 選擇的 action index
            - log_prob: Tensor, log probability
            - value: Tensor, state value V(s)
            - entropy: Tensor, policy entropy
        """
        N = story_features.shape[0]

        # Get action scores from policy network
        action_scores = self.policy_net(story_features)  # [N]

        # Get state value
        value = self.value_net(global_features)  # [1]

        # Create valid actions mask
        valid_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        invalid_indices = structure.already_minimum_section_story_indexes
        if len(invalid_indices) > 0:
            valid_mask[invalid_indices] = False

        # Mask invalid actions
        masked_scores = action_scores.masked_fill(~valid_mask, -1e9)

        # Compute probabilities
        probs = F.softmax(masked_scores, dim=0)  # [N]
        dist = Categorical(probs)

        # Sample or greedy
        if greedy:
            action = probs.argmax()
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Entropy for exploration bonus
        entropy = dist.entropy()

        return action.item(), log_prob, value, entropy

    def compute_returns(self, rewards, gamma):
        """
        計算 Monte Carlo returns

        輸入:
            - rewards: List[float], length T
            - gamma: discount factor

        輸出:
            - returns: Tensor [T]
        """
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        R = 0

        # Reverse iterate
        for t in reversed(range(T)):
            R = rewards[t] + gamma * R
            returns[t] = R

        return returns

    def compute_advantages(self, returns, values):
        """
        計算 advantages (normalized)

        輸入:
            - returns: Tensor [T]
            - values: List[Tensor], length T

        輸出:
            - advantages: Tensor [T]
        """
        values_tensor = torch.cat(values).squeeze()  # [T]
        advantages = returns - values_tensor

        # Normalize for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
```

---

### Layer 3: A2C Agent

```python
class A2CAgent(BasePGAgent):
    """
    Advantage Actor-Critic (A2C)

    特點:
    - On-policy learning
    - 每個 episode 更新一次 (accumulate_episodes=1)
    - Entropy bonus with exponential annealing
    - StateGNN + Policy + Value 一起訓練
    """
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 value_loss_coef: float = 0.5,
                 entropy_coef_initial: float = 0.01,
                 entropy_decay: float = 0.99,
                 max_grad_norm: float = 0.5,
                 accumulate_episodes: int = 1,
                 device: str = "cuda",
                 logger = None):

        super().__init__(
            node_feature_dim, edge_feature_dim, hidden_dim, num_layers,
            gamma, entropy_coef_initial, entropy_decay,
            accumulate_episodes, device, logger
        )

        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Single optimizer for all networks (一起訓練)
        self.optimizer = optim.Adam(
            list(self.state_gnn.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

    def update(self):
        """
        A2C update

        當收集到 accumulate_episodes 個 episodes 後調用

        Returns:
            - loss_dict: Dict with loss components
        """
        if len(self.buffer) < self.accumulate_episodes:
            return {}  # 尚未收集足夠的 episodes

        episodes = self.buffer.get_all_episodes()

        all_log_probs = []
        all_entropies = []
        all_values = []
        all_returns = []
        all_advantages = []

        # Process each episode
        for episode in episodes:
            # Compute returns
            returns = self.compute_returns(episode['rewards'], self.gamma)

            # Compute advantages
            advantages = self.compute_advantages(returns, episode['values'])

            # Collect
            all_log_probs.extend(episode['log_probs'])
            all_entropies.extend(episode['entropies'])
            all_values.extend(episode['values'])
            all_returns.append(returns)
            all_advantages.append(advantages)

        # Convert to tensors
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        values = torch.cat(all_values).squeeze()
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)

        # Get current entropy coefficient
        entropy_coef = self.get_entropy_coef()

        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        # Total loss
        total_loss = (policy_loss +
                     self.value_loss_coef * value_loss +
                     entropy_coef * entropy_loss)

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.state_gnn.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        # Log
        if self.logger:
            self.logger.info(
                f"A2C Update (Episode {self._number_episodes}) - "
                f"Policy Loss: {policy_loss.item():.4f}, "
                f"Value Loss: {value_loss.item():.4f}, "
                f"Entropy: {-entropy_loss.item():.4f}, "
                f"Entropy Coef: {entropy_coef:.6f}"
            )

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'entropy_coef': entropy_coef,
            'total_loss': total_loss.item()
        }
```

---

### Layer 4: PPO Agent

```python
class PPOAgent(BasePGAgent):
    """
    Proximal Policy Optimization (PPO)

    特點:
    - Clipped surrogate objective
    - 收集多個 episodes 後更新 (accumulate_episodes=5)
    - Multiple epochs per update
    - Entropy bonus with exponential annealing
    - StateGNN + Policy + Value 一起訓練
    """
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef_initial: float = 0.01,
                 entropy_decay: float = 0.99,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 accumulate_episodes: int = 5,
                 device: str = "cuda",
                 logger = None):

        super().__init__(
            node_feature_dim, edge_feature_dim, hidden_dim, num_layers,
            gamma, entropy_coef_initial, entropy_decay,
            accumulate_episodes, device, logger
        )

        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # Single optimizer for all networks (一起訓練)
        self.optimizer = optim.Adam(
            list(self.state_gnn.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

    def update(self):
        """
        PPO update

        當收集到 accumulate_episodes 個 episodes 後調用
        進行 ppo_epochs 次更新

        Returns:
            - loss_dict: Dict with loss components
        """
        if len(self.buffer) < self.accumulate_episodes:
            return {}  # 尚未收集足夠的 episodes

        episodes = self.buffer.get_all_episodes()

        # Flatten all episodes
        all_states = []
        all_global_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_valid_masks = []

        for episode in episodes:
            # Compute returns and advantages
            returns = self.compute_returns(episode['rewards'], self.gamma)
            advantages = self.compute_advantages(returns, episode['values'])

            # Store
            all_states.extend(episode['states'])
            all_global_states.extend(episode['global_states'])
            all_actions.extend(episode['actions'])
            all_old_log_probs.extend(episode['log_probs'])
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_valid_masks.extend(episode['valid_masks'])

        # Convert to tensors
        old_log_probs = torch.stack(all_old_log_probs).detach()
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)

        # Get current entropy coefficient
        entropy_coef = self.get_entropy_coef()

        # PPO epochs
        policy_losses = []
        value_losses = []
        entropies_list = []

        for epoch in range(self.ppo_epochs):
            # Re-compute with current policy (有梯度)
            new_log_probs = []
            new_values = []
            new_entropies = []

            for t in range(len(all_states)):
                state = all_states[t]
                global_state = all_global_states[t]
                action = all_actions[t]
                valid_mask = all_valid_masks[t]

                # Forward pass (有梯度)
                action_scores = self.policy_net(state)
                masked_scores = action_scores.masked_fill(~valid_mask, -1e9)
                probs = F.softmax(masked_scores, dim=0)
                dist = Categorical(probs)

                # New log prob and entropy
                new_log_prob = dist.log_prob(torch.tensor(action, device=self.device))
                new_log_probs.append(new_log_prob)
                new_entropies.append(dist.entropy())

                # New value
                new_value = self.value_net(global_state)
                new_values.append(new_value)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.cat(new_values).squeeze()
            new_entropies = torch.stack(new_entropies)

            # PPO clipped loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns)

            # Entropy loss
            entropy_loss = -new_entropies.mean()

            # Total loss
            total_loss = (policy_loss +
                         self.value_loss_coef * value_loss +
                         entropy_coef * entropy_loss)

            # Optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.state_gnn.parameters()) +
                list(self.policy_net.parameters()) +
                list(self.value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            # Record
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies_list.append(-entropy_loss.item())

        # Log
        if self.logger:
            self.logger.info(
                f"PPO Update (Episode {self._number_episodes}) - "
                f"Policy Loss: {np.mean(policy_losses):.4f}, "
                f"Value Loss: {np.mean(value_losses):.4f}, "
                f"Entropy: {np.mean(entropies_list):.4f}, "
                f"Entropy Coef: {entropy_coef:.6f}"
            )

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies_list),
            'entropy_coef': entropy_coef,
            'total_loss': np.mean(policy_losses) + np.mean(value_losses)
        }
```

---

## 📁 詳細檔案結構

```
GraphRL_story_level/
├─ RL/
│  ├─ pg/                           # 新建資料夾: Policy Gradient 模組
│  │  ├─ __init__.py               # 新建
│  │  ├─ networks.py               # 新建: PolicyNetwork, ValueNetwork
│  │  ├─ buffer.py                 # 新建: ExperienceBuffer
│  │  ├─ base_agent.py             # 新建: BasePGAgent
│  │  ├─ a2c_agent.py              # 新建: A2CAgent
│  │  └─ ppo_agent.py              # 新建: PPOAgent
│  │
│  ├─ model.py                     # 保留不變 (StateGNN)
│  ├─ environment.py               # 保留不變
│  ├─ record.py                    # 保留不變
│  └─ ...
│
├─ train_a2c.py                    # 新建: A2C 訓練腳本
├─ train_ppo.py                    # 新建: PPO 訓練腳本
├─ train.py                        # 保留不變 (DQN)
└─ train_option_critic.py          # 保留不變
```

---

## 🔄 訓練與測試流程

### 訓練流程 (參考 train.py)

```python
for episode in range(num_episodes):
    # 1. Training episode
    score, loss_dict = train_episode(agent, env, rec, logger)

    # 2. 記錄訓練結果
    rec.training_record["score"].append(score)
    rec.learn_losses.append(loss_dict.get('total_loss', 0))

    # 3. 每 test_frequency 個 episodes 進行測試
    if (episode + 1) % args.test_frequency == 0:
        # 進行 test_runs 次 inference
        test_scores = []
        test_designs = []

        for run in range(args.test_runs):
            test_score, final_design = test_episode(agent, env, rec, logger)
            test_scores.append(test_score)
            test_designs.append(final_design)

        # 計算統計量
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)

        # 記錄 mean ± std
        rec.testing_record["score_mean"].append(mean_score)
        rec.testing_record["score_std"].append(std_score)

        # 找最好的 design
        best_idx = np.argmax(test_scores)
        best_design = test_designs[best_idx]

        # 保存最好的 design
        rec.testing_record["best_design"].append(best_design)

        # 繪製 design process (最好的那次)
        visualize_design_process(agent, env, best_design, args.ckpt_dir, episode)

        # 輸出結果
        rec.output(args.ckpt_dir)

        # 繪圖
        plot_training_testing_curves(rec, args.ckpt_dir)

        # 保存模型
        save_model(agent, args.ckpt_dir, episode)
```

### 單個 Training Episode

```python
def train_episode(agent, env, rec, logger):
    """訓練一個 episode"""
    structure = env.reset()
    rec.record_in_beginning(structure, testing=False)

    graph = structure.graph.clone()
    score = 0
    done = False

    while not done:
        original_structure = deepcopy(structure)

        # Get features (no grad)
        story_features, global_features = agent.get_features(graph, structure)

        # Choose action
        action, log_prob, value, entropy = agent.choose_action(
            story_features,
            global_features,
            structure,
            greedy=False
        )

        # Step environment
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # Store experience
        valid_mask = torch.ones(story_features.shape[0], dtype=torch.bool, device=agent.device)
        if len(structure.already_minimum_section_story_indexes) > 0:
            valid_mask[structure.already_minimum_section_story_indexes] = False

        agent.buffer.add_step(
            state=story_features,
            global_state=global_features,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            valid_mask=valid_mask
        )

        # Update
        graph = structure.graph.clone()
        score += reward

    # Finish episode
    agent.buffer.finish_episode()
    agent._number_episodes += 1

    # Update if enough episodes collected
    loss_dict = {}
    if len(agent.buffer) >= agent.accumulate_episodes:
        loss_dict = agent.update()

    # Record
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    rec.record_in_end(final_structure, env, testing=False)

    return score, loss_dict
```

### 單個 Testing Episode

```python
def test_episode(agent, env, rec, logger):
    """測試一個 episode (greedy policy)"""
    structure = env.reset(testing=True)
    rec.record_in_beginning(structure, testing=True)

    graph = structure.graph.clone()
    score = 0
    done = False
    design_process = []  # 記錄設計過程

    while not done:
        original_structure = deepcopy(structure)

        # Get features
        story_features, global_features = agent.get_features(graph, structure)

        # Choose action (greedy)
        action, _, _, _ = agent.choose_action(
            story_features,
            global_features,
            structure,
            greedy=True
        )

        # Record design step
        design_process.append({
            'structure': deepcopy(structure),
            'action': action,
            'story_level_sections': structure.story_level_sections.copy()
        })

        # Step
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        graph = structure.graph.clone()
        score += reward

    # Final structure
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    rec.record_in_end(final_structure, env, testing=True)

    return score, design_process
```

### 繪圖函數

```python
def plot_training_testing_curves(rec, ckpt_dir):
    """
    繪製訓練和測試曲線

    Training: 每個 episode 的 score
    Testing: 每 test_frequency episodes 的 mean ± std
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training curve
    episodes = range(1, len(rec.training_record["score"]) + 1)
    ax.plot(episodes, rec.training_record["score"],
            label='Training Score', alpha=0.6, color='blue')

    # Testing curve (mean ± std)
    test_episodes = range(args.test_frequency,
                         len(rec.training_record["score"]) + 1,
                         args.test_frequency)
    test_means = rec.testing_record["score_mean"]
    test_stds = rec.testing_record["score_std"]

    ax.plot(test_episodes, test_means,
            label='Testing Score (mean)', color='red', linewidth=2)
    ax.fill_between(test_episodes,
                     np.array(test_means) - np.array(test_stds),
                     np.array(test_means) + np.array(test_stds),
                     alpha=0.3, color='red', label='Testing Score (±std)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training and Testing Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ckpt_dir / 'training_testing_curves.png', dpi=150)
    plt.close()
```

---

## 🔨 實作細節

### Config 參數總覽

```python
# train_a2c.py / train_ppo.py
parser.add_argument("--hidden_dim", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=3)

# Learning
parser.add_argument("--lr", type=float, default=1e-4)  # A2C: 1e-4, PPO: 3e-4
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--value_loss_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)

# Entropy annealing
parser.add_argument("--entropy_coef_initial", type=float, default=0.01)
parser.add_argument("--entropy_decay", type=float, default=0.99)

# Update frequency
parser.add_argument("--accumulate_episodes", type=int, default=1)  # A2C: 1, PPO: 5

# PPO specific
parser.add_argument("--clip_epsilon", type=float, default=0.2)
parser.add_argument("--ppo_epochs", type=int, default=4)

# Training
parser.add_argument("--num_epoch", type=int, default=1000)
parser.add_argument("--test_frequency", type=int, default=5)  # 可調整
parser.add_argument("--test_runs", type=int, default=10)      # 可調整

# Environment
parser.add_argument("--structure_shape", type=str, default="random")
parser.add_argument("--reward_type", type=str, default="material")
parser.add_argument("--add_structure_geometry", action="store_true", default=True)
parser.add_argument("--add_response_features", action="store_true", default=False)

# Other
parser.add_argument("--ckpt_dir", type=Path, default="./Results/A2C")
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--random_seed", type=int, default=731)
```

---

## 📊 維度總整理

### StateGNN
```
Input:
  - node_feature: [num_nodes, 8]
  - edge_index: [2, num_edges]
  - edge_attr: [num_edges, 11]
  - story_batch: [num_edges]

Output:
  - story_features: [N, hidden_dim*2]
    where N = num_stories × 4 (xdir_beam, zdir_beam, outer_col, inner_col)
```

### PolicyNetwork
```
Input: story_features [N, hidden_dim*2]
Hidden: [N, hidden_dim] → [N, hidden_dim//2]
Output: action_scores [N]
```

### ValueNetwork
```
Input: global_features [1, hidden_dim*2]
Hidden: [1, hidden_dim] → [1, hidden_dim//2]
Output: value [1]
```

### Action Space (動態)
```
4-story: N = 4 × 4 = 16 actions
5-story: N = 5 × 4 = 20 actions
6-story: N = 6 × 4 = 24 actions
7-story: N = 7 × 4 = 28 actions
```

### Gradient Flow
```
Action Selection: no_grad (節省記憶體)
Update: with grad (訓練 StateGNN + Policy + Value)
```

---

## ✅ 實作完成標準

1. ✅ 所有模組可獨立測試
2. ✅ A2C 和 PPO 都能正常訓練
3. ✅ 支援動態 action size (4-7 層建築)
4. ✅ Entropy 正確 annealing
5. ✅ Testing 產生 mean ± std 圖表
6. ✅ 最佳 design process 可視化
7. ✅ 模型可保存和載入
8. ✅ 記錄格式與 train.py 一致
