# PPO Model Architecture Documentation

**作者:** Kyle
**日期:** 2025-10-24
**訓練腳本:** `train_ppo.py`
**目的:** 詳細記錄PPO模型架構、輸入輸出維度、更新方式及所有技術細節

---

## 目錄
1. [整體架構概述](#整體架構概述)
2. [模型組成](#模型組成)
3. [輸入輸出維度](#輸入輸出維度)
4. [Edge-Level Features](#edge-level-features)
5. [PPO訓練流程](#ppo訓練流程)
6. [批次優化機制](#批次優化機制)
7. [超參數設定](#超參數設定)
8. [環境與獎勵](#環境與獎勵)

---

## 整體架構概述

本項目使用 **Proximal Policy Optimization (PPO)** 算法進行結構優化設計。模型採用 **Actor-Critic** 架構，並使用 **Graph Neural Network (GNN)** 提取結構特徵。

### 核心組件
```
Environment (Structure Design Problem)
    ↓ (generates graphs with node/edge features)
StateGNN (Feature Extractor)
    ↓ (extracts story-level features)
├── PolicyNetwork (Actor)  → selects action
└── ValueNetwork (Critic)   → estimates state value
```

### 關鍵特點
- **Graph-based State Representation**: 建築結構表示為圖 (nodes = joints, edges = members)
- **Story-level Action Space**: 每個action對應一個story的某個member type (x-beam, z-beam, out-col, in-col)
- **Dynamic Action Masking**: 已達到最小斷面的story members會被mask掉
- **Batch Optimization**: 一次forward多個timesteps的graphs以解決StateGNN梯度消失問題
- **Separate Learning Rates**: StateGNN使用100倍學習率以補償較小的梯度

---

## 模型組成

### 1. StateGNN (Feature Extractor)

**檔案位置:** `RL/model.py` (class `StateGNN`)

**架構:**
```python
StateGNN(
    node_feature_dim=8,      # 若add_structure_geometry=True
    edge_feature_dim=18,     # 若add_response_features=True, 否則16
    hidden_dim=100,
    member_state_dim=100,
    num_layers=3
)
```

**網路層級:**
```
Input: Node features [num_nodes, 8]
  ↓
Encoder MLP: Linear(8, 100) + ReLU
  ↓
GAT Layers (×3):
  - GATConv(100, 100, heads=4, concat=False, edge_dim=18)
  - ReLU activation
  ↓
Decoder MLP: Linear(100, 100) + ReLU
  ↓
Node Embedding: [num_nodes, 100]
  ↓
Edge Embedding:
  - Concat [node_i, node_j, edge_attr] → [200 + 18 = 218]
  - Edge MLP: Linear(218, 100) + ReLU
  ↓
Edge Embedding: [num_edges, 100]
  ↓
Story Aggregation:
  - global_add_pool by story_batch → Story Embedding [num_stories, 100]
  - global_add_pool by graph_batch → Graph Embedding [1, 100]
  ↓
State Combination:
  - state = [story_embedding, graph_embedding] → [num_stories, 200]
```

**輸入:**
- `x`: Node features `[num_nodes, 8]`
  - Node feature維度 (8維):
    1. Node type (1-hot encoded or continuous)
    2. X coordinate
    3. Y coordinate (height)
    4. Z coordinate
    5-8. Geometric features (if `add_structure_geometry=True`)

- `edge_index`: Edge connectivity `[2, num_edges*2]` (bidirectional)

- `edge_attr`: Edge features `[num_edges*2, 18]`
  - **詳見 [Edge-Level Features](#edge-level-features) 章節**

- `batch`: Graph-level batch index `[num_edges]` (用於multi-graph batching)

- `story_batch`: Story-level batch index `[num_edges]`
  - 每個edge屬於哪個story的哪個member type
  - 範例: 一個6層樓建築有 6 stories × 4 member types = 24 unique story members

- `structure_story_ptr`: Structure boundary pointers
  - 範例: `[0, 24, 48, 72]` 表示3個建築，各有24個story members

**輸出:**
- `state`: Story-level state features `[num_stories, 200]`
  - 前100維: story自己的embedding
  - 後100維: 整個建築的global embedding (所有stories共享)

---

### 2. PolicyNetwork (Actor)

**檔案位置:** `RL/pg/networks.py` (class `PolicyNetwork`)

**架構:**
```python
PolicyNetwork(
    state_dim=200,      # StateGNN輸出維度 (hidden_dim * 2)
    hidden_dim=100
)
```

**網路層級:**
```
Input: story_features [N, 200]  (N = num_stories for current structure)
  ↓
Linear(200, 100) + ReLU
  ↓
Linear(100, 50) + ReLU
  ↓
Linear(50, 1)  # 每個story member一個分數
  ↓
Output: action_scores [N]
```

**輸入:**
- `story_features`: `[N, 200]` - 當前建築的N個story members的特徵
  - N是動態的 (取決於建築大小)
  - 範例: 6層樓 → N=24 (6 stories × 4 member types)

**輸出:**
- `action_scores`: `[N]` - 每個story member的raw score
  - 經過softmax後變成action probabilities
  - 需要進行dynamic masking (已minimum section的members被mask)

---

### 3. ValueNetwork (Critic)

**檔案位置:** `RL/pg/networks.py` (class `ValueNetwork`)

**架構:**
```python
ValueNetwork(
    state_dim=200,
    hidden_dim=100
)
```

**網路層級:**
```
Input: global_features [1, 200]
  ↓
Linear(200, 100) + ReLU
  ↓
Linear(100, 50) + ReLU
  ↓
Linear(50, 1)
  ↓
Output: value [1, 1]  # State value V(s)
```

**輸入:**
- `global_features`: `[1, 200]` - 整個建築的全局特徵
  - 由story_features通過mean pooling得到
  - `global_features = story_features.mean(dim=0, keepdim=True)`

**輸出:**
- `value`: `[1, 1]` - 當前狀態的價值估計 V(s)

---

## 輸入輸出維度

### Graph Input Dimensions

**Node Features** (`graph.x`):
```python
node_feature_dim = 8  # if add_structure_geometry=True, else 5

# 5維基礎特徵:
# 1. Node type indicator
# 2. X coordinate (normalized)
# 3. Y coordinate (height, normalized)
# 4. Z coordinate (normalized)
# 5. (additional node property)

# +3維幾何特徵 (if add_structure_geometry=True):
# 6. Span length in X direction
# 7. Span length in Z direction
# 8. Story height
```

**Edge Features** (`graph.edge_attr`):
```python
edge_feature_dim = 18  # if add_response_features=True
edge_feature_dim = 16  # if add_response_features=False

# 基礎16維特徵 (詳見下一章節):
# - Section properties (4維)
# - Member geometry (2維)
# - Member type one-hot (4維)
# - Floor/story indices (4維)
# - Column/beam indicators (2維)

# +2維response特徵 (if add_response_features=True):
# 17. Static response feature (stress ratio)
# 18. Dynamic response feature (drift ratio)
```

### Model Flow Dimensions

完整的dimension flow:

```
Structure → Graph
  ├─ Nodes: [num_nodes, 8]
  └─ Edges: [num_edges*2, 18]  # bidirectional edges
       ↓
StateGNN Forward
  ├─ Node embedding: [num_nodes, 100]
  ├─ Edge embedding: [num_edges, 100]  # 注意: 只取單向edges (::2)
  ├─ Story embedding: [num_stories, 100]  # e.g., 24 for 6-story building
  └─ Graph embedding: [1, 100]
       ↓
State Combination
  └─ State: [num_stories, 200]  # concat [story_emb, graph_emb]
       ↓
Action Selection
  ├─ Policy: story_features [num_stories, 200] → action_scores [num_stories]
  └─ Value: global_features [1, 200] → value [1, 1]
       ↓
Apply Masking
  ├─ Valid mask: [num_stories] bool tensor
  └─ Masked scores: action_scores.masked_fill(~valid_mask, -1e9)
       ↓
Sample Action
  ├─ Probs: softmax(masked_scores) → [num_stories]
  ├─ Action: Categorical(probs).sample() → scalar index
  └─ Log prob: dist.log_prob(action) → scalar
```

**範例 (6-story, 4x4 building):**
```
num_nodes = 175  # (4+1) × (4+1) × 7 levels
num_edges = 600  # beams + columns
num_stories = 24  # 6 stories × 4 member types

Dimensions:
  graph.x: [175, 8]
  graph.edge_attr: [1200, 18]  # bidirectional
  StateGNN output: [24, 200]
  action_scores: [24]
  selected action: integer in [0, 23]
```

---

## Edge-Level Features

### Edge Feature 組成 (18維)

**重要創新:** 本模型在edge features中加入了**member type**和**floor information**的編碼，使GNN能夠區分不同類型的構件和樓層位置。

```python
# 總維度: 18 (if add_response_features=True)

# ===== Section Properties (4維) =====
0. Section area (normalized)
1. Moment of inertia Ix (normalized)
2. Moment of inertia Iy (normalized)
3. Moment of inertia Iz (normalized)

# ===== Member Geometry (2維) =====
4. Member length (normalized)
5. Member orientation/angle

# ===== Member Type One-Hot Encoding (4維) =====
# 這是關鍵創新！每個edge的type被編碼為4維one-hot
6. is_x_beam (1 if x-direction beam, else 0)
7. is_z_beam (1 if z-direction beam, else 0)
8. is_outer_column (1 if outer column, else 0)
9. is_inner_column (1 if inner column, else 0)
# 範例:
#   x-beam: [1, 0, 0, 0]
#   z-beam: [0, 1, 0, 0]
#   outer-col: [0, 0, 1, 0]
#   inner-col: [0, 0, 0, 1]

# ===== Floor/Story One-Hot Encoding (4維) =====
# 將column index和beam index分別編碼
10. column_idx_0 (1st bit of column floor index)
11. column_idx_1 (2nd bit of column floor index)
12. beam_idx_0 (1st bit of beam floor index)
13. beam_idx_1 (2nd bit of beam floor index)
# 使用2-bit encoding可表示最多4個不同值
# 實際實作中可能使用更多bits以支援更多樓層

# ===== Column/Beam Indicators (2維) =====
14. is_column (1 if column, 0 if beam)
15. is_beam (1 if beam, 0 if column)

# ===== Response Features (2維, optional) =====
16. Static response (stress ratio from linear analysis)
17. Dynamic response (drift ratio from nonlinear dynamic analysis)
```

### Edge Feature 創建過程

**檔案位置:** `Structure/structure.py` (推測)

關鍵實作概念:
```python
# Pseudo-code for edge feature creation

for each member in structure:
    # Basic properties
    features = [
        member.area / area_norm,
        member.Ix / I_norm,
        member.Iy / I_norm,
        member.Iz / I_norm,
        member.length / length_norm,
        member.angle
    ]

    # Member type one-hot (4維)
    if member.type == "x_beam":
        type_onehot = [1, 0, 0, 0]
    elif member.type == "z_beam":
        type_onehot = [0, 1, 0, 0]
    elif member.type == "outer_column":
        type_onehot = [0, 0, 1, 0]
    elif member.type == "inner_column":
        type_onehot = [0, 0, 0, 1]
    features.extend(type_onehot)

    # Floor index encoding (4維)
    # 將樓層索引轉換為binary bits
    col_idx_bits = int_to_binary_bits(member.column_floor_idx, n_bits=2)
    beam_idx_bits = int_to_binary_bits(member.beam_floor_idx, n_bits=2)
    features.extend(col_idx_bits + beam_idx_bits)

    # Column/Beam indicators (2維)
    is_column = 1 if "column" in member.type else 0
    is_beam = 1 - is_column
    features.extend([is_column, is_beam])

    # Response features (optional, 2維)
    if add_response_features:
        features.extend([
            member.stress_ratio,
            member.drift_ratio
        ])

    edge_features.append(features)
```

### Story Batch Indexing

`story_batch` tensor將每個edge分配到對應的story member:

```python
# 範例: 6層建築
# story_batch: [num_edges] tensor

story_batch 編碼:
  story 0, x-beam:      index 0
  story 0, z-beam:      index 1
  story 0, outer-col:   index 2
  story 0, inner-col:   index 3
  story 1, x-beam:      index 4
  story 1, z-beam:      index 5
  ...
  story 5, inner-col:   index 23

Total unique indices = 6 stories × 4 types = 24
```

**檔案位置:** `Structure/structure.py` (in `Structure` class initialization)

這個indexing使得StateGNN能夠將edge embeddings正確聚合到對應的story member categories。

---

## PPO訓練流程

### Training Loop (train_ppo.py)

```python
for episode in range(num_epochs):
    # 1. 訓練一個episode
    score, loss_dict = train_episode(ppo_agent, env, rec, logger)

    # 2. 收集經驗到buffer
    #    (在train_episode內部，每個step都會調用buffer.add_step())

    # 3. 當收集夠5個episodes時自動更新
    if len(buffer) >= accumulate_episodes:  # accumulate_episodes=5
        ppo_agent.update()

    # 4. 定期測試
    if (episode + 1) % test_frequency == 0:
        test_scores = []
        for run in range(test_runs):
            test_score, design_process = test_episode(ppo_agent, env, rec, logger)
            test_scores.append(test_score)
```

### Single Episode Training (train_utils.py)

```python
def train_episode(agent, env, rec, logger):
    structure = env.reset()
    graph = structure.graph.clone()
    score = 0
    done = False

    while not done:
        # 1. Extract features (no grad)
        story_features, global_features = agent.get_features(graph, structure)
        # story_features: [N, 200]
        # global_features: [1, 200]

        # 2. Choose action
        action, log_prob, value, entropy = agent.choose_action(
            story_features, global_features, structure, greedy=False
        )
        # action: int (index of selected story member)
        # log_prob: tensor (for policy gradient)
        # value: tensor (critic's estimate)
        # entropy: tensor (for exploration bonus)

        # 3. Step environment
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # 4. Store experience in buffer
        agent.buffer.add_step(
            state=story_features,
            global_state=global_features,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            valid_mask=valid_mask,
            graph=graph.clone(),      # Store for recomputing features!
            structure=structure       # Store for recomputing features!
        )

        graph = structure.graph.clone()
        score += reward

    # 5. Finish episode
    agent.buffer.finish_episode()
    agent._number_episodes += 1

    # 6. Update if enough episodes collected
    if len(agent.buffer) >= agent.accumulate_episodes:
        loss_dict = agent.update()

    return score, loss_dict
```

### PPO Update (ppo_agent.py)

這是**最關鍵**的部分！

```python
def update(self):
    """
    PPO update當收集到5個episodes後調用
    進行4次epochs的優化
    """
    if len(self.buffer) < self.accumulate_episodes:
        return {}

    episodes = self.buffer.get_all_episodes()  # 取得所有5個episodes

    # ===== Step 1: Flatten all episodes =====
    all_states = []
    all_global_states = []
    all_actions = []
    all_old_log_probs = []
    all_returns = []
    all_advantages = []
    all_valid_masks = []
    all_graphs = []        # 關鍵: 儲存graphs以便重新計算features
    all_structures = []    # 關鍵: 儲存structures

    for episode in episodes:
        # Compute returns and advantages
        returns = self.compute_returns(episode['rewards'], gamma=0.99)
        advantages = self.compute_advantages(returns, episode['values'])

        # Flatten
        all_states.extend(episode['states'])
        all_actions.extend(episode['actions'])
        all_old_log_probs.extend(episode['log_probs'])
        all_returns.append(returns)
        all_advantages.append(advantages)
        all_graphs.extend(episode['graphs'])
        all_structures.extend(episode['structures'])

    # Convert to tensors
    old_log_probs = torch.stack(all_old_log_probs).detach()
    returns = torch.cat(all_returns)
    advantages = torch.cat(all_advantages)

    # ===== Step 2: PPO epochs (重複優化4次) =====
    for epoch in range(self.ppo_epochs):  # ppo_epochs=4

        # ===== BATCH OPTIMIZATION (關鍵創新!) =====
        # 一次forward所有graphs以增強StateGNN的梯度

        # 2.1 Batch所有graphs成一個大batch
        batched_graph, batched_story_batch, structure_story_ptr, num_stories = \
            self._batch_all_graphs(all_graphs, all_structures)

        # batched_graph: 包含所有timesteps的graphs合併成一個大graph
        # batched_story_batch: 調整offset後的story indices
        # structure_story_ptr: [0, n1, n1+n2, ..., total]
        # num_stories: [n1, n2, n3, ...] 每個graph有多少stories

        # 2.2 Create edge-level batch index
        edge_batch = batched_graph.batch[batched_graph.edge_index[0]]
        edge_batch_half = edge_batch[::2]  # StateGNN只用單向edges

        # 2.3 一次forward StateGNN處理所有graphs (關鍵!)
        all_story_features_batched = self.state_gnn(
            batched_graph.x,
            batched_graph.edge_index,
            batched_graph.edge_attr,
            edge_batch_half,        # Edge-level batch index
            batched_story_batch,
            structure_story_ptr
        )  # [total_stories_all_timesteps, 200]

        # 2.4 分割batched features回individual timesteps
        story_features_list = []
        global_features_list = []
        start_idx = 0
        for num in num_stories:
            end_idx = start_idx + num
            story_feat = all_story_features_batched[start_idx:end_idx]
            global_feat = story_feat.mean(dim=0, keepdim=True)
            story_features_list.append(story_feat)
            global_features_list.append(global_feat)
            start_idx = end_idx

        # 2.5 使用cached features計算loss
        new_log_probs = []
        new_values = []
        new_entropies = []

        for t in range(len(all_states)):
            story_features = story_features_list[t]  # 帶梯度!
            global_features = global_features_list[t]
            action = all_actions[t]
            valid_mask = all_valid_masks[t]

            # Forward policy and value
            action_scores = self.policy_net(story_features)
            masked_scores = action_scores.masked_fill(~valid_mask, -1e9)
            probs = F.softmax(masked_scores, dim=0)
            dist = Categorical(probs)

            new_log_prob = dist.log_prob(torch.tensor(action, device=device))
            new_log_probs.append(new_log_prob)
            new_entropies.append(dist.entropy())

            new_value = self.value_net(global_features)
            new_values.append(new_value)

        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.cat(new_values).squeeze()
        new_entropies = torch.stack(new_entropies)

        # 2.6 Compute PPO losses
        # Policy loss: clipped surrogate objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(new_values, returns)

        # Entropy loss: encourage exploration
        entropy_loss = -new_entropies.mean()

        # Total loss
        total_loss = (
            policy_loss +
            value_loss_coef * value_loss +     # value_loss_coef=0.001
            entropy_coef * entropy_loss         # entropy_coef anneals
        )

        # 2.7 Backward and optimize
        self.state_gnn_optimizer.zero_grad()
        self.policy_value_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.state_gnn.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            max_grad_norm
        )

        # Step optimizers
        self.state_gnn_optimizer.step()          # LR = 3e-4 * 100
        self.policy_value_optimizer.step()       # LR = 3e-4

    # Clear buffer
    self.buffer.clear()

    return loss_dict
```

### Key Training Details

**1. Returns Computation (unnormalized):**
```python
def compute_returns(rewards, gamma):
    T = len(rewards)
    returns = torch.zeros(T)
    R = 0
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R
        returns[t] = R
    # 不做normalize，保持絕對值scale
    return returns
```

**2. Advantages Computation (normalized):**
```python
def compute_advantages(returns, values):
    values_tensor = torch.cat(values).squeeze()
    advantages = returns - values_tensor
    # Normalize for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages
```

**3. Entropy Annealing:**
```python
def get_entropy_coef():
    return entropy_coef_initial * (entropy_decay ** number_episodes)
    # entropy_coef_initial = 0.01
    # entropy_decay = 0.995
    # 隨著訓練進行，entropy bonus逐漸減少
```

---

## 批次優化機制

### 問題背景

在PPO中，StateGNN需要被訓練，但存在梯度消失問題:
- StateGNN在action selection時使用`torch.no_grad()`
- 在update時需要重新計算features以獲得梯度
- 但如果逐個timestep forward，StateGNN只看到很小的batch → 梯度很小

### 解決方案: Batch Optimization

**關鍵想法:** 一次forward所有timesteps的graphs，增強StateGNN的梯度。

**實作步驟:**

```python
def _batch_all_graphs(all_graphs, all_structures):
    """
    將多個timesteps的graphs合併成一個大batch

    輸入:
        all_graphs: [graph1, graph2, ..., graphT]  (T個timesteps)
        all_structures: [struct1, struct2, ..., structT]

    輸出:
        batched_graph: PyG Batch object
        batched_story_batch: adjusted story indices
        structure_story_ptr: boundary pointers
        num_stories_per_graph: [n1, n2, ..., nT]
    """
    # 1. 使用PyG的Batch合併所有graphs
    data_list = [g.to(device) for g in all_graphs]
    batched_graph = Batch.from_data_list(data_list)
    # batched_graph.x: [total_nodes, 8]
    # batched_graph.edge_attr: [total_edges*2, 18]
    # batched_graph.batch: [total_nodes] node-level batch index

    # 2. 調整story_batch的offset
    story_batch_list = [s['story_batch'] for s in all_structures]
    story_batch_offset = 0
    all_story_batches = []
    num_stories_per_graph = []

    for sb in story_batch_list:
        num_stories = sb.max().item() + 1  # e.g., 24 for 6-story building
        num_stories_per_graph.append(num_stories)

        # 調整offset
        adjusted_sb = sb + story_batch_offset
        all_story_batches.append(adjusted_sb)

        story_batch_offset += num_stories

    batched_story_batch = torch.cat(all_story_batches)
    # batched_story_batch: [total_edges]
    # 範例: [0,1,2,...,23, 24,25,...,47, 48,49,...] for 3 graphs with 24 stories each

    # 3. 創建structure_story_ptr
    structure_story_ptr = [0]
    for num in num_stories_per_graph:
        structure_story_ptr.append(structure_story_ptr[-1] + num)
    # 範例: [0, 24, 48, 72] for 3 structures

    return batched_graph, batched_story_batch, structure_story_ptr, num_stories_per_graph
```

**Edge-level Batch Index:**

StateGNN需要edge-level的batch index (因為它主要處理edges):

```python
# batched_graph.batch 是 node-level: [total_nodes]
# 但StateGNN需要 edge-level batch index: [total_edges]

# 解決: 從edge的source node取batch index
edge_batch = batched_graph.batch[batched_graph.edge_index[0]]  # [total_edges*2]

# StateGNN只用單向edges (edge_index[::2])
edge_batch_half = edge_batch[::2]  # [total_edges]
```

**Forward所有graphs:**

```python
all_story_features = state_gnn(
    batched_graph.x,                 # [total_nodes, 8]
    batched_graph.edge_index,        # [2, total_edges*2]
    batched_graph.edge_attr,         # [total_edges*2, 18]
    edge_batch_half,                 # [total_edges] - edge-level batch!
    batched_story_batch,             # [total_edges] - adjusted story indices
    structure_story_ptr              # [num_structures+1]
)
# Output: [total_stories, 200]
# 範例: 如果有50個timesteps，每個24 stories → [1200, 200]
```

**分割回individual features:**

```python
story_features_list = []
start_idx = 0
for num_stories in num_stories_per_graph:
    end_idx = start_idx + num_stories
    story_feat = all_story_features[start_idx:end_idx]  # 帶梯度!
    story_features_list.append(story_feat)
    start_idx = end_idx

# 然後用這些features計算policy/value loss
# 梯度會正確地back-propagate到StateGNN
```

### 效果

- **Before batch optimization:** StateGNN gradient norm ~ 1e-6
- **After batch optimization:** StateGNN gradient norm ~ 1e-3
- **Combined with 100× learning rate:** StateGNN updates comparable to Policy/Value networks

---

## 超參數設定

### Model Hyperparameters

```python
# StateGNN
node_feature_dim = 8           # with structure geometry
edge_feature_dim = 18          # with response features, else 16
hidden_dim = 100
num_layers = 3                 # GAT layers

# Policy & Value Networks
state_dim = 200                # hidden_dim * 2
policy_hidden = 100
value_hidden = 100
```

### PPO Hyperparameters

```python
# Learning rates
lr = 3e-4                              # base learning rate
state_gnn_lr_multiplier = 100.0        # StateGNN LR = 3e-2
# Separate optimizers:
#   state_gnn_optimizer: lr = 3e-4 * 100 = 0.03
#   policy_value_optimizer: lr = 3e-4

# Discount & GAE
gamma = 0.99                           # discount factor
# (No GAE lambda, using Monte Carlo returns)

# PPO clip
clip_epsilon = 0.2                     # PPO clipping range [0.8, 1.2]

# Loss coefficients
value_loss_coef = 0.001                # very small value loss weight
entropy_coef_initial = 0.01            # initial entropy bonus
entropy_decay = 0.995                  # exponential decay per episode

# Gradient clipping
max_grad_norm = 2.0                    # gradient clipping threshold

# Update frequency
ppo_epochs = 4                         # optimization epochs per update
accumulate_episodes = 5                # collect 5 episodes before update
```

### Training Hyperparameters

```python
# Training
num_epoch = 10                         # total episodes (epoch == episode)
test_frequency = 2                     # test every 2 episodes
test_runs = 1                          # number of test runs per evaluation
random_seed = 731
```

### Environment Settings

```python
# Structure
structure_shape = "random"             # "fixed", "small_random", "random"
add_structure_geometry = True          # add 3 geometric features to nodes
add_response_features = False          # add 2 response features to edges

# Reward
reward_type = "material"               # "material", "acceleration", "displacement",
                                       # "normalized", "total", "combined"

# Nonlinear dynamic analysis
do_nonlinear_dynamic_analysis = False
check_acceleration = False
check_displacement = True
```

---

## 環境與獎勵

### Environment (RL/environment.py)

**State:** 建築結構的graph representation
- Nodes: joints (nodes of the structure)
- Edges: members (beams and columns)
- Node features: coordinates, geometry
- Edge features: section properties, member type, floor index, responses

**Action Space:** 動態 (取決於建築大小)
- Action = integer index in [0, num_stories-1]
- 每個action對應一個story member (x-beam, z-beam, outer-col, inner-col)
- Invalid actions (已minimum section) 被mask掉

**Reward:** 基於材料節省
```python
def calculate_reward(whether_pass: bool):
    if whether_pass:
        volume_saved = saved_material_record[-1]
        volume_saved_SCWB = saved_material_record_SCWB[-1]  # Strong-Column-Weak-Beam adjustment

        if "material" in reward_type:
            reward = volume_saved
            if "total" in reward_type:
                reward += volume_saved_SCWB
            if "normalized" in reward_type:
                reward /= initial_material_usage

        # Other reward types: "acceleration", "displacement", "combined"
    else:
        reward = 0.0

    return reward
```

**Done Condition:**
1. **Fail:** 結構不滿足constraints (stress, drift, etc.)
2. **Success:** 所有story members都已達到minimum section

### Step Function

```python
def step(structure, action):
    # 1. Update structure based on action
    material_saved = structure.update_action(action)

    # 2. (Optional) SCWB-driven update
    if scwb_driven_design:
        material_saved_SCWB, update_actions_SCWB = \
            strong_column_weak_beam_driven_update(structure)

    # 3. Check constraints
    # 3.1 Linear static analysis
    load_cases, static_responses = check.get_response(structure)
    whether_pass, fail_name, fail_reason = check.check_pass(...)

    # 3.2 Nonlinear dynamic analysis (optional)
    if do_nonlinear_dynamic_analysis and whether_pass:
        dynamic_responses = check_nda.get_response(structure, nda_simulator, ...)
        whether_pass, fail_name, fail_reason = check_nda.check_pass(...)

    # 4. Update graph with new response features
    structure.update_graph_GraphRL(static_response_features, dynamic_response_features)

    # 5. Calculate reward
    reward = calculate_reward(whether_pass)

    # 6. Check done
    if not whether_pass:
        done = True
    elif sum(structure.story_level_sections) == 0:
        done = True  # all minimum sections
        fail_reason = "minimum_section"
    else:
        done = False

    return structure, reward, done, fail_name, fail_reason
```

### Structure Variations

**Training:**
```python
if structure_shape == "random":
    x_span_num = random.randint(2, 7)      # 2-6 spans
    z_span_num = random.randint(2, 7)      # 2-6 spans
    x_span_len = random.randint(6, 9) * 1000  # 6-8m
    z_span_len = random.randint(6, 9) * 1000  # 6-8m
    story_num = random.randint(4, 8)       # 4-7 stories
    story_height = 3200                    # fixed 3.2m
```

**Testing:**
```python
# Fixed testing structure for consistent evaluation
x_span_num = 4
z_span_num = 4
story_num = 6
# → 6 stories × 4 member types = 24 story members
```

---

## 總結

### 模型關鍵創新點

1. **Graph-based Representation with Rich Edge Features**
   - Edge features包含member type和floor information的one-hot encoding
   - 使GNN能夠區分不同構件類型和樓層位置

2. **Story-level Hierarchical Action Space**
   - Actions對應story members而非individual members
   - 減少action space並提供結構化的設計過程

3. **Batch Optimization for StateGNN**
   - 一次forward多個timesteps以增強梯度
   - 配合100×學習率使StateGNN能有效學習

4. **Dynamic Action Masking**
   - 處理varying structure sizes
   - 防止選擇invalid actions (已minimum section)

5. **Separate Optimizers with Different Learning Rates**
   - StateGNN: lr = 0.03 (高learning rate補償小梯度)
   - Policy/Value: lr = 3e-4 (標準learning rate)

### 訓練流程總結

```
Episode Loop:
  ├─ Collect 5 episodes → Buffer
  │   └─ Each step: store (graph, structure, action, reward, log_prob, value)
  │
  ├─ PPO Update (when buffer full):
  │   ├─ Batch all graphs from 5 episodes
  │   ├─ Forward StateGNN once for all timesteps
  │   ├─ Compute policy & value losses
  │   └─ Repeat 4 epochs
  │
  └─ Test every 2 episodes:
      └─ Greedy evaluation on fixed test structure
```

### 輸入輸出維度總結

| Component | Input Dimension | Output Dimension |
|-----------|----------------|------------------|
| Graph nodes | `[num_nodes, 8]` | - |
| Graph edges | `[num_edges*2, 18]` | - |
| StateGNN | `[num_nodes, 8]`, `[num_edges*2, 18]` | `[num_stories, 200]` |
| PolicyNetwork | `[num_stories, 200]` | `[num_stories]` |
| ValueNetwork | `[1, 200]` | `[1, 1]` |
| Action | - | integer in `[0, num_stories-1]` |

---

## 附錄: 重要檔案清單

### 訓練與模型
- `train_ppo.py` - PPO訓練主程式
- `RL/pg/ppo_agent.py` - PPO agent實作 (包含batch optimization)
- `RL/pg/base_agent.py` - Base policy gradient agent
- `RL/pg/networks.py` - Policy和Value networks
- `RL/pg/buffer.py` - Experience buffer
- `RL/pg/train_utils.py` - 訓練輔助函數
- `RL/model.py` - StateGNN定義

### 環境與結構
- `RL/environment.py` - RL環境
- `Structure/structure.py` - 結構定義與graph創建
- `Structure/check.py` - 線性靜力分析檢查
- `Structure/check_nda.py` - 非線性動力分析檢查

### 實用工具
- `RL/record.py` - 訓練記錄
- `Visualization/` - 可視化工具

---

**文檔結束**

如有任何疑問或需要進一步說明，請隨時提出！
