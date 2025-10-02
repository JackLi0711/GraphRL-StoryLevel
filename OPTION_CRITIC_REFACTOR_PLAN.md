# Option-Critic 原文對齊與動態 Action Size 修正計畫

## 執行日期
創建日期: 2025-10-02
更新日期: 2025-10-02 (v2 - 加入動態 action size 架構重構)

## 計畫概述
本計畫旨在：
1. 將現有的 Option-Critic 實現與原文 (Bacon et al., 2016) 完全對齊
2. **重構 Intra-Option Policy 以支持動態 action size**（每個 episode 的樓層數可能不同）
3. 確保所有組件符合原文的理論框架

---

## 🚨 核心問題：動態 Action Size

### **問題描述**
當前系統需要處理不同樓層數的建築結構：
- Episode 1: 4層樓 → 4 floors × 4 members/floor = **16 actions**
- Episode 2: 7層樓 → 7 floors × 4 members/floor = **28 actions**
- Episode 3: 10層樓 → 10 floors × 4 members/floor = **40 actions**

### **當前架構的問題**
```python
# RL/option_critic_gnn.py line 102-103
self.options_W = nn.Parameter(torch.zeros(num_options, gnn_output_dim, num_actions))
self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
```

這個設計**硬編碼了 `num_actions`**，無法處理動態變化的 action space！

### **解決方案：Per-Story-Member Scoring**
不要直接輸出固定維度的 action logits，而是：
1. 對每個 **story member** 計算一個 score
2. Story member 的數量隨樓層數動態變化
3. Policy 輸出的是 "選擇哪個 story member 減小斷面" 的分佈

**理論等價性**：
- 原文：`π(a|s,ω)` over fixed action set A
- 我們：`π(m|s,ω)` over dynamic member set M
- 其中 action a ↔ story member m (選擇哪個構件減小斷面)

---

## 問題分析總結

### 當前實現與原文的主要差異

#### 1. 🔴 **Intra-Option Policy 架構不支持動態 Action Size**
- **當前問題**: 使用固定維度的參數矩陣 `options_W[num_options, gnn_output_dim, num_actions]`
- **需要修改**: 改為 per-member scoring network
- **影響範圍**:
  - 模型初始化
  - `get_action` 方法
  - Optimizer 參數分組
  - 測試和驗證邏輯

#### 2. ❌ **Intra-Option Policy Gradient 使用錯誤的 Advantage**
- **原文要求 (Theorem 1)**:
  ```
  ∇_θ ρ = Σ μ_Ω(s,ω) Σ_a [∂π(a|s,ω)/∂θ] * Q_U(s,ω,a)
  ```
  其中 `Q_U(s,ω,a)` 是 action-value function

- **當前實現**:
  ```python
  advantage = gt.detach() - Q[:, option]
  policy_loss = -logp * advantage
  ```

- **錯誤**: 使用了 TD target (`gt`) 減去 Q_Ω，但應該直接使用 `Q_U(s,ω,a)`

#### 3. ⚠️ **缺少 Q_U 的計算機制**
- **原文說明 (Page 4)**:
  > "Learning Q_U in addition to Q_Ω is computationally wasteful. A practical solution is to only learn Q_Ω and derive an estimate of Q_U from it."

- **推導公式**:
  ```
  Q_U(s,ω,a) = r(s,a) + γ * U(ω,s')
  U(ω,s') = (1 - β(s')) * Q_Ω(s',ω) + β(s') * V_Ω(s')
  ```

#### 4. ⚠️ **Termination Gradient 的 Baseline 不精確**
- **原文要求 (Theorem 2)**:
  ```
  A_Ω(s',ω) = Q_Ω(s',ω) - V_Ω(s')
  ```
- **當前問題**: 用 `Q_max` 近似 `V_Ω(s')`

#### 5. ✅ **Q_Ω 和 Termination Networks 不受影響**
- `Q`: `Linear(member_state_dim, num_options)` - ✅ 無需修改
- `terminations`: `Linear(member_state_dim, num_options)` - ✅ 無需修改
- 這兩個網絡只依賴 global state，與 action size 無關

---

## 修正計畫 - 詳細步驟

### **階段 0: 重構 Intra-Option Policy 為 Per-Member Scoring** 🔴 **[新增]**

這是整個計畫的**基礎**，必須最先完成！

#### Task 0.1: 修改 OptionCriticGNN 的 `__init__` 方法
**檔案**: `RL/option_critic_gnn.py`
**位置**: line 19-110

**修改內容**:

```python
def __init__(self,
             node_feature_dim: int,
             edge_feature_dim: int,
             hidden_dim: int,
             member_state_dim: int,
             num_layers: int,
             num_actions: int,  # ⚠️ 這個參數改為可選，僅用於向後兼容
             num_options: int,
             temperature: float = 1.0,
             eps_start: float = 1.0,
             eps_min: float = 0.1,
             eps_decay: int = int(1e6),
             eps_test: float = 0.05,
             device: str = 'cpu',
             testing: bool = False,
             debug_logging: bool = False):
    """
    Initialize OptionCriticGNN with support for dynamic action sizes.

    Key architectural change:
    - Instead of fixed-size action logits, we use per-story-member scoring
    - This allows handling varying numbers of story members across episodes

    Args:
        ... (same as before)
        num_actions: [DEPRECATED] Not used in new architecture, kept for compatibility
    """
    super(OptionCriticGNN, self).__init__()

    # Store parameters
    self.node_feature_dim = node_feature_dim
    self.edge_feature_dim = edge_feature_dim
    self.hidden_dim = hidden_dim
    self.member_state_dim = member_state_dim
    self.num_layers = num_layers
    self.num_actions = num_actions  # Kept for compatibility, not used
    self.num_options = num_options
    self.device = torch.device(device)
    self.testing = testing
    self.debug_logging = debug_logging

    # Exploration parameters
    self.temperature = temperature
    self.eps_min = eps_min
    self.eps_start = eps_start
    self.eps_decay = eps_decay
    self.eps_test = eps_test
    self.num_steps = 0

    # StateGNN for feature extraction
    self.state_gnn = StateGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        member_state_dim=member_state_dim,
        num_layers=num_layers
    )

    # GNN output dimension
    gnn_output_dim = member_state_dim * 2

    # =========================================================================
    # Option-Critic Components
    # =========================================================================

    # 1. Policy-Over-Options (uses global state)
    self.Q = nn.Linear(member_state_dim, num_options)

    # 2. Termination functions (uses global state)
    self.terminations = nn.Linear(member_state_dim, num_options)

    # 3. Intra-Option Policies (NEW ARCHITECTURE)
    # ❌ OLD (REMOVED):
    # self.options_W = nn.Parameter(torch.zeros(num_options, gnn_output_dim, num_actions))
    # self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

    # ✅ NEW: Per-story-member scoring networks
    # Each option has its own MLP that scores story members
    self.intra_option_policies = nn.ModuleList([
        nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Output: score for each story member
        )
        for _ in range(num_options)
    ])

    # Initialize parameters
    self._initialize_parameters()

    # Move to device
    self.to(self.device)
    self.train(not testing)
```

**關鍵變化**:
1. **刪除**: `options_W` 和 `options_b` 參數
2. **新增**: `intra_option_policies` - ModuleList of MLPs
3. 每個 MLP 輸入: `[gnn_output_dim]`，輸出: `[1]` (單個 score)
4. 應用於每個 story member，得到 `[num_story_members]` 個 scores

---

#### Task 0.2: 修改 `_initialize_parameters` 方法
**檔案**: `RL/option_critic_gnn.py`
**位置**: line 112-131

**修改內容**:

```python
def _initialize_parameters(self):
    """Initialize network parameters."""

    # ✅ NEW: Initialize intra-option policy networks
    for option_idx, option_policy in enumerate(self.intra_option_policies):
        for layer in option_policy:
            if isinstance(layer, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # ❌ OLD (REMOVED):
    # nn.init.normal_(self.options_W, mean=0.0, std=0.01)
    # nn.init.zeros_(self.options_b)

    # Initialize Q network (unchanged)
    if hasattr(self.Q, 'weight'):
        nn.init.xavier_uniform_(self.Q.weight)
        if hasattr(self.Q, 'bias') and self.Q.bias is not None:
            nn.init.zeros_(self.Q.bias)

    # Initialize termination network (unchanged)
    # Start with low termination probabilities (~0.1)
    nn.init.xavier_uniform_(self.terminations.weight)
    nn.init.constant_(self.terminations.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1
```

---

#### Task 0.3: 完全重寫 `get_action` 方法
**檔案**: `RL/option_critic_gnn.py`
**位置**: line 240-420

**新實現**:

```python
def get_action(self,
               story_level_state: torch.Tensor,
               structure_story_ptr: Optional[List[int]],
               option: int,
               valid_actions_mask: torch.Tensor = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Select action by scoring story members (NEW ARCHITECTURE).

    Key change: Instead of computing logits for fixed action space,
    we score each story member and select which one to modify.

    Args:
        story_level_state: Story-level state features
                          Shape: [total_story_member_num, gnn_output_dim]
                          - For 4-story building: [16, gnn_output_dim]
                          - For 7-story building: [28, gnn_output_dim]
                          - For 10-story building: [40, gnn_output_dim]
        structure_story_ptr: NOT USED in new architecture (kept for compatibility)
        option: Current option index
        valid_actions_mask: Boolean tensor indicating which story members can be modified
                          Shape: [total_story_member_num]
                          True = can modify this member, False = cannot

    Returns:
        (action_index, log_prob, entropy)
        action_index: Which story member to reduce (0 to total_story_member_num-1)
    """
    num_story_members = story_level_state.shape[0]

    if self.debug_logging:
        print(f"DEBUG get_action: num_story_members={num_story_members}, option={option}")
        if valid_actions_mask is not None:
            print(f"DEBUG get_action: valid_count={valid_actions_mask.sum().item()}/{num_story_members}")

    # =========================================================================
    # Step 1: Compute score for each story member using option-specific MLP
    # =========================================================================
    # story_level_state: [N, gnn_output_dim] where N = num_story_members
    # Output: [N, 1]
    scores = self.intra_option_policies[option](story_level_state)  # [N, 1]
    scores = scores.squeeze(-1)  # [N]

    if self.debug_logging:
        print(f"DEBUG get_action: scores range=[{scores.min().item():.3f}, {scores.max().item():.3f}]")

    # =========================================================================
    # Step 2: Apply valid actions mask
    # =========================================================================
    if valid_actions_mask is not None:
        # Ensure mask is on correct device
        if valid_actions_mask.device != scores.device:
            valid_actions_mask = valid_actions_mask.to(scores.device)

        # Ensure mask has correct shape [num_story_members]
        if valid_actions_mask.shape[0] != num_story_members:
            raise ValueError(
                f"Mask shape mismatch: expected [{num_story_members}], "
                f"got {valid_actions_mask.shape}"
            )

        # Mask out invalid actions with very negative scores
        scores = scores.masked_fill(~valid_actions_mask, -1e8)

        if self.debug_logging:
            print(f"DEBUG get_action: After masking, valid_scores={scores[valid_actions_mask].shape[0]}")

    # =========================================================================
    # Step 3: Convert scores to probabilities
    # =========================================================================
    action_probs = (scores / self.temperature).softmax(dim=0)  # [N]

    # Additional safety: ensure no invalid actions have probability
    if valid_actions_mask is not None:
        # Zero out probabilities for invalid actions
        action_probs = action_probs * valid_actions_mask.float()

        # Renormalize
        prob_sum = action_probs.sum()
        if prob_sum > 0:
            action_probs = action_probs / prob_sum
        else:
            # Emergency: all actions masked out
            if valid_actions_mask.sum() > 0:
                # Uniform over valid actions
                action_probs = valid_actions_mask.float() / valid_actions_mask.sum().float()
            else:
                raise ValueError("No valid actions available!")

    # =========================================================================
    # Step 4: Sample action
    # =========================================================================
    action_dist = Categorical(action_probs)
    action = action_dist.sample()

    logp = action_dist.log_prob(action)
    entropy = action_dist.entropy()

    # =========================================================================
    # Step 5: Validation
    # =========================================================================
    action_idx = action.item()

    # Check if action is valid
    if valid_actions_mask is not None:
        if not valid_actions_mask[action_idx].item():
            if self.debug_logging:
                print(f"ERROR: Selected invalid action {action_idx}!")
            # Emergency fallback
            valid_indices = torch.where(valid_actions_mask)[0]
            if len(valid_indices) > 0:
                action_idx = valid_indices[0].item()
                logp = torch.log(action_probs[action_idx] + 1e-8)
            else:
                raise ValueError("No valid actions and fallback failed!")

    if self.debug_logging:
        print(f"DEBUG get_action: Selected action={action_idx}, logp={logp.item():.4f}, entropy={entropy.item():.4f}")

    return action_idx, logp, entropy
```

**關鍵變化**:
1. **輸入**: `story_level_state` shape 可變 `[N, gnn_output_dim]`
2. **處理**: 對每個 story member 計算 score
3. **輸出**: 選擇哪個 story member (index 0 to N-1)
4. **無需**: `structure_story_ptr`（舊架構的遺留參數）

---

#### Task 0.4: 更新 Optimizer 設置
**檔案**: `RL/option_critic/trainer.py`
**位置**: `_setup_optimizers` 方法 (line 182-220)

**修改內容**:

```python
def _setup_optimizers(self):
    """Setup optimizers for different model components."""

    # =========================================================================
    # Separate parameters for different components
    # =========================================================================

    # 1. Termination parameters
    termination_params = [p for n, p in self.oc.named_parameters()
                         if n.startswith('terminations')]

    # 2. Intra-option policy parameters (NEW)
    # ❌ OLD:
    # policy_params = [self.oc.options_W, self.oc.options_b]

    # ✅ NEW: Collect all parameters from intra_option_policies ModuleList
    policy_params = []
    for option_policy in self.oc.intra_option_policies:
        policy_params.extend(option_policy.parameters())

    # 3. Critic (Q network) parameters
    critic_params = [p for n, p in self.oc.named_parameters()
                    if n.startswith('Q.')]

    # 4. StateGNN parameters (shared across all components)
    state_gnn_params = [p for n, p in self.oc.named_parameters()
                       if n.startswith('state_gnn')]

    # 5. Feature processor parameters (if exists)
    feature_params = [p for n, p in self.oc.named_parameters()
                     if n.startswith('feature_processor')]

    # =========================================================================
    # Create optimizers with different learning rates
    # =========================================================================

    gnn_lr = self.args.actor_lr * 0.1  # Lower LR for GNN
    termination_lr = self.args.actor_lr * self.args.termination_lr_ratio

    # Actor optimizer (policy + termination + shared GNN)
    actor_optimizer = optim.Adam([
        {"params": policy_params, "lr": self.args.actor_lr, "name": "policy"},
        {"params": termination_params, "lr": termination_lr, "name": "termination"},
        {"params": state_gnn_params, "lr": gnn_lr, "name": "gnn_actor"},
        {"params": feature_params, "lr": self.args.actor_lr, "name": "features_actor"},
    ])

    # Critic optimizer (Q network + shared GNN)
    critic_optimizer = optim.Adam([
        {"params": critic_params, "lr": self.args.critic_lr, "name": "critic"},
        {"params": state_gnn_params, "lr": gnn_lr, "name": "gnn_critic"},
        {"params": feature_params, "lr": self.args.critic_lr, "name": "features_critic"},
    ])

    # Log configuration
    self.logger.info(f"Optimizer configuration:")
    self.logger.info(f"  Policy parameters: {sum(p.numel() for p in policy_params)} params, LR={self.args.actor_lr}")
    self.logger.info(f"  Termination parameters: {sum(p.numel() for p in termination_params)} params, LR={termination_lr}")
    self.logger.info(f"  Critic parameters: {sum(p.numel() for p in critic_params)} params, LR={self.args.critic_lr}")
    self.logger.info(f"  StateGNN parameters: {sum(p.numel() for p in state_gnn_params)} params, LR={gnn_lr}")

    return actor_optimizer, critic_optimizer
```

---

#### Task 0.5: 更新訓練循環中的模型初始化
**檔案**: `RL/option_critic/trainer.py`
**位置**: `_setup_models` 方法 (line 132-180)

**確認修改**:

```python
def _setup_models(self):
    """Setup Option-Critic models."""
    # Reset environment to infer feature sizes
    structure = self.base_env.reset()
    graph = structure.graph
    node_feature_dim = graph.x.shape[1]
    edge_feature_dim = graph.edge_attr.shape[1]

    # ⚠️ num_actions is now IGNORED in the new architecture
    # We pass it for compatibility, but it's not used
    # The actual action space is determined dynamically by the number of story members
    A = num_actions(structure)  # This can be any value, e.g., max expected actions

    # Create models
    oc = OptionCriticGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=self.args.hidden_dim,
        member_state_dim=self.args.hidden_dim,
        num_layers=self.args.num_layers,
        num_actions=A,  # ⚠️ DEPRECATED parameter, not used
        num_options=self.args.num_options,
        temperature=self.args.temperature,
        eps_start=self.args.eps_start,
        eps_min=self.args.eps_min,
        eps_decay=self.args.eps_decay,
        eps_test=self.args.eps_test,
        device=self.device,
        testing=False,
        debug_logging=True,
    )

    # ... (rest unchanged)
```

**說明**:
- `num_actions` 參數仍然傳遞，但在新架構中不再使用
- 實際的 action space 由每個 episode 的 `num_story_members` 決定

---

### **階段 1: 添加 Q_U 估計機制**

#### Task 1.1: 在 OptionCriticGNN 添加 `compute_Q_U` 方法
**檔案**: `RL/option_critic_gnn.py`
**位置**: 在 `get_Q` 方法後 (約 line 192)

**實現內容**:
```python
def compute_Q_U(self,
                global_state: torch.Tensor,
                option: int,
                action: int,
                reward: float,
                next_global_state: torch.Tensor,
                gamma: float = 0.99) -> torch.Tensor:
    """
    Compute action-value Q_U(s,ω,a) for a specific (state, option, action) tuple.

    Following Option-Critic paper (Page 4):
    Q_U(s,ω,a) = r(s,a) + γ * U(ω,s')
    U(ω,s') = (1 - β(s')) * Q_Ω(s',ω) + β(s') * V_Ω(s')

    NOTE: 'action' here refers to the story member index selected.
    The action value is independent of the specific action index,
    as Q_U only depends on the reward and next state value.

    Args:
        global_state: Current global state [1, member_state_dim]
        option: Current option index
        action: Executed action (story member index) - not directly used in computation
        reward: Immediate reward r(s,a)
        next_global_state: Next global state [1, member_state_dim]
        gamma: Discount factor

    Returns:
        Q_U value (scalar tensor)
    """
    with torch.no_grad():
        # Compute termination probability β(s') for the current option
        next_beta = self.get_terminations(next_global_state)  # [1, num_options]
        if next_beta.dim() > 1:
            next_beta_omega = next_beta[0, option]  # Scalar
        else:
            next_beta_omega = next_beta[option]

        # Compute Q_Ω(s',ω) and V_Ω(s')
        next_Q = self.get_Q(next_global_state)  # [1, num_options]
        if next_Q.dim() > 1:
            next_Q_omega = next_Q[0, option]  # Q_Ω(s',ω)
            next_V = next_Q[0].max()  # V_Ω(s') = max_ω Q_Ω(s',ω) (greedy)
        else:
            next_Q_omega = next_Q[option]
            next_V = next_Q.max()

        # Compute U(ω,s') - value upon arrival
        U_omega = (1 - next_beta_omega) * next_Q_omega + next_beta_omega * next_V

        # Compute Q_U(s,ω,a) - action value
        Q_U = reward + gamma * U_omega

    return Q_U
```

**注意事項**:
- `action` 參數保留但不直接用於計算（符合原文公式）
- 使用 `torch.no_grad()` 因為這是用於梯度計算的輔助值
- ✅ **與動態 action size 兼容**：計算不依賴 action space 大小

---

#### Task 1.2: 添加批次版本的 `compute_Q_U_batch`
**檔案**: `RL/option_critic_gnn.py`
**位置**: 在 `compute_Q_U` 方法後

**實現內容**:
```python
def compute_Q_U_batch(self,
                      global_states: List[torch.Tensor],
                      options: torch.Tensor,
                      actions: torch.Tensor,
                      rewards: torch.Tensor,
                      next_global_states: List[torch.Tensor],
                      gamma: float = 0.99) -> torch.Tensor:
    """
    Batch version of compute_Q_U for efficient processing.

    Args:
        global_states: List of global state tensors [batch_size]
        options: Option indices [batch_size]
        actions: Action indices [batch_size]
        rewards: Rewards [batch_size]
        next_global_states: List of next global state tensors [batch_size]
        gamma: Discount factor

    Returns:
        Q_U values [batch_size]
    """
    batch_size = len(options)
    Q_U_values = []

    for i in range(batch_size):
        Q_U_i = self.compute_Q_U(
            global_states[i],
            options[i].item(),
            actions[i].item(),
            rewards[i].item(),
            next_global_states[i],
            gamma
        )
        Q_U_values.append(Q_U_i)

    return torch.stack(Q_U_values)
```

**注意事項**:
- 因為圖結構不同，無法直接批次處理，需要逐個計算
- ✅ **與動態 action size 兼容**

---

### **階段 2: 修正 Intra-Option Policy Gradient**

#### Task 2.1: 修改 `actor_loss` 函數以使用正確的 Q_U
**檔案**: `RL/option_critic_gnn.py`
**位置**: `actor_loss` 函數 (line 620-729)

**完整新實現**:
```python
def actor_loss(obs, option: int, action: int, logp: torch.Tensor, entropy: torch.Tensor,
               reward: float, done: bool, next_obs,
               model: OptionCriticGNN, model_prime: OptionCriticGNN,
               gamma: float = 0.99, termination_reg: float = 0.01,
               entropy_reg: float = 0.01) -> torch.Tensor:
    """
    Compute actor loss for Option-Critic following original paper.

    Combines:
    1. Intra-option policy gradient (Theorem 1): -log π(a|s,ω) * Q_U(s,ω,a)
    2. Termination gradient (Theorem 2): β(s',ω) * [A_Ω(s',ω) + ε]

    Args:
        obs: Current observation (graph data tuple)
        option: Current option index
        action: Executed action (story member index) - NEWLY ADDED
        logp: Log probability of selected action
        entropy: Entropy of action distribution
        reward: Reward received
        done: Whether episode terminated
        next_obs: Next observation (graph data tuple)
        model: Current model
        model_prime: Target model
        gamma: Discount factor
        termination_reg: Termination regularization weight (ε in paper)
        entropy_reg: Entropy regularization weight

    Returns:
        Actor loss (combines policy gradient and termination gradient)
    """
    # =========================================================================
    # Extract global states from graph observations
    # =========================================================================
    if isinstance(obs, (tuple, list)) and len(obs) >= 4:
        graph_x, graph_edge_index, graph_edge_attr, story_batch = obs[:4]
        next_graph_x, next_graph_edge_index, next_graph_edge_attr, next_story_batch = next_obs[:4]

        # Use global state for Q_U and termination computation
        _, state = model.get_state(graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
        _, next_state = model.get_state(next_graph_x, next_graph_edge_index,
                                       next_graph_edge_attr, next_story_batch, None)
        _, next_state_prime = model_prime.get_state(next_graph_x, next_graph_edge_index,
                                                   next_graph_edge_attr, next_story_batch, None)
    else:
        # Fallback for non-graph observations
        if isinstance(obs, (tuple, list)):
            obs = obs[0] if len(obs) == 1 else torch.stack(list(obs))
        if isinstance(next_obs, (tuple, list)):
            next_obs = next_obs[0] if len(next_obs) == 1 else torch.stack(list(next_obs))

        state = model.feature_processor(obs) if hasattr(model, 'feature_processor') else obs
        next_state = model.feature_processor(next_obs) if hasattr(model, 'feature_processor') else next_obs
        next_state_prime = model_prime.feature_processor(next_obs) if hasattr(model_prime, 'feature_processor') else next_obs

    # =========================================================================
    # Part 1: Intra-Option Policy Gradient (Theorem 1)
    # =========================================================================
    # Compute Q_U(s,ω,a) using the executed action
    Q_U = model.compute_Q_U(
        global_state=state,
        option=option,
        action=action,
        reward=reward,
        next_global_state=next_state_prime,
        gamma=gamma
    )

    # Policy gradient: -log π(a|s,ω) * Q_U(s,ω,a)
    # (Negative because we're minimizing loss, equivalent to maximizing objective)
    policy_loss = -logp * Q_U.detach() - entropy_reg * entropy

    # =========================================================================
    # Part 2: Termination Gradient (Theorem 2)
    # =========================================================================
    # Get termination probability for current option at next state
    next_termination_probs = model.get_terminations(next_state)  # [1, num_options] or [num_options]

    # Handle dimensions
    if next_termination_probs.dim() > 1:
        next_beta_omega = next_termination_probs[0, option]
    else:
        next_beta_omega = next_termination_probs[option]

    # Compute advantage A_Ω(s',ω) = Q_Ω(s',ω) - V_Ω(s')
    next_Q = model.get_Q(next_state).detach()  # [1, num_options] or [num_options]

    if next_Q.dim() > 1:
        next_Q_omega = next_Q[0, option]  # Q_Ω(s',ω)
        # V_Ω(s') = max_ω Q_Ω(s',ω) for greedy policy-over-options
        next_V = next_Q[0].max()
    else:
        next_Q_omega = next_Q[option]
        next_V = next_Q.max()

    advantage_omega = next_Q_omega - next_V

    # Termination gradient: β(s',ω) * [A_Ω(s',ω) + ε]
    # Only applies if episode hasn't terminated
    termination_loss = next_beta_omega * (advantage_omega + termination_reg) * (1 - done)

    # =========================================================================
    # Combine losses
    # =========================================================================
    # Ensure all components are scalars
    if policy_loss.dim() > 0:
        policy_loss = policy_loss.mean()
    if termination_loss.dim() > 0:
        termination_loss = termination_loss.mean()

    total_actor_loss = policy_loss + termination_loss

    # Optional debug logging
    if hasattr(model, 'debug_logging') and model.debug_logging:
        print(f"[ActorLoss] Policy: {policy_loss.item():.6f}, "
              f"Termination: {termination_loss.item():.6f}, "
              f"Total: {total_actor_loss.item():.6f}")
        print(f"[ActorLoss] Q_U: {Q_U.item():.4f}, "
              f"Advantage: {advantage_omega.item():.4f}, "
              f"β: {next_beta_omega.item():.4f}")

    return total_actor_loss
```

**關鍵變化**:
1. **新增** `action` 參數
2. **使用** `compute_Q_U` 計算 Q_U(s,ω,a)
3. **修正** policy gradient 為 `-logp * Q_U`（而非錯誤的 advantage）
4. **保持** termination gradient 使用 advantage
5. ✅ **與動態 action size 兼容**

---

#### Task 2.2: 更新訓練循環以傳遞 action
**檔案**: `RL/option_critic/trainer.py`
**位置**: `_perform_updates` 方法 (line 383-425)

**修改內容**:
```python
def _perform_updates(self, step_transitions, steps):
    """Perform actor and critic updates."""
    # Accumulate actor losses
    accumulated_actor_losses = []
    if len(step_transitions) > 0:
        for tr in step_transitions:
            a_loss = actor_loss(
                tr["obs"],
                tr["option"],
                tr["action"],  # ✅ NEW: Pass action
                tr["logp"],
                tr["entropy"],
                tr["reward"],
                tr["done"],
                tr["next_obs"],
                self.oc,
                self.oc_prime,
                self.args.gamma,
                self.args.termination_reg,
                self.args.entropy_reg
            )
            accumulated_actor_losses.append(a_loss)

    # ... (rest of method unchanged)
```

---

### **階段 3: 改進 Termination Gradient 的 V_Ω 計算**

#### Task 3.1: 添加 `compute_V_omega` 方法
**檔案**: `RL/option_critic_gnn.py`
**位置**: 在 `get_terminations` 方法後 (約 line 204)

**實現內容**:
```python
def compute_V_omega(self,
                    global_state: torch.Tensor,
                    policy_over_options: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute V_Ω(s) - the value of state s under policy-over-options.

    Two modes:
    1. If policy_over_options provided: V_Ω(s) = Σ_ω π_Ω(ω|s) * Q_Ω(s,ω)
    2. Otherwise (greedy): V_Ω(s) = max_ω Q_Ω(s,ω)

    Args:
        global_state: Global state features [1, member_state_dim]
        policy_over_options: Optional policy distribution [num_options]
                           If None, uses greedy (max) policy

    Returns:
        V_Ω(s) value (scalar tensor)
    """
    Q_omega = self.get_Q(global_state)  # [1, num_options]

    if Q_omega.dim() > 1:
        Q_omega = Q_omega.squeeze(0)  # [num_options]

    if policy_over_options is not None:
        # Expected value under policy
        V = (policy_over_options * Q_omega).sum()
    else:
        # Greedy policy (default for option-critic)
        V = Q_omega.max()

    return V
```

**注意**:
- 已在 Task 2.1 的 `actor_loss` 中使用（inline 計算）
- 可選：將 inline 代碼替換為調用此方法

---

### **階段 4: 測試動態 Action Size**

#### Task 4.1: 創建動態 action size 測試
**檔案**: 新建 `tests/test_dynamic_action_size.py`

**實現內容**:
```python
"""
Test cases for dynamic action size handling in Option-Critic.
Tests the new per-story-member scoring architecture.
"""
import torch
import pytest
from RL.option_critic_gnn import OptionCriticGNN


def create_test_model(num_options=4, hidden_dim=64):
    """Create a test model instance."""
    return OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=hidden_dim,
        member_state_dim=hidden_dim,
        num_layers=2,
        num_actions=100,  # Ignored in new architecture
        num_options=num_options,
        device='cpu',
        testing=False,
        debug_logging=False
    )


def test_intra_option_policy_architecture():
    """Test that intra-option policies are MLPs, not parameter matrices."""
    model = create_test_model()

    # Check that intra_option_policies is a ModuleList
    assert hasattr(model, 'intra_option_policies'), "Model should have intra_option_policies"
    assert isinstance(model.intra_option_policies, torch.nn.ModuleList), \
        "intra_option_policies should be ModuleList"

    # Check that old architecture is removed
    assert not hasattr(model, 'options_W'), "Old options_W should be removed"
    assert not hasattr(model, 'options_b'), "Old options_b should be removed"

    # Check number of option policies
    assert len(model.intra_option_policies) == 4, "Should have 4 option policies"

    # Check that each policy is a Sequential MLP
    for i, policy in enumerate(model.intra_option_policies):
        assert isinstance(policy, torch.nn.Sequential), \
            f"Policy {i} should be Sequential"


def test_get_action_with_varying_num_members():
    """
    Test that get_action works with different numbers of story members.
    Simulates episodes with different building heights.
    """
    model = create_test_model()

    # Test scenarios: different building heights
    test_cases = [
        {"floors": 4, "members_per_floor": 4, "total": 16},   # 4-story building
        {"floors": 7, "members_per_floor": 4, "total": 28},   # 7-story building
        {"floors": 10, "members_per_floor": 4, "total": 40},  # 10-story building
    ]

    gnn_output_dim = model.member_state_dim * 2  # 128

    for case in test_cases:
        num_members = case["total"]

        # Create dummy story-level state with varying size
        story_level_state = torch.randn(num_members, gnn_output_dim)

        # Create valid actions mask (some members can be modified)
        valid_mask = torch.ones(num_members, dtype=torch.bool)
        # Mask out last floor (cannot modify)
        members_per_floor = case["members_per_floor"]
        valid_mask[-members_per_floor:] = False

        # Get action
        action, logp, entropy = model.get_action(
            story_level_state=story_level_state,
            structure_story_ptr=None,
            option=0,
            valid_actions_mask=valid_mask
        )

        # Assertions
        assert isinstance(action, int), f"Action should be int, got {type(action)}"
        assert 0 <= action < num_members, \
            f"Action {action} out of range [0, {num_members})"
        assert valid_mask[action], \
            f"Action {action} should be valid according to mask"
        assert torch.isfinite(logp), "Log probability should be finite"
        assert torch.isfinite(entropy), "Entropy should be finite"

        print(f"✓ {case['floors']}-story building ({num_members} members): "
              f"action={action}, logp={logp.item():.4f}, entropy={entropy.item():.4f}")


def test_compute_Q_U_independent_of_action_size():
    """
    Test that Q_U computation is independent of action space size.
    """
    model = create_test_model()

    # Create dummy states
    global_state = torch.randn(1, 64)
    next_global_state = torch.randn(1, 64)

    # Test with different action indices (representing different story members)
    # The Q_U value should not depend on the action index itself,
    # only on the reward and next state
    action_indices = [0, 10, 20, 50]

    Q_U_values = []
    for action_idx in action_indices:
        Q_U = model.compute_Q_U(
            global_state=global_state,
            option=0,
            action=action_idx,
            reward=1.0,  # Same reward
            next_global_state=next_global_state,  # Same next state
            gamma=0.99
        )
        Q_U_values.append(Q_U.item())

    # All Q_U values should be identical (action index doesn't affect computation)
    for i in range(1, len(Q_U_values)):
        assert abs(Q_U_values[i] - Q_U_values[0]) < 1e-6, \
            f"Q_U should be same for all actions, got {Q_U_values}"

    print(f"✓ Q_U values independent of action index: {Q_U_values[0]:.4f}")


def test_actor_loss_with_different_action_sizes():
    """
    Test that actor_loss works correctly with different episode action sizes.
    """
    from RL.option_critic_gnn import actor_loss

    model = create_test_model()
    model_prime = create_test_model()
    model_prime.load_state_dict(model.state_dict())

    # Simulate different episode sizes
    episode_configs = [
        {"num_members": 16, "action": 5},   # 4-story
        {"num_members": 28, "action": 15},  # 7-story
        {"num_members": 40, "action": 25},  # 10-story
    ]

    for config in episode_configs:
        # Create dummy graph observations
        num_members = config["num_members"]
        action = config["action"]

        # Dummy graph data (simplified)
        graph_x = torch.randn(num_members, 10)
        graph_edge_index = torch.randint(0, num_members, (2, num_members * 2))
        graph_edge_attr = torch.randn(num_members * 2, 5)
        story_batch = torch.zeros(num_members, dtype=torch.long)

        obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
        next_obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)

        # Dummy action parameters
        option = 0
        logp = torch.tensor(-1.5, requires_grad=True)
        entropy = torch.tensor(2.0)
        reward = 1.0
        done = False

        # Compute loss
        loss = actor_loss(
            obs, option, action, logp, entropy, reward, done, next_obs,
            model, model_prime, gamma=0.99
        )

        # Assertions
        assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.requires_grad, "Loss should require gradients"

        print(f"✓ Episode with {num_members} members (action={action}): loss={loss.item():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**執行測試**:
```bash
cd /mnt/d/kyle_MD_project/optionRL/GraphRL_story_level
python -m pytest tests/test_dynamic_action_size.py -v
```

---

### **階段 5: Critic Loss 驗證**

#### Task 5.1: 確認當前 Critic Loss 是否正確
**檔案**: `RL/option_critic_gnn.py`
**位置**: `critic_loss` 函數 (line 454-617)

**分析結論**:
- ✅ 當前 critic loss 實現基本正確
- ✅ 學習 Q_Ω，符合原文 Page 4 的實踐建議
- ✅ **與動態 action size 兼容**（不依賴 action space）
- **決定**: 保持當前實現，無需修改

---

### **階段 6: 添加監控和日誌**

#### Task 6.1: 添加 Q_U 值監控
**檔案**: `RL/option_critic/trainer.py`
**位置**: `_perform_updates` 方法

**新增內容**:
```python
def _perform_updates(self, step_transitions, steps):
    """Perform actor and critic updates."""

    # Debug logging: Monitor Q_U values
    if hasattr(self.args, 'debug_logging') and self.args.debug_logging:
        if len(step_transitions) > 0:
            # Sample first 3 transitions
            for idx, tr in enumerate(step_transitions[:3]):
                _, state = self.oc.get_state(*tr["obs"][:4], None)
                _, next_state = self.oc_prime.get_state(*tr["next_obs"][:4], None)

                Q_U = self.oc.compute_Q_U(
                    global_state=state,
                    option=tr["option"],
                    action=tr["action"],
                    reward=tr["reward"],
                    next_global_state=next_state,
                    gamma=self.args.gamma
                )

                self.logger.debug(
                    f"[Q_U Monitor] Sample {idx}: "
                    f"ω={tr['option']}, a={tr['action']}, "
                    f"r={tr['reward']:.4f}, Q_U={Q_U.item():.4f}"
                )

    # ... (rest of method)
```

---

#### Task 6.2: 添加 Policy/Termination Gradient 監控
已在 Task 2.1 的 `actor_loss` 函數中實現（debug logging 部分）。

---

### **階段 7: 完整測試和驗證**

#### Task 7.1: 單元測試 - 已在 Task 4.1 完成

#### Task 7.2: 整合測試
**檔案**: `tests/test_training_pipeline.py`

**新增測試**:
```python
def test_complete_training_iteration_with_dynamic_actions():
    """
    Test a complete training iteration with episodes of different sizes.
    """
    from RL.option_critic.trainer import OptionCriticTrainer
    from RL.option_critic.config import parse_args

    # Create minimal args for testing
    class TestArgs:
        epochs = 2  # Just 2 episodes
        num_options = 2
        max_option_len = 5
        device = 'cpu'
        batch_size = 4
        actor_lr = 0.001
        critic_lr = 0.001
        gamma = 0.99
        termination_reg = 0.01
        entropy_reg = 0.01
        # ... (other required args)

    args = TestArgs()

    # Create trainer
    trainer = OptionCriticTrainer(args)

    # Run training for 2 episodes (may have different building sizes)
    for episode in range(2):
        episode_score, episode_reward, _, _, _ = trainer.train_episode(episode)

        assert isinstance(episode_score, (int, float)), "Episode score should be numeric"
        assert isinstance(episode_reward, (int, float)), "Episode reward should be numeric"

        print(f"✓ Episode {episode+1}: score={episode_score:.2f}, reward={episode_reward:.2f}")
```

---

#### Task 7.3: 實際訓練驗證
**檔案**: 使用現有的 `train_option_critic.py`

**測試步驟**:
```bash
# Test 1: Short training with dynamic building sizes
python train_option_critic.py \
    --epochs 10 \
    --num_options 2 \
    --eval_frequency 5 \
    --batch_size 16 \
    --device cpu

# Expected behavior:
# - No errors related to action size mismatch
# - Models should handle 4-story, 7-story, 10-story buildings seamlessly
# - Check logs for Q_U values and gradient magnitudes

# Test 2: Verify checkpoint loading
# - Load best_model.pt
# - Run inference on buildings with different heights
# - Confirm policy works correctly
```

---

### **階段 8: 文檔更新**

#### Task 8.1: 更新 docstring
**檔案**: `RL/option_critic_gnn.py`

**更新位置**:
1. Class `OptionCriticGNN` docstring - 說明新架構
2. `__init__` method - 標註 `num_actions` 為 deprecated
3. `get_action` method - 詳細說明 per-member scoring
4. All new methods - 完整 docstring

**示例**:
```python
class OptionCriticGNN(nn.Module):
    """
    Option-Critic network with Graph Neural Network backbone.

    **NEW ARCHITECTURE (v2)**: Supports dynamic action sizes!

    Key changes from v1:
    - Intra-option policies use per-story-member scoring (MLPs)
    - No longer uses fixed-size action logits (options_W, options_b)
    - Can handle buildings with varying numbers of floors

    Architecture components:
    1. StateGNN: Extracts features from graph-structured observations
    2. Q network: Option-value function Q_Ω(s,ω) [global state → num_options]
    3. Termination networks: β(s,ω) [global state → num_options]
    4. Intra-option policies: π(m|s,ω) [story-level state → score per member]

    The action space is dynamic:
    - 4-story building: 16 story members → 16 possible actions
    - 7-story building: 28 story members → 28 possible actions
    - Action = which story member to reduce section
    """
```

---

#### Task 8.2: 更新 README
**檔案**: `GraphRL_story_level/README.md`

**新增內容**:
```markdown
# Option-Critic for Structural Design

## Overview
本專案實現 Option-Critic Architecture (Bacon et al., 2016) 用於建築結構優化。

### Key Features

1. **動態 Action Space** 🆕
   - 支持不同樓層數的建築 (4層、7層、10層等)
   - 使用 per-story-member scoring 架構
   - Action = 選擇哪個結構構件減小斷面

2. **完整對齊原文理論**
   - Intra-Option Policy Gradient (Theorem 1)
   - Termination Gradient (Theorem 2)
   - Q_U 估計機制 (Page 4)

3. **Graph Neural Network Backbone**
   - 處理圖結構的建築模型
   - 提取 story-level 和 global-level 特徵

## Architecture

### Dynamic Action Size Handling

**Problem**: 不同樓層數 → 不同的 action 數量
- 4層樓: 4 × 4 = 16 actions
- 7層樓: 7 × 4 = 28 actions

**Solution**: Per-Member Scoring
```python
# For each option ω, we have an MLP:
score_i = MLP_ω(story_member_feature_i)  # i = 0, 1, ..., N-1

# Compute action distribution:
π(m|s,ω) = softmax([score_0, score_1, ..., score_{N-1}])

# Sample which story member to modify:
action = sample from π(m|s,ω)
```

### Model Components

```
Input: Graph(nodes, edges, features)
  ↓
StateGNN
  ↓
├─ Story-level features [N, 128] ──→ Intra-option policies (per-member scoring)
└─ Global features [1, 64] ────────→ Q_Ω, β (option values & termination)
```

## Training

```bash
python train_option_critic.py \
    --epochs 1000 \
    --num_options 4 \
    --max_option_len 10 \
    --batch_size 32 \
    --device cuda
```

## Implementation Details

### 與原文對應

| 原文符號 | 實現位置 | 說明 |
|---------|---------|------|
| `Q_Ω(s,ω)` | `model.get_Q(state)` | Option-value function |
| `Q_U(s,ω,a)` | `model.compute_Q_U(...)` | Action-value function (estimated) |
| `β(s,ω)` | `model.get_terminations(state)` | Termination probability |
| `π(a\|s,ω)` | `model.get_action(...)` | Intra-option policy (per-member scoring) |
| `V_Ω(s)` | `max Q_Ω(s,ω)` | State value (greedy policy-over-options) |

### Gradient Updates

1. **Critic**: 學習 `Q_Ω(s,ω)`
   ```python
   δ = r + γ[(1-β)Q_Ω(s',ω) + β max_ω' Q_Ω(s',ω')] - Q_Ω(s,ω)
   ```

2. **Intra-Option Policy**:
   ```python
   ∇_θ J = ∇_θ log π(a|s,ω) * Q_U(s,ω,a)
   ```

3. **Termination**:
   ```python
   ∇_ϑ J = -β(s',ω) * [Q_Ω(s',ω) - V_Ω(s') + ε]
   ```
```

---

## 執行檢查清單

### ✅ 階段 0 檢查點 (動態 Action Size 架構)
- [ ] Task 0.1: 修改 `__init__` 方法 - 移除 options_W/b，添加 intra_option_policies
- [ ] Task 0.2: 修改 `_initialize_parameters` 方法
- [ ] Task 0.3: 重寫 `get_action` 方法 - per-member scoring
- [ ] Task 0.4: 更新 optimizer 設置
- [ ] Task 0.5: 更新模型初始化邏輯
- [ ] **驗證**: 測試 4層、7層、10層樓的 action selection

### ✅ 階段 1 檢查點 (Q_U 機制)
- [ ] Task 1.1: 實現 `compute_Q_U` 方法
- [ ] Task 1.2: 實現 `compute_Q_U_batch` 方法
- [ ] **驗證**: 單元測試 Q_U 計算屬性

### ✅ 階段 2 檢查點 (Policy Gradient)
- [ ] Task 2.1: 修改 `actor_loss` - 使用 Q_U
- [ ] Task 2.2: 更新訓練循環傳遞 action
- [ ] **驗證**: Gradient 計算正確，無 NaN

### ✅ 階段 3 檢查點 (Termination Gradient)
- [ ] Task 3.1: 實現 `compute_V_omega` (可選)
- [ ] **驗證**: Termination 概率合理 (0.1-0.3)

### ✅ 階段 4 檢查點 (動態測試)
- [ ] Task 4.1: 實現動態 action size 測試套件
- [ ] **驗證**: 所有測試通過

### ✅ 階段 5 檢查點 (Critic Loss)
- [ ] Task 5.1: 確認 critic loss 正確
- [ ] **決定**: 無需修改

### ✅ 階段 6 檢查點 (監控)
- [ ] Task 6.1: 添加 Q_U 監控
- [ ] Task 6.2: 添加梯度監控
- [ ] **驗證**: 訓練日誌清晰可讀

### ✅ 階段 7 檢查點 (測試)
- [ ] Task 7.1: 單元測試通過
- [ ] Task 7.2: 整合測試通過
- [ ] Task 7.3: 完整訓練測試 (10 episodes)
- [ ] **驗證**: 無 action size 相關錯誤

### ✅ 階段 8 檢查點 (文檔)
- [ ] Task 8.1: 所有 docstring 更新
- [ ] Task 8.2: README 更新
- [ ] **驗證**: 文檔完整準確

---

## 預期改進效果

1. **✅ 支持動態 Action Size**
   - 任意樓層數的建築都可以訓練和推理
   - 模型參數不隨 action size 變化

2. **✅ 理論正確性**
   - 完全符合 Option-Critic 原文
   - Gradient 計算精確

3. **✅ 訓練穩定性**
   - 正確的 Q_U 估計
   - 合理的 advantage 計算

4. **✅ 通用性**
   - 同一模型處理不同規模的問題
   - 無需針對不同樓層數重新訓練

---

## 潛在風險和緩解措施

### 風險 1: Per-Member Scoring 的表達能力
- **問題**: MLP scoring 是否足夠表達複雜的 action preferences?
- **緩解**:
  - 使用足夠深的 MLP (3層)
  - 添加 Dropout 防止過擬合
  - 監控 action 分佈的 entropy

### 風險 2: 不同 Episode 的 Gradient 方差
- **問題**: 4層樓 vs 10層樓的 gradient scale 可能不同
- **緩解**:
  - 使用 gradient clipping
  - Normalize Q_U 值
  - 監控不同 episode 的 loss 分佈

### 風險 3: Q_U 估計誤差
- **問題**: One-step estimator 可能不準確
- **緩解**:
  - 監控 Q_U 值的範圍和變化
  - 添加 Q_U clipping
  - 考慮使用 n-step returns

---

## 參考文獻

Bacon, P. L., Harb, J., & Precup, D. (2016). The Option-Critic Architecture. arXiv preprint arXiv:1609.05140.

**關鍵公式**:
- **Equation 1**: `Q_Ω(s,ω) = Σ_a π(a|s,ω) * Q_U(s,ω,a)`
- **Equation 2**: `Q_U(s,ω,a) = r(s,a) + γ * U(ω,s')`
- **Equation 3**: `U(ω,s') = (1-β(s'))*Q_Ω(s',ω) + β(s')*V_Ω(s')`
- **Theorem 1**: Intra-Option Policy Gradient
- **Theorem 2**: Termination Gradient

---

## 附錄：架構對比

### 舊架構 (v1) ❌
```python
# Fixed action size
self.options_W = nn.Parameter(torch.zeros(num_options, gnn_output_dim, num_actions))
self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

# Action selection
logits = story_level_state @ self.options_W[option] + self.options_b[option]
# logits shape: [num_story_members, num_actions] ← Fixed size!
```

### 新架構 (v2) ✅
```python
# Dynamic action size
self.intra_option_policies = nn.ModuleList([
    MLP(gnn_output_dim → 1)  # Score per story member
    for _ in range(num_options)
])

# Action selection
scores = self.intra_option_policies[option](story_level_state)
# scores shape: [num_story_members] ← Dynamic size!
```

---

**計畫更新完成 (v2)。請按照階段 0 → 1 → 2 → ... 順序執行。**
