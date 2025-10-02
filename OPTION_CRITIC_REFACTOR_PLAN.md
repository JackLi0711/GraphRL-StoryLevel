# Option-Critic 原文對齊修正計畫

## 執行日期
創建日期: 2025-10-02

## 計畫概述
本計畫旨在將現有的 Option-Critic 實現與原文 (Bacon et al., 2016) 完全對齊，並處理動態 action size 的問題。

---

## 問題分析總結

### 當前實現與原文的主要差異

#### 1. ❌ **Critic Loss 的計算方式錯誤**
- **原文要求**: 學習 `Q_Ω(s,ω)` - option-value function
- **當前問題**:
  - 目前在 `critic_loss` 函數中計算的是批次樣本的 Q 值
  - 但缺少對 `Q_U(s,ω,a)` 的正確估計和使用
  - TD target 的計算看似正確，但沒有明確區分 Q_Ω 和 Q_U

#### 2. ❌ **Intra-Option Policy Gradient 使用錯誤的 Advantage**
- **原文要求 (Theorem 1)**:
  ```
  ∇_θ ρ = Σ μ_Ω(s,ω) Σ_a [∂π(a|s,ω)/∂θ] * Q_U(s,ω,a)
  ```
  其中 `Q_U(s,ω,a)` 是 action-value function

- **當前實現 (option_critic_gnn.py:719)**:
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

- **當前實現**: 完全缺少這個估計機制

#### 4. ⚠️ **Termination Gradient 的 Baseline 不精確**
- **原文要求 (Theorem 2)**:
  ```
  ∇_ϑ ρ = -Σ μ_Ω(s',ω) [∂β(s')/∂ϑ] * A_Ω(s',ω)
  A_Ω(s',ω) = Q_Ω(s',ω) - V_Ω(s')
  ```

- **當前實現 (option_critic_gnn.py:716)**:
  ```python
  termination_loss = option_term_prob * (Q[:, option] - Q_max + termination_reg)
  ```

- **問題**: 用 `Q_max` 近似 `V_Ω(s')`，但 `V_Ω(s')` 應該是 policy-over-options 的期望值

#### 5. ✅ **動態 Action Size 處理機制已存在**
- 當前已實現: `valid_actions_mask` 在 `get_action` 和 `rollout_option` 中
- 但需要確保所有新增的計算都能正確處理這個 mask

---

## 修正計畫 - 詳細步驟

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

    Args:
        global_state: Current global state [1, feature_dim]
        option: Current option index
        action: Executed action index
        reward: Immediate reward r(s,a)
        next_global_state: Next global state [1, feature_dim]
        gamma: Discount factor

    Returns:
        Q_U value (scalar tensor)
    """
    with torch.no_grad():
        # Compute termination probability β(s')
        next_beta = self.get_terminations(next_global_state)  # [1, num_options]
        if next_beta.dim() > 1:
            next_beta_omega = next_beta[0, option]  # Scalar for current option
        else:
            next_beta_omega = next_beta[option]

        # Compute Q_Ω(s',ω) and V_Ω(s')
        next_Q = self.get_Q(next_global_state)  # [1, num_options]
        if next_Q.dim() > 1:
            next_Q_omega = next_Q[0, option]  # Q_Ω(s',ω)
            next_V = next_Q[0].max()  # V_Ω(s') = max_ω Q_Ω(s',ω)
        else:
            next_Q_omega = next_Q[option]
            next_V = next_Q.max()

        # Compute U(ω,s')
        U_omega = (1 - next_beta_omega) * next_Q_omega + next_beta_omega * next_V

        # Compute Q_U(s,ω,a)
        Q_U = reward + gamma * U_omega

    return Q_U
```

**注意事項**:
- 使用 `torch.no_grad()` 因為這是用於計算，不需要梯度
- 處理不同的 tensor 維度 (batch 可能是 [1, ...] 或單一 [...])
- `V_Ω(s')` 用 `max_ω Q_Ω(s',ω)` 近似（符合原文 greedy policy-over-options 假設）

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
- 未來可以優化為並行處理

---

### **階段 2: 修正 Intra-Option Policy Gradient**

#### Task 2.1: 修改 `actor_loss` 函數以使用正確的 Advantage
**檔案**: `RL/option_critic_gnn.py`
**位置**: `actor_loss` 函數 (line 620-729)

**修改步驟**:

1. **保存當前執行的 action**（從 transition 獲取）
   - 目前 `actor_loss` 接收的參數沒有 `action`
   - 需要添加 `action: int` 參數

2. **計算 Q_U(s,ω,a)**:
   ```python
   # 使用新增的 compute_Q_U 方法
   Q_U = model.compute_Q_U(
       global_state=state,
       option=option,
       action=action,  # 新增的參數
       reward=reward,
       next_global_state=next_state_prime,
       gamma=gamma
   )
   ```

3. **修改 Policy Gradient**:
   ```python
   # 原文 Theorem 1: 使用 Q_U(s,ω,a) 而非 advantage
   policy_loss = -logp * Q_U - entropy_reg * entropy
   ```

**完整修改後的函數簽名**:
```python
def actor_loss(obs, option: int, action: int, logp: torch.Tensor, entropy: torch.Tensor,
               reward: float, done: bool, next_obs,
               model: OptionCriticGNN, model_prime: OptionCriticGNN,
               gamma: float = 0.99, termination_reg: float = 0.01,
               entropy_reg: float = 0.01) -> torch.Tensor:
```

**詳細實現**:
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
    2. Termination gradient (Theorem 2): -∂β(s')/∂ϑ * A_Ω(s',ω)

    Args:
        obs: Current observation
        option: Current option
        action: Executed action (NEWLY ADDED)
        logp: Log probability of selected action
        entropy: Entropy of action distribution
        reward: Reward received
        done: Whether episode terminated
        next_obs: Next observation
        model: Current model
        model_prime: Target model
        gamma: Discount factor
        termination_reg: Termination regularization weight (ε in paper)
        entropy_reg: Entropy regularization weight

    Returns:
        Actor loss (combines policy gradient and termination gradient)
    """
    # Handle graph observations
    if isinstance(obs, (tuple, list)) and len(obs) >= 4:
        graph_x, graph_edge_index, graph_edge_attr, story_batch = obs[:4]
        next_graph_x, next_graph_edge_index, next_graph_edge_attr, next_story_batch = next_obs[:4]

        # Use global state for policy gradient and termination
        _, state = model.get_state(graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
        _, next_state = model.get_state(next_graph_x, next_graph_edge_index,
                                       next_graph_edge_attr, next_story_batch, None)
        _, next_state_prime = model_prime.get_state(next_graph_x, next_graph_edge_index,
                                                   next_graph_edge_attr, next_story_batch, None)
    else:
        # For tensor observations
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
        # Compute V_Ω(s') as expectation over policy-over-options
        # For greedy policy: V_Ω(s') = max_ω Q_Ω(s',ω)
        next_V = next_Q[0].max()
    else:
        next_Q_omega = next_Q[option]
        next_V = next_Q.max()

    advantage_omega = next_Q_omega - next_V

    # Termination gradient: β(s',ω) * [A_Ω(s',ω) - ε]
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

    return total_actor_loss
```

---

#### Task 2.2: 更新 `rollout_option` 以記錄 action
**檔案**: `RL/option_critic/rollout.py`
**位置**: step_transitions 記錄部分 (line 239-276)

**確認內容**:
- ✅ 已經在記錄 `"action": action`
- 無需修改

---

#### Task 2.3: 更新訓練循環以傳遞 action
**檔案**: `RL/option_critic/trainer.py`
**位置**: `_perform_updates` 方法 (line 383-425)

**修改內容**:
```python
# 原代碼 (line 389-394)
a_loss = actor_loss(
    tr["obs"], tr["option"], tr["logp"], tr["entropy"], tr["reward"],
    tr["done"], tr["next_obs"], self.oc, self.oc_prime,
    self.args.gamma, self.args.termination_reg, self.args.entropy_reg
)

# 修改為:
a_loss = actor_loss(
    tr["obs"],
    tr["option"],
    tr["action"],  # 新增: 傳遞 action
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
        global_state: Global state features [1, feature_dim]
        policy_over_options: Optional policy distribution over options [num_options]
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

---

#### Task 3.2: 使用改進的 V_Ω 計算更新 `actor_loss`
**檔案**: `RL/option_critic_gnn.py`
**位置**: `actor_loss` 函數中的 termination gradient 部分

**修改內容**:
已在 Task 2.1 中完成，使用:
```python
next_V = model.compute_V_omega(next_state, policy_over_options=None)  # Greedy
advantage_omega = next_Q_omega - next_V
```

---

### **階段 4: 驗證動態 Action Size 相容性**

#### Task 4.1: 檢查 `compute_Q_U` 對動態 action size 的支援
**檔案**: `RL/option_critic_gnn.py`

**驗證點**:
- ✅ `compute_Q_U` 不依賴 action size（只使用 option 和 reward）
- ✅ 計算過程中不涉及 action 維度的張量操作
- ✅ 無需修改

---

#### Task 4.2: 驗證 `get_action` 的 masking 機制
**檔案**: `RL/option_critic_gnn.py`
**位置**: `get_action` 方法 (line 240-420)

**驗證點**:
- ✅ 已有 `valid_actions_mask` 參數
- ✅ 已有 mask 應用邏輯 (line 273-310)
- ✅ 已有 action 驗證和 fallback 機制 (line 375-419)
- ✅ 無需修改

---

#### Task 4.3: 添加動態 action size 的測試案例
**檔案**: 新建 `tests/test_dynamic_action_size.py`

**實現內容**:
```python
"""
Test cases for dynamic action size handling in Option-Critic.
"""
import torch
import pytest
from RL.option_critic_gnn import OptionCriticGNN


def test_compute_Q_U_with_different_action_sizes():
    """
    Test that compute_Q_U works correctly regardless of action size.
    """
    # Create model with initial action size
    model = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,  # Initial action size
        num_options=4,
        device='cpu'
    )

    # Create dummy states
    global_state = torch.randn(1, 64)
    next_global_state = torch.randn(1, 64)

    # Test with different actions (within initial range)
    for action in [0, 50, 99]:
        Q_U = model.compute_Q_U(
            global_state=global_state,
            option=0,
            action=action,
            reward=1.0,
            next_global_state=next_global_state,
            gamma=0.99
        )

        assert Q_U.shape == torch.Size([]), f"Q_U should be scalar, got {Q_U.shape}"
        assert not torch.isnan(Q_U), f"Q_U should not be NaN for action {action}"


def test_get_action_with_varying_masks():
    """
    Test that get_action respects varying valid_actions_mask.
    """
    model = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,
        num_options=4,
        device='cpu'
    )

    # Create dummy story-level state
    story_level_state = torch.randn(5, 128)  # 5 story members
    structure_story_ptr = None

    # Test with different mask sizes (simulating different episodes)
    for num_valid in [50, 75, 100]:
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:num_valid] = True  # First num_valid actions are valid

        action, logp, entropy = model.get_action(
            story_level_state=story_level_state,
            structure_story_ptr=structure_story_ptr,
            option=0,
            valid_actions_mask=mask
        )

        assert 0 <= action < num_valid, \
            f"Action {action} should be within valid range [0, {num_valid})"
        assert mask[action], f"Selected action {action} should be valid according to mask"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**執行測試**:
```bash
cd /mnt/d/kyle_MD_project/optionRL/GraphRL_story_level
python -m pytest tests/test_dynamic_action_size.py -v
```

---

### **階段 5: Critic Loss 的改進（可選）**

#### Task 5.1: 確認當前 Critic Loss 是否正確
**檔案**: `RL/option_critic_gnn.py`
**位置**: `critic_loss` 函數 (line 454-617)

**分析**:
當前實現:
```python
gt = rewards + masks * gamma * (
    (1 - next_options_term_prob) * next_Q_option +
    next_options_term_prob * next_Q_max
)
td_err = (Q_option - gt.detach()).pow(2).mul(0.5).mean()
```

**對照原文 (Algorithm 1, Page 4)**:
```
δ ← r - Q_U(s,ω,a)
if s' is non-terminal:
    δ ← δ + γ(1-β(s'))*Q_Ω(s',ω) + γβ(s')*max_ω Q_Ω(s',ω')
Q_U(s,ω,a) ← Q_U(s,ω,a) + α*δ
```

**結論**:
- ✅ 當前實現基本正確
- ⚠️ 但是在學習 `Q_Ω` 而非 `Q_U`
- 根據原文 Page 4 說明，實踐中可以只學習 `Q_Ω`，這是可接受的簡化

**決定**: 保持當前實現，無需修改

---

### **階段 6: 添加詳細的 Logging 和 Monitoring**

#### Task 6.1: 添加 Q_U 值的監控
**檔案**: `RL/option_critic/trainer.py`
**位置**: `_perform_updates` 方法

**新增內容**:
```python
# 在 actor update 之前添加
if self.args.debug_logging and len(accumulated_actor_losses) > 0:
    # Sample a few transitions to monitor Q_U values
    sample_transitions = step_transitions[:min(3, len(step_transitions))]
    for idx, tr in enumerate(sample_transitions):
        _, state = self.oc.get_state(*tr["obs"])
        _, next_state = self.oc_prime.get_state(*tr["next_obs"])

        Q_U = self.oc.compute_Q_U(
            global_state=state,
            option=tr["option"],
            action=tr["action"],
            reward=tr["reward"],
            next_global_state=next_state,
            gamma=self.args.gamma
        )

        self.logger.debug(
            f"Sample {idx}: Q_U(s,ω={tr['option']},a={tr['action']}) = {Q_U.item():.4f}, "
            f"reward = {tr['reward']:.4f}"
        )
```

---

#### Task 6.2: 監控 Policy Gradient 和 Termination Gradient 的大小
**檔案**: `RL/option_critic_gnn.py`
**位置**: `actor_loss` 函數末尾

**新增內容**:
```python
# 在 return 之前添加
if hasattr(model, 'debug_logging') and model.debug_logging:
    print(f"[ActorLoss] Policy loss: {policy_loss.item():.6f}, "
          f"Termination loss: {termination_loss.item():.6f}, "
          f"Total: {total_actor_loss.item():.6f}")
    print(f"[ActorLoss] Q_U: {Q_U.item():.4f}, "
          f"Advantage_Ω: {advantage_omega.item():.4f}, "
          f"β(s',ω): {next_beta_omega.item():.4f}")
```

---

### **階段 7: 測試和驗證**

#### Task 7.1: 單元測試 - compute_Q_U
**檔案**: `tests/test_option_critic_gnn.py`

**新增測試**:
```python
def test_compute_Q_U_properties():
    """
    Test mathematical properties of Q_U computation.
    """
    model = create_test_model()

    # Create test states
    state = torch.randn(1, 64)
    next_state = torch.randn(1, 64)

    # Test 1: Q_U should increase with reward
    Q_U_low = model.compute_Q_U(state, 0, 0, 0.0, next_state, 0.99)
    Q_U_high = model.compute_Q_U(state, 0, 0, 10.0, next_state, 0.99)
    assert Q_U_high > Q_U_low, "Q_U should increase with reward"

    # Test 2: Q_U should depend on gamma
    Q_U_gamma_1 = model.compute_Q_U(state, 0, 0, 1.0, next_state, 0.99)
    Q_U_gamma_2 = model.compute_Q_U(state, 0, 0, 1.0, next_state, 0.5)
    # Should be different (unless next value is zero)

    # Test 3: Q_U should be finite
    assert torch.isfinite(Q_U_low), "Q_U should be finite"
    assert torch.isfinite(Q_U_high), "Q_U should be finite"
```

---

#### Task 7.2: 整合測試 - 完整訓練流程
**檔案**: `tests/test_training_pipeline.py`

**新增測試**:
```python
def test_actor_loss_backward():
    """
    Test that actor_loss computes gradients correctly.
    """
    model = create_test_model()
    model_prime = create_test_model()
    model_prime.load_state_dict(model.state_dict())

    # Create dummy observation
    obs = create_dummy_graph_obs()
    next_obs = create_dummy_graph_obs()

    # Create transition
    option = 0
    action = 5
    logp = torch.tensor(-1.5, requires_grad=True)
    entropy = torch.tensor(2.0)
    reward = 1.0
    done = False

    # Compute loss
    loss = actor_loss(
        obs, option, action, logp, entropy, reward, done, next_obs,
        model, model_prime, gamma=0.99
    )

    # Check loss is scalar
    assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"

    # Check backward pass
    loss.backward()

    # Check gradients exist for policy parameters
    assert model.options_W.grad is not None, "Policy parameters should have gradients"
    assert model.terminations.weight.grad is not None, "Termination parameters should have gradients"
```

---

#### Task 7.3: 執行完整訓練驗證
**檔案**: 使用現有的 `train_option_critic.py`

**測試步驟**:
```bash
# 1. 短期訓練測試（10 episodes）
python train_option_critic.py \
    --epochs 10 \
    --num_options 2 \
    --eval_frequency 5 \
    --batch_size 16 \
    --device cpu

# 2. 檢查輸出日誌
# - 確認 Q_U 值被正確計算
# - 確認 actor_loss 包含 policy gradient 和 termination gradient
# - 確認沒有 NaN 或 Inf

# 3. 檢查 checkpoint 目錄
# - 確認 best_model.pt 被保存
# - 確認 oc_stats.json 包含完整統計
```

---

### **階段 8: 文檔更新**

#### Task 8.1: 更新 docstring
**檔案**: `RL/option_critic_gnn.py`

**更新位置**:
1. 類 `OptionCriticGNN` 的 docstring - 說明新增的方法
2. `actor_loss` 函數 - 更新參數說明和算法描述
3. 新增方法的完整 docstring

---

#### Task 8.2: 更新 README
**檔案**: `GraphRL_story_level/README.md` (如果存在)

**新增內容**:
```markdown
## Option-Critic 實現細節

本實現遵循 Bacon et al. (2016) 的 Option-Critic Architecture 原文。

### 關鍵特性

1. **Q_U 估計**: 使用 one-step estimator 從 Q_Ω 推導 Q_U
2. **Intra-Option Policy Gradient**: 直接使用 Q_U(s,ω,a) 作為梯度信號
3. **Termination Gradient**: 使用 advantage A_Ω(s,ω) = Q_Ω(s,ω) - V_Ω(s)
4. **動態 Action Size**: 支持每個 episode 不同的有效 action 集合

### 與原文的對應關係

- **Theorem 1** (Intra-Option Policy Gradient): 實現於 `actor_loss` 的 `policy_loss` 部分
- **Theorem 2** (Termination Gradient): 實現於 `actor_loss` 的 `termination_loss` 部分
- **Algorithm 1**: 實現於 `trainer.py` 的訓練循環

### 動態 Action Size 處理

每個訓練步驟:
1. `rollout_option` 根據當前 structure 生成 `valid_actions_mask`
2. `get_action` 使用 mask 過濾無效 actions
3. `compute_Q_U` 不依賴 action size，保證相容性
```

---

## 執行檢查清單

### 階段 1 檢查點
- [ ] `compute_Q_U` 方法實現完成
- [ ] `compute_Q_U_batch` 方法實現完成
- [ ] 單元測試通過

### 階段 2 檢查點
- [ ] `actor_loss` 函數簽名更新
- [ ] `actor_loss` 使用 Q_U 計算 policy gradient
- [ ] `rollout_option` 確認記錄 action
- [ ] `trainer._perform_updates` 傳遞 action 參數
- [ ] 整合測試通過

### 階段 3 檢查點
- [ ] `compute_V_omega` 方法實現完成
- [ ] `actor_loss` 使用改進的 V_Ω 計算
- [ ] Termination gradient 測試通過

### 階段 4 檢查點
- [ ] 動態 action size 測試案例實現
- [ ] 所有動態測試通過
- [ ] 驗證不同 episode 的 action size 變化

### 階段 5 檢查點
- [ ] Critic loss 分析完成
- [ ] 確認無需修改或完成修改

### 階段 6 檢查點
- [ ] Q_U 監控日誌添加
- [ ] Policy/Termination gradient 監控添加
- [ ] 訓練過程日誌清晰可讀

### 階段 7 檢查點
- [ ] 所有單元測試通過
- [ ] 整合測試通過
- [ ] 完整訓練測試成功（至少 10 episodes 無錯誤）

### 階段 8 檢查點
- [ ] 所有 docstring 更新完成
- [ ] README 更新完成
- [ ] 代碼注釋完整清晰

---

## 預期改進效果

1. **理論正確性**: 完全符合 Option-Critic 原文的數學推導
2. **訓練穩定性**: 正確的 advantage 計算應該減少梯度方差
3. **學習效率**: Q_U 提供更準確的 action-value 估計
4. **通用性**: 支持動態 action size，適用於不同樓層結構

---

## 潛在風險和緩解措施

### 風險 1: Q_U 估計不準確
- **症狀**: Q_U 值波動劇烈或出現 NaN
- **緩解**: 添加 gradient clipping，監控 Q_U 值範圍

### 風險 2: 動態 action size 導致訓練不穩定
- **症狀**: 不同 episode 的 loss 差異過大
- **緩解**: Normalize Q_U 值，使用 adaptive learning rate

### 風險 3: 修改後訓練速度變慢
- **症狀**: 每個 episode 訓練時間增加
- **緩解**: Profile 代碼，優化 compute_Q_U 的批次處理

---

## 參考文獻

Bacon, P. L., Harb, J., & Precup, D. (2016). The Option-Critic Architecture. arXiv preprint arXiv:1609.05140.

原文關鍵公式:
- **Equation 1**: `Q_Ω(s,ω) = Σ_a π(a|s,ω) * Q_U(s,ω,a)`
- **Equation 2**: `Q_U(s,ω,a) = r(s,a) + γ * Σ_s' P(s'|s,a) * U(ω,s')`
- **Equation 3**: `U(ω,s') = (1-β(s'))*Q_Ω(s',ω) + β(s')*V_Ω(s')`
- **Theorem 1**: Intra-Option Policy Gradient
- **Theorem 2**: Termination Gradient

---

## 附錄: 原文與實現對照表

| 原文符號 | 實現位置 | 說明 |
|---------|---------|------|
| `Q_Ω(s,ω)` | `model.get_Q(state)` | Option-value function |
| `Q_U(s,ω,a)` | `model.compute_Q_U(...)` | Action-value function |
| `U(ω,s')` | `compute_Q_U` 內部計算 | Value upon arrival |
| `β(s,ω)` | `model.get_terminations(state)` | Termination function |
| `π(a\|s,ω)` | `model.get_action(...)` | Intra-option policy |
| `V_Ω(s)` | `model.compute_V_omega(state)` | State value under policy-over-options |
| `A_Ω(s,ω)` | `Q_omega - V_omega` | Advantage function |

---

**計畫創建完成。請按照階段順序執行，每完成一個階段檢查點後再進行下一階段。**
