# Actor-Critic Implementation Task List

## 📋 總覽

本文件列出所有實作任務，按照依賴關係排序。每個任務包含：
- 檔案路徑
- 實作內容
- 測試方式
- 預計時間

---

## ✅ Task 1: 建立 Policy Gradient 模組資料夾

**檔案**: `RL/pg/`

**內容**:
```bash
mkdir -p RL/pg
```

**測試**: 確認資料夾存在

**時間**: 1 分鐘

---

## ✅ Task 2: 實作 PolicyNetwork

**檔案**: `RL/pg/networks.py`

**內容**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Story-level policy network (Actor)

    輸入: story_features [N, state_dim]
        - N: 當前建築的 story member 數量 (動態: 16, 20, 24, 28...)
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
        """
        Args:
            story_features: [N, state_dim]

        Returns:
            action_scores: [N]
        """
        scores = self.policy_head(story_features)  # [N, 1]
        return scores.squeeze(-1)  # [N]


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
        """
        Args:
            global_features: [1, state_dim] or [batch, state_dim]

        Returns:
            value: [1] or [batch]
        """
        return self.value_head(global_features)  # [1, 1] or [batch, 1]
```

**測試**:
```python
# Test script
state_dim = 200  # hidden_dim * 2
hidden_dim = 100

# Test PolicyNetwork with different action sizes
policy = PolicyNetwork(state_dim, hidden_dim)
for N in [16, 20, 24, 28]:
    story_features = torch.randn(N, state_dim)
    scores = policy(story_features)
    assert scores.shape == (N,), f"Expected shape ({N},), got {scores.shape}"
    print(f"✓ PolicyNetwork works for {N} actions")

# Test ValueNetwork
value_net = ValueNetwork(state_dim, hidden_dim)
global_features = torch.randn(1, state_dim)
value = value_net(global_features)
assert value.shape == (1, 1), f"Expected shape (1, 1), got {value.shape}"
print(f"✓ ValueNetwork works")
```

**時間**: 20 分鐘

---

## ✅ Task 3: 實作 ExperienceBuffer

**檔案**: `RL/pg/buffer.py`

**內容**:
```python
import torch
from typing import List, Dict

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
        """
        添加一個 step 到當前 episode

        Args:
            state: [N, state_dim] tensor
            global_state: [1, state_dim] tensor
            action: int
            reward: float
            log_prob: tensor
            value: tensor
            entropy: tensor
            valid_mask: [N] bool tensor
        """
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

**測試**:
```python
# Test script
buffer = ExperienceBuffer()

# Simulate 2 episodes
for ep in range(2):
    # Episode with 5 steps
    for step in range(5):
        state = torch.randn(16, 200)
        global_state = torch.randn(1, 200)
        action = step
        reward = 1.0
        log_prob = torch.tensor(0.1)
        value = torch.tensor([[10.0]])
        entropy = torch.tensor(0.5)
        valid_mask = torch.ones(16, dtype=torch.bool)

        buffer.add_step(state, global_state, action, reward, log_prob, value, entropy, valid_mask)

    buffer.finish_episode()

assert len(buffer) == 2, f"Expected 2 episodes, got {len(buffer)}"
episodes = buffer.get_all_episodes()
assert len(episodes[0]['rewards']) == 5, "Expected 5 steps per episode"
print("✓ ExperienceBuffer works")
```

**時間**: 15 分鐘

---

## ✅ Task 4: 實作 BasePGAgent

**檔案**: `RL/pg/base_agent.py`

**內容**: 完整的 BasePGAgent 類別（參考 PLAN.md）

**關鍵點**:
1. 引入 StateGNN from RL.model
2. 實作 get_features() with torch.no_grad()
3. 實作 choose_action() with masking
4. 實作 compute_returns() - Monte Carlo
5. 實作 compute_advantages() - normalized
6. 實作 get_entropy_coef() - exponential annealing

**測試**:
```python
# Test with dummy structure
# (需要實際的 structure object)
```

**時間**: 40 分鐘

---

## ✅ Task 5: 實作 A2CAgent

**檔案**: `RL/pg/a2c_agent.py`

**內容**: 完整的 A2CAgent 類別（參考 PLAN.md）

**關鍵點**:
1. 繼承 BasePGAgent
2. 初始化 optimizer (StateGNN + Policy + Value)
3. 實作 update() 方法
   - 處理 accumulate_episodes 個 episodes
   - 計算 policy loss, value loss, entropy loss
   - Gradient clipping
   - Optimizer step

**測試**: 需要完整環境，留到整合測試

**時間**: 30 分鐘

---

## ✅ Task 6: 實作 PPOAgent

**檔案**: `RL/pg/ppo_agent.py`

**內容**: 完整的 PPOAgent 類別（參考 PLAN.md）

**關鍵點**:
1. 繼承 BasePGAgent
2. 實作 update() 方法
   - PPO clipped loss
   - 多個 epochs
   - 重新計算 log_probs with grad

**測試**: 需要完整環境，留到整合測試

**時間**: 40 分鐘

---

## ✅ Task 7: 實作 __init__.py

**檔案**: `RL/pg/__init__.py`

**內容**:
```python
"""
Policy Gradient methods for structural design optimization
"""

from .networks import PolicyNetwork, ValueNetwork
from .buffer import ExperienceBuffer
from .base_agent import BasePGAgent
from .a2c_agent import A2CAgent
from .ppo_agent import PPOAgent

__all__ = [
    'PolicyNetwork',
    'ValueNetwork',
    'ExperienceBuffer',
    'BasePGAgent',
    'A2CAgent',
    'PPOAgent'
]
```

**測試**:
```python
from RL.pg import A2CAgent, PPOAgent
print("✓ Import successful")
```

**時間**: 5 分鐘

---

## ✅ Task 8: 實作訓練輔助函數

**檔案**: `RL/pg/train_utils.py` (新建)

**內容**:
```python
"""
Training utility functions for A2C and PPO
"""

import torch
import numpy as np
from copy import deepcopy
from pathlib import Path

def train_episode(agent, env, rec, logger):
    """
    訓練一個 episode

    Args:
        agent: A2CAgent or PPOAgent
        env: Environment
        rec: Record
        logger: Logger

    Returns:
        score: float
        loss_dict: dict
    """
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

        if logger:
            logger.info(f"Episode {agent._number_episodes+1}, Step {len(agent.buffer.rewards)}, "
                       f"Action: {action}, Reward: {reward:.4f}, Score: {score:.4f}")

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


def test_episode(agent, env, rec, logger):
    """
    測試一個 episode (greedy policy)

    Args:
        agent: A2CAgent or PPOAgent
        env: Environment
        rec: Record
        logger: Logger

    Returns:
        score: float
        design_process: list of dicts
    """
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


def plot_training_testing_curves(rec, ckpt_dir, test_frequency):
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
    if "score_mean" in rec.testing_record and len(rec.testing_record["score_mean"]) > 0:
        test_episodes = range(test_frequency,
                             len(rec.training_record["score"]) + 1,
                             test_frequency)
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


def save_model(agent, ckpt_dir, episode):
    """保存模型"""
    save_path = ckpt_dir / "models" / f"model_episode_{episode+1}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'state_gnn': agent.state_gnn.state_dict(),
        'policy_net': agent.policy_net.state_dict(),
        'value_net': agent.value_net.state_dict(),
        'episode': episode,
        'entropy_coef': agent.get_entropy_coef()
    }, save_path)

    if agent.logger:
        agent.logger.info(f"Model saved to {save_path}")
```

**時間**: 30 分鐘

---

## ✅ Task 9: 實作 train_a2c.py

**檔案**: `train_a2c.py`

**內容**: 完整的訓練腳本

**關鍵點**:
1. Argument parsing
2. Setup environment
3. Setup A2C agent
4. Training loop:
   - Train episode
   - Every test_frequency: test_runs 次 testing
   - 計算 mean ± std
   - 繪圖
   - 保存模型
5. 參考 train.py 的結構

**測試**: 實際運行訓練

**時間**: 60 分鐘

---

## ✅ Task 10: 實作 train_ppo.py

**檔案**: `train_ppo.py`

**內容**: 類似 train_a2c.py，但使用 PPOAgent

**差異**:
1. Import PPOAgent
2. 新增 PPO 專屬參數：
   - `--clip_epsilon`
   - `--ppo_epochs`
3. 預設 `--accumulate_episodes=5`
4. 預設 `--lr=3e-4`

**時間**: 30 分鐘（複製 train_a2c.py 並修改）

---

## ✅ Task 11: 更新 Record 類別

**檔案**: `RL/record.py`

**修改內容**:
在 `Record` 類別中新增測試統計欄位：

```python
class Record:
    def __init__(self):
        # ... existing ...

        # 新增：測試統計
        self.testing_record["score_mean"] = []  # 每次測試的平均分數
        self.testing_record["score_std"] = []   # 每次測試的標準差
        self.testing_record["best_designs"] = []  # 每次測試的最佳設計
```

**時間**: 10 分鐘

---

## ✅ Task 12: 整合測試 - A2C

**目標**: 驗證 A2C 可以正常訓練

**測試步驟**:
1. 準備測試環境（小規模）
```bash
python train_a2c.py \
    --num_epoch 10 \
    --test_frequency 5 \
    --test_runs 3 \
    --structure_shape fixed \
    --suffix test_a2c
```

2. 檢查項目：
   - ✅ 沒有報錯
   - ✅ Training score 有記錄
   - ✅ Testing 有執行
   - ✅ 圖表有生成
   - ✅ 模型有保存

**時間**: 30 分鐘

---

## ✅ Task 13: 整合測試 - PPO

**目標**: 驗證 PPO 可以正常訓練

**測試步驟**:
```bash
python train_ppo.py \
    --num_epoch 10 \
    --test_frequency 5 \
    --test_runs 3 \
    --accumulate_episodes 5 \
    --structure_shape fixed \
    --suffix test_ppo
```

**時間**: 30 分鐘

---

## ✅ Task 14: 動態 Action Size 測試

**目標**: 驗證可以處理不同樓層數的建築

**測試步驟**:
```bash
# 測試 random structure (4-7 層)
python train_a2c.py \
    --num_epoch 20 \
    --structure_shape random \
    --suffix test_dynamic_actions
```

**檢查**: Training log 中出現不同的 action 數量（16, 20, 24, 28）

**時間**: 20 分鐘

---

## ✅ Task 15: Entropy Annealing 驗證

**目標**: 驗證 entropy coefficient 正確衰減

**測試**:
在 training log 中檢查：
```
Episode 1: Entropy Coef: 0.010000
Episode 10: Entropy Coef: 0.009044
Episode 100: Entropy Coef: 0.003660
Episode 500: Entropy Coef: 0.000063
```

**時間**: 10 分鐘

---

## ✅ Task 16: 完整訓練測試

**目標**: 運行完整的訓練

**A2C**:
```bash
python train_a2c.py \
    --num_epoch 1000 \
    --test_frequency 5 \
    --test_runs 10 \
    --structure_shape random \
    --suffix full_training_a2c
```

**PPO**:
```bash
python train_ppo.py \
    --num_epoch 1000 \
    --test_frequency 5 \
    --test_runs 10 \
    --accumulate_episodes 5 \
    --structure_shape random \
    --suffix full_training_ppo
```

**時間**: 視訓練時間而定（數小時到數天）

---

## ✅ Task 17: 文檔完善

**內容**:
1. README 更新（如何使用 A2C/PPO）
2. 參數說明
3. 結果分析

**時間**: 30 分鐘

---

## 🔍 待確認問題清單

在實作過程中，如遇到以下問題請停下來討論：

### 1. ❓ Record 類別的結構
**問題**: 現有的 `Record` 類別是否支援我們需要的所有記錄？

**需要確認**:
- `rec.training_record` 的結構
- `rec.testing_record` 的結構
- 是否需要新增欄位

**建議**: 先查看 `RL/record.py` 確認

---

### 2. ❓ Visualization 函數
**問題**: 如何繪製 design process？

**需要確認**:
- 現有的 `visualize.visualize_design_process()` 是否可用？
- 輸入格式是什麼？

**建議**: 查看 `Visualization/visualize.py`

---

### 3. ❓ Environment 的 testing structure
**問題**: `env.reset(testing=True)` 是否總是返回相同的結構？

**需要確認**:
- 測試結構是否固定？
- 還是需要傳入 `initial_design` 參數？

**建議**: 查看 `environment.py` 的 `_init_testing_structure()`

---

### 4. ❓ Import 路徑
**問題**: 確認所有 import 是否正確

**需要確認**:
```python
from RL.model import StateGNN  # 是否正確？
from RL.environment import Environment  # 是否正確？
from RL.record import Record  # 是否正確？
```

---

### 5. ❓ Device 處理
**問題**: Tensor 的 device 轉換是否正確？

**需要確認**:
- `structure.graph` 是否需要 `.to(device)`？
- `structure.aux["story_batch"]` 是否需要 `.to(device)`？

---

### 6. ❓ StateGNN forward 的參數
**問題**: StateGNN 的 forward 簽名是否正確？

**需要確認**:
```python
story_features = self.state_gnn(
    graph.x,
    graph.edge_index,
    graph.edge_attr,
    None,  # batch - 是否可以是 None？
    structure.aux["story_batch"],
    None   # structure_story_ptr - 是否可以是 None？
)
```

---

## 📊 預計總時間

| 階段 | 時間 |
|------|------|
| Task 1-3: 基礎組件 | 35 分鐘 |
| Task 4-6: Agent 實作 | 110 分鐘 |
| Task 7-8: 輔助代碼 | 35 分鐘 |
| Task 9-10: 訓練腳本 | 90 分鐘 |
| Task 11: Record 更新 | 10 分鐘 |
| Task 12-15: 測試 | 90 分鐘 |
| Task 16: 完整訓練 | 視情況 |
| Task 17: 文檔 | 30 分鐘 |
| **總計** | **約 6.5 小時（不含完整訓練）** |

---

## ✅ 實作順序建議

### Phase 1: 核心組件（可並行）
1. Task 1-3: 基礎類別
2. Task 2 的測試

### Phase 2: Agent（順序執行）
3. Task 4: BasePGAgent
4. Task 5: A2CAgent
5. Task 6: PPOAgent
6. Task 7: __init__.py

### Phase 3: 訓練邏輯
7. Task 8: 訓練輔助函數
8. Task 9: train_a2c.py
9. Task 11: 更新 Record

### Phase 4: 測試
10. Task 12: A2C 整合測試
11. 如果 Task 12 有問題，停下來解決
12. Task 10: train_ppo.py
13. Task 13: PPO 整合測試
14. Task 14-15: 特定功能測試

### Phase 5: 完整運行
15. Task 16: 完整訓練
16. Task 17: 文檔

---

## 🚀 開始實作

**當前狀態**: 計畫已完成，等待確認

**下一步**: 請確認：
1. ✅ 是否有任何待確認問題需要先討論？
2. ✅ 是否同意實作順序？
3. ✅ 是否可以開始 Phase 1？

**開始命令**: 請回覆 "開始實作" 或指出需要修改的地方！
