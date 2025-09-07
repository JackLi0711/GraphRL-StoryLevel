# Exploration Strategies in Option-Critic Framework

This document provides a comprehensive overview of all exploration strategies and parameters used in the Option-Critic implementation for structural design optimization.

## Table of Contents
- [1. Option-Critic Core Exploration Parameters](#1-option-critic-core-exploration-parameters)
- [2. Action Selection Strategies](#2-action-selection-strategies)
- [3. Option Selection and Termination](#3-option-selection-and-termination)
- [4. Exploration Schedules and Decay Strategies](#4-exploration-schedules-and-decay-strategies)
- [5. Training vs Inference Mode Differences](#5-training-vs-inference-mode-differences)
- [6. Randomness Control](#6-randomness-control)
- [7. Hyperparameters Summary](#7-hyperparameters-summary)

---

## 1. Option-Critic Core Exploration Parameters

### 1.1 Epsilon-Greedy Parameters (`option_critic_gnn.py`)

| Parameter | Default Value | Description | Usage |
|-----------|---------------|-------------|-------|
| `eps_start` | 1.0 | Starting epsilon value for exploration | Initial high exploration |
| `eps_min` | 0.1 | Minimum epsilon value | Prevents complete exploitation |
| `eps_decay` | 1,000,000 | Epsilon decay rate (steps) | Controls exploration schedule |
| `eps_test` | 0.05 | Epsilon value during testing | Minimal exploration during evaluation |

**Epsilon Calculation Formula:**
```python
eps = eps_min + (eps_start - eps_min) * exp(-step / eps_decay)
```

### 1.2 Temperature Parameter

| Parameter | Default Value | Description | Implementation |
|-----------|---------------|-------------|----------------|
| `temperature` | 1.0 | Softmax temperature for action selection | `(logits / temperature).softmax(dim=-1)` |

**Effect:** Higher temperature → more exploration, Lower temperature → more exploitation

---

## 2. Action Selection Strategies

### 2.1 Training Mode Action Selection (`option_critic_gnn.py:294-300`)

```python
if self.testing:
    # Deterministic selection (argmax)
    action = torch.argmax(action_dist.probs, dim=-1)
else:
    # Stochastic sampling
    action = action_dist.sample()
```

### 2.2 Valid Actions Masking (`train_option_critic.py:280-320`)

**Implementation:**
```python
# Create valid actions mask
already_minimum = set(getattr(structure, 'already_minimum_section_story_indexes', []) or [])
restricted_actions = set()
if hasattr(structure, 'restrict_action_space'):
    restricted = structure.restrict_action_space()
    if restricted is not None:
        restricted_actions.update(restricted)

invalid_actions = already_minimum | restricted_actions
valid_mask = torch.ones(num_actions, dtype=torch.bool, device=device)
for invalid_action in invalid_actions:
    if 0 <= invalid_action < num_actions:
        valid_mask[invalid_action] = False

# Apply mask: invalid actions get very negative logits
logits[~valid_actions_mask] = -1e8
```

---

## 3. Option Selection and Termination

### 3.1 Option Selection Strategy (`train_option_critic.py:934-937`)

**Training:**
```python
if option_termination:
    curr_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
```

**Testing:**
```python
# Always use greedy option selection during evaluation
curr_option = greedy_option
```

### 3.2 Option Termination Strategy (`option_critic_gnn.py:208-246`)

**Training Mode:**
```python
# Probabilistic sampling
option_termination = Bernoulli(termination_prob).sample()
```

**Testing Mode:**
```python
# Threshold-based deterministic decision
option_termination = termination_prob > 0.5
```

### 3.3 Termination Network Initialization (`option_critic_gnn.py:127-129`)
```python
# Initialize with negative bias for low initial termination probabilities (~0.1)
nn.init.xavier_uniform_(self.terminations.weight)
nn.init.constant_(self.terminations.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1
```

---

## 4. Exploration Schedules and Decay Strategies

### 4.1 Available Decay Schedules (`inference.py:126-151`)

#### 4.1.1 Exponential Annealing Schedule
```python
def exponential_annealing_schedule(n, rate=0.02):
    return 1 - np.exp(-rate * n)

beta_annealing_schedule = lambda n: exponential_annealing_schedule(n, 0.02)
```

#### 4.1.2 Power Decay Schedule
```python
def power_decay_schedule(episode_number: int, decay_factor: float, minimum_epsilon: float=1e-2) -> float:
    return max(decay_factor ** episode_number, minimum_epsilon)

epsilon_decay_schedule = lambda n: power_decay_schedule(n, args.epsilon, 1e-2)
```

#### 4.1.3 Linear Decay Schedule
```python
def linear_decay_schedule(episode_number: int, total_episode: int, minimum_epsilon: float=1e-1) -> float:
    return max(1.0 - episode_number/total_episode, minimum_epsilon)

straight_decay_schedule = lambda n: linear_decay_schedule(n, args.num_epoch, 1e-1)
```

#### 4.1.4 Cosine Decay Schedule
```python
def cosine_decay_schedule(episode_number: int, total_episode: int, minimum_epsilon: float=1e-1) -> float:
    linear_decay = 1.0 - episode_number / total_episode
    cosine_decay = 0.75 * linear_decay + 0.25 * linear_decay * np.cos(np.pi / 100 * episode_number)
    return max(cosine_decay, minimum_epsilon)

periodic_decay_schedule = lambda n: cosine_decay_schedule(n, args.num_epoch, 1e-1)
```

#### 4.1.5 Constant Epsilon Schedule
```python
def constant_epsilon_schedule(episode_number: int, constant_epsilon: float=1e-1) -> float:
    return constant_epsilon

fixed_epsilon_schedule = lambda n: constant_epsilon_schedule(n, 1e-1)
```

---

## 5. Training vs Inference Mode Differences

### 5.1 Model Testing Flag Effects

| Aspect | Training Mode (`testing=False`) | Inference Mode (`testing=True`) |
|--------|--------------------------------|----------------------------------|
| **Action Selection** | `Categorical.sample()` (stochastic) | `torch.argmax()` (deterministic) |
| **Option Termination** | `Bernoulli.sample()` (probabilistic) | `termination_prob > 0.5` (threshold) |
| **Option Selection** | Epsilon-greedy | Pure greedy |
| **Epsilon Value** | Dynamic decay | Fixed `eps_test = 0.05` |

### 5.2 Environment Reset Differences

**Training:**
```python
structure = base_env.reset(testing=False)
```

**Inference:**
```python
structure = base_env.reset(testing=True)
```

### 5.3 Evaluation Function Settings (`train_option_critic.py:524-560`)

```python
# Set random seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Store original testing mode and set to testing mode
original_testing = oc_model.testing
oc_model.testing = True

try:
    # ... evaluation loop ...
finally:
    # Restore original testing mode
    oc_model.testing = original_testing
```

---

## 6. Randomness Control

### 6.1 Random Seed Settings

**Training Script (`train_option_critic.py`):**
- No fixed seed (random each run)
- Step-based epsilon updates

**Inference Script (`inference.py`):**
```python
def set_random_seed(SEED: int = 731):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
```

### 6.2 Experience Replay Buffer (`experience_replay.py:23`)
```python
self.rng = random.SystemRandom(seed)  # Default seed = 42
```

---

## 7. Hyperparameters Summary

### 7.1 Option-Critic Training Parameters (`train_option_critic.py:691-727`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--num_options` | 8 | Number of options to learn |
| `--max_option_len` | 16 | Maximum primitive actions per option |
| `--temperature` | 1.0 | Temperature for action selection |
| `--eps_start` | 1.0 | Starting epsilon for exploration |
| `--eps_min` | 0.1 | Minimum epsilon |
| `--eps_decay` | 1,000,000 | Epsilon decay rate |
| `--eps_test` | 0.05 | Epsilon for testing |
| `--gamma` | 0.99 | Discount factor |
| `--termination_reg` | 0.01 | Termination regularization weight |
| `--entropy_reg` | 0.01 | Entropy regularization weight |
| `--option_length_bonus` | 0.1 | Bonus reward for longer options |

### 7.2 DQN Inference Parameters (`inference.py:60-70`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--epsilon` | 0.99 | Epsilon decay factor |
| `--random_seed` | 731 | Fixed random seed |
| `--num_epoch` | 1000 | Total training epochs |

---

## 8. Implementation Insights

### 8.1 Why Inference Performs Better

1. **Elimination of Exploration Noise**: Pure exploitation strategy removes suboptimal exploratory actions
2. **Deterministic Decision Making**: Consistent action and option selection without randomness
3. **Stable Option Termination**: Threshold-based termination reduces unnecessary option switching
4. **Reproducible Results**: Fixed random seeds ensure consistent behavior

### 8.2 Exploration-Exploitation Balance

The framework implements a hierarchical exploration strategy:
- **High-level**: Option selection (epsilon-greedy → greedy)
- **Mid-level**: Option termination (probabilistic → deterministic)  
- **Low-level**: Action selection (sampling → argmax)

### 8.3 Termination Probability Logging (`train_option_critic.py:21-150`)

A comprehensive logging system tracks termination decisions:
```python
termination_logger.log_termination_prediction(
    option=current_option,
    termination_probs=termination_probs,
    termination_decision=option_termination,
    step=length,
    context="rollout_option"
)
```

---

## 9. Recommended Exploration Strategy Tuning

### 9.1 For Better Training Performance
- Reduce `eps_min` from 0.1 to 0.01
- Increase `eps_decay` for slower epsilon reduction
- Adjust `temperature` based on action space complexity

### 9.2 For Smoother Training-to-Inference Transition
- Implement gradual temperature annealing
- Add deterministic evaluation phases during training
- Use curriculum learning with decreasing exploration

### 9.3 For Different Problem Complexities
- **Simple environments**: Higher temperature, faster epsilon decay
- **Complex environments**: Lower temperature, slower epsilon decay
- **Sparse rewards**: Higher exploration bonuses, longer exploration phases

---

*Last Updated: January 2025*  
*Framework: Option-Critic with Graph Neural Networks for Structural Design*