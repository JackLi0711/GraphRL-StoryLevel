# OptionCriticGNN Integration Guide

This guide shows how to update `train_option_critic.py` to use the new `OptionCriticGNN` network that includes StateGNN and is fully trainable.

## Files Created

1. **`RL/option_critic_gnn.py`** - Main OptionCriticGNN class
2. **`RL/experience_replay.py`** - ReplayBuffer for graph data
3. **`RL/utils.py`** - Utility functions
4. **`test_option_critic_gnn.py`** - Test script (requires PyTorch)
5. **`syntax_check.py`** - Syntax validation script

## Key Changes Needed in train_option_critic.py

### 1. Import the New Components

Replace the existing imports with:

```python
# Replace these lines:
from RL.oc_policy_adapter import GraphStateAdapter

# With these:
from RL.option_critic_gnn import OptionCriticGNN, critic_loss, actor_loss
from RL.experience_replay import ReplayBuffer
```

### 2. Remove the Adapter and Update get_obs Function

Replace the current `get_obs` function:

```python
# OLD: Using adapter
@torch.no_grad()
def get_obs(structure, adapter, device):
    graph = structure.graph
    story_batch = structure.aux["story_batch"].to(device)
    z = adapter(
        graph_x=graph.x,
        graph_edge_index=graph.edge_index,
        graph_edge_attr=graph.edge_attr,
        story_batch=story_batch,
        structure_story_ptr=None,
    )
    return z

# NEW: Direct graph data
def get_graph_data(structure, device):
    graph = structure.graph
    story_batch = structure.aux["story_batch"].to(device)
    return (
        graph.x.to(device),
        graph.edge_index.to(device), 
        graph.edge_attr.to(device),
        story_batch,
        None  # structure_story_ptr
    )
```

### 3. Replace OptionCriticFeatures with OptionCriticGNN

```python
# OLD: Using external OptionCriticFeatures
oc = OptionCriticFeatures(
    in_features=in_features,
    num_actions=A,
    num_options=args.num_options,
    temperature=args.temperature,
    eps_start=args.eps_start,
    eps_min=args.eps_min,
    eps_decay=args.eps_decay,
    eps_test=args.eps_test,
    device=device,
    testing=False,
)

# NEW: Using OptionCriticGNN
oc = OptionCriticGNN(
    node_feature_dim=node_feature_dim,
    edge_feature_dim=edge_feature_dim,
    hidden_dim=hidden_dim,
    member_state_dim=hidden_dim,  # Same as hidden_dim
    num_layers=num_layers,
    num_actions=A,
    num_options=args.num_options,
    temperature=args.temperature,
    eps_start=args.eps_start,
    eps_min=args.eps_min,
    eps_decay=args.eps_decay,
    eps_test=args.eps_test,
    device=device,
    testing=False,
)
```

### 4. Update rollout_option Function

```python
# Update the rollout_option function to use graph data directly:

def rollout_option(structure, base_env, device, max_option_len, current_option: int, oc_model, epsilon: float = None):
    # ... existing code ...
    
    # OLD: Get aggregated observation
    obs_z = get_obs(structure, adapter, device)
    state = oc_model.get_state(obs_z)
    
    # NEW: Get graph data and state
    graph_data = get_graph_data(structure, device)
    state = oc_model.get_state(*graph_data)
    
    # ... rest of the function remains similar, but use graph_data instead of obs_z
```

### 5. Update Optimizer Configuration

```python
# OLD: Separate optimizers with adapter parameters excluded
actor_params = [oc.options_W, oc.options_b]
critic_params = [p for n, p in oc.named_parameters() if (n.startswith('Q') or n.startswith('terminations'))]
shared_params = [p for n, p in oc.named_parameters() if n.startswith('features')]

# NEW: Include all OptionCriticGNN parameters (including StateGNN)
all_params = list(oc.parameters())  # This now includes StateGNN parameters!

actor_params = [oc.options_W, oc.options_b]
critic_params = [p for n, p in oc.named_parameters() 
                if (n.startswith('Q') or n.startswith('terminations'))]
state_gnn_params = [p for n, p in oc.named_parameters() 
                   if n.startswith('state_gnn')]
feature_params = [p for n, p in oc.named_parameters() 
                 if n.startswith('feature_processor')]

actor_optimizer = optim.Adam([
    {"params": actor_params, "lr": args.actor_lr},
    {"params": state_gnn_params, "lr": args.actor_lr * 0.1},  # Lower LR for GNN
    {"params": feature_params, "lr": args.actor_lr},
])

critic_optimizer = optim.Adam([
    {"params": critic_params, "lr": args.critic_lr},
    {"params": state_gnn_params, "lr": args.critic_lr * 0.1},  # Lower LR for GNN  
    {"params": feature_params, "lr": args.critic_lr},
])
```

### 6. Update Loss Function Calls

```python
# OLD: Using imported loss functions
a_loss = actor_loss_fn(
    tr["obs_z"], curr_option, tr["logp"], tr["entropy"], 
    tr["reward"], tr["done"], tr["next_obs_z"], oc, oc_prime, args
)

c_loss = critic_loss_fn(oc, oc_prime, data_batch, args)

# NEW: Using our loss functions with proper arguments
a_loss = actor_loss(
    graph_data, curr_option, tr["logp"], tr["entropy"],
    tr["reward"], tr["done"], next_graph_data, 
    oc, oc_prime, args.gamma, args.termination_reg, args.entropy_reg
)

c_loss = critic_loss(oc, oc_prime, data_batch, args.gamma)
```

### 7. Replace ReplayBuffer

```python
# OLD: Using external ReplayBuffer
buffer = ReplayBuffer(capacity=100000)

# NEW: Using our ReplayBuffer (same interface)
buffer = ReplayBuffer(capacity=100000)
```

## Benefits of the New Implementation

1. **🎯 StateGNN is now trainable**: No more `@torch.no_grad()` decorator
2. **🔄 End-to-end training**: Graph neural network learns from RL rewards
3. **📦 Self-contained**: No dependency on external option-critic-pytorch
4. **🎛️ Flexible**: Easy to adjust GNN architecture and parameters
5. **🔧 Maintainable**: All code is in your project directory

## Testing

1. **Syntax Check**: Run `python3 syntax_check.py` ✅ (completed)
2. **Functionality Test**: Run `python3 test_option_critic_gnn.py` (requires PyTorch)
3. **Integration Test**: Update `train_option_critic.py` and run training

## Notes

- The StateGNN will now learn meaningful representations for the structural design task
- Consider using different learning rates for different components (GNN vs Option-Critic)
- Monitor training carefully as end-to-end training can be more challenging
- You may want to pre-train the StateGNN or use a lower learning rate initially