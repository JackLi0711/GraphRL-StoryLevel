"""
Test script for PolicyNetwork and ValueNetwork
"""
import sys
sys.path.append('GraphRL_story_level')

import torch
from RL.pg.networks import PolicyNetwork, ValueNetwork

# Test parameters
state_dim = 200  # hidden_dim * 2
hidden_dim = 100

print("=" * 60)
print("Testing PolicyNetwork with different action sizes")
print("=" * 60)

# Test PolicyNetwork with different action sizes
policy = PolicyNetwork(state_dim, hidden_dim)
for N in [16, 20, 24, 28]:
    story_features = torch.randn(N, state_dim)
    scores = policy(story_features)
    assert scores.shape == (N,), f"Expected shape ({N},), got {scores.shape}"
    print(f"✓ PolicyNetwork works for {N} actions - output shape: {scores.shape}")

print()
print("=" * 60)
print("Testing ValueNetwork")
print("=" * 60)

# Test ValueNetwork
value_net = ValueNetwork(state_dim, hidden_dim)
global_features = torch.randn(1, state_dim)
value = value_net(global_features)
assert value.shape == (1, 1), f"Expected shape (1, 1), got {value.shape}"
print(f"✓ ValueNetwork works - output shape: {value.shape}")

# Test with batch
batch_global_features = torch.randn(5, state_dim)
batch_values = value_net(batch_global_features)
assert batch_values.shape == (5, 1), f"Expected shape (5, 1), got {batch_values.shape}"
print(f"✓ ValueNetwork works for batch - output shape: {batch_values.shape}")

print()
print("=" * 60)
print("All tests passed! ✅")
print("=" * 60)
