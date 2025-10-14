"""
Test script for ExperienceBuffer
"""
import sys
sys.path.append('GraphRL_story_level')

import torch
from RL.pg.buffer import ExperienceBuffer

print("=" * 60)
print("Testing ExperienceBuffer")
print("=" * 60)

buffer = ExperienceBuffer()

# Simulate 2 episodes
for ep in range(2):
    print(f"\nSimulating episode {ep + 1}...")
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
    print(f"  Episode {ep + 1} finished, buffer has {len(buffer)} episodes")

assert len(buffer) == 2, f"Expected 2 episodes, got {len(buffer)}"
print(f"\n✓ Buffer has correct number of episodes: {len(buffer)}")

episodes = buffer.get_all_episodes()
assert len(episodes[0]['rewards']) == 5, "Expected 5 steps per episode"
print(f"✓ Episode 0 has correct number of steps: {len(episodes[0]['rewards'])}")

print(f"✓ Episode 0 data keys: {list(episodes[0].keys())}")

# Test clear
buffer.clear()
assert len(buffer) == 0, f"Expected 0 episodes after clear, got {len(buffer)}"
print(f"✓ Buffer cleared successfully: {len(buffer)} episodes")

print()
print("=" * 60)
print("All tests passed! ✅")
print("=" * 60)
