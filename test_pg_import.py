"""
Test import of Policy Gradient module
"""
import sys
sys.path.append('GraphRL_story_level')

print("Testing imports...")
from RL.pg import A2CAgent, PPOAgent, PolicyNetwork, ValueNetwork, ExperienceBuffer, BasePGAgent

print("✓ All imports successful!")
print(f"  - PolicyNetwork: {PolicyNetwork}")
print(f"  - ValueNetwork: {ValueNetwork}")
print(f"  - ExperienceBuffer: {ExperienceBuffer}")
print(f"  - BasePGAgent: {BasePGAgent}")
print(f"  - A2CAgent: {A2CAgent}")
print(f"  - PPOAgent: {PPOAgent}")
