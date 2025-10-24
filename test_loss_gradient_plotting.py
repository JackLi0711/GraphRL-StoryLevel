"""
Test script to verify loss and gradient plotting functionality

This script creates dummy data and tests the plotting functions
"""

import numpy as np
from pathlib import Path
from RL.record import Record
from RL.pg.train_utils import plot_loss_curves, plot_gradient_norms

# Create a test directory
test_dir = Path("./test_plots")
test_dir.mkdir(exist_ok=True)

# Create a Record object
rec = Record()

# Simulate some training data
num_episodes = 10
for i in range(1, num_episodes + 1):
    # Only record when there's an update (every 5 episodes for accumulate_episodes=5)
    if i % 5 == 0:
        rec.loss_record['episodes'].append(i)
        rec.loss_record['policy_loss'].append(0.5 + 0.1 * np.random.randn() - 0.01 * i)
        rec.loss_record['value_loss'].append(0.3 + 0.05 * np.random.randn() - 0.005 * i)
        rec.loss_record['entropy_loss'].append(0.2 + 0.02 * np.random.randn() - 0.003 * i)
        rec.loss_record['total_loss'].append(
            rec.loss_record['policy_loss'][-1] +
            rec.loss_record['value_loss'][-1] +
            rec.loss_record['entropy_loss'][-1]
        )

        rec.gradient_record['episodes'].append(i)
        # StateGNN gradient should be much smaller (before batch optimization)
        rec.gradient_record['state_gnn_grad'].append(1e-3 * (1 + 0.1 * np.random.randn()))
        rec.gradient_record['policy_grad'].append(1e-2 * (1 + 0.1 * np.random.randn()))
        rec.gradient_record['value_grad'].append(1e-2 * (1 + 0.1 * np.random.randn()))

# Test plotting functions
print("Testing plot_loss_curves...")
plot_loss_curves(rec, test_dir)
print(f"✓ Loss curves saved to {test_dir / 'loss_curves.png'}")

print("\nTesting plot_gradient_norms...")
plot_gradient_norms(rec, test_dir)
print(f"✓ Gradient norms saved to {test_dir / 'gradient_norms.png'}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
print(f"\nPlease check the plots in: {test_dir.absolute()}")
print("\nExpected plots:")
print("  1. loss_curves.png - Shows policy, value, entropy, and total losses")
print("  2. gradient_norms.png - Shows gradient norms on log scale")
