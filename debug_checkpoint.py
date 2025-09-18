#!/usr/bin/env python3
"""
Debug script to inspect checkpoint contents and test model loading.
"""

import torch
from pathlib import Path
import sys

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint file contents."""
    print(f"Inspecting checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file does not exist: {checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Successfully loaded checkpoint")

        print(f"\n📁 Checkpoint keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key])} = {checkpoint[key]}")

        # If there's a state_dict, show its structure
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n🧩 State dict structure ({len(state_dict)} parameters):")
            for key in sorted(state_dict.keys())[:10]:  # Show first 10 keys
                tensor = state_dict[key]
                print(f"  {key}: {tensor.shape}")
            if len(state_dict) > 10:
                print(f"  ... and {len(state_dict) - 10} more parameters")

        return checkpoint

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def test_option_critic_loading():
    """Test loading Option-Critic model from checkpoint."""
    from RL.option_critic_gnn import OptionCriticGNN

    # Find the most recent checkpoint
    checkpoints_dir = Path("checkpoints/option_oc")
    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return

    # Find all timestamp directories
    timestamp_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        print(f"❌ No checkpoint directories found in {checkpoints_dir}")
        return

    # Get the most recent one
    latest_dir = max(timestamp_dirs, key=lambda d: d.name)
    best_model_path = latest_dir / "best_model.pt"

    print(f"🔍 Testing with checkpoint: {best_model_path}")
    checkpoint = inspect_checkpoint(best_model_path)

    if checkpoint is None:
        return

    try:
        # Try to create a dummy Option-Critic model and load the state
        print(f"\n🧪 Testing model loading...")

        # Create dummy model with minimal parameters
        model = OptionCriticGNN(
            node_feature_dim=10,
            edge_feature_dim=12,
            hidden_dim=128,
            member_state_dim=128,
            num_layers=3,
            num_actions=28,  # typical for 7-story structure
            num_options=4,
            device='cpu'
        )

        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✅ Successfully loaded state_dict into Option-Critic model (new format)")
        elif 'Q.weight' in checkpoint:
            model.load_state_dict(checkpoint)
            print(f"✅ Successfully loaded parameters into Option-Critic model (old format)")
        else:
            print(f"❌ Unexpected checkpoint format. No 'state_dict' or 'Q.weight' found")

    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Inspect specific checkpoint
        checkpoint_path = sys.argv[1]
        inspect_checkpoint(checkpoint_path)
    else:
        # Test automatic discovery and loading
        test_option_critic_loading()