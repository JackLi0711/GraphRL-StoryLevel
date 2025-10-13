#!/usr/bin/env python3
"""
Test script for attention visualization.

This script helps you test the attention visualization functionality
with a trained model.

Usage:
    python test_attention_viz.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from Visualization.visualize_attention import visualize_model_attention


def main():
    """Test attention visualization with example parameters."""

    # Example configuration - modify these paths to your trained model
    model_path = "checkpoints/option_oc/YOUR_TIMESTAMP/best_model.pt"
    checkpoint_dir = "checkpoints/option_oc/YOUR_TIMESTAMP"

    # Check if paths exist
    if not Path(model_path).exists():
        print("="*60)
        print("ERROR: Model path does not exist!")
        print(f"Please update 'model_path' in this script to point to your trained model.")
        print(f"Current path: {model_path}")
        print("="*60)

        # Try to find available models
        checkpoints_dir = Path("checkpoints/option_oc")
        if checkpoints_dir.exists():
            print("\nAvailable checkpoint directories:")
            for subdir in sorted(checkpoints_dir.iterdir()):
                if subdir.is_dir():
                    best_model = subdir / "best_model.pt"
                    if best_model.exists():
                        print(f"  ✓ {subdir.name}/best_model.pt")
                    else:
                        print(f"  ✗ {subdir.name}/ (no best_model.pt)")
        return

    print("="*60)
    print("Testing Attention Visualization")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {checkpoint_dir}/attention_visualization/")
    print("="*60)

    # Run visualization
    structure_config = {
        'structure_shape': '4story'  # Change to match your model
    }

    try:
        visualize_model_attention(
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
            structure_config=structure_config,
            device='cuda'  # Change to 'cpu' if no GPU
        )

        print("\n" + "="*60)
        print("SUCCESS! Check the output directory for results:")
        print(f"  {checkpoint_dir}/attention_visualization/")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print("ERROR during visualization:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
