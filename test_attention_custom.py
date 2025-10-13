#!/usr/bin/env python3
"""
Quick test script for custom state attention visualization.

This script demonstrates how to use visualize_attention_custom_state.py
with example section configurations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from Visualization.visualize_attention_custom_state import visualize_attention_custom_state


def main():
    """Test attention visualization with example configurations."""

    # ========================================================================
    # CONFIGURATION - MODIFY THESE VALUES
    # ========================================================================

    model_path = "checkpoints/option_oc/OB_002/best_model.pt"
    checkpoint_dir = "checkpoints/option_oc/OB_002"
    structure_shape = "random"  # or "4story", "7story", etc.
    device = "cuda"  # or "cpu"

    # Example section configurations
    # Format: [xbeam_s1, ..., xbeam_sN, zbeam_s1, ..., zbeam_sN,
    #          outcol_s1, ..., outcol_sN, incol_s1, ..., incol_sN]

    # For 4-story building: 4 stories × 4 member types = 16 values
    example_configs = {
        "initial_thick": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        "optimized": [2, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 1, 0, 0, 0],
        "uniform_thin": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "gradient": [5, 4, 3, 2, 5, 4, 3, 2, 6, 5, 4, 3, 4, 3, 2, 1],
    }

    # ========================================================================
    # RUN VISUALIZATION
    # ========================================================================

    print("="*70)
    print("Testing Custom State Attention Visualization")
    print("="*70)

    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ ERROR: Model not found at: {model_path}")
        print("\nPlease update 'model_path' in this script to point to your trained model.")
        print("\nAvailable models:")

        checkpoints_dir = Path("checkpoints/option_oc")
        if checkpoints_dir.exists():
            for subdir in sorted(checkpoints_dir.iterdir()):
                if subdir.is_dir():
                    best_model = subdir / "best_model.pt"
                    if best_model.exists():
                        print(f"  ✓ {best_model}")
        return

    print(f"\n📁 Model: {model_path}")
    print(f"📁 Output: {checkpoint_dir}/attention_custom_*/")
    print(f"🔧 Device: {device}")
    print(f"🏢 Structure: {structure_shape}")

    # Select which configuration to run
    print("\n" + "="*70)
    print("Available Configurations:")
    print("="*70)
    for idx, (name, sections) in enumerate(example_configs.items(), 1):
        print(f"{idx}. {name:20s} - {sections}")

    print("\n" + "="*70)
    print("Running visualization for all configurations...")
    print("="*70)

    for config_name, story_sections in example_configs.items():
        print(f"\n{'='*70}")
        print(f"Processing: {config_name}")
        print(f"Sections: {story_sections}")
        print(f"{'='*70}")

        try:
            visualize_attention_custom_state(
                model_path=model_path,
                checkpoint_dir=checkpoint_dir,
                structure_shape=structure_shape,
                story_sections=story_sections,
                device=device,
                output_suffix=config_name
            )
            print(f"✅ Success: {config_name}")
        except Exception as e:
            print(f"❌ Error processing {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("All configurations processed!")
    print("="*70)
    print("\nCheck output directories:")
    for config_name in example_configs.keys():
        output_dir = Path(checkpoint_dir) / f"attention_custom_{config_name}"
        print(f"  - {output_dir}/")


if __name__ == "__main__":
    main()
