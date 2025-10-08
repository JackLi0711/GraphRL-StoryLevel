#!/usr/bin/env python3
"""
Quick test script to debug reward calculation issues.

This script tests:
1. Whether actions actually change structure sections
2. Whether material usage changes
3. Whether rewards are calculated correctly

Run with: python test_reward_debug.py
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from RL.environment import Environment
from Validation import normalization as nda_norm

def test_reward_calculation():
    """Test if rewards are being calculated correctly."""

    print("=" * 80)
    print("REWARD DEBUG TEST")
    print("=" * 80)

    # Create environment with same settings as training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}

    env = Environment(
        structure_shape='random',
        add_structure_geometry=True,
        add_response_features=True,
        reward_type='material_usage',
        scwb_driven_design=False,
        do_nonlinear_dynamic_analysis=False,
        check_acceleration=False,
        check_displacement=True,
        nda_simulator=None,
        nda_norm_dict=nda_norm_dict,
        DBE_ground_motion_set=[],
        MCE_ground_motion_set=[],
        checkpoint_dir=None,
        logger=None,
        device=device,
    )

    # Reset to get initial structure
    structure = env.reset(testing=False)

    print("\n" + "=" * 80)
    print("INITIAL STRUCTURE")
    print("=" * 80)
    print(f"Structure shape: {structure.x_span_num}x{structure.z_span_num}, {structure.story_num} stories")
    print(f"Story level sections: {structure.story_level_sections}")
    print(f"Number of story-level actions: {len(structure.story_level_actions)}")

    initial_material = structure.calculate_material_usage()
    print(f"Initial material usage: {initial_material:.6f} kg")

    # Test first 5 actions
    print("\n" + "=" * 80)
    print("TESTING ACTIONS")
    print("=" * 80)

    zero_reward_count = 0
    nonzero_reward_count = 0
    section_changed_count = 0
    material_changed_count = 0

    for action_idx in range(min(5, len(structure.story_level_actions))):
        print(f"\n{'─' * 80}")
        print(f"ACTION {action_idx}")
        print(f"{'─' * 80}")

        # Record state before action
        material_before = structure.calculate_material_usage()
        sections_before = structure.story_level_sections.copy()

        print(f"BEFORE action {action_idx}:")
        print(f"  Material: {material_before:.6f} kg")
        print(f"  Sections (first 10): {sections_before[:10]}")

        # Apply action
        try:
            structure, reward, done, fail_name, fail_reason = env.step(structure, action_idx)

            # Record state after action
            material_after = structure.calculate_material_usage()
            sections_after = structure.story_level_sections

            material_saved = material_before - material_after
            sections_changed = (sections_before != sections_after)

            print(f"AFTER action {action_idx}:")
            print(f"  Material: {material_after:.6f} kg")
            print(f"  Sections (first 10): {sections_after[:10]}")
            print(f"  Material saved: {material_saved:.6f} kg")
            print(f"  Reward from env: {reward:.6f}")
            print(f"  Done: {done}, Fail: {fail_reason}")
            print(f"  Sections changed: {sections_changed}")

            # Statistics
            if abs(reward) < 1e-6:
                zero_reward_count += 1
                print(f"  ⚠️  ZERO REWARD")
            else:
                nonzero_reward_count += 1

            if sections_changed:
                section_changed_count += 1
            else:
                print(f"  ⚠️  SECTIONS DID NOT CHANGE")

            if abs(material_saved) > 1e-6:
                material_changed_count += 1
            else:
                print(f"  ⚠️  MATERIAL DID NOT CHANGE")

            # Check reward vs material consistency
            reward_mismatch = abs(reward - material_saved) > 1e-3
            if reward_mismatch:
                print(f"  ⚠️  REWARD MISMATCH!")
                print(f"     Expected (material_saved): {material_saved:.6f}")
                print(f"     Actual (reward): {reward:.6f}")
                print(f"     Difference: {abs(reward - material_saved):.6f}")

            # If done, reset for next test
            if done:
                print(f"  Episode ended, resetting structure...")
                structure = env.reset(testing=False)

        except Exception as e:
            print(f"  ❌ ERROR applying action {action_idx}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Actions tested: {min(5, len(structure.story_level_actions))}")
    print(f"Zero rewards: {zero_reward_count}")
    print(f"Non-zero rewards: {nonzero_reward_count}")
    print(f"Sections changed: {section_changed_count}")
    print(f"Material changed: {material_changed_count}")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if zero_reward_count > 3:
        print("❌ PROBLEM: Most rewards are ZERO!")
        if section_changed_count == 0:
            print("   → Hypothesis 1 CONFIRMED: Actions are NOT changing sections")
        elif material_changed_count == 0:
            print("   → Hypothesis 2 CONFIRMED: Material calculation is broken")
        else:
            print("   → Reward calculation logic may be wrong")
    else:
        print("✅ Rewards appear to be working")

    if section_changed_count == 0:
        print("❌ PROBLEM: Sections are NOT changing after actions!")
    else:
        print(f"✅ Sections changing: {section_changed_count}/{min(5, len(structure.story_level_actions))}")

    if material_changed_count == 0:
        print("❌ PROBLEM: Material usage is NOT changing!")
    else:
        print(f"✅ Material changing: {material_changed_count}/{min(5, len(structure.story_level_actions))}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_reward_calculation()
