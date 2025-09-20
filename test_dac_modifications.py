#!/usr/bin/env python3
"""
Test script for DAC modifications.

This script tests the key modifications made to DAC:
1. Action restriction functionality
2. Structure checking for all-zeros termination
3. Integration between Agent, Environment, and Trainer
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from RL.dac.config import DACConfig
from RL.dac.agent import DACAgent
from RL.dac.environment import DACEnvironmentWrapper
from RL.environment import Environment
import logging

# Mock structure class for testing
class MockStructure:
    def __init__(self):
        self.story_level_actions = list(range(10))  # 10 possible actions
        self.already_minimum_section_story_indexes = []
        self.restricted_actions = set()

        # Add required structural properties
        self.x_span_num = 3
        self.z_span_num = 3
        self.story_num = 5
        self.story_height = 3000.0

        self.aux = {
            'story_xdir_beam_member': [],
            'story_zdir_beam_member': [],
            'story_outer_column_member': [],
            'story_inner_column_member': []
        }

    def restrict_action_space(self):
        """Mock action restriction function."""
        return list(self.restricted_actions)

    def calculate_material_usage(self):
        """Mock material usage calculation."""
        return 1000.0

    def set_all_minimum(self):
        """Set all actions to minimum for testing."""
        self.already_minimum_section_story_indexes = list(range(len(self.story_level_actions)))

    def set_restricted_actions(self, actions):
        """Set restricted actions for testing."""
        self.restricted_actions = set(actions)

    def init_graph_GraphRL(self, static_response_features=None, dynamic_response_features=None):
        """Mock graph initialization."""
        # Create a simple mock graph
        import torch
        from types import SimpleNamespace

        self.graph = SimpleNamespace()
        self.graph.x = torch.randn(10, 32)  # 10 nodes, 32 features (node_feature_dim)
        self.graph.edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
        self.graph.edge_attr = torch.randn(20, 16)  # 20 edges, 16 features (edge_feature_dim)

        # Add story_batch for environment wrapper
        self.aux['story_batch'] = torch.zeros(10, dtype=torch.long)

def test_action_restriction():
    """Test action restriction functionality."""
    print("=== Testing Action Restriction ===")

    # Create mock config and agent
    config = DACConfig()
    agent = DACAgent(config)

    # Create mock structure
    structure = MockStructure()
    structure.set_restricted_actions([0, 1, 5])  # Restrict actions 0, 1, 5

    # Test valid actions mask creation
    valid_mask = agent._create_valid_actions_mask(structure, torch.device('cpu'))

    print(f"Total actions: {len(structure.story_level_actions)}")
    print(f"Restricted actions: {structure.restrict_action_space()}")
    print(f"Valid mask: {valid_mask}")
    print(f"Valid actions count: {valid_mask.sum().item()}")

    # Check that restricted actions are indeed masked out
    for action in structure.restrict_action_space():
        assert not valid_mask[action], f"Action {action} should be invalid but mask is True"

    print("✓ Action restriction test passed!")
    return True

def test_all_indexes_zero():
    """Test all-indexes-zero termination condition."""
    print("\n=== Testing All-Indexes-Zero Termination ===")

    # Create mock config and agent
    config = DACConfig()
    agent = DACAgent(config)

    # Test with normal structure (not all minimum)
    structure = MockStructure()
    structure.already_minimum_section_story_indexes = [0, 1, 2]  # Only some actions at minimum

    should_terminate, reason = agent.check_termination_conditions(structure)
    print(f"Normal structure - Terminate: {should_terminate}, Reason: {reason}")
    assert not should_terminate, "Should not terminate when not all actions are at minimum"

    # Test with all actions at minimum
    structure.set_all_minimum()
    should_terminate, reason = agent.check_termination_conditions(structure)
    print(f"All minimum structure - Terminate: {should_terminate}, Reason: {reason}")
    assert should_terminate, "Should terminate when all actions are at minimum"
    assert reason == "all_indexes_zero", f"Expected reason 'all_indexes_zero', got '{reason}'"

    print("✓ All-indexes-zero termination test passed!")
    return True

def test_environment_integration():
    """Test environment wrapper integration."""
    print("\n=== Testing Environment Integration ===")

    # Create mock base environment
    class MockBaseEnvironment:
        def __init__(self):
            self._testing_structure = MockStructure()

        def reset(self, **kwargs):
            return MockStructure()

        def step(self, structure, action):
            # Mock step function that simulates Option-Critic behavior
            next_structure = MockStructure()
            reward = -1.0
            step_pass = True
            is_min_section = False
            fail_reason = None

            # Simulate minimum section check
            if len(next_structure.already_minimum_section_story_indexes) >= len(next_structure.story_level_actions):
                is_min_section = True

            return next_structure, reward, step_pass, is_min_section, fail_reason

    # Create environment wrapper
    base_env = MockBaseEnvironment()
    env_wrapper = DACEnvironmentWrapper(base_env)

    # Test reset
    dual_states, info = env_wrapper.reset()
    print(f"Reset successful - dual_states keys: {dual_states.keys()}")
    print(f"Info keys: {info.keys()}")

    # Test step
    action = torch.tensor([2])  # Valid action
    option = 0
    next_dual_states, rewards, done, env_info = env_wrapper.step(action, option, False)

    print(f"Step successful - rewards: {rewards}")
    print(f"Done: {done}")
    print(f"Environment info keys: {env_info.keys()}")

    print("✓ Environment integration test passed!")
    return True

def test_combined_functionality():
    """Test combined functionality simulation."""
    print("\n=== Testing Combined Functionality ===")

    try:
        # This would be a more comprehensive test if we had access to the full environment
        # For now, we just verify that the key components can be initialized together

        # Create config
        config = DACConfig()
        config.max_episodes = 1
        config.max_steps_per_episode = 5

        # Create mock base environment
        class MockBaseEnvironment:
            def __init__(self):
                self._testing_structure = MockStructure()

            def reset(self, **kwargs):
                return MockStructure()

        base_env = MockBaseEnvironment()

        # Create environment wrapper
        env_wrapper = DACEnvironmentWrapper(base_env)

        # Create agent
        agent = DACAgent(config)

        # Test initialization
        print("✓ All components initialized successfully!")

        # Test action selection with restrictions
        structure = MockStructure()
        structure.set_restricted_actions([0, 1])

        # Create dummy graph data with correct dimensions
        graph_data = {
            'x': torch.randn(10, 32),  # node_feature_dim = 32
            'edge_index': torch.randint(0, 10, (2, 20)),
            'edge_attr': torch.randn(20, 16),  # edge_feature_dim = 16
            'story_batch': torch.zeros(10, dtype=torch.long),
            'structure_story_ptr': [0, 5]
        }

        # Test action selection
        option, action, info = agent.select_option_and_action(graph_data, structure)
        print(f"Selected option: {option}, action: {action}")
        print(f"Action is valid: {action.item() not in structure.restrict_action_space()}")

        print("✓ Combined functionality test passed!")
        return True

    except Exception as e:
        print(f"✗ Combined functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing DAC Modifications")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing

    tests = [
        test_action_restriction,
        test_all_indexes_zero,
        test_environment_integration,
        test_combined_functionality
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! DAC modifications are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the modifications.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)