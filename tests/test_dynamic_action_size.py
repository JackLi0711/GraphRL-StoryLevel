"""
Test cases for dynamic action size handling in Option-Critic.
Tests the new per-story-member scoring architecture.
"""
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from RL.option_critic_gnn import OptionCriticGNN, actor_loss


def create_test_model(num_options=4, hidden_dim=64):
    """Create a test model instance."""
    return OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=hidden_dim,
        member_state_dim=hidden_dim,
        num_layers=2,
        num_actions=100,  # Ignored in new architecture
        num_options=num_options,
        device='cpu',
        testing=False,
        debug_logging=False
    )


def test_intra_option_policy_architecture():
    """Test that intra-option policies are MLPs, not parameter matrices."""
    model = create_test_model()

    # Check that intra_option_policies is a ModuleList
    assert hasattr(model, 'intra_option_policies'), "Model should have intra_option_policies"
    assert isinstance(model.intra_option_policies, torch.nn.ModuleList), \
        "intra_option_policies should be ModuleList"

    # Check that old architecture is removed
    assert not hasattr(model, 'options_W'), "Old options_W should be removed"
    assert not hasattr(model, 'options_b'), "Old options_b should be removed"

    # Check number of option policies
    assert len(model.intra_option_policies) == 4, "Should have 4 option policies"

    # Check that each policy is a Sequential MLP
    for i, policy in enumerate(model.intra_option_policies):
        assert isinstance(policy, torch.nn.Sequential), \
            f"Policy {i} should be Sequential"

    print("✓ Architecture test passed")


def test_get_action_with_varying_num_members():
    """
    Test that get_action works with different numbers of story members.
    Simulates episodes with different building heights.
    """
    model = create_test_model()

    # Test scenarios: different building heights
    test_cases = [
        {"floors": 4, "members_per_floor": 4, "total": 16},   # 4-story building
        {"floors": 7, "members_per_floor": 4, "total": 28},   # 7-story building
        {"floors": 10, "members_per_floor": 4, "total": 40},  # 10-story building
    ]

    gnn_output_dim = model.member_state_dim * 2  # 128

    for case in test_cases:
        num_members = case["total"]

        # Create dummy story-level state with varying size
        story_level_state = torch.randn(num_members, gnn_output_dim)

        # Create valid actions mask (some members can be modified)
        valid_mask = torch.ones(num_members, dtype=torch.bool)
        # Mask out last floor (cannot modify)
        members_per_floor = case["members_per_floor"]
        valid_mask[-members_per_floor:] = False

        # Get action
        action, logp, entropy = model.get_action(
            story_level_state=story_level_state,
            structure_story_ptr=None,
            option=0,
            valid_actions_mask=valid_mask
        )

        # Assertions
        assert isinstance(action, int), f"Action should be int, got {type(action)}"
        assert 0 <= action < num_members, \
            f"Action {action} out of range [0, {num_members})"
        assert valid_mask[action], \
            f"Action {action} should be valid according to mask"
        assert torch.isfinite(logp), "Log probability should be finite"
        assert torch.isfinite(entropy), "Entropy should be finite"

        print(f"✓ {case['floors']}-story building ({num_members} members): "
              f"action={action}, logp={logp.item():.4f}, entropy={entropy.item():.4f}")


def test_compute_Q_U_independent_of_action_size():
    """
    Test that Q_U computation is independent of action space size.
    """
    model = create_test_model()

    # Create dummy states
    global_state = torch.randn(1, 64)
    next_global_state = torch.randn(1, 64)

    # Test with different action indices (representing different story members)
    # The Q_U value should not depend on the action index itself,
    # only on the reward and next state
    action_indices = [0, 10, 20, 50]

    Q_U_values = []
    for action_idx in action_indices:
        Q_U = model.compute_Q_U(
            global_state=global_state,
            option=0,
            action=action_idx,
            reward=1.0,  # Same reward
            next_global_state=next_global_state,  # Same next state
            gamma=0.99
        )
        Q_U_values.append(Q_U.item())

    # All Q_U values should be identical (action index doesn't affect computation)
    for i in range(1, len(Q_U_values)):
        assert abs(Q_U_values[i] - Q_U_values[0]) < 1e-6, \
            f"Q_U should be same for all actions, got {Q_U_values}"

    print(f"✓ Q_U values independent of action index: {Q_U_values[0]:.4f}")


def test_actor_loss_with_different_action_sizes():
    """
    Test that actor_loss works correctly with different episode action sizes.
    """
    model = create_test_model()
    model_prime = create_test_model()
    model_prime.load_state_dict(model.state_dict())

    # Simulate different episode sizes
    episode_configs = [
        {"num_members": 16, "action": 5},   # 4-story
        {"num_members": 28, "action": 15},  # 7-story
        {"num_members": 40, "action": 25},  # 10-story
    ]

    for config in episode_configs:
        # Create dummy graph observations
        num_members = config["num_members"]
        action = config["action"]

        # Dummy graph data (simplified)
        graph_x = torch.randn(num_members, 10)
        graph_edge_index = torch.randint(0, num_members, (2, num_members * 2))
        graph_edge_attr = torch.randn(num_members * 2, 5)
        story_batch = torch.zeros(num_members, dtype=torch.long)

        obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
        next_obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)

        # Dummy action parameters
        option = 0
        logp = torch.tensor(-1.5, requires_grad=True)
        entropy = torch.tensor(2.0)
        reward = 1.0
        done = False

        # Compute loss
        loss = actor_loss(
            obs, option, action, logp, entropy, reward, done, next_obs,
            model, model_prime, gamma=0.99
        )

        # Assertions
        assert loss.shape == torch.Size([]), f"Loss should be scalar, got {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.requires_grad, "Loss should require gradients"

        print(f"✓ Episode with {num_members} members (action={action}): loss={loss.item():.6f}")


if __name__ == "__main__":
    print("Running dynamic action size tests...\n")

    test_intra_option_policy_architecture()
    print()

    test_get_action_with_varying_num_members()
    print()

    test_compute_Q_U_independent_of_action_size()
    print()

    test_actor_loss_with_different_action_sizes()
    print()

    print("All tests passed! ✅")
