"""
Integration tests for Option-Critic training pipeline.
Tests complete workflow with dynamic action sizes.
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from RL.option_critic_gnn import OptionCriticGNN, actor_loss, critic_loss


def test_complete_forward_pass():
    """
    Test a complete forward pass through the model.
    Simulates one step of training with varying action sizes.
    """
    print("Testing complete forward pass...")

    # Create model
    model = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,  # Ignored
        num_options=4,
        device='cpu',
        testing=False,
        debug_logging=False
    )

    # Test with different building sizes
    test_configs = [
        {"floors": 4, "members": 16},
        {"floors": 7, "members": 28},
        {"floors": 10, "members": 40},
    ]

    for config in test_configs:
        num_members = config["members"]
        gnn_output_dim = 128

        # Create dummy data
        story_level_state = torch.randn(num_members, gnn_output_dim)
        global_state = torch.randn(1, 64)

        # Test option selection
        Q_values = model.get_Q(global_state)
        assert Q_values.shape[-1] == 4, f"Should have 4 options, got {Q_values.shape}"

        # Test action selection
        valid_mask = torch.ones(num_members, dtype=torch.bool)
        valid_mask[-4:] = False  # Mask last floor

        action, logp, entropy = model.get_action(
            story_level_state, None, option=0, valid_actions_mask=valid_mask
        )

        assert 0 <= action < num_members, f"Action out of range"
        assert valid_mask[action], f"Selected invalid action"
        assert torch.isfinite(logp), "Invalid log probability"
        assert torch.isfinite(entropy), "Invalid entropy"

        print(f"✓ {config['floors']}-story building: "
              f"Q_shape={Q_values.shape}, action={action}, "
              f"logp={logp.item():.4f}, entropy={entropy.item():.4f}")

    print("✓ Complete forward pass test passed\n")


def test_loss_computation_pipeline():
    """
    Test that actor and critic losses can be computed correctly.
    """
    print("Testing loss computation pipeline...")

    model = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,
        num_options=4,
        device='cpu',
        testing=False,
        debug_logging=False
    )

    model_prime = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,
        num_options=4,
        device='cpu',
        testing=True,
        debug_logging=False
    )
    model_prime.load_state_dict(model.state_dict())

    # Test with 7-story building (28 members)
    num_members = 28

    # Create dummy graph observations
    graph_x = torch.randn(num_members, 10)
    graph_edge_index = torch.randint(0, num_members, (2, num_members * 2))
    graph_edge_attr = torch.randn(num_members * 2, 5)
    story_batch = torch.zeros(num_members, dtype=torch.long)

    obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
    next_obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)

    # Test actor loss
    option = 1
    action = 15
    logp = torch.tensor(-1.5, requires_grad=True)
    entropy = torch.tensor(2.0)
    reward = 1.0
    done = False

    a_loss = actor_loss(
        obs, option, action, logp, entropy, reward, done, next_obs,
        model, model_prime, gamma=0.99
    )

    assert a_loss.shape == torch.Size([]), "Actor loss should be scalar"
    assert torch.isfinite(a_loss), "Actor loss should be finite"
    assert a_loss.requires_grad, "Actor loss should have gradients"

    print(f"✓ Actor loss: {a_loss.item():.6f}")

    # Test critic loss with batch
    batch_size = 4
    obs_batch = (
        graph_x.repeat(batch_size, 1, 1),
        graph_edge_index.repeat(batch_size, 1, 1),
        graph_edge_attr.repeat(batch_size, 1, 1),
        story_batch.repeat(batch_size, 1)
    )
    next_obs_batch = obs_batch

    options_batch = [0, 1, 2, 3]
    rewards_batch = [1.0, 0.5, 0.0, -0.5]
    dones_batch = [False, False, False, False]

    data_batch = (obs_batch, options_batch, rewards_batch, next_obs_batch, dones_batch)

    c_loss = critic_loss(model, model_prime, data_batch, gamma=0.99)

    assert c_loss.shape == torch.Size([]), "Critic loss should be scalar"
    assert torch.isfinite(c_loss), "Critic loss should be finite"
    assert c_loss.requires_grad, "Critic loss should have gradients"

    print(f"✓ Critic loss: {c_loss.item():.6f}")
    print("✓ Loss computation pipeline test passed\n")


def test_gradient_flow():
    """
    Test that gradients flow correctly through the model.
    """
    print("Testing gradient flow...")

    model = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,
        num_options=4,
        device='cpu',
        testing=False,
        debug_logging=False
    )

    model_prime = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,
        num_options=4,
        device='cpu',
        testing=True,
        debug_logging=False
    )
    model_prime.load_state_dict(model.state_dict())

    # Create dummy data
    num_members = 16
    graph_x = torch.randn(num_members, 10)
    graph_edge_index = torch.randint(0, num_members, (2, num_members * 2))
    graph_edge_attr = torch.randn(num_members * 2, 5)
    story_batch = torch.zeros(num_members, dtype=torch.long)

    obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)
    next_obs = (graph_x, graph_edge_index, graph_edge_attr, story_batch, None)

    # Compute actor loss
    option = 0
    action = 5
    logp = torch.tensor(-1.5, requires_grad=True)
    entropy = torch.tensor(2.0)
    reward = 1.0
    done = False

    a_loss = actor_loss(
        obs, option, action, logp, entropy, reward, done, next_obs,
        model, model_prime, gamma=0.99
    )

    # Backpropagate
    a_loss.backward()

    # Check that gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"  ✓ Gradient exists for {name}: mean={param.grad.abs().mean():.6f}")

    assert has_grad, "Model should have gradients after backward pass"
    print("✓ Gradient flow test passed\n")


def test_parameter_count():
    """
    Test that new architecture has reasonable parameter count.
    """
    print("Testing parameter count...")

    model = OptionCriticGNN(
        node_feature_dim=10,
        edge_feature_dim=5,
        hidden_dim=64,
        member_state_dim=64,
        num_layers=2,
        num_actions=100,
        num_options=4,
        device='cpu',
        testing=False,
        debug_logging=False
    )

    # Count parameters by component
    policy_params = sum(p.numel() for policy in model.intra_option_policies for p in policy.parameters())
    q_params = sum(p.numel() for p in model.Q.parameters())
    termination_params = sum(p.numel() for p in model.terminations.parameters())
    gnn_params = sum(p.numel() for p in model.state_gnn.parameters())

    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Policy parameters: {policy_params:,}")
    print(f"  Q network parameters: {q_params:,}")
    print(f"  Termination parameters: {termination_params:,}")
    print(f"  StateGNN parameters: {gnn_params:,}")
    print(f"  Total parameters: {total_params:,}")

    # Verify intra_option_policies exist
    assert len(model.intra_option_policies) == 4, "Should have 4 option policies"
    assert policy_params > 0, "Policy should have parameters"

    # Verify old architecture is removed
    assert not hasattr(model, 'options_W'), "options_W should be removed"
    assert not hasattr(model, 'options_b'), "options_b should be removed"

    print("✓ Parameter count test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Option-Critic Integration Tests")
    print("=" * 60)
    print()

    test_complete_forward_pass()
    test_loss_computation_pipeline()
    test_gradient_flow()
    test_parameter_count()

    print("=" * 60)
    print("All integration tests passed! ✅")
    print("=" * 60)
