#!/usr/bin/env python3
"""
Test option preview visualization with actual model loading.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_mock_logger():
    """Create a mock logger for testing."""
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def test_option_preview_with_checkpoint():
    """Test option preview visualization with real checkpoint."""

    try:
        from RL.environment import Environment
        from RL.option_critic_gnn import OptionCriticGNN
        from Visualization.visualize import visualize_option_preview_process

        # Create mock logger
        logger = create_mock_logger()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create temporary checkpoint dir
        checkpoint_dir = Path(tempfile.mkdtemp())
        logger.info(f"Using temporary checkpoint dir: {checkpoint_dir}")

        # Create minimal environment for testing
        try:
            env = Environment(
                structure_shape="fixed",
                add_structure_geometry=True,
                add_response_features=True,
                reward_type="material",
                scwb_driven_design=False,
                do_nonlinear_dynamic_analysis=False,
                check_acceleration=False,
                check_displacement=True,
                nda_simulator=None,
                nda_norm_dict=None,
                DBE_ground_motion_set=None,
                MCE_ground_motion_set=None,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
                device=device
            )
            structure = env.reset(testing=True)
            logger.info("✅ Environment created successfully")

        except Exception as e:
            logger.error(f"❌ Failed to create environment: {e}")
            return False

        # Create mock Option-Critic model
        try:
            num_actions = len(structure.story_level_actions)
            node_dim = structure.graph.x.shape[1]
            edge_dim = structure.graph.edge_attr.shape[1]

            oc_model = OptionCriticGNN(
                node_feature_dim=node_dim,
                edge_feature_dim=edge_dim,
                hidden_dim=128,
                member_state_dim=128,
                num_layers=3,
                num_actions=num_actions,
                num_options=4,
                device=device,
                testing=True
            )
            logger.info(f"✅ Created Option-Critic model with {num_actions} actions")

        except Exception as e:
            logger.error(f"❌ Failed to create Option-Critic model: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Create mock agent
        mock_agent = type('MockAgent', (), {
            'device': device,
            'gnn': oc_model
        })()

        # Create mock episode data
        episode_data = {
            'actions': [0, 1, 2, 5, 6, 7, 10, 11],
            'options': [0, 0, 0, 1, 1, 1, 2, 2],
            'option_instances': [
                {'option_index': 0, 'instance_id': 0, 'action_step': 0},
                {'option_index': 0, 'instance_id': 0, 'action_step': 1},
                {'option_index': 0, 'instance_id': 0, 'action_step': 2},
                {'option_index': 1, 'instance_id': 1, 'action_step': 3},
                {'option_index': 1, 'instance_id': 1, 'action_step': 4},
                {'option_index': 1, 'instance_id': 1, 'action_step': 5},
                {'option_index': 2, 'instance_id': 2, 'action_step': 6},
                {'option_index': 2, 'instance_id': 2, 'action_step': 7},
            ]
        }

        episode_info = {
            'episode_number': 1,
            'round_number': 1,
            'score': 85.5,
            'avg_score': 80.0
        }

        # Create a mock checkpoint file (don't actually save it)
        mock_checkpoint_path = checkpoint_dir / "best_model.pt"

        logger.info("🧪 Testing option preview visualization...")

        # Call the visualization function
        visualize_option_preview_process(
            agent=mock_agent,
            env=env,
            logger=logger,
            save_model_path=mock_checkpoint_path,  # This won't exist, should use current weights
            episode_data=episode_data,
            episode_info=episode_info,
            save_base_dir=checkpoint_dir
        )

        # Check if output was generated
        expected_dir = checkpoint_dir / f"option_preview_process_round1_ep1_score85.50"
        if expected_dir.exists():
            png_files = list(expected_dir.glob("*.png"))
            json_files = list(expected_dir.glob("*.json"))
            gif_files = list(expected_dir.glob("*.gif"))

            logger.info(f"✅ Visualization completed successfully!")
            logger.info(f"   Generated {len(png_files)} PNG files")
            logger.info(f"   Generated {len(json_files)} JSON files")
            logger.info(f"   Generated {len(gif_files)} GIF files")
            logger.info(f"   Output directory: {expected_dir}")

            return True
        else:
            logger.error(f"❌ Expected output directory not found: {expected_dir}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Option Preview Visualization with Model Test")
    print("=" * 60)

    success = test_option_preview_with_checkpoint()

    print("=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    print("=" * 60)

    sys.exit(0 if success else 1)