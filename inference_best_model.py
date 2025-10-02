import argparse
import json
import logging
from pathlib import Path
import torch
import numpy as np

from RL.environment import Environment
from RL.option_critic_gnn import OptionCriticGNN
from Validation import normalization as nda_norm
from train_option_critic import get_graph_data, num_actions, evaluate_model


def parse_inference_args():
    parser = argparse.ArgumentParser(description="Run inference with the best trained model")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/option_oc", 
                       help="Directory containing the best model")
    parser.add_argument("--num_episodes", type=int, default=10, 
                       help="Number of episodes for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def load_best_model_for_inference(checkpoint_dir, device):
    """
    Load the best model and create the environment for inference.
    
    Returns:
        tuple: (model, environment, model_info) or (None, None, None) if failed
    """
    ckpt_dir = Path(checkpoint_dir)
    best_model_path = ckpt_dir / "best_model.pt"
    best_model_info_path = ckpt_dir / "best_model_info.json"
    
    if not best_model_path.exists() or not best_model_info_path.exists():
        print(f"Best model files not found in {checkpoint_dir}")
        return None, None, None
    
    try:
        # Load model info
        with open(best_model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        
        print(f"Loading best model from episode {model_info['episode']}")
        print(f"  Training score: {model_info['score']:.2f}")
        print(f"  Training success rate: {model_info['success_rate']:.1f}%")
        print(f"  Saved at: {model_info['timestamp']}")
        
        # Get hyperparameters
        hyperparams = model_info['hyperparameters']
        args = argparse.Namespace(**hyperparams)
        
        # Create environment using saved hyperparameters
        nda_simulator = None
        nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}
        
        base_env = Environment(
            structure_shape=args.structure_shape,
            add_structure_geometry=args.add_structure_geometry,
            add_response_features=args.add_response_features,
            reward_type=args.reward_type,
            scwb_driven_design=False,
            do_nonlinear_dynamic_analysis=False,
            check_acceleration=args.check_acceleration,
            check_displacement=args.check_displacement,
            nda_simulator=nda_simulator,
            nda_norm_dict=nda_norm_dict,
            DBE_ground_motion_set=[],
            MCE_ground_motion_set=[],
            checkpoint_dir=ckpt_dir,
            logger=None,
            device=device,
        )
        
        # Reset to get structure and infer dimensions
        structure = base_env.reset()
        graph = structure.graph
        node_feature_dim = graph.x.shape[1]
        edge_feature_dim = graph.edge_attr.shape[1]
        A = num_actions(structure)
        
        # Create model with same architecture
        model = OptionCriticGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=args.hidden_dim,
            member_state_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_actions=A,
            num_options=args.num_options,
            temperature=args.temperature,
            eps_start=args.eps_start,
            eps_min=args.eps_min,
            eps_decay=args.eps_decay,
            eps_test=args.eps_test,
            device=device,
            testing=True,  # Set to testing mode
        )
        
        # Load model weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        model.testing = True  # Ensure testing mode is set
        
        print(f"Model loaded successfully")
        print(f"  Architecture: {node_feature_dim} node features, {edge_feature_dim} edge features")
        print(f"  Hidden dim: {args.hidden_dim}, Options: {args.num_options}")
        print(f"  Action space: {A}")
        
        return model, base_env, model_info
        
    except Exception as e:
        print(f"Error loading best model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_inference(model, base_env, model_info, num_episodes, device, verbose=False):
    """
    Run inference with the loaded model.
    """
    print(f"\nRunning inference for {num_episodes} episodes...")
    print("=" * 50)
    
    # Setup logging
    logger = logging.getLogger("Inference")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Get hyperparameters
    hyperparams = model_info['hyperparameters']
    max_option_len = hyperparams['max_option_len']
    
    # Run evaluation
    try:
        avg_score, avg_episode_length, success_rate = evaluate_model(
            base_env, model, device, num_episodes, max_option_len, logger, seed=42
        )
        
        print(f"\nInference Results:")
        print(f"=" * 50)
        print(f"Episodes run: {num_episodes}")
        print(f"Average score: {avg_score:.2f}")
        print(f"Success rate: {success_rate:.1f}% ({int(success_rate * num_episodes / 100)}/{num_episodes})")
        print(f"Average episode length: {avg_episode_length:.2f} options")
        
        # Compare with training performance
        training_score = model_info['score']
        training_success_rate = model_info['success_rate']
        
        print(f"\nComparison with training performance:")
        print(f"Score: {avg_score:.2f} vs {training_score:.2f} (training) - {'✓' if avg_score >= training_score * 0.9 else '✗'}")
        print(f"Success rate: {success_rate:.1f}% vs {training_success_rate:.1f}% (training) - {'✓' if success_rate >= training_success_rate * 0.9 else '✗'}")
        
        return {
            "inference_score": avg_score,
            "inference_success_rate": success_rate,
            "inference_episode_length": avg_episode_length,
            "training_score": training_score,
            "training_success_rate": training_success_rate,
            "episodes_run": num_episodes
        }
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    args = parse_inference_args()
    
    print("=" * 60)
    print("Option-Critic Best Model Inference")
    print("=" * 60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f"Using device: {device_name}")
    
    # Load best model
    print(f"Loading best model from: {args.checkpoint_dir}")
    model, base_env, model_info = load_best_model_for_inference(args.checkpoint_dir, device)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Run inference
    results = run_inference(model, base_env, model_info, args.num_episodes, device, args.verbose)
    
    if results is not None:
        # Save results
        results_path = Path(args.checkpoint_dir) / "inference_results.json"
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {results_path}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()