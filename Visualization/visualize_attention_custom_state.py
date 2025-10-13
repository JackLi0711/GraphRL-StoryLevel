"""
GAT Attention Visualization for Custom Structure States

This script allows you to visualize attention weights for a specific structure configuration
by providing custom story_level_sections (e.g., [5, 0, 0, 1]).

Usage:
    python Visualization/visualize_attention_custom_state.py \
        --model_path checkpoints/option_oc/xxx/best_model.pt \
        --checkpoint_dir checkpoints/option_oc/xxx \
        --structure_shape 4story \
        --story_sections 5 0 0 1 3 2 1 0 4 3 2 1 2 1 0 0 \
        --device cuda

Or with JSON format:
    python Visualization/visualize_attention_custom_state.py \
        --model_path checkpoints/option_oc/xxx/best_model.pt \
        --checkpoint_dir checkpoints/option_oc/xxx \
        --structure_shape 4story \
        --story_sections_json "[5,0,0,1,3,2,1,0,4,3,2,1,2,1,0,0]" \
        --device cuda
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import project modules
from Structure.structure import Structure
from RL.option_critic_gnn import OptionCriticGNN
from RL.environment import Environment
from RL.option_critic.utils import get_graph_data
from Validation import normalization as nda_norm

# Import functions from the original script
from visualize_attention import (
    build_edge_mapping,
    extract_gat_attention,
    plot_3d_structure_attention
)


def create_structure_with_sections(structure_shape: str,
                                   story_sections: List[int],
                                   logger: logging.Logger,
                                   device: torch.device,
                                   checkpoint_dir: Path,
                                   args_dict: Dict) -> Structure:
    """
    Create a structure with specified story_level_sections.

    Args:
        structure_shape: Structure shape (e.g., '4story', 'random')
        story_sections: List of section indices for each story member
        logger: Logger instance
        device: Device to use
        checkpoint_dir: Checkpoint directory
        args_dict: Model hyperparameters

    Returns:
        Structure object with specified sections
    """
    from Structure.structure import Structure
    from Structure import check

    nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}

    # Parse structure shape to get geometry
    # First create a temporary environment to get structure geometry
    temp_env = Environment(
        structure_shape=structure_shape,
        add_structure_geometry=args_dict.get('add_structure_geometry', True),
        add_response_features=args_dict.get('add_response_features', True),
        reward_type=args_dict.get('reward_type', 'weight_saving'),
        scwb_driven_design=False,
        do_nonlinear_dynamic_analysis=False,
        check_acceleration=args_dict.get('check_acceleration', False),
        check_displacement=args_dict.get('check_displacement', False),
        nda_simulator=None,
        nda_norm_dict=nda_norm_dict,
        DBE_ground_motion_set=[],
        MCE_ground_motion_set=[],
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        device=device,
    )

    # Get structure geometry
    temp_structure = temp_env.reset()
    x_span_num = temp_structure.x_span_num
    x_span_lens = temp_structure.x_span_lens
    z_span_num = temp_structure.z_span_num
    z_span_lens = temp_structure.z_span_lens
    story_num = temp_structure.story_num
    story_height = temp_structure.story_height

    # Validate story_sections length
    expected_length = len(temp_structure.story_level_sections)
    if len(story_sections) != expected_length:
        raise ValueError(
            f"story_sections length mismatch! Expected {expected_length}, got {len(story_sections)}\n"
            f"Structure: {x_span_num}x{z_span_num}, {story_num} stories\n"
            f"Expected format: [xbeam_s1, ..., xbeam_sN, zbeam_s1, ..., zbeam_sN, "
            f"outcol_s1, ..., outcol_sN, incol_s1, ..., incol_sN]"
        )

    print(f"\nCreating structure with custom sections: {story_sections}")
    print(f"  - Structure geometry: {x_span_num}x{z_span_num}, {story_num} stories")
    print(f"  - X-beams: {story_sections[:story_num]}")
    print(f"  - Z-beams: {story_sections[story_num:story_num*2]}")
    print(f"  - Outer columns: {story_sections[story_num*2:story_num*3]}")
    print(f"  - Inner columns: {story_sections[story_num*3:]}")

    # Create analysis directory for this structure
    analysis_dir = checkpoint_dir / "temp_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Create structure directly with custom sections
    structure = Structure(
        x_span_num=x_span_num,
        x_span_lens=x_span_lens,
        z_span_num=z_span_num,
        z_span_lens=z_span_lens,
        story_num=story_num,
        story_height=story_height,
        story_level_sections=story_sections,  # Pass custom sections here
        add_structure_geometry=args_dict.get('add_structure_geometry', True),
        add_response_features=args_dict.get('add_response_features', True),
        do_nonlinear_dynamic_analysis=False,
        nda_norm_dict=nda_norm_dict,
        analysis_dir=analysis_dir
    )

    # Get structural response
    load_cases, static_responses = check.get_response(structure, analysis_dir)

    # Process response to get features
    _, static_response_features, static_response_rewards = check.process_response(structure, load_cases, static_responses)

    # No dynamic analysis for visualization
    dynamic_response_features = None

    # Initialize graph with response features
    if args_dict.get('add_response_features', True):
        structure.init_graph_GraphRL(static_response_features, dynamic_response_features)
    else:
        structure.init_graph_GraphRL()

    print(f"Structure created successfully!")

    return structure


def visualize_attention_custom_state(model_path: str,
                                     checkpoint_dir: str,
                                     structure_shape: str,
                                     story_sections: List[int],
                                     device: str = 'cuda',
                                     output_suffix: str = '') -> None:
    """
    Visualize attention for a custom structure state.

    Args:
        model_path: Path to trained model checkpoint (.pt file)
        checkpoint_dir: Directory to save visualization outputs
        structure_shape: Structure shape (e.g., '4story', 'random')
        story_sections: List of section indices
        device: Device to run on ('cuda' or 'cpu')
        output_suffix: Optional suffix for output filenames
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("GAT Attention Visualization - Custom State")
    print("="*60)
    print(f"Using device: {device}")

    # Create output directory
    sections_str = '_'.join(map(str, story_sections))
    if output_suffix:
        output_dir = Path(checkpoint_dir) / f"attention_custom_{output_suffix}"
    else:
        output_dir = Path(checkpoint_dir) / f"attention_custom_sections"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model checkpoint
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Get hyperparameters
    if 'hyperparameters' in checkpoint:
        args_dict = checkpoint['hyperparameters']
    else:
        raise ValueError("Checkpoint does not contain hyperparameters!")

    # Create logger
    logger = logging.getLogger("AttentionViz_Custom")
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    # Create structure with custom sections
    print("\nCreating structure with custom sections...")
    print(f"Structure shape: {structure_shape}")
    print(f"Story sections: {story_sections}")

    structure = create_structure_with_sections(
        structure_shape=structure_shape,
        story_sections=story_sections,
        logger=logger,
        device=device,
        checkpoint_dir=Path(checkpoint_dir),
        args_dict=args_dict
    )

    print(f"\nStructure: {structure.x_span_num}x{structure.z_span_num}, {structure.story_num} stories")
    print(f"Total members: {structure.member_number}")

    # Create model
    print("\nCreating model...")
    graph = structure.graph
    node_feature_dim = graph.x.shape[1]
    edge_feature_dim = graph.edge_attr.shape[1]

    model = OptionCriticGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args_dict.get('hidden_dim', 128),
        member_state_dim=args_dict.get('hidden_dim', 128),
        num_layers=args_dict.get('num_layers', 3),
        num_actions=100,
        num_options=args_dict.get('num_options', 4),
        temperature=args_dict.get('temperature', 1.0),
        device=device,
        testing=True,
        debug_logging=False,
    )

    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Extract attention weights
    print("\nExtracting attention weights...")
    attention_per_head = extract_gat_attention(model, structure, device)

    # Generate visualizations
    print("\nGenerating visualizations...")
    stats = {
        'model_path': str(model_path),
        'structure_shape': structure_shape,
        'story_sections': story_sections,
        'num_heads': len(attention_per_head),
        'num_members': structure.member_number,
        'heads': []
    }

    for head_idx, member_attention in enumerate(attention_per_head):
        # Save visualization
        if output_suffix:
            save_path = output_dir / f"attention_head_{head_idx}_{output_suffix}.jpg"
        else:
            save_path = output_dir / f"attention_head_{head_idx}.jpg"

        plot_3d_structure_attention(
            structure=structure,
            member_attention=member_attention,
            head_idx=head_idx,
            save_path=save_path,
            show_nodes=True
        )

        # Save statistics
        att_values = list(member_attention.values())
        head_stats = {
            'head_index': head_idx,
            'num_members': len(member_attention),
            'attention_min': float(np.min(att_values)),
            'attention_max': float(np.max(att_values)),
            'attention_mean': float(np.mean(att_values)),
            'attention_std': float(np.std(att_values)),
        }
        stats['heads'].append(head_stats)

    # Save statistics
    if output_suffix:
        stats_path = output_dir / f"attention_stats_{output_suffix}.json"
    else:
        stats_path = output_dir / "attention_stats.json"

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved statistics: {stats_path}")

    print("\n" + "="*60)
    print("Attention visualization completed successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize GAT attention for custom structure state',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using space-separated section indices
  python visualize_attention_custom_state.py \\
      --model_path checkpoints/option_oc/OB_002/best_model.pt \\
      --checkpoint_dir checkpoints/option_oc/OB_002 \\
      --structure_shape random \\
      --story_sections 5 0 0 1 3 2 1 0 4 3 2 1 2 1 0 0 \\
      --device cuda

  # Using JSON format
  python visualize_attention_custom_state.py \\
      --model_path checkpoints/option_oc/OB_002/best_model.pt \\
      --checkpoint_dir checkpoints/option_oc/OB_002 \\
      --structure_shape random \\
      --story_sections_json "[5,0,0,1,3,2,1,0,4,3,2,1,2,1,0,0]" \\
      --device cuda

  # With custom output suffix
  python visualize_attention_custom_state.py \\
      --model_path checkpoints/option_oc/OB_002/best_model.pt \\
      --checkpoint_dir checkpoints/option_oc/OB_002 \\
      --structure_shape random \\
      --story_sections 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \\
      --output_suffix "initial_thick" \\
      --device cuda
        """
    )

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory to save visualization outputs')
    parser.add_argument('--structure_shape', type=str, required=True,
                       help='Structure shape (e.g., 4story, 7story, random)')
    parser.add_argument('--story_sections', type=int, nargs='+',
                       help='Space-separated section indices (e.g., 5 0 0 1 3 2 ...)')
    parser.add_argument('--story_sections_json', type=str,
                       help='JSON array of section indices (e.g., "[5,0,0,1,3,2,...]")')
    parser.add_argument('--output_suffix', type=str, default='',
                       help='Optional suffix for output files (default: none)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Parse story_sections
    if args.story_sections_json:
        try:
            story_sections = json.loads(args.story_sections_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in --story_sections_json: {e}")
            exit(1)
    elif args.story_sections:
        story_sections = args.story_sections
    else:
        print("Error: Must provide either --story_sections or --story_sections_json")
        print("Use --help for examples")
        exit(1)

    # Validate story_sections
    if not isinstance(story_sections, list) or len(story_sections) == 0:
        print("Error: story_sections must be a non-empty list")
        exit(1)

    print(f"\nParsed story_sections: {story_sections}")
    print(f"Number of sections: {len(story_sections)}")

    # Run visualization
    visualize_attention_custom_state(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        structure_shape=args.structure_shape,
        story_sections=story_sections,
        device=args.device,
        output_suffix=args.output_suffix
    )
