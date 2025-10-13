"""
GAT Attention Visualization for Option-Critic Model

This module extracts and visualizes attention weights from the Graph Attention Network (GAT)
used in the Option-Critic model. It generates 3D structure visualizations with attention
weights shown as grayscale colors (black = high attention, white = low attention).

Usage:
    python Visualization/visualize_attention.py \
        --model_path checkpoints/option_oc/xxx/best_model.pt \
        --checkpoint_dir checkpoints/option_oc/xxx \
        --structure_shape 4story \
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


def build_edge_mapping(structure: Structure, edge_index: torch.Tensor) -> Dict[Tuple[int, int], str]:
    """
    Build mapping from graph edge_index to Structure member names.

    The graph edge_index contains bidirectional edges (i→j and j→i), while
    Structure has unique member identifiers. This function creates a mapping
    that can handle both edge directions.

    Args:
        structure: Structure object containing member information
        edge_index: [2, num_edges] tensor with node pairs

    Returns:
        edge_to_member: Dict mapping (node_i, node_j) tuple to member_name
    """
    edge_to_member = {}

    # Iterate through all members in the structure
    for member_name, member_info in structure.member_to_nodeIndex_dict.items():
        node1_idx = member_info[0]  # First node index
        node2_idx = member_info[1]  # Second node index

        # Store both directions (bidirectional)
        edge_to_member[(node1_idx, node2_idx)] = member_name
        edge_to_member[(node2_idx, node1_idx)] = member_name

    return edge_to_member


def extract_gat_attention(model: OptionCriticGNN,
                         structure: Structure,
                         device: torch.device) -> List[Dict[str, float]]:
    """
    Extract GAT attention weights from the last layer of the trained model.

    Args:
        model: Trained OptionCriticGNN model
        structure: Structure object for the building
        device: torch device

    Returns:
        attention_per_head: List of dicts, one per attention head
            Each dict maps member_name → attention_value (float)
    """
    model.eval()

    # Prepare graph data using the utility function
    graph_x, graph_edge_index, graph_edge_attr, story_batch, structure_story_ptr = get_graph_data(structure, device)

    # Extract attention weights
    with torch.no_grad():
        state, (att_edge_index, att_weights) = model.state_gnn(
            x=graph_x,
            edge_index=graph_edge_index,
            edge_attr=graph_edge_attr,
            batch=None,
            story_batch=story_batch,
            structure_story_ptr=structure_story_ptr,
            return_attention_weights=True
        )

    # att_edge_index: [2, num_edges] - edge indices for attention
    # att_weights: [num_edges, num_heads] - attention values per head

    att_edge_index = att_edge_index.cpu().numpy()
    att_weights = att_weights.cpu().numpy()

    num_heads = att_weights.shape[1]
    print(f"Extracted attention: {att_weights.shape[0]} edges, {num_heads} heads")

    # Build edge mapping (use att_edge_index as it matches the attention structure)
    edge_to_member = build_edge_mapping(structure, att_edge_index)

    # Process attention for each head
    attention_per_head = []

    for head_idx in range(num_heads):
        member_attention = {}
        member_count = {}

        # Aggregate attention for each member (handle bidirectional edges)
        for edge_idx in range(att_edge_index.shape[1]):
            node_i = att_edge_index[0, edge_idx]
            node_j = att_edge_index[1, edge_idx]
            attention_value = att_weights[edge_idx, head_idx]

            # Find corresponding member
            edge_key = (node_i, node_j)
            if edge_key in edge_to_member:
                member_name = edge_to_member[edge_key]

                # Accumulate attention (for averaging bidirectional edges)
                if member_name not in member_attention:
                    member_attention[member_name] = 0.0
                    member_count[member_name] = 0

                member_attention[member_name] += attention_value
                member_count[member_name] += 1

        # Average attention for bidirectional edges
        for member_name in member_attention:
            member_attention[member_name] /= member_count[member_name]

        attention_per_head.append(member_attention)

        # Print statistics
        att_values = list(member_attention.values())
        print(f"Head {head_idx}: {len(member_attention)} members, "
              f"attention range [{np.min(att_values):.4f}, {np.max(att_values):.4f}]")

    return attention_per_head


def plot_3d_structure_attention(structure: Structure,
                                member_attention: Dict[str, float],
                                head_idx: int,
                                save_path: Path,
                                show_nodes: bool = True) -> None:
    """
    Plot 3D structure with attention weights visualized as grayscale colors.

    Visualization design:
    - Nodes: small gray dots
    - Members (edges): lines colored by attention
        - High attention → black (dark gray)
        - Low attention → white (light gray)
        - Line width varies with attention (1.0 to 4.0)
    - Colorbar: shows attention mapping
    - Static view: elevation=20°, azimuth=45°
    - Clean style: no axes, grid, or background

    Args:
        structure: Structure object
        member_attention: Dict mapping member_name → attention_value
        head_idx: Attention head index (for title)
        save_path: Path to save JPG file
        show_nodes: Whether to show nodes as dots
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set viewing angle
    ax.view_init(elev=20, azim=60)

    # Remove axes, grid, and background
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Colormap: high attention → black, low attention → white
    cmap = plt.cm.get_cmap('gray_r')  # Reversed gray

    # Normalize attention values
    att_values = list(member_attention.values())
    if len(att_values) == 0:
        print(f"Warning: No attention values for head {head_idx}")
        return

    min_att = np.min(att_values)
    max_att = np.max(att_values)
    norm = plt.Normalize(vmin=min_att, vmax=max_att)

    # Plot nodes
    if show_nodes:
        node_coords = []
        for node_name, coord in structure.node_coord_dict.items():
            # Swap Y and Z axes: (x, y, z) → (x, z, y)
            node_coords.append([coord[0], coord[2], coord[1]])

        if len(node_coords) > 0:
            node_coords = np.array(node_coords)
            ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
                      c='gray', s=10, alpha=0.3, depthshade=False)

    # Plot members with attention-based coloring
    for member_name, attention in member_attention.items():
        # Get member endpoints
        member_info = structure.member_to_nodeIndex_dict.get(member_name)
        if member_info is None:
            continue

        node1_idx = member_info[0]
        node2_idx = member_info[1]

        # Get node coordinates
        node1_name = f"N{node1_idx + 1}"
        node2_name = f"N{node2_idx + 1}"

        coord1 = structure.node_coord_dict.get(node1_name)
        coord2 = structure.node_coord_dict.get(node2_name)

        if coord1 is None or coord2 is None:
            continue

        # Swap Y and Z axes: (x, y, z) → (x, z, y)
        coord1_swapped = [coord1[0], coord1[2], coord1[1]]
        coord2_swapped = [coord2[0], coord2[2], coord2[1]]

        # Calculate color and line width
        color = cmap(norm(attention))
        linewidth = 1.0 + 3.0 * norm(attention)  # Range: 1.0 to 4.0

        # Plot the member
        ax.plot([coord1_swapped[0], coord2_swapped[0]],
                [coord1_swapped[1], coord2_swapped[1]],
                [coord1_swapped[2], coord2_swapped[2]],
                color=color, linewidth=linewidth, alpha=0.8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Attention Weight', shrink=0.6, pad=0.1)
    cbar.ax.tick_params(labelsize=10)

    # Add title
    ax.set_title(f'GAT Attention - Head {head_idx}\n'
                 f'(Last Layer, Black=High Attention)',
                 fontsize=14, pad=20)

    # Save as JPG with white background
    plt.savefig(save_path, format='jpg', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_model_attention(model_path: str,
                              checkpoint_dir: str,
                              structure_config: Dict,
                              device: str = 'cuda') -> None:
    """
    Complete attention visualization pipeline.

    This function:
    1. Loads the trained model
    2. Creates a structure based on config
    3. Extracts attention weights from GAT last layer
    4. Generates one visualization per attention head
    5. Saves statistics to JSON

    Args:
        model_path: Path to trained model checkpoint (.pt file)
        checkpoint_dir: Directory to save visualization outputs
        structure_config: Dict with structure configuration
            Must contain 'structure_shape' key (e.g., '4story')
        device: Device to run on ('cuda' or 'cpu')

    Outputs:
        checkpoint_dir/attention_visualization/
        ├── attention_head_0.jpg
        ├── attention_head_1.jpg
        ├── attention_head_2.jpg
        ├── attention_head_3.jpg
        └── attention_stats.json
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(checkpoint_dir) / "attention_visualization"
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

    # Create environment to get structure
    print("\nCreating structure...")
    structure_shape = structure_config.get('structure_shape', args_dict.get('structure_shape', '4story'))

    # Create a simple logger for environment
    logger = logging.getLogger("AttentionViz")
    logger.setLevel(logging.WARNING)  # Only show warnings and errors to reduce noise
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    # Parse structure shape
    nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}

    env = Environment(
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
        checkpoint_dir=Path(checkpoint_dir),
        logger=logger,
        device=device,
    )

    structure = env.reset()
    print(f"Structure: {structure.x_span_num}x{structure.z_span_num}, {structure.story_num} stories")
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
        num_actions=100,  # Dummy value, not used in new architecture
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
        'num_heads': len(attention_per_head),
        'num_members': structure.member_number,
        'heads': []
    }

    for head_idx, member_attention in enumerate(attention_per_head):
        # Save visualization
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
        description='Visualize GAT attention weights from Option-Critic model'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory to save visualization outputs')
    parser.add_argument('--structure_shape', type=str, default='4story',
                       help='Structure shape (e.g., 4story, 7story)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    structure_config = {
        'structure_shape': args.structure_shape
    }

    visualize_model_attention(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        structure_config=structure_config,
        device=args.device
    )
