import numpy as np
import torch
from typing import Union, Any


def to_tensor(obs: Union[np.ndarray, torch.Tensor, Any]) -> torch.Tensor:
    """
    Convert observation to PyTorch tensor.
    
    Args:
        obs: Observation to convert (can be numpy array, tensor, or other)
        
    Returns:
        PyTorch tensor
    """
    if torch.is_tensor(obs):
        return obs.float()
    
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure tensor is on the specified device.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        Tensor on target device
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def soft_update_parameters(target_network: torch.nn.Module, 
                         source_network: torch.nn.Module, 
                         tau: float) -> None:
    """
    Soft update of target network parameters.
    
    Args:
        target_network: Network to update
        source_network: Network to copy from
        tau: Interpolation parameter (0.0 = no update, 1.0 = hard update)
    """
    for target_param, source_param in zip(target_network.parameters(), 
                                        source_network.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update_parameters(target_network: torch.nn.Module, 
                         source_network: torch.nn.Module) -> None:
    """
    Hard update of target network parameters (copy all parameters).
    
    Args:
        target_network: Network to update
        source_network: Network to copy from
    """
    target_network.load_state_dict(source_network.state_dict())


def calculate_epsilon(step: int, 
                     eps_start: float, 
                     eps_min: float, 
                     eps_decay: int) -> float:
    """
    Calculate epsilon for epsilon-greedy exploration.
    
    Args:
        step: Current step
        eps_start: Starting epsilon value
        eps_min: Minimum epsilon value
        eps_decay: Decay rate
        
    Returns:
        Current epsilon value
    """
    from math import exp
    eps = eps_min + (eps_start - eps_min) * exp(-step / eps_decay)
    return eps


def validate_graph_inputs(graph_x: torch.Tensor,
                         graph_edge_index: torch.Tensor, 
                         graph_edge_attr: torch.Tensor,
                         story_batch: torch.Tensor) -> bool:
    """
    Validate graph input tensors.
    
    Args:
        graph_x: Node features
        graph_edge_index: Edge indices
        graph_edge_attr: Edge attributes
        story_batch: Story batch indices
        
    Returns:
        True if inputs are valid
    """
    try:
        # Check dimensions
        assert graph_x.dim() == 2, f"graph_x should be 2D, got {graph_x.dim()}D"
        assert graph_edge_index.dim() == 2, f"graph_edge_index should be 2D, got {graph_edge_index.dim()}D"
        assert graph_edge_attr.dim() == 2, f"graph_edge_attr should be 2D, got {graph_edge_attr.dim()}D"
        assert story_batch.dim() == 1, f"story_batch should be 1D, got {story_batch.dim()}D"
        
        # Check shapes
        assert graph_edge_index.shape[0] == 2, f"graph_edge_index first dim should be 2, got {graph_edge_index.shape[0]}"
        
        return True
        
    except AssertionError as e:
        print(f"Graph input validation failed: {e}")
        return False