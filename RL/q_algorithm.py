import torch
import typing
from torch import nn


def select_greedy_actions(next_states: torch.Tensor,
                          action_q_network: nn.Module,
                          structure_story_ptr: typing.List[int]) -> torch.Tensor:
    """Select the greedy action for the current state given Q-network."""
    action_q_values = action_q_network(next_states).squeeze()
    actions = [torch.argmax(action_q_values[structure_story_ptr[i]:structure_story_ptr[i+1]]) for i in range(len(structure_story_ptr)-1)]
    actions = torch.tensor(actions) + structure_story_ptr[:-1]
    return actions


def evaluate_selected_actions(next_states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              value_q_network: nn.Module) -> torch.Tensor:
    """Compute the Q-values by evaluating the given actions, current states, and Q-network."""
    next_q_values = value_q_network(next_states).squeeze()[actions]
    q_values = rewards + (gamma * next_q_values * (1 - dones))
    return q_values


def q_learning_update(next_states: torch.Tensor,
                      rewards: torch.Tensor,
                      dones: torch.Tensor,
                      member_ptr: typing.List[int],
                      gamma: float,
                      q_network: nn.Module) -> torch.Tensor:
    """Q-Learning update with explicitly decoupled action selection and evaluation steps."""
    actions = select_greedy_actions(next_states, q_network, member_ptr)
    q_values = evaluate_selected_actions(next_states, actions, rewards, dones, gamma, q_network)
    return q_values


def double_q_learning_update(next_states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             structure_story_ptr: typing.List[int],
                             gamma: float,
                             action_q_network: nn.Module,
                             value_q_network: nn.Module) -> torch.Tensor:
    """
    Double Q-Learning uses action_q_network to select actions and value_q_neetwork to evaluate the selected actions.

    online_q_network: action_q_network --> choose action
    target_q_network: value_q_network  --> evaluate value
    """
    actions = select_greedy_actions(next_states, action_q_network, structure_story_ptr)
    q_values = evaluate_selected_actions(next_states, actions, rewards, dones, gamma, value_q_network)
    return q_values
    

def double_q_learning_error(states: torch.Tensor,
                            actions: torch.Tensor,
                            rewards: torch.Tensor,
                            next_states: torch.Tensor,
                            dones: torch.Tensor,
                            structure_story_ptr: typing.List[int],
                            gamma: float,
                            action_q_network: nn.Module,
                            value_q_network: nn.Module,
                            logger) -> torch.Tensor:
    """Compute the TD-Error for prioritized experience replay."""
    expected_q_values = double_q_learning_update(next_states, rewards, dones, structure_story_ptr, gamma, action_q_network, value_q_network)
    q_values = action_q_network(states).squeeze()[actions] 
    delta = expected_q_values - q_values
    # print_Q(q_values, expected_q_values, rewards, delta, logger)
    return delta




def print_Q(online_q, target_q, rewards, diff, logger):
    n = 5
    print_idxs = [i for i in range(n)] + [i for i in range(len(diff)-n, len(diff))]
    logger.info("")
    for i in print_idxs:
        logger.info(f"experience: {i+1:3d}, Q(s,a) = {online_q[i]:.5f}, r + maxQ(s',a) = {target_q[i]:.5f}, reward = {rewards[i]:.5f}, diff = {diff[i]:.5f}")
    logger.info("")
    
