import time
import torch
import typing
import numpy as np
from torch import nn
from .model import StateGNN, Q_Network, GraphEmbedding


def select_greedy_actions(next_states: torch.Tensor,
                          action_q_network: Q_Network,
                          infeasible_actions: np.ndarray[bool],
                          structure_story_ptr: typing.List[int]) -> torch.Tensor:
    """Select the greedy action for the current state given Q-network."""
    action_q_values = action_q_network.forward(next_states).squeeze()
    action_q_values[infeasible_actions] = -1.0e20
    actions = [torch.argmax(action_q_values[structure_story_ptr[i]:structure_story_ptr[i+1]]) for i in range(len(structure_story_ptr)-1)]
    actions = torch.tensor(actions) + structure_story_ptr[:-1]
    return actions


def evaluate_selected_actions(next_states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              value_q_network: Q_Network) -> torch.Tensor:
    """Compute the Q-values by evaluating the given actions, current states, and Q-network."""
    next_q_values = value_q_network.forward(next_states).squeeze()[actions]
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
                             infeasible_actions: np.ndarray[bool],
                             structure_story_ptr: typing.List[int],
                             gamma: float,
                             action_q_network: Q_Network,
                             value_q_network: Q_Network) -> torch.Tensor:
    """
    Double Q-Learning uses action_q_network to select actions and value_q_neetwork to evaluate the selected actions.
    - online_q_network: action_q_network --> choose action
    - target_q_network: value_q_network  --> evaluate value
    """
    actions = select_greedy_actions(next_states, action_q_network, infeasible_actions, structure_story_ptr)
    q_values = evaluate_selected_actions(next_states, actions, rewards, dones, gamma, value_q_network)
    return q_values


def print_Q(online_q, target_q, rewards, diff, logger):
    n = 5
    print_idxs = [i for i in range(n)] + [i for i in range(len(diff)-n, len(diff))]
    logger.info("")
    for i in print_idxs:
        logger.info(f"experience: {i+1:3d}, Q(s,a) = {online_q[i]:.5f}, r + maxQ(s',a) = {target_q[i]:.5f}, reward = {rewards[i]:.5f}, diff = {diff[i]:.5f}")
    logger.info("")
    

def double_q_learning_error(states: torch.Tensor,
                            actions: torch.Tensor,
                            rewards: torch.Tensor,
                            next_states: torch.Tensor,
                            dones: torch.Tensor,
                            infeasible_actions: np.ndarray[bool],
                            structure_story_ptr: typing.List[int],
                            gamma: float,
                            action_q_network: Q_Network,
                            value_q_network: Q_Network,
                            logger) -> torch.Tensor:
    """Compute the TD-Error for prioritized experience replay."""
    t_start = time.time()
    expected_q_values = double_q_learning_update(next_states, rewards, dones, infeasible_actions, structure_story_ptr, gamma, action_q_network, value_q_network).detach()
    q_values = action_q_network.forward(states).squeeze()[actions] 
    delta = expected_q_values - q_values
    print(f"\t{expected_q_values.requires_grad}, {q_values.requires_grad}, {delta.requires_grad}")
    #print_Q(q_values, expected_q_values, rewards, delta, logger)
    t_end = time.time()
    print(f"\tused time for q_algorithm.double_q_learning_error: {t_end - t_start:.3f} sec")
    return delta


def calc_delta(states: torch.Tensor,
               actions: torch.Tensor,
               rewards: torch.Tensor,
               next_states: torch.Tensor,
               dones: torch.Tensor,
               infeasible_actions: np.ndarray[bool],
               structure_story_ptr: typing.List[int],
               gamma: float,
               action_q_network: Q_Network,
               value_q_network: Q_Network, 
               device: torch.device,
               logger) -> torch.Tensor:
    """Compute the Japan's delta for Q-Learning."""
    # action Q-network
    Q_value = action_q_network.forward(states).squeeze()
    q_values = Q_value[actions]

    # value Q-network
    tmp = value_q_network.forward(next_states).squeeze().detach()
    tmp[infeasible_actions] = -1.0e20
    next_q_values = torch.tensor([tmp[structure_story_ptr[i]:structure_story_ptr[i+1]].max() for i in range(len(structure_story_ptr)-1)], dtype=torch.float32, device=device, requires_grad=False)
    expected_q_values = rewards + (gamma * next_q_values) * (1 - dones)

    # delta
    delta = expected_q_values - q_values

    return delta




### Japan's code ###


def calc_loss(states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor,
              infeasible_actions: np.ndarray[bool],
              structure_story_ptr: typing.List[int],
              gamma: float,
              online_model: GraphEmbedding,
              target_model: GraphEmbedding,
              logger) -> torch.Tensor:
    """Compute the loss for Q-Learning."""
    # action Q-network
    Q_value = online_model.get_Q(states)
    Q_current = Q_value[actions]

    # value Q-network
    tmp = target_model.get_Q(next_states).detach()
    tmp[infeasible_actions] = -1.0e20
    Q_max_next = torch.tensor([tmp[structure_story_ptr[i]:structure_story_ptr[i+1]].max() for i in range(len(structure_story_ptr)-1)], dtype=torch.float32, device=target_model.device, requires_grad=False)
    Q_target = rewards + (gamma * Q_max_next) * (1 - dones)

    # loss
    loss = nn.MSELoss()(Q_current, Q_target)

    return loss
