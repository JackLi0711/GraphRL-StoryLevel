import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import os
import sys
import time
import json
import logging
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Dict, Callable
sys.path.append("../Structure")

import RL.model as model
import RL.buffer as buffer
import RL.q_algorithm as q_algorithm
from RL.record import Record
from RL.environment import Environment
from RL.model import synchronize_q_networks, soft_update_q_network_parameters


# abstract class
class Agent:
    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an actio given the current state of the environment."""
        raise NotImplementedError

    def learn(self, experiences: List[buffer.Experience]) -> None:
        """Update the agent's Q Network based on a collection of recent experiences."""
        raise NotImplementedError

    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplementedError
        

class DeepQAgent(Agent):
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 batch_size: int,
                 lr: float,
                 buffer_size: int,
                 epsilon_decay_schedule: Callable[[int], float],
                 synchronize_steps: float,
                 soft_update_alpha: float,
                 gamma: float,
                 update_frequency: int,
                 add_experience_frequency: int,
                 test_frequency: int = 5,
                 restrict_action: bool = False,
                 seed: int = 731,
                 logger: logging.Logger = None,
                 pretrained_ckpt_dir = None,
                 device = "cpu") -> None:
        """
        Initialize a DeepQAgent.
        
        Parameters:
        -----------
        state_size (int): the size of the state space.
        action_size (int): the size of the action space.
        batch_size (int): number of experience tuples in each mini-batch.
        buffer_size (int): maximum number of experience tuples stored in the replay buffer.
        
        edge_embedding_dimension(int): number of features on the edge embedding (hyperparameter).
        hidden_dimension (int): number of units in the hidden layers (hyperparamter).
        
        optimizer_fn (callable): function that takes Q-network parameters and returns an optimizer.
        epsilon_decay_schdule (callable): function that takes episode number and returns epsilon.

        soft_update_alpha (float): rate at which the target q-network parameters are updated.
        gamma (float): Controls how much that agent discounts future rewards (0 < gamma <= 1).
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        
        double_dqn (bool): whether to use vanilla DQN algorithm or use the Double DQN algorithm.
        seed (int): random seed

        """
        self.restrict_action = restrict_action
        self.logger = logger
        self.device = device
        
        # set seeds for reproducbility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
        
        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state,
            "logger": self.logger
        }
        self._buffer = buffer.ExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._gamma = gamma
        
        # initialize GNN
        model_kwargs = {"node_feature_dim": node_feature_dim, "edge_feature_dim": edge_feature_dim, "hidden_dim": hidden_dim,
                        "member_state_dim": hidden_dim, "num_layers": num_layers}
        self.gnn = model.StateGNN(**model_kwargs).to(self.device)
        
        # initialize Q-Networks
        self._update_frequency = update_frequency
        self._synchronize_steps = synchronize_steps
        self._soft_update_alpha = soft_update_alpha
        self._add_experience_frequency = add_experience_frequency
        q_net_kwargs = {"member_state_dim": hidden_dim * 2, "hidden_dim": hidden_dim, "q_value_dim": 1}
        self.online_q_network = model.Q_Network(**q_net_kwargs).to(self.device)
        self.target_q_network = model.Q_Network(**q_net_kwargs).to(self.device)
        synchronize_q_networks(self.target_q_network, self.online_q_network)

        # initialize optimizer
        params = list(self.gnn.parameters()) + list(self.online_q_network.parameters())
        self._optimizer = optim.RMSprop(params, lr=lr)  # Japan: RMSprop / Tony: Adam
        
        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0
        self._backprop_count = 0
        self._test_frequency = test_frequency

        # initialize pretrained model
        if pretrained_ckpt_dir:
            _load_model(self, pretrained_ckpt_dir, self.logger)

        
    # policies
    def _uniform_random_policy(self, state: torch.Tensor, dont_select_story_indexes: List[int]) -> int:
        """Choose an action uniformly at random."""
        action = self._random_state.randint(state.shape[0])
        while action in dont_select_story_indexes:
            action = self._random_state.randint(state.shape[0])
        return action
    
    
    def _greedy_policy(self, state: torch.Tensor, dont_select_story_indexes: List[int]) -> int:
        """Choose an action that maximizes the action_values given the current state."""
        with torch.no_grad():
            q_values = self.online_q_network(state).index_fill(dim=0, index=torch.tensor(dont_select_story_indexes).to(torch.int64).to(self.device), value=-10000)
            q_val = q_values.max().cpu().item()
            action = q_values.argmax().cpu().item()
            self.logger.info(f"greed_policy's selection: {action}, Q value: {q_val}")
        return action, q_val
    
    
    def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float, dont_select_story_indexes: List[int]) -> int:
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        q_val = 0
        if self._random_state.random() < epsilon:
            action = self._uniform_random_policy(state, dont_select_story_indexes)
        else:
            action, q_val = self._greedy_policy(state, dont_select_story_indexes)
        return action, q_val
    



    def _has_sufficient_experience(self) -> bool:
        """True if agent has enough experience to train on a batch of samples; False otherwies."""
        return len(self._buffer) >= self._buffer._batch_size
    
    
    def choose_action(self, 
                      state: torch.Tensor, 
                      already_minimum_section_story_indexes: List[int],
                      dont_select_story_member_indexes: List[int]=[], 
                      greedy: bool=False) -> Tuple[int, float]:
        """
        Return the action for given state as per current policy.
        action: update_story_index
        """
        q_val = 0
        dont_select_story_indexes = list(set(already_minimum_section_story_indexes + dont_select_story_member_indexes))
        
        # if testing time, always select the greedy action
        if greedy:
            epsilon = 0
            action, q_val = self._epsilon_greedy_policy(state, epsilon, dont_select_story_indexes)
        
        # choose uniform at random if agent has insufficient experience
        elif not self._has_sufficient_experience():
            action = self._uniform_random_policy(state, dont_select_story_indexes)
        else:
            epsilon = self._epsilon_decay_schedule(self._number_episodes)
            action, q_val = self._epsilon_greedy_policy(state, epsilon, dont_select_story_indexes)
            
        return action, q_val
    
    
    def _learn(self, experiences: np.ndarray) -> None:
        """Update the agent's Q network based on a collection of prioritized experiences."""
        graphs, actions, rewards, next_graphs, dones, auxs = [vs for vs in zip(*experiences)]

        # print("type:", type(auxs))
        # print("len:", len(auxs))
        # print("sub type:", type(auxs[0]))
        # print("auxs:", auxs)
        
        # ptr
        member_numbers = [int(graph.edge_attr.shape[0]/2) for graph in graphs]  
        member_ptr = torch.tensor([sum(member_numbers[:i]) for i in range(len(member_numbers)+1)])
        
        member_batch = []
        for i, member_number in enumerate(member_numbers):
            member_batch += [i] * member_number
        member_batch = torch.tensor(member_batch).to(self.device)

        # story level pooling preparation
        structure_story_ptr = []    # if the first and second graph have 15, 12 story section, it will be [0, 15]
        story_batch = torch.zeros(member_batch.shape[0])
        story_count = 0
        for i, graph in enumerate(graphs):
            structure_story_ptr.append(story_count)
            for story_members in (auxs[i]["story_xdir_beam_member"] + auxs[i]["story_zdir_beam_member"] + auxs[i]["story_outer_column_member"] + auxs[i]["story_inner_column_member"]):
                story_batch[story_members + member_ptr[i]] = story_count
                story_count += 1
        structure_story_ptr.append(story_count)
        structure_story_ptr = torch.tensor(structure_story_ptr)
        story_batch = story_batch.to(self.device).to(torch.int64)

        # get states, next_states from graphs and next_graphs
        #print("graphs in loader:", len(graphs))
        loader = DataLoader(list(graphs), batch_size=len(graphs))
        loader_next = DataLoader(list(next_graphs), batch_size=len(next_graphs))
        
        graphs_batch = next(iter(loader)).to(self.device)
        next_graphs_batch = next(iter(loader_next)).to(self.device)

        states = self.gnn(graphs_batch.x, graphs_batch.edge_index, graphs_batch.edge_attr, member_batch, story_batch, structure_story_ptr)
        next_states = self.gnn(next_graphs_batch.x, next_graphs_batch.edge_index, next_graphs_batch.edge_attr, member_batch, story_batch, structure_story_ptr)

        # convert batch-values to tensors
        actions = torch.tensor(actions)
        actions = (actions + structure_story_ptr[:len(actions)]).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)             # [batch_size]
        dones = torch.tensor(dones).to(torch.long).to(self.device)  # [batch_size]
                
        # compute temporal difference
        deltas = q_algorithm.double_q_learning_error(states,
                                                     actions,
                                                     rewards,
                                                     next_states,
                                                     dones,
                                                     structure_story_ptr,
                                                     self._gamma,
                                                     self.online_q_network,
                                                     self.target_q_network,
                                                     self.logger)

        # compute the mean squared loss
        loss = torch.mean(deltas ** 2)
        self.logger.critical(f"loss: {loss.item()}")

        # updates the parameters of the online network
        self._optimizer.zero_grad()
        loss.backward()  # retain_graph=True
        self._optimizer.step()
        
        # synchronize online and target network
        if self._synchronize_steps:
            if self._backprop_count % self._synchronize_steps == 0:
                self.logger.info("Synchronizing online q network to target network")
                synchronize_q_networks(self.target_q_network, self.online_q_network)
            self._backprop_count += 1
        else:
            soft_update_q_network_parameters(self.target_q_network, self.online_q_network, self._soft_update_alpha)
        
        return loss
        

    def step(self,
             graph: Data,
             action: int,
             reward: float,
             next_graph: Data,
             done: bool,
             aux: Dict,
             learn_losses: List[float]) -> None:
        """
        Updates the agent's state based on feedback received from the environment.
        
        Parameters:
        -----------
        state (np.array): the previous state of the environment.
        action (int): the action taken by the agent in the previous state.
        reward (float): the reward received from the environment.
        next_state (np.array): the resulting state of the environment following the action.
        done (bool): True is the training episode is finised; false otherwise.

        """
        if self._number_timesteps % self._add_experience_frequency == 0:
            experience = buffer.Experience(graph.to("cpu"),
                                           action, 
                                           reward, 
                                           next_graph.to("cpu"), 
                                           done,
                                           aux)
            self._buffer.append(experience)
        
        if done:
            self._number_episodes += 1
            self._number_timesteps = 0
            
        else:
            self._number_timesteps += 1
            # every so often the agent should learn from experiences
            if self._number_timesteps % self._update_frequency == 0 and self._has_sufficient_experience():
                experiences = self._buffer.sample()
                loss = self._learn(experiences)    
                learn_losses.append(loss.detach().cpu().numpy().item())
    
        


def _train_an_episode(agent: DeepQAgent, 
                      env: Environment, 
                      rec: Record,
                      logger: logging.Logger) -> float:
    """Train the agent until the agent meets the terminal state."""
    structure = env.reset()  # generate a new random graph
    rec.record_in_beginning(structure, testing=False)
    graph = structure.graph.clone()
    score = 0
    done = False
    q = 0
    while not done:
        original_structure = deepcopy(structure)
        # select and perform an action
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else []
        action, q_val = agent.choose_action(state, 
                                            structure.already_minimum_section_story_indexes,
                                            dont_select_story_member_indexes)
        print(f"\n-----episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, story_level_sections: {structure.story_level_sections}, action: {action:3d}")
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # record
        score += reward
        logger.info(f"episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, action: {action:3d}, reward: {reward:4f},  acculmulate_score: {score:.4f} [ORIGINAL]")
        
        if env.saved_material_record_SCWB[-1] != 0:
            actions_SCWB = '_'.join([str(a) for a in env.update_actions_record_SCWB[-1]])
            saved_material_SCWB = env.saved_material_record_SCWB[-1]
            cumulative_saved_material_SCWB = sum(env.saved_material_record_SCWB)
            logger.info(f"episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, action: {actions_SCWB}, reward_volume: {saved_material_SCWB:4f},  acculmulate_score_volume: {cumulative_saved_material_SCWB:.4f} [SCWB]")
        
        if q == 0 and q_val > 0: q = q_val

        # get next state and save experience
        next_graph = structure.graph.clone()
        agent.step(graph, action, reward, next_graph, done, structure.aux, rec.learn_losses)
        graph = next_graph.clone()
    
    rec.Q_values.append(q)

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    final_story_level_sections = final_structure.story_level_sections
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{final_story_level_sections = }")
    logger.info(f"Episode: {agent._number_episodes:4d}, fail name: {fail_name}, fail reason: {fail_reason}")
    
    rec.record_in_end(final_structure, env, testing=False)

    score = sum(env.reward_record)
    return score


def _testing(agent: DeepQAgent, 
             env: Environment, 
             rec: Record,
             logger: logging.Logger) -> float:
    """Test agent's performance with prescribed condition and greedy policy."""
    structure = env.reset(testing=True)  # generate a fix-shaped structure
    rec.record_in_beginning(structure, testing=True)
    graph = structure.graph.clone()
    score = 0
    timestep = 0
    done = False
    while not done:
        original_structure = deepcopy(structure)
        # select and perform an action
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)

        dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else []
        action, _ = agent.choose_action(state, 
                                        structure.already_minimum_section_story_indexes,
                                        dont_select_story_member_indexes, 
                                        greedy=True)
        print(f"\n*****Testing Episode, story_level_sections: {structure.story_level_sections}, action: {action:3d}")
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # get next state
        graph = structure.graph.clone()

        # record
        score += reward
        timestep += 1
        logger.info(f"*****Testing Episode, timestep: {timestep:3d}, action: {action:3d}, reward: {reward:4f},  acculmulate_score: {score:.4f} [ORIGINAL]")

        if env.saved_material_record_SCWB[-1] != 0:
            actions_SCWB = '_'.join([str(a) for a in env.update_actions_record_SCWB[-1]])
            saved_material_SCWB = env.saved_material_record_SCWB[-1]
            cumulative_saved_material_SCWB = sum(env.saved_material_record_SCWB)
            logger.info(f"*****Testing Episode, timestep: {timestep:3d}, action: {actions_SCWB}, reward_volume: {saved_material_SCWB:4f},  acculmulate_score_volume: {cumulative_saved_material_SCWB:.4f} [SCWB]")
    
    final_structure = structure if fail_reason == "minimum_section" else original_structure
    test_final_story_level_sections = final_structure.story_level_sections
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{test_final_story_level_sections = }")
    logger.info(f"Testing, fail name: {fail_name}, fail reason: {fail_reason}")

    rec.record_in_end(final_structure, env, testing=True)
    
    score = sum(env.reward_record)
    return score


def _inference(agent: DeepQAgent, 
               env: Environment,
               chances: int,
               test_fail_names: List[str],
               test_fail_reasons: List[str], 
               logger: logging.Logger) -> float:
    """Inference: greedy policy, but can have multiple chance for fault"""
    structure = env.reset(testing=True)  # generate a fix-shaped structure
    graph = structure.graph.clone()
    score = 0
    done = False
    timestep = 0
    while not done:
        original_structure = deepcopy(structure)

        # select and perform an action
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        
        dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else []
        action, _ = agent.choose_action(state, 
                                        structure.already_minimum_section_story_indexes,
                                        dont_select_story_member_indexes, 
                                        greedy=True)
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        while done and chances > 0:
            print(f"----- Fails, current failed action: {action:3d}, current chances: {chances:3d}")
            # if greedy will fail, choose the subgreedy
            structure = deepcopy(original_structure)
            structure.already_minimum_section_story_indexes.append(action)
            dont_select = list(set(structure.already_minimum_section_story_indexes))

            if len(dont_select) >= len(structure.story_level_actions):
                done = True
                break
                
            dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else None
            action, _ = agent.choose_action(state, 
                                            dont_select, 
                                            dont_select_story_member_indexes, 
                                            greedy=True)
            structure, reward, done, fail_name, fail_reason = env.step(structure, action)
            chances -= 1

        # get next state
        graph = structure.graph.clone()

        # record
        score += reward
        timestep += 1
        logger.info(f"*****Testing Episode, timestep: {timestep:3d}, action: {action:3d}, reward: {reward:4f},  acculmulate_score: {score:.4f}")
    
        # if can't select anymore, then stop
        if len(set(structure.already_minimum_section_story_indexes)) >= len(structure.story_level_actions):
            done = True


    test_fail_names.append(fail_name)
    test_fail_reasons.append(fail_reason)
    logger.info("---> Constraint not satisfied, found optimal section:")
    logger.info(list(structure.member_section_dict.values()))
    logger.info(f"testing, fail name: {fail_name}, fail reason: {fail_reason}")
    return score




def _save_model(agent: DeepQAgent, env: Environment, name: str, logger: logging.Logger) -> None:
    save_model_path = env.checkpoint_dir / "models" / f"model_{name}.pt"
    torch.save({
        'gnn': agent.gnn.state_dict(),
        'online_q_network': agent.online_q_network.state_dict(),
        'target_q_network': agent.target_q_network.state_dict(),
        }, save_model_path)
    logger.info(f" ---> model saved to {save_model_path}\n\n\n")


def _load_model(agent: DeepQAgent, load_ckpt_dir, logger: logging.Logger) -> None:
    # theta_1, theta_2, theta_3, online_q_network, target_q_network
    save_model_path = load_ckpt_dir #/ "model.pt"
    checkpoint = torch.load(save_model_path, map_location=torch.device(agent.device))
    agent.gnn.load_state_dict(checkpoint['gnn'])    
    agent.online_q_network.load_state_dict(checkpoint['online_q_network'])    
    agent.target_q_network.load_state_dict(checkpoint['target_q_network'])    
    logger.info(f"model are loaded from {save_model_path}")


def train(agent: DeepQAgent,
          env: Environment,
          rec: Record,
          number_episodes: int,
          logger: logging.Logger):
    """Reinforcement learning training loop."""

    for i in range(number_episodes):
        score = _train_an_episode(agent, env, rec, logger)
        logger.critical(f"Episode: {i+1}, score: {score:.3f}")
        logger.critical(f"Episode: {i+1}, saved_material: {sum(env.saved_material_record):.3f}, saved_material_SCWB: {sum(env.saved_material_record_SCWB):.3f}")
        logger.critical(f"Episode: {i+1}, total_reduction_amount: {env.material_usage_record[0] - env.material_usage_record[-1]:.3f}\n\n\n")

        if (i+1) % agent._test_frequency == 0:
            test_score = _testing(agent, env, rec, logger)
            logger.critical(f"Testing, score: {test_score:.3f}")
            logger.critical(f"Testing, saved_material: {sum(env.saved_material_record):.3f}, saved_material_SCWB: {sum(env.saved_material_record_SCWB):.3f}")
            logger.critical(f"Testing, total_reduction_amount: {env.material_usage_record[0] - env.material_usage_record[-1]:.3f}\n\n\n")
            
            rec.output(env.checkpoint_dir)

            if np.argmin(rec.testing_record["final_volume"]) == len(rec.testing_record["final_volume"])-1: 
                _save_model(agent, env, name="MinimumUsage", logger=logger)
            if np.argmax(rec.testing_record["score"]) == len(rec.testing_record["score"])-1: 
                _save_model(agent, env, name="HighestScore", logger=logger)
            if (i+1) % 50 == 0: 
                _save_model(agent, env, name=f"Episode{str(i+1)}", logger=logger)


    # score_info = {
    #     "train_score": train_scores,
    #     "train_score_SCWB": train_scores_SCWB,
    #     "test_score": test_scores,
    #     "test_score_SCWB": test_scores_SCWB
    # }
    # fail_info = {
    #     "fail_name": fail_names,
    #     "fail_reason": fail_reasons,
    #     "test_fail_name": test_fail_names,
    #     "test_fail_reason": test_fail_reasons
    # }
    # test_info = {
    #     "test_action": test_actions,
    #     "test_action_SCWB": test_actions_SCWB,
    #     "test_final_material_usage": test_final_material_usages,
    #     "test_final_design": test_final_designs
    # }
    # other_info = {
    #     "learn_loss": learn_losses,
    #     "Q_value": Q_values
    # }

    # return score_info, fail_info, test_info, other_info


