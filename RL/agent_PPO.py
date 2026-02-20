import time
import torch
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Tuple, List, Dict, Callable

import RL.buffer as buffer
import RL.model as model
from RL.record import Record
from RL.environment import Environment


# abstract class
class Agent:
    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an actio given the current state of the environment."""
        raise NotImplementedError

    def learn(self, experiences: List[buffer.Experience]) -> None:
        """Update the agent based on a collection of recent experiences."""
        raise NotImplementedError

    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplementedError
    

class PPOAgent(Agent):
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lr: float, 
                 use_lr_scheduler: bool,
                 gamma: float,
                 use_gae: bool,
                 gae_tau: float,
                 target_kl: float,
                 clip_eps: float,
                 entropy_weight_schedule: Callable[[int], float],
                 actor_loss_coef: float,
                 critic_loss_coef: float,
                 max_grad_norm: float,
                 optimization_epoch: int,
                 seed: int = None,
                 logger: logging.Logger = None,
                 pretrained_model_path: str = None,
                 device = "cpu", 
        ):
        self.logger = logger
        self.device = device
        
        # Set seeds for reproducbility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize GNN
        model_kwargs = {"node_feature_dim": node_feature_dim, "edge_feature_dim": edge_feature_dim, "hidden_dim": hidden_dim, "member_state_dim": hidden_dim, "num_layers": num_layers}
        self.gnn = model.StateGNN(**model_kwargs).to(self.device)
        self.logger.critical(f"gnn: \n{self.gnn}")

        # Initialize Actor-Critic network
        actor_critic_kwargs = {"member_state_dim": hidden_dim*2, "hidden_dim": hidden_dim, "action_dim": 1}
        self.actor_critic_network = model.Categorical_ActorCritic_Network(**actor_critic_kwargs).to(self.device)
        self.logger.critical(f"actor_critic_network: \n{self.actor_critic_network}")

        # Initialize optimizer
        params_gnn = list(self.gnn.parameters())
        params_actor = list(self.actor_critic_network.actor.parameters())
        params_critic = list(self.actor_critic_network.critic.parameters())
        self.optimizer = optim.Adam(params_gnn+params_actor+params_critic, lr=lr)
        if use_lr_scheduler:
            self.use_lr_scheduler = True
            # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda current_episode: 1 - current_episode / 2000, verbose=True)

        # Initialize hyperparameters
        self.discount = gamma
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.target_kl = target_kl
        self.clip_eps = clip_eps
        self.entropy_weight_schedule = entropy_weight_schedule
        self.actor_loss_coef = actor_loss_coef
        self.critic_loss_coef = critic_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optimization_epoch = optimization_epoch

        # Initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0
        self._backprop_count = 0

        # Initialize pretrained model
        if pretrained_model_path:
            self.load_model(pretrained_model_path)

        self.storage = buffer.RolloutBuffer()


    def choose_action(self, state: torch.Tensor, infeasible_actions: List[int], greedy=False):
        infeasible_mask = torch.tensor([True if i in infeasible_actions else False for i in range(state.shape[0])], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits, value = self.actor_critic_network.forward(state)
            masked_logits = logits.masked_fill(infeasible_mask, float('-inf'))  # set infeasible actions' logits to -inf
            probs = nn.functional.softmax(masked_logits, dim=-1)  # shape: (num_actions,)
            dist = torch.distributions.Categorical(probs=probs)
            dist_probs = dist.probs.clone().reshape((4, -1)).cpu().numpy()  # shape: (4, story_num)
            for category, prob in zip(["xdir_beam", "zdir_beam", "outer_col", "inner_col"], dist_probs):
                print(f"{category}: {prob}")
            print(f"{value = }")
            entropy = dist.entropy()
            print(f"{entropy = }")
            action = dist.sample() if not greedy else probs.argmax()
            print(f"{action = }")        
            log_prob = dist.log_prob(action)
            print(f"{log_prob = }")

        return logits, value.item(), entropy.item(), action.item(), log_prob.item()
    

    def step(self):
        # compute returns and advantages
        if not self.use_gae:
            returns = []
            ret = 0
            for i in reversed(range(len(self.storage.reward))):
                ret = self.storage.reward[i] + self.discount * ret * (1 - self.storage.done[i])
                returns.insert(0, ret)
            returns = torch.tensor(returns).to(self.device)  # can be normalized by initial material usage
            values = torch.tensor(self.storage.value).to(self.device)  # can be normalized by initial material usage
            advantages = returns - values
        else:
            returns = []
            advantages = []
            gae = 0
            values = self.storage.value + [0]  # add a dummy value for the last next_state
            for i in reversed(range(len(self.storage.reward))):
                mask = 1 - self.storage.done[i]
                delta = self.storage.reward[i] + self.discount * mask * values[i+1] - values[i]
                gae = gae * self.gae_tau * self.discount * mask + delta
                advantages.insert(0, gae)
                returns.insert(0, gae+values[i])
            returns = torch.tensor(returns).to(self.device)  # can be normalized by initial material usage
            advantages = torch.tensor(advantages).to(self.device)  # can be normalized by initial material usage
        print(f"\t{advantages.mean() = :.6f}, {advantages.std() = :.6f}")
        normed_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # prepare batched data from storage
        actions = torch.tensor(self.storage.action).to(self.device)
        log_probs_old = torch.tensor(self.storage.log_prob).to(self.device)
        entropies = torch.tensor(self.storage.entropy).to(self.device)
        infeasible_masks = torch.stack(self.storage.infeasible_actions).to(self.device)  # shape: (batch_size, num_actions)

        graphs = [graph for graph in self.storage.graph]
        member_numbers = [int(graph.edge_attr.shape[0]/2) for graph in graphs]  
        member_ptr = torch.tensor([sum(member_numbers[:i]) for i in range(len(member_numbers)+1)])  # size: (batch_size + 1)
        
        member_batch = []
        for i, member_number in enumerate(member_numbers):
            member_batch += [i] * member_number
        member_batch = torch.tensor(member_batch).to(self.device)  # size: (total edge_num)

        # story level pooling preparation
        auxs = [aux for aux in self.storage.aux]
        structure_story_ptr = []  # if the first and second graph have 16, 12 story members, it will be [0, 16, 28]
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

        # get batched states from batched graphs
        loader = DataLoader(list(graphs), batch_size=len(graphs))
        graphs_batch = next(iter(loader)).to(self.device)
        states = self.gnn.forward(graphs_batch.x, graphs_batch.edge_index, graphs_batch.edge_attr, member_batch, story_batch, structure_story_ptr)  # shape: (total story_member_num, hidden_dim*2)
        print(f"\t{states.shape = }")

        batched_states = states.split((structure_story_ptr[1:] - structure_story_ptr[:-1]).tolist())  # list of tensors, each tensor shape: (story_member_num, hidden_dim*2)        
        if all(batch.shape == batched_states[0].shape for batch in batched_states):
            batched_states = torch.stack(batched_states)  # shape: (batch_size, story_member_num, hidden_dim*2)
            print(f"\t\t{batched_states.shape = }")
            batched_logits, batched_values = self.actor_critic_network.forward(batched_states)  # shape: (batch_size, num_actions), (batch_size, 1)
            masked_batched_logits = batched_logits.masked_fill(infeasible_masks, float('-inf'))  # shape: (batch_size, num_actions)
            probs = nn.functional.softmax(masked_batched_logits, dim=-1)  # shape: (batch_size, num_actions)
            dists = torch.distributions.Categorical(probs=probs)
            print(f"\t\t{dists.probs.shape = }, {batched_values.shape = }")
            # print(f"\t{dists.probs.sum(dim=1) = }")  # all of them is 1.0
            log_probs = dists.log_prob(actions)  # shape: (batch_size)
            print(f"\t\t{log_probs.shape = }")
            mean_entropy = dists.entropy().mean()  # shape: (batch_size)
            print(f"\t\t{dists.entropy().shape = }, {mean_entropy = }")
        else:
            all_dists = []
            all_values = []
            for batch in batched_states:
                dist, value = self.actor_critic_network.forward(batch.to(self.device))
                all_dists.append(dist.probs.unsqueeze(0))
                all_values.append(value.unsqueeze(0))
            batched_values = torch.stack(all_values).squeeze(-1)  # shape: (batch_size)
            print(f"{batched_values.shape = }")

            all_log_probs = []
            all_entropies = []
            actions = torch.tensor(self.storage.action).to(self.device)  # shape: (batch_size)
            for i, dist in enumerate(all_dists):
                all_log_probs.append(dist.log_prob(actions[i]))
                all_entropies.append(dist.entropy())
            log_probs = torch.stack(all_log_probs)
            print(f"{log_probs.shape = }")  # shape: (batch_size)
            mean_entropy = torch.mean(all_entropies)

        # compute losses and update networks
        # kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()  # theroetical calculation
        approx_kl = torch.mean(log_probs_old - log_probs)  # approximation
        print(f"\t{approx_kl = }")
        ratios = torch.exp(log_probs - log_probs_old)
        print(f"\t{ratios.mean() = :.4f}, {ratios.std() = :.4f}")
        obj_1 = ratios * normed_advantages
        obj_2 = torch.clamp(ratios, 1.0-self.clip_eps, 1.0+self.clip_eps) * normed_advantages
        actor_loss = torch.min(obj_1, obj_2).mean()
        entropy_weight = self.entropy_weight_schedule(self._number_episodes)
        entropy_loss = entropy_weight * mean_entropy
        critic_loss = (returns - batched_values.squeeze()).pow(2).mean()
        print(f"\t[Actor coef]: {self.actor_loss_coef:.6f} [Entropy coef]: {entropy_weight:.6f} [Critic coef]: {self.critic_loss_coef:.6f}")
        print(f"\t[Actor loss]: {actor_loss:.6f} [Mean entropy]: {mean_entropy:.6f} [Critic loss]: {critic_loss:.6f}")
        assert actor_loss.requires_grad and entropy_loss.requires_grad and critic_loss.requires_grad

        # if approx_kl > 1.5*self.target_kl:
        #     actor_loss = torch.zeros_like(actor_loss).to(self.device)
        #     print(f"\t\tEarly stopping for updating actor network")
        total_loss = self.critic_loss_coef*critic_loss - self.actor_loss_coef*actor_loss - entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()

        # compute gradient norm for monitoring
        gnn_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.gnn.parameters() if p.grad is not None]).mean().sqrt()
        actor_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.actor_critic_network.actor.parameters() if p.grad is not None]).mean().sqrt()
        critic_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.actor_critic_network.critic.parameters() if p.grad is not None]).mean().sqrt()
        # clip gradients separately to avoid cross-network interference
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.gnn.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic_network.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic_network.critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        info = {
            "returns": returns.cpu().tolist(),
            "advantages": advantages.cpu().tolist(),
            "pred_values": batched_values.squeeze().detach().cpu().tolist(),
            "mean_kl_divergence": approx_kl.detach().cpu().numpy().item(),
            "ratios": ratios.detach().cpu().tolist(),
            "actor_loss": actor_loss.detach().cpu().numpy().item(),
            "entropy_loss": entropy_loss.detach().cpu().numpy().item(),
            "critic_loss": critic_loss.detach().cpu().numpy().item(),
            "gnn_grad_norm": gnn_grad_norm.detach().cpu().numpy().item(),
            "actor_grad_norm": actor_grad_norm.detach().cpu().numpy().item(),
            "critic_grad_norm": critic_grad_norm.detach().cpu().numpy().item(),
        }

        return info


    def save_model(self, ckpt_dir: Path, name: str) -> None:
        checkpoint = {
            'gnn': self.gnn.state_dict(),
            'actor_critic': self.actor_critic_network.state_dict(),
        }
        model_path = ckpt_dir / f"model_{name}.pt"
        torch.save(checkpoint, model_path)
        self.logger.critical(f"Saved model to {model_path}\n\n\n")


    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.gnn.load_state_dict(checkpoint['gnn'])
        self.actor_critic_network.load_state_dict(checkpoint['actor_critic'])
        self.logger.critical(f"Loaded model from {model_path}")




def _train_an_episode(agent: PPOAgent, 
                      env: Environment, 
                      rec: Record,
                      logger: logging.Logger) -> float:
    """Train the agent until the agent meets the terminal state."""
    structure = env.reset()  # generate a new random graph
    rec.record_in_beginning(structure, testing=False)
    graph = structure.graph.clone()
    score = 0
    done = False
    while not done:
        original_structure = deepcopy(structure)
        # select and perform an action
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        logits, value, entropy, action, log_prob = agent.choose_action(state, structure.already_minimum_section_story_indexes)
        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"-----episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, story_level_sections: {structure.story_level_sections}, action: {action:3d} [{update_story}F {member_category}]")
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)
        score += reward
        logger.info(f"episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, action: {action:3d}, reward: {reward:4f},  acculmulate_score: {score:.4f}")
        
        # store the transition in memory
        transition = {
            "graph": original_structure.graph.clone(),
            "logits": logits,
            "value": value,
            "entropy": entropy,
            "action": action,
            "log_prob": log_prob,
            "next_graph": structure.graph.clone(),
            "reward": reward,
            "done": done,
            "infeasible_actions": torch.tensor([True if i in original_structure.already_minimum_section_story_indexes else False for i in range(state.shape[0])], dtype=torch.bool),
            "aux": structure.aux
        }
        agent.storage.store(transition)
        graph = structure.graph.clone()
        agent._number_timesteps += 1
        print()

    returns = []
    advantages = []
    pred_values = []
    mean_kl_divergences = []
    ratios = []
    actor_losses, entropy_losses, critic_losses = [], [], []
    gnn_grad_norms, actor_grad_norms, critic_grad_norms = [], [], []

    params_gnn_before = [p.clone().detach() for p in agent.gnn.parameters()]
    params_actor_before = [p.clone().detach() for p in agent.actor_critic_network.actor.parameters()]
    params_critic_before = [p.clone().detach() for p in agent.actor_critic_network.critic.parameters()]

    t_start = time.time()
    if agent.use_lr_scheduler:
        agent.lr_scheduler.step(agent._number_episodes)  # update learning rate based on episode count
    for epoch in range(agent.optimization_epoch):
        info = agent.step()
        returns.append(info["returns"])
        advantages.append(info["advantages"])
        pred_values.append(info["pred_values"])
        mean_kl_divergences.append(info["mean_kl_divergence"])
        ratios.append(info["ratios"])
        actor_losses.append(info["actor_loss"])
        entropy_losses.append(info["entropy_loss"])
        critic_losses.append(info["critic_loss"])
        gnn_grad_norms.append(info["gnn_grad_norm"])
        actor_grad_norms.append(info["actor_grad_norm"])
        critic_grad_norms.append(info["critic_grad_norm"])
        agent.logger.info(f"Optimization epoch: {epoch+1:2d}, Loss - Actor: {info['actor_loss']:.4f}, Entropy: {info['entropy_loss']:.4f}, Critic: {info['critic_loss']:.4f}")
        agent.logger.info(f"Gradient norm - GNN: {info['gnn_grad_norm']:.4f}, Actor: {info['actor_grad_norm']:.4f}, Critic: {info['critic_grad_norm']:.4f}")
    t_end = time.time()
    print(f"Total optimization time: {t_end - t_start:.3f} seconds")

    params_gnn_after = [p.clone().detach() for p in agent.gnn.parameters()]
    params_actor_after = [p.clone().detach() for p in agent.actor_critic_network.actor.parameters()]
    params_critic_after = [p.clone().detach() for p in agent.actor_critic_network.critic.parameters()]
    param_change = lambda before, after: torch.tensor([(a - b).pow(2).sum() for b, a in zip(before, after)]).mean().sqrt().item()
    gnn_change = param_change(params_gnn_before, params_gnn_after)
    actor_change = param_change(params_actor_before, params_actor_after)
    critic_change = param_change(params_critic_before, params_critic_after)
    agent.logger.info(f"Parameter changes - GNN: {gnn_change:.6f}, Actor: {actor_change:.6f}, Critic: {critic_change:.6f}")

    rec.returns["train"].append(returns)
    rec.advantages["train"].append(advantages)
    rec.entropies["train"].append(agent.storage.entropy)
    rec.pred_values.append(pred_values)
    rec.kl_divergences.append(mean_kl_divergences)
    rec.ratios.append(ratios)
    rec.losses["actor"].append(actor_losses)
    rec.losses["entropy"].append(entropy_losses)
    rec.losses["critic"].append(critic_losses)
    rec.grad_norms["gnn"].append(gnn_grad_norms)
    rec.grad_norms["actor"].append(actor_grad_norms)
    rec.grad_norms["critic"].append(critic_grad_norms)
    rec.param_changes["gnn"].append(gnn_change)
    rec.param_changes["actor"].append(actor_change)
    rec.param_changes["critic"].append(critic_change)

    agent.storage.reset()
    agent._number_episodes += 1
    agent._backprop_count += 1
    agent._number_timesteps = 0

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    final_story_level_sections = final_structure.story_level_sections
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{final_story_level_sections = }")
    logger.info(f"Episode: {agent._number_episodes:4d}, fail name: {fail_name}, fail reason: {fail_reason}")
    
    rec.record_in_end(final_structure, env, testing=False)

    score = sum(env.reward_record)
    return score


def _testing(agent: PPOAgent, 
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
            state = agent.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        
        logits, value, entropy, action, log_prob = agent.choose_action(state, structure.already_minimum_section_story_indexes, greedy=True)
        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"*****Testing Episode, story_level_sections: {structure.story_level_sections}, action: {action:3d} [{update_story}F {member_category}]")
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # store the transition in memory
        transition = {
            "graph": original_structure.graph.clone(),
            "logits": logits,
            "value": value,
            "entropy": entropy,
            "action": action,
            "log_prob": log_prob,
            "next_graph": structure.graph.clone(),
            "reward": reward,
            "done": done,
            "infeasible_actions": np.array([True if i in structure.already_minimum_section_story_indexes else False for i in range(state.shape[0])], dtype=bool),
            "aux": structure.aux
        }
        agent.storage.store(transition)
        graph = structure.graph.clone()

        # record
        score += reward
        timestep += 1
        logger.info(f"*****Testing Episode, timestep: {timestep:3d}, action: {action:3d}, reward: {reward:4f},  acculmulate_score: {score:.4f}")
        print()

    if not agent.use_gae:
        returns = []
        ret = 0
        for i in reversed(range(len(agent.storage.reward))):
            ret = agent.storage.reward[i] + agent.discount * ret * (1 - agent.storage.done[i])
            returns.insert(0, ret)
        advantages = (torch.tensor(returns) - torch.tensor(agent.storage.value)).tolist()
    else:
        returns = []
        advantages = []
        gae = 0
        values = agent.storage.value + [0]  # add a dummy value for the last next_state
        for i in reversed(range(len(agent.storage.reward))):
            mask = 1 - agent.storage.done[i]
            delta = agent.storage.reward[i] + agent.discount * mask * values[i+1] - values[i]
            gae = gae * agent.gae_tau * agent.discount * mask + delta
            advantages.insert(0, gae)
            returns.insert(0, gae+values[i])
    
    rec.returns["test"].append(returns)
    rec.advantages["test"].append(advantages)
    rec.entropies["test"].append(agent.storage.entropy)
    agent.storage.reset()

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    test_final_story_level_sections = final_structure.story_level_sections
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{test_final_story_level_sections = }")
    logger.info(f"Testing, fail name: {fail_name}, fail reason: {fail_reason}")

    rec.record_in_end(final_structure, env, testing=True)
    
    score = sum(env.reward_record)
    return score




def train(agent: PPOAgent,
          env: Environment,
          rec: Record,
          train_episode_num: int,
          test_frequency: int,
          test_episode_num: int,
          logger: logging.Logger):
    """Reinforcement learning training loop."""

    for i in range(train_episode_num):
        score = _train_an_episode(agent, env, rec, logger)
        logger.critical(f"Training Episode: {i+1}, score: {score:.3f}")
        logger.critical(f"Training Episode: {i+1}, saved_material: {sum(env.saved_material_record):.3f}, saved_material_SCWB: {sum(env.saved_material_record_SCWB):.3f}")
        logger.critical(f"Training Episode: {i+1}, total_reduction_amount: {env.material_usage_record[0] - env.material_usage_record[-1]:.3f}\n\n\n")

        if (i+1) % test_frequency == 0:
            for j in range(test_episode_num):
                test_score = _testing(agent, env, rec, logger)
                logger.critical(f"Testing Episode: {j+1}, score: {test_score:.3f}")
                logger.critical(f"Testing Episode: {j+1}, saved_material: {sum(env.saved_material_record):.3f}, saved_material_SCWB: {sum(env.saved_material_record_SCWB):.3f}")
                logger.critical(f"Testing Episode: {j+1}, total_reduction_amount: {env.material_usage_record[0] - env.material_usage_record[-1]:.3f}\n\n\n")
            
            rec.output(env.checkpoint_dir)
            final_volumes = np.array(rec.testing_record["final_volume"]).reshape((-1, test_episode_num))
            scores = np.array(rec.testing_record["score"]).reshape((-1, test_episode_num))

            if np.argmin(final_volumes.mean(axis=1)) == len(final_volumes)-1: 
                agent.save_model(env.checkpoint_dir/"models", name="MinimumUsage")
            if np.argmax(scores.mean(axis=1)) == len(final_volumes)-1: 
                agent.save_model(env.checkpoint_dir/"models", name="HighestScore")
            if (i+1) % 50 == 0: 
                agent.save_model(env.checkpoint_dir/"models", name=f"Episode{str(i+1)}")