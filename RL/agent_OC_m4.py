import time
import torch
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy
from torch import nn, optim
from torch_geometric.loader import DataLoader
from typing import Tuple, List, Dict, Callable

import RL.buffer as buffer
import RL.model as model
from RL.record import Record
from RL.environment import Environment


# abstract class
class Agent:
    def choose_action(self, state: np.array) -> int:
        raise NotImplementedError

    def learn(self, experiences: List[buffer.Experience]) -> None:
        raise NotImplementedError

    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        raise NotImplementedError


class OptionCriticAgent(Agent):
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 use_bias: bool, 
                 lr: float, 
                 use_lr_scheduler: bool,
                 gamma: float,
                 entropy_weight_schedule: Callable[[int], float],
                 actor_loss_type: str, 
                 actor_loss_coef: float,
                 critic_loss_coef: float,
                 beta_loss_coef: float,
                 gjsd_loss_coef: float,
                 termination_reg: float,
                 max_grad_norm: float,
                 optimization_epoch: int, 
                 num_options: int,
                 if_hierarchical: bool,
                 epsilon_decay_schedule: Callable[[int], float],
                 sync_target_freq: float, 
                 use_gae: bool = False,
                 gae_tau: float = 0.95,
                 target_kl: float = 0.02,
                 clip_eps: float = 0.2,
                 seed: int = None,
                 logger: logging.Logger = None,
                 pretrained_model_path: str = None,
                 device = "cpu"
        ):
        self.restrict_action = False
        self.logger = logger
        self.device = device

        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)

        model_kwargs = {"node_feature_dim": node_feature_dim, "edge_feature_dim": edge_feature_dim, "hidden_dim": hidden_dim, "member_state_dim": hidden_dim, "num_layers": num_layers}
        self.gnn = model.StateGNN(**model_kwargs).to(self.device)
        self.logger.critical(f"gnn: \n{self.gnn}")

        option_critic_kwargs = {"member_state_dim": hidden_dim*2, "hidden_dim": hidden_dim, "action_dim": 1, "num_options": num_options, "use_bias": use_bias}
        self.online_oc = model.OptionCritic(**option_critic_kwargs).to(self.device)
        self.target_oc = model.OptionCritic(**option_critic_kwargs).to(self.device)
        self._sync_target_networks()
        self.logger.critical(f"Option-Critic Network: \n{self.online_oc}")

        params = list(self.gnn.parameters()) + list(self.online_oc.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.use_lr_scheduler = use_lr_scheduler
        if self.use_lr_scheduler:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda current_episode: 1 - current_episode / 2000, verbose=True)

        self.discount = gamma
        self.entropy_weight_schedule = entropy_weight_schedule
        self.actor_loss_type = actor_loss_type
        self.actor_loss_coef = actor_loss_coef
        self.critic_loss_coef = critic_loss_coef
        self.beta_loss_coef = beta_loss_coef
        self.gjsd_loss_coef = gjsd_loss_coef
        self.termination_reg = termination_reg
        self.max_grad_norm = max_grad_norm
        self.optimization_epoch = optimization_epoch
        self.num_options = num_options
        self.if_hierarchical = if_hierarchical
        self.epsilon_decay_schedule = epsilon_decay_schedule
        self.sync_target_freq = sync_target_freq

        # PPO
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.target_kl = target_kl
        self.clip_eps = clip_eps

        self._number_episodes = 0
        self._number_timesteps = 0
        self._optimizer_step_count = 0

        if pretrained_model_path:
            self.load_model(pretrained_model_path)

        self.storage = buffer.RolloutBuffer()

    def _sync_target_networks(self) -> None:
        """Synchronize target GNN and target Option-Critic network."""
        if hasattr(self, "target_gnn"):
            self.target_gnn.load_state_dict(self.gnn.state_dict())
        if hasattr(self, "target_oc"):
            self.target_oc.load_state_dict(self.online_oc.state_dict())

    def _unpack_prediction(self, prediction) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expected model output for Milestone 4:
            `logits_all`:    `(num_options, num_actions)` or `(B, num_options, num_actions)`
            `option_values`: `(num_options,)` or `(B, num_options)`
            `betas`:         `(num_options,)` or `(B, num_options)`
        """
        if isinstance(prediction, dict):
            return prediction["logits"], prediction["q"], prediction["beta"]
        if isinstance(prediction, tuple) and len(prediction) == 3:
            return prediction
        raise RuntimeError(
            "Milestone 4 requires RL.model.OptionCritic.forward() to return "
            "(logits_all, option_values, betas). Add a termination head beta(s,o)."
        )

    def compute_gjsd(self, batched_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Jensen-Shannon Divergence (GJSD) among the option policies.

        - `batched_probs`: Tensor of shape `(num_options, num_actions)` representing the action probabilities for each option.
        """
        def entropy(probs: torch.Tensor) -> torch.Tensor:
            return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        group_mean_entropy = entropy(batched_probs.mean(dim=0))
        mean_individual_entropy = entropy(batched_probs).mean()
        gjsd = group_mean_entropy - mean_individual_entropy

        return gjsd

    def sample_option(self, state: torch.Tensor, prev_option: int, is_initial_state: bool, greedy=False):
        """
        Standard Option-Critic option selection:
            - initial state: epsilon-greedy over Q(s,o)
            - otherwise: keep previous option with probability 1-beta(s, prev_option), 
                         terminate and reselect with probability beta(s, prev_option)
        
        Returns: option_idx, option_value, beta, beta_prev, epsilon
        """
        with torch.no_grad():
            logits_all, option_values, betas = self.online_oc.forward(state)
            print(f"\t{option_values = }")
            print(f"\t{betas = }")

            epsilon = 0 if greedy else self.epsilon_decay_schedule(self._number_episodes)

            if greedy:
                greedy_option = torch.argmax(option_values).item()
                beta_prev = betas[prev_option].item()
                if is_initial_state or beta_prev >= 0.5:
                    option_idx = greedy_option  # hight chance of termination
                else:
                    option_idx = prev_option  # low chance of termination
            else:
                pi_option = torch.ones_like(option_values) * (epsilon / self.num_options)
                greedy_option = torch.argmax(option_values).item()
                pi_option[greedy_option] = 1 - epsilon + epsilon / self.num_options

                if is_initial_state:
                    probs = pi_option
                else:
                    beta_prev = betas[prev_option]
                    pi_hat_option = beta_prev * pi_option
                    pi_hat_option[prev_option] += (1 - beta_prev)
                    probs = pi_hat_option
                print(f"\t{probs = }")
                dist = torch.distributions.Categorical(probs=probs)
                option_idx = dist.sample().item()

            option_value = option_values[option_idx]
            beta = betas[option_idx]
            beta_prev = betas[prev_option]
            print(f"\t{is_initial_state = }, {epsilon = }, {prev_option = }, {option_idx = }")

        return option_idx, option_value.item(), beta.item(), beta_prev.item(), epsilon

    def choose_action(self, state: torch.Tensor, infeasible_actions: List[int], option_idx: int, greedy=False):
        """
        Given the current state and option, choose an action according to the option policy, while masking out infeasible actions.
        
        Returns: gjsd, logits, option_value, beta, entropy, action, log_prob
        """
        infeasible_mask = torch.tensor([True if i in infeasible_actions else False for i in range(state.shape[0])], dtype=torch.bool, device=self.device)
        masks = infeasible_mask.unsqueeze(0).repeat(self.num_options, 1)  # (num_actions,) --> (num_options, num_actions)
        with torch.no_grad():
            logits_all, option_values, betas = self.online_oc.forward(state)
            masked_logits = logits_all.masked_fill(masks, float('-inf'))
            probs = nn.functional.softmax(masked_logits, dim=-1)  # (num_options, num_actions)
            gjsd = self.compute_gjsd(probs)
            dist = torch.distributions.Categorical(probs=probs[option_idx])

            selected_logits = logits_all[option_idx]
            selected_value = option_values[option_idx]
            selected_beta = betas[option_idx]
            dist_probs = dist.probs.clone().reshape((4, -1)).cpu().numpy()
            for category, prob in zip(["xdir_beam", "zdir_beam", "outer_col", "inner_col"], dist_probs):
                print(f"{category}: {prob}")
            print(f"{selected_value = }, {selected_beta = }")
            entropy = dist.entropy()
            print(f"{entropy = }")
            action = dist.sample() if not greedy else dist.probs.argmax()
            print(f"{action = }")
            log_prob = dist.log_prob(action)
            print(f"{log_prob = }")

        return gjsd.item(), selected_logits, selected_value.item(), selected_beta.item(), entropy.item(), action.item(), log_prob.item()

    def _build_batched_states(self, graphs: List, auxs: List, use_target: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        member_numbers = [int(graph.edge_attr.shape[0]/2) for graph in graphs]
        member_ptr = torch.tensor([sum(member_numbers[:i]) for i in range(len(member_numbers)+1)])

        member_batch = []
        for i, member_number in enumerate(member_numbers):
            member_batch += [i] * member_number
        member_batch = torch.tensor(member_batch).to(self.device)

        structure_story_ptr = []
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

        loader = DataLoader(list(graphs), batch_size=len(graphs))
        graphs_batch = next(iter(loader)).to(self.device)
        gnn = self.target_gnn if use_target else self.gnn
        states = gnn.forward(graphs_batch.x, graphs_batch.edge_index, graphs_batch.edge_attr, member_batch, story_batch, structure_story_ptr)
        batched_states = states.split((structure_story_ptr[1:] - structure_story_ptr[:-1]).tolist())
        
        return states, batched_states
    
    def get_batched_info(self) -> dict[str, torch.Tensor]:
        # prepare batched data from storage
        actions = torch.tensor(self.storage.action).to(self.device)
        infeasible_masks = torch.stack(self.storage.infeasible_actions).to(self.device)  # (B, A)
        batched_options = torch.tensor(self.storage.option_idx).to(self.device)
        prev_options = torch.tensor(self.storage.prev_option).to(self.device)
        epsilons = torch.tensor(self.storage.eps).to(self.device)
        graphs = [graph for graph in self.storage.graph]
        next_graphs = [graph for graph in self.storage.next_graph]
        auxs = [aux for aux in self.storage.aux]

        # get current batched states
        states, batched_states = self._build_batched_states(graphs, auxs, use_target=False)
        print(f"\t{states.shape = }")

        if all(batch.shape == batched_states[0].shape for batch in batched_states):
            batched_states = torch.stack(batched_states)  # (B, A, member_state_dim)
            print(f"\t\t{batched_states.shape = }")
            batched_logits_all, batched_values_all, batched_betas_all = self.online_oc.forward(batched_states)  # (B, O, A), (B, O), (B, O)
            print(f"\t\t{batched_logits_all.shape = }, {batched_values_all.shape = }, {batched_betas_all.shape = }")

            batched_masks = infeasible_masks.unsqueeze(1).repeat(1, self.num_options, 1)  # (B, 1, A) --> (B, O, A)
            masked_batched_logits = batched_logits_all.masked_fill(batched_masks, float('-inf'))  # (B, O, A)
            batched_probs_all = nn.functional.softmax(masked_batched_logits, dim=-1)  # (B, O, A)

            all_gjsds = []
            all_probs = []
            all_option_values = []
            all_prev_option_values = []
            all_prev_betas = []
            all_state_values = []
            for probs, option_values, betas, option_idx, prev_option, eps in zip(batched_probs_all, batched_values_all, batched_betas_all, batched_options, prev_options, epsilons):
                all_gjsds.append(self.compute_gjsd(probs))
                all_probs.append(probs[option_idx])
                all_option_values.append(option_values[option_idx])
                all_prev_option_values.append(option_values[prev_option])
                all_prev_betas.append(betas[prev_option])
                v = option_values.max(dim=-1)[0] * (1 - eps) + option_values.mean(dim=-1) * eps
                all_state_values.append(v)

            mean_gjsd = torch.stack(all_gjsds).to(self.device).mean()
            print(f"\t\t{mean_gjsd = }")
            batched_probs = torch.stack(all_probs)  # (B, A)
            values = torch.stack(all_option_values)  # (B,)
            prev_values = torch.stack(all_prev_option_values)  # (B,)
            prev_betas = torch.stack(all_prev_betas)  # (B,)
            state_values = torch.stack(all_state_values)  # (B,)

            batched_dists = torch.distributions.Categorical(probs=batched_probs)
            log_probs = batched_dists.log_prob(actions)  # (B,)
            mean_entropy = batched_dists.entropy().mean()  # scalar
        else:
            all_option_values = []
            all_prev_option_values = []
            all_prev_betas = []
            all_state_values = []
            all_gjsds = []
            all_log_probs = []
            all_entropies = []
            for state, mask, option_idx, prev_option, eps, action in zip(batched_states, infeasible_masks, batched_options, prev_options, epsilons, actions):
                logits_all, option_values, betas = self.online_oc.forward(state.to(self.device))
                all_option_values.append(option_values[option_idx])
                all_prev_option_values.append(option_values[prev_option])
                all_prev_betas.append(betas[prev_option])
                v = option_values.max(dim=-1)[0] * (1 - eps) + option_values.mean(dim=-1) * eps
                all_state_values.append(v)

                masks = mask.unsqueeze(0).repeat(self.num_options, 1)  # (A,) --> (O, A)
                masked_logits = logits_all.masked_fill(masks, float("-inf"))
                probs = nn.functional.softmax(masked_logits, dim=-1)  # (O, A)
                all_gjsds.append(self.compute_gjsd(probs))

                dist = torch.distributions.Categorical(probs=probs[option_idx])
                all_log_probs.append(dist.log_prob(action))
                all_entropies.append(dist.entropy())

            values = torch.stack(all_option_values).to(self.device)
            prev_values = torch.stack(all_prev_option_values).to(self.device)
            prev_betas = torch.stack(all_prev_betas).to(self.device)
            state_values = torch.stack(all_state_values).to(self.device)
            mean_gjsd = torch.stack(all_gjsds).to(self.device).mean()
            log_probs = torch.stack(all_log_probs).to(self.device)
            mean_entropy = torch.stack(all_entropies).to(self.device).mean()

        # ===== Option-Critic return / advantage calculation =====
        # Bootstrap from target network using U(s',o) = (1-beta(s',o))*Q(s',o) + beta(s',o)*max_o Q(s',o).
        if len(self.storage.option_rollout_length) == 0:
            # flat training
            option_rollout_lengths = [1 for _ in self.storage.reward]
        else:
            # hierarchical training
            option_rollout_lengths = [max(1, int(k)) for k in self.storage.option_rollout_length]

        with torch.no_grad():
            ret = torch.tensor(0.0, device=self.device)
            _, next_states = self._build_batched_states(next_graphs, auxs, use_target=False)
            last_next_state = next_states[-1].to(self.device)
            _, next_option_values, next_betas = self.target_oc.forward(last_next_state)
            last_option = batched_options[-1]
            beta_next = next_betas[last_option]
            ret = (1 - beta_next) * next_option_values[last_option] + beta_next * next_option_values.max(dim=-1)[0]

            returns = []
            advantages = []
            for i in reversed(range(len(self.storage.reward))):
                mask = 1 - self.storage.done[i]
                discount = self.discount ** option_rollout_lengths[i]
                ret = torch.tensor(self.storage.reward[i], device=self.device, dtype=torch.float32) + discount * mask * ret
                returns.insert(0, ret)
                advantages.insert(0, ret-values[i].detach())

            returns = torch.stack(returns).to(self.device)
            advantages = torch.stack(advantages).to(self.device)
            beta_advantages = prev_values.detach() - state_values.detach() + self.termination_reg

        info = {
            "all_betas": batched_betas_all, 
            "all_values": batched_values_all, 
            "values": values, 
            "prev_betas": prev_betas, 
            "gjsd": mean_gjsd,
            "log_probs": log_probs,
            "entropy": mean_entropy, 
            "returns": returns,
            "advantages": advantages,
            "beta_advantages": beta_advantages,
        }
        return info

    def step(self):
        log_probs_old = torch.tensor(self.storage.log_prob).to(self.device)
        init_states = torch.tensor(self.storage.init_state, dtype=torch.float32).to(self.device)

        info = self.get_batched_info()
        all_betas = info["all_betas"]
        all_values = info["all_values"]
        values = info["values"]
        prev_betas = info["prev_betas"]
        mean_gjsd = info["gjsd"]
        log_probs = info["log_probs"]
        mean_entropy = info["entropy"]
        returns = info["returns"]
        advantages = info["advantages"]
        beta_advantages = info["beta_advantages"]

        # In standard Option-Critic, option selection is value-based and beta is trained by beta advantage.
        # Here the intra-option policy loss is intentionally kept PPO-style.
        approx_kl = torch.mean(log_probs_old - log_probs)
        print(f"\t{approx_kl = }")
        ratios = torch.exp(log_probs - log_probs_old)
        print(f"\t{ratios.mean() = :.4f}, {ratios.std() = :.4f}")
        normed_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(f"\t{advantages.mean() = :.4f}, {advantages.std() = :.4f}")

        if self.actor_loss_type == "PPO":
            obj_1 = ratios * normed_advantages.detach()
            obj_2 = torch.clamp(ratios, 1.0-self.clip_eps, 1.0+self.clip_eps) * normed_advantages.detach()
            actor_loss = torch.min(obj_1, obj_2).mean()
        elif self.actor_loss_type == "OC": 
            actor_loss = (log_probs * advantages.detach()).mean()
        entropy_weight = self.entropy_weight_schedule(self._number_episodes)
        entropy_loss = entropy_weight * mean_entropy
        critic_loss = (values - returns.detach()).pow(2).mul(0.5).mean()

        beta_loss = (prev_betas * beta_advantages.detach() * (1-init_states)).mean()
        
        beta_entropy_weight = 0.0
        beta_entropy = (
            prev_betas * torch.log(prev_betas + 1e-8)
            + (1-prev_betas) * torch.log(1-prev_betas + 1e-8)
        ).mean()
        
        beta_reg_coef = np.linspace(1.0, 0.01, 1000)[self._number_episodes]
        target_beta = torch.full_like(prev_betas, 0.5)
        beta_reg = nn.functional.binary_cross_entropy(prev_betas, target_beta)

        gjsd_loss = self.gjsd_loss_coef * mean_gjsd
        print(f"\t[loss coeff] A: {self.actor_loss_coef:.4f}, H_A: {entropy_weight:.4f}, C: {self.critic_loss_coef:.4f}, B: {self.beta_loss_coef:.4f}, H_B: {beta_entropy_weight:.4f}, B_reg: {beta_reg_coef:.4f}, GJSD: {self.gjsd_loss_coef:.4f}")
        print(f"\t[loss value] A: {actor_loss:.4f}, H_A: {mean_entropy:.4f}, C: {critic_loss:.4f}, B: {beta_loss:.4f}, H_B: {beta_entropy:.4f}, B_reg: {beta_reg:.4f}, GJSD: {mean_gjsd:.4f}")
        assert actor_loss.requires_grad and entropy_loss.requires_grad and critic_loss.requires_grad and beta_loss.requires_grad and beta_entropy.requires_grad and beta_reg.requires_grad and gjsd_loss.requires_grad

        total_loss = (
            - self.actor_loss_coef * actor_loss
            - entropy_loss
            + self.critic_loss_coef * critic_loss
            + self.beta_loss_coef * beta_loss
            - beta_entropy_weight * beta_entropy
            + beta_reg_coef * beta_reg
            - gjsd_loss
        )
        self.optimizer.zero_grad()
        total_loss.backward()

        gnn_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.gnn.parameters() if p.grad is not None]).mean().sqrt()
        actor_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.online_oc.fc_pi.parameters() if p.grad is not None]).mean().sqrt()
        critic_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.online_oc.option_critic.parameters() if p.grad is not None]).mean().sqrt()
        beta_grad_norm = torch.tensor([p.grad.norm().item() ** 2 for p in self.online_oc.termination.parameters() if p.grad is not None]).mean().sqrt()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.gnn.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.online_oc.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self._optimizer_step_count += 1
        if self.sync_target_freq is not None:
            if self._optimizer_step_count % self.sync_target_freq == 0:
                self.logger.info("Synchronizing online network to target network")
                self._sync_target_networks()

        info = {
            "all_betas": all_betas.detach().cpu().tolist(),
            "all_values": all_values.detach().cpu().tolist(),
            "returns": returns.detach().cpu().tolist(),
            "advantages": advantages.detach().cpu().tolist(),
            "beta_advantages": beta_advantages.detach().cpu().tolist(),
            "pred_values": values.detach().cpu().tolist(),
            "mean_kl_divergence": approx_kl.detach().cpu().numpy().item(),
            "ratios": ratios.detach().cpu().tolist(),
            "actor_loss": self.actor_loss_coef*actor_loss.detach().cpu().numpy().item(),
            "entropy_loss": entropy_loss.detach().cpu().numpy().item(),
            "critic_loss": self.critic_loss_coef*critic_loss.detach().cpu().numpy().item(),
            "beta_loss": (self.beta_loss_coef*beta_loss - beta_entropy_weight*beta_entropy + beta_reg_coef*beta_reg).detach().cpu().numpy().item(),
            "gjsd_loss": gjsd_loss.detach().cpu().numpy().item(),
            "gnn_grad_norm": gnn_grad_norm.detach().cpu().numpy().item(),
            "actor_grad_norm": actor_grad_norm.detach().cpu().numpy().item(),
            "critic_grad_norm": critic_grad_norm.detach().cpu().numpy().item(),
            "beta_grad_norm": beta_grad_norm.detach().cpu().numpy().item(),
        }

        return info

    def save_model(self, ckpt_dir: Path, name: str) -> None:
        checkpoint = {
            'gnn': self.gnn.state_dict(),
            'online_oc': self.online_oc.state_dict(),
            'target_oc': self.target_oc.state_dict(),
        }
        model_path = ckpt_dir / f"model_{name}.pt"
        torch.save(checkpoint, model_path)
        self.logger.critical(f"Saved model to {model_path}\n\n\n")

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.gnn.load_state_dict(checkpoint['gnn'])
        self.online_oc.load_state_dict(checkpoint['online_oc'])
        self.target_oc.load_state_dict(checkpoint["target_oc"])
        self.logger.critical(f"model are loaded from {model_path}")


def _flat_training(agent: OptionCriticAgent,
                   env: Environment,
                   rec: Record,
                   logger: logging.Logger) -> float:
    structure = env.reset()
    rec.record_in_beginning(structure, testing=False)
    graph = structure.graph.clone()
    score = 0
    done = False
    prev_option = 0
    is_initial_state = True
    while not done:
        original_structure = deepcopy(structure)
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        option_idx, option_value, beta, beta_prev, epsilon = agent.sample_option(state, prev_option, is_initial_state)
        gjsd, logits, value, beta, entropy, action, log_prob = agent.choose_action(state, structure.already_minimum_section_story_indexes, option_idx)
        print(f"{option_value - value = }")

        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"-----episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, story_level_sections: {structure.story_level_sections}, option: {option_idx:2d}, beta: {beta:.4f}, prev_option: {prev_option:2d}, prev_beta: {beta_prev:.4f}, action: {action:3d} [{update_story}F {member_category}]")
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)
        score += reward
        logger.info(f"episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, option: {option_idx:2d}, reward: {reward:4f}, score: {score:.4f}")

        transition = {
            # Actor-Critic
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
            "aux": original_structure.aux,
            # Option-Critic
            "option_idx": option_idx,
            "prev_option": prev_option,
            "init_state": is_initial_state,
            "eps": epsilon,
            "beta": beta,
            "beta_prev": beta_prev,
            "option_value": option_value,
            "gen_js_divergence": gjsd,
            "option_rollout_length": 1,
        }
        agent.storage.store(transition)
        graph = structure.graph.clone()
        agent._number_timesteps += 1
        prev_option = option_idx
        is_initial_state = False
        print()

    _update_and_record(agent, rec)

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{final_structure.story_level_sections = }")
    logger.info(f"Episode: {agent._number_episodes:4d}, fail name: {fail_name}, fail reason: {fail_reason}")

    rec.record_in_end(final_structure, env, testing=False)

    score = sum(env.reward_record)
    return score

def _flat_testing(agent: OptionCriticAgent,
                  env: Environment,
                  rec: Record,
                  logger: logging.Logger) -> float:
    structure = env.reset(testing=True)
    rec.record_in_beginning(structure, testing=True)
    graph = structure.graph.clone()
    score = 0
    timestep = 0
    done = False
    prev_option = 0
    is_initial_state = True
    while not done:
        original_structure = deepcopy(structure)
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        option_idx, option_value, beta, beta_prev, epsilon = agent.sample_option(state, prev_option, is_initial_state, greedy=True)
        gjsd, logits, value, beta, entropy, action, log_prob = agent.choose_action(state, structure.already_minimum_section_story_indexes, option_idx, greedy=True)
        print(f"{option_value - value = }")

        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"*****Testing Episode, timestep: {timestep+1:3d}, story_level_sections: {structure.story_level_sections}, option: {option_idx:2d}, beta: {beta:.4f}, prev_option: {prev_option:2d}, prev_beta: {beta_prev:.4f}, action: {action:3d} [{update_story}F {member_category}]")

        structure, reward, done, fail_name, fail_reason = env.step(structure, action)
        score += reward
        timestep += 1
        logger.info(f"*****Testing Episode, timestep: {timestep:3d}, option: {option_idx:2d}, reward: {reward:4f}, score: {score:.4f}")

        transition = {
            # Actor-Critic
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
            "aux": original_structure.aux,
            # Option-Critic
            "option_idx": option_idx,
            "prev_option": prev_option,
            "init_state": is_initial_state,
            "eps": epsilon,
            "beta": beta,
            "beta_prev": beta_prev,
            "option_value": option_value,
            "gen_js_divergence": gjsd,
            "option_rollout_length": 1,
        }
        agent.storage.store(transition)
        graph = structure.graph.clone()
        prev_option = option_idx
        is_initial_state = False
        print()

    _record_testing_rollout(agent, rec)

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{final_structure.story_level_sections = }")
    logger.info(f"Testing, fail name: {fail_name}, fail reason: {fail_reason}")

    rec.record_in_end(final_structure, env, testing=True)

    score = sum(env.reward_record)
    return score


def _hierarchical_training(agent: OptionCriticAgent,
                           env: Environment,
                           rec: Record,
                           logger: logging.Logger) -> float:
    structure = env.reset()
    rec.record_in_beginning(structure, testing=False)
    graph = structure.graph.clone()
    score = 0
    done = False
    prev_option = 0
    is_initial_state = True
    option_rollout_length = 0
    while not done:
        original_structure = deepcopy(structure)

        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        option_idx, option_value, beta, beta_prev, epsilon = agent.sample_option(state, prev_option, is_initial_state)
        gjsd, logits, value, beta, entropy, action, log_prob = agent.choose_action(state, structure.already_minimum_section_story_indexes, option_idx)
        
        if (option_idx != prev_option) or is_initial_state:
            initial_logits, initial_value, initial_beta, initial_action = logits, value, beta, action
            initial_gjsd, initial_entropy, initial_log_prob = gjsd, entropy, log_prob
            print(f"{option_value - initial_value = }")
        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"-----episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps+1:3d}, story_level_sections: {structure.story_level_sections}, option: {option_idx:2d}, beta: {beta:.4f}, prev_option: {prev_option:2d}, prev_beta: {beta_prev:.4f}, action: {action:3d} [{update_story}F {member_category}]")

        # cheap update (no analysis)            
        structure, _ = env.cheap_step(structure, action)
        graph = structure.graph.clone()
        agent._number_timesteps += 1
        option_rollout_length += 1 if (option_idx == prev_option) or is_initial_state else 0

        # expensive update (with analysis) when the same option rollout ends
        if (option_idx != prev_option) or np.all(np.array(structure.story_level_sections) == 0):
            structure, reward, done, fail_name, fail_reason = env.expensive_step(structure)
            score += reward
            logger.info(f"episode: {agent._number_episodes+1:4d}, timestep: {agent._number_timesteps:3d}, option: {option_idx:2d}, reward: {reward:4f}, score: {score:.4f}")

            transition = {
                # Actor-Critic
                "graph": original_structure.graph.clone(),
                "logits": initial_logits,
                "value": initial_value,
                "entropy": initial_entropy,
                "action": initial_action,
                "log_prob": initial_log_prob,
                "next_graph": structure.graph.clone(),
                "reward": reward,
                "done": done,
                "infeasible_actions": torch.tensor([True if i in original_structure.already_minimum_section_story_indexes else False for i in range(state.shape[0])], dtype=torch.bool),
                "aux": original_structure.aux,
                # Option-Critic
                "option_idx": option_idx,
                "prev_option": prev_option,
                "init_state": is_initial_state,
                "eps": epsilon,
                "beta": initial_beta,
                "beta_prev": beta_prev,
                "option_value": option_value,
                "gen_js_divergence": initial_gjsd,
                "option_rollout_length": option_rollout_length,
            }
            agent.storage.store(transition)
            graph = structure.graph.clone()
            option_rollout_length = 1

        prev_option = option_idx
        is_initial_state = False

    _update_and_record(agent, rec)

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{final_structure.story_level_sections = }")
    logger.info(f"Episode: {agent._number_episodes:4d}, fail name: {fail_name}, fail reason: {fail_reason}")

    rec.record_in_end(final_structure, env, testing=False)

    score = sum(env.reward_record)
    return score

def _hierarchical_testing(agent: OptionCriticAgent,
                          env: Environment,
                          rec: Record,
                          logger: logging.Logger) -> float:
    structure = env.reset(testing=True)
    rec.record_in_beginning(structure, testing=True)
    graph = structure.graph.clone()
    score = 0
    timestep = 0
    done = False
    prev_option = 0
    is_initial_state = True
    option_rollout_length = 0
    while not done:
        original_structure = deepcopy(structure)

        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn.forward(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)
        option_idx, option_value, beta, beta_prev, epsilon = agent.sample_option(state, prev_option, is_initial_state, greedy=True)
        gjsd, logits, value, beta, entropy, action, log_prob = agent.choose_action(state, structure.already_minimum_section_story_indexes, option_idx, greedy=True)
        
        if (option_idx != prev_option) or is_initial_state:
            initial_logits, initial_value, initial_beta, initial_action = logits, value, beta, action
            initial_gjsd, initial_entropy, initial_log_prob = gjsd, entropy, log_prob
            print(f"{option_value - initial_value = }")
        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        print(f"*****Testing Episode, timestep: {timestep+1:3d}, story_level_sections: {structure.story_level_sections}, option: {option_idx:2d}, beta: {beta:.4f}, prev_option: {prev_option:2d}, prev_beta: {beta_prev:.4f}, action: {action:3d} [{update_story}F {member_category}]")

        # cheap update (no analysis)            
        structure, _ = env.cheap_step(structure, action)
        graph = structure.graph.clone()
        timestep += 1
        option_rollout_length += 1 if (option_idx == prev_option) or is_initial_state else 0

        # expensive update (with analysis) when the same option rollout ends
        if (option_idx != prev_option) or np.all(np.array(structure.story_level_sections) == 0):
            structure, reward, done, fail_name, fail_reason = env.expensive_step(structure)
            score += reward
            logger.info(f"*****Testing, timestep: {timestep:3d}, option: {option_idx:2d}, reward: {reward:4f}, score: {score:.4f}")

            transition = {
                # Actor-Critic
                "graph": original_structure.graph.clone(),
                "logits": initial_logits,
                "value": initial_value,
                "entropy": initial_entropy,
                "action": initial_action,
                "log_prob": initial_log_prob,
                "next_graph": structure.graph.clone(),
                "reward": reward,
                "done": done,
                "infeasible_actions": torch.tensor([True if i in original_structure.already_minimum_section_story_indexes else False for i in range(state.shape[0])], dtype=torch.bool),
                "aux": original_structure.aux,
                # Option-Critic
                "option_idx": option_idx,
                "prev_option": prev_option,
                "init_state": is_initial_state,
                "eps": epsilon,
                "beta": initial_beta,
                "beta_prev": beta_prev,
                "option_value": option_value,
                "gen_js_divergence": initial_gjsd,
                "option_rollout_length": option_rollout_length,
            }
            agent.storage.store(transition)
            graph = structure.graph.clone()
            option_rollout_length = 1

        prev_option = option_idx
        is_initial_state = False

    _record_testing_rollout(agent, rec)

    final_structure = structure if fail_reason == "minimum_section" else original_structure
    logger.info("---> Constraint not satisfied, found optimal section at previous timestep")
    logger.info(f"{final_structure.story_level_sections = }")
    logger.info(f"Testing, fail name: {fail_name}, fail reason: {fail_reason}")

    rec.record_in_end(final_structure, env, testing=True)

    score = sum(env.reward_record)
    return score


def _update_and_record(agent: OptionCriticAgent, rec: Record):
    all_betas = []
    all_values = []
    returns = []
    advantages = []
    beta_advantages = []
    pred_values = []
    mean_kl_divergences = []
    ratios = []
    actor_losses, entropy_losses, critic_losses, beta_losses, gjsd_losses = [], [], [], [], []
    gnn_grad_norms, actor_grad_norms, critic_grad_norms, beta_grad_norms = [], [], [], []

    params_gnn_before = [p.clone().detach() for p in agent.gnn.parameters()]
    params_actor_before = [p.clone().detach() for p in agent.online_oc.fc_pi.parameters()]
    params_critic_before = [p.clone().detach() for p in agent.online_oc.option_critic.parameters()]
    params_beta_before = [p.clone().detach() for p in agent.online_oc.termination.parameters()]

    t_start = time.time()
    if agent.use_lr_scheduler:
        agent.lr_scheduler.step(agent._number_episodes)
    for epoch in range(agent.optimization_epoch):
        info = agent.step()
        all_betas.append(info["all_betas"])
        all_values.append(info["all_values"])
        returns.append(info["returns"])
        advantages.append(info["advantages"])
        beta_advantages.append(info["beta_advantages"])
        pred_values.append(info["pred_values"])
        mean_kl_divergences.append(info["mean_kl_divergence"])
        ratios.append(info["ratios"])
        actor_losses.append(info["actor_loss"])
        entropy_losses.append(info["entropy_loss"])
        critic_losses.append(info["critic_loss"])
        beta_losses.append(info["beta_loss"])
        gjsd_losses.append(info["gjsd_loss"])
        gnn_grad_norms.append(info["gnn_grad_norm"])
        actor_grad_norms.append(info["actor_grad_norm"])
        critic_grad_norms.append(info["critic_grad_norm"])
        beta_grad_norms.append(info["beta_grad_norm"])
        agent.logger.info(f"Optim epoch: {epoch+1:2d}, Loss - Actor: {info['actor_loss']:.4f}, Entropy: {info['entropy_loss']:.4f}, Critic: {info['critic_loss']:.4f}, Beta: {info['beta_loss']:.4f}, GJSD: {info['gjsd_loss']:.4f}")
        agent.logger.info(f"Grad norm - GNN: {info['gnn_grad_norm']:.4f}, Actor: {info['actor_grad_norm']:.4f}, Critic: {info['critic_grad_norm']:.4f}, Beta: {info['beta_grad_norm']:.4f}")
    t_end = time.time()
    print(f"Total optimization time: {t_end - t_start:.3f} seconds")

    params_gnn_after = [p.clone().detach() for p in agent.gnn.parameters()]
    params_actor_after = [p.clone().detach() for p in agent.online_oc.fc_pi.parameters()]
    params_critic_after = [p.clone().detach() for p in agent.online_oc.option_critic.parameters()]
    params_beta_after = [p.clone().detach() for p in agent.online_oc.termination.parameters()]
    param_change = lambda before, after: torch.tensor([(a - b).pow(2).sum() for b, a in zip(before, after)]).mean().sqrt().item()
    gnn_change = param_change(params_gnn_before, params_gnn_after)
    actor_change = param_change(params_actor_before, params_actor_after)
    critic_change = param_change(params_critic_before, params_critic_after)
    beta_change = param_change(params_beta_before, params_beta_after)
    agent.logger.info(f"Param change - GNN: {gnn_change:.6f}, Actor: {actor_change:.6f}, Critic: {critic_change:.6f}, Beta: {beta_change:.6f}")

    rec.returns["train"].append(returns)
    rec.advantages["train"].append(advantages)
    rec.beta_advantages["train"].append(beta_advantages)
    rec.entropies["train"].append(agent.storage.entropy)
    rec.pred_values.append(pred_values)
    rec.kl_divergences.append(mean_kl_divergences)
    rec.ratios.append(ratios)
    rec.losses["actor"].append(actor_losses)
    rec.losses["entropy"].append(entropy_losses)
    rec.losses["critic"].append(critic_losses)
    rec.losses["beta"].append(beta_losses)
    rec.losses["gjsd"].append(gjsd_losses)
    rec.grad_norms["gnn"].append(gnn_grad_norms)
    rec.grad_norms["actor"].append(actor_grad_norms)
    rec.grad_norms["critic"].append(critic_grad_norms)
    rec.grad_norms["beta"].append(beta_grad_norms)
    rec.param_changes["gnn"].append(gnn_change)
    rec.param_changes["actor"].append(actor_change)
    rec.param_changes["critic"].append(critic_change)
    rec.param_changes["beta"].append(beta_change)
    rec.option_indices["train"].append(agent.storage.option_idx)
    rec.option_values["train"].append(agent.storage.option_value)
    rec.gen_js_divergences["train"].append(agent.storage.gen_js_divergence)
    rec.option_rollout_lengths["train"].append(agent.storage.option_rollout_length)
    rec.all_betas["train"].append(all_betas)
    rec.all_values["train"].append(all_values)

    agent.storage.reset()
    agent._number_episodes += 1
    agent._number_timesteps = 0

def _record_testing_rollout(agent: OptionCriticAgent, rec: Record):
    info = agent.get_batched_info()
    all_betas = info["all_betas"]
    all_values = info["all_values"]
    values = info["values"]
    prev_betas = info["prev_betas"]
    mean_gjsd = info["gjsd"]
    log_probs = info["log_probs"]
    mean_entropy = info["entropy"]
    returns = info["returns"]
    advantages = info["advantages"]
    beta_advantages = info["beta_advantages"]

    rec.returns["test"].append(returns.detach().cpu().tolist())
    rec.advantages["test"].append(advantages.detach().cpu().tolist())
    rec.beta_advantages["test"].append(beta_advantages.detach().cpu().tolist())
    rec.entropies["test"].append(agent.storage.entropy)
    rec.option_indices["test"].append(agent.storage.option_idx)
    rec.option_values["test"].append(agent.storage.option_value)
    rec.gen_js_divergences["test"].append(agent.storage.gen_js_divergence)
    rec.option_rollout_lengths["test"].append(agent.storage.option_rollout_length)
    rec.all_betas["test"].append(all_betas.detach().cpu().tolist())
    rec.all_values["test"].append(all_values.detach().cpu().tolist())
    agent.storage.reset()


def train(agent: OptionCriticAgent,
          env: Environment,
          rec: Record,
          train_episode_num: int,
          test_frequency: int,
          test_episode_num: int,
          logger: logging.Logger):

    for i in range(train_episode_num):
        if agent.if_hierarchical:
            score = _hierarchical_training(agent, env, rec, logger)
        else:
            score = _flat_training(agent, env, rec, logger)
        logger.critical(f"Training Episode: {i+1}, score: {score:.3f}")
        logger.critical(f"Training Episode: {i+1}, saved_material: {sum(env.saved_material_record):.3f}, saved_material_SCWB: {sum(env.saved_material_record_SCWB):.3f}")
        logger.critical(f"Training Episode: {i+1}, total_reduction_amount: {env.material_usage_record[0] - env.material_usage_record[-1]:.3f}\n\n\n")

        if (i+1) % test_frequency == 0:
            for j in range(test_episode_num):
                if agent.if_hierarchical:
                    agent.logger.critical(f"Testing with learned option-selection policy...")
                    test_score = _hierarchical_testing(agent, env, rec, logger)
                    logger.critical(f"Testing Episode: {j+1}, Learned Option Policy")
                else:
                    test_score = _flat_testing(agent, env, rec, logger)
                    logger.critical(f"Testing Episode: {j+1}, Learned Option Policy")
                logger.critical(f"score: {test_score:.3f}")
                logger.critical(f"saved_material: {sum(env.saved_material_record):.3f}, saved_material_SCWB: {sum(env.saved_material_record_SCWB):.3f}")
                logger.critical(f"total_reduction_amount: {env.material_usage_record[0] - env.material_usage_record[-1]:.3f}\n\n\n")

            rec.output(env.checkpoint_dir)

            if len(rec.testing_record["final_volume"]) > 0:
                final_volumes = np.array(rec.testing_record["final_volume"])
                scores = np.array(rec.testing_record["score"])

                if np.argmin(final_volumes) == len(final_volumes)-1:
                    agent.save_model(env.checkpoint_dir/"models", name="MinimumUsage")
                if np.argmax(scores) == len(scores)-1:
                    agent.save_model(env.checkpoint_dir/"models", name="HighestScore")

            if (i+1) % 100 == 0:
                agent.save_model(env.checkpoint_dir/"models", name=f"Episode{str(i+1)}")
