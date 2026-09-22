"""
Rollout: run a trained agent greedily on one structure, one Design Step at a time.

`design_steps()` is the single Rollout loop shared by every consumer
(matplotlib inference output, the Rollout demo exporter, and a future live backend; see ADR-0004).
It never prints; consumers decide what to show.
"""
import time
import typing
import numpy as np
import torch

from copy import deepcopy
from dataclasses import dataclass, field

from RL import agent_DQN, agent_PPO, agent_OC, agent_OC_m3


@dataclass
class DesignStep:
    index: int
    # decision (filled before env.step)
    structure_before: typing.Any        # deepcopy of the structure before this step
    raw_preference: np.ndarray          # Q values (DQN) or policy logits (PPO / OC), one per story-level member group
    vis_values: np.ndarray              # values used to colour the legacy matplotlib frame
    infeasible: typing.List[int]        # group indexes that could not be selected
    action: int
    option: typing.Optional[int] = None  # Option-Critic only
    choose_time: float = 0.0
    # outcome (filled after env.step)
    structure_after: typing.Any = None  # the live structure after env.step (keeps changing in later steps)
    sections_after: typing.List[int] = field(default_factory=list)
    volume_after: float = 0.0           # m3
    reward: float = 0.0
    passed: bool = True
    done: bool = False
    fail_name: typing.Optional[str] = None
    fail_reason: typing.Optional[str] = None
    check_detail: typing.Optional[dict] = None
    step_time: float = 0.0


def agent_family(agent) -> str:
    """'DQN', 'PPO', 'OC' or 'OC_m3'."""
    if isinstance(agent, agent_DQN.DeepQAgent):
        return "DQN"
    if isinstance(agent, agent_PPO.PPOAgent):
        return "PPO"
    if isinstance(agent, agent_OC.OptionCriticAgent):
        return "OC"
    if isinstance(agent, agent_OC_m3.OptionCriticAgent):
        return "OC_m3"
    raise ValueError("agent type not supported.")


def _infeasible_actions(agent, structure) -> typing.List[int]:
    dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else []
    return list(set(structure.already_minimum_section_story_indexes + dont_select_story_member_indexes))


def _decide(agent, family: str, state, infeasible_actions):
    """Greedy decision -> (action, option, raw_preference, vis_values)."""
    option_idx = None
    if family == "DQN":
        action, _ = agent.choose_action(state, infeasible_actions, greedy=True)
        q_values = agent.online_q_network(state).squeeze().detach().cpu().numpy()
        vis_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))  # noamalize q_values to [0, 1]
        raw = q_values
    elif family == "PPO":
        logits, _, _, action, _ = agent.choose_action(state, infeasible_actions, greedy=True)
        vis_values = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()  # use action probabilities as vis values
        raw = logits.squeeze().detach().cpu().numpy()
    elif family == "OC":
        option_idx = 0
        _, logits, _, _, action, _ = agent.choose_action(state, infeasible_actions, option_idx, greedy=True)
        vis_values = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
        raw = logits.squeeze().detach().cpu().numpy()
    else:  # OC_m3
        option_idx, _ = agent.choose_option(state, greedy=True)
        _, logits, _, _, action, _ = agent.choose_action(state, infeasible_actions, option_idx, greedy=True)
        vis_values = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
        raw = logits.squeeze().detach().cpu().numpy()
    return int(action), option_idx, np.asarray(raw, dtype=float), vis_values


def design_steps(agent, env, structure,
                 before_step: typing.Optional[typing.Callable[[DesignStep, typing.Any], None]] = None
                 ) -> typing.Iterator[DesignStep]:
    """
    Yield one DesignStep per step until the Rollout is done.

    `env.init_records(structure)` must already have been called.
    `before_step(step, structure)` runs after the decision and before env.step,
    so consumers can draw the pre-step structure (the legacy frames do this).
    """
    family = agent_family(agent)
    graph = structure.graph.clone()
    done = False
    index = 0
    while not done:
        structure_before = deepcopy(structure)
        # go through gnn and get state
        with torch.no_grad():
            graph = graph.to(agent.device)
            state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(agent.device), None)

        t_start = time.time()
        infeasible_actions = _infeasible_actions(agent, structure)
        action, option_idx, raw, vis_values = _decide(agent, family, state, infeasible_actions)
        choose_time = time.time() - t_start

        step = DesignStep(index=index, structure_before=structure_before, raw_preference=raw, vis_values=vis_values,
                          infeasible=sorted(int(i) for i in infeasible_actions), action=action, option=option_idx,
                          choose_time=choose_time)
        if before_step is not None:
            before_step(step, structure)

        # update structure
        t_start = time.time()
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)
        step.step_time = time.time() - t_start
        step.structure_after = structure
        step.sections_after = [int(s) for s in structure.story_level_sections]
        step.volume_after = float(structure.calculate_material_usage())
        step.reward = reward
        step.passed = bool(getattr(env, "last_step_passed", True))
        step.fail_name = fail_name
        step.fail_reason = fail_reason
        step.check_detail = None if step.passed else getattr(env, "last_check_detail", None)

        # check if can't select anymore
        if len(_infeasible_actions(agent, structure)) >= len(structure.story_level_actions):
            done = True
        step.done = bool(done)
        graph = structure.graph.clone()
        index += 1
        yield step
