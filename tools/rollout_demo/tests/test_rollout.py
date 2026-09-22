"""Seam 1: the Rollout generator (RL.rollout.design_steps) and the Rollout JSON document built from it.

執行（repo 根目錄，訓練用的 Python 環境）:
    python -m unittest discover -s tools/rollout_demo/tests -p "test_*.py"

agent / env / structure 全部是 stub：不載入訓練好的模型、不跑 OpenSees、不讀 Results/。
"""
import json
import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "rollout_demo"))

from RL import agent_DQN, agent_PPO, rollout  # noqa: E402
import rollout_doc  # noqa: E402

STORIES = 2
CATEGORIES = ["xdir_beam", "zdir_beam", "outer_column", "inner_column"]
GROUPS = STORIES * len(CATEGORIES)
CATALOG = {"beam": [f"B{i}" for i in range(4)], "column": [f"C{i}" for i in range(4)]}


class _Passthrough:
    """Stands in for torch_geometric graphs and batch tensors: .clone()/.to() return itself."""
    x = edge_index = edge_attr = None

    def clone(self):
        return self

    def to(self, device):
        return self


class StubStructure:
    """Two stories, one member per group; group g has volume = section index + 1 (m3)."""

    def __init__(self, sections):
        self.story_level_sections = list(sections)
        self.story_num = STORIES
        self.x_span_num, self.z_span_num = 1, 1
        self.x_span_lens, self.z_span_lens = [6000], [8000]
        self.story_height = 3200
        self.story_level_categories = [c for c in CATEGORIES for _ in range(STORIES)]
        self.story_level_actions = [[g] for g in range(GROUPS)]
        self.node_coord_dict = {f"N{i + 1}": (float(i), 0.0, 0.0) for i in range(GROUPS + 1)}
        self.member_to_nodeIndex_dict = {f"E{g + 1}": [g, g + 1] for g in range(GROUPS)}
        self.graph = _Passthrough()
        self.aux = {"story_batch": _Passthrough()}

    @property
    def already_minimum_section_story_indexes(self):
        return [g for g, s in enumerate(self.story_level_sections) if s == 0]

    def restrict_action_space(self):
        return []

    def calculate_material_usage(self):
        return float(sum(s + 1 for s in self.story_level_sections))


class StubEnv:
    """Reduces the chosen group by one; `failing_steps` maps a step index to a check detail that fails it."""

    def __init__(self, failing_steps=None):
        self.failing_steps = failing_steps or {}
        self.calls = 0

    def step(self, structure, action):
        structure.story_level_sections[action] -= 1
        detail = self.failing_steps.get(self.calls)
        self.calls += 1
        self.last_step_passed = detail is None
        self.last_check_detail = detail
        if detail is not None:
            return structure, -1.0, True, detail["load_case"], detail["check"]
        if sum(structure.story_level_sections) == 0:
            return structure, 1.0, True, None, "minimum_section"
        return structure, 1.0, False, None, None


def _scores(structure):
    """Deterministic preference: favour the group with the largest section, ties to the lowest index."""
    return torch.tensor([float(s) * 10 - g * 0.1 for g, s in enumerate(structure.story_level_sections)])


class StubDQN(agent_DQN.DeepQAgent):
    def __init__(self):  # no network, no buffer
        self.device = "cpu"
        self.restrict_action = False
        self.structure = None
        self.gnn = lambda *args: None

    def choose_action(self, state, infeasible_actions, greedy=False):
        q = self.online_q_network(state).clone()
        q[infeasible_actions] = -math.inf
        return int(q.argmax()), float(q.max())

    def online_q_network(self, state):
        return _scores(self.structure)


class StubPPO(agent_PPO.PPOAgent):
    def __init__(self):
        self.device = "cpu"
        self.restrict_action = False
        self.structure = None
        self.gnn = lambda *args: None

    def choose_action(self, state, infeasible_actions, greedy=False):
        logits = _scores(self.structure)
        masked = logits.clone()
        masked[infeasible_actions] = -math.inf
        return logits, 0.0, 0.0, int(masked.argmax()), 0.0


def run_rollout(agent, env, sections):
    structure = StubStructure(sections)
    agent.structure = structure  # the stub agent reads the live structure instead of a GNN state
    initial = StubStructure(sections)
    steps = list(rollout.design_steps(agent, env, structure))
    run = {"name": "2026_01_01__00_00_00__Stub", "path": "Results/Stub", "algorithm": "DQN", "model_file": "m.pt"}
    return steps, rollout_doc.build_document(run, rollout.agent_family(agent), initial, steps, CATALOG)


DRIFT = {"check": "drift_ratio", "load_case": "EQX+", "value": 0.0061, "limit": 0.005, "comparator": ">"}


class RejectedRolloutTest(unittest.TestCase):
    def setUp(self):
        # all groups at section 1; the 3rd Design Step fails the drift check
        self.steps, self.doc = run_rollout(StubDQN(), StubEnv({2: DRIFT}), [1] * GROUPS)
        self.records = self.doc["steps"]

    def test_stops_at_the_rejected_step(self):
        self.assertEqual(len(self.records), 3)
        self.assertEqual([r["passed"] for r in self.records], [True, True, False])

    def test_rejected_step_carries_the_check_detail(self):
        self.assertEqual(self.records[-1]["rejection"], DRIFT)
        self.assertIsNone(self.records[-1]["saving_ratio_after"])
        self.assertIsNone(self.records[0]["rejection"])

    def test_final_design_is_the_one_before_the_rejected_step(self):
        end = self.doc["end"]
        self.assertEqual(end["reason"], "rejected")
        self.assertEqual(end["final_sections"], self.records[1]["sections_after"])
        self.assertAlmostEqual(end["final_volume_m3"], self.records[1]["volume_m3_after"])
        self.assertAlmostEqual(end["saving_ratio"], self.records[1]["saving_ratio_after"])

    def test_saving_ratio_is_relative_to_the_initial_volume(self):
        initial = self.doc["initial"]["volume_m3"]
        self.assertEqual(initial, 2.0 * GROUPS)
        for r in self.records[:2]:
            self.assertAlmostEqual(r["saving_ratio_after"], (initial - r["volume_m3_after"]) / initial)

    def test_chosen_group_and_its_sections(self):
        first = self.records[0]
        self.assertEqual(first["action"], 0)
        self.assertEqual((first["section_before"], first["section_after"]), (1, 0))
        self.assertEqual(first["sections_after"][0], 0)

    def test_minimum_section_groups_are_infeasible_with_no_preference(self):
        second = self.records[1]
        self.assertIn(0, second["infeasible"])
        self.assertIsNone(second["preference"][0])
        self.assertTrue(all(v is not None for g, v in enumerate(second["preference"]) if g not in second["infeasible"]))

    def test_dqn_preference_is_raw_q_value(self):
        self.assertEqual(self.doc["preference_kind"], "q_value")
        self.assertAlmostEqual(self.records[0]["preference"][3], 10 - 0.3, places=5)


class MinimumSectionRolloutTest(unittest.TestCase):
    def setUp(self):
        self.steps, self.doc = run_rollout(StubPPO(), StubEnv(), [1] * GROUPS)

    def test_ends_at_minimum_section_without_rejection(self):
        self.assertEqual(len(self.doc["steps"]), GROUPS)
        self.assertTrue(all(r["passed"] for r in self.doc["steps"]))
        end = self.doc["end"]
        self.assertEqual(end["reason"], "minimum_section")
        self.assertEqual(end["final_sections"], [0] * GROUPS)
        self.assertAlmostEqual(end["saving_ratio"], 0.5)

    def test_ppo_preference_is_masked_action_probability(self):
        self.assertEqual(self.doc["preference_kind"], "action_probability")
        for r in self.doc["steps"]:
            feasible = [v for v in r["preference"] if v is not None]
            self.assertAlmostEqual(sum(feasible), 1.0)
            self.assertEqual(max(range(GROUPS), key=lambda g: -1 if r["preference"][g] is None else r["preference"][g]), r["action"])

    def test_generator_marks_only_the_last_step_done(self):
        self.assertEqual([s.done for s in self.steps], [False] * (GROUPS - 1) + [True])


class DocumentShapeTest(unittest.TestCase):
    def setUp(self):
        _, self.doc = run_rollout(StubDQN(), StubEnv({0: DRIFT}), [2] * GROUPS)

    def test_top_level_fields_and_json_round_trip(self):
        for key in ["schema_version", "run", "geometry", "topology", "groups", "section_catalog",
                    "preference_kind", "initial", "steps", "end"]:
            self.assertIn(key, self.doc)
        self.assertEqual(self.doc["schema_version"], 1)
        self.assertEqual(json.loads(json.dumps(self.doc, allow_nan=False)), self.doc)

    def test_groups_are_story_and_category_in_action_order(self):
        groups = self.doc["groups"]
        self.assertEqual(len(groups), GROUPS)
        self.assertEqual(groups[0], {"story": 1, "category": "xdir_beam", "kind": "beam"})
        self.assertEqual(groups[1], {"story": 2, "category": "xdir_beam", "kind": "beam"})
        self.assertEqual(groups[-1], {"story": 2, "category": "inner_column", "kind": "column"})

    def test_topology_links_members_to_groups(self):
        topo = self.doc["topology"]
        self.assertEqual(len(topo["nodes"]), GROUPS + 1)
        self.assertEqual(topo["members"][3], [3, 4, 3])

    def test_rejected_first_step_keeps_the_initial_design(self):
        end = self.doc["end"]
        self.assertEqual(end["final_sections"], [2] * GROUPS)
        self.assertEqual(end["saving_ratio"], 0.0)

    def test_other_agent_families_are_refused(self):
        with self.assertRaises(ValueError):
            rollout_doc.build_document({}, "OC_m3", StubStructure([1] * GROUPS), [], CATALOG)


if __name__ == "__main__":
    unittest.main()
