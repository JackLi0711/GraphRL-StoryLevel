"""
Turn a Rollout (a list of RL.rollout.DesignStep) into the Rollout JSON document the demo page replays.

The document format is the contract between Python and the page (ADR-0004); bump SCHEMA_VERSION on breaking changes.
This module only reads plain attributes of the structure, so tests can feed it stubs.
"""
import numpy as np

SCHEMA_VERSION = 1
PREFERENCE_KIND = {"DQN": "q_value", "PPO": "action_probability"}
CATEGORY_KIND = {"xdir_beam": "beam", "zdir_beam": "beam", "outer_column": "column", "inner_column": "column"}


def _preference(family: str, raw, infeasible) -> list:
    """Action Preference per group: Q value (DQN) or masked action probability (PPO); None for infeasible groups."""
    raw = np.asarray(raw, dtype=float).reshape(-1)
    feasible = np.array([i not in infeasible for i in range(raw.shape[0])])
    if family == "PPO":
        values = np.full(raw.shape, np.nan)
        if feasible.any():
            logits = raw[feasible]
            exp = np.exp(logits - logits.max())
            values[feasible] = exp / exp.sum()
    else:
        values = raw
    return [float(v) if ok else None for v, ok in zip(values, feasible)]


def _topology(structure) -> dict:
    node_names = sorted(structure.node_coord_dict.keys(), key=lambda name: int(name[1:]))
    member_group = {}
    for group, members in enumerate(structure.story_level_actions):
        for member_index in members:
            member_group[int(member_index)] = group
    members = []
    for member_name in sorted(structure.member_to_nodeIndex_dict.keys(), key=lambda name: int(name[1:])):
        node1, node2 = structure.member_to_nodeIndex_dict[member_name][:2]
        members.append([int(node1), int(node2), member_group.get(int(member_name[1:]) - 1)])
    return {
        "nodes": [[float(c) for c in structure.node_coord_dict[name]] for name in node_names],
        "members": members,  # [node index, node index, group index or None]
    }


def _saving_ratio(initial_volume: float, volume: float) -> float:
    return (initial_volume - volume) / initial_volume


def build_document(run: dict, family: str, initial_structure, steps: list, section_catalog: dict) -> dict:
    """
    run: {"name", "path", "algorithm", "model_file"}
    initial_structure: the structure before the first Design Step
    section_catalog: {"beam": [designation, ...], "column": [designation, ...]}, index-aligned with section indexes
    """
    if family not in PREFERENCE_KIND:
        raise ValueError(f"Rollout documents support DQN and PPO only, got {family}")
    s = initial_structure
    initial_sections = [int(v) for v in s.story_level_sections]
    initial_volume = float(s.calculate_material_usage())

    records = []
    for step in steps:
        infeasible = [int(i) for i in step.infeasible]
        records.append({
            "index": int(step.index),
            "preference": _preference(family, step.raw_preference, infeasible),
            "infeasible": infeasible,
            "action": int(step.action),
            "section_before": int(step.structure_before.story_level_sections[step.action]),
            "section_after": int(step.sections_after[step.action]),
            "sections_after": [int(v) for v in step.sections_after],
            "volume_m3_after": float(step.volume_after),
            "saving_ratio_after": _saving_ratio(initial_volume, step.volume_after) if step.passed else None,
            "passed": bool(step.passed),
            "rejection": None if step.passed else _rejection(step),
        })

    passed = [r for r in records if r["passed"]]
    final_sections = passed[-1]["sections_after"] if passed else initial_sections
    final_volume = passed[-1]["volume_m3_after"] if passed else initial_volume
    if records and not records[-1]["passed"]:
        reason = "rejected"
    elif sum(final_sections) == 0:
        reason = "minimum_section"
    else:
        reason = "no_feasible_action"

    return {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "geometry": {
            "x_span_num": int(s.x_span_num), "z_span_num": int(s.z_span_num), "story_num": int(s.story_num),
            "x_span_lens": [float(v) for v in s.x_span_lens], "z_span_lens": [float(v) for v in s.z_span_lens],
            "story_height": float(s.story_height),
        },
        "topology": _topology(s),
        "groups": [
            {"story": g % s.story_num + 1, "category": category, "kind": CATEGORY_KIND[category]}
            for g, category in enumerate(s.story_level_categories)
        ],
        "section_catalog": section_catalog,
        "preference_kind": PREFERENCE_KIND[family],
        "initial": {"sections": initial_sections, "volume_m3": initial_volume},
        "steps": records,
        "end": {
            "reason": reason,
            "final_sections": final_sections,
            "final_volume_m3": final_volume,
            "saving_ratio": _saving_ratio(initial_volume, final_volume),
        },
    }


def _rejection(step) -> dict:
    detail = step.check_detail or {}
    return {
        "check": detail.get("check", step.fail_reason),
        "load_case": detail.get("load_case", step.fail_name),
        "value": detail.get("value"),
        "limit": detail.get("limit"),
        "comparator": detail.get("comparator"),
    }
