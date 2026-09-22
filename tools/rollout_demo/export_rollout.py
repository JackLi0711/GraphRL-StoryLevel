"""
Run a Rollout of a trained Run on the Testing Geometry and write it as a Rollout JSON document.

Usage (repo root, in the training environment):
    python tools/rollout_demo/export_rollout.py                      # the default DQN and PPO Runs
    python tools/rollout_demo/export_rollout.py --run <Run dir> [--model <.pt>]

The agent and environment are built by the inference scripts themselves (load_agent_and_env),
so the exported Rollout is the one terminal inference would produce on the Testing Geometry.
"""
import json
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)  # the inference scripts resolve RL/, Visualization/ ... relative to the working directory
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOL_DIR))

import rollout_doc  # noqa: E402

DEFAULT_RUNS = [
    "Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/DQN_Experiment_Jack/2026_01_05__11_59_47__DouDQN_MatReward_SoftUpdate_LinearDecay010_Buffer10000_Batch256_Epoch1000",
    "Results/AdjustedMoreSections/RandomShape/OpenSees_RSA/2026_03_10__13_19_18__PPO_MatReward_UseGAE095_LR5e-4_ActLossCoef10_CriLossCoef001_EntroWei01to001_OptimEpoch5_Episode1000",
]
# [x_span_num, z_span_num, story_num, x_span_length, z_span_length, story_height]
TESTING_GEOMETRY = [4, 4, 6, 6000, 8000, 3200]


def detect_algorithm(train_args: dict) -> str:
    if "use_gae" in train_args:
        return "PPO"
    if "model_type" in train_args:
        return "DQN"
    raise ValueError("cannot tell whether this Run is DQN or PPO from train_args.json")


def load_agent_and_env(algorithm: str, run_dir: Path, model_path: Path):
    import inference
    import inference_PPO
    script = inference if algorithm == "DQN" else inference_PPO
    argv = sys.argv
    sys.argv = [script.__file__, "--ckpt_dir", str(run_dir), "--trained_model_path", str(model_path)]
    try:
        args = script.parse_args()
    finally:
        sys.argv = argv
    return script.load_agent_and_env(args)


def export(run_dir: Path, model_path: Path, out_dir: Path) -> Path:
    from RL import rollout
    from Structure.structure import Structure
    from Structure.sections import beam_sections, column_sections

    with open(run_dir / "train_args.json", "r") as f:
        algorithm = detect_algorithm(json.load(f))
    agent, env = load_agent_and_env(algorithm, run_dir, model_path)

    x_span_num, z_span_num, story_num, x_span_len, z_span_len, story_height = TESTING_GEOMETRY
    structure = Structure(
        x_span_num=x_span_num, x_span_lens=[x_span_len] * x_span_num,
        z_span_num=z_span_num, z_span_lens=[z_span_len] * z_span_num,
        story_num=story_num, story_height=story_height,
        story_level_sections=None,
        add_structure_geometry=env.add_structure_geometry,
        add_response_features=env.add_response_features,
        do_nonlinear_dynamic_analysis=env.do_nonlinear_dynamic_analysis,
        nda_norm_dict=env.nda_norm_dict,
        analysis_dir=env.checkpoint_dir / "Modal_Analysis",
    )
    env.init_records(structure)
    initial_structure = deepcopy(structure)
    steps = []
    for step in rollout.design_steps(agent, env, structure):
        steps.append(step)
        print(f"Design Step {step.index}: group {step.action}, passed={step.passed}")

    try:
        run_path = run_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        run_path = run_dir.as_posix()
    run = {"name": run_dir.name, "path": run_path, "algorithm": algorithm, "model_file": model_path.name}
    catalog = {"beam": [s["name"] for s in beam_sections], "column": [s["name"] for s in column_sections]}
    document = rollout_doc.build_document(run, rollout.agent_family(agent), initial_structure, steps, catalog)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algorithm.lower()}__{run_dir.name[:20]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=1)
    end = document["end"]
    print(f"wrote {out_path} ({len(steps)} Design Steps, end: {end['reason']}, Saving Ratio {end['saving_ratio']:.3f})")
    return out_path


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run", type=Path, action="append", help="Run directory; repeatable (default: the DQN and PPO demo Runs)")
    parser.add_argument("--model", type=Path, default=None, help="model file (default: <Run>/models/model_HighestScore.pt); only with a single --run")
    parser.add_argument("--out_dir", type=Path, default=TOOL_DIR / "data")
    args = parser.parse_args()
    runs = args.run or [Path(p) for p in DEFAULT_RUNS]
    if args.model is not None and len(runs) != 1:
        parser.error("--model needs exactly one --run")
    for run_dir in runs:
        run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
        model_path = args.model or run_dir / "models" / "model_HighestScore.pt"
        export(run_dir, model_path, args.out_dir.resolve())


if __name__ == "__main__":
    main()
