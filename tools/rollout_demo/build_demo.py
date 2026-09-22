"""
Inline every Rollout JSON document into the page template, producing one self-contained rollout_demo.html.

Usage (repo root; no training environment needed):
    python tools/rollout_demo/build_demo.py
    python tools/rollout_demo/build_demo.py --data_dir <dir> --out <file.html>
"""
import json
import re
from argparse import ArgumentParser
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
TEMPLATE = TOOL_DIR / "template.html"
ALGORITHM_ORDER = {"DQN": 0, "PPO": 1}
PLACEHOLDER = re.compile(r"^const ROLLOUTS = .*;\r?$", re.M)


def script_json(obj) -> str:
    """JSON that is safe inside <script> (a '</script>' in a string must not end the tag)."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def load_rollouts(data_dir: Path) -> list:
    rollouts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(data_dir.glob("*.json"))]
    if not rollouts:
        raise SystemExit(f"no Rollout JSON documents in {data_dir}; run export_rollout.py first")
    return sorted(rollouts, key=lambda r: (ALGORITHM_ORDER.get(r["run"]["algorithm"], 9), r["run"]["name"]))


def render(rollouts: list, template: str) -> str:
    if not PLACEHOLDER.search(template):
        raise SystemExit("template has no 'const ROLLOUTS = ...;' line")
    return PLACEHOLDER.sub(lambda _: "const ROLLOUTS = " + script_json(rollouts) + ";", template, count=1)


def main() -> None:
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data_dir", type=Path, default=TOOL_DIR / "data")
    parser.add_argument("--out", type=Path, default=TOOL_DIR / "rollout_demo.html")
    args = parser.parse_args()
    rollouts = load_rollouts(args.data_dir)
    args.out.write_text(render(rollouts, TEMPLATE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {args.out} ({len(rollouts)} Rollouts: {', '.join(r['run']['algorithm'] for r in rollouts)})")


if __name__ == "__main__":
    main()
