"""Tests for build_index.read_args: the json-vs-log consistency check (spec C2).

執行（repo 根目錄）:
    python -m unittest discover -s tools/checkpoint_index/tests -p "test_*.py"

只用暫存目錄裡造出來的假 Run，不讀 Results/。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_index  # noqa: E402


def make_run(root: Path, json_args=None, namespace=None) -> Path:
    run = root / "2026_01_01__00_00_00__TestRun"
    run.mkdir()
    lines = ["2026-01-01 00:00:00,000 - Graph-RL - CRITICAL - Results/x\n"]
    if namespace is not None:
        lines.append("2026-01-01 00:00:00,000 - Graph-RL - CRITICAL - " + namespace + "\n")
    lines.append("2026-01-01 00:00:00,000 - Graph-RL - CRITICAL - Device: Tesla V100-SXM2-32GB\n")
    (run / "record.log").write_text("".join(lines), encoding="utf-8")
    if json_args is not None:
        (run / "train_args.json").write_text(json.dumps(json_args), encoding="utf-8")
    return run


class ReadArgsConsistencyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_matching_sources_produce_no_warning(self):
        run = make_run(self.root,
                       json_args={"lr": 0.0005, "gamma": 0.99, "comment": ["a", "b"]},
                       namespace="Namespace(lr=0.0005, gamma=0.99, comment=['a', 'b'])")
        args, source, warnings = build_index.read_args(run)
        self.assertEqual(source, "json+log")
        self.assertEqual(warnings, [])

    def test_differing_value_is_reported(self):
        run = make_run(self.root,
                       json_args={"lr": 0.001, "gamma": 0.99},
                       namespace="Namespace(lr=0.0005, gamma=0.99)")
        args, source, warnings = build_index.read_args(run)
        self.assertEqual(len(warnings), 1)
        self.assertIn("lr", warnings[0])
        self.assertEqual(args["lr"], 0.001, "train_args.json must win over the log")

    def test_ckpt_dir_difference_is_not_reported(self):
        # Filing 搬移 Run 後兩邊記下的路徑不同是預期中的事（ADR-0002）
        run = make_run(self.root,
                       json_args={"lr": 0.0005, "ckpt_dir": "Results/OpenSees_RSA/OC_Experiment/run"},
                       namespace="Namespace(lr=0.0005, ckpt_dir=PosixPath('Results/OpenSees_RSA/run'))")
        _, _, warnings = build_index.read_args(run)
        self.assertEqual(warnings, [])

    def test_multiple_mismatches_are_reported_in_stable_order(self):
        run = make_run(self.root,
                       json_args={"lr": 0.001, "gamma": 0.9, "hidden_dim": 100},
                       namespace="Namespace(lr=0.0005, gamma=0.99, hidden_dim=100)")
        _, _, warnings = build_index.read_args(run)
        self.assertEqual([w.split(":")[1].split()[0] for w in warnings], ["gamma", "lr"])

    def test_log_only_run_has_no_consistency_warning(self):
        run = make_run(self.root, namespace="Namespace(lr=0.0005, ckpt_dir=PosixPath('x'))")
        args, source, warnings = build_index.read_args(run)
        self.assertEqual(source, "log")
        self.assertEqual(args["ckpt_dir"], "x")
        self.assertEqual(warnings, [])


class OperatorRulesTest(unittest.TestCase):
    """Operator = 地點-人（CONTEXT.md: Operator，ADR-0003）。"""

    def q(self, operator, device, date="2026_03_01"):
        return build_index.qualify_operator(operator, device, date)

    def test_local_machine_is_always_local_jack(self):
        self.assertEqual(self.q("Kyle", "RTX4080"), "Local-Jack")
        self.assertEqual(self.q("unknown", "RTX4080", "2025_05_01"), "Local-Jack")

    def test_server_keeps_person_from_folder(self):
        self.assertEqual(self.q("Kyle", "V100", "2026_05_01"), "Server-Kyle")
        self.assertEqual(self.q("Jack", "RTX3080", "2025_05_01"), "Server-Jack")

    def test_unfiled_server_run_inferred_by_year(self):
        self.assertEqual(self.q("unknown", "V100", "2025_12_31"), "Server-Kyle")
        self.assertEqual(self.q("unknown", "V100", "2024_01_01"), "Server-Kyle")
        self.assertEqual(self.q("unknown", "V100", "2026_01_01"), "Server-Jack")
        # 以年份為界，不是逐年列舉：2027 以後仍為 Jack
        self.assertEqual(self.q("unknown", "RTX3080", "2027_06_01"), "Server-Jack")

    def test_unfiled_server_run_without_date_stays_unknown(self):
        self.assertEqual(self.q("unknown", "V100", None), "Server-unknown")

    def test_unrecognised_device_is_not_guessed(self):
        self.assertEqual(self.q("Jack", "A100"), "Unknown-Jack")
        self.assertEqual(self.q("unknown", "A100", "2025_01_01"), "Unknown-unknown")
        self.assertEqual(self.q("Kyle", "unknown"), "Unknown-Kyle")

    def test_unrecognised_device_produces_warning(self):
        self.assertEqual(build_index.device_warnings("V100"), [])
        self.assertEqual(build_index.device_warnings("RTX4080"), [])
        [w] = build_index.device_warnings("NVIDIA A100")
        self.assertIn("NVIDIA A100", w)

    def test_every_known_device_has_a_location(self):
        for full, (short, location) in build_index.DEVICES.items():
            self.assertIn(location, ("Server", "Local"), full)
            self.assertEqual(build_index.device_warnings(short), [])

    def test_read_device_maps_full_name_to_short_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp))  # make_run 寫入 Tesla V100-SXM2-32GB
            self.assertEqual(build_index.read_device(run), "V100")


class EmbedJsonTest(unittest.TestCase):
    def test_script_close_tag_cannot_end_the_script_block(self):
        out = build_index.embed_json({"comment": "</script><script>alert(1)</script>"})
        self.assertNotIn("</script>", out)
        self.assertEqual(json.loads(out)["comment"], "</script><script>alert(1)</script>")

    def test_non_ascii_kept_readable(self):
        self.assertIn("附註", build_index.embed_json({"note": "附註"}))


if __name__ == "__main__":
    unittest.main()
