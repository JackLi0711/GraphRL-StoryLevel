"""Build a searchable index of training Runs under Results/.

READ-ONLY: 這支程式只讀 Results/ 底下的檔案，絕不寫入、改名或搬移任何東西。
輸出只有本目錄下的兩個衍生檔：index.json（索引資料）與 index.html（內嵌資料的單檔檢視頁面）。

用法:
    python tools/checkpoint_index/build_index.py
    python tools/checkpoint_index/build_index.py --scope <path> --out <file>
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key_map import canonicalize, NON_PARAMETER_KEYS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE = "Results/AdjustedMoreSections/RandomShape/OpenSees_RSA"
DEFAULT_OUT = Path(__file__).resolve().parent / "index.json"

# Convergence Gap 超過此值視為未收斂 (ADR-0001)
UNCONVERGED_GAP = 0.1

# record.log 裡的 GPU 全名 -> (短名, 訓練地點)。新增機器時只改這裡。
# 地點決定 Operator 的前綴（CONTEXT.md: Operator）：Local 一律是 Jack 的本機。
DEVICES = {
    "Tesla V100-SXM2-32GB":    ("V100",    "Server"),
    "NVIDIA GeForce RTX 3080": ("RTX3080", "Server"),
    "NVIDIA GeForce RTX 4080": ("RTX4080", "Local"),
}
DEVICE_LOCATION = {short: location for short, location in DEVICES.values()}

# 從這一年起 Jack 的 server 帳號才有可運作的 OpenSees 環境；之前的 server Run 都在 Kyle 帳號下（ADR-0003）
JACK_SERVER_ACCOUNT_FROM_YEAR = 2026


# --------------------------------------------------------------------------
# args: 兩種來源
# --------------------------------------------------------------------------

def _log_head(run: Path, lines: int = 31) -> str:
    out = []
    with open(run / "record.log", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            out.append(line)
            if i >= lines:
                break
    return "".join(out)


def _parse_namespace(text: str) -> dict:
    """Parse the Namespace(...) repr that argparse dumps into the log.

    Namespace 的 repr 完整落在單獨一行；必須只取那一行，
    把後續的 log 一起餵給 ast.parse 會是語法錯誤。
    """
    line = next(l for l in text.splitlines() if "Namespace(" in l)
    src = line[line.index("Namespace("):].rstrip()
    parsed = {}
    for kw in ast.parse(src, mode="eval").body.keywords:
        try:
            parsed[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            # PosixPath('x') 等建構式呼叫不是字面量，取出其中的字串
            raw = ast.unparse(kw.value)
            m = re.match(r"(?:Posix|Windows)Path\('(.*)'\)$", raw)
            parsed[kw.arg] = m.group(1) if m else raw
    return parsed


def read_args(run: Path):
    """Return (args, source, warnings). train_args.json 優先。"""
    warnings = []
    json_args = None
    ns_args = None

    tj = run / "train_args.json"
    if tj.exists():
        json_args = json.loads(tj.read_text(encoding="utf-8"))

    head = _log_head(run)
    if "Namespace(" in head:
        try:
            ns_args = _parse_namespace(head)
        except Exception as exc:  # noqa: BLE001
            warnings.append("Namespace 解析失敗: {}".format(exc))

    if json_args is not None and ns_args is not None:
        # 兩種來源都有的 Run 是唯一能驗證 Namespace 解析器正確性的樣本；不一致會顯示在檢視頁面上。
        # ckpt_dir 不比對：Filing 會搬移 Run，json 與 log 各自記下的路徑本來就可能不同，
        # 而且兩者都不是 Run 的身分（ADR-0002），差異不代表資料有問題。
        for k in sorted((set(json_args) & set(ns_args)) - {"ckpt_dir"}):
            if json_args[k] != ns_args[k]:
                warnings.append(
                    "args 來源不一致: {} json={!r} log={!r}".format(
                        k, json_args[k], ns_args[k]))
        return json_args, "json+log", warnings
    if json_args is not None:
        return json_args, "json", warnings
    if ns_args is not None:
        return ns_args, "log", warnings
    warnings.append("找不到任何 args 來源")
    return {}, "none", warnings


def detect_algorithm(args: dict) -> str:
    if "option_num" in args:
        return "OC"
    if "clip_eps" in args or "gae_tau" in args:
        return "PPO"
    if "buffer_size" in args or "per_alpha" in args:
        return "DQN"
    return "unknown"


def read_device(run: Path) -> str:
    m = re.search(r"Device: (.+)", _log_head(run))
    if not m:
        return "unknown"
    full = m.group(1).strip()
    return DEVICES[full][0] if full in DEVICES else full


# --------------------------------------------------------------------------
# Filing: purpose / operator (見 CONTEXT.md)
# --------------------------------------------------------------------------

def parse_filing(rel_parent: str):
    """rel_parent 是 Run 相對於 scope 的所在資料夾（'' 表示直接位於 scope 底下）。"""
    if not rel_parent:
        # 未經 Filing：沒有資料夾標籤可讀，因此 operator 不明
        return "Unfiled", "unknown"
    parts = rel_parent.split("_")
    if "Experiment" in parts:
        purpose = "Experiment"
    elif "Reproduction" in parts:
        purpose = "Reproduction"
    else:
        purpose = "Unfiled"
    # 人名為最後一段；沒有人名後綴的分組資料夾是 Jack 自己建立的（CONTEXT.md: Operator）
    tail = parts[-1]
    operator = "Jack" if tail in ("Experiment", "Reproduction") else tail
    return purpose, operator


def qualify_operator(operator: str, device: str, date) -> str:
    """依訓練地點替 operator 加上前綴（地點由 DEVICES 決定）。

    Local  -> Local-Jack（不論資料夾標籤）
    Server -> Server-{operator}；沒有資料夾人名可讀的 Run 依年份推定帳號：
              JACK_SERVER_ACCOUNT_FROM_YEAR 以前為 Kyle，之後（含）為 Jack（ADR-0003）
    無法辨識的機器 -> Unknown-{operator}，不猜 server 或 local（另見 device_warnings）
    """
    location = DEVICE_LOCATION.get(device)
    if location == "Local":
        return "Local-Jack"
    if location == "Server":
        if operator == "unknown" and date:
            operator = "Kyle" if int(date[:4]) < JACK_SERVER_ACCOUNT_FROM_YEAR else "Jack"
        return "Server-{}".format(operator)
    return "Unknown-{}".format(operator)


def device_warnings(device: str) -> list:
    if device in DEVICE_LOCATION:
        return []
    return ["無法辨識的 device「{}」：不知道是 server 還是 local，operator 標為 Unknown-…；"
            "請在 build_index.py 的 DEVICES 加入此機器".format(device)]


# --------------------------------------------------------------------------
# 指標 (CONTEXT.md: Saving Ratio / Convergence Gap, ADR-0001)
# --------------------------------------------------------------------------

def read_metrics(run: Path):
    path = run / "testing_record.txt"
    if not path.exists():
        return {}, ["缺少 testing_record.txt"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    iv, fv = rec["initial_volume"], rec["final_volume"]
    ratios = [(a - b) / a for a, b in zip(iv, fv)]
    if not ratios:
        return {}, ["testing_record.txt 沒有測試點"]

    best_i = max(range(len(ratios)), key=ratios.__getitem__)
    geom = rec["geometry"][best_i]
    # 先捨入再相減，否則 UI 顯示的 gap 會對不上 best - last 的最後一位
    best = round(ratios[best_i], 5)
    last = round(ratios[-1], 5)
    gap = round(best - last, 5)

    return {
        "test_points": len(ratios),
        "best": best,
        "last": last,
        "gap": gap,
        "unconverged": gap > UNCONVERGED_GAP,
        "best_index": best_i,
        "score_best": round(rec["score"][best_i], 5),
        "initial_volume": round(iv[best_i], 4),
        "final_volume": round(fv[best_i], 4),
        "story_num": geom[2],
        "initial_design": rec["initial_design"][best_i],
        "final_design": rec["final_design"][best_i],
        "fail_reason": rec["fail_reason"][best_i],
        "fail_reason_counts": dict(collections.Counter(rec["fail_reason"])),
    }, []


# --------------------------------------------------------------------------

def norm_path(p: str) -> str:
    return p.replace("\\", "/").rstrip("/")


def classify_fields(records) -> dict:
    """三層欄位：universal / algorithm-specific / invariant。"""
    n = len(records)
    present = collections.Counter()
    values = collections.defaultdict(set)
    by_algo = collections.defaultdict(set)
    for r in records:
        for k, v in r["args"].items():
            present[k] += 1
            by_algo[k].add(r["algorithm"])
            try:
                values[k].add(json.dumps(v, sort_keys=True))
            except TypeError:
                values[k].add(repr(v))

    universal, algo_specific, invariant = [], {}, []
    for k, count in present.items():
        if len(values[k]) == 1:
            invariant.append(k)
        elif count == n:
            universal.append(k)
        else:
            algo_specific.setdefault("/".join(sorted(by_algo[k])), []).append(k)

    return {
        "universal": sorted(universal),
        "algorithm_specific": {k: sorted(v) for k, v in sorted(algo_specific.items())},
        "invariant": sorted(invariant),
    }


def build(scope_rel: str) -> dict:
    scope = REPO_ROOT / scope_rel
    if not scope.is_dir():
        raise SystemExit("scope 不存在: {}".format(scope))

    # 只看 scope 底下第 1、2 層：Run 要嘛直接位於 scope（Unfiled），
    # 要嘛在一層分組資料夾內。不遞迴下探 Run 內部 —— 那裡的路徑長度
    # 會超過 Windows MAX_PATH 而讓 rglob 直接拋 FileNotFoundError。
    runs = sorted(
        set(p.parent for p in scope.glob("*/record.log"))
        | set(p.parent for p in scope.glob("*/*/record.log"))
    )
    records = []
    for run in runs:
        rel = run.relative_to(scope).as_posix()
        rel_parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        name = run.name

        raw_args, source, warns = read_args(run)
        args = canonicalize(raw_args)
        metrics, mwarns = read_metrics(run)
        warns = warns + mwarns

        purpose, operator = parse_filing(rel_parent)
        device = read_device(run)
        warns = warns + device_warnings(device)
        date = name[:10] if re.match(r"\d{4}_\d{2}_\d{2}", name) else None
        operator = qualify_operator(operator, device, date)
        recorded = args.get("ckpt_dir")
        actual = run.relative_to(REPO_ROOT).as_posix()
        moved = bool(recorded) and norm_path(str(recorded)) != actual

        records.append({
            "id": actual,
            "name": name,
            "group": rel_parent or "(none)",
            "date": date,
            "algorithm": detect_algorithm(args),
            "purpose": purpose,
            "operator": operator,
            "device": device,
            "args_source": source,
            "moved": moved,
            "recorded_ckpt_dir": norm_path(str(recorded)) if recorded else None,
            "comment": args.get("comment"),
            "args": {k: v for k, v in args.items() if k not in NON_PARAMETER_KEYS},
            "metrics": metrics,
            "warnings": warns,
        })

    return {"scope": scope_rel, "runs": records, "fields": classify_fields(records)}


ANNOTATIONS_PATH = Path(__file__).resolve().parent / "annotations.json"


def load_annotations(runs) -> dict:
    """讀取使用者的星號 / 附註（由檢視頁面寫入），原樣內嵌到 HTML。

    annotations.json 是人工撰寫的原始資料，應納入 git；本程式只讀不改。
    Run 被 Filing 搬移後 key（路徑）會失效，頁面會依資料夾名稱自動重新對應
    （ADR-0002 記載的唯一例外），這裡只負責回報。
    """
    if not ANNOTATIONS_PATH.exists():
        return {"version": 2, "annotations": {}}
    data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    entries = data.get("annotations", {})
    ids = {r["id"] for r in runs}
    names = collections.Counter(r["id"].rsplit("/", 1)[-1] for r in runs)
    moved = [k for k in entries if k not in ids and names[k.rsplit("/", 1)[-1]] == 1]
    orphan = [k for k in entries if k not in ids and k not in moved]
    print("annotations: {} entries ({} matched by folder name after move, {} orphaned)".format(
        len(entries), len(moved), len(orphan)))
    for k in orphan:
        print("    orphan: {}".format(k))
    return data


VIEWER_TEMPLATE = Path(__file__).resolve().parent / "viewer_template.html"


def embed_json(obj) -> str:
    """序列化成可安全放進 <script> 的 JSON（避免字串中的 </script> 提早結束標籤）。"""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def render_viewer(index: dict, annotations: dict) -> str:
    """把索引與標註內嵌進檢視頁面範本。

    資料必須內嵌：用 fetch() 讀外部 json 在 file:// 下會被 CORS 擋掉，
    內嵌才能達成「雙擊就開、不需要伺服器」。
    """
    return (VIEWER_TEMPLATE.read_text(encoding="utf-8")
            .replace("/*__INDEX_JSON__*/{}", embed_json(index), 1)
            .replace("/*__ANNOTATIONS_JSON__*/{}", embed_json(annotations), 1))


def print_summary(index: dict) -> None:
    runs = index["runs"]
    for axis in ("algorithm", "purpose", "operator", "device", "args_source"):
        print("  {:12s}".format(axis), dict(collections.Counter(r[axis] for r in runs)))
    print("  {:12s}".format("moved"), dict(collections.Counter(r["moved"] for r in runs)))
    print("  {:12s}".format("unconverged"),
          dict(collections.Counter(r["metrics"].get("unconverged") for r in runs)))
    fields = index["fields"]
    print("  fields: universal={} invariant={} algo-specific={}".format(
        len(fields["universal"]), len(fields["invariant"]),
        {k: len(v) for k, v in fields["algorithm_specific"].items()}))
    flagged = [r for r in runs if r["warnings"]]
    print("  runs with warnings: {}".format(len(flagged)))
    for r in flagged[:10]:
        print("    {}: {}".format(r["name"][:50], r["warnings"]))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    cli = parser.parse_args()

    index = build(cli.scope)
    cli.out.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("indexed {} runs -> {} ({:.0f} KB)".format(
        len(index["runs"]), cli.out, cli.out.stat().st_size / 1024))

    html_out = cli.out.with_suffix(".html")
    html_out.write_text(render_viewer(index, load_annotations(index["runs"])), encoding="utf-8")
    print("viewer -> {} ({:.0f} KB)".format(html_out, html_out.stat().st_size / 1024))

    print_summary(index)


if __name__ == "__main__":
    main()
