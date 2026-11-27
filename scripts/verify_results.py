"""Check completed runs for identical test sets and split leakage.

For every target scenario, all completed runs (both models, all conditions) must
share the same test recordings and the same test windows (labels and target
lengths in the same order). Within each run, no recording may appear in more
than one split; a held-out run may not train or validate on any of the target's
attack recordings, nor validate on any target recording; a single-scenario seen
run may only use the target's recordings. Exits with code 1 on any failure.

    python scripts/verify_results.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import experiment_condition, load_config  # noqa: E402


def load_runs(root: Path) -> dict:
    runs = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = load_config(path.parent / "config.yaml")
        exp = result["experiment"]
        runs[(exp["model"], experiment_condition(cfg), exp["scenario"])] = (path.parent, result, cfg)
    return runs


def check_run(key, result) -> list[str]:
    model, condition, scenario = key
    manifest = {split: {tuple(r) for r in recs} for split, recs in result["data"]["recording_manifest"].items()}
    errors = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = manifest[a] & manifest[b]
        if shared:
            errors.append(f"{key}: {len(shared)} recording(s) in both {a} and {b}")
    if any(r[0] != scenario for r in manifest["test"]):
        errors.append(f"{key}: test split contains other scenarios")
    if condition == "held_out":
        for split in ("train", "val"):
            if any(r[0] == scenario and r[1] == "test_attack" for r in manifest[split]):
                errors.append(f"{key}: target attack recording in {split}")
        if any(r[0] == scenario for r in manifest["val"]):
            errors.append(f"{key}: target recording in held-out validation")
    if condition == "seen":
        for split in ("train", "val"):
            if any(r[0] != scenario for r in manifest[split]):
                errors.append(f"{key}: single-scenario seen run uses other scenarios in {split}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output", help="optional JSON report path")
    args = ap.parse_args()
    runs = load_runs(Path(args.results_root))
    if not runs:
        raise SystemExit(f"No completed runs under {args.results_root}")

    errors: list[str] = []
    by_scenario = defaultdict(list)
    for key, (run_dir, result, cfg) in runs.items():
        errors += check_run(key, result)
        by_scenario[key[2]].append((key, run_dir, result, cfg))

    data_cfgs = {json.dumps(cfg["data"], sort_keys=True) for _, _, cfg in runs.values()}
    seeds = {cfg["seed"] for _, _, cfg in runs.values()}
    if len(data_cfgs) != 1 or len(seeds) != 1:
        errors.append("runs differ in preprocessing settings or seed")

    per_scenario = {}
    for scenario, items in sorted(by_scenario.items()):
        ref_key, ref_dir, ref_result, _ = items[0]
        ref_manifest = sorted(map(tuple, ref_result["data"]["recording_manifest"]["test"]))
        ref_pred = np.load(ref_dir / "test_predictions.npz")
        for key, run_dir, result, _ in items[1:]:
            if sorted(map(tuple, result["data"]["recording_manifest"]["test"])) != ref_manifest:
                errors.append(f"{key}: test recordings differ from {ref_key}")
            pred = np.load(run_dir / "test_predictions.npz")
            for name in ("label", "target_length"):
                if not np.array_equal(pred[name], ref_pred[name]):
                    errors.append(f"{key}: test {name} array differs from {ref_key}")
        per_scenario[scenario] = {"runs": len(items), "test_recordings": len(ref_manifest),
                                  "test_windows": int(len(ref_pred["label"])),
                                  "malicious_windows": int(ref_pred["label"].sum())}

    # Not an error: seen_all runs trained before model reuse existed, or resumed from
    # a checkpoint (dropout RNG state is not checkpointed), can end at different weights.
    notes = []
    for model in sorted({k[0] for k in runs}):
        groups = defaultdict(list)
        for k, (_, r, _) in runs.items():
            if k[0] == model and k[1] == "seen_all":
                groups[(round(r["test"]["threshold"], 12), len(r["history"]))].append(k[2])
        if len(groups) > 1:
            notes.append(f"{model}: {len(groups)} distinct seen_all models: "
                         + "; ".join(f"tau={t:.4f}, {n} epochs: {', '.join(v)}" for (t, n), v in groups.items()))

    report = {"runs": len(runs), "scenarios": per_scenario, "notes": notes, "errors": errors,
              "passed": not errors}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for scenario, info in per_scenario.items():
        print(f"{scenario:22s} runs={info['runs']} test_recordings={info['test_recordings']} "
              f"test_windows={info['test_windows']} malicious={info['malicious_windows']}")
    for note in notes:
        print(f"[note] {note}")
    for error in errors:
        print(f"[FAIL] {error}")
    print(f"[{'ok' if not errors else 'failed'}] {len(runs)} runs checked, {len(errors)} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
