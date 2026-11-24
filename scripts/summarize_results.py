"""Export per-scenario metrics and seen-minus-held-out gaps from completed runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import experiment_condition, load_config  # noqa: E402

METRICS = ("f1", "auroc", "auprc", "fpr", "precision", "recall", "positive_rate", "n")


def summarize(root: Path) -> tuple[list[dict], list[dict]]:
    rows, runs = [], {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = load_config(path.parent / "config.yaml")
        exp = result["experiment"]
        condition = experiment_condition(cfg)
        key = (exp["model"], condition, exp["scenario"])
        if key in runs:
            raise ValueError(f"Duplicate experiment: {key}")
        runs[key] = (result, cfg)
        for length, metrics in {"all": result["test"], **result["test_by_length"]}.items():
            rows.append({"model": exp["model"], "condition": condition,
                         "scenario": exp["scenario"], "target_length": length,
                         **{name: metrics[name] for name in METRICS}, "threshold": metrics["threshold"]})
    comparisons = []
    for (model, condition, scenario), (held, held_cfg) in runs.items():
        if condition != "held_out":
            continue
        for seen_condition in ("seen", "seen_all"):
            pair = runs.get((model, seen_condition, scenario))
            if pair is None:
                continue
            seen, seen_cfg = pair
            # A gap is meaningful only when both runs share preprocessing and test recordings.
            if (held_cfg["seed"] != seen_cfg["seed"] or held_cfg["data"] != seen_cfg["data"]
                    or held["data"]["recording_manifest"]["test"] != seen["data"]["recording_manifest"]["test"]):
                raise ValueError(f"Cannot compare different test data: {model}/{scenario}/{seen_condition}")
            held_metrics = {"all": held["test"], **held["test_by_length"]}
            seen_metrics = {"all": seen["test"], **seen["test_by_length"]}
            for length, metrics in held_metrics.items():
                control = seen_metrics[length]
                if metrics["n"] != control["n"] or metrics["positive_rate"] != control["positive_rate"]:
                    raise ValueError(f"Mismatched test windows: {model}/{scenario}/{length}")
                comparisons.append({"model": model, "scenario": scenario, "seen_condition": seen_condition,
                                    "target_length": length,
                                    **{f"{metric}_gap": control[metric] - metrics[metric]
                                       for metric in ("f1", "auroc", "auprc", "fpr")}})
    return rows, comparisons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()
    root = Path(args.results_root)
    rows, comparisons = summarize(root)
    if not rows:
        raise FileNotFoundError(f"No completed runs under {root}")
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for name, values, fields in (
        ("metrics.csv", rows, ["model", "condition", "scenario", "target_length", *METRICS, "threshold"]),
        ("comparison.csv", comparisons, ["model", "scenario", "seen_condition", "target_length",
                                         "f1_gap", "auroc_gap", "auprc_gap", "fpr_gap"]),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    print(f"[done] {len(rows)} metric rows, {len(comparisons)} comparison rows -> {output}")


if __name__ == "__main__":
    main()
