"""Run-to-run variance from the distinct Seen (all) models already trained.

Seen (all) trains on the same pooled data for every target, so every saved
Seen (all) model of one architecture had identical training data. Runs that
resumed from a checkpoint ended at different weights (dropout RNG state is not
checkpointed), which gives several independently finished models. This script
groups the saved model.pt files by content, evaluates every distinct model on
every target's test windows, and reports how much the metrics move between
models. Each model must reproduce its own run's saved test predictions before
any other number is reported.

    python scripts/variance_check.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped \
        --config configs/local-gpu.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

import lidds_gen.data.experiment as experiment  # noqa: E402
from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from lidds_gen.data.vocab import Vocabulary  # noqa: E402
from lidds_gen.evaluation.metrics import compute_metrics  # noqa: E402
from lidds_gen.models import build_model  # noqa: E402
from lidds_gen.training import predict  # noqa: E402
from lidds_gen.training.loop import resolve_device  # noqa: E402

METRICS = ("auroc", "auprc", "f1")


def distinct_models(root: Path, model: str) -> list[dict]:
    """Saved Seen (all) models grouped by file content (identical weights = one model)."""
    groups = defaultdict(list)
    for path in sorted((root / model / "seen_all").glob("*/model.pt")):
        groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.parent)
    models = []
    for digest, dirs in groups.items():
        result = json.loads((dirs[0] / "metrics.json").read_text(encoding="utf-8"))
        models.append({"id": digest[:8], "dirs": dirs, "tau": result["test"]["threshold"],
                       "epochs": len(result["history"]),
                       "best_val_loss": min(h["val_loss"] for h in result["history"])})
    models.sort(key=lambda m: m["dirs"][0].name)
    for i, m in enumerate(models):
        m["name"] = f"run {chr(ord('A') + i)}"
    return models


def test_split(cfg: dict, scenario: str, vocab: Vocabulary):
    """The target's test windows, encoded with the pooled vocabulary.

    Built through the cheap single-scenario path (test windows are identical in
    every condition), with vocabulary construction replaced by the pooled one.
    """
    cfg = copy.deepcopy(cfg)
    cfg["experiment"].update(setting="seen", seen_scope="single", scenario=scenario)
    original = experiment.Vocabulary.build
    experiment.Vocabulary.build = classmethod(lambda cls, windows: vocab)
    try:
        _, splits, _ = experiment.build_experiment(cfg)
    finally:
        experiment.Vocabulary.build = original
    return splits["test"]


def key(metric: str, model: dict) -> str:
    return f"{metric} {model['name']}"


def spread(values: list[float]) -> float:
    return float(max(values) - min(values))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--config", required=True, help="config whose paths point at the processed caches")
    ap.add_argument("--model", default="lstm", choices=("lstm", "transformer"))
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--output", default="results_summary")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    root, out = Path(args.results_root), Path(args.output)

    models = distinct_models(root, args.model)
    if len(models) < 2:
        raise SystemExit(f"Only {len(models)} distinct {args.model} Seen (all) model(s); nothing to compare")
    vocab_files = {(d / "vocab.json").read_bytes() for m in models for d in m["dirs"]}
    if len(vocab_files) != 1:
        raise SystemExit("Seen (all) runs have different vocabularies; models are not comparable")
    vocab = Vocabulary.load(models[0]["dirs"][0] / "vocab.json")
    cfg = load_config(args.config)
    run_cfg = load_config(models[0]["dirs"][0] / "config.yaml")
    device = resolve_device(args.device)
    nets = {}
    for m in models:
        net = build_model(args.model, len(vocab), run_cfg["model"], run_cfg["data"]["max_length"])
        net.load_state_dict(torch.load(m["dirs"][0] / "model.pt", map_location="cpu", weights_only=True))
        nets[m["name"]] = net.to(device)
        print(f"[model] {m['name']} ({m['id']}): tau={m['tau']:.4f}, {m['epochs']} epochs, "
              f"best val loss {m['best_val_loss']:.4f}, trained in {', '.join(d.name for d in m['dirs'])}")

    rows, checks = [], []
    for scenario in args.scenarios:
        test = test_split(cfg, scenario, vocab)
        labels = np.asarray(test.labels)
        saved = np.load(root / args.model / "seen_all" / scenario / "test_predictions.npz")
        if not (np.array_equal(labels, saved["label"])
                and np.array_equal(np.asarray(test.target_lengths), saved["target_length"])):
            raise ValueError(f"{scenario}: rebuilt test windows do not match saved predictions")
        row = {"scenario": scenario, "test_windows": len(labels)}
        for m in models:
            prob = predict(nets[m["name"]], test, run_cfg, device)
            if any(d.name == scenario for d in m["dirs"]):
                # The run that trained this model evaluated this target: predictions must match.
                diff = float(np.max(np.abs(prob - saved["prob"])))
                checks.append((m["name"], scenario, diff))
                if diff > 1e-4:
                    raise ValueError(f"{m['name']} does not reproduce saved predictions on {scenario} "
                                     f"(max difference {diff:.2e})")
            metrics = compute_metrics(labels, prob, m["tau"])
            for name in METRICS:
                row[key(name, m)] = metrics[name]
        for name in METRICS:
            row[f"{name} range"] = spread([row[key(name, m)] for m in models])
        rows.append(row)
        aurocs = ", ".join(f"{row[key('auroc', m)]:.3f}" for m in models)
        print(f"[done] {scenario}: AUROC {aurocs} (range {row['auroc range']:.3f})")

    # Compare the spread with the Seen (all) minus Held-out gap on the same targets.
    gaps = {}
    for scenario in args.scenarios:
        a = root / args.model / "seen_all" / scenario / "metrics.json"
        b = root / args.model / "held_out" / scenario / "metrics.json"
        if a.exists() and b.exists():
            gaps[scenario] = (json.loads(a.read_text(encoding="utf-8"))["test"]["auroc"]
                              - json.loads(b.read_text(encoding="utf-8"))["test"]["auroc"])

    out.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "test_windows"] + [key(n, m) for n in METRICS for m in models] \
        + [f"{n} range" for n in METRICS]
    with (out / f"variance_{args.model}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def mean_over(key):
        return float(np.mean([r[key] for r in rows]))

    label = {"lstm": "LSTM", "transformer": "Transformer"}[args.model]
    lines = [f"# Run-to-run variance: {label} Seen (all)", "",
             f"Generated by `scripts/variance_check.py`. The {len(models)} distinct {label} Seen (all) models below "
             "were trained on identical data with identical settings; they differ only because some runs resumed "
             "from a checkpoint (the dropout random state is not checkpointed). They are not independent random "
             "seeds, so this is likely a lower bound on seed-to-seed variation.", "",
             "| Model | Trained in run(s) | Epochs | Best val loss | τ |", "|---|---|---|---|---|"]
    lines += [f"| {m['name']} | {', '.join(d.name for d in m['dirs'])} | {m['epochs']} | "
              f"{m['best_val_loss']:.4f} | {m['tau']:.4f} |" for m in models]
    lines += ["", f"Check: each model reproduced its own run's saved test predictions on {len(checks)} "
              f"(model, target) pairs (largest difference {max(c[2] for c in checks):.1e}).", "",
              "## Test AUROC of each model on each target", "",
              "| Target | " + " | ".join(m["name"] for m in models) + " | Range | Seen (all) − Held-out gap |",
              "|---|" + "---|" * (len(models) + 2)]
    for r in rows:
        gap = gaps.get(r["scenario"])
        cells = " | ".join(f"{r[key('auroc', m)]:.3f}" for m in models)
        gap_cell = "–" if gap is None else f"{gap:+.3f}"
        lines.append(f"| {r['scenario']} | {cells} | {r['auroc range']:.3f} | {gap_cell} |")
    lines.append("| **Mean** | " + " | ".join(f"{mean_over(key('auroc', m)):.3f}" for m in models)
                 + f" | {mean_over('auroc range'):.3f} | "
                 + (f"{np.mean(list(gaps.values())):+.3f}" if gaps else "–") + " |")
    lines += ["", "## Spread between models (max − min), over targets", "",
              "| Metric | Mean range | Largest range | Target with largest range |", "|---|---|---|---|"]
    for name in METRICS:
        worst = max(rows, key=lambda r: r[f"{name} range"])
        lines.append(f"| {name.upper()} | {mean_over(f'{name} range'):.3f} | {worst[f'{name} range']:.3f} | "
                     f"{worst['scenario']} |")
    mean_aurocs = [mean_over(key("auroc", m)) for m in models]
    lines += ["", f"Mean AUROC over the {len(rows)} targets ranges from {min(mean_aurocs):.3f} to "
              f"{max(mean_aurocs):.3f} across models (spread {spread(mean_aurocs):.3f})."]
    if gaps:
        larger = sum(abs(gaps[r["scenario"]]) > r["auroc range"] for r in rows if r["scenario"] in gaps)
        lines.append(f"The Seen (all) − Held-out AUROC gap is larger than the between-model range for "
                     f"{larger} of {len(gaps)} targets.")
    (out / f"variance_{args.model}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out / f'variance_{args.model}.md'}")


if __name__ == "__main__":
    main()
