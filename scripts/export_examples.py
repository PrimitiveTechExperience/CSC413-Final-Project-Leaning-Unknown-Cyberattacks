"""Successful and unsuccessful test predictions, decoded back to system calls.

For each target scenario, the test windows are rebuilt with the single-scenario
config (the test windows are identical in every condition; this is checked
against the saved labels and lengths). Each example lists the window's
probability from every completed run, so the same window can be compared across
models and conditions. Also reports how different benign and malicious test
windows are by system-call frequency (total variation distance) and which calls
are over-represented in malicious windows.

    python scripts/export_examples.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped \
        --config configs/local-gpu.yaml --scenarios CVE-2012-2122 ZipSlip
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from lidds_gen.data.experiment import build_experiment  # noqa: E402

MODELS = ("lstm", "transformer")
CONDITIONS = ("seen", "seen_all", "held_out")
LABEL = {"lstm": "LSTM", "transformer": "Transformer", "seen": "Seen", "seen_all": "Seen (all)",
         "held_out": "Held-out"}


def run_length(calls: list[str], limit: int = 30) -> str:
    """Compress repeated calls: read read read write -> read×3 write."""
    out, i = [], 0
    while i < len(calls) and len(out) < limit:
        j = i
        while j < len(calls) and calls[j] == calls[i]:
            j += 1
        out.append(calls[i] + (f"×{j - i}" if j - i > 1 else ""))
        i = j
    return " ".join(out) + (" …" if i < len(calls) else "")


def call_profile(windows, labels):
    counts = {0: Counter(), 1: Counter()}
    for calls, y in zip(windows, labels):
        counts[int(y)].update(calls)
    totals = {k: sum(c.values()) for k, c in counts.items()}
    names = set(counts[0]) | set(counts[1])
    p = {k: {n: counts[k][n] / max(totals[k], 1) for n in names} for k in (0, 1)}
    tv = 0.5 * sum(abs(p[1][n] - p[0][n]) for n in names)
    lift = sorted(((p[1][n] - p[0][n], n) for n in names), reverse=True)
    over = [{"call": n, "malicious_share": round(p[1][n], 4), "benign_share": round(p[0][n], 4)}
            for d, n in lift[:5] if d > 0]
    return tv, over


def pick(prob, label, mask, order):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    ranked = idx[np.argsort(prob[idx], kind="stable")]
    return int(ranked[-1] if order == "max" else ranked[0] if order == "min" else ranked[len(ranked) // 2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--config", required=True, help="config whose paths point at the processed caches")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--model", default="transformer", choices=MODELS, help="model used to pick examples")
    ap.add_argument("--output", default="results_summary/examples")
    args = ap.parse_args()
    root, out = Path(args.results_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for scenario in args.scenarios:
        preds, taus = {}, {}
        for m in MODELS:
            for c in CONDITIONS:
                d = root / m / c / scenario
                if (d / "metrics.json").exists():
                    preds[(m, c)] = np.load(d / "test_predictions.npz")
                    taus[(m, c)] = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["test"]["threshold"]
        if (args.model, "held_out") not in preds or (args.model, "seen_all") not in preds:
            print(f"[skip] {scenario}: {args.model} seen_all/held_out runs not complete")
            continue
        cfg = load_config(args.config)
        cfg = copy.deepcopy(cfg)
        cfg["experiment"].update(setting="seen", seen_scope="single", scenario=scenario, model=args.model)
        vocab, splits, _ = build_experiment(cfg)
        test = splits["test"]
        ref = next(iter(preds.values()))
        if not (np.array_equal(np.asarray(test.labels), ref["label"])
                and np.array_equal(np.asarray(test.target_lengths), ref["target_length"])):
            raise ValueError(f"{scenario}: rebuilt test windows do not match saved predictions")
        windows = [[vocab.itos[i] for i in seq] for seq in test.sequences]
        labels = np.asarray(test.labels)
        tv, over = call_profile(windows, labels)

        held = preds[(args.model, "held_out")]["prob"]
        seen = preds[(args.model, "seen_all")]["prob"]
        t_held, t_seen = taus[(args.model, "held_out")], taus[(args.model, "seen_all")]
        caught = (labels == 1) & (held >= t_held)
        alarms = (labels == 0) & (held >= t_held)
        cases = {
            ("detected without seeing this attack (held-out true positive, median score)" if caught.any() else
             "highest-scoring attack window when held out (still below τ: no held-out detections)"):
                pick(held, labels, caught if caught.any() else labels == 1, "median" if caught.any() else "max"),
            "missed when held out but caught when seen (held-out false negative)":
                pick(held, labels, (labels == 1) & (held < t_held) & (seen >= t_seen), "median"),
            "missed in every condition (false negative, lowest seen (all) score)":
                pick(seen, labels, (labels == 1) & (seen < t_seen) & (held < t_held), "min"),
            ("false alarm when held out (benign, highest score)" if alarms.any() else
             "highest-scoring benign window when held out (still below τ: no false alarms)"):
                pick(held, labels, labels == 0, "max"),
            "typical benign window (true negative)": pick(held, labels, (labels == 0) & (held < t_held), "median"),
        }
        examples = []
        for title, i in cases.items():
            if i is None:
                continue
            examples.append({"case": title, "index": i, "label": int(labels[i]),
                             "target_length": int(test.target_lengths[i]), "calls": len(windows[i]),
                             "unknown_tokens": windows[i].count("<unk>"),
                             "probabilities": {f"{LABEL[m]} {LABEL[c]}": round(float(p["prob"][i]), 4)
                                               for (m, c), p in preds.items()},
                             "thresholds": {f"{LABEL[m]} {LABEL[c]}": round(float(t), 4) for (m, c), t in taus.items()},
                             "sequence": run_length(windows[i])})
        report = {"scenario": scenario, "model_used_to_pick": args.model, "test_windows": int(len(labels)),
                  "malicious_windows": int(labels.sum()), "call_distribution_tv_distance": round(tv, 4),
                  "over_represented_in_malicious": over, "examples": examples}
        (out / f"{scenario}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[done] {scenario}: TV distance {tv:.3f}, {len(examples)} examples")

    # Combined Markdown from every scenario exported so far.
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.json"))]
    reports.sort(key=lambda r: SCENARIOS.index(r["scenario"]) if r["scenario"] in SCENARIOS else 99)
    lines = ["# Test prediction examples", "",
             "Windows are decoded system-call sequences (runs of the same call are shown as `call×n`; at most 30 "
             "groups). Probabilities are P(malicious) from each completed run; a window is flagged when its "
             "probability is at least that run's threshold τ, chosen on validation data.", "",
             "## How different are benign and malicious test windows?", "",
             "Total variation (TV) distance between the system-call frequency distributions of benign and "
             "malicious test windows (0 = identical mix of calls, 1 = no calls in common).", "",
             "| Scenario | TV distance | Calls most over-represented in malicious windows |", "|---|---|---|"]
    for r in reports:
        calls = ", ".join(f"`{o['call']}` ({100 * o['malicious_share']:.1f}% vs {100 * o['benign_share']:.1f}%)"
                          for o in r["over_represented_in_malicious"][:3])
        lines.append(f"| {r['scenario']} | {r['call_distribution_tv_distance']:.3f} | {calls} |")
    for r in reports:
        lines += ["", f"## {r['scenario']}", "",
                  f"{r['test_windows']:,} test windows, {r['malicious_windows']:,} malicious. "
                  f"Examples picked with the {LABEL[r['model_used_to_pick']]} runs.", ""]
        for e in r["examples"]:
            probs = ", ".join(f"{k} {v:.3f}{'*' if v >= e['thresholds'][k] else ''}"
                              for k, v in e["probabilities"].items())
            lines += [f"**{e['case'][0].upper() + e['case'][1:]}** "
                      f"({'malicious' if e['label'] else 'benign'}, {e['calls']} calls)", "",
                      f"P(malicious): {probs} (* = flagged)", "", f"`{e['sequence']}`", ""]
    (out.parent / "examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out.parent / 'examples.md'}")


if __name__ == "__main__":
    main()
