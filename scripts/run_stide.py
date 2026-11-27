"""STIDE baseline (Hofmeyr et al., 1998) on each target's held-out test windows.

STIDE stores every length-n run of system calls (n-gram) seen in benign training
traces and scores a window by the fraction of its n-grams that were never seen.
It never sees an attack, so every result is "held-out" by construction and is
directly comparable with the neural Held-out condition.

Per target scenario (one STIDE database per application, as in the original method):
  train : windows from the target's *benign* training recordings
  val   : windows from the target's *benign* validation recordings (threshold only)
  test  : exactly the test windows the LSTM / Transformer runs were scored on

The recording re-split and the per-split window sampling depend only on the seed,
the scenario and the split, so the test windows are rebuilt here without loading
the other 14 scenarios. They are checked against the neural runs'
``test_predictions.npz`` (labels and lengths, in order) when those exist.

AUROC and AUPRC are threshold-free. For F1 / precision / recall / FPR the threshold
tau is the lowest score whose false-positive rate on benign validation windows is at
most --max-fpr (validation has no attacks, so tau cannot be chosen by F1 as for the
neural models).

    python scripts/run_stide.py --config configs/local-gpu.yaml
    python scripts/run_stide.py --config configs/local-gpu.yaml --scenarios ZipSlip --ngram 6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from lidds_gen.data.experiment import SPLITS, _split_seed  # noqa: E402
from lidds_gen.data.splits import resplit_recordings  # noqa: E402
from lidds_gen.data.store import (  # noqa: E402
    load_scenario_metadata, load_scenario_windows, preprocessing_settings,
)
from lidds_gen.evaluation import compute_metrics, metrics_by_length  # noqa: E402

NEURAL_MODELS = ("lstm", "transformer")


# --------------------------------------------------------------------------- data

def stide_windows(cfg: dict[str, Any], scenario: str) -> tuple[dict[str, dict[str, Any]], int]:
    """Benign train/val windows and all test windows of one scenario, sampled exactly
    as ``build_experiment`` samples them. Returns ({split: windows}, vocabulary size)."""
    root = Path(cfg["paths"]["processed_root"])
    data = cfg["data"]
    if "max_windows_per_split" not in data:
        raise ValueError("data.max_windows_per_split is missing; use the config the grid was run with")
    caps = data["max_windows_per_split"] or {}
    meta = load_scenario_metadata(root, scenario)
    if meta.get("preprocessing") != preprocessing_settings(cfg):
        raise ValueError(f"{scenario}: preprocessing config differs from cache; use the grid's config")
    infos = resplit_recordings(meta["recordings"], seed=cfg["seed"], ratios=data["test_resplit"])
    # STIDE never sees an attack recording: train/val keep benign recordings only.
    membership = {r.key: r.split for r in infos if r.split == "test" or not r.exploit}

    store = load_scenario_windows(root, scenario)
    rec_split = np.array([membership.get(r.key, "") for r in store.recordings])
    window_split = rec_split[store.recording_ids]
    out = {}
    for split in SPLITS:
        candidates = np.flatnonzero(window_split == split)
        if not len(candidates):
            raise ValueError(f"{scenario}: no {split} windows")
        cap = caps.get(split)
        if cap is not None and len(candidates) > int(cap):
            # Same generator and draw as build_experiment, so the capped test sample
            # matches the neural runs window for window.
            rng = np.random.default_rng(_split_seed(cfg["seed"], scenario, split))
            candidates = np.sort(rng.choice(candidates, size=int(cap), replace=False))
        out[split] = {
            "sequences": [np.asarray(store.call_ids[store.offsets[i]:store.offsets[i + 1]], dtype=np.int64)
                          for i in candidates],
            "labels": store.labels[candidates].astype(int),
            "target_lengths": store.target_lengths[candidates].astype(int),
        }
    for split in ("train", "val"):
        if out[split]["labels"].any():
            raise ValueError(f"{scenario}: malicious window in benign {split} recordings")
    return out, len(store.tokens)


def check_test_windows(compare_root: Path | None, scenario: str, labels: np.ndarray,
                       target_lengths: np.ndarray) -> list[str]:
    """Compare with the neural held-out runs' test windows; returns the runs checked."""
    if compare_root is None:
        return []
    checked = []
    for model in NEURAL_MODELS:
        path = compare_root / model / "held_out" / scenario / "test_predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as ref:
            same = (np.array_equal(ref["label"], labels)
                    and np.array_equal(ref["target_length"], target_lengths))
        if not same:
            raise ValueError(f"{scenario}: test windows differ from {path}; "
                             "use the same config (seed, caps, processed_root) as the grid")
        checked.append(str(path))
    return checked


# ------------------------------------------------------------------------- STIDE

def gram_codes(windows: Sequence[np.ndarray], n: int, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense integer code for every n-gram of every window (equal codes = equal n-grams).

    Returns (codes, grams per window). A window shorter than n becomes a single gram
    padded with the token ``vocab_size``, so it only matches an equally short window.
    Codes are only comparable within one call, so pass all splits together.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    pad = vocab_size
    padded = [w if len(w) >= n else np.concatenate([w, np.full(n - len(w), pad, dtype=np.int64)])
              for w in windows]
    lengths = np.array([len(w) for w in padded], dtype=np.int64)
    counts = lengths - n + 1
    flat = np.concatenate(padded).astype(np.int64)
    window_start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    gram_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    total = int(counts.sum())
    starts = np.repeat(window_start, counts) + (np.arange(total) - np.repeat(gram_start, counts))
    # Extend codes one position at a time and re-densify, so they never overflow.
    code = flat[starts]
    for j in range(1, n):
        code = code * (pad + 1) + flat[starts + j]
        code = np.unique(code, return_inverse=True)[1].reshape(-1)
    return code, counts


def stide_scores(train: Sequence[np.ndarray], evaluate: Sequence[Sequence[np.ndarray]], n: int,
                 vocab_size: int) -> tuple[list[np.ndarray], int]:
    """Fraction of each window's n-grams absent from the benign training database.
    Returns (scores for each sequence list in ``evaluate``, database size)."""
    groups = [list(train)] + [list(e) for e in evaluate]
    codes, counts = gram_codes([w for g in groups for w in g], n, vocab_size)
    window_bounds = np.cumsum([0] + [len(g) for g in groups])
    gram_bounds = np.concatenate([[0], np.cumsum(counts)])[window_bounds]
    known = np.unique(codes[:gram_bounds[1]])
    out = []
    for g in range(1, len(groups)):
        c = codes[gram_bounds[g]:gram_bounds[g + 1]]
        idx = np.minimum(np.searchsorted(known, c), len(known) - 1)
        miss = (known[idx] != c).astype(np.float64)
        per_window = counts[window_bounds[g]:window_bounds[g + 1]]
        offsets = np.concatenate([[0], np.cumsum(per_window)[:-1]])
        out.append(np.add.reduceat(miss, offsets) / per_window)
    return out, int(len(known))


def fpr_threshold(benign_scores: np.ndarray, max_fpr: float) -> float:
    """Lowest tau with P(score >= tau) <= max_fpr on benign validation windows."""
    if not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must be in [0, 1)")
    s = np.sort(np.asarray(benign_scores, dtype=np.float64))
    candidates = np.unique(np.append(s, np.nextafter(s[-1], np.inf)))
    fpr = 1.0 - np.searchsorted(s, candidates, side="left") / len(s)
    return float(candidates[np.argmax(fpr <= max_fpr)])


# --------------------------------------------------------------------------- runs

def run_scenario(cfg: dict[str, Any], scenario: str, ngrams: Sequence[int], max_fpr: float,
                 out_root: Path, compare_root: Path | None, overwrite: bool) -> list[dict[str, Any]]:
    todo = [n for n in ngrams
            if overwrite or not (out_root / f"stide_n{n}" / "held_out" / scenario / "metrics.json").exists()]
    if not todo:
        print(f"[skip] {scenario}: all n finished", flush=True)
        return []
    start = time.perf_counter()
    splits, vocab_size = stide_windows(cfg, scenario)
    test = splits["test"]
    checked = check_test_windows(compare_root, scenario, test["labels"], test["target_lengths"])
    if compare_root is not None and not checked:
        print(f"[warn] {scenario}: no neural held-out test_predictions.npz under {compare_root}; "
              "test windows not cross-checked", flush=True)
    load_seconds = time.perf_counter() - start
    rows = []
    for n in todo:
        t0 = time.perf_counter()
        (val_score, test_score), db_size = stide_scores(
            splits["train"]["sequences"], [splits["val"]["sequences"], test["sequences"]], n, vocab_size)
        tau = fpr_threshold(val_score, max_fpr)
        metrics = compute_metrics(test["labels"], test_score, tau)
        run_dir = out_root / f"stide_n{n}" / "held_out" / scenario
        run_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "experiment": {"model": f"stide_n{n}", "setting": "held_out", "scenario": scenario,
                           "ngram": n, "score": "fraction of n-grams absent from benign training",
                           "threshold_rule": f"benign validation FPR <= {max_fpr}"},
            "data": {split: {"windows": len(v["labels"]), "malicious_windows": int(v["labels"].sum())}
                     for split, v in splits.items()},
            "database_ngrams": db_size,
            "val_benign_fpr_at_tau": float(np.mean(val_score >= tau)),
            "checked_against": checked,
            "runtime": {"load_seconds": load_seconds, "score_seconds": time.perf_counter() - t0},
            "test": metrics,
            "test_by_length": metrics_by_length(test["labels"], test_score, test["target_lengths"], tau),
        }
        np.savez_compressed(run_dir / "test_predictions.npz", prob=test_score, label=test["labels"],
                            target_length=test["target_lengths"])
        (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{scenario:22s} n={n:<3d} AUROC {metrics['auroc']:.3f}  AUPRC {metrics['auprc']:.3f}  "
              f"F1 {metrics['f1']:.3f}  FPR {metrics['fpr']:.3f}  ({db_size} n-grams)", flush=True)
        rows.append(results)
    return rows


SUMMARY_FIELDS = ("Scenario", "n", "AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR", "Threshold",
                  "Test windows", "Malicious test windows", "Database n-grams")


def write_summary(out_root: Path, scenarios: Sequence[str]) -> Path:
    """Per-scenario rows plus a mean row per n, from every finished run on disk."""
    rows = []
    for path in sorted(out_root.glob("stide_n*/held_out/*/metrics.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["experiment"]["scenario"] not in scenarios:
            continue
        t = r["test"]
        rows.append({"Scenario": r["experiment"]["scenario"], "n": r["experiment"]["ngram"],
                     "AUROC": t["auroc"], "AUPRC": t["auprc"], "F1": t["f1"], "Precision": t["precision"],
                     "Recall": t["recall"], "FPR": t["fpr"], "Threshold": t["threshold"],
                     "Test windows": t["n"], "Malicious test windows": r["data"]["test"]["malicious_windows"],
                     "Database n-grams": r["database_ngrams"]})
    order = {s: i for i, s in enumerate(scenarios)}
    rows.sort(key=lambda row: (row["n"], order[row["Scenario"]]))
    out_rows = []
    for n in sorted({row["n"] for row in rows}):
        group = [row for row in rows if row["n"] == n]
        out_rows += group
        mean = {"Scenario": f"Mean ({len(group)} scenarios)", "n": n}
        for key in ("AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR"):
            mean[key] = float(np.nanmean([row[key] for row in group]))
        out_rows.append(mean)
    path = out_root / "stide_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    return path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml",
                    help="the same config the LSTM/Transformer grid was run with")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--scenarios", nargs="*", default=None, help="targets (default: all configured)")
    ap.add_argument("--ngram", nargs="+", type=int, default=[3, 6, 10],
                    help="n-gram lengths (6 is the classic STIDE choice)")
    ap.add_argument("--max-fpr", type=float, default=0.01,
                    help="false-positive budget on benign validation windows for tau")
    ap.add_argument("--out", default=None, help="output root (default: <results_root>_stide)")
    ap.add_argument("--compare-root", default=None,
                    help="neural results root to cross-check test windows (default: paths.results_root)")
    ap.add_argument("--no-compare", action="store_true", help="skip the cross-check")
    ap.add_argument("--overwrite", action="store_true", help="recompute finished runs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    configured = list(cfg["data"].get("scenarios", SCENARIOS))
    scenarios = args.scenarios or configured
    unknown = set(scenarios) - set(configured)
    if unknown:
        raise SystemExit(f"Unknown scenarios: {sorted(unknown)}")
    results_root = Path(cfg["paths"]["results_root"])
    out_root = Path(args.out) if args.out else results_root.with_name(results_root.name + "_stide")
    compare_root = None if args.no_compare else Path(args.compare_root or results_root)
    if compare_root is not None and out_root.resolve() == compare_root.resolve():
        raise SystemExit("--out must differ from the neural results root (summary scripts glob it)")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing STIDE results to {out_root}", flush=True)

    failed = {}
    for scenario in scenarios:
        try:
            run_scenario(cfg, scenario, args.ngram, args.max_fpr, out_root, compare_root, args.overwrite)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed[scenario] = repr(exc)
            print(f"[fail] {scenario}: {exc!r}", flush=True)
    summary = write_summary(out_root, scenarios)
    print(f"Summary: {summary}", flush=True)
    if failed:
        (out_root / "failed_runs.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(failed)} scenario(s) failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
