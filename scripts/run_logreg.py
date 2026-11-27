"""Bag-of-calls logistic regression baseline in the same three conditions as the neural models.

Each window becomes a vector of system-call frequencies (count of each call / window
length), so call order is discarded. A class-balanced, L2-regularized logistic
regression is trained on exactly the windows the LSTM / Transformer saw, and tau is
picked on validation the same way (``evaluation.threshold_metric``):

  Seen       : target scenario only (train / val / test)
  Seen (all) : all scenarios' train / val (one model, reused for every target)
  Held-out   : all scenarios, with the target's attack recordings removed from train
               and the target removed from validation

The re-split and the per-split window sampling depend only on the seed, the scenario,
the split and which recordings are eligible, so the windows are rebuilt here while
loading each scenario's cache once (instead of once per run). Every run is checked
against the neural run of the same condition and target: train / val / test window
and malicious counts from ``dataset_summary.json`` and test labels and lengths from
``test_predictions.npz``.

    python scripts/run_logreg.py --config configs/local-gpu.yaml
    python scripts/run_logreg.py --config configs/local-gpu.yaml --scenarios ZipSlip --conditions held_out
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from lidds_gen.data.experiment import SPLITS, _split_seed  # noqa: E402
from lidds_gen.data.splits import resplit_recordings  # noqa: E402
from lidds_gen.data.store import (  # noqa: E402
    load_scenario_metadata, load_scenario_windows, preprocessing_settings,
)
from lidds_gen.evaluation import compute_metrics, metrics_by_length, select_threshold  # noqa: E402

CONDITIONS = ("seen", "seen_all", "held_out")
LABEL = {"seen": "Seen", "seen_all": "Seen (all)", "held_out": "Held-out"}
NEURAL_MODELS = ("lstm", "transformer")


@dataclass
class Block:
    """Bag-of-calls features for a set of windows, in build_experiment's order."""
    X: sparse.csr_matrix
    y: np.ndarray
    target_lengths: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def stack(blocks: Sequence[Block]) -> Block:
    return Block(sparse.vstack([b.X for b in blocks], format="csr"),
                 np.concatenate([b.y for b in blocks]), np.concatenate([b.target_lengths for b in blocks]))


# --------------------------------------------------------------------------- data

def call_names(cfg: dict[str, Any], scenarios: Sequence[str]) -> list[str]:
    """One feature per system-call name seen in any scenario's cache."""
    root = Path(cfg["paths"]["processed_root"])
    return sorted({t for s in scenarios for t in load_scenario_metadata(root, s)["tokens"]})


def bag_of_calls(store, candidates: np.ndarray, to_feature: np.ndarray, n_features: int) -> sparse.csr_matrix:
    """Rows = windows, columns = call names, values = frequency of the call in the window."""
    starts = store.offsets[candidates].astype(np.int64)
    sizes = store.offsets[candidates + 1].astype(np.int64) - starts
    total = int(sizes.sum())
    row_start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    positions = np.repeat(starts, sizes) + (np.arange(total) - np.repeat(row_start, sizes))
    columns = to_feature[np.asarray(store.call_ids[positions], dtype=np.int64)]
    values = np.repeat(1.0 / np.maximum(sizes, 1), sizes)
    X = sparse.csr_matrix((values, columns, np.concatenate([[0], np.cumsum(sizes)])),
                          shape=(len(candidates), n_features))
    X.sum_duplicates()
    return X


def sample(candidates: np.ndarray, cap: int | None, seed: int, scenario: str, split: str) -> np.ndarray:
    """The same draw as build_experiment for this (scenario, split) and candidate set."""
    if cap is not None and len(candidates) > int(cap):
        rng = np.random.default_rng(_split_seed(seed, scenario, split))
        candidates = np.sort(rng.choice(candidates, size=int(cap), replace=False))
    return candidates


def scenario_blocks(cfg: dict[str, Any], scenario: str, names: list[str]) -> dict[str, Block]:
    """train / val / test over all of the scenario's recordings (used by Seen, Seen (all),
    and by Held-out for non-target scenarios), plus ``train_benign``: training windows of
    benign recordings only (the target's contribution in Held-out)."""
    root = Path(cfg["paths"]["processed_root"])
    data = cfg["data"]
    if "max_windows_per_split" not in data:
        raise ValueError("data.max_windows_per_split is missing; use the config the grid was run with")
    caps = data["max_windows_per_split"] or {}
    meta = load_scenario_metadata(root, scenario)
    if meta.get("preprocessing") != preprocessing_settings(cfg):
        raise ValueError(f"{scenario}: preprocessing config differs from cache; use the grid's config")
    infos = resplit_recordings(meta["recordings"], seed=cfg["seed"], ratios=data["test_resplit"])
    split_of = {r.key: r.split for r in infos}
    exploit = {r.key: r.exploit for r in infos}

    store = load_scenario_windows(root, scenario)
    window_split = np.array([split_of[r.key] for r in store.recordings])[store.recording_ids]
    window_exploit = np.array([exploit[r.key] for r in store.recordings])[store.recording_ids]
    feature = {name: i for i, name in enumerate(names)}
    to_feature = np.array([feature[t] for t in store.tokens], dtype=np.int64)
    candidate_sets = {split: np.flatnonzero(window_split == split) for split in SPLITS}
    candidate_sets["train_benign"] = np.flatnonzero((window_split == "train") & ~window_exploit)
    out = {}
    for name, candidates in candidate_sets.items():
        split = "train" if name == "train_benign" else name
        chosen = sample(candidates, caps.get(split), cfg["seed"], scenario, split)
        out[name] = Block(bag_of_calls(store, chosen, to_feature, len(names)),
                          store.labels[chosen].astype(int), store.target_lengths[chosen].astype(int))
    return out


def condition_splits(blocks: dict[str, dict[str, Block]], condition: str, target: str) -> dict[str, Block]:
    """Mirror select_setting: which scenarios' windows each condition trains and validates on."""
    others = [s for s in blocks if s != target]
    if condition == "seen":
        train, val = [blocks[target]["train"]], [blocks[target]["val"]]
    elif condition == "seen_all":
        train, val = [blocks[s]["train"] for s in blocks], [blocks[s]["val"] for s in blocks]
    elif condition == "held_out":
        train = [blocks[s]["train"] for s in others] + [blocks[target]["train_benign"]]
        val = [blocks[s]["val"] for s in others]
    else:
        raise ValueError(f"Unknown condition {condition!r}")
    return {"train": stack([b for b in train if len(b)]), "val": stack([b for b in val if len(b)]),
            "test": blocks[target]["test"]}


def check_against_neural(compare_root: Path | None, condition: str, target: str,
                         splits: dict[str, Block]) -> list[str]:
    """Same windows as the neural run? Returns the files checked; raises on a mismatch."""
    if compare_root is None:
        return []
    checked = []
    for model in NEURAL_MODELS:
        run = compare_root / model / condition / target
        summary_path = run / "dataset_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))["splits"]
            for split, block in splits.items():
                expected = (summary[split]["windows"], summary[split]["malicious_windows"])
                if expected != (len(block), int(block.y.sum())):
                    raise ValueError(f"{condition}/{target} {split}: {len(block)} windows "
                                     f"({int(block.y.sum())} malicious), {summary_path} has {expected}")
            checked.append(str(summary_path))
        pred_path = run / "test_predictions.npz"
        if pred_path.is_file():
            with np.load(pred_path, allow_pickle=False) as ref:
                if not (np.array_equal(ref["label"], splits["test"].y)
                        and np.array_equal(ref["target_length"], splits["test"].target_lengths)):
                    raise ValueError(f"{condition}/{target}: test windows differ from {pred_path}")
            checked.append(str(pred_path))
    return checked


# -------------------------------------------------------------------------- model

def fit(train: Block, C: float, max_iter: int, seed: int):
    """Standardized frequencies (no centering, keeps sparsity) -> balanced logistic regression,
    the linear analogue of the neural models' class-weighted BCE."""
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train.X, train.y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def top_calls(model, names: list[str], k: int = 10) -> dict[str, list[list[Any]]]:
    """Calls with the largest weights (per standard deviation of their frequency)."""
    coef = model[-1].coef_[0]
    order = np.argsort(coef)
    return {"malicious": [[names[i], float(coef[i])] for i in order[::-1][:k]],
            "benign": [[names[i], float(coef[i])] for i in order[:k]]}


def evaluate(model, converged: bool, train_seconds: float, splits: dict[str, Block], cfg: dict[str, Any],
             condition: str, target: str, names: list[str], checked: list[str], run_dir: Path,
             C: float) -> dict[str, Any]:
    val_prob = model.predict_proba(splits["val"].X)[:, 1]
    tau = select_threshold(splits["val"].y, val_prob, cfg["evaluation"]["threshold_metric"])
    test = splits["test"]
    test_prob = model.predict_proba(test.X)[:, 1]
    results = {
        "experiment": {"model": "logreg", "setting": condition, "scenario": target,
                       "features": "bag of calls (frequency per window)", "C": C,
                       "class_weight": "balanced", "threshold_metric": cfg["evaluation"]["threshold_metric"]},
        "data": {split: {"windows": len(b), "malicious_windows": int(b.y.sum())} for split, b in splits.items()},
        "converged": converged,
        "checked_against": checked,
        "top_calls": top_calls(model, names),
        "runtime": {"training_seconds": train_seconds},
        "test": compute_metrics(test.y, test_prob, tau),
        "test_by_length": metrics_by_length(test.y, test_prob, test.target_lengths, tau),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "test_predictions.npz", prob=test_prob, label=test.y,
                        target_length=test.target_lengths)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    t = results["test"]
    print(f"{LABEL[condition]:11s} {target:22s} AUROC {t['auroc']:.3f}  AUPRC {t['auprc']:.3f}  "
          f"F1 {t['f1']:.3f}  FPR {t['fpr']:.3f}{'' if converged else '  [not converged]'}", flush=True)
    return results


# ------------------------------------------------------------------------ summary

METRICS = (("AUROC", "auroc"), ("AUPRC", "auprc"), ("F1", "f1"), ("Precision", "precision"),
           ("Recall", "recall"), ("FPR", "fpr"))


def write_summary(out_root: Path, targets: Sequence[str]) -> tuple[Path, Path]:
    """logreg_per_scenario.csv (one row per target and condition) and logreg_overall.csv
    (mean over targets, same columns as results_summary/overall.csv)."""
    runs = {}
    for path in out_root.glob("logreg/*/*/metrics.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["experiment"]["scenario"] in targets:
            runs[(r["experiment"]["setting"], r["experiment"]["scenario"])] = r
    per_path, overall_path = out_root / "logreg_per_scenario.csv", out_root / "logreg_overall.csv"
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Scenario", "Condition", *[m for m, _ in METRICS], "Threshold", "Converged"])
        for target in targets:
            for c in CONDITIONS:
                if (c, target) in runs:
                    r = runs[(c, target)]
                    w.writerow([target, LABEL[c], *[f"{r['test'][k]:.4f}" for _, k in METRICS],
                                f"{r['test']['threshold']:.4f}", r["converged"]])
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Condition", "Runs", *[m for m, _ in METRICS]])
        for c in CONDITIONS:
            group = [r for (cond, _), r in runs.items() if cond == c]
            if group:
                w.writerow(["LogReg (bag of calls)", LABEL[c], len(group),
                            *[f"{np.nanmean([r['test'][k] for r in group]):.3f}" for _, k in METRICS]])
    return per_path, overall_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml",
                    help="the same config the LSTM/Transformer grid was run with")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--scenarios", nargs="*", default=None, help="targets (default: all configured)")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 regularization strength")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--out", default=None, help="output root (default: <results_root>_logreg)")
    ap.add_argument("--compare-root", default=None,
                    help="neural results root to cross-check windows (default: paths.results_root)")
    ap.add_argument("--no-compare", action="store_true", help="skip the cross-check")
    ap.add_argument("--overwrite", action="store_true", help="recompute finished runs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    pool = list(cfg["data"].get("scenarios", SCENARIOS))
    targets = args.scenarios or pool
    unknown = set(targets) - set(pool)
    if unknown:
        raise SystemExit(f"Unknown scenarios: {sorted(unknown)}")
    results_root = Path(cfg["paths"]["results_root"])
    out_root = Path(args.out) if args.out else results_root.with_name(results_root.name + "_logreg")
    compare_root = None if args.no_compare else Path(args.compare_root or results_root)
    if compare_root is not None and out_root.resolve() == compare_root.resolve():
        raise SystemExit("--out must differ from the neural results root (summary scripts glob it)")
    print(f"Writing logistic-regression results to {out_root}", flush=True)

    def run_dir(condition: str, target: str) -> Path:
        return out_root / "logreg" / condition / target

    todo = [(c, t) for c in args.conditions for t in targets
            if args.overwrite or not (run_dir(c, t) / "metrics.json").exists()]
    if todo:
        names = call_names(cfg, pool)
        blocks = {}
        for scenario in pool:  # Seen (all) and Held-out train on every scenario
            t0 = time.perf_counter()
            blocks[scenario] = scenario_blocks(cfg, scenario, names)
            print(f"[load] {scenario}: {time.perf_counter() - t0:.0f}s", flush=True)

    failed, shared = {}, None
    for condition, target in todo:
        try:
            splits = condition_splits(blocks, condition, target)
            checked = check_against_neural(compare_root, condition, target, splits)
            if compare_root is not None and not checked:
                print(f"[warn] {condition}/{target}: no neural run under {compare_root}; not cross-checked",
                      flush=True)
            if condition == "seen_all" and shared is not None:
                model, converged, seconds = shared  # same pooled train / val for every target
            else:
                t0 = time.perf_counter()
                model, converged = fit(splits["train"], args.C, args.max_iter, cfg["seed"])
                seconds = time.perf_counter() - t0
                if condition == "seen_all":
                    shared = (model, converged, seconds)
            evaluate(model, converged, seconds, splits, cfg, condition, target, names, checked,
                     run_dir(condition, target), args.C)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed[f"{condition}/{target}"] = repr(exc)
            print(f"[fail] {condition}/{target}: {exc!r}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    per_path, overall_path = write_summary(out_root, targets)
    print(f"Summary: {overall_path}\n         {per_path}", flush=True)
    if failed:
        (out_root / "failed_runs.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(failed)} run(s) failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
