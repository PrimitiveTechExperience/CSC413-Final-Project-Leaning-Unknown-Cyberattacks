"""Train one model for one (setting, target scenario) and evaluate on the target's test split.

    python scripts/run_experiment.py --set experiment.model=transformer experiment.setting=seen \
        experiment.scenario=CVE-2017-7529
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lidds_gen.config import dump_config, experiment_run_dir, load_config  # noqa: E402
from lidds_gen.data.experiment import build_experiment  # noqa: E402
from lidds_gen.data.report import write_dataset_summary  # noqa: E402
from lidds_gen.evaluation import compute_metrics, metrics_by_length, select_threshold  # noqa: E402
from lidds_gen.models import build_model  # noqa: E402
from lidds_gen.training import predict, train_model  # noqa: E402
from lidds_gen.training.loop import resolve_device  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    run_start = time.perf_counter()
    exp = cfg["experiment"]
    device = resolve_device(cfg["training"]["device"])
    run_dir = experiment_run_dir(cfg)
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Completed run already exists: {run_dir}; choose a new paths.results_root")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "progress.json").unlink(missing_ok=True)  # from an interrupted attempt
    dump_config(cfg, run_dir / "config.yaml")
    set_seed(cfg["seed"])

    assembly_start = time.perf_counter()
    vocab, splits, summary = build_experiment(cfg)
    assembly_seconds = time.perf_counter() - assembly_start
    vocab.save(run_dir / "vocab.json")
    write_dataset_summary(summary, run_dir)
    print(json.dumps({"vocab_size": len(vocab), "splits": {
        split: {key: value for key, value in values.items() if key != "by_length"}
        for split, values in summary["splits"].items()
    }}, indent=2))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(exp["model"], len(vocab), cfg["model"], cfg["data"]["max_length"])
    train_start = time.perf_counter()
    progress_path = run_dir / "progress.json"

    def write_progress(status: dict) -> None:
        # Live status for monitoring only; metrics.json remains the completion marker.
        partial = progress_path.with_suffix(".json.tmp")
        partial.write_text(json.dumps({**status, "train_windows": len(splits["train"]),
                                       "val_windows": len(splits["val"]), "time": time.time()}),
                           encoding="utf-8")
        partial.replace(progress_path)

    # Resume only from a checkpoint written by this exact configuration.
    checkpoint_key = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    model, history = train_model(model, splits["train"], splits["val"], cfg, progress=write_progress,
                                 checkpoint=run_dir / "checkpoint.pt", checkpoint_key=checkpoint_key)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_start
    torch.save(model.state_dict(), run_dir / "model.pt")

    evaluation_start = time.perf_counter()
    val_prob = predict(model, splits["val"], cfg, device)
    tau = select_threshold(splits["val"].labels, val_prob, cfg["evaluation"]["threshold_metric"])

    test = splits["test"]
    test_prob = predict(model, test, cfg, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "assembly_seconds": assembly_seconds, "training_seconds": training_seconds,
        "evaluation_seconds": time.perf_counter() - evaluation_start,
        "elapsed_seconds": time.perf_counter() - run_start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }
    results = {
        "experiment": exp,
        "data": summary,
        "history": history,
        "runtime": runtime,
        "test": compute_metrics(test.labels, test_prob, tau),
        "test_by_length": metrics_by_length(test.labels, test_prob, test.target_lengths, tau),
    }
    np.savez_compressed(run_dir / "test_predictions.npz", prob=test_prob, label=np.asarray(test.labels),
                        target_length=np.asarray(test.target_lengths))
    runtime["elapsed_seconds"] = time.perf_counter() - run_start
    # metrics.json is the completion marker used by run_all.py; write it last.
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "checkpoint.pt").unlink(missing_ok=True)
    print(json.dumps(results["test"], indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
