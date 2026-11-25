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
from lidds_gen.data import SCENARIOS  # noqa: E402
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


def write_progress_file(path: Path, payload: dict) -> bool:
    """Best-effort status write. On Windows the replace fails while a reader has the
    file open, so retry briefly and then skip this update; it must never stop training."""
    partial = path.with_suffix(".json.tmp")
    for attempt in range(5):
        try:
            partial.write_text(json.dumps(payload), encoding="utf-8")
            partial.replace(path)
            return True
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    return False


def _without_target(cfg: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg, default=str))
    out["experiment"].pop("scenario", None)
    return out


def find_shared_model(cfg: dict[str, Any], run_dir: Path) -> Path | None:
    """The all-scenario seen control trains and validates on the same pooled data for
    every target (only the test split is target-specific), so it is the same model each
    time. Return a completed run of that control whose config differs only in the target
    and whose vocabulary is identical, so its trained model can be evaluated here."""
    exp = cfg["experiment"]
    if exp["setting"] != "seen" or exp.get("seen_scope", "single") != "all":
        return None
    import yaml
    want = _without_target(cfg)
    vocab_bytes = (run_dir / "vocab.json").read_bytes()
    for scenario in cfg["data"].get("scenarios", SCENARIOS):
        donor = run_dir.parent / scenario
        if donor == run_dir or not all((donor / f).exists() for f in ("metrics.json", "model.pt", "vocab.json")):
            continue
        metrics = json.loads((donor / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("runtime", {}).get("reused_model_from"):
            continue  # only reuse a model that was actually trained there
        donor_cfg = yaml.safe_load((donor / "config.yaml").read_text(encoding="utf-8"))
        if _without_target(donor_cfg) == want and (donor / "vocab.json").read_bytes() == vocab_bytes:
            return donor
    return None


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
        write_progress_file(progress_path, {**status, "train_windows": len(splits["train"]),
                                            "val_windows": len(splits["val"]), "time": time.time()})

    try:
        donor = find_shared_model(cfg, run_dir)
    except Exception as exc:  # noqa: BLE001 - a lookup problem must fall back to training, not fail
        print(f"[warn] shared-model lookup failed ({exc!r}); training instead", flush=True)
        donor = None
    if donor is not None:
        print(f"[reuse] all-scenario seen model trained in {donor}", flush=True)
        model.load_state_dict(torch.load(donor / "model.pt", map_location="cpu", weights_only=True))
        model.to(device)
        history = json.loads((donor / "metrics.json").read_text(encoding="utf-8"))["history"]
    else:
        # Resume only from a checkpoint written by this exact configuration.
        checkpoint_key = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
        model, history = train_model(model, splits["train"], splits["val"], cfg, progress=write_progress,
                                     checkpoint=run_dir / "checkpoint.pt", checkpoint_key=checkpoint_key)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_start
    if donor is None:
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
        # Set when training was skipped; history and model.pt are the donor run's.
        "reused_model_from": str(donor) if donor is not None else None,
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
