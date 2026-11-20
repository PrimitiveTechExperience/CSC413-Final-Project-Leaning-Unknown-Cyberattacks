"""Train one model for one (setting, target scenario) and evaluate on the target's test split.

    python scripts/run_experiment.py --set experiment.model=transformer experiment.setting=seen \
        experiment.scenario=CVE-2017-7529
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lidds_gen.config import dump_config, load_config  # noqa: E402
from lidds_gen.data.experiment import build_experiment  # noqa: E402
from lidds_gen.evaluation import compute_metrics, metrics_by_length, select_threshold  # noqa: E402
from lidds_gen.models import build_model  # noqa: E402
from lidds_gen.training import predict, train_model  # noqa: E402
from lidds_gen.training.loop import resolve_device  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    exp = cfg["experiment"]
    run_dir = Path(cfg["paths"]["results_root"]) / exp["model"] / exp["setting"] / exp["scenario"]
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    set_seed(cfg["seed"])

    vocab, splits, summary = build_experiment(cfg)
    vocab.save(run_dir / "vocab.json")
    print(json.dumps(summary, indent=2))

    model = build_model(exp["model"], len(vocab), cfg["model"], cfg["data"]["max_length"])
    model, history = train_model(model, splits["train"], splits["val"], cfg)
    torch.save(model.state_dict(), run_dir / "model.pt")

    device = resolve_device(cfg["training"]["device"])
    val_prob = predict(model, splits["val"], cfg, device)
    tau = select_threshold(splits["val"].labels, val_prob, cfg["evaluation"]["threshold_metric"])

    test = splits["test"]
    test_prob = predict(model, test, cfg, device)
    results = {
        "experiment": exp,
        "data": summary,
        "history": history,
        "test": compute_metrics(test.labels, test_prob, tau),
        "test_by_length": metrics_by_length(test.labels, test_prob, test.target_lengths, tau),
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "test_predictions.npz", prob=test_prob, label=np.asarray(test.labels),
                        target_length=np.asarray(test.target_lengths))
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
