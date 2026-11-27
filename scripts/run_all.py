"""Full grid: two models x seen / held-out x 15 targets, with an optional seen-all control.

Completed runs (with an existing metrics.json) are skipped, so the grid can be resumed.
Each run executes in its own process (memory and CUDA state reset between runs) and
resumes from its last epoch checkpoint if interrupted. A failed run is recorded in
<results_root>/failed_runs.json and the grid continues; the exit code is 1 if any failed.

    python scripts/run_all.py --models lstm --scenarios CVE-2014-0160 ZipSlip
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidds_gen.config import dump_config, experiment_run_dir, load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from lidds_gen.training.loop import learning_rate  # noqa: E402

RUN_EXPERIMENT = Path(__file__).resolve().parent / "run_experiment.py"


def effective_config(cfg: dict) -> dict:
    """Config with the learning rate resolved for this run's model, so adding a
    per-model override for one model does not invalidate the other's finished runs."""
    out = copy.deepcopy(cfg)
    out["training"]["lr"] = learning_rate(cfg)
    out["training"].pop("lr_by_model", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="*", default=["lstm", "transformer"])
    ap.add_argument("--settings", nargs="*", default=["seen", "held_out"])
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--include-seen-control", action="store_true",
                    help="also run the all-scenario seen control alongside the configured baseline")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    base = load_config(args.config, args.set)
    results_root = Path(base["paths"]["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    failures_path = results_root / "failed_runs.json"
    failures = []
    for model in args.models:
        for setting in args.settings:
            for scenario in args.scenarios:
                scopes = [base["experiment"].get("seen_scope", "single")]
                if setting == "seen" and args.include_seen_control:
                    scopes = ["single", "all"]
                for scope in scopes:
                    cfg = copy.deepcopy(base)
                    cfg["experiment"].update(model=model, setting=setting, scenario=scenario, seen_scope=scope)
                    out = experiment_run_dir(cfg) / "metrics.json"
                    if out.exists():
                        saved = load_config(out.parent / "config.yaml")
                        if effective_config(saved) != effective_config(cfg):
                            raise ValueError(f"Existing run has different config: {out.parent}; "
                                             "choose a new paths.results_root")
                        print(f"[skip] {out}")
                        continue
                    print(f"[run] {model} / {out.parent.parent.name} / {scenario}", flush=True)
                    pending = results_root / "_pending_config.yaml"
                    dump_config(cfg, pending)
                    code = subprocess.run([sys.executable, "-u", str(RUN_EXPERIMENT), "--config", str(pending)]).returncode
                    if code != 0:
                        print(f"[failed] {model} / {out.parent.parent.name} / {scenario} (exit {code}); continuing",
                              flush=True)
                        failures.append({"model": model, "condition": out.parent.parent.name, "scenario": scenario,
                                         "exit_code": code, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (results_root / "_pending_config.yaml").unlink(missing_ok=True)
    if failures:
        print(f"[done] {len(failures)} run(s) failed; see {failures_path}. Rerun this command to retry them.")
        sys.exit(1)
    failures_path.unlink(missing_ok=True)
    print("[done] all runs complete")


if __name__ == "__main__":
    main()
