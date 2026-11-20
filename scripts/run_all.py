"""Full grid: {lstm, transformer} x {seen, held_out} x 15 target scenarios.

Completed runs (with an existing metrics.json) are skipped, so the grid can be resumed.

    python scripts/run_all.py --models lstm --scenarios CVE-2014-0160 ZipSlip
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from run_experiment import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="*", default=["lstm", "transformer"])
    ap.add_argument("--settings", nargs="*", default=["seen", "held_out"])
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    base = load_config(args.config, args.set)
    for model in args.models:
        for setting in args.settings:
            for scenario in args.scenarios:
                cfg = copy.deepcopy(base)
                cfg["experiment"].update(model=model, setting=setting, scenario=scenario)
                out = Path(cfg["paths"]["results_root"]) / model / setting / scenario / "metrics.json"
                if out.exists():
                    print(f"[skip] {out}")
                    continue
                print(f"[run] {model} / {setting} / {scenario}")
                run(cfg)


if __name__ == "__main__":
    main()
