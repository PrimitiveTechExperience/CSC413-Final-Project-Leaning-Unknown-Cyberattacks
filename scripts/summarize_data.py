"""Export split/class/window-length counts and recording membership before training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import dump_config, experiment_condition, load_config  # noqa: E402
from lidds_gen.data.experiment import build_experiment  # noqa: E402
from lidds_gen.data.report import write_dataset_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    _, _, summary = build_experiment(cfg)
    root = (Path(cfg["paths"]["results_root"]) / "data_summary"
            / experiment_condition(cfg) / cfg["experiment"]["scenario"])
    dump_config(cfg, root / "config.yaml")
    write_dataset_summary(summary, root)
    print(f"[done] Dataset counts and recording manifest -> {root}")


if __name__ == "__main__":
    main()
