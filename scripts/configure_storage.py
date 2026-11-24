"""Write a local experiment config with all dataset and result paths on one drive."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import dump_config, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="storage folder, e.g. D:/CSC413-LIDDS")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", help="default: configs/local[-gpu][-pilot].yaml")
    ap.add_argument("--pilot", action="store_true", help="two scenarios, five recordings per partition, one epoch")
    ap.add_argument("--gpu", action="store_true", help="require CUDA and separate GPU result/config paths")
    ap.add_argument("--batch-size", type=int, help="use a batch size verified on the target device")
    ap.add_argument("--results-name", help="results folder under <root>/results (default: pilot/full[_gpu])")
    args = ap.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        ap.error("batch size must be positive")
    cfg = load_config(args.config)
    root = Path(args.root).resolve()
    results_name = "pilot" if args.pilot else "full"
    if args.gpu:
        results_name += "_gpu"
        cfg["training"]["device"] = "cuda"
    if args.results_name:
        results_name = args.results_name
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    cfg["paths"].update(raw_root=str(root / "raw" / "LID-DS-2021"),
                        processed_root=str(root / ("processed_pilot" if args.pilot else "processed")),
                        results_root=str(root / "results" / results_name))
    if args.pilot:
        cfg["data"].update(scenarios=["CVE-2014-0160", "CVE-2020-23839"], max_recordings_per_partition=5)
        cfg["training"]["epochs"] = 1
    name = "local" + ("-gpu" if args.gpu else "") + ("-pilot" if args.pilot else "") + ".yaml"
    output = args.output or str(Path("configs") / name)
    dump_config(cfg, output)
    print(f"[done] {output}")
    for key, value in cfg["paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
