"""Parse raw LID-DS-2021 recordings into per-scenario window caches.

    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --scenarios CVE-2014-0160 --set data.max_recordings_per_partition=5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS, ScenarioWindowWriter, iter_scenario_recordings, windows_for_recording  # noqa: E402
from lidds_gen.data.recording import list_scenario_zips  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d, paths = cfg["data"], cfg["paths"]
    limit = d.get("max_recordings_per_partition")

    for scenario in args.scenarios:
        n = len(list_scenario_zips(paths["raw_root"], scenario, limit))
        if n == 0:
            print(f"[skip] {scenario}: no recordings under {paths['raw_root']}/{scenario}")
            continue
        writer = ScenarioWindowWriter(scenario)
        recs = iter_scenario_recordings(paths["raw_root"], scenario, d["recon_step_keywords"], limit)
        for rec in tqdm(recs, total=n, desc=scenario):
            if rec.exploit and rec.attack_start_ns is None:
                print(f"[warn] {scenario}/{rec.name}: exploit recording without a non-recon attack time")
            writer.add(rec, windows_for_recording(
                rec,
                seed=cfg["seed"],
                mode=d["window_mode"],
                lengths=d["window_lengths"],
                fixed_length=d["fixed_length"],
                keep_short_final=d["keep_short_final_window"],
            ))
        print(f"[done] {scenario} -> {writer.save(paths['processed_root'])}")


if __name__ == "__main__":
    main()
