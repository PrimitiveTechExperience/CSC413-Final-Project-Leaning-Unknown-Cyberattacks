"""Parse raw LID-DS-2021 recordings into per-scenario window caches.

    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --scenarios CVE-2014-0160 --set data.max_recordings_per_partition=5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS, ScenarioWindowWriter, iter_scenario_recordings, windows_for_recording  # noqa: E402
from lidds_gen.data.recording import list_scenario_zips  # noqa: E402
from lidds_gen.data.store import preprocessing_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d, paths = cfg["data"], cfg["paths"]
    limit = d.get("max_recordings_per_partition")
    scenarios = args.scenarios or d.get("scenarios", SCENARIOS)
    listings = {scenario: list_scenario_zips(paths["raw_root"], scenario, limit) for scenario in scenarios}
    for scenario, listing in listings.items():
        missing = {"training", "validation", "test_normal", "test_attack"} - {part for _, part in listing}
        if missing:
            raise FileNotFoundError(f"{scenario}: missing recording partitions {sorted(missing)} "
                                    f"under {paths['raw_root']}; check the dataset extraction and raw_root")
    for scenario, listing in listings.items():
        n = len(listing)
        writer = ScenarioWindowWriter(scenario, preprocessing_settings(cfg), spill_dir=paths["processed_root"])
        audit = []
        recs = iter_scenario_recordings(paths["raw_root"], scenario, d["recon_step_keywords"], limit)
        for rec in tqdm(recs, total=n, desc=scenario):
            if rec.exploit and rec.attack_start_ns is None:
                print(f"[warn] {scenario}/{rec.name}: exploit recording without a non-recon attack time")
            windows = windows_for_recording(
                rec,
                seed=cfg["seed"],
                mode=d["window_mode"],
                lengths=d["window_lengths"],
                fixed_length=d["fixed_length"],
                keep_short_final=d["keep_short_final_window"],
            )
            writer.add(rec, windows)
            audit.append({
                "name": rec.name, "partition": rec.partition, "exploit": rec.exploit,
                "warmup_end_ns": rec.warmup_end_ns, "attack_start_ns": rec.attack_start_ns,
                "attack_steps": rec.attack_steps, "retained_entry_events": len(rec.syscalls),
                "threads": len({call.thread_id for call in rec.syscalls}),
                "windows": len(windows), "malicious_windows": sum(w.label for w in windows),
                "short_final_windows": sum(len(w.calls) < w.target_length for w in windows),
            })
            if rec.exploit and not any(w.label for w in windows):
                print(f"[warn] {scenario}/{rec.name}: attack recording produced no malicious windows")
        print(f"[done] {scenario} -> {writer.save(paths['processed_root'])}")
        audit_path = Path(paths["processed_root"]) / f"{scenario}.audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
