"""Export dataset counts without requiring model training or torch."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_dataset_summary(summary: dict, root: str | Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ["split", "target_length", "vocab_size", "recordings", "benign_recordings",
              "attack_recordings", "windows_before_cap", "windows", "benign_windows", "malicious_windows",
              "malicious_window_rate", "unknown_token_rate", "short_final_windows"]
    with (root / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, counts in summary["splits"].items():
            overall = {key: value for key, value in counts.items()
                       if key not in ("by_length", "windows_available")}
            overall["windows_before_cap"] = sum(counts.get("windows_available", {}).values())
            writer.writerow({"split": split, "target_length": "all",
                             "vocab_size": summary["vocab_size"], **overall})
            for length, values in counts["by_length"].items():
                writer.writerow({"split": split, "target_length": length, **values})
