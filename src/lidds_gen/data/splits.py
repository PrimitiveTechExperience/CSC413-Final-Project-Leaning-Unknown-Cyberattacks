"""Split whole recordings before selecting seen / held-out experiment pools."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
import random
from typing import Sequence


@dataclass(frozen=True)
class RecordingInfo:
    scenario: str
    name: str
    partition: str
    exploit: bool
    split: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return self.scenario, self.partition, self.name


def resplit_recordings(
    recordings: Sequence[RecordingInfo], seed: int = 413,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> list[RecordingInfo]:
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("Split ratios must be three positive fractions summing to 1")
    if len({r.key for r in recordings}) != len(recordings):
        raise ValueError("Duplicate recording identities")
    assignments = {}
    groups = defaultdict(list)
    for r in recordings:
        if r.partition in ("training", "validation"):
            if r.exploit:
                raise ValueError(f"Expected official benign partition: {r.key}")
            assignments[r.key] = "train" if r.partition == "training" else "val"
        elif r.partition in ("test_normal", "test_attack"):
            groups[(r.scenario, r.exploit)].append(r)
        else:
            raise ValueError(f"Unknown partition {r.partition!r}")
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda r: r.key)
        random.Random(f"{seed}:{group[0]}:{group[1]}").shuffle(items)
        # Largest remainder allocation preserves all recordings and exact 14/3/3
        # counts for a 20-recording stratum; tiny strata may have empty splits.
        counts = [int(len(items) * ratio) for ratio in ratios]
        order = sorted(range(3), key=lambda i: -(len(items) * ratios[i] - counts[i]))
        for i in order[:len(items) - sum(counts)]:
            counts[i] += 1
        start = 0
        for split, count in zip(("train", "val", "test"), counts):
            for r in items[start:start + count]:
                assignments[r.key] = split
            start += count
    return [replace(r, split=assignments[r.key]) for r in recordings]


def select_setting(
    recordings: Sequence[RecordingInfo], setting: str, scenario: str,
    seen_scope: str = "all",
) -> dict[str, list[RecordingInfo]]:
    if setting not in ("seen", "held_out") or seen_scope not in ("all", "single"):
        raise ValueError("Invalid experiment setting or seen_scope")
    if not any(r.scenario == scenario for r in recordings):
        raise ValueError(f"Target scenario {scenario!r} not found")
    out = {"train": [], "val": [], "test": []}
    for r in recordings:
        if r.split not in out:
            raise ValueError("Recordings must be re-split before experiment selection")
        if r.split == "test":
            if r.scenario == scenario:
                out["test"].append(r)
        elif setting == "held_out":
            # Target benign recordings are allowed for training only. Threshold
            # selection and early stopping use other scenarios' validation data.
            if r.scenario != scenario or (r.split == "train" and not r.exploit):
                out[r.split].append(r)
        elif seen_scope == "all" or r.scenario == scenario:
            out[r.split].append(r)
    return out
