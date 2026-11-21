"""Compact per-scenario caches with recording identity for leakage-free splitting.

Cache token IDs are storage codes, not the model vocabulary. The latter is fit
after recording selection. Arrays never contain pickled Python objects.
"""
from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .recording import Recording
from .splits import RecordingInfo
from .windows import Window


def preprocessing_settings(cfg: dict) -> dict:
    """Record settings that determine cached calls, windows, and labels."""
    keys = ("window_mode", "window_lengths", "fixed_length", "keep_short_final_window",
            "recon_step_keywords", "max_recordings_per_partition")
    return {"seed": cfg["seed"], **{k: cfg["data"].get(k) for k in keys}}


@dataclass
class ScenarioWindows:
    recordings: list[RecordingInfo]
    tokens: list[str]
    call_ids: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    target_lengths: np.ndarray
    recording_ids: np.ndarray
    preprocessing: dict | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def window_calls(self, i: int) -> list[str]:
        return [self.tokens[int(t)] for t in self.call_ids[self.offsets[i]:self.offsets[i + 1]]]


class ScenarioWindowWriter:
    def __init__(self, scenario: str, preprocessing: dict | None = None):
        self.scenario = scenario
        self.preprocessing = preprocessing
        self.recordings = []
        self._keys = set()
        self._tokens = {}
        self._calls = array("I")
        self._offsets = array("Q", [0])
        self._labels = array("B")
        self._lengths = array("I")
        self._recording_ids = array("I")

    def add(self, recording: Recording, windows: Iterable[Window]) -> None:
        info = RecordingInfo(recording.scenario, recording.name, recording.partition, recording.exploit)
        if recording.scenario != self.scenario or info.key in self._keys:
            raise ValueError("Wrong scenario or duplicate recording")
        self._keys.add(info.key)
        rec_id = len(self.recordings)
        self.recordings.append(info)
        for window in windows:
            if not window.calls or len(window.calls) > window.target_length:
                raise ValueError("Invalid window length")
            if window.label not in (0, 1) or (window.label and not recording.exploit):
                raise ValueError("Invalid window label")
            for name in window.calls:
                self._calls.append(self._tokens.setdefault(name, len(self._tokens)))
            self._offsets.append(len(self._calls))
            self._labels.append(window.label)
            self._lengths.append(window.target_length)
            self._recording_ids.append(rec_id)

    def save(self, root: str | Path) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.scenario}.npz"
        np.savez_compressed(
            path, call_ids=np.asarray(self._calls, dtype=np.uint32),
            offsets=np.asarray(self._offsets, dtype=np.int64),
            labels=np.asarray(self._labels, dtype=np.uint8),
            target_lengths=np.asarray(self._lengths, dtype=np.int32),
            recording_ids=np.asarray(self._recording_ids, dtype=np.int32),
        )
        path.with_suffix(".json").write_text(json.dumps({
            "version": 1, "scenario": self.scenario, "tokens": list(self._tokens),
            "recordings": [asdict(r) for r in self.recordings],
            "preprocessing": self.preprocessing,
        }, indent=2), encoding="utf-8")
        return path


def load_scenario_windows(root: str | Path, scenario: str) -> ScenarioWindows:
    path = Path(root) / f"{scenario}.npz"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("version") != 1 or meta["scenario"] != scenario:
        raise ValueError(f"{path}: incompatible cache metadata")
    with np.load(path, allow_pickle=False) as arrays:
        return ScenarioWindows(
            [RecordingInfo(**r) for r in meta["recordings"]], meta["tokens"],
            arrays["call_ids"], arrays["offsets"], arrays["labels"],
            arrays["target_lengths"], arrays["recording_ids"],
            meta.get("preprocessing"),
        )
