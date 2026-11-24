"""Compact per-scenario caches with recording identity for leakage-free splitting.

Cache token IDs are storage codes, not the model vocabulary. The latter is fit
after recording selection. Arrays never contain pickled Python objects.
"""
from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
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
    """Accumulates windows; call IDs spill to a temporary file in fixed-size chunks so
    large scenarios (hundreds of millions of calls) never need one huge in-memory array."""

    SPILL_CALLS = 16 * 1024 * 1024  # 64 MiB of uint32 per flush

    def __init__(self, scenario: str, preprocessing: dict | None = None,
                 spill_dir: str | Path | None = None):
        self.scenario = scenario
        self.preprocessing = preprocessing
        self.recordings = []
        self._keys = set()
        self._tokens = {}
        self._calls = array("I")
        self._spill_dir = spill_dir
        self._spill = None
        self._spilled = 0
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
            self._offsets.append(self._spilled + len(self._calls))
            self._labels.append(window.label)
            self._lengths.append(window.target_length)
            self._recording_ids.append(rec_id)
            if len(self._calls) >= self.SPILL_CALLS:
                self._flush()

    def _flush(self) -> None:
        if not self._calls:
            return
        if self._spill is None:
            if self._spill_dir is not None:
                Path(self._spill_dir).mkdir(parents=True, exist_ok=True)
            self._spill = tempfile.NamedTemporaryFile(
                prefix=f"{self.scenario}.calls.", suffix=".tmp", dir=self._spill_dir, delete=False)
        self._calls.tofile(self._spill)
        self._spilled += len(self._calls)
        self._calls = array("I")

    def save(self, root: str | Path) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.scenario}.npz"
        spill_path = None
        if self._spill is not None:
            self._flush()
            self._spill.close()
            spill_path = self._spill.name
            # memmap: numpy writes it into the zip in buffered chunks, not all at once.
            call_ids = np.memmap(spill_path, dtype=np.uint32, mode="r", shape=(self._spilled,))
        else:
            call_ids = np.asarray(self._calls, dtype=np.uint32)
        try:
            np.savez_compressed(
                path, call_ids=call_ids,
                offsets=np.asarray(self._offsets, dtype=np.int64),
                labels=np.asarray(self._labels, dtype=np.uint8),
                target_lengths=np.asarray(self._lengths, dtype=np.int32),
                recording_ids=np.asarray(self._recording_ids, dtype=np.int32),
            )
        finally:
            if spill_path is not None:
                del call_ids  # release the mapping so Windows can delete the file
                os.remove(spill_path)
                self._spill = None
        path.with_suffix(".json").write_text(json.dumps({
            "version": 1, "scenario": self.scenario, "tokens": list(self._tokens),
            "recordings": [asdict(r) for r in self.recordings],
            "preprocessing": self.preprocessing,
        }, indent=2), encoding="utf-8")
        return path


def load_scenario_metadata(root: str | Path, scenario: str) -> dict:
    """Recordings, storage tokens, and preprocessing settings without loading arrays."""
    path = Path(root) / f"{scenario}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("version") != 1 or meta["scenario"] != scenario:
        raise ValueError(f"{path}: incompatible cache metadata")
    meta["recordings"] = [RecordingInfo(**r) for r in meta["recordings"]]
    return meta


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
