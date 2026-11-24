"""Assemble cached windows into recording-disjoint, encoded experiment splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from .splits import resplit_recordings, select_setting
from .store import load_scenario_metadata, load_scenario_windows, preprocessing_settings
from .vocab import UNK_ID, Vocabulary


@dataclass
class EncodedSplit:
    sequences: list[np.ndarray]
    labels: list[int]
    target_lengths: list[int]
    scenarios: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def class_counts(self) -> tuple[int, int]:
        positive = int(np.sum(self.labels))
        return len(self) - positive, positive


SPLITS = ("train", "val", "test")


def _split_seed(seed: int, scenario: str, split: str) -> list[int]:
    # Stable across runs and settings (unlike hash()), so the target's test sample
    # is identical in seen and held-out runs.
    return [seed, zlib.crc32(scenario.encode("utf-8")), SPLITS.index(split)]


def build_experiment(cfg: dict[str, Any]):
    root = Path(cfg["paths"]["processed_root"])
    exp, data = cfg["experiment"], cfg["data"]
    # Require the full configured grid; silently training on an incomplete pool
    # would change the held-out experiment. Tests/debug jobs can specify a subset.
    from . import SCENARIOS
    scenarios = data.get("scenarios", SCENARIOS)
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Duplicate scenarios")
    if exp["setting"] == "seen" and exp.get("seen_scope", "single") == "single":
        scenarios = [exp["scenario"]]
    if "max_windows_per_split" not in data:
        # Explicit null means uncapped; a missing key means a stale config copy.
        raise ValueError("data.max_windows_per_split is missing; regenerate the config "
                         "(scripts/configure_storage.py) or set it to null for no cap")
    caps = data["max_windows_per_split"] or {}
    if set(caps) - set(SPLITS) or any(c is not None and int(c) <= 0 for c in caps.values()):
        raise ValueError("max_windows_per_split takes positive caps for train/val/test")
    metas = {s: load_scenario_metadata(root, s) for s in scenarios}
    expected = preprocessing_settings(cfg)
    for scenario, meta in metas.items():
        if meta.get("preprocessing") != expected:
            raise ValueError(f"{scenario}: preprocessing config differs from cache; rerun prepare_data.py")
    infos = resplit_recordings(
        [r for meta in metas.values() for r in meta["recordings"]],
        seed=cfg["seed"], ratios=data["test_resplit"],
    )
    selected = select_setting(infos, exp["setting"], exp["scenario"], exp.get("seen_scope", "single"))
    recording_keys = [{r.key for r in selected[split]} for split in SPLITS]
    if any(recording_keys[i] & recording_keys[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Recording leakage between experiment splits")
    membership = {r.key: split for split, recs in selected.items() for r in recs}

    # Load one scenario at a time and keep only the sampled windows (storage IDs),
    # so peak memory is one scenario's cache rather than all fifteen.
    sampled = {split: [] for split in SPLITS}  # (scenario, storage ids, label, target length)
    available = {split: {} for split in SPLITS}
    for scenario in scenarios:
        store = load_scenario_windows(root, scenario)
        rec_split = np.array([membership.get(r.key, "") for r in store.recordings])
        window_split = rec_split[store.recording_ids]
        sizes = np.diff(store.offsets)
        for split in SPLITS:
            candidates = np.flatnonzero(window_split == split)
            if not len(candidates):
                continue
            if sizes[candidates].max() > data["max_length"]:
                raise ValueError("Cached window exceeds max_length; regenerate the cache")
            available[split][scenario] = int(len(candidates))
            cap = caps.get(split)
            if cap is not None and len(candidates) > int(cap):
                rng = np.random.default_rng(_split_seed(cfg["seed"], scenario, split))
                candidates = np.sort(rng.choice(candidates, size=int(cap), replace=False))
            for i in candidates:
                ids = np.array(store.call_ids[store.offsets[i]:store.offsets[i + 1]])
                sampled[split].append((scenario, ids, int(store.labels[i]), int(store.target_lengths[i])))
        metas[scenario]["tokens"] = store.tokens
        del store, rec_split, window_split, sizes
    for split, items in sampled.items():
        if not items:
            raise ValueError(f"Empty {split} split; provide more recordings / check scenario selection")

    # Vocabulary from sampled training windows only (no validation/test leakage).
    train_ids = {s: set() for s in scenarios}
    for scenario, ids, _, _ in sampled["train"]:
        train_ids[scenario].update(np.unique(ids).tolist())
    vocab = Vocabulary.build([metas[s]["tokens"][i] for i in sorted(ids)] for s, ids in train_ids.items())
    # Translate local storage IDs once per scenario instead of per system call.
    mappings = {s: np.asarray(vocab.encode(meta["tokens"]), dtype=np.int64) for s, meta in metas.items()}
    splits, summary = {}, {"vocab_size": len(vocab), "window_caps": {s: caps.get(s) for s in SPLITS},
                           "splits": {}, "recording_manifest": {}}
    for split, items in sampled.items():
        sequences, labels, lengths, names = [], [], [], []
        for scenario, ids, label, length in items:
            sequences.append(mappings[scenario][ids])
            labels.append(label)
            lengths.append(length)
            names.append(scenario)
        sampled[split] = None  # release storage-ID copies once encoded
        encoded = EncodedSplit(sequences, labels, lengths, names)
        splits[split] = encoded
        n0, n1 = encoded.class_counts()
        tokens = sum(len(seq) for seq in sequences)
        summary["splits"][split] = {
            "recordings": len(selected[split]),
            "benign_recordings": sum(not r.exploit for r in selected[split]),
            "attack_recordings": sum(r.exploit for r in selected[split]),
            "windows_available": available[split], "windows": len(encoded), "benign_windows": n0, "malicious_windows": n1,
            "malicious_window_rate": n1 / len(encoded),
            "unknown_token_rate": sum(int(np.sum(seq == UNK_ID)) for seq in sequences) / tokens,
            "by_length": {},
        }
        for length in sorted(set(lengths)):
            positions = [i for i, target in enumerate(lengths) if target == length]
            positive = sum(labels[i] for i in positions)
            summary["splits"][split]["by_length"][str(length)] = {
                "windows": len(positions), "benign_windows": len(positions) - positive,
                "malicious_windows": positive, "malicious_window_rate": positive / len(positions),
                "short_final_windows": sum(len(sequences[i]) < length for i in positions),
            }
        summary["recording_manifest"][split] = [list(r.key) for r in sorted(selected[split], key=lambda r: r.key)]
    if not all(splits["train"].class_counts()):
        raise ValueError("Training split must contain both benign and malicious windows")
    return vocab, splits, summary
