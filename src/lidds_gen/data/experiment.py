"""Assemble cached windows into recording-disjoint, encoded experiment splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .splits import resplit_recordings, select_setting
from .store import load_scenario_windows, preprocessing_settings
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


def build_experiment(cfg: dict[str, Any]):
    root = Path(cfg["paths"]["processed_root"])
    exp, data = cfg["experiment"], cfg["data"]
    # Require the full configured grid; silently training on an incomplete pool
    # would change the held-out experiment. Tests/debug jobs can specify a subset.
    from . import SCENARIOS
    scenarios = data.get("scenarios", SCENARIOS)
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Duplicate scenarios")
    if exp["setting"] == "seen" and exp.get("seen_scope", "all") == "single":
        scenarios = [exp["scenario"]]
    stores = {s: load_scenario_windows(root, s) for s in scenarios}
    expected = preprocessing_settings(cfg)
    for scenario, store in stores.items():
        if store.preprocessing != expected:
            raise ValueError(f"{scenario}: preprocessing config differs from cache; rerun prepare_data.py")
    infos = resplit_recordings(
        [r for store in stores.values() for r in store.recordings],
        seed=cfg["seed"], ratios=data["test_resplit"],
    )
    selected = select_setting(infos, exp["setting"], exp["scenario"], exp.get("seen_scope", "all"))
    membership = {r.key: split for split, recs in selected.items() for r in recs}
    indices = {split: [] for split in selected}
    for scenario, store in stores.items():
        for i, rec_id in enumerate(store.recording_ids):
            split = membership.get(store.recordings[int(rec_id)].key)
            if split is not None:
                if store.offsets[i + 1] - store.offsets[i] > data["max_length"]:
                    raise ValueError("Cached window exceeds max_length; regenerate the cache")
                indices[split].append((scenario, i))
    for split, items in indices.items():
        if not items:
            raise ValueError(f"Empty {split} split; provide more recordings / check scenario selection")

    vocab = Vocabulary.build(stores[s].window_calls(i) for s, i in indices["train"])
    # Translate local storage IDs once per scenario instead of per system call.
    mappings = {s: np.asarray(vocab.encode(store.tokens), dtype=np.int64) for s, store in stores.items()}
    splits, summary = {}, {"vocab_size": len(vocab), "splits": {}}
    for split, items in indices.items():
        sequences, labels, lengths, names = [], [], [], []
        for scenario, i in items:
            store = stores[scenario]
            ids = store.call_ids[store.offsets[i]:store.offsets[i + 1]]
            sequences.append(mappings[scenario][ids])
            labels.append(int(store.labels[i]))
            lengths.append(int(store.target_lengths[i]))
            names.append(scenario)
        encoded = EncodedSplit(sequences, labels, lengths, names)
        splits[split] = encoded
        n0, n1 = encoded.class_counts()
        tokens = sum(len(seq) for seq in sequences)
        summary["splits"][split] = {
            "recordings": len(selected[split]),
            "benign_recordings": sum(not r.exploit for r in selected[split]),
            "attack_recordings": sum(r.exploit for r in selected[split]),
            "windows": len(encoded), "benign_windows": n0, "malicious_windows": n1,
            "unknown_token_rate": sum(int(np.sum(seq == UNK_ID)) for seq in sequences) / tokens,
        }
    if not all(splits["train"].class_counts()):
        raise ValueError("Training split must contain both benign and malicious windows")
    return vocab, splits, summary
