"""Deterministic vocabulary, fitted exclusively on selected training windows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

PAD_ID, UNK_ID = 0, 1


class Vocabulary:
    def __init__(self, tokens: list[str]):
        if tokens[:2] != ["<pad>", "<unk>"] or len(set(tokens)) != len(tokens):
            raise ValueError("Vocabulary must have unique tokens and reserved pad/unk IDs")
        self.itos = tokens
        self.stoi = {s: i for i, s in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, windows: Iterable[Iterable[str]]) -> Vocabulary:
        tokens = {s for window in windows for s in window}
        if tokens.intersection(("<pad>", "<unk>")):
            raise ValueError("System-call names collide with reserved tokens")
        return cls(["<pad>", "<unk>"] + sorted(tokens))

    def encode(self, calls: Iterable[str]) -> list[int]:
        return [self.stoi.get(s, UNK_ID) for s in calls]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
