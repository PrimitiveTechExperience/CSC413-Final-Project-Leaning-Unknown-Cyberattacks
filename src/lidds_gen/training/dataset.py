"""Torch dataset over encoded windows, padded per batch (max 200 calls)."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..data.experiment import EncodedSplit
from ..data.vocab import PAD_ID


class WindowDataset(Dataset):
    def __init__(self, split: EncodedSplit, max_length: int = 200):
        self.split = split
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int):
        seq = torch.as_tensor(self.split.sequences[i][: self.max_length], dtype=torch.long)
        return seq, float(self.split.labels[i])


def collate_windows(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    tokens = torch.full((len(seqs), int(lengths.max())), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tokens[i, : len(s)] = s
    return tokens, lengths, torch.tensor(labels, dtype=torch.float32)
