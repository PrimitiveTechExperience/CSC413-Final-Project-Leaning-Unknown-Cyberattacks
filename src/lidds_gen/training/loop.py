"""Class-weighted BCE training with early stopping on validation loss (Sec. 4.4)."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.experiment import EncodedSplit
from .dataset import WindowDataset, collate_windows


def class_weights(split: EncodedSplit) -> tuple[float, float]:
    """``(w0, w1)`` from training counts only: w_c = N / (2 N_c)."""
    n0, n1 = split.class_counts()
    n = n0 + n1
    return (n / (2 * n0) if n0 else 1.0, n / (2 * n1) if n1 else 1.0)


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, w0: float, w1: float) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (per * torch.where(y > 0.5, w1, w0)).mean()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(split: EncodedSplit, cfg: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        WindowDataset(split, cfg["data"]["max_length"]),
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_windows,
        num_workers=cfg["training"]["num_workers"],
        generator=g,
    )


@torch.no_grad()
def predict(model: nn.Module, split: EncodedSplit, cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for tokens, lengths, _ in _loader(split, cfg, shuffle=False, seed=0):
        probs.append(torch.sigmoid(model(tokens.to(device), lengths.to(device))).cpu())
    return torch.cat(probs).numpy() if probs else np.zeros(0)


@torch.no_grad()
def _val_loss(model, split, cfg, device, w0, w1) -> float:
    model.eval()
    total, n = 0.0, 0
    for tokens, lengths, y in _loader(split, cfg, shuffle=False, seed=0):
        y = y.to(device)
        total += weighted_bce(model(tokens.to(device), lengths.to(device)), y, w0, w1).item() * len(y)
        n += len(y)
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    train: EncodedSplit,
    val: EncodedSplit,
    cfg: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train and return the model restored to its best-validation-loss epoch, plus history."""
    tcfg = cfg["training"]
    device = resolve_device(tcfg["device"])
    model.to(device)
    w0, w1 = class_weights(train)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    best_loss, best_state, patience, history = float("inf"), None, 0, []
    for epoch in range(tcfg["epochs"]):
        model.train()
        running, seen = 0.0, 0
        for tokens, lengths, y in tqdm(
            _loader(train, cfg, shuffle=True, seed=cfg["seed"] + epoch),
            desc=f"epoch {epoch}", leave=False,
        ):
            tokens, lengths, y = tokens.to(device), lengths.to(device), y.to(device)
            loss = weighted_bce(model(tokens, lengths), y, w0, w1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)

        val_loss = _val_loss(model, val, cfg, device, w0, w1)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= tcfg["early_stopping_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
