"""Class-weighted BCE training with early stopping on validation loss (Sec. 4.4)."""
from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Callable

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
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use a CUDA-enabled PyTorch environment")
    return device


def learning_rate(cfg: dict[str, Any]) -> float:
    """``training.lr``, unless ``training.lr_by_model`` sets one for this model."""
    tcfg = cfg["training"]
    model = cfg.get("experiment", {}).get("model")
    return float((tcfg.get("lr_by_model") or {}).get(model, tcfg["lr"]))


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
    progress: Callable[[dict[str, Any]], None] | None = None,
    checkpoint: str | Path | None = None,
    checkpoint_key: str | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train and return the model restored to its best-validation-loss epoch, plus history.

    ``progress`` (optional) receives live status for monitoring: phase, epoch,
    batch counts, and the history so far. It is throttled to about once per 2 s.

    ``checkpoint`` (optional) is saved after every epoch and resumed from if it
    exists with the same ``checkpoint_key`` (a fingerprint of the run's config), so
    an interrupted run continues from its last completed epoch.
    """
    last_report = [0.0]

    def report(force: bool = False, **status: Any) -> None:
        if progress is None or (not force and time.perf_counter() - last_report[0] < 2.0):
            return
        last_report[0] = time.perf_counter()
        progress({**status, "epochs": tcfg["epochs"], "history": history})

    tcfg = cfg["training"]
    device = resolve_device(tcfg["device"])
    model.to(device)
    w0, w1 = class_weights(train)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate(cfg), weight_decay=tcfg["weight_decay"])

    best_loss, best_state, patience, history = float("inf"), None, 0, []
    start_epoch, stopped = 0, False
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("key") == checkpoint_key:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            best_loss, best_state, patience = state["best_loss"], state["best_state"], state["patience"]
            history, stopped = state["history"], state["stopped"]
            start_epoch = len(history)
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
            print(f"[resume] from checkpoint after epoch {start_epoch}", flush=True)

    def save_checkpoint(stop: bool) -> None:
        if checkpoint is None:
            return
        partial = Path(str(checkpoint) + ".tmp")
        torch.save({"key": checkpoint_key, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "best_loss": best_loss, "best_state": best_state, "patience": patience,
                    "history": history, "stopped": stop, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
                   partial)
        # Windows refuses the replace while another process (antivirus, indexer) has the
        # file open. Retry, then keep the previous checkpoint rather than stop training.
        for attempt in range(10):
            try:
                partial.replace(checkpoint)
                return
            except OSError:
                time.sleep(0.2 * (attempt + 1))
        print(f"[warn] could not update {checkpoint}; keeping the previous checkpoint", flush=True)

    for epoch in range(start_epoch, tcfg["epochs"]):
        if stopped:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        running, seen = 0.0, 0
        loader = _loader(train, cfg, shuffle=True, seed=cfg["seed"] + epoch)
        report(True, phase="train", epoch=epoch, batch=0, batches=len(loader))
        for batch, (tokens, lengths, y) in enumerate(tqdm(loader, desc=f"epoch {epoch}", leave=False), 1):
            tokens, lengths, y = tokens.to(device), lengths.to(device), y.to(device)
            loss = weighted_bce(model(tokens, lengths), y, w0, w1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
            report(phase="train", epoch=epoch, batch=batch, batches=len(loader),
                   train_loss=running / max(seen, 1))

        report(True, phase="validate", epoch=epoch, batch=len(loader), batches=len(loader),
               train_loss=running / max(seen, 1))
        val_loss = _val_loss(model, val, cfg, device, w0, w1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_loss,
                        "epoch_seconds": time.perf_counter() - epoch_start})
        if val_loss < best_loss:
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        stop = patience >= tcfg["early_stopping_patience"]
        save_checkpoint(stop)
        if stop:
            break
        report(True, phase="epoch_done", epoch=epoch, batch=len(loader), batches=len(loader))

    report(True, phase="evaluate", epoch=len(history) - 1)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
