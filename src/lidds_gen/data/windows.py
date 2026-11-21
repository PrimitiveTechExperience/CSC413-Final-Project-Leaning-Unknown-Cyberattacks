"""Deterministic non-overlapping windows within a recording's thread streams."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import random
from typing import Sequence

from .recording import Recording, Syscall


@dataclass
class Window:
    thread_id: int
    calls: list[str]
    label: int
    target_length: int


def make_windows(
    calls: Sequence[Syscall], recording: Recording, rng: random.Random,
    lengths: Sequence[int], keep_short_final: bool = True,
) -> list[Window]:
    if not lengths or any(not isinstance(n, int) or n <= 0 for n in lengths):
        raise ValueError("Window lengths must be positive integers")
    streams = defaultdict(list)
    for call in calls:
        streams[call.thread_id].append(call)
    windows = []
    for tid in sorted(streams):
        stream = streams[tid]
        start = 0
        while start < len(stream):
            length = rng.choice(lengths)
            chunk = stream[start:start + length]
            if len(chunk) < length and not keep_short_final:
                break
            malicious = recording.exploit and recording.attack_start_ns is not None and any(
                c.timestamp_ns >= recording.attack_start_ns for c in chunk
            )
            windows.append(Window(tid, [c.name for c in chunk], int(malicious), length))
            start += length
    return windows


def windows_for_recording(
    recording: Recording, seed: int = 413, mode: str = "variable",
    lengths: Sequence[int] = (50, 100, 200), fixed_length: int = 100,
    keep_short_final: bool = True,
) -> list[Window]:
    if mode not in ("fixed", "variable"):
        raise ValueError(f"Unknown window mode {mode!r}")
    # Stable across processes and independent of traversal order / Python hash seed.
    identity = f"{seed}:{recording.scenario}:{recording.partition}:{recording.name}"
    derived_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    return make_windows(recording.syscalls, recording, random.Random(derived_seed),
                        [fixed_length] if mode == "fixed" else lengths, keep_short_final)
