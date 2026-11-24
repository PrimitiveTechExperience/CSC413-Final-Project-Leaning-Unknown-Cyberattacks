"""Read LID-DS-2021 ZIPs without extracting logs or using call arguments.

Format reference: LID-DS/LID-DS, dataloader/{syscall,recording}_2021.py.
Trace timestamps are integer nanoseconds; metadata absolute times are seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Sequence
import json
import zipfile


@dataclass(frozen=True, slots=True)
class Syscall:
    timestamp_ns: int
    thread_id: int
    name: str


@dataclass
class Recording:
    name: str
    scenario: str
    partition: str
    exploit: bool
    warmup_end_ns: int | None
    attack_start_ns: int | None
    syscalls: list[Syscall]
    attack_steps: list[dict] = field(default_factory=list)


PARTITIONS = {
    "training": "training", "validation": "validation",
    "test_normal": "test/normal", "test_attack": "test/normal_and_attack",
}


def _absolute_ns(event: dict) -> int:
    return int(Decimal(str(event["absolute"])) * 10**9)


def _split_spaced_process_name(line: str) -> list[str]:
    """Re-split a line whose process name contains spaces (e.g. ``puma 002``).

    Columns are: timestamp, uid, pid, process name, thread id, syscall, direction,
    args. Find the first direction marker whose thread-id column (two before it) is
    an integer, then join the name tokens. Returns the unchanged split if none fits.
    """
    tokens = line.split()
    for i in range(7, len(tokens)):
        if tokens[i] in (">", "<") and tokens[i - 2].isdigit():
            rest = line.split(maxsplit=i + 1)[i + 1:]
            return [*tokens[:3], " ".join(tokens[3:i - 2]), *tokens[i - 2:i + 1], *rest]
    return tokens


def load_recording(
    path: str | Path, scenario: str, partition: str,
    recon_keywords: Sequence[str] = (),
) -> Recording:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        # Match by basename so ZIPs containing a directory are also supported.
        def member(suffix: str) -> str:
            matches = [n for n in archive.namelist() if Path(n).name == path.stem + suffix]
            if len(matches) != 1:
                raise ValueError(f"{path}: expected one {path.stem + suffix} member")
            return matches[0]

        with archive.open(member(".json")) as stream:
            meta = json.load(stream, parse_float=Decimal)
        exploit = meta["exploit"]
        if not isinstance(exploit, bool):
            raise ValueError(f"{path}: exploit must be a boolean")
        timing = meta["time"]
        warmup = _absolute_ns(timing["warmup_end"])
        steps = timing.get("exploit", [])
        if exploit and not steps:
            raise ValueError(f"{path}: exploit recording has no attack timing")
        keywords = tuple(k.lower() for k in recon_keywords)
        step_audit = [{"name": step.get("name", ""), "timestamp_ns": _absolute_ns(step),
                       "reconnaissance": any(k in step.get("name", "").lower() for k in keywords)}
                      for step in steps] if exploit else []
        attack_times = [step["timestamp_ns"] for step in step_audit if not step["reconnaissance"]]
        attack = min(attack_times) if attack_times else None
        calls = []
        with archive.open(member(".sc")) as stream:
            for lineno, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                fields = raw.decode("utf-8").split(maxsplit=7)
                try:
                    if len(fields) >= 7 and fields[6] not in (">", "<"):
                        fields = _split_spaced_process_name(raw.decode("utf-8"))
                    if len(fields) < 7 or fields[6] not in (">", "<"):
                        raise ValueError("invalid syscall columns")
                    if fields[6] == "<":
                        continue
                    timestamp, thread = int(fields[0]), int(fields[4])
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid syscall entry") from exc
                if timestamp >= warmup:
                    calls.append(Syscall(timestamp, thread, fields[5]))
    return Recording(path.stem, scenario, partition, exploit, warmup, attack, calls, step_audit)


def list_scenario_zips(
    raw_root: str | Path, scenario: str, limit: int | None = None,
) -> list[tuple[Path, str]]:
    if limit is not None and limit <= 0:
        raise ValueError("max_recordings_per_partition must be positive")
    out = []
    for partition, folder in PARTITIONS.items():
        paths = sorted((Path(raw_root) / scenario / folder).glob("*.zip"))
        out.extend((p, partition) for p in (paths[:limit] if limit is not None else paths))
    return out


def iter_scenario_recordings(
    raw_root: str | Path, scenario: str, recon_keywords: Sequence[str] = (),
    limit: int | None = None,
) -> Iterator[Recording]:
    for path, partition in list_scenario_zips(raw_root, scenario, limit):
        yield load_recording(path, scenario, partition, recon_keywords)
