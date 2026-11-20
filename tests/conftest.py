import json
import zipfile
from pathlib import Path

import pytest

T0 = 1_631_000_000  # seconds


def sc_line(ts_ns: int, tid: int, name: str, direction: str = ">") -> str:
    return f"{ts_ns} 0 100 apache2 {tid} {name} {direction} fd=3 size=10"


def write_recording(path: Path, lines: list[str], meta: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{path.stem}.sc", "\n".join(lines) + "\n")
        zf.writestr(f"{path.stem}.json", json.dumps(meta))
    return path


@pytest.fixture
def attack_zip(tmp_path: Path) -> Path:
    """Two threads, warm-up ends at T0+1s, recon at T0+2s, exploit at T0+3s."""
    ns = 10**9
    lines = [
        sc_line(T0 * ns + int(0.5 * ns), 1, "read"),          # before warm-up: dropped
        sc_line((T0 + 1) * ns + 1, 1, "open"),
        sc_line((T0 + 1) * ns + 2, 1, "open", "<"),           # exit event: dropped
        sc_line((T0 + 1) * ns + 3, 2, "mmap"),
        sc_line((T0 + 2) * ns + 1, 1, "socket"),              # recon (still benign)
        sc_line((T0 + 3) * ns + 1, 2, "execve"),              # malicious
        sc_line((T0 + 3) * ns + 2, 1, "write"),               # malicious
    ]
    meta = {
        "exploit": True,
        "time": {
            "warmup_end": {"absolute": T0 + 1.0},
            "exploit": [
                {"absolute": T0 + 2.0, "name": "port_scan"},
                {"absolute": T0 + 3.0, "name": "exploit"},
            ],
        },
    }
    return write_recording(tmp_path / "rec_attack.zip", lines, meta)
