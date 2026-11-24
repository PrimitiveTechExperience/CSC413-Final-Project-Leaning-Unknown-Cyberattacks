"""Extract scenario archives while keeping individual recording ZIPs compressed."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402
from lidds_gen.data.recording import PARTITIONS  # noqa: E402


def extract_scenario(source: Path, root: Path, scenario: str) -> tuple[int, int]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = []
        for member in archive.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if not name.parts or name.parts[0] != scenario or member.is_dir():
                continue
            if name.is_absolute() or any(part in ("..", ".") or ":" in part for part in name.parts):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if name.suffix != ".zip" or name.parent.relative_to(scenario).as_posix() not in PARTITIONS.values():
                continue
            target = (root / Path(*name.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
            members.append((member, target))
        if not members:
            raise ValueError(f"No recording ZIPs in {source}; check the scenario name and archive layout")
        pending = [(member, target) for member, target in members
                   if not (target.exists() and target.stat().st_size == member.file_size
                           and zipfile.is_zipfile(target))]
        required = sum(member.file_size for member, _ in pending)
        if shutil.disk_usage(root).free < required + 1024**3:
            raise OSError(f"Insufficient space to extract {source}: need {required / 1024**3:.1f} GiB + 1 GiB reserve")
        for member, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".zip.partial")
            with archive.open(member) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            partial.replace(target)
    return len(pending), len(members) - len(pending)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--archives-root", required=True)
    ap.add_argument("--scenarios", nargs="+")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for scenario in args.scenarios or cfg["data"].get("scenarios", SCENARIOS):
        source = Path(args.archives_root) / f"{scenario}.zip"
        written, skipped = extract_scenario(source, Path(cfg["paths"]["raw_root"]), scenario)
        print(f"[done] {scenario}: extracted {written}, already present {skipped}", flush=True)


if __name__ == "__main__":
    main()
