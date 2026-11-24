"""Scenario extraction must retain recordings and reject paths outside the destination."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extract_scenarios.py"
SPEC = importlib.util.spec_from_file_location("extract_scenarios", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract_keeps_recordings_compressed_and_is_resumable(tmp_path, attack_zip):
    outer = tmp_path / "S.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.write(attack_zip, "S/test/normal_and_attack/rec_attack.zip")
        archive.writestr("__MACOSX/S/test/normal_and_attack/._rec_attack.zip", b"metadata")
    root = tmp_path / "raw"
    assert MODULE.extract_scenario(outer, root, "S") == (1, 0)
    assert list(root.rglob("*.zip")) == [root / "S/test/normal_and_attack/rec_attack.zip"]
    assert not list(root.rglob("*.sc"))
    assert MODULE.extract_scenario(outer, root, "S") == (0, 1)


def test_extract_rejects_path_traversal(tmp_path):
    outer = tmp_path / "S.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("S/../../outside.zip", b"bad")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        MODULE.extract_scenario(outer, tmp_path / "raw", "S")
    assert not (tmp_path / "outside.zip").exists()
