"""Exercise raw ZIPs through experiment assembly and the training CLI."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import T0, sc_line, write_recording
from lidds_gen.config import load_config
from lidds_gen.data import iter_scenario_recordings, load_recording, windows_for_recording
from lidds_gen.data.experiment import build_experiment
from lidds_gen.data.recording import list_scenario_zips
from lidds_gen.data.splits import RecordingInfo, resplit_recordings, select_setting
from lidds_gen.data.store import ScenarioWindowWriter, preprocessing_settings
from lidds_gen.data.vocab import UNK_ID

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline_config(tmp_path):
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["paths"] = {"raw_root": str(tmp_path / "raw"),
                    "processed_root": str(tmp_path / "processed"),
                    "results_root": str(tmp_path / "results")}
    cfg["data"].update(scenarios=["A", "B"], window_mode="fixed", fixed_length=2)
    cfg["experiment"].update(scenario="A", setting="held_out")
    cfg["model"].update(embed_dim=8, dropout=0.0)
    cfg["model"]["lstm"].update(hidden_size=8, num_layers=1)
    cfg["model"]["transformer"].update(num_layers=1, num_heads=2, ff_dim=16)
    cfg["training"].update(epochs=1, batch_size=128, device="cpu")
    for scenario in ("A", "B"):
        for partition, count, exploit in (
            ("training", 2, False), ("validation", 2, False),
            ("test/normal", 20, False), ("test/normal_and_attack", 20, True),
        ):
            for i in range(count):
                name = f"{scenario}_{partition.replace('/', '_')}_{i}"
                # Each recording has a distinct marker to expose recording leaks.
                names = ["read", f"attack_{scenario}" if exploit else "read", name]
                lines = [sc_line((T0 + j) * 10**9, 1, s) for j, s in enumerate(names)]
                meta = {"exploit": exploit, "time": {
                    "warmup_end": {"absolute": T0},
                    "exploit": [{"absolute": T0 + 1, "name": "exploit"}] if exploit else [],
                }}
                write_recording(Path(cfg["paths"]["raw_root"]) / scenario / partition / f"{name}.zip",
                                lines, meta)
        writer = ScenarioWindowWriter(scenario, preprocessing_settings(cfg))
        for rec in iter_scenario_recordings(cfg["paths"]["raw_root"], scenario):
            writer.add(rec, windows_for_recording(rec, seed=cfg["seed"], mode="fixed", fixed_length=2))
        writer.save(cfg["paths"]["processed_root"])
    return cfg


def test_assembly_has_no_held_out_or_vocabulary_leakage(pipeline_config):
    cfg = pipeline_config
    vocab, splits, summary = build_experiment(cfg)
    assert "attack_A" not in vocab.stoi
    assert "attack_B" in vocab.stoi
    assert set(splits["val"].scenarios) == {"B"}
    assert set(splits["test"].scenarios) == {"A"}
    assert any(s == "A" and y == 0 for s, y in zip(splits["train"].scenarios, splits["train"].labels))
    assert not any(s == "A" and y == 1 for s, y in zip(splits["train"].scenarios, splits["train"].labels))
    assert any(UNK_ID in seq for seq in splits["test"].sequences)
    assert summary["splits"]["train"]["unknown_token_rate"] == 0
    assert summary["splits"]["test"]["unknown_token_rate"] > 0

    # Marker calls from validation/test must not enter the learned vocabulary.
    infos = []
    for scenario in ("A", "B"):
        infos.extend(RecordingInfo(r.scenario, r.name, r.partition, r.exploit)
                     for r in iter_scenario_recordings(cfg["paths"]["raw_root"], scenario))
    selected = select_setting(resplit_recordings(infos, cfg["seed"]), "held_out", "A")
    assert all(r.name not in vocab.stoi for r in selected["val"] + selected["test"])

    # Both architectures/settings evaluate exactly the same target recordings.
    seen_cfg = copy.deepcopy(cfg)
    seen_cfg["experiment"].update(setting="seen", seen_scope="single")
    seen_vocab, seen, _ = build_experiment(seen_cfg)
    assert set(seen["train"].scenarios) == {"A"}
    assert "attack_A" in seen_vocab.stoi
    assert seen["test"].labels == splits["test"].labels
    assert seen["test"].target_lengths == splits["test"].target_lengths


def test_recordings_do_not_cross_splits_and_selection_is_order_independent():
    infos = [RecordingInfo("A", str(i), "test_attack", True) for i in range(20)]
    first = resplit_recordings(infos, seed=413)
    second = resplit_recordings(list(reversed(infos)), seed=413)
    assert {r.key: r.split for r in first} == {r.key: r.split for r in second}
    keys = [{r.key for r in first if r.split == s} for s in ("train", "val", "test")]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    with pytest.raises(ValueError, match="Duplicate recording"):
        resplit_recordings(infos + infos[:1])


def test_changed_preprocessing_rejects_stale_cache(pipeline_config):
    pipeline_config["data"]["fixed_length"] = 3
    with pytest.raises(ValueError, match="rerun prepare_data"):
        build_experiment(pipeline_config)


def test_missing_scenario_is_not_silently_dropped(pipeline_config):
    pipeline_config["data"]["scenarios"].append("missing")
    with pytest.raises(FileNotFoundError):
        build_experiment(pipeline_config)


def test_partition_limit_is_deterministic(pipeline_config):
    root = pipeline_config["paths"]["raw_root"]
    paths = list_scenario_zips(root, "A", limit=2)
    assert len(paths) == 8
    assert paths == list_scenario_zips(root, "A", limit=2)


def test_parser_preserves_nanosecond_boundaries(tmp_path):
    ns = T0 * 10**9
    meta = {"exploit": True, "time": {
        "warmup_end": {"absolute": f"{T0}.000000001"},
        "exploit": [{"absolute": f"{T0}.000000002", "name": "exploit"}],
    }}
    path = write_recording(tmp_path / "precise.zip",
                           [sc_line(ns + i, 1, "read") for i in range(3)], meta)
    rec = load_recording(path, "A", "test_attack")
    assert [c.timestamp_ns for c in rec.syscalls] == [ns + 1, ns + 2]
    assert [w.label for w in windows_for_recording(rec, mode="fixed", fixed_length=1)] == [0, 1]


def test_missing_attack_timing_cannot_silently_become_benign(tmp_path):
    path = write_recording(tmp_path / "invalid.zip", [], {
        "exploit": True, "time": {"warmup_end": {"absolute": T0}, "exploit": []},
    })
    with pytest.raises(ValueError, match="no attack timing"):
        load_recording(path, "A", "test_attack")


def test_prepare_and_train_cli_for_both_models_and_settings(pipeline_config, tmp_path):
    pytest.importorskip("torch")
    import yaml

    cfg = pipeline_config
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    prepared = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"),
                               "--config", str(path)], capture_output=True, text=True)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_all.py"), "--config", str(path),
        "--scenarios", "A", "--set", "experiment.seen_scope=single",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for model in ("lstm", "transformer"):
        for setting in ("seen", "held_out"):
            out = Path(cfg["paths"]["results_root"]) / model / setting / "A"
            metrics = json.loads((out / "metrics.json").read_text())
            assert len(metrics["history"]) == 1
            assert np.isfinite(metrics["history"][0]["train_loss"])
            assert metrics["test"]["n"] == 12
            assert 0 <= metrics["test"]["threshold"] <= 1
            assert set(metrics["test_by_length"]) == {"2"}
            assert (out / "model.pt").is_file()
            assert (out / "vocab.json").is_file()
            with np.load(out / "test_predictions.npz") as predictions:
                assert len(predictions["prob"]) == 12
