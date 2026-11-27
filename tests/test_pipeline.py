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
    test_counts = summary["splits"]["test"]
    assert test_counts["by_length"]["2"]["short_final_windows"] == 6
    assert test_counts["by_length"]["2"]["malicious_windows"] == 6
    assert test_counts["malicious_window_rate"] == 0.5
    manifest = summary["recording_manifest"]
    membership = [set(map(tuple, manifest[s])) for s in ("train", "val", "test")]
    assert not (membership[0] & membership[1] or membership[0] & membership[2] or membership[1] & membership[2])

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
        "--scenarios", "A", "--include-seen-control",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for model in ("lstm", "transformer"):
        for setting in ("seen", "seen_all", "held_out"):
            out = Path(cfg["paths"]["results_root"]) / model / setting / "A"
            metrics = json.loads((out / "metrics.json").read_text())
            assert len(metrics["history"]) == 1
            assert np.isfinite(metrics["history"][0]["train_loss"])
            assert metrics["history"][0]["epoch_seconds"] > 0
            assert metrics["runtime"]["device"] == "cpu"
            assert metrics["runtime"]["training_seconds"] > 0
            assert metrics["test"]["n"] == 12
            assert 0 <= metrics["test"]["threshold"] <= 1
            assert set(metrics["test_by_length"]) == {"2"}
            assert (out / "model.pt").is_file()
            assert (out / "vocab.json").is_file()
            assert (out / "dataset_summary.csv").is_file()
            with np.load(out / "test_predictions.npz") as predictions:
                assert len(predictions["prob"]) == 12
    # Resuming must skip all three conditions, but reject changed hyperparameters.
    resumed = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_all.py"), "--config", str(path),
        "--scenarios", "A", "--include-seen-control",
    ], capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert resumed.stdout.count("[skip]") == 6
    changed = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_all.py"), "--config", str(path),
        "--scenarios", "A", "--set", "training.lr=0.02",
    ], capture_output=True, text=True)
    assert changed.returncode != 0
    assert "different config" in changed.stderr
    aggregated = subprocess.run([
        sys.executable, str(ROOT / "scripts/summarize_results.py"),
        "--results-root", cfg["paths"]["results_root"],
    ], capture_output=True, text=True)
    assert aggregated.returncode == 0, aggregated.stdout + aggregated.stderr
    import csv
    summary_path = Path(cfg["paths"]["results_root"]) / "summary/comparison.csv"
    with summary_path.open(newline="") as stream:
        comparisons = list(csv.DictReader(stream))
    assert len(comparisons) == 8  # 2 models x 2 seen controls x (overall + length 2).
    assert {row["seen_condition"] for row in comparisons} == {"seen", "seen_all"}
    # Identical window counts alone must not allow different recording sets to be paired.
    metrics_path = Path(cfg["paths"]["results_root"]) / "lstm/seen/A/metrics.json"
    bad_metrics = json.loads(metrics_path.read_text())
    bad_metrics["data"]["recording_manifest"]["test"][0][2] = "different-recording"
    metrics_path.write_text(json.dumps(bad_metrics))
    rejected = subprocess.run([
        sys.executable, str(ROOT / "scripts/summarize_results.py"),
        "--results-root", cfg["paths"]["results_root"],
    ], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "different test data" in rejected.stderr


def test_dataset_summary_cli_without_training(pipeline_config, tmp_path):
    import csv
    import yaml

    path = tmp_path / "summary_config.yaml"
    path.write_text(yaml.safe_dump(pipeline_config))
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/summarize_data.py"), "--config", str(path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = Path(pipeline_config["paths"]["results_root"]) / "data_summary" / "held_out" / "A"
    summary = json.loads((out / "dataset_summary.json").read_text())
    assert summary["vocab_size"] > 2
    assert summary["splits"]["train"]["unknown_token_rate"] == 0
    with (out / "dataset_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["target_length"] for row in rows} == {"all", "2"}


def test_window_caps_are_per_scenario_deterministic_and_keep_target_test_fixed(pipeline_config):
    uncapped = copy.deepcopy(pipeline_config)
    uncapped["data"]["max_windows_per_split"] = None
    _, full, full_summary = build_experiment(uncapped)

    capped = copy.deepcopy(pipeline_config)
    capped["data"]["max_windows_per_split"] = {"train": 7, "val": 3, "test": 5}
    _, a, summary = build_experiment(capped)
    _, b, _ = build_experiment(copy.deepcopy(capped))
    for split, cap in (("train", 7), ("val", 3), ("test", 5)):
        per_scenario = {s: a[split].scenarios.count(s) for s in set(a[split].scenarios)}
        available = summary["splits"][split]["windows_available"]
        assert all(per_scenario[s] == min(cap, available[s]) for s in per_scenario)
        assert sum(available.values()) == len(full[split])
        assert [x.tolist() for x in a[split].sequences] == [x.tolist() for x in b[split].sequences]

    seen = copy.deepcopy(capped)
    seen["experiment"].update(setting="seen", seen_scope="single")
    _, seen_splits, _ = build_experiment(seen)
    # Same target test windows in both settings, so gaps compare like with like.
    assert a["test"].labels == seen_splits["test"].labels
    assert a["test"].target_lengths == seen_splits["test"].target_lengths
    assert len(a["test"]) == 5


def test_invalid_window_cap_is_rejected(pipeline_config):
    pipeline_config["data"]["max_windows_per_split"] = {"training": 10}
    with pytest.raises(ValueError, match="max_windows_per_split"):
        build_experiment(pipeline_config)


def test_interrupted_training_resumes_from_epoch_checkpoint(pipeline_config, tmp_path):
    import torch
    from lidds_gen.models import build_model
    from lidds_gen.training import train_model

    cfg = copy.deepcopy(pipeline_config)
    cfg["training"].update(epochs=3, early_stopping_patience=10)
    _, splits, _ = build_experiment(cfg)
    vocab, _, _ = build_experiment(cfg)

    def fresh():
        torch.manual_seed(0)
        return build_model("lstm", len(vocab), cfg["model"], cfg["data"]["max_length"])

    _, full = train_model(fresh(), splits["train"], splits["val"], cfg)

    class Interrupt(Exception):
        pass

    def stop_after_first_epoch(status):
        if status.get("phase") == "epoch_done" and status["epoch"] == 0:
            raise Interrupt

    ckpt = tmp_path / "checkpoint.pt"
    with pytest.raises(Interrupt):
        train_model(fresh(), splits["train"], splits["val"], cfg, progress=stop_after_first_epoch,
                    checkpoint=ckpt, checkpoint_key="k")
    assert ckpt.exists()
    _, resumed = train_model(fresh(), splits["train"], splits["val"], cfg, checkpoint=ckpt, checkpoint_key="k")
    assert len(resumed) == 3
    for a, b in zip(full, resumed):  # same trajectory as an uninterrupted run
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-5)
        assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-5)
    # A checkpoint from a different configuration is ignored, not resumed.
    _, other = train_model(fresh(), splits["train"], splits["val"], cfg, checkpoint=ckpt, checkpoint_key="other")
    assert other[0]["train_loss"] == pytest.approx(full[0]["train_loss"], rel=1e-5)


def test_run_all_records_failed_run_and_continues(pipeline_config, tmp_path):
    import yaml
    cfg = pipeline_config
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_all.py"), "--config", str(path),
        "--models", "lstm", "--settings", "seen", "--scenarios", "MISSING", "A",
    ], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    failed = json.loads((Path(cfg["paths"]["results_root"]) / "failed_runs.json").read_text())
    assert [f["scenario"] for f in failed] == ["MISSING"]
    # The grid continued past the failure.
    assert (Path(cfg["paths"]["results_root"]) / "lstm" / "seen" / "A" / "metrics.json").exists()
    assert not (Path(cfg["paths"]["results_root"]) / "_pending_config.yaml").exists()


def test_progress_write_survives_locked_target(tmp_path, monkeypatch):
    # On Windows the dashboard reading progress.json blocks the replace; that must not crash training.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_experiment

    target = tmp_path / "progress.json"

    def locked(self, other):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(run_experiment.time, "sleep", lambda s: None)
    assert run_experiment.write_progress_file(target, {"epoch": 1}) is False
    monkeypatch.undo()
    assert run_experiment.write_progress_file(target, {"epoch": 2}) is True
    assert json.loads(target.read_text(encoding="utf-8"))["epoch"] == 2


def test_all_scenario_seen_control_reuses_trained_model_with_same_results(pipeline_config, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_experiment

    def control(scenario, results):
        cfg = copy.deepcopy(pipeline_config)
        cfg["paths"]["results_root"] = str(tmp_path / results)
        cfg["experiment"].update(model="lstm", setting="seen", seen_scope="all", scenario=scenario)
        return run_experiment.run(cfg)

    first = control("A", "shared")
    assert first["runtime"]["reused_model_from"] is None
    reused = control("B", "shared")
    assert reused["runtime"]["reused_model_from"].endswith("A")
    assert not (tmp_path / "shared/lstm/seen_all/B/model.pt").exists()
    trained = control("B", "separate")  # B trained from scratch instead
    assert trained["runtime"]["reused_model_from"] is None
    def losses(history):  # epoch_seconds is wall-clock and naturally differs
        return [(h["train_loss"], h["val_loss"]) for h in history]

    assert losses(reused["history"]) == losses(trained["history"])
    for key in ("auroc", "f1", "fpr", "threshold"):
        if key in trained["test"]:
            assert reused["test"][key] == pytest.approx(trained["test"][key], abs=1e-6)
    # Single-scenario seen and held-out runs never reuse.
    cfg = copy.deepcopy(pipeline_config)
    cfg["paths"]["results_root"] = str(tmp_path / "shared")
    cfg["experiment"].update(model="lstm", setting="held_out", scenario="B")
    assert run_experiment.run(cfg)["runtime"]["reused_model_from"] is None


def test_per_model_learning_rate_overrides_shared_rate(pipeline_config):
    from lidds_gen.training.loop import learning_rate

    cfg = copy.deepcopy(pipeline_config)
    cfg["training"]["lr"] = 1e-3
    cfg["training"].pop("lr_by_model", None)
    cfg["experiment"]["model"] = "transformer"
    assert learning_rate(cfg) == 1e-3  # no override configured
    cfg["training"]["lr_by_model"] = {"transformer": 1e-4}
    assert learning_rate(cfg) == 1e-4
    cfg["experiment"]["model"] = "lstm"
    assert learning_rate(cfg) == 1e-3


def test_run_all_compares_effective_learning_rate(pipeline_config):
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_all import effective_config

    saved = copy.deepcopy(pipeline_config)
    saved["experiment"]["model"] = "lstm"
    saved["training"]["lr"] = 1e-3
    current = copy.deepcopy(saved)
    current["training"]["lr_by_model"] = {"transformer": 3e-4}
    # Adding an override for the other model keeps finished LSTM runs valid ...
    assert effective_config(saved) == effective_config(current)
    # ... but changing the rate this model trains with is still rejected.
    current["training"]["lr_by_model"]["lstm"] = 5e-4
    assert effective_config(saved) != effective_config(current)
