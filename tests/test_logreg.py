"""Bag-of-calls logistic regression: same windows as the neural runs in every condition."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lidds_gen.data.experiment import build_experiment
from lidds_gen.data.report import write_dataset_summary
from test_pipeline import pipeline_config  # noqa: F401 - pytest fixture

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_logreg", ROOT / "scripts/run_logreg.py")
logreg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logreg)

CONDITION_CFG = {"seen": ("seen", "single"), "seen_all": ("seen", "all"), "held_out": ("held_out", "single")}


def neural_splits(cfg, condition, target):
    cfg = copy.deepcopy(cfg)
    setting, scope = CONDITION_CFG[condition]
    cfg["experiment"].update(setting=setting, seen_scope=scope, scenario=target)
    return build_experiment(cfg)


@pytest.mark.parametrize("caps", [None, {"train": 7, "val": 3, "test": 5}])
def test_every_condition_rebuilds_the_neural_windows(pipeline_config, caps):  # noqa: F811
    cfg = copy.deepcopy(pipeline_config)
    cfg["data"]["max_windows_per_split"] = caps
    pool = cfg["data"]["scenarios"]
    names = logreg.call_names(cfg, pool)
    blocks = {s: logreg.scenario_blocks(cfg, s, names) for s in pool}
    for condition in logreg.CONDITIONS:
        for target in pool:
            vocab, expected, _ = neural_splits(cfg, condition, target)
            ours = logreg.condition_splits(blocks, condition, target)
            for split in ("train", "val", "test"):
                block, ref = ours[split], expected[split]
                assert len(block) == len(ref), (condition, target, split)
                assert block.y.sum() == sum(ref.labels), (condition, target, split)
            # Test rows in the same order, with the same calls (unknown calls map to <unk> there).
            test, ref = ours["test"], expected["test"]
            assert test.y.tolist() == ref.labels
            assert test.target_lengths.tolist() == ref.target_lengths
            for row, seq in zip(test.X, ref.sequences):
                freq = {names[j]: v for j, v in zip(row.indices, row.data)}
                calls = [vocab.itos[i] for i in seq]
                for name in set(calls) - {"<unk>"}:
                    assert freq.pop(name) == pytest.approx(calls.count(name) / len(calls))
                assert sum(freq.values()) == pytest.approx(calls.count("<unk>") / len(calls))


def test_held_out_never_trains_on_target_attacks_or_validates_on_target(pipeline_config):  # noqa: F811
    cfg = pipeline_config
    names = logreg.call_names(cfg, ["A", "B"])
    blocks = {s: logreg.scenario_blocks(cfg, s, names) for s in ("A", "B")}
    held = logreg.condition_splits(blocks, "held_out", "A")
    attack_a = names.index("attack_A")
    assert held["train"].X[:, attack_a].nnz == 0
    assert held["val"].X[:, attack_a].nnz == 0
    assert len(held["val"]) == len(blocks["B"]["val"])


def test_cli_runs_all_conditions_and_rejects_different_windows(pipeline_config, tmp_path):  # noqa: F811
    cfg = pipeline_config
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    # Fake neural held-out run for A with the real summary and test labels.
    _, splits, summary = neural_splits(cfg, "held_out", "A")
    ref = Path(cfg["paths"]["results_root"]) / "lstm" / "held_out" / "A"
    write_dataset_summary(summary, ref)
    np.savez_compressed(ref / "test_predictions.npz", prob=np.zeros(len(splits["test"])),
                        label=np.array(splits["test"].labels), target_length=np.array(splits["test"].target_lengths))

    out = tmp_path / "logreg_out"
    logreg.main(["--config", str(config), "--out", str(out)])
    for condition in logreg.CONDITIONS:
        for target in ("A", "B"):
            metrics = json.loads((out / "logreg" / condition / target / "metrics.json").read_text())
            assert 0 <= metrics["test"]["auroc"] <= 1
            assert metrics["top_calls"]["malicious"]
    held = json.loads((out / "logreg" / "held_out" / "A" / "metrics.json").read_text())
    assert len(held["checked_against"]) == 2
    with open(out / "logreg_overall.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["Condition"], r["Runs"]) for r in rows] == [("Seen", "2"), ("Seen (all)", "2"), ("Held-out", "2")]
    # Seen (all) is one shared model: identical predictions on a shared target are expected,
    # and the fit is reused, so both runs record the same training time.
    a = json.loads((out / "logreg" / "seen_all" / "A" / "metrics.json").read_text())
    b = json.loads((out / "logreg" / "seen_all" / "B" / "metrics.json").read_text())
    assert a["runtime"] == b["runtime"]

    summary["splits"]["train"]["malicious_windows"] += 1
    write_dataset_summary(summary, ref)
    with pytest.raises(SystemExit, match="1 run"):
        logreg.main(["--config", str(config), "--out", str(tmp_path / "again")])
