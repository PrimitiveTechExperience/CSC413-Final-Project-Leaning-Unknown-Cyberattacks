"""STIDE baseline: scoring, threshold rule, and identical test windows to the neural runs."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lidds_gen.data.experiment import build_experiment
from test_pipeline import pipeline_config  # noqa: F401 - pytest fixture

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_stide", ROOT / "scripts/run_stide.py")
stide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stide)


def seqs(*windows):
    return [np.array(w, dtype=np.int64) for w in windows]


def test_scores_are_fraction_of_unseen_ngrams():
    train = seqs([1, 2, 3, 4], [1, 2])
    test = seqs([1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 9], [1, 2], [2, 1], [])
    (score,), db = stide.stide_scores(train, [test], n=3, vocab_size=10)
    # Grams: (1,2,3) (2,3,4) seen; (4,3,2) (3,2,1) unseen; (2,3,9) unseen.
    # Short windows are padded and only match identical short training windows.
    np.testing.assert_allclose(score, [0, 1, 0.5, 0, 1, 1])
    assert db == 3


def test_codes_are_exact_for_long_ngrams():
    # Base^n would overflow int64 for n=12 with a large vocabulary; codes must not collide.
    rng = np.random.default_rng(0)
    windows = [rng.integers(0, 500, size=40) for _ in range(50)]
    codes, counts = stide.gram_codes(windows, 12, 500)
    grams = [tuple(w[i:i + 12]) for w in windows for i in range(len(w) - 11)]
    assert len(codes) == len(grams) == counts.sum()
    assert len(set(codes.tolist())) == len(set(grams))
    assert len(set(zip(grams, codes.tolist()))) == len(set(grams))


def test_threshold_respects_benign_fpr_budget():
    val = np.array([0, 0, 0, 0.5])
    assert stide.fpr_threshold(val, 0.25) == 0.5
    tau = stide.fpr_threshold(val, 0.0)
    assert tau > 0.5 and np.mean(val >= tau) == 0
    assert stide.fpr_threshold(np.zeros(5), 0.01) > 0


def test_test_windows_match_neural_held_out_and_train_is_benign(pipeline_config):  # noqa: F811
    cfg = copy.deepcopy(pipeline_config)
    cfg["data"]["max_windows_per_split"] = {"train": 7, "val": 3, "test": 5}
    _, splits, _ = build_experiment(cfg)
    windows, vocab_size = stide.stide_windows(cfg, "A")
    assert windows["test"]["labels"].tolist() == splits["test"].labels
    assert windows["test"]["target_lengths"].tolist() == splits["test"].target_lengths
    assert not windows["train"]["labels"].any() and not windows["val"]["labels"].any()
    assert vocab_size > 0


def test_cli_writes_runs_summary_and_rejects_mismatched_test_windows(pipeline_config, tmp_path):  # noqa: F811
    cfg = pipeline_config
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    _, splits, _ = build_experiment(cfg)
    ref = Path(cfg["paths"]["results_root"]) / "lstm" / "held_out" / "A"
    ref.mkdir(parents=True)
    np.savez_compressed(ref / "test_predictions.npz", prob=np.zeros(len(splits["test"])),
                        label=np.array(splits["test"].labels), target_length=np.array(splits["test"].target_lengths))

    out = tmp_path / "stide"
    stide.main(["--config", str(config), "--scenarios", "A", "--ngram", "1", "2", "--out", str(out)])
    metrics = json.loads((out / "stide_n2" / "held_out" / "A" / "metrics.json").read_text())
    assert metrics["test"]["n"] == len(splits["test"])
    assert metrics["checked_against"] == [str(ref / "test_predictions.npz")]
    assert 0 <= metrics["test"]["auroc"] <= 1
    with open(out / "stide_summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["Scenario"] for r in rows] == ["A", "Mean (1 scenarios)"] * 2

    # Different test labels mean a different config: refuse instead of comparing.
    np.savez_compressed(ref / "test_predictions.npz", prob=np.zeros(len(splits["test"])),
                        label=1 - np.array(splits["test"].labels),
                        target_length=np.array(splits["test"].target_lengths))
    with pytest.raises(SystemExit, match="1 scenario"):
        stide.main(["--config", str(config), "--scenarios", "A", "--ngram", "2",
                    "--out", str(tmp_path / "stide2")])
