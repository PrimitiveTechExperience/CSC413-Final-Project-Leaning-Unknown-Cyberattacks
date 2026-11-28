import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json  # noqa: E402

import analyze_results  # noqa: E402
import export_examples  # noqa: E402
import variance_check  # noqa: E402
import verify_results  # noqa: E402


def manifest(train, val, test):
    return {"data": {"recording_manifest": {"train": train, "val": val, "test": test}}}


def test_verify_accepts_clean_held_out_run():
    run = manifest(train=[["A", "training", "a1"], ["B", "test_attack", "b1"]],
                   val=[["B", "validation", "b2"]],
                   test=[["A", "test_attack", "a2"], ["A", "test_normal", "a3"]])
    assert verify_results.check_run(("lstm", "held_out", "A"), run) == []


def test_verify_flags_target_attack_and_shared_recordings():
    run = manifest(train=[["A", "test_attack", "a1"], ["A", "test_normal", "a3"]],
                   val=[["A", "validation", "a4"]],
                   test=[["A", "test_attack", "a2"], ["A", "test_normal", "a3"]])
    errors = verify_results.check_run(("lstm", "held_out", "A"), run)
    assert any("both train and test" in e for e in errors)
    assert any("target attack recording in train" in e for e in errors)
    assert any("target recording in held-out validation" in e for e in errors)


def test_verify_flags_other_scenarios_in_single_seen_run():
    run = manifest(train=[["A", "training", "a1"], ["B", "training", "b1"]],
                   val=[["A", "validation", "a4"]], test=[["A", "test_attack", "a2"]])
    assert any("other scenarios" in e for e in verify_results.check_run(("lstm", "seen", "A"), run))


def test_sign_test_drops_ties_and_is_two_sided():
    assert analyze_results.sign_test([0.1, 0.2, 0.0, -0.1]) == (2, 1, 1.0)
    wins, losses, p = analyze_results.sign_test([0.1] * 10)
    assert (wins, losses) == (10, 0) and abs(p - 2 / 2 ** 10) < 1e-12


def test_run_length_compresses_repeats():
    assert export_examples.run_length(["read", "read", "write", "read"]) == "read×2 write read"
    assert export_examples.run_length(["a", "b"] * 20, limit=3) == "a b a …"


def test_call_profile_distance_bounds():
    tv, over = export_examples.call_profile([["a", "a"], ["b", "b"]], [0, 1])
    assert tv == 1.0 and over[0]["call"] == "b"
    tv, _ = export_examples.call_profile([["a", "b"], ["b", "a"]], [0, 1])
    assert tv == 0.0


def test_variance_groups_identical_model_files(tmp_path):
    runs = {"A": b"weights-1", "B": b"weights-2", "C": b"weights-1"}
    for target, blob in runs.items():
        d = tmp_path / "lstm" / "seen_all" / target
        d.mkdir(parents=True)
        (d / "model.pt").write_bytes(blob)
        tau = 0.9 if blob == b"weights-1" else 0.8
        (d / "metrics.json").write_text(json.dumps(
            {"test": {"threshold": tau}, "history": [{"val_loss": 0.5}, {"val_loss": 0.4}]}))
    models = variance_check.distinct_models(tmp_path, "lstm")
    assert [m["name"] for m in models] == ["run A", "run B"]
    assert [[d.name for d in m["dirs"]] for m in models] == [["A", "C"], ["B"]]
    assert models[0]["tau"] == 0.9 and models[0]["epochs"] == 2 and models[0]["best_val_loss"] == 0.4
    assert abs(variance_check.spread([0.70, 0.72, 0.71]) - 0.02) < 1e-12


def test_stide_tables_pair_stide_with_held_out(tmp_path):
    for n, scenario, auroc in ((6, "A", 0.9), (6, "B", 0.5)):
        run = tmp_path / f"stide_n{n}" / "held_out" / scenario
        run.mkdir(parents=True)
        test = {"auroc": auroc, "auprc": 0.4, "f1": 0.1, "precision": 0.2, "recall": 0.1, "fpr": 0.01}
        (run / "metrics.json").write_text(json.dumps({"test": test}), encoding="utf-8")
    stide = analyze_results.load_stide(tmp_path)
    assert set(stide) == {(6, "A"), (6, "B")}

    def neural(auroc):
        return {"result": {"test": {"auroc": auroc, "auprc": 0.3, "f1": 0.0, "precision": 0.0,
                                    "recall": 0.0, "fpr": 0.0}}}

    runs = {("lstm", "held_out", "A"): neural(0.8), ("lstm", "held_out", "B"): neural(0.6),
            ("lstm", "seen_all", "A"): neural(0.95), ("lstm", "seen_all", "B"): neural(0.7)}
    t = analyze_results.baseline_tables(runs, stide, {}, ["A", "B"])
    lstm = next(r for r in t["stide_comparison"][1] if r[1] == "LSTM")
    assert lstm[2:5] == ["0.000", "1/1", "1.0000"]  # +0.1 on A, -0.1 on B
    assert lstm[8:10] == ["-0.125", "0/2"]
    assert t["held_out_auroc_by_scenario"][1][-1][1] == "0.700"
    assert "logreg_comparison" not in t
    assert analyze_results.load_stide(tmp_path / "missing") == {}


def test_logreg_loaded_and_compared_in_same_condition(tmp_path):
    for condition, scenario, auroc in (("held_out", "A", 0.7), ("seen_all", "A", 0.6)):
        run = tmp_path / "logreg" / condition / scenario
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps({"test": {"auroc": auroc, "auprc": 0.2}}), encoding="utf-8")
    logreg = analyze_results.load_logreg(tmp_path)
    assert set(logreg) == {("logreg", "held_out", "A"), ("logreg", "seen_all", "A")}

    runs = {("lstm", c, "A"): {"result": {"test": {"auroc": v, "auprc": 0.3}}}
            for c, v in (("held_out", 0.65), ("seen_all", 0.9))}
    t = analyze_results.baseline_tables(runs, {}, logreg, ["A"])
    rows = {(r[0], r[1]): r for r in t["logreg_comparison"][1]}
    assert rows[("Held-out", "LSTM")][2:4] == ["0.050", "1/0"]
    assert rows[("Seen (all)", "LSTM")][2:4] == ["-0.300", "0/1"]
    assert t["logreg_auroc_by_scenario"][1][0] == ["A", "–", "0.600", "0.700"]
    assert "stide_comparison" not in t
