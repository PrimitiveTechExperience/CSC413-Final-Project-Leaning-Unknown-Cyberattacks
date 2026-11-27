import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_results  # noqa: E402
import export_examples  # noqa: E402
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
