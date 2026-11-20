import random

from lidds_gen.data import (
    RecordingInfo,
    Vocabulary,
    load_recording,
    resplit_recordings,
    select_setting,
    windows_for_recording,
)
from lidds_gen.data.recording import Recording, Syscall
from lidds_gen.data.store import ScenarioWindowWriter, load_scenario_windows
from lidds_gen.data.vocab import UNK_ID
from lidds_gen.data.windows import make_windows


def test_parse_entry_only_after_warmup(attack_zip):
    rec = load_recording(attack_zip, "S", "test_attack", recon_keywords=["scan"])
    assert [c.name for c in rec.syscalls] == ["open", "mmap", "socket", "execve", "write"]
    assert rec.exploit
    # Recon step is skipped: attack starts at the second exploit step.
    assert rec.attack_start_ns == (1_631_000_000 + 3) * 10**9


def test_without_recon_keywords_first_step_is_attack(attack_zip):
    rec = load_recording(attack_zip, "S", "test_attack")
    assert rec.attack_start_ns == (1_631_000_000 + 2) * 10**9


def test_windows_are_per_thread_and_labelled(attack_zip):
    rec = load_recording(attack_zip, "S", "test_attack", recon_keywords=["scan"])
    wins = windows_for_recording(rec, seed=0, mode="fixed", fixed_length=2)
    by_tid = {(w.thread_id, tuple(w.calls)): w.label for w in wins}
    assert by_tid == {
        (1, ("open", "socket")): 0,
        (1, ("write",)): 1,
        (2, ("mmap", "execve")): 1,
    }


def _rec(n: int, attack_at: int | None = None) -> Recording:
    calls = [Syscall(i, 7, f"c{i % 3}") for i in range(n)]
    return Recording("r", "S", "training", attack_at is not None, None, attack_at, calls)


def test_variable_windows_cover_stream_and_are_deterministic():
    rec = _rec(1000)
    a = windows_for_recording(rec, seed=1, lengths=(50, 100, 200))
    b = windows_for_recording(rec, seed=1, lengths=(50, 100, 200))
    assert [len(w.calls) for w in a] == [len(w.calls) for w in b]
    assert sum(len(w.calls) for w in a) == 1000
    assert {w.target_length for w in a} <= {50, 100, 200}
    assert all(len(w.calls) == w.target_length for w in a[:-1])


def test_drop_short_final_window():
    wins = make_windows(_rec(5).syscalls, _rec(5), random.Random(0), [2], keep_short_final=False)
    assert [len(w.calls) for w in wins] == [2, 2]


def test_vocab_unknown_and_roundtrip(tmp_path):
    v = Vocabulary.build([["read", "open"], ["read"]])
    assert v.encode(["open", "ptrace"])[1] == UNK_ID
    v.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").itos == v.itos


def _infos():
    out = []
    for s in ("A", "B"):
        out += [RecordingInfo(s, f"{s}_tr{i}", "training", False) for i in range(4)]
        out += [RecordingInfo(s, f"{s}_va{i}", "validation", False) for i in range(2)]
        out += [RecordingInfo(s, f"{s}_tn{i}", "test_normal", False) for i in range(20)]
        out += [RecordingInfo(s, f"{s}_ta{i}", "test_attack", True) for i in range(20)]
    return out


def test_resplit_is_stratified_and_deterministic():
    a = resplit_recordings(_infos(), seed=3)
    b = resplit_recordings(_infos(), seed=3)
    assert a == b
    attack_A = [r for r in a if r.scenario == "A" and r.exploit]
    assert [sum(r.split == s for r in attack_A) for s in ("train", "val", "test")] == [14, 3, 3]
    assert all(r.split == "train" for r in a if r.partition == "training")


def test_held_out_excludes_target_attacks_only():
    recs = resplit_recordings(_infos(), seed=0)
    sel = select_setting(recs, "held_out", "A")
    assert not any(r.scenario == "A" and r.exploit for r in sel["train"] + sel["val"])
    assert any(r.scenario == "A" and not r.exploit for r in sel["train"])
    assert any(r.scenario == "B" and r.exploit for r in sel["train"])
    assert all(r.scenario == "B" for r in sel["val"])
    assert all(r.scenario == "A" and r.split == "test" for r in sel["test"])

    seen = select_setting(recs, "seen", "A")
    assert sel["test"] == seen["test"]
    assert any(r.scenario == "A" and r.exploit for r in seen["train"])
    single = select_setting(recs, "seen", "A", seen_scope="single")
    assert all(r.scenario == "A" for r in single["train"] + single["val"])


def test_store_roundtrip(tmp_path, attack_zip):
    rec = load_recording(attack_zip, "S", "test_attack", recon_keywords=["scan"])
    wins = windows_for_recording(rec, seed=0, mode="fixed", fixed_length=2)
    w = ScenarioWindowWriter("S")
    w.add(rec, wins)
    w.save(tmp_path)
    sw = load_scenario_windows(tmp_path, "S")
    assert len(sw) == len(wins)
    assert [sw.window_calls(i) for i in range(len(sw))] == [x.calls for x in wins]
    assert sw.labels.tolist() == [x.label for x in wins]
    assert sw.recordings[0].exploit
