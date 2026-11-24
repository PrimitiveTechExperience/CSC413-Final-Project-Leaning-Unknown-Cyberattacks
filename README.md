# CSC413 Final Project: Learning Unknown Cyberattacks

**Beyond Known Attacks: Can Deep Learning Models Detect Previously Unknown Cyberattacks?**

We train an LSTM and a Transformer encoder to classify system-call windows from
[LID-DS-2021](https://github.com/LID-DS/LID-DS) as benign or malicious, and compare
two settings per target scenario:

| Setting    | Target scenario's attacks in training? | Threshold chosen on            |
|------------|----------------------------------------|--------------------------------|
| `seen`     | yes (single target scenario)            | target scenario's validation   |
| `seen_all` | yes (all-scenario control)              | validation of all scenarios    |
| `held_out` | no (its benign recordings stay)        | validation of the other 14       |

The held-out setting tests transfer to an unfamiliar attack in a familiar application.
The single-scenario seen baseline also changes training diversity; its gap alone
does not isolate memorization. The all-scenario seen control helps assess the
effect of including the target's attacks, with training size and validation
composition remaining limitations.
Windows are 50 / 100 / 200 calls (variable per window), so test metrics are also
reported per window length. See [Proposal/proposal.tex](Proposal/proposal.tex) for the full design.

## Layout

```
configs/default.yaml        all hyper-parameters and paths
src/lidds_gen/
  data/                     torch-free pipeline
    recording.py            parse .zip recordings (.sc + .json), entry events after warm-up
    windows.py              per-thread, non-overlapping, variable-length windows + labels
    store.py                per-scenario .npz cache of windows
    splits.py               70/15/15 re-split of official test recordings; seen / held-out
    vocab.py                vocabulary from training windows only, <pad>=0, <unk>=1
    experiment.py           assemble encoded train/val/test for one experiment
  models/                   LSTMClassifier, TransformerClassifier (both return logits)
  training/                 dataset/collate, class-weighted BCE, early stopping
  evaluation/metrics.py     threshold selection (val only), F1/AUROC/AUPRC/FPR, per-length
scripts/
  prepare_data.py           raw dataset -> data/processed/<scenario>.{npz,json}
  configure_storage.py      local full/pilot configs for another drive
  summarize_data.py         split/class/length counts and recording manifest
  extract_scenarios.py      unpack scenario archives; retain recording ZIPs
  summarize_results.py      metric tables and seen-minus-held-out gaps
  benchmark_gpu.py          full-length synthetic GPU capacity and timing check
  run_experiment.py         one (model, setting, scenario) run -> results/...
  run_all.py                full 2 x 2 x 15 grid, resumable
tests/                      unit tests on synthetic recordings
Proposal/                   LaTeX proposal (NeurIPS 2026 template)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"     # or: pip install -r requirements.txt
pytest
```

For GPU training, install the CUDA build of PyTorch from <https://pytorch.org> first.

## Data

Download LID-DS-2021 via the link in the [LID-DS README](https://github.com/LID-DS/LID-DS)
and extract it so the tree looks like:

```
data/raw/LID-DS-2021/<scenario>/training/*.zip
                                /validation/*.zip
                                /test/normal/*.zip
                                /test/normal_and_attack/*.zip
```

The root `data/`, `results/` and checkpoints are git-ignored; the source package
`src/lidds_gen/data/` is tracked.

## Running

```bash
python scripts/prepare_data.py                         # parse + window all 15 scenarios
python scripts/run_experiment.py --set experiment.model=transformer \
    experiment.setting=held_out experiment.scenario=ZipSlip
python scripts/run_all.py                              # full grid
```

Any config value can be overridden with `--set dotted.key=value`. For a quick
debug pass use `--set data.max_recordings_per_partition=5 training.epochs=1`
when preparing data and running experiments. Changing the seed, window settings,
reconnaissance keywords, or recording limit requires re-running `prepare_data.py`;
experiment assembly rejects caches prepared with different settings.

The default seen setting follows the proposal's single-scenario baseline.
Pass `--set experiment.seen_scope=all` for the all-scenario control, which writes
to `seen_all/` instead of `seen/`. Use `run_all.py --include-seen-control` to run
both seen scopes alongside the held-out setting (90 runs for the full grid).
Held-out training keeps the target's benign training recordings, excludes all
of its attack recordings, and uses only other scenarios for validation.

### Window caps

The full dataset has about 35.9M windows (4.3B calls, 15.7 GiB as cached arrays),
which neither fits in 16 GB of RAM nor trains in reasonable time. After the
recording-level split, `data.max_windows_per_split` (default 100k train, 50k
val, 50k test) keeps a uniform random sample of each scenario's windows in each
split. The seed depends only on the global seed, scenario, and split, so:

- validation and test keep their natural class ratio (in expectation);
- the target's test windows are identical in `seen`, `seen_all`, and `held_out`
  runs, so seen-minus-held-out gaps compare the same windows;
- no scenario dominates the pooled training set by size (CVE-2017-12635_6 alone
  has 11.3M windows).

Scenarios are loaded one at a time and only sampled windows are kept, so peak
memory is about 6.6 GiB for a held-out run. `dataset_summary.json` records the
caps and `windows_available` (per scenario, before capping) for each split. Set
a split's cap to `null` to keep all its windows. Configs must state the key
explicitly; a config copied before this option existed is rejected rather than
silently run uncapped. Regenerate local configs with `configure_storage.py`.

To debug held-out training with a smaller pool, set `data.scenarios` explicitly
(for example, `--set 'data.scenarios=[CVE-2014-0160,ZipSlip]'`) in both preparation
and training commands. This changes the experiment; full results require all 15
scenarios. `run_all.py --scenarios ...` selects evaluation targets only.

Preprocessing follows the upstream [syscall columns](https://github.com/LID-DS/LID-DS/blob/master/dataloader/syscall_2021.py)
and [recording metadata](https://github.com/LID-DS/LID-DS/blob/master/dataloader/recording_2021.py).
Missing exploit timing raises an error instead of silently assigning benign
labels. If every timed step matches a reconnaissance keyword, the recording
produces benign windows and preparation prints a warning. The keyword policy
was checked against the full dataset's step names: only CVE-2017-12635_6's
`port-scan` step matches, so its labels start at `privilege-escalation`;
CVE-2020-23839's `trick admin` step is part of the exploit. CVE-2020-13942 and
Juice-Shop have a single `attack` step in their metadata.
Per-length results group by the requested window length; retained final windows
can contain fewer calls.

Each run writes `results/<model>/<condition>/<scenario>/` containing `config.yaml`,
`vocab.json`, `model.pt`, `metrics.json` (overall and per-window-length test metrics,
training history, window counts, unknown-token rate), `dataset_summary.json`,
`dataset_summary.csv`, and `test_predictions.npz`. The summary includes recording
membership for each split, class counts and malicious proportions by target
window length, and counts of retained short final windows. Completed runs are
never overwritten; resuming with changed settings requires a new results root.

## Using another drive and running a pilot

Keep the downloaded scenario archives on the larger drive. Extract each scenario
archive once into `<storage>/raw/LID-DS-2021/`, preserving the scenario and
partition directories shown above. Keep the individual recording ZIPs compressed:
the parser reads their `.sc` and `.json` members directly.

Generate local configs without moving the source repository:

```powershell
python scripts/configure_storage.py --root D:/CSC413-LIDDS
python scripts/configure_storage.py --root D:/CSC413-LIDDS --pilot
python scripts/extract_scenarios.py --config configs/local-pilot.yaml --archives-root D:/CSC413-LIDDS/downloads
```

These git-ignored configs put raw data, processed caches, results, and model
checkpoints on D:. The pilot uses CVE-2014-0160 (single-step) and CVE-2020-23839
(multi-step), at most five recordings per official partition, and one epoch:

```powershell
python scripts/prepare_data.py --config configs/local-pilot.yaml
python scripts/summarize_data.py --config configs/local-pilot.yaml
python scripts/run_all.py --config configs/local-pilot.yaml --scenarios CVE-2014-0160 CVE-2020-23839 --include-seen-control
python scripts/summarize_results.py --results-root D:/CSC413-LIDDS/results/pilot
```

Preparation writes `<scenario>.audit.json` alongside each cache, with exploitation
step names, timestamps, reconnaissance exclusions, retained entry-event counts,
and window counts per recording. Inspect multi-step attack timing and warnings
before full training. The timing policy labels all post-exploitation calls in an
attack recording as malicious, including background activity; per-thread
windowing does not remove that source of label noise.

The pilot is a pipeline check with a reduced training pool, not a result for the
proposal's 15-scenario experiment. After validating the pilot, use the full config:

```powershell
python scripts/extract_scenarios.py --config configs/local.yaml --archives-root D:/CSC413-LIDDS/downloads
python scripts/prepare_data.py --config configs/local.yaml
python scripts/run_all.py --config configs/local.yaml --include-seen-control
python scripts/summarize_results.py --results-root D:/CSC413-LIDDS/results/full
```

The result summary writes `summary/metrics.csv` and `summary/comparison.csv`.
Gaps are seen minus held-out; a positive FPR gap means more false alarms in the
seen setting. Overall and per-length metrics are kept separate, and comparisons
reject runs with different preprocessing settings or test recording membership.

## GPU setup and benchmarking

Use a CUDA-enabled PyTorch build that supports the installed NVIDIA driver and
GPU architecture. The [official installation instructions](https://pytorch.org/get-started/locally/)
and [version-specific commands](https://pytorch.org/get-started/previous-versions/)
list the available builds. A separate environment on the data drive keeps the
larger GPU packages off the source drive; the local machine setup uses
`D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe`.

The tested Windows/GTX 1660 Ti setup uses **PyTorch 2.9.1 + CUDA 12.6**.
The 2.14.1 CUDA build completed model computations but crashed during process
shutdown on this machine; the 2.9.1 build passed the same capacity checks and
exited normally. Install the pinned CUDA build before installing this project:

```powershell
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe -m pip install -e ".[dev]"
```

Before choosing a batch size, check training memory with full-length synthetic
windows using the project architectures and Adam optimizer:

```powershell
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/benchmark_gpu.py --config configs/local.yaml --output D:/CSC413-LIDDS/results/gpu_capacity.json
```

This is a capacity and throughput check, not a detection experiment. It includes
forward/backward passes and optimizer state, reports out-of-memory cases, and
writes peak allocated/reserved CUDA memory and seconds per step. Confirm the
selected batch size with the real-data pilot before full training.

Generate GPU configs after selecting a batch size (replace `256` below with the
verified size). They share the existing raw/processed caches and write to
`results/pilot_gpu` and `results/full_gpu` on the storage drive:

```powershell
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/configure_storage.py --root D:/CSC413-LIDDS --gpu --batch-size 256
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/configure_storage.py --root D:/CSC413-LIDDS --gpu --pilot --batch-size 256
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/run_all.py --config configs/local-gpu-pilot.yaml --scenarios CVE-2014-0160 CVE-2020-23839 --include-seen-control
```

For GPU runs, set `training.device=cuda`; this fails clearly when CUDA is
unavailable. Use a separate `paths.results_root` when rerunning an existing CPU
experiment. Every new run records its device, PyTorch/CUDA versions, phase
timings, peak CUDA memory, and per-epoch training/validation duration in
`metrics.json`. That completion marker is written after the predictions finish
saving, so interrupted runs are not mistaken for completed experiments.

## Authors

Ryan H. Lin, Sebastian Sperneac, Arman Hovhannisyan (University of Toronto Mississauga)
