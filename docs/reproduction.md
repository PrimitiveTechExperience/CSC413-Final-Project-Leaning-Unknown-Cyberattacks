# Reproducing the experiments

This page covers everything from downloading LID-DS-2021 to regenerating the
tables and figures in the README. GPU-specific setup is in
[gpu_setup.md](gpu_setup.md).

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,analysis]"     # or: pip install -r requirements.txt
pytest
```

`analysis` adds matplotlib, which is only needed for `scripts/analyze_results.py`.
For GPU training, install the CUDA build of PyTorch first (see
[gpu_setup.md](gpu_setup.md)).

## 2. Get the data

Download LID-DS-2021 via the link in the [LID-DS README](https://github.com/LID-DS/LID-DS)
and extract it so the tree looks like:

```
data/raw/LID-DS-2021/<scenario>/training/*.zip
                                /validation/*.zip
                                /test/normal/*.zip
                                /test/normal_and_attack/*.zip
```

Keep the individual recording ZIPs compressed: the parser reads their `.sc` and
`.json` members directly. The root `data/`, `results/` and checkpoints are
git-ignored.

### Keeping data on another drive

The full dataset, caches and results need tens of GB. To keep them on a larger
drive without moving the repository, generate git-ignored local configs that
point there, then extract each scenario archive once:

```powershell
python scripts/configure_storage.py --root D:/CSC413-LIDDS
python scripts/configure_storage.py --root D:/CSC413-LIDDS --pilot
python scripts/extract_scenarios.py --config configs/local.yaml --archives-root D:/CSC413-LIDDS/downloads
```

`extract_scenarios.py` preserves the scenario and partition directories shown
above, keeps recording ZIPs compressed, rejects path traversal, and can be
re-run safely.

## 3. Prepare the windows

```bash
python scripts/prepare_data.py --config configs/local.yaml   # parse + window all 15 scenarios
```

Preparation writes `<scenario>.{npz,json}` caches and `<scenario>.audit.json`,
which lists exploitation step names, timestamps, reconnaissance exclusions,
retained entry-event counts, and window counts per recording. Inspect multi-step
attack timing and warnings before training. The timing policy labels all
post-exploitation calls in an attack recording as malicious, including
background activity; per-thread windowing does not remove that source of label
noise.

Preprocessing follows the upstream [syscall columns](https://github.com/LID-DS/LID-DS/blob/master/dataloader/syscall_2021.py)
and [recording metadata](https://github.com/LID-DS/LID-DS/blob/master/dataloader/recording_2021.py).
Missing exploit timing raises an error instead of silently assigning benign
labels. If every timed step matches a reconnaissance keyword, the recording
produces benign windows and preparation prints a warning. The keyword policy
was checked against the full dataset's step names: only CVE-2017-12635_6's
`port-scan` step matches, so its labels start at `privilege-escalation`;
CVE-2020-23839's `trick admin` step is part of the exploit. CVE-2020-13942 and
Juice-Shop have a single `attack` step in their metadata.

Changing the seed, window settings, reconnaissance keywords, or recording limit
requires re-running `prepare_data.py`; experiment assembly rejects caches
prepared with different settings. `scripts/summarize_data.py` exports
split/class/length counts and the recording manifest for one experiment without
training.

## 4. Pilot run (pipeline check)

The pilot uses CVE-2014-0160 (single-step) and CVE-2020-23839 (multi-step), at
most five recordings per official partition, and one epoch:

```powershell
python scripts/extract_scenarios.py --config configs/local-pilot.yaml --archives-root D:/CSC413-LIDDS/downloads
python scripts/prepare_data.py --config configs/local-pilot.yaml
python scripts/summarize_data.py --config configs/local-pilot.yaml
python scripts/run_all.py --config configs/local-pilot.yaml --scenarios CVE-2014-0160 CVE-2020-23839 --include-seen-control
python scripts/summarize_results.py --results-root D:/CSC413-LIDDS/results/pilot
```

The pilot is a pipeline check with a reduced training pool, not a result for
the 15-scenario experiment. For a quick debug pass on the default config, use
`--set data.max_recordings_per_partition=5 training.epochs=1` for both
preparation and training. To debug held-out training with a smaller pool, set
`data.scenarios` explicitly (for example,
`--set 'data.scenarios=[CVE-2014-0160,ZipSlip]'`) in both preparation and
training commands; full results require all 15 scenarios. `run_all.py
--scenarios ...` selects evaluation targets only.

### Window caps

The full dataset has about 35.9M windows (4.3B calls, 15.7 GiB as cached
arrays), which neither fits in 16 GB of RAM nor trains in reasonable time. After
the recording-level split, `data.max_windows_per_split` (default 100k train, 50k
val, 50k test) keeps a uniform random sample of each scenario's windows in each
split. The seed depends only on the global seed, scenario, and split, so the
target's test windows are identical in every condition. Scenarios are loaded
one at a time and only sampled windows are kept, so peak memory is about
6.6 GiB for a held-out run.

`dataset_summary.json` records the caps and `windows_available` (per scenario,
before capping) for each split. Set a split's cap to `null` to keep all its
windows. Configs must state the key explicitly; a config copied before this
option existed is rejected rather than silently run uncapped. Regenerate local
configs with `configure_storage.py`.

## 5. Full grid

```powershell
python scripts/run_all.py --config configs/local-gpu.yaml --include-seen-control
```

This runs 2 models × 3 conditions × 15 target scenarios = 90 runs. Each run
executes in its own process and writes
`results/<model>/<condition>/<scenario>/` containing `config.yaml`, `vocab.json`,
`model.pt`, `metrics.json` (overall and per-window-length test metrics, training
history, window counts, runtime), `dataset_summary.{json,csv}`, and
`test_predictions.npz`. `metrics.json` is written last, so an interrupted run is
never mistaken for a finished one. The dataset summary includes recording
membership for each split, class counts and malicious proportions by target
window length, and counts of retained short final windows. Per-length results
group by the requested window length; retained final windows can contain fewer
calls.

The grid is resumable: finished runs are skipped, and an interrupted run resumes
from its last epoch checkpoint. A failed run is recorded in `failed_runs.json`
and the grid continues. Finished runs are never overwritten; `run_all.py`
refuses to resume when the config has changed (comparing the learning rate each
model actually trains with), so changed settings need a new
`paths.results_root`. A single run can be launched with
`scripts/run_experiment.py --set experiment.model=transformer experiment.setting=held_out experiment.scenario=ZipSlip`.

The Seen (all) control trains and validates on the same pooled data for every
target, so the first finished Seen (all) run's model is reused for the other
targets (about 2 minutes instead of about 100). Reuse happens only when the
configs differ in the target alone and the vocabularies are identical;
`metrics.json` then records `runtime.reused_model_from`.

Our full grid took about 83 GPU-hours on a GTX 1660 Ti (see
[gpu_setup.md](gpu_setup.md) for timings).

## 6. Verify, summarize, plot

```powershell
python scripts/verify_results.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped
python scripts/summarize_results.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped
python scripts/analyze_results.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped
python scripts/export_examples.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped --config configs/local-gpu.yaml
python scripts/variance_check.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped --config configs/local-gpu.yaml
```

- `verify_results.py` checks that every condition and model is evaluated on the
  same test recordings and the same test windows (labels and lengths in the same
  order), that no recording appears in two splits, that held-out runs never see
  the target's attack recordings and never validate on the target, and that
  single-scenario seen runs only use the target. It exits with code 1 on any
  problem.
- `summarize_results.py` writes `summary/metrics.csv` and `summary/comparison.csv`
  under the results root. Gaps are seen minus held-out; a positive FPR gap means
  more false alarms in the seen setting. Overall and per-length metrics are kept
  separate, and comparisons reject runs with different preprocessing settings or
  test recording membership.
- `analyze_results.py` writes the tables in [`results_summary/`](../results_summary/results.md)
  and the figures in [`figures/`](../figures/).
- `export_examples.py` rebuilds each target's test windows, checks them against
  the saved predictions, and writes decoded example windows and per-scenario
  call-frequency statistics to [`results_summary/examples.md`](../results_summary/examples.md).
  Loading the largest scenarios needs several GB of RAM.
- `variance_check.py` groups the saved Seen (all) `model.pt` files by content
  (all were trained on identical data), checks that each reproduces its own
  run's saved test predictions, evaluates every distinct model on every
  target, and writes the spread between them to
  [`results_summary/variance_lstm.md`](../results_summary/variance_lstm.md).
  About 10 minutes on a GPU.

## 7. Baselines (optional, CPU only)

Both baselines need the prepared caches and the same config the grid was run
with, and both cross-check their windows against the neural runs under
`paths.results_root` (a mismatch fails that run). Their output goes next to,
not inside, `paths.results_root`, so the summary scripts above are unaffected.
Finished runs are skipped on a rerun (`--overwrite` recomputes).

### Bag-of-calls logistic regression

```powershell
python scripts/run_logreg.py --config configs/local-gpu.yaml
```

Each window becomes the frequency of each system call in it (order discarded),
standardized, and a class-balanced L2 logistic regression is trained in the same
Seen, Seen (all) and Held-out conditions as the neural models, on the same
windows. τ is chosen on validation by `evaluation.threshold_metric`, as for the
neural models, so all metrics compare directly. Each scenario's cache is loaded
once; Seen (all) is fit once and reused for every target. The run checks train,
validation and test window and malicious counts against each neural run's
`dataset_summary.json` and test labels against its `test_predictions.npz`.
Writes `<results_root>_logreg/logreg/<condition>/<scenario>/{metrics.json,test_predictions.npz}`
(`metrics.json` includes the 10 calls weighted most towards malicious and
benign), `logreg_overall.csv` (same columns as `results_summary/overall.csv`)
and `logreg_per_scenario.csv`. Peak memory is similar to one held-out run.

### STIDE

STIDE (Hofmeyr et al., 1998) stores every
length-n system-call sequence seen in a target's benign training recordings and
scores each test window by the fraction of its n-grams that were never seen. It
never sees an attack, so it is a held-out result by construction:

```powershell
python scripts/run_stide.py --config configs/local-gpu.yaml
```

- Uses n = 3, 6 and 10 by default (`--ngram 6` for the classic setting only).
- Learns from the target's benign training recordings only (one database per
  application, as in the original method); the score is the fraction of a
  window's n-grams never seen there.
- The threshold τ keeps the false-positive rate on benign validation windows at
  or below 1% (`--max-fpr`). AUROC and AUPRC do not depend on it.
- Writes `<results_root>_stide/stide_n<n>/held_out/<scenario>/{metrics.json,test_predictions.npz}`
  and `<results_root>_stide/stide_summary.csv` (per scenario plus a mean row per n).
  Compare its AUROC/AUPRC with the Held-out rows; its F1 uses a different
  threshold rule from the neural models, so treat it as secondary.

`analyze_results.py` picks up both baselines' output automatically (or pass
`--stide-root` / `--logreg-root`) and adds their tables to `results_summary/`
and `figures/stide_vs_held_out.png` and `figures/logreg_vs_neural.png`.
