# CSC413 Final Project: Learning Unknown Cyberattacks

**Beyond Known Attacks: Can Deep Learning Models Detect Previously Unknown Cyberattacks?**

We train an LSTM and a Transformer encoder to classify system-call windows from
[LID-DS-2021](https://github.com/LID-DS/LID-DS) as benign or malicious, and compare
two settings per target scenario:

| Setting    | Target scenario's attacks in training? | Threshold chosen on            |
|------------|----------------------------------------|--------------------------------|
| `seen`     | yes                                    | validation of training scenarios |
| `held_out` | no (its benign recordings stay)        | validation of the other 14       |

The seen vs. held-out gap measures how much detection relies on memorised attacks.
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

`data/`, `results/` and checkpoints are git-ignored.

## Running

```bash
python scripts/prepare_data.py                         # parse + window all 15 scenarios
python scripts/run_experiment.py --set experiment.model=transformer \
    experiment.setting=held_out experiment.scenario=ZipSlip
python scripts/run_all.py                              # full grid
```

Any config value can be overridden with `--set dotted.key=value`. For a quick
debug pass use `--set data.max_recordings_per_partition=5 training.epochs=1`.

Each run writes `results/<model>/<setting>/<scenario>/` containing `config.yaml`,
`vocab.json`, `model.pt`, `metrics.json` (overall and per-window-length test metrics,
training history, window counts, unknown-token rate) and `test_predictions.npz`.

## Authors

Ryan H. Lin, Sebastian Sperneac, Arman Hovhannisyan (University of Toronto Mississauga)
