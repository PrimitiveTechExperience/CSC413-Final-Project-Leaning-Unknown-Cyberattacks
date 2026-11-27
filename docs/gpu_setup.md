# GPU setup, benchmarking and troubleshooting

## Installing PyTorch with CUDA

Use a CUDA-enabled PyTorch build that supports the installed NVIDIA driver and
GPU architecture. The [official installation instructions](https://pytorch.org/get-started/locally/)
and [version-specific commands](https://pytorch.org/get-started/previous-versions/)
list the available builds. A separate environment on the data drive keeps the
larger GPU packages off the source drive; our setup uses
`D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe`.

The tested Windows / GTX 1660 Ti (6 GB) setup uses **PyTorch 2.9.1 + CUDA 12.6**.
Install the pinned CUDA build before installing this project:

```powershell
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe -m pip install -e ".[dev]"
```

## Choosing a batch size

Before choosing a batch size, check training memory with full-length synthetic
windows using the project architectures and the Adam optimizer:

```powershell
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/benchmark_gpu.py --config configs/local.yaml --output D:/CSC413-LIDDS/results/gpu_capacity.json
```

This is a capacity and throughput check, not a detection experiment. It includes
forward/backward passes and optimizer state, reports out-of-memory cases, and
writes peak allocated/reserved CUDA memory and seconds per step. Confirm the
selected batch size with the real-data pilot before full training. We used 256.

Generate GPU configs after selecting a batch size. They share the existing
raw/processed caches and write to separate results folders:

```powershell
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/configure_storage.py --root D:/CSC413-LIDDS --gpu --batch-size 256
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/configure_storage.py --root D:/CSC413-LIDDS --gpu --pilot --batch-size 256
D:/CSC413-LIDDS/venv-gpu/Scripts/python.exe scripts/run_all.py --config configs/local-gpu-pilot.yaml --scenarios CVE-2014-0160 CVE-2020-23839 --include-seen-control
```

For GPU runs, set `training.device=cuda`; this fails clearly when CUDA is
unavailable instead of silently training on the CPU. Use a separate
`paths.results_root` when rerunning an existing CPU experiment. Every run
records its device, PyTorch/CUDA versions, phase timings, peak CUDA memory, and
per-epoch duration in `metrics.json`.

Typical timings on the GTX 1660 Ti with batch size 256: a single-scenario Seen
run takes under 35 minutes, and a pooled run (Seen (all) or Held-out, about
1.29M training windows) takes 7–8 minutes per epoch for the LSTM and about
11–12 minutes per epoch for the Transformer, so 1–5 hours per run. Peak
reserved CUDA memory stayed under 2.1 GB. The whole grid took about 83 GPU-hours
with Seen (all) model reuse.

## Troubleshooting

- **PyTorch 2.14.1 crashes on exit.** The 2.14.1 CUDA build completed model
  computations but crashed during process shutdown on our Windows machine; the
  2.9.1 build passed the same capacity checks and exited normally.
- **`PermissionError [WinError 5]` when replacing `progress.json` or the
  checkpoint.** On Windows a file cannot be replaced while another process
  (for example a dashboard polling progress) has it open. Progress writes are
  best-effort with retries, and checkpoint writes retry and keep the previous
  checkpoint if the file stays locked, so training continues.
- **Resumed runs are not bit-identical to uninterrupted ones.** Checkpoints save
  the model, optimizer, early-stopping state and history, and the data order is
  seeded per epoch, but the dropout random state is not checkpointed. A resumed
  run therefore follows a different, equally valid trajectory. In our grid,
  three LSTM Seen (all) runs resumed after an interruption before model reuse
  existed; together with the uninterrupted runs they produced three slightly
  different LSTM Seen (all) models (best validation loss 0.2999–0.3030).
  `scripts/verify_results.py` lists which targets used which model.
- **Out of memory on the host.** The full dataset is about 35.9M windows
  (15.7 GiB as cached arrays). The per-scenario window caps
  (`data.max_windows_per_split`) keep peak RAM at about 6.6 GiB for a pooled run.
  Avoid running `export_examples.py` on the largest scenarios while training.
