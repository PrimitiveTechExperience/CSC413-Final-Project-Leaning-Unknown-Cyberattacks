"""Measure GPU capacity with full-length synthetic windows before real-data training."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lidds_gen.config import load_config  # noqa: E402
from lidds_gen.models import build_model  # noqa: E402
from lidds_gen.training import weighted_bce  # noqa: E402
from lidds_gen.training.loop import resolve_device  # noqa: E402


def measure(cfg: dict, name: str, batch_size: int, steps: int, device: torch.device) -> dict:
    torch.manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats(device)
    length = cfg["data"]["max_length"]
    model = build_model(name, 256, cfg["model"], length).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"],
                                 weight_decay=cfg["training"]["weight_decay"])
    tokens = torch.randint(2, 256, (batch_size, length), device=device)
    lengths = torch.full((batch_size,), length, dtype=torch.long, device=device)
    labels = (torch.arange(batch_size, device=device) % 2).float()

    def step():
        optimizer.zero_grad()
        loss = weighted_bce(model(tokens, lengths), labels, 1.0, 1.0)
        loss.backward()
        optimizer.step()
        return loss

    step()  # Warm up kernels and allocate optimizer state.
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        loss = step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"Nonfinite benchmark loss: {name}, batch {batch_size}")
    return {"model": name, "batch_size": batch_size, "window_length": length, "status": "ok",
            "seconds_per_step": seconds / steps, "windows_per_second": batch_size * steps / seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--models", nargs="+", choices=["lstm", "transformer"], default=["lstm", "transformer"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.steps < 1 or any(size < 1 for size in args.batch_sizes):
        ap.error("steps and batch sizes must be positive")
    cfg = load_config(args.config)
    device = resolve_device("cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": "synthetic_full_length_capacity_check", "torch_version": str(torch.__version__),
              "cuda_version": torch.version.cuda, "device": torch.cuda.get_device_name(device),
              "capability": list(torch.cuda.get_device_capability(device)),
              "cpu_threads": torch.get_num_threads(), "steps": args.steps, "results": []}
    for model in args.models:
        for size in args.batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                result = measure(cfg, model, size, args.steps, device)
            except torch.cuda.OutOfMemoryError:
                result = {"model": model, "batch_size": size, "status": "out_of_memory"}
            report["results"].append(result)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
