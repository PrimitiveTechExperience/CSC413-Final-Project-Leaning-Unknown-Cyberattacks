"""Tables and figures for the report from a completed (or partial) results grid.

Writes Markdown/CSV tables to --output and PNG figures to --figures:

    python scripts/analyze_results.py --results-root D:/CSC413-LIDDS/results/full_gpu_capped

Tables: overall means, per-scenario metrics, generalization gaps, LSTM vs
Transformer, per-window-length AUROC, error counts, training summary, dataset
statistics, and parameter counts. Figures: loss curves, per-scenario AUROC,
AUROC gaps, per-length AUROC, and test score distributions. If
scripts/run_stide.py or scripts/run_logreg.py output exists (default
<results-root>_stide / _logreg), baseline tables and figures are added. Missing runs are left blank, so the
script can be run before the grid has finished.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidds_gen.config import experiment_condition, load_config  # noqa: E402
from lidds_gen.data import SCENARIOS  # noqa: E402

MODELS = ("lstm", "transformer")
CONDITIONS = ("seen", "seen_all", "held_out")
LENGTHS = ("50", "100", "200")
LABEL = {"lstm": "LSTM", "transformer": "Transformer", "logreg": "LogReg", "seen": "Seen", "seen_all": "Seen (all)",
         "held_out": "Held-out"}
SHORT = {"Bruteforce_CWE-307": "Bruteforce", "CWE-89-SQL-injection": "SQL injection",
         "PHP_CWE-434": "PHP upload", "EPS_CWE-434": "EPS upload"}

# Chart styling: light surface, recessive axes, fixed categorical order.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE, PINK = "#3c4f65", "#b8667c"  # charcoal blue, rosy charcoal pink (CVD-checked pair)
SERIES = (BLUE, PINK)
MODEL_COLOR = {"lstm": BLUE, "transformer": PINK}
# Two hues only: both seen conditions are blue (Seen (all) hollow/dashed), Held-out is pink.
COND_STYLE = {
    "seen": {"color": BLUE, "face": BLUE, "marker": "o", "ls": "-"},
    "seen_all": {"color": BLUE, "face": SURFACE, "marker": "o", "ls": "--"},
    "held_out": {"color": PINK, "face": PINK, "marker": "D", "ls": "-"},
}


def load_runs(root: Path) -> dict:
    runs = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = load_config(path.parent / "config.yaml")
        exp = result["experiment"]
        runs[(exp["model"], experiment_condition(cfg), exp["scenario"])] = {
            "dir": path.parent, "result": result, "cfg": cfg}
    return runs


def fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def markdown(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def write_csv(path: Path, headers, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def metric(runs, model, condition, scenario, name, length=None):
    run = runs.get((model, condition, scenario))
    if run is None:
        return None
    block = run["result"]["test"] if length is None else run["result"]["test_by_length"].get(length)
    return None if block is None else block.get(name)


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else None


def sign_test(diffs) -> tuple[int, int, float]:
    """Two-sided exact sign test; ties are dropped."""
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    n = wins + losses
    if n == 0:
        return wins, losses, float("nan")
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n
    return wins, losses, min(1.0, 2 * tail)


def best_epoch(history) -> int:
    losses = [h["val_loss"] for h in history]
    return int(np.argmin(losses)) + 1


# ---------------------------------------------------------------- tables
def tables(runs, scenarios) -> dict[str, tuple[list, list]]:
    t = {}
    names = ("auroc", "auprc", "f1", "precision", "recall", "fpr")

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            count = sum((model, condition, s) in runs for s in scenarios)
            rows.append([LABEL[model], LABEL[condition], count]
                        + [fmt(mean([metric(runs, model, condition, s, n) for s in scenarios])) for n in names])
    t["overall"] = (["Model", "Condition", "Runs", "AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR"], rows)

    for name in ("auroc", "auprc", "f1", "fpr"):
        headers = ["Scenario"] + [f"{LABEL[m]} {LABEL[c]}" for m in MODELS for c in CONDITIONS]
        rows = [[s] + [fmt(metric(runs, m, c, s, name)) for m in MODELS for c in CONDITIONS] for s in scenarios]
        rows.append(["**Mean**"] + [fmt(mean([metric(runs, m, c, s, name) for s in scenarios]))
                                    for m in MODELS for c in CONDITIONS])
        t[f"per_scenario_{name}"] = (headers, rows)

    rows = []
    for model in MODELS:
        for name in ("auroc", "auprc", "f1"):
            row = [LABEL[model], name.upper()]
            for a, b in (("seen", "held_out"), ("seen_all", "held_out"), ("seen", "seen_all")):
                diffs = [metric(runs, model, a, s, name) - metric(runs, model, b, s, name) for s in scenarios
                         if metric(runs, model, a, s, name) is not None and metric(runs, model, b, s, name) is not None]
                wins, losses, p = sign_test(diffs)
                row += [fmt(mean(diffs)), f"{wins}/{losses}", fmt(p, 4)]
            rows.append(row)
    t["gaps_summary"] = (["Model", "Metric", "Seen − Held-out", "+/−", "p", "Seen (all) − Held-out", "+/−", "p",
                          "Seen − Seen (all)", "+/−", "p"], rows)

    headers = ["Scenario"] + [f"{LABEL[m]} {g}" for m in MODELS for g in ("Seen − HO", "Seen (all) − HO")]
    rows = []
    for s in scenarios:
        row = [s]
        for m in MODELS:
            ho = metric(runs, m, "held_out", s, "auroc")
            for c in ("seen", "seen_all"):
                v = metric(runs, m, c, s, "auroc")
                row.append(fmt(None if v is None or ho is None else v - ho))
        rows.append(row)
    t["gaps_auroc_by_scenario"] = (headers, rows)

    rows = []
    for condition in CONDITIONS:
        for name in ("auroc", "auprc", "f1"):
            diffs = [metric(runs, "transformer", condition, s, name) - metric(runs, "lstm", condition, s, name)
                     for s in scenarios if metric(runs, "transformer", condition, s, name) is not None
                     and metric(runs, "lstm", condition, s, name) is not None]
            wins, losses, p = sign_test(diffs)
            rows.append([LABEL[condition], name.upper(), len(diffs), fmt(mean(diffs)), wins, losses, fmt(p, 4)])
    t["model_comparison"] = (["Condition", "Metric", "Scenarios", "Transformer − LSTM", "Transformer better",
                              "LSTM better", "Sign-test p"], rows)

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            rows.append([LABEL[model], LABEL[condition]]
                        + [fmt(mean([metric(runs, model, condition, s, "auroc", L) for s in scenarios]))
                           for L in LENGTHS]
                        + [fmt(mean([metric(runs, model, condition, s, "f1", L) for s in scenarios]))
                           for L in LENGTHS])
    t["by_length"] = (["Model", "Condition"] + [f"AUROC {L}" for L in LENGTHS] + [f"F1 {L}" for L in LENGTHS], rows)

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            c = defaultdict(int)
            for s in scenarios:
                for k in ("tp", "fp", "fn", "tn"):
                    c[k] += metric(runs, model, condition, s, k) or 0
            fnr = c["fn"] / max(c["fn"] + c["tp"], 1)
            fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
            zero = sum((metric(runs, model, condition, s, "tp") == 0) for s in scenarios
                       if (model, condition, s) in runs)
            rows.append([LABEL[model], LABEL[condition], fmt(c["tp"]), fmt(c["fp"]), fmt(c["fn"]), fmt(c["tn"]),
                         fmt(fpr), fmt(fnr), zero])
    t["errors"] = (["Model", "Condition", "TP", "FP", "FN", "TN", "Pooled FPR", "Pooled FNR",
                    "Scenarios with no detections"], rows)

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            for s in scenarios:
                run = runs.get((model, condition, s))
                if run is None:
                    continue
                r = run["result"]
                reused = r["runtime"].get("reused_model_from")
                rows.append([LABEL[model], LABEL[condition], s, len(r["history"]), best_epoch(r["history"]),
                             fmt(min(h["val_loss"] for h in r["history"]), 4), fmt(r["test"]["threshold"], 4),
                             fmt(r["runtime"]["elapsed_seconds"] / 3600, 2),
                             Path(reused).name if reused else ""])
    t["training"] = (["Model", "Condition", "Scenario", "Epochs", "Best epoch", "Best val loss", "τ",
                      "Hours", "Model reused from"], rows)
    return t


def load_stide(root: Path | None) -> dict:
    """{(n, scenario): metrics.json} from scripts/run_stide.py, or {} when there is no STIDE output."""
    if root is None or not root.is_dir():
        return {}
    stide = {}
    for path in sorted(root.glob("stide_n*/held_out/*/metrics.json")):
        n = int(path.parents[2].name.removeprefix("stide_n"))
        stide[(n, path.parent.name)] = json.loads(path.read_text(encoding="utf-8"))
    return stide


def stide_metric(stide, n, scenario, name):
    m = stide.get((n, scenario))
    return None if m is None else m["test"].get(name)


def load_logreg(root: Path | None) -> dict:
    """{("logreg", condition, scenario): {"result": metrics.json}} from scripts/run_logreg.py, or {}."""
    if root is None or not root.is_dir():
        return {}
    return {("logreg", path.parents[1].name, path.parent.name):
            {"result": json.loads(path.read_text(encoding="utf-8"))}
            for path in sorted(root.glob("logreg/*/*/metrics.json"))}


def paired(get_a, get_b, scenarios) -> list:
    """Mean of a − b over scenarios, wins/losses, and sign-test p (scenarios missing either side are skipped)."""
    diffs = []
    for s in scenarios:
        a, b = get_a(s), get_b(s)
        if a is not None and b is not None:
            diffs.append(a - b)
    wins, losses, p = sign_test(diffs)
    return [fmt(mean(diffs)), f"{wins}/{losses}", fmt(p, 4)]


def baseline_tables(runs, stide, logreg, scenarios) -> dict[str, tuple[list, list]]:
    """STIDE (held-out by construction: it never sees an attack) and the bag-of-calls logistic regression
    (trained in the same three conditions as the neural models) next to the neural runs."""
    t = {}
    allruns = {**runs, **logreg}
    ngrams = sorted({n for n, _ in stide})
    models = MODELS + (("logreg",) if logreg else ())
    names = ("auroc", "auprc", "f1", "precision", "recall", "fpr")

    rows = [[LABEL[m], LABEL[c], sum((m, c, s) in allruns for s in scenarios)]
            + [fmt(mean([metric(allruns, m, c, s, k) for s in scenarios])) for k in names]
            for c in CONDITIONS for m in models]
    rows += [[f"STIDE n = {n}", LABEL["held_out"], sum((n, s) in stide for s in scenarios)]
             + [fmt(mean([stide_metric(stide, n, s, k) for s in scenarios])) for k in names] for n in ngrams]
    t["baselines_overall"] = (["Method", "Condition", "Runs", "AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR"],
                              rows)

    cols = [(f"STIDE n = {n}", lambda s, k, n=n: stide_metric(stide, n, s, k)) for n in ngrams]
    cols += [(f"{LABEL[m]} Held-out", lambda s, k, m=m: metric(allruns, m, "held_out", s, k))
             for m in reversed(models)]
    for name in ("auroc", "auprc"):
        rows = [[s] + [fmt(get(s, name)) for _, get in cols] for s in scenarios]
        rows.append(["**Mean**"] + [fmt(mean([get(s, name) for s in scenarios])) for _, get in cols])
        t[f"held_out_{name}_by_scenario"] = (["Scenario"] + [h for h, _ in cols], rows)

    if stide:
        rows = []
        for n in ngrams:
            for m in MODELS:
                row = [f"STIDE n = {n}", LABEL[m]]
                for c, k in (("held_out", "auroc"), ("held_out", "auprc"), ("seen_all", "auroc")):
                    row += paired(lambda s: stide_metric(stide, n, s, k), lambda s: metric(runs, m, c, s, k), scenarios)
                rows.append(row)
        t["stide_comparison"] = (["STIDE", "Neural model", "AUROC − Held-out", "+/−", "p", "AUPRC − Held-out",
                                  "+/−", "p", "AUROC − Seen (all)", "+/−", "p"], rows)

    if logreg:
        rows = [[s] + [fmt(metric(logreg, "logreg", c, s, "auroc")) for c in CONDITIONS] for s in scenarios]
        rows.append(["**Mean**"] + [fmt(mean([metric(logreg, "logreg", c, s, "auroc") for s in scenarios]))
                                    for c in CONDITIONS])
        t["logreg_auroc_by_scenario"] = (["Scenario"] + [f"LogReg {LABEL[c]}" for c in CONDITIONS], rows)
        rows = []
        for c in CONDITIONS:
            for m in MODELS:
                row = [LABEL[c], LABEL[m]]
                for k in ("auroc", "auprc"):
                    row += paired(lambda s: metric(logreg, "logreg", c, s, k), lambda s: metric(runs, m, c, s, k),
                                  scenarios)
                rows.append(row)
        t["logreg_comparison"] = (["Condition", "Neural model", "AUROC LogReg − neural", "+/−", "p",
                                   "AUPRC LogReg − neural", "+/−", "p"], rows)
    return t


def dataset_table(runs, scenarios):
    rows = []
    for s in scenarios:
        run = runs.get(("lstm", "seen", s)) or runs.get(("transformer", "seen", s))
        if run is None:
            continue
        sp = run["result"]["data"]["splits"]
        row = [s]
        for split in ("train", "val", "test"):
            row.append(f"{sp[split]['recordings']} ({sp[split]['attack_recordings']})")
        for split in ("train", "val", "test"):
            row.append(fmt(sum(sp[split]["windows_available"].values())))
        for split in ("train", "val", "test"):
            row.append(fmt(sp[split]["windows"]))
        for split in ("train", "val", "test"):
            row.append(f"{100 * sp[split]['malicious_window_rate']:.1f}")
        rows.append(row)
    headers = (["Scenario", "Train recs (attack)", "Val recs (attack)", "Test recs (attack)",
                "Train windows available", "Val windows available", "Test windows available",
                "Train windows used", "Val windows used", "Test windows used",
                "Train % malicious", "Val % malicious", "Test % malicious"])
    pooled = next((runs[k] for k in sorted(runs) if k[1] == "seen_all"), None)
    totals = None
    if pooled is not None:
        sp = pooled["result"]["data"]["splits"]
        totals = {split: {k: sp[split][k] for k in ("recordings", "attack_recordings", "windows",
                                                     "malicious_window_rate")}
                  for split in ("train", "val")}
        totals["train"]["windows_available"] = sum(sp["train"]["windows_available"].values())
        totals["val"]["windows_available"] = sum(sp["val"]["windows_available"].values())
        totals["by_length"] = {split: sp[split]["by_length"] for split in ("train", "val")}
    return (headers, rows), totals


def parameter_table(runs):
    import torch  # noqa: F401  (only needed here)
    from lidds_gen.models.lstm import LSTMClassifier
    from lidds_gen.models.transformer import TransformerClassifier

    cfg = next(iter(runs.values()))["cfg"]
    m = cfg["model"]
    vocab = defaultdict(set)
    for (model, condition, _), run in runs.items():
        vocab["pooled" if condition != "seen" else "single"].add(run["result"]["data"]["vocab_size"])
    pooled_v = max(vocab["pooled"]) if vocab["pooled"] else 119

    def build(name, v):
        if name == "lstm":
            return LSTMClassifier(v, m["embed_dim"], m["lstm"]["hidden_size"], m["lstm"]["num_layers"], m["dropout"])
        return TransformerClassifier(v, m["embed_dim"], m["transformer"]["num_heads"], m["transformer"]["num_layers"],
                                     m["transformer"]["ff_dim"], m["dropout"], cfg["data"]["max_length"])

    rows, detail = [], []
    for name in MODELS:
        net = build(name, pooled_v)
        groups = defaultdict(int)
        for pname, p in net.named_parameters():
            groups[pname.split(".")[0]] += p.numel()
        total = sum(groups.values())
        single = sorted(sum(p.numel() for p in build(name, v).parameters()) for v in vocab["single"])
        rows.append([LABEL[name], fmt(total), fmt(total - groups["embedding"]), fmt(groups["embedding"]),
                     f"{fmt(single[0])}–{fmt(single[-1])}" if single else "–"])
        for g, n in groups.items():
            detail.append([LABEL[name], g, fmt(n)])
    headers = ["Model", f"Parameters (vocab {pooled_v})", "Excluding embedding", "Embedding",
               "Single-scenario runs"]
    vocab_note = {k: sorted(v) for k, v in vocab.items()}
    return (headers, rows), (["Model", "Module", "Parameters"], detail), vocab_note


# ---------------------------------------------------------------- figures
def style(ax, grid_axis="y", integer_x=False):
    from matplotlib.ticker import MaxNLocator

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(runs, scenarios, out: Path, plt, stide=None, logreg=None):
    stide, logreg = stide or {}, logreg or {}
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK2, "axes.titlesize": 9,
                         "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "legend.frameon": False, "legend.fontsize": 8})
    written = []

    # Loss curves for the pooled (all-scenario) control, one model per line colour.
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for model in MODELS:
        donor = next((runs[k] for k in sorted(runs) if k[0] == model and k[1] == "seen_all"
                      and not runs[k]["result"]["runtime"].get("reused_model_from")), None)
        if donor is None:
            continue
        h = donor["result"]["history"]
        ep = [e["epoch"] + 1 for e in h]
        ax.plot(ep, [e["train_loss"] for e in h], color=MODEL_COLOR[model], lw=2, ls="--",
                label=f"{LABEL[model]} train")
        ax.plot(ep, [e["val_loss"] for e in h], color=MODEL_COLOR[model], lw=2, label=f"{LABEL[model]} validation")
        b = best_epoch(h)
        ax.plot([b], [h[b - 1]["val_loss"]], "o", ms=6, color=MODEL_COLOR[model], mec="#fcfcfb", mew=1.5)
    style(ax, integer_x=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Class-weighted BCE loss")
    ax.set_title("Seen (all) control: training and validation loss (dot = kept epoch)", loc="left")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "loss_seen_all.png", dpi=200)
    plt.close(fig)
    written.append("loss_seen_all.png")

    # Validation loss per scenario: small multiples, both models in each panel.
    for condition in ("seen", "held_out"):
        cols = 5
        rows_n = math.ceil(len(scenarios) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.0 * rows_n), squeeze=False)
        for ax, s in zip(axes.flat, scenarios):
            for model in MODELS:
                run = runs.get((model, condition, s))
                if run is None:
                    continue
                h = run["result"]["history"]
                ep = [e["epoch"] + 1 for e in h]
                ax.plot(ep, [e["train_loss"] for e in h], color=MODEL_COLOR[model], lw=1.2, ls="--")
                ax.plot(ep, [e["val_loss"] for e in h], color=MODEL_COLOR[model], lw=2)
            style(ax, integer_x=True)
            ax.set_title(SHORT.get(s, s), loc="left")
        for ax in list(axes.flat)[len(scenarios):]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=MODEL_COLOR[m], lw=2) for m in MODELS]
        handles += [plt.Line2D([], [], color=MUTED, lw=2), plt.Line2D([], [], color=MUTED, lw=1.2, ls="--")]
        fig.legend(handles, [LABEL[m] for m in MODELS] + ["validation", "train"], loc="upper right", ncol=4)
        fig.suptitle(f"{LABEL[condition]}: loss per epoch by target scenario", x=0.01, ha="left", color=INK)
        fig.supxlabel("Epoch", color=INK2, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"loss_{condition}.png", dpi=200)
        plt.close(fig)
        written.append(f"loss_{condition}.png")

    # Per-scenario AUROC, one panel per model, one marker colour per condition.
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    y = np.arange(len(scenarios))[::-1]
    for ax, model in zip(axes, MODELS):
        for i, condition in enumerate(CONDITIONS):
            vals = [metric(runs, model, condition, s, "auroc") for s in scenarios]
            xs = [v if v is not None else np.nan for v in vals]
            st = COND_STYLE[condition]
            ax.scatter(xs, y + (1 - i) * 0.22, s=36 if st["marker"] == "o" else 30, marker=st["marker"],
                       facecolor=st["face"], edgecolor=st["color"], linewidth=1.4, label=LABEL[condition], zorder=3)
        ax.axvline(0.5, color=MUTED, lw=1, ls=":", zorder=1)
        ax.set_yticks(y, [SHORT.get(s, s) for s in scenarios])
        ax.set_xlim(0.3, 1.01)
        style(ax, grid_axis="x")
        ax.set_title(LABEL[model], loc="left")
        ax.set_xlabel("Test AUROC (dotted line = chance)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "auroc_by_scenario.png", dpi=200)
    plt.close(fig)
    written.append("auroc_by_scenario.png")

    # AUROC gap (Seen (all) minus Held-out) per scenario, both models.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    h = 0.38
    for i, model in enumerate(MODELS):
        gaps = []
        for s in scenarios:
            a, b = metric(runs, model, "seen_all", s, "auroc"), metric(runs, model, "held_out", s, "auroc")
            gaps.append(np.nan if a is None or b is None else a - b)
        ax.barh(y + (0.5 - i) * h, gaps, height=h - 0.04, color=MODEL_COLOR[model], label=LABEL[model])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [SHORT.get(s, s) for s in scenarios])
    style(ax, grid_axis="x")
    ax.set_xlabel("AUROC gap: Seen (all) − Held-out")
    ax.set_title("How much AUROC is lost when the target's attacks are withheld", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "auroc_gap.png", dpi=200)
    plt.close(fig)
    written.append("auroc_gap.png")

    # Mean AUROC by window length.
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        for condition in CONDITIONS:
            vals = [mean([metric(runs, model, condition, s, "auroc", L) for s in scenarios]) for L in LENGTHS]
            st = COND_STYLE[condition]
            ax.plot([int(L) for L in LENGTHS], [np.nan if v is None else v for v in vals], marker=st["marker"],
                    ms=7, lw=2, ls=st["ls"], color=st["color"], mfc=st["face"], mec=st["color"], mew=1.5,
                    label=LABEL[condition])
        ax.set_xticks([50, 100, 200])
        style(ax)
        ax.set_title(LABEL[model], loc="left")
        ax.set_xlabel("Window length (calls)")
    axes[0].set_ylabel("Mean test AUROC over scenarios")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "auroc_by_length.png", dpi=200)
    plt.close(fig)
    written.append("auroc_by_length.png")

    # Score distributions: where Held-out transfer works, where it fails.
    picks = [s for s in ("CVE-2012-2122", "Bruteforce_CWE-307", "ZipSlip") if s in scenarios]
    model = "transformer"
    fig, axes = plt.subplots(len(picks), 2, figsize=(8, 2.0 * len(picks)), squeeze=False, sharex=True)
    bins = np.linspace(0, 1, 41)
    for r, s in enumerate(picks):
        for c, condition in enumerate(("seen_all", "held_out")):
            ax = axes[r][c]
            run = runs.get((model, condition, s))
            if run is None:
                ax.axis("off")
                continue
            pred = np.load(run["dir"] / "test_predictions.npz")
            for label, color, name in ((0, BLUE, "benign"), (1, PINK, "malicious")):
                p = pred["prob"][pred["label"] == label]
                ax.hist(p, bins=bins, weights=np.full(len(p), 1 / max(len(p), 1)), color=color, alpha=0.75,
                        label=name, edgecolor="#fcfcfb", linewidth=0.4)
            tau = run["result"]["test"]["threshold"]
            ax.axvline(tau, color=INK, lw=1, ls="--")
            auroc = run["result"]["test"]["auroc"]
            ax.set_title(f"{SHORT.get(s, s)} · {LABEL[condition]} · AUROC {auroc:.2f}", loc="left")
            style(ax)
    axes[0][0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("Predicted probability of malicious (dashed = τ)")
    fig.supylabel("Share of class", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=200)
    plt.close(fig)
    written.append("score_distributions.png")

    # STIDE (classic n = 6) against each neural Held-out model, one point per target.
    if any(n == 6 for n, _ in stide):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot([0.4, 1.0], [0.4, 1.0], color=MUTED, lw=1, ls=":", zorder=1)
        for model in MODELS:
            pts = [(stide_metric(stide, 6, s, "auroc"), metric(runs, model, "held_out", s, "auroc"), s)
                   for s in scenarios]
            pts = [p for p in pts if p[0] is not None and p[1] is not None]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=38, color=MODEL_COLOR[model],
                       edgecolor=SURFACE, linewidth=1.2, label=f"{LABEL[model]} Held-out", zorder=3)
        for s, dx, dy, ha in (("CVE-2017-7529", -0.012, 0.0, "right"), ("CVE-2020-23839", 0.0, -0.04, "center"),
                              ("CVE-2017-12635_6", -0.012, 0.012, "right"), ("Bruteforce_CWE-307", -0.012, 0.02, "right")):
            x = stide_metric(stide, 6, s, "auroc")
            ys = [metric(runs, m, "held_out", s, "auroc") for m in MODELS]
            if x is not None and all(v is not None for v in ys):
                name = "CouchDB (both models)" if s == "CVE-2017-12635_6" else SHORT.get(s, s)
                ax.annotate(name, (x + dx, max(ys) + dy if dy >= 0 else min(ys) + dy), ha=ha,
                            va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.set_aspect("equal")
        style(ax, grid_axis="both")
        ax.set_xlabel("STIDE (n = 6) test AUROC")
        ax.set_ylabel("Neural Held-out test AUROC")
        ax.set_title("Unseen attacks: supervised models vs STIDE\n(above the dotted line = neural model better)",
                     loc="left")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "stide_vs_held_out.png", dpi=200)
        plt.close(fig)
        written.append("stide_vs_held_out.png")

    # Bag-of-calls logistic regression against each neural model, condition by condition.
    if logreg:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
        for ax, condition in zip(axes, CONDITIONS):
            ax.plot([0.4, 1.0], [0.4, 1.0], color=MUTED, lw=1, ls=":", zorder=1)
            for model in MODELS:
                pts = [(metric(logreg, "logreg", condition, s, "auroc"), metric(runs, model, condition, s, "auroc"))
                       for s in scenarios]
                pts = [p for p in pts if p[0] is not None and p[1] is not None]
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=MODEL_COLOR[model],
                           edgecolor=SURFACE, linewidth=1.2, label=LABEL[model], zorder=3)
            ax.set_xlim(0.4, 1.0)
            ax.set_ylim(0.4, 1.0)
            ax.set_aspect("equal")
            style(ax, grid_axis="both")
            ax.set_title(LABEL[condition], loc="left")
            ax.set_xlabel("Bag-of-calls LogReg AUROC")
        axes[0].set_ylabel("Neural model AUROC")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2)
        fig.suptitle("Neural models vs bag-of-calls logistic regression per target "
                     "(above the dotted line = neural model better)", x=0.01, ha="left", color=INK, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / "logreg_vs_neural.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        written.append("logreg_vs_neural.png")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output", default="results_summary")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--stide-root", default=None,
                    help="scripts/run_stide.py output (default: <results-root>_stide, skipped if missing)")
    ap.add_argument("--logreg-root", default=None,
                    help="scripts/run_logreg.py output (default: <results-root>_logreg, skipped if missing)")
    args = ap.parse_args()
    root = Path(args.results_root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No completed runs under {args.results_root}")
    scenarios = [s for s in SCENARIOS if any(k[2] == s for k in runs)]
    stide = load_stide(Path(args.stide_root) if args.stide_root else root.with_name(root.name + "_stide"))
    logreg = load_logreg(Path(args.logreg_root) if args.logreg_root else root.with_name(root.name + "_logreg"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    t = tables(runs, scenarios)
    if stide or logreg:
        t.update(baseline_tables(runs, stide, logreg, scenarios))
    t["dataset"], pooled = dataset_table(runs, scenarios)
    t["parameters"], t["parameters_by_module"], vocab_sizes = parameter_table(runs)
    for name, (headers, rows) in t.items():
        write_csv(out / f"{name}.csv", headers, rows)

    sections = [
        ("Overall test metrics (mean over target scenarios)", "overall"),
        ("Test AUROC by scenario", "per_scenario_auroc"),
        ("Test AUPRC by scenario", "per_scenario_auprc"),
        ("Test F1 by scenario (threshold τ from validation)", "per_scenario_f1"),
        ("Test false-positive rate by scenario", "per_scenario_fpr"),
        ("Generalization gaps (mean over scenarios; +/− = scenarios where the first condition is higher/lower; "
         "two-sided sign test)", "gaps_summary"),
        ("AUROC gaps by scenario", "gaps_auroc_by_scenario"),
        ("Transformer vs LSTM (paired by scenario)", "model_comparison"),
        ("Mean test AUROC and F1 by requested window length", "by_length"),
        ("Errors at τ, pooled over scenarios", "errors"),
        ("Baselines next to the neural models (mean over target scenarios; STIDE τ keeps benign validation "
         "FPR ≤ 1%, LogReg τ is chosen like the neural models)", "baselines_overall"),
        ("Held-out test AUROC by scenario, baselines and neural models", "held_out_auroc_by_scenario"),
        ("Held-out test AUPRC by scenario, baselines and neural models", "held_out_auprc_by_scenario"),
        ("STIDE minus neural model (mean over scenarios; +/− = scenarios where STIDE is higher/lower; "
         "two-sided sign test)", "stide_comparison"),
        ("Bag-of-calls logistic regression: test AUROC by scenario and condition", "logreg_auroc_by_scenario"),
        ("Logistic regression minus neural model in the same condition (+/− = scenarios where LogReg is "
         "higher/lower; two-sided sign test)", "logreg_comparison"),
        ("Training summary", "training"),
        ("Dataset per scenario (single-scenario splits; windows used after the per-scenario caps)", "dataset"),
        ("Parameter counts", "parameters"),
        ("Parameters by module (pooled vocabulary)", "parameters_by_module"),
    ]
    complete = len(runs)
    doc = ["# Results summary", "",
           f"Generated by `scripts/analyze_results.py` from {complete} completed run(s). "
           "Tables are also saved as CSV in this folder; figures are in `figures/`.", ""]
    for title, key in sections:
        if key in t:
            doc += [f"## {title}", "", markdown(*t[key]), ""]
    if pooled:
        doc += ["## Pooled training data (all 15 scenarios, Seen (all) control)", "",
                markdown(["Split", "Recordings (attack)", "Windows available", "Windows used", "% malicious"],
                         [[split, f"{pooled[split]['recordings']} ({pooled[split]['attack_recordings']})",
                           fmt(pooled[split]["windows_available"]), fmt(pooled[split]["windows"]),
                           f"{100 * pooled[split]['malicious_window_rate']:.1f}"] for split in ("train", "val")]),
                "", "Windows by requested length (train): "
                + ", ".join(f"{L}: {fmt(v['windows'])} ({100 * v['malicious_window_rate']:.1f}% malicious)"
                            for L, v in pooled["by_length"]["train"].items()), ""]
    doc += [f"Vocabulary sizes: single-scenario {vocab_sizes.get('single')}, pooled {vocab_sizes.get('pooled')}.", ""]
    (out / "results.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"[done] tables -> {out / 'results.md'}")

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = Path(args.figures)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name in figures(runs, scenarios, fig_dir, plt, stide, logreg):
            print(f"[done] figure -> {fig_dir / name}")


if __name__ == "__main__":
    main()
