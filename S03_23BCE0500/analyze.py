#!/usr/bin/env python3
"""
S03 analysis: raw runs -> medians, derived metrics, optimal-split table, manual-format CSV, figures.

Reads  results/s03_raw_runs.csv  (written by run_experiment.py)
Writes results/s03_results_median.csv, results/s03_optimal_split.csv,
       results/s03_results_manual_format.csv, results/figures/*.png|svg

All derived metrics (speedup, efficiency, utilisation, ...) are computed here from the raw measurements.

Definitions
  Execution_Time      makespan of one run in ms (start of GPU enqueue -> CPU and GPU both finished)
  Speedup             median CPU-serial makespan (1 thread, CPU share 100%) / makespan
  Speedup_vs_GPU_only median GPU-only makespan (CPU share 0%, same sweep) / makespan
  Resource_Count      CPU threads actually given work + 1 if the GPU got work
  Efficiency          Speedup / Resource_Count (the GPU counts as one resource unit; not comparable with
                      a pure CPU parallel efficiency)
  Partition efficiency ideal makespan / measured makespan, where ideal = 1 / (1/T_cpu_only + 1/T_gpu_only)
                      (perfect overlap of the two devices, no extra overhead)
  CPU utilisation     sum of per-thread busy time / (threads x makespan)
  GPU kernel util.    summed kernel time / makespan;   GPU active = GPU span (H2D+kernel+D2H) / makespan
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette
C_CPU, C_GPU, C_HYB, C_RED, C_GREY, C_PURPLE = "#E69F00", "#0072B2", "#009E73", "#D55E00", "#6b6b6b", "#CC79A7"

BASE_KEY = ["workload", "tiles", "tile_dim", "passes", "streams", "chunk_tiles", "pinned"]
CFG_KEY = BASE_KEY + ["threads_req"]
NUMERIC = ["tiles", "tile_dim", "passes", "threads_req", "threads_actual", "streams", "chunk_tiles", "pinned", "cpu_frac",
           "tiles_cpu", "tiles_gpu", "rep", "makespan_ms", "sched_ms", "cpu_wall_ms", "cpu_busy_sum_ms", "gpu_h2d_ms",
           "gpu_kernel_ms", "gpu_d2h_ms", "gpu_span_ms", "sync_wait_ms", "max_rel_err", "host_bytes", "device_bytes"]


def load(path):
    raw = pd.read_csv(path)
    for c in NUMERIC:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["cpu_frac"] = raw["cpu_frac"].round(4)
    return raw


def add_baselines(df):
    """df has one row per (config, cpu_frac) with median metrics. Adds baselines and derived metrics."""
    cpu_only = df[df.cpu_frac == 1.0].set_index(CFG_KEY)["makespan_ms"].rename("cpu_only_ms")
    gpu_only = df[df.cpu_frac == 0.0].set_index(CFG_KEY)["makespan_ms"].rename("gpu_only_ms")
    serial = (df[(df.cpu_frac == 1.0) & (df.threads_req == 1)].set_index(BASE_KEY)["makespan_ms"].rename("serial_ms"))
    df = df.join(cpu_only, on=CFG_KEY).join(gpu_only, on=CFG_KEY).join(serial, on=BASE_KEY)
    df["speedup_vs_serial"] = df.serial_ms / df.makespan_ms
    df["speedup_vs_gpu_only"] = df.gpu_only_ms / df.makespan_ms
    df["speedup_vs_cpu_only"] = df.cpu_only_ms / df.makespan_ms
    df["resource_count"] = np.where(df.tiles_cpu > 0, df.threads_req, 0) + np.where(df.tiles_gpu > 0, 1, 0)
    df["efficiency"] = df.speedup_vs_serial / df.resource_count
    df["ideal_ms"] = 1.0 / (1.0 / df.cpu_only_ms + 1.0 / df.gpu_only_ms)
    df["partition_eff"] = df.ideal_ms / df.makespan_ms
    thr = df.threads_req.where(df.tiles_cpu > 0, np.nan)
    df["cpu_util"] = (df.cpu_busy_sum_ms / (thr * df.makespan_ms)).fillna(0.0)
    df["gpu_kernel_util"] = df.gpu_kernel_ms / df.makespan_ms
    df["gpu_active"] = df.gpu_span_ms / df.makespan_ms
    df["comm_ms"] = df.gpu_h2d_ms + df.gpu_d2h_ms
    df["transfer_frac"] = (df.comm_ms / df.gpu_span_ms).where(df.gpu_span_ms > 0, np.nan)
    df["imbalance_ms"] = (df.cpu_wall_ms - df.gpu_span_ms).abs().where((df.tiles_cpu > 0) & (df.tiles_gpu > 0), np.nan)
    return df


def summarise(raw):
    g = raw.groupby(CFG_KEY + ["cpu_frac"])
    cols = [c for c in NUMERIC if c not in CFG_KEY + ["cpu_frac", "rep"]]
    med = g[cols].median().reset_index()
    spread = g["makespan_ms"].agg(["count", "min", "max", "mean", "std"]).reset_index()
    spread.columns = CFG_KEY + ["cpu_frac", "n_reps", "makespan_min", "makespan_max", "makespan_mean", "makespan_std"]
    q = g["makespan_ms"].quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = CFG_KEY + ["cpu_frac", "makespan_q1", "makespan_q3"]
    med = med.merge(spread, on=CFG_KEY + ["cpu_frac"]).merge(q, on=CFG_KEY + ["cpu_frac"])
    med["makespan_cv"] = med.makespan_std / med.makespan_mean
    return add_baselines(med)


def optimal_table(med):
    rows = []
    for key, g in med.groupby(CFG_KEY):
        g = g.sort_values("cpu_frac")
        if g.cpu_frac.min() > 0 or g.cpu_frac.max() < 1:
            continue
        best = g.loc[g.makespan_ms.idxmin()]
        near = g[g.makespan_ms <= 1.02 * best.makespan_ms]
        gpu = g[g.cpu_frac == 0].iloc[0]
        cpu = g[g.cpu_frac == 1].iloc[0]
        hyb = g[(g.cpu_frac > 0) & (g.cpu_frac < 1)]
        best_h = hyb.loc[hyb.makespan_ms.idxmin()] if len(hyb) else best
        gain = (gpu.makespan_ms - best_h.makespan_ms) / gpu.makespan_ms
        noise = 2 * max(np.nan_to_num(best_h.makespan_cv), np.nan_to_num(gpu.makespan_cv))
        single = gpu if gpu.makespan_ms <= cpu.makespan_ms else cpu
        gain_single = (single.makespan_ms - best_h.makespan_ms) / single.makespan_ms
        noise_single = 2 * max(np.nan_to_num(best_h.makespan_cv), np.nan_to_num(single.makespan_cv))
        pred = gpu.makespan_ms / (cpu.makespan_ms + gpu.makespan_ms)
        rows.append(dict(zip(CFG_KEY, key)) | dict(
            gpu_only_ms=gpu.makespan_ms, cpu_only_ms=cpu.makespan_ms, serial_ms=best.serial_ms,
            best_frac=best.cpu_frac, best_ms=best.makespan_ms,
            best_hybrid_frac=best_h.cpu_frac, best_hybrid_ms=best_h.makespan_ms,
            near_opt_lo=near.cpu_frac.min(), near_opt_hi=near.cpu_frac.max(),
            speedup_best_vs_gpu_only=gpu.makespan_ms / best.makespan_ms,
            speedup_best_vs_cpu_only=cpu.makespan_ms / best.makespan_ms,
            speedup_best_vs_serial=best.serial_ms / best.makespan_ms,
            hybrid_gain_vs_gpu_only=gain, noise_threshold=noise,
            hybrid_beats_gpu_only=bool(gain > max(noise, 0.02)),
            hybrid_gain_vs_best_single=gain_single,
            hybrid_beats_best_single=bool(gain_single > max(noise_single, 0.02)),
            predicted_frac=pred, ideal_ms=best.ideal_ms, partition_eff_best=best.partition_eff,
            faster_single_device="GPU-only" if gpu.makespan_ms <= cpu.makespan_ms else "CPU-only",
            gpu_only_transfer_frac=gpu.transfer_frac, cv_best=best.makespan_cv, cv_gpu_only=gpu.makespan_cv,
            sync_ms_at_best=best.sync_wait_ms, sched_ms_at_best=best.sched_ms,
            cpu_util_at_best=best.cpu_util, gpu_active_at_best=best.gpu_active))
    return pd.DataFrame(rows)


def manual_format(raw, med):
    """One CSV row per run with the columns required by the case-study manual (+ extras)."""
    m = med[CFG_KEY + ["cpu_frac", "serial_ms", "gpu_only_ms", "cpu_only_ms"]]
    r = raw.merge(m, on=CFG_KEY + ["cpu_frac"], how="left")
    used_thr = np.where(r.tiles_cpu > 0, r.threads_req, 0)
    r["Resource_Count"] = used_thr + np.where(r.tiles_gpu > 0, 1, 0)
    r["Speedup"] = r.serial_ms / r.makespan_ms
    model = np.where(r.tiles_gpu == 0, "OpenMP (CPU-only)", np.where(r.tiles_cpu == 0, "CUDA (GPU-only)", "OpenMP+CUDA hybrid"))
    out = pd.DataFrame({
        "Student_ID": r.Student_ID, "Batch_ID": r.Batch_ID,
        "Workload": "tileblur_p" + r.passes.astype(int).astype(str),
        "Dataset": r.Dataset,
        "Problem_Size": (r.tiles * r.tile_dim * r.tile_dim).astype("int64"),
        "Algorithm": "3x3 box blur x" + r.passes.astype(int).astype(str) + " passes, CPU/GPU tile partitioning",
        "Parallel_Model": model, "Hardware": r.Hardware,
        "Resource_Count": r.Resource_Count, "Threads": r.threads_req,
        "Execution_Time": r.makespan_ms,
        "Speedup": r.Speedup, "Efficiency": r.Speedup / r.Resource_Count,
        "Communication_Time": r.gpu_h2d_ms + r.gpu_d2h_ms,
        "Synchronization_Time": r.sync_wait_ms, "Scheduling_Time": r.sched_ms,
        "Memory_Use": (r.host_bytes + r.device_bytes) / 1048576.0,
        "Energy_if_available": np.nan, "Run_ID": r.Run_ID,
        # extras (units: time in ms, Memory_Use in MiB, Problem_Size in pixels)
        "Tiles": r.tiles.astype(int), "Tile_Dim": r.tile_dim.astype(int), "Passes": r.passes.astype(int),
        "CPU_Fraction": r.cpu_frac, "Tiles_CPU": r.tiles_cpu.astype(int), "Tiles_GPU": r.tiles_gpu.astype(int),
        "Streams": r.streams.astype(int), "Pinned_Host_Memory": r.pinned.astype(int), "Repetition": r.rep.astype(int),
        "Baseline_for_Speedup": "CPU serial (1 thread, CPU share 100%), median of repetitions",
        "Speedup_vs_GPU_only": r.gpu_only_ms / r.makespan_ms, "Speedup_vs_CPU_only_same_threads": r.cpu_only_ms / r.makespan_ms,
        "CPU_Compute_Wall_ms": r.cpu_wall_ms, "CPU_Busy_Sum_ms": r.cpu_busy_sum_ms,
        "GPU_H2D_ms": r.gpu_h2d_ms, "GPU_Kernel_ms": r.gpu_kernel_ms, "GPU_D2H_ms": r.gpu_d2h_ms, "GPU_Span_ms": r.gpu_span_ms,
        "Max_Rel_Error": r.max_rel_err})
    return out


# ------------------------------------------------------------------ plotting
def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 110, "savefig.dpi": 200})


def save(fig, figdir, name):
    fig.savefig(figdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def headline(med):
    tiles = med.tiles.max()
    th = med.threads_req.max()
    return med[(med.tiles == tiles) & (med.threads_req == th)].sort_values(["passes", "cpu_frac"]), int(tiles), int(th)


def fig_main(med, figdir):
    h, tiles, th = headline(med)
    pass_list = sorted(h.passes.unique())
    fig, axes = plt.subplots(len(pass_list), 3, figsize=(7.4, 1.95 * len(pass_list) + 0.3), squeeze=False)
    for i, p in enumerate(pass_list):
        g = h[h.passes == p].sort_values("cpu_frac")
        x = g.cpu_frac * 100
        best = g.loc[g.makespan_ms.idxmin()]
        ax = axes[i, 0]
        ax.fill_between(x, g.makespan_min, g.makespan_max, color=C_HYB, alpha=0.18, lw=0)
        ax.plot(x, g.makespan_ms, "o-", color=C_HYB, ms=3, lw=1.5, label="makespan")
        ax.plot(x, g.cpu_wall_ms.where(g.tiles_cpu > 0), "--", color=C_CPU, lw=1, label="CPU part")
        ax.plot(x, g.gpu_span_ms.where(g.tiles_gpu > 0), "--", color=C_GPU, lw=1, label="GPU part (H2D+kernel+D2H)")
        ax.plot([best.cpu_frac * 100], [best.makespan_ms], "*", color=C_RED, ms=9, zorder=5, label="best split")
        ax.annotate(f"best: {best.cpu_frac * 100:.0f}% CPU", (best.cpu_frac * 100, best.makespan_ms), textcoords="offset points",
                    xytext=(0, -13 if best.cpu_frac < 0.5 else 8), ha="center", fontsize=6.5, color=C_RED)
        ax.set_ylabel(f"time (ms)\npasses = {int(p)}")
        ax.set_yscale("log")
        if i == 0:
            ax.set_title("Time vs CPU share")
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=5.5)
        ax = axes[i, 1]
        ax.plot(x, g.cpu_util * 100, "o-", color=C_CPU, ms=3, label="CPU busy")
        ax.plot(x, g.gpu_kernel_util * 100, "s-", color=C_GPU, ms=3, label="GPU kernel")
        ax.plot(x, g.gpu_active * 100, "^--", color=C_PURPLE, ms=3, lw=1, label="GPU active (incl. copies)")
        ax.axvline(best.cpu_frac * 100, color=C_RED, lw=0.8, ls=":")
        ax.set_ylim(-3, 108)
        if i == 0:
            ax.set_title("Utilisation vs CPU share")
        ax.set_ylabel("% of makespan")
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=5.5)
        ax = axes[i, 2]
        ax.plot(x, g.speedup_vs_serial, "o-", color=C_HYB, ms=3, label="vs CPU serial")
        ax.plot(x, g.speedup_vs_gpu_only, "s-", color=C_GPU, ms=3, label="vs GPU-only")
        ax.axhline(1, color=C_GREY, lw=0.8)
        ax.axvline(best.cpu_frac * 100, color=C_RED, lw=0.8, ls=":")
        ax.set_yscale("log")
        if i == 0:
            ax.set_title("Speedup vs CPU share")
        ax.set_ylabel("speedup (x)")
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=5.5)
        if i == len(pass_list) - 1:
            for a in axes[i]:
                a.set_xlabel("share of tiles on CPU (%)")
    fig.suptitle(f"{tiles} tiles of {int(h.tile_dim.iloc[0])}x{int(h.tile_dim.iloc[0])}, {th} CPU threads (median of repetitions, band = min-max)", fontsize=8, y=1.0)
    fig.tight_layout()
    save(fig, figdir, "fig1_split_time_util_speedup")


def fig_breakdown(med, figdir):
    h, tiles, th = headline(med)
    pass_list = sorted(h.passes.unique())
    fig, axes = plt.subplots(1, len(pass_list), figsize=(7.4, 2.35), squeeze=False)
    for j, p in enumerate(pass_list):
        g = h[h.passes == p].sort_values("cpu_frac").reset_index(drop=True)
        ax = axes[0, j]
        x = np.arange(len(g))
        w = 0.38
        ax.bar(x - w / 2, g.cpu_wall_ms, w, color=C_CPU, label="CPU compute")
        ax.bar(x + w / 2, g.gpu_h2d_ms, w, color="#56B4E9", label="GPU H2D copy")
        ax.bar(x + w / 2, g.gpu_kernel_ms, w, bottom=g.gpu_h2d_ms, color=C_GPU, label="GPU kernels")
        ax.bar(x + w / 2, g.gpu_d2h_ms, w, bottom=g.gpu_h2d_ms + g.gpu_kernel_ms, color="#003f66", label="GPU D2H copy")
        ax.plot(x, g.makespan_ms, "k_", ms=9, mew=1.6, label="makespan")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f * 100:.0f}" for f in g.cpu_frac])
        ax.set_xlabel("share of tiles on CPU (%)")
        ax.set_title(f"passes = {int(p)}")
        if j == 0:
            ax.set_ylabel("time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Where the time goes: CPU compute vs GPU copy and kernel time ({tiles} tiles, {th} threads)", fontsize=8, y=1.02)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.1))
    save(fig, figdir, "fig2_time_breakdown")


def fig_optimal(opt, figdir):
    if opt.empty:
        return
    th_max, t_max = opt.threads_req.max(), opt.tiles.max()
    pass_list = sorted(opt.passes.unique())
    cols = [C_GPU, C_CPU, C_HYB, C_RED, C_PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.25))
    ax = axes[0]
    for k, p in enumerate(pass_list):
        g = opt[(opt.passes == p) & (opt.tiles == t_max)].sort_values("threads_req")
        ax.plot(g.threads_req, g.best_frac * 100, "o-", color=cols[k % 5], ms=4, label=f"measured, passes {int(p)}")
        ax.plot(g.threads_req, g.predicted_frac * 100, "x--", color=cols[k % 5], ms=5, lw=0.9, label=f"predicted, passes {int(p)}")
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("optimal CPU share (%)")
    ax.set_title(f"Optimal share vs threads ({int(t_max)} tiles)")
    ax.set_xticks(sorted(opt.threads_req.unique()))
    ax.legend(frameon=False, fontsize=5.5)
    ax = axes[1]
    for k, p in enumerate(pass_list):
        g = opt[(opt.passes == p) & (opt.threads_req == th_max)].sort_values("tiles")
        ax.plot(g.tiles, g.best_frac * 100, "o-", color=cols[k % 5], ms=4, label=f"measured, passes {int(p)}")
        ax.plot(g.tiles, g.predicted_frac * 100, "x--", color=cols[k % 5], ms=5, lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("problem size (tiles)")
    ax.set_title(f"Optimal share vs size ({int(th_max)} threads)")
    ax.legend(frameon=False, fontsize=5.5)
    ax = axes[2]
    for k, p in enumerate(pass_list):
        g = opt[(opt.passes == p) & (opt.tiles == t_max)].sort_values("threads_req")
        ax.plot(g.threads_req, g.speedup_best_vs_gpu_only, "o-", color=cols[k % 5], ms=4, label=f"vs GPU-only, passes {int(p)}")
        ax.plot(g.threads_req, g.speedup_best_vs_cpu_only, "s--", color=cols[k % 5], ms=3.5, lw=0.9, label=f"vs CPU-only, passes {int(p)}")
    ax.axhline(1, color=C_GREY, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("speedup of best split (x)")
    ax.set_title("Gain of best split")
    ax.set_xticks(sorted(opt.threads_req.unique()))
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    save(fig, figdir, "fig3_optimal_split_trends")


def fig_all_curves(med, figdir):
    pass_list = sorted(med.passes.unique())
    tile_list = sorted(med.tiles.unique())
    fig, axes = plt.subplots(len(pass_list), len(tile_list), figsize=(3.0 * len(tile_list), 2.3 * len(pass_list)), squeeze=False)
    cm = plt.get_cmap("viridis")
    ths = sorted(med.threads_req.unique())
    for i, p in enumerate(pass_list):
        for j, n in enumerate(tile_list):
            ax = axes[i, j]
            for k, t in enumerate(ths):
                g = med[(med.passes == p) & (med.tiles == n) & (med.threads_req == t)].sort_values("cpu_frac")
                ax.plot(g.cpu_frac * 100, g.makespan_ms, "o-", ms=2.5, lw=1.2, color=cm(k / max(1, len(ths) - 1) * 0.85), label=f"{int(t)} threads")
            ax.set_yscale("log")
            ax.set_title(f"passes {int(p)}, {int(n)} tiles")
            if j == 0:
                ax.set_ylabel("makespan (ms)")
            if i == len(pass_list) - 1:
                ax.set_xlabel("CPU share (%)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, figdir, "fig4_all_configurations")


def fig_overhead(med, figdir):
    h, tiles, th = headline(med)
    pass_list = sorted(h.passes.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for k, p in enumerate(pass_list):
        g = h[h.passes == p].sort_values("cpu_frac")
        c = [C_GPU, C_CPU, C_HYB, C_RED][k % 4]
        axes[0].plot(g.cpu_frac * 100, g.sync_wait_ms, "o-", ms=3, color=c, label=f"passes {int(p)}")
        axes[1].plot(g.cpu_frac * 100, g.sched_ms, "o-", ms=3, color=c, label=f"passes {int(p)}")
    axes[0].set_title("Host wait for GPU at the end (synchronisation)")
    axes[1].set_title("Partition + GPU enqueue time (scheduling)")
    for a in axes:
        a.set_xlabel("share of tiles on CPU (%)")
        a.set_ylabel("ms")
        a.legend(frameon=False)
    fig.tight_layout()
    save(fig, figdir, "fig5_sync_and_scheduling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/s03_raw_runs.csv")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    out = Path(a.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    style()

    raw = load(a.input)
    med = summarise(raw)
    opt = optimal_table(med)
    med.to_csv(out / "s03_results_median.csv", index=False)
    opt.to_csv(out / "s03_optimal_split.csv", index=False)
    manual_format(raw, med).to_csv(out / "s03_results_manual_format.csv", index=False)

    bad = raw[(raw.max_rel_err.notna()) & (raw.max_rel_err > 1e-5)]
    print(f"{len(raw)} runs, {raw.groupby(CFG_KEY).ngroups} configurations, correctness failures: {len(bad)} "
          f"(max error {raw.max_rel_err.max():.2e})")
    fig_main(med, figdir)
    fig_breakdown(med, figdir)
    fig_optimal(opt, figdir)
    fig_all_curves(med, figdir)
    fig_overhead(med, figdir)

    show = opt[["passes", "tiles", "threads_req", "gpu_only_ms", "cpu_only_ms", "best_frac", "best_ms",
                "speedup_best_vs_gpu_only", "predicted_frac", "hybrid_beats_gpu_only"]].round(3)
    print(show.to_string(index=False))
    print(f"\nFiles written to {out}/ and {figdir}/")


if __name__ == "__main__":
    main()
