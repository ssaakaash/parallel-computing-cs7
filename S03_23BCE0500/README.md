# CS7 S03: CPU-GPU Workload Partitioning (BCSE412L Parallel Computing)

Question: is CPU+GPU execution always better than GPU-only execution, and which split of the work minimises the makespan?

Workload: N independent image tiles (T x T float32). Each tile gets P passes of a 3x3 box blur (clamped edges inside the tile).
A share `f` of the tiles runs on the CPU (OpenMP), the remaining tiles run on the GPU (CUDA, chunked copies, one kernel per pass).
`f = 0` is the GPU-only baseline, `f = 1` is the CPU-only baseline, CPU-serial (1 thread, `f = 1`) is the baseline for speedup.

## Files

| File | Purpose |
|---|---|
| `s03_bench.cu` | CUDA + OpenMP benchmark. Prints one raw CSV row per (CPU share, repetition). |
| `run_experiment.py` | Compiles the benchmark, records the environment, runs the sweep, writes `results/s03_raw_runs.csv`. |
| `analyze.py` | Computes medians and derived metrics from the raw rows, finds the optimal split, writes the CSVs and figures. |
| `make_report.py` | Builds the individual technical report (.docx) from the results. |
| `s03_kaggle.ipynb` | Notebook that writes all of the above and runs everything on Kaggle. |

## Reproduce (Kaggle, GPU accelerator on, Internet on)

```
nvcc -O3 -std=c++14 -arch=sm_75 -Xcompiler -fopenmp,-O3,-march=native s03_bench.cu -o s03_bench   # done by run_experiment.py
python run_experiment.py --quick --outdir results_quick     # 2-4 minute sanity run, not for reporting
python run_experiment.py --student-id <REG_NO> --batch-id <BATCH_ID>
python analyze.py
python make_report.py --name "<NAME>" --reg <REG_NO> --batch "<BATCH>"
```

Defaults of the full run: tiles 256, 1024, 4096 (64 MiB, 256 MiB, 1 GiB of input), tile 256 x 256, passes 1 and 16,
CPU threads 1, 2, 4, CPU share 0 to 100% in 10% steps, 7 repetitions, 64-tile chunks, 1 CUDA stream, pinned host memory.
Useful options: `--fine` (5% steps), `--passes 1 4 16`, `--streams 1 4`, `--pinned 0 1`, `--reps 10`.
`run_experiment.py` detects the GPU architecture with `nvidia-smi` (sm_75 for the Tesla T4, sm_60 for a P100).
The host-only build (`--cpu-only-build`) exists to test the scripts without a GPU. Its timings are meaningless and `make_report.py` refuses to use them.

## Outputs (in `results/`)

* `system_info.txt`: GPU, driver, CPU, RAM, compiler versions, compile command, sweep settings.
* `s03_raw_runs.csv`: raw measurements, one row per run (nothing derived).
* `s03_results_manual_format.csv`: one row per run with the columns required by the manual
  (Student_ID, Batch_ID, Workload, Dataset, Problem_Size, Algorithm, Parallel_Model, Hardware, Resource_Count, Threads, Execution_Time,
  Speedup, Efficiency, Communication_Time, Synchronization_Time, Scheduling_Time, Memory_Use, Energy_if_available, Run_ID) plus extra columns.
* `s03_results_median.csv`: medians per point with min, max, quartiles, coefficient of variation, baselines and derived metrics.
* `s03_optimal_split.csv`: one row per configuration: best share, near-optimal range, gains, predicted share, flags for "beats GPU-only" and "beats the faster single device".
* `figures/*.png` and `figures/*.svg`: figures (SVG is the editable form; `analyze.py` is the plotting source).

Units: times in ms, `Memory_Use` in MiB (host input+output plus GPU chunk buffers), `Problem_Size` in pixels (tiles x T x T), `Energy_if_available` is empty (not measured).

## Metric definitions

* `Execution_Time`: makespan of one run, from the start of the GPU enqueue until both CPU and GPU parts have finished.
* `Speedup`: median CPU-serial makespan (1 thread, CPU share 100%) divided by the run's makespan. Baseline is also stated in `Baseline_for_Speedup`.
  `Speedup_vs_GPU_only` and `Speedup_vs_CPU_only_same_threads` give the other two baselines.
* `Resource_Count`: CPU threads that received work, plus 1 if the GPU received work. `Efficiency` = `Speedup` / `Resource_Count`.
  The GPU counts as one resource unit, so this is not comparable with a pure CPU parallel efficiency.
* `Communication_Time`: GPU host-to-device plus device-to-host copy time (CUDA events, summed over chunks). CPU shares use shared memory, so no communication.
* `Synchronization_Time`: time the host spends blocked waiting for the GPU after its own CPU share has finished.
* `Scheduling_Time`: partitioning plus time to enqueue all GPU work.
* Partition efficiency: `1 / (1/T_cpu_only + 1/T_gpu_only)` divided by the measured makespan (perfect overlap, no overhead = 1).
* CPU utilisation: summed per-thread busy time / (threads x makespan). GPU kernel utilisation: summed kernel time / makespan.
* Predicted share: `f* = T_gpu_only / (T_cpu_only + T_gpu_only)`, computed from the two single-device medians.

## Notes

* GPU output is compared with the CPU result for every CPU share (`max_rel_err` in the raw file, first repetition of each point). Runs above 1e-5 are flagged by the driver.
* GPU buffers are allocated once per configuration and excluded from the timing. Host buffers are pinned by default so that copies are asynchronous.
* Repetitions alternate the sweep direction to spread slow drift (clock speed, shared cloud load) over all shares.
* Do not mix results from different hardware without labelling them. `Hardware` is stored in every row.
