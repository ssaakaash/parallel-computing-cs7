#!/usr/bin/env python3
"""
S03 (CS7) CPU-GPU workload partitioning: experiment driver.

Compiles s03_bench.cu, records the environment, sweeps
    passes x problem size (tiles) x CPU thread count x CPU share,
repeats every point, and stores one raw row per run in results/s03_raw_runs.csv.
Derived metrics (speedup, efficiency, utilisation) are NOT computed here; analyze.py does that
from the raw rows.

Usage (from the folder containing s03_bench.cu):
    python run_experiment.py --quick          # 2-4 minute sanity run
    python run_experiment.py                  # full experiment (defaults below)
    python run_experiment.py --student-id 23BCE0434 --batch-id B07
"""
import argparse, csv, datetime, io, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path


def sh(cmd, check=False, env=None):
    try:
        cp = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True, env=env)
    except Exception as e:  # tool missing
        return 1, "", str(e)
    if check and cp.returncode != 0:
        print(cp.stdout, cp.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {cmd}")
    return cp.returncode, cp.stdout.strip(), cp.stderr.strip()


def detect_environment():
    info = {}
    rc, out, _ = sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")
    if rc == 0 and out:
        first = out.splitlines()[0].split(",")
        info["gpu_name"] = first[0].strip()
        info["gpu_mem"] = first[1].strip() if len(first) > 1 else "?"
        info["driver"] = first[2].strip() if len(first) > 2 else "?"
        cc = first[3].strip() if len(first) > 3 else ""
        info["arch"] = "sm_" + cc.replace(".", "") if re.match(r"^\d+\.\d+$", cc) else "sm_75"
        info["gpu_count"] = len(out.splitlines())
    else:
        info.update(gpu_name="none detected", gpu_mem="?", driver="?", arch="sm_75", gpu_count=0)
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or "unknown"
    info["cpu_name"] = cpu
    info["logical_cpus"] = os.cpu_count() or 1
    mem_kb = avail_kb = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_kb = int(line.split()[1])
            if line.startswith("MemAvailable"):
                avail_kb = int(line.split()[1])
    except OSError:
        pass
    info["ram_gb"] = round(mem_kb / 1048576, 1)
    info["ram_avail_gb"] = round(avail_kb / 1048576, 1)
    info["nvcc"] = sh("nvcc --version")[1].replace("\n", " | ")
    info["gcc"] = sh("g++ --version")[1].splitlines()[0] if shutil.which("g++") else "g++ not found"
    info["os"] = platform.platform()
    info["python"] = sys.version.split()[0]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, nargs="+", default=[256, 1024, 4096], help="problem sizes: number of tiles")
    ap.add_argument("--tile", type=int, default=256, help="tile edge length in pixels (floats)")
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 16], help="blur passes per tile (arithmetic intensity)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="OpenMP threads for the CPU share")
    ap.add_argument("--streams", type=int, nargs="+", default=[1], help="CUDA streams (1 = serialised offload)")
    ap.add_argument("--pinned", type=int, nargs="+", default=[1], help="1 = pinned host memory, 0 = pageable")
    ap.add_argument("--chunk", type=int, default=64, help="tiles per transfer chunk")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--fine", action="store_true", help="CPU share step 0.05 instead of 0.1")
    ap.add_argument("--quick", action="store_true", help="small sanity sweep, do not use for reported results")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--cpu-only-build", action="store_true", help="build the host-only shim (pipeline test, no GPU)")
    ap.add_argument("--student-id", default="23BCE0434")
    ap.add_argument("--batch-id", default="BATCH_ID")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--arch", default="auto")
    args = ap.parse_args()

    if args.quick:
        args.tiles, args.passes, args.threads, args.reps = [64, 256], [1, 8], [1, 4], 3

    root = Path(__file__).resolve().parent
    src = root / "s03_bench.cu"
    exe = root / ("s03_bench_cpu" if args.cpu_only_build else "s03_bench")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise SystemExit("s03_bench.cu must be in the same folder as this script")

    env_info = detect_environment()
    arch = env_info["arch"] if args.arch == "auto" else args.arch
    threads = [t for t in args.threads if t <= env_info["logical_cpus"]]
    if threads != args.threads:
        print(f"note: dropping thread counts above the {env_info['logical_cpus']} logical CPUs available -> {threads}")
    if not threads:
        threads = [env_info["logical_cpus"]]

    if not args.skip_compile:
        if args.cpu_only_build:
            compile_cmd = ["g++", "-O3", "-fopenmp", "-march=native", "-DNO_CUDA", "-x", "c++", str(src), "-o", str(exe)]
        else:
            compile_cmd = ["nvcc", "-O3", "-std=c++14", f"-arch={arch}", "-Xcompiler", "-fopenmp,-O3,-march=native",
                           str(src), "-o", str(exe)]
        print("+", " ".join(compile_cmd), flush=True)
        sh(compile_cmd, check=True)
    else:
        compile_cmd = ["(skipped; reused existing binary)"]
    if not exe.exists():
        raise SystemExit("binary missing; compile first")

    hardware = f"{env_info['gpu_name']} + {env_info['cpu_name']} ({env_info['logical_cpus']} logical CPUs)"
    if args.cpu_only_build:
        hardware = "NO GPU (host-only shim) + " + env_info["cpu_name"]
    (out / "system_info.txt").write_text(
        "S03 environment record\n"
        f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"GPU: {env_info['gpu_name']} | memory {env_info['gpu_mem']} | driver {env_info['driver']} | target {arch} | GPUs visible: {env_info['gpu_count']}\n"
        f"CPU: {env_info['cpu_name']} | logical CPUs {env_info['logical_cpus']}\n"
        f"RAM: {env_info['ram_gb']} GB total, {env_info['ram_avail_gb']} GB available at start\n"
        f"OS: {env_info['os']}\nPython: {env_info['python']}\n"
        f"CUDA compiler: {env_info['nvcc']}\nHost compiler: {env_info['gcc']}\n"
        f"Compile command: {' '.join(compile_cmd)}\n"
        f"Runtime: OMP_DYNAMIC=FALSE, CUDA_VISIBLE_DEVICES=0, OMP thread counts {threads}\n"
        f"Sweep: tiles={args.tiles} tile={args.tile} passes={args.passes} streams={args.streams} pinned={args.pinned} "
        f"chunk={args.chunk} reps={args.reps} fine={args.fine} quick={args.quick}\n"
    )

    step = 0.05 if args.fine else 0.1
    nfrac = int(round(1 / step))
    fracs = ",".join(f"{i * step:.2f}" for i in range(nfrac + 1))

    run_env = os.environ.copy()
    run_env["OMP_DYNAMIC"] = "FALSE"
    run_env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    configs = [(p, n, th, st, pin) for p in args.passes for n in args.tiles for th in threads
               for st in args.streams for pin in args.pinned]
    raw_path = out / "s03_raw_runs.csv"
    if raw_path.exists():
        raw_path.rename(out / f"s03_raw_runs_previous_{int(time.time())}.csv")
    cols = None
    t_start = time.time()
    run_counter = 0
    print(f"{len(configs)} configurations x {nfrac + 1} CPU shares x {args.reps} repetitions", flush=True)

    for ci, (p, n, th, st, pin) in enumerate(configs, 1):
        need_gb = 3.2 * n * args.tile * args.tile * 4 / 1e9
        if env_info["ram_avail_gb"] and need_gb > env_info["ram_avail_gb"] * 0.8:
            print(f"skip tiles={n}: needs about {need_gb:.1f} GB host RAM, only {env_info['ram_avail_gb']} GB available")
            continue
        cmd = [str(exe), "--tiles", str(n), "--tile", str(args.tile), "--passes", str(p), "--threads", str(th),
               "--streams", str(st), "--chunk", str(args.chunk), "--pinned", str(pin), "--reps", str(args.reps),
               "--fracs", fracs, "--workload", "tileblur"]
        t0 = time.time()
        cp = subprocess.run(cmd, text=True, capture_output=True, env=run_env)
        if cp.returncode != 0:
            print(cp.stdout[-2000:], cp.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"benchmark failed for {cmd}")
        rows = list(csv.DictReader(io.StringIO(cp.stdout)))
        cfg_id = f"p{p}_n{n}_t{th}_s{st}_pin{pin}"
        for r in rows:
            run_counter += 1
            r["Student_ID"] = args.student_id
            r["Batch_ID"] = args.batch_id
            r["Dataset"] = f"synthetic_tiles_{args.tile}x{args.tile}_float32"
            r["Hardware"] = hardware
            r["Run_ID"] = f"{cfg_id}_f{float(r['cpu_frac']):.2f}_r{r['rep']}"
        if cols is None:
            cols = ["Run_ID", "Student_ID", "Batch_ID", "Dataset", "Hardware"] + [c for c in rows[0].keys()
                                                                                    if c not in ("Run_ID", "Student_ID", "Batch_ID", "Dataset", "Hardware")]
        with open(raw_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if f.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        bad = [r for r in rows if r["max_rel_err"] != "nan" and float(r["max_rel_err"]) > 1e-5]
        dt = time.time() - t0
        done = time.time() - t_start
        eta = done / ci * (len(configs) - ci)
        flag = f"  WARNING: {len(bad)} runs failed the correctness check" if bad else ""
        print(f"[{ci}/{len(configs)}] {cfg_id}: {len(rows)} rows in {dt:.0f}s, elapsed {done / 60:.1f} min, ETA {eta / 60:.1f} min{flag}", flush=True)

    print(f"\nRaw rows -> {raw_path} ({run_counter} runs)\nEnvironment -> {out / 'system_info.txt'}\nNext: python analyze.py")


if __name__ == "__main__":
    main()
