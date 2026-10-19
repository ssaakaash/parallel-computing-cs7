// S03 - CPU-GPU workload partitioning benchmark (CUDA + OpenMP)
//
// Workload: N independent image tiles, each T x T floats. Every tile is processed by P passes of a
// 3x3 box blur (clamped edges inside the tile). Tiles are independent, so any subset can run on
// either device. A fraction f_cpu of the tiles goes to the CPU (OpenMP), the rest to the GPU (CUDA).
//   f_cpu = 1.0 -> CPU-only baseline,  f_cpu = 0.0 -> GPU-only baseline,  otherwise hybrid.
//
// The GPU work (H2D copy, P kernels, D2H copy, in chunks) is enqueued asynchronously first, then the
// host runs its own share with OpenMP while the GPU works, and finally waits for the GPU.
// Raw per-run measurements are printed as CSV rows on stdout, one row per (cpu_frac, rep).
//
// Build (GPU):   nvcc -O3 -std=c++14 -arch=sm_75 -Xcompiler -fopenmp,-O3,-march=native s03_bench.cu -o s03_bench
// Build (no GPU, pipeline test only):  g++ -O3 -fopenmp -march=native -DNO_CUDA -x c++ s03_bench.cu -o s03_bench_cpu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>

#ifndef NO_CUDA
#include <cuda_runtime.h>
#else
// ---- host-only shim: lets the whole pipeline be tested on a machine without a GPU.
// The "GPU" runs synchronously on the host, so timings from this build say nothing about real hardware.
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
typedef int cudaStream_t;
struct ShimEvent { double t; };
typedef ShimEvent* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
#define cudaHostAllocDefault 0
static const char* cudaGetErrorString(cudaError_t) { return "no error"; }
static cudaError_t cudaMalloc(void** p, size_t n) { *p = malloc(n); return *p ? 0 : 2; }
static cudaError_t cudaFree(void* p) { free(p); return 0; }
static cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) { *p = malloc(n); return *p ? 0 : 2; }
static cudaError_t cudaFreeHost(void* p) { free(p); return 0; }
static cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = 0; return 0; }
static cudaError_t cudaStreamDestroy(cudaStream_t) { return 0; }
static cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new ShimEvent{0.0}; return 0; }
static cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return 0; }
static cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) { e->t = omp_get_wtime() * 1e3; return 0; }
static cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) { *ms = (float)(b->t - a->t); return 0; }
static cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return 0; }
static cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
static cudaError_t cudaDeviceSynchronize() { return 0; }
static cudaError_t cudaGetLastError() { return 0; }
static cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) { memcpy(d, s, n); return 0; }
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ------------------------------------------------------------------ GPU kernel
#ifndef NO_CUDA
// 3x3 box blur on a batch of tiles; blockIdx.z selects the tile. Summation order matches the CPU code.
__global__ void k_blur(const float* __restrict__ in, float* __restrict__ out, int T) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= T || y >= T) return;
    size_t base = (size_t)blockIdx.z * T * T;
    const float* src = in + base;
    float s = 0.f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int xx = min(max(x + dx, 0), T - 1), yy = min(max(y + dy, 0), T - 1);
            s += src[(size_t)yy * T + xx];
        }
    out[base + (size_t)y * T + x] = s / 9.f;
}
#endif

// ------------------------------------------------------------------ CPU kernel (same arithmetic order)
static inline void blur_edge_px(const float* r0, const float* r1, const float* r2, float* o, int T, int x) {
    int xm = x > 0 ? x - 1 : 0, xp = x < T - 1 ? x + 1 : T - 1;
    float s = 0.f;
    s += r0[xm]; s += r0[x]; s += r0[xp];
    s += r1[xm]; s += r1[x]; s += r1[xp];
    s += r2[xm]; s += r2[x]; s += r2[xp];
    o[x] = s / 9.f;
}

static void blur_tile_cpu(const float* __restrict__ in, float* __restrict__ out, int T) {
    for (int y = 0; y < T; y++) {
        int ym = y > 0 ? y - 1 : 0, yp = y < T - 1 ? y + 1 : T - 1;
        const float* r0 = in + (size_t)ym * T;
        const float* r1 = in + (size_t)y * T;
        const float* r2 = in + (size_t)yp * T;
        float* o = out + (size_t)y * T;
        blur_edge_px(r0, r1, r2, o, T, 0);
#pragma omp simd
        for (int x = 1; x < T - 1; x++) {
            float s = 0.f;
            s += r0[x - 1]; s += r0[x]; s += r0[x + 1];
            s += r1[x - 1]; s += r1[x]; s += r1[x + 1];
            s += r2[x - 1]; s += r2[x]; s += r2[x + 1];
            o[x] = s / 9.f;
        }
        blur_edge_px(r0, r1, r2, o, T, T - 1);
    }
}

// P passes, ping-pong between `out` and `tmp` so that the final pass lands in `out`.
static void cpu_tile(const float* in, float* out, float* tmp, int T, int P) {
    const float* src = in;
    for (int i = 0; i < P; i++) {
        float* dst = (((P - 1 - i) & 1) == 0) ? out : tmp;
        blur_tile_cpu(src, dst, T);
        src = dst;
    }
}

// Runs tiles [first, first+count) on the CPU with `th` OpenMP threads; per-thread busy time -> busy[id*8].
static int cpu_process(const float* h_in, float* h_out, int first, int count, int T, int P, int th, double* busy) {
    if (count <= 0) return 0;
    size_t tp = (size_t)T * T;
    int team = 0;
#pragma omp parallel num_threads(th)
    {
        static thread_local std::vector<float> tmp;
        if (tmp.size() < tp) tmp.resize(tp);
        int id = omp_get_thread_num();
        double b = 0.0;
#pragma omp single
        team = omp_get_num_threads();
#pragma omp for schedule(dynamic, 1)
        for (int k = 0; k < count; k++) {
            double t0 = omp_get_wtime();
            size_t off = (size_t)(first + k) * tp;
            cpu_tile(h_in + off, h_out + off, tmp.data(), T, P);
            b += omp_get_wtime() - t0;
        }
        busy[id * 8] = b;
    }
    return team;
}

// ------------------------------------------------------------------ GPU pipeline
struct GpuCtx {
    int S = 1, chunk = 64, maxChunks = 0;
    std::vector<cudaStream_t> st;
    std::vector<float*> dA, dB;
    std::vector<cudaEvent_t> ev;      // 4 events per chunk: before H2D, after H2D, after kernels, after D2H
    cudaEvent_t evStart = nullptr;
    std::vector<cudaEvent_t> evEnd;   // one per stream
};

static void launch_blur(const float* in, float* out, int T, int nt, cudaStream_t st) {
#ifndef NO_CUDA
    dim3 b(16, 16), g((T + 15) / 16, (T + 15) / 16, nt);
    k_blur<<<g, b, 0, st>>>(in, out, T);
#else
    (void)st;
    for (int t = 0; t < nt; t++) blur_tile_cpu(in + (size_t)t * T * T, out + (size_t)t * T * T, T);
#endif
}

static void gpu_setup(GpuCtx& g, int S, int chunk, int maxTiles, size_t tp) {
    g.S = S; g.chunk = chunk;
    g.maxChunks = (maxTiles + chunk - 1) / chunk;
    g.st.resize(S); g.dA.resize(S); g.dB.resize(S); g.evEnd.resize(S);
    size_t bytes = (size_t)chunk * tp * sizeof(float);
    for (int s = 0; s < S; s++) {
        CK(cudaStreamCreate(&g.st[s]));
        CK(cudaMalloc((void**)&g.dA[s], bytes));
        CK(cudaMalloc((void**)&g.dB[s], bytes));
        CK(cudaEventCreate(&g.evEnd[s]));
    }
    g.ev.resize((size_t)4 * g.maxChunks);
    for (auto& e : g.ev) CK(cudaEventCreate(&e));
    CK(cudaEventCreate(&g.evStart));
}

static void gpu_teardown(GpuCtx& g) {
    for (int s = 0; s < g.S; s++) {
        CK(cudaFree(g.dA[s])); CK(cudaFree(g.dB[s]));
        CK(cudaStreamDestroy(g.st[s])); CK(cudaEventDestroy(g.evEnd[s]));
    }
    for (auto& e : g.ev) CK(cudaEventDestroy(e));
    CK(cudaEventDestroy(g.evStart));
}

// Enqueue tiles [first, first+count) onto the GPU streams (returns immediately; work runs asynchronously).
static int gpu_enqueue(GpuCtx& g, const float* h_in, float* h_out, int first, int count, int T, int P) {
    size_t tp = (size_t)T * T;
    int nch = (count + g.chunk - 1) / g.chunk;
    CK(cudaEventRecord(g.evStart, g.st[0]));
    for (int s = 1; s < g.S; s++) CK(cudaStreamWaitEvent(g.st[s], g.evStart, 0));
    for (int c = 0; c < nch; c++) {
        int s = c % g.S;
        int t0 = first + c * g.chunk;
        int nt = std::min(g.chunk, first + count - t0);
        size_t off = (size_t)t0 * tp, bytes = (size_t)nt * tp * sizeof(float);
        cudaEvent_t* e = &g.ev[(size_t)4 * c];
        CK(cudaEventRecord(e[0], g.st[s]));
        CK(cudaMemcpyAsync(g.dA[s], h_in + off, bytes, cudaMemcpyHostToDevice, g.st[s]));
        CK(cudaEventRecord(e[1], g.st[s]));
        float *a = g.dA[s], *b = g.dB[s];
        for (int p = 0; p < P; p++) { launch_blur(a, b, T, nt, g.st[s]); std::swap(a, b); }
        CK(cudaGetLastError());
        CK(cudaEventRecord(e[2], g.st[s]));
        CK(cudaMemcpyAsync(h_out + off, a, bytes, cudaMemcpyDeviceToHost, g.st[s]));   // `a` holds the latest output
        CK(cudaEventRecord(e[3], g.st[s]));
    }
    for (int s = 0; s < g.S; s++) CK(cudaEventRecord(g.evEnd[s], g.st[s]));
    return nch;
}

struct GpuTimes { double h2d = 0, kern = 0, d2h = 0, span = 0; };

static GpuTimes gpu_collect(GpuCtx& g, int nch) {
    GpuTimes r;
    float ms;
    for (int c = 0; c < nch; c++) {
        cudaEvent_t* e = &g.ev[(size_t)4 * c];
        CK(cudaEventElapsedTime(&ms, e[0], e[1])); r.h2d += ms;
        CK(cudaEventElapsedTime(&ms, e[1], e[2])); r.kern += ms;
        CK(cudaEventElapsedTime(&ms, e[2], e[3])); r.d2h += ms;
    }
    for (int s = 0; s < g.S; s++) {
        CK(cudaEventElapsedTime(&ms, g.evStart, g.evEnd[s]));
        r.span = std::max(r.span, (double)ms);
    }
    return r;
}

// ------------------------------------------------------------------ helpers
static void init_input(float* p, size_t n) {
#pragma omp parallel for
    for (long long i = 0; i < (long long)n; i++) {
        uint32_t x = (uint32_t)i * 2654435761u + 12345u;
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
        p[i] = (x >> 8) / 16777216.0f;
    }
}

static double max_rel_err(const float* a, const float* ref, size_t n) {
    double mx = 0.0, mr = 1e-6;
#pragma omp parallel for reduction(max : mx, mr)
    for (long long i = 0; i < (long long)n; i++) {
        double d = std::fabs((double)a[i] - (double)ref[i]);
        double r = std::fabs((double)ref[i]);
        if (d > mx) mx = d;
        if (r > mr) mr = r;
    }
    return mx / mr;
}

static std::string get_arg(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

static std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) v.push_back(atof(s.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

struct RunResult {
    int nC = 0, nG = 0, team = 0;
    double makespan = 0, sched = 0, cpu_wall = 0, cpu_busy = 0, sync_wait = 0;
    GpuTimes g;
};

// One timed execution of the whole workload with CPU share f_cpu.
static RunResult run_once(GpuCtx& g, const float* h_in, float* h_out, int N, int T, int P, int th, double f_cpu) {
    RunResult r;
    r.nC = (int)llround(f_cpu * N);
    r.nC = std::max(0, std::min(N, r.nC));
    r.nG = N - r.nC;                       // GPU takes tiles [0, nG), CPU takes [nG, N)
    std::vector<double> busy((size_t)(th + 1) * 8, 0.0);
    CK(cudaDeviceSynchronize());           // start from an idle GPU (untimed)

    double t0 = omp_get_wtime();
    int nch = 0;
    if (r.nG > 0) nch = gpu_enqueue(g, h_in, h_out, 0, r.nG, T, P);
    double t_enq = omp_get_wtime();
    if (r.nC > 0) r.team = cpu_process(h_in, h_out, r.nG, r.nC, T, P, th, busy.data());
    double t_cpu = omp_get_wtime();
    if (r.nG > 0) for (int s = 0; s < g.S; s++) CK(cudaStreamSynchronize(g.st[s]));
    double t1 = omp_get_wtime();

    r.makespan = (t1 - t0) * 1e3;
    r.sched = (t_enq - t0) * 1e3;
    r.cpu_wall = r.nC > 0 ? (t_cpu - t_enq) * 1e3 : 0.0;
    r.sync_wait = r.nG > 0 ? (t1 - t_cpu) * 1e3 : 0.0;
    for (int i = 0; i < r.team && i < th; i++) r.cpu_busy += busy[i * 8] * 1e3;
    if (r.nG > 0) r.g = gpu_collect(g, nch);
    return r;
}

int main(int argc, char** argv) {
    int N = atoi(get_arg(argc, argv, "--tiles", "256").c_str());
    int T = atoi(get_arg(argc, argv, "--tile", "256").c_str());
    int P = atoi(get_arg(argc, argv, "--passes", "1").c_str());
    int th = atoi(get_arg(argc, argv, "--threads", "4").c_str());
    int S = atoi(get_arg(argc, argv, "--streams", "1").c_str());
    int chunk = atoi(get_arg(argc, argv, "--chunk", "64").c_str());
    int pinned = atoi(get_arg(argc, argv, "--pinned", "1").c_str());
    int reps = atoi(get_arg(argc, argv, "--reps", "5").c_str());
    int header = atoi(get_arg(argc, argv, "--header", "1").c_str());
    std::string workload = get_arg(argc, argv, "--workload", "tileblur");
    std::vector<double> fracs = parse_list(get_arg(argc, argv, "--fracs", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"));
    if (N < 1 || T < 4 || P < 1 || th < 1 || S < 1 || chunk < 1 || reps < 1 || fracs.empty()) {
        fprintf(stderr, "bad arguments\n");
        return 1;
    }
    omp_set_dynamic(0);
    CK(cudaFree(0));                       // force CUDA context creation before any timing

    size_t tp = (size_t)T * T, n_el = (size_t)N * tp, bytes = n_el * sizeof(float);
    float *h_in = nullptr, *h_out = nullptr;
    if (pinned) { CK(cudaHostAlloc((void**)&h_in, bytes, cudaHostAllocDefault)); CK(cudaHostAlloc((void**)&h_out, bytes, cudaHostAllocDefault)); }
    else { h_in = (float*)malloc(bytes); h_out = (float*)malloc(bytes); }
    std::vector<float> ref(n_el);
    if (!h_in || !h_out) { fprintf(stderr, "host allocation failed\n"); return 1; }
    init_input(h_in, n_el);

    int maxth = omp_get_max_threads();
    std::vector<double> busy0((size_t)(std::max(1, maxth) + 1) * 8, 0.0);
    cpu_process(h_in, ref.data(), 0, N, T, P, std::max(1, maxth), busy0.data());   // CPU reference result

    GpuCtx g;
    gpu_setup(g, S, chunk, N, tp);
    double dev_bytes = (double)S * 2 * chunk * tp * sizeof(float);

    if (header)
        printf("workload,tiles,tile_dim,passes,threads_req,threads_actual,streams,chunk_tiles,pinned,cpu_frac,tiles_cpu,tiles_gpu,rep,"
               "makespan_ms,sched_ms,cpu_wall_ms,cpu_busy_sum_ms,gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,gpu_span_ms,sync_wait_ms,"
               "max_rel_err,host_bytes,device_bytes\n");

    // warm-up: GPU clocks, OpenMP thread pool, page faults
    const double warm[3] = {0.0, 0.5, 1.0};
    for (double w : warm) run_once(g, h_in, h_out, N, T, P, th, w);

    for (int rep = 0; rep < reps; rep++) {
        std::vector<double> order = fracs;
        if (rep & 1) std::reverse(order.begin(), order.end());   // alternate sweep direction to spread drift
        for (double f : order) {
            bool check = (rep == 0);
            if (check) memset(h_out, 0, bytes);
            RunResult r = run_once(g, h_in, h_out, N, T, P, th, f);
            char errbuf[32];
            if (check) snprintf(errbuf, sizeof errbuf, "%.3e", max_rel_err(h_out, ref.data(), n_el));
            else snprintf(errbuf, sizeof errbuf, "nan");
            printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%.0f,%.0f\n",
                   workload.c_str(), N, T, P, th, r.team, S, chunk, pinned, f, r.nC, r.nG, rep + 1,
                   r.makespan, r.sched, r.cpu_wall, r.cpu_busy, r.g.h2d, r.g.kern, r.g.d2h, r.g.span, r.sync_wait,
                   errbuf, 2.0 * bytes, dev_bytes);
            fflush(stdout);
        }
        fprintf(stderr, "[%s N=%d P=%d th=%d] rep %d/%d done\n", workload.c_str(), N, P, th, rep + 1, reps);
    }

    gpu_teardown(g);
    if (pinned) { CK(cudaFreeHost(h_in)); CK(cudaFreeHost(h_out)); } else { free(h_in); free(h_out); }
    return 0;
}
