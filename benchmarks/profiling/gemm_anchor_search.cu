// Anchor search for Riley's anchored cuBLASLt GEMMs (A100, BF16).
//
// usage: gemm_anchor_search K N [top]
//
// Riley binds every row count of a projection to one reduction: the anchor,
// cuBLASLt's first heuristic at 128 rows unless kAnchorOverrides in
// kernels/src/gemm.cu names another, plus table entries that reproduce the
// anchor's bytes faster at given row counts. This program enumerates every
// configuration Riley's recording contract accepts (split-K 1, or in-place and
// output-type split-K; 256-byte alignment; 16 MiB workspace) and:
//
// 1. times all of them at 256 rows (the prompt step) and 512;
// 2. takes the fastest `top` that are row-count invariant (the rows of a
//    256-row product equal those of 1-, 7-, 64-, 128- and 512-row products)
//    and groups them by output bytes;
// 3. for each group and for the current anchor, finds per row count the
//    fastest configuration with the group's bytes.
//
// Plan layouts match RileyCudaGemmPlan: TN over column-major W[K,N], X[K,M],
// Y[N,M] with FP32 compute and BF16 data.
//
// The byte classes are exact; the times are not. On the random probe data the
// Qwen2.5-7B gate|up anchor took 443 us at 256 rows against 288 us in the
// server, which reversed its order with another tiling. Profile a candidate in
// the server before adding it to the table.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#define CHECK_CUDA(x)                                                      \
  do {                                                                     \
    cudaError_t err_ = (x);                                                   \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(err_));         \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)
#define CHECK_LT(x)                                                        \
  do {                                                                     \
    cublasStatus_t s = (x);                                                \
    if (s != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "%s: %d\n", #x, static_cast<int>(s));           \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

constexpr size_t kWorkspace = 16u << 20;
constexpr uint32_t kAlignment = 256;
constexpr uint64_t kMaxRows = 512;

struct Config {
  int32_t id;
  uint32_t tile, stages, split_k, reduction;
  std::string name() const {
    char b[96];
    std::snprintf(b, sizeof b, "{%d, %u, %u, %u, %u}", id, tile, stages,
                  split_k, reduction);
    return b;
  }
  bool operator<(const Config& o) const {
    return std::tie(id, tile, stages, split_k, reduction) <
           std::tie(o.id, o.tile, o.stages, o.split_k, o.reduction);
  }
  bool operator==(const Config& o) const {
    return !(*this < o) && !(o < *this);
  }
};

cublasLtHandle_t handle;
cublasLtMatmulDesc_t operation;
uint64_t K, N;
void *weight, *input, *output, *workspace;

struct Layouts {
  cublasLtMatrixLayout_t w, x, y;
};
std::map<uint64_t, Layouts> layout_cache;

Layouts layouts(uint64_t m) {
  auto it = layout_cache.find(m);
  if (it != layout_cache.end()) return it->second;
  Layouts l{};
  CHECK_LT(cublasLtMatrixLayoutCreate(&l.w, CUDA_R_16BF, K, N, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&l.x, CUDA_R_16BF, K, m, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&l.y, CUDA_R_16BF, N, m, N));
  layout_cache[m] = l;
  return l;
}

bool make_algo(const Config& c, cublasLtMatmulAlgo_t* a) {
  if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                             CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                             CUDA_R_16BF, c.id, a) != CUBLAS_STATUS_SUCCESS)
    return false;
  uint32_t swizzle = 0, custom = 0;
  return cublasLtMatmulAlgoConfigSetAttribute(a, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                              &c.tile, sizeof c.tile) == 0 &&
         cublasLtMatmulAlgoConfigSetAttribute(
             a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &c.stages, sizeof c.stages) == 0 &&
         cublasLtMatmulAlgoConfigSetAttribute(
             a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &c.split_k, sizeof c.split_k) == 0 &&
         cublasLtMatmulAlgoConfigSetAttribute(
             a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &c.reduction,
             sizeof c.reduction) == 0 &&
         cublasLtMatmulAlgoConfigSetAttribute(
             a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof swizzle) == 0 &&
         cublasLtMatmulAlgoConfigSetAttribute(
             a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom) == 0;
}

bool aligned(const cublasLtMatmulAlgo_t* a) {
  for (auto cap : {CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
                   CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
                   CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
                   CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES}) {
    uint32_t v = 0;
    size_t w = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(a, cap, &v, sizeof v, &w) != 0)
      return false;
    if (v != 0 && kAlignment % v != 0) return false;
  }
  return true;
}

bool valid(const Config& c, uint64_t m, cublasLtMatmulAlgo_t* a) {
  if (!make_algo(c, a) || !aligned(a)) return false;
  Layouts l = layouts(m);
  cublasLtMatmulHeuristicResult_t r{};
  return cublasLtMatmulAlgoCheck(handle, operation, l.w, l.x, l.y, l.y, a,
                                 &r) == CUBLAS_STATUS_SUCCESS &&
         r.state == CUBLAS_STATUS_SUCCESS && r.workspaceSize <= kWorkspace;
}

bool run(const cublasLtMatmulAlgo_t& a, uint64_t m) {
  const float alpha = 1.0F, beta = 0.0F;
  Layouts l = layouts(m);
  return cublasLtMatmul(handle, operation, &alpha, weight, l.w, input, l.x,
                        &beta, output, l.y, output, l.y, &a, workspace,
                        kWorkspace, nullptr) == CUBLAS_STATUS_SUCCESS;
}

// Mean microseconds of `reps` launches after warm-up; negative on failure.
double time_us(const Config& c, uint64_t m, int reps) {
  cublasLtMatmulAlgo_t a{};
  if (!valid(c, m, &a)) return -1;
  for (int i = 0; i < 3; ++i)
    if (!run(a, m)) {
      (void)cudaGetLastError();
      return -1;
    }
  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for (int i = 0; i < reps; ++i) run(a, m);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 1000.0 * ms / reps;
}

bool bits(const Config& c, uint64_t m, std::vector<uint16_t>* out) {
  cublasLtMatmulAlgo_t a{};
  if (!valid(c, m, &a)) return false;
  CHECK_CUDA(cudaMemset(output, 0xff, N * m * 2));
  if (!run(a, m) || cudaDeviceSynchronize() != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  out->resize(N * m);
  CHECK_CUDA(cudaMemcpy(out->data(), output, N * m * 2, cudaMemcpyDeviceToHost));
  return true;
}

std::vector<Config> enumerate() {
  std::vector<Config> all;
  int ids[256];
  int count = 0;
  CHECK_LT(cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                    CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                                    CUDA_R_16BF, 256, ids, &count));
  for (int i = 0; i < count; ++i) {
    cublasLtMatmulAlgo_t a{};
    if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                               CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, ids[i], &a) != 0)
      continue;
    size_t bytes = 0;
    std::vector<uint32_t> tiles, stages;
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr,
                                      0, &bytes);
    tiles.resize(bytes / 4);
    if (bytes)
      cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS,
                                        tiles.data(), bytes, &bytes);
    if (tiles.empty()) tiles.push_back(0);
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                      nullptr, 0, &bytes);
    stages.resize(bytes / 4);
    if (bytes)
      cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                        stages.data(), bytes, &bytes);
    if (stages.empty()) stages.push_back(0);
    int32_t split_support = 0;
    uint32_t reductions = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                      &split_support, sizeof split_support,
                                      &bytes);
    cublasLtMatmulAlgoCapGetAttribute(
        &a, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reductions,
        sizeof reductions, &bytes);
    for (uint32_t t : tiles)
      for (uint32_t s : stages) {
        all.push_back({ids[i], t, s, 1, CUBLASLT_REDUCTION_SCHEME_NONE});
        if (!split_support) continue;
        for (uint32_t k : {2u, 3u, 4u, 6u, 8u})
          for (uint32_t r : {uint32_t(CUBLASLT_REDUCTION_SCHEME_INPLACE),
                             uint32_t(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE)})
            if (reductions & r) all.push_back({ids[i], t, s, k, r});
      }
  }
  return all;
}

// Riley's anchor: first heuristic at 128 rows, kept when its reduction is
// allowed (none, in-place or output-type split-K), else forced to split 1.
Config riley_anchor() {
  Layouts l = layouts(128);
  cublasLtMatmulPreference_t pref;
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = kWorkspace;
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t r[16];
  int n = 0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, operation, l.w, l.x, l.y,
                                          l.y, pref, 16, r, &n));
  for (int i = 0; i < n; ++i) {
    Config c{};
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_ID,
                                         &c.id, sizeof c.id, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo,
                                         CUBLASLT_ALGO_CONFIG_TILE_ID, &c.tile,
                                         sizeof c.tile, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo,
                                         CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &c.stages, sizeof c.stages, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo,
                                         CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &c.split_k, sizeof c.split_k, &w);
    cublasLtMatmulAlgoConfigGetAttribute(
        &r[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &c.reduction,
        sizeof c.reduction, &w);
    if (c.split_k > 1 && c.reduction != CUBLASLT_REDUCTION_SCHEME_INPLACE &&
        c.reduction != CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE) {
      c.split_k = 1;
      c.reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    }
    cublasLtMatmulAlgo_t a{};
    if (valid(c, 128, &a)) return c;
  }
  std::fprintf(stderr, "no anchor\n");
  std::exit(1);
}

void fill(void* device, size_t count, uint64_t seed, uint32_t base,
          uint32_t span) {
  std::vector<uint16_t> host(count);
  uint64_t state = seed;
  for (auto& v : host) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const uint32_t b = static_cast<uint32_t>(state >> 24);
    v = static_cast<uint16_t>(((b >> 31) << 15) |
                              ((base + (b >> 8) % span) << 7) | (b & 0x7f));
  }
  CHECK_CUDA(cudaMemcpy(device, host.data(), count * 2, cudaMemcpyHostToDevice));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s K N [top]\n", argv[0]);
    return 2;
  }
  K = std::strtoull(argv[1], nullptr, 10);
  N = std::strtoull(argv[2], nullptr, 10);
  const size_t top = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 24;
  CHECK_LT(cublasLtCreate(&handle));
  CHECK_LT(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CHECK_LT(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  CHECK_CUDA(cudaMalloc(&weight, K * N * 2));
  CHECK_CUDA(cudaMalloc(&input, K * kMaxRows * 2));
  CHECK_CUDA(cudaMalloc(&output, N * kMaxRows * 2));
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspace));
  fill(weight, K * N, 0x9e3779b97f4a7c15ULL ^ (K * 1315423911ULL + N), 0x76, 5);
  fill(input, K * kMaxRows, 0x51ed270b2c8f1d3bULL ^ K, 0x7c, 6);

  const Config anchor = riley_anchor();
  // gemm_anchor_search K N time M[,M...] ID,TILE,STAGES,SPLIT,RED ...:
  // interleaved 200-launch timings of named configurations, five rounds,
  // best round reported, plus whether each matches the first one's bytes.
  if (argc > 4 && std::string(argv[3]) == "time") {
    std::vector<uint64_t> rows;
    for (char* p = argv[4]; *p;) {
      rows.push_back(std::strtoull(p, &p, 10));
      if (*p == ',') ++p;
    }
    std::vector<Config> named;
    for (int i = 5; i < argc; ++i) {
      Config c{};
      if (std::sscanf(argv[i], "%d,%u,%u,%u,%u", &c.id, &c.tile, &c.stages,
                      &c.split_k, &c.reduction) == 5)
        named.push_back(c);
    }
    for (int warm = 0; warm < 3; ++warm) time_us(named[0], 256, 200);
    for (uint64_t m : rows) {
      std::vector<double> best(named.size(), 1e30);
      for (int round = 0; round < 5; ++round)
        for (size_t i = 0; i < named.size(); ++i) {
          const double t = time_us(named[i], m, 200);
          if (t > 0) best[i] = std::min(best[i], t);
        }
      std::vector<uint16_t> first, got;
      const bool have_first = bits(named[0], m, &first);
      for (size_t i = 0; i < named.size(); ++i) {
        const bool same = have_first && bits(named[i], m, &got) && got == first;
        std::printf("m=%-4llu %-24s %8.1f us %s\n", (unsigned long long)m,
                    named[i].name().c_str(), best[i] > 1e29 ? -1.0 : best[i],
                    i == 0 ? "" : (same ? "same-bytes" : "DIFFERENT"));
      }
    }
    return 0;
  }
  const std::vector<Config> all = enumerate();
  std::printf("shape %llu -> %llu: %zu configurations, anchor %s\n",
              (unsigned long long)K, (unsigned long long)N, all.size(),
              anchor.name().c_str());

  // Times of every valid configuration per row count, fastest first.
  std::map<uint64_t, std::vector<std::pair<double, Config>>> timed;
  auto timings = [&](uint64_t m) -> std::vector<std::pair<double, Config>>& {
    auto it = timed.find(m);
    if (it != timed.end()) return it->second;
    std::vector<std::pair<double, Config>> v;
    const int reps = m >= 128 ? 10 : 20;
    for (const Config& c : all) {
      const double t = time_us(c, m, reps);
      if (t > 0) v.push_back({t, c});
    }
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.first < b.first; });
    // Re-time the leaders to cut noise from the single pass.
    for (size_t i = 0; i < std::min<size_t>(v.size(), 64); ++i)
      v[i].first = std::min(v[i].first, time_us(v[i].second, m, reps * 2));
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.first < b.first; });
    return timed[m] = v;
  };
  auto time_of = [&](const Config& c, uint64_t m) {
    for (auto& [t, x] : timings(m))
      if (x == c) return t;
    return -1.0;
  };

  auto& at256 = timings(256);
  std::printf("256 rows: %zu valid; anchor %.1f us; fastest:\n", at256.size(),
              time_of(anchor, 256));
  for (size_t i = 0; i < std::min<size_t>(at256.size(), 12); ++i)
    std::printf("  %8.1f us  %s  (512 rows %.1f us)\n", at256[i].first,
                at256[i].second.name().c_str(),
                time_of(at256[i].second, 512));

  // Row-count invariance and byte classes of the fastest configurations.
  const uint64_t probe_rows[] = {1, 7, 64, 128, 512};
  std::vector<Config> reps;
  std::vector<std::vector<uint16_t>> rep_bits;
  std::vector<Config> candidates = {anchor};
  for (size_t i = 0; i < at256.size() && candidates.size() < top + 1; ++i)
    if (!(at256[i].second == anchor)) candidates.push_back(at256[i].second);
  for (const Config& c : candidates) {
    std::vector<uint16_t> full;
    if (!bits(c, 256, &full)) continue;
    bool invariant = true;
    for (uint64_t m : probe_rows) {
      std::vector<uint16_t> part;
      if (!bits(c, m, &part)) {
        invariant = false;
        break;
      }
      const uint64_t rows = std::min<uint64_t>(m, 256);
      if (!std::equal(part.begin(), part.begin() + N * rows, full.begin())) {
        invariant = false;
        break;
      }
    }
    if (!invariant) {
      std::printf("  not row invariant: %s\n", c.name().c_str());
      continue;
    }
    bool seen = false;
    for (auto& b : rep_bits) seen = seen || b == full;
    if (seen) continue;
    reps.push_back(c);
    rep_bits.push_back(full);
  }

  const uint64_t grid[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 512};
  for (size_t r = 0; r < reps.size(); ++r) {
    std::printf("class %zu rep %s%s\n", r, reps[r].name().c_str(),
                reps[r] == anchor ? " (current anchor)" : "");
    double weighted = 0;
    for (uint64_t m : grid) {
      std::vector<uint16_t> want;
      if (!bits(reps[r], m, &want)) {
        std::printf("  m=%-4llu rep invalid\n", (unsigned long long)m);
        continue;
      }
      auto& list = timings(m);
      double best = -1;
      Config pick{};
      size_t checked = 0;
      for (auto& [t, c] : list) {
        if (++checked > 400) break;
        std::vector<uint16_t> got;
        if (bits(c, m, &got) && got == want) {
          best = t;
          pick = c;
          break;
        }
      }
      std::printf("  m=%-4llu best-equivalent %8.1f us %s  (unconstrained %.1f us)\n",
                  (unsigned long long)m, best, best > 0 ? pick.name().c_str() : "-",
                  list.empty() ? -1.0 : list[0].first);
    }
    (void)weighted;
  }
  return 0;
}
