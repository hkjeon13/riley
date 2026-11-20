// Reproduce the measured layer-zero plan, then compare alternative reduction plans.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        std::exit(1);
    }
}

void check_lt(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLASLt status %d\n", message, int(status));
        std::exit(1);
    }
}

using Words = std::vector<uint16_t>;
Words read(const std::string& path, size_t count) {
    Words data(count);
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint16_t))),
            "missing or truncated BF16 operand");
    require(input.peek() == std::char_traits<char>::eof(), "unexpected BF16 trailing data");
    return data;
}

void write(const std::string& path, const Words& data) {
    std::ofstream output(path, std::ios::binary);
    require(bool(output.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint16_t))),
            "could not write BF16 replay");
}

constexpr cublasLtMatmulAlgoConfigAttributes_t kFields[] = {
    CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
};
using Signature = std::array<uint32_t, 7>;

Signature signature(const cublasLtMatmulAlgo_t& algorithm) {
    Signature values{};
    for (size_t index = 0; index < values.size(); ++index) {
        size_t bytes = 0;
        check_lt(cublasLtMatmulAlgoConfigGetAttribute(
                     &algorithm, kFields[index], &values[index], sizeof(uint32_t), &bytes),
                 "read algorithm attribute");
        require(bytes == sizeof(uint32_t), "unexpected algorithm attribute width");
    }
    return values;
}

uint64_t numerical_flags(const cublasLtMatmulAlgo_t& algorithm) {
    uint64_t flags = 0;
    size_t bytes = 0;
    check_lt(cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                              &flags, sizeof(flags), &bytes), "read numerical flags");
    require(bytes == sizeof(flags), "unexpected numerical flags width");
    return flags;
}
}  // namespace

int main(int argc, char** argv) {
    require(argc == 13, "usage: replay ROOT N K WS ID TILE STAGES SPLIT REDUCTION SWIZZLE CUSTOM FLAGS");
    const std::string root = argv[1];
    const size_t n = std::stoull(argv[2]), k = std::stoull(argv[3]);
    const size_t recorded_workspace = std::stoull(argv[4]);
    require(n == 3584 && (k == 3584 || k == 18944), "unexpected projection shape");
    Signature original{};
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = uint32_t(std::stoul(argv[5 + i]));
    }
    const auto flags = std::stoull(argv[12]);

    int visible = 0, version = 0;
    cudaDeviceProp device{};
    check_cuda(cudaGetDeviceCount(&visible), "visible devices");
    check_cuda(cudaGetDeviceProperties(&device, 0), "device properties");
    check_cuda(cudaRuntimeGetVersion(&version), "CUDA version");
    require(visible == 1 && device.major == 8 && device.minor == 0
            && version == 12040 && cublasLtGetVersion() == 120405, "runtime differs from recorded plan");
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weights{}, input{}, output{};
    check_lt(cublasLtCreate(&handle), "create cuBLASLt");
    check_lt(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "FP32 compute descriptor");
    const cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
             "transpose stored weights");
    check_lt(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)),
             "set input layout");
    check_lt(cublasLtMatrixLayoutCreate(&weights, CUDA_R_16BF, k, n, k), "weight layout");
    check_lt(cublasLtMatrixLayoutCreate(&input, CUDA_R_16BF, k, 1, k), "input layout");
    check_lt(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, n, 1, n), "output layout");

    cublasLtMatmulAlgo_t recorded{};
    check_lt(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF,
                                  CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, original[0], &recorded),
             "initialize recorded plan");
    for (size_t i = 1; i < original.size(); ++i) {
        check_lt(cublasLtMatmulAlgoConfigSetAttribute(&recorded, kFields[i], &original[i], sizeof(original[i])),
                 "configure recorded plan");
    }
    require(signature(recorded) == original && numerical_flags(recorded) == flags, "recorded plan differs");
    cublasLtMatmulHeuristicResult_t check{};
    check_lt(cublasLtMatmulAlgoCheck(handle, operation, weights, input, output, output, &recorded, &check),
             "verify recorded plan");
    require(check.state == CUBLAS_STATUS_SUCCESS && check.workspaceSize == recorded_workspace,
            "recorded workspace differs");

    struct Plan {
        std::string name;
        cublasLtMatmulAlgo_t algorithm;
        size_t workspace;
    };
    std::vector<Plan> plans{{"recorded", recorded, recorded_workspace}};
    std::set<Signature> seen{original};
    constexpr size_t kCandidateCap = 16 * 1024 * 1024;
    if (original[3] > 1 && original[4] == CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE) {
        auto candidate = recorded;
        const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        cublasLtMatmulHeuristicResult_t alternative{};
        const auto configured = cublasLtMatmulAlgoConfigSetAttribute(
            &candidate, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction));
        const auto verified = configured == CUBLAS_STATUS_SUCCESS
            ? cublasLtMatmulAlgoCheck(handle, operation, weights, input, output, output, &candidate, &alternative)
            : configured;
        if (verified == CUBLAS_STATUS_SUCCESS && alternative.state == CUBLAS_STATUS_SUCCESS
            && alternative.workspaceSize <= kCandidateCap) {
            auto expected = original;
            expected[4] = reduction;
            require(signature(candidate) == expected && numerical_flags(candidate) == flags,
                    "isolated reduction control changed another attribute");
            plans.push_back({"fixed-reduction", candidate, alternative.workspaceSize});
            seen.insert(expected);
        } else {
            std::fprintf(stderr, "same-plan FP32 reduction unavailable: %d, plan status %d\n",
                         int(verified), int(alternative.state));
        }
    }
    cublasLtMatmulPreference_t pref{};
    check_lt(cublasLtMatmulPreferenceCreate(&pref), "create workspace preference");
    check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &kCandidateCap, sizeof(kCandidateCap)), "set candidate workspace");
    constexpr int kMax = 32;
    cublasLtMatmulHeuristicResult_t list[kMax]{};
    int returned = 0;
    check_lt(cublasLtMatmulAlgoGetHeuristic(handle, operation, weights, input, output, output,
                                           pref, kMax, list, &returned), "enumerate projection plans");
    require(returned >= 0 && returned <= kMax, "invalid heuristic count");
    std::printf("heuristics %d\n", returned);
    for (int i = 0; i < returned; ++i) {
        const auto& item = list[i];
        if (item.state != CUBLAS_STATUS_SUCCESS || item.workspaceSize > kCandidateCap) {
            continue;
        }
        const auto code = signature(item.algo);
        const bool unsplit = code[3] <= 1 && code[4] == CUBLASLT_REDUCTION_SCHEME_NONE;
        const bool compute_type = code[3] > 1 && code[4] == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        if ((unsplit || compute_type) && seen.insert(code).second) {
            plans.push_back({"heuristic-" + std::to_string(i), item.algo, item.workspaceSize});
        }
    }

    void *w = nullptr, *x = nullptr, *y = nullptr, *workspace = nullptr;
    check_cuda(cudaMalloc(&w, n * k * 2), "allocate weights");
    check_cuda(cudaMalloc(&x, k * 2), "allocate input");
    check_cuda(cudaMalloc(&y, n * 2), "allocate output");
    check_cuda(cudaMalloc(&workspace, std::max(kCandidateCap, recorded_workspace)), "allocate workspace");
    const auto host_weights = read(root + "/weights.bf16", n * k);
    check_cuda(cudaMemcpy(w, host_weights.data(), n * k * 2, cudaMemcpyHostToDevice), "upload weights");
    const float alpha = 1, beta = 0;
    for (const auto& plan : plans) {
        std::printf("%s", plan.name.c_str());
        for (const auto field : signature(plan.algorithm)) {
            std::printf(" %u", field);
        }
        std::printf(" %llu %zu\n", static_cast<unsigned long long>(numerical_flags(plan.algorithm)), plan.workspace);
        for (const char* scenario : {"669-9", "1023-67"}) {
            const std::string prefix = root + "/" + scenario;
            const auto host_input = read(prefix + ".input.bf16", k);
            check_cuda(cudaMemcpy(x, host_input.data(), k * 2, cudaMemcpyHostToDevice), "upload recorded input");
            Words first(n), second(n);
            for (auto* destination : {&first, &second}) {
                check_lt(cublasLtMatmul(handle, operation, &alpha, w, weights, x, input, &beta,
                                       y, output, y, output, &plan.algorithm,
                                       plan.workspace ? workspace : nullptr, plan.workspace, nullptr),
                         "replay cuBLASLt projection");
                check_cuda(cudaMemcpy(destination->data(), y, n * 2, cudaMemcpyDeviceToHost), "read BF16 result");
            }
            require(first == second, "projection replay differed between repeats");
            if (plan.name == "recorded") {
                require(first == read(prefix + ".riley.bf16", n), "recorded plan did not reproduce the served output");
            }
            write(prefix + "." + plan.name + ".bf16", first);
        }
    }
    check_cuda(cudaFree(workspace), "release workspace");
    check_cuda(cudaFree(y), "release output");
    check_cuda(cudaFree(x), "release input");
    check_cuda(cudaFree(w), "release weights");
    check_lt(cublasLtMatmulPreferenceDestroy(pref), "close preference");
    check_lt(cublasLtMatrixLayoutDestroy(output), "close output layout");
    check_lt(cublasLtMatrixLayoutDestroy(input), "close input layout");
    check_lt(cublasLtMatrixLayoutDestroy(weights), "close weight layout");
    check_lt(cublasLtMatmulDescDestroy(operation), "close operation");
    check_lt(cublasLtDestroy(handle), "close handle");
}
