// Replay the observed Riley cuBLASLt plan for two captured BF16 input rows.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr uint64_t kHidden = 3584;
constexpr uint64_t kOutput = 4608;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

void check_lt(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLASLt status %d\n", operation, int(status));
        std::exit(1);
    }
}

void require(bool pass, const char* detail) {
    if (!pass) {
        std::fprintf(stderr, "%s\n", detail);
        std::exit(1);
    }
}

std::vector<uint16_t> read_words(const std::string& path, size_t count) {
    std::vector<uint16_t> data(count);
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint16_t))),
            "missing or truncated BF16 operand");
    require(input.peek() == std::char_traits<char>::eof(), "unexpected bytes after BF16 operand");
    return data;
}

void set_algorithm(cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoConfigAttributes_t attribute, uint32_t value) {
    check_lt(cublasLtMatmulAlgoConfigSetAttribute(&algorithm, attribute, &value, sizeof(value)),
             "set recorded algorithm attribute");
}
}  // namespace

int main(int argc, char** argv) {
    require(argc == 2, "usage: replay_riley_lt OPERAND_DIRECTORY");
    const std::string root = argv[1];
    int device_count = 0, version = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "visible device count");
    check_cuda(cudaRuntimeGetVersion(&version), "runtime version");
    cudaDeviceProp device{};
    check_cuda(cudaGetDeviceProperties(&device, 0), "device geometry");
    require(device_count == 1 && device.major == 8 && device.minor == 0, "expected one SM80 device");
    require(version == 12040 && cublasLtGetVersion() == 120405, "loaded CUDA/cuBLASLt versions differ from the Riley receipt");

    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weights{}, input{}, output{};
    cublasLtMatmulAlgo_t algorithm{};
    check_lt(cublasLtCreate(&handle), "create cuBLASLt");
    check_lt(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create FP32 compute descriptor");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transpose weights");
    check_lt(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "input layout");
    check_lt(cublasLtMatrixLayoutCreate(&weights, CUDA_R_16BF, kHidden, kOutput, kHidden), "weights layout");
    check_lt(cublasLtMatrixLayoutCreate(&input, CUDA_R_16BF, kHidden, 1, kHidden), "input layout");
    check_lt(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, kOutput, 1, kOutput), "output layout");
    check_lt(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF,
                                  CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 21, &algorithm), "recorded plan ID");
    set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, 15);
    set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, 18);
    set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
    set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0);
    set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0);
    set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0);
    cublasLtMatmulHeuristicResult_t checked{};
    check_lt(cublasLtMatmulAlgoCheck(handle, operation, weights, input, output, output, &algorithm, &checked),
             "check recorded plan");
    require(checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == 0, "recorded plan workspace differs");

    void *w = nullptr, *x = nullptr, *y = nullptr;
    check_cuda(cudaMalloc(&w, kHidden * kOutput * 2), "allocate weights");
    check_cuda(cudaMalloc(&x, kHidden * 2), "allocate input");
    check_cuda(cudaMalloc(&y, kOutput * 2), "allocate output");
    const auto host_weights = read_words(root + "/weights.bf16", kHidden * kOutput);
    check_cuda(cudaMemcpy(w, host_weights.data(), kHidden * kOutput * 2, cudaMemcpyHostToDevice), "copy weights");
    const float alpha = 1.0F, beta = 0.0F;
    for (const char* name : {"669-9", "1023-67"}) {
        const auto host_input = read_words(root + "/" + name + ".input.bf16", kHidden);
        check_cuda(cudaMemcpy(x, host_input.data(), kHidden * 2, cudaMemcpyHostToDevice), "copy captured input");
        check_lt(cublasLtMatmul(handle, operation, &alpha, w, weights, x, input, &beta,
                               y, output, y, output, &algorithm, nullptr, 0, nullptr), "execute recorded Riley plan");
        std::vector<uint16_t> host_output(kOutput);
        check_cuda(cudaMemcpy(host_output.data(), y, kOutput * 2, cudaMemcpyDeviceToHost), "copy BF16 result");
        std::ofstream file(root + "/" + name + ".native-lt.bf16", std::ios::binary);
        require(bool(file.write(reinterpret_cast<const char*>(host_output.data()), kOutput * 2)), "write BF16 result");
    }
    check_cuda(cudaFree(w), "release weights");
    check_cuda(cudaFree(x), "release input");
    check_cuda(cudaFree(y), "release output");
    check_lt(cublasLtMatrixLayoutDestroy(weights), "close weight layout");
    check_lt(cublasLtMatrixLayoutDestroy(input), "close input layout");
    check_lt(cublasLtMatrixLayoutDestroy(output), "close output layout");
    check_lt(cublasLtMatmulDescDestroy(operation), "close operation");
    check_lt(cublasLtDestroy(handle), "close cuBLASLt");
}
