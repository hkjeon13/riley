int main(int argc, char** argv) {
    require(argc == 13, "usage: bucket-probe ROOT N K ID TILE STAGES SPLIT REDUCTION SWIZZLE CUSTOM FLAGS LABEL");
    const std::string root = argv[1], label = argv[12];
    const size_t n = std::stoull(argv[2]), k = std::stoull(argv[3]);
    require(n == 3584 && (k == 3584 || k == 18944), "unexpected projection geometry");
    Signature original{};
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = uint32_t(std::stoul(argv[4 + i]));
    }
    const auto flags = std::stoull(argv[11]);
    int devices = 0, version = 0;
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceCount(&devices), "visible devices");
    check_cuda(cudaGetDeviceProperties(&prop, 0), "device properties");
    check_cuda(cudaRuntimeGetVersion(&version), "CUDA version");
    require(devices == 1 && prop.major == 8 && prop.minor == 0
            && version == 12040 && cublasLtGetVersion() == 120405, "unexpected GPU/library tuple");

    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weights{};
    check_lt(cublasLtCreate(&handle), "initialize cuBLASLt");
    check_lt(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "set compute type");
    const cublasOperation_t trans = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "weights transpose");
    check_lt(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)), "input order");
    check_lt(cublasLtMatrixLayoutCreate(&weights, CUDA_R_16BF, k, n, k), "weight layout");
    cublasLtMatmulAlgo_t algo{};
    check_lt(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                  CUDA_R_16BF, CUDA_R_16BF, original[0], &algo), "initialize candidate");
    for (size_t i = 1; i < original.size(); ++i) {
        check_lt(cublasLtMatmulAlgoConfigSetAttribute(&algo, kFields[i], &original[i], sizeof(original[i])),
                 "configure candidate");
    }
    require(signature(algo) == original && numerical_flags(algo) == flags, "candidate plan differs");
    const Words input_rows[] = {read(root + "/669-9.input.bf16", k), read(root + "/1023-67.input.bf16", k)};
    const Words expected[] = {read(root + "/669-9." + label + ".bf16", n),
                              read(root + "/1023-67." + label + ".bf16", n)};
    const Words matrix = read(root + "/weights.bf16", n * k);
    constexpr size_t cap = 16 * 1024 * 1024;
    void *w = nullptr, *x = nullptr, *y = nullptr, *workspace = nullptr;
    check_cuda(cudaMalloc(&w, n * k * 2), "allocate weights");
    check_cuda(cudaMalloc(&x, 256 * k * 2), "allocate input");
    check_cuda(cudaMalloc(&y, 256 * n * 2), "allocate output");
    check_cuda(cudaMalloc(&workspace, cap), "allocate workspace");
    check_cuda(cudaMemcpy(w, matrix.data(), matrix.size() * 2, cudaMemcpyHostToDevice), "upload weights");

    const float alpha = 1, beta = 0;
    for (const size_t m : {1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256}) {
        cublasLtMatrixLayout_t input{}, output{};
        check_lt(cublasLtMatrixLayoutCreate(&input, CUDA_R_16BF, k, m, k), "input layout");
        check_lt(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, n, m, n), "output layout");
        cublasLtMatmulHeuristicResult_t check{};
        const auto status = cublasLtMatmulAlgoCheck(handle, operation, weights, input, output, output, &algo, &check);
        if (status != CUBLAS_STATUS_SUCCESS || check.state != CUBLAS_STATUS_SUCCESS || check.workspaceSize > cap) {
            require(m != 1, "candidate no longer accepts recorded M=1");
            std::printf("%zu unsupported %d %d %zu\n", m, int(status), int(check.state), check.workspaceSize);
        } else {
            Words x_host(m * k), y_host(m * n);
            size_t mismatched_rows = 0, mismatched_words = 0;
            for (size_t phase = 0; phase < 2; ++phase) {
                for (size_t i = 0; i < m; ++i) {
                    const Words& row = input_rows[(i + phase) % 2];
                    std::copy(row.begin(), row.end(), x_host.begin() + i * k);
                }
                check_cuda(cudaMemcpy(x, x_host.data(), x_host.size() * 2, cudaMemcpyHostToDevice), "upload batch");
                check_lt(cublasLtMatmul(handle, operation, &alpha, w, weights, x, input, &beta, y, output, y,
                                       output, &algo, check.workspaceSize ? workspace : nullptr, check.workspaceSize, nullptr),
                         "run fixed candidate at shape");
                check_cuda(cudaMemcpy(y_host.data(), y, y_host.size() * 2, cudaMemcpyDeviceToHost), "read batch");
                for (size_t i = 0; i < m; ++i) {
                    const Words& row = expected[(i + phase) % 2];
                    size_t changed = 0;
                    for (size_t j = 0; j < n; ++j) {
                        changed += y_host[i * n + j] != row[j];
                    }
                    mismatched_words += changed;
                    mismatched_rows += changed != 0;
                }
            }
            require(m != 1 || mismatched_words == 0, "candidate disagrees with its archived M=1 output");
            std::printf("%zu ran %zu %zu %zu\n", m, check.workspaceSize, mismatched_rows, mismatched_words);
        }
        check_lt(cublasLtMatrixLayoutDestroy(output), "release output layout");
        check_lt(cublasLtMatrixLayoutDestroy(input), "release input layout");
    }
    check_cuda(cudaFree(workspace), "release workspace");
    check_cuda(cudaFree(y), "release output");
    check_cuda(cudaFree(x), "release input");
    check_cuda(cudaFree(w), "release weights");
    check_lt(cublasLtMatrixLayoutDestroy(weights), "release weight layout");
    check_lt(cublasLtMatmulDescDestroy(operation), "release operation");
    check_lt(cublasLtDestroy(handle), "release cuBLASLt");
}
