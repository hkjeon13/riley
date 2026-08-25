#include "rustinfer_cuda.h"

#define RUSTINFER_STRINGIFY_INNER(value) #value
#define RUSTINFER_STRINGIFY(value) RUSTINFER_STRINGIFY_INNER(value)

namespace {

// Compile-time metadata only: this translation unit intentionally contains no
// __global__, __device__, CUDA Runtime, or CUDA Driver API use.
constexpr char kBuildInfo[] =
    "rustinfer-cuda-native abi=" RUSTINFER_STRINGIFY(
        RUSTINFER_CUDA_ABI_VERSION) " nvcc="
    RUSTINFER_STRINGIFY(__CUDACC_VER_MAJOR__) "."
    RUSTINFER_STRINGIFY(__CUDACC_VER_MINOR__) "."
    RUSTINFER_STRINGIFY(__CUDACC_VER_BUILD__);

}  // namespace

extern "C" uint32_t rustinfer_cuda_abi_version(void) noexcept {
  return RUSTINFER_CUDA_ABI_VERSION;
}

extern "C" const char* rustinfer_cuda_build_info(void) noexcept {
  return kBuildInfo;
}
