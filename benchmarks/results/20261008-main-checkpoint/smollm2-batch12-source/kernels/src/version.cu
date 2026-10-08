#include "riley_cuda.h"

#define RILEY_STRINGIFY_INNER(value) #value
#define RILEY_STRINGIFY(value) RILEY_STRINGIFY_INNER(value)

namespace {

// Compile-time metadata only: this translation unit intentionally contains no
// __global__, __device__, CUDA Runtime, or CUDA Driver API use.
constexpr char kBuildInfo[] =
    "riley-cuda-native abi=" RILEY_STRINGIFY(
        RILEY_CUDA_ABI_VERSION) " nvcc="
    RILEY_STRINGIFY(__CUDACC_VER_MAJOR__) "."
    RILEY_STRINGIFY(__CUDACC_VER_MINOR__) "."
    RILEY_STRINGIFY(__CUDACC_VER_BUILD__);

}  // namespace

extern "C" uint32_t riley_cuda_abi_version(void) noexcept {
  return RILEY_CUDA_ABI_VERSION;
}

extern "C" const char* riley_cuda_build_info(void) noexcept {
  return kBuildInfo;
}
