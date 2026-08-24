#ifndef RUSTINFER_CUDA_H_
#define RUSTINFER_CUDA_H_

#include <stdint.h>

#define RUSTINFER_CUDA_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

// Host-only ABI metadata. These functions do not initialize or query a device.
uint32_t rustinfer_cuda_abi_version(void);
const char* rustinfer_cuda_build_info(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RUSTINFER_CUDA_H_
