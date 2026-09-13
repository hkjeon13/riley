#pragma once
#include <stdint.h>
// Internal raw ABI. Owner/context/extent/stream validation belongs to recorder.
#ifdef __cplusplus
extern "C" {
#define RILEY_FI_NOEXCEPT noexcept
#else
#define RILEY_FI_NOEXCEPT
#endif
uint64_t riley_flashinfer_decode_workspace_bytes(void) RILEY_FI_NOEXCEPT;
int riley_flashinfer_decode_prepare(void*, const void*, const void*, void*, uint64_t,
                                   uint32_t, uint32_t, void*) RILEY_FI_NOEXCEPT;
int riley_flashinfer_decode_run(void*, const void*, const void*, const void*, void*,
                               void*, uint64_t) RILEY_FI_NOEXCEPT;
int riley_flashinfer_mixed_prepare(void*, const void*, void*, uint64_t, uint32_t,
                                  uint32_t, uint32_t, void*) RILEY_FI_NOEXCEPT;
int riley_flashinfer_mixed_run(void*, const void*, const void*, const void*, void*,
                              void*, uint64_t) RILEY_FI_NOEXCEPT;
uint64_t riley_flashinfer_prefill_workspace_bytes(void) RILEY_FI_NOEXCEPT;
int riley_flashinfer_prefill_prepare(void*, const void*, void*, uint64_t,
                                    uint32_t, uint32_t, void*) RILEY_FI_NOEXCEPT;
int riley_flashinfer_prefill_run(void*, const void*, const void*, const void*, void*,
                                void*, uint64_t) RILEY_FI_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef RILEY_FI_NOEXCEPT
