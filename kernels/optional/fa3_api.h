#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#define RILEY_FA3_NOEXCEPT noexcept
#else
#define RILEY_FA3_NOEXCEPT
#endif
// Experimental raw ABI, version 1. Not registered as an exact serving backend.
// Q/O: packed [total_q,9,64] BF16. K/V: [physical_pages,3,16,64] BF16.
// Mode 0: causal ragged prefill, mode 1: one query per request decode.
// All host arrays are copied at create; graph identity includes their contents.
typedef struct {
    uint32_t abi_version, mode, batch, total_q, physical_pages;
    const int32_t *q_indptr;       // batch + 1 entries, starts at zero
    const int32_t *kv_lengths;    // batch entries, 1..4096
    const int32_t *page_indptr;    // batch + 1 entries, starts at zero
    const int32_t *pages;         // exactly page_entries physical page IDs
    uint32_t page_entries;
} RileyFa3Spec;
typedef struct {
    const void *q, *k, *v;
    void *o;
    uint64_t q_bytes, k_bytes, v_bytes, o_bytes;
} RileyFa3Buffers;
typedef struct RileyFa3Plan RileyFa3Plan;
enum {
    RILEY_FA3_OK = 0, RILEY_FA3_INVALID = 1, RILEY_FA3_UNSUPPORTED = 2,
    RILEY_FA3_CUDA = 3, RILEY_FA3_CONTEXT = 4, RILEY_FA3_CAPTURE = 5,
    RILEY_FA3_INTERNAL = 6
};
// Host-only structural validation. Host pointers must reference their stated arrays.
int riley_fa3_validate(const RileyFa3Spec *, uint64_t *workspace_bytes) RILEY_FA3_NOEXCEPT;
// Cold operation, no capture. Explicit non-default stream; nonaliasing cudaMalloc
// device buffers on current context. Caller retains buffers/stream until destroy.
int riley_fa3_create(const RileyFa3Spec *, const RileyFa3Buffers *, void *stream,
                     RileyFa3Plan **) RILEY_FA3_NOEXCEPT;
// Same context/stream, serialized calls. No allocation, host copies or sync.
// Enqueue success is not completion; downstream work observes stream ordering.
int riley_fa3_enqueue(RileyFa3Plan *) RILEY_FA3_NOEXCEPT;
// Cold operation. Destroy all graphs retaining this plan FIRST. Synchronizes the
// bound stream before freeing. On failure, retain the pointer and its leases.
int riley_fa3_destroy(RileyFa3Plan **) RILEY_FA3_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef RILEY_FA3_NOEXCEPT
