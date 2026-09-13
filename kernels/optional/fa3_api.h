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
// bound stream before freeing. A close-stage CUDA failure permanently poisons
// the plan: retain pointer/leases, never retry a possibly completed free.
int riley_fa3_destroy(RileyFa3Plan **) RILEY_FA3_NOEXCEPT;
// Internal model-recorder ABI: external workspace is retained/accounted by the
// existing Riley graph reservation. Return values here are cudaError_t codes.
// Metadata prepare is SM89-compatible and graph-safe. Invalid packets publish
// zero query lengths and set status, including when a bad page is in a suffix.
uint64_t riley_fa3_model_workspace_bytes(void) RILEY_FA3_NOEXCEPT;
int riley_fa3_model_metadata_prepare(void *stream, const void *packet,
    uint64_t packet_bytes, void *workspace, uint64_t workspace_bytes,
    uint32_t physical, uint32_t capacity, uint32_t context, void *status) RILEY_FA3_NOEXCEPT;
// Cold prepare_only=1 checks SM90 capabilities and sets kernel attributes, with
// no attention launch. Normal calls require that preparation and exact stable
// shape/addresses. Caller validates context, extents, alignment and all leases.
int riley_fa3_model_attention(void *stream, const void *q, const void *k,
    const void *v, void *out, void *workspace, uint64_t workspace_bytes,
    uint32_t physical, uint32_t capacity, uint32_t context, uint32_t prepare_only) RILEY_FA3_NOEXCEPT;
// Schedule once after metadata prepare, then attention resets only its semaphore
// for each sequential layer. Not safe for concurrent layers sharing workspace.
int riley_fa3_model_schedule(void *stream, void *workspace, uint64_t workspace_bytes,
    uint32_t capacity, uint32_t context) RILEY_FA3_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef RILEY_FA3_NOEXCEPT
