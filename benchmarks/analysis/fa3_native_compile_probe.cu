// Build/link probe only. This does not expose a production ABI or launch attention.
#include "flash_fwd_launch_template.h"
#include <cstdio>

// Instantiate actual upstream BF16 causal dense attention, including TMA/GMMA.
// Q/KV head counts are runtime parameters; head dimension is fixed at 64.
void riley_fa3_dense_compile_probe(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<90, 64, 64, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
                  true, false, false, false, false, false, false,
                  false, false, false>(params, stream);
}

// PagedKVNonTMA avoids claiming TMA page-layout compatibility for Riley page16.
// Varlen scheduler preparation is linked from the same immutable upstream tree.
void riley_fa3_paged_prefill_compile_probe(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<90, 64, 64, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
                  true, false, false, true, true, false, false,
                  true, false, false>(params, stream);
}

void riley_fa3_paged_decode_compile_probe(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<90, 64, 64, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
                  false, false, false, true, true, false, false,
                  true, false, false>(params, stream);
}

int main() {
    bool cuda_caught = false, cutlass_caught = false;
    try { CHECK_CUDA(cudaErrorInvalidValue); }
    catch (const RileyFa3CudaError &e) { cuda_caught = e.status == cudaErrorInvalidValue; }
    try { CHECK_CUTLASS(cutlass::Status::kErrorInvalidProblem); }
    catch (const RileyFa3CutlassError &e) {
        cutlass_caught = e.status == cutlass::Status::kErrorInvalidProblem;
    }
    if (!cuda_caught || !cutlass_caught) return 1;
    std::puts("{\"native_error_contract\":\"pass\",\"attention_runtime\":\"not_run\",\"production_adapter\":false}");
    return 0;
}
