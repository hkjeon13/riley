//! Cold storage for N2/N4 captures under the P128/M1 aggregate owner.
use super::*;
use riley_cuda::{CudaContext, CudaPreparedStridedGemm, CudaStridedGemmConfig};
pub(super) struct MultiDecodeParents {
    pub(super) scratch: [Vec<CudaDeviceBuffer>; 2],
    pub(super) plans: Vec<CudaPreparedStridedGemm>,
    pub(super) staging: Vec<CudaPinnedHostBuffer>,
}
impl MultiDecodeParents {
    pub(super) fn prepare(context: &CudaContext) -> LlamaBatchExecutorResult<Self> {
        let cuda = |e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e);
        let mut scratch = [Vec::new(), Vec::new()];
        let mut plans = Vec::new();
        let mut staging = Vec::new();
        for (index, bucket) in [2u32, 4].into_iter().enumerate() {
            let b = u64::from(bucket);
            let result = 640 + 98304 * b;
            for bytes in [
                1152 * b,
                1152 * b,
                1920 * b,
                1152 * b,
                1152 * b,
                1152 * b,
                2304 * b,
                6144 * b,
                3072 * b,
                1152 * b,
                98304 * b,
                1280,
                result,
                8 * b,
            ] {
                scratch[index].push(context.allocate_device_buffer(bytes).map_err(cuda)?);
            }
            staging.push(
                context
                    .allocate_pinned_host_buffer(2 * result)
                    .map_err(cuda)?,
            );
            for (n, k) in [
                (960, 576),
                (3072, 576),
                (576, 576),
                (576, 1536),
                (49152, 576),
            ] {
                plans.push(
                    context
                        .prepare_strided_gemm(
                            CudaStridedGemmConfig::new(n, k, bucket, false).map_err(cuda)?,
                        )
                        .map_err(cuda)?,
                );
            }
        }
        Ok(Self {
            scratch,
            plans,
            staging,
        })
    }
}
