//! Cold preparation of separately owned, qualified packed decode projections.
//!
//! This module is a child of `graph_decode_full`. It never changes an original
//! weight or runs a projection. All copies finish before preparation returns.

use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::forward::LlamaForwardError;
use crate::llama::{ExecutionSite, LlamaOp, PreparedLlamaBatchExecutor};
use riley_cuda::{
    CudaContext, CudaDType, CudaDeviceBuffer, CudaError, CudaGemmConfig, CudaPinnedHostBuffer,
    CudaPreparedGemm, CudaStream,
};
use sha2::{Digest, Sha256};

const LAYERS: usize = 30;
const HIDDEN: u64 = 576;
const KEY_VALUE: u64 = 192;
const INTERMEDIATE: u64 = 1536;
const PACKED_WIDTHS: [u64; 2] = [HIDDEN + 2 * KEY_VALUE, 2 * INTERMEDIATE];
const WORKSPACE_CAP: u64 = 16 * 1024 * 1024;
const MAX_COPY_CHUNK: u64 = 1024 * 1024;
const BUFFER_COUNT: usize = 2 * LAYERS + 2;

pub(super) struct PackedDecodeParents {
    /// Interleaved QKV and gate/up weights for thirty layers, then their two outputs.
    pub(super) buffers: Vec<CudaDeviceBuffer>,
    /// M1/N960/K576 followed by M1/N3072/K576.
    pub(super) plans: Vec<CudaPreparedGemm>,
    pub(super) digest: [u8; 32],
    pub(super) device_bytes: u64,
}

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "packed decode projections",
        reason,
    }
}

fn cuda(error: CudaError) -> LlamaBatchExecutorError {
    cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), error)
}

fn host_allocation(requested_bytes: u64) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::HostAllocation {
        resource: LlamaBatchExecutorResource::HostWorkspace,
        requested_bytes,
    }
}

fn host_bytes(byte_len: u64) -> LlamaBatchExecutorResult<Vec<u8>> {
    let len = usize::try_from(byte_len).map_err(|_| rejected("host byte length overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| host_allocation(byte_len))?;
    bytes.resize(len, 0);
    Ok(bytes)
}

fn hash_text(digest: &mut Sha256, text: &str) {
    digest.update((text.len() as u64).to_le_bytes());
    digest.update(text.as_bytes());
}

/// Serialize every public config and selected metadata field explicitly. This
/// intentionally avoids Debug output, Rust object bytes, and pointer addresses.
fn hash_plan(
    digest: &mut Sha256,
    index: usize,
    plan: &CudaPreparedGemm,
) -> LlamaBatchExecutorResult<()> {
    hash_text(digest, "plan");
    digest.update((index as u64).to_le_bytes());
    digest.update(plan.device_ordinal().to_le_bytes());
    let config = plan.config();
    for value in [
        config.m(),
        config.n(),
        config.k(),
        config.max_workspace_bytes(),
        config.input_bytes(),
        config.weight_bytes(),
        config.output_bytes(),
    ] {
        digest.update(value.to_le_bytes());
    }
    hash_text(digest, config.reduction_policy().id());
    // Serialize the actual types without relying on Rust enum discriminants.
    for (field, dtype) in [
        ("input", config.input_dtype()),
        ("weight", config.weight_dtype()),
        ("accumulator", config.accumulator_dtype()),
        ("output", config.output_dtype()),
    ] {
        hash_text(digest, field);
        hash_text(
            digest,
            match dtype {
                CudaDType::BF16 => "BF16",
                CudaDType::F32 => "F32",
                _ => return Err(rejected("unsupported packed plan scalar type")),
            },
        );
    }
    digest.update([u8::from(config.deterministic())]);

    let metadata = plan.algorithm_metadata();
    digest.update(metadata.backend_id().to_le_bytes());
    digest.update(metadata.algorithm_id().to_le_bytes());
    for value in [
        metadata.tile_id(),
        metadata.stages_id(),
        metadata.split_k(),
        metadata.reduction_scheme(),
        metadata.cta_swizzling(),
        metadata.custom_option(),
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update([u8::from(metadata.deterministic())]);
    digest.update(metadata.workspace_bytes().to_le_bytes());
    digest.update(metadata.numerical_implementation_flags().to_le_bytes());
    let (major, minor) = metadata.compute_capability();
    digest.update(major.to_le_bytes());
    digest.update(minor.to_le_bytes());
    digest.update(metadata.runtime_version().to_le_bytes());
    digest.update(metadata.cublaslt_version().to_le_bytes());
    let (m, n, k) = metadata.dimensions();
    for value in [m, n, k] {
        digest.update(value.to_le_bytes());
    }
    Ok(())
}

fn validate_plan(
    plan: &CudaPreparedGemm,
    config: CudaGemmConfig,
    context: &CudaContext,
) -> LlamaBatchExecutorResult<()> {
    let metadata = plan.algorithm_metadata();
    if plan.is_poisoned()
        || plan.config() != config
        || plan.device_ordinal() != context.device_ordinal()
        || metadata.dimensions() != (1, config.n(), HIDDEN)
        || metadata.compute_capability() != (8, 9)
        || metadata.runtime_version() != 13000
        || metadata.cublaslt_version() != 130101
        || metadata.backend_id() != 1
        || metadata.algorithm_id() != 13
        || metadata.custom_option() != 74
        || metadata.tile_id() != 0
        || metadata.stages_id() != 0
        || metadata.cta_swizzling() != 0
        || metadata.split_k() != 1
        || metadata.reduction_scheme() != 0
        || metadata.workspace_bytes() != 0
        || !metadata.deterministic()
        || metadata.numerical_implementation_flags() != 131585
    {
        return Err(rejected(
            "packed plan differs from the qualified projection metadata",
        ));
    }
    Ok(())
}

fn read_weight(
    buffer: &mut CudaDeviceBuffer,
    bytes: &mut [u8],
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
    chunk_len: usize,
) -> LlamaBatchExecutorResult<()> {
    if buffer.byte_len() != bytes.len() as u64 {
        return Err(rejected("original projection weight byte length differs"));
    }
    for (index, chunk) in bytes.chunks_mut(chunk_len).enumerate() {
        buffer
            .download_to_slice((index * chunk_len) as u64, chunk, staging, stream)
            .map_err(cuda)?;
    }
    Ok(())
}

impl PackedDecodeParents {
    pub(super) fn prepare(
        executor: &mut PreparedLlamaBatchExecutor,
        context: &CudaContext,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<Self> {
        let forward = &mut executor.owner.forward;
        let dimensions = forward.plan.dimensions();
        if executor.owner.poisoned
            || forward.is_poisoned()
            || !executor.config.vllm_smol_p128_graph()
            || forward.plan.sequence_length() != 1
            || forward.plan.layers().len() != LAYERS
            || dimensions.hidden_size() as u64 != HIDDEN
            || dimensions.key_value_width() as u64 != KEY_VALUE
            || dimensions.intermediate_size() as u64 != INTERMEDIATE
            || dimensions.vocabulary_size() != 49152
            || dimensions.query_heads() != 9
            || dimensions.key_value_heads() != 3
            || dimensions.head_dimension() != 64
            || context.compute_capability() != (8, 9)
            || forward.plan.layers().iter().any(|layer| {
                layer.query_bias().is_some()
                    || layer.key_bias().is_some()
                    || layer.value_bias().is_some()
                    || layer.output_bias().is_some()
            })
        {
            return Err(rejected(
                "qualified healthy SmolLM2 P128 geometry and profile required",
            ));
        }
        let chunk_len = usize::try_from(forward.io_staging.byte_len().min(MAX_COPY_CHUNK))
            .map_err(|_| rejected("staging byte length overflow"))?;
        if chunk_len == 0 {
            return Err(rejected("packing requires nonempty reusable staging"));
        }
        let mut readback = host_bytes(chunk_len as u64)?;
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(2)
            .map_err(|_| host_allocation(2 * size_of::<CudaPreparedGemm>() as u64))?;
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(BUFFER_COUNT)
            .map_err(|_| host_allocation((BUFFER_COUNT * size_of::<CudaDeviceBuffer>()) as u64))?;
        let mut digest = Sha256::new();
        hash_text(&mut digest, "riley.packed-decode-parents.v1");
        for value in [LAYERS as u64, HIDDEN, KEY_VALUE, INTERMEDIATE] {
            digest.update(value.to_le_bytes());
        }
        for (index, width) in PACKED_WIDTHS.into_iter().enumerate() {
            // Match the qualified numerical probe: generic strict-no-split
            // selection under the same 16 MiB cap, followed by exact metadata.
            let config = CudaGemmConfig::new(1, width, HIDDEN, WORKSPACE_CAP).map_err(cuda)?;
            let plan = context.prepare_gemm(config).map_err(cuda)?;
            validate_plan(&plan, config, context)?;
            hash_plan(&mut digest, index, &plan)?;
            plans.push(plan);
        }

        let mut device_bytes = 0_u64;
        for layer_index in 0..LAYERS {
            let layer = &forward.plan.layers()[layer_index];
            let ids = [
                layer.query_weight(),
                layer.key_weight(),
                layer.value_weight(),
                layer.gate_weight(),
                layer.up_weight(),
            ];
            let widths = [HIDDEN, KEY_VALUE, KEY_VALUE, INTERMEDIATE, INTERMEDIATE];
            for (group_index, range) in [0..3, 3..5].into_iter().enumerate() {
                let packed_index = 2 * layer_index + group_index;
                let byte_len = plans[group_index].config().weight_bytes();
                let mut bytes = host_bytes(byte_len)?;
                hash_text(&mut digest, "packed-weight");
                for value in [
                    packed_index as u64,
                    layer_index as u64,
                    group_index as u64,
                    byte_len,
                    range.len() as u64,
                ] {
                    digest.update(value.to_le_bytes());
                }
                let mut offset = 0;
                for projection in range {
                    let id = ids[projection];
                    let source_bytes = widths[projection] * HIDDEN * 2;
                    digest.update((id.index() as u64).to_le_bytes());
                    digest.update(source_bytes.to_le_bytes());
                    let end = offset + source_bytes as usize;
                    let original = forward.weights.borrow_graph_weight(id).map_err(|source| {
                        LlamaForwardError::Weight {
                            site: Some(ExecutionSite::global(LlamaOp::IterationCompletion)),
                            source,
                        }
                    })?;
                    read_weight(
                        original,
                        &mut bytes[offset..end],
                        &mut forward.io_staging,
                        stream,
                        chunk_len,
                    )?;
                    offset = end;
                }
                if offset != bytes.len() {
                    return Err(rejected("packed projection row concatenation differs"));
                }
                let mut packed = context.allocate_device_buffer(byte_len).map_err(cuda)?;
                for (index, chunk) in bytes.chunks(chunk_len).enumerate() {
                    let offset = (index * chunk_len) as u64;
                    packed
                        .upload_from_slice(offset, chunk, &mut forward.io_staging, stream)
                        .map_err(cuda)?;
                    packed
                        .download_to_slice(
                            offset,
                            &mut readback[..chunk.len()],
                            &mut forward.io_staging,
                            stream,
                        )
                        .map_err(cuda)?;
                    if &readback[..chunk.len()] != chunk {
                        return Err(rejected(
                            "packed upload readback differs from original weight bytes",
                        ));
                    }
                    // Bind actual verified device content, not only its host source.
                    digest.update(&readback[..chunk.len()]);
                }
                device_bytes = device_bytes
                    .checked_add(packed.byte_len())
                    .ok_or_else(|| rejected("packed device footprint overflow"))?;
                buffers.push(packed);
            }
        }
        for (group_index, plan) in plans.iter().enumerate() {
            let byte_len = plan.config().output_bytes();
            let output = context.allocate_device_buffer(byte_len).map_err(cuda)?;
            hash_text(&mut digest, "uninitialized-output-size-only");
            digest.update(((2 * LAYERS + group_index) as u64).to_le_bytes());
            digest.update(output.byte_len().to_le_bytes());
            device_bytes = device_bytes
                .checked_add(output.byte_len())
                .ok_or_else(|| rejected("packed device footprint overflow"))?;
            buffers.push(output);
        }
        hash_text(&mut digest, "resource-footprint");
        for value in [buffers.len() as u64, plans.len() as u64, device_bytes, 0] {
            // The final field is the selected additional workspace footprint.
            digest.update(value.to_le_bytes());
        }
        Ok(Self {
            buffers,
            plans,
            digest: digest.finalize().into(),
            device_bytes,
        })
    }

    /// The caller must close the graph reservation before releasing its parents.
    /// Attempt every close, preserving the first error while later closes run.
    pub(super) fn close(self) -> LlamaBatchExecutorResult<()> {
        let mut first = None;
        for plan in self.plans {
            if let Err(error) = plan.close() {
                first.get_or_insert_with(|| cuda(error));
            }
        }
        for buffer in self.buffers {
            if let Err(error) = buffer.close() {
                first.get_or_insert_with(|| cuda(error));
            }
        }
        first.map_or(Ok(()), Err)
    }
}
