//! Cold-owned CUDA resources for the Llama continuous-batch executor.
//!
//! This module is the sole lifetime boundary for the uploaded forward owner,
//! optional shape GEMM variants, paged KV storage, absolute `RoPE` tables,
//! metadata input, and output workspaces.  It deliberately does not import
//! scheduler or server policy.  The batch-executor facade retains logical
//! iteration state and borrows these resources for dispatch.

use riley_cuda::{
    CudaContext, CudaDType, CudaDeviceBuffer, CudaStream, RopeTableParams, rope_table,
};
use riley_model::LoadedModel;

use super::super::batch::{LlamaBatchMetadataConfig, PreparedLlamaBatchMetadata};
use super::super::forward::{LlamaRopeTableProfile, PreparedLlamaForward, span_mut};
use super::super::{ExecutionSite, LlamaOp};
use super::buffers::{
    BatchDeviceInput, BatchHostInput, allocate_packed_device_input, allocate_packed_host_input,
    allocate_synchronous_device_input, allocate_synchronous_host_input, close_device_input,
    close_host_input,
};
use super::config::{BatchMetadataTransport, PreparedLlamaBatchExecutorConfig};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    cuda_error as owner_cuda, record_close, usize_u64,
};
use super::gemm_plan::{PreparedLlamaBatchShape, prepare_shape_variants};
use super::host::allocate_zeroed_host_bytes;
use super::metadata::PackedIterationLayout;
use super::output::{GREEDY_RESULT_BYTES, greedy_result_capacity_bytes, output_logits_bytes};
use super::rope::{build_absolute_cpu_rope_tables, build_absolute_rope_angles};
use crate::paged_kv::KvLayout;

const F32_BYTES: u64 = 4;
pub(in crate::llama) const SUPPORTED_HEAD_DIMENSION: usize = 64;

/// Host workspace coupled to the prepared device input and greedy result
/// storage.  This remains private to the Llama executor boundary even though
/// dispatch borrows it while an iteration is executing.
pub(in crate::llama) struct BatchHostWorkspace {
    pub(in crate::llama) input: BatchHostInput,
    pub(in crate::llama) greedy_results: Box<[u8]>,
}

/// One cold-prepared owner of all reusable CUDA-side batch resources.
///
/// Shape variants hold only their plan/GEMM descriptors; they share this
/// owner's uploaded weights, paged KV arena, `RoPE`, metadata, and output
/// workspaces rather than allocating duplicates.
pub(in crate::llama) struct PreparedLlamaBatchOwner {
    pub(in crate::llama) metadata: PreparedLlamaBatchMetadata,
    pub(in crate::llama) forward: PreparedLlamaForward,
    pub(in crate::llama) shape_variants: Box<[PreparedLlamaBatchShape]>,
    pub(in crate::llama) layout: KvLayout,
    pub(in crate::llama) key_cache: CudaDeviceBuffer,
    pub(in crate::llama) value_cache: CudaDeviceBuffer,
    pub(in crate::llama) absolute_rope_cos: CudaDeviceBuffer,
    pub(in crate::llama) absolute_rope_sin: CudaDeviceBuffer,
    pub(in crate::llama) device_input: BatchDeviceInput,
    pub(in crate::llama) gathered_logits: Option<CudaDeviceBuffer>,
    pub(in crate::llama) greedy_results: Option<CudaDeviceBuffer>,
    pub(in crate::llama) host: BatchHostWorkspace,
    pub(in crate::llama) poisoned: bool,
}

impl PreparedLlamaBatchOwner {
    /// Cold opt-in bridge for the existing per-operation metadata transport.
    /// The returned graph borrows actual executor storage, including both full
    /// KV parents. Packed-slab metadata and inventory promotion remain blocked.
    #[allow(dead_code)] // Admission remains disabled until complete C07 evidence.
    pub(in crate::llama) fn prepare_parent_attention_graph<'a>(
        &'a mut self,
        stream: &'a mut CudaStream,
        layer_index: usize,
        batch: riley_cuda::PackedBatchHostV1<'_>,
        query_heads: u64,
        output_rows: u64,
    ) -> LlamaBatchExecutorResult<riley_cuda::ParentAttentionGraph<'a>> {
        let site = ExecutionSite::layer(layer_index, LlamaOp::RaggedPagedAttention);
        if self.poisoned || self.layout.head_dimension() != 64 {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "attention graph parent binding",
                reason: "owner is poisoned or not D64",
            });
        }
        let layer = riley_cuda::AttentionParentLayer::new(
            self.layout.layer_count() as u64,
            layer_index as u64,
            self.layout.physical_block_count() as u64,
            self.layout.key_value_head_count() as u64,
        )
        .map_err(|source| owner_cuda(site, source))?;
        if self.layout.layer_byte_offset(layer_index) != Some(layer.byte_offset())
            || self.layout.layer_stride_bytes() != layer.byte_len()
            || self.layout.bytes_per_kind() != layer.parent_byte_len()
        {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "attention graph parent binding",
                reason: "KV layout differs from native layer geometry",
            });
        }
        let BatchDeviceInput::PerOperation(metadata) = &mut self.device_input else {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "attention graph parent binding",
                reason: "packed metadata slab binding is not admitted",
            });
        };
        riley_cuda::ParentAttentionGraph::prepare(
            riley_cuda::ParentAttentionResources {
                stream,
                query: &mut self.forward.buffers.hidden_rotary,
                key_parent: &mut self.key_cache,
                value_parent: &mut self.value_cache,
                output: &mut self.forward.buffers.hidden_context,
                sequence_block_offsets: &mut metadata.sequence_block_offsets,
                block_ids: &mut metadata.physical_block_ids,
                valid_tokens: &mut metadata.valid_tokens,
                row_sequence_slots: &mut metadata.row_sequence_slots,
                row_positions: &mut metadata.row_positions,
            },
            layer,
            batch,
            query_heads,
            output_rows,
            0.125,
        )
        .map_err(|source| owner_cuda(site, source))
    }

    /// Captures canonical C07 metadata using actual model-owned query and KV.
    /// This cold API measures same-owner parity before exposing any evidence.
    #[cfg(feature = "cuda")]
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(in crate::llama) fn prepare_c07_attention_graph<'a>(
        &'a mut self,
        stream: &'a mut CudaStream,
        device: &'a mut crate::llama::graph_decode_exact_device_slab::PureDecodeGraphV1ExactDeviceSlab,
        staging: &mut riley_cuda::CudaPinnedHostBuffer,
        source: crate::llama::graph_decode_exact_host_slab::PureDecodeGraphV1ExactHostSlabLease<'_>,
        expected: crate::llama::graph_decode_layout::PureDecodeGraphMetadataLayout,
        layer_index: usize,
        batch: riley_cuda::PackedBatchHostV1<'_>,
    ) -> LlamaBatchExecutorResult<
        crate::llama::graph_decode_attention_owner::C07AttentionGraphOwner<'a>,
    > {
        if self.poisoned {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "C07 attention owner",
                reason: "executor is poisoned",
            });
        }
        let query_heads = self.forward.plan.dimensions().query_heads() as u64;
        crate::llama::graph_decode_attention_owner::C07AttentionGraphOwner::prepare(
            stream,
            &mut self.forward.buffers.hidden_rotary,
            &mut self.key_cache,
            &mut self.value_cache,
            &mut self.forward.buffers.hidden_context,
            device,
            staging,
            source,
            expected,
            self.layout,
            layer_index,
            batch,
            query_heads,
            &mut self.poisoned,
        )
    }

    /// G02C cold M=1 capture of the actual final norm workspace and weight.
    /// Profile rejection happens before CUDA work; per-layer Norm evidence is
    /// never substituted for this final normalization operation.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(in crate::llama) fn prepare_c07_final_norm_graph<'a>(
        &'a mut self,
        stream: &'a mut CudaStream,
    ) -> LlamaBatchExecutorResult<
        crate::llama::graph_decode_final_norm_owner::C07FinalNormGraphOwner<'a>,
    > {
        use crate::llama::graph_decode_final_norm_owner::{
            C07FinalNormGraphOwner, FinalNormBinding,
        };
        let rejected = |reason| LlamaBatchExecutorError::InvalidConfiguration {
            field: "C07 final norm owner",
            reason,
        };
        if self.poisoned || self.forward.is_poisoned() {
            return Err(rejected("executor is poisoned"));
        }
        let plan = &self.forward.plan;
        if plan.sequence_length() != 1 {
            return Err(rejected("initial final norm owner requires exact M=1"));
        }
        let hidden = plan.dimensions().hidden_size();
        let weight_id = plan.final_norm_weight();
        let binding = FinalNormBinding::new(
            self.forward.rms_norm_profile(),
            1,
            hidden as u64,
            plan.final_norm_epsilon(),
            weight_id,
        )
        .map_err(|_| rejected("final norm profile or geometry has no exact capture evidence"))?;
        let metadata = self
            .forward
            .weights
            .physical_metadata(weight_id)
            .ok_or_else(|| rejected("final norm weight belongs to a different uploaded owner"))?;
        if metadata.dtype != riley_tensor::DType::BF16
            || metadata.shape != [hidden]
            || metadata.byte_len != hidden as u64 * 2
        {
            return Err(rejected("final norm weight dtype or shape differs"));
        }
        let weight = self
            .forward
            .weights
            .borrow_graph_weight(weight_id)
            .map_err(|_| rejected("final norm weight cannot be borrowed"))?;
        C07FinalNormGraphOwner::prepare(
            stream,
            &mut self.forward.buffers.hidden_current,
            weight,
            &mut self.forward.buffers.hidden_norm,
            &mut self.forward.io_staging,
            binding,
            &mut self.poisoned,
        )
    }

    /// Uploads weights and reserves every CUDA and host resource reused by a
    /// prepared continuous-batch executor.
    ///
    /// The caller has already normalized and validated the host-only config.
    /// This method preserves the established cold allocation and rollback
    /// order: forward first, then optional shape plans, KV/`RoPE`, metadata,
    /// outputs, and host workspace.
    #[allow(clippy::too_many_lines)]
    pub(in crate::llama) fn prepare(
        model: &LoadedModel,
        context: &CudaContext,
        stream: &mut CudaStream,
        config: PreparedLlamaBatchExecutorConfig,
    ) -> LlamaBatchExecutorResult<Self> {
        let spec = model.spec();
        let attention = spec
            .blocks()
            .first()
            .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                field: "model.blocks",
                reason: "the validated model must contain at least one decoder layer",
            })?
            .attention();
        if attention.head_dimension() != SUPPORTED_HEAD_DIMENSION {
            return Err(LlamaBatchExecutorError::UnsupportedHeadDimension {
                expected: SUPPORTED_HEAD_DIMENSION,
                actual: attention.head_dimension(),
            });
        }
        let bounds = config.metadata();
        if bounds.max_input_tokens() > spec.max_sequence_length() {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "max_input_tokens",
                reason: "maximum dense rows must not exceed the model sequence length",
            });
        }

        let metadata = PreparedLlamaBatchMetadata::prepare(bounds)?;
        let mut forward = PreparedLlamaForward::prepare(
            model,
            context,
            stream,
            bounds.max_input_tokens(),
            config.forward(),
        )?;
        let shape_variants = match prepare_shape_variants(
            model,
            context,
            &forward,
            config.shape_policy(),
            config.configured_shape_buckets(),
        ) {
            Ok(variants) => variants,
            Err(error) => {
                let _ = forward.close();
                return Err(error);
            }
        };
        let required_gemm_workspace_bytes = shape_variants.iter().fold(
            forward.gemms.maximum_workspace_bytes(),
            |required, shape| required.max(shape.gemms.maximum_workspace_bytes()),
        );
        if let Err(error) =
            forward.ensure_batch_shape_gemm_workspace(context, required_gemm_workspace_bytes)
        {
            for shape in shape_variants {
                let _ = shape.close();
            }
            let _ = forward.close();
            return Err(LlamaBatchExecutorError::Forward(error));
        }
        let dimensions = forward.plan.dimensions();
        if dimensions.head_dimension() != SUPPORTED_HEAD_DIMENSION {
            return Err(LlamaBatchExecutorError::UnsupportedHeadDimension {
                expected: SUPPORTED_HEAD_DIMENSION,
                actual: dimensions.head_dimension(),
            });
        }
        let layout = KvLayout::checked(
            forward.plan.layers().len(),
            bounds.physical_block_count(),
            dimensions.key_value_heads(),
            dimensions.head_dimension(),
        )?;

        let key_cache = allocate_device(
            context,
            layout.bytes_per_kind(),
            ExecutionSite::layer(0, LlamaOp::KvCacheWrite),
        )?;
        let value_cache = allocate_device(
            context,
            layout.bytes_per_kind(),
            ExecutionSite::layer(0, LlamaOp::KvCacheWrite),
        )?;
        let rope_bytes_per_kind = checked_product_u64(
            &[
                usize_u64(
                    spec.max_sequence_length(),
                    LlamaBatchExecutorResource::RopeCos,
                )?,
                usize_u64(
                    dimensions.head_dimension() / 2,
                    LlamaBatchExecutorResource::RopeCos,
                )?,
                F32_BYTES,
            ],
            LlamaBatchExecutorResource::RopeCos,
        )?;
        let mut absolute_rope_cos = allocate_device(
            context,
            rope_bytes_per_kind,
            ExecutionSite::layer(0, LlamaOp::QueryRope),
        )?;
        let mut absolute_rope_sin = allocate_device(
            context,
            rope_bytes_per_kind,
            ExecutionSite::layer(0, LlamaOp::QueryRope),
        )?;
        let rope_site = ExecutionSite::layer(0, LlamaOp::QueryRope);
        if forward.rope_table_profile() == LlamaRopeTableProfile::HuggingFaceCuda {
            let rope_angles = build_absolute_rope_angles(
                spec.max_sequence_length(),
                dimensions.head_dimension(),
                forward.plan.rope_theta(),
            )?;
            absolute_rope_cos
                .upload_from_slice(0, &rope_angles, &mut forward.io_staging, stream)
                .map_err(|source| owner_cuda(rope_site, source))?;
            let mut rope_table_params = RopeTableParams {
                angles_cos: span_mut(
                    &mut absolute_rope_cos,
                    CudaDType::F32,
                    rope_bytes_per_kind,
                    rope_site,
                )?,
                sin: span_mut(
                    &mut absolute_rope_sin,
                    CudaDType::F32,
                    rope_bytes_per_kind,
                    rope_site,
                )?,
                element_count: rope_bytes_per_kind / F32_BYTES,
            };
            rope_table(&mut rope_table_params, stream)
                .map_err(|source| owner_cuda(rope_site, source))?;
        } else {
            let (rope_cos, rope_sin) = build_absolute_cpu_rope_tables(
                spec.max_sequence_length(),
                dimensions.head_dimension(),
                forward.plan.rope_theta(),
            )?;
            absolute_rope_cos
                .upload_from_slice(0, &rope_cos, &mut forward.io_staging, stream)
                .map_err(|source| owner_cuda(rope_site, source))?;
            absolute_rope_sin
                .upload_from_slice(0, &rope_sin, &mut forward.io_staging, stream)
                .map_err(|source| owner_cuda(rope_site, source))?;
        }

        let device_input = allocate_device_input(context, bounds, config.metadata_transport())?;
        let gathered_logits_capacity_bytes =
            output_logits_bytes(bounds.max_output_slots(), dimensions.vocabulary_size())?;
        let gathered_logits = if bounds.max_output_slots() == 0 {
            None
        } else {
            Some(allocate_device(
                context,
                gathered_logits_capacity_bytes,
                ExecutionSite::global(LlamaOp::OutputGather),
            )?)
        };
        let greedy_result_capacity_bytes = greedy_result_capacity_bytes(bounds.max_output_slots())?;
        let greedy_results = if bounds.max_output_slots() == 0 {
            None
        } else {
            Some(allocate_device(
                context,
                greedy_result_capacity_bytes,
                ExecutionSite::global(LlamaOp::OutputGather),
            )?)
        };
        let host = allocate_host_workspace(context, bounds, config.metadata_transport())?;

        Ok(Self {
            metadata,
            forward,
            shape_variants,
            layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            poisoned: false,
        })
    }

    /// Explicitly closes every resource in the original observable order.
    ///
    /// All cleanup attempts run even after an error.  The first batch-resource
    /// failure wins over a shape-plan failure, which wins over the reused
    /// forward owner's failure.  `Drop` remains best-effort only.
    #[allow(clippy::too_many_lines)]
    pub(in crate::llama) fn close(self) -> LlamaBatchExecutorResult<()> {
        let Self {
            metadata: _,
            forward,
            shape_variants,
            layout: _,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            poisoned: _,
        } = self;
        let mut first = None;
        record_close(
            &mut first,
            LlamaBatchExecutorResource::KeyCache,
            key_cache.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::ValueCache,
            value_cache.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::RopeCos,
            absolute_rope_cos.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::RopeSin,
            absolute_rope_sin.close(),
        );
        record_first_error(&mut first, close_device_input(device_input));
        if let Some(buffer) = gathered_logits {
            record_close(
                &mut first,
                LlamaBatchExecutorResource::GatheredLogits,
                buffer.close(),
            );
        }
        if let Some(buffer) = greedy_results {
            record_close(
                &mut first,
                LlamaBatchExecutorResource::GreedyResults,
                buffer.close(),
            );
        }
        record_first_error(&mut first, close_host_input(host.input));

        let mut shape_error = None;
        for shape in shape_variants {
            record_first_error(&mut shape_error, shape.close().err());
        }
        let forward_error = forward
            .close()
            .map_err(LlamaBatchExecutorError::Forward)
            .err();
        finish_close_errors(first, shape_error, forward_error)
    }
}

fn allocate_device_input(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
    transport: BatchMetadataTransport,
) -> LlamaBatchExecutorResult<BatchDeviceInput> {
    match transport {
        BatchMetadataTransport::Synchronous => allocate_synchronous_device_input(context, bounds),
        BatchMetadataTransport::PackedAsync => {
            let capacity = PackedIterationLayout::capacity(bounds)?.total_bytes;
            allocate_packed_device_input(context, capacity)
        }
    }
}

fn allocate_host_workspace(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
    transport: BatchMetadataTransport,
) -> LlamaBatchExecutorResult<BatchHostWorkspace> {
    let input = match transport {
        BatchMetadataTransport::Synchronous => allocate_synchronous_host_input(bounds)?,
        BatchMetadataTransport::PackedAsync => {
            let capacity = PackedIterationLayout::capacity(bounds)?.total_bytes;
            allocate_packed_host_input(context, capacity)?
        }
    };
    Ok(BatchHostWorkspace {
        input,
        greedy_results: allocate_zeroed_host_bytes(bounds.max_output_slots(), GREEDY_RESULT_BYTES)?,
    })
}

fn allocate_device(
    context: &CudaContext,
    byte_len: u64,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<CudaDeviceBuffer> {
    context
        .allocate_device_buffer(byte_len)
        .map_err(|source| owner_cuda(site, source))
}

fn checked_product_u64(
    values: &[u64],
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<u64> {
    values.iter().try_fold(1_u64, |product, &value| {
        product
            .checked_mul(value)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })
    })
}

fn record_first_error(
    first: &mut Option<LlamaBatchExecutorError>,
    candidate: Option<LlamaBatchExecutorError>,
) {
    if first.is_none() {
        *first = candidate;
    }
}

fn finish_close_errors(
    first: Option<LlamaBatchExecutorError>,
    shape_error: Option<LlamaBatchExecutorError>,
    forward_error: Option<LlamaBatchExecutorError>,
) -> LlamaBatchExecutorResult<()> {
    match (first, shape_error, forward_error) {
        (Some(error), _, _) | (None, Some(error), _) | (None, None, Some(error)) => Err(error),
        (None, None, None) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{finish_close_errors, record_first_error};
    use crate::llama::executor::error::LlamaBatchExecutorError;

    fn failure(field: &'static str) -> LlamaBatchExecutorError {
        LlamaBatchExecutorError::InvalidConfiguration {
            field,
            reason: "test cleanup precedence",
        }
    }

    #[test]
    fn cleanup_precedence_keeps_the_first_resource_error() {
        let mut first = None;
        record_first_error(&mut first, Some(failure("key_cache")));
        record_first_error(&mut first, Some(failure("value_cache")));

        let error = finish_close_errors(first, Some(failure("shape")), Some(failure("forward")))
            .expect_err("the first resource close failure must win");
        assert!(matches!(
            error,
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "key_cache",
                ..
            }
        ));
    }

    #[test]
    fn cleanup_precedence_keeps_shape_before_forward_error() {
        let error = finish_close_errors(None, Some(failure("shape")), Some(failure("forward")))
            .expect_err("a shape close failure must win before forward close");
        assert!(matches!(
            error,
            LlamaBatchExecutorError::InvalidConfiguration { field: "shape", .. }
        ));
    }

    #[test]
    fn cleanup_succeeds_when_every_close_succeeds() {
        finish_close_errors(None, None, None).expect("all successful close results must succeed");
    }
}
