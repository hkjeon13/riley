//! Execution dispatch for the Llama batch executor.
//!
//! This module consumes cold-prepared, borrowed resources to perform one
//! iteration: metadata transport, command-batch lifecycle, exact forward
//! dispatch, and output primitives. The enclosing owner retains public API,
//! resource lifetime, output publication, failure-state policy, allocation,
//! and explicit close ordering.

use std::mem;

use riley_cuda::{
    AttentionReductionProfile, Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut,
    CudaCommandStream, CudaDType, CudaDeviceBuffer, CudaExecutionStream, CudaStream,
    EmbeddingParams, GatedMultiplyParams, IndexedRopeParams, PackedBatchHostV1, PackedBatchV1,
    RaggedPagedAttentionParams, RaggedPagedKvCacheWriteParams, ResidualAddParams,
    ResidualRmsNormParams, RmsNormParams, RowGatherParams, SiluParams, deterministic_bf16_argmax,
    embedding, fixed37_ragged_paged_attention, gated_multiply, grouped_ragged_paged_attention,
    indexed_rope, ragged_paged_attention, ragged_paged_kv_cache_write, residual_add, row_gather,
    silu,
};

use super::super::batch::LlamaPackedBatchMetadata;
use super::super::forward::{
    ForwardBuffers, GemmPlans, LlamaForwardError, LlamaRmsNormProfile, PreparedLlamaForward,
    execute_gemm, execute_profile_residual_rms_norm, execute_profile_rms_norm,
    execute_projection_bias, poison_for_cuda_error, span, span_mut, weight_span,
};
use super::super::{ExecutionSite, LlamaExecutionPlan, LlamaOp};
use super::buffers::{BatchDeviceInput, BatchHostInput, U16_BYTES, U32_BYTES};
use super::config::{
    BatchMetadataTransport, ExecutionCompletionImplementation, PreparedLlamaBatchExecutorConfig,
    RaggedAttentionImplementation, ResidualNormImplementation,
};
use super::device_views::{packed_device_views, per_operation_device_views};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    checked_byte_len, cuda_error as batch_cuda, cuda_error as dispatch_cuda, usize_u64,
};
use super::gemm_plan::PreparedLlamaBatchShape;
use super::metadata::{PackedIterationLayout, encode_u16, encode_u32, pack_iteration_input};
use super::output::{greedy_result_bytes, output_logits_bytes};
use super::rope::absolute_rope_position_count;
use crate::cuda_weights::CudaUploadedWeights;
use crate::paged_kv::KvLayout;

const BF16_BYTES: u64 = 2;

/// Tracks whether a command submission could have mutated iteration state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::llama) enum BatchDispatchDisposition {
    /// Validation or setup failed before a command batch began.
    #[default]
    PreDispatch,
    /// A command batch was opened, so partial device-side mutation is possible.
    CommandSubmissionStarted,
}

impl BatchDispatchDisposition {
    /// Returns whether the enclosing iteration may have been partially mutated.
    #[must_use]
    pub(in crate::llama) const fn mutation_may_have_occurred(self) -> bool {
        matches!(self, Self::CommandSubmissionStarted)
    }
}

/// Runs one borrowed command-batch body and always observes completion.
///
/// The caller retains metadata preflight, the command body, and failure-state
/// decisions. Once the native batch begins, this guard records the established
/// mutation-unknown disposition before exposing its non-replaceable proxy.
pub(in crate::llama) fn execute_iteration_command_batch<'stream, F>(
    stream: &'stream mut CudaStream,
    dispatch_disposition: &mut BatchDispatchDisposition,
    body: F,
) -> LlamaBatchExecutorResult<()>
where
    F: for<'batch> FnOnce(&mut CudaCommandStream<'batch, 'stream>) -> LlamaBatchExecutorResult<()>,
{
    let completion_site = ExecutionSite::global(LlamaOp::IterationCompletion);
    let mut command_batch = stream
        .begin_command_batch()
        .map_err(|source| dispatch_cuda(completion_site, source))?;
    *dispatch_disposition = BatchDispatchDisposition::CommandSubmissionStarted;
    let body_result = {
        let mut commands = command_batch.commands();
        body(&mut commands)
    };
    let completion_result = command_batch
        .finish()
        .map_err(|source| dispatch_cuda(completion_site, source));
    match completion_result {
        Err(error) => Err(error),
        Ok(()) => body_result,
    }
}

/// Borrowed state for one non-empty output gather and optional greedy argmax.
pub(in crate::llama) struct OutputPrimitiveDispatch<'a> {
    pub(in crate::llama) logits: &'a CudaDeviceBuffer,
    pub(in crate::llama) logits_byte_len: u64,
    pub(in crate::llama) vocabulary_size: usize,
    pub(in crate::llama) dense_rows: usize,
    pub(in crate::llama) output_indices: Option<CudaBufferSpan<'a>>,
    pub(in crate::llama) output_indices_host: &'a [u32],
    pub(in crate::llama) output_count: usize,
    pub(in crate::llama) gathered_logits: &'a mut Option<CudaDeviceBuffer>,
    pub(in crate::llama) greedy_results: &'a mut Option<CudaDeviceBuffer>,
    pub(in crate::llama) produce_greedy_tokens: bool,
}

/// Runs output row gather and, when requested, exact deterministic argmax.
///
/// The caller invokes this only for a non-empty output count and retains all
/// state transitions and post-dispatch failure decisions.
pub(in crate::llama) fn dispatch_output_primitives<S: CudaExecutionStream + ?Sized>(
    dispatch: OutputPrimitiveDispatch<'_>,
    stream: &mut S,
) -> LlamaBatchExecutorResult<()> {
    let OutputPrimitiveDispatch {
        logits,
        logits_byte_len,
        vocabulary_size,
        dense_rows,
        output_indices,
        output_indices_host,
        output_count,
        gathered_logits,
        greedy_results,
        produce_greedy_tokens,
    } = dispatch;
    let output_indices = output_indices.ok_or(LlamaBatchExecutorError::InvalidConfiguration {
        field: "output_token_indices",
        reason: "non-empty output has no cold-prepared device index buffer",
    })?;
    let site = ExecutionSite::global(LlamaOp::OutputGather);
    {
        let output =
            gathered_logits
                .as_mut()
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "gathered_logits",
                    reason: "non-empty output has no cold-prepared device buffer",
                })?;
        let mut params = RowGatherParams {
            input: span(logits, CudaDType::BF16, logits_byte_len, site)?,
            row_indices: output_indices,
            row_indices_host: output_indices_host,
            output: CudaBufferSpanMut::new(
                output,
                CudaDType::BF16,
                0,
                output_logits_bytes(output_count, vocabulary_size)?,
            )
            .map_err(|source| dispatch_cuda(site, source))?,
            input_row_count: usize_u64(dense_rows, LlamaBatchExecutorResource::GatheredLogits)?,
            column_count: usize_u64(vocabulary_size, LlamaBatchExecutorResource::GatheredLogits)?,
        };
        row_gather(&mut params, stream).map_err(|source| dispatch_cuda(site, source))?;
    }
    if produce_greedy_tokens {
        let logits =
            gathered_logits
                .as_ref()
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "gathered_logits",
                    reason: "greedy selection requires gathered logits",
                })?;
        let results =
            greedy_results
                .as_mut()
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "greedy_results",
                    reason: "non-empty output has no cold-prepared greedy result buffer",
                })?;
        let mut argmax = Bf16ArgmaxParams {
            logits: CudaBufferSpan::new(
                logits,
                CudaDType::BF16,
                0,
                output_logits_bytes(output_count, vocabulary_size)?,
            )
            .map_err(|source| dispatch_cuda(site, source))?,
            results: CudaBufferSpanMut::new(
                results,
                CudaDType::U32,
                0,
                usize_u64(
                    greedy_result_bytes(output_count)?,
                    LlamaBatchExecutorResource::GreedyResults,
                )?,
            )
            .map_err(|source| dispatch_cuda(site, source))?,
            row_count: usize_u64(output_count, LlamaBatchExecutorResource::GreedyResults)?,
            vocabulary_size: usize_u64(vocabulary_size, LlamaBatchExecutorResource::GreedyResults)?,
        };
        deterministic_bf16_argmax(&mut argmax, stream)
            .map_err(|source| dispatch_cuda(site, source))?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::large_types_passed_by_value
)]
pub(in crate::llama) fn execute_packed(
    packed: LlamaPackedBatchMetadata<'_>,
    config: PreparedLlamaBatchExecutorConfig,
    dense_rows: usize,
    forward: &mut PreparedLlamaForward,
    shape_variants: &mut [PreparedLlamaBatchShape],
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    device: &mut BatchDeviceInput,
    gathered_logits: &mut Option<CudaDeviceBuffer>,
    greedy_results: &mut Option<CudaDeviceBuffer>,
    host_input: &mut BatchHostInput,
    produce_greedy_tokens: bool,
    dispatch_disposition: &mut BatchDispatchDisposition,
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    let bounds = config.metadata();
    let active = packed.total_input_tokens();
    if dense_rows != forward.plan.sequence_length()
        && !shape_variants
            .iter()
            .any(|shape| shape.dense_rows == dense_rows)
    {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "shape_variants",
            reason: "selected dense-row bucket was not prepared",
        });
    }
    let metadata_site = ExecutionSite::global(LlamaOp::BatchMetadataUpload);
    let host_batch = PackedBatchHostV1::new(
        packed.block_row_offsets(),
        packed.physical_block_ids(),
        packed.valid_tokens(),
        packed.row_sequence_slots(),
        packed.position_ids(),
        usize_u64(
            bounds.physical_block_count(),
            LlamaBatchExecutorResource::PhysicalBlockIds,
        )?,
    )
    .map_err(|source| batch_cuda(metadata_site, source))?;
    let packed_layout = match (&mut *host_input, &mut *device, config.metadata_transport()) {
        (
            BatchHostInput::PerOperation(host),
            BatchDeviceInput::PerOperation(device),
            BatchMetadataTransport::Synchronous,
        ) => {
            host.padded_tokens[..dense_rows].fill(0);
            host.padded_tokens[..active].copy_from_slice(packed.input_token_ids());
            upload_batch_tokens(forward, &host.padded_tokens[..dense_rows], stream)?;
            encode_u32(packed.block_row_offsets(), &mut host.sequence_block_offsets);
            encode_u32(packed.physical_block_ids(), &mut host.physical_block_ids);
            encode_u16(packed.valid_tokens(), &mut host.valid_tokens);
            encode_u32(packed.row_sequence_slots(), &mut host.row_sequence_slots);
            encode_u32(packed.position_ids(), &mut host.row_positions);
            encode_u32(
                packed.output_token_indices(),
                &mut host.output_token_indices,
            );

            upload_prefix(
                &mut device.sequence_block_offsets,
                &host.sequence_block_offsets,
                packed.block_row_offsets().len() * U32_BYTES,
                &mut forward.io_staging,
                stream,
                metadata_site,
            )?;
            upload_prefix(
                &mut device.physical_block_ids,
                &host.physical_block_ids,
                packed.physical_block_ids().len() * U32_BYTES,
                &mut forward.io_staging,
                stream,
                metadata_site,
            )?;
            upload_prefix(
                &mut device.valid_tokens,
                &host.valid_tokens,
                packed.valid_tokens().len() * U16_BYTES,
                &mut forward.io_staging,
                stream,
                metadata_site,
            )?;
            upload_prefix(
                &mut device.row_sequence_slots,
                &host.row_sequence_slots,
                active * U32_BYTES,
                &mut forward.io_staging,
                stream,
                metadata_site,
            )?;
            upload_prefix(
                &mut device.row_positions,
                &host.row_positions,
                active * U32_BYTES,
                &mut forward.io_staging,
                stream,
                metadata_site,
            )?;
            if packed.output_count() != 0 {
                let output_indices = device.output_token_indices.as_mut().ok_or(
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "output_token_indices",
                        reason: "non-empty output has no cold-prepared device index buffer",
                    },
                )?;
                upload_prefix(
                    output_indices,
                    &host.output_token_indices,
                    packed.output_count() * U32_BYTES,
                    &mut forward.io_staging,
                    stream,
                    metadata_site,
                )?;
            }
            None
        }
        (
            BatchHostInput::IterationBatch(host),
            BatchDeviceInput::IterationBatch { slab },
            BatchMetadataTransport::PackedAsync,
        ) => {
            let layout = PackedIterationLayout::for_batch(&packed, dense_rows)?;
            layout.validate_capacity(host.bytes.len())?;
            layout.validate_u64_capacity(slab.byte_len())?;
            pack_iteration_input(&packed, dense_rows, layout, &mut host.bytes)?;
            host.pinned
                .write(0, &host.bytes[..layout.total_bytes])
                .map_err(|source| batch_cuda(metadata_site, source))?;
            Some(layout)
        }
        _ => {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                reason: "cold-prepared host/device input transport does not match configuration",
            });
        }
    };

    let mut execute_iteration_body = |batch: PackedBatchV1<'_>,
                                      token_ids: Option<CudaBufferSpan<'_>>,
                                      output_indices: Option<CudaBufferSpan<'_>>,
                                      stream: &mut dyn CudaExecutionStream|
     -> LlamaBatchExecutorResult<()> {
        let rms_norm_profile = forward.rms_norm_profile();
        let PreparedLlamaForward {
            plan: maximum_plan,
            weights,
            gemms: maximum_gemms,
            buffers,
            ..
        } = forward;
        let (plan, gemms) = if dense_rows == maximum_plan.sequence_length() {
            (&*maximum_plan, maximum_gemms)
        } else {
            let shape = shape_variants
                .iter_mut()
                .find(|shape| shape.dense_rows == dense_rows)
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "shape_variants",
                    reason: "selected dense-row bucket was not prepared",
                })?;
            (&shape.plan, &mut shape.gemms)
        };
        execute_fixed_graph(
            plan,
            weights,
            gemms,
            buffers,
            config.residual_norm_implementation(),
            rms_norm_profile,
            config.ragged_attention_reduction_profile(),
            config.ragged_attention_implementation(),
            layout,
            key_cache,
            value_cache,
            rope_cos,
            rope_sin,
            token_ids,
            batch,
            packed.position_ids(),
            stream,
        )?;

        if packed.output_count() != 0 {
            dispatch_output_primitives(
                OutputPrimitiveDispatch {
                    logits: &buffers.logits,
                    logits_byte_len: plan.workspace_spec().logits_bytes(),
                    vocabulary_size: plan.dimensions().vocabulary_size(),
                    dense_rows,
                    output_indices,
                    output_indices_host: packed.output_token_indices(),
                    output_count: packed.output_count(),
                    gathered_logits,
                    greedy_results,
                    produce_greedy_tokens: produce_greedy_tokens,
                },
                stream,
            )?;
        }
        Ok(())
    };

    match (
        config.execution_completion_implementation(),
        config.metadata_transport(),
    ) {
        (ExecutionCompletionImplementation::PerOperation, BatchMetadataTransport::Synchronous) => {
            let BatchDeviceInput::PerOperation(device) = &*device else {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "metadata_transport",
                    reason: "synchronous execution has no per-operation device metadata",
                });
            };
            let views = per_operation_device_views(host_batch, device, &packed, metadata_site)?;
            execute_iteration_body(
                views.batch,
                views.token_ids,
                views.output_token_indices,
                stream,
            )
        }
        (
            ExecutionCompletionImplementation::IterationBatch,
            BatchMetadataTransport::Synchronous,
        ) => {
            let BatchDeviceInput::PerOperation(device) = &*device else {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "metadata_transport",
                    reason: "synchronous execution has no per-operation device metadata",
                });
            };
            let views = per_operation_device_views(host_batch, device, &packed, metadata_site)?;
            execute_iteration_command_batch(stream, dispatch_disposition, |commands| {
                execute_iteration_body(
                    views.batch,
                    views.token_ids,
                    views.output_token_indices,
                    commands,
                )
            })
        }
        (
            ExecutionCompletionImplementation::IterationBatch,
            BatchMetadataTransport::PackedAsync,
        ) => {
            let layout = packed_layout.ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                field: "packed_iteration_layout",
                reason: "packed async input was not prepared before command-batch begin",
            })?;
            let copy_byte_len = usize_u64(
                layout.total_bytes,
                LlamaBatchExecutorResource::PackedIterationInput,
            )?;
            match (&*device, &*host_input) {
                (BatchDeviceInput::IterationBatch { slab }, BatchHostInput::IterationBatch(_)) => {
                    // Resolve every dtype, alignment, range, host-shape, and
                    // output-view check before the first H2D submission. The
                    // same immutable descriptors are rebound after enqueue;
                    // no shape or offset can change inside the command batch.
                    let _preflight_views =
                        packed_device_views(host_batch, slab, &packed, layout, metadata_site)?;
                }
                _ => {
                    return Err(LlamaBatchExecutorError::InvalidConfiguration {
                        field: "metadata_transport",
                        reason: "packed async execution has no packed host/device slab",
                    });
                }
            }
            execute_iteration_command_batch(stream, dispatch_disposition, |commands| {
                match (&mut *device, &*host_input) {
                    (
                        BatchDeviceInput::IterationBatch { slab },
                        BatchHostInput::IterationBatch(host),
                    ) => {
                        let copy_result = slab
                            .copy_from_pinned_in_command_batch(
                                0,
                                &host.pinned,
                                0,
                                copy_byte_len,
                                commands,
                            )
                            .map_err(|source| batch_cuda(metadata_site, source));
                        match copy_result {
                            Err(error) => Err(error),
                            Ok(()) => {
                                match packed_device_views(
                                    host_batch,
                                    slab,
                                    &packed,
                                    layout,
                                    metadata_site,
                                ) {
                                    Err(error) => Err(error),
                                    Ok(views) => execute_iteration_body(
                                        views.batch,
                                        views.token_ids,
                                        views.output_token_indices,
                                        commands,
                                    ),
                                }
                            }
                        }
                    }
                    _ => Err(LlamaBatchExecutorError::InvalidConfiguration {
                        field: "metadata_transport",
                        reason: "packed async execution has no packed host/device slab",
                    }),
                }
            })
        }
        (ExecutionCompletionImplementation::PerOperation, BatchMetadataTransport::PackedAsync) => {
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                reason: "packed async metadata requires iteration-batch completion",
            })
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::large_types_passed_by_value,
    clippy::similar_names
)]
fn execute_fixed_graph<S: CudaExecutionStream + ?Sized>(
    plan: &LlamaExecutionPlan,
    weights: &CudaUploadedWeights,
    gemms: &mut GemmPlans,
    buffers: &mut ForwardBuffers,
    residual_norm_implementation: ResidualNormImplementation,
    rms_norm_profile: LlamaRmsNormProfile,
    attention_reduction_profile: AttentionReductionProfile,
    attention_implementation: RaggedAttentionImplementation,
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    token_ids: Option<CudaBufferSpan<'_>>,
    batch: PackedBatchV1<'_>,
    positions_host: &[u32],
    stream: &mut S,
) -> LlamaBatchExecutorResult<()> {
    let dense_rows = usize_u64(
        plan.sequence_length(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let dimensions = plan.dimensions();
    let hidden = usize_u64(
        dimensions.hidden_size(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let key_value_width = usize_u64(
        dimensions.key_value_width(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let query_heads = usize_u64(
        dimensions.query_heads(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let key_value_heads = usize_u64(
        dimensions.key_value_heads(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let head_size = usize_u64(
        dimensions.head_dimension(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let max_positions =
        absolute_rope_position_count(rope_cos.byte_len(), dimensions.head_dimension())?;
    let hidden_elements = plan.workspace_spec().hidden_buffer_bytes() / BF16_BYTES;
    let intermediate_elements = plan.workspace_spec().intermediate_buffer_bytes() / BF16_BYTES;

    let embedding_site = ExecutionSite::global(LlamaOp::Embedding);
    let embedding_weight = weight_span(weights, plan.embedding_weight(), embedding_site)?;
    {
        let token_ids = match token_ids {
            Some(token_ids) => token_ids,
            None => span(
                &buffers.token_ids,
                CudaDType::U32,
                plan.workspace_spec().token_ids_bytes(),
                embedding_site,
            )?,
        };
        let mut params = EmbeddingParams {
            table: embedding_weight,
            token_ids,
            output: span_mut(
                &mut buffers.hidden_current,
                CudaDType::BF16,
                plan.workspace_spec().hidden_buffer_bytes(),
                embedding_site,
            )?,
            error_scratch: span_mut(
                &mut buffers.embedding_error_scratch,
                CudaDType::U8,
                plan.workspace_spec().embedding_error_scratch_bytes(),
                embedding_site,
            )?,
            token_count: dense_rows,
            vocabulary_size: usize_u64(
                dimensions.vocabulary_size(),
                LlamaBatchExecutorResource::GatheredLogits,
            )?,
            hidden_size: hidden,
        };
        embedding(&mut params, stream).map_err(|source| {
            LlamaBatchExecutorError::Forward(LlamaForwardError::Embedding {
                site: embedding_site,
                source,
            })
        })?;
    }

    for layer in plan.layers() {
        let layer_index = layer.index();
        let input_norm_site = ExecutionSite::layer(layer_index, LlamaOp::InputNorm);
        let input_norm_weight = weight_span(weights, layer.input_norm_weight(), input_norm_site)?;
        {
            let mut params = RmsNormParams {
                input: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    input_norm_site,
                )?,
                weight: input_norm_weight,
                output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    input_norm_site,
                )?,
                row_count: dense_rows,
                hidden_size: hidden,
                epsilon: layer.input_norm_epsilon(),
            };
            execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
                .map_err(|source| batch_cuda(input_norm_site, source))?;
        }

        let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
        execute_gemm(
            &mut gemms.hidden,
            &buffers.hidden_norm,
            weight_span(weights, layer.query_weight(), query_site)?,
            &mut buffers.hidden_projection,
            &mut buffers.gemm_workspace,
            stream,
            query_site,
        )?;
        execute_projection_bias(
            weights,
            layer.query_bias(),
            &mut buffers.hidden_projection,
            dense_rows,
            hidden,
            stream,
            query_site,
        )?;
        let key_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
        execute_gemm(
            &mut gemms.key_value,
            &buffers.hidden_norm,
            weight_span(weights, layer.key_weight(), key_site)?,
            &mut buffers.key_raw,
            &mut buffers.gemm_workspace,
            stream,
            key_site,
        )?;
        execute_projection_bias(
            weights,
            layer.key_bias(),
            &mut buffers.key_raw,
            dense_rows,
            key_value_width,
            stream,
            key_site,
        )?;
        let value_site = ExecutionSite::layer(layer_index, LlamaOp::ValueProjection);
        execute_gemm(
            &mut gemms.key_value,
            &buffers.hidden_norm,
            weight_span(weights, layer.value_weight(), value_site)?,
            &mut buffers.value_raw,
            &mut buffers.gemm_workspace,
            stream,
            value_site,
        )?;
        execute_projection_bias(
            weights,
            layer.value_bias(),
            &mut buffers.value_raw,
            dense_rows,
            key_value_width,
            stream,
            value_site,
        )?;

        let query_rope_site = ExecutionSite::layer(layer_index, LlamaOp::QueryRope);
        {
            let mut params = IndexedRopeParams {
                input: span(
                    &buffers.hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    query_rope_site,
                )?,
                cos: CudaBufferSpan::new(rope_cos, CudaDType::F32, 0, rope_cos.byte_len())
                    .map_err(|source| batch_cuda(query_rope_site, source))?,
                sin: CudaBufferSpan::new(rope_sin, CudaDType::F32, 0, rope_sin.byte_len())
                    .map_err(|source| batch_cuda(query_rope_site, source))?,
                positions: batch.device_row_positions(),
                positions_host,
                output: span_mut(
                    &mut buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    query_rope_site,
                )?,
                head_count: query_heads,
                head_size,
                rotary_dimension: head_size,
                table_position_count: max_positions,
            };
            indexed_rope(&mut params, stream)
                .map_err(|source| batch_cuda(query_rope_site, source))?;
        }
        let key_rope_site = ExecutionSite::layer(layer_index, LlamaOp::KeyRope);
        {
            let mut params = IndexedRopeParams {
                input: span(
                    &buffers.key_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    key_rope_site,
                )?,
                cos: CudaBufferSpan::new(rope_cos, CudaDType::F32, 0, rope_cos.byte_len())
                    .map_err(|source| batch_cuda(key_rope_site, source))?,
                sin: CudaBufferSpan::new(rope_sin, CudaDType::F32, 0, rope_sin.byte_len())
                    .map_err(|source| batch_cuda(key_rope_site, source))?,
                positions: batch.device_row_positions(),
                positions_host,
                output: span_mut(
                    &mut buffers.key_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    key_rope_site,
                )?,
                head_count: key_value_heads,
                head_size,
                rotary_dimension: head_size,
                table_position_count: max_positions,
            };
            indexed_rope(&mut params, stream)
                .map_err(|source| batch_cuda(key_rope_site, source))?;
        }

        let cache_site = ExecutionSite::layer(layer_index, LlamaOp::KvCacheWrite);
        let layer_offset = layout.layer_byte_offset(layer_index).ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "KV layer offset",
                reason: "decoder layer lies outside the prepared KV layout",
            },
        )?;
        {
            let mut params = RaggedPagedKvCacheWriteParams {
                key_source: span(
                    &buffers.key_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    cache_site,
                )?,
                value_source: span(
                    &buffers.value_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    cache_site,
                )?,
                key_pool: CudaBufferSpanMut::new(
                    key_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(cache_site, source))?,
                value_pool: CudaBufferSpanMut::new(
                    value_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(cache_site, source))?,
                batch,
                key_value_head_count: key_value_heads,
                head_size,
            };
            ragged_paged_kv_cache_write(&mut params, stream)
                .map_err(|source| batch_cuda(cache_site, source))?;
        }

        let attention_site = ExecutionSite::layer(layer_index, LlamaOp::RaggedPagedAttention);
        {
            let mut params = RaggedPagedAttentionParams {
                query: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_site,
                )?,
                key_pool: CudaBufferSpan::new(
                    key_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                value_pool: CudaBufferSpan::new(
                    value_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                output: span_mut(
                    &mut buffers.hidden_context,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_site,
                )?,
                batch,
                query_head_count: query_heads,
                key_value_head_count: key_value_heads,
                head_size,
                output_row_count: dense_rows,
                scale: 1.0 / (head_size as f32).sqrt(),
            };
            match attention_reduction_profile {
                AttentionReductionProfile::CanonicalV1 => match attention_implementation {
                    RaggedAttentionImplementation::Legacy => {
                        ragged_paged_attention(&mut params, stream)
                    }
                    RaggedAttentionImplementation::GroupedHeads => {
                        grouped_ragged_paged_attention(&mut params, stream)
                    }
                },
                AttentionReductionProfile::FixedContiguous37BalancedV1 => {
                    fixed37_ragged_paged_attention(&mut params, stream)
                }
            }
            .map_err(|source| batch_cuda(attention_site, source))?;
        }

        let output_site = ExecutionSite::layer(layer_index, LlamaOp::OutputProjection);
        execute_gemm(
            &mut gemms.hidden,
            &buffers.hidden_context,
            weight_span(weights, layer.output_weight(), output_site)?,
            &mut buffers.hidden_projection,
            &mut buffers.gemm_workspace,
            stream,
            output_site,
        )?;
        execute_projection_bias(
            weights,
            layer.output_bias(),
            &mut buffers.hidden_projection,
            dense_rows,
            hidden,
            stream,
            output_site,
        )?;
        let attention_residual_site = ExecutionSite::layer(layer_index, LlamaOp::AttentionResidual);
        let post_norm_site = ExecutionSite::layer(layer_index, LlamaOp::PostAttentionNorm);
        match residual_norm_implementation {
            ResidualNormImplementation::Separate => {
                let mut residual = ResidualAddParams {
                    left: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    right: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    element_count: hidden_elements,
                };
                residual_add(&mut residual, stream)
                    .map_err(|source| batch_cuda(attention_residual_site, source))?;
                let mut norm = RmsNormParams {
                    input: span(
                        &buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    weight: weight_span(
                        weights,
                        layer.post_attention_norm_weight(),
                        post_norm_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    row_count: dense_rows,
                    hidden_size: hidden,
                    epsilon: layer.post_attention_norm_epsilon(),
                };
                execute_profile_rms_norm(rms_norm_profile, &mut norm, stream)
                    .map_err(|source| batch_cuda(post_norm_site, source))?;
            }
            ResidualNormImplementation::Fused => {
                let mut fused = ResidualRmsNormParams {
                    left: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    right: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    weight: weight_span(
                        weights,
                        layer.post_attention_norm_weight(),
                        post_norm_site,
                    )?,
                    residual_output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    normalized_output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    row_count: dense_rows,
                    hidden_size: hidden,
                    epsilon: layer.post_attention_norm_epsilon(),
                };
                execute_profile_residual_rms_norm(rms_norm_profile, &mut fused, stream)
                    .map_err(|source| batch_cuda(post_norm_site, source))?;
            }
        }
        let gate_site = ExecutionSite::layer(layer_index, LlamaOp::GateProjection);
        execute_gemm(
            &mut gemms.intermediate,
            &buffers.hidden_norm,
            weight_span(weights, layer.gate_weight(), gate_site)?,
            &mut buffers.gate_raw,
            &mut buffers.gemm_workspace,
            stream,
            gate_site,
        )?;
        let up_site = ExecutionSite::layer(layer_index, LlamaOp::UpProjection);
        execute_gemm(
            &mut gemms.intermediate,
            &buffers.hidden_norm,
            weight_span(weights, layer.up_weight(), up_site)?,
            &mut buffers.up_raw,
            &mut buffers.gemm_workspace,
            stream,
            up_site,
        )?;
        let silu_site = ExecutionSite::layer(layer_index, LlamaOp::Silu);
        {
            let mut params = SiluParams {
                input: span(
                    &buffers.gate_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    silu_site,
                )?,
                output: span_mut(
                    &mut buffers.gate_activated,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    silu_site,
                )?,
                element_count: intermediate_elements,
            };
            silu(&mut params, stream).map_err(|source| batch_cuda(silu_site, source))?;
        }
        let gated_site = ExecutionSite::layer(layer_index, LlamaOp::GatedMultiply);
        {
            let mut params = GatedMultiplyParams {
                activated_gate: span(
                    &buffers.gate_activated,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                up: span(
                    &buffers.up_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                output: span_mut(
                    &mut buffers.gated_product,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                element_count: intermediate_elements,
            };
            gated_multiply(&mut params, stream).map_err(|source| batch_cuda(gated_site, source))?;
        }
        let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
        execute_gemm(
            &mut gemms.down,
            &buffers.gated_product,
            weight_span(weights, layer.down_weight(), down_site)?,
            &mut buffers.hidden_current,
            &mut buffers.gemm_workspace,
            stream,
            down_site,
        )?;
        let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);
        {
            let mut params = ResidualAddParams {
                left: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                right: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                output: span_mut(
                    &mut buffers.hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                element_count: hidden_elements,
            };
            residual_add(&mut params, stream)
                .map_err(|source| batch_cuda(mlp_residual_site, source))?;
        }
        mem::swap(&mut buffers.hidden_current, &mut buffers.hidden_projection);
    }

    let final_norm_site = ExecutionSite::global(LlamaOp::FinalNorm);
    {
        let mut params = RmsNormParams {
            input: span(
                &buffers.hidden_current,
                CudaDType::BF16,
                plan.workspace_spec().hidden_buffer_bytes(),
                final_norm_site,
            )?,
            weight: weight_span(weights, plan.final_norm_weight(), final_norm_site)?,
            output: span_mut(
                &mut buffers.hidden_norm,
                CudaDType::BF16,
                plan.workspace_spec().hidden_buffer_bytes(),
                final_norm_site,
            )?,
            row_count: dense_rows,
            hidden_size: hidden,
            epsilon: plan.final_norm_epsilon(),
        };
        execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
            .map_err(|source| batch_cuda(final_norm_site, source))?;
    }
    let lm_head_site = ExecutionSite::global(LlamaOp::LmHead);
    execute_gemm(
        &mut gemms.lm_head,
        &buffers.hidden_norm,
        weight_span(weights, plan.lm_head_weight(), lm_head_site)?,
        &mut buffers.logits,
        &mut buffers.gemm_workspace,
        stream,
        lm_head_site,
    )?;
    Ok(())
}

fn upload_batch_tokens(
    forward: &mut PreparedLlamaForward,
    token_ids: &[u32],
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    if token_ids.len() == forward.plan.sequence_length() {
        return forward.upload_tokens(token_ids, stream).map_err(Into::into);
    }
    let byte_len = checked_byte_len(
        token_ids.len(),
        U32_BYTES,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    if byte_len > forward.token_bytes.len() {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "dense_rows",
            reason: "selected token prefix exceeds the shared maximum buffer",
        });
    }
    encode_u32(token_ids, &mut forward.token_bytes[..byte_len]);
    forward.tokens_ready = false;
    forward.output_ready = false;
    let site = ExecutionSite::global(LlamaOp::Embedding);
    match forward.buffers.token_ids.upload_from_slice(
        0,
        &forward.token_bytes[..byte_len],
        &mut forward.io_staging,
        stream,
    ) {
        Ok(()) => {
            forward.tokens_ready = true;
            Ok(())
        }
        Err(source) => {
            poison_for_cuda_error(&mut forward.poisoned, &source);
            Err(batch_cuda(site, source))
        }
    }
}

fn upload_prefix(
    destination: &mut CudaDeviceBuffer,
    source: &[u8],
    byte_len: usize,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    stream: &mut CudaStream,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    destination
        .upload_from_slice(0, &source[..byte_len], staging, stream)
        .map_err(|source| batch_cuda(site, source))
}

#[cfg(test)]
mod tests {
    use super::BatchDispatchDisposition;

    #[test]
    fn dispatch_disposition_distinguishes_preflight_from_unknown_mutation() {
        let mut disposition = BatchDispatchDisposition::PreDispatch;
        assert!(!disposition.mutation_may_have_occurred());

        disposition = BatchDispatchDisposition::CommandSubmissionStarted;
        assert!(disposition.mutation_may_have_occurred());
    }
}
