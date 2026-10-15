//! Execution dispatch for the Llama batch executor.
//!
//! This module consumes cold-prepared, borrowed resources to perform one
//! iteration: metadata transport, command-batch lifecycle, exact forward
//! dispatch, and output primitives. The enclosing owner retains public API,
//! resource lifetime, output publication, failure-state policy, allocation,
//! and explicit close ordering.

#[cfg(feature="cuda-cublas-gemm-probe")]
use riley_cuda::{hugging_face_qwen_head128_rms_norm,qkv_split_bias_rope_hf_head128};
#[cfg(feature="cuda-cublas-gemm-probe")]
use riley_cuda::hf_pack_separate_qkv;
use std::mem;

use riley_cuda::{
    AttentionReductionProfile, Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut,
    CudaCommandStream, CudaDType, CudaDeviceBuffer, CudaExecutionStream, CudaStepGraph, CudaStream,
    DecodePartialReductionOrder, EmbeddingParams, GatedMultiplyParams, IndexedRopeParams,
    MergedGatedMultiplyParams, NativeBf16RaggedPagedSplitGqaD128TwoStageParams, PackedBatchHostV1,
    PackedBatchV1, QkvRopeParams, QkvSplitBiasParams, RaggedPagedAttentionParams,
    RaggedPagedKvCacheWriteParams, ResidualAddParams, ResidualRmsNormParams, RmsNormParams,
    RowGatherParams, SiluParams, abort_step_capture, begin_step_capture, deterministic_bf16_argmax,
    deterministic_bf16_argmax_bounded, embedding, embedding_deferred_report, end_step_capture,
    fixed37_ragged_paged_attention, gated_multiply, grouped_ragged_paged_attention, indexed_rope,
    launch_step_graph, native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma,
    native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma,
    native_bf16_ragged_paged_split_gqa_d128_two_stage_shared_kv, qkv_split_bias,
    qkv_split_bias_rope, ragged_paged_attention, ragged_paged_kv_cache_write, residual_add,
    row_gather, set_step_capture_dynamic_counts, silu, silu_gated_multiply,
    silu_gated_multiply_merged_rows, validate_replayed_decode_step,
};

use super::super::batch::{LlamaBatchRowKind, LlamaPackedBatchMetadata};
use super::super::forward::{
    ForwardBuffers, GemmPlans, LlamaForwardError, LlamaRmsNormProfile, PreparedLlamaForward,
    PreparedLlamaGemm, execute_gemm_with_row_groups, execute_profile_residual_rms_norm, execute_profile_rms_norm, execute_profile_rms_norm_grouped,
    execute_projection_bias, execute_projection_with_row_groups, poison_for_cuda_error,
    span, span_mut, weight_span,
};
use super::super::{ExecutionSite, LlamaExecutionPlan, LlamaOp};
use super::buffers::{
    BatchDeviceInput, BatchHostInput, EMBEDDING_REPORT_BYTES, MAX_STAGED_COPIES,
    PerOperationDeviceMetadata, PerOperationHostWorkspace, STEP_COUNT_WORDS, StagedCopy,
    StagedTarget, U16_BYTES, U32_BYTES, UPLOADED_BLOCK_IDS, UPLOADED_BLOCK_OFFSETS,
    UPLOADED_OUTPUT_INDICES, UPLOADED_ROW_SLOTS, UploadedBytes,
};
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
use super::owner::NativeD128TwoStageWorkspace;
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
    // Everything is submitted; host work that does not feed this iteration
    // overlaps the GPU from here until the batch completes.
    crate::submit_hook::run();
    crate::host_trace::lap_here(crate::host_trace::SUBMIT);
    let completion_result = command_batch
        .finish()
        .map_err(|source| dispatch_cuda(completion_site, source));
    crate::host_trace::lap_here(crate::host_trace::WAIT);
    match completion_result {
        Err(error) => Err(error),
        Ok(()) => body_result,
    }
}

/// Output-row LM-head buckets up to this many rows are interchangeable in
/// cost, so the largest one that fits is used for every smaller output count.
const OUTPUT_ROW_LM_HEAD_WIDE_ROWS: usize = 32;

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
    pub(in crate::llama) greedy_candidate_count: Option<usize>,
    /// The LM head already wrote the output rows into `gathered_logits`.
    pub(in crate::llama) logits_already_gathered: bool,
}

/// Output-row LM head of a mixed step: where the output rows' normalized
/// hidden states come from and which anchored plan computes their logits.
pub(in crate::llama) struct OutputRowLmHead<'a> {
    pub(in crate::llama) gemm: &'a mut PreparedLlamaGemm,
    pub(in crate::llama) output_indices: CudaBufferSpan<'a>,
    pub(in crate::llama) output_indices_host: &'a [u32],
    pub(in crate::llama) gathered_logits: &'a mut CudaDeviceBuffer,
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
        greedy_candidate_count,
        logits_already_gathered,
    } = dispatch;
    let output_indices = output_indices.ok_or(LlamaBatchExecutorError::InvalidConfiguration {
        field: "output_token_indices",
        reason: "non-empty output has no cold-prepared device index buffer",
    })?;
    let site = ExecutionSite::global(LlamaOp::OutputGather);
    if !logits_already_gathered {
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
        match greedy_candidate_count {
            Some(count) => deterministic_bf16_argmax_bounded(
                &mut argmax,
                usize_u64(count, LlamaBatchExecutorResource::GreedyResults)?,
                stream,
            ),
            None => deterministic_bf16_argmax(&mut argmax, stream),
        }
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
    mut native_d128_two_stage_workspace: Option<&mut NativeD128TwoStageWorkspace>,
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
    crate::host_trace::lap_here(crate::host_trace::PACK);
    let packed_layout = match (&mut *host_input, &mut *device, config.metadata_transport()) {
        (
            BatchHostInput::PerOperation(host),
            BatchDeviceInput::PerOperation(device),
            BatchMetadataTransport::Synchronous,
        ) if host.staging.is_some() => {
            // Stage everything in pinned memory; the command batch copies it.
            let _ = device;
            stage_step_uploads(host, forward, &packed, config, dense_rows, metadata_site)?;
            None
        }
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

            upload_prefix_if_changed(
                &mut device.sequence_block_offsets,
                &host.sequence_block_offsets,
                packed.block_row_offsets().len() * U32_BYTES,
                &mut host.uploaded[UPLOADED_BLOCK_OFFSETS],
                &mut forward.io_staging,
                stream,
                metadata_site,
            )?;
            upload_prefix_if_changed(
                &mut device.physical_block_ids,
                &host.physical_block_ids,
                packed.physical_block_ids().len() * U32_BYTES,
                &mut host.uploaded[UPLOADED_BLOCK_IDS],
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
            upload_prefix_if_changed(
                &mut device.row_sequence_slots,
                &host.row_sequence_slots,
                active * U32_BYTES,
                &mut host.uploaded[UPLOADED_ROW_SLOTS],
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
            if config.decode_step_capture() {
                // Recorded steps read these two counts on the device. They
                // are copied only when they differ from the device's copy.
                let active_count = u32::try_from(active).map_err(|_| {
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "row_positions",
                        reason: "active row count does not fit u32",
                    }
                })?;
                let output_count = u32::try_from(packed.output_count()).map_err(|_| {
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "output_token_indices",
                        reason: "output row count does not fit u32",
                    }
                })?;
                let step_counts = [active_count, output_count];
                if host.uploaded_step_counts != Some(step_counts) {
                    host.uploaded_step_counts = None;
                    let mut counts = [0_u8; STEP_COUNT_WORDS * U32_BYTES];
                    counts[..U32_BYTES].copy_from_slice(&active_count.to_ne_bytes());
                    counts[U32_BYTES..].copy_from_slice(&output_count.to_ne_bytes());
                    device
                        .row_positions
                        .upload_from_slice(
                            (bounds.max_input_tokens() * U32_BYTES) as u64,
                            &counts,
                            &mut forward.io_staging,
                            stream,
                        )
                        .map_err(|source| batch_cuda(metadata_site, source))?;
                    host.uploaded_step_counts = Some(step_counts);
                }
            }
            if config.prompt_attention_mma() {
                // One kind word per launched token row, after the positions and
                // the step counts: 1 marks a prompt token that takes the
                // tensor-core producer. Only a step with such rows reads the
                // kinds, so only such a step copies them.
                let row_kinds_offset =
                    ((bounds.max_input_tokens() + STEP_COUNT_WORDS) * U32_BYTES) as u64;
                if has_tensor_core_prompt_rows(&packed, prompt_row_min_position(config)) {
                    for (row, kind) in host
                        .row_kinds
                        .chunks_exact_mut(U32_BYTES)
                        .take(active)
                        .enumerate()
                    {
                        kind.copy_from_slice(
                            &u32::from(is_tensor_core_prompt_row(
                                &packed,
                                row,
                                prompt_row_min_position(config),
                            ))
                            .to_ne_bytes(),
                        );
                    }
                    host.row_kinds[active * U32_BYTES..dense_rows * U32_BYTES].fill(0);
                    let row_kind_bytes = fill_generated_row_list(
                        config,
                        bounds,
                        &mut host.row_kinds,
                        active,
                        dense_rows,
                    )?;
                    device
                        .row_positions
                        .upload_from_slice(
                            row_kinds_offset,
                            &host.row_kinds[..row_kind_bytes],
                            &mut forward.io_staging,
                            stream,
                        )
                        .map_err(|source| batch_cuda(metadata_site, source))?;
                }
            }
            if packed.output_count() != 0 {
                let output_indices = device.output_token_indices.as_mut().ok_or(
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "output_token_indices",
                        reason: "non-empty output has no cold-prepared device index buffer",
                    },
                )?;
                upload_prefix_if_changed(
                    output_indices,
                    &host.output_token_indices,
                    packed.output_count() * U32_BYTES,
                    &mut host.uploaded[UPLOADED_OUTPUT_INDICES],
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

    crate::host_trace::lap_here(crate::host_trace::UPLOAD);
    // A step is recordable when its outputs are selected on the device. The
    // recorded kernels read the active and output row counts from the device,
    // so one graph per (dense shape, output LM-head shape) serves decode-only
    // and prefill-mixed steps alike.
    let step_capture_eligible = config.decode_step_capture()
        && produce_greedy_tokens
        && active != 0
        && active <= dense_rows;
    if let Some(prepared)=native_d128_two_stage_workspace.as_deref_mut().and_then(|w|w.materialized.as_mut()) {
        riley_cuda::prepare_materialized_ragged_attention_metadata(host_batch,&prepared.reference,
            &mut forward.buffers.hidden_context,&mut prepared.valid,&mut prepared.staging,
            &mut prepared.valid_host,&mut prepared.encoded_valid,&prepared.zero_host,stream)
            .map_err(|e|batch_cuda(metadata_site,e))?;
    }
    let mut execute_iteration_body = |batch: PackedBatchV1<'_>,
                                      token_ids: Option<CudaBufferSpan<'_>>,
                                      output_indices: Option<CudaBufferSpan<'_>>,
                                      in_command_batch: bool,
                                      step_counts: Option<CudaBufferSpan<'_>>,
                                      stream: &mut dyn CudaExecutionStream|
     -> LlamaBatchExecutorResult<()> {
        let rms_norm_profile = forward.rms_norm_profile();
        let PreparedLlamaForward {
            plan: maximum_plan,
            weights,
            gemms: maximum_gemms,
            batch_output_lm_head,
            buffers,
            captured_steps: maximum_captured_steps,
            captured_step_budget,
            ..
        } = forward;
        // A mixed step's output rows are fewer than its dense rows. Their LM
        // head can run on the smallest prepared bucket that holds them (and
        // fits the gathered-logits buffer) instead of on every dense row.
        let mut output_rows_bucket = 0_usize;
        let output_rows_index = if batch_output_lm_head.is_none() && config.output_row_lm_head()
            && packed.output_count() != 0
            && packed.output_count() < dense_rows
        {
            shape_variants
                .iter()
                .enumerate()
                .filter(|(_, shape)| {
                    shape.dense_rows >= packed.output_count()
                        && shape.dense_rows < dense_rows
                        && shape.dense_rows <= bounds.max_output_slots()
                })
                // Up to `OUTPUT_ROW_LM_HEAD_WIDE_ROWS` the largest fitting
                // bucket is taken, beyond it the smallest: the LM head reads
                // the same weights for any small row count, and one output
                // shape per dense shape means one recorded step instead of
                // one per small bucket.
                .min_by_key(|(_, shape)| {
                    if shape.dense_rows <= OUTPUT_ROW_LM_HEAD_WIDE_ROWS {
                        (false, OUTPUT_ROW_LM_HEAD_WIDE_ROWS - shape.dense_rows)
                    } else {
                        (true, shape.dense_rows)
                    }
                })
                .map(|(index, shape)| {
                    output_rows_bucket = shape.dense_rows;
                    index
                })
        } else {
            None
        };
        let (plan, gemms, mut captured_steps, output_lm_head) =
            if dense_rows == maximum_plan.sequence_length() {
                let output_lm_head = match output_rows_index {
                    Some(index) => Some(&mut shape_variants[index].gemms.lm_head),
                    None => None,
                };
                (
                    &*maximum_plan,
                    maximum_gemms,
                    Some(maximum_captured_steps),
                    output_lm_head,
                )
            } else {
                let current_index = shape_variants
                    .iter()
                    .position(|shape| shape.dense_rows == dense_rows)
                    .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                        field: "shape_variants",
                        reason: "selected dense-row bucket was not prepared",
                    })?;
                let (shape, output_lm_head) = match output_rows_index {
                    Some(index) if index < current_index => {
                        let (lower, upper) = shape_variants.split_at_mut(current_index);
                        (&mut upper[0], Some(&mut lower[index].gemms.lm_head))
                    }
                    Some(index) if index > current_index => {
                        let (lower, upper) = shape_variants.split_at_mut(index);
                        (&mut lower[current_index], Some(&mut upper[0].gemms.lm_head))
                    }
                    _ => (&mut shape_variants[current_index], None),
                };
                (
                    &shape.plan,
                    &mut shape.gemms,
                    Some(&mut shape.captured_steps),
                    output_lm_head,
                )
            };
        let output_lm_head = if packed.output_count() != 0 {
            batch_output_lm_head.as_mut().or(output_lm_head)
        } else { output_lm_head };
        let logits_already_gathered = output_lm_head.is_some();
        let output_rows_extent = output_rows_index.and_then(|index| {
            if logits_already_gathered {
                Some(output_rows_bucket)
            } else {
                let _ = index;
                None
            }
        });
        let output_row_lm_head = match (output_lm_head, output_indices, gathered_logits.as_mut()) {
            (Some(gemm), Some(output_indices), Some(gathered_logits)) => Some(OutputRowLmHead {
                gemm,
                output_indices,
                output_indices_host: packed.output_token_indices(),
                gathered_logits,
            }),
            (Some(_), _, _) => {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "gathered_logits",
                    reason: "output-row LM head has no cold-prepared output buffers",
                });
            }
            (None, _, _) => None,
        };
        if !(step_capture_eligible && in_command_batch && step_counts.is_some()) {
            captured_steps = None;
        }
        // The recorded launch extents depend on the dense shape (this plan)
        // and on where the LM head runs: nowhere read (no outputs), on every
        // dense row, or on one output-row bucket.
        let (capture_key, output_extent_rows) = if packed.output_count() == 0 {
            (0_usize, 1_usize)
        } else {
            match output_rows_extent {
                Some(rows) => (rows + 1, rows),
                None => (1, dense_rows),
            }
        };
        // A step without prompt rows does not launch the tensor-core prompt
        // producer at all, so it is recorded separately from one that does.
        let prompt_rows_in_step = config.prompt_attention_mma()
            && has_tensor_core_prompt_rows(&packed, prompt_row_min_position(config));
        let capture_key = capture_key * 2 + usize::from(prompt_rows_in_step);
        let recorded = captured_steps.as_deref().and_then(|steps| {
            steps
                .iter()
                .find(|(key, _)| *key == capture_key)
                .map(|(_, graph)| graph)
        });
        let replaying = recorded.is_some();
        let capturing = captured_steps.is_some() && !replaying && *captured_step_budget != 0;
        crate::host_trace::note_step_kind(capturing, replaying);
        // Staged step uploads also move the embedding into the recorded step.
        let embedding_in_step = config.staged_step_uploads() && (capturing || replaying);
        let step_capture = match (recorded, step_counts) {
            (Some(graph), _) => StepCaptureMode::Replay(graph, packed.output_token_indices()),
            (None, Some(counts)) if capturing => StepCaptureMode::Capture {
                counts,
                output_rows: output_extent_rows,
            },
            _ => StepCaptureMode::Disabled,
        };
        let fixed = execute_fixed_graph(
            plan,
            weights,
            gemms,
            buffers,
            config.residual_norm_implementation(),
            rms_norm_profile,
            config.ragged_attention_reduction_profile(),
            config.ragged_attention_implementation(),
            config.d128_global_merge(),
            config.d128_tensor_core_producer(),
            prompt_rows_in_step.then(|| {
                let kinds_first_word = (bounds.max_input_tokens() + STEP_COUNT_WORDS) as u64;
                // The generated-row list sits behind the kinds region.
                let generated_rows = config.dense_prompt_attention().then(|| {
                    (
                        kinds_first_word + bounds.max_input_tokens() as u64,
                        bounds.max_rows() as u64,
                    )
                });
                (kinds_first_word, generated_rows)
            }),
            config.fused_silu_multiply(),
            config.fused_qkv_rope(),
            config.fused_kv_write(),
            config.fused_mlp_residual_norm(),
            embedding_in_step,
            native_d128_two_stage_workspace.as_deref_mut(),
            layout,
            key_cache,
            value_cache,
            rope_cos,
            rope_sin,
            token_ids,
            batch,
            packed.position_ids(),
            step_capture,
            output_row_lm_head,
            stream,
        );
        // Staged step uploads: the batch also copies this iteration's results
        // back into the pinned staging buffer, after everything that produces
        // them and outside any recording.
        let copy_staged_outputs = |greedy_results: &Option<CudaDeviceBuffer>,
                                   embedding_error_scratch: &CudaDeviceBuffer,
                                   stream: &mut dyn CudaExecutionStream|
         -> LlamaBatchExecutorResult<()> {
            let BatchHostInput::PerOperation(host) = &*host_input else {
                return Ok(());
            };
            let Some(staging) = host.staging.as_ref() else {
                return Ok(());
            };
            let site = ExecutionSite::global(LlamaOp::OutputGather);
            if produce_greedy_tokens && packed.output_count() != 0 {
                let byte_len = greedy_result_bytes(packed.output_count())?;
                let results = greedy_results.as_ref().ok_or(
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "greedy_results",
                        reason: "non-empty output has no cold-prepared greedy result buffer",
                    },
                )?;
                results
                    .copy_to_pinned_in_active_command_batch(
                        0,
                        staging,
                        host.staged_output_offset,
                        byte_len as u64,
                        stream,
                    )
                    .map_err(|source| batch_cuda(site, source))?;
                host.staged_greedy_bytes.set(byte_len);
            }
            if embedding_in_step {
                embedding_error_scratch
                    .copy_to_pinned_in_active_command_batch(
                        0,
                        staging,
                        host.staged_report_offset,
                        EMBEDDING_REPORT_BYTES as u64,
                        stream,
                    )
                    .map_err(|source| batch_cuda(site, source))?;
                host.staged_report.set(true);
            }
            Ok(())
        };
        if replaying {
            fixed?;
            return if in_command_batch {
                copy_staged_outputs(&*greedy_results, &buffers.embedding_error_scratch, stream)
            } else {
                Ok(())
            };
        }
        if let Err(error) = fixed {
            if capturing {
                let _ = abort_step_capture(stream);
            }
            return Err(error);
        }

        let outputs = if packed.output_count() != 0 {
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
                    greedy_candidate_count: config.greedy_candidate_count(),
                    logits_already_gathered,
                },
                stream,
            )
        } else {
            Ok(())
        };
        if !capturing {
            outputs?;
            return if in_command_batch {
                copy_staged_outputs(&*greedy_results, &buffers.embedding_error_scratch, stream)
            } else {
                Ok(())
            };
        }
        // Nothing recorded has run yet: finish the capture, then launch it
        // for this step and keep it for later ones.
        let capture_site = ExecutionSite::global(LlamaOp::IterationCompletion);
        if let Err(error) = outputs {
            let _ = abort_step_capture(stream);
            return Err(error);
        }
        let graph = end_step_capture(stream).map_err(|source| batch_cuda(capture_site, source))?;
        launch_step_graph(stream, &graph).map_err(|source| batch_cuda(capture_site, source))?;
        if let Some(steps) = captured_steps {
            steps.push((capture_key, graph));
            *captured_step_budget -= 1;
        }
        copy_staged_outputs(&*greedy_results, &buffers.embedding_error_scratch, stream)
    };

    let result = match (
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
                false,
                None,
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
            let step_counts = if config.decode_step_capture() {
                Some(
                    CudaBufferSpan::new(
                        &device.row_positions,
                        CudaDType::U32,
                        (bounds.max_input_tokens() * U32_BYTES) as u64,
                        (STEP_COUNT_WORDS * U32_BYTES) as u64,
                    )
                    .map_err(|source| batch_cuda(metadata_site, source))?,
                )
            } else {
                None
            };
            // Staged step uploads: the embedding reads the executor's own
            // token buffer, filled by the first copy of the batch.
            let staged_token_ids = match (&*host_input, device.token_ids.as_ref()) {
                (BatchHostInput::PerOperation(host), Some(buffer)) if host.staging.is_some() => {
                    Some(
                        CudaBufferSpan::new(
                            buffer,
                            CudaDType::U32,
                            0,
                            (dense_rows * U32_BYTES) as u64,
                        )
                        .map_err(|source| batch_cuda(metadata_site, source))?,
                    )
                }
                (BatchHostInput::PerOperation(host), None) if host.staging.is_some() => {
                    return Err(LlamaBatchExecutorError::InvalidConfiguration {
                        field: "staged_step_uploads",
                        reason: "staged step uploads have no cold-prepared device token buffer",
                    });
                }
                _ => None,
            };
            execute_iteration_command_batch(stream, dispatch_disposition, |commands| {
                if let (BatchHostInput::PerOperation(host), true) =
                    (&*host_input, staged_token_ids.is_some())
                {
                    let copied = match host.staging.as_ref() {
                        Some(staging) => copy_staged_step_uploads(
                            device,
                            staging,
                            &host.staged[..host.staged_count],
                            commands,
                            metadata_site,
                        ),
                        None => Ok(()),
                    };
                    if let Err(error) = copied {
                        return Err(error);
                    }
                }
                execute_iteration_body(
                    views.batch,
                    staged_token_ids.or(views.token_ids),
                    views.output_token_indices,
                    true,
                    step_counts,
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
                                        false,
                                        None,
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
    };
    if result.is_ok() {
        check_staged_embedding_report(host_input)?;
    }
    result
}

/// Checks the embedding error report the command batch copied back into the
/// pinned staging buffer (the embedding ran inside the recorded step, so
/// nothing read it on the way). A token outside the vocabulary cannot pass the
/// scheduler's own bounds check; if the device still reports one the
/// iteration has run with that row left unembedded and is failed here.
fn check_staged_embedding_report(host_input: &mut BatchHostInput) -> LlamaBatchExecutorResult<()> {
    let site = ExecutionSite::global(LlamaOp::Embedding);
    let BatchHostInput::PerOperation(host) = host_input else {
        return Ok(());
    };
    if !host.staged_report.replace(false) {
        return Ok(());
    }
    let offset = host.staged_report_offset;
    let staging = host
        .staging
        .as_mut()
        .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
            field: "staged_step_uploads",
            reason: "a staged embedding report has no pinned buffer",
        })?;
    let mut report = [0_u8; EMBEDDING_REPORT_BYTES];
    staging
        .read(offset, &mut report)
        .map_err(|source| batch_cuda(site, source))?;
    let word = |start: usize| {
        u32::from_ne_bytes([
            report[start],
            report[start + 1],
            report[start + 2],
            report[start + 3],
        ])
    };
    // struct_size, code, then two u64 fields and a zero word.
    if word(0) as usize != EMBEDDING_REPORT_BYTES || word(4) != 0 {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "token_ids",
            reason: "the device embedding reported an out-of-range token or a malformed report",
        });
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::large_types_passed_by_value,
    clippy::similar_names
)]
/// How one iteration relates to a recorded pure-decode step.
#[derive(Clone, Copy)]
enum StepCaptureMode<'a> {
    /// Run every operation eagerly.
    Disabled,
    /// Run the embedding eagerly, then start recording the remaining
    /// operations; the caller ends the capture and launches the graph. The
    /// recorded kernels read the step's row counts from `counts` and launch
    /// over this plan's dense rows and `output_rows`.
    Capture {
        counts: CudaBufferSpan<'a>,
        output_rows: usize,
    },
    /// Run the embedding eagerly, then replay the recorded remainder.
    ///
    /// Carries the host mirror of the output row indices, which the replay
    /// validates in place of the skipped per-operation wrappers.
    Replay(&'a CudaStepGraph, &'a [u32]),
}

fn execute_fixed_graph<S: CudaExecutionStream + ?Sized>(
    plan: &LlamaExecutionPlan,
    weights: &CudaUploadedWeights,
    gemms: &mut GemmPlans,
    buffers: &mut ForwardBuffers,
    residual_norm_implementation: ResidualNormImplementation,
    rms_norm_profile: LlamaRmsNormProfile,
    attention_reduction_profile: AttentionReductionProfile,
    attention_implementation: RaggedAttentionImplementation,
    d128_global_merge: bool,
    d128_tensor_core_producer: bool,
    prompt_row_kinds_first_word: Option<(u64, Option<(u64, u64)>)>,
    fused_silu_multiply: bool,
    fused_qkv_rope: bool,
    fused_kv_write: bool,
    fused_mlp_residual_norm: bool,
    embedding_in_recorded_step: bool,
    mut native_d128_two_stage_workspace: Option<&mut NativeD128TwoStageWorkspace>,
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    token_ids: Option<CudaBufferSpan<'_>>,
    batch: PackedBatchV1<'_>,
    positions_host: &[u32],
    step_capture: StepCaptureMode<'_>,
    output_row_lm_head: Option<OutputRowLmHead<'_>>,
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
    // Width of the query projection and of the attention output; wider than
    // `hidden` for a few models, which then need the merged Q|K|V projection.
    let query_width = usize_u64(
        dimensions.query_width(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let query_buffer_bytes = plan.workspace_spec().query_buffer_bytes();
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
    // With the gather inside the recorded step its error report stays on the
    // device and the caller reads it after the iteration (see
    // `check_deferred_embedding_report`); a replayed step launches nothing here.
    let embedding_in_recorded_step =
        embedding_in_recorded_step && !matches!(step_capture, StepCaptureMode::Disabled);
    let run_embedding = |deferred_report: bool,
                         buffers: &mut ForwardBuffers,
                         stream: &mut S|
     -> LlamaBatchExecutorResult<()> {
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
        if deferred_report {
            embedding_deferred_report(&mut params, stream)
        } else {
            embedding(&mut params, stream)
        }
        .map_err(|source| {
            LlamaBatchExecutorError::Forward(LlamaForwardError::Embedding {
                site: embedding_site,
                source,
            })
        })
    };
    if !embedding_in_recorded_step {
        run_embedding(false, &mut *buffers, &mut *stream)?;
    }

    // An embedding that reads its error report back on the host has run
    // eagerly above; everything after it is recordable.
    let capture_site = ExecutionSite::global(LlamaOp::IterationCompletion);
    match step_capture {
        StepCaptureMode::Disabled => {}
        StepCaptureMode::Capture {
            counts,
            output_rows,
        } => {
            begin_step_capture(stream).map_err(|source| batch_cuda(capture_site, source))?;
            let output_rows = usize_u64(output_rows, LlamaBatchExecutorResource::HostWorkspace)?;
            if let Err(source) =
                set_step_capture_dynamic_counts(stream, counts, dense_rows, output_rows)
            {
                let _ = abort_step_capture(stream);
                return Err(batch_cuda(capture_site, source));
            }
            if embedding_in_recorded_step {
                if let Err(error) = run_embedding(true, &mut *buffers, &mut *stream) {
                    let _ = abort_step_capture(stream);
                    return Err(error);
                }
            }
        }
        StepCaptureMode::Replay(graph, output_indices_host) => {
            validate_replayed_decode_step(
                batch,
                positions_host,
                max_positions,
                output_indices_host,
                dense_rows,
                native_d128_two_stage_workspace
                    .as_deref()
                    .map(|workspace| workspace.partial_state_capacity),
            )
            .map_err(|source| batch_cuda(capture_site, source))?;
            return launch_step_graph(stream, graph)
                .map_err(|source| batch_cuda(capture_site, source));
        }
    }

    #[cfg(feature="cuda-cublas-gemm-probe")]
    let hf_wide_norm=matches!(hidden,2048|2560)&&native_d128_two_stage_workspace.as_ref().and_then(|w|w.materialized.as_ref()).is_some_and(|m|m.hf.is_some());
    #[cfg(not(feature="cuda-cublas-gemm-probe"))]let hf_wide_norm=false;
    if hf_wide_norm && residual_norm_implementation!=ResidualNormImplementation::Separate {
      return Err(LlamaBatchExecutorError::InvalidConfiguration{field:"hf_wide_norm",reason:"private request RMSNorm requires separate residual/norm"});
    }
    let norm_after_mlp = fused_mlp_residual_norm
        && residual_norm_implementation == ResidualNormImplementation::Fused
        && !plan.layers().is_empty();
    for (ordinal, layer) in plan.layers().iter().enumerate() {
        let (hidden_current, hidden_projection) = layer_hidden_roles(
            ordinal,
            &mut buffers.hidden_current,
            &mut buffers.hidden_projection,
        );
        let layer_index = layer.index();
        let input_norm_site = ExecutionSite::layer(layer_index, LlamaOp::InputNorm);
        let input_norm_weight = weight_span(weights, layer.input_norm_weight(), input_norm_site)?;
        // With the fused MLP residual norm the previous layer already wrote
        // this layer's normalized input.
        if !(norm_after_mlp && ordinal > 0) {
            let mut params = RmsNormParams {
                input: span(
                    hidden_current,
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
            execute_profile_rms_norm_grouped(rms_norm_profile,&mut params,stream,batch.host().row_sequence_slots(),hf_wide_norm)
                .map_err(|source| batch_cuda(input_norm_site, source))?;
        }

        let rope_in_split = fused_qkv_rope && gemms.merged_qkv_plan().is_some();
        if gemms.merged_qkv_plan().is_some() {
            // One GEMM over the concatenated Q|K|V weights. `gate_raw` is idle
            // until the MLP and is wide enough to hold the merged rows.
            let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            let merged_width = query_width + 2 * key_value_width;
            let merged_bytes = dense_rows * merged_width * 2;
            let missing_bias = LlamaForwardError::InvalidConfiguration {
                field: "projection_bias_mode",
                reason: "merged QKV projection requires Q/K/V biases, or no bias with a per-head query/key norm and the fused RoPE split",
            };
            // With biases: split + bias. Without them, with a per-head norm
            // (Qwen3): the split kernel normalizes every query and key head
            // instead and the bias spans carry the two norm scales. Without
            // either (Llama 3): the bias is a vector of negative zeros.
            let (query_bias, key_bias, value_bias, head_norm_epsilon) = match (
                layer.query_bias(),
                layer.key_bias(),
                layer.value_bias(),
                layer.query_key_norm(),
            ) {
                (Some(query_bias), Some(key_bias), Some(value_bias), None) => (
                    weight_span(weights, query_bias, query_site)?,
                    weight_span(weights, key_bias, query_site)?,
                    weight_span(weights, value_bias, query_site)?,
                    None,
                ),
                (None, None, None, Some((query_scale, key_scale, epsilon)))
                    if rope_in_split && head_size == 128 =>
                {
                    (
                        weight_span(weights, query_scale, query_site)?,
                        weight_span(weights, key_scale, query_site)?,
                        weight_span(weights, key_scale, query_site)?,
                        Some(epsilon),
                    )
                }
                (None, None, None, None) => {
                    let zeros = weights.negative_zero_bias().ok_or(missing_bias)?;
                    let zeros_span = |elements: u64| {
                        CudaBufferSpan::new(zeros, CudaDType::BF16, 0, elements * 2)
                            .map_err(|source| batch_cuda(query_site, source))
                    };
                    (
                        zeros_span(query_width)?,
                        zeros_span(key_value_width)?,
                        zeros_span(key_value_width)?,
                        None,
                    )
                }
                _ => return Err(missing_bias.into()),
            };
            #[cfg(feature="cuda-cublas-gemm-probe")]
            let separate=if let Some((query,key_value))=gemms.hf_separate_projection_plans(){
              execute_gemm_with_row_groups(query,&buffers.hidden_norm,weight_span(weights,layer.query_weight(),query_site)?,&mut buffers.hidden_context,&mut buffers.gemm_workspace,stream,query_site,batch.host().row_sequence_slots(),None)?;
              execute_gemm_with_row_groups(key_value,&buffers.hidden_norm,weight_span(weights,layer.key_weight(),query_site)?,&mut buffers.key_raw,&mut buffers.gemm_workspace,stream,query_site,batch.host().row_sequence_slots(),None)?;
              execute_gemm_with_row_groups(key_value,&buffers.hidden_norm,weight_span(weights,layer.value_weight(),query_site)?,&mut buffers.value_raw,&mut buffers.gemm_workspace,stream,query_site,batch.host().row_sequence_slots(),None)?;
              hf_pack_separate_qkv(span(&buffers.hidden_context,CudaDType::BF16,dense_rows*query_width*2,query_site)?,span(&buffers.key_raw,CudaDType::BF16,dense_rows*key_value_width*2,query_site)?,span(&buffers.value_raw,CudaDType::BF16,dense_rows*key_value_width*2,query_site)?,span_mut(&mut buffers.gate_raw,CudaDType::BF16,merged_bytes,query_site)?,dense_rows,query_width,key_value_width,stream).map_err(|e|batch_cuda(query_site,e))?;
              true
            }else{false};
            #[cfg(not(feature="cuda-cublas-gemm-probe"))]let separate=false;
            if !separate {
            let merged_weight =
                weights
                    .merged_qkv(layer_index)
                    .ok_or(LlamaForwardError::InvalidConfiguration {
                        field: "projection_bias_mode",
                        reason: "merged QKV weights were not uploaded for this layer",
                    })?;
            let merged_weight =
                CudaBufferSpan::new(merged_weight, CudaDType::BF16, 0, merged_weight.byte_len())
                    .map_err(|source| batch_cuda(query_site, source))?;
            if let Some(plan) = gemms.merged_qkv_plan() {
                execute_gemm_with_row_groups(
                    plan,
                    &buffers.hidden_norm,
                    merged_weight,
                    &mut buffers.gate_raw,
                    &mut buffers.gemm_workspace,
                    stream,
                    query_site,
                batch.host().row_sequence_slots(),None,
)?;
            }
            }
            let mut params = QkvSplitBiasParams {
                merged: span(&buffers.gate_raw, CudaDType::BF16, merged_bytes, query_site)?,
                query_bias,
                key_bias,
                value_bias,
                query: span_mut(
                    if fused_qkv_rope {
                        &mut buffers.hidden_rotary
                    } else {
                        &mut *hidden_projection
                    },
                    CudaDType::BF16,
                    dense_rows * query_width * 2,
                    query_site,
                )?,
                key: span_mut(
                    if fused_qkv_rope {
                        &mut buffers.key_rotary
                    } else {
                        &mut buffers.key_raw
                    },
                    CudaDType::BF16,
                    dense_rows * key_value_width * 2,
                    query_site,
                )?,
                value: span_mut(
                    &mut buffers.value_raw,
                    CudaDType::BF16,
                    dense_rows * key_value_width * 2,
                    query_site,
                )?,
                row_count: dense_rows,
                query_width,
                key_value_width,
            };
            if fused_qkv_rope {
                let rope = QkvRopeParams {
                    cos: CudaBufferSpan::new(rope_cos, CudaDType::F32, 0, rope_cos.byte_len())
                        .map_err(|source| batch_cuda(query_site, source))?,
                    sin: CudaBufferSpan::new(rope_sin, CudaDType::F32, 0, rope_sin.byte_len())
                        .map_err(|source| batch_cuda(query_site, source))?,
                    positions: batch.device_row_positions(),
                    positions_host,
                    head_size,
                    table_position_count: max_positions,
                    // The KV cache write below consumes the deferred split.
                    defer_for_kv_write: fused_kv_write,
                    head_norm_epsilon,
                };
                #[cfg(feature="cuda-cublas-gemm-probe")]
                let split_result=if hf_wide_norm && head_norm_epsilon.is_some(){qkv_split_bias_rope_hf_head128(&mut params,&rope,stream)}else{qkv_split_bias_rope(&mut params,&rope,stream)};
                #[cfg(not(feature="cuda-cublas-gemm-probe"))]
                let split_result=qkv_split_bias_rope(&mut params,&rope,stream);
                split_result.map_err(|source|batch_cuda(query_site,source))?;
            } else {
                qkv_split_bias(&mut params, stream)
                    .map_err(|source| batch_cuda(query_site, source))?;
            }
        } else {
            let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            if query_width != hidden {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "projection_bias_mode",
                    reason: "a query projection wider than the hidden size requires the merged Q|K|V projection with the fused RoPE split",
                });
            }
            let (strict, fused, requests) = gemms.query_projection_plans_with_requests();
            execute_projection_with_row_groups(
                strict,
                fused,
                weights,
                layer.query_bias(),
                &buffers.hidden_norm,
                weight_span(weights, layer.query_weight(), query_site)?,
                hidden_projection,
                &mut buffers.gemm_workspace,
                dense_rows,
                hidden,
                stream,
                query_site,
            batch.host().row_sequence_slots(),requests,
)?;
            let key_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
            let (strict, fused, requests) = gemms.key_value_projection_plans_with_requests();
            execute_projection_with_row_groups(
                strict,
                fused,
                weights,
                layer.key_bias(),
                &buffers.hidden_norm,
                weight_span(weights, layer.key_weight(), key_site)?,
                &mut buffers.key_raw,
                &mut buffers.gemm_workspace,
                dense_rows,
                key_value_width,
                stream,
                key_site,
            batch.host().row_sequence_slots(),requests,
)?;
            let value_site = ExecutionSite::layer(layer_index, LlamaOp::ValueProjection);
            let (strict, fused, requests) = gemms.key_value_projection_plans_with_requests();
            execute_projection_with_row_groups(
                strict,
                fused,
                weights,
                layer.value_bias(),
                &buffers.hidden_norm,
                weight_span(weights, layer.value_weight(), value_site)?,
                &mut buffers.value_raw,
                &mut buffers.gemm_workspace,
                dense_rows,
                key_value_width,
                stream,
                value_site,
            batch.host().row_sequence_slots(),requests,
)?;
        }

        // Qwen3 RMS-normalizes every projected query head and key head before
        // the rotary embedding. `hidden_norm` is idle from here to the
        // post-attention norm and `gate_raw` until the MLP, so they hold the
        // normalized query and key.
        // The merged split with fused RoPE has already normalized both.
        let query_key_normed = layer.query_key_norm().is_some() && !rope_in_split;
        #[cfg(feature="cuda-cublas-gemm-probe")]
        let hf_head_norm = head_size==128 && native_d128_two_stage_workspace.as_ref()
            .and_then(|w|w.materialized.as_ref()).is_some_and(|m|m.hf.is_some());
        if let (Some((query_scale, key_scale, epsilon)), false) =
            (layer.query_key_norm(), rope_in_split)
        {
            if gemms.merged_qkv_plan().is_some() {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "projection_bias_mode",
                    reason: "a per-head query/key norm on merged Q|K|V needs the fused RoPE split",
                });
            }
            let query_norm_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            let mut params = RmsNormParams {
                input: span(
                    hidden_projection,
                    CudaDType::BF16,
                    dense_rows * query_heads * head_size * 2,
                    query_norm_site,
                )?,
                weight: weight_span(weights, query_scale, query_norm_site)?,
                output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    dense_rows * query_heads * head_size * 2,
                    query_norm_site,
                )?,
                row_count: dense_rows * query_heads,
                hidden_size: head_size,
                epsilon,
            };
            #[cfg(feature="cuda-cublas-gemm-probe")]
            let norm_result=if hf_head_norm { hugging_face_qwen_head128_rms_norm(&mut params,stream) } else { execute_profile_rms_norm(rms_norm_profile,&mut params,stream) };
            #[cfg(not(feature="cuda-cublas-gemm-probe"))]
            let norm_result=execute_profile_rms_norm(rms_norm_profile,&mut params,stream);
            norm_result.map_err(|source|batch_cuda(query_norm_site,source))?;
            let key_norm_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
            let mut params = RmsNormParams {
                input: span(
                    &buffers.key_raw,
                    CudaDType::BF16,
                    dense_rows * key_value_width * 2,
                    key_norm_site,
                )?,
                weight: weight_span(weights, key_scale, key_norm_site)?,
                output: span_mut(
                    &mut buffers.gate_raw,
                    CudaDType::BF16,
                    dense_rows * key_value_width * 2,
                    key_norm_site,
                )?,
                row_count: dense_rows * key_value_heads,
                hidden_size: head_size,
                epsilon,
            };
            #[cfg(feature="cuda-cublas-gemm-probe")]
            let norm_result=if hf_head_norm { hugging_face_qwen_head128_rms_norm(&mut params,stream) } else { execute_profile_rms_norm(rms_norm_profile,&mut params,stream) };
            #[cfg(not(feature="cuda-cublas-gemm-probe"))]
            let norm_result=execute_profile_rms_norm(rms_norm_profile,&mut params,stream);
            norm_result.map_err(|source|batch_cuda(key_norm_site,source))?;
        }

        let query_rope_site = ExecutionSite::layer(layer_index, LlamaOp::QueryRope);
        if !rope_in_split {
            let mut params = IndexedRopeParams {
                input: span(
                    if query_key_normed {
                        &buffers.hidden_norm
                    } else {
                        &*hidden_projection
                    },
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
        if !rope_in_split {
            let mut params = IndexedRopeParams {
                input: span(
                    if query_key_normed {
                        &buffers.gate_raw
                    } else {
                        &buffers.key_raw
                    },
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
        let materialized_reference = native_d128_two_stage_workspace.as_ref().is_some_and(|w|w.materialized.is_some());
        if materialized_reference {
            if !matches!(step_capture, StepCaptureMode::Disabled) {
                return Err(LlamaBatchExecutorError::InvalidConfiguration { field: "materialized reference attention", reason: "temporary reference resources cannot be graph captured" });
            }
            let prepared = native_d128_two_stage_workspace.as_deref_mut().and_then(|w|w.materialized.as_mut()).expect("cold prepared reference workspace");
            #[cfg(feature="cuda-cublas-gemm-probe")]
            let mut hf_executed=false;
            #[cfg(feature="cuda-cublas-gemm-probe")]
            if let Some(hf)=prepared.hf.as_mut() {
                let host=batch.host();let positions=host.row_positions();let slots=host.row_sequence_slots();let mut begin=0;
                while begin<positions.len() {
                    let slot=slots[begin] as usize;let mut end=begin+1;
                    while end<positions.len() && slots[end] as usize==slot {end+=1;}
                    if positions[begin..end].windows(2).any(|adjacent|adjacent[0].checked_add(1)!=Some(adjacent[1])) {return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"hf_attention_rows",reason:"query rows must be contiguous within a sequence"});}
                    let m=(end-begin) as u64;let t=u64::from(positions[end-1])+1;let table_begin=host.sequence_block_offsets()[slot] as u64;let table_len=t.div_ceil(16);
                    if table_begin+table_len>host.block_ids().len() as u64 {return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"hf_attention_table",reason:"logical prefix exceeds packed block table"});}
                    let ids=batch.device_block_ids();let row_bytes=query_width*2;let rb=(begin as u64)*row_bytes;
                    hf.plan.execute_hf_attention(
                        CudaBufferSpan::new(&buffers.hidden_rotary,CudaDType::BF16,rb,m*row_bytes).map_err(|e|batch_cuda(attention_site,e))?,
                        CudaBufferSpan::new(key_cache,CudaDType::BF16,layer_offset,layout.layer_stride_bytes()).map_err(|e|batch_cuda(attention_site,e))?,
                        CudaBufferSpan::new(value_cache,CudaDType::BF16,layer_offset,layout.layer_stride_bytes()).map_err(|e|batch_cuda(attention_site,e))?,
                        ids.subspan(table_begin*4,table_len*4).map_err(|e|batch_cuda(attention_site,e))?,
                        CudaBufferSpanMut::new(&mut buffers.hidden_context,CudaDType::BF16,rb,m*row_bytes).map_err(|e|batch_cuda(attention_site,e))?,
                        &mut hf.scratch,m,t,layout.physical_block_count() as u64,dimensions.query_heads() as u64,dimensions.key_value_heads() as u64,dimensions.head_dimension() as u64,stream,
                    ).map_err(|e|batch_cuda(attention_site,e))?;
                    begin=end;
                }
                hf_executed=true;
            }
            #[cfg(not(feature="cuda-cublas-gemm-probe"))]
            let hf_executed=false;
            if !hf_executed && attention_implementation == RaggedAttentionImplementation::OwnedHfEagerBf16V1 {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"owned-hf-eager-bf16-v1",reason:"owned attention workspace missing; fallback is forbidden"});
            }
            if !hf_executed {
            riley_cuda::prepared_materialized_ragged_attention(
                span(&buffers.hidden_rotary,CudaDType::BF16,query_buffer_bytes,attention_site)?,
                CudaBufferSpan::new(key_cache,CudaDType::BF16,layer_offset,layout.layer_stride_bytes()).map_err(|e|batch_cuda(attention_site,e))?,
                CudaBufferSpan::new(value_cache,CudaDType::BF16,layer_offset,layout.layer_stride_bytes()).map_err(|e|batch_cuda(attention_site,e))?,
                &mut buffers.hidden_context,batch,
                &prepared.reference,&mut prepared.scores,&prepared.valid,
                &mut prepared.valid_host,&mut prepared.duplicate_scratch,stream,
            ).map_err(|e|batch_cuda(attention_site,e))?;
            }
        }
        if materialized_reference {
            // Output active rows already produced by the prepared reference.
        } else if attention_implementation
            == RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage
        {
            if attention_reduction_profile != AttentionReductionProfile::CanonicalV1 {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "ragged_attention_implementation",
                    reason: "native D128 two-stage attention requires canonical-v1 reductions",
                });
            }
            let workspace = native_d128_two_stage_workspace.as_deref_mut().ok_or(
                LlamaBatchExecutorError::InvalidConfiguration {
                    field: "native_d128_two_stage_workspace",
                    reason: "native D128 two-stage selection has no prepared workspace",
                },
            )?;
            let partial_state_bytes = workspace.partial_states.byte_len();
            let reduction_step_bytes = workspace.reduction_steps.byte_len();
            let reduction_normalizer_bytes = workspace.reduction_normalizers.byte_len();
            let mut params = NativeBf16RaggedPagedSplitGqaD128TwoStageParams {
                query: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    query_buffer_bytes,
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
                partial_states: CudaBufferSpanMut::new(
                    &mut workspace.partial_states,
                    CudaDType::F32,
                    0,
                    partial_state_bytes,
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                reduction_steps: CudaBufferSpanMut::new(
                    &mut workspace.reduction_steps,
                    CudaDType::F32,
                    0,
                    reduction_step_bytes,
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                reduction_normalizers: CudaBufferSpanMut::new(
                    &mut workspace.reduction_normalizers,
                    CudaDType::F32,
                    0,
                    reduction_normalizer_bytes,
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                output: span_mut(
                    &mut buffers.hidden_context,
                    CudaDType::BF16,
                    query_buffer_bytes,
                    attention_site,
                )?,
                batch,
                query_head_count: query_heads,
                key_value_head_count: key_value_heads,
                head_size,
                output_row_count: dense_rows,
                partial_state_capacity: workspace.partial_state_capacity,
                scale: 1.0 / (head_size as f32).sqrt(),
                reduction_order: DecodePartialReductionOrder::LogicalAscending,
            };
            // Shared-KV producer geometry: bit-identical states, one K/V read
            // per KV head instead of one per query head.
            if let (true, true, Some((first_word, generated_rows))) = (
                d128_global_merge,
                d128_tensor_core_producer,
                prompt_row_kinds_first_word,
            ) {
                native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma(
                    &mut params,
                    first_word,
                    generated_rows,
                    stream,
                )
            } else if d128_global_merge && d128_tensor_core_producer {
                native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge(
                    &mut params,
                    stream,
                )
            } else if let (true, Some((first_word, _))) =
                (d128_global_merge, prompt_row_kinds_first_word)
            {
                native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma(
                    &mut params,
                    first_word,
                    stream,
                )
            } else if d128_global_merge {
                native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge(&mut params, stream)
            } else {
                native_bf16_ragged_paged_split_gqa_d128_two_stage_shared_kv(&mut params, stream)
            }
            .map_err(|source| batch_cuda(attention_site, source))?;
        } else {
            let mut params = RaggedPagedAttentionParams {
                query: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    query_buffer_bytes,
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
                    query_buffer_bytes,
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
                    RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage | RaggedAttentionImplementation::MaterializedBf16ReferenceD128 | RaggedAttentionImplementation::MaterializedBf16ReferenceD64 | RaggedAttentionImplementation::OwnedHfEagerBf16V1 => {
                        return Err(LlamaBatchExecutorError::InvalidConfiguration {
                            field: "ragged_attention_implementation",
                            reason: "native D128 two-stage dispatch escaped its explicit branch",
                        });
                    }
                },
                AttentionReductionProfile::FixedContiguous37BalancedV1 => {
                    fixed37_ragged_paged_attention(&mut params, stream)
                }
            }
            .map_err(|source| batch_cuda(attention_site, source))?;
        }

        let output_site = ExecutionSite::layer(layer_index, LlamaOp::OutputProjection);
        execute_gemm_with_row_groups(
            gemms.output_projection_plan(),
            &buffers.hidden_context,
            weight_span(weights, layer.output_weight(), output_site)?,
            hidden_projection,
            &mut buffers.gemm_workspace,
            stream,
            output_site,
        batch.host().row_sequence_slots(),None,
)?;
        execute_projection_bias(
            weights,
            layer.output_bias(),
            hidden_projection,
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
                        hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        attention_residual_site,
                    )?,
                    right: span(
                        hidden_projection,
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
                execute_profile_rms_norm_grouped(rms_norm_profile,&mut norm,stream,batch.host().row_sequence_slots(),hf_wide_norm)
                    .map_err(|source| batch_cuda(post_norm_site, source))?;
            }
            ResidualNormImplementation::Fused => {
                let mut fused = ResidualRmsNormParams {
                    left: span(
                        hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    right: span(
                        hidden_projection,
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
        if gemms.merged_gate_up_plan().is_some() {
            // One GEMM over the concatenated gate|up weights; each output row
            // holds the raw gate followed by the up projection.
            if !fused_silu_multiply {
                return Err(LlamaForwardError::InvalidConfiguration {
                    field: "merged_gate_up",
                    reason: "merged gate|up projection requires the fused SiLU multiply",
                }
                .into());
            }
            let merged_weight = weights.merged_gate_up(layer_index).ok_or(
                LlamaForwardError::InvalidConfiguration {
                    field: "merged_gate_up",
                    reason: "merged gate|up weights were not uploaded for this layer",
                },
            )?;
            let merged_weight =
                CudaBufferSpan::new(merged_weight, CudaDType::BF16, 0, merged_weight.byte_len())
                    .map_err(|source| batch_cuda(gate_site, source))?;
            if let Some(merged_plan) = gemms.merged_gate_up_plan() {
                execute_gemm_with_row_groups(
                    merged_plan,
                    &buffers.hidden_norm,
                    merged_weight,
                    &mut buffers.gate_raw,
                    &mut buffers.gemm_workspace,
                    stream,
                    gate_site,
                batch.host().row_sequence_slots(),None,
)?;
            }
            let gated_site = ExecutionSite::layer(layer_index, LlamaOp::GatedMultiply);
            let mut params = MergedGatedMultiplyParams {
                merged: span(
                    &buffers.gate_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes() * 2,
                    gated_site,
                )?,
                output: span_mut(
                    &mut buffers.gated_product,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                element_count: intermediate_elements,
                row_width: plan.dimensions().intermediate_size() as u64,
            };
            silu_gated_multiply_merged_rows(&mut params, stream)
                .map_err(|source| batch_cuda(gated_site, source))?;
        } else {
            execute_gemm_with_row_groups(
                &mut gemms.intermediate,
                &buffers.hidden_norm,
                weight_span(weights, layer.gate_weight(), gate_site)?,
                &mut buffers.gate_raw,
                &mut buffers.gemm_workspace,
                stream,
                gate_site,
            batch.host().row_sequence_slots(),None,
)?;
            let up_site = ExecutionSite::layer(layer_index, LlamaOp::UpProjection);
            execute_gemm_with_row_groups(
                &mut gemms.intermediate,
                &buffers.hidden_norm,
                weight_span(weights, layer.up_weight(), up_site)?,
                &mut buffers.up_raw,
                &mut buffers.gemm_workspace,
                stream,
                up_site,
            batch.host().row_sequence_slots(),None,
)?;
            let silu_site = ExecutionSite::layer(layer_index, LlamaOp::Silu);
            if !fused_silu_multiply {
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
                        if fused_silu_multiply {
                            &buffers.gate_raw
                        } else {
                            &buffers.gate_activated
                        },
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
                if fused_silu_multiply {
                    silu_gated_multiply(&mut params, stream)
                        .map_err(|source| batch_cuda(gated_site, source))?;
                } else {
                    gated_multiply(&mut params, stream)
                        .map_err(|source| batch_cuda(gated_site, source))?;
                }
            }
        }
        let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
        execute_gemm_with_row_groups(
            &mut gemms.down,
            &buffers.gated_product,
            weight_span(weights, layer.down_weight(), down_site)?,
            hidden_current,
            &mut buffers.gemm_workspace,
            stream,
            down_site,
        batch.host().row_sequence_slots(),None,
)?;
        let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);
        if norm_after_mlp {
            // The norm that follows this residual: the next layer's input norm,
            // or the final norm after the last layer.
            let (next_weight, next_epsilon, next_site) = match plan.layers().get(ordinal + 1) {
                Some(next) => (
                    next.input_norm_weight(),
                    next.input_norm_epsilon(),
                    ExecutionSite::layer(next.index(), LlamaOp::InputNorm),
                ),
                None => (
                    plan.final_norm_weight(),
                    plan.final_norm_epsilon(),
                    ExecutionSite::global(LlamaOp::FinalNorm),
                ),
            };
            let mut fused = ResidualRmsNormParams {
                left: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                right: span(
                    hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                weight: weight_span(weights, next_weight, next_site)?,
                residual_output: span_mut(
                    hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                normalized_output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    next_site,
                )?,
                row_count: dense_rows,
                hidden_size: hidden,
                epsilon: next_epsilon,
            };
            execute_profile_residual_rms_norm(rms_norm_profile, &mut fused, stream)
                .map_err(|source| batch_cuda(next_site, source))?;
        } else {
            let mut params = ResidualAddParams {
                left: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                right: span(
                    hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                output: span_mut(
                    hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                element_count: hidden_elements,
            };
            residual_add(&mut params, stream)
                .map_err(|source| batch_cuda(mlp_residual_site, source))?;
        }
    }

    // Preserve the public owner roles after an odd number of layers. Within
    // the chain, physical owners never move; a future native recorder can use
    // the same parity mapping without retaining conflicting per-op borrows.
    if plan.layers().len() % 2 != 0 {
        mem::swap(&mut buffers.hidden_current, &mut buffers.hidden_projection);
    }

    let final_norm_site = ExecutionSite::global(LlamaOp::FinalNorm);
    if !norm_after_mlp {
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
        execute_profile_rms_norm_grouped(rms_norm_profile,&mut params,stream,batch.host().row_sequence_slots(),hf_wide_norm)
            .map_err(|source| batch_cuda(final_norm_site, source))?;
    }
    let lm_head_site = ExecutionSite::global(LlamaOp::LmHead);
    // No token is requested from this partial prefill: the next iteration
    // consumes only KV. Batch-only storage deliberately has no dense logits.
    if buffers.logits.byte_len() == 0 && output_row_lm_head.is_none() {
        return Ok(());
    }
    if let Some(output_rows) = output_row_lm_head {
        // `hidden_rotary` is idle after the last layer: it receives the output
        // rows' normalized hidden states in output order. Rows past the output
        // count keep earlier contents; their logits are never read.
        let gather_site = ExecutionSite::global(LlamaOp::OutputGather);
        let gathered_rows = u64::try_from(output_rows.output_indices_host.len()).map_err(|_| {
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_token_indices",
                reason: "output row count does not fit u64",
            }
        })?;
        {
            let mut params = RowGatherParams {
                input: span(
                    &buffers.hidden_norm,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    gather_site,
                )?,
                row_indices: output_rows.output_indices,
                row_indices_host: output_rows.output_indices_host,
                output: span_mut(
                    &mut buffers.hidden_rotary,
                    CudaDType::BF16,
                    gathered_rows * hidden * 2,
                    gather_site,
                )?,
                input_row_count: dense_rows,
                column_count: hidden,
            };
            row_gather(&mut params, stream).map_err(|source| batch_cuda(gather_site, source))?;
        }
        execute_gemm_with_row_groups(
            output_rows.gemm,
            &buffers.hidden_rotary,
            weight_span(weights, plan.lm_head_weight(), lm_head_site)?,
            output_rows.gathered_logits,
            &mut buffers.gemm_workspace,
            stream,
            lm_head_site,
        batch.host().row_sequence_slots(),Some(gathered_rows as usize),
)?;
        return Ok(());
    }
    execute_gemm_with_row_groups(
        &mut gemms.lm_head,
        &buffers.hidden_norm,
        weight_span(weights, plan.lm_head_weight(), lm_head_site)?,
        &mut buffers.logits,
        &mut buffers.gemm_workspace,
        stream,
        lm_head_site,
    batch.host().row_sequence_slots(),Some(batch.host().active_row_count() as usize),
)?;
    Ok(())
}

/// First position whose prompt token takes the tensor-core attention producer.
/// The split is by the token's own position, so a token takes the same path in
/// every batch and chunk. Below it the shared-KV producer is the cheaper one
/// for the few rows of a prompt that barely crosses the bound, and a step
/// whose prompt rows all sit below it launches nothing extra.
const TENSOR_CORE_PROMPT_MIN_POSITION: u32 = 256;

/// The first prompt position that takes the prompt-row producer: every prompt
/// row with the dense prompt attention, else the bound above.
const fn prompt_row_min_position(config: PreparedLlamaBatchExecutorConfig) -> u32 {
    if config.dense_prompt_attention() {
        0
    } else {
        TENSOR_CORE_PROMPT_MIN_POSITION
    }
}

fn is_tensor_core_prompt_row(
    packed: &LlamaPackedBatchMetadata<'_>,
    row: usize,
    min_position: u32,
) -> bool {
    let slot = packed.row_sequence_slots()[row] as usize;
    packed
        .row_kind_codes()
        .get(slot)
        .is_some_and(|code| *code == LlamaBatchRowKind::Prefill.code())
        && packed.position_ids()[row] >= min_position
}

/// With the dense prompt attention, writes the list of the step's rows that
/// are not prompt rows (`[count, rows...]`, U32) behind the row kinds, at
/// word `max_input_tokens`, and returns how many bytes of `row_kinds` the
/// step copies to the device: the kinds alone, or the kinds region and the
/// list. `row_kinds` already holds this step's kinds.
fn fill_generated_row_list(
    config: PreparedLlamaBatchExecutorConfig,
    bounds: crate::llama::LlamaBatchMetadataConfig,
    row_kinds: &mut [u8],
    active: usize,
    dense_rows: usize,
) -> LlamaBatchExecutorResult<usize> {
    if !config.dense_prompt_attention() {
        return Ok(dense_rows * U32_BYTES);
    }
    let list_base = bounds.max_input_tokens() * U32_BYTES;
    let (kinds, list) = row_kinds.split_at_mut(list_base);
    let mut count = 0_usize;
    for (row, kind) in kinds.chunks_exact(U32_BYTES).take(active).enumerate() {
        if kind != 1_u32.to_ne_bytes() {
            let slot = (1 + count) * U32_BYTES;
            let Some(word) = list.get_mut(slot..slot + U32_BYTES) else {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "dense_prompt_attention",
                    reason: "a step has more generated rows than sequence rows",
                });
            };
            word.copy_from_slice(&(row as u32).to_ne_bytes());
            count += 1;
        }
    }
    list[..U32_BYTES].copy_from_slice(&(count as u32).to_ne_bytes());
    Ok(list_base + (1 + count) * U32_BYTES)
}

fn has_tensor_core_prompt_rows(packed: &LlamaPackedBatchMetadata<'_>, min_position: u32) -> bool {
    !packed.prefill_row_indices().is_empty()
        && (0..packed.total_input_tokens())
            .any(|row| is_tensor_core_prompt_row(packed, row, min_position))
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

/// Writes one run of bytes to the pinned staging buffer and records where the
/// command batch has to copy it.
#[allow(clippy::too_many_arguments)]
fn stage_bytes(
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    staged: &mut [StagedCopy; MAX_STAGED_COPIES],
    staged_count: &mut usize,
    cursor: &mut u64,
    target: StagedTarget,
    destination_offset: u64,
    bytes: &[u8],
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    if *staged_count == MAX_STAGED_COPIES {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "staged_step_uploads",
            reason: "an iteration staged more copies than the fixed capacity",
        });
    }
    staging
        .write(*cursor, bytes)
        .map_err(|source| batch_cuda(site, source))?;
    let byte_len = bytes.len() as u64;
    staged[*staged_count] = StagedCopy {
        target,
        destination_offset,
        staging_offset: *cursor,
        byte_len,
    };
    *staged_count += 1;
    *cursor += byte_len.div_ceil(8) * 8;
    Ok(())
}

/// The staged form of the synchronous token and metadata uploads: the same
/// bytes for the same device buffers, written to pinned memory now and copied
/// by [`copy_staged_step_uploads`] inside the iteration's command batch.
fn stage_step_uploads(
    host: &mut PerOperationHostWorkspace,
    forward: &mut PreparedLlamaForward,
    packed: &LlamaPackedBatchMetadata<'_>,
    config: PreparedLlamaBatchExecutorConfig,
    dense_rows: usize,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    let bounds = config.metadata();
    let active = packed.total_input_tokens();
    let PerOperationHostWorkspace {
        sequence_block_offsets,
        physical_block_ids,
        valid_tokens,
        row_sequence_slots,
        row_positions,
        output_token_indices,
        uploaded_step_counts,
        row_kinds,
        uploaded,
        staging,
        staged,
        staged_count,
        staged_greedy_bytes,
        staged_report,
        ..
    } = host;
    let staging = staging
        .as_mut()
        .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
            field: "staged_step_uploads",
            reason: "staged step uploads have no cold-prepared pinned buffer",
        })?;
    *staged_count = 0;
    staged_greedy_bytes.set(0);
    staged_report.set(false);
    let mut cursor = 0_u64;

    let token_byte_len = checked_byte_len(
        dense_rows,
        U32_BYTES,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    if token_byte_len > forward.token_bytes.len() || active > dense_rows {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "dense_rows",
            reason: "selected token prefix exceeds the shared maximum buffer",
        });
    }
    forward.tokens_ready = false;
    forward.output_ready = false;
    forward.token_bytes[..token_byte_len].fill(0);
    encode_u32(
        packed.input_token_ids(),
        &mut forward.token_bytes[..active * U32_BYTES],
    );
    stage_bytes(
        staging,
        staged,
        staged_count,
        &mut cursor,
        StagedTarget::Tokens,
        0,
        &forward.token_bytes[..token_byte_len],
        site,
    )?;
    forward.tokens_ready = true;

    encode_u32(packed.block_row_offsets(), sequence_block_offsets);
    encode_u32(packed.physical_block_ids(), physical_block_ids);
    encode_u16(packed.valid_tokens(), valid_tokens);
    encode_u32(packed.row_sequence_slots(), row_sequence_slots);
    encode_u32(packed.position_ids(), row_positions);
    encode_u32(packed.output_token_indices(), output_token_indices);

    // Metadata that often repeats from one step to the next is staged only
    // when its bytes changed, as in the synchronous path.
    for (target, kind, bytes) in [
        (
            StagedTarget::BlockOffsets,
            UPLOADED_BLOCK_OFFSETS,
            &sequence_block_offsets[..packed.block_row_offsets().len() * U32_BYTES],
        ),
        (
            StagedTarget::BlockIds,
            UPLOADED_BLOCK_IDS,
            &physical_block_ids[..packed.physical_block_ids().len() * U32_BYTES],
        ),
        (
            StagedTarget::RowSlots,
            UPLOADED_ROW_SLOTS,
            &row_sequence_slots[..active * U32_BYTES],
        ),
        (
            StagedTarget::OutputIndices,
            UPLOADED_OUTPUT_INDICES,
            &output_token_indices[..packed.output_count() * U32_BYTES],
        ),
    ] {
        if bytes.is_empty() || uploaded[kind].as_slice() == bytes {
            continue;
        }
        uploaded[kind].clear();
        stage_bytes(
            staging,
            staged,
            staged_count,
            &mut cursor,
            target,
            0,
            bytes,
            site,
        )?;
        if bytes.len() <= uploaded[kind].capacity() {
            uploaded[kind].extend_from_slice(bytes);
        }
    }
    stage_bytes(
        staging,
        staged,
        staged_count,
        &mut cursor,
        StagedTarget::ValidTokens,
        0,
        &valid_tokens[..packed.valid_tokens().len() * U16_BYTES],
        site,
    )?;
    stage_bytes(
        staging,
        staged,
        staged_count,
        &mut cursor,
        StagedTarget::RowPositions,
        0,
        &row_positions[..active * U32_BYTES],
        site,
    )?;
    if config.decode_step_capture() {
        let invalid_count = |field| LlamaBatchExecutorError::InvalidConfiguration {
            field,
            reason: "row count does not fit u32",
        };
        let step_counts = [
            u32::try_from(active).map_err(|_| invalid_count("row_positions"))?,
            u32::try_from(packed.output_count())
                .map_err(|_| invalid_count("output_token_indices"))?,
        ];
        if *uploaded_step_counts != Some(step_counts) {
            *uploaded_step_counts = None;
            let mut counts = [0_u8; STEP_COUNT_WORDS * U32_BYTES];
            counts[..U32_BYTES].copy_from_slice(&step_counts[0].to_ne_bytes());
            counts[U32_BYTES..].copy_from_slice(&step_counts[1].to_ne_bytes());
            stage_bytes(
                staging,
                staged,
                staged_count,
                &mut cursor,
                StagedTarget::RowPositions,
                (bounds.max_input_tokens() * U32_BYTES) as u64,
                &counts,
                site,
            )?;
            *uploaded_step_counts = Some(step_counts);
        }
    }
    if config.prompt_attention_mma()
        && has_tensor_core_prompt_rows(packed, prompt_row_min_position(config))
    {
        for (row, kind) in row_kinds
            .chunks_exact_mut(U32_BYTES)
            .take(active)
            .enumerate()
        {
            kind.copy_from_slice(
                &u32::from(is_tensor_core_prompt_row(
                    packed,
                    row,
                    prompt_row_min_position(config),
                ))
                .to_ne_bytes(),
            );
        }
        row_kinds[active * U32_BYTES..dense_rows * U32_BYTES].fill(0);
        let row_kind_bytes =
            fill_generated_row_list(config, bounds, row_kinds, active, dense_rows)?;
        stage_bytes(
            staging,
            staged,
            staged_count,
            &mut cursor,
            StagedTarget::RowPositions,
            ((bounds.max_input_tokens() + STEP_COUNT_WORDS) * U32_BYTES) as u64,
            &row_kinds[..row_kind_bytes],
            site,
        )?;
    }
    Ok(())
}

/// Enqueues the copies recorded by [`stage_step_uploads`] in the command
/// batch; they complete with the batch, before any kernel that reads them.
fn copy_staged_step_uploads<'stream>(
    device: &'stream PerOperationDeviceMetadata,
    staging: &'stream riley_cuda::CudaPinnedHostBuffer,
    staged: &[StagedCopy],
    commands: &mut CudaCommandStream<'_, 'stream>,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    for copy in staged {
        let destination = match copy.target {
            StagedTarget::Tokens => device.token_ids.as_ref(),
            StagedTarget::BlockOffsets => Some(&device.sequence_block_offsets),
            StagedTarget::BlockIds => Some(&device.physical_block_ids),
            StagedTarget::ValidTokens => Some(&device.valid_tokens),
            StagedTarget::RowSlots => Some(&device.row_sequence_slots),
            StagedTarget::RowPositions => Some(&device.row_positions),
            StagedTarget::OutputIndices => device.output_token_indices.as_ref(),
        }
        .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
            field: "staged_step_uploads",
            reason: "a staged copy names a device buffer that was not cold-prepared",
        })?;
        destination
            .copy_from_pinned_in_command_batch(
                copy.destination_offset,
                staging,
                copy.staging_offset,
                copy.byte_len,
                commands,
            )
            .map_err(|source| batch_cuda(site, source))?;
    }
    Ok(())
}

/// [`upload_prefix`] that is skipped when the device already holds exactly
/// these bytes from the previous copy of the same metadata.
fn upload_prefix_if_changed(
    destination: &mut CudaDeviceBuffer,
    source: &[u8],
    byte_len: usize,
    uploaded: &mut UploadedBytes,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    stream: &mut CudaStream,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    if byte_len != 0 && uploaded.as_slice() == &source[..byte_len] {
        return Ok(());
    }
    uploaded.clear();
    upload_prefix(destination, source, byte_len, staging, stream, site)?;
    if byte_len <= uploaded.capacity() {
        uploaded.extend_from_slice(&source[..byte_len]);
    }
    Ok(())
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

/// Selects physical scratch owners without moving either allocation. The final
/// output of one layer is the next layer's current input. This is ordinal
/// parity, independent of any model-supplied layer identifier.
fn layer_hidden_roles<'a, T>(
    ordinal: usize,
    first: &'a mut T,
    second: &'a mut T,
) -> (&'a mut T, &'a mut T) {
    if ordinal % 2 == 0 {
        (first, second)
    } else {
        (second, first)
    }
}

#[cfg(test)]
mod hidden_role_tests {
    use super::layer_hidden_roles;

    #[derive(Debug, PartialEq, Eq)]
    struct Owner {
        identity: u64,
        value: u64,
    }

    // Compare against the previous dispatch semantics across odd/even layer
    // counts and repeated iterations. Values model preserved residual input,
    // projection scratch overwrite, down output, and final residual output.
    #[test]
    fn physical_roles_preserve_legacy_chain_and_continuation() {
        fn layer(current: &mut Owner, projection: &mut Owner, ordinal: usize) {
            let residual = current.value;
            projection.value = residual.wrapping_mul(3).wrapping_add(ordinal as u64);
            let attention_residual = residual.wrapping_add(projection.value);
            current.value = attention_residual.wrapping_mul(7);
            projection.value = attention_residual.wrapping_add(current.value);
        }
        for count in [0, 1, 2, 3, 29, 30, 31, 64] {
            let mut legacy = (
                Owner {
                    identity: 11,
                    value: 0,
                },
                Owner {
                    identity: 22,
                    value: 0,
                },
            );
            let mut fixed = (
                Owner {
                    identity: 11,
                    value: 0,
                },
                Owner {
                    identity: 22,
                    value: 0,
                },
            );
            for iteration in 0..16 {
                legacy.0.value = iteration;
                fixed.0.value = iteration;
                let identities = (fixed.0.identity, fixed.1.identity);
                for ordinal in 0..count {
                    layer(&mut legacy.0, &mut legacy.1, ordinal);
                    std::mem::swap(&mut legacy.0, &mut legacy.1);
                    let (current, projection) =
                        layer_hidden_roles(ordinal, &mut fixed.0, &mut fixed.1);
                    layer(current, projection, ordinal);
                    assert_eq!(projection.value, legacy.0.value);
                    assert_eq!(projection.identity, legacy.0.identity);
                    assert_eq!((fixed.0.identity, fixed.1.identity), identities);
                }
                if count % 2 != 0 {
                    std::mem::swap(&mut fixed.0, &mut fixed.1);
                }
                assert_eq!(fixed, legacy, "layers={count} iteration={iteration}");
            }
        }
    }
}
