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
    CudaCommandStream, CudaDType, CudaDeviceBuffer, CudaExecutionStream, CudaStepGraph, CudaStream,
    DecodePartialReductionOrder, EmbeddingParams, GatedMultiplyParams, IndexedRopeParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams, PackedBatchHostV1, PackedBatchV1,
    QkvRopeParams, QkvSplitBiasParams, RaggedPagedAttentionParams, RaggedPagedKvCacheWriteParams,
    ResidualAddParams, ResidualRmsNormParams, RmsNormParams, RowGatherParams, SiluParams,
    abort_step_capture, begin_step_capture, deterministic_bf16_argmax,
    deterministic_bf16_argmax_bounded, embedding, end_step_capture, fixed37_ragged_paged_attention,
    gated_multiply, grouped_ragged_paged_attention, indexed_rope, launch_step_graph,
    native_bf16_ragged_paged_split_gqa_d128_global_merge_prompt_mma,
    native_bf16_ragged_paged_split_gqa_d128_shared_kv_global_merge,
    native_bf16_ragged_paged_split_gqa_d128_two_stage_shared_kv, qkv_split_bias,
    qkv_split_bias_rope, ragged_paged_attention, ragged_paged_kv_cache_write, residual_add,
    row_gather, set_step_capture_dynamic_counts, silu, silu_gated_multiply,
    validate_replayed_decode_step,
};

use super::super::batch::{LlamaBatchRowKind, LlamaPackedBatchMetadata};
use super::super::forward::{
    ForwardBuffers, GemmPlans, LlamaForwardError, LlamaRmsNormProfile, PreparedLlamaForward,
    PreparedLlamaGemm, execute_gemm, execute_profile_residual_rms_norm, execute_profile_rms_norm,
    execute_projection_bias, execute_projection_with_optional_bias_epilogue, poison_for_cuda_error,
    span, span_mut, weight_span,
};
use super::super::{ExecutionSite, LlamaExecutionPlan, LlamaOp};
use super::buffers::{BatchDeviceInput, BatchHostInput, STEP_COUNT_WORDS, U16_BYTES, U32_BYTES};
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
                if has_tensor_core_prompt_rows(&packed) {
                    for row in 0..active {
                        host.row_kinds[row * U32_BYTES..(row + 1) * U32_BYTES].copy_from_slice(
                            &u32::from(is_tensor_core_prompt_row(&packed, row)).to_ne_bytes(),
                        );
                    }
                    host.row_kinds[active * U32_BYTES..dense_rows * U32_BYTES].fill(0);
                    device
                        .row_positions
                        .upload_from_slice(
                            row_kinds_offset,
                            &host.row_kinds[..dense_rows * U32_BYTES],
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

    // A step is recordable when its outputs are selected on the device. The
    // recorded kernels read the active and output row counts from the device,
    // so one graph per (dense shape, output LM-head shape) serves decode-only
    // and prefill-mixed steps alike.
    let step_capture_eligible = config.decode_step_capture()
        && produce_greedy_tokens
        && active != 0
        && active <= dense_rows;
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
            buffers,
            captured_steps: maximum_captured_steps,
            captured_step_budget,
            ..
        } = forward;
        // A mixed step's output rows are fewer than its dense rows. Their LM
        // head can run on the smallest prepared bucket that holds them (and
        // fits the gathered-logits buffer) instead of on every dense row.
        let mut output_rows_bucket = 0_usize;
        let output_rows_index = if config.output_row_lm_head()
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
                .min_by_key(|(_, shape)| shape.dense_rows)
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
        let prompt_rows_in_step =
            config.prompt_attention_mma() && has_tensor_core_prompt_rows(&packed);
        let capture_key = capture_key * 2 + usize::from(prompt_rows_in_step);
        let recorded = captured_steps.as_deref().and_then(|steps| {
            steps
                .iter()
                .find(|(key, _)| *key == capture_key)
                .map(|(_, graph)| graph)
        });
        let replaying = recorded.is_some();
        let capturing = captured_steps.is_some() && !replaying && *captured_step_budget != 0;
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
            prompt_rows_in_step
                .then(|| (bounds.max_input_tokens() + STEP_COUNT_WORDS) as u64),
            config.fused_silu_multiply(),
            config.fused_qkv_rope(),
            config.fused_kv_write(),
            config.fused_mlp_residual_norm(),
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
        if replaying {
            return fixed;
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
            return outputs;
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
            execute_iteration_command_batch(stream, dispatch_disposition, |commands| {
                execute_iteration_body(
                    views.batch,
                    views.token_ids,
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
    }
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
    prompt_row_kinds_first_word: Option<u64>,
    fused_silu_multiply: bool,
    fused_qkv_rope: bool,
    fused_kv_write: bool,
    fused_mlp_residual_norm: bool,
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

    // The embedding reads its error report back on the host, so it always
    // runs eagerly; everything after it is recordable.
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
            execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
                .map_err(|source| batch_cuda(input_norm_site, source))?;
        }

        let rope_in_split = fused_qkv_rope && gemms.merged_qkv_plan().is_some();
        if gemms.merged_qkv_plan().is_some() {
            // One GEMM over the concatenated Q|K|V weights. `gate_raw` is idle
            // until the MLP and is wide enough to hold the merged rows.
            let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            let merged_width = hidden + 2 * key_value_width;
            let merged_bytes = dense_rows * merged_width * 2;
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
            let missing_bias = LlamaForwardError::InvalidConfiguration {
                field: "projection_bias_mode",
                reason: "merged QKV projection requires Q/K/V biases",
            };
            let (Some(query_bias), Some(key_bias), Some(value_bias)) =
                (layer.query_bias(), layer.key_bias(), layer.value_bias())
            else {
                return Err(missing_bias.into());
            };
            if let Some(plan) = gemms.merged_qkv_plan() {
                execute_gemm(
                    plan,
                    &buffers.hidden_norm,
                    merged_weight,
                    &mut buffers.gate_raw,
                    &mut buffers.gemm_workspace,
                    stream,
                    query_site,
                )?;
            }
            let mut params = QkvSplitBiasParams {
                merged: span(&buffers.gate_raw, CudaDType::BF16, merged_bytes, query_site)?,
                query_bias: weight_span(weights, query_bias, query_site)?,
                key_bias: weight_span(weights, key_bias, query_site)?,
                value_bias: weight_span(weights, value_bias, query_site)?,
                query: span_mut(
                    if fused_qkv_rope {
                        &mut buffers.hidden_rotary
                    } else {
                        &mut *hidden_projection
                    },
                    CudaDType::BF16,
                    dense_rows * hidden * 2,
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
                query_width: hidden,
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
                };
                qkv_split_bias_rope(&mut params, &rope, stream)
                    .map_err(|source| batch_cuda(query_site, source))?;
            } else {
                qkv_split_bias(&mut params, stream)
                    .map_err(|source| batch_cuda(query_site, source))?;
            }
        } else {
            let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            let (strict, fused) = gemms.query_projection_plans();
            execute_projection_with_optional_bias_epilogue(
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
            )?;
            let key_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
            let (strict, fused) = gemms.key_value_projection_plans();
            execute_projection_with_optional_bias_epilogue(
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
            )?;
            let value_site = ExecutionSite::layer(layer_index, LlamaOp::ValueProjection);
            let (strict, fused) = gemms.key_value_projection_plans();
            execute_projection_with_optional_bias_epilogue(
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
            )?;
        }

        let query_rope_site = ExecutionSite::layer(layer_index, LlamaOp::QueryRope);
        if !rope_in_split {
            let mut params = IndexedRopeParams {
                input: span(
                    hidden_projection,
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
        if attention_implementation
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
                    plan.workspace_spec().hidden_buffer_bytes(),
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
            if let (true, Some(first_word)) = (d128_global_merge, prompt_row_kinds_first_word) {
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
                    RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage => {
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
        execute_gemm(
            &mut gemms.hidden,
            &buffers.hidden_context,
            weight_span(weights, layer.output_weight(), output_site)?,
            hidden_projection,
            &mut buffers.gemm_workspace,
            stream,
            output_site,
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
                execute_profile_rms_norm(rms_norm_profile, &mut norm, stream)
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
        let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
        execute_gemm(
            &mut gemms.down,
            &buffers.gated_product,
            weight_span(weights, layer.down_weight(), down_site)?,
            hidden_current,
            &mut buffers.gemm_workspace,
            stream,
            down_site,
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
        execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
            .map_err(|source| batch_cuda(final_norm_site, source))?;
    }
    let lm_head_site = ExecutionSite::global(LlamaOp::LmHead);
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
        execute_gemm(
            output_rows.gemm,
            &buffers.hidden_rotary,
            weight_span(weights, plan.lm_head_weight(), lm_head_site)?,
            output_rows.gathered_logits,
            &mut buffers.gemm_workspace,
            stream,
            lm_head_site,
        )?;
        return Ok(());
    }
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

/// First position whose prompt token takes the tensor-core attention producer.
/// The split is by the token's own position, so a token takes the same path in
/// every batch and chunk. Below it the shared-KV producer is the cheaper one
/// for the few rows of a prompt that barely crosses the bound, and a step
/// whose prompt rows all sit below it launches nothing extra.
const TENSOR_CORE_PROMPT_MIN_POSITION: u32 = 256;

fn is_tensor_core_prompt_row(packed: &LlamaPackedBatchMetadata<'_>, row: usize) -> bool {
    let slot = packed.row_sequence_slots()[row] as usize;
    packed
        .row_kind_codes()
        .get(slot)
        .is_some_and(|code| *code == LlamaBatchRowKind::Prefill.code())
        && packed.position_ids()[row] >= TENSOR_CORE_PROMPT_MIN_POSITION
}

fn has_tensor_core_prompt_rows(packed: &LlamaPackedBatchMetadata<'_>) -> bool {
    !packed.prefill_row_indices().is_empty()
        && (0..packed.total_input_tokens()).any(|row| is_tensor_core_prompt_row(packed, row))
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
