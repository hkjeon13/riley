//! Borrowed CUDA dispatch bindings for the Llama batch executor.
//!
//! The enclosing owner keeps metadata transport, fixed-graph execution,
//! output-ready state, failure routing, allocation, and close ordering. This
//! component runs borrowed command-batch completion guards and binds already
//! prepared output buffers after the fixed graph has produced logits.

use riley_cuda::{
    Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut, CudaCommandStream, CudaDType,
    CudaDeviceBuffer, CudaExecutionStream, CudaStream, RowGatherParams, deterministic_bf16_argmax,
    row_gather,
};

use super::super::forward::span;
use super::super::{ExecutionSite, LlamaOp};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    cuda_error as dispatch_cuda, usize_u64,
};
use super::output::{greedy_result_bytes, output_logits_bytes};

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
