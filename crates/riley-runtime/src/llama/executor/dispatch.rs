//! Borrowed CUDA output primitives for the Llama batch executor.
//!
//! The enclosing owner keeps metadata transport, fixed-graph execution,
//! output-ready state, failure routing, allocation, and close ordering. This
//! component binds already prepared output buffers after the fixed graph has
//! produced logits.

use riley_cuda::{
    Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaDeviceBuffer,
    CudaExecutionStream, RowGatherParams, deterministic_bf16_argmax, row_gather,
};

use super::super::forward::span;
use super::super::{ExecutionSite, LlamaOp};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    cuda_error as dispatch_cuda,
};
use super::output::{greedy_result_bytes, output_logits_bytes};

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

fn usize_u64(value: usize, resource: LlamaBatchExecutorResource) -> LlamaBatchExecutorResult<u64> {
    u64::try_from(value).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow { resource })
}
