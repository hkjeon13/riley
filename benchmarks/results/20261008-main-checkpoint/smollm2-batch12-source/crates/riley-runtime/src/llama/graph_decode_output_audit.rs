//! Cold actual output-chain capture, including D2H into original pinned staging.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::{
    buffers::BatchDeviceInput,
    error::{LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error},
    metadata::PackedIterationLayout,
    output::decode_greedy_tokens,
};
use crate::llama::{ExecutionSite, LlamaBatchRow, LlamaOp};
use riley_cuda::{
    BorrowedOutputGraph, BorrowedOutputResources, CudaDeviceBuffer, CudaPinnedHostBuffer,
    CudaStream,
};
fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 output audit",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    #[allow(clippy::too_many_lines)] // Keep the snapshot/capture/restore transaction together.
    pub(crate) fn audit_c07_output(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<u32> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("output audit needs healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1 || self.output_count != 1 {
            return Err(rejected("output audit requires M=1 one output"));
        }
        let packed = self.owner.metadata.pack(rows)?;
        let layout = PackedIterationLayout::for_batch(&packed, 1)?;
        let indices = packed.output_token_indices();
        if indices.len() != 1 {
            return Err(rejected("output map must contain one row"));
        }
        let BatchDeviceInput::IterationBatch { slab } = &mut self.owner.device_input else {
            return Err(rejected("packed slab required"));
        };
        layout.validate_u64_capacity(slab.byte_len())?;
        let f = &mut self.owner.forward;
        let gathered = self
            .owner
            .gathered_logits
            .as_mut()
            .ok_or_else(|| rejected("gathered owner missing"))?;
        let results = self
            .owner
            .greedy_results
            .as_mut()
            .ok_or_else(|| rejected("result owner missing"))?;
        let vocab = f.plan.dimensions().vocabulary_size();
        let outcome = (|| -> Result<Vec<u8>, AuditError> {
            let slab_before = read(slab, &mut f.io_staging, stream)?;
            let offset = layout.output_token_indices.offset;
            if slab_before[offset..offset + 4] != indices[0].to_ne_bytes() {
                return Err(AuditError::Invalid("stale device output map"));
            }
            let input = read(&mut f.buffers.logits, &mut f.io_staging, stream)?;
            let gathered_before = read(gathered, &mut f.io_staging, stream)?;
            let results_before = read(results, &mut f.io_staging, stream)?;
            let mut pinned_before = vec![
                0;
                usize::try_from(f.io_staging.byte_len())
                    .map_err(|_| AuditError::Invalid("host range"))?
            ];
            f.io_staging.read(0, &mut pinned_before)?;
            let mut graph = BorrowedOutputGraph::prepare(
                BorrowedOutputResources {
                    stream,
                    input: &mut f.buffers.logits,
                    indices: slab,
                    gathered,
                    output: results,
                    pinned: &mut f.io_staging,
                },
                1,
                indices,
                vocab as u64,
                offset as u64,
            )?;
            let mut records = Vec::new();
            for _ in 0..16 {
                records = graph.replay()?;
            }
            graph.close()?;
            let mut pinned_after = vec![0; pinned_before.len()];
            f.io_staging.read(0, &mut pinned_after)?;
            if pinned_after[8..] != pinned_before[8..] || pinned_after[..8] != records {
                return Err(AuditError::Invalid("pinned result or tail mismatch"));
            }
            if read(slab, &mut f.io_staging, stream)? != slab_before
                || read(&mut f.buffers.logits, &mut f.io_staging, stream)? != input
            {
                return Err(AuditError::Invalid("output input/metadata changed"));
            }
            if read(gathered, &mut f.io_staging, stream)? != input
                || read(results, &mut f.io_staging, stream)?[..8] != records
            {
                return Err(AuditError::Invalid("gather/result mismatch"));
            }
            gathered.upload_from_slice(0, &gathered_before, &mut f.io_staging, stream)?;
            results.upload_from_slice(0, &results_before, &mut f.io_staging, stream)?;
            if read(gathered, &mut f.io_staging, stream)? != gathered_before
                || read(results, &mut f.io_staging, stream)? != results_before
            {
                return Err(AuditError::Invalid("output restoration failed"));
            }
            f.io_staging.write(0, &pinned_before)?;
            Ok(records)
        })();
        let records = outcome.map_err(|error| {
            self.owner.poisoned = true;
            match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => {
                    cuda_error(ExecutionSite::global(LlamaOp::OutputGather), source)
                }
            }
        })?;
        let mut tokens = [0];
        if let Err(error) = decode_greedy_tokens(&records, vocab, &mut tokens) {
            if !matches!(error, LlamaBatchExecutorError::GreedyLogitsNonFinite { .. }) {
                self.owner.poisoned = true;
            }
            return Err(error);
        }
        Ok(tokens[0])
    }
}
fn read(
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> Result<Vec<u8>, AuditError> {
    let mut bytes =
        vec![0; usize::try_from(buffer.byte_len()).map_err(|_| AuditError::Invalid("host range"))?];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}
#[derive(Debug)]
enum AuditError {
    Cuda(riley_cuda::CudaError),
    Invalid(&'static str),
}
impl From<riley_cuda::CudaError> for AuditError {
    fn from(e: riley_cuda::CudaError) -> Self {
        Self::Cuda(e)
    }
}
