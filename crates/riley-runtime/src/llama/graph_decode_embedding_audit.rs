//! Cold actual embedding owner audit; full decode admission remains separate.
use super::{LlamaBatchExecutorError, LlamaBatchExecutorResult, PreparedLlamaBatchExecutor};
use crate::llama::LlamaBatchRow;
use crate::llama::executor::buffers::BatchDeviceInput;
use crate::llama::executor::metadata::PackedIterationLayout;
use riley_cuda::{
    Bf16EmbeddingStatusD2HStatus, BorrowedEmbeddingGraph, BorrowedEmbeddingResources,
    CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream, EmbeddingGraphGeometry,
};
fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 embedding audit",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    /// A caller-owned cold report avoids adding allocations to normal dispatch.
    pub(crate) fn audit_c07_embedding(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
        report: &mut CudaPinnedHostBuffer,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("audit requires a healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1 {
            return Err(rejected("initial embedding audit requires M=1"));
        }
        let packed = self.owner.metadata.pack(rows)?;
        if packed.total_input_tokens() != 1 {
            return Err(rejected("audit requires one active token"));
        }
        let layout = PackedIterationLayout::for_batch(&packed, 1)?;
        if layout.token_ids.offset != 0 || layout.token_ids.byte_len != 4 {
            return Err(rejected(
                "native capture requires a token prefix at byte zero",
            ));
        }
        let BatchDeviceInput::IterationBatch { slab } = &mut self.owner.device_input else {
            return Err(rejected("audit requires actual packed token metadata"));
        };
        layout.validate_u64_capacity(slab.byte_len())?;
        let forward = &mut self.owner.forward;
        let dims = forward.plan.dimensions();
        let weight_id = forward.plan.embedding_weight();
        let meta = forward
            .weights
            .physical_metadata(weight_id)
            .ok_or_else(|| rejected("embedding weight belongs to another owner"))?;
        if meta.dtype != riley_tensor::DType::BF16
            || meta.shape != [dims.vocabulary_size(), dims.hidden_size()]
        {
            return Err(rejected("embedding physical table shape or dtype differs"));
        }
        let table = forward
            .weights
            .borrow_graph_weight(weight_id)
            .map_err(|_| rejected("embedding table cannot be borrowed"))?;
        let geometry = EmbeddingGraphGeometry {
            tokens: 1,
            vocabulary: dims.vocabulary_size() as u64,
            hidden: dims.hidden_size() as u64,
        };
        let result = probe(
            BorrowedEmbeddingResources {
                stream,
                table,
                token_ids: slab,
                output: &mut forward.buffers.hidden_current,
                error_scratch: &mut forward.buffers.embedding_error_scratch,
                report,
            },
            geometry,
            packed.input_token_ids()[0],
            &mut forward.io_staging,
        );
        if let Err(error) = result {
            self.owner.poisoned = true;
            return Err(match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => crate::llama::executor::error::cuda_error(
                    crate::llama::ExecutionSite::global(crate::llama::LlamaOp::Embedding),
                    source,
                ),
            });
        }
        Ok(())
    }
}
fn read(
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> Result<Vec<u8>, AuditError> {
    let n = usize::try_from(buffer.byte_len())
        .map_err(|_| AuditError::Invalid("buffer exceeds host address space"))?;
    let mut bytes = vec![0; n];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}
fn probe(
    resources: BorrowedEmbeddingResources<'_>,
    geometry: EmbeddingGraphGeometry,
    token: u32,
    staging: &mut CudaPinnedHostBuffer,
) -> Result<(), AuditError> {
    let BorrowedEmbeddingResources {
        stream,
        table,
        token_ids,
        output,
        error_scratch,
        report,
    } = resources;
    let input_before = read(token_ids, staging, stream)?;
    if input_before[..4] != token.to_le_bytes() || u64::from(token) >= geometry.vocabulary {
        return Err(AuditError::Invalid(
            "actual packed token differs or is outside vocabulary",
        ));
    }
    let table_before = read(table, staging, stream)?;
    let output_before = read(output, staging, stream)?;
    let scratch_before = read(error_scratch, staging, stream)?;
    let mut graph = BorrowedEmbeddingGraph::prepare(
        BorrowedEmbeddingResources {
            stream,
            table,
            token_ids,
            output,
            error_scratch,
            report,
        },
        geometry,
    )?;
    for _ in 0..16 {
        if graph.replay()? != Bf16EmbeddingStatusD2HStatus::Success {
            return Err(AuditError::Invalid(
                "valid embedding token produced an error status",
            ));
        }
    }
    graph.close()?;
    let actual = read(output, staging, stream)?;
    let bytes = usize::try_from(geometry.hidden * 2)
        .map_err(|_| AuditError::Invalid("row exceeds host address space"))?;
    let start = usize::try_from(u64::from(token) * geometry.hidden * 2)
        .map_err(|_| AuditError::Invalid("row offset exceeds host address space"))?;
    if actual[..bytes] != table_before[start..start + bytes]
        || actual[bytes..] != output_before[bytes..]
        || read(table, staging, stream)? != table_before
        || read(token_ids, staging, stream)? != input_before
    {
        return Err(AuditError::Invalid(
            "embedding row parity or owner preservation failed",
        ));
    }
    output.upload_from_slice(0, &output_before, staging, stream)?;
    error_scratch.upload_from_slice(0, &scratch_before, staging, stream)?;
    if read(output, staging, stream)? != output_before
        || read(error_scratch, staging, stream)? != scratch_before
    {
        return Err(AuditError::Invalid(
            "embedding workspace restoration failed",
        ));
    }
    Ok(())
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
