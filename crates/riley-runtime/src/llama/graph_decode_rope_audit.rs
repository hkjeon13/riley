//! Cold Q/K `RoPE` audit using the actual executor's packed position allocation.
//! No in-flight layer activation or full-graph admission claim.
use super::{LlamaBatchExecutorError, LlamaBatchExecutorResult, PreparedLlamaBatchExecutor};
use crate::llama::LlamaBatchRow;
use crate::llama::executor::buffers::BatchDeviceInput;
use crate::llama::executor::metadata::PackedIterationLayout;
use riley_cuda::{
    BorrowedRopeGraph, BorrowedRopeResources, CudaBufferSpan, CudaBufferSpanMut, CudaDType,
    CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream, IndexedRopeParams, RopeGraphGeometry,
    indexed_rope,
};

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 RoPE audit",
        reason,
    }
}

impl PreparedLlamaBatchExecutor {
    pub(crate) fn audit_c07_rope(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("audit needs a healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1 {
            return Err(rejected("initial RoPE audit requires exact M=1"));
        }
        let packed = self.owner.metadata.pack(rows)?;
        if packed.total_input_tokens() != 1 {
            return Err(rejected(
                "initial RoPE audit requires exactly one active row",
            ));
        }
        let layout = PackedIterationLayout::for_batch(&packed, 1)?;
        let positions_host = packed.position_ids();
        let BatchDeviceInput::IterationBatch { slab } = &mut self.owner.device_input else {
            return Err(rejected("audit requires the actual packed metadata slab"));
        };
        layout.validate_u64_capacity(slab.byte_len())?;
        let forward = &mut self.owner.forward;
        let dimensions = forward.plan.dimensions();
        let head_size = dimensions.head_dimension() as u64;
        let shape = RopeGraphGeometry {
            rows: 1,
            heads: dimensions.query_heads() as u64,
            head_size,
            rotary_dimension: head_size,
            table_positions: self.owner.absolute_rope_cos.byte_len() / (head_size / 2 * 4),
            positions_byte_offset: layout.row_positions.offset as u64,
        };
        let result = (|| {
            let actual = read(slab, &mut forward.io_staging, stream)?;
            let offset = layout.row_positions.offset;
            if actual[offset..offset + 4] != positions_host[0].to_le_bytes() {
                return Err(AuditError::Invalid(
                    "packed device position differs from this iteration's host mirror",
                ));
            }
            probe(
                BorrowedRopeResources {
                    stream,
                    input: &mut forward.buffers.hidden_projection,
                    cos: &mut self.owner.absolute_rope_cos,
                    sin: &mut self.owner.absolute_rope_sin,
                    positions: slab,
                    output: &mut forward.buffers.hidden_rotary,
                },
                shape,
                positions_host,
                &mut forward.io_staging,
            )?;
            probe(
                BorrowedRopeResources {
                    stream,
                    input: &mut forward.buffers.key_raw,
                    cos: &mut self.owner.absolute_rope_cos,
                    sin: &mut self.owner.absolute_rope_sin,
                    positions: slab,
                    output: &mut forward.buffers.key_rotary,
                },
                RopeGraphGeometry {
                    heads: dimensions.key_value_heads() as u64,
                    ..shape
                },
                positions_host,
                &mut forward.io_staging,
            )
        })();
        if let Err(source) = result {
            self.owner.poisoned = true;
            return Err(match source {
                AuditError::Cuda(source) => crate::llama::executor::error::cuda_error(
                    crate::llama::ExecutionSite::global(crate::llama::LlamaOp::QueryRope),
                    source,
                ),
                AuditError::Invalid(reason) => rejected(reason),
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
    let len = usize::try_from(buffer.byte_len())
        .map_err(|_| AuditError::Invalid("buffer exceeds host address space"))?;
    let mut bytes = vec![0; len];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}

fn probe(
    resources: BorrowedRopeResources<'_>,
    shape: RopeGraphGeometry,
    positions_host: &[u32],
    staging: &mut CudaPinnedHostBuffer,
) -> Result<(), AuditError> {
    let BorrowedRopeResources {
        stream,
        input,
        cos,
        sin,
        positions,
        output,
    } = resources;
    let original = read(output, staging, stream)?;
    let before = [
        read(input, staging, stream)?,
        read(cos, staging, stream)?,
        read(sin, staging, stream)?,
        read(positions, staging, stream)?,
    ];
    let mut graph = BorrowedRopeGraph::prepare(
        BorrowedRopeResources {
            stream,
            input,
            cos,
            sin,
            positions,
            output,
        },
        shape,
        positions_host,
    )?;
    for _ in 0..16 {
        graph.replay()?;
    }
    graph.close()?;
    let captured = read(output, staging, stream)?;
    let bytes = shape.rows * shape.heads * shape.head_size * 2;
    indexed_rope(
        &mut IndexedRopeParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, bytes)?,
            cos: CudaBufferSpan::new(cos, CudaDType::F32, 0, cos.byte_len())?,
            sin: CudaBufferSpan::new(sin, CudaDType::F32, 0, sin.byte_len())?,
            positions: CudaBufferSpan::new(
                positions,
                CudaDType::U32,
                shape.positions_byte_offset,
                shape.rows * 4,
            )?,
            positions_host,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, bytes)?,
            head_count: shape.heads,
            head_size: shape.head_size,
            rotary_dimension: shape.rotary_dimension,
            table_position_count: shape.table_positions,
        },
        stream,
    )?;
    let eager = read(output, staging, stream)?;
    let after = [
        read(input, staging, stream)?,
        read(cos, staging, stream)?,
        read(sin, staging, stream)?,
        read(positions, staging, stream)?,
    ];
    let used = usize::try_from(bytes)
        .map_err(|_| AuditError::Invalid("RoPE span exceeds host address space"))?;
    if captured != eager
        || before != after
        || captured[used..] != original[used..]
        || captured[..used]
            .chunks_exact(2)
            .any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(AuditError::Invalid(
            "graph/eager parity, finite output or owner preservation failed",
        ));
    }
    output.upload_from_slice(0, &original, staging, stream)?;
    if read(output, staging, stream)? != original {
        return Err(AuditError::Invalid("output restoration failed"));
    }
    Ok(())
}

#[derive(Debug)]
enum AuditError {
    Cuda(riley_cuda::CudaError),
    Invalid(&'static str),
}
impl From<riley_cuda::CudaError> for AuditError {
    fn from(source: riley_cuda::CudaError) -> Self {
        Self::Cuda(source)
    }
}
