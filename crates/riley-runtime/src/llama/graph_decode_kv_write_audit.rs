//! Cold all-layer KV write mapping audit; reused sources are not in-flight activations.
use super::{LlamaBatchExecutorError, LlamaBatchExecutorResult, PreparedLlamaBatchExecutor};
use crate::llama::LlamaBatchRow;
use crate::llama::executor::buffers::BatchDeviceInput;
use crate::llama::executor::metadata::PackedIterationLayout;
use riley_cuda::{
    AttentionParentLayer, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream,
    PackedAttentionMetadataLayout, PackedBatchHostV1, PackedParentKvWriteGraph,
    PackedParentKvWriteResources,
};

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 KV write audit",
        reason,
    }
}

impl PreparedLlamaBatchExecutor {
    /// Writes and restores each real KV layer using an independent CPU scatter oracle.
    #[allow(clippy::too_many_lines)] // Keep mutation, validation and restoration in one transaction.
    pub(crate) fn audit_c07_kv_write(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("audit requires a healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1
            || self.owner.layout.head_dimension() != 64
        {
            return Err(rejected("initial KV write audit requires M=1 and D64"));
        }
        let packed = self.owner.metadata.pack(rows)?;
        if packed.total_input_tokens() != 1 || packed.row_count() != 1 {
            return Err(rejected(
                "initial KV write audit requires one sequence and token",
            ));
        }
        let layout = PackedIterationLayout::for_batch(&packed, 1)?;
        let BatchDeviceInput::IterationBatch { slab } = &mut self.owner.device_input else {
            return Err(rejected("audit requires packed metadata"));
        };
        layout.validate_u64_capacity(slab.byte_len())?;
        let kv = self.owner.layout;
        let result = (|| -> Result<(), AuditError> {
            let batch = PackedBatchHostV1::new(
                packed.block_row_offsets(),
                packed.physical_block_ids(),
                packed.valid_tokens(),
                packed.row_sequence_slots(),
                packed.position_ids(),
                kv.physical_block_count() as u64,
            )?;
            let fields = [
                layout.sequence_block_offsets,
                layout.physical_block_ids,
                layout.valid_tokens,
                layout.row_sequence_slots,
                layout.row_positions,
            ];
            let metadata = PackedAttentionMetadataLayout::new(
                slab.byte_len(),
                fields.map(|r| (r.offset as u64, r.byte_len as u64)),
            )?;
            let position = packed.position_ids()[0] as usize;
            let slot = packed.row_sequence_slots()[0] as usize;
            let logical = packed.block_row_offsets()[slot] as usize + position / 16;
            let physical = *packed
                .physical_block_ids()
                .get(logical)
                .ok_or(AuditError::Invalid("position has no physical block"))?
                as usize;
            let forward = &mut self.owner.forward;
            let original_k = read(&mut self.owner.key_cache, &mut forward.io_staging, stream)?;
            let original_v = read(&mut self.owner.value_cache, &mut forward.io_staging, stream)?;
            let source_k = read(
                &mut forward.buffers.key_rotary,
                &mut forward.io_staging,
                stream,
            )?;
            let source_v = read(
                &mut forward.buffers.value_raw,
                &mut forward.io_staging,
                stream,
            )?;
            let original_metadata = read(slab, &mut forward.io_staging, stream)?;
            for layer in 0..kv.layer_count() {
                let geometry = AttentionParentLayer::new(
                    kv.layer_count() as u64,
                    layer as u64,
                    kv.physical_block_count() as u64,
                    kv.key_value_head_count() as u64,
                )?;
                if kv.layer_byte_offset(layer) != Some(geometry.byte_offset())
                    || kv.layer_stride_bytes() != geometry.byte_len()
                    || kv.bytes_per_kind() != geometry.parent_byte_len()
                {
                    return Err(AuditError::Invalid("native and executor KV layout differ"));
                }
                let mut graph = PackedParentKvWriteGraph::prepare(
                    PackedParentKvWriteResources {
                        stream,
                        key_source: &mut forward.buffers.key_rotary,
                        value_source: &mut forward.buffers.value_raw,
                        key_parent: &mut self.owner.key_cache,
                        value_parent: &mut self.owner.value_cache,
                        metadata_slab: slab,
                    },
                    geometry,
                    metadata,
                    batch,
                    &mut forward.io_staging,
                )?;
                for _ in 0..8 {
                    graph.replay()?;
                }
                graph.close()?;
                for (pool, original, source) in [
                    (&mut self.owner.key_cache, &original_k, &source_k),
                    (&mut self.owner.value_cache, &original_v, &source_v),
                ] {
                    let mut expected = original.clone();
                    let base = usize::try_from(geometry.byte_offset())
                        .map_err(|_| AuditError::Invalid("layer offset exceeds host range"))?;
                    for head in 0..kv.key_value_head_count() {
                        let dst = base
                            + (physical * kv.key_value_head_count() * 16 * 64
                                + head * 16 * 64
                                + (position % 16) * 64)
                                * 2;
                        let src = head * 64 * 2;
                        expected[dst..dst + 128].copy_from_slice(&source[src..src + 128]);
                    }
                    if read(pool, &mut forward.io_staging, stream)? != expected {
                        return Err(AuditError::Invalid(
                            "KV write differs from CPU scatter or changed another layer",
                        ));
                    }
                    pool.upload_from_slice(0, original, &mut forward.io_staging, stream)?;
                    if read(pool, &mut forward.io_staging, stream)? != *original {
                        return Err(AuditError::Invalid("KV parent restoration failed"));
                    }
                }
                if read(slab, &mut forward.io_staging, stream)? != original_metadata
                    || read(
                        &mut forward.buffers.key_rotary,
                        &mut forward.io_staging,
                        stream,
                    )? != source_k
                    || read(
                        &mut forward.buffers.value_raw,
                        &mut forward.io_staging,
                        stream,
                    )? != source_v
                {
                    return Err(AuditError::Invalid(
                        "KV write modified a source or metadata",
                    ));
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.owner.poisoned = true;
            return Err(match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => crate::llama::executor::error::cuda_error(
                    crate::llama::ExecutionSite::global(crate::llama::LlamaOp::KvCacheWrite),
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
    let len = usize::try_from(buffer.byte_len())
        .map_err(|_| AuditError::Invalid("buffer exceeds host range"))?;
    let mut bytes = vec![0; len];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
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
