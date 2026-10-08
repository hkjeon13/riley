//! Live C07 attention evidence from canonical metadata and measured byte parity.
//! No global capability is promoted: evidence belongs to this retained owner.

use riley_cuda::{
    AttentionParentLayer, CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaStream, PackedAttentionMetadataLayout, PackedBatchHostV1,
    PackedBatchV1, PackedParentAttentionResources, ParentAttentionGraph,
    RaggedPagedAttentionParams, grouped_ragged_paged_attention,
};
use sha2::{Digest, Sha256};

use super::executor::error::{LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error};
use super::graph::GraphOperatorCapability;
use super::graph_decode_capture_inventory::{
    PureDecodeGraphV1CaptureCapabilityInventory, PureDecodeGraphV1CaptureOperation,
};
use super::graph_decode_exact_device_slab::{
    PureDecodeGraphV1ExactDeviceSlab, pure_decode_graph_v1_exact_metadata_layouts_match,
};
use super::graph_decode_exact_host_slab::PureDecodeGraphV1ExactHostSlabLease;
use super::graph_decode_layout::{
    PureDecodeGraphMetadataField as Field, PureDecodeGraphMetadataLayout,
};
use super::{ExecutionSite, LlamaOp};
use crate::paged_kv::KvLayout;

const FIELDS: [Field; 5] = [
    Field::SequenceBlockOffsets,
    Field::PhysicalBlockIds,
    Field::ValidTokens,
    Field::RowSequenceSlots,
    Field::PositionIds,
];

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 attention owner",
        reason,
    }
}

/// A retained native graph and same-allocation eager/graph parity evidence.
/// The fields cannot be created from a supplied boolean or scalar allocation ID.
pub(crate) struct C07AttentionGraphOwner<'a> {
    graph: Option<ParentAttentionGraph<'a>>,
    executor_poisoned: &'a mut bool,
    layout: PureDecodeGraphMetadataLayout,
    kv_layout: KvLayout,
    layer_index: usize,
    query_heads: u64,
    payload_sha256: [u8; 32],
    output_sha256: [u8; 32],
    terminal: bool,
}

impl<'a> C07AttentionGraphOwner<'a> {
    /// Cold prepares canonical metadata, measures eager/graph parity, closes the
    /// probe, then recaptures unchanged inputs under the same exclusive borrows.
    /// Output/metadata downloads occur only after known graph close. Extra work
    /// is cold validation and never enters the normal decode dispatch path.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn prepare(
        stream: &'a mut CudaStream,
        query: &'a mut CudaDeviceBuffer,
        key: &'a mut CudaDeviceBuffer,
        value: &'a mut CudaDeviceBuffer,
        output: &'a mut CudaDeviceBuffer,
        device: &'a mut PureDecodeGraphV1ExactDeviceSlab,
        staging: &mut CudaPinnedHostBuffer,
        source: PureDecodeGraphV1ExactHostSlabLease<'_>,
        expected_layout: PureDecodeGraphMetadataLayout,
        kv_layout: KvLayout,
        layer_index: usize,
        batch: PackedBatchHostV1<'_>,
        query_heads: u64,
        executor_poisoned: &'a mut bool,
    ) -> LlamaBatchExecutorResult<Self> {
        let site = ExecutionSite::layer(layer_index, LlamaOp::RaggedPagedAttention);
        if *executor_poisoned {
            return Err(rejected("executor is poisoned"));
        }
        let mut cuda = |e| {
            super::forward::poison_for_cuda_error(executor_poisoned, &e);
            cuda_error(site, e)
        };
        if !pure_decode_graph_v1_exact_metadata_layouts_match(expected_layout, source.layout())
            || !pure_decode_graph_v1_exact_metadata_layouts_match(expected_layout, device.layout())
            || source.geometry_digest() != expected_layout.geometry_digest()
            || device.geometry_digest() != expected_layout.geometry_digest()
        {
            return Err(rejected("complete metadata layouts differ"));
        }
        if kv_layout.head_dimension() != 64
            || batch.active_row_count() != u64::from(expected_layout.bucket_rows())
        {
            return Err(rejected(
                "requires exact pure-decode bucket rows and D64 KV layout",
            ));
        }
        let layer = AttentionParentLayer::new(
            kv_layout.layer_count() as u64,
            layer_index as u64,
            kv_layout.physical_block_count() as u64,
            kv_layout.key_value_head_count() as u64,
        )
        .map_err(&mut cuda)?;
        if kv_layout.layer_byte_offset(layer_index) != Some(layer.byte_offset())
            || kv_layout.layer_stride_bytes() != layer.byte_len()
            || kv_layout.bytes_per_kind() != layer.parent_byte_len()
        {
            return Err(rejected("KV layout differs from native layer span"));
        }
        let regions = FIELDS.map(|field| {
            let region = expected_layout.region(field);
            (region.offset(), region.byte_len())
        });
        let metadata_layout =
            PackedAttentionMetadataLayout::new(expected_layout.total_bytes(), regions)
                .map_err(&mut cuda)?;
        let slab = device.borrow_for_attention_capture();
        let rows = u64::from(expected_layout.bucket_rows());
        // The packed prepare validates payload bytes against batch before H2D.
        // This first capture is the parity probe and carries no inventory evidence.
        let mut probe = ParentAttentionGraph::prepare_packed(
            PackedParentAttentionResources {
                stream,
                query,
                key_parent: key,
                value_parent: value,
                output,
                metadata_slab: slab,
            },
            layer,
            metadata_layout,
            source.bytes(),
            staging,
            batch,
            query_heads,
            rows,
            0.125,
        )
        .map_err(&mut cuda)?;
        probe.replay().map_err(&mut cuda)?;
        probe.close().map_err(&mut cuda)?;
        let output_bytes = rows
            .checked_mul(query_heads)
            .and_then(|n| n.checked_mul(128))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| rejected("output length overflow"))?;
        let mut graph_bytes = vec![0; output_bytes];
        output
            .download_to_slice(0, &mut graph_bytes, staging, stream)
            .map_err(&mut cuda)?;
        let mut slab_bytes = vec![0; source.bytes().len()];
        slab.download_to_slice(0, &mut slab_bytes, staging, stream)
            .map_err(&mut cuda)?;
        if slab_bytes != source.bytes() {
            return Err(rejected("graph mutated canonical metadata"));
        }
        let view = |i: usize, dtype| CudaBufferSpan::new(slab, dtype, regions[i].0, regions[i].1);
        let packed = PackedBatchV1::new(
            batch,
            view(0, CudaDType::U32).map_err(&mut cuda)?,
            view(1, CudaDType::U32).map_err(&mut cuda)?,
            view(2, CudaDType::U16).map_err(&mut cuda)?,
            view(3, CudaDType::U32).map_err(&mut cuda)?,
            view(4, CudaDType::U32).map_err(&mut cuda)?,
        )
        .map_err(&mut cuda)?;
        grouped_ragged_paged_attention(
            &mut RaggedPagedAttentionParams {
                query: CudaBufferSpan::new(query, CudaDType::BF16, 0, output_bytes as u64)
                    .map_err(&mut cuda)?,
                key_pool: CudaBufferSpan::new(
                    key,
                    CudaDType::BF16,
                    layer.byte_offset(),
                    layer.byte_len(),
                )
                .map_err(&mut cuda)?,
                value_pool: CudaBufferSpan::new(
                    value,
                    CudaDType::BF16,
                    layer.byte_offset(),
                    layer.byte_len(),
                )
                .map_err(&mut cuda)?,
                output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, output_bytes as u64)
                    .map_err(&mut cuda)?,
                batch: packed,
                query_head_count: query_heads,
                key_value_head_count: kv_layout.key_value_head_count() as u64,
                head_size: 64,
                output_row_count: rows,
                scale: 0.125,
            },
            stream,
        )
        .map_err(&mut cuda)?;
        stream.synchronize().map_err(&mut cuda)?;
        let mut eager_bytes = vec![0; output_bytes];
        output
            .download_to_slice(0, &mut eager_bytes, staging, stream)
            .map_err(&mut cuda)?;
        if eager_bytes != graph_bytes {
            return Err(rejected("same-owner eager/graph output bytes differ"));
        }
        // An all-NaN match must not become readiness evidence.
        if eager_bytes.chunks_exact(2).any(|word| {
            let bits = u16::from_le_bytes([word[0], word[1]]);
            bits & 0x7f80 == 0x7f80
        }) {
            return Err(rejected("attention parity output is not finite"));
        }
        let payload_sha256 = Sha256::digest(&slab_bytes).into();
        let output_sha256 = Sha256::digest(&eager_bytes).into();
        let graph = ParentAttentionGraph::prepare_packed(
            PackedParentAttentionResources {
                stream,
                query,
                key_parent: key,
                value_parent: value,
                output,
                metadata_slab: slab,
            },
            layer,
            metadata_layout,
            source.bytes(),
            staging,
            batch,
            query_heads,
            rows,
            0.125,
        )
        .map_err(&mut cuda)?;
        Ok(Self {
            graph: Some(graph),
            executor_poisoned,
            layout: expected_layout,
            kv_layout,
            layer_index,
            query_heads,
            payload_sha256,
            output_sha256,
            terminal: false,
        })
    }

    /// Supplies Attention only for this live, parity-checked layout/layer owner.
    /// Other slots retain their evidence; a partial chain remains Unknown.
    pub(crate) fn bind_inventory(
        &self,
        inventory: PureDecodeGraphV1CaptureCapabilityInventory,
        layout: PureDecodeGraphMetadataLayout,
        kv_layout: KvLayout,
        layer_index: usize,
        query_heads: u64,
    ) -> PureDecodeGraphV1CaptureCapabilityInventory {
        let matched = !self.terminal
            && self.layout == layout
            && self.kv_layout == kv_layout
            && self.layer_index == layer_index
            && self.query_heads == query_heads;
        inventory.with_capability(
            PureDecodeGraphV1CaptureOperation::Attention,
            if inventory.capability_for(PureDecodeGraphV1CaptureOperation::Attention)
                == GraphOperatorCapability::Unsupported
            {
                GraphOperatorCapability::Unsupported
            } else if matched {
                GraphOperatorCapability::Supported
            } else {
                GraphOperatorCapability::Unknown
            },
        )
    }

    pub(crate) fn replay(&mut self) -> riley_cuda::CudaResult<()> {
        let result = self
            .graph
            .as_mut()
            .expect("live owner retains its graph")
            .replay();
        if result.is_err() {
            self.terminal = true;
            *self.executor_poisoned = true;
        }
        result
    }

    pub(crate) fn parity_hashes(&self) -> ([u8; 32], [u8; 32]) {
        (self.payload_sha256, self.output_sha256)
    }

    pub(crate) fn close(mut self) -> riley_cuda::CudaResult<()> {
        let result = self
            .graph
            .take()
            .expect("live owner retains its graph")
            .close();
        if result.is_err() {
            *self.executor_poisoned = true;
        }
        result
    }
}

impl Drop for C07AttentionGraphOwner<'_> {
    fn drop(&mut self) {
        if let Some(graph) = self.graph.take() {
            if graph.close().is_err() {
                *self.executor_poisoned = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph_decode_exact_host_slab::{
        PureDecodeGraphV1ExactHostSlab, PureDecodeGraphV1ExactHostSlabWrite,
    };
    use crate::llama::graph_decode_layout::PureDecodeGraphMetadataLayoutSpec;
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

    #[test]
    #[ignore = "requires CUDA GPU; exact canonical C07 attention owner"]
    fn c07_packed_attention_owner_parity_binds_only_matching_live_inventory()
    -> Result<(), Box<dyn std::error::Error>> {
        let context = riley_cuda::CudaRuntime::initialize()?
            .device(0)?
            .create_context()?;
        let mut stream = context.create_stream()?;
        let layout = PureDecodeGraphMetadataLayout::try_new(
            PureDecodeGraphMetadataLayoutSpec::new(1, 1, 3, 5),
        )?;
        let bad_layout = PureDecodeGraphMetadataLayout::try_new(
            PureDecodeGraphMetadataLayoutSpec::new(1, 1, 4, 5),
        )?;
        let kv = KvLayout::checked(3, 1, 1, 64)?;
        let mut metadata =
            PreparedLlamaBatchMetadata::prepare(LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)?)?;
        let rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Decode,
            &[7],
            1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &[0], &[1], 1),
            Some(0),
        )];
        let packed = metadata.pack(&rows)?;
        let mut host = PureDecodeGraphV1ExactHostSlab::prepare(layout)?;
        let source = match host
            .write_exact_v1_leased(&packed, &[1, 2, 3], &[0; 5])
            .map_err(|e| format!("canonical write: {e:?}"))?
        {
            PureDecodeGraphV1ExactHostSlabWrite::Written(source) => source,
            _ => return Err("canonical fixture rejected".into()),
        };
        let batch = PackedBatchHostV1::new(&[0, 1], &[0], &[1], &[0], &[0], 1)?;
        let mut device = PureDecodeGraphV1ExactDeviceSlab::prepare(&context, layout)?;
        let mut staging = context.allocate_pinned_host_buffer(kv.bytes_per_kind())?;
        let mut query = context.allocate_device_buffer(128)?;
        let mut output = context.allocate_device_buffer(128)?;
        let mut key = context.allocate_device_buffer(kv.bytes_per_kind())?;
        let mut value = context.allocate_device_buffer(kv.bytes_per_kind())?;
        let ones = [0x80, 0x3f].repeat(kv.bytes_per_kind() as usize / 2);
        query.upload_from_slice(0, &ones[..128], &mut staging, &mut stream)?;
        key.upload_from_slice(0, &ones, &mut staging, &mut stream)?;
        value.upload_from_slice(0, &ones, &mut staging, &mut stream)?;
        let stable = context.allocation_stats()?;
        let mut poisoned = false;
        assert!(
            C07AttentionGraphOwner::prepare(
                &mut stream,
                &mut query,
                &mut key,
                &mut value,
                &mut output,
                &mut device,
                &mut staging,
                source,
                bad_layout,
                kv,
                1,
                batch,
                1,
                &mut poisoned,
            )
            .is_err()
        );
        let mut owner = C07AttentionGraphOwner::prepare(
            &mut stream,
            &mut query,
            &mut key,
            &mut value,
            &mut output,
            &mut device,
            &mut staging,
            source,
            layout,
            kv,
            1,
            batch,
            1,
            &mut poisoned,
        )?;
        let empty = PureDecodeGraphV1CaptureCapabilityInventory::default();
        let bound = owner.bind_inventory(empty, layout, kv, 1, 1);
        assert_eq!(
            bound.capability_for(PureDecodeGraphV1CaptureOperation::Attention),
            GraphOperatorCapability::Supported
        );
        assert_eq!(
            bound.operator_capability(),
            GraphOperatorCapability::Unknown
        );
        assert_eq!(
            owner
                .bind_inventory(empty, bad_layout, kv, 1, 1)
                .capability_for(PureDecodeGraphV1CaptureOperation::Attention),
            GraphOperatorCapability::Unknown
        );
        assert_eq!(
            owner
                .bind_inventory(empty, layout, kv, 0, 1)
                .capability_for(PureDecodeGraphV1CaptureOperation::Attention),
            GraphOperatorCapability::Unknown
        );
        for _ in 0..32 {
            owner.replay()?;
        }
        let hashes = owner.parity_hashes();
        assert_eq!(hashes.0, <[u8; 32]>::from(Sha256::digest(source.bytes())));
        assert_eq!(hashes.1, <[u8; 32]>::from(Sha256::digest(&ones[..128])));
        eprintln!(
            "canonical metadata/output SHA256: {:02x?} {:02x?}",
            hashes.0, hashes.1
        );
        owner.close()?;
        assert_eq!(context.allocation_stats()?, stable);
        device.close()?;
        for buffer in [query, output, key, value] {
            buffer.close()?;
        }
        staging.close()?;
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }
}
