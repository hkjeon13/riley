//! Parent KV layer capture with exclusive Rust borrows and native graph leases.

#[cfg(feature = "cuda")]
use super::{
    CudaGraphCaptureMode, begin_deferred_capture_contexts, finish_deferred_capture_contexts,
    validate_graph_grouped_ragged_paged_attention_bf16_capture_preflight,
};
use crate::{CudaDeviceBuffer, CudaError, CudaResult, CudaStream, PackedBatchHostV1};

/// Fixed D64/page16 BF16 parent geometry. Byte spans are derived, never supplied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionParentLayer {
    layers: u64,
    layer: u64,
    blocks: u64,
    kv_heads: u64,
    stride: u64,
    total: u64,
}

impl AttentionParentLayer {
    /// Checks layer bounds and every multiplication before any CUDA operation.
    ///
    /// # Errors
    /// Rejects zero dimensions, out-of-range layers and arithmetic overflow.
    pub fn new(layers: u64, layer: u64, blocks: u64, kv_heads: u64) -> CudaResult<Self> {
        const OP: &str = "AttentionParentLayer::new";
        let stride = blocks
            .checked_mul(kv_heads)
            .and_then(|n| n.checked_mul(16 * 64 * 2));
        let total = stride.and_then(|n| n.checked_mul(layers));
        match (stride, total) {
            (Some(stride), Some(total)) if stride > 0 && total > 0 && layer < layers => Ok(Self {
                layers,
                layer,
                blocks,
                kv_heads,
                stride,
                total,
            }),
            _ => Err(CudaError::out_of_range(
                OP,
                "invalid or overflowing parent layer geometry",
            )),
        }
    }

    /// Exact immutable layer byte offset within both parent allocations.
    #[must_use]
    pub const fn byte_offset(self) -> u64 {
        self.layer * self.stride
    }
    /// Exact layer byte length.
    #[must_use]
    pub const fn byte_len(self) -> u64 {
        self.stride
    }
    /// Exact complete parent byte length.
    #[must_use]
    pub const fn parent_byte_len(self) -> u64 {
        self.total
    }
}

/// Exclusive borrow of the actual stream, KV parents and fixed metadata.
/// Native leases additionally prevent reuse if a safe caller forgets the graph.
pub struct ParentAttentionResources<'a> {
    /// Stream used for capture and every replay.
    pub stream: &'a mut CudaStream,
    /// Fixed query allocation.
    pub query: &'a mut CudaDeviceBuffer,
    /// Complete multi-layer K parent.
    pub key_parent: &'a mut CudaDeviceBuffer,
    /// Complete multi-layer V parent.
    pub value_parent: &'a mut CudaDeviceBuffer,
    /// Fixed output allocation.
    pub output: &'a mut CudaDeviceBuffer,
    /// Packed CSR offsets allocation.
    pub sequence_block_offsets: &'a mut CudaDeviceBuffer,
    /// Packed physical block IDs allocation.
    pub block_ids: &'a mut CudaDeviceBuffer,
    /// Packed valid-token counts allocation.
    pub valid_tokens: &'a mut CudaDeviceBuffer,
    /// Packed row sequence slots allocation.
    pub row_sequence_slots: &'a mut CudaDeviceBuffer,
    /// Packed row positions allocation.
    pub row_positions: &'a mut CudaDeviceBuffer,
}

/// One immutable attention layer graph. This does not admit full decode graphs.
/// All buffers remain inaccessible until explicit close or successful Drop.
pub struct ParentAttentionGraph<'a> {
    // Native drops before the borrowed resources are released.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: RetainedAttentionResources<'a>,
    terminal: bool,
}

#[cfg(feature = "cuda")]
struct CaptureGuard {
    native: crate::ffi::GraphCaptureHandle,
    active: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active {
            let transition = self.native.abort_with_transition();
            if transition.resource_release_known {
                finish_deferred_capture_contexts();
            }
        }
    }
}

impl<'a> ParentAttentionGraph<'a> {
    /// Captures one node and instantiates it using exact opaque parent owners.
    /// The host batch is an admission witness only; device metadata must already
    /// contain the corresponding bytes. No graph capability is promoted here.
    ///
    /// # Errors
    /// Rejects mismatched geometry, context, capacity or active native leases.
    /// CUDA capture or instantiation errors retain uncertain leases fail-closed.
    pub fn prepare(
        resources: ParentAttentionResources<'a>,
        layer: AttentionParentLayer,
        batch: PackedBatchHostV1<'_>,
        query_heads: u64,
        output_rows: u64,
        scale: f32,
    ) -> CudaResult<Self> {
        const OP: &str = "ParentAttentionGraph::prepare";
        if layer.blocks != batch.physical_block_count()
            || resources.key_parent.byte_len() != layer.total
            || resources.value_parent.byte_len() != layer.total
        {
            return Err(CudaError::out_of_range(
                OP,
                "parent allocation or metadata geometry mismatch",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            validate_graph_grouped_ragged_paged_attention_bf16_capture_preflight(
                resources.stream,
                resources.query,
                resources.key_parent,
                resources.value_parent,
                resources.output,
                resources.sequence_block_offsets,
                resources.block_ids,
                resources.valid_tokens,
                resources.row_sequence_slots,
                resources.row_positions,
                batch,
                query_heads,
                layer.kv_heads,
                output_rows,
                scale,
                OP,
            )?;
            let native = resources
                .stream
                .native
                .begin_graph_parent_layer_attention_bf16_capture(
                    resources.query.native_handle(),
                    resources.key_parent.native_handle(),
                    resources.value_parent.native_handle(),
                    resources.output.native_handle(),
                    resources.sequence_block_offsets.native_handle(),
                    resources.block_ids.native_handle(),
                    resources.valid_tokens.native_handle(),
                    resources.row_sequence_slots.native_handle(),
                    resources.row_positions.native_handle(),
                    batch.sequence_count(),
                    batch.block_count(),
                    batch.active_row_count(),
                    batch.physical_block_count(),
                    query_heads,
                    layer.kv_heads,
                    output_rows,
                    scale,
                    CudaGraphCaptureMode::ThreadLocal as u32,
                    layer.layers,
                    layer.layer,
                )?;
            let native = instantiate_attention_capture(native)?;
            Ok(Self {
                native,
                resources: RetainedAttentionResources::Separate(resources),
                terminal: false,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (resources, query_heads, output_rows, scale);
            Err(CudaError::unavailable(OP))
        }
    }

    /// Replays and waits for known completion. Any error permanently blocks replay.
    ///
    /// # Errors
    /// Rejects terminal graphs and propagates launch or completion failures.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "ParentAttentionGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            let mut launch = self.native.launch(&mut self.resources.stream().native)?;
            launch.complete()?;
            self.terminal = false;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(OP))
        }
    }

    /// Releases the native graph leases before returning parent access to caller.
    ///
    /// # Errors
    /// Rejects terminal graphs and propagates native destruction failures.
    pub fn close(mut self) -> CudaResult<()> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "ParentAttentionGraph::close",
                "graph is terminal",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self;
            Err(CudaError::unavailable("ParentAttentionGraph::close"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AttentionParentLayer;

    #[test]
    fn parent_layer_geometry_rejects_zero_bounds_and_overflow() {
        for args in [
            (0, 0, 1, 1),
            (2, 2, 1, 1),
            (2, 0, 0, 1),
            (2, 0, 1, 0),
            (u64::MAX, 0, 1, 1),
            (2, 0, u64::MAX, 1),
            (2, 0, 1, u64::MAX),
        ] {
            assert!(AttentionParentLayer::new(args.0, args.1, args.2, args.3).is_err());
        }
        let layer = AttentionParentLayer::new(3, 2, 6, 3).unwrap();
        assert_eq!(layer.byte_len(), 6 * 3 * 16 * 64 * 2);
        assert_eq!(layer.byte_offset(), 2 * layer.byte_len());
        assert_eq!(layer.parent_byte_len(), 3 * layer.byte_len());
    }
}

#[cfg(feature = "cuda")]
fn instantiate_attention_capture(
    native: crate::ffi::GraphCaptureHandle,
) -> CudaResult<crate::ffi::GraphExecHandle> {
    begin_deferred_capture_contexts();
    let mut capture = CaptureGuard {
        native,
        active: true,
    };
    capture
        .native
        .enqueue_grouped_ragged_paged_attention_bf16()?;
    let transition = capture.native.end();
    capture.active = !transition.owner_consumed;
    if transition.resource_release_known {
        finish_deferred_capture_contexts();
    }
    let mut graph = transition.result?;
    graph.instantiate()
}

/// Fixed field regions in one metadata parent: CSR offsets, block IDs,
/// valid tokens, row slots, positions. Field order need not match byte order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedAttentionMetadataLayout {
    bytes: u64,
    regions: [(u64, u64); 5],
}

impl PackedAttentionMetadataLayout {
    /// Validates full field capacities, alignment, overflow and non-overlap.
    ///
    /// # Errors
    /// Rejects empty, misaligned, overlapping or out-of-bounds regions.
    pub fn new(bytes: u64, regions: [(u64, u64); 5]) -> CudaResult<Self> {
        let widths = [4, 4, 2, 4, 4];
        for (i, &(offset, len)) in regions.iter().enumerate() {
            if len == 0
                || offset % widths[i] != 0
                || len % widths[i] != 0
                || offset.checked_add(len).is_none_or(|end| end > bytes)
            {
                return Err(CudaError::out_of_range(
                    "PackedAttentionMetadataLayout::new",
                    "invalid metadata field range",
                ));
            }
            for &(other, other_len) in &regions[..i] {
                if offset < other + other_len && other < offset + len {
                    return Err(CudaError::invalid_argument(
                        "PackedAttentionMetadataLayout::new",
                        "overlapping metadata fields",
                    ));
                }
            }
        }
        Ok(Self { bytes, regions })
    }

    fn validate_payload(self, bytes: &[u8], batch: PackedBatchHostV1<'_>) -> CudaResult<()> {
        const OP: &str = "PackedAttentionMetadataLayout::validate_payload";
        if u64::try_from(bytes.len()).ok() != Some(self.bytes) {
            return Err(CudaError::out_of_range(
                OP,
                "payload does not match exact slab size",
            ));
        }
        let u32_fields = [
            (0, batch.sequence_block_offsets()),
            (1, batch.block_ids()),
            (3, batch.row_sequence_slots()),
            (4, batch.row_positions()),
        ];
        for (i, words) in u32_fields {
            self.validate_field(bytes, i, words.iter().flat_map(|word| word.to_le_bytes()))?;
        }
        self.validate_field(
            bytes,
            2,
            batch
                .valid_tokens()
                .iter()
                .flat_map(|word| word.to_le_bytes()),
        )
    }

    fn validate_field(
        self,
        bytes: &[u8],
        field: usize,
        expected: impl Iterator<Item = u8>,
    ) -> CudaResult<()> {
        let (offset, len) = self.regions[field];
        let start = usize::try_from(offset)
            .map_err(|_| CudaError::out_of_range("packed metadata", "offset is not addressable"))?;
        for (index, byte) in expected.enumerate() {
            if index as u64 >= len || bytes.get(start + index) != Some(&byte) {
                return Err(CudaError::invalid_argument(
                    "packed metadata",
                    "payload differs from validated batch witness",
                ));
            }
        }
        Ok(())
    }
}

/// Actual parent allocations borrowed for a packed-metadata attention graph.
pub struct PackedParentAttentionResources<'a> {
    /// Capture/replay stream.
    pub stream: &'a mut CudaStream,
    /// Fixed BF16 query.
    pub query: &'a mut CudaDeviceBuffer,
    /// Complete K parent.
    pub key_parent: &'a mut CudaDeviceBuffer,
    /// Complete V parent.
    pub value_parent: &'a mut CudaDeviceBuffer,
    /// Fixed BF16 output.
    pub output: &'a mut CudaDeviceBuffer,
    /// One metadata parent, exclusively leased once by native capture.
    pub metadata_slab: &'a mut CudaDeviceBuffer,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
enum RetainedAttentionResources<'a> {
    Separate(ParentAttentionResources<'a>),
    Packed(PackedParentAttentionResources<'a>),
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl RetainedAttentionResources<'_> {
    fn stream(&mut self) -> &mut CudaStream {
        match self {
            Self::Separate(r) => r.stream,
            Self::Packed(r) => r.stream,
        }
    }
}

impl<'a> ParentAttentionGraph<'a> {
    /// Stages the exact validated payload synchronously, then captures from the
    /// same exclusively borrowed slab. No caller-declared freshness is trusted.
    ///
    /// # Errors
    /// Rejects layout/content/capacity/context mismatches and propagates CUDA
    /// upload, capture or instantiate failures without returning graph evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_packed(
        resources: PackedParentAttentionResources<'a>,
        layer: AttentionParentLayer,
        metadata: PackedAttentionMetadataLayout,
        payload: &[u8],
        staging: &mut crate::CudaPinnedHostBuffer,
        batch: PackedBatchHostV1<'_>,
        query_heads: u64,
        output_rows: u64,
        scale: f32,
    ) -> CudaResult<Self> {
        const OP: &str = "ParentAttentionGraph::prepare_packed";
        metadata.validate_payload(payload, batch)?;
        let query_bytes = batch
            .active_row_count()
            .checked_mul(query_heads)
            .and_then(|n| n.checked_mul(128));
        let output_bytes = output_rows
            .checked_mul(query_heads)
            .and_then(|n| n.checked_mul(128));
        if layer.blocks != batch.physical_block_count()
            || query_heads == 0
            || query_heads % layer.kv_heads != 0
            || output_rows < batch.active_row_count()
            || !scale.is_finite()
            || scale <= 0.0
            || resources.key_parent.byte_len() != layer.total
            || resources.value_parent.byte_len() != layer.total
            || resources.metadata_slab.byte_len() != metadata.bytes
            || query_bytes.is_none_or(|n| n > resources.query.byte_len())
            || output_bytes.is_none_or(|n| n > resources.output.byte_len())
        {
            return Err(CudaError::out_of_range(
                OP,
                "packed attention geometry or allocation mismatch",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let buffers = [
                &*resources.query,
                &*resources.key_parent,
                &*resources.value_parent,
                &*resources.output,
                &*resources.metadata_slab,
            ];
            for (i, buffer) in buffers.iter().enumerate() {
                super::ensure_same_context(&resources.stream.context, buffer.context_owner(), OP)?;
                buffer.ensure_idle_for_operation(OP)?;
                for other in &buffers[..i] {
                    if buffer
                        .native_handle()
                        .same_allocation(other.native_handle())
                    {
                        return Err(CudaError::invalid_argument(
                            OP,
                            "parent allocations must be distinct",
                        ));
                    }
                }
            }
            resources
                .metadata_slab
                .upload_from_slice(0, payload, staging, resources.stream)?;
            let offsets = metadata.regions.map(|region| region.0);
            let native = resources
                .stream
                .native
                .begin_graph_packed_parent_attention_bf16_capture(
                    resources.query.native_handle(),
                    resources.key_parent.native_handle(),
                    resources.value_parent.native_handle(),
                    resources.output.native_handle(),
                    resources.metadata_slab.native_handle(),
                    &offsets,
                    batch.sequence_count(),
                    batch.block_count(),
                    batch.active_row_count(),
                    batch.physical_block_count(),
                    query_heads,
                    layer.kv_heads,
                    output_rows,
                    scale,
                    CudaGraphCaptureMode::ThreadLocal as u32,
                    layer.layers,
                    layer.layer,
                )?;
            let native = instantiate_attention_capture(native)?;
            Ok(Self {
                native,
                resources: RetainedAttentionResources::Packed(resources),
                terminal: false,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (resources, staging);
            Err(CudaError::unavailable(OP))
        }
    }
}

#[cfg(test)]
mod packed_tests {
    use super::{PackedAttentionMetadataLayout, PackedBatchHostV1};

    #[test]
    fn packed_fields_reject_overlap_alignment_overflow_and_truncation() {
        let valid = [(16, 8), (32, 4), (40, 2), (8, 4), (0, 4)];
        assert!(PackedAttentionMetadataLayout::new(64, valid).is_ok());
        for invalid in [
            (0, (18, 8)),
            (1, (16, 4)),
            (2, (63, 2)),
            (3, (u64::MAX - 3, 8)),
            (4, (0, 0)),
        ] {
            let mut regions = valid;
            regions[invalid.0] = invalid.1;
            assert!(PackedAttentionMetadataLayout::new(64, regions).is_err());
        }
    }

    #[test]
    fn packed_payload_must_match_batch_before_upload() {
        let batch = PackedBatchHostV1::new(&[0, 1], &[0], &[1], &[0], &[0], 1).unwrap();
        let layout =
            PackedAttentionMetadataLayout::new(64, [(16, 8), (32, 4), (40, 2), (8, 4), (0, 4)])
                .unwrap();
        let mut payload = [0_u8; 64];
        payload[20..24].copy_from_slice(&1_u32.to_le_bytes());
        payload[40..42].copy_from_slice(&1_u16.to_le_bytes());
        assert!(layout.validate_payload(&payload, batch).is_ok());
        payload[40] = 2;
        assert!(layout.validate_payload(&payload, batch).is_err());
        assert!(layout.validate_payload(&payload[..63], batch).is_err());
        let short =
            PackedAttentionMetadataLayout::new(64, [(16, 4), (32, 4), (40, 2), (8, 4), (0, 4)])
                .unwrap();
        assert!(short.validate_payload(&payload, batch).is_err());
    }
}

/// Exclusive actual sources, complete KV parents, metadata parent and stream.
pub struct PackedParentKvWriteResources<'a> {
    /// Capture/replay stream.
    pub stream: &'a mut CudaStream,
    /// Dense rotated BF16 key source.
    pub key_source: &'a mut CudaDeviceBuffer,
    /// Dense BF16 value source.
    pub value_source: &'a mut CudaDeviceBuffer,
    /// Complete key cache allocation.
    pub key_parent: &'a mut CudaDeviceBuffer,
    /// Complete value cache allocation.
    pub value_parent: &'a mut CudaDeviceBuffer,
    /// Actual packed metadata allocation; no upload or replacement occurs.
    pub metadata_slab: &'a mut CudaDeviceBuffer,
}

/// One D64/page16 KV write retaining whole parents through known completion.
pub struct PackedParentKvWriteGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: PackedParentKvWriteResources<'a>,
    terminal: bool,
}

impl<'a> PackedParentKvWriteGraph<'a> {
    /// Validates actual device metadata against the batch before capturing.
    ///
    /// # Errors
    /// Rejects foreign/busy owners, mismatched metadata or invalid geometry.
    pub fn prepare(
        resources: PackedParentKvWriteResources<'a>,
        layer: AttentionParentLayer,
        metadata: PackedAttentionMetadataLayout,
        batch: PackedBatchHostV1<'_>,
        staging: &mut crate::CudaPinnedHostBuffer,
    ) -> CudaResult<Self> {
        const OP: &str = "PackedParentKvWriteGraph::prepare";
        let source_bytes = batch
            .active_row_count()
            .checked_mul(layer.kv_heads)
            .and_then(|n| n.checked_mul(128));
        if layer.blocks != batch.physical_block_count()
            || resources.key_parent.byte_len() != layer.total
            || resources.value_parent.byte_len() != layer.total
            || resources.metadata_slab.byte_len() != metadata.bytes
            || source_bytes.is_none_or(|n| {
                n == 0
                    || n > resources.key_source.byte_len()
                    || n > resources.value_source.byte_len()
            })
        {
            return Err(CudaError::out_of_range(
                OP,
                "KV write parent geometry or source capacity mismatch",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let buffers = [
                &*resources.key_source,
                &*resources.value_source,
                &*resources.key_parent,
                &*resources.value_parent,
                &*resources.metadata_slab,
            ];
            for (i, buffer) in buffers.iter().enumerate() {
                super::ensure_same_context(&resources.stream.context, buffer.context_owner(), OP)?;
                buffer.ensure_idle_for_operation(OP)?;
                for other in &buffers[..i] {
                    if buffer
                        .native_handle()
                        .same_allocation(other.native_handle())
                    {
                        return Err(CudaError::invalid_argument(
                            OP,
                            "KV write parents and sources must be distinct",
                        ));
                    }
                }
            }
            let mut actual = vec![
                0;
                usize::try_from(metadata.bytes).map_err(|_| {
                    CudaError::out_of_range(OP, "metadata exceeds host address space")
                })?
            ];
            resources
                .metadata_slab
                .download_to_slice(0, &mut actual, staging, resources.stream)?;
            metadata.validate_payload(&actual, batch)?;
            let offsets = metadata.regions.map(|region| region.0);
            let native = resources
                .stream
                .native
                .begin_graph_packed_parent_kv_write_bf16_capture(
                    resources.key_source.native_handle(),
                    resources.value_source.native_handle(),
                    resources.key_parent.native_handle(),
                    resources.value_parent.native_handle(),
                    resources.metadata_slab.native_handle(),
                    &offsets,
                    batch.sequence_count(),
                    batch.block_count(),
                    batch.active_row_count(),
                    batch.physical_block_count(),
                    layer.kv_heads,
                    CudaGraphCaptureMode::ThreadLocal as u32,
                    layer.layers,
                    layer.layer,
                )?;
            begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_ragged_paged_kv_cache_write_bf16()?;
            let transition = capture.native.end();
            capture.active = !transition.owner_consumed;
            if transition.resource_release_known {
                finish_deferred_capture_contexts();
            }
            let mut graph = transition.result?;
            let native = graph.instantiate()?;
            Ok(Self {
                native,
                resources,
                terminal: false,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (resources, staging);
            Err(CudaError::unavailable(OP))
        }
    }

    /// Replays once and waits for completion before releasing any parent.
    ///
    /// # Errors
    /// CUDA failure makes this owner terminal.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "PackedParentKvWriteGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            self.native
                .launch(&mut self.resources.stream.native)?
                .complete()?;
            self.terminal = false;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        Err(CudaError::unavailable(OP))
    }

    /// Destroys the graph before returning access to borrowed parents.
    ///
    /// # Errors
    /// Rejects terminal state and propagates destruction failure.
    pub fn close(mut self) -> CudaResult<()> {
        const OP: &str = "PackedParentKvWriteGraph::close";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self;
            Err(CudaError::unavailable(OP))
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod kv_native_tests {
    #[test]
    #[ignore = "requires CUDA; native parent/metadata preflight"]
    fn native_packed_kv_write_rejects_parent_and_metadata_substitution()
    -> Result<(), Box<dyn std::error::Error>> {
        let context = crate::CudaRuntime::initialize()?
            .device(0)?
            .create_context()?;
        let mut stream = context.create_stream()?;
        let k = context.allocate_device_buffer(128)?;
        let v = context.allocate_device_buffer(128)?;
        let kp = context.allocate_device_buffer(8192)?;
        let vp = context.allocate_device_buffer(8192)?;
        let meta = context.allocate_device_buffer(64)?;
        let stats = context.allocation_stats()?;
        for (layers, layer, offsets) in [
            (2, 2, [0, 8, 12, 16, 20]),
            (1, 0, [0, 8, 12, 16, 20]),
            (0, 0, [0, 8, 12, 16, 20]),
            (2, 0, [0, 0, 12, 16, 20]),
            (2, 0, [1, 8, 12, 16, 20]),
            (2, 0, [0, 8, 12, 16, u64::MAX - 3]),
        ] {
            assert!(
                stream
                    .native
                    .begin_graph_packed_parent_kv_write_bf16_capture(
                        k.native_handle(),
                        v.native_handle(),
                        kp.native_handle(),
                        vp.native_handle(),
                        meta.native_handle(),
                        &offsets,
                        1,
                        1,
                        1,
                        2,
                        1,
                        super::CudaGraphCaptureMode::ThreadLocal as u32,
                        layers,
                        layer
                    )
                    .is_err()
            );
            assert_eq!(context.allocation_stats()?, stats);
        }
        k.close()?;
        v.close()?;
        kp.close()?;
        vp.close()?;
        meta.close()?;
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }
}
