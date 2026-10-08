//! C07 attention-graph binding to executor-owned parent KV allocations.
//!
//! C05-19 currently owns isolated whole K/V buffers by value.  The executor,
//! however, owns one larger K allocation and one larger V allocation and
//! addresses one layer through [`KvLayout::layer_byte_offset`].  This module
//! is the CPU-only contract that bridges those two ownership models without
//! exposing a device pointer or caller-provided byte span.  A future native
//! ABI owner must consume this exact binding before it can capture or replay
//! an attention graph.
//!
//! It deliberately does not allocate CUDA memory, call CUDA, update the C07
//! inventory, or admit graph replay.  In particular, a successfully created
//! binding is not C05-19 GPU parity evidence and must not make `Attention`
//! capture-capable.

use super::graph::GraphMetadataLayoutSignature;
use crate::paged_kv::{KV_BLOCK_SIZE, KvLayout};

/// Opaque cold identity of one executor-owned device allocation.
///
/// This identifier is an ownership token, not a device address.  The native
/// ABI must receive an opaque allocation owner/lease corresponding to this
/// identity; passing a raw pointer and a byte length is outside this contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AttentionGraphParentAllocationId(u64);

impl AttentionGraphParentAllocationId {
    /// Creates one non-zero cold allocation identity.
    #[must_use]
    pub(crate) const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }
}

/// Opaque CUDA-context identity selected during cold preparation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AttentionGraphContextId(u64);

impl AttentionGraphContextId {
    /// Creates one non-zero context identity.
    #[must_use]
    pub(crate) const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }
}

/// Opaque CUDA-stream identity selected during cold preparation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AttentionGraphStreamId(u64);

impl AttentionGraphStreamId {
    /// Creates one non-zero stream identity.
    #[must_use]
    pub(crate) const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }
}

/// Exact device/context/stream ownership required by one graph capture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AttentionGraphExecutionScope {
    device_ordinal: u32,
    context: AttentionGraphContextId,
    stream: AttentionGraphStreamId,
}

impl AttentionGraphExecutionScope {
    /// Creates a closed capture scope from cold executor ownership facts.
    #[must_use]
    pub(crate) const fn new(
        device_ordinal: u32,
        context: AttentionGraphContextId,
        stream: AttentionGraphStreamId,
    ) -> Self {
        Self {
            device_ordinal,
            context,
            stream,
        }
    }
}

/// One executor-owned allocation that may be borrowed by an attention graph.
///
/// `active_lease` is only a CPU lifecycle mirror.  The future CUDA wrapper
/// must hold the corresponding native allocation lease until graph launch
/// completion is known; it must never reconstruct this state from an address.
#[derive(Debug)]
pub(crate) struct AttentionGraphParentAllocation {
    id: AttentionGraphParentAllocationId,
    byte_len: u64,
    scope: AttentionGraphExecutionScope,
    closed: bool,
    active_lease: bool,
}

impl AttentionGraphParentAllocation {
    /// Registers one already-created executor allocation for graph binding.
    pub(crate) fn try_new(
        id: AttentionGraphParentAllocationId,
        byte_len: u64,
        scope: AttentionGraphExecutionScope,
    ) -> AttentionGraphParentBindingResult<Self> {
        if byte_len == 0 {
            return Err(AttentionGraphParentBindingError::ZeroParentAllocation { id });
        }
        Ok(Self {
            id,
            byte_len,
            scope,
            closed: false,
            active_lease: false,
        })
    }

    /// Explicitly retires the allocation after every graph lease was released.
    pub(crate) fn close(&mut self) -> AttentionGraphParentBindingResult<()> {
        if self.closed {
            return Err(AttentionGraphParentBindingError::ParentAlreadyClosed { id: self.id });
        }
        if self.active_lease {
            return Err(AttentionGraphParentBindingError::ParentLeaseActive { id: self.id });
        }
        self.closed = true;
        Ok(())
    }

    fn ensure_available(&self) -> AttentionGraphParentBindingResult<()> {
        if self.closed {
            return Err(AttentionGraphParentBindingError::ParentAlreadyClosed { id: self.id });
        }
        if self.active_lease {
            return Err(AttentionGraphParentBindingError::ParentLeaseActive { id: self.id });
        }
        Ok(())
    }
}

/// Cold metadata allocation and its canonical packed-layout identity.
#[derive(Debug)]
pub(crate) struct AttentionGraphMetadataAllocation {
    id: AttentionGraphParentAllocationId,
    scope: AttentionGraphExecutionScope,
    layout: GraphMetadataLayoutSignature,
    closed: bool,
    active_lease: bool,
}

impl AttentionGraphMetadataAllocation {
    /// Registers the existing packed metadata allocation used by the executor.
    #[must_use]
    pub(crate) const fn new(
        id: AttentionGraphParentAllocationId,
        scope: AttentionGraphExecutionScope,
        layout: GraphMetadataLayoutSignature,
    ) -> Self {
        Self {
            id,
            scope,
            layout,
            closed: false,
            active_lease: false,
        }
    }

    /// Explicitly retires metadata only after graph leases were released.
    pub(crate) fn close(&mut self) -> AttentionGraphParentBindingResult<()> {
        if self.closed {
            return Err(AttentionGraphParentBindingError::ParentAlreadyClosed { id: self.id });
        }
        if self.active_lease {
            return Err(AttentionGraphParentBindingError::ParentLeaseActive { id: self.id });
        }
        self.closed = true;
        Ok(())
    }

    fn ensure_available(&self) -> AttentionGraphParentBindingResult<()> {
        if self.closed {
            return Err(AttentionGraphParentBindingError::ParentAlreadyClosed { id: self.id });
        }
        if self.active_lease {
            return Err(AttentionGraphParentBindingError::ParentLeaseActive { id: self.id });
        }
        Ok(())
    }
}

/// Immutable one-layer subrange derived only from the checked executor layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AttentionGraphLayerSpan {
    parent: AttentionGraphParentAllocationId,
    layer_index: usize,
    byte_offset: u64,
    byte_len: u64,
}

impl AttentionGraphLayerSpan {
    /// Returns the parent allocation identity retained by the graph lease.
    #[must_use]
    pub(crate) const fn parent(self) -> AttentionGraphParentAllocationId {
        self.parent
    }

    /// Returns the cold-selected decoder layer.
    #[must_use]
    pub(crate) const fn layer_index(self) -> usize {
        self.layer_index
    }

    /// Returns the trusted layer offset derived from [`KvLayout`].
    #[must_use]
    pub(crate) const fn byte_offset(self) -> u64 {
        self.byte_offset
    }

    /// Returns the trusted exact layer length derived from [`KvLayout`].
    #[must_use]
    pub(crate) const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// Cold request for one fixed grouped paged-attention capture binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AttentionGraphCaptureRequest {
    expected_key_parent: AttentionGraphParentAllocationId,
    expected_value_parent: AttentionGraphParentAllocationId,
    expected_metadata_parent: AttentionGraphParentAllocationId,
    scope: AttentionGraphExecutionScope,
    layer_index: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_dimension: usize,
    metadata_layout: GraphMetadataLayoutSignature,
}

impl AttentionGraphCaptureRequest {
    /// Creates a request from executor-owned identities and immutable geometry.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) const fn new(
        expected_key_parent: AttentionGraphParentAllocationId,
        expected_value_parent: AttentionGraphParentAllocationId,
        expected_metadata_parent: AttentionGraphParentAllocationId,
        scope: AttentionGraphExecutionScope,
        layer_index: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_dimension: usize,
        metadata_layout: GraphMetadataLayoutSignature,
    ) -> Self {
        Self {
            expected_key_parent,
            expected_value_parent,
            expected_metadata_parent,
            scope,
            layer_index,
            query_head_count,
            key_value_head_count,
            head_dimension,
            metadata_layout,
        }
    }
}

/// Closed preflight rejection for a parent-span graph binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum AttentionGraphParentBindingError {
    /// A parent registration tried to describe a zero-byte allocation.
    ZeroParentAllocation {
        id: AttentionGraphParentAllocationId,
    },
    /// The supplied allocation identity does not match the cold request.
    WrongParent {
        role: &'static str,
        expected: AttentionGraphParentAllocationId,
        actual: AttentionGraphParentAllocationId,
    },
    /// K and V identities alias, making their immutable layer spans overlap.
    OverlappingKeyValueParents {
        id: AttentionGraphParentAllocationId,
    },
    /// A parent was closed before capture admission.
    ParentAlreadyClosed {
        id: AttentionGraphParentAllocationId,
    },
    /// A parent already has a graph lease and cannot be reused or closed.
    ParentLeaseActive {
        id: AttentionGraphParentAllocationId,
    },
    /// Device ordinal differs from the capture scope.
    WrongDevice {
        role: &'static str,
        expected: u32,
        actual: u32,
    },
    /// CUDA context identity differs from the capture scope.
    WrongContext {
        role: &'static str,
        expected: AttentionGraphContextId,
        actual: AttentionGraphContextId,
    },
    /// CUDA stream identity differs from the capture scope.
    WrongStream {
        role: &'static str,
        expected: AttentionGraphStreamId,
        actual: AttentionGraphStreamId,
    },
    /// The metadata allocation was prepared under a different canonical layout.
    MetadataLayoutMismatch,
    /// C05-19 only admits non-zero grouped-query geometry with QH divisible by KVH.
    InvalidHeadGeometry {
        query_head_count: usize,
        key_value_head_count: usize,
    },
    /// The currently reviewed graph path is fixed to D64 and page size 16.
    UnsupportedPagedGeometry {
        head_dimension: usize,
        block_size: usize,
    },
    /// The caller requested a layer outside the checked parent layout.
    LayerOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
    /// A checked layer span cannot be represented in the parent allocation.
    LayerSpanOutOfRange {
        role: &'static str,
        parent: AttentionGraphParentAllocationId,
        byte_offset: u64,
        byte_len: u64,
        parent_byte_len: u64,
    },
    /// Graph close was requested while a launch completion is still outstanding.
    GraphLaunchInFlight,
    /// A graph lease was released more than once.
    GraphLeaseAlreadyReleased,
}

/// Result of one CPU-only parent-span binding transition.
pub(crate) type AttentionGraphParentBindingResult<T> = Result<T, AttentionGraphParentBindingError>;

/// One graph lease retaining exact K/V layer spans and executor metadata.
///
/// The mutable parent borrows make parent close or reuse impossible in safe
/// Rust until this lease is released.  `launch` narrows that rule further: its
/// completion token must finish before graph close can release the parents.
#[must_use]
pub(crate) struct AttentionGraphParentBinding<'parent> {
    key_parent: &'parent mut AttentionGraphParentAllocation,
    value_parent: &'parent mut AttentionGraphParentAllocation,
    metadata_parent: &'parent mut AttentionGraphMetadataAllocation,
    key_span: AttentionGraphLayerSpan,
    value_span: AttentionGraphLayerSpan,
    released: bool,
    launch_in_flight: bool,
}

impl<'parent> AttentionGraphParentBinding<'parent> {
    /// Returns the exact K span permitted to the future native graph owner.
    #[must_use]
    pub(crate) const fn key_span(&self) -> AttentionGraphLayerSpan {
        self.key_span
    }

    /// Returns the exact V span permitted to the future native graph owner.
    #[must_use]
    pub(crate) const fn value_span(&self) -> AttentionGraphLayerSpan {
        self.value_span
    }

    /// Starts one modeled replay and retains the graph lease through completion.
    pub(crate) fn launch(
        &mut self,
    ) -> AttentionGraphParentBindingResult<AttentionGraphParentLaunch<'_, 'parent>> {
        if self.released {
            return Err(AttentionGraphParentBindingError::GraphLeaseAlreadyReleased);
        }
        if self.launch_in_flight {
            return Err(AttentionGraphParentBindingError::GraphLaunchInFlight);
        }
        self.launch_in_flight = true;
        Ok(AttentionGraphParentLaunch {
            binding: self,
            finished: false,
        })
    }

    /// Releases the graph lease after known graph completion.
    pub(crate) fn close(&mut self) -> AttentionGraphParentBindingResult<()> {
        if self.released {
            return Err(AttentionGraphParentBindingError::GraphLeaseAlreadyReleased);
        }
        if self.launch_in_flight {
            return Err(AttentionGraphParentBindingError::GraphLaunchInFlight);
        }
        self.release_once();
        Ok(())
    }

    fn release_once(&mut self) {
        if !self.released {
            self.key_parent.active_lease = false;
            self.value_parent.active_lease = false;
            self.metadata_parent.active_lease = false;
            self.released = true;
        }
    }
}

impl Drop for AttentionGraphParentBinding<'_> {
    fn drop(&mut self) {
        // Best effort only: an explicit close is the observation point.  This
        // CPU mirror never calls CUDA, but mirrors the eventual native owner's
        // requirement to retain every parent until it owns completion evidence.
        if !self.launch_in_flight {
            self.release_once();
        }
    }
}

/// In-flight completion guard for one modeled attention graph replay.
#[must_use]
pub(crate) struct AttentionGraphParentLaunch<'binding, 'parent> {
    binding: &'binding mut AttentionGraphParentBinding<'parent>,
    finished: bool,
}

impl AttentionGraphParentLaunch<'_, '_> {
    /// Marks graph completion known and permits a later graph close.
    pub(crate) fn finish(mut self) {
        self.binding.launch_in_flight = false;
        self.finished = true;
    }
}

impl Drop for AttentionGraphParentLaunch<'_, '_> {
    fn drop(&mut self) {
        // A forgotten completion must remain fail-closed: graph close and
        // parent release keep seeing an in-flight launch rather than assuming
        // that the device finished work.
        if !self.finished {
            self.binding.launch_in_flight = true;
        }
    }
}

/// Binds executor parents to one immutable C05-19 layer span.
///
/// No caller supplies a raw address, offset, or length.  The only admitted
/// span comes from the trusted `KvLayout`, so overflow, a different layer, and
/// a parent capacity mismatch are all rejected before native capture.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_attention_graph_parent_layer<'parent>(
    key_parent: &'parent mut AttentionGraphParentAllocation,
    value_parent: &'parent mut AttentionGraphParentAllocation,
    metadata_parent: &'parent mut AttentionGraphMetadataAllocation,
    layout: KvLayout,
    request: AttentionGraphCaptureRequest,
) -> AttentionGraphParentBindingResult<AttentionGraphParentBinding<'parent>> {
    validate_parent(
        "key",
        key_parent,
        request.expected_key_parent,
        request.scope,
    )?;
    validate_parent(
        "value",
        value_parent,
        request.expected_value_parent,
        request.scope,
    )?;
    validate_metadata_parent(metadata_parent, request)?;
    if key_parent.id == value_parent.id {
        return Err(
            AttentionGraphParentBindingError::OverlappingKeyValueParents { id: key_parent.id },
        );
    }
    if request.query_head_count == 0
        || request.key_value_head_count == 0
        || request.query_head_count % request.key_value_head_count != 0
        || request.key_value_head_count != layout.key_value_head_count()
    {
        return Err(AttentionGraphParentBindingError::InvalidHeadGeometry {
            query_head_count: request.query_head_count,
            key_value_head_count: request.key_value_head_count,
        });
    }
    if request.head_dimension != 64
        || layout.head_dimension() != 64
        || layout.block_size() != KV_BLOCK_SIZE
    {
        return Err(AttentionGraphParentBindingError::UnsupportedPagedGeometry {
            head_dimension: request.head_dimension,
            block_size: layout.block_size(),
        });
    }
    if request.layer_index >= layout.layer_count() {
        return Err(AttentionGraphParentBindingError::LayerOutOfRange {
            layer_index: request.layer_index,
            layer_count: layout.layer_count(),
        });
    }
    let byte_offset = layout.layer_byte_offset(request.layer_index).ok_or(
        AttentionGraphParentBindingError::LayerOutOfRange {
            layer_index: request.layer_index,
            layer_count: layout.layer_count(),
        },
    )?;
    let byte_len = layout.layer_stride_bytes();
    validate_span("key", key_parent, byte_offset, byte_len)?;
    validate_span("value", value_parent, byte_offset, byte_len)?;

    key_parent.active_lease = true;
    value_parent.active_lease = true;
    metadata_parent.active_lease = true;
    Ok(AttentionGraphParentBinding {
        key_span: AttentionGraphLayerSpan {
            parent: key_parent.id,
            layer_index: request.layer_index,
            byte_offset,
            byte_len,
        },
        value_span: AttentionGraphLayerSpan {
            parent: value_parent.id,
            layer_index: request.layer_index,
            byte_offset,
            byte_len,
        },
        key_parent,
        value_parent,
        metadata_parent,
        released: false,
        launch_in_flight: false,
    })
}

fn validate_parent(
    role: &'static str,
    parent: &AttentionGraphParentAllocation,
    expected_id: AttentionGraphParentAllocationId,
    expected_scope: AttentionGraphExecutionScope,
) -> AttentionGraphParentBindingResult<()> {
    if parent.id != expected_id {
        return Err(AttentionGraphParentBindingError::WrongParent {
            role,
            expected: expected_id,
            actual: parent.id,
        });
    }
    parent.ensure_available()?;
    validate_scope(role, parent.scope, expected_scope)
}

fn validate_metadata_parent(
    parent: &AttentionGraphMetadataAllocation,
    request: AttentionGraphCaptureRequest,
) -> AttentionGraphParentBindingResult<()> {
    if parent.id != request.expected_metadata_parent {
        return Err(AttentionGraphParentBindingError::WrongParent {
            role: "metadata",
            expected: request.expected_metadata_parent,
            actual: parent.id,
        });
    }
    parent.ensure_available()?;
    validate_scope("metadata", parent.scope, request.scope)?;
    if parent.layout != request.metadata_layout {
        return Err(AttentionGraphParentBindingError::MetadataLayoutMismatch);
    }
    Ok(())
}

fn validate_scope(
    role: &'static str,
    actual: AttentionGraphExecutionScope,
    expected: AttentionGraphExecutionScope,
) -> AttentionGraphParentBindingResult<()> {
    if actual.device_ordinal != expected.device_ordinal {
        return Err(AttentionGraphParentBindingError::WrongDevice {
            role,
            expected: expected.device_ordinal,
            actual: actual.device_ordinal,
        });
    }
    if actual.context != expected.context {
        return Err(AttentionGraphParentBindingError::WrongContext {
            role,
            expected: expected.context,
            actual: actual.context,
        });
    }
    if actual.stream != expected.stream {
        return Err(AttentionGraphParentBindingError::WrongStream {
            role,
            expected: expected.stream,
            actual: actual.stream,
        });
    }
    Ok(())
}

fn validate_span(
    role: &'static str,
    parent: &AttentionGraphParentAllocation,
    byte_offset: u64,
    byte_len: u64,
) -> AttentionGraphParentBindingResult<()> {
    let Some(end) = byte_offset.checked_add(byte_len) else {
        return Err(AttentionGraphParentBindingError::LayerSpanOutOfRange {
            role,
            parent: parent.id,
            byte_offset,
            byte_len,
            parent_byte_len: parent.byte_len,
        });
    };
    if end > parent.byte_len {
        return Err(AttentionGraphParentBindingError::LayerSpanOutOfRange {
            role,
            parent: parent.id,
            byte_offset,
            byte_len,
            parent_byte_len: parent.byte_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionGraphCaptureRequest, AttentionGraphContextId, AttentionGraphExecutionScope,
        AttentionGraphMetadataAllocation, AttentionGraphParentAllocation,
        AttentionGraphParentAllocationId, AttentionGraphParentBindingError, AttentionGraphStreamId,
        bind_attention_graph_parent_layer,
    };
    use crate::llama::graph::GraphMetadataLayoutSignature;
    use crate::paged_kv::KvLayout;

    const KEY: AttentionGraphParentAllocationId = match AttentionGraphParentAllocationId::new(1) {
        Some(value) => value,
        None => panic!("test identity must be non-zero"),
    };
    const VALUE: AttentionGraphParentAllocationId = match AttentionGraphParentAllocationId::new(2) {
        Some(value) => value,
        None => panic!("test identity must be non-zero"),
    };
    const METADATA: AttentionGraphParentAllocationId =
        match AttentionGraphParentAllocationId::new(3) {
            Some(value) => value,
            None => panic!("test identity must be non-zero"),
        };
    const CONTEXT: AttentionGraphContextId = match AttentionGraphContextId::new(11) {
        Some(value) => value,
        None => panic!("test identity must be non-zero"),
    };
    const OTHER_CONTEXT: AttentionGraphContextId = match AttentionGraphContextId::new(13) {
        Some(value) => value,
        None => panic!("test identity must be non-zero"),
    };
    const STREAM: AttentionGraphStreamId = match AttentionGraphStreamId::new(12) {
        Some(value) => value,
        None => panic!("test identity must be non-zero"),
    };
    const OTHER_STREAM: AttentionGraphStreamId = match AttentionGraphStreamId::new(14) {
        Some(value) => value,
        None => panic!("test identity must be non-zero"),
    };

    fn scope() -> AttentionGraphExecutionScope {
        AttentionGraphExecutionScope::new(0, CONTEXT, STREAM)
    }

    fn layout() -> KvLayout {
        KvLayout::checked(3, 8, 2, 64).expect("fixed C05-19 geometry")
    }

    fn metadata_layout() -> GraphMetadataLayoutSignature {
        GraphMetadataLayoutSignature::new(1, [7; 32])
    }

    fn request(layer_index: usize) -> AttentionGraphCaptureRequest {
        AttentionGraphCaptureRequest::new(
            KEY,
            VALUE,
            METADATA,
            scope(),
            layer_index,
            8,
            2,
            64,
            metadata_layout(),
        )
    }

    fn parents() -> (
        AttentionGraphParentAllocation,
        AttentionGraphParentAllocation,
        AttentionGraphMetadataAllocation,
    ) {
        let layout = layout();
        (
            AttentionGraphParentAllocation::try_new(KEY, layout.bytes_per_kind(), scope())
                .expect("K parent"),
            AttentionGraphParentAllocation::try_new(VALUE, layout.bytes_per_kind(), scope())
                .expect("V parent"),
            AttentionGraphMetadataAllocation::new(METADATA, scope(), metadata_layout()),
        )
    }

    #[test]
    fn exact_binding_derives_one_immutable_layer_span_from_the_parent_layout() {
        let (mut key, mut value, mut metadata) = parents();
        let layout = layout();
        let binding = bind_attention_graph_parent_layer(
            &mut key,
            &mut value,
            &mut metadata,
            layout,
            request(2),
        )
        .expect("exact layer must bind");

        let expected_offset = layout.layer_byte_offset(2).expect("valid layer");
        assert_eq!(binding.key_span().parent(), KEY);
        assert_eq!(binding.value_span().parent(), VALUE);
        assert_eq!(binding.key_span().layer_index(), 2);
        assert_eq!(binding.key_span().byte_offset(), expected_offset);
        assert_eq!(binding.value_span().byte_offset(), expected_offset);
        assert_eq!(binding.key_span().byte_len(), layout.layer_stride_bytes());
    }

    #[test]
    fn wrong_parent_and_aliasing_key_value_parent_reject_before_lease() {
        let (mut key, mut value, mut metadata) = parents();
        let wrong_key = AttentionGraphParentAllocationId::new(9).expect("non-zero");
        let wrong_request = AttentionGraphCaptureRequest::new(
            wrong_key,
            VALUE,
            METADATA,
            scope(),
            0,
            8,
            2,
            64,
            metadata_layout(),
        );
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                wrong_request,
            ),
            Err(AttentionGraphParentBindingError::WrongParent { role: "key", .. })
        ));

        let mut alias_key =
            AttentionGraphParentAllocation::try_new(KEY, layout().bytes_per_kind(), scope())
                .expect("K parent");
        let mut alias_value =
            AttentionGraphParentAllocation::try_new(KEY, layout().bytes_per_kind(), scope())
                .expect("aliased V parent model");
        let alias_request = AttentionGraphCaptureRequest::new(
            KEY,
            KEY,
            METADATA,
            scope(),
            0,
            8,
            2,
            64,
            metadata_layout(),
        );
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut alias_key,
                &mut alias_value,
                &mut metadata,
                layout(),
                alias_request,
            ),
            Err(AttentionGraphParentBindingError::OverlappingKeyValueParents { id: KEY })
        ));
    }

    #[test]
    fn parent_capacity_and_layer_bounds_reject_before_lease() {
        let (mut key, mut value, mut metadata) = parents();
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                request(3),
            ),
            Err(AttentionGraphParentBindingError::LayerOutOfRange {
                layer_index: 3,
                layer_count: 3,
            })
        ));

        let mut short_key =
            AttentionGraphParentAllocation::try_new(KEY, layout().bytes_per_kind() - 1, scope())
                .expect("non-zero short parent");
        let (mut replacement_key, mut replacement_value, mut replacement_metadata) = parents();
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut short_key,
                &mut replacement_value,
                &mut replacement_metadata,
                layout(),
                request(2),
            ),
            Err(AttentionGraphParentBindingError::LayerSpanOutOfRange { role: "key", .. })
        ));
        replacement_key.close().expect("unused parent closes");
    }

    #[test]
    fn scope_and_metadata_mismatch_reject_before_lease() {
        let (mut key, mut value, mut metadata) = parents();
        let wrong_scope = AttentionGraphExecutionScope::new(1, CONTEXT, STREAM);
        let wrong_device = AttentionGraphCaptureRequest::new(
            KEY,
            VALUE,
            METADATA,
            wrong_scope,
            0,
            8,
            2,
            64,
            metadata_layout(),
        );
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                wrong_device,
            ),
            Err(AttentionGraphParentBindingError::WrongDevice { role: "key", .. })
        ));

        let wrong_context = AttentionGraphCaptureRequest::new(
            KEY,
            VALUE,
            METADATA,
            AttentionGraphExecutionScope::new(0, OTHER_CONTEXT, STREAM),
            0,
            8,
            2,
            64,
            metadata_layout(),
        );
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                wrong_context,
            ),
            Err(AttentionGraphParentBindingError::WrongContext { role: "key", .. })
        ));

        let wrong_stream = AttentionGraphCaptureRequest::new(
            KEY,
            VALUE,
            METADATA,
            AttentionGraphExecutionScope::new(0, CONTEXT, OTHER_STREAM),
            0,
            8,
            2,
            64,
            metadata_layout(),
        );
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                wrong_stream,
            ),
            Err(AttentionGraphParentBindingError::WrongStream { role: "key", .. })
        ));

        let bad_metadata_layout = GraphMetadataLayoutSignature::new(2, [8; 32]);
        let mismatch = AttentionGraphCaptureRequest::new(
            KEY,
            VALUE,
            METADATA,
            scope(),
            0,
            8,
            2,
            64,
            bad_metadata_layout,
        );
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                mismatch,
            ),
            Err(AttentionGraphParentBindingError::MetadataLayoutMismatch)
        ));
    }

    #[test]
    fn closed_parent_is_rejected_and_completion_precedes_graph_release() {
        let (mut key, mut value, mut metadata) = parents();
        key.close().expect("idle parent closes");
        assert!(matches!(
            bind_attention_graph_parent_layer(
                &mut key,
                &mut value,
                &mut metadata,
                layout(),
                request(0),
            ),
            Err(AttentionGraphParentBindingError::ParentAlreadyClosed { id: KEY })
        ));

        let (mut key, mut value, mut metadata) = parents();
        let mut binding = bind_attention_graph_parent_layer(
            &mut key,
            &mut value,
            &mut metadata,
            layout(),
            request(0),
        )
        .expect("binding");
        drop(binding.launch().expect("one launch"));
        assert_eq!(
            binding.close(),
            Err(AttentionGraphParentBindingError::GraphLaunchInFlight)
        );
        drop(binding);
        assert_eq!(
            key.close(),
            Err(AttentionGraphParentBindingError::ParentLeaseActive { id: KEY })
        );

        let (mut key, mut value, mut metadata) = parents();
        let mut binding = bind_attention_graph_parent_layer(
            &mut key,
            &mut value,
            &mut metadata,
            layout(),
            request(0),
        )
        .expect("binding");
        binding.launch().expect("one launch").finish();
        binding.close().expect("completion permits graph close");
        assert_eq!(
            binding.close(),
            Err(AttentionGraphParentBindingError::GraphLeaseAlreadyReleased)
        );
        drop(binding);
        key.close().expect("released K parent closes");
        value.close().expect("released V parent closes");
        metadata.close().expect("released metadata parent closes");
    }
}
