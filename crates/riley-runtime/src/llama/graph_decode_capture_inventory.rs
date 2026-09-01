//! Closed, cold capture-capability inventory for the C07 pure-decode chain.
//!
//! C06 accepts one aggregate [`GraphOperatorCapability`] fact, but that fact
//! must not be synthesized from an unreviewed forward path.  This C07-28
//! value records every canonical pure-decode operation separately and reduces
//! them fail-closed: every operation must be `Supported`; an `Unsupported`
//! operation wins over any `Unknown` operation; and all other combinations
//! remain `Unknown`.
//!
//! The inventory is only a value-level evidence boundary.  It opens no CUDA
//! capture, owns no runtime resource, changes no registry, and does not
//! connect C07's metadata H2D graph to production decode execution.  A later
//! operation-specific C05 vertical slice must establish native capture
//! lifecycle and parity before it supplies `Supported` here.

use super::graph::GraphOperatorCapability;

/// Number of canonical operations in the initial C07 pure-decode chain.
///
/// Keep this count and [`PureDecodeGraphV1CaptureOperation::ALL`] synchronized
/// whenever the reviewed chain changes.  A fixed array makes omission,
/// duplication, and dynamic inventory construction impossible at this layer.
pub(crate) const PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT: usize = 14;

/// One canonical operation that must be reviewed before full-graph capture.
///
/// The variants describe operation classes, not CUDA handles or execution
/// sites.  They intentionally remain separate even when a future execution
/// implementation happens to fuse several of them, because that change needs
/// a new capture-safety review.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1CaptureOperation {
    /// Fixed exact metadata transfer into the graph-owned device slab.
    MetadataH2d,
    /// Token embedding lookup and validation-status production.
    Embedding,
    /// Per-layer RMS normalization before projection work.
    Norm,
    /// Per-layer query, key, value, output, and MLP projection GEMMs.
    LayerProjectionGemm,
    /// Rotary position embedding application.
    Rope,
    /// Key/value cache writes for the decode iteration.
    KvWrite,
    /// Decode attention computation.
    Attention,
    /// C07 V1's out-of-place BF16 SiLU from gate projection to activated gate.
    MlpSiluBf16,
    /// Elementwise product of the activated gate and up-projection outputs.
    MlpGatedMultiply,
    /// Attention and MLP residual additions.
    Residual,
    /// Final output normalization.
    FinalNorm,
    /// Language-model head projection after final normalization.
    LmHead,
    /// GPU greedy sampling/output handling after language-model projection.
    GpuGreedy,
    /// The selected token/status transfer or completion dependency boundary.
    CompletionBoundary,
}

impl PureDecodeGraphV1CaptureOperation {
    /// Canonical stable operation order used by the fixed inventory.
    pub(crate) const ALL: [Self; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT] = [
        Self::MetadataH2d,
        Self::Embedding,
        Self::Norm,
        Self::LayerProjectionGemm,
        Self::Rope,
        Self::KvWrite,
        Self::Attention,
        Self::MlpSiluBf16,
        Self::MlpGatedMultiply,
        Self::Residual,
        Self::FinalNorm,
        Self::LmHead,
        Self::GpuGreedy,
        Self::CompletionBoundary,
    ];

    /// Returns this operation's fixed capability-array index.
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::MetadataH2d => 0,
            Self::Embedding => 1,
            Self::Norm => 2,
            Self::LayerProjectionGemm => 3,
            Self::Rope => 4,
            Self::KvWrite => 5,
            Self::Attention => 6,
            Self::MlpSiluBf16 => 7,
            Self::MlpGatedMultiply => 8,
            Self::Residual => 9,
            Self::FinalNorm => 10,
            Self::LmHead => 11,
            Self::GpuGreedy => 12,
            Self::CompletionBoundary => 13,
        }
    }

    /// Returns every canonical operation in its stable inventory order.
    #[must_use]
    pub(crate) const fn all() -> [Self; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT] {
        Self::ALL
    }
}

/// Immutable capability evidence for the complete C07 pure-decode chain.
///
/// Constructing this value does not claim that any operation has been reviewed.
/// The default is deliberately all-unknown, and the aggregate remains unknown
/// until a later cold owner provides explicit evidence for every slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PureDecodeGraphV1CaptureCapabilityInventory {
    capabilities: [GraphOperatorCapability; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT],
}

impl PureDecodeGraphV1CaptureCapabilityInventory {
    /// Creates the complete fixed inventory from canonical operation slots.
    #[must_use]
    pub(crate) const fn new(
        capabilities: [GraphOperatorCapability; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT],
    ) -> Self {
        Self { capabilities }
    }

    /// Returns the reviewed capability attached to one canonical operation.
    #[must_use]
    pub(crate) const fn capability_for(
        self,
        operation: PureDecodeGraphV1CaptureOperation,
    ) -> GraphOperatorCapability {
        self.capabilities[operation.index()]
    }

    /// Returns a copy whose one named operation has replacement evidence.
    ///
    /// This remains a cold immutable value transform.  It does not mutate an
    /// executor or convert a capability value into capture/replay permission.
    #[must_use]
    pub(crate) const fn with_capability(
        mut self,
        operation: PureDecodeGraphV1CaptureOperation,
        capability: GraphOperatorCapability,
    ) -> Self {
        self.capabilities[operation.index()] = capability;
        self
    }

    /// Reduces all operation evidence into C06's one fail-closed fact.
    ///
    /// An explicit unsupported operation is more informative than an unknown
    /// one, so it has precedence.  A future capability variant is neither
    /// equal to `Supported` nor to `Unsupported` and therefore remains safely
    /// unknown without changing this reducer.
    #[must_use]
    pub(crate) fn operator_capability(self) -> GraphOperatorCapability {
        if self
            .capabilities
            .iter()
            .any(|capability| *capability == GraphOperatorCapability::Unsupported)
        {
            return GraphOperatorCapability::Unsupported;
        }
        if self
            .capabilities
            .iter()
            .all(|capability| *capability == GraphOperatorCapability::Supported)
        {
            GraphOperatorCapability::Supported
        } else {
            GraphOperatorCapability::Unknown
        }
    }
}

impl Default for PureDecodeGraphV1CaptureCapabilityInventory {
    fn default() -> Self {
        Self::new([GraphOperatorCapability::Unknown; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT, PureDecodeGraphV1CaptureCapabilityInventory,
        PureDecodeGraphV1CaptureOperation,
    };
    use crate::llama::graph::GraphOperatorCapability;

    fn inventory(
        capability: GraphOperatorCapability,
    ) -> PureDecodeGraphV1CaptureCapabilityInventory {
        PureDecodeGraphV1CaptureCapabilityInventory::new(
            [capability; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT],
        )
    }

    #[test]
    fn canonical_operations_have_dense_stable_slots() {
        let operations = PureDecodeGraphV1CaptureOperation::all();
        let expected = [
            PureDecodeGraphV1CaptureOperation::MetadataH2d,
            PureDecodeGraphV1CaptureOperation::Embedding,
            PureDecodeGraphV1CaptureOperation::Norm,
            PureDecodeGraphV1CaptureOperation::LayerProjectionGemm,
            PureDecodeGraphV1CaptureOperation::Rope,
            PureDecodeGraphV1CaptureOperation::KvWrite,
            PureDecodeGraphV1CaptureOperation::Attention,
            PureDecodeGraphV1CaptureOperation::MlpSiluBf16,
            PureDecodeGraphV1CaptureOperation::MlpGatedMultiply,
            PureDecodeGraphV1CaptureOperation::Residual,
            PureDecodeGraphV1CaptureOperation::FinalNorm,
            PureDecodeGraphV1CaptureOperation::LmHead,
            PureDecodeGraphV1CaptureOperation::GpuGreedy,
            PureDecodeGraphV1CaptureOperation::CompletionBoundary,
        ];
        assert_eq!(operations, expected);
        assert_eq!(expected.len(), PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT);
        for (index, operation) in expected.into_iter().enumerate() {
            assert_eq!(operation.index(), index);
        }
    }

    #[test]
    fn default_inventory_leaves_every_operation_unknown() {
        let inventory = PureDecodeGraphV1CaptureCapabilityInventory::default();
        assert_eq!(
            inventory.operator_capability(),
            GraphOperatorCapability::Unknown
        );
        for operation in PureDecodeGraphV1CaptureOperation::all() {
            assert_eq!(
                inventory.capability_for(operation),
                GraphOperatorCapability::Unknown
            );
        }
    }

    #[test]
    fn only_complete_supported_evidence_is_admitted() {
        assert_eq!(
            inventory(GraphOperatorCapability::Supported).operator_capability(),
            GraphOperatorCapability::Supported
        );
    }

    #[test]
    fn each_unknown_operation_keeps_the_chain_unknown() {
        let all_supported = inventory(GraphOperatorCapability::Supported);
        for operation in PureDecodeGraphV1CaptureOperation::all() {
            let changed =
                all_supported.with_capability(operation, GraphOperatorCapability::Unknown);
            assert_eq!(
                changed.operator_capability(),
                GraphOperatorCapability::Unknown,
                "{operation:?} must not inherit capture approval from the other operations",
            );
            assert_eq!(
                all_supported.capability_for(operation),
                GraphOperatorCapability::Supported,
                "cold replacement must not mutate the original inventory",
            );
        }
    }

    #[test]
    fn each_unsupported_operation_rejects_the_chain() {
        let all_supported = inventory(GraphOperatorCapability::Supported);
        for operation in PureDecodeGraphV1CaptureOperation::all() {
            assert_eq!(
                all_supported
                    .with_capability(operation, GraphOperatorCapability::Unsupported)
                    .operator_capability(),
                GraphOperatorCapability::Unsupported,
                "{operation:?} must reject full-graph capture until it has a reviewed replacement",
            );
        }
    }

    #[test]
    fn unsupported_evidence_wins_over_unknown_evidence() {
        let inventory = inventory(GraphOperatorCapability::Supported)
            .with_capability(
                PureDecodeGraphV1CaptureOperation::Attention,
                GraphOperatorCapability::Unknown,
            )
            .with_capability(
                PureDecodeGraphV1CaptureOperation::KvWrite,
                GraphOperatorCapability::Unsupported,
            );
        assert_eq!(
            inventory.operator_capability(),
            GraphOperatorCapability::Unsupported
        );
    }
}
