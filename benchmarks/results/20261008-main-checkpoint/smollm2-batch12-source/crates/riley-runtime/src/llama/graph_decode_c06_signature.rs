//! Checked C07 V1 composition into one complete C06 graph-cache identity.
//!
//! C07-20 consumes only C07-19's opaque partial identity plus an independently
//! cold-prepared C06 static identity. It asks C06 to compare their full
//! metadata-layout identities before it assembles a complete cache-key value.
//! It never rewrites the static identity or performs registry, dispatch,
//! capture, allocation, or execution work.

use super::graph::{
    GraphSignature, GraphStaticMetadataLayoutMismatch, GraphStaticSignature,
    compose_graph_signature_checked_metadata,
};
use super::graph_decode_c06_identity::{
    PureDecodeGraphV1C06Identity, PureDecodeGraphV1C06IdentityBinding,
};
use super::graph_decode_preflight::PureDecodeGraphV1Ineligibility;

/// C07 facts paired with one checked complete C06 graph-cache identity.
///
/// The original partial identity is retained so the C07 binding and its
/// metadata/iteration components remain inspectable without rebuilding them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) struct PureDecodeGraphV1C06SignatureBinding {
    identity: PureDecodeGraphV1C06IdentityBinding,
    signature: GraphSignature,
}

impl PureDecodeGraphV1C06SignatureBinding {
    /// Returns the original checked C07-to-C06 partial identity.
    pub(crate) const fn identity(self) -> PureDecodeGraphV1C06IdentityBinding {
        self.identity
    }

    /// Returns the complete immutable C06 cache-key value.
    #[must_use]
    pub(crate) const fn signature(self) -> GraphSignature {
        self.signature
    }
}

/// Closed C07-20 result for checked complete C06 identity assembly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
#[allow(clippy::large_enum_variant)] // C07 preserves allocation-free value composition.
pub(crate) enum PureDecodeGraphV1C06Signature {
    /// The C07 partial identity and static C06 metadata identity matched exactly.
    Bound(PureDecodeGraphV1C06SignatureBinding),
    /// The V1 candidate was already outside C07 before static identity inspection.
    Ineligible(PureDecodeGraphV1Ineligibility),
}

/// Result of composing one C07-19 partial identity with one static C06 value.
pub(crate) type PureDecodeGraphV1C06SignatureResult =
    Result<PureDecodeGraphV1C06Signature, GraphStaticMetadataLayoutMismatch>;

/// Assembles a complete C06 graph-cache key after exact metadata verification.
///
/// A bound C07-19 value carries the only C07 metadata and iteration facts this
/// slice accepts. The independently supplied static value must already have
/// the same full metadata layout identity; C06 rejects a schema or digest
/// mismatch without replacing any part of that static value. Other static
/// facts remain caller-owned cold identity and are preserved, not validated,
/// by this C07 bridge. An ineligible candidate is returned unchanged before
/// static metadata comparison or key construction.
pub(crate) fn compose_pure_decode_graph_v1_c06_signature(
    identity: &PureDecodeGraphV1C06Identity,
    static_signature: GraphStaticSignature,
) -> PureDecodeGraphV1C06SignatureResult {
    match *identity {
        PureDecodeGraphV1C06Identity::Bound(identity) => compose_graph_signature_checked_metadata(
            static_signature,
            identity.metadata_layout_signature(),
            identity.iteration_signature(),
        )
        .map(|signature| {
            PureDecodeGraphV1C06Signature::Bound(PureDecodeGraphV1C06SignatureBinding {
                identity,
                signature,
            })
        }),
        PureDecodeGraphV1C06Identity::Ineligible(reason) => {
            Ok(PureDecodeGraphV1C06Signature::Ineligible(reason))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1C06Signature, PureDecodeGraphV1C06SignatureBinding,
        compose_pure_decode_graph_v1_c06_signature,
    };
    use crate::llama::graph::{
        GraphComputeType, GraphDataType, GraphDeviceSignature, GraphGemmPlanSetId,
        GraphGeometrySignature, GraphImplementationId, GraphImplementationSignature,
        GraphIterationSignature, GraphLayoutSignature, GraphMetadataLayoutSignature,
        GraphModelArchitecture, GraphModelSignature, GraphReductionPolicyId,
        GraphRevisionFingerprint, GraphSamplingBackend, GraphSignature, GraphStaticSignature,
        GraphTensorSignature, GraphWorkloadStage,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBinding;
    use crate::llama::graph_decode_c06_identity::{
        PureDecodeGraphV1C06Identity, PureDecodeGraphV1C06IdentityBinding,
        bind_pure_decode_graph_v1_c06_identity,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;
    use crate::llama::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
    use crate::llama::graph_decode_preflight_binding::PureDecodeGraphV1LayoutBinding;

    fn layout(
        bucket_rows: u32,
        block_entry_capacity: u64,
        header_bytes: u64,
        control_status_bytes: u64,
    ) -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(
            bucket_rows,
            block_entry_capacity,
            header_bytes,
            control_status_bytes,
        ))
        .expect("valid fixed C07 metadata layout")
    }

    fn partial_identity(
        active_rows: u32,
        layout: PureDecodeGraphMetadataLayout,
        sampling_backend: GraphSamplingBackend,
    ) -> PureDecodeGraphV1C06Identity {
        let padding = plan_pure_decode_graph_padding(active_rows)
            .expect("supported active rows select an exact C07 bucket");
        let binding = PureDecodeGraphV1LayoutBinding::Bound(
            PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("matching exact C07 bucket must bind"),
        );
        bind_pure_decode_graph_v1_c06_identity(&binding, sampling_backend)
    }

    fn expect_partial_bound(
        identity: &PureDecodeGraphV1C06Identity,
    ) -> PureDecodeGraphV1C06IdentityBinding {
        match *identity {
            PureDecodeGraphV1C06Identity::Bound(binding) => binding,
            PureDecodeGraphV1C06Identity::Ineligible(reason) => {
                panic!("expected a bound C07 partial identity, got {reason:?}")
            }
        }
    }

    fn expect_bound(
        result: &PureDecodeGraphV1C06Signature,
    ) -> PureDecodeGraphV1C06SignatureBinding {
        match *result {
            PureDecodeGraphV1C06Signature::Bound(binding) => binding,
            PureDecodeGraphV1C06Signature::Ineligible(reason) => {
                panic!("expected a complete C06 identity, got {reason:?}")
            }
        }
    }

    fn static_signature(
        metadata_layout: GraphMetadataLayoutSignature,
        revision: u8,
    ) -> GraphStaticSignature {
        let revision_u32 = u32::from(revision);

        GraphStaticSignature::new(
            GraphModelSignature::new(
                GraphModelArchitecture::LlamaDecoder,
                revision_u32,
                GraphRevisionFingerprint::from_bytes([revision; 32]),
                revision_u32,
            ),
            GraphDeviceSignature::new(8, 9, 12_804, 12_804, revision_u32),
            GraphTensorSignature::new(
                GraphDataType::BFloat16,
                GraphDataType::BFloat16,
                GraphComputeType::Float32,
            ),
            GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 128),
            GraphLayoutSignature::new(8_192, 16, revision_u32, metadata_layout),
            GraphImplementationSignature::new(
                GraphImplementationId::new(revision_u32),
                GraphImplementationId::new(revision_u32 + 1),
                GraphImplementationId::new(revision_u32 + 2),
                GraphImplementationId::new(revision_u32 + 3),
                GraphGemmPlanSetId::new(revision_u32),
                GraphReductionPolicyId::new(revision_u32),
            ),
        )
    }

    #[test]
    fn every_supported_c07_candidate_assembles_one_exact_complete_c06_signature() {
        for active_rows in 1..=32 {
            let bucket = plan_pure_decode_graph_padding(active_rows)
                .expect("supported active rows select an exact C07 bucket")
                .bucket_rows();
            let layout = layout(bucket, u64::from(bucket), 1, 1);
            let partial = partial_identity(active_rows, layout, GraphSamplingBackend::GpuGreedy);
            let partial_binding = expect_partial_bound(&partial);
            let static_signature = static_signature(partial_binding.metadata_layout_signature(), 1);

            let composed = compose_pure_decode_graph_v1_c06_signature(&partial, static_signature)
                .expect("matching C07 and C06 metadata identities must compose");
            let bound = expect_bound(&composed);

            assert_eq!(bound.identity(), partial_binding);
            assert_eq!(
                bound.signature(),
                GraphSignature::new(static_signature, partial_binding.iteration_signature()),
            );
        }
    }

    #[test]
    fn schema_or_digest_mismatch_rejects_without_rebuilding_static_identity() {
        let partial = partial_identity(3, layout(4, 4, 1, 1), GraphSamplingBackend::GpuGreedy);
        let expected = expect_partial_bound(&partial).metadata_layout_signature();

        for actual in [
            GraphMetadataLayoutSignature::new(expected.schema_version() + 1, *expected.digest()),
            GraphMetadataLayoutSignature::new(expected.schema_version(), [0xE7; 32]),
        ] {
            assert!(
                compose_pure_decode_graph_v1_c06_signature(&partial, static_signature(actual, 1))
                    .is_err(),
                "C07 metadata must exactly match the independent static identity"
            );
        }
    }

    #[test]
    fn ineligible_candidate_short_circuits_before_static_metadata_comparison() {
        let reason = PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 };

        assert_eq!(
            compose_pure_decode_graph_v1_c06_signature(
                &PureDecodeGraphV1C06Identity::Ineligible(reason),
                static_signature(GraphMetadataLayoutSignature::new(99, [0xA5; 32]), 1),
            ),
            Ok(PureDecodeGraphV1C06Signature::Ineligible(reason)),
        );
    }

    #[test]
    fn matching_metadata_preserves_unvalidated_static_facts_in_the_complete_key() {
        let partial = partial_identity(4, layout(4, 4, 1, 1), GraphSamplingBackend::Unsupported);
        let metadata_layout = expect_partial_bound(&partial).metadata_layout_signature();
        let first_static = static_signature(metadata_layout, 1);
        let second_static = static_signature(metadata_layout, 2);
        let first_composed = compose_pure_decode_graph_v1_c06_signature(&partial, first_static)
            .expect("matching metadata must compose");
        let first = expect_bound(&first_composed).signature();
        let second_composed = compose_pure_decode_graph_v1_c06_signature(&partial, second_static)
            .expect("matching metadata must compose");
        let second = expect_bound(&second_composed).signature();

        assert_eq!(first.static_signature(), first_static);
        assert_eq!(second.static_signature(), second_static);
        assert_eq!(first.iteration(), second.iteration());
        assert_eq!(
            first.iteration(),
            GraphIterationSignature::new(
                GraphWorkloadStage::PureDecode,
                4,
                GraphSamplingBackend::Unsupported,
            )
        );
        assert_ne!(first, second);
        assert_ne!(first.fingerprint(), second.fingerprint());
    }
}
