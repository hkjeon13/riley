//! C07 V1 candidate-to-C06 partial graph identity composition.
//!
//! C07-19 accepts only the closed C07-7 layout-binding result. A bound
//! candidate retains its C07 padding topology while deriving the two C06
//! identity components that C07 can establish: a layout-derived metadata
//! signature and a pure-decode iteration signature. It neither completes a
//! whole graph signature nor performs dispatch, capture, or execution.

use super::graph::{
    GraphIterationSignature, GraphMetadataLayoutSignature, GraphSamplingBackend, GraphWorkloadStage,
};
use super::graph_decode_binding::PureDecodeGraphMetadataBinding;
use super::graph_decode_layout_signature::pure_decode_graph_v1_metadata_layout_signature;
use super::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
use super::graph_decode_preflight_binding::PureDecodeGraphV1LayoutBinding;

/// C07 facts paired with the C06 identity components they can establish.
///
/// This value retains the original C07 binding so the active-row and trailing
/// padding topology remains available alongside the bucket-level C06
/// iteration identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) struct PureDecodeGraphV1C06IdentityBinding {
    metadata_binding: PureDecodeGraphMetadataBinding,
    metadata_layout_signature: GraphMetadataLayoutSignature,
    iteration_signature: GraphIterationSignature,
}

impl PureDecodeGraphV1C06IdentityBinding {
    /// Returns the original exact C07 layout and padding binding.
    #[must_use]
    pub(crate) const fn metadata_binding(self) -> PureDecodeGraphMetadataBinding {
        self.metadata_binding
    }

    /// Returns the C06 metadata identity derived from the bound C07 layout.
    #[must_use]
    pub(crate) const fn metadata_layout_signature(self) -> GraphMetadataLayoutSignature {
        self.metadata_layout_signature
    }

    /// Returns the C06 pure-decode bucket and observed sampling identity.
    #[must_use]
    pub(crate) const fn iteration_signature(self) -> GraphIterationSignature {
        self.iteration_signature
    }
}

/// Closed C06 partial-identity result for one C07-7 V1 layout binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
#[allow(clippy::large_enum_variant)] // C07 keeps this value-only composition allocation-free.
pub(crate) enum PureDecodeGraphV1C06Identity {
    /// A C07 candidate with exact layout/padding binding has partial C06 identity.
    Bound(PureDecodeGraphV1C06IdentityBinding),
    /// The V1 candidate was already outside C07 before C06 identity construction.
    Ineligible(PureDecodeGraphV1Ineligibility),
}

/// Derives C06 partial identity from one already closed C07-7 result.
///
/// A bound value derives its metadata identity directly from the C07 layout
/// and names the exact C07 bucket as a pure-decode iteration. The sampling
/// backend is an observed caller fact, not a capture-safety assertion; C06
/// retains responsibility for rejecting unsupported sampling during dispatch.
/// An ineligible value preserves its typed reason and never constructs C06
/// identity components.
pub(crate) fn bind_pure_decode_graph_v1_c06_identity(
    binding: &PureDecodeGraphV1LayoutBinding,
    sampling_backend: GraphSamplingBackend,
) -> PureDecodeGraphV1C06Identity {
    match *binding {
        PureDecodeGraphV1LayoutBinding::Bound(metadata_binding) => {
            let layout = metadata_binding.layout();
            let metadata_layout_signature = pure_decode_graph_v1_metadata_layout_signature(layout);
            let iteration_signature = GraphIterationSignature::new(
                GraphWorkloadStage::PureDecode,
                layout.bucket_rows(),
                sampling_backend,
            );
            PureDecodeGraphV1C06Identity::Bound(PureDecodeGraphV1C06IdentityBinding {
                metadata_binding,
                metadata_layout_signature,
                iteration_signature,
            })
        }
        PureDecodeGraphV1LayoutBinding::Ineligible(reason) => {
            PureDecodeGraphV1C06Identity::Ineligible(reason)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1C06Identity, PureDecodeGraphV1C06IdentityBinding,
        bind_pure_decode_graph_v1_c06_identity,
    };
    use crate::llama::graph::{GraphIterationSignature, GraphSamplingBackend, GraphWorkloadStage};
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBinding;
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_layout_signature::pure_decode_graph_v1_metadata_layout_signature;
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

    fn bound(
        active_rows: u32,
        layout: PureDecodeGraphMetadataLayout,
    ) -> PureDecodeGraphV1LayoutBinding {
        let padding = plan_pure_decode_graph_padding(active_rows)
            .expect("supported active rows select an exact C07 bucket");
        PureDecodeGraphV1LayoutBinding::Bound(
            PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("matching C07 bucket must bind"),
        )
    }

    fn expect_bound(
        identity: &PureDecodeGraphV1C06Identity,
    ) -> PureDecodeGraphV1C06IdentityBinding {
        match identity {
            PureDecodeGraphV1C06Identity::Bound(binding) => *binding,
            PureDecodeGraphV1C06Identity::Ineligible(reason) => {
                panic!("expected a bound C07 candidate, got {reason:?}")
            }
        }
    }

    #[test]
    fn every_supported_c07_candidate_derives_its_exact_c06_partial_identity() {
        for active_rows in 1..=32 {
            let bucket = plan_pure_decode_graph_padding(active_rows)
                .expect("supported active rows select an exact C07 bucket")
                .bucket_rows();
            let layout = layout(bucket, u64::from(bucket), 1, 1);
            let c07_binding = bound(active_rows, layout);
            let identity = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
                &c07_binding,
                GraphSamplingBackend::GpuGreedy,
            ));

            assert_eq!(
                identity.metadata_binding(),
                match c07_binding {
                    PureDecodeGraphV1LayoutBinding::Bound(binding) => binding,
                    PureDecodeGraphV1LayoutBinding::Ineligible(_) => unreachable!(),
                }
            );
            assert_eq!(
                identity.metadata_layout_signature(),
                pure_decode_graph_v1_metadata_layout_signature(layout),
            );
            assert_eq!(
                identity.iteration_signature(),
                GraphIterationSignature::new(
                    GraphWorkloadStage::PureDecode,
                    bucket,
                    GraphSamplingBackend::GpuGreedy,
                ),
            );
        }
    }

    #[test]
    fn ineligible_reason_is_preserved_without_constructing_partial_identity() {
        let reason = PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 };

        assert_eq!(
            bind_pure_decode_graph_v1_c06_identity(
                &PureDecodeGraphV1LayoutBinding::Ineligible(reason),
                GraphSamplingBackend::Unsupported,
            ),
            PureDecodeGraphV1C06Identity::Ineligible(reason),
        );
    }

    #[test]
    fn observed_sampling_backend_is_part_of_iteration_identity_without_admission_claim() {
        let layout = layout(4, 4, 1, 1);
        let greedy = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
            &bound(3, layout),
            GraphSamplingBackend::GpuGreedy,
        ));
        let unsupported = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
            &bound(3, layout),
            GraphSamplingBackend::Unsupported,
        ));

        assert_eq!(
            greedy.iteration_signature(),
            GraphIterationSignature::new(
                GraphWorkloadStage::PureDecode,
                4,
                GraphSamplingBackend::GpuGreedy,
            ),
        );
        assert_eq!(
            unsupported.iteration_signature(),
            GraphIterationSignature::new(
                GraphWorkloadStage::PureDecode,
                4,
                GraphSamplingBackend::Unsupported,
            ),
        );
        assert_ne!(
            greedy.iteration_signature(),
            unsupported.iteration_signature()
        );
        assert_eq!(
            greedy.metadata_layout_signature(),
            unsupported.metadata_layout_signature(),
        );
    }

    #[test]
    fn same_bucket_padding_topology_is_retained_while_c06_iteration_bucket_is_shared() {
        let layout = layout(8, 8, 1, 1);
        let padded = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
            &bound(5, layout),
            GraphSamplingBackend::GpuGreedy,
        ));
        let exact = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
            &bound(8, layout),
            GraphSamplingBackend::GpuGreedy,
        ));

        assert_ne!(padded.metadata_binding(), exact.metadata_binding());
        assert_eq!(padded.metadata_binding().padding_plan().padding_rows(), 3,);
        assert_eq!(exact.metadata_binding().padding_plan().padding_rows(), 0,);
        assert_eq!(padded.iteration_signature(), exact.iteration_signature());
        assert_eq!(
            padded.metadata_layout_signature(),
            exact.metadata_layout_signature(),
        );
    }

    #[test]
    fn distinct_cold_geometry_changes_metadata_identity_without_changing_bucket_identity() {
        let first = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
            &bound(4, layout(4, 4, 1, 1)),
            GraphSamplingBackend::GpuGreedy,
        ));
        let second = expect_bound(&bind_pure_decode_graph_v1_c06_identity(
            &bound(4, layout(4, 5, 1, 1)),
            GraphSamplingBackend::GpuGreedy,
        ));

        assert_ne!(
            first.metadata_layout_signature(),
            second.metadata_layout_signature(),
        );
        assert_eq!(first.iteration_signature(), second.iteration_signature());
    }
}
