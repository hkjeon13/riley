//! Exact bucket binding between cold metadata geometry and trailing padding.
//!
//! This C07-4 contract only verifies that two already validated value objects
//! describe the same fixed bucket. It neither selects a bucket nor maps caller
//! rows, writes metadata, allocates storage, or grants graph execution rights.

use super::graph_decode_layout::{
    PureDecodeGraphMetadataGeometryDigest, PureDecodeGraphMetadataLayout,
};
use super::graph_decode_padding::PureDecodeGraphPaddingPlan;

/// Closed rejection when a cold metadata layout and iteration padding disagree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphMetadataBindingError {
    /// Both value inputs are valid but name different exact graph buckets.
    LayoutPaddingBucketMismatch {
        /// Bucket embedded in the cold metadata layout.
        layout_bucket: u32,
        /// Bucket selected by the trailing-padding plan.
        padding_bucket: u32,
    },
}

/// Result of exact cold-layout and dynamic-padding binding.
pub(crate) type PureDecodeGraphMetadataBindingResult<T> =
    Result<T, PureDecodeGraphMetadataBindingError>;

/// One value-only binding with a layout-derived cold geometry identity.
///
/// The digest is calculated from `layout` at construction so a caller cannot
/// pair this binding with a stale or unrelated geometry digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PureDecodeGraphMetadataBinding {
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    padding: PureDecodeGraphPaddingPlan,
}

impl PureDecodeGraphMetadataBinding {
    /// Binds two value contracts only when their exact C07 buckets match.
    ///
    /// This slice makes no hot-path lifecycle claim. A future owner may retain
    /// cold layout facts or memoize bindings only after its own lifecycle
    /// review. This function does not retry selection, replace a mismatch with
    /// a larger bucket, or validate metadata contents.
    pub(crate) fn try_new(
        layout: PureDecodeGraphMetadataLayout,
        padding: PureDecodeGraphPaddingPlan,
    ) -> PureDecodeGraphMetadataBindingResult<Self> {
        if layout.bucket_rows() != padding.bucket_rows() {
            return Err(
                PureDecodeGraphMetadataBindingError::LayoutPaddingBucketMismatch {
                    layout_bucket: layout.bucket_rows(),
                    padding_bucket: padding.bucket_rows(),
                },
            );
        }
        let geometry_digest = layout.geometry_digest();
        Ok(Self {
            layout,
            geometry_digest,
            padding,
        })
    }

    /// Returns the exact cold metadata layout used to derive this binding.
    #[must_use]
    pub(crate) const fn layout(self) -> PureDecodeGraphMetadataLayout {
        self.layout
    }

    /// Returns the cold geometry identity derived from the bound layout.
    #[must_use]
    pub(crate) const fn geometry_digest(self) -> PureDecodeGraphMetadataGeometryDigest {
        self.geometry_digest
    }

    /// Returns the iteration-local trailing-padding topology.
    #[must_use]
    pub(crate) const fn padding_plan(self) -> PureDecodeGraphPaddingPlan {
        self.padding
    }
}

#[cfg(test)]
mod tests {
    use super::{PureDecodeGraphMetadataBinding, PureDecodeGraphMetadataBindingError};
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;

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

    fn padding(active_rows: u32) -> crate::llama::graph_decode_padding::PureDecodeGraphPaddingPlan {
        plan_pure_decode_graph_padding(active_rows).expect("supported C07 active rows")
    }

    #[test]
    fn matching_bucket_binds_layout_padding_and_layout_derived_digest() {
        let layout = layout(4, 16, 5, 5);
        let expected_digest = layout.geometry_digest();
        let padding = padding(3);
        let binding = PureDecodeGraphMetadataBinding::try_new(layout, padding)
            .expect("M4 cold layout must bind A3 to M4 padding");

        assert_eq!(binding.layout(), layout);
        assert_eq!(binding.geometry_digest(), expected_digest);
        assert_eq!(binding.padding_plan(), padding);
    }

    #[test]
    fn mismatched_buckets_fail_closed_without_reselection_or_fallback() {
        assert_eq!(
            PureDecodeGraphMetadataBinding::try_new(layout(4, 16, 5, 5), padding(5)),
            Err(
                PureDecodeGraphMetadataBindingError::LayoutPaddingBucketMismatch {
                    layout_bucket: 4,
                    padding_bucket: 8,
                }
            )
        );
        assert_eq!(
            PureDecodeGraphMetadataBinding::try_new(layout(8, 16, 5, 5), padding(3)),
            Err(
                PureDecodeGraphMetadataBindingError::LayoutPaddingBucketMismatch {
                    layout_bucket: 8,
                    padding_bucket: 4,
                }
            )
        );
    }

    #[test]
    fn every_supported_active_count_binds_only_its_exact_selected_bucket() {
        for active_rows in 1..=32 {
            let padding = padding(active_rows);
            let layout = layout(padding.bucket_rows(), 32, 5, 5);
            let binding = PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("selected C07 bucket must bind itself");
            assert_eq!(binding.layout().bucket_rows(), padding.bucket_rows());
            assert_eq!(binding.padding_plan().active_rows(), active_rows);
        }
    }

    #[test]
    fn same_bucket_different_active_rows_share_geometry_but_not_padding() {
        let layout = layout(8, 16, 5, 5);
        let padded =
            PureDecodeGraphMetadataBinding::try_new(layout, padding(5)).expect("A5 selects M8");
        let exact =
            PureDecodeGraphMetadataBinding::try_new(layout, padding(8)).expect("A8 selects M8");

        assert_ne!(padded, exact);
        assert_eq!(padded.geometry_digest(), exact.geometry_digest());
        assert_eq!(padded.padding_plan().padding_rows(), 3);
        assert_eq!(exact.padding_plan().padding_rows(), 0);
    }

    #[test]
    fn same_bucket_different_cold_geometry_binds_but_keeps_distinct_digests() {
        let padding = padding(8);
        let first = PureDecodeGraphMetadataBinding::try_new(layout(8, 16, 5, 5), padding)
            .expect("first M8 geometry binds");
        let second = PureDecodeGraphMetadataBinding::try_new(layout(8, 17, 5, 5), padding)
            .expect("second M8 geometry binds");

        assert_ne!(first.geometry_digest(), second.geometry_digest());
        assert_eq!(first.padding_plan(), second.padding_plan());
    }

    #[test]
    fn unsupported_active_counts_produce_no_padding_input_for_binding() {
        for active_rows in [0, 33, u32::MAX] {
            assert_eq!(plan_pure_decode_graph_padding(active_rows), None);
        }
    }

    #[test]
    fn binding_is_a_copyable_value_only_contract() {
        let binding = PureDecodeGraphMetadataBinding::try_new(layout(4, 16, 5, 5), padding(3))
            .expect("matching M4 binding");
        let copied_binding = binding;
        assert_eq!(copied_binding, binding);
    }
}
