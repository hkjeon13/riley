//! Exact C07-to-C06 cold metadata-layout identity bridge.
//!
//! C07-18 derives the generic C06 metadata-layout signature from one exact
//! C07 cold layout. It establishes only schema/digest identity; it does not
//! construct a whole graph signature, own an address, admit capture, or run
//! graph work.

use super::graph::GraphMetadataLayoutSignature;
use super::graph_decode_layout::PureDecodeGraphMetadataLayout;

/// Derives the generic C06 metadata-layout identity for one exact C07 layout.
///
/// The schema version and digest come directly from the C07 canonical cold
/// geometry. There is no caller-supplied digest, allocation, dynamic batch
/// fact, address, registry lookup, capture admission, or execution.
#[must_use]
pub(crate) fn pure_decode_graph_v1_metadata_layout_signature(
    layout: PureDecodeGraphMetadataLayout,
) -> GraphMetadataLayoutSignature {
    GraphMetadataLayoutSignature::new(
        PureDecodeGraphMetadataLayout::schema_version(),
        *layout.geometry_digest().as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::pure_decode_graph_v1_metadata_layout_signature;
    use crate::llama::graph_decode_layout::{
        PURE_DECODE_GRAPH_METADATA_LAYOUT_SCHEMA_VERSION, PureDecodeGraphMetadataLayout,
        PureDecodeGraphMetadataLayoutSpec,
    };

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
        .expect("representable exact C07 cold layout")
    }

    #[test]
    fn every_exact_c07_bucket_derives_its_own_c06_metadata_identity() {
        for bucket_rows in [1, 2, 4, 8, 16, 32] {
            let layout = layout(bucket_rows, u64::from(bucket_rows), 5, 7);
            let expected_digest = layout.geometry_digest();
            let signature = pure_decode_graph_v1_metadata_layout_signature(layout);

            assert_eq!(
                signature.schema_version(),
                PURE_DECODE_GRAPH_METADATA_LAYOUT_SCHEMA_VERSION
            );
            assert_eq!(signature.digest(), expected_digest.as_bytes());
        }
    }

    #[test]
    fn every_changed_cold_geometry_changes_the_derived_c06_identity() {
        let baseline = pure_decode_graph_v1_metadata_layout_signature(layout(8, 16, 5, 7));

        for changed in [
            layout(8, 17, 5, 7),
            layout(8, 16, 6, 7),
            layout(8, 16, 5, 8),
            layout(16, 16, 5, 7),
        ] {
            assert_ne!(
                baseline,
                pure_decode_graph_v1_metadata_layout_signature(changed)
            );
        }
    }

    #[test]
    fn same_cold_layout_derives_one_deterministic_copyable_identity() {
        let layout = layout(4, 8, 5, 7);
        let first = pure_decode_graph_v1_metadata_layout_signature(layout);
        let second = pure_decode_graph_v1_metadata_layout_signature(layout);
        let copied = first;

        assert_eq!(first, second);
        assert_eq!(copied, first);
    }
}
