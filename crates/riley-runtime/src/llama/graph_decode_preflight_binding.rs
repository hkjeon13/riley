//! C07-6 candidate and C07-4 cold-layout composition.
//!
//! This C07-7 value-only step preserves an ineligible V1 candidate unchanged,
//! or binds its already selected padding plan to one caller-supplied cold
//! layout. It does not inspect or transform V1 metadata itself.

use super::graph_decode_binding::{
    PureDecodeGraphMetadataBinding, PureDecodeGraphMetadataBindingError,
};
use super::graph_decode_layout::PureDecodeGraphMetadataLayout;
use super::graph_decode_preflight::{PureDecodeGraphV1Ineligibility, PureDecodeGraphV1Preflight};

/// Closed result of composing one V1 candidate result with one cold layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
#[allow(clippy::large_enum_variant)] // C07 keeps this closed composition allocation-free.
pub(crate) enum PureDecodeGraphV1LayoutBinding {
    /// The eligible V1 candidate and cold layout name the same exact bucket.
    Bound(PureDecodeGraphMetadataBinding),
    /// The V1 candidate was already ineligible before layout binding.
    Ineligible(PureDecodeGraphV1Ineligibility),
}

/// Result of binding one closed V1 candidate result to one cold layout.
pub(crate) type PureDecodeGraphV1LayoutBindingResult =
    Result<PureDecodeGraphV1LayoutBinding, PureDecodeGraphMetadataBindingError>;

/// Binds an eligible V1 candidate result to one exact cold layout bucket.
///
/// An ineligible result is returned unchanged without consulting the layout.
/// An eligible result delegates only to the C07-4 exact bucket binder, so a
/// mismatch remains its typed error rather than causing reselection or a
/// larger-bucket fallback.
pub(crate) fn bind_pure_decode_graph_v1_preflight(
    preflight: PureDecodeGraphV1Preflight,
    layout: PureDecodeGraphMetadataLayout,
) -> PureDecodeGraphV1LayoutBindingResult {
    match preflight {
        PureDecodeGraphV1Preflight::Eligible(padding) => {
            PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .map(PureDecodeGraphV1LayoutBinding::Bound)
        }
        PureDecodeGraphV1Preflight::Ineligible(reason) => {
            Ok(PureDecodeGraphV1LayoutBinding::Ineligible(reason))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PureDecodeGraphV1LayoutBinding, bind_pure_decode_graph_v1_preflight};
    use crate::llama::graph_decode_binding::{
        PureDecodeGraphMetadataBinding, PureDecodeGraphMetadataBindingError,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;
    use crate::llama::graph_decode_preflight::{
        PureDecodeGraphV1Ineligibility, PureDecodeGraphV1Preflight,
    };

    fn layout(bucket_rows: u32) -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(
            bucket_rows,
            u64::from(bucket_rows),
            1,
            1,
        ))
        .expect("every catalog bucket has a minimal valid cold layout")
    }

    #[test]
    fn every_supported_candidate_binds_its_exact_cold_bucket_and_digest() {
        for active_rows in 1..=32 {
            let padding = plan_pure_decode_graph_padding(active_rows)
                .expect("every supported active row count has a padding plan");
            let layout = layout(padding.bucket_rows());
            let expected = PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("matching exact bucket must bind");

            assert_eq!(
                bind_pure_decode_graph_v1_preflight(
                    PureDecodeGraphV1Preflight::Eligible(padding),
                    layout,
                ),
                Ok(PureDecodeGraphV1LayoutBinding::Bound(expected)),
            );
        }
    }

    #[test]
    fn eligible_candidate_preserves_the_typed_layout_bucket_mismatch() {
        let padding = plan_pure_decode_graph_padding(3).expect("A3 selects M4");

        assert_eq!(
            bind_pure_decode_graph_v1_preflight(
                PureDecodeGraphV1Preflight::Eligible(padding),
                layout(8),
            ),
            Err(
                PureDecodeGraphMetadataBindingError::LayoutPaddingBucketMismatch {
                    layout_bucket: 8,
                    padding_bucket: 4,
                }
            ),
        );
    }

    #[test]
    fn ineligible_candidate_short_circuits_before_layout_binding() {
        let reason = PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
            prefill_rows: 1,
            prefill_tokens: 1,
        };

        assert_eq!(
            bind_pure_decode_graph_v1_preflight(
                PureDecodeGraphV1Preflight::Ineligible(reason),
                layout(8),
            ),
            Ok(PureDecodeGraphV1LayoutBinding::Ineligible(reason)),
        );
    }
}
