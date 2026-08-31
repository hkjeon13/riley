//! Strict no-padding projection of V1 native fields into C07 field order.
//!
//! This C07-8 boundary accepts only an already validated V1 pure-decode batch
//! whose active rows and block entries exactly fill one cold C07 layout. It
//! returns borrowed native slices only; it does not define any missing-tail
//! values or fixed metadata bytes.

use super::batch::LlamaPackedBatchMetadata;
use super::graph_decode_binding::{
    PureDecodeGraphMetadataBinding, PureDecodeGraphMetadataBindingError,
};
use super::graph_decode_layout::{PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout};
use super::graph_decode_preflight::{
    PureDecodeGraphV1Ineligibility, preflight_pure_decode_graph_v1,
};
use super::graph_decode_preflight_binding::{
    PureDecodeGraphV1LayoutBinding, bind_pure_decode_graph_v1_preflight,
};

/// Borrowed V1 native fields that exactly fill one C07 cold layout.
///
/// Header and control/status bytes are deliberately absent. This view carries
/// no row or block padding, and does not itself authorize fixed-slab writing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactNativeFields<'a> {
    binding: PureDecodeGraphMetadataBinding,
    token_ids: &'a [u32],
    position_ids: &'a [u32],
    row_sequence_slots: &'a [u32],
    sequence_block_offsets: &'a [u32],
    physical_block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    output_token_indices: &'a [u32],
}

impl<'a> PureDecodeGraphV1ExactNativeFields<'a> {
    /// Returns the exact cold layout and dynamic padding facts for this view.
    #[must_use]
    pub(crate) const fn binding(self) -> PureDecodeGraphMetadataBinding {
        self.binding
    }

    /// Returns one V1 input token per exact C07 bucket row.
    #[must_use]
    pub(crate) const fn token_ids(self) -> &'a [u32] {
        self.token_ids
    }

    /// Returns one V1 absolute position per exact C07 bucket row.
    #[must_use]
    pub(crate) const fn position_ids(self) -> &'a [u32] {
        self.position_ids
    }

    /// Returns one V1 token-to-row slot per exact C07 bucket row.
    #[must_use]
    pub(crate) const fn row_sequence_slots(self) -> &'a [u32] {
        self.row_sequence_slots
    }

    /// Returns the exact V1 block CSR offsets for the cold C07 bucket.
    #[must_use]
    pub(crate) const fn sequence_block_offsets(self) -> &'a [u32] {
        self.sequence_block_offsets
    }

    /// Returns the V1 physical block IDs that exactly fill cold capacity.
    #[must_use]
    pub(crate) const fn physical_block_ids(self) -> &'a [u32] {
        self.physical_block_ids
    }

    /// Returns V1 valid-token counts aligned with physical block IDs.
    #[must_use]
    pub(crate) const fn valid_tokens(self) -> &'a [u16] {
        self.valid_tokens
    }

    /// Returns one V1 output-token index per exact C07 bucket row.
    #[must_use]
    pub(crate) const fn output_token_indices(self) -> &'a [u32] {
        self.output_token_indices
    }
}

/// Closed reason a V1 batch cannot be projected without creating tail values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactProjectionIneligibility {
    /// The C07-6 V1 pure-decode classifier rejected the batch unchanged.
    Preflight(PureDecodeGraphV1Ineligibility),
    /// The selected C07 bucket has row lanes whose values are not defined here.
    PaddingRequired {
        /// Active V1 pure-decode rows in the iteration.
        active_rows: u32,
        /// Exact cold C07 bucket rows selected for the iteration.
        bucket_rows: u32,
    },
    /// One V1 native field did not exactly fill its fixed C07 field capacity.
    NativeFieldLengthMismatch {
        /// Fixed C07 field whose V1 native slice length differed.
        field: PureDecodeGraphMetadataField,
        /// Exact fixed C07 field element capacity.
        expected: u64,
        /// Actual V1 native slice length on this host.
        actual: usize,
    },
}

/// Closed C07-8 projection result for one V1 metadata view and cold layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
#[allow(clippy::large_enum_variant)] // C07 keeps the closed result allocation-free.
pub(crate) enum PureDecodeGraphV1ExactProjection<'a> {
    /// Every native field exactly fills the selected cold C07 layout.
    Projected(PureDecodeGraphV1ExactNativeFields<'a>),
    /// The batch needs a later padding or field-capacity contract.
    Ineligible(PureDecodeGraphV1ExactProjectionIneligibility),
}

/// Result of strict V1 native-field projection into one cold C07 layout.
pub(crate) type PureDecodeGraphV1ExactProjectionResult<'a> =
    Result<PureDecodeGraphV1ExactProjection<'a>, PureDecodeGraphMetadataBindingError>;

/// Projects V1 native fields only when no row or block tail must be invented.
///
/// C07-6 classifies the base-validated metadata first, and C07-7 binds its
/// eligible plan to the supplied cold layout. A C07-4 layout mismatch remains
/// an error. Once bound, this function requires zero row padding and exact
/// native field lengths for every non-opaque C07 field.
pub(crate) fn project_pure_decode_graph_v1_exact<'metadata>(
    metadata: &LlamaPackedBatchMetadata<'metadata>,
    layout: PureDecodeGraphMetadataLayout,
) -> PureDecodeGraphV1ExactProjectionResult<'metadata> {
    let preflight = preflight_pure_decode_graph_v1(metadata);
    match bind_pure_decode_graph_v1_preflight(preflight, layout)? {
        PureDecodeGraphV1LayoutBinding::Bound(binding) => {
            Ok(project_bound_v1_native_fields(metadata, &binding))
        }
        PureDecodeGraphV1LayoutBinding::Ineligible(reason) => {
            Ok(PureDecodeGraphV1ExactProjection::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(reason),
            ))
        }
    }
}

fn project_bound_v1_native_fields<'metadata>(
    metadata: &LlamaPackedBatchMetadata<'metadata>,
    binding: &PureDecodeGraphMetadataBinding,
) -> PureDecodeGraphV1ExactProjection<'metadata> {
    let padding = binding.padding_plan();
    if padding.padding_rows() != 0 {
        return PureDecodeGraphV1ExactProjection::Ineligible(
            PureDecodeGraphV1ExactProjectionIneligibility::PaddingRequired {
                active_rows: padding.active_rows(),
                bucket_rows: padding.bucket_rows(),
            },
        );
    }

    let bucket_rows = u64::from(padding.bucket_rows());
    let sequence_block_offsets = bucket_rows + 1;
    let block_entry_capacity = binding.layout().block_entry_capacity();
    for (field, expected, actual) in [
        (
            PureDecodeGraphMetadataField::TokenIds,
            bucket_rows,
            metadata.input_token_ids().len(),
        ),
        (
            PureDecodeGraphMetadataField::PositionIds,
            bucket_rows,
            metadata.position_ids().len(),
        ),
        (
            PureDecodeGraphMetadataField::RowSequenceSlots,
            bucket_rows,
            metadata.row_sequence_slots().len(),
        ),
        (
            PureDecodeGraphMetadataField::SequenceBlockOffsets,
            sequence_block_offsets,
            metadata.block_row_offsets().len(),
        ),
        (
            PureDecodeGraphMetadataField::PhysicalBlockIds,
            block_entry_capacity,
            metadata.physical_block_ids().len(),
        ),
        (
            PureDecodeGraphMetadataField::ValidTokens,
            block_entry_capacity,
            metadata.valid_tokens().len(),
        ),
        (
            PureDecodeGraphMetadataField::OutputTokenIndices,
            bucket_rows,
            metadata.output_token_indices().len(),
        ),
    ] {
        if u64::try_from(actual).ok() != Some(expected) {
            return PureDecodeGraphV1ExactProjection::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::NativeFieldLengthMismatch {
                    field,
                    expected,
                    actual,
                },
            );
        }
    }

    PureDecodeGraphV1ExactProjection::Projected(PureDecodeGraphV1ExactNativeFields {
        binding: *binding,
        token_ids: metadata.input_token_ids(),
        position_ids: metadata.position_ids(),
        row_sequence_slots: metadata.row_sequence_slots(),
        sequence_block_offsets: metadata.block_row_offsets(),
        physical_block_ids: metadata.physical_block_ids(),
        valid_tokens: metadata.valid_tokens(),
        output_token_indices: metadata.output_token_indices(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1ExactProjection, PureDecodeGraphV1ExactProjectionIneligibility,
        project_pure_decode_graph_v1_exact,
    };
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBindingError;
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout,
        PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

    #[derive(Debug, Eq, PartialEq)]
    struct MetadataSnapshot {
        schema_version: u16,
        block_table_schema_version: u16,
        sequence_tags: Vec<u64>,
        row_kind_codes: Vec<u8>,
        input_row_offsets: Vec<u32>,
        input_token_ids: Vec<u32>,
        position_ids: Vec<u32>,
        row_sequence_slots: Vec<u32>,
        block_row_offsets: Vec<u32>,
        physical_block_ids: Vec<u32>,
        valid_tokens: Vec<u16>,
        logical_lengths: Vec<u32>,
        output_slots_by_row: Vec<u32>,
        output_row_indices: Vec<u32>,
        output_token_indices: Vec<u32>,
        prefill_row_indices: Vec<u32>,
        decode_row_indices: Vec<u32>,
        prefill_token_count: usize,
        decode_token_count: usize,
    }

    fn metadata_snapshot(metadata: &LlamaPackedBatchMetadata<'_>) -> MetadataSnapshot {
        MetadataSnapshot {
            schema_version: metadata.schema_version(),
            block_table_schema_version: metadata.block_table_schema_version(),
            sequence_tags: metadata.sequence_tags().to_vec(),
            row_kind_codes: metadata.row_kind_codes().to_vec(),
            input_row_offsets: metadata.input_row_offsets().to_vec(),
            input_token_ids: metadata.input_token_ids().to_vec(),
            position_ids: metadata.position_ids().to_vec(),
            row_sequence_slots: metadata.row_sequence_slots().to_vec(),
            block_row_offsets: metadata.block_row_offsets().to_vec(),
            physical_block_ids: metadata.physical_block_ids().to_vec(),
            valid_tokens: metadata.valid_tokens().to_vec(),
            logical_lengths: metadata.logical_lengths().to_vec(),
            output_slots_by_row: metadata.output_slots_by_row().to_vec(),
            output_row_indices: metadata.output_row_indices().to_vec(),
            output_token_indices: metadata.output_token_indices().to_vec(),
            prefill_row_indices: metadata.prefill_row_indices().to_vec(),
            decode_row_indices: metadata.decode_row_indices().to_vec(),
            prefill_token_count: metadata.prefill_token_count(),
            decode_token_count: metadata.decode_token_count(),
        }
    }

    fn layout(bucket_rows: u32, block_entry_capacity: u64) -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(
            bucket_rows,
            block_entry_capacity,
            1,
            1,
        ))
        .expect("fixture cold layout must be valid")
    }

    fn decode_config<const ROWS: usize>() -> LlamaBatchMetadataConfig {
        LlamaBatchMetadataConfig::new(ROWS, ROWS, ROWS, ROWS, ROWS)
            .expect("one-token decode fixture must fit exact cold bounds")
    }

    fn decode_rows<'a, const ROWS: usize>(
        token_ids: &'a [u32; ROWS],
        physical_block_ids: &'a [u32; ROWS],
        valid_tokens: &'a [u16; ROWS],
    ) -> [LlamaBatchRow<'a>; ROWS] {
        std::array::from_fn(|index| {
            let index_u32 = u32::try_from(index).expect("fixture index must fit u32");
            LlamaBatchRow::new(
                u64::from(index_u32) + 1,
                LlamaBatchRowKind::Decode,
                &token_ids[index..=index],
                1,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &physical_block_ids[index..=index],
                    &valid_tokens[index..=index],
                    1,
                ),
                Some(index_u32),
            )
        })
    }

    fn assert_exact_projection<const ROWS: usize>() {
        let token_ids = [7; ROWS];
        let physical_block_ids = std::array::from_fn(|index| {
            u32::try_from(index).expect("fixture block index must fit u32")
        });
        let valid_tokens = [1; ROWS];
        let rows = decode_rows(&token_ids, &physical_block_ids, &valid_tokens);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<ROWS>())
            .expect("decode fixture must prepare");
        let metadata = prepared
            .pack(&rows)
            .expect("one-token decode rows must base-validate");
        let before = metadata_snapshot(&metadata);
        let rows_u32 = u32::try_from(ROWS).expect("fixture row count must fit u32");
        let fields = match project_pure_decode_graph_v1_exact(
            &metadata,
            layout(rows_u32, u64::from(rows_u32)),
        )
        .expect("matching exact layout must bind")
        {
            PureDecodeGraphV1ExactProjection::Projected(fields) => fields,
            PureDecodeGraphV1ExactProjection::Ineligible(reason) => {
                panic!("exact native fields unexpectedly rejected: {reason:?}");
            }
        };

        assert_eq!(fields.binding().padding_plan().active_rows(), rows_u32);
        assert_eq!(fields.binding().padding_plan().bucket_rows(), rows_u32);
        assert_eq!(fields.binding().padding_plan().padding_rows(), 0);
        assert_eq!(fields.token_ids(), metadata.input_token_ids());
        assert_eq!(fields.position_ids(), metadata.position_ids());
        assert_eq!(fields.row_sequence_slots(), metadata.row_sequence_slots());
        assert_eq!(
            fields.sequence_block_offsets(),
            metadata.block_row_offsets()
        );
        assert_eq!(fields.physical_block_ids(), metadata.physical_block_ids());
        assert_eq!(fields.valid_tokens(), metadata.valid_tokens());
        assert_eq!(
            fields.output_token_indices(),
            metadata.output_token_indices()
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    #[test]
    fn every_exact_initial_bucket_projects_all_seven_native_fields_without_mutation() {
        assert_exact_projection::<1>();
        assert_exact_projection::<2>();
        assert_exact_projection::<4>();
        assert_exact_projection::<8>();
        assert_exact_projection::<16>();
        assert_exact_projection::<32>();
    }

    #[test]
    fn padded_rows_remain_ineligible_before_any_native_field_projection() {
        let token_ids = [7; 3];
        let physical_block_ids = [0, 1, 2];
        let valid_tokens = [1; 3];
        let rows = decode_rows(&token_ids, &physical_block_ids, &valid_tokens);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<3>())
            .expect("three-row decode fixture must prepare");
        let metadata = prepared
            .pack(&rows)
            .expect("three one-token decode rows must base-validate");
        let before = metadata_snapshot(&metadata);

        assert_eq!(
            project_pure_decode_graph_v1_exact(&metadata, layout(4, 4)),
            Ok(PureDecodeGraphV1ExactProjection::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::PaddingRequired {
                    active_rows: 3,
                    bucket_rows: 4,
                }
            )),
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    #[test]
    fn exact_rows_reject_a_shorter_dynamic_block_span_than_cold_capacity() {
        let token_ids = [7; 4];
        let physical_block_ids = [0, 1, 2, 3];
        let valid_tokens = [1; 4];
        let rows = decode_rows(&token_ids, &physical_block_ids, &valid_tokens);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<4>())
            .expect("four-row decode fixture must prepare");
        let metadata = prepared
            .pack(&rows)
            .expect("four one-token decode rows must base-validate");
        let before = metadata_snapshot(&metadata);

        assert_eq!(
            project_pure_decode_graph_v1_exact(&metadata, layout(4, 5)),
            Ok(PureDecodeGraphV1ExactProjection::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::NativeFieldLengthMismatch {
                    field: PureDecodeGraphMetadataField::PhysicalBlockIds,
                    expected: 5,
                    actual: 4,
                }
            )),
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    #[test]
    fn prefill_and_over_catalog_batches_preserve_preflight_reason_before_layout_use() {
        let prefill_tokens = [10];
        let prefill_blocks = [0];
        let prefill_valid = [1];
        let prefill_rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Prefill,
            &prefill_tokens,
            1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &prefill_blocks, &prefill_valid, 1),
            Some(0),
        )];
        let mut prefill_prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<1>())
            .expect("one-token prefill fixture must prepare");
        let prefill = prefill_prepared
            .pack(&prefill_rows)
            .expect("one-token prefill must base-validate");
        let prefill_before = metadata_snapshot(&prefill);

        assert_eq!(
            project_pure_decode_graph_v1_exact(&prefill, layout(8, 8)),
            Ok(PureDecodeGraphV1ExactProjection::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(
                    PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                        prefill_rows: 1,
                        prefill_tokens: 1,
                    }
                )
            )),
        );
        assert_eq!(metadata_snapshot(&prefill), prefill_before);

        let token_ids = [7; 33];
        let physical_block_ids = std::array::from_fn(|index| {
            u32::try_from(index).expect("fixture block index must fit u32")
        });
        let valid_tokens = [1; 33];
        let rows = decode_rows(&token_ids, &physical_block_ids, &valid_tokens);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<33>())
            .expect("33-row decode fixture must prepare");
        let metadata = prepared
            .pack(&rows)
            .expect("33 one-token decode rows must base-validate");
        let before = metadata_snapshot(&metadata);

        assert_eq!(
            project_pure_decode_graph_v1_exact(&metadata, layout(32, 32)),
            Ok(PureDecodeGraphV1ExactProjection::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(
                    PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 }
                )
            )),
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    #[test]
    fn eligible_batch_preserves_c07_7_typed_layout_mismatch() {
        let token_ids = [7; 3];
        let physical_block_ids = [0, 1, 2];
        let valid_tokens = [1; 3];
        let rows = decode_rows(&token_ids, &physical_block_ids, &valid_tokens);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<3>())
            .expect("three-row decode fixture must prepare");
        let metadata = prepared
            .pack(&rows)
            .expect("three one-token decode rows must base-validate");
        let before = metadata_snapshot(&metadata);

        assert_eq!(
            project_pure_decode_graph_v1_exact(&metadata, layout(8, 8)),
            Err(
                PureDecodeGraphMetadataBindingError::LayoutPaddingBucketMismatch {
                    layout_bucket: 8,
                    padding_bucket: 4,
                }
            ),
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }
}
