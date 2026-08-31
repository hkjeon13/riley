//! Read-only C07 eligibility for one already validated pure-decode batch view.
//!
//! This C07-6 classifier reads the existing V1 metadata facts only. It neither
//! adapts that ABI, writes fixed metadata, chooses placeholder values, nor
//! grants any execution or capture authority.

use super::batch::LlamaPackedBatchMetadata;
use super::graph_decode_padding::{PureDecodeGraphPaddingPlan, plan_pure_decode_graph_padding};

/// Closed result of checking one base-validated V1 metadata view for C07.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) enum PureDecodeGraphV1Preflight {
    /// The view is an exact pure-decode candidate with a checked C07 bucket.
    Eligible(PureDecodeGraphPaddingPlan),
    /// The view remains outside C07 and must not select a graph bucket here.
    Ineligible(PureDecodeGraphV1Ineligibility),
}

/// Closed reason an already validated V1 metadata view is not a C07 candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1Ineligibility {
    /// At least one prefill row or token is present in the iteration.
    PrefillWorkPresent {
        /// Number of prefill rows reported by the validated metadata view.
        prefill_rows: usize,
        /// Number of prefill tokens reported by the validated metadata view.
        prefill_tokens: usize,
    },
    /// The decode-row count does not cover every iteration row.
    DecodeRowCountMismatch {
        /// Total iteration row count.
        row_count: usize,
        /// Rows classified as decode by the validated metadata view.
        decode_rows: usize,
    },
    /// Decode-token facts are not exactly one token per iteration row.
    DecodeTokenShapeMismatch {
        /// Total iteration row count.
        row_count: usize,
        /// Total flattened input-token count.
        total_input_tokens: usize,
        /// Flattened token count classified as decode.
        decode_tokens: usize,
    },
    /// The validated metadata does not request one output per iteration row.
    OutputCountMismatch {
        /// Total iteration row count.
        row_count: usize,
        /// Number of requested output slots.
        output_count: usize,
    },
    /// The host row count cannot be represented by the C07 catalog width.
    RowCountNotRepresentable {
        /// Host row count that did not fit the C07 catalog width.
        row_count: usize,
    },
    /// The exact active-row count has no initial C07 bucket.
    UnsupportedActiveRows {
        /// Checked active-row count passed to the C07 catalog.
        active_rows: u32,
    },
}

/// Reads one already validated V1 metadata view without changing it.
///
/// Eligibility requires no prefill work, every row and token classified as
/// decode, exactly one input token and output per row, a checked `u32` active
/// count, and an existing C07 bucket. This function never replaces an
/// unsupported count with a larger bucket.
pub(crate) fn preflight_pure_decode_graph_v1(
    metadata: &LlamaPackedBatchMetadata<'_>,
) -> PureDecodeGraphV1Preflight {
    preflight_counts(PureDecodeGraphV1Counts::from_metadata(metadata))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PureDecodeGraphV1Counts {
    row_count: usize,
    prefill_rows: usize,
    decode_rows: usize,
    prefill_tokens: usize,
    total_input_tokens: usize,
    decode_tokens: usize,
    output_count: usize,
}

impl PureDecodeGraphV1Counts {
    fn from_metadata(metadata: &LlamaPackedBatchMetadata<'_>) -> Self {
        let metadata = *metadata;
        Self {
            row_count: metadata.row_count(),
            prefill_rows: metadata.prefill_row_count(),
            decode_rows: metadata.decode_row_count(),
            prefill_tokens: metadata.prefill_token_count(),
            total_input_tokens: metadata.total_input_tokens(),
            decode_tokens: metadata.decode_token_count(),
            output_count: metadata.output_count(),
        }
    }
}

fn preflight_counts(counts: PureDecodeGraphV1Counts) -> PureDecodeGraphV1Preflight {
    if counts.prefill_rows != 0 || counts.prefill_tokens != 0 {
        return PureDecodeGraphV1Preflight::Ineligible(
            PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                prefill_rows: counts.prefill_rows,
                prefill_tokens: counts.prefill_tokens,
            },
        );
    }
    if counts.decode_rows != counts.row_count {
        return PureDecodeGraphV1Preflight::Ineligible(
            PureDecodeGraphV1Ineligibility::DecodeRowCountMismatch {
                row_count: counts.row_count,
                decode_rows: counts.decode_rows,
            },
        );
    }
    if counts.decode_tokens != counts.total_input_tokens
        || counts.total_input_tokens != counts.row_count
    {
        return PureDecodeGraphV1Preflight::Ineligible(
            PureDecodeGraphV1Ineligibility::DecodeTokenShapeMismatch {
                row_count: counts.row_count,
                total_input_tokens: counts.total_input_tokens,
                decode_tokens: counts.decode_tokens,
            },
        );
    }
    if counts.output_count != counts.row_count {
        return PureDecodeGraphV1Preflight::Ineligible(
            PureDecodeGraphV1Ineligibility::OutputCountMismatch {
                row_count: counts.row_count,
                output_count: counts.output_count,
            },
        );
    }
    let Ok(active_rows) = u32::try_from(counts.row_count) else {
        return PureDecodeGraphV1Preflight::Ineligible(
            PureDecodeGraphV1Ineligibility::RowCountNotRepresentable {
                row_count: counts.row_count,
            },
        );
    };
    match plan_pure_decode_graph_padding(active_rows) {
        Some(plan) => PureDecodeGraphV1Preflight::Eligible(plan),
        None => PureDecodeGraphV1Preflight::Ineligible(
            PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1Counts, PureDecodeGraphV1Ineligibility, PureDecodeGraphV1Preflight,
        preflight_counts, preflight_pure_decode_graph_v1,
    };
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
    };
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
        row_count: usize,
        prefill_row_count: usize,
        decode_row_count: usize,
        prefill_token_count: usize,
        total_input_tokens: usize,
        decode_token_count: usize,
        output_count: usize,
    }

    fn metadata_snapshot(metadata: &LlamaPackedBatchMetadata<'_>) -> MetadataSnapshot {
        let metadata = *metadata;
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
            row_count: metadata.row_count(),
            prefill_row_count: metadata.prefill_row_count(),
            decode_row_count: metadata.decode_row_count(),
            prefill_token_count: metadata.prefill_token_count(),
            total_input_tokens: metadata.total_input_tokens(),
            decode_token_count: metadata.decode_token_count(),
            output_count: metadata.output_count(),
        }
    }

    fn decode_config<const ROWS: usize>() -> LlamaBatchMetadataConfig {
        LlamaBatchMetadataConfig::new(ROWS, ROWS, ROWS, ROWS, ROWS)
            .expect("one-token decode fixture must fit its exact cold bounds")
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

    #[test]
    fn every_supported_active_count_is_a_read_only_exact_or_padded_candidate() {
        let token_ids = [7; 32];
        let physical_block_ids = std::array::from_fn(|index| {
            u32::try_from(index).expect("fixture block index must fit u32")
        });
        let valid_tokens = [1; 32];
        let rows = decode_rows(&token_ids, &physical_block_ids, &valid_tokens);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(decode_config::<32>())
            .expect("decode fixture must prepare");

        for active_rows in 1..=32 {
            let active_rows_u32 =
                u32::try_from(active_rows).expect("fixture row count must fit u32");
            let expected_bucket = match active_rows {
                1 => 1,
                2 => 2,
                3..=4 => 4,
                5..=8 => 8,
                9..=16 => 16,
                17..=32 => 32,
                _ => unreachable!("fixture only iterates the initial C07 catalog range"),
            };
            let metadata = prepared
                .pack(&rows[..active_rows])
                .expect("one-token decode rows must base-validate");
            let before = metadata_snapshot(&metadata);
            let plan = match preflight_pure_decode_graph_v1(&metadata) {
                PureDecodeGraphV1Preflight::Eligible(plan) => plan,
                PureDecodeGraphV1Preflight::Ineligible(reason) => {
                    panic!("supported pure-decode metadata unexpectedly rejected: {reason:?}");
                }
            };

            assert_eq!(plan.active_rows(), active_rows_u32);
            assert_eq!(plan.bucket_rows(), expected_bucket);
            assert_eq!(plan.active_rows() + plan.padding_rows(), plan.bucket_rows());
            assert_eq!(metadata_snapshot(&metadata), before);
        }
    }

    #[test]
    fn over_catalog_pure_decode_batch_stays_ineligible_without_maximum_fallback() {
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
            preflight_pure_decode_graph_v1(&metadata),
            PureDecodeGraphV1Preflight::Ineligible(
                PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 }
            )
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    #[test]
    fn one_token_prefill_with_an_output_stays_ineligible_without_writing_metadata() {
        let tokens = [10];
        let physical_blocks = [0];
        let valid_tokens = [1];
        let rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Prefill,
            &tokens,
            1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &physical_blocks, &valid_tokens, 1),
            Some(0),
        )];
        let config = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)
            .expect("one-token prefill fixture must fit its exact cold bounds");
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(config).expect("prefill fixture must prepare");
        let metadata = prepared
            .pack(&rows)
            .expect("one-token prefill must base-validate");
        let before = metadata_snapshot(&metadata);

        assert_eq!(
            preflight_pure_decode_graph_v1(&metadata),
            PureDecodeGraphV1Preflight::Ineligible(
                PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                    prefill_rows: 1,
                    prefill_tokens: 1,
                }
            )
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    const PURE_COUNTS: PureDecodeGraphV1Counts = PureDecodeGraphV1Counts {
        row_count: 4,
        prefill_rows: 0,
        decode_rows: 4,
        prefill_tokens: 0,
        total_input_tokens: 4,
        decode_tokens: 4,
        output_count: 4,
    };

    fn assert_ineligible(
        counts: PureDecodeGraphV1Counts,
        expected: PureDecodeGraphV1Ineligibility,
    ) {
        assert_eq!(
            preflight_counts(counts),
            PureDecodeGraphV1Preflight::Ineligible(expected)
        );
    }

    #[test]
    fn count_classifier_accepts_coherent_counts_and_rejects_empty() {
        assert!(matches!(
            preflight_counts(PURE_COUNTS),
            PureDecodeGraphV1Preflight::Eligible(_)
        ));
        assert_ineligible(
            PureDecodeGraphV1Counts {
                row_count: 0,
                prefill_rows: 0,
                decode_rows: 0,
                prefill_tokens: 0,
                total_input_tokens: 0,
                decode_tokens: 0,
                output_count: 0,
            },
            PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 0 },
        );
    }

    #[test]
    fn count_classifier_rejects_every_non_pure_decode_shape_closed() {
        assert_ineligible(
            PureDecodeGraphV1Counts {
                prefill_rows: 1,
                prefill_tokens: 1,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                prefill_rows: 1,
                prefill_tokens: 1,
            },
        );
        assert_ineligible(
            PureDecodeGraphV1Counts {
                prefill_rows: 1,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                prefill_rows: 1,
                prefill_tokens: 0,
            },
        );
        assert_ineligible(
            PureDecodeGraphV1Counts {
                prefill_tokens: 1,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                prefill_rows: 0,
                prefill_tokens: 1,
            },
        );
        assert_ineligible(
            PureDecodeGraphV1Counts {
                decode_rows: 3,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::DecodeRowCountMismatch {
                row_count: 4,
                decode_rows: 3,
            },
        );
        assert_ineligible(
            PureDecodeGraphV1Counts {
                decode_tokens: 3,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::DecodeTokenShapeMismatch {
                row_count: 4,
                total_input_tokens: 4,
                decode_tokens: 3,
            },
        );
        assert_ineligible(
            PureDecodeGraphV1Counts {
                total_input_tokens: 3,
                decode_tokens: 3,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::DecodeTokenShapeMismatch {
                row_count: 4,
                total_input_tokens: 3,
                decode_tokens: 3,
            },
        );
        assert_ineligible(
            PureDecodeGraphV1Counts {
                output_count: 3,
                ..PURE_COUNTS
            },
            PureDecodeGraphV1Ineligibility::OutputCountMismatch {
                row_count: 4,
                output_count: 3,
            },
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn count_classifier_closes_when_host_rows_do_not_fit_the_c07_width() {
        let row_count = usize::MAX;
        assert_eq!(
            preflight_counts(PureDecodeGraphV1Counts {
                row_count,
                prefill_rows: 0,
                decode_rows: row_count,
                prefill_tokens: 0,
                total_input_tokens: row_count,
                decode_tokens: row_count,
                output_count: row_count,
            }),
            PureDecodeGraphV1Preflight::Ineligible(
                PureDecodeGraphV1Ineligibility::RowCountNotRepresentable { row_count }
            )
        );
    }
}
