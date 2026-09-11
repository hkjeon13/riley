//! Exact V1 C07 metadata writing into a caller-owned fixed slab.
//!
//! This C07-11 wrapper connects the prior exact source contracts to C07-5's
//! checked fixed-slab writer. It creates no storage, interprets no opaque
//! bytes, and does not transfer or execute graph work.

use super::batch::LlamaPackedBatchMetadata;
use super::graph_decode_binding::PureDecodeGraphMetadataBindingError;
use super::graph_decode_exact_field_sources::compose_pure_decode_graph_v1_exact_field_sources;
use super::graph_decode_exact_projection::{
    PureDecodeGraphV1ExactProjection, PureDecodeGraphV1ExactProjectionIneligibility,
    project_pure_decode_graph_v1_exact,
};
use super::graph_decode_exact_sources::{
    PureDecodeGraphV1ExactMetadataSources, PureDecodeGraphV1ExactOpaqueSourceError,
};
use super::graph_decode_layout::PureDecodeGraphMetadataLayout;
use super::graph_decode_packer::{
    PureDecodeGraphMetadataPackError, pack_pure_decode_graph_metadata_le,
};

/// Closed result after attempting to write exact V1 metadata into one C07 slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) enum PureDecodeGraphV1ExactSlabWrite {
    /// The caller-owned fixed slab received all nine exact source fields.
    Written,
    /// C07-8 rejected the V1 shape before opaque or destination validation.
    Ineligible(PureDecodeGraphV1ExactProjectionIneligibility),
}

/// Closed failure while preparing or writing one exact V1 C07 slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactSlabWriteError {
    /// C07-7 rejected an otherwise eligible V1 candidate's cold layout.
    LayoutBinding(PureDecodeGraphMetadataBindingError),
    /// C07-9 rejected a caller-owned opaque region length.
    OpaqueSource(PureDecodeGraphV1ExactOpaqueSourceError),
    /// C07-5 rejected the caller-owned fixed slab or field source.
    Pack(PureDecodeGraphMetadataPackError),
}

/// Result of exact V1 C07 fixed-slab writing.
pub(crate) type PureDecodeGraphV1ExactSlabWriteResult<T> =
    Result<T, PureDecodeGraphV1ExactSlabWriteError>;

/// Writes exact V1 C07 metadata in canonical little-endian fixed-slab order.
///
/// The sequence is C07-8 projection, C07-9 opaque-length binding, C07-10
/// source composition, then C07-5 writing. Every ineligible or error outcome
/// occurs before C07-5 changes the caller-owned destination.
pub(crate) fn write_pure_decode_graph_v1_exact_metadata_le(
    metadata: &LlamaPackedBatchMetadata<'_>,
    layout: PureDecodeGraphMetadataLayout,
    header: &[u8],
    control_status: &[u8],
    destination: &mut [u8],
) -> PureDecodeGraphV1ExactSlabWriteResult<PureDecodeGraphV1ExactSlabWrite> {
    let native = match project_pure_decode_graph_v1_exact(metadata, layout)
        .map_err(PureDecodeGraphV1ExactSlabWriteError::LayoutBinding)?
    {
        PureDecodeGraphV1ExactProjection::Projected(native) => native,
        PureDecodeGraphV1ExactProjection::Ineligible(reason) => {
            return Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(reason));
        }
    };
    let opaque = PureDecodeGraphV1ExactMetadataSources::try_new(&native, header, control_status)
        .map_err(PureDecodeGraphV1ExactSlabWriteError::OpaqueSource)?;
    let source = compose_pure_decode_graph_v1_exact_field_sources(&opaque);
    let binding = source.binding();
    pack_pure_decode_graph_metadata_le(&binding, source.field_sources(), destination)
        .map_err(PureDecodeGraphV1ExactSlabWriteError::Pack)?;
    Ok(PureDecodeGraphV1ExactSlabWrite::Written)
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1ExactSlabWrite, PureDecodeGraphV1ExactSlabWriteError,
        write_pure_decode_graph_v1_exact_metadata_le,
    };
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBindingError;
    use crate::llama::graph_decode_exact_projection::PureDecodeGraphV1ExactProjectionIneligibility;
    use crate::llama::graph_decode_exact_sources::{
        PureDecodeGraphV1ExactOpaqueField, PureDecodeGraphV1ExactOpaqueSourceError,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout,
        PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_packer::PureDecodeGraphMetadataPackError;
    use crate::llama::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
    use crate::paged_kv::{BLOCK_TABLE_V1_VERSION, KV_BLOCK_SIZE};

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
        .expect("fixture cold layout must be valid")
    }

    fn exact_layout() -> PureDecodeGraphMetadataLayout {
        layout(2, 2, 3, 5)
    }

    fn pack_two_distinct_exact_decodes(
        prepared: &mut PreparedLlamaBatchMetadata,
    ) -> LlamaPackedBatchMetadata<'_> {
        let token_ids = [0x0102_0304, 0xa0b0_c0d0];
        let physical_block_ids = [2, 3];
        let valid_tokens = [7, 13];
        let rows = [
            LlamaBatchRow::new(
                1,
                LlamaBatchRowKind::Decode,
                &token_ids[0..1],
                7,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &physical_block_ids[0..1],
                    &valid_tokens[0..1],
                    7,
                ),
                Some(1),
            ),
            LlamaBatchRow::new(
                2,
                LlamaBatchRowKind::Decode,
                &token_ids[1..2],
                13,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &physical_block_ids[1..2],
                    &valid_tokens[1..2],
                    13,
                ),
                Some(0),
            ),
        ];
        prepared
            .pack(&rows)
            .expect("two distinct one-token decode rows must base-validate")
    }

    fn pack_single_block_decode_rows<const ROWS: usize>(
        prepared: &mut PreparedLlamaBatchMetadata,
    ) -> LlamaPackedBatchMetadata<'_> {
        let token_ids = [7; ROWS];
        let physical_block_ids: [u32; ROWS] = std::array::from_fn(|index| {
            u32::try_from(index).expect("fixture physical block index must fit u32")
        });
        let valid_tokens = [1; ROWS];
        let rows: [LlamaBatchRow<'_>; ROWS] = std::array::from_fn(|index| {
            let index_u32 = u32::try_from(index).expect("fixture row index must fit u32");
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
        });
        prepared
            .pack(&rows)
            .expect("one-token decode rows must base-validate")
    }

    fn pack_two_decode_rows_with_three_blocks(
        prepared: &mut PreparedLlamaBatchMetadata,
    ) -> LlamaPackedBatchMetadata<'_> {
        let token_ids = [7, 8];
        let physical_block_ids = [0, 1, 2];
        let full_block_tokens =
            u16::try_from(KV_BLOCK_SIZE).expect("KV block size must fit fixture u16");
        let first_valid_tokens = [full_block_tokens, 1];
        let second_valid_tokens = [7];
        let first_target_length =
            u32::try_from(KV_BLOCK_SIZE + 1).expect("fixture target length must fit u32");
        let rows = [
            LlamaBatchRow::new(
                1,
                LlamaBatchRowKind::Decode,
                &token_ids[0..1],
                first_target_length,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &physical_block_ids[0..2],
                    &first_valid_tokens,
                    first_target_length,
                ),
                Some(0),
            ),
            LlamaBatchRow::new(
                2,
                LlamaBatchRowKind::Decode,
                &token_ids[1..2],
                7,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &physical_block_ids[2..3],
                    &second_valid_tokens,
                    7,
                ),
                Some(1),
            ),
        ];
        prepared
            .pack(&rows)
            .expect("two decode rows with three blocks must base-validate")
    }

    fn pack_one_prefill(prepared: &mut PreparedLlamaBatchMetadata) -> LlamaPackedBatchMetadata<'_> {
        let token_ids = [10];
        let physical_block_ids = [0];
        let valid_tokens = [1];
        let rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Prefill,
            &token_ids,
            1,
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                1,
            ),
            Some(0),
        )];
        prepared
            .pack(&rows)
            .expect("one-token prefill row must base-validate")
    }

    fn field_bytes(
        destination: &[u8],
        layout: PureDecodeGraphMetadataLayout,
        field: PureDecodeGraphMetadataField,
    ) -> &[u8] {
        let region = layout.region(field);
        let start = usize::try_from(region.offset()).expect("fixture offset must fit usize");
        let byte_len = usize::try_from(region.byte_len()).expect("fixture length must fit usize");
        &destination[start..start + byte_len]
    }

    fn read_u32s(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("exact u32 bytes")))
            .collect()
    }

    fn read_u16s(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("exact u16 bytes")))
            .collect()
    }

    fn assert_canonical_written_fields(
        destination: &[u8],
        layout: PureDecodeGraphMetadataLayout,
        metadata: &LlamaPackedBatchMetadata<'_>,
        header: &[u8],
        control_status: &[u8],
    ) {
        assert_eq!(
            field_bytes(destination, layout, PureDecodeGraphMetadataField::Header),
            header
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::TokenIds
            )),
            metadata.input_token_ids()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::PositionIds
            )),
            metadata.position_ids()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::RowSequenceSlots
            )),
            metadata.row_sequence_slots()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::SequenceBlockOffsets
            )),
            metadata.block_row_offsets()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::PhysicalBlockIds
            )),
            metadata.physical_block_ids()
        );
        assert_eq!(
            read_u16s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::ValidTokens
            )),
            metadata.valid_tokens()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::OutputTokenIndices
            )),
            metadata.output_token_indices()
        );
        assert_eq!(
            field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::ControlStatus
            ),
            control_status
        );
    }

    #[test]
    fn exact_m2_b2_metadata_writes_canonical_bytes_and_preserves_callers() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("two-row exact fixture must fit cold bounds");
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(config).expect("fixture must prepare");
        let metadata = pack_two_distinct_exact_decodes(&mut prepared);
        let metadata_before = metadata_snapshot(&metadata);
        let layout = exact_layout();
        let header = [0xa0, 0xa1, 0xa2];
        let control_status = [0xc0, 0xc1, 0xc2, 0xc3, 0xc4];
        let header_before = header;
        let control_status_before = control_status;
        let slab_len =
            usize::try_from(layout.total_bytes()).expect("fixture slab length must fit usize");
        let mut destination = vec![0xab; slab_len + 3];
        let tail_before = destination[slab_len..].to_vec();

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &metadata,
                layout,
                &header,
                &control_status,
                &mut destination,
            ),
            Ok(PureDecodeGraphV1ExactSlabWrite::Written)
        );

        assert_eq!(metadata.input_token_ids(), &[0x0102_0304, 0xa0b0_c0d0]);
        assert_eq!(metadata.position_ids(), &[6, 12]);
        assert_eq!(metadata.row_sequence_slots(), &[0, 1]);
        assert_eq!(metadata.block_row_offsets(), &[0, 1, 2]);
        assert_eq!(metadata.physical_block_ids(), &[2, 3]);
        assert_eq!(metadata.valid_tokens(), &[7, 13]);
        assert_eq!(metadata.output_token_indices(), &[1, 0]);
        assert_canonical_written_fields(&destination, layout, &metadata, &header, &control_status);
        assert_eq!(&destination[slab_len..], tail_before);
        assert_eq!(metadata_snapshot(&metadata), metadata_before);
        assert_eq!(header, header_before);
        assert_eq!(control_status, control_status_before);
    }

    #[test]
    fn prefill_and_over_catalog_batches_short_circuit_before_opaque_or_destination_checks() {
        let prefill_config = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)
            .expect("prefill fixture must fit cold bounds");
        let mut prefill_prepared =
            PreparedLlamaBatchMetadata::prepare(prefill_config).expect("fixture must prepare");
        let prefill = pack_one_prefill(&mut prefill_prepared);
        let prefill_before = metadata_snapshot(&prefill);
        let mut prefill_destination = [0xab];
        let prefill_destination_before = prefill_destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &prefill,
                layout(8, 8, 3, 5),
                &[0; 2],
                &[0; 4],
                &mut prefill_destination,
            ),
            Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(
                    PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                        prefill_rows: 1,
                        prefill_tokens: 1,
                    }
                )
            ))
        );
        assert_eq!(metadata_snapshot(&prefill), prefill_before);
        assert_eq!(prefill_destination, prefill_destination_before);

        let over_catalog_config = LlamaBatchMetadataConfig::new(33, 33, 33, 33, 33)
            .expect("over-catalog fixture must fit cold bounds");
        let mut over_catalog_prepared =
            PreparedLlamaBatchMetadata::prepare(over_catalog_config).expect("fixture must prepare");
        let over_catalog = pack_single_block_decode_rows::<33>(&mut over_catalog_prepared);
        let over_catalog_before = metadata_snapshot(&over_catalog);
        let mut over_catalog_destination = [0xab];
        let over_catalog_destination_before = over_catalog_destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &over_catalog,
                layout(32, 32, 3, 5),
                &[0; 2],
                &[0; 4],
                &mut over_catalog_destination,
            ),
            Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(
                    PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 }
                )
            ))
        );
        assert_eq!(metadata_snapshot(&over_catalog), over_catalog_before);
        assert_eq!(over_catalog_destination, over_catalog_destination_before);
    }

    #[test]
    fn row_and_block_tails_short_circuit_before_opaque_or_destination_checks() {
        let padded_config = LlamaBatchMetadataConfig::new(3, 3, 3, 3, 3)
            .expect("padded-row fixture must fit cold bounds");
        let mut padded_prepared =
            PreparedLlamaBatchMetadata::prepare(padded_config).expect("fixture must prepare");
        let padded = pack_single_block_decode_rows::<3>(&mut padded_prepared);
        let padded_before = metadata_snapshot(&padded);
        let mut padded_destination = [0xab];
        let padded_destination_before = padded_destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &padded,
                layout(4, 4, 3, 5),
                &[0; 2],
                &[0; 4],
                &mut padded_destination,
            ),
            Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::PaddingRequired {
                    active_rows: 3,
                    bucket_rows: 4,
                }
            ))
        );
        assert_eq!(metadata_snapshot(&padded), padded_before);
        assert_eq!(padded_destination, padded_destination_before);

        let exact_config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact-row fixture must fit cold bounds");
        let mut exact_prepared =
            PreparedLlamaBatchMetadata::prepare(exact_config).expect("fixture must prepare");
        let exact = pack_two_distinct_exact_decodes(&mut exact_prepared);
        let exact_before = metadata_snapshot(&exact);
        let mut short_block_destination = [0xab];
        let short_block_destination_before = short_block_destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &exact,
                layout(2, 3, 3, 5),
                &[0; 2],
                &[0; 4],
                &mut short_block_destination,
            ),
            Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::NativeFieldLengthMismatch {
                    field: PureDecodeGraphMetadataField::PhysicalBlockIds,
                    expected: 3,
                    actual: 2,
                }
            ))
        );
        assert_eq!(metadata_snapshot(&exact), exact_before);
        assert_eq!(short_block_destination, short_block_destination_before);

        let long_block_config = LlamaBatchMetadataConfig::new(2, 2, 3, 2, 3)
            .expect("long-block fixture must fit cold bounds");
        let mut long_block_prepared =
            PreparedLlamaBatchMetadata::prepare(long_block_config).expect("fixture must prepare");
        let long_block = pack_two_decode_rows_with_three_blocks(&mut long_block_prepared);
        let long_block_before = metadata_snapshot(&long_block);
        let mut long_block_destination = [0xab];
        let long_block_destination_before = long_block_destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &long_block,
                exact_layout(),
                &[0; 2],
                &[0; 4],
                &mut long_block_destination,
            ),
            Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::NativeFieldLengthMismatch {
                    field: PureDecodeGraphMetadataField::PhysicalBlockIds,
                    expected: 2,
                    actual: 3,
                }
            ))
        );
        assert_eq!(metadata_snapshot(&long_block), long_block_before);
        assert_eq!(long_block_destination, long_block_destination_before);
    }

    #[test]
    fn eligible_layout_mismatch_preserves_c07_7_error_before_opaque_or_destination_checks() {
        let config = LlamaBatchMetadataConfig::new(3, 3, 3, 3, 3)
            .expect("layout-mismatch fixture must fit cold bounds");
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(config).expect("fixture must prepare");
        let metadata = pack_single_block_decode_rows::<3>(&mut prepared);
        let metadata_before = metadata_snapshot(&metadata);
        let mut destination = [0xab];
        let destination_before = destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &metadata,
                layout(8, 8, 3, 5),
                &[0; 2],
                &[0; 4],
                &mut destination,
            ),
            Err(PureDecodeGraphV1ExactSlabWriteError::LayoutBinding(
                PureDecodeGraphMetadataBindingError::LayoutPaddingBucketMismatch {
                    layout_bucket: 8,
                    padding_bucket: 4,
                }
            ))
        );
        assert_eq!(metadata_snapshot(&metadata), metadata_before);
        assert_eq!(destination, destination_before);
    }

    #[test]
    fn opaque_length_errors_precede_destination_checks_with_header_then_control_order() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("opaque-error fixture must fit cold bounds");
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(config).expect("fixture must prepare");
        let metadata = pack_two_distinct_exact_decodes(&mut prepared);
        let metadata_before = metadata_snapshot(&metadata);
        let mut header_destination = [0xab];
        let header_destination_before = header_destination;

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &metadata,
                exact_layout(),
                &[0; 2],
                &[0; 4],
                &mut header_destination,
            ),
            Err(PureDecodeGraphV1ExactSlabWriteError::OpaqueSource(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::Header,
                    expected: 3,
                    actual: 2,
                }
            ))
        );
        assert_eq!(header_destination, header_destination_before);

        let mut control_destination = [0xab];
        let control_destination_before = control_destination;
        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &metadata,
                exact_layout(),
                &[0; 3],
                &[0; 4],
                &mut control_destination,
            ),
            Err(PureDecodeGraphV1ExactSlabWriteError::OpaqueSource(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::ControlStatus,
                    expected: 5,
                    actual: 4,
                }
            ))
        );
        assert_eq!(metadata_snapshot(&metadata), metadata_before);
        assert_eq!(control_destination, control_destination_before);
    }

    #[test]
    fn valid_exact_sources_preserve_original_destination_too_small_error_without_writing() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("small-destination fixture must fit cold bounds");
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(config).expect("fixture must prepare");
        let metadata = pack_two_distinct_exact_decodes(&mut prepared);
        let metadata_before = metadata_snapshot(&metadata);
        let layout = exact_layout();
        let required = layout.total_bytes();
        let available =
            usize::try_from(required - 1).expect("fixture short slab length must fit usize");
        let mut destination = vec![0xab; available];
        let destination_before = destination.clone();

        assert_eq!(
            write_pure_decode_graph_v1_exact_metadata_le(
                &metadata,
                layout,
                &[0; 3],
                &[0; 5],
                &mut destination,
            ),
            Err(PureDecodeGraphV1ExactSlabWriteError::Pack(
                PureDecodeGraphMetadataPackError::DestinationTooSmall {
                    required,
                    available,
                }
            ))
        );
        assert_eq!(metadata_snapshot(&metadata), metadata_before);
        assert_eq!(destination, destination_before);
    }
}
