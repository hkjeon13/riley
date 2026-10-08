//! Borrow-only composition of exact V1 metadata into C07-5 field order.
//!
//! This C07-10 bridge reborrows C07-9's successful nine-field grouping into
//! the fixed C07-5 source shape. It does not repeat prior validation, write a
//! slab, interpret opaque bytes, or grant graph execution rights.

use super::graph_decode_binding::PureDecodeGraphMetadataBinding;
use super::graph_decode_exact_sources::PureDecodeGraphV1ExactMetadataSources;
use super::graph_decode_packer::PureDecodeGraphMetadataFieldSources;

/// One exact C07-5 field-source view paired with its originating C07 binding.
///
/// The fields retain the shortest borrow of the C07-9 grouping, so a caller
/// cannot pair them with a different cold layout binding without explicitly
/// leaving this value boundary.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactFieldSourceBinding<'view> {
    binding: PureDecodeGraphMetadataBinding,
    fields: PureDecodeGraphMetadataFieldSources<'view>,
}

impl<'view> PureDecodeGraphV1ExactFieldSourceBinding<'view> {
    /// Returns the exact cold binding proven by the upstream C07-8/9 inputs.
    #[must_use]
    pub(crate) const fn binding(self) -> PureDecodeGraphMetadataBinding {
        self.binding
    }

    /// Returns all nine fields in C07-5's canonical fixed-source order.
    #[must_use]
    pub(crate) const fn field_sources(self) -> PureDecodeGraphMetadataFieldSources<'view> {
        self.fields
    }
}

/// Reborrows one successful C07-9 grouping into C07-5's nine-field source.
///
/// C07-8 already proved the seven native field capacities, and C07-9 already
/// proved the two opaque byte capacities. This bridge deliberately has no
/// error path and does not repeat either proof.
pub(crate) fn compose_pure_decode_graph_v1_exact_field_sources<'metadata, 'opaque, 'view>(
    source: &'view PureDecodeGraphV1ExactMetadataSources<'metadata, 'opaque>,
) -> PureDecodeGraphV1ExactFieldSourceBinding<'view>
where
    'metadata: 'view,
    'opaque: 'view,
{
    let native = source.native_fields();
    PureDecodeGraphV1ExactFieldSourceBinding {
        binding: native.binding(),
        fields: PureDecodeGraphMetadataFieldSources::new(
            source.header(),
            native.token_ids(),
            native.position_ids(),
            native.row_sequence_slots(),
            native.sequence_block_offsets(),
            native.physical_block_ids(),
            native.valid_tokens(),
            native.output_token_indices(),
            source.control_status(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::compose_pure_decode_graph_v1_exact_field_sources;
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph_decode_exact_projection::{
        PureDecodeGraphV1ExactNativeFields, PureDecodeGraphV1ExactProjection,
        project_pure_decode_graph_v1_exact,
    };
    use crate::llama::graph_decode_exact_sources::PureDecodeGraphV1ExactMetadataSources;
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout,
        PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_packer::pack_pure_decode_graph_metadata_le;
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

    fn layout() -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(2, 2, 3, 5))
            .expect("fixture cold layout must be valid")
    }

    fn pack_two_exact_decodes(
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
            .expect("two one-token decode rows must base-validate")
    }

    fn exact_native<'metadata>(
        metadata: &LlamaPackedBatchMetadata<'metadata>,
    ) -> PureDecodeGraphV1ExactNativeFields<'metadata> {
        match project_pure_decode_graph_v1_exact(metadata, layout())
            .expect("matching exact layout must bind")
        {
            PureDecodeGraphV1ExactProjection::Projected(native) => native,
            PureDecodeGraphV1ExactProjection::Ineligible(reason) => {
                panic!("exact native fields unexpectedly rejected: {reason:?}");
            }
        }
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

    fn assert_distinct_native_fixture_values(native: &PureDecodeGraphV1ExactNativeFields<'_>) {
        assert_eq!(native.token_ids(), &[0x0102_0304, 0xa0b0_c0d0]);
        assert_eq!(native.position_ids(), &[6, 12]);
        assert_eq!(native.row_sequence_slots(), &[0, 1]);
        assert_eq!(native.sequence_block_offsets(), &[0, 1, 2]);
        assert_eq!(native.physical_block_ids(), &[2, 3]);
        assert_eq!(native.valid_tokens(), &[7, 13]);
        assert_eq!(native.output_token_indices(), &[1, 0]);
    }

    fn assert_canonical_packed_field_mapping(
        destination: &[u8],
        layout: PureDecodeGraphMetadataLayout,
        native: &PureDecodeGraphV1ExactNativeFields<'_>,
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
            native.token_ids()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::PositionIds
            )),
            native.position_ids()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::RowSequenceSlots
            )),
            native.row_sequence_slots()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::SequenceBlockOffsets
            )),
            native.sequence_block_offsets()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::PhysicalBlockIds
            )),
            native.physical_block_ids()
        );
        assert_eq!(
            read_u16s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::ValidTokens
            )),
            native.valid_tokens()
        );
        assert_eq!(
            read_u32s(field_bytes(
                destination,
                layout,
                PureDecodeGraphMetadataField::OutputTokenIndices
            )),
            native.output_token_indices()
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
    fn successful_c07_9_grouping_reborrows_all_nine_fields_in_canonical_packer_order() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("two-token exact decode fixture must fit cold bounds");
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(config).expect("fixture must prepare");
        let metadata = pack_two_exact_decodes(&mut prepared);
        let metadata_before = metadata_snapshot(&metadata);
        let header = [0xa0, 0xa1, 0xa2];
        let control_status = [0xc0, 0xc1, 0xc2, 0xc3, 0xc4];
        let header_before = header;
        let control_status_before = control_status;
        let native = exact_native(&metadata);
        assert_distinct_native_fixture_values(&native);
        let exact =
            PureDecodeGraphV1ExactMetadataSources::try_new(&native, &header, &control_status)
                .expect("exact opaque regions must bind");

        let composed = compose_pure_decode_graph_v1_exact_field_sources(&exact);
        let binding = composed.binding();
        let cold_layout = binding.layout();
        assert_eq!(binding, native.binding());
        let slab_len =
            usize::try_from(cold_layout.total_bytes()).expect("fixture slab length must fit usize");
        let mut destination = vec![0xab; slab_len + 3];
        let tail_before = destination[slab_len..].to_vec();
        pack_pure_decode_graph_metadata_le(&binding, composed.field_sources(), &mut destination)
            .expect("exact C07-10 source must satisfy the C07-5 packer");

        assert_canonical_packed_field_mapping(
            &destination,
            cold_layout,
            &native,
            &header,
            &control_status,
        );
        assert_eq!(&destination[slab_len..], tail_before);
        assert_eq!(metadata_snapshot(&metadata), metadata_before);
        assert_eq!(header, header_before);
        assert_eq!(control_status, control_status_before);
    }
}
