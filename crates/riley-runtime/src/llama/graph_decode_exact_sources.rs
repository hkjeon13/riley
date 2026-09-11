//! Exact opaque-region binding for projected V1 pure-decode metadata.
//!
//! This C07-9 value boundary joins C07-8's already exact native fields with
//! caller-owned header and control/status bytes only after their cold-layout
//! capacities match. It does not interpret any opaque byte, construct a fixed
//! packer source, write a slab, or make graph execution safe.

use super::graph_decode_exact_projection::PureDecodeGraphV1ExactNativeFields;
use super::graph_decode_layout::PureDecodeGraphMetadataField;

/// One opaque C07 metadata region accepted by the exact V1 source boundary.
///
/// The variants intentionally exclude all seven native fields, whose exact
/// capacity was already proven by C07-8.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactOpaqueField {
    /// Versioned fixed header bytes.
    Header,
    /// Versioned fixed control and status bytes.
    ControlStatus,
}

impl PureDecodeGraphV1ExactOpaqueField {
    const fn layout_field(self) -> PureDecodeGraphMetadataField {
        match self {
            Self::Header => PureDecodeGraphMetadataField::Header,
            Self::ControlStatus => PureDecodeGraphMetadataField::ControlStatus,
        }
    }
}

/// Closed rejection while pairing exact V1 native fields with opaque bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactOpaqueSourceError {
    /// A caller-owned opaque byte slice did not fill its exact cold region.
    FieldLengthMismatch {
        /// Opaque C07 region whose byte length differed.
        field: PureDecodeGraphV1ExactOpaqueField,
        /// Exact cold byte capacity declared by the bound layout.
        expected: u64,
        /// Caller-owned byte length on this host.
        actual: usize,
    },
}

/// Result of binding caller-owned opaque bytes to exact V1 native fields.
pub(crate) type PureDecodeGraphV1ExactOpaqueSourceResult<T> =
    Result<T, PureDecodeGraphV1ExactOpaqueSourceError>;

/// Borrowed exact V1 native fields and separately borrowed opaque C07 bytes.
///
/// The V1 and opaque inputs deliberately retain independent lifetimes. Success
/// proves only both opaque slice lengths; it does not define their byte
/// semantics, padding sentinels, ownership, address stability, or any write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactMetadataSources<'metadata, 'opaque> {
    native: PureDecodeGraphV1ExactNativeFields<'metadata>,
    header: &'opaque [u8],
    control_status: &'opaque [u8],
}

impl<'metadata, 'opaque> PureDecodeGraphV1ExactMetadataSources<'metadata, 'opaque> {
    /// Binds exact native V1 fields to two exact-length opaque cold regions.
    ///
    /// The header is checked before control/status so a call with both lengths
    /// wrong has a stable, closed first failure. No opaque byte is interpreted.
    pub(crate) fn try_new(
        native: &PureDecodeGraphV1ExactNativeFields<'metadata>,
        header: &'opaque [u8],
        control_status: &'opaque [u8],
    ) -> PureDecodeGraphV1ExactOpaqueSourceResult<Self> {
        validate_opaque_field_length(native, PureDecodeGraphV1ExactOpaqueField::Header, header)?;
        validate_opaque_field_length(
            native,
            PureDecodeGraphV1ExactOpaqueField::ControlStatus,
            control_status,
        )?;
        Ok(Self {
            native: *native,
            header,
            control_status,
        })
    }

    /// Returns the C07-8 native fields whose exact capacities were preserved.
    pub(crate) const fn native_fields(self) -> PureDecodeGraphV1ExactNativeFields<'metadata> {
        self.native
    }

    /// Returns caller-owned header bytes with the exact cold capacity.
    #[must_use]
    pub(crate) const fn header(self) -> &'opaque [u8] {
        self.header
    }

    /// Returns caller-owned control/status bytes with the exact cold capacity.
    #[must_use]
    pub(crate) const fn control_status(self) -> &'opaque [u8] {
        self.control_status
    }
}

fn validate_opaque_field_length(
    native: &PureDecodeGraphV1ExactNativeFields<'_>,
    field: PureDecodeGraphV1ExactOpaqueField,
    bytes: &[u8],
) -> PureDecodeGraphV1ExactOpaqueSourceResult<()> {
    let expected = native
        .binding()
        .layout()
        .region(field.layout_field())
        .byte_len();
    let actual = bytes.len();
    if u64::try_from(actual).ok() != Some(expected) {
        return Err(
            PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                field,
                expected,
                actual,
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1ExactMetadataSources, PureDecodeGraphV1ExactOpaqueField,
        PureDecodeGraphV1ExactOpaqueSourceError,
    };
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph_decode_exact_projection::{
        PureDecodeGraphV1ExactNativeFields, PureDecodeGraphV1ExactProjection,
        project_pure_decode_graph_v1_exact,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
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
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(1, 1, 3, 5))
            .expect("fixture cold layout must be valid")
    }

    fn one_row_config() -> LlamaBatchMetadataConfig {
        LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)
            .expect("one-token exact decode fixture must fit cold bounds")
    }

    fn pack_one_exact_decode(
        prepared: &mut PreparedLlamaBatchMetadata,
    ) -> LlamaPackedBatchMetadata<'_> {
        let token_ids = [7];
        let physical_block_ids = [0];
        let valid_tokens = [1];
        let rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Decode,
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
            .expect("one-token decode row must base-validate")
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

    #[test]
    fn exact_native_fields_bind_arbitrary_exact_length_opaque_bytes_without_mutation() {
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(one_row_config()).expect("fixture must prepare");
        let metadata = pack_one_exact_decode(&mut prepared);
        let before = metadata_snapshot(&metadata);
        let header = [0xa0, 0xa1, 0xa2];
        let control_status = [0xc0, 0xc1, 0xc2, 0xc3, 0xc4];
        let header_before = header;
        let control_status_before = control_status;
        let native = exact_native(&metadata);

        let sources =
            PureDecodeGraphV1ExactMetadataSources::try_new(&native, &header, &control_status)
                .expect("exact opaque regions must bind");

        assert_eq!(sources.native_fields(), native);
        assert_eq!(sources.header(), &header);
        assert_eq!(sources.control_status(), &control_status);
        assert_eq!(metadata_snapshot(&metadata), before);
        assert_eq!(header, header_before);
        assert_eq!(control_status, control_status_before);
    }

    #[test]
    fn header_length_mismatches_fail_closed_before_control_status() {
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(one_row_config()).expect("fixture must prepare");
        let metadata = pack_one_exact_decode(&mut prepared);
        let before = metadata_snapshot(&metadata);
        let native = exact_native(&metadata);

        assert_eq!(
            PureDecodeGraphV1ExactMetadataSources::try_new(&native, &[0; 2], &[0; 5]),
            Err(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::Header,
                    expected: 3,
                    actual: 2,
                }
            )
        );
        assert_eq!(
            PureDecodeGraphV1ExactMetadataSources::try_new(&native, &[0; 4], &[0; 4]),
            Err(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::Header,
                    expected: 3,
                    actual: 4,
                }
            )
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }

    #[test]
    fn control_status_length_mismatches_fail_after_an_exact_header() {
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(one_row_config()).expect("fixture must prepare");
        let metadata = pack_one_exact_decode(&mut prepared);
        let before = metadata_snapshot(&metadata);
        let native = exact_native(&metadata);

        assert_eq!(
            PureDecodeGraphV1ExactMetadataSources::try_new(&native, &[0; 3], &[0; 4]),
            Err(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::ControlStatus,
                    expected: 5,
                    actual: 4,
                }
            )
        );
        assert_eq!(
            PureDecodeGraphV1ExactMetadataSources::try_new(&native, &[0; 3], &[0; 6]),
            Err(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::ControlStatus,
                    expected: 5,
                    actual: 6,
                }
            )
        );
        assert_eq!(metadata_snapshot(&metadata), before);
    }
}
