//! Fixed-offset metadata slab writing for a validated C07 binding.
//!
//! This C07-5 primitive copies already prepared fixed-length values into a
//! caller-owned byte slice. It does not adapt the current batch ABI, choose
//! placeholder values, allocate memory, or invoke graph execution.

use super::graph_decode_binding::PureDecodeGraphMetadataBinding;
use super::graph_decode_layout::{PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout};

const U16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const FIELD_COUNT: usize = PureDecodeGraphMetadataField::ALL.len();

/// Borrowed values for every region in a fixed pure-decode metadata layout.
///
/// Every slice must already include exactly the layout's fixed capacity. In
/// particular, values for trailing placeholder lanes are caller-owned inputs;
/// this type does not define their sentinel or kernel-mask semantics.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PureDecodeGraphMetadataFieldSources<'a> {
    header: &'a [u8],
    token_ids: &'a [u32],
    position_ids: &'a [u32],
    row_sequence_slots: &'a [u32],
    sequence_block_offsets: &'a [u32],
    physical_block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    output_token_indices: &'a [u32],
    control_status: &'a [u8],
}

impl<'a> PureDecodeGraphMetadataFieldSources<'a> {
    /// Creates borrowed values for one fully populated fixed metadata slab.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // The canonical fixed field order is intentionally explicit.
    pub(crate) const fn new(
        header: &'a [u8],
        token_ids: &'a [u32],
        position_ids: &'a [u32],
        row_sequence_slots: &'a [u32],
        sequence_block_offsets: &'a [u32],
        physical_block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        output_token_indices: &'a [u32],
        control_status: &'a [u8],
    ) -> Self {
        Self {
            header,
            token_ids,
            position_ids,
            row_sequence_slots,
            sequence_block_offsets,
            physical_block_ids,
            valid_tokens,
            output_token_indices,
            control_status,
        }
    }
}

/// Closed failure while validating or writing a fixed metadata slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphMetadataPackError {
    /// A fixed layout quantity cannot be represented by this host's slice API.
    LayoutLengthNotAddressable {
        /// Layout byte length or offset that did not fit the host representation.
        value: u64,
    },
    /// The caller-owned destination is smaller than the required fixed slab.
    DestinationTooSmall {
        /// Exact fixed byte length declared by the layout.
        required: u64,
        /// Available destination byte length on this host.
        available: usize,
    },
    /// One source slice does not have the exact fixed field element count.
    FieldByteLengthMismatch {
        /// Fixed-layout field whose source length differed.
        field: PureDecodeGraphMetadataField,
        /// Exact fixed byte length declared by the layout.
        expected: u64,
        /// Actual byte length derived from the source slice.
        actual: u64,
    },
    /// One source slice's byte length overflowed or did not fit the wire width.
    SourceByteLengthOverflow {
        /// Fixed-layout field whose source length could not be represented.
        field: PureDecodeGraphMetadataField,
    },
    /// A layout region is not divisible by its declared wire element width.
    FieldByteLengthMisaligned {
        /// Fixed-layout field with an invalid byte length.
        field: PureDecodeGraphMetadataField,
        /// Declared region byte length.
        byte_len: u64,
    },
}

/// Result of fixed pure-decode metadata slab writing.
pub(crate) type PureDecodeGraphMetadataPackResult<T> = Result<T, PureDecodeGraphMetadataPackError>;

/// Host-addressable region bounds validated before a destination is changed.
#[derive(Clone, Copy)]
struct ValidatedPackLayout {
    slab_len: usize,
    bounds: [(usize, usize); FIELD_COUNT],
}

impl ValidatedPackLayout {
    const fn bounds(self, field: PureDecodeGraphMetadataField) -> (usize, usize) {
        self.bounds[field_index(field)]
    }
}

/// Writes a fully populated fixed metadata slab in canonical little-endian form.
///
/// All destination and source lengths are checked before any destination byte is
/// changed. On success, only the required slab prefix is zeroed first,
/// including alignment gaps, then each fixed field is copied into its declared
/// region. Any caller-owned destination tail remains unchanged.
pub(crate) fn pack_pure_decode_graph_metadata_le(
    binding: &PureDecodeGraphMetadataBinding,
    source: PureDecodeGraphMetadataFieldSources<'_>,
    destination: &mut [u8],
) -> PureDecodeGraphMetadataPackResult<()> {
    let layout = binding.layout();
    let validated_layout = validate_pack_layout(&layout, destination)?;
    validate_source(&layout, source)?;

    let destination = &mut destination[..validated_layout.slab_len];
    destination.fill(0);
    write_bytes(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::Header),
        source.header,
    );
    write_u32s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::TokenIds),
        source.token_ids,
    );
    write_u32s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::PositionIds),
        source.position_ids,
    );
    write_u32s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::RowSequenceSlots),
        source.row_sequence_slots,
    );
    write_u32s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::SequenceBlockOffsets),
        source.sequence_block_offsets,
    );
    write_u32s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::PhysicalBlockIds),
        source.physical_block_ids,
    );
    write_u16s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::ValidTokens),
        source.valid_tokens,
    );
    write_u32s(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::OutputTokenIndices),
        source.output_token_indices,
    );
    write_bytes(
        destination,
        validated_layout.bounds(PureDecodeGraphMetadataField::ControlStatus),
        source.control_status,
    );
    Ok(())
}

fn validate_pack_layout(
    layout: &PureDecodeGraphMetadataLayout,
    destination: &[u8],
) -> PureDecodeGraphMetadataPackResult<ValidatedPackLayout> {
    let slab_len = usize::try_from(layout.total_bytes()).map_err(|_| {
        PureDecodeGraphMetadataPackError::LayoutLengthNotAddressable {
            value: layout.total_bytes(),
        }
    })?;
    if destination.len() < slab_len {
        return Err(PureDecodeGraphMetadataPackError::DestinationTooSmall {
            required: layout.total_bytes(),
            available: destination.len(),
        });
    }
    let mut bounds = [(0, 0); FIELD_COUNT];
    for field in PureDecodeGraphMetadataField::ALL {
        bounds[field_index(field)] = checked_region_bounds(layout, field, slab_len)?;
    }
    Ok(ValidatedPackLayout { slab_len, bounds })
}

fn validate_source(
    layout: &PureDecodeGraphMetadataLayout,
    source: PureDecodeGraphMetadataFieldSources<'_>,
) -> PureDecodeGraphMetadataPackResult<()> {
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::Header,
        source.header.len(),
        1,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::TokenIds,
        source.token_ids.len(),
        U32_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::PositionIds,
        source.position_ids.len(),
        U32_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::RowSequenceSlots,
        source.row_sequence_slots.len(),
        U32_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::SequenceBlockOffsets,
        source.sequence_block_offsets.len(),
        U32_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::PhysicalBlockIds,
        source.physical_block_ids.len(),
        U32_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::ValidTokens,
        source.valid_tokens.len(),
        U16_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::OutputTokenIndices,
        source.output_token_indices.len(),
        U32_BYTES,
    )?;
    validate_field_length(
        layout,
        PureDecodeGraphMetadataField::ControlStatus,
        source.control_status.len(),
        1,
    )
}

fn validate_field_length(
    layout: &PureDecodeGraphMetadataLayout,
    field: PureDecodeGraphMetadataField,
    actual: usize,
    element_bytes: usize,
) -> PureDecodeGraphMetadataPackResult<()> {
    let byte_len = layout.region(field).byte_len();
    let element_bytes_u64 = u64::try_from(element_bytes)
        .map_err(|_| PureDecodeGraphMetadataPackError::SourceByteLengthOverflow { field })?;
    if byte_len % element_bytes_u64 != 0 {
        return Err(
            PureDecodeGraphMetadataPackError::FieldByteLengthMisaligned { field, byte_len },
        );
    }
    let actual = actual
        .checked_mul(element_bytes)
        .ok_or(PureDecodeGraphMetadataPackError::SourceByteLengthOverflow { field })?;
    let actual = u64::try_from(actual)
        .map_err(|_| PureDecodeGraphMetadataPackError::SourceByteLengthOverflow { field })?;
    if actual != byte_len {
        return Err(PureDecodeGraphMetadataPackError::FieldByteLengthMismatch {
            field,
            expected: byte_len,
            actual,
        });
    }
    Ok(())
}

const fn field_index(field: PureDecodeGraphMetadataField) -> usize {
    match field {
        PureDecodeGraphMetadataField::Header => 0,
        PureDecodeGraphMetadataField::TokenIds => 1,
        PureDecodeGraphMetadataField::PositionIds => 2,
        PureDecodeGraphMetadataField::RowSequenceSlots => 3,
        PureDecodeGraphMetadataField::SequenceBlockOffsets => 4,
        PureDecodeGraphMetadataField::PhysicalBlockIds => 5,
        PureDecodeGraphMetadataField::ValidTokens => 6,
        PureDecodeGraphMetadataField::OutputTokenIndices => 7,
        PureDecodeGraphMetadataField::ControlStatus => 8,
    }
}

fn write_bytes(destination: &mut [u8], bounds: (usize, usize), values: &[u8]) {
    let (start, end) = bounds;
    destination[start..end].copy_from_slice(values);
}

fn write_u32s(destination: &mut [u8], bounds: (usize, usize), values: &[u32]) {
    let (start, end) = bounds;
    for (bytes, value) in destination[start..end]
        .chunks_exact_mut(U32_BYTES)
        .zip(values)
    {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u16s(destination: &mut [u8], bounds: (usize, usize), values: &[u16]) {
    let (start, end) = bounds;
    for (bytes, value) in destination[start..end]
        .chunks_exact_mut(U16_BYTES)
        .zip(values)
    {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn checked_region_bounds(
    layout: &PureDecodeGraphMetadataLayout,
    field: PureDecodeGraphMetadataField,
    slab_len: usize,
) -> PureDecodeGraphMetadataPackResult<(usize, usize)> {
    let region = layout.region(field);
    let offset = usize::try_from(region.offset()).map_err(|_| {
        PureDecodeGraphMetadataPackError::LayoutLengthNotAddressable {
            value: region.offset(),
        }
    })?;
    let byte_len = usize::try_from(region.byte_len()).map_err(|_| {
        PureDecodeGraphMetadataPackError::LayoutLengthNotAddressable {
            value: region.byte_len(),
        }
    })?;
    let end = offset.checked_add(byte_len).ok_or(
        PureDecodeGraphMetadataPackError::LayoutLengthNotAddressable {
            value: layout.total_bytes(),
        },
    )?;
    if end > slab_len {
        return Err(
            PureDecodeGraphMetadataPackError::LayoutLengthNotAddressable {
                value: layout.total_bytes(),
            },
        );
    }
    Ok((offset, end))
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphMetadataFieldSources, PureDecodeGraphMetadataPackError,
        pack_pure_decode_graph_metadata_le,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBinding;
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout,
        PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;

    fn binding() -> PureDecodeGraphMetadataBinding {
        let layout = PureDecodeGraphMetadataLayout::try_new(
            PureDecodeGraphMetadataLayoutSpec::new(4, 6, 3, 3),
        )
        .expect("valid M4 B6 layout");
        let padding = plan_pure_decode_graph_padding(3).expect("A3 selects M4");
        PureDecodeGraphMetadataBinding::try_new(layout, padding).expect("matching M4 binding")
    }

    #[allow(clippy::too_many_arguments)] // Mirrors the explicit fixed-field source constructor.
    fn source<'a>(
        header: &'a [u8],
        token_ids: &'a [u32],
        position_ids: &'a [u32],
        row_sequence_slots: &'a [u32],
        sequence_block_offsets: &'a [u32],
        physical_block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        output_token_indices: &'a [u32],
        control_status: &'a [u8],
    ) -> PureDecodeGraphMetadataFieldSources<'a> {
        PureDecodeGraphMetadataFieldSources::new(
            header,
            token_ids,
            position_ids,
            row_sequence_slots,
            sequence_block_offsets,
            physical_block_ids,
            valid_tokens,
            output_token_indices,
            control_status,
        )
    }

    const EXACT_SOURCE_LENGTHS: [usize; 9] = [3, 4, 4, 4, 5, 6, 6, 4, 3];
    static BYTE_VALUES: [u8; 4] = [0x10, 0x20, 0x30, 0x40];
    static U32_VALUES: [u32; 7] = [
        0x0102_0304,
        0x1112_1314,
        0x2122_2324,
        0x3132_3334,
        0x4142_4344,
        0x5152_5354,
        0x6162_6364,
    ];
    static U16_VALUES: [u16; 7] = [0x0102, 0x1112, 0x2122, 0x3132, 0x4142, 0x5152, 0x6162];

    fn source_with_lengths(lengths: [usize; 9]) -> PureDecodeGraphMetadataFieldSources<'static> {
        source(
            &BYTE_VALUES[..lengths[0]],
            &U32_VALUES[..lengths[1]],
            &U32_VALUES[..lengths[2]],
            &U32_VALUES[..lengths[3]],
            &U32_VALUES[..lengths[4]],
            &U32_VALUES[..lengths[5]],
            &U16_VALUES[..lengths[6]],
            &U32_VALUES[..lengths[7]],
            &BYTE_VALUES[..lengths[8]],
        )
    }

    const fn field_element_bytes(field: PureDecodeGraphMetadataField) -> u64 {
        match field {
            PureDecodeGraphMetadataField::Header | PureDecodeGraphMetadataField::ControlStatus => 1,
            PureDecodeGraphMetadataField::ValidTokens => 2,
            PureDecodeGraphMetadataField::TokenIds
            | PureDecodeGraphMetadataField::PositionIds
            | PureDecodeGraphMetadataField::RowSequenceSlots
            | PureDecodeGraphMetadataField::SequenceBlockOffsets
            | PureDecodeGraphMetadataField::PhysicalBlockIds
            | PureDecodeGraphMetadataField::OutputTokenIndices => 4,
        }
    }

    #[test]
    fn writes_fixed_fields_little_endian_and_zeroes_alignment_gaps() {
        let header = [0xa0, 0xa1, 0xa2];
        let token_ids = [10, 20, 30, 40];
        let position_ids = [100, 101, 102, 103];
        let row_sequence_slots = [7, 6, 5, 4];
        let sequence_block_offsets = [0, 2, 3, 5, 5];
        let physical_block_ids = [11, 12, 13, 14, 15, 16];
        let valid_tokens = [31, 32, 33, 34, 35, 36];
        let output_token_indices = [3, 2, 1, 0];
        let control_status = [0xc0, 0xc1, 0xc2];
        let mut destination = [0xee; 129];

        pack_pure_decode_graph_metadata_le(
            &binding(),
            source(
                &header,
                &token_ids,
                &position_ids,
                &row_sequence_slots,
                &sequence_block_offsets,
                &physical_block_ids,
                &valid_tokens,
                &output_token_indices,
                &control_status,
            ),
            &mut destination,
        )
        .expect("exact fixed source and slab must write");

        assert_eq!(&destination[0..3], &header);
        assert_eq!(destination[3], 0);
        assert_eq!(&destination[4..8], &token_ids[0].to_le_bytes());
        assert_eq!(&destination[16..20], &token_ids[3].to_le_bytes());
        assert_eq!(
            &destination[52..56],
            &sequence_block_offsets[0].to_le_bytes()
        );
        assert_eq!(&destination[72..76], &physical_block_ids[0].to_le_bytes());
        assert_eq!(&destination[96..98], &valid_tokens[0].to_le_bytes());
        assert_eq!(
            &destination[108..112],
            &output_token_indices[0].to_le_bytes()
        );
        assert_eq!(&destination[124..127], &control_status);
        assert_eq!(destination[127], 0);
        assert_eq!(destination[128], 0xee);
    }

    #[test]
    fn rejects_every_short_or_long_source_field_before_mutating_destination() {
        let layout = binding().layout();
        for (index, field) in PureDecodeGraphMetadataField::ALL.into_iter().enumerate() {
            let expected = layout.region(field).byte_len();
            let element_bytes = field_element_bytes(field);
            for actual_length in [
                EXACT_SOURCE_LENGTHS[index] - 1,
                EXACT_SOURCE_LENGTHS[index] + 1,
            ] {
                let mut lengths = EXACT_SOURCE_LENGTHS;
                lengths[index] = actual_length;
                let mut destination = [0xa5; 129];
                let actual = u64::try_from(actual_length).expect("test source length must fit u64")
                    * element_bytes;

                assert_eq!(
                    pack_pure_decode_graph_metadata_le(
                        &binding(),
                        source_with_lengths(lengths),
                        &mut destination,
                    ),
                    Err(PureDecodeGraphMetadataPackError::FieldByteLengthMismatch {
                        field,
                        expected,
                        actual,
                    })
                );
                assert_eq!(destination, [0xa5; 129]);
            }
        }
    }

    #[test]
    fn copies_exact_sized_values_without_interpreting_metadata_semantics() {
        let header = [0xfe, 0xed, 0xfa];
        let token_ids = [u32::MAX, 0, 42, 0xdead_beef];
        let position_ids = [u32::MAX, 1, 0, 99];
        let row_sequence_slots = [9, 0, u32::MAX, 1];
        let sequence_block_offsets = [u32::MAX, 7, 0, 99, 1];
        let physical_block_ids = [u32::MAX, 0, 77, 2, 1, 99];
        let valid_tokens = [u16::MAX, 0, 17, 3, 1, 99];
        let output_token_indices = [u32::MAX, 0, 4, 2];
        let control_status = [0xba, 0xad, 0xf0];
        let mut destination = [0x5a; 128];

        pack_pure_decode_graph_metadata_le(
            &binding(),
            source(
                &header,
                &token_ids,
                &position_ids,
                &row_sequence_slots,
                &sequence_block_offsets,
                &physical_block_ids,
                &valid_tokens,
                &output_token_indices,
                &control_status,
            ),
            &mut destination,
        )
        .expect("exact fixed field widths are the sole C07-5 acceptance rule");

        assert_eq!(&destination[16..20], &token_ids[3].to_le_bytes());
        assert_eq!(
            &destination[52..56],
            &sequence_block_offsets[0].to_le_bytes()
        );
        assert_eq!(&destination[72..76], &physical_block_ids[0].to_le_bytes());
        assert_eq!(&destination[96..98], &valid_tokens[0].to_le_bytes());
        assert_eq!(
            &destination[108..112],
            &output_token_indices[0].to_le_bytes()
        );
    }

    #[test]
    fn rejects_wrong_destination_length_before_source_mutation() {
        let header = [0xa0, 0xa1, 0xa2];
        let token_ids = [10, 20, 30, 40];
        let position_ids = [100, 101, 102, 103];
        let row_sequence_slots = [7, 6, 5, 4];
        let sequence_block_offsets = [0, 2, 3, 5, 5];
        let physical_block_ids = [11, 12, 13, 14, 15, 16];
        let valid_tokens = [31, 32, 33, 34, 35, 36];
        let output_token_indices = [3, 2, 1, 0];
        let control_status = [0xc0, 0xc1, 0xc2];
        let mut destination = [0xa5; 128];

        assert_eq!(
            pack_pure_decode_graph_metadata_le(
                &binding(),
                source(
                    &header,
                    &token_ids,
                    &position_ids,
                    &row_sequence_slots,
                    &sequence_block_offsets,
                    &physical_block_ids,
                    &valid_tokens,
                    &output_token_indices,
                    &control_status,
                ),
                &mut destination[..127],
            ),
            Err(PureDecodeGraphMetadataPackError::DestinationTooSmall {
                required: 128,
                available: 127,
            })
        );
        assert_eq!(destination, [0xa5; 128]);
    }
}
