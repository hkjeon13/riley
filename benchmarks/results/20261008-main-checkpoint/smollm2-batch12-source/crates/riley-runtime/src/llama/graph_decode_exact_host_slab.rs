//! Cold-owned, aligned host storage for one exact V1 C07 metadata slab.
//!
//! C07-12 creates the persistent host allocation that C07-11 deliberately
//! leaves caller-owned. The exposed payload is exactly one cold layout and is
//! aligned at runtime without relying on `Box<[u8]>`'s element alignment.
//! Writes reuse that allocation through C07-11; this module neither transfers
//! the bytes nor creates a graph execution path.

use std::error;
use std::fmt;

use super::batch::LlamaPackedBatchMetadata;
use super::graph_decode_exact_projection::PureDecodeGraphV1ExactProjectionIneligibility;
use super::graph_decode_exact_slab_writer::{
    PureDecodeGraphV1ExactSlabWrite, PureDecodeGraphV1ExactSlabWriteError,
    PureDecodeGraphV1ExactSlabWriteResult, write_pure_decode_graph_v1_exact_metadata_le,
};
use super::graph_decode_layout::{
    PureDecodeGraphMetadataGeometryDigest, PureDecodeGraphMetadataLayout,
};

/// Closed failure while cold-preparing one exact V1 C07 host slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactHostSlabPrepareError {
    /// The layout's exact payload length cannot be addressed on this host.
    LayoutLengthNotAddressable {
        /// Fixed C07 payload bytes declared by the cold layout.
        total_bytes: u64,
    },
    /// The layout's base-alignment value cannot be represented on this host.
    RequiredBaseAlignmentNotAddressable {
        /// Fixed C07 base alignment declared by the cold layout.
        required_alignment: u64,
    },
    /// The fixed layout supplied an invalid host base alignment.
    InvalidBaseAlignment {
        /// Fixed C07 base alignment after host-width conversion.
        required_alignment: usize,
    },
    /// Adding enough bytes to align an interior payload overflowed.
    BackingLengthOverflow {
        /// Fixed C07 payload bytes declared by the cold layout.
        total_bytes: u64,
        /// Fixed C07 base alignment declared by the cold layout.
        required_alignment: u64,
    },
    /// The aligned backing allocation length cannot be addressed on this host.
    BackingLengthNotAddressable {
        /// Exact backing bytes requested for the cold allocation.
        requested_bytes: u64,
    },
    /// The host could not allocate the one cold backing byte allocation.
    HostAllocation {
        /// Exact backing bytes requested for the cold allocation.
        requested_bytes: u64,
    },
    /// No aligned payload range fit inside the freshly allocated backing bytes.
    CannotAlignStorage {
        /// Fixed C07 base alignment after host-width conversion.
        required_alignment: usize,
    },
}

impl fmt::Display for PureDecodeGraphV1ExactHostSlabPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutLengthNotAddressable { total_bytes } => {
                write!(
                    formatter,
                    "exact C07 host-slab payload {total_bytes} bytes is not addressable"
                )
            }
            Self::RequiredBaseAlignmentNotAddressable { required_alignment } => write!(
                formatter,
                "exact C07 host-slab base alignment {required_alignment} is not addressable"
            ),
            Self::InvalidBaseAlignment { required_alignment } => write!(
                formatter,
                "exact C07 host-slab base alignment {required_alignment} is not a nonzero power of two"
            ),
            Self::BackingLengthOverflow {
                total_bytes,
                required_alignment,
            } => write!(
                formatter,
                "exact C07 host-slab backing length overflowed for {total_bytes} payload bytes and {required_alignment}-byte alignment"
            ),
            Self::BackingLengthNotAddressable { requested_bytes } => write!(
                formatter,
                "exact C07 host-slab backing {requested_bytes} bytes is not addressable"
            ),
            Self::HostAllocation { requested_bytes } => write!(
                formatter,
                "could not allocate {requested_bytes} exact C07 host-slab backing bytes"
            ),
            Self::CannotAlignStorage { required_alignment } => write!(
                formatter,
                "could not select a {required_alignment}-byte-aligned exact C07 host-slab payload"
            ),
        }
    }
}

impl error::Error for PureDecodeGraphV1ExactHostSlabPrepareError {}

/// Result of cold preparation for one exact V1 C07 host slab.
pub(crate) type PureDecodeGraphV1ExactHostSlabPrepareResult<T> =
    Result<T, PureDecodeGraphV1ExactHostSlabPrepareError>;

/// Read-only proof of one just-written exact V1 host-slab payload.
///
/// The lease carries the payload with the same cold layout and geometry digest
/// that own it. Its borrow prevents another write, owner move, or owner drop
/// until the lease ends. It proves only a successful host write; it does not
/// establish opaque-byte semantics, pinned storage, transfer completion,
/// device contents, graph readiness, or execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactHostSlabLease<'slab> {
    layout: &'slab PureDecodeGraphMetadataLayout,
    geometry_digest: &'slab PureDecodeGraphMetadataGeometryDigest,
    bytes: &'slab [u8],
}

impl<'slab> PureDecodeGraphV1ExactHostSlabLease<'slab> {
    /// Returns the exact cold layout that owns these borrowed bytes.
    #[must_use]
    pub(crate) const fn layout(self) -> PureDecodeGraphMetadataLayout {
        *self.layout
    }

    /// Returns the geometry identity stored with the originating cold owner.
    #[must_use]
    pub(crate) const fn geometry_digest(self) -> PureDecodeGraphMetadataGeometryDigest {
        *self.geometry_digest
    }

    /// Returns the exact aligned payload written by the originating owner.
    #[must_use]
    pub(crate) const fn bytes(self) -> &'slab [u8] {
        self.bytes
    }
}

/// Closed result of an exact V1 write that may yield a read-only host lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
#[allow(clippy::large_enum_variant)] // The closed no-allocation write result retains its lease.
pub(crate) enum PureDecodeGraphV1ExactHostSlabWrite<'slab> {
    /// C07-11 wrote the exact payload and it is now borrowed read-only.
    Written(PureDecodeGraphV1ExactHostSlabLease<'slab>),
    /// C07-11 rejected the V1 batch before changing this host payload.
    Ineligible(PureDecodeGraphV1ExactProjectionIneligibility),
}

/// Result of an exact V1 write that may yield a read-only host lease.
pub(crate) type PureDecodeGraphV1ExactHostSlabWriteResult<'slab> =
    Result<PureDecodeGraphV1ExactHostSlabWrite<'slab>, PureDecodeGraphV1ExactSlabWriteError>;

/// One cold-owned, fixed-address host payload for an exact V1 C07 layout.
///
/// The private backing allocation contains up to `alignment - 1` extra bytes;
/// the exposed interior payload is exactly `layout.total_bytes()` and begins at
/// a checked layout-aligned address. Moving this owner moves only the `Box`
/// handle, never its allocation, and no public API can resize or replace that
/// allocation. The payload address remains stable until this owner is dropped.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactHostSlab {
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    backing: Box<[u8]>,
    payload_offset: usize,
    payload_len: usize,
}

impl PureDecodeGraphV1ExactHostSlab {
    /// Cold-prepares one zeroed, layout-aligned, fixed-address host payload.
    ///
    /// `Box<[u8]>` guarantees byte alignment only, so this method reserves an
    /// interior alignment window and verifies the chosen payload range before
    /// retaining the backing allocation. It performs the sole allocation for
    /// this owner; later exact writes do not allocate.
    pub(crate) fn prepare(
        layout: PureDecodeGraphMetadataLayout,
    ) -> PureDecodeGraphV1ExactHostSlabPrepareResult<Self> {
        let total_bytes = layout.total_bytes();
        let payload_len = usize::try_from(total_bytes).map_err(|_| {
            PureDecodeGraphV1ExactHostSlabPrepareError::LayoutLengthNotAddressable { total_bytes }
        })?;
        let required_alignment_u64 = PureDecodeGraphMetadataLayout::required_base_alignment();
        let required_alignment = usize::try_from(required_alignment_u64).map_err(|_| {
            PureDecodeGraphV1ExactHostSlabPrepareError::RequiredBaseAlignmentNotAddressable {
                required_alignment: required_alignment_u64,
            }
        })?;
        if !required_alignment.is_power_of_two() {
            return Err(
                PureDecodeGraphV1ExactHostSlabPrepareError::InvalidBaseAlignment {
                    required_alignment,
                },
            );
        }
        let backing_bytes = total_bytes.checked_add(required_alignment_u64 - 1).ok_or(
            PureDecodeGraphV1ExactHostSlabPrepareError::BackingLengthOverflow {
                total_bytes,
                required_alignment: required_alignment_u64,
            },
        )?;
        let backing_len = usize::try_from(backing_bytes).map_err(|_| {
            PureDecodeGraphV1ExactHostSlabPrepareError::BackingLengthNotAddressable {
                requested_bytes: backing_bytes,
            }
        })?;
        let mut backing = Vec::new();
        backing.try_reserve_exact(backing_len).map_err(|_| {
            PureDecodeGraphV1ExactHostSlabPrepareError::HostAllocation {
                requested_bytes: backing_bytes,
            }
        })?;
        backing.resize(backing_len, 0);
        let backing = backing.into_boxed_slice();
        let payload_offset = backing.as_ptr().align_offset(required_alignment);
        let payload_end = payload_offset
            .checked_add(payload_len)
            .filter(|payload_end| *payload_end <= backing.len())
            .ok_or(
                PureDecodeGraphV1ExactHostSlabPrepareError::CannotAlignStorage {
                    required_alignment,
                },
            )?;
        debug_assert!(payload_end <= backing.len());
        Ok(Self {
            layout,
            geometry_digest: layout.geometry_digest(),
            backing,
            payload_offset,
            payload_len,
        })
    }

    /// Returns the immutable exact cold layout owned by this payload.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        self.layout
    }

    /// Returns the cold geometry identity stored with this fixed payload.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        self.geometry_digest
    }

    /// Returns the exact aligned payload whose address stays stable while owned.
    #[must_use]
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.backing[self.payload_offset..self.payload_offset + self.payload_len]
    }

    /// Reuses this cold payload for one checked exact V1 metadata write.
    ///
    /// This is a single direct C07-11 call with the owner's stored layout. Its
    /// result and error identity, ordering, and no-mutation behavior are
    /// preserved; this owner adds neither validation nor allocation on writes.
    pub(crate) fn write_exact_v1(
        &mut self,
        metadata: &LlamaPackedBatchMetadata<'_>,
        header: &[u8],
        control_status: &[u8],
    ) -> PureDecodeGraphV1ExactSlabWriteResult<PureDecodeGraphV1ExactSlabWrite> {
        write_pure_decode_graph_v1_exact_metadata_le(
            metadata,
            self.layout,
            header,
            control_status,
            &mut self.backing[self.payload_offset..self.payload_offset + self.payload_len],
        )
    }

    /// Writes exact V1 metadata and borrows the successful payload read-only.
    ///
    /// This calls the existing C07-12 write boundary exactly once. A successful
    /// write becomes a lease of this same owner's layout, digest, and exact
    /// payload. Every C07-11 ineligible or error result keeps its original
    /// identity and precedence without creating a lease.
    pub(crate) fn write_exact_v1_leased<'slab>(
        &'slab mut self,
        metadata: &LlamaPackedBatchMetadata<'_>,
        header: &[u8],
        control_status: &[u8],
    ) -> PureDecodeGraphV1ExactHostSlabWriteResult<'slab> {
        match self.write_exact_v1(metadata, header, control_status)? {
            PureDecodeGraphV1ExactSlabWrite::Written => Ok(
                PureDecodeGraphV1ExactHostSlabWrite::Written(self.successful_write_lease()),
            ),
            PureDecodeGraphV1ExactSlabWrite::Ineligible(reason) => {
                Ok(PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason))
            }
        }
    }

    fn successful_write_lease(&self) -> PureDecodeGraphV1ExactHostSlabLease<'_> {
        PureDecodeGraphV1ExactHostSlabLease {
            layout: &self.layout,
            geometry_digest: &self.geometry_digest,
            bytes: self.bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1ExactHostSlab, PureDecodeGraphV1ExactHostSlabLease,
        PureDecodeGraphV1ExactHostSlabPrepareError, PureDecodeGraphV1ExactHostSlabWrite,
    };
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph_decode_exact_projection::PureDecodeGraphV1ExactProjectionIneligibility;
    use crate::llama::graph_decode_exact_slab_writer::{
        PureDecodeGraphV1ExactSlabWrite, PureDecodeGraphV1ExactSlabWriteError,
    };
    use crate::llama::graph_decode_exact_sources::{
        PureDecodeGraphV1ExactOpaqueField, PureDecodeGraphV1ExactOpaqueSourceError,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataField, PureDecodeGraphMetadataLayout,
        PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

    fn exact_layout() -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(2, 2, 3, 5))
            .expect("exact M2/B2 fixture layout must be valid")
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
            .expect("two exact decode rows must base-validate")
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
            .expect("one prefill row must base-validate")
    }

    fn field_bytes(
        bytes: &[u8],
        layout: PureDecodeGraphMetadataLayout,
        field: PureDecodeGraphMetadataField,
    ) -> &[u8] {
        let region = layout.region(field);
        let start = usize::try_from(region.offset()).expect("fixture offset must fit usize");
        let len = usize::try_from(region.byte_len()).expect("fixture field length must fit usize");
        &bytes[start..start + len]
    }

    fn read_u32s(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("fixture u32 bytes")))
            .collect()
    }

    fn read_u16s(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("fixture u16 bytes")))
            .collect()
    }

    #[test]
    fn prepare_exposes_an_exact_zeroed_layout_aligned_payload() {
        let layout = exact_layout();
        let slab = PureDecodeGraphV1ExactHostSlab::prepare(layout)
            .expect("exact fixture host slab must prepare");
        let required_alignment =
            usize::try_from(PureDecodeGraphMetadataLayout::required_base_alignment())
                .expect("fixture alignment fits usize");

        assert_eq!(slab.layout(), layout);
        assert_eq!(slab.geometry_digest(), layout.geometry_digest());
        assert_eq!(
            slab.bytes().len(),
            usize::try_from(layout.total_bytes()).expect("fixture payload length fits usize")
        );
        assert_eq!((slab.bytes().as_ptr() as usize) % required_alignment, 0);
        assert!(slab.bytes().iter().all(|byte| *byte == 0));
    }

    #[test]
    fn exact_write_forwards_canonical_bytes_through_the_cold_owner() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact fixture must fit metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config)
            .expect("exact fixture must prepare metadata");
        let metadata = pack_two_exact_decodes(&mut prepared);
        let layout = exact_layout();
        let mut slab = PureDecodeGraphV1ExactHostSlab::prepare(layout)
            .expect("exact fixture host slab must prepare");
        let header = [0xa0, 0xa1, 0xa2];
        let control_status = [0xc0, 0xc1, 0xc2, 0xc3, 0xc4];

        assert_eq!(
            slab.write_exact_v1(&metadata, &header, &control_status),
            Ok(PureDecodeGraphV1ExactSlabWrite::Written)
        );
        assert_eq!(
            field_bytes(slab.bytes(), layout, PureDecodeGraphMetadataField::Header),
            header
        );
        assert_eq!(
            read_u32s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::TokenIds,
            )),
            metadata.input_token_ids()
        );
        assert_eq!(
            read_u32s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::PositionIds,
            )),
            metadata.position_ids()
        );
        assert_eq!(
            read_u32s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::RowSequenceSlots,
            )),
            metadata.row_sequence_slots()
        );
        assert_eq!(
            read_u32s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::SequenceBlockOffsets,
            )),
            metadata.block_row_offsets()
        );
        assert_eq!(
            read_u32s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::PhysicalBlockIds,
            )),
            metadata.physical_block_ids()
        );
        assert_eq!(
            read_u16s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::ValidTokens,
            )),
            metadata.valid_tokens()
        );
        assert_eq!(
            read_u32s(field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::OutputTokenIndices,
            )),
            metadata.output_token_indices()
        );
        assert_eq!(
            field_bytes(
                slab.bytes(),
                layout,
                PureDecodeGraphMetadataField::ControlStatus,
            ),
            control_status
        );
    }

    #[test]
    fn moving_the_owner_and_reusing_exact_writes_preserves_payload_address() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact fixture must fit metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config)
            .expect("exact fixture must prepare metadata");
        let metadata = pack_two_exact_decodes(&mut prepared);
        let slab = PureDecodeGraphV1ExactHostSlab::prepare(exact_layout())
            .expect("exact fixture host slab must prepare");
        let original_address = slab.bytes().as_ptr() as usize;
        let mut slab = move_owner(slab);

        assert_eq!(slab.bytes().as_ptr() as usize, original_address);
        assert_eq!(
            slab.write_exact_v1(&metadata, &[1, 2, 3], &[4, 5, 6, 7, 8]),
            Ok(PureDecodeGraphV1ExactSlabWrite::Written)
        );
        assert_eq!(slab.bytes().as_ptr() as usize, original_address);
        assert_eq!(
            slab.write_exact_v1(&metadata, &[8, 7, 6], &[5, 4, 3, 2, 1]),
            Ok(PureDecodeGraphV1ExactSlabWrite::Written)
        );
        assert_eq!(slab.bytes().as_ptr() as usize, original_address);
    }

    #[test]
    fn preflight_ineligibility_preserves_c07_11_identity_and_cold_payload() {
        let config = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)
            .expect("prefill fixture must fit metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config)
            .expect("prefill fixture must prepare metadata");
        let metadata = pack_one_prefill(&mut prepared);
        let mut slab = PureDecodeGraphV1ExactHostSlab::prepare(exact_layout())
            .expect("exact fixture host slab must prepare");
        let address = slab.bytes().as_ptr() as usize;
        let before = slab.bytes().to_vec();

        assert_eq!(
            slab.write_exact_v1(&metadata, &[0; 2], &[0; 4]),
            Ok(PureDecodeGraphV1ExactSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(
                    PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                        prefill_rows: 1,
                        prefill_tokens: 1,
                    }
                )
            ))
        );
        assert_eq!(slab.bytes().as_ptr() as usize, address);
        assert_eq!(slab.bytes(), before);
    }

    #[test]
    fn opaque_error_preserves_c07_11_identity_and_cold_payload() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact fixture must fit metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config)
            .expect("exact fixture must prepare metadata");
        let metadata = pack_two_exact_decodes(&mut prepared);
        let mut slab = PureDecodeGraphV1ExactHostSlab::prepare(exact_layout())
            .expect("exact fixture host slab must prepare");
        assert_eq!(
            slab.write_exact_v1(&metadata, &[1, 2, 3], &[4, 5, 6, 7, 8]),
            Ok(PureDecodeGraphV1ExactSlabWrite::Written)
        );
        let address = slab.bytes().as_ptr() as usize;
        let before = slab.bytes().to_vec();

        assert_eq!(
            slab.write_exact_v1(&metadata, &[0; 2], &[0; 4]),
            Err(PureDecodeGraphV1ExactSlabWriteError::OpaqueSource(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::Header,
                    expected: 3,
                    actual: 2,
                }
            ))
        );
        assert_eq!(slab.bytes().as_ptr() as usize, address);
        assert_eq!(slab.bytes(), before);
    }

    #[test]
    fn leased_write_binds_the_just_written_payload_to_its_layout_and_digest() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact fixture must fit metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config)
            .expect("exact fixture must prepare metadata");
        let metadata = pack_two_exact_decodes(&mut prepared);
        let layout = exact_layout();
        let mut slab = PureDecodeGraphV1ExactHostSlab::prepare(layout)
            .expect("exact fixture host slab must prepare");
        let payload_address = slab.bytes().as_ptr() as usize;
        let lease = match slab
            .write_exact_v1_leased(
                &metadata,
                &[0xa0, 0xa1, 0xa2],
                &[0xc0, 0xc1, 0xc2, 0xc3, 0xc4],
            )
            .expect("exact fixture leased write must succeed")
        {
            PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
            PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                panic!("exact fixture unexpectedly ineligible: {reason:?}");
            }
        };

        assert_eq!(lease.layout(), layout);
        assert_eq!(lease.geometry_digest(), layout.geometry_digest());
        assert_eq!(lease.bytes().as_ptr() as usize, payload_address);
        assert_eq!(
            field_bytes(lease.bytes(), layout, PureDecodeGraphMetadataField::Header),
            &[0xa0, 0xa1, 0xa2]
        );
        assert_eq!(
            field_bytes(
                lease.bytes(),
                layout,
                PureDecodeGraphMetadataField::ControlStatus,
            ),
            &[0xc0, 0xc1, 0xc2, 0xc3, 0xc4]
        );
        let moved_lease = move_lease(lease);
        assert_eq!(moved_lease.bytes().as_ptr() as usize, payload_address);
    }

    #[test]
    fn ending_a_lease_allows_a_later_exact_write_at_the_same_payload_address() {
        let config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact fixture must fit metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config)
            .expect("exact fixture must prepare metadata");
        let metadata = pack_two_exact_decodes(&mut prepared);
        let mut slab = PureDecodeGraphV1ExactHostSlab::prepare(exact_layout())
            .expect("exact fixture host slab must prepare");
        let first_address = {
            let lease = match slab
                .write_exact_v1_leased(&metadata, &[1, 2, 3], &[4, 5, 6, 7, 8])
                .expect("first leased write must succeed")
            {
                PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
                PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                    panic!("exact fixture unexpectedly ineligible: {reason:?}");
                }
            };
            lease.bytes().as_ptr() as usize
        };
        let second_lease = match slab
            .write_exact_v1_leased(&metadata, &[8, 7, 6], &[5, 4, 3, 2, 1])
            .expect("second leased write must succeed")
        {
            PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
            PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                panic!("exact fixture unexpectedly ineligible: {reason:?}");
            }
        };

        assert_eq!(second_lease.bytes().as_ptr() as usize, first_address);
        assert_eq!(
            field_bytes(
                second_lease.bytes(),
                exact_layout(),
                PureDecodeGraphMetadataField::Header,
            ),
            &[8, 7, 6]
        );
    }

    #[test]
    fn leased_write_preserves_ineligible_and_opaque_error_identity_without_mutation() {
        let prefill_config = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)
            .expect("prefill fixture must fit metadata bounds");
        let mut prefill_prepared = PreparedLlamaBatchMetadata::prepare(prefill_config)
            .expect("prefill fixture must prepare metadata");
        let prefill = pack_one_prefill(&mut prefill_prepared);
        let mut prefill_slab = PureDecodeGraphV1ExactHostSlab::prepare(exact_layout())
            .expect("exact fixture host slab must prepare");
        let prefill_address = prefill_slab.bytes().as_ptr() as usize;
        let prefill_before = prefill_slab.bytes().to_vec();

        assert_eq!(
            prefill_slab.write_exact_v1_leased(&prefill, &[0; 2], &[0; 4]),
            Ok(PureDecodeGraphV1ExactHostSlabWrite::Ineligible(
                PureDecodeGraphV1ExactProjectionIneligibility::Preflight(
                    PureDecodeGraphV1Ineligibility::PrefillWorkPresent {
                        prefill_rows: 1,
                        prefill_tokens: 1,
                    }
                )
            ))
        );
        assert_eq!(prefill_slab.bytes().as_ptr() as usize, prefill_address);
        assert_eq!(prefill_slab.bytes(), prefill_before);

        let exact_config = LlamaBatchMetadataConfig::new(2, 2, 2, 2, 4)
            .expect("exact fixture must fit metadata bounds");
        let mut exact_prepared = PreparedLlamaBatchMetadata::prepare(exact_config)
            .expect("exact fixture must prepare metadata");
        let exact = pack_two_exact_decodes(&mut exact_prepared);
        let mut exact_slab = PureDecodeGraphV1ExactHostSlab::prepare(exact_layout())
            .expect("exact fixture host slab must prepare");
        assert_eq!(
            exact_slab.write_exact_v1(&exact, &[1, 2, 3], &[4, 5, 6, 7, 8]),
            Ok(PureDecodeGraphV1ExactSlabWrite::Written)
        );
        let exact_address = exact_slab.bytes().as_ptr() as usize;
        let exact_before = exact_slab.bytes().to_vec();

        assert_eq!(
            exact_slab.write_exact_v1_leased(&exact, &[0; 2], &[0; 4]),
            Err(PureDecodeGraphV1ExactSlabWriteError::OpaqueSource(
                PureDecodeGraphV1ExactOpaqueSourceError::FieldLengthMismatch {
                    field: PureDecodeGraphV1ExactOpaqueField::Header,
                    expected: 3,
                    actual: 2,
                }
            ))
        );
        assert_eq!(exact_slab.bytes().as_ptr() as usize, exact_address);
        assert_eq!(exact_slab.bytes(), exact_before);
    }

    #[test]
    fn prepare_error_is_a_closed_error_type() {
        let error =
            PureDecodeGraphV1ExactHostSlabPrepareError::HostAllocation { requested_bytes: 9 };
        assert_eq!(
            error.to_string(),
            "could not allocate 9 exact C07 host-slab backing bytes"
        );
    }

    fn move_owner(owner: PureDecodeGraphV1ExactHostSlab) -> PureDecodeGraphV1ExactHostSlab {
        owner
    }

    fn move_lease(
        lease: PureDecodeGraphV1ExactHostSlabLease<'_>,
    ) -> PureDecodeGraphV1ExactHostSlabLease<'_> {
        lease
    }
}
