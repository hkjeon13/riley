//! Deterministic host-side ownership and block-table state for exact paged KV.
//!
//! This module deliberately owns no CUDA resources. It prepares stable host
//! arrays that a CUDA owner can copy into preallocated device table storage.
//! Reservation, device mutation, and logical publication are separate steps:
//! callers reserve blocks, execute against [`BlockTableV1`], and then commit or
//! roll back the detached [`SequenceReservation`] token.

use std::error;
use std::fmt;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Fixed token capacity of every PR10 physical block.
pub const KV_BLOCK_SIZE: usize = 16;
/// Address-translation table format understood by the PR10 exact path.
pub const BLOCK_TABLE_V1_VERSION: u16 = 1;
/// Initial optional block-metadata descriptor format.
pub const OPTIONAL_BLOCK_METADATA_V1_VERSION: u16 = 1;

const BF16_BYTES: u64 = 2;
const EMPTY_PHYSICAL_BLOCK_ID: u32 = u32::MAX;

static NEXT_POOL_COOKIE: AtomicU64 = AtomicU64::new(1);

/// Result type for host-side paged-KV state management.
pub type PagedKvResult<T> = Result<T, PagedKvError>;

/// Checked configuration, ownership, reservation, or allocation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PagedKvError {
    InvalidConfiguration {
        field: &'static str,
        reason: &'static str,
    },
    ArithmeticOverflow {
        field: &'static str,
    },
    HostAllocation {
        resource: &'static str,
        requested_elements: usize,
    },
    PoolCookieExhausted,
    SequenceIdExhausted,
    GenerationExhausted {
        physical_index: u32,
    },
    ForeignPool {
        expected_cookie: u64,
        actual_cookie: u64,
    },
    OutOfBlocks {
        requested_blocks: usize,
        free_blocks: usize,
    },
    CapacityExceeded {
        requested_tokens: usize,
        maximum_tokens: usize,
    },
    LengthRegression {
        committed_tokens: usize,
        requested_tokens: usize,
    },
    InvalidTruncateLength {
        committed_tokens: usize,
        requested_tokens: usize,
    },
    ReservationInProgress,
    NoReservationInProgress,
    ReservationMismatch,
    ReservationNonceExhausted,
    Poisoned,
    CorruptSequenceState {
        reason: &'static str,
    },
    InvalidLogicalBlock {
        logical_block_index: usize,
        block_count: usize,
    },
    InvalidPhysicalBlock {
        physical_index: u32,
        block_count: usize,
    },
    StaleBlock {
        physical_index: u32,
        handle_generation: u64,
        current_generation: u64,
    },
    DoubleFree {
        physical_index: u32,
        generation: u64,
    },
    WrongOwner {
        physical_index: u32,
        expected: SequenceId,
        actual: Option<SequenceId>,
    },
    UnsupportedMetadataVersion {
        version: u16,
    },
}

#[allow(clippy::too_many_lines)]
impl fmt::Display for PagedKvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid paged-KV configuration {field}: {reason}"
                )
            }
            Self::ArithmeticOverflow { field } => {
                write!(formatter, "paged-KV arithmetic overflow for {field}")
            }
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "could not reserve {requested_elements} host elements for {resource}"
            ),
            Self::PoolCookieExhausted => formatter.write_str("paged-KV pool cookies exhausted"),
            Self::SequenceIdExhausted => {
                formatter.write_str("paged-KV sequence identifiers exhausted")
            }
            Self::GenerationExhausted { physical_index } => write!(
                formatter,
                "generation exhausted for physical KV block {physical_index}"
            ),
            Self::ForeignPool {
                expected_cookie,
                actual_cookie,
            } => write!(
                formatter,
                "paged-KV handle belongs to pool {actual_cookie}, expected {expected_cookie}"
            ),
            Self::OutOfBlocks {
                requested_blocks,
                free_blocks,
            } => write!(
                formatter,
                "paged-KV pool has {free_blocks} free blocks but {requested_blocks} are required"
            ),
            Self::CapacityExceeded {
                requested_tokens,
                maximum_tokens,
            } => write!(
                formatter,
                "paged-KV sequence length {requested_tokens} exceeds fixed capacity {maximum_tokens}"
            ),
            Self::LengthRegression {
                committed_tokens,
                requested_tokens,
            } => write!(
                formatter,
                "reservation target {requested_tokens} is below committed length {committed_tokens}"
            ),
            Self::InvalidTruncateLength {
                committed_tokens,
                requested_tokens,
            } => write!(
                formatter,
                "truncate target {requested_tokens} exceeds committed length {committed_tokens}"
            ),
            Self::ReservationInProgress => {
                formatter.write_str("a paged-KV reservation is already in progress")
            }
            Self::NoReservationInProgress => {
                formatter.write_str("no paged-KV reservation is in progress")
            }
            Self::ReservationMismatch => {
                formatter.write_str("paged-KV reservation token does not match the active owner")
            }
            Self::ReservationNonceExhausted => {
                formatter.write_str("paged-KV reservation nonces exhausted")
            }
            Self::Poisoned => formatter.write_str("the paged-KV sequence is poisoned"),
            Self::CorruptSequenceState { reason } => {
                write!(formatter, "corrupt paged-KV sequence state: {reason}")
            }
            Self::InvalidLogicalBlock {
                logical_block_index,
                block_count,
            } => write!(
                formatter,
                "logical block {logical_block_index} is outside active table length {block_count}"
            ),
            Self::InvalidPhysicalBlock {
                physical_index,
                block_count,
            } => write!(
                formatter,
                "physical block {physical_index} is outside pool length {block_count}"
            ),
            Self::StaleBlock {
                physical_index,
                handle_generation,
                current_generation,
            } => write!(
                formatter,
                "stale physical block {physical_index} generation {handle_generation}; current generation is {current_generation}"
            ),
            Self::DoubleFree {
                physical_index,
                generation,
            } => write!(
                formatter,
                "physical block {physical_index} generation {generation} was already freed"
            ),
            Self::WrongOwner {
                physical_index,
                expected,
                actual,
            } => write!(
                formatter,
                "physical block {physical_index} owner is {actual:?}, expected {expected:?}"
            ),
            Self::UnsupportedMetadataVersion { version } => {
                write!(
                    formatter,
                    "unsupported optional block metadata version {version}"
                )
            }
        }
    }
}

impl error::Error for PagedKvError {}

/// Checked byte layout for `[layer, physical_block, kv_head, token, head_dim]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvLayout {
    layer_count: usize,
    physical_block_count: usize,
    key_value_head_count: usize,
    head_dimension: usize,
    head_stride_bytes: u64,
    block_stride_bytes: u64,
    layer_stride_bytes: u64,
    bytes_per_kind: u64,
    bytes_per_physical_block: u64,
    total_bytes: u64,
}

impl KvLayout {
    /// Builds the fixed PR10 BF16 K/V pool layout.
    ///
    /// # Errors
    ///
    /// Returns for zero dimensions, a physical block count that cannot be
    /// represented by the U32 table ABI, or checked byte-arithmetic overflow.
    pub fn checked(
        layer_count: usize,
        physical_block_count: usize,
        key_value_head_count: usize,
        head_dimension: usize,
    ) -> PagedKvResult<Self> {
        for (field, value) in [
            ("layer_count", layer_count),
            ("physical_block_count", physical_block_count),
            ("key_value_head_count", key_value_head_count),
            ("head_dimension", head_dimension),
        ] {
            if value == 0 {
                return Err(PagedKvError::InvalidConfiguration {
                    field,
                    reason: "must be non-zero",
                });
            }
        }
        if physical_block_count > u32::MAX as usize {
            return Err(PagedKvError::InvalidConfiguration {
                field: "physical_block_count",
                reason: "must fit the U32 physical block table",
            });
        }

        let layers = checked_u64(layer_count, "layer_count")?;
        let blocks = checked_u64(physical_block_count, "physical_block_count")?;
        let heads = checked_u64(key_value_head_count, "key_value_head_count")?;
        let dimension = checked_u64(head_dimension, "head_dimension")?;
        let head_stride_bytes = checked_u64(KV_BLOCK_SIZE, "block_size")?
            .checked_mul(dimension)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or(PagedKvError::ArithmeticOverflow {
                field: "head_stride_bytes",
            })?;
        let block_stride_bytes =
            heads
                .checked_mul(head_stride_bytes)
                .ok_or(PagedKvError::ArithmeticOverflow {
                    field: "block_stride_bytes",
                })?;
        let layer_stride_bytes =
            blocks
                .checked_mul(block_stride_bytes)
                .ok_or(PagedKvError::ArithmeticOverflow {
                    field: "layer_stride_bytes",
                })?;
        let bytes_per_kind =
            layers
                .checked_mul(layer_stride_bytes)
                .ok_or(PagedKvError::ArithmeticOverflow {
                    field: "bytes_per_kind",
                })?;
        let bytes_per_physical_block = layers
            .checked_mul(block_stride_bytes)
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(PagedKvError::ArithmeticOverflow {
                field: "bytes_per_physical_block",
            })?;
        let total_bytes =
            bytes_per_kind
                .checked_mul(2)
                .ok_or(PagedKvError::ArithmeticOverflow {
                    field: "total_bytes",
                })?;

        Ok(Self {
            layer_count,
            physical_block_count,
            key_value_head_count,
            head_dimension,
            head_stride_bytes,
            block_stride_bytes,
            layer_stride_bytes,
            bytes_per_kind,
            bytes_per_physical_block,
            total_bytes,
        })
    }

    #[must_use]
    pub const fn layer_count(self) -> usize {
        self.layer_count
    }

    #[must_use]
    pub const fn physical_block_count(self) -> usize {
        self.physical_block_count
    }

    #[must_use]
    pub const fn key_value_head_count(self) -> usize {
        self.key_value_head_count
    }

    #[must_use]
    pub const fn head_dimension(self) -> usize {
        self.head_dimension
    }

    #[must_use]
    pub const fn block_size(self) -> usize {
        KV_BLOCK_SIZE
    }

    #[must_use]
    pub const fn head_stride_bytes(self) -> u64 {
        self.head_stride_bytes
    }

    #[must_use]
    pub const fn block_stride_bytes(self) -> u64 {
        self.block_stride_bytes
    }

    #[must_use]
    pub const fn layer_stride_bytes(self) -> u64 {
        self.layer_stride_bytes
    }

    #[must_use]
    pub const fn bytes_per_kind(self) -> u64 {
        self.bytes_per_kind
    }

    #[must_use]
    pub const fn bytes_per_physical_block(self) -> u64 {
        self.bytes_per_physical_block
    }

    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }

    /// Returns the byte offset of one layer in either K or V allocation.
    #[must_use]
    pub fn layer_byte_offset(self, layer_index: usize) -> Option<u64> {
        if layer_index >= self.layer_count {
            return None;
        }
        u64::try_from(layer_index)
            .ok()
            .and_then(|layer| layer.checked_mul(self.layer_stride_bytes))
    }

    /// Returns the byte offset in either the K or V allocation.
    #[must_use]
    pub fn element_byte_offset(
        self,
        layer_index: usize,
        physical_block_index: usize,
        key_value_head_index: usize,
        token_in_block: usize,
    ) -> Option<u64> {
        if layer_index >= self.layer_count
            || physical_block_index >= self.physical_block_count
            || key_value_head_index >= self.key_value_head_count
            || token_in_block >= KV_BLOCK_SIZE
        {
            return None;
        }
        let layer = u64::try_from(layer_index).ok()?;
        let block = u64::try_from(physical_block_index).ok()?;
        let head = u64::try_from(key_value_head_index).ok()?;
        let token = u64::try_from(token_in_block).ok()?;
        let token_stride = u64::try_from(self.head_dimension)
            .ok()?
            .checked_mul(BF16_BYTES)?;
        layer
            .checked_mul(self.layer_stride_bytes)?
            .checked_add(block.checked_mul(self.block_stride_bytes)?)?
            .checked_add(head.checked_mul(self.head_stride_bytes)?)?
            .checked_add(token.checked_mul(token_stride)?)
    }
}

/// Pool-local request identity assigned without caller-controlled aliasing.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId {
    pool_cookie: u64,
    value: u64,
}

impl SequenceId {
    #[must_use]
    pub const fn pool_cookie(self) -> u64 {
        self.pool_cookie
    }

    #[must_use]
    pub const fn value(self) -> u64 {
        self.value
    }
}

/// Non-cloneable proof that no live [`SequenceState`] remains for an owner.
#[derive(Debug)]
#[must_use = "pass the orphan token back to its pool to reclaim physical blocks"]
pub struct SequenceReclaimToken {
    owner: SequenceId,
}

impl SequenceReclaimToken {
    #[must_use]
    pub const fn sequence_id(&self) -> SequenceId {
        self.owner
    }
}

/// Opaque physical allocation identity bound to one pool generation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId {
    pool_cookie: u64,
    physical_index: u32,
    generation: u64,
}

impl BlockId {
    #[must_use]
    pub const fn pool_cookie(self) -> u64 {
        self.pool_cookie
    }

    #[must_use]
    pub const fn physical_index(self) -> u32 {
        self.physical_index
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Extensible metadata capability identifier, intentionally separate from V1 addresses.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MetadataKind(u16);

impl MetadataKind {
    pub const KEY_BOUNDS: Self = Self(1);
    pub const VALUE_NORM_BOUND: Self = Self(2);
    pub const QUANTIZATION_SCALE: Self = Self(3);
    pub const RESIDENCY: Self = Self(4);
    pub const PREFIX_REUSE: Self = Self(5);

    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Address-like token and byte range interpreted only by the owning backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpaqueDeviceView {
    address: u64,
    byte_len: u64,
}

impl OpaqueDeviceView {
    #[must_use]
    pub const fn new(address: u64, byte_len: u64) -> Self {
        Self { address, byte_len }
    }

    #[must_use]
    pub const fn address(self) -> u64 {
        self.address
    }

    #[must_use]
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.byte_len == 0
    }
}

/// Optional sidecar descriptor. The pool binds it to a block generation.
#[derive(Debug, Eq, PartialEq)]
pub struct OptionalBlockMetadata {
    version: u16,
    kind: MetadataKind,
    device_view: OpaqueDeviceView,
}

impl OptionalBlockMetadata {
    #[must_use]
    pub const fn new(version: u16, kind: MetadataKind, device_view: OpaqueDeviceView) -> Self {
        Self {
            version,
            kind,
            device_view,
        }
    }

    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    #[must_use]
    pub const fn kind(&self) -> MetadataKind {
        self.kind
    }

    #[must_use]
    pub const fn device_view(&self) -> OpaqueDeviceView {
        self.device_view
    }
}

/// Borrowed metadata whose lifetime prevents concurrent free or block reuse.
#[derive(Debug)]
pub struct BlockMetadataView<'a> {
    block_id: BlockId,
    metadata: &'a OptionalBlockMetadata,
}

impl<'a> BlockMetadataView<'a> {
    #[must_use]
    pub const fn block_id(&self) -> BlockId {
        self.block_id
    }

    #[must_use]
    pub const fn metadata(&self) -> &'a OptionalBlockMetadata {
        self.metadata
    }
}

struct BlockSlot {
    generation: u64,
    owner: Option<SequenceId>,
    sidecar: Option<OptionalBlockMetadata>,
}

/// Snapshot of pool accounting with no device synchronization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvBlockPoolStats {
    physical_block_count: usize,
    allocated_block_count: usize,
    free_block_count: usize,
    high_water_mark: usize,
    usable_kv_bytes: u64,
    host_pool_metadata_bytes: u64,
    sidecar_count: usize,
    sidecar_device_bytes: u64,
    lifetime_allocation_count: u64,
    total_allocation_latency_ns: u64,
    maximum_allocation_latency_ns: u64,
}

impl KvBlockPoolStats {
    #[must_use]
    pub const fn physical_block_count(self) -> usize {
        self.physical_block_count
    }

    #[must_use]
    pub const fn allocated_block_count(self) -> usize {
        self.allocated_block_count
    }

    #[must_use]
    pub const fn free_block_count(self) -> usize {
        self.free_block_count
    }

    #[must_use]
    pub const fn high_water_mark(self) -> usize {
        self.high_water_mark
    }

    /// Total preallocated K/V pool capacity, not current logical usage.
    #[must_use]
    pub const fn usable_kv_bytes(self) -> u64 {
        self.usable_kv_bytes
    }

    #[must_use]
    pub const fn host_pool_metadata_bytes(self) -> u64 {
        self.host_pool_metadata_bytes
    }

    #[must_use]
    pub const fn sidecar_count(self) -> usize {
        self.sidecar_count
    }

    /// Sum of declared sidecar device-view lengths.
    ///
    /// Overlapping or shared opaque views are counted once per descriptor, so
    /// this is metadata accounting rather than unique physical allocation.
    #[must_use]
    pub const fn sidecar_device_bytes(self) -> u64 {
        self.sidecar_device_bytes
    }

    /// Number of successful physical-block leases over this pool's lifetime.
    #[must_use]
    pub const fn lifetime_allocation_count(self) -> u64 {
        self.lifetime_allocation_count
    }

    /// Saturating sum of successful free-list pop, generation, and owner-bind latency.
    #[must_use]
    pub const fn total_allocation_latency_ns(self) -> u64 {
        self.total_allocation_latency_ns
    }

    /// Largest successful free-list pop, generation, and owner-bind latency.
    #[must_use]
    pub const fn maximum_allocation_latency_ns(self) -> u64 {
        self.maximum_allocation_latency_ns
    }
}

/// Deterministic, fixed-capacity physical block allocator.
pub struct KvBlockPool {
    cookie: u64,
    layout: KvLayout,
    slots: Box<[BlockSlot]>,
    free_list: Vec<u32>,
    allocated_block_count: usize,
    high_water_mark: usize,
    next_sequence_id: u64,
    sidecar_count: usize,
    sidecar_device_bytes: u64,
    lifetime_allocation_count: u64,
    total_allocation_latency_ns: u64,
    maximum_allocation_latency_ns: u64,
}

impl fmt::Debug for KvBlockPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KvBlockPool")
            .field("cookie", &self.cookie)
            .field("layout", &self.layout)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl KvBlockPool {
    /// Preallocates every slot and the deterministic LIFO free list.
    ///
    /// Initial allocation order is physical `0, 1, ...`. Releasing a sequence
    /// in reverse logical order makes the same blocks available in that order
    /// again, keeping repeated reset runs deterministic.
    ///
    /// # Errors
    ///
    /// Returns if the process-wide pool cookie or either host allocation fails.
    pub fn new(layout: KvLayout) -> PagedKvResult<Self> {
        let cookie = next_pool_cookie()?;
        let block_count = layout.physical_block_count();
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(block_count)
            .map_err(|_| PagedKvError::HostAllocation {
                resource: "pool_slots",
                requested_elements: block_count,
            })?;
        for _ in 0..block_count {
            slots.push(BlockSlot {
                generation: 0,
                owner: None,
                sidecar: None,
            });
        }
        let mut free_list = Vec::new();
        free_list
            .try_reserve_exact(block_count)
            .map_err(|_| PagedKvError::HostAllocation {
                resource: "free_list",
                requested_elements: block_count,
            })?;
        for index in (0..block_count).rev() {
            free_list.push(u32::try_from(index).map_err(|_| {
                PagedKvError::InvalidConfiguration {
                    field: "physical_block_count",
                    reason: "must fit the U32 physical block table",
                }
            })?);
        }
        Ok(Self {
            cookie,
            layout,
            slots: slots.into_boxed_slice(),
            free_list,
            allocated_block_count: 0,
            high_water_mark: 0,
            next_sequence_id: 1,
            sidecar_count: 0,
            sidecar_device_bytes: 0,
            lifetime_allocation_count: 0,
            total_allocation_latency_ns: 0,
            maximum_allocation_latency_ns: 0,
        })
    }

    #[must_use]
    pub const fn cookie(&self) -> u64 {
        self.cookie
    }

    #[must_use]
    pub const fn layout(&self) -> KvLayout {
        self.layout
    }

    #[must_use]
    pub fn stats(&self) -> KvBlockPoolStats {
        let per_block = mem::size_of::<BlockSlot>()
            .checked_add(mem::size_of::<u32>())
            .and_then(|bytes| bytes.checked_mul(self.slots.len()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        KvBlockPoolStats {
            physical_block_count: self.slots.len(),
            allocated_block_count: self.allocated_block_count,
            free_block_count: self.free_list.len(),
            high_water_mark: self.high_water_mark,
            usable_kv_bytes: self.layout.total_bytes(),
            host_pool_metadata_bytes: per_block,
            sidecar_count: self.sidecar_count,
            sidecar_device_bytes: self.sidecar_device_bytes,
            lifetime_allocation_count: self.lifetime_allocation_count,
            total_allocation_latency_ns: self.total_allocation_latency_ns,
            maximum_allocation_latency_ns: self.maximum_allocation_latency_ns,
        }
    }

    /// Creates a pool-bound sequence with fixed, preallocated host table arrays.
    ///
    /// # Errors
    ///
    /// Returns for an unrepresentable maximum length, sequence-ID exhaustion,
    /// or host table allocation failure.
    pub fn create_sequence(
        &mut self,
        maximum_logical_length: usize,
    ) -> PagedKvResult<SequenceState> {
        if maximum_logical_length == 0 {
            return Err(PagedKvError::InvalidConfiguration {
                field: "maximum_logical_length",
                reason: "must be non-zero",
            });
        }
        let maximum = u32::try_from(maximum_logical_length).map_err(|_| {
            PagedKvError::InvalidConfiguration {
                field: "maximum_logical_length",
                reason: "must fit the U32 block table",
            }
        })?;
        let sequence_value = self.next_sequence_id;
        let next = sequence_value
            .checked_add(1)
            .ok_or(PagedKvError::SequenceIdExhausted)?;
        let sequence = SequenceState::new(
            self.cookie,
            SequenceId {
                pool_cookie: self.cookie,
                value: sequence_value,
            },
            maximum,
        )?;
        self.next_sequence_id = next;
        Ok(sequence)
    }

    /// Validates pool, physical index, generation, allocation state, and owner.
    ///
    /// # Errors
    ///
    /// Distinguishes foreign, stale, already-freed, and wrong-owner handles.
    pub fn validate_block(&self, block: BlockId, owner: SequenceId) -> PagedKvResult<()> {
        self.validate_block_index(block)?;
        if owner.pool_cookie() != self.cookie {
            return Err(PagedKvError::ForeignPool {
                expected_cookie: self.cookie,
                actual_cookie: owner.pool_cookie(),
            });
        }
        let slot = &self.slots[block.physical_index as usize];
        if slot.generation != block.generation {
            return Err(PagedKvError::StaleBlock {
                physical_index: block.physical_index,
                handle_generation: block.generation,
                current_generation: slot.generation,
            });
        }
        match slot.owner {
            None => Err(PagedKvError::DoubleFree {
                physical_index: block.physical_index,
                generation: block.generation,
            }),
            Some(actual) if actual != owner => Err(PagedKvError::WrongOwner {
                physical_index: block.physical_index,
                expected: owner,
                actual: Some(actual),
            }),
            Some(_) => Ok(()),
        }
    }

    /// Reclaims every block owned by a consumed sequence state.
    ///
    /// Normal owners should use [`SequenceState::close`] or `reset`. This scan
    /// is the explicit cancellation seam for a higher-level owner that first
    /// consumed the state with [`SequenceState::abandon_for_reclaim`].
    ///
    /// # Errors
    ///
    /// Returns when the orphan proof belongs to another pool.
    pub fn reclaim_sequence(&mut self, token: &SequenceReclaimToken) -> PagedKvResult<usize> {
        let owner = token.owner;
        if owner.pool_cookie() != self.cookie {
            return Err(PagedKvError::ForeignPool {
                expected_cookie: self.cookie,
                actual_cookie: owner.pool_cookie(),
            });
        }
        let mut released = 0;
        for index in (0..self.slots.len()).rev() {
            if self.slots[index].owner == Some(owner) {
                self.release_slot(index);
                released += 1;
            }
        }
        Ok(released)
    }

    fn validate_block_index(&self, block: BlockId) -> PagedKvResult<()> {
        if block.pool_cookie != self.cookie {
            return Err(PagedKvError::ForeignPool {
                expected_cookie: self.cookie,
                actual_cookie: block.pool_cookie,
            });
        }
        if block.physical_index as usize >= self.slots.len() {
            return Err(PagedKvError::InvalidPhysicalBlock {
                physical_index: block.physical_index,
                block_count: self.slots.len(),
            });
        }
        Ok(())
    }

    fn allocate_block(&mut self, owner: SequenceId) -> PagedKvResult<BlockId> {
        let started = Instant::now();
        let block = self.allocate_block_inner(owner)?;
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.lifetime_allocation_count = self.lifetime_allocation_count.saturating_add(1);
        self.total_allocation_latency_ns = self.total_allocation_latency_ns.saturating_add(elapsed);
        self.maximum_allocation_latency_ns = self.maximum_allocation_latency_ns.max(elapsed);
        Ok(block)
    }

    fn allocate_block_inner(&mut self, owner: SequenceId) -> PagedKvResult<BlockId> {
        let physical_index = self.free_list.pop().ok_or(PagedKvError::OutOfBlocks {
            requested_blocks: 1,
            free_blocks: 0,
        })?;
        let slot = &mut self.slots[physical_index as usize];
        debug_assert!(slot.owner.is_none());
        debug_assert!(slot.sidecar.is_none());
        let Some(generation) = slot.generation.checked_add(1) else {
            self.free_list.push(physical_index);
            return Err(PagedKvError::GenerationExhausted { physical_index });
        };
        slot.generation = generation;
        slot.owner = Some(owner);
        self.allocated_block_count += 1;
        self.high_water_mark = self.high_water_mark.max(self.allocated_block_count);
        Ok(BlockId {
            pool_cookie: self.cookie,
            physical_index,
            generation,
        })
    }

    fn release_block(&mut self, block: BlockId, owner: SequenceId) -> PagedKvResult<()> {
        self.validate_block(block, owner)?;
        self.release_slot(block.physical_index as usize);
        Ok(())
    }

    fn release_slot(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        debug_assert!(slot.owner.is_some());
        if let Some(sidecar) = slot.sidecar.take() {
            self.sidecar_count -= 1;
            self.sidecar_device_bytes -= sidecar.device_view().byte_len();
        }
        slot.owner = None;
        self.allocated_block_count -= 1;
        debug_assert!(self.free_list.len() < self.free_list.capacity());
        self.free_list
            .push(u32::try_from(index).expect("layout checked physical U32 index"));
    }

    fn attach_sidecar(
        &mut self,
        block: BlockId,
        owner: SequenceId,
        metadata: OptionalBlockMetadata,
    ) -> PagedKvResult<()> {
        self.validate_block(block, owner)?;
        if metadata.version() != OPTIONAL_BLOCK_METADATA_V1_VERSION {
            return Err(PagedKvError::UnsupportedMetadataVersion {
                version: metadata.version(),
            });
        }
        let slot = &self.slots[block.physical_index as usize];
        let prior_bytes = slot
            .sidecar
            .as_ref()
            .map_or(0, |sidecar| sidecar.device_view().byte_len());
        let base = self
            .sidecar_device_bytes
            .checked_sub(prior_bytes)
            .expect("tracked sidecar bytes include the replaced descriptor");
        let updated = base.checked_add(metadata.device_view().byte_len()).ok_or(
            PagedKvError::ArithmeticOverflow {
                field: "sidecar_device_bytes",
            },
        )?;
        let slot = &mut self.slots[block.physical_index as usize];
        if slot.sidecar.is_none() {
            self.sidecar_count += 1;
        }
        slot.sidecar = Some(metadata);
        self.sidecar_device_bytes = updated;
        Ok(())
    }

    fn sidecar(
        &self,
        block: BlockId,
        owner: SequenceId,
    ) -> PagedKvResult<Option<BlockMetadataView<'_>>> {
        self.validate_block(block, owner)?;
        Ok(self.slots[block.physical_index as usize]
            .sidecar
            .as_ref()
            .map(|metadata| BlockMetadataView {
                block_id: block,
                metadata,
            }))
    }

    fn clear_sidecar(&mut self, block: BlockId, owner: SequenceId) -> PagedKvResult<()> {
        self.validate_block(block, owner)?;
        let slot = &mut self.slots[block.physical_index as usize];
        if let Some(sidecar) = slot.sidecar.take() {
            self.sidecar_count -= 1;
            self.sidecar_device_bytes -= sidecar.device_view().byte_len();
        }
        Ok(())
    }
}

/// Borrowed, exact V1 address-translation table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockTableV1<'a> {
    physical_block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    logical_length: u32,
}

impl<'a> BlockTableV1<'a> {
    #[must_use]
    pub const fn version(self) -> u16 {
        BLOCK_TABLE_V1_VERSION
    }

    #[must_use]
    pub const fn physical_block_ids(self) -> &'a [u32] {
        self.physical_block_ids
    }

    #[must_use]
    pub const fn valid_tokens(self) -> &'a [u16] {
        self.valid_tokens
    }

    #[must_use]
    pub const fn logical_length(self) -> u32 {
        self.logical_length
    }

    #[must_use]
    pub const fn block_count(self) -> usize {
        self.physical_block_ids.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingReservation {
    nonce: u64,
    previous_block_count: usize,
    target_logical_length: u32,
}

/// Detached authorization to publish or abandon one tentative table state.
///
/// Dropping this value does not silently guess whether device mutation began.
/// The sequence remains locked until the caller explicitly rolls it back,
/// poisons it, resets it, or invokes abandoned-reservation recovery.
#[derive(Debug)]
#[must_use = "a paged-KV reservation must be committed, rolled back, or poisoned"]
pub struct SequenceReservation {
    pool_cookie: u64,
    sequence_id: SequenceId,
    nonce: u64,
    previous_block_count: usize,
    target_logical_length: u32,
}

impl SequenceReservation {
    #[must_use]
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    #[must_use]
    pub const fn target_logical_length(&self) -> u32 {
        self.target_logical_length
    }
}

/// Summary returned after the tentative logical length becomes visible.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReservationCommit {
    previous_logical_length: u32,
    logical_length: u32,
    allocated_block_count: usize,
}

impl ReservationCommit {
    #[must_use]
    pub const fn previous_logical_length(self) -> u32 {
        self.previous_logical_length
    }

    #[must_use]
    pub const fn logical_length(self) -> u32 {
        self.logical_length
    }

    #[must_use]
    pub const fn allocated_block_count(self) -> usize {
        self.allocated_block_count
    }
}

/// One pool-bound sequence with fixed-capacity host block-table backing arrays.
#[derive(Debug)]
#[must_use = "release allocated blocks with close/reset or reclaim the sequence from its pool"]
pub struct SequenceState {
    pool_cookie: u64,
    sequence_id: SequenceId,
    maximum_logical_length: u32,
    block_ids: Box<[Option<BlockId>]>,
    physical_block_ids: Box<[u32]>,
    valid_tokens: Box<[u16]>,
    allocated_block_count: usize,
    logical_length: u32,
    next_reservation_nonce: u64,
    pending: Option<PendingReservation>,
    poisoned: bool,
}

impl SequenceState {
    fn new(
        pool_cookie: u64,
        sequence_id: SequenceId,
        maximum_logical_length: u32,
    ) -> PagedKvResult<Self> {
        let maximum_blocks = blocks_for_length(maximum_logical_length as usize);
        Ok(Self {
            pool_cookie,
            sequence_id,
            maximum_logical_length,
            block_ids: boxed_filled(maximum_blocks, None, "sequence_block_handles")?,
            physical_block_ids: boxed_filled(
                maximum_blocks,
                EMPTY_PHYSICAL_BLOCK_ID,
                "sequence_physical_ids",
            )?,
            valid_tokens: boxed_filled(maximum_blocks, 0, "sequence_valid_tokens")?,
            allocated_block_count: 0,
            logical_length: 0,
            next_reservation_nonce: 1,
            pending: None,
            poisoned: false,
        })
    }

    #[must_use]
    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    #[must_use]
    pub const fn pool_cookie(&self) -> u64 {
        self.pool_cookie
    }

    #[must_use]
    pub const fn maximum_logical_length(&self) -> u32 {
        self.maximum_logical_length
    }

    #[must_use]
    pub const fn logical_length(&self) -> u32 {
        self.logical_length
    }

    #[must_use]
    pub const fn allocated_block_count(&self) -> usize {
        self.allocated_block_count
    }

    #[must_use]
    pub const fn maximum_block_count(&self) -> usize {
        self.block_ids.len()
    }

    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[must_use]
    pub const fn has_pending_reservation(&self) -> bool {
        self.pending.is_some()
    }

    /// U32 IDs plus U16 valid counts in the fully preallocated V1 table arrays.
    #[must_use]
    pub fn host_table_capacity_bytes(&self) -> u64 {
        u64::try_from(self.maximum_block_count())
            .ok()
            .and_then(|blocks| blocks.checked_mul(6))
            .unwrap_or(u64::MAX)
    }

    /// Preallocated host payload for handles plus the exact U32/U16 V1 table.
    #[must_use]
    pub fn host_state_capacity_bytes(&self) -> u64 {
        let bytes_per_block = mem::size_of::<Option<BlockId>>()
            .checked_add(mem::size_of::<u32>())
            .and_then(|bytes| bytes.checked_add(mem::size_of::<u16>()));
        bytes_per_block
            .and_then(|bytes| bytes.checked_mul(self.maximum_block_count()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX)
    }

    /// Token slots allocated but unused by the committed logical length.
    #[must_use]
    pub fn internal_fragmentation_tokens(&self) -> usize {
        blocks_for_length(self.logical_length as usize)
            .saturating_mul(KV_BLOCK_SIZE)
            .saturating_sub(self.logical_length as usize)
    }

    /// Returns one opaque committed block identity for diagnostics/metadata.
    #[must_use]
    pub fn block_id(&self, logical_block_index: usize) -> Option<BlockId> {
        let committed_blocks = blocks_for_length(self.logical_length as usize);
        if logical_block_index >= committed_blocks {
            return None;
        }
        self.block_ids[logical_block_index]
    }

    /// Returns the currently committed V1 table.
    ///
    /// # Errors
    ///
    /// A poisoned owner or forgotten active reservation is not publishable.
    pub fn block_table(&self) -> PagedKvResult<BlockTableV1<'_>> {
        self.ensure_readable()?;
        let block_count = blocks_for_length(self.logical_length as usize);
        Ok(self.table_view(block_count, self.logical_length))
    }

    /// Returns the tentative table authorized by a matching reservation token.
    ///
    /// # Errors
    ///
    /// Returns if the token belongs to another pool/sequence/reservation.
    pub fn reserved_block_table(
        &self,
        reservation: &SequenceReservation,
    ) -> PagedKvResult<BlockTableV1<'_>> {
        self.validate_reservation(reservation)?;
        Ok(self.table_view(
            self.allocated_block_count,
            reservation.target_logical_length,
        ))
    }

    /// Reserves all physical blocks required by `target_logical_length`.
    ///
    /// The committed logical length is unchanged. All arrays and free-list
    /// storage were allocated during construction; this hot operation performs
    /// no host heap allocation. OOM is preflighted, and any later allocator
    /// error releases every tentative block before returning.
    ///
    /// # Errors
    ///
    /// Returns for poison, a pending transaction, pool mismatch, length
    /// regression/capacity, insufficient free blocks, or generation exhaustion.
    // HOT_RESERVATION_BEGIN
    pub fn reserve_to(
        &mut self,
        pool: &mut KvBlockPool,
        target_logical_length: usize,
    ) -> PagedKvResult<SequenceReservation> {
        self.ensure_mutable(pool)?;
        if self.pending.is_some() {
            return Err(PagedKvError::ReservationInProgress);
        }
        if target_logical_length < self.logical_length as usize {
            return Err(PagedKvError::LengthRegression {
                committed_tokens: self.logical_length as usize,
                requested_tokens: target_logical_length,
            });
        }
        if target_logical_length > self.maximum_logical_length as usize {
            return Err(PagedKvError::CapacityExceeded {
                requested_tokens: target_logical_length,
                maximum_tokens: self.maximum_logical_length as usize,
            });
        }
        let target =
            u32::try_from(target_logical_length).map_err(|_| PagedKvError::CapacityExceeded {
                requested_tokens: target_logical_length,
                maximum_tokens: self.maximum_logical_length as usize,
            })?;
        let target_blocks = blocks_for_length(target_logical_length);
        let additional_blocks = target_blocks.saturating_sub(self.allocated_block_count);
        if additional_blocks > pool.free_list.len() {
            return Err(PagedKvError::OutOfBlocks {
                requested_blocks: additional_blocks,
                free_blocks: pool.free_list.len(),
            });
        }
        if self.allocated_block_count != 0 {
            let tail = self.allocated_block(self.allocated_block_count - 1)?;
            pool.validate_block(tail, self.sequence_id)?;
        }
        let nonce = self.next_reservation_nonce;
        let next_nonce = nonce
            .checked_add(1)
            .ok_or(PagedKvError::ReservationNonceExhausted)?;
        let previous_block_count = self.allocated_block_count;
        while self.allocated_block_count < target_blocks {
            match pool.allocate_block(self.sequence_id) {
                Ok(block) => {
                    let index = self.allocated_block_count;
                    self.block_ids[index] = Some(block);
                    self.physical_block_ids[index] = block.physical_index();
                    self.valid_tokens[index] = 0;
                    self.allocated_block_count += 1;
                }
                Err(error) => {
                    self.rollback_new_blocks(pool, previous_block_count)?;
                    return Err(error);
                }
            }
        }
        self.fill_valid_tokens(target_logical_length, target_blocks);
        let pending = PendingReservation {
            nonce,
            previous_block_count,
            target_logical_length: target,
        };
        self.pending = Some(pending);
        self.next_reservation_nonce = next_nonce;
        Ok(SequenceReservation {
            pool_cookie: self.pool_cookie,
            sequence_id: self.sequence_id,
            nonce,
            previous_block_count,
            target_logical_length: target,
        })
    }

    /// Publishes the tentative logical length after successful device execution.
    ///
    /// Sidecars for blocks whose content range changed are invalidated before
    /// logical publication. The logical length is the final mutation.
    ///
    /// # Errors
    ///
    /// Returns for a mismatched reservation or stale internal block handle.
    pub fn commit(
        &mut self,
        pool: &mut KvBlockPool,
        reservation: SequenceReservation,
    ) -> PagedKvResult<ReservationCommit> {
        self.ensure_pool(pool)?;
        self.validate_reservation(&reservation)?;
        let previous_logical_length = self.logical_length;
        let first_changed = if reservation.target_logical_length > previous_logical_length {
            previous_logical_length as usize / KV_BLOCK_SIZE
        } else {
            self.allocated_block_count
        };
        for index in first_changed..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.validate_block(block, self.sequence_id)?;
        }
        for index in first_changed..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.clear_sidecar(block, self.sequence_id)?;
        }
        self.pending = None;
        self.logical_length = reservation.target_logical_length;
        drop(reservation);
        Ok(ReservationCommit {
            previous_logical_length,
            logical_length: self.logical_length,
            allocated_block_count: self.allocated_block_count,
        })
    }

    /// Abandons a reservation known not to have mutated device cache content.
    ///
    /// # Errors
    ///
    /// Returns for a mismatched reservation or stale tentative block handle.
    pub fn rollback(
        &mut self,
        pool: &mut KvBlockPool,
        reservation: SequenceReservation,
    ) -> PagedKvResult<()> {
        self.ensure_pool(pool)?;
        self.validate_reservation(&reservation)?;
        drop(reservation);
        self.rollback_pending(pool)
    }

    /// Rolls back tentative blocks and poisons the owner after device mutation failure.
    ///
    /// # Errors
    ///
    /// Returns for a mismatched reservation or stale tentative block handle.
    pub fn poison(
        &mut self,
        pool: &mut KvBlockPool,
        reservation: SequenceReservation,
    ) -> PagedKvResult<()> {
        self.ensure_pool(pool)?;
        self.validate_reservation(&reservation)?;
        self.poisoned = true;
        for index in 0..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.validate_block(block, self.sequence_id)?;
        }
        for index in 0..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.clear_sidecar(block, self.sequence_id)?;
        }
        drop(reservation);
        self.rollback_pending(pool)
    }

    /// Safely recovers a lost reservation token by rolling back and poisoning.
    ///
    /// # Errors
    ///
    /// Returns if no reservation exists, the pool differs, or a handle is stale.
    pub fn rollback_abandoned_reservation(&mut self, pool: &mut KvBlockPool) -> PagedKvResult<()> {
        self.ensure_pool(pool)?;
        if self.pending.is_none() {
            return Err(PagedKvError::NoReservationInProgress);
        }
        self.poisoned = true;
        self.rollback_pending(pool)
    }
    // HOT_RESERVATION_END

    /// Shrinks committed state and releases now-unused tail blocks.
    ///
    /// # Errors
    ///
    /// Returns for poison, pending reservation, pool mismatch, invalid target,
    /// or stale ownership state. Validation completes before any release.
    pub fn truncate_to(
        &mut self,
        pool: &mut KvBlockPool,
        target_logical_length: usize,
    ) -> PagedKvResult<()> {
        self.ensure_mutable(pool)?;
        if self.pending.is_some() {
            return Err(PagedKvError::ReservationInProgress);
        }
        if target_logical_length > self.logical_length as usize {
            return Err(PagedKvError::InvalidTruncateLength {
                committed_tokens: self.logical_length as usize,
                requested_tokens: target_logical_length,
            });
        }
        let target_blocks = blocks_for_length(target_logical_length);
        for index in target_blocks..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.validate_block(block, self.sequence_id)?;
        }
        if target_blocks != 0 && target_logical_length != self.logical_length as usize {
            let tail = self.allocated_block(target_blocks - 1)?;
            pool.validate_block(tail, self.sequence_id)?;
        }
        if target_blocks != 0 && target_logical_length != self.logical_length as usize {
            let tail = self.allocated_block(target_blocks - 1)?;
            pool.clear_sidecar(tail, self.sequence_id)?;
        }
        self.release_tail(pool, target_blocks)?;
        self.logical_length = u32::try_from(target_logical_length).map_err(|_| {
            PagedKvError::CorruptSequenceState {
                reason: "truncate target does not fit committed U32 length",
            }
        })?;
        self.fill_valid_tokens(target_logical_length, target_blocks);
        Ok(())
    }

    /// Releases all blocks, clears poison/pending state, and preserves arrays.
    ///
    /// # Errors
    ///
    /// Returns for pool mismatch or stale ownership. All handles are validated
    /// before the first committed block is released.
    pub fn reset(&mut self, pool: &mut KvBlockPool) -> PagedKvResult<()> {
        self.ensure_pool(pool)?;
        if self.pending.is_some() {
            self.rollback_pending(pool)?;
        }
        for index in 0..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.validate_block(block, self.sequence_id)?;
        }
        self.release_tail(pool, 0)?;
        self.logical_length = 0;
        self.pending = None;
        self.poisoned = false;
        Ok(())
    }

    /// Returns every physical block while retaining retryable empty host state.
    ///
    /// # Errors
    ///
    /// Returns the same cleanup failures as [`Self::reset`].
    pub fn close(&mut self, pool: &mut KvBlockPool) -> PagedKvResult<()> {
        self.reset(pool)
    }

    /// Consumes the host table without releasing its blocks and returns the
    /// only token accepted by the pool's orphan-reclamation scan.
    ///
    /// This models cancellation after ownership transfer: once called, safe
    /// code cannot continue reading a stale block table for the reclaimed
    /// sequence. Prefer [`Self::close`] when ordinary cleanup is available.
    pub fn abandon_for_reclaim(self) -> SequenceReclaimToken {
        SequenceReclaimToken {
            owner: self.sequence_id,
        }
    }

    /// Associates optional V1 metadata with one committed block generation.
    ///
    /// # Errors
    ///
    /// Returns for poison/pending state, invalid logical index, stale owner, or
    /// an unsupported descriptor version. Empty V1 device views are valid.
    pub fn attach_sidecar(
        &self,
        pool: &mut KvBlockPool,
        logical_block_index: usize,
        metadata: OptionalBlockMetadata,
    ) -> PagedKvResult<()> {
        self.ensure_readable()?;
        self.ensure_pool(pool)?;
        let block = self.committed_block(logical_block_index)?;
        pool.attach_sidecar(block, self.sequence_id, metadata)
    }

    /// Borrows optional metadata bound to the current block generation.
    ///
    /// # Errors
    ///
    /// Returns for poison/pending state, invalid logical index, or stale owner.
    pub fn sidecar<'a>(
        &self,
        pool: &'a KvBlockPool,
        logical_block_index: usize,
    ) -> PagedKvResult<Option<BlockMetadataView<'a>>> {
        self.ensure_readable()?;
        self.ensure_pool(pool)?;
        let block = self.committed_block(logical_block_index)?;
        pool.sidecar(block, self.sequence_id)
    }

    fn ensure_pool(&self, pool: &KvBlockPool) -> PagedKvResult<()> {
        if pool.cookie != self.pool_cookie {
            return Err(PagedKvError::ForeignPool {
                expected_cookie: self.pool_cookie,
                actual_cookie: pool.cookie,
            });
        }
        Ok(())
    }

    fn ensure_mutable(&self, pool: &KvBlockPool) -> PagedKvResult<()> {
        self.ensure_pool(pool)?;
        if self.poisoned {
            return Err(PagedKvError::Poisoned);
        }
        Ok(())
    }

    fn ensure_readable(&self) -> PagedKvResult<()> {
        if self.poisoned {
            return Err(PagedKvError::Poisoned);
        }
        if self.pending.is_some() {
            return Err(PagedKvError::ReservationInProgress);
        }
        Ok(())
    }

    fn validate_reservation(&self, reservation: &SequenceReservation) -> PagedKvResult<()> {
        if reservation.pool_cookie != self.pool_cookie
            || reservation.sequence_id != self.sequence_id
        {
            return Err(PagedKvError::ReservationMismatch);
        }
        let Some(pending) = self.pending else {
            return Err(PagedKvError::NoReservationInProgress);
        };
        if pending.nonce != reservation.nonce
            || pending.previous_block_count != reservation.previous_block_count
            || pending.target_logical_length != reservation.target_logical_length
        {
            return Err(PagedKvError::ReservationMismatch);
        }
        Ok(())
    }

    fn committed_block(&self, logical_block_index: usize) -> PagedKvResult<BlockId> {
        let block_count = blocks_for_length(self.logical_length as usize);
        if logical_block_index >= block_count {
            return Err(PagedKvError::InvalidLogicalBlock {
                logical_block_index,
                block_count,
            });
        }
        self.allocated_block(logical_block_index)
    }

    fn allocated_block(&self, index: usize) -> PagedKvResult<BlockId> {
        self.block_ids
            .get(index)
            .copied()
            .flatten()
            .ok_or(PagedKvError::CorruptSequenceState {
                reason: "allocated table prefix is missing a block handle",
            })
    }

    fn table_view(&self, block_count: usize, logical_length: u32) -> BlockTableV1<'_> {
        BlockTableV1 {
            physical_block_ids: &self.physical_block_ids[..block_count],
            valid_tokens: &self.valid_tokens[..block_count],
            logical_length,
        }
    }

    fn fill_valid_tokens(&mut self, logical_length: usize, block_count: usize) {
        for index in 0..block_count {
            let block_start = index * KV_BLOCK_SIZE;
            self.valid_tokens[index] = u16::try_from(
                logical_length
                    .saturating_sub(block_start)
                    .min(KV_BLOCK_SIZE),
            )
            .expect("fixed block size fits U16");
        }
        for index in block_count..self.allocated_block_count {
            self.valid_tokens[index] = 0;
        }
    }

    fn rollback_pending(&mut self, pool: &mut KvBlockPool) -> PagedKvResult<()> {
        let pending = self.pending.ok_or(PagedKvError::NoReservationInProgress)?;
        self.rollback_new_blocks(pool, pending.previous_block_count)?;
        let committed_blocks = blocks_for_length(self.logical_length as usize);
        self.fill_valid_tokens(self.logical_length as usize, committed_blocks);
        self.pending = None;
        Ok(())
    }

    fn rollback_new_blocks(
        &mut self,
        pool: &mut KvBlockPool,
        previous_block_count: usize,
    ) -> PagedKvResult<()> {
        for index in previous_block_count..self.allocated_block_count {
            let block = self.allocated_block(index)?;
            pool.validate_block(block, self.sequence_id)?;
        }
        self.release_tail(pool, previous_block_count)
    }

    fn release_tail(
        &mut self,
        pool: &mut KvBlockPool,
        retained_block_count: usize,
    ) -> PagedKvResult<()> {
        while self.allocated_block_count > retained_block_count {
            let index = self.allocated_block_count - 1;
            let block = self.allocated_block(index)?;
            pool.release_block(block, self.sequence_id)?;
            self.block_ids[index] = None;
            self.physical_block_ids[index] = EMPTY_PHYSICAL_BLOCK_ID;
            self.valid_tokens[index] = 0;
            self.allocated_block_count = index;
        }
        Ok(())
    }
}

fn blocks_for_length(logical_length: usize) -> usize {
    logical_length.div_ceil(KV_BLOCK_SIZE)
}

fn checked_u64(value: usize, field: &'static str) -> PagedKvResult<u64> {
    u64::try_from(value).map_err(|_| PagedKvError::ArithmeticOverflow { field })
}

fn boxed_filled<T: Clone>(
    length: usize,
    value: T,
    resource: &'static str,
) -> PagedKvResult<Box<[T]>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| PagedKvError::HostAllocation {
            resource,
            requested_elements: length,
        })?;
    values.resize(length, value);
    Ok(values.into_boxed_slice())
}

fn next_pool_cookie() -> PagedKvResult<u64> {
    NEXT_POOL_COOKIE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| PagedKvError::PoolCookieExhausted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(blocks: usize) -> KvBlockPool {
        KvBlockPool::new(KvLayout::checked(2, blocks, 3, 64).expect("valid test layout"))
            .expect("test pool")
    }

    fn reserve_and_commit(sequence: &mut SequenceState, pool: &mut KvBlockPool, target: usize) {
        let reservation = sequence.reserve_to(pool, target).expect("reserve");
        let table = sequence
            .reserved_block_table(&reservation)
            .expect("reserved table");
        assert_eq!(
            table.logical_length(),
            u32::try_from(target).expect("test target fits U32")
        );
        sequence.commit(pool, reservation).expect("commit");
    }

    #[test]
    fn layout_is_checked_layer_block_head_token_major() {
        let layout = KvLayout::checked(2, 5, 3, 64).expect("layout");
        assert_eq!(layout.block_size(), 16);
        assert_eq!(layout.head_stride_bytes(), 16 * 64 * 2);
        assert_eq!(layout.block_stride_bytes(), 3 * 16 * 64 * 2);
        assert_eq!(layout.layer_stride_bytes(), 5 * 3 * 16 * 64 * 2);
        assert_eq!(layout.bytes_per_kind(), 2 * 5 * 3 * 16 * 64 * 2);
        assert_eq!(layout.total_bytes(), 2 * layout.bytes_per_kind());
        assert_eq!(layout.bytes_per_physical_block(), 2 * 2 * 3 * 16 * 64 * 2);
        assert_eq!(layout.element_byte_offset(0, 0, 0, 0), Some(0));
        assert_eq!(
            layout.element_byte_offset(1, 2, 1, 3),
            Some(
                layout.layer_stride_bytes()
                    + 2 * layout.block_stride_bytes()
                    + layout.head_stride_bytes()
                    + 3 * 64 * 2
            )
        );
        assert_eq!(layout.element_byte_offset(2, 0, 0, 0), None);
        assert_eq!(layout.element_byte_offset(0, 5, 0, 0), None);
        assert_eq!(layout.element_byte_offset(0, 0, 3, 0), None);
        assert_eq!(layout.element_byte_offset(0, 0, 0, 16), None);
    }

    #[test]
    fn block_boundaries_publish_exact_v1_tables_and_reuse_deterministically() {
        let mut pool = pool(9);
        let mut sequence = pool.create_sequence(129).expect("sequence");
        for target in [1, 15, 16, 17, 31, 32, 33, 128, 129] {
            sequence.reset(&mut pool).expect("reset");
            let reservation = sequence.reserve_to(&mut pool, target).expect("reserve");
            let table = sequence
                .reserved_block_table(&reservation)
                .expect("pending table");
            assert_eq!(table.version(), BLOCK_TABLE_V1_VERSION);
            assert_eq!(
                table.logical_length(),
                u32::try_from(target).expect("test target fits U32")
            );
            assert_eq!(table.block_count(), blocks_for_length(target));
            assert_eq!(
                table.physical_block_ids(),
                &(0..u32::try_from(blocks_for_length(target)).expect("small test"))
                    .collect::<Vec<_>>()
            );
            let expected_last = if target % KV_BLOCK_SIZE == 0 {
                KV_BLOCK_SIZE
            } else {
                target % KV_BLOCK_SIZE
            };
            assert!(
                table.valid_tokens()[..table.block_count() - 1]
                    .iter()
                    .all(|&tokens| tokens == u16::try_from(KV_BLOCK_SIZE).unwrap())
            );
            assert_eq!(
                table.valid_tokens().last(),
                Some(&u16::try_from(expected_last).expect("test tail fits U16"))
            );
            let pending_ids = table.physical_block_ids().to_vec();
            let pending_valid_tokens = table.valid_tokens().to_vec();
            sequence.commit(&mut pool, reservation).expect("commit");
            let committed = sequence.block_table().expect("table");
            assert_eq!(
                committed.logical_length(),
                u32::try_from(target).expect("test target fits U32")
            );
            assert_eq!(committed.physical_block_ids(), pending_ids);
            assert_eq!(committed.valid_tokens(), pending_valid_tokens);
            assert_eq!(
                sequence.internal_fragmentation_tokens(),
                blocks_for_length(target) * KV_BLOCK_SIZE - target
            );
        }
        sequence.reset(&mut pool).expect("final reset");
        let stats = pool.stats();
        assert_eq!(stats.allocated_block_count(), 0);
        assert_eq!(stats.free_block_count(), 9);
        assert_eq!(stats.high_water_mark(), 9);
        assert_eq!(stats.lifetime_allocation_count(), 29);
        assert!(stats.total_allocation_latency_ns() >= stats.maximum_allocation_latency_ns());
        assert_eq!(
            sequence.host_state_capacity_bytes(),
            u64::try_from(
                sequence.maximum_block_count()
                    * (mem::size_of::<Option<BlockId>>()
                        + mem::size_of::<u32>()
                        + mem::size_of::<u16>())
            )
            .expect("small test payload")
        );
    }

    #[test]
    fn oom_and_explicit_rollback_leave_committed_state_unchanged() {
        let mut pool = pool(3);
        let mut sequence = pool.create_sequence(64).expect("sequence");
        reserve_and_commit(&mut sequence, &mut pool, 17);
        let committed_ids = sequence
            .block_table()
            .expect("table")
            .physical_block_ids()
            .to_vec();
        let before = pool.stats();
        assert!(matches!(
            sequence.reserve_to(&mut pool, 49),
            Err(PagedKvError::OutOfBlocks {
                requested_blocks: 2,
                free_blocks: 1,
            })
        ));
        assert_eq!(pool.stats(), before);
        assert_eq!(sequence.logical_length(), 17);
        assert_eq!(
            sequence.block_table().expect("table").physical_block_ids(),
            committed_ids
        );

        let reservation = sequence.reserve_to(&mut pool, 33).expect("reserve one");
        assert_eq!(pool.stats().allocated_block_count(), 3);
        assert_eq!(
            sequence
                .reserved_block_table(&reservation)
                .expect("pending")
                .valid_tokens(),
            &[16, 16, 1]
        );
        sequence.rollback(&mut pool, reservation).expect("rollback");
        assert_eq!(
            pool.stats().allocated_block_count(),
            before.allocated_block_count()
        );
        assert_eq!(pool.stats().free_block_count(), before.free_block_count());
        assert_eq!(pool.stats().high_water_mark(), 3);
        assert_eq!(sequence.logical_length(), 17);
        assert_eq!(
            sequence.block_table().expect("table").valid_tokens(),
            &[16, 1]
        );
    }

    #[test]
    fn device_failure_poison_rolls_back_tentative_blocks_and_invalidates_sidecars() {
        let mut pool = pool(3);
        let mut sequence = pool.create_sequence(32).expect("sequence");
        reserve_and_commit(&mut sequence, &mut pool, 15);
        sequence
            .attach_sidecar(
                &mut pool,
                0,
                OptionalBlockMetadata::new(
                    OPTIONAL_BLOCK_METADATA_V1_VERSION,
                    MetadataKind::KEY_BOUNDS,
                    OpaqueDeviceView::new(0x1000, 64),
                ),
            )
            .expect("sidecar");
        let reservation = sequence.reserve_to(&mut pool, 17).expect("tentative block");
        assert_eq!(pool.stats().allocated_block_count(), 2);
        sequence
            .poison(&mut pool, reservation)
            .expect("poison and rollback");
        assert!(sequence.is_poisoned());
        assert!(!sequence.has_pending_reservation());
        assert_eq!(sequence.logical_length(), 15);
        assert_eq!(sequence.allocated_block_count(), 1);
        assert_eq!(pool.stats().allocated_block_count(), 1);
        assert_eq!(pool.stats().free_block_count(), 2);
        assert_eq!(pool.stats().sidecar_count(), 0);
        assert_eq!(sequence.block_table(), Err(PagedKvError::Poisoned));

        sequence.reset(&mut pool).expect("reset clears poison");
        assert!(!sequence.is_poisoned());
        assert_eq!(sequence.logical_length(), 0);
        assert_eq!(pool.stats().allocated_block_count(), 0);
        assert_eq!(pool.stats().free_block_count(), 3);
    }

    #[test]
    fn stale_double_free_wrong_owner_and_foreign_pool_are_distinct() {
        let mut first_pool = pool(1);
        let mut first = first_pool.create_sequence(16).expect("first sequence");
        let second = first_pool.create_sequence(16).expect("second sequence");
        reserve_and_commit(&mut first, &mut first_pool, 1);
        let old = first.block_id(0).expect("block");
        let mut other_pool = pool(1);
        let foreign_owner = other_pool
            .create_sequence(16)
            .expect("foreign first sequence");
        assert_eq!(
            first.sequence_id().value(),
            foreign_owner.sequence_id().value()
        );
        assert_ne!(first.sequence_id(), foreign_owner.sequence_id());
        assert_eq!(
            first_pool.validate_block(old, foreign_owner.sequence_id()),
            Err(PagedKvError::ForeignPool {
                expected_cookie: first_pool.cookie(),
                actual_cookie: foreign_owner.sequence_id().pool_cookie(),
            })
        );
        assert_eq!(
            first_pool.validate_block(old, second.sequence_id()),
            Err(PagedKvError::WrongOwner {
                physical_index: 0,
                expected: second.sequence_id(),
                actual: Some(first.sequence_id()),
            })
        );
        first.reset(&mut first_pool).expect("reset");
        assert_eq!(
            first_pool.validate_block(old, first.sequence_id()),
            Err(PagedKvError::DoubleFree {
                physical_index: 0,
                generation: old.generation(),
            })
        );
        let mut reused = second;
        reserve_and_commit(&mut reused, &mut first_pool, 1);
        let current = reused.block_id(0).expect("reused block");
        assert_eq!(current.physical_index(), old.physical_index());
        assert!(current.generation() > old.generation());
        assert_eq!(
            first_pool.validate_block(old, first.sequence_id()),
            Err(PagedKvError::StaleBlock {
                physical_index: 0,
                handle_generation: old.generation(),
                current_generation: current.generation(),
            })
        );
        assert_eq!(
            other_pool.validate_block(current, reused.sequence_id()),
            Err(PagedKvError::ForeignPool {
                expected_cookie: other_pool.cookie(),
                actual_cookie: first_pool.cookie(),
            })
        );
    }

    #[test]
    fn sidecar_is_optional_versioned_generation_bound_and_invalidated() {
        let mut pool = pool(2);
        let mut sequence = pool.create_sequence(32).expect("sequence");
        reserve_and_commit(&mut sequence, &mut pool, 1);
        assert!(
            sequence
                .sidecar(&pool, 0)
                .expect("absent sidecar")
                .is_none()
        );

        let empty = OptionalBlockMetadata::new(
            OPTIONAL_BLOCK_METADATA_V1_VERSION,
            MetadataKind::KEY_BOUNDS,
            OpaqueDeviceView::new(0, 0),
        );
        sequence
            .attach_sidecar(&mut pool, 0, empty)
            .expect("empty V1 sidecar");
        let view = sequence
            .sidecar(&pool, 0)
            .expect("sidecar lookup")
            .expect("sidecar");
        assert!(view.metadata().device_view().is_empty());
        assert_eq!(pool.stats().sidecar_count(), 1);

        let unknown =
            OptionalBlockMetadata::new(9, MetadataKind::from_raw(99), OpaqueDeviceView::new(7, 11));
        assert_eq!(
            sequence.attach_sidecar(&mut pool, 0, unknown),
            Err(PagedKvError::UnsupportedMetadataVersion { version: 9 })
        );
        assert_eq!(pool.stats().sidecar_count(), 1);

        reserve_and_commit(&mut sequence, &mut pool, 2);
        assert!(
            sequence
                .sidecar(&pool, 0)
                .expect("invalidated lookup")
                .is_none()
        );
        sequence
            .attach_sidecar(
                &mut pool,
                0,
                OptionalBlockMetadata::new(
                    OPTIONAL_BLOCK_METADATA_V1_VERSION,
                    MetadataKind::VALUE_NORM_BOUND,
                    OpaqueDeviceView::new(0x1000, 64),
                ),
            )
            .expect("attach sidecar");
        let old = sequence.block_id(0).expect("old block");
        sequence.reset(&mut pool).expect("free invalidates sidecar");
        assert_eq!(pool.stats().sidecar_count(), 0);
        assert_eq!(pool.stats().sidecar_device_bytes(), 0);

        let mut reused = pool.create_sequence(16).expect("reuse sequence");
        reserve_and_commit(&mut reused, &mut pool, 1);
        assert_eq!(reused.block_id(0).expect("new block").physical_index(), 0);
        assert!(reused.sidecar(&pool, 0).expect("fresh block").is_none());
        assert!(matches!(
            pool.validate_block(old, sequence.sequence_id()),
            Err(PagedKvError::StaleBlock { .. })
        ));
    }

    #[test]
    fn lost_reservation_requires_poisoned_recovery_then_reset() {
        let mut pool = pool(2);
        let mut sequence = pool.create_sequence(32).expect("sequence");
        let reservation = sequence.reserve_to(&mut pool, 17).expect("reserve");
        drop(reservation);
        assert!(matches!(
            sequence.reserve_to(&mut pool, 1),
            Err(PagedKvError::ReservationInProgress)
        ));
        sequence
            .rollback_abandoned_reservation(&mut pool)
            .expect("recover");
        assert!(sequence.is_poisoned());
        assert_eq!(sequence.allocated_block_count(), 0);
        assert_eq!(sequence.block_table(), Err(PagedKvError::Poisoned));
        sequence.reset(&mut pool).expect("reset poison");
        assert!(!sequence.is_poisoned());
        reserve_and_commit(&mut sequence, &mut pool, 1);
    }

    #[test]
    fn truncate_reset_close_and_dropped_owner_account_exactly() {
        let mut pool = pool(6);
        let mut sequence = pool.create_sequence(64).expect("sequence");
        reserve_and_commit(&mut sequence, &mut pool, 49);
        assert_eq!(pool.stats().allocated_block_count(), 4);
        let mut wrong_pool = self::pool(1);
        assert_eq!(
            sequence.close(&mut wrong_pool),
            Err(PagedKvError::ForeignPool {
                expected_cookie: pool.cookie(),
                actual_cookie: wrong_pool.cookie(),
            })
        );
        assert_eq!(pool.stats().allocated_block_count(), 4);
        assert_eq!(
            sequence
                .block_table()
                .expect("close failure preserves owner")
                .logical_length(),
            49
        );
        sequence.truncate_to(&mut pool, 16).expect("truncate");
        assert_eq!(sequence.block_table().expect("table").valid_tokens(), &[16]);
        assert_eq!(pool.stats().allocated_block_count(), 1);
        sequence.close(&mut pool).expect("close");
        assert_eq!(pool.stats().allocated_block_count(), 0);

        let mut dropped = pool.create_sequence(32).expect("dropped owner");
        reserve_and_commit(&mut dropped, &mut pool, 17);
        let reclaim = dropped.abandon_for_reclaim();
        assert_eq!(pool.stats().allocated_block_count(), 2);
        assert_eq!(pool.reclaim_sequence(&reclaim).expect("reclaim orphan"), 2);
        assert_eq!(
            pool.reclaim_sequence(&reclaim).expect("idempotent reclaim"),
            0
        );
        assert_eq!(pool.stats().allocated_block_count(), 0);
        assert_eq!(pool.stats().free_block_count(), 6);

        let mut foreign_pool = self::pool(1);
        let mut foreign = foreign_pool.create_sequence(16).expect("foreign owner");
        reserve_and_commit(&mut foreign, &mut foreign_pool, 1);
        let foreign_cookie = foreign.sequence_id().pool_cookie();
        let foreign_reclaim = foreign.abandon_for_reclaim();
        assert_eq!(
            pool.reclaim_sequence(&foreign_reclaim),
            Err(PagedKvError::ForeignPool {
                expected_cookie: pool.cookie(),
                actual_cookie: foreign_cookie,
            })
        );
        assert_eq!(pool.stats().allocated_block_count(), 0);
        assert_eq!(
            foreign_pool
                .reclaim_sequence(&foreign_reclaim)
                .expect("foreign token remains usable with its owner pool"),
            1
        );
    }

    #[test]
    fn deterministic_property_sequence_preserves_accounting_and_tables() {
        let mut pool = pool(12);
        let mut sequence = pool.create_sequence(129).expect("sequence");
        let mut state = 0x5eed_u64;
        for _ in 0..512 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let target = usize::try_from(state % 130).expect("small target");
            if target >= sequence.logical_length() as usize {
                let reservation = sequence.reserve_to(&mut pool, target).expect("reserve");
                if state & 1 == 0 {
                    sequence.commit(&mut pool, reservation).expect("commit");
                } else {
                    sequence.rollback(&mut pool, reservation).expect("rollback");
                }
            } else {
                sequence.truncate_to(&mut pool, target).expect("truncate");
            }
            let table = sequence.block_table().expect("committed table");
            assert_eq!(
                table.block_count(),
                blocks_for_length(table.logical_length() as usize)
            );
            assert_eq!(table.physical_block_ids().len(), table.valid_tokens().len());
            assert_eq!(pool.stats().allocated_block_count(), table.block_count());
            assert_eq!(
                pool.stats().allocated_block_count() + pool.stats().free_block_count(),
                pool.stats().physical_block_count()
            );
            assert!(
                table
                    .valid_tokens()
                    .iter()
                    .all(|&valid| valid > 0 && valid <= 16)
            );
            assert_eq!(
                table
                    .valid_tokens()
                    .iter()
                    .map(|&value| u32::from(value))
                    .sum::<u32>(),
                table.logical_length()
            );
        }
        sequence.reset(&mut pool).expect("reset");
        assert_eq!(pool.stats().allocated_block_count(), 0);
    }

    #[test]
    fn hot_reservation_source_contains_no_heap_preparation() {
        let source = include_str!("paged_kv.rs");
        let begin = source
            .find("// HOT_RESERVATION_BEGIN")
            .expect("hot reservation marker");
        let end = source
            .find("// HOT_RESERVATION_END")
            .expect("hot reservation end marker");
        let hot = &source[begin..end];
        for forbidden in [
            "Vec::new",
            "Box::new",
            "try_reserve",
            ".resize(",
            ".collect(",
            "format!",
        ] {
            assert!(
                !hot.contains(forbidden),
                "hot reservation source contains allocation token {forbidden:?}"
            );
        }
    }
}
