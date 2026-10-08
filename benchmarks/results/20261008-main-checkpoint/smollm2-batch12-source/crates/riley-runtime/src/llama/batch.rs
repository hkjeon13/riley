//! Bounded host metadata for one mixed Llama prefill/decode iteration.
//!
//! This module deliberately stops at the scheduler-to-kernel boundary. It
//! validates borrowed row descriptions and packs them into preallocated CSR
//! buffers, but it neither owns scheduler state nor invokes a single-request
//! executor. Device upload and batched CUDA kernels can consume the resulting
//! [`LlamaPackedBatchMetadata`] without changing this host contract.

use std::error;
use std::fmt;

use crate::paged_kv::{BLOCK_TABLE_V1_VERSION, BlockTableV1, KV_BLOCK_SIZE};

/// Version of the packed Llama batch metadata contract.
pub const LLAMA_BATCH_METADATA_V1_VERSION: u16 = 1;

/// Per-row output-slot marker used when no sampled output is requested.
///
/// Valid output slots are dense zero-based indices and can therefore never be
/// this value.
pub const LLAMA_BATCH_NO_OUTPUT_SLOT: u32 = u32::MAX;

/// Result type for bounded Llama batch metadata preparation and packing.
pub type LlamaBatchResult<T> = Result<T, LlamaBatchError>;

/// Stable failure from bounded host batch preparation or validation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaBatchError {
    /// A cold preparation bound is zero, inconsistent, or not transportable.
    InvalidConfiguration {
        /// Configuration field that failed validation.
        field: &'static str,
        /// Stable explanation of the violated invariant.
        reason: &'static str,
    },
    /// One iteration exceeded a cold-prepared capacity.
    CapacityExceeded {
        /// Bounded resource that was exceeded.
        resource: &'static str,
        /// Elements required by the attempted iteration.
        requested: usize,
        /// Elements available after preparation.
        capacity: usize,
    },
    /// A checked count, offset, or position calculation overflowed.
    ArithmeticOverflow {
        /// Calculation that could not be represented.
        field: &'static str,
    },
    /// A cold host allocation could not reserve its exact element capacity.
    HostAllocation {
        /// Buffer whose allocation failed.
        resource: &'static str,
        /// Elements requested from the allocator.
        requested_elements: usize,
    },
    /// An input transport object used an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Transport object whose version was rejected.
        resource: &'static str,
        /// Version accepted by this runtime build.
        expected: u16,
        /// Version supplied by the caller.
        actual: u16,
    },
    /// Borrowed row metadata violated a structural execution invariant.
    InvalidBatch {
        /// Field or relation that failed validation.
        field: &'static str,
        /// Stable explanation of the violated invariant.
        reason: &'static str,
    },
}

impl fmt::Display for LlamaBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid Llama batch configuration {field}: {reason}"
                )
            }
            Self::CapacityExceeded {
                resource,
                requested,
                capacity,
            } => write!(
                formatter,
                "Llama batch {resource} capacity exceeded: requested={requested} capacity={capacity}"
            ),
            Self::ArithmeticOverflow { field } => {
                write!(formatter, "Llama batch arithmetic overflow for {field}")
            }
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "could not reserve {requested_elements} host elements for Llama batch {resource}"
            ),
            Self::UnsupportedSchemaVersion {
                resource,
                expected,
                actual,
            } => write!(
                formatter,
                "unsupported {resource} schema version: expected={expected} actual={actual}"
            ),
            Self::InvalidBatch { field, reason } => {
                write!(formatter, "invalid Llama batch {field}: {reason}")
            }
        }
    }
}

impl error::Error for LlamaBatchError {}

/// Work performed by one packed batch row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LlamaBatchRowKind {
    /// One bounded prompt-ingestion chunk.
    Prefill = 0,
    /// One autoregressive input token for an active sequence.
    Decode = 1,
}

impl LlamaBatchRowKind {
    /// Stable byte code stored in the packed row-kind vector.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Borrowed scheduler-independent V1 block-table transport fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaBatchBlockTable<'a> {
    schema_version: u16,
    physical_block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    logical_length: u32,
}

impl<'a> LlamaBatchBlockTable<'a> {
    /// Describes borrowed block-table fields for later validation by
    /// [`PreparedLlamaBatchMetadata::pack`].
    #[must_use]
    pub const fn new(
        schema_version: u16,
        physical_block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        logical_length: u32,
    ) -> Self {
        Self {
            schema_version,
            physical_block_ids,
            valid_tokens,
            logical_length,
        }
    }

    /// Adapts the runtime's canonical borrowed V1 table without allocation.
    #[must_use]
    pub const fn from_v1(table: BlockTableV1<'a>) -> Self {
        Self::new(
            table.version(),
            table.physical_block_ids(),
            table.valid_tokens(),
            table.logical_length(),
        )
    }

    /// Block-table transport schema version.
    #[must_use]
    pub const fn schema_version(self) -> u16 {
        self.schema_version
    }

    /// Physical block indices in logical-token order.
    #[must_use]
    pub const fn physical_block_ids(self) -> &'a [u32] {
        self.physical_block_ids
    }

    /// Valid-token counts corresponding one-for-one with physical blocks.
    #[must_use]
    pub const fn valid_tokens(self) -> &'a [u16] {
        self.valid_tokens
    }

    /// Logical length addressed by the table.
    #[must_use]
    pub const fn logical_length(self) -> u32 {
        self.logical_length
    }
}

/// Borrowed description of one sequence row in an iteration.
///
/// `sequence_tag` is opaque to the runtime. It exists only to reject accidental
/// duplicate rows and to preserve deterministic host-side routing metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaBatchRow<'a> {
    sequence_tag: u64,
    kind: LlamaBatchRowKind,
    input_token_ids: &'a [u32],
    target_logical_length: u32,
    block_table: LlamaBatchBlockTable<'a>,
    output_slot: Option<u32>,
}

impl<'a> LlamaBatchRow<'a> {
    /// Describes one row for later validation and packing.
    #[must_use]
    pub const fn new(
        sequence_tag: u64,
        kind: LlamaBatchRowKind,
        input_token_ids: &'a [u32],
        target_logical_length: u32,
        block_table: LlamaBatchBlockTable<'a>,
        output_slot: Option<u32>,
    ) -> Self {
        Self {
            sequence_tag,
            kind,
            input_token_ids,
            target_logical_length,
            block_table,
            output_slot,
        }
    }

    /// Opaque caller-provided sequence identity.
    #[must_use]
    pub const fn sequence_tag(self) -> u64 {
        self.sequence_tag
    }

    /// Prefill or decode work represented by this row.
    #[must_use]
    pub const fn kind(self) -> LlamaBatchRowKind {
        self.kind
    }

    /// Input token slice for this iteration only.
    #[must_use]
    pub const fn input_token_ids(self) -> &'a [u32] {
        self.input_token_ids
    }

    /// Logical sequence length after successful execution.
    #[must_use]
    pub const fn target_logical_length(self) -> u32 {
        self.target_logical_length
    }

    /// Borrowed physical block-table snapshot for the target length.
    #[must_use]
    pub const fn block_table(self) -> LlamaBatchBlockTable<'a> {
        self.block_table
    }

    /// Dense output-routing slot, or none for a non-sampling prefill chunk.
    #[must_use]
    pub const fn output_slot(self) -> Option<u32> {
        self.output_slot
    }
}

/// Cold bounds for reusable packed batch metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaBatchMetadataConfig {
    max_rows: usize,
    max_input_tokens: usize,
    max_block_entries: usize,
    max_output_slots: usize,
    physical_block_count: usize,
}

impl LlamaBatchMetadataConfig {
    /// Validates explicit upper bounds for all reusable host buffers.
    ///
    /// `max_output_slots` may be zero for prefill-only metadata. Every other
    /// capacity must be positive. Physical blocks are exclusive across rows in
    /// PR13 because prefix sharing is outside the current contract.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaBatchError::InvalidConfiguration`] when the capacities
    /// cannot represent at least one non-empty token and block table per row,
    /// do not fit V1 `u32` offsets, or permit more output slots than rows.
    pub fn new(
        max_rows: usize,
        max_input_tokens: usize,
        max_block_entries: usize,
        max_output_slots: usize,
        physical_block_count: usize,
    ) -> LlamaBatchResult<Self> {
        validate_positive("max_rows", max_rows)?;
        validate_positive("max_input_tokens", max_input_tokens)?;
        validate_positive("max_block_entries", max_block_entries)?;
        validate_positive("physical_block_count", physical_block_count)?;
        validate_u32_bound("max_rows", max_rows)?;
        validate_u32_bound("max_input_tokens", max_input_tokens)?;
        validate_u32_bound("max_block_entries", max_block_entries)?;
        validate_u32_bound("max_output_slots", max_output_slots)?;
        validate_u32_bound("physical_block_count", physical_block_count)?;
        if max_input_tokens < max_rows {
            return Err(LlamaBatchError::InvalidConfiguration {
                field: "max_input_tokens",
                reason: "must permit at least one input token per row",
            });
        }
        if max_block_entries < max_rows {
            return Err(LlamaBatchError::InvalidConfiguration {
                field: "max_block_entries",
                reason: "must permit at least one physical block per row",
            });
        }
        if max_output_slots > max_rows {
            return Err(LlamaBatchError::InvalidConfiguration {
                field: "max_output_slots",
                reason: "cannot exceed max_rows",
            });
        }
        if max_block_entries > physical_block_count {
            return Err(LlamaBatchError::InvalidConfiguration {
                field: "max_block_entries",
                reason: "cannot exceed the exclusive physical block pool",
            });
        }
        max_rows
            .checked_add(1)
            .ok_or(LlamaBatchError::ArithmeticOverflow {
                field: "row offset capacity",
            })?;
        Ok(Self {
            max_rows,
            max_input_tokens,
            max_block_entries,
            max_output_slots,
            physical_block_count,
        })
    }

    /// Maximum sequence rows in one packed iteration.
    #[must_use]
    pub const fn max_rows(self) -> usize {
        self.max_rows
    }

    /// Maximum flattened input tokens in one packed iteration.
    #[must_use]
    pub const fn max_input_tokens(self) -> usize {
        self.max_input_tokens
    }

    /// Maximum flattened block-table entries in one packed iteration.
    #[must_use]
    pub const fn max_block_entries(self) -> usize {
        self.max_block_entries
    }

    /// Maximum sampled output slots in one packed iteration.
    #[must_use]
    pub const fn max_output_slots(self) -> usize {
        self.max_output_slots
    }

    /// Number of addressable exclusive physical KV blocks.
    #[must_use]
    pub const fn physical_block_count(self) -> usize {
        self.physical_block_count
    }
}

/// Prepared logical capacities reported without exposing allocator internals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaBatchBufferCapacities {
    rows: usize,
    input_tokens: usize,
    row_sequence_slots: usize,
    block_entries: usize,
    output_slots: usize,
    physical_block_marks: usize,
}

impl LlamaBatchBufferCapacities {
    /// Capacity shared by every per-row buffer.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Capacity shared by flattened token and position buffers.
    #[must_use]
    pub const fn input_tokens(self) -> usize {
        self.input_tokens
    }

    /// Capacity of the flattened token-to-row index buffer.
    #[must_use]
    pub const fn row_sequence_slots(self) -> usize {
        self.row_sequence_slots
    }

    /// Capacity shared by flattened physical-ID and valid-token buffers.
    #[must_use]
    pub const fn block_entries(self) -> usize {
        self.block_entries
    }

    /// Capacity shared by slot-to-row and slot-to-token buffers.
    #[must_use]
    pub const fn output_slots(self) -> usize {
        self.output_slots
    }

    /// Size of the reusable physical-block duplicate-detection map.
    #[must_use]
    pub const fn physical_block_marks(self) -> usize {
        self.physical_block_marks
    }
}

/// Reusable owner of all host buffers needed to pack one bounded iteration.
///
/// [`Self::prepare`] is the only allocating operation. Once it succeeds,
/// [`Self::pack`] clears and reuses the same buffers and performs no host
/// allocation for either valid or rejected batches.
#[derive(Debug)]
pub struct PreparedLlamaBatchMetadata {
    config: LlamaBatchMetadataConfig,
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
    validation_sequence_tags: Vec<u64>,
    output_slot_marks: Vec<u64>,
    physical_block_marks: Vec<u64>,
    validation_epoch: u64,
}

impl PreparedLlamaBatchMetadata {
    /// Allocates every bounded host buffer required by [`Self::pack`].
    ///
    /// # Errors
    ///
    /// Returns [`LlamaBatchError::HostAllocation`] if any exact capacity cannot
    /// be reserved. No partially prepared object is returned.
    pub fn prepare(config: LlamaBatchMetadataConfig) -> LlamaBatchResult<Self> {
        let row_offsets =
            config
                .max_rows
                .checked_add(1)
                .ok_or(LlamaBatchError::ArithmeticOverflow {
                    field: "row offset capacity",
                })?;
        Ok(Self {
            config,
            sequence_tags: reserve_vec(config.max_rows, "sequence tags")?,
            row_kind_codes: reserve_vec(config.max_rows, "row kind codes")?,
            input_row_offsets: reserve_vec(row_offsets, "input row offsets")?,
            input_token_ids: reserve_vec(config.max_input_tokens, "input token IDs")?,
            position_ids: reserve_vec(config.max_input_tokens, "position IDs")?,
            row_sequence_slots: reserve_vec(config.max_input_tokens, "token row sequence slots")?,
            block_row_offsets: reserve_vec(row_offsets, "block row offsets")?,
            physical_block_ids: reserve_vec(config.max_block_entries, "physical block IDs")?,
            valid_tokens: reserve_vec(config.max_block_entries, "block valid tokens")?,
            logical_lengths: reserve_vec(config.max_rows, "logical lengths")?,
            output_slots_by_row: reserve_vec(config.max_rows, "per-row output slots")?,
            output_row_indices: reserve_vec(config.max_output_slots, "output row indices")?,
            output_token_indices: reserve_vec(config.max_output_slots, "output token indices")?,
            prefill_row_indices: reserve_vec(config.max_rows, "prefill row indices")?,
            decode_row_indices: reserve_vec(config.max_rows, "decode row indices")?,
            validation_sequence_tags: reserve_vec(config.max_rows, "validation sequence tags")?,
            output_slot_marks: zeroed_vec(config.max_output_slots, "output slot marks")?,
            physical_block_marks: zeroed_vec(config.physical_block_count, "physical block marks")?,
            validation_epoch: 0,
        })
    }

    /// Cold bounds with which this reusable packer was prepared.
    #[must_use]
    pub const fn config(&self) -> LlamaBatchMetadataConfig {
        self.config
    }

    /// Logical capacities guaranteed not to grow during [`Self::pack`].
    #[must_use]
    pub const fn capacities(&self) -> LlamaBatchBufferCapacities {
        LlamaBatchBufferCapacities {
            rows: self.config.max_rows,
            input_tokens: self.config.max_input_tokens,
            row_sequence_slots: self.config.max_input_tokens,
            block_entries: self.config.max_block_entries,
            output_slots: self.config.max_output_slots,
            physical_block_marks: self.config.physical_block_count,
        }
    }

    /// Validates and packs one mixed prefill/decode iteration without allocating.
    ///
    /// Rows retain caller order. Separate prefill/decode row-index vectors make
    /// stable kind-specific dispatch possible without regrouping the packed CSR
    /// arrays. Output slots must be unique and exactly dense in `0..N`; decode
    /// rows always require one slot, while intermediate prefill chunks may omit
    /// one. A slot maps both to its source row and to that row's last flattened
    /// input token, which is the position whose logits are sampled.
    ///
    /// # Errors
    ///
    /// Returns a capacity, schema, arithmetic, or structural error before a
    /// view is issued. A later valid call can always reuse the prepared object.
    pub fn pack<'batch>(
        &'batch mut self,
        rows: &[LlamaBatchRow<'_>],
    ) -> LlamaBatchResult<LlamaPackedBatchMetadata<'batch>> {
        // HOT_BATCH_PACK_BEGIN
        let validated = self.validate(rows)?;
        self.clear_packed();
        self.output_row_indices
            .resize(validated.output_count, LLAMA_BATCH_NO_OUTPUT_SLOT);
        self.output_token_indices
            .resize(validated.output_count, LLAMA_BATCH_NO_OUTPUT_SLOT);
        self.input_row_offsets.push(0);
        self.block_row_offsets.push(0);

        let mut token_offset = 0_u32;
        let mut block_offset = 0_u32;
        for (row_index, row) in rows.iter().enumerate() {
            let row_index_u32 = checked_u32(row_index, "row index")?;
            let input_count = checked_u32(row.input_token_ids.len(), "row input token count")?;
            let block_count =
                checked_u32(row.block_table.physical_block_ids.len(), "row block count")?;
            let start_position = row.target_logical_length.checked_sub(input_count).ok_or(
                LlamaBatchError::ArithmeticOverflow {
                    field: "row start position",
                },
            )?;
            token_offset = token_offset.checked_add(input_count).ok_or(
                LlamaBatchError::ArithmeticOverflow {
                    field: "input CSR offset",
                },
            )?;
            block_offset = block_offset.checked_add(block_count).ok_or(
                LlamaBatchError::ArithmeticOverflow {
                    field: "block CSR offset",
                },
            )?;

            self.sequence_tags.push(row.sequence_tag);
            self.row_kind_codes.push(row.kind.code());
            self.input_token_ids.extend_from_slice(row.input_token_ids);
            self.position_ids
                .extend(start_position..row.target_logical_length);
            for _ in row.input_token_ids {
                self.row_sequence_slots.push(row_index_u32);
            }
            self.input_row_offsets.push(token_offset);
            self.physical_block_ids
                .extend_from_slice(row.block_table.physical_block_ids);
            self.valid_tokens
                .extend_from_slice(row.block_table.valid_tokens);
            self.block_row_offsets.push(block_offset);
            self.logical_lengths.push(row.target_logical_length);

            match row.kind {
                LlamaBatchRowKind::Prefill => self.prefill_row_indices.push(row_index_u32),
                LlamaBatchRowKind::Decode => self.decode_row_indices.push(row_index_u32),
            }

            if let Some(slot) = row.output_slot {
                let slot_index = checked_usize(slot, "output slot index")?;
                self.output_slots_by_row.push(slot);
                self.output_row_indices[slot_index] = row_index_u32;
                self.output_token_indices[slot_index] =
                    token_offset
                        .checked_sub(1)
                        .ok_or(LlamaBatchError::ArithmeticOverflow {
                            field: "output token index",
                        })?;
            } else {
                self.output_slots_by_row.push(LLAMA_BATCH_NO_OUTPUT_SLOT);
            }
        }

        debug_assert_eq!(self.input_token_ids.len(), validated.total_input_tokens);
        debug_assert_eq!(self.position_ids.len(), validated.total_input_tokens);
        debug_assert_eq!(self.row_sequence_slots.len(), validated.total_input_tokens);
        debug_assert_eq!(self.physical_block_ids.len(), validated.total_block_entries);
        debug_assert_eq!(self.valid_tokens.len(), validated.total_block_entries);
        debug_assert_eq!(self.prefill_row_indices.len(), validated.prefill_rows);
        debug_assert_eq!(self.decode_row_indices.len(), validated.decode_rows);
        // HOT_BATCH_PACK_END
        Ok(self.view(validated))
    }

    #[allow(clippy::too_many_lines)]
    fn validate(&mut self, rows: &[LlamaBatchRow<'_>]) -> LlamaBatchResult<ValidatedBatch> {
        if rows.is_empty() {
            return Err(LlamaBatchError::InvalidBatch {
                field: "rows",
                reason: "an iteration must contain at least one row",
            });
        }
        ensure_capacity("rows", rows.len(), self.config.max_rows)?;
        self.advance_validation_epoch();
        let epoch = self.validation_epoch;
        self.validation_sequence_tags.clear();

        let mut total_input_tokens = 0_usize;
        let mut total_block_entries = 0_usize;
        let mut output_count = 0_usize;
        let mut prefill_rows = 0_usize;
        let mut decode_rows = 0_usize;
        let mut prefill_tokens = 0_usize;
        let mut decode_tokens = 0_usize;

        for row in rows {
            if self.validation_sequence_tags.contains(&row.sequence_tag) {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "sequence_tag",
                    reason: "a sequence may appear at most once per iteration",
                });
            }
            self.validation_sequence_tags.push(row.sequence_tag);
            if row.input_token_ids.is_empty() {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "input_token_ids",
                    reason: "every row must contain at least one input token",
                });
            }
            if row.kind == LlamaBatchRowKind::Decode && row.input_token_ids.len() != 1 {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "input_token_ids",
                    reason: "decode rows must contain exactly one input token",
                });
            }
            if row.kind == LlamaBatchRowKind::Decode && row.output_slot.is_none() {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "output_slot",
                    reason: "decode rows must request one sampled output",
                });
            }

            let input_count = row.input_token_ids.len();
            let target_length = checked_usize(row.target_logical_length, "target logical length")?;
            if target_length < input_count {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "target_logical_length",
                    reason: "cannot be shorter than the row input slice",
                });
            }
            total_input_tokens = total_input_tokens.checked_add(input_count).ok_or(
                LlamaBatchError::ArithmeticOverflow {
                    field: "total input tokens",
                },
            )?;
            ensure_capacity(
                "input tokens",
                total_input_tokens,
                self.config.max_input_tokens,
            )?;

            match row.kind {
                LlamaBatchRowKind::Prefill => {
                    prefill_rows =
                        prefill_rows
                            .checked_add(1)
                            .ok_or(LlamaBatchError::ArithmeticOverflow {
                                field: "prefill row count",
                            })?;
                    prefill_tokens = prefill_tokens.checked_add(input_count).ok_or(
                        LlamaBatchError::ArithmeticOverflow {
                            field: "prefill token count",
                        },
                    )?;
                }
                LlamaBatchRowKind::Decode => {
                    decode_rows =
                        decode_rows
                            .checked_add(1)
                            .ok_or(LlamaBatchError::ArithmeticOverflow {
                                field: "decode row count",
                            })?;
                    decode_tokens = decode_tokens.checked_add(input_count).ok_or(
                        LlamaBatchError::ArithmeticOverflow {
                            field: "decode token count",
                        },
                    )?;
                }
            }

            self.validate_block_table(row, target_length, epoch)?;
            total_block_entries = total_block_entries
                .checked_add(row.block_table.physical_block_ids.len())
                .ok_or(LlamaBatchError::ArithmeticOverflow {
                    field: "total block entries",
                })?;
            ensure_capacity(
                "block entries",
                total_block_entries,
                self.config.max_block_entries,
            )?;

            if let Some(slot) = row.output_slot {
                let slot_index = checked_usize(slot, "output slot index")?;
                let requested =
                    slot_index
                        .checked_add(1)
                        .ok_or(LlamaBatchError::ArithmeticOverflow {
                            field: "output slot extent",
                        })?;
                ensure_capacity("output slot index", requested, self.config.max_output_slots)?;
                if self.output_slot_marks[slot_index] == epoch {
                    return Err(LlamaBatchError::InvalidBatch {
                        field: "output_slot",
                        reason: "output slots must be unique within an iteration",
                    });
                }
                self.output_slot_marks[slot_index] = epoch;
                output_count =
                    output_count
                        .checked_add(1)
                        .ok_or(LlamaBatchError::ArithmeticOverflow {
                            field: "output slot count",
                        })?;
            }
        }

        ensure_capacity("output slots", output_count, self.config.max_output_slots)?;
        if self.output_slot_marks[..output_count]
            .iter()
            .any(|&mark| mark != epoch)
        {
            return Err(LlamaBatchError::InvalidBatch {
                field: "output_slot",
                reason: "output slots must be dense and zero-based",
            });
        }

        Ok(ValidatedBatch {
            total_input_tokens,
            total_block_entries,
            output_count,
            prefill_rows,
            decode_rows,
            prefill_tokens,
            decode_tokens,
        })
    }

    fn validate_block_table(
        &mut self,
        row: &LlamaBatchRow<'_>,
        target_length: usize,
        epoch: u64,
    ) -> LlamaBatchResult<()> {
        let table = row.block_table;
        if table.schema_version != BLOCK_TABLE_V1_VERSION {
            return Err(LlamaBatchError::UnsupportedSchemaVersion {
                resource: "Llama batch block table",
                expected: BLOCK_TABLE_V1_VERSION,
                actual: table.schema_version,
            });
        }
        if table.logical_length != row.target_logical_length {
            return Err(LlamaBatchError::InvalidBatch {
                field: "block_table.logical_length",
                reason: "must equal the row target logical length",
            });
        }
        if table.physical_block_ids.len() != table.valid_tokens.len() {
            return Err(LlamaBatchError::InvalidBatch {
                field: "block_table",
                reason: "physical IDs and valid-token vectors must have equal length",
            });
        }
        let required_blocks = target_length.div_ceil(KV_BLOCK_SIZE);
        if table.physical_block_ids.len() != required_blocks {
            return Err(LlamaBatchError::InvalidBatch {
                field: "block_table",
                reason: "block count must exactly address the target logical length",
            });
        }

        let mut valid_token_sum = 0_usize;
        for (block_index, (&physical_id, &valid_tokens)) in table
            .physical_block_ids
            .iter()
            .zip(table.valid_tokens)
            .enumerate()
        {
            let physical_index = checked_usize(physical_id, "physical block index")?;
            if physical_index >= self.config.physical_block_count {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "block_table.physical_block_ids",
                    reason: "physical block ID is outside the prepared pool",
                });
            }
            if self.physical_block_marks[physical_index] == epoch {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "block_table.physical_block_ids",
                    reason: "physical block IDs must be unique across the iteration",
                });
            }
            self.physical_block_marks[physical_index] = epoch;

            let expected_valid = if block_index + 1 == required_blocks {
                let tail = target_length % KV_BLOCK_SIZE;
                if tail == 0 { KV_BLOCK_SIZE } else { tail }
            } else {
                KV_BLOCK_SIZE
            };
            if usize::from(valid_tokens) != expected_valid {
                return Err(LlamaBatchError::InvalidBatch {
                    field: "block_table.valid_tokens",
                    reason: "must contain full blocks followed by one canonical tail",
                });
            }
            valid_token_sum = valid_token_sum
                .checked_add(usize::from(valid_tokens))
                .ok_or(LlamaBatchError::ArithmeticOverflow {
                    field: "block table valid-token sum",
                })?;
        }
        if valid_token_sum != target_length {
            return Err(LlamaBatchError::InvalidBatch {
                field: "block_table.valid_tokens",
                reason: "valid-token sum must equal the target logical length",
            });
        }
        Ok(())
    }

    fn advance_validation_epoch(&mut self) {
        if let Some(next) = self.validation_epoch.checked_add(1) {
            self.validation_epoch = next;
        } else {
            self.output_slot_marks.fill(0);
            self.physical_block_marks.fill(0);
            self.validation_epoch = 1;
        }
    }

    fn clear_packed(&mut self) {
        self.sequence_tags.clear();
        self.row_kind_codes.clear();
        self.input_row_offsets.clear();
        self.input_token_ids.clear();
        self.position_ids.clear();
        self.row_sequence_slots.clear();
        self.block_row_offsets.clear();
        self.physical_block_ids.clear();
        self.valid_tokens.clear();
        self.logical_lengths.clear();
        self.output_slots_by_row.clear();
        self.output_row_indices.clear();
        self.output_token_indices.clear();
        self.prefill_row_indices.clear();
        self.decode_row_indices.clear();
    }

    fn view(&self, validated: ValidatedBatch) -> LlamaPackedBatchMetadata<'_> {
        LlamaPackedBatchMetadata {
            schema_version: LLAMA_BATCH_METADATA_V1_VERSION,
            block_table_schema_version: BLOCK_TABLE_V1_VERSION,
            sequence_tags: &self.sequence_tags,
            row_kind_codes: &self.row_kind_codes,
            input_row_offsets: &self.input_row_offsets,
            input_token_ids: &self.input_token_ids,
            position_ids: &self.position_ids,
            row_sequence_slots: &self.row_sequence_slots,
            block_row_offsets: &self.block_row_offsets,
            physical_block_ids: &self.physical_block_ids,
            valid_tokens: &self.valid_tokens,
            logical_lengths: &self.logical_lengths,
            output_slots_by_row: &self.output_slots_by_row,
            output_row_indices: &self.output_row_indices,
            output_token_indices: &self.output_token_indices,
            prefill_row_indices: &self.prefill_row_indices,
            decode_row_indices: &self.decode_row_indices,
            prefill_token_count: validated.prefill_tokens,
            decode_token_count: validated.decode_tokens,
        }
    }
}

/// Borrowed packed CSR view valid until the reusable owner is packed again.
///
/// `input_row_offsets` indexes `input_token_ids`, `position_ids`, and
/// `row_sequence_slots`.
/// `block_row_offsets` indexes both `physical_block_ids` and `valid_tokens`.
/// Every offsets vector has `row_count + 1` entries and ends at the associated
/// flattened vector length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "packed batch metadata must be consumed before the owner is reused"]
pub struct LlamaPackedBatchMetadata<'a> {
    schema_version: u16,
    block_table_schema_version: u16,
    sequence_tags: &'a [u64],
    row_kind_codes: &'a [u8],
    input_row_offsets: &'a [u32],
    input_token_ids: &'a [u32],
    position_ids: &'a [u32],
    row_sequence_slots: &'a [u32],
    block_row_offsets: &'a [u32],
    physical_block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    logical_lengths: &'a [u32],
    output_slots_by_row: &'a [u32],
    output_row_indices: &'a [u32],
    output_token_indices: &'a [u32],
    prefill_row_indices: &'a [u32],
    decode_row_indices: &'a [u32],
    prefill_token_count: usize,
    decode_token_count: usize,
}

impl<'a> LlamaPackedBatchMetadata<'a> {
    /// Packed metadata schema version.
    #[must_use]
    pub const fn schema_version(self) -> u16 {
        self.schema_version
    }

    /// Block-table schema version required by every packed row.
    #[must_use]
    pub const fn block_table_schema_version(self) -> u16 {
        self.block_table_schema_version
    }

    /// Number of mixed sequence rows.
    #[must_use]
    pub const fn row_count(self) -> usize {
        self.sequence_tags.len()
    }

    /// Opaque sequence identities in caller row order.
    #[must_use]
    pub const fn sequence_tags(self) -> &'a [u64] {
        self.sequence_tags
    }

    /// Stable [`LlamaBatchRowKind::code`] values in caller row order.
    #[must_use]
    pub const fn row_kind_codes(self) -> &'a [u8] {
        self.row_kind_codes
    }

    /// CSR offsets into both token and position arrays.
    #[must_use]
    pub const fn input_row_offsets(self) -> &'a [u32] {
        self.input_row_offsets
    }

    /// Flattened input token IDs in caller row order.
    #[must_use]
    pub const fn input_token_ids(self) -> &'a [u32] {
        self.input_token_ids
    }

    /// Flattened absolute positions aligned one-for-one with input tokens.
    #[must_use]
    pub const fn position_ids(self) -> &'a [u32] {
        self.position_ids
    }

    /// Flattened caller row index aligned one-for-one with input tokens.
    ///
    /// This is the native batch ABI's direct token-to-sequence indirection;
    /// [`Self::input_row_offsets`] remains available for CSR row slicing.
    #[must_use]
    pub const fn row_sequence_slots(self) -> &'a [u32] {
        self.row_sequence_slots
    }

    /// Total flattened input tokens.
    #[must_use]
    pub const fn total_input_tokens(self) -> usize {
        self.input_token_ids.len()
    }

    /// CSR offsets into both block ID and valid-token arrays.
    #[must_use]
    pub const fn block_row_offsets(self) -> &'a [u32] {
        self.block_row_offsets
    }

    /// Flattened physical KV block IDs in caller row order.
    #[must_use]
    pub const fn physical_block_ids(self) -> &'a [u32] {
        self.physical_block_ids
    }

    /// Flattened valid-token counts aligned with physical block IDs.
    #[must_use]
    pub const fn valid_tokens(self) -> &'a [u16] {
        self.valid_tokens
    }

    /// Target logical length for every row.
    #[must_use]
    pub const fn logical_lengths(self) -> &'a [u32] {
        self.logical_lengths
    }

    /// Output slot for every row, using [`LLAMA_BATCH_NO_OUTPUT_SLOT`] when absent.
    #[must_use]
    pub const fn output_slots_by_row(self) -> &'a [u32] {
        self.output_slots_by_row
    }

    /// Dense slot-to-row mapping, indexed directly by output slot.
    #[must_use]
    pub const fn output_row_indices(self) -> &'a [u32] {
        self.output_row_indices
    }

    /// Dense slot-to-flattened-token mapping for logits selection.
    #[must_use]
    pub const fn output_token_indices(self) -> &'a [u32] {
        self.output_token_indices
    }

    /// Number of sampled outputs expected from the iteration.
    #[must_use]
    pub const fn output_count(self) -> usize {
        self.output_row_indices.len()
    }

    /// Caller row indices for prefill work, preserving relative row order.
    #[must_use]
    pub const fn prefill_row_indices(self) -> &'a [u32] {
        self.prefill_row_indices
    }

    /// Caller row indices for decode work, preserving relative row order.
    #[must_use]
    pub const fn decode_row_indices(self) -> &'a [u32] {
        self.decode_row_indices
    }

    /// Number of prefill rows.
    #[must_use]
    pub const fn prefill_row_count(self) -> usize {
        self.prefill_row_indices.len()
    }

    /// Number of decode rows.
    #[must_use]
    pub const fn decode_row_count(self) -> usize {
        self.decode_row_indices.len()
    }

    /// Flattened tokens belonging to prefill rows.
    #[must_use]
    pub const fn prefill_token_count(self) -> usize {
        self.prefill_token_count
    }

    /// Flattened tokens belonging to decode rows.
    #[must_use]
    pub const fn decode_token_count(self) -> usize {
        self.decode_token_count
    }

    /// Input-token slice for one row, or none when `row_index` is out of range.
    #[must_use]
    pub fn input_tokens_for_row(self, row_index: usize) -> Option<&'a [u32]> {
        csr_row(self.input_row_offsets, self.input_token_ids, row_index)
    }

    /// Absolute-position slice for one row, or none when out of range.
    #[must_use]
    pub fn positions_for_row(self, row_index: usize) -> Option<&'a [u32]> {
        csr_row(self.input_row_offsets, self.position_ids, row_index)
    }

    /// Physical block-ID slice for one row, or none when out of range.
    #[must_use]
    pub fn physical_blocks_for_row(self, row_index: usize) -> Option<&'a [u32]> {
        csr_row(self.block_row_offsets, self.physical_block_ids, row_index)
    }

    /// Valid-token slice for one row, or none when out of range.
    #[must_use]
    pub fn valid_tokens_for_row(self, row_index: usize) -> Option<&'a [u16]> {
        csr_row(self.block_row_offsets, self.valid_tokens, row_index)
    }

    /// Decodes one per-row output slot; absent and out-of-range rows return none.
    #[must_use]
    pub fn output_slot_for_row(self, row_index: usize) -> Option<u32> {
        self.output_slots_by_row
            .get(row_index)
            .copied()
            .filter(|&slot| slot != LLAMA_BATCH_NO_OUTPUT_SLOT)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ValidatedBatch {
    total_input_tokens: usize,
    total_block_entries: usize,
    output_count: usize,
    prefill_rows: usize,
    decode_rows: usize,
    prefill_tokens: usize,
    decode_tokens: usize,
}

fn validate_positive(field: &'static str, value: usize) -> LlamaBatchResult<()> {
    if value == 0 {
        Err(LlamaBatchError::InvalidConfiguration {
            field,
            reason: "must be greater than zero",
        })
    } else {
        Ok(())
    }
}

fn validate_u32_bound(field: &'static str, value: usize) -> LlamaBatchResult<()> {
    if u32::try_from(value).is_ok() {
        Ok(())
    } else {
        Err(LlamaBatchError::InvalidConfiguration {
            field,
            reason: "must fit the V1 u32 transport representation",
        })
    }
}

fn ensure_capacity(
    resource: &'static str,
    requested: usize,
    capacity: usize,
) -> LlamaBatchResult<()> {
    if requested <= capacity {
        Ok(())
    } else {
        Err(LlamaBatchError::CapacityExceeded {
            resource,
            requested,
            capacity,
        })
    }
}

fn checked_u32(value: usize, field: &'static str) -> LlamaBatchResult<u32> {
    u32::try_from(value).map_err(|_| LlamaBatchError::ArithmeticOverflow { field })
}

fn checked_usize(value: u32, field: &'static str) -> LlamaBatchResult<usize> {
    usize::try_from(value).map_err(|_| LlamaBatchError::ArithmeticOverflow { field })
}

fn reserve_vec<T>(capacity: usize, resource: &'static str) -> LlamaBatchResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| LlamaBatchError::HostAllocation {
            resource,
            requested_elements: capacity,
        })?;
    Ok(values)
}

fn zeroed_vec(capacity: usize, resource: &'static str) -> LlamaBatchResult<Vec<u64>> {
    let mut values = reserve_vec(capacity, resource)?;
    values.resize(capacity, 0);
    Ok(values)
}

fn csr_row<'a, T>(offsets: &[u32], values: &'a [T], row_index: usize) -> Option<&'a [T]> {
    let start = checked_usize(*offsets.get(row_index)?, "CSR row start").ok()?;
    let end_index = row_index.checked_add(1)?;
    let end = checked_usize(*offsets.get(end_index)?, "CSR row end").ok()?;
    values.get(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: u16 = 16;

    fn config() -> LlamaBatchMetadataConfig {
        LlamaBatchMetadataConfig::new(4, 8, 8, 4, 16).expect("valid config")
    }

    #[test]
    fn configuration_rejects_unrepresentable_or_inconsistent_bounds() {
        assert!(matches!(
            LlamaBatchMetadataConfig::new(0, 1, 1, 0, 1),
            Err(LlamaBatchError::InvalidConfiguration {
                field: "max_rows",
                ..
            })
        ));
        assert!(matches!(
            LlamaBatchMetadataConfig::new(2, 1, 2, 1, 2),
            Err(LlamaBatchError::InvalidConfiguration {
                field: "max_input_tokens",
                ..
            })
        ));
        assert!(matches!(
            LlamaBatchMetadataConfig::new(2, 2, 2, 3, 2),
            Err(LlamaBatchError::InvalidConfiguration {
                field: "max_output_slots",
                ..
            })
        ));
        assert!(matches!(
            LlamaBatchMetadataConfig::new(2, 2, 3, 2, 2),
            Err(LlamaBatchError::InvalidConfiguration {
                field: "max_block_entries",
                ..
            })
        ));
    }

    #[test]
    fn mixed_rows_pack_csr_positions_and_dense_output_routes() {
        let prefill_tokens = [10, 11, 12];
        let prefill_ids = [2];
        let prefill_valid = [3];
        let decode_tokens = [20];
        let decode_ids = [4, 5];
        let decode_valid = [FULL, 1];
        let rows = [
            LlamaBatchRow::new(
                41,
                LlamaBatchRowKind::Prefill,
                &prefill_tokens,
                3,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &prefill_ids, &prefill_valid, 3),
                Some(1),
            ),
            LlamaBatchRow::new(
                42,
                LlamaBatchRowKind::Decode,
                &decode_tokens,
                17,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &decode_ids, &decode_valid, 17),
                Some(0),
            ),
        ];
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config()).expect("prepare");
        let packed = prepared.pack(&rows).expect("pack");

        assert_eq!(packed.schema_version(), LLAMA_BATCH_METADATA_V1_VERSION);
        assert_eq!(packed.block_table_schema_version(), BLOCK_TABLE_V1_VERSION);
        assert_eq!(packed.row_count(), 2);
        assert_eq!(packed.sequence_tags(), &[41, 42]);
        assert_eq!(
            packed.row_kind_codes(),
            &[
                LlamaBatchRowKind::Prefill.code(),
                LlamaBatchRowKind::Decode.code()
            ]
        );
        assert_eq!(packed.input_row_offsets(), &[0, 3, 4]);
        assert_eq!(packed.input_token_ids(), &[10, 11, 12, 20]);
        assert_eq!(packed.position_ids(), &[0, 1, 2, 16]);
        assert_eq!(packed.row_sequence_slots(), &[0, 0, 0, 1]);
        assert_eq!(packed.block_row_offsets(), &[0, 1, 3]);
        assert_eq!(packed.physical_block_ids(), &[2, 4, 5]);
        assert_eq!(packed.valid_tokens(), &[3, FULL, 1]);
        assert_eq!(packed.logical_lengths(), &[3, 17]);
        assert_eq!(packed.output_slots_by_row(), &[1, 0]);
        assert_eq!(packed.output_row_indices(), &[1, 0]);
        assert_eq!(packed.output_token_indices(), &[3, 2]);
        assert_eq!(packed.prefill_row_indices(), &[0]);
        assert_eq!(packed.decode_row_indices(), &[1]);
        assert_eq!(packed.prefill_token_count(), 3);
        assert_eq!(packed.decode_token_count(), 1);
        assert_eq!(packed.input_tokens_for_row(0), Some(&prefill_tokens[..]));
        assert_eq!(packed.positions_for_row(1), Some(&[16][..]));
        assert_eq!(packed.physical_blocks_for_row(1), Some(&decode_ids[..]));
        assert_eq!(packed.valid_tokens_for_row(1), Some(&decode_valid[..]));
        assert_eq!(packed.output_slot_for_row(0), Some(1));
        assert_eq!(packed.output_slot_for_row(3), None);
    }

    #[test]
    fn intermediate_prefill_can_pack_without_output_capacity() {
        let cfg = LlamaBatchMetadataConfig::new(1, 2, 1, 0, 1).expect("config");
        let tokens = [7, 8];
        let ids = [0];
        let valid = [2];
        let rows = [LlamaBatchRow::new(
            7,
            LlamaBatchRowKind::Prefill,
            &tokens,
            2,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids, &valid, 2),
            None,
        )];
        let mut prepared = PreparedLlamaBatchMetadata::prepare(cfg).expect("prepare");
        let packed = prepared.pack(&rows).expect("pack");

        assert_eq!(packed.output_count(), 0);
        assert_eq!(packed.output_slots_by_row(), &[LLAMA_BATCH_NO_OUTPUT_SLOT]);
        assert_eq!(packed.output_slot_for_row(0), None);
    }

    #[test]
    fn rejects_decode_shape_and_output_slot_errors() {
        let tokens = [1, 2];
        let ids = [0];
        let valid = [2];
        let table = LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids, &valid, 2);
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config()).expect("prepare");

        let wrong_width = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Decode,
            &tokens,
            2,
            table,
            Some(0),
        )];
        assert!(matches!(
            prepared.pack(&wrong_width),
            Err(LlamaBatchError::InvalidBatch {
                field: "input_token_ids",
                ..
            })
        ));

        let one = [1];
        let one_valid = [1];
        let decode_table = LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids, &one_valid, 1);
        let missing_slot = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Decode,
            &one,
            1,
            decode_table,
            None,
        )];
        assert!(matches!(
            prepared.pack(&missing_slot),
            Err(LlamaBatchError::InvalidBatch {
                field: "output_slot",
                ..
            })
        ));

        let ids_b = [1];
        let rows = [
            LlamaBatchRow::new(
                1,
                LlamaBatchRowKind::Prefill,
                &one,
                1,
                decode_table,
                Some(0),
            ),
            LlamaBatchRow::new(
                2,
                LlamaBatchRowKind::Prefill,
                &one,
                1,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids_b, &one_valid, 1),
                Some(2),
            ),
        ];
        assert!(matches!(
            prepared.pack(&rows),
            Err(LlamaBatchError::InvalidBatch {
                field: "output_slot",
                reason: "output slots must be dense and zero-based"
            })
        ));
    }

    #[test]
    fn rejects_noncanonical_or_aliased_block_tables() {
        let token = [1];
        let ids = [0];
        let wrong_valid = [2];
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config()).expect("prepare");
        let wrong_tail = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Prefill,
            &token,
            1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids, &wrong_valid, 1),
            Some(0),
        )];
        assert!(matches!(
            prepared.pack(&wrong_tail),
            Err(LlamaBatchError::InvalidBatch {
                field: "block_table.valid_tokens",
                ..
            })
        ));

        let valid = [1];
        let rows = [
            LlamaBatchRow::new(
                1,
                LlamaBatchRowKind::Prefill,
                &token,
                1,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids, &valid, 1),
                Some(0),
            ),
            LlamaBatchRow::new(
                2,
                LlamaBatchRowKind::Prefill,
                &token,
                1,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids, &valid, 1),
                Some(1),
            ),
        ];
        assert!(matches!(
            prepared.pack(&rows),
            Err(LlamaBatchError::InvalidBatch {
                field: "block_table.physical_block_ids",
                reason: "physical block IDs must be unique across the iteration"
            })
        ));

        let foreign = [16];
        let outside_pool = [LlamaBatchRow::new(
            3,
            LlamaBatchRowKind::Prefill,
            &token,
            1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &foreign, &valid, 1),
            Some(0),
        )];
        assert!(matches!(
            prepared.pack(&outside_pool),
            Err(LlamaBatchError::InvalidBatch {
                field: "block_table.physical_block_ids",
                reason: "physical block ID is outside the prepared pool"
            })
        ));
    }

    #[test]
    fn rejects_duplicate_sequences_and_capacity_overflow() {
        let token = [1];
        let valid = [1];
        let ids_a = [0];
        let ids_b = [1];
        let duplicate = [
            LlamaBatchRow::new(
                9,
                LlamaBatchRowKind::Prefill,
                &token,
                1,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids_a, &valid, 1),
                None,
            ),
            LlamaBatchRow::new(
                9,
                LlamaBatchRowKind::Prefill,
                &token,
                1,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids_b, &valid, 1),
                None,
            ),
        ];
        let small = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 2).expect("config");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(small).expect("prepare");
        assert!(matches!(
            prepared.pack(&duplicate),
            Err(LlamaBatchError::CapacityExceeded {
                resource: "rows",
                requested: 2,
                capacity: 1
            })
        ));

        let mut prepared = PreparedLlamaBatchMetadata::prepare(config()).expect("prepare");
        assert!(matches!(
            prepared.pack(&duplicate),
            Err(LlamaBatchError::InvalidBatch {
                field: "sequence_tag",
                ..
            })
        ));
    }

    #[test]
    fn repeated_pack_reuses_every_prepared_allocation() {
        let tokens_a = [1, 2];
        let ids_a = [0];
        let valid_a = [2];
        let tokens_b = [3];
        let ids_b = [1];
        let valid_b = [1];
        let rows = [
            LlamaBatchRow::new(
                1,
                LlamaBatchRowKind::Prefill,
                &tokens_a,
                2,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids_a, &valid_a, 2),
                Some(0),
            ),
            LlamaBatchRow::new(
                2,
                LlamaBatchRowKind::Decode,
                &tokens_b,
                1,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &ids_b, &valid_b, 1),
                Some(1),
            ),
        ];
        let mut prepared = PreparedLlamaBatchMetadata::prepare(config()).expect("prepare");
        let sequence_ptr = prepared.sequence_tags.as_ptr();
        let kind_ptr = prepared.row_kind_codes.as_ptr();
        let input_offsets_ptr = prepared.input_row_offsets.as_ptr();
        let token_ptr = prepared.input_token_ids.as_ptr();
        let position_ptr = prepared.position_ids.as_ptr();
        let row_sequence_slot_ptr = prepared.row_sequence_slots.as_ptr();
        let block_offsets_ptr = prepared.block_row_offsets.as_ptr();
        let block_ptr = prepared.physical_block_ids.as_ptr();
        let valid_ptr = prepared.valid_tokens.as_ptr();
        let logical_ptr = prepared.logical_lengths.as_ptr();
        let slot_ptr = prepared.output_slots_by_row.as_ptr();
        let output_row_ptr = prepared.output_row_indices.as_ptr();
        let output_token_ptr = prepared.output_token_indices.as_ptr();
        let prefill_ptr = prepared.prefill_row_indices.as_ptr();
        let decode_ptr = prepared.decode_row_indices.as_ptr();
        let sequence_validation_ptr = prepared.validation_sequence_tags.as_ptr();
        let slot_marks_ptr = prepared.output_slot_marks.as_ptr();
        let block_marks_ptr = prepared.physical_block_marks.as_ptr();

        for _ in 0..32 {
            let packed = prepared.pack(&rows).expect("pack");
            assert_eq!(packed.total_input_tokens(), 3);
            assert_eq!(packed.output_count(), 2);
        }

        assert_eq!(sequence_ptr, prepared.sequence_tags.as_ptr());
        assert_eq!(kind_ptr, prepared.row_kind_codes.as_ptr());
        assert_eq!(input_offsets_ptr, prepared.input_row_offsets.as_ptr());
        assert_eq!(token_ptr, prepared.input_token_ids.as_ptr());
        assert_eq!(position_ptr, prepared.position_ids.as_ptr());
        assert_eq!(row_sequence_slot_ptr, prepared.row_sequence_slots.as_ptr());
        assert_eq!(block_offsets_ptr, prepared.block_row_offsets.as_ptr());
        assert_eq!(block_ptr, prepared.physical_block_ids.as_ptr());
        assert_eq!(valid_ptr, prepared.valid_tokens.as_ptr());
        assert_eq!(logical_ptr, prepared.logical_lengths.as_ptr());
        assert_eq!(slot_ptr, prepared.output_slots_by_row.as_ptr());
        assert_eq!(output_row_ptr, prepared.output_row_indices.as_ptr());
        assert_eq!(output_token_ptr, prepared.output_token_indices.as_ptr());
        assert_eq!(prefill_ptr, prepared.prefill_row_indices.as_ptr());
        assert_eq!(decode_ptr, prepared.decode_row_indices.as_ptr());
        assert_eq!(
            sequence_validation_ptr,
            prepared.validation_sequence_tags.as_ptr()
        );
        assert_eq!(slot_marks_ptr, prepared.output_slot_marks.as_ptr());
        assert_eq!(block_marks_ptr, prepared.physical_block_marks.as_ptr());
        assert_eq!(prepared.capacities().rows(), 4);
        assert_eq!(prepared.capacities().input_tokens(), 8);
        assert_eq!(prepared.capacities().row_sequence_slots(), 8);
    }

    #[test]
    fn hot_pack_source_contains_no_preparation_or_allocation() {
        let source = include_str!("batch.rs");
        let begin = source
            .find("// HOT_BATCH_PACK_BEGIN")
            .expect("hot batch pack begin marker");
        let end = source
            .find("// HOT_BATCH_PACK_END")
            .expect("hot batch pack end marker");
        let hot = &source[begin..end];

        for forbidden in [
            "Vec::",
            "Box::",
            "vec!",
            ".collect(",
            "String::",
            "format!",
            "HashMap",
            "HashSet",
            "try_reserve",
        ] {
            assert!(
                !hot.contains(forbidden),
                "hot batch pack source contains forbidden allocation token {forbidden:?}"
            );
        }
        assert!(
            hot.contains("self.row_sequence_slots.push(row_index_u32)"),
            "hot pack must emit the native token-to-sequence row map"
        );
        assert!(
            hot.contains("self.input_row_offsets.push(token_offset)"),
            "hot pack must retain the host CSR token offsets"
        );
    }
}
