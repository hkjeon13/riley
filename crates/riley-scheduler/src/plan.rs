//! Immutable, owned scheduler-to-runtime plans and runtime-to-scheduler results.

use std::collections::HashSet;

use riley_runtime::paged_kv::{BLOCK_TABLE_V1_VERSION, BlockTableV1, KV_BLOCK_SIZE};

use crate::error::{SchedulerError, SchedulerResult};

/// Schema version emitted and accepted by this crate build.
pub const ITERATION_SCHEMA_VERSION: u16 = 1;

macro_rules! nonzero_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(u64);

        impl $name {
            /// Creates an identifier, rejecting the reserved zero value.
            #[must_use]
            pub const fn new(value: u64) -> Option<Self> {
                if value == 0 { None } else { Some(Self(value)) }
            }

            /// Returns the scheduler-issued integer value.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

nonzero_id!(RequestId, "Scheduler-issued request identity.");
nonzero_id!(
    IterationId,
    "Scheduler-issued immutable iteration identity."
);

/// Stable output-routing index within an iteration plan.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OutputSlot(u32);

impl OutputSlot {
    /// Creates an output slot. Zero is a valid first slot.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the runtime-facing slot index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stage of model work represented by one item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkKind {
    /// One bounded chunk of prompt ingestion.
    Prefill,
    /// One autoregressive input token for an active sequence.
    Decode,
}

/// Owned snapshot of the physical KV blocks visible to one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedBlockTable {
    request_id: RequestId,
    schema_version: u16,
    physical_block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
    logical_length: u32,
}

impl OwnedBlockTable {
    /// Creates and validates an immutable V1 executor transport table.
    ///
    /// # Errors
    ///
    /// Returns a schema, allocation, or structural error when the vectors are
    /// not the canonical V1 representation of `logical_length`.
    pub fn new(
        request_id: RequestId,
        schema_version: u16,
        physical_block_ids: Vec<u32>,
        valid_tokens: Vec<u16>,
        logical_length: u32,
    ) -> SchedulerResult<Self> {
        if schema_version != BLOCK_TABLE_V1_VERSION {
            return Err(SchedulerError::UnsupportedSchemaVersion {
                resource: "block table",
                expected: BLOCK_TABLE_V1_VERSION,
                actual: schema_version,
            });
        }
        if physical_block_ids.len() != valid_tokens.len() {
            return Err(SchedulerError::InvalidPlan {
                field: "block_tables",
                reason: "physical IDs and valid-token vectors must have equal length",
            });
        }
        let required_blocks = (logical_length as usize).div_ceil(KV_BLOCK_SIZE);
        if physical_block_ids.len() != required_blocks {
            return Err(SchedulerError::InvalidPlan {
                field: "block_tables",
                reason: "block count must exactly address the logical length",
            });
        }
        let mut unique_ids = HashSet::new();
        unique_ids
            .try_reserve(physical_block_ids.len())
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "block table physical ID set",
                requested_elements: physical_block_ids.len(),
            })?;
        let mut token_sum = 0_usize;
        for (index, (&physical_id, &valid)) in
            physical_block_ids.iter().zip(&valid_tokens).enumerate()
        {
            if !unique_ids.insert(physical_id) {
                return Err(SchedulerError::InvalidPlan {
                    field: "block_tables",
                    reason: "physical block IDs must be unique within one table",
                });
            }
            let expected = if index + 1 == required_blocks {
                let tail = logical_length as usize % KV_BLOCK_SIZE;
                if tail == 0 { KV_BLOCK_SIZE } else { tail }
            } else {
                KV_BLOCK_SIZE
            };
            if usize::from(valid) != expected {
                return Err(SchedulerError::InvalidPlan {
                    field: "block_tables",
                    reason: "valid-token counts must be full blocks followed by one canonical tail",
                });
            }
            token_sum = token_sum.checked_add(usize::from(valid)).ok_or(
                SchedulerError::ArithmeticOverflow {
                    field: "block table valid-token sum",
                },
            )?;
        }
        if token_sum != logical_length as usize {
            return Err(SchedulerError::InvalidPlan {
                field: "block_tables",
                reason: "valid-token sum must equal logical length",
            });
        }
        Ok(Self {
            request_id,
            schema_version,
            physical_block_ids,
            valid_tokens,
            logical_length,
        })
    }

    /// Fallibly copies the exact transport fields from a borrowed runtime table.
    ///
    /// # Errors
    ///
    /// Returns when either bounded vector cannot be allocated or the borrowed
    /// table violates the same invariants as [`Self::new`].
    pub fn copy_from_v1(request_id: RequestId, table: BlockTableV1<'_>) -> SchedulerResult<Self> {
        let block_count = table.block_count();
        let mut physical_block_ids = Vec::new();
        physical_block_ids
            .try_reserve_exact(block_count)
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "owned block table physical IDs",
                requested_elements: block_count,
            })?;
        physical_block_ids.extend_from_slice(table.physical_block_ids());
        let mut valid_tokens = Vec::new();
        valid_tokens.try_reserve_exact(block_count).map_err(|_| {
            SchedulerError::HostAllocation {
                resource: "owned block table valid tokens",
                requested_elements: block_count,
            }
        })?;
        valid_tokens.extend_from_slice(table.valid_tokens());
        Self::new(
            request_id,
            table.version(),
            physical_block_ids,
            valid_tokens,
            table.logical_length(),
        )
    }

    /// Request whose KV ownership this table snapshots.
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Executor transport schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Executor transport version, matching [`BlockTableV1::version`].
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.schema_version
    }

    /// Physical block indices in logical-token order.
    #[must_use]
    pub fn physical_block_ids(&self) -> &[u32] {
        &self.physical_block_ids
    }

    /// Valid token count in each corresponding physical block.
    #[must_use]
    pub fn valid_tokens(&self) -> &[u16] {
        &self.valid_tokens
    }

    /// Logical length addressed by this snapshot.
    #[must_use]
    pub const fn logical_length(&self) -> u32 {
        self.logical_length
    }
}

/// One request's work in an immutable iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkItem {
    request_id: RequestId,
    kind: WorkKind,
    input_tokens: Vec<u32>,
    target_logical_length: usize,
    block_table_index: usize,
    output_slot: Option<OutputSlot>,
}

impl WorkItem {
    /// Creates an owned item and validates invariants independent of its plan.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidPlan`] for empty input, a target length
    /// shorter than the supplied input, or a decode item not containing exactly
    /// one token.
    pub fn new(
        request_id: RequestId,
        kind: WorkKind,
        input_tokens: Vec<u32>,
        target_logical_length: usize,
        block_table_index: usize,
        output_slot: Option<OutputSlot>,
    ) -> SchedulerResult<Self> {
        if input_tokens.is_empty() {
            return Err(SchedulerError::InvalidPlan {
                field: "input_tokens",
                reason: "a work item must contain at least one input token",
            });
        }
        if target_logical_length < input_tokens.len() {
            return Err(SchedulerError::InvalidPlan {
                field: "target_logical_length",
                reason: "must be at least the input token count",
            });
        }
        if kind == WorkKind::Decode && input_tokens.len() != 1 {
            return Err(SchedulerError::InvalidPlan {
                field: "input_tokens",
                reason: "decode work must contain exactly one token",
            });
        }
        Ok(Self {
            request_id,
            kind,
            input_tokens,
            target_logical_length,
            block_table_index,
            output_slot,
        })
    }

    /// Request to which this work belongs.
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Whether this item is prefill or decode work.
    #[must_use]
    pub const fn kind(&self) -> WorkKind {
        self.kind
    }

    /// Owned input tokens exposed as a borrow for runtime execution.
    #[must_use]
    pub fn input_tokens(&self) -> &[u32] {
        &self.input_tokens
    }

    /// Logical sequence length after this item commits successfully.
    #[must_use]
    pub const fn target_logical_length(&self) -> usize {
        self.target_logical_length
    }

    /// Index into [`IterationPlan::block_tables`].
    #[must_use]
    pub const fn block_table_index(&self) -> usize {
        self.block_table_index
    }

    /// Slot to which the runtime must route a sampled token, when any.
    #[must_use]
    pub const fn output_slot(&self) -> Option<OutputSlot> {
        self.output_slot
    }
}

/// Immutable, self-contained input to one runtime execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IterationPlan {
    schema_version: u16,
    iteration_id: IterationId,
    prefill_items: Vec<WorkItem>,
    decode_items: Vec<WorkItem>,
    total_tokens: usize,
    block_tables: Vec<OwnedBlockTable>,
    output_slots: Vec<OutputSlot>,
}

impl IterationPlan {
    /// Builds a version-one plan and validates all cross-vector references.
    ///
    /// # Errors
    ///
    /// Returns a schema, arithmetic, host-allocation, or structural plan error
    /// before the plan is observable by the runtime.
    pub fn new(
        iteration_id: IterationId,
        prefill_items: Vec<WorkItem>,
        decode_items: Vec<WorkItem>,
        block_tables: Vec<OwnedBlockTable>,
    ) -> SchedulerResult<Self> {
        Self::from_version(
            ITERATION_SCHEMA_VERSION,
            iteration_id,
            prefill_items,
            decode_items,
            block_tables,
        )
    }

    /// Builds a plan while validating an explicitly transported schema version.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::UnsupportedSchemaVersion`] unless `schema_version`
    /// is [`ITERATION_SCHEMA_VERSION`], or another checked plan validation error.
    #[allow(clippy::too_many_lines)]
    pub fn from_version(
        schema_version: u16,
        iteration_id: IterationId,
        prefill_items: Vec<WorkItem>,
        decode_items: Vec<WorkItem>,
        block_tables: Vec<OwnedBlockTable>,
    ) -> SchedulerResult<Self> {
        validate_schema("iteration plan", schema_version)?;
        if prefill_items.is_empty() && decode_items.is_empty() {
            return Err(SchedulerError::InvalidPlan {
                field: "items",
                reason: "an iteration must contain work",
            });
        }

        let mut total_tokens = 0_usize;
        let item_count = prefill_items.len().checked_add(decode_items.len()).ok_or(
            SchedulerError::ArithmeticOverflow {
                field: "iteration item count",
            },
        )?;
        let mut referenced_block_tables =
            validate_iteration_block_tables(&block_tables, item_count)?;
        let mut request_ids = HashSet::new();
        request_ids
            .try_reserve(item_count)
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "iteration request identity set",
                requested_elements: item_count,
            })?;
        let mut output_slots = Vec::new();
        output_slots
            .try_reserve_exact(item_count)
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "iteration output slots",
                requested_elements: item_count,
            })?;
        let mut unique_slots = HashSet::new();
        unique_slots
            .try_reserve(item_count)
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "iteration output slot set",
                requested_elements: item_count,
            })?;

        for (expected_kind, items) in [
            (WorkKind::Prefill, prefill_items.as_slice()),
            (WorkKind::Decode, decode_items.as_slice()),
        ] {
            for item in items {
                if item.kind != expected_kind {
                    return Err(SchedulerError::InvalidPlan {
                        field: "work kind",
                        reason: "item is stored in the wrong stage vector",
                    });
                }
                if expected_kind == WorkKind::Decode && item.output_slot.is_none() {
                    return Err(SchedulerError::InvalidPlan {
                        field: "output_slots",
                        reason: "decode work must request one sampled output",
                    });
                }
                if !request_ids.insert(item.request_id) {
                    return Err(SchedulerError::InvalidPlan {
                        field: "request_id",
                        reason: "a request may appear at most once per iteration",
                    });
                }
                let table = block_tables.get(item.block_table_index).ok_or(
                    SchedulerError::InvalidPlan {
                        field: "block_table_index",
                        reason: "work item references a missing block table",
                    },
                )?;
                if referenced_block_tables[item.block_table_index] {
                    return Err(SchedulerError::InvalidPlan {
                        field: "block_table_index",
                        reason: "each block table must be referenced exactly once",
                    });
                }
                referenced_block_tables[item.block_table_index] = true;
                if table.request_id != item.request_id {
                    return Err(SchedulerError::InvalidPlan {
                        field: "block_table_index",
                        reason: "work item references another request's block table",
                    });
                }
                if table.logical_length as usize != item.target_logical_length {
                    return Err(SchedulerError::InvalidPlan {
                        field: "block_tables",
                        reason: "table logical length must equal the work target",
                    });
                }
                total_tokens = total_tokens.checked_add(item.input_tokens.len()).ok_or(
                    SchedulerError::ArithmeticOverflow {
                        field: "iteration token count",
                    },
                )?;
                if let Some(slot) = item.output_slot {
                    if !unique_slots.insert(slot) {
                        return Err(SchedulerError::InvalidPlan {
                            field: "output_slots",
                            reason: "output slots must be unique within an iteration",
                        });
                    }
                    output_slots.push(slot);
                }
            }
        }
        if referenced_block_tables.iter().any(|referenced| !referenced) {
            return Err(SchedulerError::InvalidPlan {
                field: "block_tables",
                reason: "each block table must be referenced exactly once",
            });
        }
        let output_slot_bound =
            u32::try_from(output_slots.len()).map_err(|_| SchedulerError::ArithmeticOverflow {
                field: "iteration output slot count",
            })?;
        if output_slots
            .iter()
            .any(|slot| slot.get() >= output_slot_bound)
        {
            return Err(SchedulerError::InvalidPlan {
                field: "output_slots",
                reason: "output slots must be dense and zero-based",
            });
        }
        output_slots.sort_unstable_by_key(|slot| slot.get());

        Ok(Self {
            schema_version,
            iteration_id,
            prefill_items,
            decode_items,
            total_tokens,
            block_tables,
            output_slots,
        })
    }

    /// Transport schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Unique identifier required on runtime feedback.
    #[must_use]
    pub const fn iteration_id(&self) -> IterationId {
        self.iteration_id
    }

    /// Prompt chunks, in deterministic execution order.
    #[must_use]
    pub fn prefill_items(&self) -> &[WorkItem] {
        &self.prefill_items
    }

    /// Autoregressive items, in deterministic execution order.
    #[must_use]
    pub fn decode_items(&self) -> &[WorkItem] {
        &self.decode_items
    }

    /// Sum of all input-token slice lengths.
    #[must_use]
    pub const fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Owned, versioned block-table snapshots referenced by work items.
    #[must_use]
    pub fn block_tables(&self) -> &[OwnedBlockTable] {
        &self.block_tables
    }

    /// Dense zero-based slots in canonical ascending sampling order.
    #[must_use]
    pub fn output_slots(&self) -> &[OutputSlot] {
        &self.output_slots
    }

    /// Number of requests represented by the plan.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.prefill_items.len() + self.decode_items.len()
    }
}

fn validate_iteration_block_tables(
    block_tables: &[OwnedBlockTable],
    item_count: usize,
) -> SchedulerResult<Vec<bool>> {
    if block_tables.len() != item_count {
        return Err(SchedulerError::InvalidPlan {
            field: "block_tables",
            reason: "each work item must own exactly one referenced block table",
        });
    }

    let block_entry_count = block_tables.iter().try_fold(0_usize, |count, table| {
        count.checked_add(table.physical_block_ids.len())
    });
    let Some(block_entry_count) = block_entry_count else {
        return Err(SchedulerError::ArithmeticOverflow {
            field: "iteration physical block entry count",
        });
    };
    let mut physical_block_ids = HashSet::new();
    physical_block_ids
        .try_reserve(block_entry_count)
        .map_err(|_| SchedulerError::HostAllocation {
            resource: "iteration physical block identity set",
            requested_elements: block_entry_count,
        })?;
    for table in block_tables {
        for &physical_block_id in &table.physical_block_ids {
            if !physical_block_ids.insert(physical_block_id) {
                return Err(SchedulerError::InvalidPlan {
                    field: "block_tables",
                    reason: "physical block IDs must be unique across the iteration",
                });
            }
        }
    }

    let mut referenced_block_tables = Vec::new();
    referenced_block_tables
        .try_reserve_exact(block_tables.len())
        .map_err(|_| SchedulerError::HostAllocation {
            resource: "iteration block table reference map",
            requested_elements: block_tables.len(),
        })?;
    referenced_block_tables.resize(block_tables.len(), false);
    Ok(referenced_block_tables)
}

/// Runtime output routed back to one stable plan slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IterationOutput {
    slot: OutputSlot,
    token_id: u32,
    stop: bool,
}

impl IterationOutput {
    /// Creates one sampled-token result.
    #[must_use]
    pub const fn new(slot: OutputSlot, token_id: u32, stop: bool) -> Self {
        Self {
            slot,
            token_id,
            stop,
        }
    }

    /// Plan output slot receiving this token.
    #[must_use]
    pub const fn slot(self) -> OutputSlot {
        self.slot
    }

    /// Sampled token identifier.
    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    /// Whether generation should stop after committing this token.
    #[must_use]
    pub const fn stop(self) -> bool {
        self.stop
    }
}

/// Versioned runtime feedback for one immutable iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IterationResult {
    schema_version: u16,
    iteration_id: IterationId,
    outputs: Vec<IterationOutput>,
    gpu_execution_ns: u64,
    gpu_idle_gap_ns: u64,
}

impl IterationResult {
    /// Builds version-one runtime feedback.
    ///
    /// # Errors
    ///
    /// Returns an invalid-result error when an output slot occurs twice.
    pub fn new(
        iteration_id: IterationId,
        outputs: Vec<IterationOutput>,
        gpu_execution_ns: u64,
        gpu_idle_gap_ns: u64,
    ) -> SchedulerResult<Self> {
        Self::from_version(
            ITERATION_SCHEMA_VERSION,
            iteration_id,
            outputs,
            gpu_execution_ns,
            gpu_idle_gap_ns,
        )
    }

    /// Builds feedback while validating an explicitly transported schema version.
    ///
    /// # Errors
    ///
    /// Returns a schema, allocation, or duplicate-output error.
    pub fn from_version(
        schema_version: u16,
        iteration_id: IterationId,
        outputs: Vec<IterationOutput>,
        gpu_execution_ns: u64,
        gpu_idle_gap_ns: u64,
    ) -> SchedulerResult<Self> {
        validate_schema("iteration result", schema_version)?;
        let mut slots = HashSet::new();
        slots
            .try_reserve(outputs.len())
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "iteration result output slot set",
                requested_elements: outputs.len(),
            })?;
        for output in &outputs {
            if !slots.insert(output.slot) {
                return Err(SchedulerError::InvalidIterationResult {
                    field: "outputs",
                    reason: "an output slot may occur at most once",
                });
            }
        }
        Ok(Self {
            schema_version,
            iteration_id,
            outputs,
            gpu_execution_ns,
            gpu_idle_gap_ns,
        })
    }

    /// Builds feedback from an internal producer that already proved outputs
    /// are ordered densely by slot in `0..outputs.len()`.
    ///
    /// This allocation-free path is crate-private so external transports must
    /// continue through [`Self::new`] or [`Self::from_version`].
    #[must_use]
    pub(crate) fn from_dense_outputs(
        iteration_id: IterationId,
        outputs: Vec<IterationOutput>,
        gpu_execution_ns: u64,
        gpu_idle_gap_ns: u64,
    ) -> Self {
        debug_assert!(
            outputs
                .iter()
                .enumerate()
                .all(|(index, output)| { u32::try_from(index).ok() == Some(output.slot().get()) })
        );
        Self {
            schema_version: ITERATION_SCHEMA_VERSION,
            iteration_id,
            outputs,
            gpu_execution_ns,
            gpu_idle_gap_ns,
        }
    }

    /// Transport schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Plan identifier to which this feedback belongs.
    #[must_use]
    pub const fn iteration_id(&self) -> IterationId {
        self.iteration_id
    }

    /// Sampled outputs routed by stable slot.
    #[must_use]
    pub fn outputs(&self) -> &[IterationOutput] {
        &self.outputs
    }

    /// Runtime-reported GPU execution duration.
    #[must_use]
    pub const fn gpu_execution_ns(&self) -> u64 {
        self.gpu_execution_ns
    }

    /// Runtime-reported gap since the preceding GPU iteration.
    #[must_use]
    pub const fn gpu_idle_gap_ns(&self) -> u64 {
        self.gpu_idle_gap_ns
    }
}

fn validate_schema(resource: &'static str, actual: u16) -> SchedulerResult<()> {
    if actual == ITERATION_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(SchedulerError::UnsupportedSchemaVersion {
            resource,
            expected: ITERATION_SCHEMA_VERSION,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use riley_runtime::paged_kv::BLOCK_TABLE_V1_VERSION;

    use crate::SchedulerError;

    use super::{
        IterationId, IterationOutput, IterationPlan, IterationResult, OutputSlot, OwnedBlockTable,
        RequestId, WorkItem, WorkKind,
    };

    fn table(request: u64, physical_block_id: u32) -> OwnedBlockTable {
        OwnedBlockTable::new(
            RequestId::new(request).unwrap(),
            BLOCK_TABLE_V1_VERSION,
            vec![physical_block_id],
            vec![1],
            1,
        )
        .unwrap()
    }

    fn prefill(request: u64, block_table_index: usize) -> WorkItem {
        WorkItem::new(
            RequestId::new(request).unwrap(),
            WorkKind::Prefill,
            vec![u32::try_from(request).unwrap()],
            1,
            block_table_index,
            None,
        )
        .unwrap()
    }

    fn decode(request: u64, block_table_index: usize, output_slot: Option<u32>) -> WorkItem {
        WorkItem::new(
            RequestId::new(request).unwrap(),
            WorkKind::Decode,
            vec![u32::try_from(request).unwrap()],
            1,
            block_table_index,
            output_slot.map(OutputSlot::new),
        )
        .unwrap()
    }

    #[test]
    fn zero_is_reserved_for_scheduler_ids() {
        assert!(RequestId::new(0).is_none());
        assert_eq!(IterationId::new(7).unwrap().get(), 7);
    }

    #[test]
    fn result_rejects_duplicate_output_slots() {
        let output = IterationOutput::new(OutputSlot::new(0), 42, false);
        let result = IterationResult::new(IterationId::new(1).unwrap(), vec![output, output], 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn owned_v1_table_requires_canonical_full_and_tail_counts() {
        let request_id = RequestId::new(1).unwrap();
        let table = OwnedBlockTable::new(
            request_id,
            BLOCK_TABLE_V1_VERSION,
            vec![7, 3],
            vec![16, 1],
            17,
        )
        .unwrap();
        assert_eq!(table.physical_block_ids(), &[7, 3]);
        assert_eq!(table.valid_tokens(), &[16, 1]);
        assert_eq!(table.logical_length(), 17);

        assert!(
            OwnedBlockTable::new(
                request_id,
                BLOCK_TABLE_V1_VERSION,
                vec![7, 3],
                vec![15, 2],
                17,
            )
            .is_err()
        );
    }

    #[test]
    fn iteration_rejects_physical_block_aliases_across_requests() {
        let error = IterationPlan::new(
            IterationId::new(1).unwrap(),
            vec![prefill(1, 0), prefill(2, 1)],
            Vec::new(),
            vec![table(1, 7), table(2, 7)],
        )
        .unwrap_err();

        assert_eq!(
            error,
            SchedulerError::InvalidPlan {
                field: "block_tables",
                reason: "physical block IDs must be unique across the iteration",
            }
        );
    }

    #[test]
    fn iteration_requires_one_block_table_per_work_item() {
        let error = IterationPlan::new(
            IterationId::new(1).unwrap(),
            vec![prefill(1, 0)],
            Vec::new(),
            vec![table(1, 7), table(2, 8)],
        )
        .unwrap_err();

        assert_eq!(
            error,
            SchedulerError::InvalidPlan {
                field: "block_tables",
                reason: "each work item must own exactly one referenced block table",
            }
        );
    }

    #[test]
    fn iteration_rejects_duplicate_block_table_references() {
        let error = IterationPlan::new(
            IterationId::new(1).unwrap(),
            vec![prefill(1, 0), prefill(2, 0)],
            Vec::new(),
            vec![table(1, 7), table(2, 8)],
        )
        .unwrap_err();

        assert_eq!(
            error,
            SchedulerError::InvalidPlan {
                field: "block_table_index",
                reason: "each block table must be referenced exactly once",
            }
        );
    }

    #[test]
    fn iteration_accepts_permuted_exactly_once_block_table_references() {
        let plan = IterationPlan::new(
            IterationId::new(1).unwrap(),
            vec![prefill(1, 1), prefill(2, 0)],
            Vec::new(),
            vec![table(2, 8), table(1, 7)],
        )
        .unwrap();

        assert_eq!(plan.batch_size(), 2);
        assert_eq!(plan.prefill_items()[0].block_table_index(), 1);
        assert_eq!(plan.prefill_items()[1].block_table_index(), 0);
    }

    #[test]
    fn iteration_requires_a_sampled_output_for_decode_work() {
        let error = IterationPlan::new(
            IterationId::new(1).unwrap(),
            Vec::new(),
            vec![decode(1, 0, None)],
            vec![table(1, 7)],
        )
        .unwrap_err();

        assert_eq!(
            error,
            SchedulerError::InvalidPlan {
                field: "output_slots",
                reason: "decode work must request one sampled output",
            }
        );
    }

    #[test]
    fn iteration_requires_dense_zero_based_output_slots() {
        let error = IterationPlan::new(
            IterationId::new(1).unwrap(),
            Vec::new(),
            vec![decode(1, 0, Some(1))],
            vec![table(1, 7)],
        )
        .unwrap_err();

        assert_eq!(
            error,
            SchedulerError::InvalidPlan {
                field: "output_slots",
                reason: "output slots must be dense and zero-based",
            }
        );
    }

    #[test]
    fn mixed_stage_output_slots_are_canonicalized_for_sampling() {
        let prefill_with_output = WorkItem::new(
            RequestId::new(5).unwrap(),
            WorkKind::Prefill,
            vec![5],
            1,
            0,
            Some(OutputSlot::new(4)),
        )
        .unwrap();
        let plan = IterationPlan::new(
            IterationId::new(1).unwrap(),
            vec![prefill_with_output],
            vec![
                decode(1, 1, Some(0)),
                decode(2, 2, Some(1)),
                decode(3, 3, Some(2)),
                decode(4, 4, Some(3)),
            ],
            vec![
                table(5, 5),
                table(1, 1),
                table(2, 2),
                table(3, 3),
                table(4, 4),
            ],
        )
        .unwrap();

        assert_eq!(
            plan.output_slots(),
            &[
                OutputSlot::new(0),
                OutputSlot::new(1),
                OutputSlot::new(2),
                OutputSlot::new(3),
                OutputSlot::new(4),
            ]
        );
        assert_eq!(
            plan.prefill_items()[0].output_slot(),
            Some(OutputSlot::new(4))
        );
    }
}
