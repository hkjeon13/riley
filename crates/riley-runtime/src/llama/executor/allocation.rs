//! Cold allocation-accounting for the Llama continuous-batch executor.
//!
//! This module aggregates immutable byte and allocation-count facts only after
//! the enclosing owner has prepared every resource. It neither allocates nor
//! owns CUDA, host-workspace, KV, model, stream, or dispatch state.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use super::super::batch::LlamaBatchMetadataConfig;
use super::super::forward::PreparedLlamaAllocationReport;
use super::buffers::{U16_BYTES, U32_BYTES};
use super::config::BatchMetadataTransport;
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    checked_byte_len, usize_u64,
};
use super::metadata::{PackedIterationLayout, sequence_block_offset_count};
use super::output::GREEDY_RESULT_BYTES;

const PER_OPERATION_BASE_DEVICE_ALLOCATIONS: u64 = 9;
const ITERATION_BATCH_BASE_DEVICE_ALLOCATIONS: u64 = 5;

/// Exact owned allocation totals after cold batch preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaBatchAllocationReport {
    forward: PreparedLlamaAllocationReport,
    kv_cache_bytes: u64,
    rope_table_bytes: u64,
    packed_metadata_device_bytes: u64,
    batch_input_device_bytes: u64,
    gathered_logits_capacity_bytes: u64,
    greedy_result_capacity_bytes: u64,
    additional_device_bytes: u64,
    total_device_bytes: u64,
    additional_device_allocation_count: u64,
    total_device_allocation_count: u64,
    host_workspace_bytes: u64,
    total_pinned_host_bytes: u64,
    total_pinned_host_allocation_count: u64,
}

impl PreparedLlamaBatchAllocationReport {
    #[must_use]
    pub const fn forward(self) -> PreparedLlamaAllocationReport {
        self.forward
    }

    #[must_use]
    pub const fn kv_cache_bytes(self) -> u64 {
        self.kv_cache_bytes
    }

    #[must_use]
    pub const fn rope_table_bytes(self) -> u64 {
        self.rope_table_bytes
    }

    #[must_use]
    pub const fn packed_metadata_device_bytes(self) -> u64 {
        self.packed_metadata_device_bytes
    }

    /// Device bytes owned by the selected batch-input transport.
    ///
    /// Per-operation completion owns the six established metadata buffers;
    /// iteration completion owns one aligned token-plus-metadata slab.
    #[must_use]
    pub const fn batch_input_device_bytes(self) -> u64 {
        self.batch_input_device_bytes
    }

    #[must_use]
    pub const fn gathered_logits_capacity_bytes(self) -> u64 {
        self.gathered_logits_capacity_bytes
    }

    #[must_use]
    pub const fn greedy_result_capacity_bytes(self) -> u64 {
        self.greedy_result_capacity_bytes
    }

    #[must_use]
    pub const fn additional_device_bytes(self) -> u64 {
        self.additional_device_bytes
    }

    #[must_use]
    pub const fn total_device_bytes(self) -> u64 {
        self.total_device_bytes
    }

    #[must_use]
    pub const fn additional_device_allocation_count(self) -> u64 {
        self.additional_device_allocation_count
    }

    #[must_use]
    pub const fn total_device_allocation_count(self) -> u64 {
        self.total_device_allocation_count
    }

    #[must_use]
    pub const fn host_workspace_bytes(self) -> u64 {
        self.host_workspace_bytes
    }

    #[must_use]
    pub const fn pinned_host_bytes(self) -> u64 {
        self.total_pinned_host_bytes
    }

    #[must_use]
    pub const fn pinned_host_allocation_count(self) -> u64 {
        self.total_pinned_host_allocation_count
    }
}

/// Aggregates the exact cold allocation contract after all owners exist.
///
/// The caller retains every resource and invokes this only after successful
/// allocation, so this function derives transport-specific host and pinned
/// capacity from the same immutable bounds used by those owners.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(in crate::llama) fn build_batch_allocation_report(
    forward: PreparedLlamaAllocationReport,
    bounds: LlamaBatchMetadataConfig,
    transport: BatchMetadataTransport,
    kv_cache_bytes: u64,
    rope_bytes_per_kind: u64,
    gathered_logits_capacity_bytes: u64,
    greedy_result_capacity_bytes: u64,
) -> LlamaBatchExecutorResult<PreparedLlamaBatchAllocationReport> {
    let offset_count = sequence_block_offset_count(bounds.max_rows())?;
    let sequence_block_offsets_bytes = checked_byte_len(
        offset_count,
        U32_BYTES,
        LlamaBatchExecutorResource::SequenceBlockOffsets,
    )?;
    let physical_block_ids_bytes = checked_byte_len(
        bounds.max_block_entries(),
        U32_BYTES,
        LlamaBatchExecutorResource::PhysicalBlockIds,
    )?;
    let valid_tokens_bytes = checked_byte_len(
        bounds.max_block_entries(),
        U16_BYTES,
        LlamaBatchExecutorResource::ValidTokens,
    )?;
    let row_sequence_slots_bytes = checked_byte_len(
        bounds.max_input_tokens(),
        U32_BYTES,
        LlamaBatchExecutorResource::RowSequenceSlots,
    )?;
    let row_positions_bytes = checked_byte_len(
        bounds.max_input_tokens(),
        U32_BYTES,
        LlamaBatchExecutorResource::RowPositions,
    )?;
    let output_token_indices_bytes = checked_byte_len(
        bounds.max_output_slots(),
        U32_BYTES,
        LlamaBatchExecutorResource::OutputTokenIndices,
    )?;
    let packed_metadata_device_bytes = [
        sequence_block_offsets_bytes,
        physical_block_ids_bytes,
        valid_tokens_bytes,
        row_sequence_slots_bytes,
        row_positions_bytes,
        output_token_indices_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total
            .checked_add(usize_u64(bytes, LlamaBatchExecutorResource::HostWorkspace)?)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })
    })?;
    let packed_iteration_capacity = match transport {
        BatchMetadataTransport::Synchronous => 0,
        BatchMetadataTransport::PackedAsync => PackedIterationLayout::capacity(bounds)?.total_bytes,
    };
    let batch_input_device_bytes = match transport {
        BatchMetadataTransport::Synchronous => packed_metadata_device_bytes,
        BatchMetadataTransport::PackedAsync => usize_u64(
            packed_iteration_capacity,
            LlamaBatchExecutorResource::PackedIterationInput,
        )?,
    };
    let rope_table_bytes =
        rope_bytes_per_kind
            .checked_mul(2)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeSin,
            })?;
    let additional_device_bytes = kv_cache_bytes
        .checked_add(rope_table_bytes)
        .and_then(|bytes| bytes.checked_add(batch_input_device_bytes))
        .and_then(|bytes| bytes.checked_add(gathered_logits_capacity_bytes))
        .and_then(|bytes| bytes.checked_add(greedy_result_capacity_bytes))
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let total_device_bytes = forward
        .total_device_bytes()
        .checked_add(additional_device_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let (base_allocations, output_allocations) = match transport {
        BatchMetadataTransport::Synchronous => (
            PER_OPERATION_BASE_DEVICE_ALLOCATIONS,
            u64::from(bounds.max_output_slots() != 0) * 3,
        ),
        BatchMetadataTransport::PackedAsync => (
            ITERATION_BATCH_BASE_DEVICE_ALLOCATIONS,
            u64::from(bounds.max_output_slots() != 0) * 2,
        ),
    };
    let additional_device_allocation_count = base_allocations
        .checked_add(output_allocations)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let total_device_allocation_count = forward
        .device_allocation_count()
        .checked_add(additional_device_allocation_count)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let input_host_workspace_bytes = match transport {
        BatchMetadataTransport::Synchronous => [
            bounds.max_input_tokens().checked_mul(U32_BYTES),
            Some(sequence_block_offsets_bytes),
            Some(physical_block_ids_bytes),
            Some(valid_tokens_bytes),
            Some(row_sequence_slots_bytes),
            Some(row_positions_bytes),
            Some(output_token_indices_bytes),
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes.ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::HostWorkspace,
                })?)
                .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::HostWorkspace,
                })
        })?,
        BatchMetadataTransport::PackedAsync => packed_iteration_capacity,
    };
    let greedy_result_host_bytes = bounds
        .max_output_slots()
        .checked_mul(GREEDY_RESULT_BYTES)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::HostWorkspace,
        })?;
    let host_workspace_bytes = usize_u64(
        input_host_workspace_bytes
            .checked_add(greedy_result_host_bytes)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })?,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let (additional_pinned_host_bytes, additional_pinned_host_allocation_count) = match transport {
        BatchMetadataTransport::Synchronous => (0, 0),
        BatchMetadataTransport::PackedAsync => (batch_input_device_bytes, 1),
    };
    let total_pinned_host_bytes = forward
        .pinned_host_bytes()
        .checked_add(additional_pinned_host_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::PinnedIterationInput,
        })?;
    let total_pinned_host_allocation_count = forward
        .pinned_host_allocation_count()
        .checked_add(additional_pinned_host_allocation_count)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::PinnedIterationInput,
        })?;
    Ok(PreparedLlamaBatchAllocationReport {
        forward,
        kv_cache_bytes,
        rope_table_bytes,
        packed_metadata_device_bytes,
        batch_input_device_bytes,
        gathered_logits_capacity_bytes,
        greedy_result_capacity_bytes,
        additional_device_bytes,
        total_device_bytes,
        additional_device_allocation_count,
        total_device_allocation_count,
        host_workspace_bytes,
        total_pinned_host_bytes,
        total_pinned_host_allocation_count,
    })
}
