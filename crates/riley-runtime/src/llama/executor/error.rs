//! Stable error and resource vocabulary for the Llama batch-executor facade.
//!
//! This module describes externally observable failure categories and resource
//! names, but owns no prepared executor state.  Other executor components may
//! depend on this vocabulary without depending on the batch-owner module.

use std::error;
use std::fmt;

use riley_cuda::CudaError;

use super::super::batch::LlamaBatchError;
use super::super::error::ExecutionSite;
use super::super::forward::LlamaForwardError;
use crate::paged_kv::PagedKvError;

/// Result type for continuous-batch preparation, execution, transfer, and close.
pub type LlamaBatchExecutorResult<T> = Result<T, LlamaBatchExecutorError>;

/// Extra owning resource held by the prepared executor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaBatchExecutorResource {
    KeyCache,
    ValueCache,
    RopeCos,
    RopeSin,
    SequenceBlockOffsets,
    PhysicalBlockIds,
    ValidTokens,
    RowSequenceSlots,
    RowPositions,
    OutputTokenIndices,
    PackedIterationInput,
    PinnedIterationInput,
    GatheredLogits,
    GreedyResults,
    HostWorkspace,
}

impl LlamaBatchExecutorResource {
    const fn name(self) -> &'static str {
        match self {
            Self::KeyCache => "shared_key_cache",
            Self::ValueCache => "shared_value_cache",
            Self::RopeCos => "absolute_rope_cos",
            Self::RopeSin => "absolute_rope_sin",
            Self::SequenceBlockOffsets => "batch_sequence_block_offsets",
            Self::PhysicalBlockIds => "batch_physical_block_ids",
            Self::ValidTokens => "batch_valid_tokens",
            Self::RowSequenceSlots => "batch_row_sequence_slots",
            Self::RowPositions => "batch_row_positions",
            Self::OutputTokenIndices => "batch_output_token_indices",
            Self::PackedIterationInput => "batch_packed_iteration_input",
            Self::PinnedIterationInput => "batch_pinned_iteration_input",
            Self::GatheredLogits => "batch_gathered_logits",
            Self::GreedyResults => "batch_greedy_results",
            Self::HostWorkspace => "batch_host_workspace",
        }
    }
}

impl fmt::Display for LlamaBatchExecutorResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Structured failure from the shape-bucketed continuous-batch executor.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlamaBatchExecutorError {
    Metadata(LlamaBatchError),
    Forward(LlamaForwardError),
    PagedKv(PagedKvError),
    InvalidConfiguration {
        field: &'static str,
        reason: &'static str,
    },
    UnsupportedHeadDimension {
        expected: usize,
        actual: usize,
    },
    InvalidBatch {
        field: &'static str,
        reason: &'static str,
    },
    TokenOutOfRange {
        position: usize,
        token_id: u32,
        vocabulary_size: usize,
    },
    PositionOutOfRange {
        row: usize,
        position: u32,
        maximum: usize,
    },
    Cuda {
        site: ExecutionSite,
        source: CudaError,
    },
    HostAllocation {
        resource: LlamaBatchExecutorResource,
        requested_bytes: u64,
    },
    ArithmeticOverflow {
        resource: LlamaBatchExecutorResource,
    },
    OutputNotReady,
    Poisoned,
    InvalidDownloadLength {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    GreedyLogitsNonFinite {
        output_index: usize,
    },
    InvalidGreedyResult {
        output_index: usize,
        status: u32,
        token_id: u32,
    },
    Cleanup {
        resource: LlamaBatchExecutorResource,
        source: CudaError,
    },
}

impl fmt::Display for LlamaBatchExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metadata(source) => source.fmt(formatter),
            Self::Forward(source) => source.fmt(formatter),
            Self::PagedKv(source) => source.fmt(formatter),
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid Llama batch executor configuration {field}: {reason}"
                )
            }
            Self::UnsupportedHeadDimension { expected, actual } => write!(
                formatter,
                "continuous-batch attention requires head dimension {expected}, got {actual}"
            ),
            Self::InvalidBatch { field, reason } => {
                write!(
                    formatter,
                    "invalid executable Llama batch {field}: {reason}"
                )
            }
            Self::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "batch token ID {token_id} at flattened position {position} is outside vocabulary 0..{vocabulary_size}"
            ),
            Self::PositionOutOfRange {
                row,
                position,
                maximum,
            } => write!(
                formatter,
                "batch row {row} uses absolute position {position} outside 0..{maximum}"
            ),
            Self::Cuda { site, source } => write!(formatter, "{site}: {source}"),
            Self::HostAllocation {
                resource,
                requested_bytes,
            } => write!(
                formatter,
                "could not reserve {requested_bytes} host bytes for {resource}"
            ),
            Self::ArithmeticOverflow { resource } => {
                write!(formatter, "byte arithmetic overflow for {resource}")
            }
            Self::OutputNotReady => formatter
                .write_str("gathered batch logits are unavailable before successful execution"),
            Self::Poisoned => formatter.write_str(
                "the Llama batch executor was poisoned by a native CUDA execution failure",
            ),
            Self::InvalidDownloadLength {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "batch-logit destination has {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::GreedyLogitsNonFinite { output_index } => write!(
                formatter,
                "greedy output row {output_index} contains a non-finite logit"
            ),
            Self::InvalidGreedyResult {
                output_index,
                status,
                token_id,
            } => write!(
                formatter,
                "greedy output row {output_index} has invalid native result status={status} token_id={token_id}"
            ),
            Self::Cleanup { resource, source } => {
                write!(formatter, "could not close {resource}: {source}")
            }
        }
    }
}

impl error::Error for LlamaBatchExecutorError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Metadata(source) => Some(source),
            Self::Forward(source) => Some(source),
            Self::PagedKv(source) => Some(source),
            Self::Cuda { source, .. } | Self::Cleanup { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<LlamaBatchError> for LlamaBatchExecutorError {
    fn from(source: LlamaBatchError) -> Self {
        Self::Metadata(source)
    }
}

impl From<LlamaForwardError> for LlamaBatchExecutorError {
    fn from(source: LlamaForwardError) -> Self {
        Self::Forward(source)
    }
}

impl From<PagedKvError> for LlamaBatchExecutorError {
    fn from(source: PagedKvError) -> Self {
        Self::PagedKv(source)
    }
}
