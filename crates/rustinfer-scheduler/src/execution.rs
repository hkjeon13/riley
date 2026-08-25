//! Fail-closed bridge from immutable scheduler plans to the Llama batch runtime.
//!
//! The scheduler remains the owner of logical KV reservations. This module
//! borrows an [`IterationPlan`], builds runtime-neutral Llama rows, and (with
//! the `cuda` feature) performs one synchronous execute/download transaction.
//! It never commits or aborts scheduler state on the caller's behalf: instead,
//! successful downloads expose commit data and failures expose the only safe
//! [`ExecutionAbort`] disposition, when one can be proven.

use std::error;
use std::fmt;

#[cfg(any(feature = "cuda", test))]
use rustinfer_runtime::llama::LlamaBatchMetadataConfig;
use rustinfer_runtime::llama::{LlamaBatchBlockTable, LlamaBatchRow, LlamaBatchRowKind};

#[cfg(feature = "cuda")]
use rustinfer_runtime::llama::{LlamaBatchExecutorError, PreparedLlamaBatchExecutor};
#[cfg(feature = "cuda")]
use rustinfer_runtime::{CudaContext, CudaError, CudaEvent, CudaStream};

use crate::plan::{
    ITERATION_SCHEMA_VERSION, IterationId, IterationOutput, IterationPlan, IterationResult,
    OutputSlot, WorkItem, WorkKind,
};
use crate::scheduler::ExecutionAbort;

const BF16_BYTES: usize = 2;

/// Result type for scheduler-to-runtime adaptation and commit-data assembly.
pub type IterationAdapterResult<T> = Result<T, IterationAdapterError>;

/// A checked plan-adaptation, runtime-bound, or commit-data failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum IterationAdapterError {
    /// The immutable plan does not satisfy the runtime bridge contract.
    InvalidPlan {
        /// Invalid plan field or relationship.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// The plan exceeds one of the executor's cold-prepared bounds.
    CapacityExceeded {
        /// Capacity being checked.
        resource: &'static str,
        /// Elements required by the plan.
        requested: usize,
        /// Elements available in the prepared executor.
        capacity: usize,
    },
    /// Checked size or index arithmetic failed.
    ArithmeticOverflow {
        /// Quantity that could not be represented.
        field: &'static str,
    },
    /// A bounded host allocation could not be reserved before dispatch.
    HostAllocation {
        /// Buffer being allocated.
        resource: &'static str,
        /// Number of requested elements.
        requested_elements: usize,
    },
    /// An input token is outside the prepared model vocabulary.
    TokenOutOfRange {
        /// Flattened input-token position.
        position: usize,
        /// Invalid token identifier.
        token_id: u32,
        /// Prepared vocabulary size.
        vocabulary_size: usize,
    },
    /// A row reaches beyond the prepared absolute `RoPE` table.
    PositionOutOfRange {
        /// Scheduler request identity used as the runtime sequence tag.
        sequence_tag: u64,
        /// Exclusive target sequence length.
        target_logical_length: u32,
        /// Number of prepared absolute positions.
        maximum_position_count: usize,
    },
    /// Runtime output metadata differs from the plan that was dispatched.
    InvalidRuntimeOutput {
        /// Runtime output field that violated the bridge invariant.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// Sampled-token count differs from the number of downloaded output rows.
    InvalidSampleCount {
        /// Number of output rows requiring samples.
        expected: usize,
        /// Number of supplied samples.
        actual: usize,
    },
    /// A sampled token cannot belong to the model vocabulary.
    SampleTokenOutOfRange {
        /// Dense output slot whose sample is invalid.
        slot: OutputSlot,
        /// Invalid token identifier.
        token_id: u32,
        /// Prepared vocabulary size.
        vocabulary_size: usize,
    },
    /// The fixed-M Llama executor rejected or failed the iteration.
    #[cfg(feature = "cuda")]
    Runtime(Box<LlamaBatchExecutorError>),
    /// A dispatch-boundary CUDA event operation failed.
    #[cfg(feature = "cuda")]
    CudaTiming {
        /// Stable event operation name.
        operation: &'static str,
        /// Native CUDA event failure.
        source: CudaError,
    },
    /// Stream quiescence could not be established after a dispatch attempt.
    #[cfg(feature = "cuda")]
    Synchronization {
        /// Failure observed before the synchronization attempt, when any.
        preceding: Option<Box<Self>>,
        /// CUDA stream synchronization failure.
        source: CudaError,
    },
}

impl fmt::Display for IterationAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan { field, reason } => {
                write!(
                    formatter,
                    "invalid executable iteration plan {field}: {reason}"
                )
            }
            Self::CapacityExceeded {
                resource,
                requested,
                capacity,
            } => write!(
                formatter,
                "iteration {resource} capacity exceeded: requested={requested} capacity={capacity}"
            ),
            Self::ArithmeticOverflow { field } => {
                write!(
                    formatter,
                    "iteration adapter arithmetic overflow for {field}"
                )
            }
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "could not reserve {requested_elements} host elements for iteration {resource}"
            ),
            Self::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "iteration token ID {token_id} at flattened position {position} is outside vocabulary 0..{vocabulary_size}"
            ),
            Self::PositionOutOfRange {
                sequence_tag,
                target_logical_length,
                maximum_position_count,
            } => write!(
                formatter,
                "iteration sequence {sequence_tag} targets length {target_logical_length}, beyond {maximum_position_count} prepared positions"
            ),
            Self::InvalidRuntimeOutput { field, reason } => {
                write!(
                    formatter,
                    "invalid runtime iteration output {field}: {reason}"
                )
            }
            Self::InvalidSampleCount { expected, actual } => write!(
                formatter,
                "iteration has {expected} output rows but received {actual} sampled tokens"
            ),
            Self::SampleTokenOutOfRange {
                slot,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "sampled token ID {token_id} for output slot {} is outside vocabulary 0..{vocabulary_size}",
                slot.get()
            ),
            #[cfg(feature = "cuda")]
            Self::Runtime(source) => write!(formatter, "Llama batch execution failed: {source}"),
            #[cfg(feature = "cuda")]
            Self::CudaTiming { operation, source } => {
                write!(
                    formatter,
                    "CUDA iteration timing {operation} failed: {source}"
                )
            }
            #[cfg(feature = "cuda")]
            Self::Synchronization { preceding, source } => {
                if let Some(preceding) = preceding {
                    write!(
                        formatter,
                        "{preceding}; CUDA stream synchronization also failed: {source}"
                    )
                } else {
                    write!(formatter, "CUDA stream synchronization failed: {source}")
                }
            }
        }
    }
}

impl error::Error for IterationAdapterError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Runtime(source) => Some(source.as_ref()),
            #[cfg(feature = "cuda")]
            Self::CudaTiming { source, .. } => Some(source),
            #[cfg(feature = "cuda")]
            Self::Synchronization { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Borrowed, allocation-bounded runtime rows prepared from one immutable plan.
///
/// Row order is always all prefill items followed by all decode items, exactly
/// matching [`IterationPlan`]. Physical block tables and token slices remain
/// borrowed from the plan, so the value must not outlive that plan.
#[derive(Debug)]
pub struct PreparedLlamaIteration<'plan> {
    iteration_id: IterationId,
    rows: Vec<LlamaBatchRow<'plan>>,
    total_input_tokens: usize,
    total_block_entries: usize,
    output_count: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    commit_outputs: Vec<IterationOutput>,
}

impl<'plan> PreparedLlamaIteration<'plan> {
    /// Converts and defensively revalidates an immutable scheduler plan.
    ///
    /// All allocations needed for row adaptation and later commit-output
    /// assembly are reserved here, before any device dispatch is possible.
    ///
    /// # Errors
    ///
    /// Returns for a schema/order/reference/routing invariant violation,
    /// checked arithmetic failure, or bounded host allocation failure.
    pub fn prepare(plan: &'plan IterationPlan) -> IterationAdapterResult<Self> {
        if plan.schema_version() != ITERATION_SCHEMA_VERSION {
            return Err(IterationAdapterError::InvalidPlan {
                field: "schema_version",
                reason: "plan version differs from the scheduler runtime bridge",
            });
        }
        if plan.batch_size() == 0 {
            return Err(IterationAdapterError::InvalidPlan {
                field: "items",
                reason: "an executable iteration must contain at least one row",
            });
        }

        let mut rows = reserve_vec(plan.batch_size(), "runtime rows")?;
        let commit_outputs = reserve_vec(plan.output_slots().len(), "commit outputs")?;

        let mut total_input_tokens = 0_usize;
        let mut total_block_entries = 0_usize;
        for (expected_kind, items) in [
            (WorkKind::Prefill, plan.prefill_items()),
            (WorkKind::Decode, plan.decode_items()),
        ] {
            for item in items {
                let row = adapt_item(plan, item, expected_kind)?;
                total_input_tokens = total_input_tokens
                    .checked_add(item.input_tokens().len())
                    .ok_or(IterationAdapterError::ArithmeticOverflow {
                        field: "total input tokens",
                    })?;
                total_block_entries = total_block_entries
                    .checked_add(row.block_table().physical_block_ids().len())
                    .ok_or(IterationAdapterError::ArithmeticOverflow {
                        field: "total block entries",
                    })?;
                rows.push(row);
            }
        }

        if total_input_tokens != plan.total_tokens() {
            return Err(IterationAdapterError::InvalidPlan {
                field: "total_tokens",
                reason: "stored total differs from the adapted row token count",
            });
        }
        if rows.len() != plan.batch_size() {
            return Err(IterationAdapterError::InvalidPlan {
                field: "batch_size",
                reason: "stored batch size differs from the adapted row count",
            });
        }

        Ok(Self {
            iteration_id: plan.iteration_id(),
            rows,
            total_input_tokens,
            total_block_entries,
            output_count: plan.output_slots().len(),
            commit_outputs,
        })
    }

    /// Scheduler iteration identity carried into all commit or abort data.
    #[must_use]
    pub const fn iteration_id(&self) -> IterationId {
        self.iteration_id
    }

    /// Runtime rows in deterministic prefill-then-decode order.
    #[must_use]
    pub fn rows(&self) -> &[LlamaBatchRow<'plan>] {
        &self.rows
    }

    /// Total flattened input-token count.
    #[must_use]
    pub const fn total_input_tokens(&self) -> usize {
        self.total_input_tokens
    }

    /// Total flattened physical block-table entries.
    #[must_use]
    pub const fn total_block_entries(&self) -> usize {
        self.total_block_entries
    }

    /// Number of dense output rows expected from the executor.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    #[cfg(any(feature = "cuda", test))]
    fn validate_executor_bounds(
        &self,
        metadata: LlamaBatchMetadataConfig,
        vocabulary_size: usize,
        maximum_position_count: usize,
    ) -> IterationAdapterResult<()> {
        ensure_capacity("rows", self.rows.len(), metadata.max_rows())?;
        ensure_capacity(
            "input tokens",
            self.total_input_tokens,
            metadata.max_input_tokens(),
        )?;
        ensure_capacity(
            "block entries",
            self.total_block_entries,
            metadata.max_block_entries(),
        )?;
        ensure_capacity(
            "output slots",
            self.output_count,
            metadata.max_output_slots(),
        )?;

        let mut flattened_position = 0_usize;
        for row in &self.rows {
            for &token_id in row.input_token_ids() {
                if usize::try_from(token_id)
                    .ok()
                    .is_none_or(|token| token >= vocabulary_size)
                {
                    return Err(IterationAdapterError::TokenOutOfRange {
                        position: flattened_position,
                        token_id,
                        vocabulary_size,
                    });
                }
                flattened_position = flattened_position.checked_add(1).ok_or(
                    IterationAdapterError::ArithmeticOverflow {
                        field: "flattened token position",
                    },
                )?;
            }
            if usize::try_from(row.target_logical_length())
                .ok()
                .is_none_or(|target| target > maximum_position_count)
            {
                return Err(IterationAdapterError::PositionOutOfRange {
                    sequence_tag: row.sequence_tag(),
                    target_logical_length: row.target_logical_length(),
                    maximum_position_count,
                });
            }
            for &physical_id in row.block_table().physical_block_ids() {
                if usize::try_from(physical_id)
                    .ok()
                    .is_none_or(|block| block >= metadata.physical_block_count())
                {
                    return Err(IterationAdapterError::CapacityExceeded {
                        resource: "physical block index",
                        requested: usize::try_from(physical_id)
                            .unwrap_or(usize::MAX)
                            .saturating_add(1),
                        capacity: metadata.physical_block_count(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// One sampled token aligned to a dense downloaded output slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledIterationToken {
    token_id: u32,
    stop: bool,
}

impl SampledIterationToken {
    /// Creates one scheduler-facing sampled token.
    #[must_use]
    pub const fn new(token_id: u32, stop: bool) -> Self {
        Self { token_id, stop }
    }

    /// Sampled vocabulary token identifier.
    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    /// Whether generation must stop after this token commits.
    #[must_use]
    pub const fn stop(self) -> bool {
        self.stop
    }
}

/// Runtime timing fields attached to a scheduler commit result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IterationTiming {
    gpu_execution_ns: u64,
    gpu_idle_gap_ns: u64,
}

/// Reusable CUDA-event owner for exact iteration execution and idle timings.
///
/// The measured execution interval begins after plan adaptation, executor-bound
/// validation, and host output allocation. It ends after the logits D2H copy is
/// enqueued. The end event is synchronized before durations are published, and
/// the next iteration's idle gap is the same-stream interval from that end
/// event to the next dispatch start event.
#[cfg(feature = "cuda")]
pub struct LlamaIterationCudaTimer {
    start: CudaEvent,
    end: CudaEvent,
    previous_end: CudaEvent,
    has_previous_end: bool,
}

#[cfg(feature = "cuda")]
impl LlamaIterationCudaTimer {
    /// Creates the three timing-enabled events reused by one execution lane.
    ///
    /// # Errors
    ///
    /// Returns the first event-creation failure after explicitly attempting to
    /// close every event already created by this constructor.
    pub fn prepare(context: &CudaContext) -> Result<Self, CudaError> {
        let start = context.create_event()?;
        let end = match context.create_event() {
            Ok(event) => event,
            Err(error) => {
                let _ = start.close();
                return Err(error);
            }
        };
        let previous_end = match context.create_event() {
            Ok(event) => event,
            Err(error) => {
                let _ = end.close();
                let _ = start.close();
                return Err(error);
            }
        };
        Ok(Self {
            start,
            end,
            previous_end,
            has_previous_end: false,
        })
    }

    fn record_start(&mut self, stream: &mut CudaStream) -> IterationAdapterResult<()> {
        self.start
            .record(stream)
            .map_err(|source| IterationAdapterError::CudaTiming {
                operation: "start record",
                source,
            })
    }

    fn record_end_and_measure(
        &mut self,
        stream: &mut CudaStream,
    ) -> IterationAdapterResult<IterationTiming> {
        self.end
            .record(stream)
            .map_err(|source| IterationAdapterError::CudaTiming {
                operation: "end record",
                source,
            })?;
        self.end
            .synchronize()
            .map_err(|source| IterationAdapterError::CudaTiming {
                operation: "end synchronize",
                source,
            })?;
        let gpu_execution_ns = elapsed_event_ns(&self.start, &self.end, "execution elapsed")?;
        let gpu_idle_gap_ns = if self.has_previous_end {
            elapsed_event_ns(&self.previous_end, &self.start, "idle elapsed")?
        } else {
            0
        };
        std::mem::swap(&mut self.end, &mut self.previous_end);
        self.has_previous_end = true;
        Ok(IterationTiming::new(gpu_execution_ns, gpu_idle_gap_ns))
    }

    fn invalidate_previous_end(&mut self) {
        self.has_previous_end = false;
    }

    /// Starts a new observation window whose first iteration has no idle gap.
    ///
    /// This does not record or destroy an event. It only prevents time outside
    /// the caller's next measured window from being attributed to that window.
    pub fn reset_window(&mut self) {
        self.invalidate_previous_end();
    }

    /// Explicitly destroys every owned CUDA event.
    ///
    /// # Errors
    ///
    /// Returns the first close error after attempting all three closes.
    pub fn close(self) -> Result<(), CudaError> {
        let Self {
            start,
            end,
            previous_end,
            has_previous_end: _,
        } = self;
        let mut first_error = None;
        for event in [start, end, previous_end] {
            if let Err(error) = event.close() {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "cuda")]
fn elapsed_event_ns(
    start: &CudaEvent,
    end: &CudaEvent,
    operation: &'static str,
) -> IterationAdapterResult<u64> {
    let elapsed_ms = start
        .elapsed_ms(end)
        .map_err(|source| IterationAdapterError::CudaTiming { operation, source })?;
    if !elapsed_ms.is_finite() || elapsed_ms.is_sign_negative() {
        return Err(IterationAdapterError::InvalidRuntimeOutput {
            field: "CUDA event elapsed milliseconds",
            reason: "timing result must be finite and non-negative",
        });
    }
    let elapsed_ns = f64::from(elapsed_ms) * 1_000_000.0;
    if elapsed_ns >= 18_446_744_073_709_551_616.0 {
        return Err(IterationAdapterError::ArithmeticOverflow {
            field: "CUDA event elapsed nanoseconds",
        });
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rounded_ns = elapsed_ns.round() as u64;
    Ok(rounded_ns)
}

impl IterationTiming {
    /// Creates externally measured GPU timing metadata.
    #[must_use]
    pub const fn new(gpu_execution_ns: u64, gpu_idle_gap_ns: u64) -> Self {
        Self {
            gpu_execution_ns,
            gpu_idle_gap_ns,
        }
    }

    /// Runtime-reported GPU execution duration.
    #[must_use]
    pub const fn gpu_execution_ns(self) -> u64 {
        self.gpu_execution_ns
    }

    /// Gap since the preceding GPU iteration.
    #[must_use]
    pub const fn gpu_idle_gap_ns(self) -> u64 {
        self.gpu_idle_gap_ns
    }
}

/// Owned native-endian BF16 logits downloaded in dense output-slot order.
///
/// Device work and the owning stream have completed before this value is
/// returned. It is therefore always safe either to build an [`IterationResult`]
/// or to abort its iteration as [`ExecutionAbort::DeviceQuiescedMutationUnknown`].
#[derive(Debug)]
pub struct DownloadedLlamaIteration {
    iteration_id: IterationId,
    vocabulary_size: usize,
    output_count: usize,
    logits_bf16_native: Vec<u8>,
    commit_outputs: Vec<IterationOutput>,
}

impl DownloadedLlamaIteration {
    /// Scheduler iteration identity matching these logits.
    #[must_use]
    pub const fn iteration_id(&self) -> IterationId {
        self.iteration_id
    }

    /// Vocabulary width of every output row.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Number of dense output rows.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    /// Complete native-endian BF16 `[O,V]` logit storage.
    #[must_use]
    pub fn logits_bf16_native(&self) -> &[u8] {
        &self.logits_bf16_native
    }

    /// Returns one vocabulary-wide BF16 row for a dense plan output slot.
    ///
    /// # Errors
    ///
    /// Returns when `slot` is not among this iteration's dense `0..O` slots or
    /// checked row-byte arithmetic fails.
    pub fn logits_for_slot(&self, slot: OutputSlot) -> IterationAdapterResult<&[u8]> {
        let index =
            usize::try_from(slot.get()).map_err(|_| IterationAdapterError::ArithmeticOverflow {
                field: "output slot index",
            })?;
        if index >= self.output_count {
            return Err(IterationAdapterError::InvalidRuntimeOutput {
                field: "output slot",
                reason: "requested slot is outside the downloaded dense output range",
            });
        }
        let row_bytes = self.vocabulary_size.checked_mul(BF16_BYTES).ok_or(
            IterationAdapterError::ArithmeticOverflow {
                field: "logit row byte length",
            },
        )?;
        let start =
            index
                .checked_mul(row_bytes)
                .ok_or(IterationAdapterError::ArithmeticOverflow {
                    field: "logit row byte offset",
                })?;
        let end =
            start
                .checked_add(row_bytes)
                .ok_or(IterationAdapterError::ArithmeticOverflow {
                    field: "logit row byte range",
                })?;
        self.logits_bf16_native
            .get(start..end)
            .ok_or(IterationAdapterError::InvalidRuntimeOutput {
                field: "logits",
                reason: "downloaded storage is shorter than its declared shape",
            })
    }

    /// Safe scheduler abort data if sampling or result publication cannot finish.
    #[must_use]
    pub const fn abort_data(&self) -> (IterationId, ExecutionAbort) {
        (
            self.iteration_id,
            ExecutionAbort::DeviceQuiescedMutationUnknown,
        )
    }

    /// Consumes sampled tokens in dense slot order and builds commit feedback.
    ///
    /// The output vector was reserved before device dispatch, so successful
    /// assembly performs no allocation. A validation failure returns this
    /// downloaded iteration to the caller, preserving both retry and safe abort.
    ///
    /// # Errors
    ///
    /// Returns an owning [`IterationCommitFailure`] for a sample-count mismatch
    /// or token outside the downloaded model vocabulary.
    pub fn into_result(
        mut self,
        samples: &[SampledIterationToken],
        timing: IterationTiming,
    ) -> Result<IterationResult, IterationCommitFailure> {
        let output_count = self.output_count;
        if samples.len() != output_count {
            return Err(IterationCommitFailure::new(
                self,
                IterationAdapterError::InvalidSampleCount {
                    expected: output_count,
                    actual: samples.len(),
                },
            ));
        }
        let Ok(output_count_u32) = u32::try_from(output_count) else {
            return Err(IterationCommitFailure::new(
                self,
                IterationAdapterError::ArithmeticOverflow {
                    field: "commit output count",
                },
            ));
        };
        for (raw_slot, sample) in (0..output_count_u32).zip(samples.iter().copied()) {
            if usize::try_from(sample.token_id)
                .ok()
                .is_none_or(|token| token >= self.vocabulary_size)
            {
                let vocabulary_size = self.vocabulary_size;
                return Err(IterationCommitFailure::new(
                    self,
                    IterationAdapterError::SampleTokenOutOfRange {
                        slot: OutputSlot::new(raw_slot),
                        token_id: sample.token_id,
                        vocabulary_size,
                    },
                ));
            }
        }
        for (raw_slot, sample) in (0..output_count_u32).zip(samples.iter().copied()) {
            self.commit_outputs.push(IterationOutput::new(
                OutputSlot::new(raw_slot),
                sample.token_id,
                sample.stop,
            ));
        }
        Ok(IterationResult::from_dense_outputs(
            self.iteration_id,
            self.commit_outputs,
            timing.gpu_execution_ns,
            timing.gpu_idle_gap_ns,
        ))
    }
}

/// Owning, retryable failure while converting downloaded logits to commit data.
#[derive(Debug)]
pub struct IterationCommitFailure {
    iteration: DownloadedLlamaIteration,
    error: IterationAdapterError,
}

impl IterationCommitFailure {
    fn new(iteration: DownloadedLlamaIteration, error: IterationAdapterError) -> Self {
        Self { iteration, error }
    }

    /// Downloaded logits retained for corrected sampling or inspection.
    #[must_use]
    pub const fn iteration(&self) -> &DownloadedLlamaIteration {
        &self.iteration
    }

    /// Commit-data validation failure.
    #[must_use]
    pub const fn error(&self) -> &IterationAdapterError {
        &self.error
    }

    /// Safe scheduler abort data for the completed device iteration.
    #[must_use]
    pub const fn abort_data(&self) -> (IterationId, ExecutionAbort) {
        self.iteration.abort_data()
    }

    /// Recovers the logits and validation error.
    #[must_use]
    pub fn into_parts(self) -> (DownloadedLlamaIteration, IterationAdapterError) {
        (self.iteration, self.error)
    }
}

impl fmt::Display for IterationCommitFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl error::Error for IterationCommitFailure {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Failure from one execute/download transaction plus safe settlement advice.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct IterationExecutionFailure {
    iteration_id: IterationId,
    abort: Option<ExecutionAbort>,
    error: IterationAdapterError,
}

#[cfg(feature = "cuda")]
impl IterationExecutionFailure {
    fn new(
        iteration_id: IterationId,
        abort: Option<ExecutionAbort>,
        error: IterationAdapterError,
    ) -> Self {
        Self {
            iteration_id,
            abort,
            error,
        }
    }

    /// Iteration that failed to produce commit data.
    #[must_use]
    pub const fn iteration_id(&self) -> IterationId {
        self.iteration_id
    }

    /// Safe scheduler disposition, or none when stream quiescence was not proven.
    #[must_use]
    pub const fn abort_disposition(&self) -> Option<ExecutionAbort> {
        self.abort
    }

    /// Safe `(iteration, disposition)` pair for [`crate::Scheduler::abort_iteration`].
    #[must_use]
    pub const fn abort_data(&self) -> Option<(IterationId, ExecutionAbort)> {
        match self.abort {
            Some(abort) => Some((self.iteration_id, abort)),
            None => None,
        }
    }

    /// Underlying adaptation, runtime, or synchronization failure.
    #[must_use]
    pub const fn error(&self) -> &IterationAdapterError {
        &self.error
    }

    /// Consumes the settlement wrapper and returns the underlying failure.
    #[must_use]
    pub fn into_error(self) -> IterationAdapterError {
        self.error
    }
}

#[cfg(feature = "cuda")]
impl fmt::Display for IterationExecutionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.abort {
            Some(abort) => write!(
                formatter,
                "iteration {} failed ({abort:?}): {}",
                self.iteration_id.get(),
                self.error
            ),
            None => write!(
                formatter,
                "iteration {} failed without a safe scheduler settlement: {}",
                self.iteration_id.get(),
                self.error
            ),
        }
    }
}

#[cfg(feature = "cuda")]
impl error::Error for IterationExecutionFailure {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Executes one scheduler plan and synchronously downloads dense BF16 logits.
///
/// Adaptation, executor-bound validation, and all output allocations finish
/// before `PreparedLlamaBatchExecutor::execute` can mutate reserved KV ranges.
/// The stream is explicitly synchronized on every post-dispatch path.
///
/// # Errors
///
/// Returns [`IterationExecutionFailure`] with `NotDispatched` when rollback is
/// safe, `DeviceQuiescedMutationUnknown` after a synchronized dispatch failure,
/// or no disposition when CUDA stream quiescence itself cannot be established.
#[cfg(feature = "cuda")]
pub fn execute_llama_iteration(
    plan: &IterationPlan,
    executor: &mut PreparedLlamaBatchExecutor,
    stream: &mut CudaStream,
) -> Result<DownloadedLlamaIteration, IterationExecutionFailure> {
    execute_llama_iteration_inner(plan, executor, stream, None).map(|(downloaded, _)| downloaded)
}

/// Executes one scheduler plan with exact same-stream CUDA event timings.
///
/// The returned timing covers dispatch through the logits D2H copy. Its idle
/// field is zero for the timer's first successful iteration and otherwise
/// measures from the preceding successful iteration's end event to this
/// iteration's dispatch start event.
///
/// # Errors
///
/// Uses the same settlement contract as [`execute_llama_iteration`]. A timing
/// event failure before dispatch is `NotDispatched`; a post-dispatch event
/// failure is classified only after stream quiescence is attempted.
#[cfg(feature = "cuda")]
pub fn execute_llama_iteration_timed(
    plan: &IterationPlan,
    executor: &mut PreparedLlamaBatchExecutor,
    stream: &mut CudaStream,
    timer: &mut LlamaIterationCudaTimer,
) -> Result<(DownloadedLlamaIteration, IterationTiming), IterationExecutionFailure> {
    execute_llama_iteration_inner(plan, executor, stream, Some(timer))
}

#[cfg(feature = "cuda")]
fn execute_llama_iteration_inner(
    plan: &IterationPlan,
    executor: &mut PreparedLlamaBatchExecutor,
    stream: &mut CudaStream,
    mut timer: Option<&mut LlamaIterationCudaTimer>,
) -> Result<(DownloadedLlamaIteration, IterationTiming), IterationExecutionFailure> {
    let iteration_id = plan.iteration_id();
    let prepared = PreparedLlamaIteration::prepare(plan).map_err(|error| {
        IterationExecutionFailure::new(iteration_id, Some(ExecutionAbort::NotDispatched), error)
    })?;
    let vocabulary_size = executor.vocabulary_size();
    let maximum_position_count = executor.maximum_position_count().map_err(|source| {
        IterationExecutionFailure::new(
            iteration_id,
            Some(ExecutionAbort::NotDispatched),
            IterationAdapterError::Runtime(Box::new(source)),
        )
    })?;
    prepared
        .validate_executor_bounds(
            executor.config().metadata(),
            vocabulary_size,
            maximum_position_count,
        )
        .map_err(|error| {
            IterationExecutionFailure::new(iteration_id, Some(ExecutionAbort::NotDispatched), error)
        })?;

    let output_bytes = executor
        .output_byte_len_for(prepared.output_count)
        .map_err(|source| {
            IterationExecutionFailure::new(
                iteration_id,
                Some(ExecutionAbort::NotDispatched),
                IterationAdapterError::Runtime(Box::new(source)),
            )
        })?;
    let mut logits_bf16_native =
        zeroed_vec(output_bytes, "downloaded BF16 logits").map_err(|error| {
            IterationExecutionFailure::new(iteration_id, Some(ExecutionAbort::NotDispatched), error)
        })?;

    if let Some(timer) = timer.as_deref_mut() {
        timer.record_start(stream).map_err(|error| {
            IterationExecutionFailure::new(iteration_id, Some(ExecutionAbort::NotDispatched), error)
        })?;
    }
    if let Err(source) = executor.execute(prepared.rows(), stream) {
        invalidate_timer(&mut timer);
        return Err(classify_runtime_failure(
            iteration_id,
            IterationAdapterError::Runtime(Box::new(source)),
            stream,
        ));
    }
    if executor.output_count() != prepared.output_count {
        invalidate_timer(&mut timer);
        return Err(classify_runtime_failure(
            iteration_id,
            IterationAdapterError::InvalidRuntimeOutput {
                field: "output_count",
                reason: "executor output count differs from the immutable plan",
            },
            stream,
        ));
    }
    if let Err(source) = executor.download_logits(&mut logits_bf16_native, stream) {
        invalidate_timer(&mut timer);
        return Err(classify_runtime_failure(
            iteration_id,
            IterationAdapterError::Runtime(Box::new(source)),
            stream,
        ));
    }
    let timing = match timer.as_deref_mut() {
        Some(timer) => match timer.record_end_and_measure(stream) {
            Ok(timing) => timing,
            Err(error) => {
                timer.invalidate_previous_end();
                return Err(classify_runtime_failure(iteration_id, error, stream));
            }
        },
        None => IterationTiming::default(),
    };
    if let Err(source) = stream.synchronize() {
        invalidate_timer(&mut timer);
        return Err(IterationExecutionFailure::new(
            iteration_id,
            None,
            IterationAdapterError::Synchronization {
                preceding: None,
                source,
            },
        ));
    }

    Ok((
        DownloadedLlamaIteration {
            iteration_id,
            vocabulary_size,
            output_count: prepared.output_count,
            logits_bf16_native,
            commit_outputs: prepared.commit_outputs,
        },
        timing,
    ))
}

#[cfg(feature = "cuda")]
fn invalidate_timer(timer: &mut Option<&mut LlamaIterationCudaTimer>) {
    if let Some(timer) = timer.as_deref_mut() {
        timer.invalidate_previous_end();
    }
}

#[cfg(feature = "cuda")]
fn classify_runtime_failure(
    iteration_id: IterationId,
    error: IterationAdapterError,
    stream: &mut CudaStream,
) -> IterationExecutionFailure {
    match stream.synchronize() {
        Ok(()) => IterationExecutionFailure::new(
            iteration_id,
            Some(ExecutionAbort::DeviceQuiescedMutationUnknown),
            error,
        ),
        Err(source) => IterationExecutionFailure::new(
            iteration_id,
            None,
            IterationAdapterError::Synchronization {
                preceding: Some(Box::new(error)),
                source,
            },
        ),
    }
}

fn adapt_item<'plan>(
    plan: &'plan IterationPlan,
    item: &'plan WorkItem,
    expected_kind: WorkKind,
) -> IterationAdapterResult<LlamaBatchRow<'plan>> {
    if item.kind() != expected_kind {
        return Err(IterationAdapterError::InvalidPlan {
            field: "work kind",
            reason: "item is stored in the wrong stage vector",
        });
    }
    if item.input_tokens().is_empty() {
        return Err(IterationAdapterError::InvalidPlan {
            field: "input_tokens",
            reason: "every runtime row must contain at least one input token",
        });
    }
    if expected_kind == WorkKind::Decode
        && (item.input_tokens().len() != 1 || item.output_slot().is_none())
    {
        return Err(IterationAdapterError::InvalidPlan {
            field: "decode item",
            reason: "decode rows require exactly one input token and one output slot",
        });
    }
    let target_logical_length = u32::try_from(item.target_logical_length()).map_err(|_| {
        IterationAdapterError::ArithmeticOverflow {
            field: "target logical length",
        }
    })?;
    let table = plan.block_tables().get(item.block_table_index()).ok_or(
        IterationAdapterError::InvalidPlan {
            field: "block_table_index",
            reason: "work item references a missing block table",
        },
    )?;
    if table.request_id() != item.request_id() {
        return Err(IterationAdapterError::InvalidPlan {
            field: "block_table_index",
            reason: "work item references another request's block table",
        });
    }
    if table.logical_length() != target_logical_length {
        return Err(IterationAdapterError::InvalidPlan {
            field: "block table logical length",
            reason: "block table length differs from the work target",
        });
    }
    let kind = match expected_kind {
        WorkKind::Prefill => LlamaBatchRowKind::Prefill,
        WorkKind::Decode => LlamaBatchRowKind::Decode,
    };
    Ok(LlamaBatchRow::new(
        item.request_id().get(),
        kind,
        item.input_tokens(),
        target_logical_length,
        LlamaBatchBlockTable::new(
            table.schema_version(),
            table.physical_block_ids(),
            table.valid_tokens(),
            table.logical_length(),
        ),
        item.output_slot().map(OutputSlot::get),
    ))
}

#[cfg(any(feature = "cuda", test))]
fn ensure_capacity(
    resource: &'static str,
    requested: usize,
    capacity: usize,
) -> IterationAdapterResult<()> {
    if requested <= capacity {
        Ok(())
    } else {
        Err(IterationAdapterError::CapacityExceeded {
            resource,
            requested,
            capacity,
        })
    }
}

fn reserve_vec<T>(capacity: usize, resource: &'static str) -> IterationAdapterResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| IterationAdapterError::HostAllocation {
            resource,
            requested_elements: capacity,
        })?;
    Ok(values)
}

#[cfg(feature = "cuda")]
fn zeroed_vec(length: usize, resource: &'static str) -> IterationAdapterResult<Vec<u8>> {
    let mut values = reserve_vec(length, resource)?;
    values.resize(length, 0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use rustinfer_runtime::paged_kv::BLOCK_TABLE_V1_VERSION;

    use super::{
        DownloadedLlamaIteration, IterationAdapterError, IterationTiming, PreparedLlamaIteration,
        SampledIterationToken,
    };
    use crate::plan::{
        IterationId, IterationPlan, OutputSlot, OwnedBlockTable, RequestId, WorkItem, WorkKind,
    };

    fn request(value: u64) -> RequestId {
        RequestId::new(value).expect("nonzero request")
    }

    fn table(request_id: RequestId, block: u32, length: u32) -> OwnedBlockTable {
        let block_count = usize::try_from(length)
            .expect("small length")
            .div_ceil(rustinfer_runtime::paged_kv::KV_BLOCK_SIZE);
        let ids = (0..block_count)
            .map(|offset| block + u32::try_from(offset).expect("small offset"))
            .collect();
        let valid = (0..block_count)
            .map(|index| {
                let remaining = usize::try_from(length).expect("small length")
                    - index * rustinfer_runtime::paged_kv::KV_BLOCK_SIZE;
                u16::try_from(remaining.min(rustinfer_runtime::paged_kv::KV_BLOCK_SIZE))
                    .expect("block width fits u16")
            })
            .collect();
        OwnedBlockTable::new(request_id, BLOCK_TABLE_V1_VERSION, ids, valid, length)
            .expect("valid table")
    }

    fn mixed_plan() -> IterationPlan {
        let prefill_id = request(1);
        let decode_id = request(2);
        IterationPlan::new(
            IterationId::new(9).expect("nonzero iteration"),
            vec![
                WorkItem::new(prefill_id, WorkKind::Prefill, vec![10, 11], 2, 0, None)
                    .expect("valid prefill"),
            ],
            vec![
                WorkItem::new(
                    decode_id,
                    WorkKind::Decode,
                    vec![12],
                    17,
                    1,
                    Some(OutputSlot::new(0)),
                )
                .expect("valid decode"),
            ],
            vec![table(prefill_id, 0, 2), table(decode_id, 4, 17)],
        )
        .expect("valid mixed plan")
    }

    #[test]
    fn plan_is_adapted_in_prefill_then_decode_order_without_copying_payloads() {
        let plan = mixed_plan();
        let prepared = PreparedLlamaIteration::prepare(&plan).expect("adapt plan");

        assert_eq!(prepared.iteration_id().get(), 9);
        assert_eq!(prepared.total_input_tokens(), 3);
        assert_eq!(prepared.total_block_entries(), 3);
        assert_eq!(prepared.output_count(), 1);
        assert_eq!(prepared.rows().len(), 2);
        assert_eq!(prepared.rows()[0].sequence_tag(), 1);
        assert_eq!(prepared.rows()[0].input_token_ids(), &[10, 11]);
        assert_eq!(prepared.rows()[0].output_slot(), None);
        assert_eq!(prepared.rows()[1].sequence_tag(), 2);
        assert_eq!(prepared.rows()[1].input_token_ids(), &[12]);
        assert_eq!(prepared.rows()[1].output_slot(), Some(0));
        assert_eq!(
            prepared.rows()[1].block_table().physical_block_ids(),
            &[4, 5]
        );
    }

    #[test]
    fn plan_boundary_rejects_nondense_slots_and_decode_without_output() {
        let request_id = request(1);
        let nondense = IterationPlan::new(
            IterationId::new(1).expect("iteration"),
            vec![
                WorkItem::new(
                    request_id,
                    WorkKind::Prefill,
                    vec![1],
                    1,
                    0,
                    Some(OutputSlot::new(7)),
                )
                .expect("work item"),
            ],
            Vec::new(),
            vec![table(request_id, 0, 1)],
        )
        .expect_err("plan boundary must reject a non-dense slot");
        assert!(matches!(
            nondense,
            crate::SchedulerError::InvalidPlan {
                field: "output_slots",
                ..
            }
        ));

        let decode_without_output = IterationPlan::new(
            IterationId::new(2).expect("iteration"),
            Vec::new(),
            vec![
                WorkItem::new(request_id, WorkKind::Decode, vec![1], 1, 0, None)
                    .expect("work item"),
            ],
            vec![table(request_id, 0, 1)],
        )
        .expect_err("plan boundary must reject output-free decode");
        assert!(matches!(
            decode_without_output,
            crate::SchedulerError::InvalidPlan {
                field: "output_slots",
                ..
            }
        ));
    }

    #[test]
    fn physical_blocks_must_be_exclusive_across_rows() {
        let first = request(1);
        let second = request(2);
        let error = IterationPlan::new(
            IterationId::new(3).expect("iteration"),
            vec![
                WorkItem::new(first, WorkKind::Prefill, vec![1], 1, 0, None).expect("first item"),
                WorkItem::new(second, WorkKind::Prefill, vec![2], 1, 1, None).expect("second item"),
            ],
            Vec::new(),
            vec![table(first, 0, 1), table(second, 0, 1)],
        )
        .expect_err("plan boundary must reject cross-row duplicate blocks");

        assert!(matches!(
            error,
            crate::SchedulerError::InvalidPlan {
                field: "block_tables",
                ..
            }
        ));
    }

    #[test]
    fn executor_bounds_are_rechecked_without_device_work() {
        let plan = mixed_plan();
        let prepared = PreparedLlamaIteration::prepare(&plan).expect("adapt plan");
        let too_few_rows = rustinfer_runtime::llama::LlamaBatchMetadataConfig::new(1, 3, 3, 1, 8)
            .expect("valid bounds");
        assert!(matches!(
            prepared.validate_executor_bounds(too_few_rows, 32, 32),
            Err(IterationAdapterError::CapacityExceeded {
                resource: "rows",
                requested: 2,
                capacity: 1,
            })
        ));

        let bounds = rustinfer_runtime::llama::LlamaBatchMetadataConfig::new(2, 3, 3, 1, 8)
            .expect("valid bounds");
        assert!(matches!(
            prepared.validate_executor_bounds(bounds, 12, 32),
            Err(IterationAdapterError::TokenOutOfRange { token_id: 12, .. })
        ));
        assert!(matches!(
            prepared.validate_executor_bounds(bounds, 32, 16),
            Err(IterationAdapterError::PositionOutOfRange {
                target_logical_length: 17,
                ..
            })
        ));
        assert!(matches!(
            prepared.validate_executor_bounds(bounds, 32, 32),
            Ok(())
        ));
    }

    #[test]
    fn downloaded_rows_map_dense_slots_to_commit_results_without_reallocation() {
        let mut commit_outputs = Vec::with_capacity(2);
        commit_outputs.clear();
        let downloaded = DownloadedLlamaIteration {
            iteration_id: IterationId::new(7).expect("iteration"),
            vocabulary_size: 3,
            output_count: 2,
            logits_bf16_native: vec![0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
            commit_outputs,
        };

        assert_eq!(
            downloaded
                .logits_for_slot(OutputSlot::new(0))
                .expect("slot zero"),
            &[0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            downloaded
                .logits_for_slot(OutputSlot::new(1))
                .expect("slot one"),
            &[10, 11, 12, 13, 14, 15]
        );
        assert!(downloaded.logits_for_slot(OutputSlot::new(2)).is_err());

        let result = downloaded
            .into_result(
                &[
                    SampledIterationToken::new(2, false),
                    SampledIterationToken::new(1, true),
                ],
                IterationTiming::new(80, 5),
            )
            .expect("commit data");
        assert_eq!(result.iteration_id().get(), 7);
        assert_eq!(result.gpu_execution_ns(), 80);
        assert_eq!(result.gpu_idle_gap_ns(), 5);
        assert_eq!(result.outputs()[0].slot(), OutputSlot::new(0));
        assert_eq!(result.outputs()[0].token_id(), 2);
        assert!(!result.outputs()[0].stop());
        assert_eq!(result.outputs()[1].slot(), OutputSlot::new(1));
        assert_eq!(result.outputs()[1].token_id(), 1);
        assert!(result.outputs()[1].stop());
    }

    #[test]
    fn commit_failure_retains_download_and_safe_abort_data() {
        let downloaded = DownloadedLlamaIteration {
            iteration_id: IterationId::new(8).expect("iteration"),
            vocabulary_size: 2,
            output_count: 1,
            logits_bf16_native: vec![0; 4],
            commit_outputs: Vec::with_capacity(1),
        };
        let failure = downloaded
            .into_result(&[], IterationTiming::default())
            .expect_err("sample count must fail");
        assert!(matches!(
            failure.error(),
            IterationAdapterError::InvalidSampleCount {
                expected: 1,
                actual: 0,
            }
        ));
        assert_eq!(failure.abort_data().0.get(), 8);
        assert_eq!(
            failure.abort_data().1,
            crate::ExecutionAbort::DeviceQuiescedMutationUnknown
        );
        assert_eq!(failure.iteration().logits_bf16_native(), &[0; 4]);
    }
}
