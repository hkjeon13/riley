//! Stable scheduler errors and the crate-local result alias.

use std::error;
use std::fmt;

use crate::plan::{IterationId, RequestId};

/// Result type for scheduler admission, planning, and accounting operations.
pub type SchedulerResult<T> = Result<T, SchedulerError>;

/// A checked scheduler configuration, capacity, state, or protocol failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SchedulerError {
    /// A scheduler configuration value violates a documented invariant.
    InvalidConfiguration {
        /// Invalid configuration field.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// Checked integer arithmetic failed before state was mutated.
    ArithmeticOverflow {
        /// Counter, size, or capacity being calculated.
        field: &'static str,
    },
    /// A bounded host allocation could not reserve its full capacity.
    HostAllocation {
        /// Buffer whose allocation failed.
        resource: &'static str,
        /// Requested element capacity.
        requested_elements: usize,
    },
    /// A previous failed operation produced terminal notifications that the
    /// caller must recover before issuing another mutating operation.
    PendingCompletions {
        /// Number of recoverable notifications in the bounded outbox.
        count: usize,
    },
    /// A terminal operation would exceed the scheduler-owned completion outbox.
    CompletionBacklogCapacity {
        /// Fixed outbox capacity established at scheduler construction.
        limit: usize,
        /// Notifications already waiting for recovery.
        pending: usize,
        /// Additional notifications the operation may produce.
        needed: usize,
    },
    /// Admission was attempted after shutdown began.
    SchedulerClosed,
    /// A consuming close needs an explicit executor safety assertion for the
    /// outstanding immutable plan.
    CloseDispositionRequired {
        /// Iteration that must be completed or explicitly aborted.
        iteration_id: IterationId,
    },
    /// A scheduler-issued identifier cannot advance without wrapping.
    IdentifierExhausted {
        /// Identifier namespace that was exhausted.
        kind: &'static str,
    },
    /// A caller-provided clock moved backwards.
    ClockRegression {
        /// Most recent accepted monotonic timestamp.
        previous_ns: u64,
        /// Rejected monotonic timestamp.
        current_ns: u64,
    },
    /// The bounded waiting-request queue is full.
    WaitingQueueFull {
        /// Configured maximum number of waiting requests.
        limit: usize,
    },
    /// Admitting a prompt would exceed the bounded waiting-token total.
    WaitingTokenLimit {
        /// Configured maximum number of waiting prompt tokens.
        limit: usize,
        /// Total that the admission would require.
        requested: usize,
    },
    /// A request cannot fit the configured per-sequence token limit.
    SequenceTokenLimit {
        /// Configured maximum sequence length.
        limit: usize,
        /// Total prompt and generation capacity requested.
        requested: usize,
    },
    /// The active-sequence admission bound is exhausted.
    ActiveSequenceLimit {
        /// Configured maximum number of active sequences.
        limit: usize,
    },
    /// Promised KV capacity would exceed the scheduler's fixed quota.
    KvCapacityExceeded {
        /// Number of additional blocks requested.
        requested_blocks: usize,
        /// Number of unpromised blocks remaining.
        available_blocks: usize,
    },
    /// A queued request exceeded the configured admission deadline.
    AdmissionTimedOut {
        /// Timed-out request.
        request_id: RequestId,
        /// Time spent waiting for admission.
        waited_ns: u64,
    },
    /// No request with this scheduler-issued identifier exists.
    UnknownRequest {
        /// Unknown request identifier.
        request_id: RequestId,
    },
    /// A request state transition is not part of the scheduler state machine.
    InvalidStateTransition {
        /// Request whose transition was rejected.
        request_id: RequestId,
        /// Stable source-state name.
        from: &'static str,
        /// Stable destination-state name.
        to: &'static str,
    },
    /// Planning was requested while another immutable plan is outstanding.
    IterationInFlight {
        /// Outstanding iteration identifier.
        iteration_id: IterationId,
    },
    /// Runtime feedback arrived when no iteration is outstanding.
    NoIterationInFlight,
    /// Runtime feedback names a stale, replayed, or otherwise unexpected plan.
    UnexpectedIteration {
        /// Outstanding iteration identifier.
        expected: IterationId,
        /// Identifier carried by runtime feedback.
        actual: IterationId,
    },
    /// A plan or runtime-result DTO violates its versioned schema.
    UnsupportedSchemaVersion {
        /// DTO whose schema is unsupported.
        resource: &'static str,
        /// Only version accepted by this crate build.
        expected: u16,
        /// Version found in the DTO.
        actual: u16,
    },
    /// A plan assembled inside the scheduler violates a structural invariant.
    InvalidPlan {
        /// Plan field or relationship that is invalid.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// Runtime feedback does not exactly match the outstanding plan contract.
    InvalidIterationResult {
        /// Result field or relationship that is invalid.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// A scheduler metric sample or gauge snapshot is internally inconsistent.
    InvalidMetricSample {
        /// Metric field or relationship that is invalid.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// A monotonically increasing metric counter would overflow.
    MetricOverflow {
        /// Metric that could not be advanced.
        metric: &'static str,
    },
    /// A paged-KV ownership or reservation operation failed.
    PagedKv(riley_runtime::paged_kv::PagedKvError),
}

#[allow(clippy::too_many_lines)]
impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid scheduler configuration {field}: {reason}"
                )
            }
            Self::ArithmeticOverflow { field } => {
                write!(formatter, "scheduler arithmetic overflow for {field}")
            }
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "could not reserve {requested_elements} host elements for {resource}"
            ),
            Self::PendingCompletions { count } => write!(
                formatter,
                "scheduler has {count} pending completion notifications; drain them before mutating state"
            ),
            Self::CompletionBacklogCapacity {
                limit,
                pending,
                needed,
            } => write!(
                formatter,
                "completion outbox capacity {limit} cannot hold {needed} additional notifications with {pending} already pending"
            ),
            Self::SchedulerClosed => formatter.write_str("scheduler admission is closed"),
            Self::CloseDispositionRequired { iteration_id } => write!(
                formatter,
                "closing scheduler requires an abort disposition for in-flight iteration {}",
                iteration_id.get()
            ),
            Self::IdentifierExhausted { kind } => {
                write!(formatter, "scheduler {kind} identifiers exhausted")
            }
            Self::ClockRegression {
                previous_ns,
                current_ns,
            } => write!(
                formatter,
                "scheduler monotonic clock regressed from {previous_ns} ns to {current_ns} ns"
            ),
            Self::WaitingQueueFull { limit } => {
                write!(
                    formatter,
                    "scheduler waiting queue reached its {limit}-request limit"
                )
            }
            Self::WaitingTokenLimit { limit, requested } => write!(
                formatter,
                "waiting prompts require {requested} tokens, exceeding the {limit}-token limit"
            ),
            Self::SequenceTokenLimit { limit, requested } => write!(
                formatter,
                "request requires {requested} sequence tokens, exceeding the {limit}-token limit"
            ),
            Self::ActiveSequenceLimit { limit } => write!(
                formatter,
                "scheduler reached its {limit}-sequence active limit"
            ),
            Self::KvCapacityExceeded {
                requested_blocks,
                available_blocks,
            } => write!(
                formatter,
                "request promises {requested_blocks} KV blocks but only {available_blocks} remain"
            ),
            Self::AdmissionTimedOut {
                request_id,
                waited_ns,
            } => write!(
                formatter,
                "request {} timed out after waiting {waited_ns} ns for admission",
                request_id.get()
            ),
            Self::UnknownRequest { request_id } => {
                write!(formatter, "unknown scheduler request {}", request_id.get())
            }
            Self::InvalidStateTransition {
                request_id,
                from,
                to,
            } => write!(
                formatter,
                "request {} cannot transition from {from} to {to}",
                request_id.get()
            ),
            Self::IterationInFlight { iteration_id } => write!(
                formatter,
                "iteration {} is still in flight",
                iteration_id.get()
            ),
            Self::NoIterationInFlight => formatter.write_str("no scheduler iteration is in flight"),
            Self::UnexpectedIteration { expected, actual } => write!(
                formatter,
                "runtime result names iteration {}, expected {}",
                actual.get(),
                expected.get()
            ),
            Self::UnsupportedSchemaVersion {
                resource,
                expected,
                actual,
            } => write!(
                formatter,
                "unsupported {resource} schema version {actual}; expected {expected}"
            ),
            Self::InvalidPlan { field, reason } => {
                write!(formatter, "invalid iteration plan {field}: {reason}")
            }
            Self::InvalidIterationResult { field, reason } => {
                write!(formatter, "invalid iteration result {field}: {reason}")
            }
            Self::InvalidMetricSample { field, reason } => {
                write!(
                    formatter,
                    "invalid scheduler metric sample {field}: {reason}"
                )
            }
            Self::MetricOverflow { metric } => {
                write!(formatter, "scheduler metric {metric} overflowed")
            }
            Self::PagedKv(source) => write!(formatter, "paged-KV scheduler failure: {source}"),
        }
    }
}

impl error::Error for SchedulerError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::PagedKv(source) => Some(source),
            _ => None,
        }
    }
}

impl From<riley_runtime::paged_kv::PagedKvError> for SchedulerError {
    fn from(source: riley_runtime::paged_kv::PagedKvError) -> Self {
        Self::PagedKv(source)
    }
}
