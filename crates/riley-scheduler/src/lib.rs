//! Bounded Rust-native admission and continuous-iteration planning.
//!
//! The scheduler owns host-side request and paged-KV metadata. It produces an
//! immutable iteration plan, but does not launch GPU work itself.
//!
//! # Example
//!
//! ```
//! use riley_runtime::paged_kv::KvLayout;
//! use riley_scheduler::{
//!     ExecutionAbort, OverloadPolicy, RequestDescriptor, RequestFinishReason,
//!     RequestState, Scheduler, SchedulerConfig,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // This example only creates host-side metadata; it does not initialize CUDA.
//! let layout = KvLayout::checked(1, 8, 1, 64)?;
//! let config = SchedulerConfig {
//!     max_waiting_requests: 4,
//!     max_waiting_prompt_tokens: 64,
//!     max_active_sequences: 2,
//!     max_sequence_tokens: 32,
//!     iteration_token_budget: 8,
//!     max_prefill_chunk_tokens: 8,
//!     aging_threshold_ns: 1_000,
//!     overload_policy: OverloadPolicy::Wait,
//!     admission_timeout_ns: None,
//!     max_promised_kv_blocks: 8,
//!     metrics_window_samples: 16,
//! };
//! let mut scheduler = Scheduler::new(config, layout)?;
//!
//! let submission = scheduler.submit(RequestDescriptor::new(vec![1, 2, 3], 1), 0)?;
//! assert_eq!(submission.state(), RequestState::Admitted);
//!
//! let planning = scheduler.plan_iteration(1)?;
//! assert!(planning.completions().is_empty());
//! let iteration_id = planning
//!     .plan()
//!     .expect("one admitted request produces a plan")
//!     .iteration_id();
//!
//! // No executor was called, so the detached KV reservation is safe to roll back.
//! let aborted = scheduler.abort_iteration(iteration_id, ExecutionAbort::NotDispatched, 2)?;
//! assert!(aborted.completions().is_empty());
//! assert!(aborted.settlement_failures().is_empty());
//! assert_eq!(
//!     scheduler.request_state(submission.request_id()),
//!     Some(RequestState::Admitted),
//! );
//!
//! let closed = scheduler.close(3, None)?;
//! assert!(closed.settlement_failures().is_empty());
//! assert_eq!(closed.completions().len(), 1);
//! assert_eq!(closed.completions()[0].reason(), RequestFinishReason::Cancelled);
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod error;
pub mod execution;
pub mod metrics;
pub mod plan;
mod scheduler;

pub use config::{OverloadPolicy, SchedulerConfig};
pub use error::{SchedulerError, SchedulerResult};
pub use execution::{
    DownloadedLlamaIteration, IterationAdapterError, IterationAdapterResult,
    IterationCommitFailure, IterationTiming, PreparedLlamaIteration, SampledIterationToken,
};
#[cfg(feature = "cuda")]
pub use execution::{
    IterationExecutionFailure, LlamaIterationCudaTimer, execute_llama_iteration,
    execute_llama_iteration_timed,
};
pub use metrics::{
    IterationMetricSample, MetricWindowSnapshot, SchedulerGauges, SchedulerMetricsSnapshot,
};
pub use plan::{
    ITERATION_SCHEMA_VERSION, IterationId, IterationOutput, IterationPlan, IterationResult,
    OutputSlot, OwnedBlockTable, RequestId, WorkItem, WorkKind,
};
pub use scheduler::{
    CancellationOutcome, ExecutionAbort, IterationUpdates, PlanningOutput, RequestCompletion,
    RequestDescriptor, RequestFinishReason, RequestSettlementFailure, RequestSnapshot,
    RequestState, Scheduler, SchedulerCloseFailure, SchedulerCloseOutput, Submission, TokenEvent,
};

/// Returns lower-layer build information without starting a scheduler.
///
/// # Errors
///
/// Propagates a native-contract error from a CUDA-enabled runtime.
pub fn runtime_build_info() -> riley_core::Result<riley_runtime::BuildInfo> {
    riley_runtime::build_info()
}
