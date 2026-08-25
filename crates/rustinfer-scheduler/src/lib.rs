//! Bounded Rust-native admission and continuous-iteration planning.

pub mod config;
pub mod error;
pub mod metrics;
pub mod plan;
mod scheduler;

pub use config::{OverloadPolicy, SchedulerConfig};
pub use error::{SchedulerError, SchedulerResult};
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
pub fn runtime_build_info() -> rustinfer_core::Result<rustinfer_runtime::BuildInfo> {
    rustinfer_runtime::build_info()
}
