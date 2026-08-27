//! Bounded inference worker and request-local streaming protocol.
//!
//! The HTTP layer never owns scheduler or CUDA state.  It submits one
//! transport-independent request through a bounded command queue and receives
//! ordered events through a request-local bounded channel.  Cancellation is a
//! request-local atomic bit, so disconnect handling does not depend on finding
//! spare capacity in the command queue.

use std::collections::BTreeMap;
use std::error;
use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{
    self, Receiver, RecvError, RecvTimeoutError, SyncSender, TryRecvError, TrySendError,
};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use riley_scheduler::SchedulerMetricsSnapshot;

use crate::domain::{
    GenerationEvent, GenerationRequest, ModelMetadata, RequestMetadata, ServiceErrorClass,
};

const REQUEST_LIVE: u8 = 0;
const REQUEST_CANCELLED: u8 = 1;
const REQUEST_TERMINAL: u8 = 2;
const REQUEST_CANCELLING: u8 = 3;

const LIFECYCLE_STARTING: u8 = 0;
const LIFECYCLE_READY: u8 = 1;
const LIFECYCLE_DRAINING: u8 = 2;
const LIFECYCLE_STOPPED: u8 = 3;
const LIFECYCLE_FAILED: u8 = 4;

/// Stable worker-local request key.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct EngineRequestId(u64);

impl EngineRequestId {
    /// Numeric identity used only inside the process.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Fixed bounds and timing policy for one inference worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineConfig {
    /// Number of admission commands that may wait for the worker.
    pub command_queue_capacity: usize,
    /// Number of events buffered independently for each request.
    pub event_channel_capacity: usize,
    /// Maximum requests retained by the engine protocol at once.
    pub max_inflight_requests: usize,
    /// Maximum time a submitter waits for backend admission.
    pub admission_timeout: Duration,
    /// Worker wait between empty backend ticks.
    pub idle_poll_interval: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            command_queue_capacity: 256,
            event_channel_capacity: 16,
            max_inflight_requests: 256,
            admission_timeout: Duration::from_secs(5),
            idle_poll_interval: Duration::from_millis(1),
        }
    }
}

impl EngineConfig {
    fn validate(self) -> Result<Self, EngineError> {
        if self.command_queue_capacity == 0 {
            return Err(EngineError::InvalidConfiguration {
                field: "command_queue_capacity",
                reason: "must be at least one",
            });
        }
        if self.event_channel_capacity == 0 {
            return Err(EngineError::InvalidConfiguration {
                field: "event_channel_capacity",
                reason: "must be at least one",
            });
        }
        if self.max_inflight_requests == 0 {
            return Err(EngineError::InvalidConfiguration {
                field: "max_inflight_requests",
                reason: "must be at least one",
            });
        }
        if self.admission_timeout.is_zero() {
            return Err(EngineError::InvalidConfiguration {
                field: "admission_timeout",
                reason: "must be non-zero",
            });
        }
        if self.idle_poll_interval.is_zero() {
            return Err(EngineError::InvalidConfiguration {
                field: "idle_poll_interval",
                reason: "must be non-zero",
            });
        }
        Ok(self)
    }
}

/// Sanitized engine failure suitable for HTTP status mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EngineError {
    /// A cold engine setting is invalid.
    InvalidConfiguration {
        /// Invalid field.
        field: &'static str,
        /// Stable explanation.
        reason: &'static str,
    },
    /// A request selected a model other than the loaded model.
    ModelNotFound,
    /// Bounded admission capacity is exhausted.
    Overloaded,
    /// The worker did not finish admission before the configured deadline.
    Timeout,
    /// Admission is disabled because shutdown has started.
    ShuttingDown,
    /// The worker stopped unexpectedly.
    WorkerUnavailable,
    /// A request violated a model/backend invariant.
    InvalidRequest,
    /// A non-sensitive internal failure classification.
    Internal,
}

impl EngineError {
    /// Stable service error class when this failure can be represented by a
    /// generation event.
    #[must_use]
    pub const fn service_class(&self) -> ServiceErrorClass {
        match self {
            Self::Overloaded => ServiceErrorClass::Overloaded,
            Self::Timeout => ServiceErrorClass::Timeout,
            Self::ShuttingDown => ServiceErrorClass::ShuttingDown,
            Self::ModelNotFound | Self::InvalidRequest => ServiceErrorClass::InvalidRequest,
            Self::InvalidConfiguration { .. } | Self::WorkerUnavailable | Self::Internal => {
                ServiceErrorClass::Internal
            }
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(formatter, "invalid engine configuration {field}: {reason}")
            }
            Self::ModelNotFound => formatter.write_str("requested model is not loaded"),
            Self::Overloaded => formatter.write_str("inference admission is overloaded"),
            Self::Timeout => formatter.write_str("inference admission timed out"),
            Self::ShuttingDown => formatter.write_str("inference engine is shutting down"),
            Self::WorkerUnavailable => formatter.write_str("inference worker is unavailable"),
            Self::InvalidRequest => formatter.write_str("generation request is invalid"),
            Self::Internal => formatter.write_str("internal inference failure"),
        }
    }
}

impl error::Error for EngineError {}

/// Internal backend failure.  `detail` is retained for server-side diagnostics
/// and is never copied into [`GenerationEvent`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendError {
    class: ServiceErrorClass,
    detail: String,
}

impl BackendError {
    /// Creates a backend failure with a sanitized public class and private
    /// diagnostic detail.
    #[must_use]
    pub fn new(class: ServiceErrorClass, detail: impl Into<String>) -> Self {
        Self {
            class,
            detail: detail.into(),
        }
    }

    /// Public event classification.
    #[must_use]
    pub const fn class(&self) -> ServiceErrorClass {
        self.class
    }

    /// Server-side diagnostic.  HTTP encoders must not expose this string.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "backend {:?}: {}", self.class, self.detail)
    }
}

impl error::Error for BackendError {}

#[cfg(any(feature = "cuda", test))]
fn private_request_error(operation: &'static str) -> BackendError {
    BackendError::new(ServiceErrorClass::InvalidRequest, operation)
}

#[cfg(any(feature = "cuda", test))]
fn visible_utf8_prefix(bytes: &[u8]) -> &str {
    let valid_length =
        std::str::from_utf8(bytes).map_or_else(|source| source.valid_up_to(), str::len);
    std::str::from_utf8(&bytes[..valid_length]).unwrap_or("")
}

/// One event returned by a backend after its state transition is committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendEvent {
    request_id: EngineRequestId,
    event: GenerationEvent,
}

impl BackendEvent {
    /// Associates an ordered domain event with a worker request.
    #[must_use]
    pub const fn new(request_id: EngineRequestId, event: GenerationEvent) -> Self {
        Self { request_id, event }
    }

    /// Target request.
    #[must_use]
    pub const fn request_id(&self) -> EngineRequestId {
        self.request_id
    }

    /// Borrowed domain event.
    #[must_use]
    pub const fn event(&self) -> &GenerationEvent {
        &self.event
    }

    fn into_parts(self) -> (EngineRequestId, GenerationEvent) {
        (self.request_id, self.event)
    }
}

/// Backend protocol owned and called exclusively by the inference worker.
///
/// `step` must return token events only after the backend's authoritative
/// scheduler commit succeeds.  A call must be finite so cancellation and
/// shutdown are observed at iteration boundaries.
pub trait EngineBackend: Send + 'static {
    /// Validates, tokenizes, and admits a request into backend-owned state.
    ///
    /// # Errors
    ///
    /// Returns a request-scoped admission failure without retaining the key.
    fn admit(
        &mut self,
        request_id: EngineRequestId,
        metadata: &RequestMetadata,
        request: &GenerationRequest,
    ) -> Result<(), BackendError>;

    /// Cancels queued or active work at a safe iteration boundary.
    ///
    /// Returned events are published in vector order.
    ///
    /// # Errors
    ///
    /// Returns when backend ownership cannot be safely reclaimed.
    fn cancel(&mut self, request_id: EngineRequestId) -> Result<Vec<BackendEvent>, BackendError>;

    /// Advances at most one finite backend iteration.
    ///
    /// # Errors
    ///
    /// Returns a fatal backend error.  The worker fails all live requests and
    /// enters shutdown.
    fn step(&mut self) -> Result<Vec<BackendEvent>, BackendError>;

    /// Whether no admitted work remains.
    #[must_use]
    fn is_idle(&self) -> bool;

    /// Returns the scheduler's sanitized, bounded operational snapshot when
    /// the backend owns one.
    ///
    /// # Errors
    ///
    /// Returns when the backend cannot produce a consistent snapshot.
    fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, BackendError> {
        Ok(EngineMetricsSnapshot::default())
    }

    /// Returns metrics captured only after a successful explicit shutdown.
    #[must_use]
    fn final_metrics_snapshot(&self) -> EngineMetricsSnapshot {
        EngineMetricsSnapshot::default()
    }

    /// Cancels remaining work and explicitly closes owned resources.
    ///
    /// # Errors
    ///
    /// Returns after attempting backend cleanup when an ownership or device
    /// close failed.
    fn shutdown(&mut self) -> Result<Vec<BackendEvent>, BackendError>;
}

struct RequestControl {
    cancellation: Arc<AtomicU8>,
    events: SyncSender<GenerationEvent>,
    stats: Arc<EngineStats>,
}

impl Drop for RequestControl {
    fn drop(&mut self) {
        self.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

enum Command {
    Submit {
        request_id: EngineRequestId,
        metadata: RequestMetadata,
        request: GenerationRequest,
        cancellation: Arc<AtomicU8>,
        events: SyncSender<GenerationEvent>,
        admitted: SyncSender<Result<(), EngineError>>,
    },
    Metrics {
        response: SyncSender<Result<EngineMetricsSnapshot, EngineError>>,
    },
    Wake,
}

struct Lifecycle {
    state: AtomicU8,
    notification_lock: Mutex<()>,
    notification: Condvar,
}

impl Lifecycle {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(LIFECYCLE_STARTING),
            notification_lock: Mutex::new(()),
            notification: Condvar::new(),
        }
    }

    fn load(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    fn store(&self, state: u8) {
        self.state.store(state, Ordering::Release);
        self.notification.notify_all();
    }

    fn begin_shutdown(&self) {
        let _ = self.state.compare_exchange(
            LIFECYCLE_READY,
            LIFECYCLE_DRAINING,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.notification.notify_all();
    }

    fn wait_until(&self, deadline: Duration, predicate: impl Fn(u8) -> bool) -> bool {
        if predicate(self.load()) {
            return true;
        }
        let guard = match self.notification_lock.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let (_guard, _) = match self
            .notification
            .wait_timeout_while(guard, deadline, |()| !predicate(self.load()))
        {
            Ok(result) => result,
            Err(poisoned) => poisoned.into_inner(),
        };
        predicate(self.load())
    }
}

struct EngineInner {
    config: EngineConfig,
    model: ModelMetadata,
    commands: SyncSender<Command>,
    lifecycle: Arc<Lifecycle>,
    next_request_id: AtomicU64,
    stats: Arc<EngineStats>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

struct EngineStats {
    active_requests: AtomicUsize,
    waiting_requests: AtomicUsize,
    final_metrics: Mutex<Option<EngineMetricsSnapshot>>,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            active_requests: AtomicUsize::new(0),
            waiting_requests: AtomicUsize::new(0),
            final_metrics: Mutex::new(None),
        }
    }
}

impl EngineStats {
    fn store_final_metrics(&self, metrics: &EngineMetricsSnapshot) {
        *self
            .final_metrics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(*metrics);
    }

    fn final_metrics(&self) -> Option<EngineMetricsSnapshot> {
        *self
            .final_metrics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Cloneable handle to the single inference worker.
#[derive(Clone)]
pub struct InferenceEngine {
    inner: Arc<EngineInner>,
}

/// Short compatibility name used by the service layer.
pub type Engine = InferenceEngine;

/// Lock-free lifecycle and bounded-load snapshot for readiness endpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineStatus {
    /// Worker initialization completed successfully.
    pub ready: bool,
    /// New admissions are currently accepted.
    pub accepting: bool,
    /// Requests admitted into the backend.
    pub active_requests: usize,
    /// Submit commands waiting for backend admission.
    pub waiting_requests: usize,
}

/// Native allocation gauges sampled from the backend's owned CUDA context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineAllocationMetrics {
    /// Bytes in live device allocations.
    pub device_live_bytes: u64,
    /// Number of live device allocations.
    pub device_live_allocations: u64,
    /// Bytes in live pinned-host allocations.
    pub pinned_host_live_bytes: u64,
    /// Number of live pinned-host allocations.
    pub pinned_host_live_allocations: u64,
}

/// Maximum number of cold-prepared dense-row buckets exposed by one engine.
///
/// This mirrors the runtime executor's fixed inline capacity without making
/// the no-CUDA server API depend on CUDA-only runtime types.
pub const ENGINE_BATCH_SHAPE_BUCKET_CAPACITY: usize = 10;

/// Active and selected dense-row facts for the latest committed iteration.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineBatchShapeObservation {
    /// Flattened input-token rows executed by the runtime.
    pub active_rows: usize,
    /// Cold-prepared dense rows selected for execution.
    pub selected_dense_rows: usize,
    /// Inactive rows executed only to fill the selected dense shape.
    pub padding_rows: usize,
}

/// Committed execution counters and GPU latency for one dense-row bucket.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineBatchShapeBucketMetrics {
    /// Exact dense-row count represented by this bucket.
    pub dense_rows: usize,
    /// Successfully committed iterations that selected this bucket.
    pub hit_count: u64,
    /// Committed iterations with one CUDA-event latency sample.
    pub latency_sample_count: u64,
    /// Sum of committed GPU execution durations in nanoseconds.
    pub gpu_execution_ns_total: u64,
    /// Largest committed GPU execution duration in nanoseconds.
    pub gpu_execution_ns_maximum: u64,
    /// Most recent committed GPU execution duration in nanoseconds.
    pub gpu_execution_ns_last: u64,
}

impl EngineBatchShapeBucketMetrics {
    /// Mean committed GPU execution duration in nanoseconds.
    #[must_use]
    pub const fn gpu_execution_ns_average(self) -> Option<u64> {
        if self.latency_sample_count == 0 {
            None
        } else {
            Some(self.gpu_execution_ns_total / self.latency_sample_count)
        }
    }
}

/// Fixed-capacity batch-shape metrics collected on the exclusive CUDA worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineBatchShapeMetricsSnapshot {
    /// Sticky signal that a non-authoritative shape metric could not be updated.
    pub metrics_degraded: bool,
    /// Latest successfully committed iteration, or none before the first hit.
    pub last: Option<EngineBatchShapeObservation>,
    /// Number of valid entries in [`Self::buckets`].
    pub bucket_count: usize,
    /// Cold-prepared buckets followed by zeroed unused entries.
    pub buckets: [EngineBatchShapeBucketMetrics; ENGINE_BATCH_SHAPE_BUCKET_CAPACITY],
}

impl Default for EngineBatchShapeMetricsSnapshot {
    fn default() -> Self {
        Self {
            metrics_degraded: false,
            last: None,
            bucket_count: 0,
            buckets: [EngineBatchShapeBucketMetrics::default(); ENGINE_BATCH_SHAPE_BUCKET_CAPACITY],
        }
    }
}

impl EngineBatchShapeMetricsSnapshot {
    /// Cold-prepared bucket entries in ascending dense-row order.
    #[must_use]
    pub fn prepared_buckets(&self) -> &[EngineBatchShapeBucketMetrics] {
        self.buckets
            .split_at(self.bucket_count.min(ENGINE_BATCH_SHAPE_BUCKET_CAPACITY))
            .0
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BatchShapeExecutionSample {
    active_rows: usize,
    selected_dense_rows: usize,
    padding_rows: usize,
    gpu_execution_ns: u64,
}

#[cfg(any(feature = "cuda", test))]
fn prepare_batch_shape_metrics(
    dense_rows: impl ExactSizeIterator<Item = usize>,
) -> Result<EngineBatchShapeMetricsSnapshot, &'static str> {
    let bucket_count = dense_rows.len();
    if bucket_count == 0 {
        return Err("batch-shape metrics require at least one prepared bucket");
    }
    if bucket_count > ENGINE_BATCH_SHAPE_BUCKET_CAPACITY {
        return Err("batch-shape metrics exceed the fixed bucket capacity");
    }
    let mut snapshot = EngineBatchShapeMetricsSnapshot {
        bucket_count,
        ..EngineBatchShapeMetricsSnapshot::default()
    };
    let mut previous = 0_usize;
    for (index, dense_rows) in dense_rows.enumerate() {
        if dense_rows == 0 || dense_rows <= previous {
            return Err("batch-shape metric buckets must be positive and strictly increasing");
        }
        snapshot.buckets[index].dense_rows = dense_rows;
        previous = dense_rows;
    }
    Ok(snapshot)
}

/// Records only a scheduler-committed iteration. Staged execution facts are
/// intentionally discarded when sampling, result construction, or scheduler
/// commit fails before an iteration metric is produced.
#[cfg(any(feature = "cuda", test))]
fn record_committed_batch_shape(
    snapshot: &mut EngineBatchShapeMetricsSnapshot,
    staged: Option<BatchShapeExecutionSample>,
    iteration_metric_recorded: bool,
) {
    if !iteration_metric_recorded {
        return;
    }
    let Some(staged) = staged else {
        snapshot.metrics_degraded = true;
        return;
    };
    if staged.active_rows == 0
        || staged.active_rows > staged.selected_dense_rows
        || staged.selected_dense_rows - staged.active_rows != staged.padding_rows
    {
        snapshot.metrics_degraded = true;
        return;
    }
    let Some(bucket_index) = snapshot
        .prepared_buckets()
        .iter()
        .position(|bucket| bucket.dense_rows == staged.selected_dense_rows)
    else {
        snapshot.metrics_degraded = true;
        return;
    };
    let bucket = snapshot.buckets[bucket_index];
    let Some(next_hit_count) = bucket.hit_count.checked_add(1) else {
        snapshot.metrics_degraded = true;
        return;
    };
    let Some(next_latency_sample_count) = bucket.latency_sample_count.checked_add(1) else {
        snapshot.metrics_degraded = true;
        return;
    };
    let Some(next_gpu_execution_ns_total) = bucket
        .gpu_execution_ns_total
        .checked_add(staged.gpu_execution_ns)
    else {
        snapshot.metrics_degraded = true;
        return;
    };

    snapshot.buckets[bucket_index] = EngineBatchShapeBucketMetrics {
        dense_rows: bucket.dense_rows,
        hit_count: next_hit_count,
        latency_sample_count: next_latency_sample_count,
        gpu_execution_ns_total: next_gpu_execution_ns_total,
        gpu_execution_ns_maximum: bucket.gpu_execution_ns_maximum.max(staged.gpu_execution_ns),
        gpu_execution_ns_last: staged.gpu_execution_ns,
    };
    snapshot.last = Some(EngineBatchShapeObservation {
        active_rows: staged.active_rows,
        selected_dense_rows: staged.selected_dense_rows,
        padding_rows: staged.padding_rows,
    });
}

/// Bounded operational snapshot produced on the exclusive backend worker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineMetricsSnapshot {
    /// Scheduler counters, gauges, and fixed-window p95 values when available.
    pub scheduler: Option<SchedulerMetricsSnapshot>,
    /// Committed CUDA batch-shape counters and latency when available.
    pub batch_shapes: Option<EngineBatchShapeMetricsSnapshot>,
    /// CUDA allocation gauges when a native context is active.
    pub allocation: Option<EngineAllocationMetrics>,
}

impl fmt::Debug for InferenceEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InferenceEngine")
            .field("model", &self.inner.model)
            .field("config", &self.inner.config)
            .field("ready", &self.is_ready())
            .finish_non_exhaustive()
    }
}

impl InferenceEngine {
    /// Starts one worker and transfers exclusive backend ownership to it.
    ///
    /// # Errors
    ///
    /// Returns for invalid engine/model metadata or host thread creation.
    pub fn start<B: EngineBackend>(
        model: ModelMetadata,
        config: EngineConfig,
        backend: B,
    ) -> Result<Self, EngineError> {
        model
            .validate()
            .map_err(|_| EngineError::InvalidConfiguration {
                field: "model",
                reason: "model metadata is invalid",
            })?;
        let config = config.validate()?;
        let (command_tx, command_rx) = mpsc::sync_channel(config.command_queue_capacity);
        let lifecycle = Arc::new(Lifecycle::new());
        let worker_lifecycle = Arc::clone(&lifecycle);
        let stats = Arc::new(EngineStats::default());
        let worker_stats = Arc::clone(&stats);
        let worker = thread::Builder::new()
            .name("riley-engine".to_owned())
            .spawn(move || {
                worker_main(
                    backend,
                    config,
                    &command_rx,
                    &worker_lifecycle,
                    &worker_stats,
                );
            })
            .map_err(|_| EngineError::Internal)?;
        let inner = Arc::new(EngineInner {
            config,
            model,
            commands: command_tx,
            lifecycle,
            next_request_id: AtomicU64::new(1),
            stats,
            worker: Mutex::new(Some(worker)),
        });
        let engine = Self { inner };
        if !engine
            .inner
            .lifecycle
            .wait_until(config.admission_timeout, |state| {
                state != LIFECYCLE_STARTING
            })
        {
            engine.begin_shutdown();
            return Err(EngineError::Timeout);
        }
        if !engine.is_ready() {
            return Err(EngineError::WorkerUnavailable);
        }
        Ok(engine)
    }

    /// Public metadata for the loaded model.
    #[must_use]
    pub fn model_metadata(&self) -> &ModelMetadata {
        &self.inner.model
    }

    /// Whether the worker currently accepts new work.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.lifecycle.load() == LIFECYCLE_READY
    }

    /// Current lifecycle and bounded admission counts.
    #[must_use]
    pub fn status(&self) -> EngineStatus {
        let lifecycle = self.inner.lifecycle.load();
        EngineStatus {
            ready: lifecycle == LIFECYCLE_READY,
            accepting: lifecycle == LIFECYCLE_READY,
            active_requests: self.inner.stats.active_requests.load(Ordering::Acquire),
            waiting_requests: self.inner.stats.waiting_requests.load(Ordering::Acquire),
        }
    }

    /// Requests one bounded operational metrics snapshot from the worker.
    ///
    /// Backends without a scheduler or CUDA context leave those fields empty.
    /// The request shares the bounded command queue and never reads backend
    /// state concurrently with an execution iteration.
    ///
    /// # Errors
    ///
    /// Returns for overload, shutdown, worker failure, or a snapshot timeout.
    pub fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, EngineError> {
        if !self.is_ready() {
            return Err(match self.inner.lifecycle.load() {
                LIFECYCLE_DRAINING | LIFECYCLE_STOPPED => EngineError::ShuttingDown,
                _ => EngineError::WorkerUnavailable,
            });
        }
        let (response, receiver) = mpsc::sync_channel(1);
        match self.inner.commands.try_send(Command::Metrics { response }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(EngineError::Overloaded),
            Err(TrySendError::Disconnected(_)) => return Err(EngineError::WorkerUnavailable),
        }
        receiver
            .recv_timeout(
                self.inner
                    .config
                    .admission_timeout
                    .min(Duration::from_millis(250)),
            )
            .map_err(|error| match error {
                RecvTimeoutError::Timeout => EngineError::Timeout,
                RecvTimeoutError::Disconnected => EngineError::WorkerUnavailable,
            })?
    }

    /// Returns the backend snapshot captured after successful explicit close.
    #[must_use]
    pub fn final_metrics_snapshot(&self) -> Option<EngineMetricsSnapshot> {
        self.inner.stats.final_metrics()
    }

    /// Submits one request without blocking on generation.
    ///
    /// Command admission is non-blocking; backend admission waits only for the
    /// configured finite timeout.  Dropping the returned handle atomically
    /// cancels the request.
    ///
    /// # Errors
    ///
    /// Returns for a model mismatch, bounded overload, admission timeout,
    /// invalid backend input, or shutdown.
    pub fn submit(&self, request: GenerationRequest) -> Result<GenerationHandle, EngineError> {
        if request.model_id != self.inner.model.model_id {
            return Err(EngineError::ModelNotFound);
        }
        if !self.is_ready() {
            return Err(match self.inner.lifecycle.load() {
                LIFECYCLE_DRAINING | LIFECYCLE_STOPPED => EngineError::ShuttingDown,
                _ => EngineError::WorkerUnavailable,
            });
        }
        let raw_id = self
            .inner
            .next_request_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| EngineError::WorkerUnavailable)?;
        let request_id = EngineRequestId(raw_id);
        let metadata = RequestMetadata::new(
            format!("cmpl-{raw_id:016x}"),
            request.model_id.clone(),
            unix_seconds(),
        )
        .map_err(|_| EngineError::Internal)?;
        self.submit_with_metadata(request_id, metadata, request)
    }

    fn submit_with_metadata(
        &self,
        request_id: EngineRequestId,
        metadata: RequestMetadata,
        request: GenerationRequest,
    ) -> Result<GenerationHandle, EngineError> {
        let cancellation = Arc::new(AtomicU8::new(REQUEST_LIVE));
        let (event_tx, event_rx) = mpsc::sync_channel(self.inner.config.event_channel_capacity);
        let (admitted_tx, admitted_rx) = mpsc::sync_channel(1);
        let command = Command::Submit {
            request_id,
            metadata: metadata.clone(),
            request,
            cancellation: Arc::clone(&cancellation),
            events: event_tx,
            admitted: admitted_tx,
        };
        self.inner
            .stats
            .waiting_requests
            .fetch_add(1, Ordering::AcqRel);
        match self.inner.commands.try_send(command) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.inner
                    .stats
                    .waiting_requests
                    .fetch_sub(1, Ordering::AcqRel);
                return Err(EngineError::Overloaded);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.inner
                    .stats
                    .waiting_requests
                    .fetch_sub(1, Ordering::AcqRel);
                return Err(EngineError::WorkerUnavailable);
            }
        }
        match admitted_rx.recv_timeout(self.inner.config.admission_timeout) {
            Ok(Ok(())) => Ok(GenerationHandle {
                request_id,
                metadata: Some(metadata),
                cancellation: Some(Arc::new(EngineCancellation {
                    state: cancellation,
                    wake: self.inner.commands.clone(),
                })),
                events: Some(event_rx),
            }),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => {
                cancellation.store(REQUEST_CANCELLED, Ordering::Release);
                let _ = self.inner.commands.try_send(Command::Wake);
                Err(EngineError::Timeout)
            }
            Err(RecvTimeoutError::Disconnected) => Err(EngineError::WorkerUnavailable),
        }
    }

    /// Stops accepting requests.  The worker cancels outstanding requests at
    /// safe backend iteration boundaries and explicitly closes backend state.
    pub fn begin_shutdown(&self) {
        self.inner.lifecycle.begin_shutdown();
        let _ = self.inner.commands.try_send(Command::Wake);
    }

    /// Waits for graceful cleanup and joins the worker.
    ///
    /// # Errors
    ///
    /// Returns on timeout or backend/worker failure.
    pub fn shutdown(&self, timeout: Duration) -> Result<(), EngineError> {
        self.begin_shutdown();
        if !self.inner.lifecycle.wait_until(timeout, |state| {
            matches!(state, LIFECYCLE_STOPPED | LIFECYCLE_FAILED)
        }) {
            return Err(EngineError::Timeout);
        }
        let failed = self.inner.lifecycle.load() == LIFECYCLE_FAILED;
        let handle = match self.inner.worker.lock() {
            Ok(mut worker) => worker.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(handle) = handle {
            handle.join().map_err(|_| EngineError::WorkerUnavailable)?;
        }
        if failed {
            Err(EngineError::Internal)
        } else {
            Ok(())
        }
    }
}

/// Receiver and cancellation guard for one generation request.
pub struct GenerationHandle {
    request_id: EngineRequestId,
    metadata: Option<RequestMetadata>,
    cancellation: Option<Arc<EngineCancellation>>,
    events: Option<Receiver<GenerationEvent>>,
}

/// Cloneable cancellation capability detached from the event receiver.
pub struct EngineCancellation {
    state: Arc<AtomicU8>,
    wake: SyncSender<Command>,
}

impl fmt::Debug for EngineCancellation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineCancellation")
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl EngineCancellation {
    /// Requests cancellation without depending on command-queue capacity.
    pub fn cancel(&self) {
        let _ = self.state.compare_exchange(
            REQUEST_LIVE,
            REQUEST_CANCELLED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.wake.try_send(Command::Wake);
    }

    /// Whether cancellation is requested or already being applied.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire),
            REQUEST_CANCELLED | REQUEST_CANCELLING
        )
    }
}

impl fmt::Debug for GenerationHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationHandle")
            .field("request_id", &self.request_id)
            .field("metadata", &self.metadata())
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl GenerationHandle {
    /// Worker-local numeric request identity.
    #[must_use]
    pub const fn request_id(&self) -> EngineRequestId {
        self.request_id
    }

    /// Stable API response metadata.
    ///
    /// # Panics
    ///
    /// Panics only after internal ownership has already been transferred with
    /// [`Self::into_parts`], which consumes the handle and cannot leave a
    /// callable value behind in safe Rust.
    #[must_use]
    pub fn metadata(&self) -> &RequestMetadata {
        self.metadata
            .as_ref()
            .expect("generation metadata is present until ownership transfer")
    }

    /// Atomically requests cancellation.  This succeeds even when the bounded
    /// command queue is full; the optional wake command only reduces latency.
    pub fn cancel(&self) {
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
        }
    }

    /// Whether cancellation was requested before a terminal event.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(|cancellation| cancellation.is_cancelled())
    }

    /// Blocks until the next ordered event or worker disconnect.
    ///
    /// # Errors
    ///
    /// Returns after the worker drops this request's event sender.
    ///
    /// # Panics
    ///
    /// Panics only after the receiver has been transferred by the consuming
    /// [`Self::into_parts`] operation.
    pub fn recv(&self) -> Result<GenerationEvent, RecvError> {
        self.events
            .as_ref()
            .expect("generation receiver is present until ownership transfer")
            .recv()
    }

    /// Blocks for at most `timeout` for the next event.
    ///
    /// # Errors
    ///
    /// Distinguishes timeout from worker disconnect.
    ///
    /// # Panics
    ///
    /// Panics only after the receiver has been transferred by the consuming
    /// [`Self::into_parts`] operation.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<GenerationEvent, RecvTimeoutError> {
        self.events
            .as_ref()
            .expect("generation receiver is present until ownership transfer")
            .recv_timeout(timeout)
    }

    /// Non-blocking event receive.
    ///
    /// # Errors
    ///
    /// Distinguishes an empty live channel from disconnect.
    ///
    /// # Panics
    ///
    /// Panics only after the receiver has been transferred by the consuming
    /// [`Self::into_parts`] operation.
    pub fn try_recv(&self) -> Result<GenerationEvent, TryRecvError> {
        self.events
            .as_ref()
            .expect("generation receiver is present until ownership transfer")
            .try_recv()
    }

    /// Transfers response metadata, the sole event receiver, and the detached
    /// cancellation capability to a service adapter.  The original drop guard
    /// is disarmed by the transfer.
    ///
    /// # Panics
    ///
    /// Panics only if an internal ownership field was previously taken, which
    /// cannot happen through the public safe API.
    #[must_use]
    pub fn into_parts(
        mut self,
    ) -> (
        RequestMetadata,
        Receiver<GenerationEvent>,
        Arc<EngineCancellation>,
    ) {
        let metadata = self
            .metadata
            .take()
            .expect("generation metadata can be transferred once");
        let events = self
            .events
            .take()
            .expect("generation receiver can be transferred once");
        let cancellation = self
            .cancellation
            .take()
            .expect("generation cancellation can be transferred once");
        (metadata, events, cancellation)
    }
}

impl Drop for GenerationHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn worker_main<B: EngineBackend>(
    mut backend: B,
    config: EngineConfig,
    commands: &Receiver<Command>,
    lifecycle: &Lifecycle,
    stats: &Arc<EngineStats>,
) {
    let mut requests = BTreeMap::<EngineRequestId, RequestControl>::new();
    lifecycle.store(LIFECYCLE_READY);
    let mut failed = false;

    loop {
        if lifecycle.load() == LIFECYCLE_DRAINING {
            reject_pending_commands(commands, stats);
            match backend.shutdown() {
                Ok(events) => {
                    publish_events(&mut backend, &mut requests, events);
                    stats.store_final_metrics(&backend.final_metrics_snapshot());
                }
                Err(error) => {
                    eprintln!("riley engine shutdown failure: {}", error.detail());
                    failed = true;
                }
            }
            fail_remaining(
                &mut requests,
                if failed {
                    ServiceErrorClass::Internal
                } else {
                    ServiceErrorClass::ShuttingDown
                },
            );
            break;
        }

        process_cancellations(&mut backend, &mut requests, lifecycle, &mut failed);
        if failed {
            lifecycle.store(LIFECYCLE_DRAINING);
            continue;
        }

        let mut received_command = false;
        loop {
            match commands.try_recv() {
                Ok(command) => {
                    received_command = true;
                    process_command(
                        command,
                        &mut backend,
                        &mut requests,
                        config.max_inflight_requests,
                        lifecycle,
                        stats,
                    );
                    process_cancellations(&mut backend, &mut requests, lifecycle, &mut failed);
                    if failed || lifecycle.load() == LIFECYCLE_DRAINING {
                        break;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    lifecycle.store(LIFECYCLE_DRAINING);
                    break;
                }
            }
        }
        if failed || lifecycle.load() == LIFECYCLE_DRAINING {
            continue;
        }

        if !backend.is_idle() {
            match backend.step() {
                Ok(events) => publish_events(&mut backend, &mut requests, events),
                Err(error) => {
                    eprintln!("riley engine backend failure: {}", error.detail());
                    fail_remaining(&mut requests, error.class());
                    failed = true;
                    lifecycle.store(LIFECYCLE_DRAINING);
                }
            }
            continue;
        }

        if received_command {
            continue;
        }
        match commands.recv_timeout(config.idle_poll_interval) {
            Ok(command) => process_command(
                command,
                &mut backend,
                &mut requests,
                config.max_inflight_requests,
                lifecycle,
                stats,
            ),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => lifecycle.store(LIFECYCLE_DRAINING),
        }
    }

    stats.waiting_requests.store(0, Ordering::Release);
    lifecycle.store(if failed {
        LIFECYCLE_FAILED
    } else {
        LIFECYCLE_STOPPED
    });
}

fn process_command<B: EngineBackend>(
    command: Command,
    backend: &mut B,
    requests: &mut BTreeMap<EngineRequestId, RequestControl>,
    maximum_requests: usize,
    lifecycle: &Lifecycle,
    stats: &Arc<EngineStats>,
) {
    let Command::Submit {
        request_id,
        metadata,
        request,
        cancellation,
        events,
        admitted,
    } = command
    else {
        if let Command::Metrics { response } = command {
            let result = backend.metrics_snapshot().map_err(|error| {
                eprintln!("riley metrics snapshot failure: {}", error.detail());
                EngineError::Internal
            });
            let _ = response.try_send(result);
        }
        return;
    };
    stats.waiting_requests.fetch_sub(1, Ordering::AcqRel);
    if lifecycle.load() != LIFECYCLE_READY {
        let _ = admitted.try_send(Err(EngineError::ShuttingDown));
        return;
    }
    if cancellation.load(Ordering::Acquire) != REQUEST_LIVE {
        let _ = admitted.try_send(Err(EngineError::Timeout));
        return;
    }
    if requests.len() >= maximum_requests {
        let _ = admitted.try_send(Err(EngineError::Overloaded));
        return;
    }
    match backend.admit(request_id, &metadata, &request) {
        Ok(()) => {
            let replaced = requests.insert(
                request_id,
                RequestControl {
                    cancellation,
                    events,
                    stats: Arc::clone(stats),
                },
            );
            debug_assert!(replaced.is_none());
            stats.active_requests.fetch_add(1, Ordering::AcqRel);
            let _ = admitted.try_send(Ok(()));
        }
        Err(error) => {
            eprintln!("riley request admission failure: {}", error.detail());
            let mapped = match error.class() {
                ServiceErrorClass::Overloaded => EngineError::Overloaded,
                ServiceErrorClass::Timeout | ServiceErrorClass::Cancelled => EngineError::Timeout,
                ServiceErrorClass::ShuttingDown => EngineError::ShuttingDown,
                ServiceErrorClass::InvalidRequest => EngineError::InvalidRequest,
                ServiceErrorClass::Internal => EngineError::Internal,
            };
            let _ = admitted.try_send(Err(mapped));
        }
    }
}

fn process_cancellations<B: EngineBackend>(
    backend: &mut B,
    requests: &mut BTreeMap<EngineRequestId, RequestControl>,
    lifecycle: &Lifecycle,
    failed: &mut bool,
) {
    let cancelled: Vec<_> = requests
        .iter()
        .filter_map(|(&request_id, control)| {
            control
                .cancellation
                .compare_exchange(
                    REQUEST_CANCELLED,
                    REQUEST_CANCELLING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
                .then_some(request_id)
        })
        .collect();
    for request_id in cancelled {
        match backend.cancel(request_id) {
            Ok(events) => publish_events(backend, requests, events),
            Err(error) => {
                eprintln!("riley cancellation failure: {}", error.detail());
                fail_remaining(requests, error.class());
                *failed = true;
                lifecycle.store(LIFECYCLE_DRAINING);
                return;
            }
        }
    }
}

fn publish_events<B: EngineBackend>(
    backend: &mut B,
    requests: &mut BTreeMap<EngineRequestId, RequestControl>,
    events: Vec<BackendEvent>,
) {
    let mut cancel_after_publish = Vec::new();
    for backend_event in events {
        let (request_id, event) = backend_event.into_parts();
        let terminal = matches!(
            event,
            GenerationEvent::Finished { .. } | GenerationEvent::Failed { .. }
        );
        let Some(control) = requests.get(&request_id) else {
            continue;
        };
        if matches!(
            control.cancellation.load(Ordering::Acquire),
            REQUEST_CANCELLED | REQUEST_CANCELLING
        ) && !terminal
        {
            continue;
        }
        let sent = if matches!(&event, GenerationEvent::TokenDelta { text } if text.is_empty()) {
            true
        } else {
            match control.events.try_send(event) {
                Ok(()) => true,
                Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
            }
        };
        if terminal {
            if let Some(control) = requests.remove(&request_id) {
                control
                    .cancellation
                    .store(REQUEST_TERMINAL, Ordering::Release);
            }
        } else if !sent
            && control
                .cancellation
                .compare_exchange(
                    REQUEST_LIVE,
                    REQUEST_CANCELLING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            cancel_after_publish.push(request_id);
        }
    }
    for request_id in cancel_after_publish {
        match backend.cancel(request_id) {
            Ok(events) => publish_events(backend, requests, events),
            Err(error) => {
                eprintln!(
                    "riley slow-consumer cancellation failure: {}",
                    error.detail()
                );
                if let Some(control) = requests.remove(&request_id) {
                    let _ = control.events.try_send(GenerationEvent::Failed {
                        class: error.class(),
                    });
                    control
                        .cancellation
                        .store(REQUEST_TERMINAL, Ordering::Release);
                }
            }
        }
    }
}

fn fail_remaining(
    requests: &mut BTreeMap<EngineRequestId, RequestControl>,
    class: ServiceErrorClass,
) {
    for (_, control) in std::mem::take(requests) {
        let _ = control.events.try_send(GenerationEvent::Failed { class });
        control
            .cancellation
            .store(REQUEST_TERMINAL, Ordering::Release);
    }
}

fn reject_pending_commands(commands: &Receiver<Command>, stats: &EngineStats) {
    loop {
        match commands.try_recv() {
            Ok(Command::Submit { admitted, .. }) => {
                stats.waiting_requests.fetch_sub(1, Ordering::AcqRel);
                let _ = admitted.try_send(Err(EngineError::ShuttingDown));
            }
            Ok(Command::Metrics { response }) => {
                let _ = response.try_send(Err(EngineError::ShuttingDown));
            }
            Ok(Command::Wake) => {}
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
        }
    }
}

impl crate::service::RequestCancellation for EngineCancellation {
    fn cancel(&self) {
        Self::cancel(self);
    }
}

impl crate::service::CompletionBackend for InferenceEngine {
    fn model_metadata(&self) -> ModelMetadata {
        InferenceEngine::model_metadata(self).clone()
    }

    fn status(&self) -> crate::service::BackendStatus {
        let status = InferenceEngine::status(self);
        crate::service::BackendStatus {
            ready: status.ready,
            accepting: status.accepting,
            active_requests: status.active_requests,
            waiting_requests: status.waiting_requests,
        }
    }

    fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, ServiceErrorClass> {
        InferenceEngine::metrics_snapshot(self).map_err(|error| error.service_class())
    }

    fn final_metrics_snapshot(&self) -> Option<EngineMetricsSnapshot> {
        InferenceEngine::final_metrics_snapshot(self)
    }

    fn submit(
        &self,
        request: GenerationRequest,
    ) -> Result<crate::service::SubmittedRequest, ServiceErrorClass> {
        let handle =
            InferenceEngine::submit(self, request).map_err(|error| error.service_class())?;
        let (metadata, events, cancellation) = handle.into_parts();
        let cancellation: Arc<dyn crate::service::RequestCancellation> = cancellation;
        Ok(crate::service::SubmittedRequest::new(
            metadata,
            events,
            cancellation,
        ))
    }

    fn begin_shutdown(&self) {
        InferenceEngine::begin_shutdown(self);
    }

    fn shutdown(&self, deadline: Instant) -> Result<(), ServiceErrorClass> {
        let timeout = deadline.saturating_duration_since(Instant::now());
        if timeout.is_zero() {
            return Err(ServiceErrorClass::Timeout);
        }
        InferenceEngine::shutdown(self, timeout).map_err(|error| error.service_class())
    }
}

#[cfg(feature = "cuda")]
mod cuda_backend {
    use std::collections::VecDeque;
    use std::time::Instant;

    use riley_model::{DecodeOptions, EncodeOptions, LoadedModel};
    use riley_runtime::generation::{
        FinishReason as RuntimeFinishReason, GenerationRequest as RuntimeGenerationRequest,
        GenerationState,
    };
    use riley_runtime::llama::{
        MAX_LLAMA_BATCH_SHAPE_BUCKETS, PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig,
    };
    use riley_runtime::sampling::{SamplingParams, SamplingWorkspace, TokenConstraints};
    use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
    use riley_scheduler::{
        ExecutionAbort, LlamaIterationCudaTimer, RequestCompletion, RequestDescriptor,
        RequestFinishReason, RequestId as SchedulerRequestId, SampledIterationToken, Scheduler,
        SchedulerConfig, SchedulerError, execute_llama_iteration_greedy_timed_with_workspace,
        execute_llama_iteration_timed,
    };

    use crate::domain::{
        FinishReason, GenerationEvent, GenerationRequest, ModelMetadata, RequestMetadata,
        ServiceErrorClass, TokenUsage,
    };

    use super::{
        BackendError, BackendEvent, BatchShapeExecutionSample, ENGINE_BATCH_SHAPE_BUCKET_CAPACITY,
        EngineAllocationMetrics, EngineBackend, EngineBatchShapeMetricsSnapshot, EngineConfig,
        EngineError, EngineMetricsSnapshot, EngineRequestId, InferenceEngine,
        prepare_batch_shape_metrics, private_request_error, record_committed_batch_shape,
        visible_utf8_prefix,
    };

    const _: () = assert!(ENGINE_BATCH_SHAPE_BUCKET_CAPACITY == MAX_LLAMA_BATCH_SHAPE_BUCKETS);

    /// Cold CUDA, scheduler, and fixed-batch preparation settings.
    #[derive(Clone, Debug)]
    pub struct CudaBackendConfig {
        /// Visible CUDA device ordinal.
        pub device_ordinal: u32,
        /// Bounded continuous scheduler policy.
        pub scheduler: SchedulerConfig,
        /// Fixed-M runtime metadata and forward policy.
        pub executor: PreparedLlamaBatchExecutorConfig,
        /// Enables exact device-side greedy selection for eligible requests.
        pub gpu_greedy: bool,
    }

    /// Fully prepared ownership bundle transferred into the engine worker.
    pub struct CudaEngineResources {
        metadata: ModelMetadata,
        model: LoadedModel,
        scheduler: Scheduler,
        context: CudaContext,
        stream: CudaStream,
        executor: PreparedLlamaBatchExecutor,
        timer: LlamaIterationCudaTimer,
        gpu_greedy: bool,
    }

    impl std::fmt::Debug for CudaEngineResources {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("CudaEngineResources")
                .field("metadata", &self.metadata)
                .field("scheduler", &self.scheduler)
                .field("executor", &self.executor)
                .finish_non_exhaustive()
        }
    }

    impl CudaEngineResources {
        /// Initializes CUDA, prepares the fixed batch executor, and constructs
        /// the scheduler from the executor's exact paged-KV layout.
        ///
        /// No model execution occurs during this cold preparation.
        ///
        /// # Errors
        ///
        /// Returns a private diagnostic for invalid metadata, device/runtime
        /// initialization, executor preparation, or scheduler construction.
        pub fn prepare(
            metadata: ModelMetadata,
            model: LoadedModel,
            config: CudaBackendConfig,
        ) -> Result<Self, BackendError> {
            metadata
                .validate()
                .map_err(|source| internal(format!("invalid model metadata: {source}")))?;
            let model_context = model.spec().max_sequence_length();
            if metadata.context_window_tokens > model_context {
                return Err(internal(format!(
                    "public context window {} exceeds model bound {model_context}",
                    metadata.context_window_tokens
                )));
            }
            let runtime = CudaRuntime::initialize()
                .map_err(|source| internal(format!("CUDA initialization failed: {source}")))?;
            let device = runtime
                .device(config.device_ordinal)
                .map_err(|source| internal(format!("CUDA device selection failed: {source}")))?;
            let context = device
                .create_context()
                .map_err(|source| internal(format!("CUDA context creation failed: {source}")))?;
            let mut stream = context
                .create_stream()
                .map_err(|source| internal(format!("CUDA stream creation failed: {source}")))?;
            let executor =
                PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config.executor)
                    .map_err(|source| {
                        internal(format!("batch executor preparation failed: {source}"))
                    })?;
            let scheduler = Scheduler::new(config.scheduler, executor.kv_layout())
                .map_err(|source| internal(format!("scheduler preparation failed: {source}")))?;
            let timer = LlamaIterationCudaTimer::prepare(&context)
                .map_err(|source| internal(format!("CUDA timing preparation failed: {source}")))?;
            Ok(Self {
                metadata,
                model,
                scheduler,
                context,
                stream,
                executor,
                timer,
                gpu_greedy: config.gpu_greedy,
            })
        }

        /// Loaded model metadata copied before resources move to the worker.
        #[must_use]
        pub const fn metadata(&self) -> &ModelMetadata {
            &self.metadata
        }
    }

    impl InferenceEngine {
        /// Starts the production CUDA backend and transfers every scheduler and
        /// CUDA owner to the worker thread.
        ///
        /// # Errors
        ///
        /// Returns for invalid engine configuration or backend scratch
        /// preparation.
        pub fn start_cuda(
            resources: CudaEngineResources,
            config: EngineConfig,
        ) -> Result<Self, EngineError> {
            let metadata = resources.metadata.clone();
            let backend = CudaBackend::new(resources).map_err(|error| {
                eprintln!("riley CUDA backend setup failure: {}", error.detail());
                EngineError::Internal
            })?;
            Self::start(metadata, config, backend)
        }
    }

    struct CudaRequest {
        engine_id: EngineRequestId,
        scheduler_id: SchedulerRequestId,
        state: GenerationState,
        decoded_token_bytes: Vec<u8>,
        prompt_tokens: u64,
        cancellation_deferred: bool,
        pre_iteration_cancel_delta: String,
    }

    struct PendingToken {
        scheduler_id: SchedulerRequestId,
        token_id: u32,
        text_delta: String,
    }

    struct CudaBackend {
        metadata: ModelMetadata,
        model: LoadedModel,
        scheduler: Option<Scheduler>,
        context: Option<CudaContext>,
        stream: Option<CudaStream>,
        executor: Option<PreparedLlamaBatchExecutor>,
        timer: Option<LlamaIterationCudaTimer>,
        sampling: SamplingWorkspace,
        allowed_tokens: Vec<bool>,
        addressable_tokens: usize,
        gpu_greedy: bool,
        greedy_token_workspace: Vec<u32>,
        requests: Vec<CudaRequest>,
        samples: Vec<SampledIterationToken>,
        pending_tokens: Vec<PendingToken>,
        pending_events: VecDeque<BackendEvent>,
        batch_shapes: EngineBatchShapeMetricsSnapshot,
        final_metrics: EngineMetricsSnapshot,
        clock: Instant,
    }

    impl CudaBackend {
        fn new(resources: CudaEngineResources) -> Result<Self, BackendError> {
            let vocabulary_size = resources.executor.vocabulary_size();
            let addressable_tokens = resources.model.tokenizer().addressable_token_count();
            if addressable_tokens == 0 || addressable_tokens > vocabulary_size {
                return Err(internal(format!(
                    "tokenizer addressable bound {addressable_tokens} is outside model vocabulary {vocabulary_size}"
                )));
            }
            let sampling = SamplingWorkspace::new(vocabulary_size)
                .map_err(|source| internal(format!("sampling preparation failed: {source}")))?;
            let mut allowed_tokens = Vec::new();
            allowed_tokens
                .try_reserve_exact(vocabulary_size)
                .map_err(|_| internal("allowed-token mask allocation failed"))?;
            allowed_tokens.resize(vocabulary_size, false);
            let output_capacity = resources.executor.config().metadata().max_output_slots();
            let mut greedy_token_workspace = Vec::new();
            greedy_token_workspace
                .try_reserve_exact(output_capacity)
                .map_err(|_| internal("greedy-token workspace allocation failed"))?;
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(output_capacity)
                .map_err(|_| internal("sample staging allocation failed"))?;
            let mut pending_tokens = Vec::new();
            pending_tokens
                .try_reserve_exact(output_capacity)
                .map_err(|_| internal("token publication staging allocation failed"))?;
            let request_capacity = resources
                .scheduler
                .config()
                .max_waiting_requests
                .checked_add(resources.scheduler.config().max_active_sequences)
                .ok_or_else(|| internal("request capacity overflowed"))?;
            let mut requests = Vec::new();
            requests
                .try_reserve_exact(request_capacity)
                .map_err(|_| internal("request state allocation failed"))?;
            let mut pending_events = VecDeque::new();
            pending_events
                .try_reserve_exact(request_capacity)
                .map_err(|_| internal("pending event allocation failed"))?;
            let batch_shapes = prepare_batch_shape_metrics(
                resources
                    .executor
                    .shape_bucket_hits()
                    .iter()
                    .map(|bucket| bucket.dense_rows()),
            )
            .map_err(internal)?;
            Ok(Self {
                metadata: resources.metadata,
                model: resources.model,
                scheduler: Some(resources.scheduler),
                context: Some(resources.context),
                stream: Some(resources.stream),
                executor: Some(resources.executor),
                timer: Some(resources.timer),
                sampling,
                allowed_tokens,
                addressable_tokens,
                gpu_greedy: resources.gpu_greedy,
                greedy_token_workspace,
                requests,
                samples,
                pending_tokens,
                pending_events,
                batch_shapes,
                final_metrics: EngineMetricsSnapshot::default(),
                clock: Instant::now(),
            })
        }

        fn now_ns(&self) -> u64 {
            u64::try_from(self.clock.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }

        fn scheduler_mut(&mut self) -> Result<&mut Scheduler, BackendError> {
            self.scheduler
                .as_mut()
                .ok_or_else(|| internal("scheduler is already closed"))
        }

        fn request_index_by_engine(&self, request_id: EngineRequestId) -> Option<usize> {
            self.requests
                .iter()
                .position(|request| request.engine_id == request_id)
        }

        fn request_index_by_scheduler(&self, request_id: SchedulerRequestId) -> Option<usize> {
            self.requests
                .iter()
                .position(|request| request.scheduler_id == request_id)
        }

        fn process_completion(
            &mut self,
            completion: &RequestCompletion,
        ) -> Result<Vec<BackendEvent>, BackendError> {
            let Some(index) = self.request_index_by_scheduler(completion.request_id()) else {
                return Ok(Vec::new());
            };
            let mut request = self.requests.swap_remove(index);
            let mut events = Vec::new();
            events
                .try_reserve_exact(2)
                .map_err(|_| internal("completion event allocation failed"))?;
            if completion.reason() == RequestFinishReason::Cancelled {
                let delta = if request.cancellation_deferred {
                    request.pre_iteration_cancel_delta.as_str()
                } else if request.state.finish_reason().is_none() {
                    request.state.cancel().map_err(|source| {
                        internal(format!("cancellation flush failed: {source}"))
                    })?
                } else {
                    ""
                };
                if !delta.is_empty() {
                    events.push(BackendEvent::new(
                        request.engine_id,
                        GenerationEvent::TokenDelta {
                            text: delta.to_owned(),
                        },
                    ));
                }
            }
            match completion.reason() {
                RequestFinishReason::Length
                | RequestFinishReason::Stop
                | RequestFinishReason::Cancelled => {
                    let usage = TokenUsage::new(
                        request.prompt_tokens,
                        u64::try_from(completion.generated_token_ids().len())
                            .map_err(|_| internal("completion token count overflowed"))?,
                    )
                    .map_err(|_| internal("usage arithmetic overflowed"))?;
                    let reason = match completion.reason() {
                        RequestFinishReason::Length => FinishReason::Length,
                        RequestFinishReason::Stop => FinishReason::Stop,
                        RequestFinishReason::Cancelled => FinishReason::Cancelled,
                        RequestFinishReason::AdmissionTimeout
                        | RequestFinishReason::ExecutorFailure => unreachable!(),
                    };
                    events.push(BackendEvent::new(
                        request.engine_id,
                        GenerationEvent::Finished { reason, usage },
                    ));
                }
                RequestFinishReason::AdmissionTimeout => events.push(BackendEvent::new(
                    request.engine_id,
                    GenerationEvent::Failed {
                        class: ServiceErrorClass::Timeout,
                    },
                )),
                RequestFinishReason::ExecutorFailure => events.push(BackendEvent::new(
                    request.engine_id,
                    GenerationEvent::Failed {
                        class: ServiceErrorClass::Internal,
                    },
                )),
            }
            Ok(events)
        }

        fn process_completions(
            &mut self,
            completions: &[RequestCompletion],
        ) -> Result<Vec<BackendEvent>, BackendError> {
            let mut events = Vec::new();
            events
                .try_reserve(completions.len().saturating_mul(2))
                .map_err(|_| internal("completion batch allocation failed"))?;
            for completion in completions {
                events.extend(self.process_completion(completion)?);
            }
            Ok(events)
        }

        fn sample_iteration(
            &mut self,
            plan: &riley_scheduler::IterationPlan,
            downloaded: &riley_scheduler::DownloadedLlamaIteration,
            gpu_greedy: bool,
        ) -> Result<(), BackendError> {
            self.samples.clear();
            self.pending_tokens.clear();
            for &slot in plan.output_slots() {
                let item = plan
                    .prefill_items()
                    .iter()
                    .chain(plan.decode_items())
                    .find(|item| item.output_slot() == Some(slot))
                    .ok_or_else(|| internal("output slot has no scheduler work item"))?;
                let request_index = self
                    .request_index_by_scheduler(item.request_id())
                    .ok_or_else(|| internal("iteration references unknown request"))?;
                let request = &mut self.requests[request_index];
                let (token_id, token_logprob) = if gpu_greedy {
                    (
                        downloaded.greedy_token_for_slot(slot).map_err(|source| {
                            internal(format!("greedy token routing failed: {source}"))
                        })?,
                        Some(0.0),
                    )
                } else {
                    let logits = downloaded
                        .logits_for_slot(slot)
                        .map_err(|source| internal(format!("logit routing failed: {source}")))?;
                    self.allowed_tokens.fill(false);
                    self.allowed_tokens[..self.addressable_tokens].fill(true);
                    for &masked_id in request.state.masked_finish_token_ids() {
                        let index = usize::try_from(masked_id)
                            .map_err(|_| internal("masked token ID cannot index vocabulary"))?;
                        let allowed = self
                            .allowed_tokens
                            .get_mut(index)
                            .ok_or_else(|| internal("masked token ID is outside vocabulary"))?;
                        *allowed = false;
                    }
                    let distribution = self
                        .sampling
                        .process_bf16_native(
                            logits,
                            TokenConstraints::AllowedMask(&self.allowed_tokens),
                            request.state.history_token_ids(),
                            request.state.request().sampling_params,
                        )
                        .map_err(|source| internal(format!("sampling failed: {source}")))?;
                    let sample =
                        distribution
                            .sample(request.state.sampling_rng().map_err(|source| {
                                internal(format!("sampling RNG failed: {source}"))
                            })?)
                            .map_err(|source| internal(format!("sampling RNG failed: {source}")))?;
                    (sample.token_id(), sample.token_logprob())
                };
                let needs_decoding = request
                    .state
                    .token_needs_decoding(token_id)
                    .map_err(|source| internal(format!("token acceptance failed: {source}")))?;
                let decoded_count = if needs_decoding {
                    self.model
                        .tokenizer()
                        .decode_token_bytes_into(
                            token_id,
                            DecodeOptions {
                                skip_special_tokens: true,
                            },
                            &mut request.decoded_token_bytes,
                        )
                        .map_err(|source| internal(format!("token decoding failed: {source}")))?
                } else {
                    0
                };
                let decoded =
                    needs_decoding.then_some(&request.decoded_token_bytes[..decoded_count]);
                let generated = request
                    .state
                    .accept_token(token_id, token_logprob, decoded)
                    .map_err(|source| internal(format!("token acceptance failed: {source}")))?;
                let stop = matches!(
                    generated.finish_reason(),
                    Some(
                        RuntimeFinishReason::Eos
                            | RuntimeFinishReason::StopToken
                            | RuntimeFinishReason::StopString
                    )
                );
                self.samples
                    .push(SampledIterationToken::new(token_id, stop));
                self.pending_tokens.push(PendingToken {
                    scheduler_id: item.request_id(),
                    token_id,
                    text_delta: generated.text_delta().to_owned(),
                });
            }
            Ok(())
        }

        #[allow(clippy::float_cmp)]
        fn iteration_gpu_greedy_eligible(
            &self,
            plan: &riley_scheduler::IterationPlan,
        ) -> Result<bool, BackendError> {
            if !self.gpu_greedy || self.addressable_tokens != self.sampling.vocabulary_size() {
                return Ok(false);
            }
            for &slot in plan.output_slots() {
                let item = plan
                    .prefill_items()
                    .iter()
                    .chain(plan.decode_items())
                    .find(|item| item.output_slot() == Some(slot))
                    .ok_or_else(|| internal("output slot has no scheduler work item"))?;
                let request_index = self
                    .request_index_by_scheduler(item.request_id())
                    .ok_or_else(|| internal("iteration references unknown request"))?;
                let state = &self.requests[request_index].state;
                let params = state.request().sampling_params;
                if params.temperature != 0.0
                    || params.repetition_penalty != 1.0
                    || !state.masked_finish_token_ids().is_empty()
                {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        fn snapshot_cancel_deltas(
            &mut self,
            plan: &riley_scheduler::IterationPlan,
        ) -> Result<(), BackendError> {
            for item in plan.prefill_items().iter().chain(plan.decode_items()) {
                let request_index = self
                    .request_index_by_scheduler(item.request_id())
                    .ok_or_else(|| internal("iteration references unknown request"))?;
                let request = &mut self.requests[request_index];
                request.pre_iteration_cancel_delta.clear();
                request
                    .pre_iteration_cancel_delta
                    .push_str(visible_utf8_prefix(
                        request.state.stop_state().pending_bytes(),
                    ));
            }
            Ok(())
        }

        fn settle_execution_failure(
            &mut self,
            failure: &riley_scheduler::IterationExecutionFailure,
        ) -> Result<Vec<BackendEvent>, BackendError> {
            let Some((iteration_id, abort)) = failure.abort_data() else {
                return Err(internal(format!(
                    "executor failed without proven stream quiescence: {}",
                    failure.error()
                )));
            };
            let now_ns = self.now_ns();
            let updates = self
                .scheduler_mut()?
                .abort_iteration(iteration_id, abort, now_ns)
                .map_err(|source| internal(format!("iteration abort failed: {source}")))?;
            if abort == ExecutionAbort::NotDispatched {
                return Err(internal(format!(
                    "iteration failed before dispatch: {}",
                    failure.error()
                )));
            }
            self.process_completions(updates.completions())
        }

        fn close_resources(&mut self) -> Result<(), BackendError> {
            let mut first_error = None;
            if self
                .scheduler
                .as_ref()
                .is_some_and(|scheduler| scheduler.inflight_iteration_id().is_some())
            {
                let synchronized = self
                    .stream
                    .as_mut()
                    .ok_or_else(|| internal("in-flight scheduler has no CUDA stream"))?
                    .synchronize();
                match synchronized {
                    Ok(()) => {
                        let iteration = self
                            .scheduler
                            .as_ref()
                            .and_then(Scheduler::inflight_iteration_id)
                            .ok_or_else(|| internal("in-flight iteration disappeared"))?;
                        let now_ns = self.now_ns();
                        if let Err(source) = self.scheduler_mut()?.abort_iteration(
                            iteration,
                            ExecutionAbort::DeviceQuiescedMutationUnknown,
                            now_ns,
                        ) {
                            first_error = Some(format!("final iteration abort failed: {source}"));
                        }
                    }
                    Err(source) => {
                        first_error = Some(format!(
                            "CUDA stream quiescence could not be proven during shutdown: {source}"
                        ));
                    }
                }
            }
            if first_error.is_none() {
                if let Some(scheduler) = self.scheduler.take() {
                    if let Err(source) = scheduler.close(self.now_ns(), None) {
                        first_error = Some(format!("scheduler close failed: {source}"));
                    }
                }
            }
            if let Some(executor) = self.executor.take() {
                if let Err(source) = executor.close() {
                    first_error.get_or_insert_with(|| format!("executor close failed: {source}"));
                }
            }
            if let Some(timer) = self.timer.take() {
                if let Err(source) = timer.close() {
                    first_error.get_or_insert_with(|| format!("CUDA timer close failed: {source}"));
                }
            }
            if let Some(mut stream) = self.stream.take() {
                if let Err(source) = stream.synchronize() {
                    first_error
                        .get_or_insert_with(|| format!("stream synchronize failed: {source}"));
                }
                if let Err(source) = stream.close() {
                    first_error.get_or_insert_with(|| format!("stream close failed: {source}"));
                }
            }
            if first_error.is_none() {
                match self.context.as_ref() {
                    Some(context) => match context.allocation_stats() {
                        Ok(allocation) => {
                            self.final_metrics = EngineMetricsSnapshot {
                                scheduler: None,
                                batch_shapes: Some(self.batch_shapes),
                                allocation: Some(EngineAllocationMetrics {
                                    device_live_bytes: allocation.device_live_bytes(),
                                    device_live_allocations: allocation.device_live_allocations(),
                                    pinned_host_live_bytes: allocation.pinned_host_live_bytes(),
                                    pinned_host_live_allocations: allocation
                                        .pinned_host_live_allocations(),
                                }),
                            };
                        }
                        Err(source) => {
                            first_error = Some(format!(
                                "final allocation metrics failed before context close: {source}"
                            ));
                        }
                    },
                    None => {
                        first_error = Some("CUDA context is already closed".to_owned());
                    }
                }
            }
            if let Some(context) = self.context.take() {
                if let Err(source) = context.close() {
                    first_error.get_or_insert_with(|| format!("context close failed: {source}"));
                }
            }
            match first_error {
                Some(detail) => Err(internal(detail)),
                None => Ok(()),
            }
        }
    }

    impl EngineBackend for CudaBackend {
        fn admit(
            &mut self,
            request_id: EngineRequestId,
            metadata: &RequestMetadata,
            request: &GenerationRequest,
        ) -> Result<(), BackendError> {
            if request.model_id != self.metadata.model_id {
                return Err(private_request_error("request selected a different model"));
            }
            if request.max_new_tokens == 0
                || request.max_new_tokens > self.metadata.max_output_tokens
            {
                return Err(private_request_error(
                    "requested output token count is outside server bounds",
                ));
            }
            let prompt_token_ids = self
                .model
                .tokenizer()
                .encode(
                    &request.prompt,
                    EncodeOptions {
                        add_special_tokens: true,
                    },
                )
                .map_err(|_| private_request_error("prompt tokenization failed"))?;
            let total_tokens = prompt_token_ids
                .len()
                .checked_add(request.max_new_tokens)
                .ok_or_else(|| private_request_error("request token count overflowed"))?;
            if total_tokens > self.model.spec().max_sequence_length() {
                return Err(private_request_error(
                    "request exceeds model context window",
                ));
            }
            let prompt_tokens = u64::try_from(prompt_token_ids.len())
                .map_err(|_| internal("prompt token count overflowed"))?;
            let runtime_request = RuntimeGenerationRequest {
                request_id: metadata.request_id.as_bytes().to_vec(),
                seed: request.sampling.seed.unwrap_or(0),
                prompt_token_ids: prompt_token_ids.clone(),
                sampling_params: SamplingParams {
                    temperature: request.sampling.temperature,
                    top_k: None,
                    top_p: Some(f64::from(request.sampling.top_p)),
                    repetition_penalty: 1.0,
                },
                min_new_tokens: 0,
                max_new_tokens: request.max_new_tokens,
                eos_token_ids: self.model.spec().special_tokens().eos().to_vec(),
                stop_token_ids: Vec::new(),
                stop_strings: request.stop_sequences.clone(),
            };
            let state = GenerationState::new(
                runtime_request,
                self.sampling.vocabulary_size(),
                self.model.tokenizer().maximum_decoded_token_bytes(),
            )
            .map_err(|_| private_request_error("generation request validation failed"))?;
            let mut decoded_token_bytes = Vec::new();
            let maximum_bytes = self.model.tokenizer().maximum_decoded_token_bytes();
            decoded_token_bytes
                .try_reserve_exact(maximum_bytes)
                .map_err(|_| internal("decoded-token buffer allocation failed"))?;
            decoded_token_bytes.resize(maximum_bytes, 0);
            let mut pre_iteration_cancel_delta = String::new();
            pre_iteration_cancel_delta
                .try_reserve_exact(maximum_bytes)
                .map_err(|_| internal("cancellation delta allocation failed"))?;
            let now_ns = self.now_ns();
            let submission = self
                .scheduler_mut()?
                .submit(
                    RequestDescriptor::new(prompt_token_ids, request.max_new_tokens),
                    now_ns,
                )
                .map_err(|source| scheduler_admission_error(&source))?;
            self.requests.push(CudaRequest {
                engine_id: request_id,
                scheduler_id: submission.request_id(),
                state,
                decoded_token_bytes,
                prompt_tokens,
                cancellation_deferred: false,
                pre_iteration_cancel_delta,
            });
            Ok(())
        }

        fn cancel(
            &mut self,
            request_id: EngineRequestId,
        ) -> Result<Vec<BackendEvent>, BackendError> {
            let Some(index) = self.request_index_by_engine(request_id) else {
                return Ok(Vec::new());
            };
            let scheduler_id = self.requests[index].scheduler_id;
            let now_ns = self.now_ns();
            let outcome = self
                .scheduler_mut()?
                .cancel(scheduler_id, now_ns)
                .map_err(|source| internal(format!("scheduler cancellation failed: {source}")))?;
            if outcome.deferred_until_iteration_settles() {
                self.requests[index].cancellation_deferred = true;
                return Ok(Vec::new());
            }
            match outcome.completion().cloned() {
                Some(completion) => self.process_completion(&completion),
                None => Ok(Vec::new()),
            }
        }

        #[allow(clippy::too_many_lines)]
        fn step(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
            if let Some(event) = self.pending_events.pop_front() {
                return Ok(vec![event]);
            }
            let now_ns = self.now_ns();
            let planning = self
                .scheduler_mut()?
                .plan_iteration(now_ns)
                .map_err(|source| internal(format!("iteration planning failed: {source}")))?;
            let (plan, completions) = planning.into_parts();
            let mut events = self.process_completions(&completions)?;
            let Some(plan) = plan else {
                return Ok(events);
            };
            self.snapshot_cancel_deltas(&plan)?;
            let gpu_greedy = self.iteration_gpu_greedy_eligible(&plan)?;
            let expected_active_rows = plan.total_tokens();
            let (mut downloaded, timing, staged_shape) = {
                let executor = self
                    .executor
                    .as_mut()
                    .ok_or_else(|| internal("batch executor is already closed"))?;
                let stream = self
                    .stream
                    .as_mut()
                    .ok_or_else(|| internal("CUDA stream is already closed"))?;
                let timer = self
                    .timer
                    .as_mut()
                    .ok_or_else(|| internal("CUDA timer is already closed"))?;
                let execution = if gpu_greedy {
                    execute_llama_iteration_greedy_timed_with_workspace(
                        &plan,
                        executor,
                        stream,
                        timer,
                        &mut self.greedy_token_workspace,
                    )
                } else {
                    execute_llama_iteration_timed(&plan, executor, stream, timer)
                };
                match execution {
                    Ok((downloaded, timing)) => {
                        let staged_shape = executor
                            .last_shape_observation()
                            .filter(|shape| shape.active_rows() == expected_active_rows)
                            .map(|shape| BatchShapeExecutionSample {
                                active_rows: shape.active_rows(),
                                selected_dense_rows: shape.selected_dense_rows(),
                                padding_rows: shape.padding_rows(),
                                gpu_execution_ns: timing.gpu_execution_ns(),
                            });
                        (downloaded, timing, staged_shape)
                    }
                    Err(failure) => {
                        events.extend(self.settle_execution_failure(&failure)?);
                        return Ok(events);
                    }
                }
            };
            let sampling = self.sample_iteration(&plan, &downloaded, gpu_greedy);
            if gpu_greedy {
                if let Err(source) =
                    downloaded.restore_greedy_token_workspace(&mut self.greedy_token_workspace)
                {
                    let (iteration, abort) = downloaded.abort_data();
                    let detail = format!("greedy-token workspace recovery failed: {source}");
                    let now_ns = self.now_ns();
                    self.scheduler_mut()?
                        .abort_iteration(iteration, abort, now_ns)
                        .map_err(|abort_source| {
                            internal(format!(
                                "workspace-recovery abort failed after {detail}: {abort_source}"
                            ))
                        })?;
                    return Err(internal(detail));
                }
            }
            if let Err(error) = sampling {
                let (iteration, abort) = downloaded.abort_data();
                let now_ns = self.now_ns();
                let _ = self
                    .scheduler_mut()?
                    .abort_iteration(iteration, abort, now_ns)
                    .map_err(|source| internal(format!("sampling abort failed: {source}")))?;
                return Err(error);
            }
            let result = match downloaded.into_result(&self.samples, timing) {
                Ok(result) => result,
                Err(failure) => {
                    let (iteration, abort) = failure.abort_data();
                    let detail = format!("iteration result construction failed: {failure}");
                    let now_ns = self.now_ns();
                    self.scheduler_mut()?
                        .abort_iteration(iteration, abort, now_ns)
                        .map_err(|source| {
                            internal(format!(
                                "result-construction abort failed after {detail}: {source}"
                            ))
                        })?;
                    return Err(internal(detail));
                }
            };
            let now_ns = self.now_ns();
            let updates = match self.scheduler_mut()?.complete_iteration(&result, now_ns) {
                Ok(updates) => updates,
                Err(source) => {
                    let detail = format!("scheduler commit failed: {source}");
                    if self
                        .scheduler
                        .as_ref()
                        .and_then(Scheduler::inflight_iteration_id)
                        == Some(result.iteration_id())
                    {
                        let now_ns = self.now_ns();
                        self.scheduler_mut()?
                            .abort_iteration(
                                result.iteration_id(),
                                ExecutionAbort::DeviceQuiescedMutationUnknown,
                                now_ns,
                            )
                            .map_err(|abort_source| {
                                internal(format!(
                                    "commit abort failed after {detail}: {abort_source}"
                                ))
                            })?;
                    }
                    return Err(internal(detail));
                }
            };
            record_committed_batch_shape(
                &mut self.batch_shapes,
                staged_shape,
                updates.iteration_metric().is_some(),
            );

            // External publication starts only after the authoritative commit
            // above returns successfully.
            for token in updates.token_events() {
                let pending = self
                    .pending_tokens
                    .iter()
                    .find(|pending| pending.scheduler_id == token.request_id())
                    .ok_or_else(|| internal("committed token has no staged publication"))?;
                if pending.token_id != token.token_id() {
                    return Err(internal("committed token differs from staged sample"));
                }
                let request_index = self
                    .request_index_by_scheduler(token.request_id())
                    .ok_or_else(|| internal("committed token targets unknown request"))?;
                events.push(BackendEvent::new(
                    self.requests[request_index].engine_id,
                    GenerationEvent::TokenDelta {
                        text: pending.text_delta.clone(),
                    },
                ));
            }
            events.extend(self.process_completions(updates.completions())?);
            if !updates.settlement_failures().is_empty() {
                eprintln!(
                    "riley scheduler contained {} settlement failure(s)",
                    updates.settlement_failures().len()
                );
            }
            Ok(events)
        }

        fn is_idle(&self) -> bool {
            self.requests.is_empty() && self.pending_events.is_empty()
        }

        fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, BackendError> {
            let scheduler = self
                .scheduler
                .as_ref()
                .ok_or_else(|| internal("scheduler is already closed"))?
                .metrics_snapshot()
                .map_err(|source| internal(format!("scheduler metrics failed: {source}")))?;
            let allocation = self
                .context
                .as_ref()
                .ok_or_else(|| internal("CUDA context is already closed"))?
                .allocation_stats()
                .map_err(|source| internal(format!("allocation metrics failed: {source}")))?;
            Ok(EngineMetricsSnapshot {
                scheduler: Some(scheduler),
                batch_shapes: Some(self.batch_shapes),
                allocation: Some(EngineAllocationMetrics {
                    device_live_bytes: allocation.device_live_bytes(),
                    device_live_allocations: allocation.device_live_allocations(),
                    pinned_host_live_bytes: allocation.pinned_host_live_bytes(),
                    pinned_host_live_allocations: allocation.pinned_host_live_allocations(),
                }),
            })
        }

        fn final_metrics_snapshot(&self) -> EngineMetricsSnapshot {
            self.final_metrics
        }

        fn shutdown(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
            if self.scheduler.is_none()
                && self.executor.is_none()
                && self.timer.is_none()
                && self.stream.is_none()
                && self.context.is_none()
            {
                return Ok(Vec::new());
            }
            if let Some(scheduler) = self.scheduler.as_mut() {
                scheduler.begin_shutdown();
            }
            let request_ids: Vec<_> = self
                .requests
                .iter()
                .map(|request| request.engine_id)
                .collect();
            let mut events = Vec::new();
            for request_id in request_ids {
                events.extend(self.cancel(request_id)?);
            }
            self.close_resources()?;
            Ok(events)
        }
    }

    fn scheduler_admission_error(source: &SchedulerError) -> BackendError {
        let class = match source {
            SchedulerError::WaitingQueueFull { .. }
            | SchedulerError::WaitingTokenLimit { .. }
            | SchedulerError::ActiveSequenceLimit { .. }
            | SchedulerError::KvCapacityExceeded { .. } => ServiceErrorClass::Overloaded,
            SchedulerError::SchedulerClosed => ServiceErrorClass::ShuttingDown,
            SchedulerError::SequenceTokenLimit { .. } => ServiceErrorClass::InvalidRequest,
            _ => ServiceErrorClass::Internal,
        };
        BackendError::new(class, format!("scheduler admission failed: {source}"))
    }

    fn internal(detail: impl Into<String>) -> BackendError {
        BackendError::new(ServiceErrorClass::Internal, detail)
    }
}

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaBackendConfig, CudaEngineResources};

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::domain::{
        FinishReason, GenerationEvent, GenerationRequest, ModelMetadata, RequestMetadata,
        SamplingParameters, ServiceErrorClass, TokenUsage,
    };

    use super::{
        BackendError, BackendEvent, BatchShapeExecutionSample, EngineBackend,
        EngineBatchShapeMetricsSnapshot, EngineConfig, EngineError, EngineRequestId,
        InferenceEngine, prepare_batch_shape_metrics, private_request_error,
        record_committed_batch_shape, visible_utf8_prefix,
    };

    #[derive(Debug)]
    struct MockRequest {
        prompt_tokens: u64,
        generated: u64,
        remaining: usize,
    }

    #[derive(Default)]
    struct MockCounters {
        admitted: AtomicUsize,
        cancelled: AtomicUsize,
        shutdowns: AtomicUsize,
    }

    struct MockBackend {
        requests: BTreeMap<EngineRequestId, MockRequest>,
        pending: VecDeque<BackendEvent>,
        counters: Arc<MockCounters>,
        step_delay: Duration,
    }

    impl MockBackend {
        fn new(counters: Arc<MockCounters>, step_delay: Duration) -> Self {
            Self {
                requests: BTreeMap::new(),
                pending: VecDeque::new(),
                counters,
                step_delay,
            }
        }

        fn cancellation_events(&mut self, request_id: EngineRequestId) -> Vec<BackendEvent> {
            let Some(request) = self.requests.remove(&request_id) else {
                return Vec::new();
            };
            self.counters.cancelled.fetch_add(1, Ordering::AcqRel);
            vec![BackendEvent::new(
                request_id,
                GenerationEvent::Finished {
                    reason: FinishReason::Cancelled,
                    usage: TokenUsage::new(request.prompt_tokens, request.generated)
                        .expect("small mock usage"),
                },
            )]
        }
    }

    impl EngineBackend for MockBackend {
        fn admit(
            &mut self,
            request_id: EngineRequestId,
            _metadata: &RequestMetadata,
            request: &GenerationRequest,
        ) -> Result<(), BackendError> {
            self.counters.admitted.fetch_add(1, Ordering::AcqRel);
            self.requests.insert(
                request_id,
                MockRequest {
                    prompt_tokens: u64::try_from(request.prompt.len()).unwrap_or(u64::MAX),
                    generated: 0,
                    remaining: request.max_new_tokens,
                },
            );
            Ok(())
        }

        fn cancel(
            &mut self,
            request_id: EngineRequestId,
        ) -> Result<Vec<BackendEvent>, BackendError> {
            Ok(self.cancellation_events(request_id))
        }

        fn step(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
            if !self.step_delay.is_zero() {
                thread::sleep(self.step_delay);
            }
            if let Some(event) = self.pending.pop_front() {
                return Ok(vec![event]);
            }
            let Some((&request_id, request)) = self.requests.iter_mut().next() else {
                return Ok(Vec::new());
            };
            if request.remaining == 0 {
                let request = self.requests.remove(&request_id).expect("request exists");
                return Ok(vec![BackendEvent::new(
                    request_id,
                    GenerationEvent::Finished {
                        reason: FinishReason::Length,
                        usage: TokenUsage::new(request.prompt_tokens, request.generated)
                            .expect("small mock usage"),
                    },
                )]);
            }
            request.remaining -= 1;
            request.generated += 1;
            let finished = request.remaining == 0;
            let usage = TokenUsage::new(request.prompt_tokens, request.generated)
                .expect("small mock usage");
            let mut events = vec![BackendEvent::new(
                request_id,
                GenerationEvent::TokenDelta {
                    text: "x".to_owned(),
                },
            )];
            if finished {
                self.requests.remove(&request_id);
                events.push(BackendEvent::new(
                    request_id,
                    GenerationEvent::Finished {
                        reason: FinishReason::Length,
                        usage,
                    },
                ));
            }
            Ok(events)
        }

        fn is_idle(&self) -> bool {
            self.requests.is_empty() && self.pending.is_empty()
        }

        fn shutdown(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
            self.counters.shutdowns.fetch_add(1, Ordering::AcqRel);
            let ids: Vec<_> = self.requests.keys().copied().collect();
            let mut events = Vec::new();
            for request_id in ids {
                events.extend(self.cancellation_events(request_id));
            }
            Ok(events)
        }
    }

    fn model() -> ModelMetadata {
        ModelMetadata {
            model_id: "mock-model".to_owned(),
            created_unix_seconds: 1,
            owned_by: "riley".to_owned(),
            context_window_tokens: 1024,
            max_output_tokens: 128,
        }
    }

    fn request(max_new_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            model_id: "mock-model".to_owned(),
            prompt: "hello".to_owned(),
            max_new_tokens,
            sampling: SamplingParameters {
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(7),
            },
            stop_sequences: Vec::new(),
            stream: true,
        }
    }

    fn engine(
        counters: Arc<MockCounters>,
        event_capacity: usize,
        maximum_requests: usize,
        step_delay: Duration,
    ) -> InferenceEngine {
        InferenceEngine::start(
            model(),
            EngineConfig {
                command_queue_capacity: 16,
                event_channel_capacity: event_capacity,
                max_inflight_requests: maximum_requests,
                admission_timeout: Duration::from_secs(1),
                idle_poll_interval: Duration::from_millis(1),
            },
            MockBackend::new(counters, step_delay),
        )
        .expect("mock engine")
    }

    #[test]
    fn metrics_snapshot_is_serialized_through_the_worker() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(counters, 2, 2, Duration::ZERO);
        assert_eq!(engine.final_metrics_snapshot(), None);
        assert_eq!(
            engine.metrics_snapshot().expect("metrics snapshot"),
            super::EngineMetricsSnapshot::default()
        );
        engine.shutdown(Duration::from_secs(1)).expect("shutdown");
        assert_eq!(
            engine.final_metrics_snapshot(),
            Some(super::EngineMetricsSnapshot::default())
        );
    }

    #[test]
    fn staged_shape_metrics_count_only_committed_iterations() {
        let mut metrics =
            prepare_batch_shape_metrics([8].into_iter()).expect("fixed-maximum bucket");
        let staged = Some(BatchShapeExecutionSample {
            active_rows: 3,
            selected_dense_rows: 8,
            padding_rows: 5,
            gpu_execution_ns: 70,
        });

        // A D2H failure returns before staging. Sampling, result-construction,
        // and scheduler-commit failures may retain this local staged value, but
        // none produces the authoritative iteration metric required below.
        record_committed_batch_shape(&mut metrics, staged, false);
        assert_eq!(metrics.prepared_buckets()[0].hit_count, 0);
        assert_eq!(metrics.last, None);

        record_committed_batch_shape(&mut metrics, staged, true);
        let bucket = metrics.prepared_buckets()[0];
        assert_eq!(bucket.hit_count, 1);
        assert_eq!(bucket.latency_sample_count, 1);
        assert_eq!(bucket.gpu_execution_ns_total, 70);
        assert_eq!(bucket.gpu_execution_ns_average(), Some(70));
        assert_eq!(bucket.gpu_execution_ns_maximum, 70);
        assert_eq!(bucket.gpu_execution_ns_last, 70);
        assert_eq!(
            metrics.last,
            Some(super::EngineBatchShapeObservation {
                active_rows: 3,
                selected_dense_rows: 8,
                padding_rows: 5,
            })
        );
    }

    #[test]
    fn active_shape_buckets_route_without_hot_path_growth() {
        let mut metrics =
            prepare_batch_shape_metrics([1, 2, 4, 8].into_iter()).expect("active-row buckets");
        record_committed_batch_shape(
            &mut metrics,
            Some(BatchShapeExecutionSample {
                active_rows: 3,
                selected_dense_rows: 4,
                padding_rows: 1,
                gpu_execution_ns: 11,
            }),
            true,
        );

        assert_eq!(metrics.bucket_count, 4);
        assert_eq!(
            metrics
                .prepared_buckets()
                .iter()
                .map(|bucket| (bucket.dense_rows, bucket.hit_count))
                .collect::<Vec<_>>(),
            [(1, 0), (2, 0), (4, 1), (8, 0)]
        );
        assert!(!std::mem::needs_drop::<EngineBatchShapeMetricsSnapshot>());

        let source = include_str!("engine.rs");
        let start = source
            .find("fn record_committed_batch_shape(")
            .expect("shape recorder source");
        let end = source[start..]
            .find("/// Bounded operational snapshot")
            .map(|offset| start + offset)
            .expect("shape recorder source end");
        let hot_path = &source[start..end];
        for forbidden in ["Vec<", "Box<", "String", "format!", "collect(", "push("] {
            assert!(
                !hot_path.contains(forbidden),
                "shape hot path must not contain allocation primitive {forbidden}"
            );
        }
    }

    #[test]
    fn shape_metric_overflow_is_isolated_as_degraded_observability() {
        let mut metrics =
            prepare_batch_shape_metrics([8].into_iter()).expect("fixed-maximum bucket");
        metrics.buckets[0].hit_count = u64::MAX;
        let before = metrics.buckets[0];

        record_committed_batch_shape(
            &mut metrics,
            Some(BatchShapeExecutionSample {
                active_rows: 8,
                selected_dense_rows: 8,
                padding_rows: 0,
                gpu_execution_ns: 1,
            }),
            true,
        );

        assert!(metrics.metrics_degraded);
        assert_eq!(metrics.buckets[0], before);
        assert_eq!(metrics.last, None);
    }

    #[test]
    fn committed_shape_without_matching_flattened_rows_degrades_without_a_hit() {
        let mut metrics =
            prepare_batch_shape_metrics([8].into_iter()).expect("fixed-maximum bucket");

        // The CUDA staging boundary converts a mismatch between the runtime's
        // active rows and IterationPlan::total_tokens into a missing sample.
        record_committed_batch_shape(&mut metrics, None, true);

        assert!(metrics.metrics_degraded);
        assert_eq!(metrics.prepared_buckets()[0].hit_count, 0);
    }

    #[test]
    fn events_are_ordered_and_terminal_is_last() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(Arc::clone(&counters), 8, 4, Duration::ZERO);
        let handle = engine.submit(request(3)).expect("admitted");
        for _ in 0..3 {
            assert_eq!(
                handle.recv_timeout(Duration::from_secs(1)),
                Ok(GenerationEvent::TokenDelta {
                    text: "x".to_owned()
                })
            );
        }
        assert!(matches!(
            handle.recv_timeout(Duration::from_secs(1)),
            Ok(GenerationEvent::Finished {
                reason: FinishReason::Length,
                usage
            }) if usage.completion_tokens() == 3
        ));
        assert!(handle.recv_timeout(Duration::from_millis(10)).is_err());
        engine.shutdown(Duration::from_secs(1)).expect("shutdown");
    }

    #[test]
    fn model_mismatch_and_live_capacity_are_rejected() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(Arc::clone(&counters), 8, 1, Duration::from_millis(20));
        let mut wrong = request(1);
        wrong.model_id = "other".to_owned();
        assert!(matches!(
            engine.submit(wrong),
            Err(EngineError::ModelNotFound)
        ));
        let first = engine.submit(request(100)).expect("first admitted");
        assert!(matches!(
            engine.submit(request(1)),
            Err(EngineError::Overloaded)
        ));
        first.cancel();
        drop(first);
        engine.shutdown(Duration::from_secs(2)).expect("shutdown");
    }

    #[test]
    fn drop_guard_cancels_without_relying_on_command_capacity() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(Arc::clone(&counters), 4, 2, Duration::from_millis(5));
        let handle = engine.submit(request(100)).expect("admitted");
        drop(handle);
        let deadline = Instant::now() + Duration::from_secs(1);
        while counters.cancelled.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(counters.cancelled.load(Ordering::Acquire), 1);
        engine.shutdown(Duration::from_secs(1)).expect("shutdown");
    }

    #[test]
    fn slow_consumer_is_cancelled_when_request_channel_is_full() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(Arc::clone(&counters), 1, 2, Duration::ZERO);
        let handle = engine.submit(request(100)).expect("admitted");
        let deadline = Instant::now() + Duration::from_secs(1);
        while counters.cancelled.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(counters.cancelled.load(Ordering::Acquire), 1);
        assert!(matches!(
            handle.recv_timeout(Duration::from_secs(1)),
            Ok(GenerationEvent::TokenDelta { .. })
        ));
        engine.shutdown(Duration::from_secs(1)).expect("shutdown");
    }

    #[test]
    fn graceful_shutdown_rejects_new_work_and_closes_backend() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(Arc::clone(&counters), 8, 4, Duration::from_millis(5));
        let handle = engine.submit(request(100)).expect("admitted");
        engine.begin_shutdown();
        assert!(matches!(
            engine.submit(request(1)),
            Err(EngineError::ShuttingDown)
        ));
        engine.shutdown(Duration::from_secs(2)).expect("shutdown");
        assert_eq!(counters.shutdowns.load(Ordering::Acquire), 1);
        loop {
            match handle
                .recv_timeout(Duration::from_secs(1))
                .expect("shutdown terminal event")
            {
                GenerationEvent::TokenDelta { .. } => {}
                GenerationEvent::Finished {
                    reason: FinishReason::Cancelled,
                    ..
                }
                | GenerationEvent::Failed {
                    class: ServiceErrorClass::ShuttingDown,
                } => break,
                event => panic!("unexpected shutdown event: {event:?}"),
            }
        }
    }

    #[test]
    fn concurrent_requests_finish_without_deadlock() {
        let counters = Arc::new(MockCounters::default());
        let engine = engine(Arc::clone(&counters), 8, 8, Duration::ZERO);
        let mut clients = Vec::new();
        for _ in 0..8 {
            let engine = engine.clone();
            clients.push(thread::spawn(move || {
                let handle = engine.submit(request(2)).expect("admitted");
                let mut tokens = 0;
                loop {
                    match handle
                        .recv_timeout(Duration::from_secs(2))
                        .expect("worker event")
                    {
                        GenerationEvent::TokenDelta { .. } => tokens += 1,
                        GenerationEvent::Finished { .. } => break tokens,
                        GenerationEvent::Failed { class } => {
                            panic!("unexpected failure: {class:?}")
                        }
                    }
                }
            }));
        }
        for client in clients {
            assert_eq!(client.join().expect("client thread"), 2);
        }
        engine.shutdown(Duration::from_secs(1)).expect("shutdown");
        assert_eq!(counters.admitted.load(Ordering::Acquire), 8);
    }

    #[test]
    fn user_input_diagnostics_do_not_retain_prompt_text() {
        let secret = "private-customer-prompt-7c9d";
        let error = private_request_error("prompt tokenization failed");
        assert_eq!(error.detail(), "prompt tokenization failed");
        assert!(!error.detail().contains(secret));
    }

    #[test]
    fn deferred_cancel_checkpoint_excludes_current_uncommitted_token() {
        let previous_pending = "previous committed stop-prefix".as_bytes();
        let checkpoint = visible_utf8_prefix(previous_pending).to_owned();
        let mut pending_after_sample = previous_pending.to_vec();
        pending_after_sample.extend_from_slice(b" current-uncommitted-token");

        assert_eq!(checkpoint, "previous committed stop-prefix");
        assert!(!checkpoint.contains("uncommitted"));
        assert!(
            visible_utf8_prefix(&pending_after_sample).contains("uncommitted"),
            "control demonstrates why the pre-iteration checkpoint is required"
        );

        let incomplete = [b'o', b'k', 0xe2, 0x82];
        assert_eq!(visible_utf8_prefix(&incomplete), "ok");
    }
}
