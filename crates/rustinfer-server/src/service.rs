//! Bounded, dependency-free HTTP service for the initial completion API.
//!
//! The listener owns a bounded connection queue and a fixed worker set. Each
//! connection carries one close-delimited HTTP/1.1 request, so socket lifetime
//! is also the cancellation lifetime for submitted generation work.
//! Streaming writes detect decode-time disconnects and use a one-millisecond
//! peer probe only on idle event polls. Aggregate responses additionally probe
//! every 16 deltas, bounding service-consumed work after disconnect to 16
//! deltas plus the backend's bounded channel and current execution iteration.

use std::collections::VecDeque;
use std::error;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::domain::{
    FinishReason, GenerationEvent, GenerationRequest, ModelMetadata, RequestLimits,
    RequestMetadata, ServiceErrorClass,
};
use crate::engine::EngineMetricsSnapshot;
use crate::http::{
    HttpLimits, HttpMethod, HttpReadError, HttpRequest, read_request, write_response,
    write_sse_head,
};
use crate::openai::{
    ApiError, CompletionRequest, CompletionResponse, ErrorObject, ErrorResponse, ModelListResponse,
    ModelObject, SseStreamEncoder, normalize_completion_request,
};

const JSON_CONTENT_TYPE: &str = "application/json; charset=utf-8";
const CONNECTION_POLL_INTERVAL: Duration = Duration::from_millis(10);
const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(25);
const DISCONNECT_PEEK_TIMEOUT: Duration = Duration::from_millis(1);
const NON_STREAMING_PROBE_DELTA_INTERVAL: u64 = 16;

/// Terminal transport-level state recorded without prompt or generated text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestObservationStatus {
    /// A successful terminal event and response were delivered.
    Completed,
    /// A sanitized backend or service error ended the request.
    Failed,
    /// The client socket closed or stopped accepting bytes.
    ClientDisconnected,
}

/// Bounded, transport-independent metadata for one admitted request.
///
/// This type intentionally has no prompt or generated-text field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestObservation {
    /// Stable request identifier assigned by the backend.
    pub request_id: String,
    /// Public model identifier.
    pub model_id: String,
    /// Time spent inside backend admission.
    pub queue_wait: Duration,
    /// Time from admission completion to the first visible delta.
    pub time_to_first_token: Option<Duration>,
    /// Backend-reported generated token count, or observed deltas before error.
    pub tokens_generated: u64,
    /// Successful or backend-provided terminal reason, when available.
    pub finish_reason: Option<FinishReason>,
    /// Transport-level completion class.
    pub status: RequestObservationStatus,
    /// Stable failure class with all backend detail removed.
    pub error_class: Option<ServiceErrorClass>,
    /// Active backend requests at observation time.
    pub active_requests: usize,
    /// Waiting backend requests at observation time.
    pub waiting_requests: usize,
}

/// Snapshot of the bounded recent-observation buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationSnapshot {
    /// Oldest-to-newest retained observations.
    pub observations: Vec<RequestObservation>,
    /// Observations evicted because the fixed buffer was full.
    pub dropped_observations: u64,
}

#[derive(Debug)]
struct ObservationBuffer {
    capacity: usize,
    state: Mutex<ObservationState>,
}

#[derive(Debug)]
struct ObservationState {
    entries: VecDeque<RequestObservation>,
    dropped: u64,
}

#[derive(Debug, Default)]
struct ServiceMetrics {
    cancellations: AtomicU64,
    disconnects: AtomicU64,
    overloads: AtomicU64,
}

impl ServiceMetrics {
    fn record_overload(&self) {
        let _ = self
            .overloads
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            });
    }

    fn record_terminal(
        &self,
        status: RequestObservationStatus,
        error_class: Option<ServiceErrorClass>,
    ) {
        if status == RequestObservationStatus::ClientDisconnected {
            let _ = self
                .disconnects
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_add(1)
                });
        }
        if matches!(
            error_class,
            Some(
                ServiceErrorClass::Cancelled
                    | ServiceErrorClass::Timeout
                    | ServiceErrorClass::ShuttingDown
            )
        ) {
            let _ = self
                .cancellations
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_add(1)
                });
        }
    }
}

impl ObservationBuffer {
    fn try_new(capacity: usize) -> Result<Self, ServerStartError> {
        let mut entries = VecDeque::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|_| ServerStartError::HostAllocation {
                resource: "request observation buffer",
            })?;
        Ok(Self {
            capacity,
            state: Mutex::new(ObservationState {
                entries,
                dropped: 0,
            }),
        })
    }

    fn record(&self, observation: RequestObservation) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.entries.len() == self.capacity {
            let _ = state.entries.pop_front();
            state.dropped = state.dropped.saturating_add(1);
        }
        state.entries.push_back(observation);
    }

    fn snapshot(&self) -> ObservationSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ObservationSnapshot {
            observations: state.entries.iter().cloned().collect(),
            dropped_observations: state.dropped,
        }
    }

    fn dropped(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .dropped
    }
}

/// Cancellation operation attached to one admitted generation request.
pub trait RequestCancellation: Send + Sync {
    /// Requests cancellation at the backend's next safe reclamation point.
    fn cancel(&self);
}

/// Cheap, non-blocking backend state used by readiness and admission checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendStatus {
    /// Model execution is initialized and able to make progress.
    pub ready: bool,
    /// New requests may be submitted.
    pub accepting: bool,
    /// Requests currently holding active execution state.
    pub active_requests: usize,
    /// Requests admitted but waiting for execution capacity.
    pub waiting_requests: usize,
}

/// Backend boundary kept independent of all HTTP and `OpenAI` DTOs.
pub trait CompletionBackend: Send + Sync {
    /// Returns public metadata for the single loaded model.
    fn model_metadata(&self) -> ModelMetadata;

    /// Returns a non-blocking readiness and load snapshot.
    fn status(&self) -> BackendStatus;

    /// Returns a bounded operational snapshot without exposing request text.
    /// Backends without scheduler or native allocation state return empty
    /// optional fields in the snapshot.
    ///
    /// # Errors
    ///
    /// Returns a stable class when a consistent snapshot cannot be produced.
    fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, ServiceErrorClass> {
        Ok(EngineMetricsSnapshot::default())
    }

    /// Admits a normalized request and returns its bounded event stream.
    ///
    /// # Errors
    ///
    /// Returns only a stable public failure class. Backend diagnostics must be
    /// logged internally and must not cross this boundary.
    fn submit(&self, request: GenerationRequest) -> Result<SubmittedRequest, ServiceErrorClass>;

    /// Stops new admission without waiting for active requests.
    fn begin_shutdown(&self);

    /// Reclaims backend resources no later than the supplied service deadline.
    ///
    /// # Errors
    ///
    /// Returns a stable class if shutdown could not complete cleanly.
    fn shutdown(&self, deadline: Instant) -> Result<(), ServiceErrorClass>;
}

/// One backend submission guarded by cancellation-on-drop.
pub struct SubmittedRequest {
    metadata: RequestMetadata,
    events: Receiver<GenerationEvent>,
    cancellation: Arc<dyn RequestCancellation>,
    armed: bool,
}

impl SubmittedRequest {
    /// Creates an armed submission. Dropping it before [`Self::disarm`]
    /// requests cancellation.
    #[must_use]
    pub fn new(
        metadata: RequestMetadata,
        events: Receiver<GenerationEvent>,
        cancellation: Arc<dyn RequestCancellation>,
    ) -> Self {
        Self {
            metadata,
            events,
            cancellation,
            armed: true,
        }
    }

    /// Stable metadata used by both response modes.
    #[must_use]
    pub const fn metadata(&self) -> &RequestMetadata {
        &self.metadata
    }

    /// Waits for one generation event for at most `timeout`.
    ///
    /// # Errors
    ///
    /// Returns timeout or channel-disconnection state from the bounded backend
    /// event channel.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<GenerationEvent, RecvTimeoutError> {
        self.events.recv_timeout(timeout)
    }

    /// Attempts to receive an event without blocking.
    ///
    /// # Errors
    ///
    /// Returns empty or disconnected state from the backend event channel.
    pub fn try_recv(&self) -> Result<GenerationEvent, TryRecvError> {
        self.events.try_recv()
    }

    /// Marks a terminal event as observed, suppressing cancellation on drop.
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl fmt::Debug for SubmittedRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubmittedRequest")
            .field("metadata", &self.metadata)
            .field("armed", &self.armed)
            .finish_non_exhaustive()
    }
}

impl Drop for SubmittedRequest {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// Resource and timeout bounds for [`start_server`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServerConfig {
    /// Listener address. Port zero is allowed for loopback tests.
    pub bind_address: SocketAddr,
    /// Fixed number of connection workers.
    pub worker_threads: usize,
    /// Maximum accepted sockets waiting for a worker.
    pub connection_queue_capacity: usize,
    /// Maximum inactivity while framing one request.
    pub read_timeout: Duration,
    /// Absolute wall-clock bound for receiving headers and the declared body.
    pub framing_timeout: Duration,
    /// Maximum inactivity while writing headers, JSON, or one SSE frame.
    pub write_timeout: Duration,
    /// End-to-end event-wait deadline after backend admission.
    pub request_timeout: Duration,
    /// Backend drain deadline used by [`ServerHandle::shutdown`].
    pub shutdown_grace: Duration,
    /// HTTP framing bounds.
    pub http_limits: HttpLimits,
    /// DTO normalization bounds.
    pub request_limits: RequestLimits,
    /// Maximum accumulated UTF-8 bytes in a non-streaming completion.
    pub maximum_non_streaming_bytes: usize,
    /// Number of recent request observations retained in memory.
    pub observation_capacity: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: SocketAddr::from(([127, 0, 0, 1], 8080)),
            worker_threads: 8,
            connection_queue_capacity: 128,
            read_timeout: Duration::from_secs(10),
            framing_timeout: Duration::from_secs(15),
            write_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(120),
            shutdown_grace: Duration::from_secs(30),
            http_limits: HttpLimits::default(),
            request_limits: RequestLimits::default(),
            maximum_non_streaming_bytes: 16 * 1_024 * 1_024,
            observation_capacity: 1_024,
        }
    }
}

impl ServerConfig {
    fn validate(self) -> Result<Self, ServerStartError> {
        if self.worker_threads == 0 {
            return Err(ServerStartError::InvalidConfig {
                field: "worker_threads",
            });
        }
        if self.connection_queue_capacity == 0 {
            return Err(ServerStartError::InvalidConfig {
                field: "connection_queue_capacity",
            });
        }
        for (field, duration) in [
            ("read_timeout", self.read_timeout),
            ("framing_timeout", self.framing_timeout),
            ("write_timeout", self.write_timeout),
            ("request_timeout", self.request_timeout),
            ("shutdown_grace", self.shutdown_grace),
        ] {
            if duration.is_zero() {
                return Err(ServerStartError::InvalidConfig { field });
            }
        }
        if self.maximum_non_streaming_bytes == 0 {
            return Err(ServerStartError::InvalidConfig {
                field: "maximum_non_streaming_bytes",
            });
        }
        if self.observation_capacity == 0 {
            return Err(ServerStartError::InvalidConfig {
                field: "observation_capacity",
            });
        }
        if self.request_limits.max_model_bytes == 0
            || self.request_limits.max_prompt_bytes == 0
            || self.request_limits.max_output_tokens == 0
            || (self.request_limits.max_stop_sequences > 0
                && (self.request_limits.max_stop_sequence_bytes == 0
                    || self.request_limits.max_total_stop_bytes == 0))
        {
            return Err(ServerStartError::InvalidConfig {
                field: "request_limits",
            });
        }
        self.http_limits
            .validate()
            .map_err(|_| ServerStartError::InvalidConfig {
                field: "http_limits",
            })?;
        Ok(self)
    }
}

/// Failure to bind or validate an HTTP service.
#[derive(Debug)]
#[non_exhaustive]
pub enum ServerStartError {
    /// One configured resource bound cannot produce a bounded service.
    InvalidConfig { field: &'static str },
    /// Public backend model metadata is invalid.
    InvalidModelMetadata,
    /// A cold service allocation could not be reserved.
    HostAllocation { resource: &'static str },
    /// Listener creation or setup failed.
    Io(io::Error),
}

impl fmt::Display for ServerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field } => write!(formatter, "invalid server config {field}"),
            Self::InvalidModelMetadata => formatter.write_str("invalid public model metadata"),
            Self::HostAllocation { resource } => {
                write!(formatter, "could not allocate {resource}")
            }
            Self::Io(source) => write!(formatter, "could not start HTTP listener: {source}"),
        }
    }
}

impl error::Error for ServerStartError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::InvalidConfig { .. }
            | Self::InvalidModelMetadata
            | Self::HostAllocation { .. } => None,
        }
    }
}

impl From<io::Error> for ServerStartError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

/// Failure observed while joining the service or draining its backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ServerShutdownError {
    /// A service thread panicked while processing a connection.
    ThreadPanicked,
    /// The backend did not shut down cleanly.
    Backend(ServiceErrorClass),
    /// The configured global shutdown deadline elapsed before one stage ended.
    DeadlineExceeded { stage: &'static str },
    /// The bounded backend-shutdown coordinator thread could not be started.
    CoordinatorUnavailable,
}

impl fmt::Display for ServerShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadPanicked => formatter.write_str("an HTTP service thread panicked"),
            Self::Backend(class) => write!(formatter, "backend shutdown failed: {class:?}"),
            Self::DeadlineExceeded { stage } => {
                write!(
                    formatter,
                    "shutdown deadline exceeded while waiting for {stage}"
                )
            }
            Self::CoordinatorUnavailable => {
                formatter.write_str("could not start the backend shutdown coordinator")
            }
        }
    }
}

impl error::Error for ServerShutdownError {}

/// Running listener and worker ownership.
pub struct ServerHandle {
    local_address: SocketAddr,
    stopping: Arc<AtomicBool>,
    backend: Arc<dyn CompletionBackend>,
    shutdown_grace: Duration,
    listener_thread: Option<JoinHandle<()>>,
    worker_threads: Vec<JoinHandle<()>>,
    connections: Arc<ConnectionRegistry>,
    observations: Arc<ObservationBuffer>,
    metrics: Arc<ServiceMetrics>,
    joined: bool,
}

#[derive(Debug)]
struct ConnectionRegistry {
    slots: Mutex<Vec<Option<RegisteredConnection>>>,
}

#[derive(Debug)]
struct RegisteredConnection {
    stream: TcpStream,
    phase: ConnectionPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnectionPhase {
    Framing,
    Serving,
}

impl ConnectionRegistry {
    fn try_new(capacity: usize) -> Result<Self, ServerStartError> {
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| ServerStartError::HostAllocation {
                resource: "connection registry",
            })?;
        slots.resize_with(capacity, || None);
        Ok(Self {
            slots: Mutex::new(slots),
        })
    }

    fn register(&self, stream: &TcpStream) -> io::Result<usize> {
        let duplicate = stream.try_clone()?;
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some((slot, entry)) = slots
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.is_none())
        else {
            return Err(io::Error::other("connection registry capacity exhausted"));
        };
        *entry = Some(RegisteredConnection {
            stream: duplicate,
            phase: ConnectionPhase::Framing,
        });
        Ok(slot)
    }

    fn mark_serving(&self, slot: usize) {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(Some(connection)) = slots.get_mut(slot) {
            connection.phase = ConnectionPhase::Serving;
        }
    }

    fn unregister(&self, slot: usize) {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = slots.get_mut(slot) {
            *entry = None;
        }
    }

    fn shutdown_all(&self) {
        let slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for connection in slots.iter().flatten() {
            let _ = connection.stream.shutdown(Shutdown::Both);
        }
    }

    fn shutdown_framing(&self) {
        let slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for connection in slots.iter().flatten() {
            if connection.phase == ConnectionPhase::Framing {
                let _ = connection.stream.shutdown(Shutdown::Both);
            }
        }
    }
}

#[derive(Debug)]
struct ConnectionJob {
    stream: TcpStream,
    registry_slot: usize,
}

impl ServerHandle {
    /// Address selected by the operating system after binding.
    #[must_use]
    pub const fn local_address(&self) -> SocketAddr {
        self.local_address
    }

    /// Returns a consistent clone of recent request observations.
    #[must_use]
    pub fn observations(&self) -> ObservationSnapshot {
        self.observations.snapshot()
    }

    /// Returns the same sanitized, fixed-shape snapshot exposed at `/metrics`.
    ///
    /// # Errors
    ///
    /// Returns a stable class if the backend worker cannot produce a
    /// consistent point-in-time snapshot.
    pub fn metrics_snapshot(&self) -> Result<OperationalMetricsSnapshot, ServiceErrorClass> {
        operational_metrics(self.backend.as_ref(), &self.metrics, &self.observations)
    }

    /// Begins graceful shutdown, joins every service thread, then drains the
    /// backend through the configured deadline.
    ///
    /// # Errors
    ///
    /// Returns a stable backend shutdown class or reports a service panic.
    pub fn shutdown(mut self) -> Result<(), ServerShutdownError> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> Result<(), ServerShutdownError> {
        if self.joined {
            return Ok(());
        }
        // This is a one-shot ownership transition. If a deadline expires,
        // unfinished threads are detached instead of making Drop retry an
        // unbounded join.
        self.joined = true;
        self.stopping.store(true, Ordering::Release);
        self.backend.begin_shutdown();
        let deadline = Instant::now()
            .checked_add(self.shutdown_grace)
            .unwrap_or_else(Instant::now);
        self.connections.shutdown_framing();

        let (backend_sender, backend_receiver) = mpsc::sync_channel(1);
        let backend = Arc::clone(&self.backend);
        let backend_thread = thread::Builder::new()
            .name("rustinfer-backend-shutdown".to_owned())
            .spawn(move || {
                let _ = backend_sender.send(backend.shutdown(deadline));
            })
            .map_err(|_| ServerShutdownError::CoordinatorUnavailable)?;

        let mut thread_panicked = false;
        if let Some(handle) = self.listener_thread.take() {
            if join_until(handle, deadline, "listener thread")? {
                thread_panicked = true;
            }
        }
        let force_close_reserve = self
            .shutdown_grace
            .checked_div(4)
            .unwrap_or(Duration::ZERO)
            .min(Duration::from_millis(250));
        let graceful_transport_deadline = deadline
            .checked_sub(force_close_reserve)
            .unwrap_or_else(Instant::now);
        while self
            .worker_threads
            .iter()
            .any(|handle| !handle.is_finished())
            && Instant::now() < graceful_transport_deadline
        {
            thread::sleep(Duration::from_millis(1));
        }
        self.connections.shutdown_all();
        for handle in self.worker_threads.drain(..) {
            if join_until(handle, deadline, "HTTP worker thread")? {
                thread_panicked = true;
            }
        }

        let backend_result = match backend_receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Disconnected) => return Err(ServerShutdownError::ThreadPanicked),
            Err(TryRecvError::Empty) => backend_receiver
                .recv_timeout(remaining_until(deadline).ok_or(
                    ServerShutdownError::DeadlineExceeded {
                        stage: "backend shutdown",
                    },
                )?)
                .map_err(|error| match error {
                    RecvTimeoutError::Timeout => ServerShutdownError::DeadlineExceeded {
                        stage: "backend shutdown",
                    },
                    RecvTimeoutError::Disconnected => ServerShutdownError::ThreadPanicked,
                })?,
        };
        if join_until(backend_thread, deadline, "backend shutdown thread")? {
            thread_panicked = true;
        }

        if let Err(class) = backend_result {
            return Err(ServerShutdownError::Backend(class));
        }
        if thread_panicked {
            return Err(ServerShutdownError::ThreadPanicked);
        }
        Ok(())
    }
}

fn remaining_until(deadline: Instant) -> Option<Duration> {
    deadline.checked_duration_since(Instant::now())
}

fn join_until(
    handle: JoinHandle<()>,
    deadline: Instant,
    stage: &'static str,
) -> Result<bool, ServerShutdownError> {
    while !handle.is_finished() {
        let Some(remaining) = remaining_until(deadline) else {
            return Err(ServerShutdownError::DeadlineExceeded { stage });
        };
        thread::sleep(remaining.min(Duration::from_millis(1)));
    }
    Ok(handle.join().is_err())
}

impl fmt::Debug for ServerHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerHandle")
            .field("local_address", &self.local_address)
            .field("stopping", &self.stopping.load(Ordering::Acquire))
            .field("worker_threads", &self.worker_threads.len())
            .finish_non_exhaustive()
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        let _ = self.shutdown_inner();
    }
}

/// Starts a bounded HTTP service on `config.bind_address`.
///
/// # Errors
///
/// Returns an error for invalid bounds, invalid public model metadata, a cold
/// host-allocation failure, or a listener bind/setup failure.
pub fn start_server(
    config: ServerConfig,
    backend: Arc<dyn CompletionBackend>,
) -> Result<ServerHandle, ServerStartError> {
    let config = config.validate()?;
    backend
        .model_metadata()
        .validate()
        .map_err(|_| ServerStartError::InvalidModelMetadata)?;

    let listener = TcpListener::bind(config.bind_address)?;
    listener.set_nonblocking(true)?;
    let local_address = listener.local_addr()?;
    let stopping = Arc::new(AtomicBool::new(false));
    let connection_capacity = config
        .worker_threads
        .checked_add(config.connection_queue_capacity)
        .ok_or(ServerStartError::InvalidConfig {
            field: "connection capacity",
        })?;
    let connections = Arc::new(ConnectionRegistry::try_new(connection_capacity)?);
    let (connection_sender, connection_receiver) =
        mpsc::sync_channel::<ConnectionJob>(config.connection_queue_capacity);
    let connection_receiver = Arc::new(Mutex::new(connection_receiver));
    let observations = Arc::new(ObservationBuffer::try_new(config.observation_capacity)?);
    let metrics = Arc::new(ServiceMetrics::default());

    let mut worker_threads = Vec::new();
    worker_threads
        .try_reserve_exact(config.worker_threads)
        .map_err(|_| ServerStartError::HostAllocation {
            resource: "HTTP worker handle table",
        })?;
    for worker_index in 0..config.worker_threads {
        let receiver = Arc::clone(&connection_receiver);
        let worker_backend = Arc::clone(&backend);
        let worker_stopping = Arc::clone(&stopping);
        let worker_observations = Arc::clone(&observations);
        let worker_metrics = Arc::clone(&metrics);
        let worker_connections = Arc::clone(&connections);
        worker_threads.push(
            thread::Builder::new()
                .name(format!("rustinfer-http-{worker_index}"))
                .spawn(move || {
                    worker_loop(
                        &receiver,
                        worker_backend.as_ref(),
                        &worker_stopping,
                        &worker_connections,
                        &worker_observations,
                        &worker_metrics,
                        config,
                    );
                })?,
        );
    }

    let listener_stopping = Arc::clone(&stopping);
    let listener_connections = Arc::clone(&connections);
    let listener_metrics = Arc::clone(&metrics);
    let listener_thread = thread::Builder::new()
        .name("rustinfer-listener".to_owned())
        .spawn(move || {
            listener_loop(
                &listener,
                &connection_sender,
                &listener_stopping,
                &listener_connections,
                &listener_metrics,
                config.write_timeout,
            );
        })?;

    Ok(ServerHandle {
        local_address,
        stopping,
        backend,
        shutdown_grace: config.shutdown_grace,
        listener_thread: Some(listener_thread),
        worker_threads,
        connections,
        observations,
        metrics,
        joined: false,
    })
}

fn listener_loop(
    listener: &TcpListener,
    sender: &SyncSender<ConnectionJob>,
    stopping: &AtomicBool,
    connections: &ConnectionRegistry,
    metrics: &ServiceMetrics,
    write_timeout: Duration,
) {
    while !stopping.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                if stopping.load(Ordering::Acquire) {
                    let _ = write_api_error(&mut stream, &ApiError::ShuttingDown);
                    let _ = stream.shutdown(Shutdown::Both);
                    break;
                }
                if stream.set_write_timeout(Some(write_timeout)).is_err() {
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                }
                let _ = stream.set_nodelay(true);
                let Ok(registry_slot) = connections.register(&stream) else {
                    metrics.record_overload();
                    let _ = write_api_error(&mut stream, &ApiError::Overloaded);
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                };
                match sender.try_send(ConnectionJob {
                    stream,
                    registry_slot,
                }) {
                    Ok(()) => {}
                    Err(TrySendError::Full(mut job)) => {
                        metrics.record_overload();
                        connections.unregister(job.registry_slot);
                        let _ = write_api_error(&mut job.stream, &ApiError::Overloaded);
                        let _ = job.stream.shutdown(Shutdown::Both);
                    }
                    Err(TrySendError::Disconnected(mut job)) => {
                        connections.unregister(job.registry_slot);
                        let _ = write_api_error(&mut job.stream, &ApiError::ShuttingDown);
                        let _ = job.stream.shutdown(Shutdown::Both);
                        break;
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(CONNECTION_POLL_INTERVAL);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
}

fn worker_loop(
    receiver: &Mutex<Receiver<ConnectionJob>>,
    backend: &dyn CompletionBackend,
    stopping: &AtomicBool,
    connections: &ConnectionRegistry,
    observations: &Arc<ObservationBuffer>,
    metrics: &Arc<ServiceMetrics>,
    config: ServerConfig,
) {
    loop {
        let job_result = receiver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(CONNECTION_POLL_INTERVAL);
        match job_result {
            Ok(mut job) => {
                if stopping.load(Ordering::Acquire) {
                    let _ = write_api_error(&mut job.stream, &ApiError::ShuttingDown);
                    let _ = job.stream.shutdown(Shutdown::Both);
                } else {
                    let context = ConnectionContext {
                        backend,
                        stopping,
                        connections,
                        observations,
                        metrics,
                        config,
                    };
                    handle_connection(&mut job.stream, job.registry_slot, &context);
                    let _ = job.stream.shutdown(Shutdown::Both);
                }
                connections.unregister(job.registry_slot);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

struct ConnectionContext<'a> {
    backend: &'a dyn CompletionBackend,
    stopping: &'a AtomicBool,
    connections: &'a ConnectionRegistry,
    observations: &'a Arc<ObservationBuffer>,
    metrics: &'a Arc<ServiceMetrics>,
    config: ServerConfig,
}

fn handle_connection(
    stream: &mut TcpStream,
    registry_slot: usize,
    context: &ConnectionContext<'_>,
) {
    if stream
        .set_read_timeout(Some(context.config.read_timeout))
        .is_err()
        || stream
            .set_write_timeout(Some(context.config.write_timeout))
            .is_err()
    {
        return;
    }
    let framing_deadline = Instant::now()
        .checked_add(context.config.framing_timeout)
        .unwrap_or_else(Instant::now);
    let mut reader = FramingReader {
        stream,
        inactivity_timeout: context.config.read_timeout,
        deadline: framing_deadline,
    };
    let request = match read_request(&mut reader, context.config.http_limits) {
        Ok(request) => request,
        Err(error) => {
            if !is_peer_gone(&error) {
                let _ = write_http_read_error(stream, &error);
            }
            return;
        }
    };
    context.connections.mark_serving(registry_slot);
    route_request(
        stream,
        context.backend,
        context.stopping,
        context.observations,
        context.metrics,
        context.config,
        &request,
    );
}

struct FramingReader<'a> {
    stream: &'a mut TcpStream,
    inactivity_timeout: Duration,
    deadline: Instant,
}

impl Read for FramingReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "framing deadline elapsed"))?;
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "framing deadline elapsed",
            ));
        }
        self.stream
            .set_read_timeout(Some(self.inactivity_timeout.min(remaining)))?;
        self.stream.read(buffer)
    }
}

fn route_request(
    stream: &mut TcpStream,
    backend: &dyn CompletionBackend,
    stopping: &AtomicBool,
    observations: &Arc<ObservationBuffer>,
    metrics: &Arc<ServiceMetrics>,
    config: ServerConfig,
    request: &HttpRequest,
) {
    let path = request.target().split('?').next().unwrap_or_default();
    match (request.method(), path) {
        (HttpMethod::Get, "/healthz") => write_health(stream, backend.status(), false),
        (HttpMethod::Get, "/readyz") => write_health(stream, backend.status(), true),
        (HttpMethod::Get, "/metrics") => {
            write_operational_metrics(stream, backend, metrics, observations);
        }
        (HttpMethod::Get, "/v1/models") => write_model_list(stream, backend),
        (HttpMethod::Get, path) if path.starts_with("/v1/models/") => {
            write_model(stream, backend, &path["/v1/models/".len()..]);
        }
        (HttpMethod::Post, "/v1/completions") => {
            handle_completion(
                stream,
                backend,
                stopping,
                observations,
                metrics,
                config,
                request.body(),
            );
        }
        (HttpMethod::Get, "/v1/completions")
        | (HttpMethod::Post, "/healthz" | "/readyz" | "/metrics" | "/v1/models") => {
            let _ = write_public_error(
                stream,
                405,
                "method_not_allowed",
                "the endpoint does not accept this method",
                None,
            );
        }
        _ => {
            let _ = write_public_error(stream, 404, "not_found", "endpoint not found", None);
        }
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    ready: bool,
    accepting: bool,
    active_requests: usize,
    waiting_requests: usize,
}

/// Native live-allocation gauges in the operational metrics response.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct OperationalAllocationMetrics {
    /// Number of live CUDA device allocations.
    pub device_live_count: u64,
    /// Bytes retained by live CUDA device allocations.
    pub device_live_bytes: u64,
    /// Number of live pinned-host allocations.
    pub pinned_live_count: u64,
    /// Bytes retained by live pinned-host allocations.
    pub pinned_live_bytes: u64,
}

/// Monotonic service counters in the operational metrics response.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct OperationalCounters {
    /// Cancellation terminals observed by the scheduler or transport.
    pub cancellations: u64,
    /// Client disconnect terminals observed by the service.
    pub disconnects: u64,
    /// Requests or connections rejected at bounded overload boundaries.
    pub overloads: u64,
    /// Observation samples evicted from the fixed-capacity ring.
    pub dropped_samples: u64,
}

/// Fixed, prompt-free JSON contract used by release and soak gates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct OperationalMetricsSnapshot {
    /// Requests currently admitted to backend ownership.
    pub active_requests: u64,
    /// Requests waiting in bounded admission state.
    pub waiting_requests: u64,
    /// Physical KV blocks currently allocated.
    pub kv_allocated_blocks: u64,
    /// Native allocation gauges.
    pub allocation: OperationalAllocationMetrics,
    /// Monotonic service counters.
    pub counters: OperationalCounters,
}

fn operational_metrics(
    backend: &dyn CompletionBackend,
    metrics: &ServiceMetrics,
    observations: &ObservationBuffer,
) -> Result<OperationalMetricsSnapshot, ServiceErrorClass> {
    let status = backend.status();
    let backend_metrics = backend.metrics_snapshot()?;
    let scheduler_cancellations = backend_metrics
        .scheduler
        .map_or(0, |snapshot| snapshot.requests_cancelled);
    let kv_allocated_blocks = backend_metrics.scheduler.map_or(0, |snapshot| {
        u64::try_from(snapshot.gauges.allocated_kv_blocks).unwrap_or(u64::MAX)
    });
    let allocation = backend_metrics.allocation.unwrap_or_default();
    Ok(OperationalMetricsSnapshot {
        active_requests: u64::try_from(status.active_requests).unwrap_or(u64::MAX),
        waiting_requests: u64::try_from(status.waiting_requests).unwrap_or(u64::MAX),
        kv_allocated_blocks,
        allocation: OperationalAllocationMetrics {
            device_live_count: allocation.device_live_allocations,
            device_live_bytes: allocation.device_live_bytes,
            pinned_live_count: allocation.pinned_host_live_allocations,
            pinned_live_bytes: allocation.pinned_host_live_bytes,
        },
        counters: OperationalCounters {
            cancellations: scheduler_cancellations
                .max(metrics.cancellations.load(Ordering::Acquire)),
            disconnects: metrics.disconnects.load(Ordering::Acquire),
            overloads: metrics.overloads.load(Ordering::Acquire),
            dropped_samples: observations.dropped(),
        },
    })
}

fn write_operational_metrics(
    stream: &mut TcpStream,
    backend: &dyn CompletionBackend,
    metrics: &ServiceMetrics,
    observations: &ObservationBuffer,
) {
    match operational_metrics(backend, metrics, observations) {
        Ok(snapshot) => {
            let _ = write_json(stream, 200, &snapshot);
        }
        Err(_) => {
            let _ = write_public_error(
                stream,
                503,
                "metrics_unavailable",
                "operational metrics are temporarily unavailable",
                None,
            );
        }
    }
}

fn write_health(stream: &mut TcpStream, backend: BackendStatus, readiness: bool) {
    let available = backend.ready && backend.accepting;
    let status = if readiness && !available { 503 } else { 200 };
    let body = HealthResponse {
        status: if available { "ok" } else { "unavailable" },
        ready: backend.ready,
        accepting: backend.accepting,
        active_requests: backend.active_requests,
        waiting_requests: backend.waiting_requests,
    };
    let _ = write_json(stream, status, &body);
}

fn write_model_list(stream: &mut TcpStream, backend: &dyn CompletionBackend) {
    let metadata = backend.model_metadata();
    let _ = write_json(stream, 200, &ModelListResponse::single(&metadata));
}

fn write_model(stream: &mut TcpStream, backend: &dyn CompletionBackend, requested_id: &str) {
    let metadata = backend.model_metadata();
    if requested_id == metadata.model_id {
        let _ = write_json(stream, 200, &ModelObject::from(&metadata));
    } else {
        let _ = write_public_error(
            stream,
            404,
            "model_not_found",
            "model not found",
            Some("model"),
        );
    }
}

fn handle_completion(
    stream: &mut TcpStream,
    backend: &dyn CompletionBackend,
    stopping: &AtomicBool,
    observations: &Arc<ObservationBuffer>,
    metrics: &Arc<ServiceMetrics>,
    config: ServerConfig,
    body: &[u8],
) {
    if stopping.load(Ordering::Acquire) {
        let _ = write_api_error(stream, &ApiError::ShuttingDown);
        return;
    }
    let Ok(request) = serde_json::from_slice::<CompletionRequest>(body) else {
        let _ = write_public_error(
            stream,
            400,
            "invalid_json",
            "request body must be valid completion JSON",
            None,
        );
        return;
    };
    let request = match normalize_completion_request(request, config.request_limits) {
        Ok(request) => request,
        Err(error) => {
            let _ = write_api_error(stream, &ApiError::InvalidRequest(error));
            return;
        }
    };
    let model = backend.model_metadata();
    if request.model_id != model.model_id {
        let _ = write_public_error(
            stream,
            404,
            "model_not_found",
            "model not found",
            Some("model"),
        );
        return;
    }
    let status = backend.status();
    if !status.accepting {
        let _ = write_api_error(stream, &ApiError::ShuttingDown);
        return;
    }
    if !status.ready {
        let _ = write_public_error(
            stream,
            503,
            "model_unavailable",
            "the model is not ready",
            None,
        );
        return;
    }
    if stream
        .set_read_timeout(Some(DISCONNECT_PEEK_TIMEOUT))
        .is_err()
    {
        return;
    }
    let streaming = request.stream;
    let admission_started = Instant::now();
    let submitted = match backend.submit(request) {
        Ok(submitted) => submitted,
        Err(class) => {
            if class == ServiceErrorClass::Overloaded {
                metrics.record_overload();
            }
            let _ = write_api_error(stream, &ApiError::from(class));
            return;
        }
    };
    let queue_wait = admission_started.elapsed();
    let tracker = RequestTracker::new(
        submitted.metadata().clone(),
        queue_wait,
        Arc::clone(observations),
        Arc::clone(metrics),
    );
    let deadline = Instant::now()
        .checked_add(config.request_timeout)
        .unwrap_or_else(Instant::now);
    if streaming {
        stream_completion(stream, submitted, backend, stopping, deadline, tracker);
    } else {
        collect_completion(
            stream,
            submitted,
            backend,
            stopping,
            deadline,
            config.maximum_non_streaming_bytes,
            tracker,
        );
    }
}

struct RequestTracker {
    metadata: RequestMetadata,
    queue_wait: Duration,
    admitted_at: Instant,
    time_to_first_token: Option<Duration>,
    delta_events: u64,
    observations: Arc<ObservationBuffer>,
    metrics: Arc<ServiceMetrics>,
}

impl RequestTracker {
    fn new(
        metadata: RequestMetadata,
        queue_wait: Duration,
        observations: Arc<ObservationBuffer>,
        metrics: Arc<ServiceMetrics>,
    ) -> Self {
        Self {
            metadata,
            queue_wait,
            admitted_at: Instant::now(),
            time_to_first_token: None,
            delta_events: 0,
            observations,
            metrics,
        }
    }

    fn token_delta(&mut self) {
        if self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(self.admitted_at.elapsed());
        }
        self.delta_events = self.delta_events.saturating_add(1);
    }

    fn finish(
        self,
        backend: &dyn CompletionBackend,
        status: RequestObservationStatus,
        finish_reason: Option<FinishReason>,
        error_class: Option<ServiceErrorClass>,
        tokens_generated: Option<u64>,
    ) {
        let backend_status = backend.status();
        self.metrics.record_terminal(status, error_class);
        self.observations.record(RequestObservation {
            request_id: self.metadata.request_id,
            model_id: self.metadata.model_id,
            queue_wait: self.queue_wait,
            time_to_first_token: self.time_to_first_token,
            tokens_generated: tokens_generated.unwrap_or(self.delta_events),
            finish_reason,
            status,
            error_class,
            active_requests: backend_status.active_requests,
            waiting_requests: backend_status.waiting_requests,
        });
    }
}

#[allow(clippy::too_many_lines)]
fn stream_completion(
    stream: &mut TcpStream,
    mut submitted: SubmittedRequest,
    backend: &dyn CompletionBackend,
    stopping: &AtomicBool,
    deadline: Instant,
    mut tracker: RequestTracker,
) {
    if write_sse_head(stream).is_err() {
        tracker.finish(
            backend,
            RequestObservationStatus::ClientDisconnected,
            None,
            Some(ServiceErrorClass::Cancelled),
            None,
        );
        return;
    }
    let mut encoder = SseStreamEncoder::new(submitted.metadata().clone());
    loop {
        let Some(wait) = event_wait(stopping, deadline) else {
            if client_disconnected(stream) {
                tracker.finish(
                    backend,
                    RequestObservationStatus::ClientDisconnected,
                    None,
                    Some(ServiceErrorClass::Cancelled),
                    None,
                );
                return;
            }
            let error = if stopping.load(Ordering::Acquire) {
                ApiError::ShuttingDown
            } else {
                ApiError::Timeout
            };
            let _ = write_stream_error(stream, &mut encoder, &error);
            tracker.finish(
                backend,
                RequestObservationStatus::Failed,
                None,
                Some(if matches!(error, ApiError::ShuttingDown) {
                    ServiceErrorClass::ShuttingDown
                } else {
                    ServiceErrorClass::Timeout
                }),
                None,
            );
            return;
        };
        match submitted.recv_timeout(wait) {
            Ok(_) if stopping.load(Ordering::Acquire) => {
                if client_disconnected(stream) {
                    tracker.finish(
                        backend,
                        RequestObservationStatus::ClientDisconnected,
                        None,
                        Some(ServiceErrorClass::Cancelled),
                        None,
                    );
                    return;
                }
                let delivered =
                    write_stream_error(stream, &mut encoder, &ApiError::ShuttingDown).is_ok();
                tracker.finish(
                    backend,
                    if delivered {
                        RequestObservationStatus::Failed
                    } else {
                        RequestObservationStatus::ClientDisconnected
                    },
                    None,
                    Some(if delivered {
                        ServiceErrorClass::ShuttingDown
                    } else {
                        ServiceErrorClass::Cancelled
                    }),
                    None,
                );
                return;
            }
            Ok(event) => {
                if matches!(event, GenerationEvent::TokenDelta { .. }) {
                    tracker.token_delta();
                }
                let terminal = !matches!(event, GenerationEvent::TokenDelta { .. });
                let (finish_reason, error_class, completion_tokens) = match &event {
                    GenerationEvent::TokenDelta { .. } => (None, None, None),
                    GenerationEvent::Finished { reason, usage } => (
                        Some(*reason),
                        match reason {
                            FinishReason::Cancelled => Some(ServiceErrorClass::Cancelled),
                            FinishReason::Error => Some(ServiceErrorClass::Internal),
                            FinishReason::Stop | FinishReason::Length => None,
                        },
                        Some(usage.completion_tokens()),
                    ),
                    GenerationEvent::Failed { class } => (None, Some(*class), None),
                };
                let Ok(frame) = encoder.encode_event(&event) else {
                    let _ = write_stream_error(stream, &mut encoder, &ApiError::Internal);
                    tracker.finish(
                        backend,
                        RequestObservationStatus::Failed,
                        finish_reason,
                        Some(ServiceErrorClass::Internal),
                        completion_tokens,
                    );
                    return;
                };
                if stream.write_all(frame.as_bytes()).is_err() {
                    tracker.finish(
                        backend,
                        RequestObservationStatus::ClientDisconnected,
                        finish_reason,
                        Some(ServiceErrorClass::Cancelled),
                        completion_tokens,
                    );
                    return;
                }
                if terminal {
                    let Ok(done) = encoder.encode_done() else {
                        tracker.finish(
                            backend,
                            RequestObservationStatus::Failed,
                            finish_reason,
                            Some(ServiceErrorClass::Internal),
                            completion_tokens,
                        );
                        return;
                    };
                    if stream.write_all(done.as_bytes()).is_err() || stream.flush().is_err() {
                        tracker.finish(
                            backend,
                            RequestObservationStatus::ClientDisconnected,
                            finish_reason,
                            Some(ServiceErrorClass::Cancelled),
                            completion_tokens,
                        );
                        return;
                    }
                    submitted.disarm();
                    tracker.finish(
                        backend,
                        if error_class.is_some() {
                            RequestObservationStatus::Failed
                        } else {
                            RequestObservationStatus::Completed
                        },
                        finish_reason,
                        error_class,
                        completion_tokens,
                    );
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if client_disconnected(stream) {
                    tracker.finish(
                        backend,
                        RequestObservationStatus::ClientDisconnected,
                        None,
                        Some(ServiceErrorClass::Cancelled),
                        None,
                    );
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = write_stream_error(stream, &mut encoder, &ApiError::Internal);
                tracker.finish(
                    backend,
                    RequestObservationStatus::Failed,
                    None,
                    Some(ServiceErrorClass::Internal),
                    None,
                );
                return;
            }
        }
    }
}

fn write_stream_error(
    stream: &mut TcpStream,
    encoder: &mut SseStreamEncoder,
    error: &ApiError,
) -> io::Result<()> {
    let frame = encoder
        .encode_error(error)
        .map_err(|_| io::Error::other("SSE error encoding failed"))?;
    stream.write_all(frame.as_bytes())?;
    let done = encoder
        .encode_done()
        .map_err(|_| io::Error::other("SSE terminal encoding failed"))?;
    stream.write_all(done.as_bytes())?;
    stream.flush()
}

#[allow(clippy::too_many_lines)]
fn collect_completion(
    stream: &mut TcpStream,
    mut submitted: SubmittedRequest,
    backend: &dyn CompletionBackend,
    stopping: &AtomicBool,
    deadline: Instant,
    maximum_bytes: usize,
    mut tracker: RequestTracker,
) {
    let mut text = String::new();
    loop {
        let Some(wait) = event_wait(stopping, deadline) else {
            if client_disconnected(stream) {
                tracker.finish(
                    backend,
                    RequestObservationStatus::ClientDisconnected,
                    None,
                    Some(ServiceErrorClass::Cancelled),
                    None,
                );
                return;
            }
            let error = if stopping.load(Ordering::Acquire) {
                ApiError::ShuttingDown
            } else {
                ApiError::Timeout
            };
            let _ = write_api_error(stream, &error);
            tracker.finish(
                backend,
                RequestObservationStatus::Failed,
                None,
                Some(if matches!(error, ApiError::ShuttingDown) {
                    ServiceErrorClass::ShuttingDown
                } else {
                    ServiceErrorClass::Timeout
                }),
                None,
            );
            return;
        };
        match submitted.recv_timeout(wait) {
            Ok(_) if stopping.load(Ordering::Acquire) => {
                if client_disconnected(stream) {
                    tracker.finish(
                        backend,
                        RequestObservationStatus::ClientDisconnected,
                        None,
                        Some(ServiceErrorClass::Cancelled),
                        None,
                    );
                    return;
                }
                let delivered = write_api_error(stream, &ApiError::ShuttingDown).is_ok();
                tracker.finish(
                    backend,
                    if delivered {
                        RequestObservationStatus::Failed
                    } else {
                        RequestObservationStatus::ClientDisconnected
                    },
                    None,
                    Some(if delivered {
                        ServiceErrorClass::ShuttingDown
                    } else {
                        ServiceErrorClass::Cancelled
                    }),
                    None,
                );
                return;
            }
            Ok(GenerationEvent::TokenDelta { text: delta }) => {
                tracker.token_delta();
                if tracker.delta_events % NON_STREAMING_PROBE_DELTA_INTERVAL == 0
                    && client_disconnected(stream)
                {
                    tracker.finish(
                        backend,
                        RequestObservationStatus::ClientDisconnected,
                        None,
                        Some(ServiceErrorClass::Cancelled),
                        None,
                    );
                    return;
                }
                let Some(new_length) = text.len().checked_add(delta.len()) else {
                    let _ = write_api_error(stream, &ApiError::Internal);
                    tracker.finish(
                        backend,
                        RequestObservationStatus::Failed,
                        None,
                        Some(ServiceErrorClass::Internal),
                        None,
                    );
                    return;
                };
                if new_length > maximum_bytes {
                    let _ = write_api_error(stream, &ApiError::Internal);
                    tracker.finish(
                        backend,
                        RequestObservationStatus::Failed,
                        None,
                        Some(ServiceErrorClass::Internal),
                        None,
                    );
                    return;
                }
                if text.try_reserve(delta.len()).is_err() {
                    let _ = write_api_error(stream, &ApiError::Internal);
                    tracker.finish(
                        backend,
                        RequestObservationStatus::Failed,
                        None,
                        Some(ServiceErrorClass::Internal),
                        None,
                    );
                    return;
                }
                text.push_str(&delta);
            }
            Ok(GenerationEvent::Finished { reason, usage }) => {
                let response =
                    match CompletionResponse::new(submitted.metadata(), text, reason, usage) {
                        Ok(response) => write_json(stream, 200, &response),
                        Err(_) => write_api_error(
                            stream,
                            &match reason {
                                FinishReason::Cancelled => ApiError::Cancelled,
                                FinishReason::Error | FinishReason::Stop | FinishReason::Length => {
                                    ApiError::Internal
                                }
                            },
                        ),
                    };
                if response.is_ok() {
                    submitted.disarm();
                }
                let delivered = response.is_ok();
                tracker.finish(
                    backend,
                    if delivered {
                        match reason {
                            FinishReason::Stop | FinishReason::Length => {
                                RequestObservationStatus::Completed
                            }
                            FinishReason::Cancelled | FinishReason::Error => {
                                RequestObservationStatus::Failed
                            }
                        }
                    } else {
                        RequestObservationStatus::ClientDisconnected
                    },
                    Some(reason),
                    match reason {
                        FinishReason::Cancelled => Some(ServiceErrorClass::Cancelled),
                        FinishReason::Error => Some(ServiceErrorClass::Internal),
                        FinishReason::Stop | FinishReason::Length => None,
                    },
                    Some(usage.completion_tokens()),
                );
                return;
            }
            Ok(GenerationEvent::Failed { class }) => {
                let delivered = write_api_error(stream, &ApiError::from(class)).is_ok();
                if delivered {
                    submitted.disarm();
                }
                tracker.finish(
                    backend,
                    if delivered {
                        RequestObservationStatus::Failed
                    } else {
                        RequestObservationStatus::ClientDisconnected
                    },
                    None,
                    Some(class),
                    None,
                );
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                if client_disconnected(stream) {
                    tracker.finish(
                        backend,
                        RequestObservationStatus::ClientDisconnected,
                        None,
                        Some(ServiceErrorClass::Cancelled),
                        None,
                    );
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = write_api_error(stream, &ApiError::Internal);
                tracker.finish(
                    backend,
                    RequestObservationStatus::Failed,
                    None,
                    Some(ServiceErrorClass::Internal),
                    None,
                );
                return;
            }
        }
    }
}

fn event_wait(stopping: &AtomicBool, deadline: Instant) -> Option<Duration> {
    if stopping.load(Ordering::Acquire) {
        return None;
    }
    let remaining = deadline.checked_duration_since(Instant::now())?;
    if remaining.is_zero() {
        None
    } else {
        Some(remaining.min(EVENT_POLL_INTERVAL))
    }
}

fn write_json<T: Serialize>(stream: &mut TcpStream, status: u16, value: &T) -> io::Result<()> {
    let body = serde_json::to_vec(value).map_err(|_| io::Error::other("JSON encoding failed"))?;
    write_response(stream, status, JSON_CONTENT_TYPE, &body)
}

fn write_api_error(stream: &mut TcpStream, error: &ApiError) -> io::Result<()> {
    write_json(stream, error.status_code(), &error.response())
}

fn write_public_error(
    stream: &mut TcpStream,
    status: u16,
    code: &str,
    message: &str,
    parameter: Option<&str>,
) -> io::Result<()> {
    write_json(
        stream,
        status,
        &ErrorResponse {
            error: ErrorObject {
                message: message.to_owned(),
                kind: if status < 500 {
                    "invalid_request_error".to_owned()
                } else {
                    "server_error".to_owned()
                },
                param: parameter.map(str::to_owned),
                code: code.to_owned(),
            },
        },
    )
}

fn write_http_read_error(stream: &mut TcpStream, error: &HttpReadError) -> io::Result<()> {
    let status = match error {
        HttpReadError::Io(source)
            if matches!(
                source.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ) =>
        {
            408
        }
        _ => error.status_code(),
    };
    let (code, message) = match status {
        405 => ("method_not_allowed", "the HTTP method is not supported"),
        408 => ("request_timeout", "request framing timed out"),
        411 => ("length_required", "Content-Length is required"),
        413 => ("request_too_large", "request body exceeds the server limit"),
        415 => (
            "unsupported_media_type",
            "Content-Type must be application/json",
        ),
        431 => (
            "headers_too_large",
            "request headers exceed the server limit",
        ),
        501 => (
            "unsupported_framing",
            "request transfer framing is unsupported",
        ),
        500 => ("internal_error", "the server encountered an internal error"),
        _ => ("malformed_request", "request framing is malformed"),
    };
    write_public_error(stream, status, code, message, None)
}

fn is_peer_gone(error: &HttpReadError) -> bool {
    match error {
        HttpReadError::UnexpectedEof { .. } => true,
        HttpReadError::Io(source) => matches!(
            source.kind(),
            io::ErrorKind::BrokenPipe
                | io::ErrorKind::ConnectionAborted
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::UnexpectedEof
        ),
        _ => false,
    }
}

fn client_disconnected(stream: &TcpStream) -> bool {
    let mut byte = [0_u8; 1];
    match stream.peek(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
            ) =>
        {
            false
        }
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{Shutdown, SocketAddr, TcpStream};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use serde_json::{Value, json};

    use super::{
        BackendStatus, CompletionBackend, ObservationBuffer, OperationalMetricsSnapshot,
        RequestCancellation, RequestObservation, RequestObservationStatus, ServerConfig,
        ServerShutdownError, SubmittedRequest, start_server,
    };
    use crate::domain::{
        FinishReason, GenerationEvent, GenerationRequest, ModelMetadata, RequestLimits,
        RequestMetadata, ServiceErrorClass, TokenUsage,
    };

    const TEST_TIMEOUT: Duration = Duration::from_secs(3);

    enum Script {
        Complete(Vec<String>),
        WaitForCancellation,
        CancelOnShutdown,
        FloodUntilDisconnected,
        Reject(ServiceErrorClass),
    }

    struct TestCancellation {
        cancelled: AtomicBool,
        cancellation_count: Arc<AtomicUsize>,
    }

    impl TestCancellation {
        fn new(cancellation_count: Arc<AtomicUsize>) -> Self {
            Self {
                cancelled: AtomicBool::new(false),
                cancellation_count,
            }
        }

        fn is_cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Acquire)
        }
    }

    impl RequestCancellation for TestCancellation {
        fn cancel(&self) {
            if !self.cancelled.swap(true, Ordering::AcqRel) {
                self.cancellation_count.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    struct TestBackend {
        model: ModelMetadata,
        scripts: Mutex<VecDeque<Script>>,
        ready: AtomicBool,
        accepting: AtomicBool,
        active: Arc<AtomicUsize>,
        waiting: AtomicUsize,
        next_id: AtomicU64,
        cancellation_count: Arc<AtomicUsize>,
        shutdown_signal: Arc<AtomicBool>,
        shutdown_called: AtomicBool,
        shutdown_delay: Mutex<Option<Duration>>,
    }

    impl TestBackend {
        fn new(scripts: impl IntoIterator<Item = Script>) -> Arc<Self> {
            Arc::new(Self {
                model: ModelMetadata {
                    model_id: "fixture-model".to_owned(),
                    created_unix_seconds: 42,
                    owned_by: "rustinfer-tests".to_owned(),
                    context_window_tokens: 4_096,
                    max_output_tokens: 1_024,
                },
                scripts: Mutex::new(scripts.into_iter().collect()),
                ready: AtomicBool::new(true),
                accepting: AtomicBool::new(true),
                active: Arc::new(AtomicUsize::new(0)),
                waiting: AtomicUsize::new(0),
                next_id: AtomicU64::new(1),
                cancellation_count: Arc::new(AtomicUsize::new(0)),
                shutdown_signal: Arc::new(AtomicBool::new(false)),
                shutdown_called: AtomicBool::new(false),
                shutdown_delay: Mutex::new(None),
            })
        }

        fn cancellations(&self) -> usize {
            self.cancellation_count.load(Ordering::Acquire)
        }

        fn delay_shutdown_by(&self, delay: Duration) {
            *self
                .shutdown_delay
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(delay);
        }
    }

    impl CompletionBackend for TestBackend {
        fn model_metadata(&self) -> ModelMetadata {
            self.model.clone()
        }

        fn status(&self) -> BackendStatus {
            BackendStatus {
                ready: self.ready.load(Ordering::Acquire),
                accepting: self.accepting.load(Ordering::Acquire),
                active_requests: self.active.load(Ordering::Acquire),
                waiting_requests: self.waiting.load(Ordering::Acquire),
            }
        }

        fn submit(
            &self,
            _request: GenerationRequest,
        ) -> Result<SubmittedRequest, ServiceErrorClass> {
            let script = self
                .scripts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_front()
                .unwrap_or_else(|| Script::Complete(vec!["default".to_owned()]));
            if let Script::Reject(class) = script {
                return Err(class);
            }
            if !self.accepting.load(Ordering::Acquire) {
                return Err(ServiceErrorClass::ShuttingDown);
            }

            let id = self.next_id.fetch_add(1, Ordering::AcqRel);
            let metadata = RequestMetadata::new(format!("cmpl-{id}"), &self.model.model_id, 42)
                .expect("fixture metadata");
            let (sender, receiver) = std::sync::mpsc::sync_channel(2);
            let cancellation =
                Arc::new(TestCancellation::new(Arc::clone(&self.cancellation_count)));
            let worker_cancellation = Arc::clone(&cancellation);
            let active = Arc::clone(&self.active);
            let shutdown_signal = Arc::clone(&self.shutdown_signal);
            active.fetch_add(1, Ordering::AcqRel);
            thread::spawn(move || {
                match script {
                    Script::Complete(deltas) => {
                        let completion_tokens =
                            u64::try_from(deltas.len()).expect("test token count fits u64");
                        for text in deltas {
                            if sender.send(GenerationEvent::TokenDelta { text }).is_err() {
                                active.fetch_sub(1, Ordering::AcqRel);
                                return;
                            }
                        }
                        let usage = TokenUsage::new(2, completion_tokens).expect("fixture usage");
                        let _ = sender.send(GenerationEvent::Finished {
                            reason: FinishReason::Stop,
                            usage,
                        });
                    }
                    Script::WaitForCancellation => {
                        while !worker_cancellation.is_cancelled() {
                            thread::sleep(Duration::from_millis(2));
                        }
                    }
                    Script::CancelOnShutdown => {
                        while !shutdown_signal.load(Ordering::Acquire) {
                            thread::sleep(Duration::from_millis(2));
                        }
                        let usage = TokenUsage::new(2, 0).expect("fixture usage");
                        let _ = sender.send(GenerationEvent::Finished {
                            reason: FinishReason::Cancelled,
                            usage,
                        });
                    }
                    Script::FloodUntilDisconnected => {
                        let text = "x".repeat(4 * 1_024);
                        while !worker_cancellation.is_cancelled() {
                            if sender
                                .send(GenerationEvent::TokenDelta { text: text.clone() })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                    Script::Reject(_) => unreachable!("rejections return before worker spawn"),
                }
                active.fetch_sub(1, Ordering::AcqRel);
            });
            let cancellation: Arc<dyn RequestCancellation> = cancellation;
            Ok(SubmittedRequest::new(metadata, receiver, cancellation))
        }

        fn begin_shutdown(&self) {
            self.accepting.store(false, Ordering::Release);
            self.shutdown_signal.store(true, Ordering::Release);
        }

        fn shutdown(&self, deadline: Instant) -> Result<(), ServiceErrorClass> {
            self.shutdown_called.store(true, Ordering::Release);
            if let Some(delay) = *self
                .shutdown_delay
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
            {
                thread::sleep(delay);
                return Ok(());
            }
            while self.active.load(Ordering::Acquire) != 0 {
                if Instant::now() >= deadline {
                    return Err(ServiceErrorClass::Timeout);
                }
                thread::sleep(Duration::from_millis(2));
            }
            Ok(())
        }
    }

    fn test_config() -> ServerConfig {
        ServerConfig {
            bind_address: SocketAddr::from(([127, 0, 0, 1], 0)),
            worker_threads: 4,
            connection_queue_capacity: 32,
            read_timeout: Duration::from_millis(500),
            framing_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_millis(500),
            request_timeout: Duration::from_secs(2),
            shutdown_grace: Duration::from_secs(2),
            http_limits: crate::http::HttpLimits::default(),
            request_limits: RequestLimits {
                max_prompt_bytes: 1_024,
                max_output_tokens: 64,
                ..RequestLimits::default()
            },
            maximum_non_streaming_bytes: 1_024 * 1_024,
            observation_capacity: 64,
        }
    }

    fn post_body(body: &[u8]) -> Vec<u8> {
        let mut request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        request.extend_from_slice(body);
        request
    }

    fn send_request(address: SocketAddr, request: &[u8]) -> Vec<u8> {
        let mut stream = TcpStream::connect(address).expect("connect loopback server");
        stream
            .set_read_timeout(Some(TEST_TIMEOUT))
            .expect("set client timeout");
        stream.write_all(request).expect("write request");
        let mut response = Vec::new();
        stream.read_to_end(&mut response).expect("read response");
        response
    }

    fn response_status(response: &[u8]) -> u16 {
        let head_end = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .expect("response head terminator");
        let head = std::str::from_utf8(&response[..head_end]).expect("ASCII response head");
        head.split_ascii_whitespace()
            .nth(1)
            .expect("response status")
            .parse()
            .expect("numeric response status")
    }

    fn response_body(response: &[u8]) -> &[u8] {
        let head_end = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .expect("response head terminator");
        &response[head_end + 4..]
    }

    #[track_caller]
    fn wait_until(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + TEST_TIMEOUT;
        while !predicate() {
            assert!(Instant::now() < deadline, "condition timed out");
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn observation_buffer_evicts_oldest_metadata_as_one_consistent_state() {
        let buffer = ObservationBuffer::try_new(1).expect("allocate observation buffer");
        for request_id in ["cmpl-old", "cmpl-new"] {
            buffer.record(RequestObservation {
                request_id: request_id.to_owned(),
                model_id: "fixture-model".to_owned(),
                queue_wait: Duration::ZERO,
                time_to_first_token: None,
                tokens_generated: 0,
                finish_reason: None,
                status: RequestObservationStatus::Failed,
                error_class: Some(ServiceErrorClass::Internal),
                active_requests: 0,
                waiting_requests: 0,
            });
        }
        let snapshot = buffer.snapshot();
        assert_eq!(snapshot.dropped_observations, 1);
        assert_eq!(snapshot.observations.len(), 1);
        assert_eq!(snapshot.observations[0].request_id, "cmpl-new");
    }

    #[test]
    fn streaming_and_non_streaming_text_match_and_are_observed() {
        let backend = TestBackend::new([
            Script::Complete(vec!["hello".to_owned(), " world".to_owned()]),
            Script::Complete(vec!["hello".to_owned(), " world".to_owned()]),
        ]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let server = start_server(test_config(), backend_trait).expect("start server");

        let non_stream = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "test",
            "stream": false
        }))
        .expect("encode request");
        let response = send_request(server.local_address(), &post_body(&non_stream));
        assert_eq!(response_status(&response), 200);
        let json: Value =
            serde_json::from_slice(response_body(&response)).expect("completion JSON");
        assert_eq!(json["choices"][0]["text"], "hello world");

        let stream = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "test",
            "stream": true
        }))
        .expect("encode request");
        let response = send_request(server.local_address(), &post_body(&stream));
        assert_eq!(response_status(&response), 200);
        let body = std::str::from_utf8(response_body(&response)).expect("UTF-8 SSE");
        assert!(body.ends_with("data: [DONE]\n\n"));
        let streamed_text: String = body
            .split("\n\n")
            .filter_map(|frame| frame.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .filter_map(|value| value["choices"][0]["text"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(streamed_text, json["choices"][0]["text"]);

        let snapshot = server.observations();
        assert_eq!(snapshot.observations.len(), 2);
        assert_eq!(snapshot.dropped_observations, 0);
        for observation in snapshot.observations {
            assert_eq!(observation.model_id, "fixture-model");
            assert!(observation.request_id.starts_with("cmpl-"));
            assert!(observation.time_to_first_token.is_some());
            assert_eq!(observation.tokens_generated, 2);
            assert_eq!(observation.finish_reason, Some(FinishReason::Stop));
            assert_eq!(observation.status, RequestObservationStatus::Completed);
            assert_eq!(observation.error_class, None);
        }
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn readiness_health_and_model_metadata_are_separate() {
        let backend = TestBackend::new([]);
        backend.ready.store(false, Ordering::Release);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let server = start_server(test_config(), backend_trait).expect("start server");

        let health = send_request(
            server.local_address(),
            b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&health), 200);
        let readiness = send_request(
            server.local_address(),
            b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&readiness), 503);

        backend.ready.store(true, Ordering::Release);
        let readiness = send_request(
            server.local_address(),
            b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&readiness), 200);
        let metrics = send_request(
            server.local_address(),
            b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&metrics), 200);
        let metrics_json: Value =
            serde_json::from_slice(response_body(&metrics)).expect("metrics JSON");
        assert_eq!(
            metrics_json
                .as_object()
                .expect("metrics object")
                .keys()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>(),
            [
                "active_requests",
                "allocation",
                "counters",
                "kv_allocated_blocks",
                "waiting_requests",
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(metrics_json["active_requests"], 0);
        assert_eq!(metrics_json["waiting_requests"], 0);
        assert_eq!(metrics_json["kv_allocated_blocks"], 0);
        assert_eq!(metrics_json["allocation"]["device_live_count"], 0);
        assert_eq!(metrics_json["allocation"]["device_live_bytes"], 0);
        assert_eq!(metrics_json["allocation"]["pinned_live_count"], 0);
        assert_eq!(metrics_json["allocation"]["pinned_live_bytes"], 0);
        assert_eq!(
            server.metrics_snapshot().expect("snapshot"),
            OperationalMetricsSnapshot::default()
        );
        let models = send_request(
            server.local_address(),
            b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&models), 200);
        let body: Value = serde_json::from_slice(response_body(&models)).expect("model JSON");
        assert_eq!(body["data"][0]["id"], "fixture-model");

        let model = send_request(
            server.local_address(),
            b"GET /v1/models/fixture-model HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&model), 200);
        let missing = send_request(
            server.local_address(),
            b"GET /v1/models/secret HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&missing), 404);
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn malformed_bounded_and_unavailable_requests_are_sanitized() {
        let backend = TestBackend::new([Script::Complete(vec!["12345".to_owned()])]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let mut config = test_config();
        config.request_limits.max_prompt_bytes = 4;
        config.http_limits.maximum_body_bytes = 256;
        config.maximum_non_streaming_bytes = 4;
        let server = start_server(config, backend_trait).expect("start server");

        let malformed = send_request(server.local_address(), &post_body(b"{not-json}"));
        assert_eq!(response_status(&malformed), 400);
        let malformed_body = std::str::from_utf8(response_body(&malformed)).expect("error UTF-8");
        assert!(malformed_body.contains("invalid_json"));
        assert!(!malformed_body.contains("line 1"));

        let unsupported_method = send_request(
            server.local_address(),
            b"PUT /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&unsupported_method), 405);
        assert!(
            std::str::from_utf8(response_body(&unsupported_method))
                .expect("error UTF-8")
                .contains("method_not_allowed")
        );

        let oversized_prompt = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "12345"
        }))
        .expect("encode request");
        let response = send_request(server.local_address(), &post_body(&oversized_prompt));
        assert_eq!(response_status(&response), 400);
        assert!(
            std::str::from_utf8(response_body(&response))
                .expect("error UTF-8")
                .contains("request_too_large")
        );

        let declared_too_large = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 257\r\n\r\n";
        let response = send_request(server.local_address(), declared_too_large);
        assert_eq!(response_status(&response), 413);

        let valid = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "1234"
        }))
        .expect("encode request");
        let response = send_request(server.local_address(), &post_body(&valid));
        assert_eq!(response_status(&response), 500);
        assert!(
            std::str::from_utf8(response_body(&response))
                .expect("error UTF-8")
                .contains("internal_error")
        );
        wait_until(|| backend.cancellations() == 1);

        backend.ready.store(false, Ordering::Release);
        let response = send_request(server.local_address(), &post_body(&valid));
        assert_eq!(response_status(&response), 503);
        assert!(
            std::str::from_utf8(response_body(&response))
                .expect("error UTF-8")
                .contains("model_unavailable")
        );
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn backend_and_connection_queue_overload_return_429() {
        let backend = TestBackend::new([
            Script::Reject(ServiceErrorClass::InvalidRequest),
            Script::Reject(ServiceErrorClass::Overloaded),
        ]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let mut config = test_config();
        config.worker_threads = 1;
        config.connection_queue_capacity = 1;
        config.read_timeout = Duration::from_millis(250);
        let server = start_server(config, backend_trait).expect("start server");

        let body = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "ok"
        }))
        .expect("encode request");
        let response = send_request(server.local_address(), &post_body(&body));
        assert_eq!(response_status(&response), 400);
        let public_error = std::str::from_utf8(response_body(&response)).expect("error UTF-8");
        assert!(public_error.contains("invalid_request"));
        assert!(!public_error.contains("internal"));

        let response = send_request(server.local_address(), &post_body(&body));
        assert_eq!(response_status(&response), 429);
        assert!(
            std::str::from_utf8(response_body(&response))
                .expect("error UTF-8")
                .contains("overloaded")
        );

        let first = TcpStream::connect(server.local_address()).expect("first slow connection");
        thread::sleep(Duration::from_millis(30));
        let mut second = TcpStream::connect(server.local_address()).expect("queued connection");
        second
            .write_all(b"GET /healthz HTTP/1.1\r\n")
            .expect("write partial queued request");
        thread::sleep(Duration::from_millis(30));
        let mut excess = (0..8)
            .map(|_| {
                let mut stream =
                    TcpStream::connect(server.local_address()).expect("excess connection");
                stream
                    .set_read_timeout(Some(TEST_TIMEOUT))
                    .expect("set timeout");
                stream.write_all(&post_body(&body)).expect("write request");
                stream
            })
            .collect::<Vec<_>>();
        let mut statuses = Vec::with_capacity(excess.len());
        for stream in &mut excess {
            let mut response = Vec::new();
            stream.read_to_end(&mut response).expect("read overload");
            statuses.push(response_status(&response));
        }
        assert!(statuses.contains(&429));
        assert!(
            server
                .metrics_snapshot()
                .expect("operational metrics")
                .counters
                .overloads
                >= 2
        );
        drop(first);
        drop(second);
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn absolute_framing_deadline_rejects_trickle_and_releases_worker() {
        let backend = TestBackend::new([]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let mut config = test_config();
        config.worker_threads = 1;
        config.connection_queue_capacity = 1;
        config.read_timeout = Duration::from_millis(500);
        config.framing_timeout = Duration::from_millis(100);
        let server = start_server(config, backend_trait).expect("start server");

        let mut trickle = TcpStream::connect(server.local_address()).expect("connect trickle");
        trickle
            .set_read_timeout(Some(TEST_TIMEOUT))
            .expect("set trickle timeout");
        // Each byte arrives well inside the 500 ms inactivity timeout, but
        // the request remains incomplete across the 100 ms absolute bound.
        for byte in b"GET /healthz HTTP/1.1\r\n".iter().take(7) {
            if trickle.write_all(std::slice::from_ref(byte)).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(15));
        }
        let _ = trickle.shutdown(Shutdown::Write);
        let mut response = Vec::new();
        if let Err(error) = trickle.read_to_end(&mut response) {
            assert_eq!(error.kind(), std::io::ErrorKind::ConnectionReset);
        }
        assert_eq!(response_status(&response), 408);

        let response = send_request(
            server.local_address(),
            b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(response_status(&response), 200);
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn shutdown_interrupts_partial_requests_within_global_grace() {
        let backend = TestBackend::new([]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let mut config = test_config();
        config.worker_threads = 1;
        config.connection_queue_capacity = 1;
        config.read_timeout = Duration::from_secs(30);
        config.framing_timeout = Duration::from_secs(30);
        config.shutdown_grace = Duration::from_millis(150);
        let server = start_server(config, backend_trait).expect("start server");

        let mut partial = TcpStream::connect(server.local_address()).expect("connect partial");
        partial
            .write_all(b"POST /v1/completions HTTP/1.1\r\n")
            .expect("write partial request");
        thread::sleep(Duration::from_millis(30));

        let started = Instant::now();
        server.shutdown().expect("bounded graceful shutdown");
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "shutdown exceeded its bounded grace: {:?}",
            started.elapsed()
        );
        assert!(backend.shutdown_called.load(Ordering::Acquire));
    }

    #[test]
    fn shutdown_deadline_contains_a_backend_that_violates_its_contract() {
        let backend = TestBackend::new([]);
        backend.delay_shutdown_by(Duration::from_secs(1));
        let backend_trait: Arc<dyn CompletionBackend> = backend;
        let mut config = test_config();
        config.shutdown_grace = Duration::from_millis(80);
        let server = start_server(config, backend_trait).expect("start server");

        let started = Instant::now();
        assert!(matches!(
            server.shutdown(),
            Err(ServerShutdownError::DeadlineExceeded {
                stage: "backend shutdown"
            })
        ));
        assert!(
            started.elapsed() < Duration::from_millis(400),
            "misbehaving backend escaped shutdown deadline: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn request_timeout_and_client_disconnect_cancel_backend_work() {
        let backend = TestBackend::new([
            Script::WaitForCancellation,
            Script::WaitForCancellation,
            Script::FloodUntilDisconnected,
            Script::FloodUntilDisconnected,
        ]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let mut config = test_config();
        config.request_timeout = Duration::from_millis(80);
        config.write_timeout = Duration::from_millis(100);
        config.maximum_non_streaming_bytes = 64 * 1_024 * 1_024;
        let server = start_server(config, backend_trait).expect("start server");

        let non_stream = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "ok"
        }))
        .expect("encode request");
        let response = send_request(server.local_address(), &post_body(&non_stream));
        assert_eq!(response_status(&response), 408);
        wait_until(|| backend.cancellations() >= 1);
        wait_until(|| backend.active.load(Ordering::Acquire) == 0);

        let mut before_first_event =
            TcpStream::connect(server.local_address()).expect("connect waiting request");
        before_first_event
            .write_all(&post_body(&non_stream))
            .expect("write waiting request");
        wait_until(|| backend.active.load(Ordering::Acquire) == 1);
        before_first_event
            .shutdown(Shutdown::Both)
            .expect("disconnect before first event");
        drop(before_first_event);
        wait_until(|| backend.cancellations() >= 2);
        wait_until(|| backend.active.load(Ordering::Acquire) == 0);

        // Aggregate mode has no response write during decode, so it probes
        // every 16 consumed deltas and cancels within that documented bound.
        let mut aggregate_decode =
            TcpStream::connect(server.local_address()).expect("connect aggregate decode");
        aggregate_decode
            .write_all(&post_body(&non_stream))
            .expect("write aggregate request");
        wait_until(|| backend.active.load(Ordering::Acquire) == 1);
        aggregate_decode
            .shutdown(Shutdown::Both)
            .expect("disconnect aggregate decode");
        drop(aggregate_decode);
        wait_until(|| backend.cancellations() >= 3);
        wait_until(|| backend.active.load(Ordering::Acquire) == 0);

        let stream_body = serde_json::to_vec(&json!({
            "model": "fixture-model",
            "prompt": "ok",
            "stream": true
        }))
        .expect("encode request");
        let mut client = TcpStream::connect(server.local_address()).expect("connect stream");
        client
            .write_all(&post_body(&stream_body))
            .expect("write stream request");
        let mut reader = BufReader::new(client.try_clone().expect("clone stream"));
        let mut line = String::new();
        loop {
            line.clear();
            reader.read_line(&mut line).expect("read SSE head");
            if line == "\r\n" {
                break;
            }
        }
        client.shutdown(Shutdown::Both).expect("disconnect client");
        drop(reader);
        drop(client);
        wait_until(|| backend.cancellations() >= 4);

        let observations = server.observations();
        assert!(observations.observations.iter().any(|observation| {
            observation.error_class == Some(ServiceErrorClass::Timeout)
                && observation.status == RequestObservationStatus::Failed
        }));
        assert!(observations.observations.iter().any(|observation| {
            observation.status == RequestObservationStatus::ClientDisconnected
        }));
        let metrics = server.metrics_snapshot().expect("operational metrics");
        assert!(metrics.counters.cancellations >= 4);
        assert!(metrics.counters.disconnects >= 3);
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn concurrent_clients_finish_without_deadlock() {
        const CLIENTS: usize = 24;
        let scripts = (0..CLIENTS)
            .map(|index| Script::Complete(vec![format!("answer-{index}")]))
            .collect::<Vec<_>>();
        let backend = TestBackend::new(scripts);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let mut config = test_config();
        config.connection_queue_capacity = CLIENTS;
        let server = start_server(config, backend_trait).expect("start server");
        let address = server.local_address();

        let clients = (0..CLIENTS)
            .map(|_| {
                thread::spawn(move || {
                    let body = serde_json::to_vec(&json!({
                        "model": "fixture-model",
                        "prompt": "parallel"
                    }))
                    .expect("encode request");
                    response_status(&send_request(address, &post_body(&body)))
                })
            })
            .collect::<Vec<_>>();
        for client in clients {
            assert_eq!(client.join().expect("client thread"), 200);
        }
        wait_until(|| server.observations().observations.len() == CLIENTS);
        assert_eq!(backend.cancellations(), 0);
        server.shutdown().expect("graceful shutdown");
    }

    #[test]
    fn graceful_shutdown_stops_admission_cancels_active_and_joins() {
        let backend = TestBackend::new([Script::CancelOnShutdown]);
        let backend_trait: Arc<dyn CompletionBackend> = backend.clone();
        let server = start_server(test_config(), backend_trait).expect("start server");
        let address = server.local_address();
        let client = thread::spawn(move || {
            let body = serde_json::to_vec(&json!({
                "model": "fixture-model",
                "prompt": "wait"
            }))
            .expect("encode request");
            send_request(address, &post_body(&body))
        });
        wait_until(|| backend.active.load(Ordering::Acquire) == 1);

        server.shutdown().expect("graceful shutdown");
        let response = client.join().expect("client thread");
        assert_eq!(response_status(&response), 503);
        assert!(backend.shutdown_called.load(Ordering::Acquire));
        assert_eq!(backend.cancellations(), 1);
        assert_eq!(backend.active.load(Ordering::Acquire), 0);
    }
}
