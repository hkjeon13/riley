//! CUDA Graph ABI vocabulary and fail-closed lifecycle policy.
//!
//! C05-4 owns real thread-local capture begin and one-shot abort/recovery while
//! keeping graph end, instantiate, and replay behind a later resource-lifetime
//! slice. CPU vocabulary/lifecycle validation still fails closed.

use std::marker::PhantomData;
#[cfg(any(feature = "cuda", test))]
use std::mem::{align_of, offset_of, size_of};
use std::num::NonZeroU64;
use std::rc::Rc;
#[cfg(feature = "cuda")]
use std::{cell::RefCell, sync::Arc};

#[cfg(feature = "cuda")]
use crate::runtime::{ContextInner, ensure_same_context};
use crate::{
    CudaDeviceBuffer, CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult,
    CudaStream,
};

// A native close deferred into a live capture still owns a RileyCudaContext
// child, but a foreign safe wrapper may otherwise drop the final Arc backing
// that native context before abort gets a chance to drain the close. Keep the
// corresponding Rust context leases in the capture thread until native abort
// has proved all deferred cleanup. ThreadLocal capture itself is !Send, so one
// thread-local ledger is enough and intentionally never crosses host threads.
#[cfg(feature = "cuda")]
struct DeferredCaptureContexts {
    active: bool,
    contexts: Vec<Arc<ContextInner>>,
}

#[cfg(feature = "cuda")]
impl DeferredCaptureContexts {
    const fn new() -> Self {
        Self {
            active: false,
            contexts: Vec::new(),
        }
    }
}

#[cfg(feature = "cuda")]
thread_local! {
    static DEFERRED_CAPTURE_CONTEXTS: RefCell<DeferredCaptureContexts> =
        const { RefCell::new(DeferredCaptureContexts::new()) };
}

#[cfg(feature = "cuda")]
fn begin_deferred_capture_contexts() {
    DEFERRED_CAPTURE_CONTEXTS.with(|ledger| {
        let mut ledger = ledger.borrow_mut();
        debug_assert!(
            !ledger.active,
            "native capture admission must reject a second safe capture on one thread"
        );
        ledger.active = true;
    });
}

/// Retains a safe context owner when its native child may hand its close to
/// the currently active capture. This is deliberately crate-private: public
/// resource wrappers call it from Drop before their native handle's Drop runs.
#[cfg(feature = "cuda")]
pub(crate) fn retain_context_for_active_graph_capture(context: &Arc<ContextInner>) -> bool {
    DEFERRED_CAPTURE_CONTEXTS.with(|ledger| {
        let mut ledger = ledger.borrow_mut();
        if !ledger.active {
            return false;
        }
        ledger.contexts.push(Arc::clone(context));
        true
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn has_active_graph_capture() -> bool {
    DEFERRED_CAPTURE_CONTEXTS.with(|ledger| ledger.borrow().active)
}

#[cfg(feature = "cuda")]
fn finish_deferred_capture_contexts() {
    let contexts = DEFERRED_CAPTURE_CONTEXTS.with(|ledger| {
        let mut ledger = ledger.borrow_mut();
        // Mark this false before Arc destruction: a ContextInner Drop can make
        // a native close call, and it must never append a new lease while the
        // successful native abort is dismantling this ledger.
        ledger.active = false;
        std::mem::take(&mut ledger.contexts)
    });
    // Drop after the RefCell borrow is gone. A final ContextInner destructor
    // may close its native context, which consults the same TLS capture state.
    drop(contexts);
}

/// The only CUDA Graph capture mode currently admitted by the Riley ABI.
///
/// CUDA's raw capture-mode numeric values are intentionally not exposed. More
/// permissive modes require a separate ownership and thread-safety review.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum CudaGraphCaptureMode {
    /// Confine capture invalidation and dependency tracking to the calling
    /// thread's capture domain.
    ThreadLocal = 1,
}

/// Per-operation CUDA Graph capture admission result.
///
/// `Unknown` is the default and is denied. This value does not claim that an
/// entire stream, context, or library is capture-capable; a future operation
/// descriptor must report it for the exact operation being admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum CudaGraphCaptureCapability {
    /// No reviewed capability result is available for this exact operation.
    Unknown = 0,
    /// The exact operation is known not to be capture-safe.
    Unsupported = 1,
    /// The exact operation is reviewed and admitted for graph capture.
    Supported = 2,
}

impl CudaGraphCaptureCapability {
    /// Whether this exact operation is allowed to enter graph capture.
    #[must_use]
    pub const fn admits_capture(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Rejects an unsupported or unreviewed capture operation before CUDA work.
    ///
    /// # Errors
    ///
    /// Returns a not-supported error unless this capability is `Supported`.
    pub fn require_capture_admission(self, operation: &'static str) -> CudaResult<()> {
        match self {
            Self::Supported => Ok(()),
            Self::Unsupported => Err(CudaError::new(
                CudaErrorKind::NotSupported,
                CudaErrorDomain::Rust,
                CudaErrorStage::Prepare,
                0,
                operation,
                "the operation is declared unsupported during CUDA Graph capture",
            )),
            Self::Unknown => Err(CudaError::new(
                CudaErrorKind::NotSupported,
                CudaErrorDomain::Rust,
                CudaErrorStage::Prepare,
                0,
                operation,
                "the operation's CUDA Graph capture capability is unknown and is denied",
            )),
        }
    }
}

/// Detailed phase for a graph-related outcome.
///
/// `Unknown` preserves a newer native ABI value conservatively rather than
/// treating it as a successful or reusable state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CudaGraphStage {
    /// Starting stream capture.
    CaptureBegin,
    /// Enqueuing a capture-admitted operation.
    CaptureEnqueue,
    /// Finishing stream capture.
    CaptureEnd,
    /// Aborting an invalidated or failed capture.
    CaptureAbort,
    /// Instantiating a captured graph.
    Instantiate,
    /// Updating a closed, supported graph-node descriptor.
    Update,
    /// Submitting a graph executable to a stream.
    Launch,
    /// Observing graph-launch completion.
    Completion,
    /// Closing a capture, graph, or graph executable.
    Close,
    /// A future ABI stage that this Rust wrapper does not yet understand.
    Unknown(u32),
}

/// Private Rust mirror of the fixed C ABI graph-failure companion record.
///
/// This contains no native handle and is never public API. The CUDA FFI
/// boundary supplies it immediately after a graph operation so the decoder
/// below can reject malformed companion metadata before a future graph owner
/// reasons about lifecycle evidence.
#[cfg(any(feature = "cuda", test))]
#[repr(C)]
pub(crate) struct RawGraphErrorInfo {
    struct_size: u32,
    graph_stage: u32,
    capture_id: u64,
    exec_id: u64,
    submission_started: u8,
    completion_known: u8,
    resource_release_known: u8,
    poisoned: u8,
    reserved0: u32,
    reserved: [u64; 3],
}

#[cfg(any(feature = "cuda", test))]
impl RawGraphErrorInfo {
    /// Required stable prefix size of the v1 companion record.
    pub(crate) const ABI_SIZE: u32 = 56;

    /// Creates an exact v1 zero-initialized companion output buffer.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            struct_size: Self::ABI_SIZE,
            graph_stage: 0,
            capture_id: 0,
            exec_id: 0,
            submission_started: 0,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 0,
            reserved0: 0,
            reserved: [0; 3],
        }
    }

    /// Returns the native-reported companion record size.
    #[must_use]
    pub(crate) const fn struct_size(&self) -> u32 {
        self.struct_size
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) const fn capture_abort_for_test(resource_release_known: u8, poisoned: u8) -> Self {
        Self {
            struct_size: Self::ABI_SIZE,
            graph_stage: 4,
            capture_id: 1,
            exec_id: 0,
            submission_started: 0,
            completion_known: 0,
            resource_release_known,
            poisoned,
            reserved0: 0,
            reserved: [0; 3],
        }
    }
}

#[cfg(any(feature = "cuda", test))]
const _: () = assert!(size_of::<RawGraphErrorInfo>() == RawGraphErrorInfo::ABI_SIZE as usize);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(align_of::<RawGraphErrorInfo>() == 8);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, struct_size) == 0);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, graph_stage) == 4);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, capture_id) == 8);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, exec_id) == 16);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, submission_started) == 24);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, completion_known) == 25);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, resource_release_known) == 26);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, poisoned) == 27);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, reserved0) == 28);
#[cfg(any(feature = "cuda", test))]
const _: () = assert!(offset_of!(RawGraphErrorInfo, reserved) == 32);

/// Public, decoded companion metadata for a graph operation outcome.
///
/// This information augments, rather than replaces, the stable generic
/// [`CudaError`](crate::CudaError). A graph action will expose an instance only
/// after the native graph ABI is linked; callers cannot manufacture IDs or
/// claim completion through this value type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaGraphFailureInfo {
    stage: Option<CudaGraphStage>,
    capture_id: Option<NonZeroU64>,
    exec_id: Option<NonZeroU64>,
    flags: u8,
}

impl CudaGraphFailureInfo {
    const SUBMISSION_STARTED: u8 = 1;
    const COMPLETION_KNOWN: u8 = 1 << 1;
    const RESOURCE_RELEASE_KNOWN: u8 = 1 << 2;
    const POISONED: u8 = 1 << 3;

    /// Detailed graph phase, or `None` when no graph phase was reached.
    #[must_use]
    pub const fn stage(&self) -> Option<CudaGraphStage> {
        self.stage
    }

    /// Capture identity assigned by the native owner, if any.
    #[must_use]
    pub const fn capture_id(&self) -> Option<NonZeroU64> {
        self.capture_id
    }

    /// Graph-exec identity assigned by the native owner, if any.
    #[must_use]
    pub const fn exec_id(&self) -> Option<NonZeroU64> {
        self.exec_id
    }

    /// Whether CUDA graph launch submission was attempted.
    #[must_use]
    pub const fn submission_started(&self) -> bool {
        self.flags & Self::SUBMISSION_STARTED != 0
    }

    /// Whether a completion boundary was observed unambiguously.
    #[must_use]
    pub const fn completion_known(&self) -> bool {
        self.flags & Self::COMPLETION_KNOWN != 0
    }

    /// Whether every transient graph-resource lease cleanup outcome is known.
    #[must_use]
    pub const fn resource_release_known(&self) -> bool {
        self.flags & Self::RESOURCE_RELEASE_KNOWN != 0
    }

    /// Whether the relevant graph, graph exec, or stream must not be reused.
    #[must_use]
    pub const fn poisoned(&self) -> bool {
        self.flags & Self::POISONED != 0
    }

    /// Whether this is the exact empty C05-1 capture-begin stub receipt.
    ///
    /// The generic decoder accepts a forward-compatible larger companion
    /// record. C05-1 deliberately remains stricter: its native stub must
    /// report the exact v1 record with capture-begin stage and no ownership or
    /// completion evidence.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty_capture_begin_attempt(&self) -> bool {
        matches!(self.stage, Some(CudaGraphStage::CaptureBegin))
            && self.capture_id.is_none()
            && self.exec_id.is_none()
            && self.flags == 0
    }
}

/// Decodes one graph-failure companion record from the native ABI.
///
/// The required v1 prefix is accepted from a future larger record, but every
/// reserved field and ABI boolean is checked exactly. Unknown graph stages are
/// retained as unknown rather than treated as successful lifecycle evidence.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn decode_graph_failure_info(
    raw: &RawGraphErrorInfo,
) -> CudaResult<CudaGraphFailureInfo> {
    if raw.struct_size < RawGraphErrorInfo::ABI_SIZE {
        return Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Validation,
            0,
            "CudaGraphFailureInfo::decode",
            "native graph error metadata is smaller than the required prefix",
        ));
    }
    if raw.reserved0 != 0 || raw.reserved != [0; 3] {
        return Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Validation,
            0,
            "CudaGraphFailureInfo::decode",
            "native graph error metadata has a non-zero reserved field",
        ));
    }

    let mut flags = 0;
    if decode_graph_abi_bool("submission_started", raw.submission_started)? {
        flags |= CudaGraphFailureInfo::SUBMISSION_STARTED;
    }
    if decode_graph_abi_bool("completion_known", raw.completion_known)? {
        flags |= CudaGraphFailureInfo::COMPLETION_KNOWN;
    }
    if decode_graph_abi_bool("resource_release_known", raw.resource_release_known)? {
        flags |= CudaGraphFailureInfo::RESOURCE_RELEASE_KNOWN;
    }
    if decode_graph_abi_bool("poisoned", raw.poisoned)? {
        flags |= CudaGraphFailureInfo::POISONED;
    }

    Ok(CudaGraphFailureInfo {
        stage: decode_graph_stage(raw.graph_stage),
        capture_id: NonZeroU64::new(raw.capture_id),
        exec_id: NonZeroU64::new(raw.exec_id),
        flags,
    })
}

#[cfg(any(feature = "cuda", test))]
const fn decode_graph_stage(value: u32) -> Option<CudaGraphStage> {
    match value {
        0 => None,
        1 => Some(CudaGraphStage::CaptureBegin),
        2 => Some(CudaGraphStage::CaptureEnqueue),
        3 => Some(CudaGraphStage::CaptureEnd),
        4 => Some(CudaGraphStage::CaptureAbort),
        5 => Some(CudaGraphStage::Instantiate),
        6 => Some(CudaGraphStage::Update),
        7 => Some(CudaGraphStage::Launch),
        8 => Some(CudaGraphStage::Completion),
        9 => Some(CudaGraphStage::Close),
        unknown => Some(CudaGraphStage::Unknown(unknown)),
    }
}

#[cfg(any(feature = "cuda", test))]
fn decode_graph_abi_bool(name: &'static str, value: u8) -> CudaResult<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Validation,
            0,
            "CudaGraphFailureInfo::decode",
            format!("native graph error metadata has invalid {name} boolean"),
        )),
    }
}

/// Pure lifecycle state used by the future native graph owners.
///
/// This value does not own a CUDA handle or submit any device work. It is a
/// CPU-side fail-closed guard for the native transition sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CudaGraphLifecycleState {
    /// No capture has begun and no native graph resource is owned.
    Uninitialized,
    /// Stream capture is active.
    Capturing,
    /// Capture ended successfully and a graph is ready to instantiate.
    Captured,
    /// A graph executable is ready for update or launch.
    Instantiated,
    /// Graph launch submission occurred and completion remains unresolved.
    Launching,
    /// All known graph resources were released.
    Closed,
    /// Completion or resource state is ambiguous; reuse and close are denied.
    Poisoned,
}

/// CPU-only lifecycle validator for a future native CUDA Graph owner.
///
/// The validator never calls CUDA. A later owner embeds it before every native
/// operation so an invalid transition is rejected before device mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaGraphLifecycle {
    state: CudaGraphLifecycleState,
}

impl Default for CudaGraphLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaGraphLifecycle {
    /// Starts with no native graph resource.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: CudaGraphLifecycleState::Uninitialized,
        }
    }

    /// Current pure lifecycle state.
    #[must_use]
    pub const fn state(self) -> CudaGraphLifecycleState {
        self.state
    }

    /// Records successful admission to stream capture.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless the lifecycle is uninitialized.
    pub fn begin_capture(&mut self) -> CudaResult<()> {
        self.transition(
            CudaGraphLifecycleState::Uninitialized,
            CudaGraphLifecycleState::Capturing,
            "CudaGraphCapture::begin",
        )
    }

    /// Records an operation admitted while capture is active.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless capture is active.
    pub fn enqueue_capture_operation(&mut self) -> CudaResult<()> {
        self.require_state(
            CudaGraphLifecycleState::Capturing,
            "CudaGraphCapture::enqueue_capture_operation",
        )
    }

    /// Records successful capture completion.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless capture is active.
    pub fn end_capture(&mut self) -> CudaResult<()> {
        self.transition(
            CudaGraphLifecycleState::Capturing,
            CudaGraphLifecycleState::Captured,
            "CudaGraphCapture::end",
        )
    }

    /// Aborts active capture and closes its owner after recovery is confirmed.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless capture is active.
    pub fn abort_capture(&mut self) -> CudaResult<()> {
        self.transition(
            CudaGraphLifecycleState::Capturing,
            CudaGraphLifecycleState::Closed,
            "CudaGraphCapture::abort",
        )
    }

    /// Records successful graph executable instantiation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless a captured graph is owned.
    pub fn instantiate(&mut self) -> CudaResult<()> {
        self.transition(
            CudaGraphLifecycleState::Captured,
            CudaGraphLifecycleState::Instantiated,
            "CapturedGraph::instantiate",
        )
    }

    /// Validates a closed, supported update against an instantiated exec.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless an executable is instantiated.
    pub fn update(&mut self) -> CudaResult<()> {
        self.require_state(CudaGraphLifecycleState::Instantiated, "GraphExec::update")
    }

    /// Records graph executable launch submission.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless an executable is instantiated.
    pub fn launch(&mut self) -> CudaResult<()> {
        self.transition(
            CudaGraphLifecycleState::Instantiated,
            CudaGraphLifecycleState::Launching,
            "GraphExec::launch",
        )
    }

    /// Records whether a launched graph's completion and transient lease
    /// release are known.
    ///
    /// An unknown completion or release poisons the lifecycle instead of
    /// permitting eager reuse or cleanup.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error unless a launch is in flight.
    pub fn observe_completion(
        &mut self,
        completion_known: bool,
        resource_release_known: bool,
    ) -> CudaResult<()> {
        self.require_state(CudaGraphLifecycleState::Launching, "GraphLaunch::complete")?;
        self.state = if completion_known && resource_release_known {
            CudaGraphLifecycleState::Instantiated
        } else {
            CudaGraphLifecycleState::Poisoned
        };
        Ok(())
    }

    /// Conservatively marks a non-closed lifecycle as non-reusable.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error after close.
    pub fn poison(&mut self) -> CudaResult<()> {
        if self.state == CudaGraphLifecycleState::Closed {
            return Err(self.invalid_transition("CudaGraphLifecycle::poison"));
        }
        self.state = CudaGraphLifecycleState::Poisoned;
        Ok(())
    }

    /// Releases only a lifecycle whose completion and resource state are known.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error for active capture, in-flight launch,
    /// poisoned state, or repeated close.
    pub fn close(&mut self) -> CudaResult<()> {
        match self.state {
            CudaGraphLifecycleState::Uninitialized
            | CudaGraphLifecycleState::Captured
            | CudaGraphLifecycleState::Instantiated => {
                self.state = CudaGraphLifecycleState::Closed;
                Ok(())
            }
            CudaGraphLifecycleState::Capturing
            | CudaGraphLifecycleState::Launching
            | CudaGraphLifecycleState::Closed
            | CudaGraphLifecycleState::Poisoned => {
                Err(self.invalid_transition("CudaGraphLifecycle::close"))
            }
        }
    }

    fn transition(
        &mut self,
        expected: CudaGraphLifecycleState,
        next: CudaGraphLifecycleState,
        operation: &'static str,
    ) -> CudaResult<()> {
        self.require_state(expected, operation)?;
        self.state = next;
        Ok(())
    }

    fn require_state(
        self,
        expected: CudaGraphLifecycleState,
        operation: &'static str,
    ) -> CudaResult<()> {
        if self.state == expected {
            Ok(())
        } else {
            Err(self.invalid_transition(operation))
        }
    }

    fn invalid_transition(self, operation: &'static str) -> CudaError {
        CudaError::invalid_state(
            operation,
            format!(
                "graph lifecycle transition is invalid while state is {:?}",
                self.state
            ),
        )
    }
}

/// Borrowed owner of one active thread-local CUDA Graph capture.
///
/// The owner holds both the native capture handle and the mutable stream
/// borrow. A live capture is deliberately `!Send + !Sync`: CUDA requires a
/// thread-local capture to end on its begin thread, and recovery must never be
/// detached from that owner. C05-4 exposes only [`Self::abort`]; graph end,
/// instantiate, and replay wait for C05-5's retained-resource contract.
///
/// ```compile_fail
/// fn cannot_query_stream_while_capturing(stream: &mut riley_cuda::CudaStream) {
///     let capture = stream
///         .begin_graph_capture(riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     let _ = stream.query();
///     drop(capture);
/// }
/// ```
///
/// ```compile_fail
/// fn cannot_synchronize_stream_while_capturing(stream: &mut riley_cuda::CudaStream) {
///     let capture = stream
///         .begin_graph_capture(riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     let _ = stream.synchronize();
///     drop(capture);
/// }
/// ```
///
/// ```compile_fail
/// fn cannot_begin_a_batch_while_capturing(stream: &mut riley_cuda::CudaStream) {
///     let capture = stream
///         .begin_graph_capture(riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     let _ = stream.begin_command_batch();
///     drop(capture);
/// }
/// ```
///
/// ```compile_fail
/// fn cannot_close_stream_while_capturing(stream: &mut riley_cuda::CudaStream) {
///     let capture = stream
///         .begin_graph_capture(riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     let _ = stream.close();
///     drop(capture);
/// }
/// ```
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::GraphCapture<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::GraphCapture<'static>>();
/// ```
pub struct GraphCapture<'stream> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    stream: &'stream mut CudaStream,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl GraphCapture<'_> {
    /// Terminates capture, discards its graph, and restores the stream only
    /// when native recovery is fully known.
    ///
    /// The owner is marked consumed only after native reports that it consumed
    /// the in/out capture pointer. A documented pre-CUDA validation rejection
    /// leaves the owner active, so Drop can still perform its normal abort
    /// recovery; a CUDA end attempt is never retried.
    ///
    /// # Errors
    ///
    /// Returns a native close/recovery error. On an ambiguous native outcome
    /// the stream is intentionally retained busy rather than reused.
    pub fn abort(mut self) -> CudaResult<()> {
        self.abort_once()
    }

    fn abort_once(&mut self) -> CudaResult<()> {
        if !self.active {
            return Ok(());
        }
        // Keep the concrete mutable borrow observable in both feature modes;
        // its lifetime is the safe stream lease even though native abort owns
        // the actual CUDA transition.
        let _ = &mut *self.stream;
        #[cfg(feature = "cuda")]
        {
            let transition = self.native.abort_with_transition();
            if transition.resource_release_known {
                // A valid native companion can prove that abort ended capture,
                // destroyed its transient graph, drained every deferred close,
                // and released the exact native owner even when CUDA reports a
                // deferred non-success status. Only then may the Rust context
                // leases used by those deferred native children be dropped.
                finish_deferred_capture_contexts();
            }
            self.active = !transition.owner_consumed;
            transition.result
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable("GraphCapture::abort"))
        }
    }
}

impl Drop for GraphCapture<'_> {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// Borrowed owner of the C05-5 fixed-address f32 fill capture.
///
/// This is deliberately separate from [`GraphCapture`]: it is the only graph
/// capture entry point that admits a CUDA operation. It retains exclusive
/// mutable borrows of both the capture stream and one preallocated device
/// buffer through graph instantiation and every replay, so safe Rust cannot
/// close, reuse, or move either resource while native graph ownership relies
/// on their addresses.
///
/// ```compile_fail
/// fn cannot_use_the_capture_stream(
///     stream: &mut riley_cuda::CudaStream,
///     buffer: &mut riley_cuda::CudaDeviceBuffer,
/// ) {
///     let capture = stream
///         .begin_graph_fill_capture(buffer, 1, riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     let _ = stream.query();
///     drop(capture);
/// }
/// ```
///
/// ```compile_fail
/// fn cannot_close_the_capture_buffer(
///     stream: &mut riley_cuda::CudaStream,
///     buffer: &mut riley_cuda::CudaDeviceBuffer,
/// ) {
///     let capture = stream
///         .begin_graph_fill_capture(buffer, 1, riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     buffer.close().unwrap();
///     drop(capture);
/// }
/// ```
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::GraphFillCapture<'static, 'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::GraphFillCapture<'static, 'static>>();
/// ```
pub struct GraphFillCapture<'stream, 'buffer> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    stream: Option<&'stream mut CudaStream>,
    buffer: Option<&'buffer mut CudaDeviceBuffer>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl<'stream, 'buffer> GraphFillCapture<'stream, 'buffer> {
    /// Enqueues one fixed-address f32 fill that was admitted when this capture
    /// began. Each value becomes an immutable graph-node parameter; C05-5
    /// intentionally exposes no dynamic update API.
    ///
    /// # Errors
    ///
    /// Multiple successful enqueue calls are allowed. A native enqueue failure
    /// keeps this capture owned so Drop can perform the single abort/recovery
    /// attempt rather than allowing a retry after an ambiguous CUDA outcome.
    pub fn enqueue_fill(&mut self, value: f32) -> CudaResult<()> {
        if !self.active {
            return Err(CudaError::invalid_state(
                "GraphFillCapture::enqueue_fill",
                "the graph fill capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                "GraphFillCapture::enqueue_fill",
                "a prior graph fill enqueue failed and this capture must be aborted",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_fill(value);
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = value;
            Err(CudaError::unavailable("GraphFillCapture::enqueue_fill"))
        }
    }

    /// Ends capture and returns the captured graph while retaining the same
    /// stream and buffer borrows for the graph's full lifetime.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error without ending capture if no fill was
    /// enqueued. Native end failures consume/poison the one-shot owner; only
    /// proven native resource release clears deferred Rust context leases.
    pub fn end(mut self) -> CudaResult<CapturedGraph<'stream, 'buffer>> {
        if !self.active {
            return Err(CudaError::invalid_state(
                "GraphFillCapture::end",
                "the graph fill capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                "GraphFillCapture::end",
                "a prior graph fill enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                "GraphFillCapture::end",
                "capture end requires at least one successful fixed-fill enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let transition = self.native.end();
            if transition.resource_release_known {
                finish_deferred_capture_contexts();
            }
            self.active = !transition.owner_consumed;
            let native = transition.result?;
            let stream = self.stream.take().ok_or_else(|| {
                CudaError::invalid_state(
                    "GraphFillCapture::end",
                    "the graph fill capture lost its stream borrow",
                )
            })?;
            let buffer = self.buffer.take().ok_or_else(|| {
                CudaError::invalid_state(
                    "GraphFillCapture::end",
                    "the graph fill capture lost its device-buffer borrow",
                )
            })?;
            Ok(CapturedGraph {
                native,
                stream: Some(stream),
                buffer: Some(buffer),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Keep the retained borrows structurally live in the CPU-only
            // build as well: this public type must have the same ownership
            // shape regardless of whether its native implementation is
            // linked.
            let _ = (&mut self.stream, &mut self.buffer);
            self.active = false;
            Err(CudaError::unavailable("GraphFillCapture::end"))
        }
    }

    /// Ends and discards this capture through the same one-shot recovery path
    /// used by Drop.
    ///
    /// # Errors
    ///
    /// Returns a native recovery error. A valid companion record can still
    /// release deferred Rust context leases even when CUDA reports an earlier
    /// deferred non-success status.
    pub fn abort(mut self) -> CudaResult<()> {
        self.abort_once()
    }

    fn abort_once(&mut self) -> CudaResult<()> {
        if !self.active {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            let transition = self.native.abort_with_transition();
            if transition.resource_release_known {
                finish_deferred_capture_contexts();
            }
            self.active = !transition.owner_consumed;
            transition.result
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable("GraphFillCapture::abort"))
        }
    }
}

impl Drop for GraphFillCapture<'_, '_> {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// Captured, uninstantiated f32-fill CUDA Graph.
///
/// The graph still retains its capture stream and fixed output buffer. Calling
/// [`Self::instantiate`] transfers those exact borrows into [`GraphExec`];
/// calling [`Self::close`] instead destroys the graph and returns the borrows
/// when this value is dropped.
///
/// ```compile_fail
/// fn cannot_reuse_resources_while_graph_is_live(
///     stream: &mut riley_cuda::CudaStream,
///     buffer: &mut riley_cuda::CudaDeviceBuffer,
/// ) {
///     let mut capture = stream
///         .begin_graph_fill_capture(buffer, 1, riley_cuda::CudaGraphCaptureMode::ThreadLocal)
///         .unwrap();
///     capture.enqueue_fill(1.0).unwrap();
///     let graph = capture.end().unwrap();
///     let _ = stream.query();
///     drop(graph);
/// }
/// ```
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::CapturedGraph<'static, 'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::CapturedGraph<'static, 'static>>();
/// ```
pub struct CapturedGraph<'stream, 'buffer> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    stream: Option<&'stream mut CudaStream>,
    buffer: Option<&'buffer mut CudaDeviceBuffer>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl<'stream, 'buffer> CapturedGraph<'stream, 'buffer> {
    /// Instantiates this captured graph exactly once.
    ///
    /// # Errors
    ///
    /// A native instantiate ambiguity consumes this safe owner and intentionally
    /// leaves its native resource leases fail-closed instead of retrying.
    pub fn instantiate(mut self) -> CudaResult<GraphExec<'stream, 'buffer>> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let stream = self.stream.take().ok_or_else(|| {
                CudaError::invalid_state(
                    "CapturedGraph::instantiate",
                    "the captured graph lost its stream borrow",
                )
            })?;
            let buffer = self.buffer.take().ok_or_else(|| {
                CudaError::invalid_state(
                    "CapturedGraph::instantiate",
                    "the captured graph lost its device-buffer borrow",
                )
            })?;
            Ok(GraphExec {
                native,
                stream: Some(stream),
                buffer: Some(buffer),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            Err(CudaError::unavailable("CapturedGraph::instantiate"))
        }
    }

    /// Explicitly destroys this captured graph.
    ///
    /// # Errors
    ///
    /// A close ambiguity is deliberately not retried; native retains the
    /// graph's resource leases fail-closed.
    pub fn close(mut self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            Err(CudaError::unavailable("CapturedGraph::close"))
        }
    }
}

/// Instantiated fixed-fill CUDA Graph executable.
///
/// It retains its exact capture stream and output buffer. Only
/// [`Self::launch`] may use that stream until this executable is explicitly
/// closed or dropped.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::GraphExec<'static, 'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::GraphExec<'static, 'static>>();
/// ```
pub struct GraphExec<'stream, 'buffer> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    stream: Option<&'stream mut CudaStream>,
    buffer: Option<&'buffer mut CudaDeviceBuffer>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl<'stream, 'buffer> GraphExec<'stream, 'buffer> {
    /// Launches one replay on the exact stream retained by capture.
    ///
    /// The graph executable itself keeps the stream and fixed buffer borrowed,
    /// so there is deliberately no stream argument through which a foreign
    /// same-context stream could be substituted.
    ///
    /// # Errors
    ///
    /// Returns a native launch error. An ambiguous native launch retains its
    /// native completion owner and graph resource leases fail-closed.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<GraphLaunch<'exec, 'stream, 'buffer>> {
        #[cfg(feature = "cuda")]
        {
            if self.buffer.is_none() {
                return Err(CudaError::invalid_state(
                    "GraphExec::launch",
                    "the graph exec lost its fixed device-buffer borrow",
                ));
            }
            let native = {
                let stream = self.stream.as_deref_mut().ok_or_else(|| {
                    CudaError::invalid_state(
                        "GraphExec::launch",
                        "the graph exec lost its capture-stream borrow",
                    )
                })?;
                self.native.launch(&mut stream.native)?
            };
            Ok(GraphLaunch {
                native,
                exec: self,
                active: true,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("GraphExec::launch"))
        }
    }

    /// Explicitly destroys this executable after every launch has completed.
    ///
    /// # Errors
    ///
    /// The Rust launch borrow prevents this call while a [`GraphLaunch`] is
    /// live. Native independently rejects raw-ABI close while launch state is
    /// in flight or poisoned.
    pub fn close(mut self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            Err(CudaError::unavailable("GraphExec::close"))
        }
    }
}

/// Borrowed completion owner for one graph replay.
///
/// ```compile_fail
/// fn cannot_close_or_relaunch_an_exec(
///     exec: &mut riley_cuda::GraphExec<'_, '_>,
/// ) {
///     let launch = exec.launch().unwrap();
///     exec.close().unwrap();
///     drop(launch);
/// }
/// ```
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::GraphLaunch<'static, 'static, 'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::GraphLaunch<'static, 'static, 'static>>();
/// ```
pub struct GraphLaunch<'exec, 'stream, 'buffer> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut GraphExec<'stream, 'buffer>,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl GraphLaunch<'_, '_, '_> {
    /// Waits for replay completion exactly once.
    ///
    /// # Errors
    ///
    /// An ambiguous completion never retries CUDA synchronization. Native
    /// retains the graph exec and its resource leases fail-closed.
    pub fn finish(mut self) -> CudaResult<()> {
        self.complete_once()
    }

    fn complete_once(&mut self) -> CudaResult<()> {
        // This is a deliberate exclusive reborrow: GraphLaunch holds it for
        // its entire lifetime, preventing exec launch/close/reuse until the
        // completion owner is consumed or dropped.
        let _ = &mut *self.exec;
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            self.native.complete()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("GraphLaunch::finish"))
        }
    }
}

impl Drop for GraphLaunch<'_, '_, '_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

/// A by-value stream and fixed device-buffer pair recovered from a known graph
/// close.
///
/// This bundle is deliberately not a graph capability. It merely returns the
/// exact resources that an [`OwnedGraphExec`] or [`OwnedCapturedGraph`] held
/// while native graph leases kept them busy. Call [`Self::into_parts`] to use
/// them normally, or [`Self::close`] when the cold resources are no longer
/// needed.
pub struct OwnedGraphFillResources {
    stream: CudaStream,
    buffer: CudaDeviceBuffer,
}

impl OwnedGraphFillResources {
    fn new(stream: CudaStream, buffer: CudaDeviceBuffer) -> Self {
        Self { stream, buffer }
    }

    /// Returns the exact stream and device buffer after a known graph release.
    #[must_use]
    pub fn into_parts(self) -> (CudaStream, CudaDeviceBuffer) {
        let Self { stream, buffer } = self;
        (stream, buffer)
    }

    /// Explicitly destroys both recovered cold resources.
    ///
    /// The device buffer closes before its stream, matching the ordinary
    /// caller-owned cleanup order. If either native close is ambiguous, its
    /// existing one-shot fail-closed contract remains in force.
    ///
    /// # Errors
    ///
    /// Returns a resource-close error. Use [`Self::into_parts`] when callers
    /// need to observe each resource close independently.
    pub fn close(self) -> CudaResult<()> {
        let (stream, buffer) = self.into_parts();
        buffer.close()?;
        stream.close()
    }
}

/// Error from beginning a by-value fixed-fill graph capture.
///
/// Rust-side preflight failures return the untouched stream/buffer pair through
/// [`Self::into_resources`]. Once the native capture entry point has been
/// called, this error deliberately withholds the pair: native may have entered
/// capture or retained its raw leases while recovering a deferred CUDA error.
/// In that case the values are dropped into the existing fail-closed native
/// ownership path rather than being made reusable without evidence.
#[must_use]
pub struct OwnedGraphFillCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphFillResources>,
}

impl OwnedGraphFillCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphFillResources) -> Self {
        Self {
            error,
            resources: Some(resources),
        }
    }

    #[cfg(feature = "cuda")]
    fn terminal(error: CudaError) -> Self {
        Self {
            error,
            resources: None,
        }
    }

    /// The original CUDA validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched values only when capture had not entered native
    /// ownership.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphFillResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphFillCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphFillCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphFillCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph fill capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphFillCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address f32 CUDA Graph capture.
///
/// Unlike [`GraphFillCapture`], this state moves the exact stream and device
/// buffer into the graph owner. It therefore has no self-referential borrow and
/// can move into a cold, same-thread graph holder after capture. It remains
/// `!Send + !Sync`: a thread-local capture must end on its begin thread.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphFillCapture>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphFillCapture>();
/// ```
pub struct OwnedGraphFillCapture {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    // Native must drop before these raw-resource wrappers. An ambiguous native
    // capture owns their raw addresses and must keep them fail-closed.
    stream: Option<CudaStream>,
    buffer: Option<CudaDeviceBuffer>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphFillCapture {
    /// Enqueues one immutable f32 fill node on this by-value capture.
    ///
    /// # Errors
    ///
    /// A native enqueue failure makes this capture terminal; Drop performs the
    /// same one-shot abort/recovery path as the borrowed C05-5 owner.
    pub fn enqueue_fill(&mut self, value: f32) -> CudaResult<()> {
        if !self.active {
            return Err(CudaError::invalid_state(
                "OwnedGraphFillCapture::enqueue_fill",
                "the owned graph fill capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                "OwnedGraphFillCapture::enqueue_fill",
                "a prior graph fill enqueue failed and this capture must be aborted",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_fill(value);
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = value;
            Err(CudaError::unavailable(
                "OwnedGraphFillCapture::enqueue_fill",
            ))
        }
    }

    /// Ends capture and transfers the moved resources into an owned graph.
    ///
    /// # Errors
    ///
    /// The consuming error path never returns raw resources after a native end
    /// attempt. Native graph/capture recovery retains any uncertain leases
    /// fail-closed, exactly as C05-5's borrowed owner does.
    pub fn end(mut self) -> CudaResult<OwnedCapturedGraph> {
        if !self.active {
            return Err(CudaError::invalid_state(
                "OwnedGraphFillCapture::end",
                "the owned graph fill capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                "OwnedGraphFillCapture::end",
                "a prior graph fill enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                "OwnedGraphFillCapture::end",
                "capture end requires at least one successful fixed-fill enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let transition = self.native.end();
            if transition.resource_release_known {
                finish_deferred_capture_contexts();
            }
            self.active = !transition.owner_consumed;
            let native = transition.result?;
            let resources = take_owned_graph_fill_resources(
                &mut self.stream,
                &mut self.buffer,
                "OwnedGraphFillCapture::end",
            )?;
            let (stream, buffer) = resources.into_parts();
            Ok(OwnedCapturedGraph {
                native,
                stream: Some(stream),
                buffer: Some(buffer),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            self.active = false;
            Err(CudaError::unavailable("OwnedGraphFillCapture::end"))
        }
    }

    /// Aborts capture and returns the moved resources only after native
    /// recovery has proved their graph leases released.
    ///
    /// # Errors
    ///
    /// An error retains no reusable resources. Any uncertain native owner is
    /// deliberately left in its existing fail-closed state.
    pub fn abort(mut self) -> CudaResult<OwnedGraphFillResources> {
        self.abort_once()?;
        take_owned_graph_fill_resources(
            &mut self.stream,
            &mut self.buffer,
            "OwnedGraphFillCapture::abort",
        )
    }

    fn abort_once(&mut self) -> CudaResult<()> {
        if !self.active {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            let transition = self.native.abort_with_transition();
            if transition.resource_release_known {
                finish_deferred_capture_contexts();
            }
            self.active = !transition.owner_consumed;
            transition.result
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable("OwnedGraphFillCapture::abort"))
        }
    }
}

impl Drop for OwnedGraphFillCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value captured f32-fill CUDA Graph awaiting instantiate or close.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedCapturedGraph>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedCapturedGraph>();
/// ```
pub struct OwnedCapturedGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    stream: Option<CudaStream>,
    buffer: Option<CudaDeviceBuffer>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedGraph {
    /// Instantiates this graph exactly once while retaining its moved
    /// stream/buffer pair.
    ///
    /// # Errors
    ///
    /// An instantiate error is terminal for this consuming owner; no resource
    /// pair is returned without native release evidence.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_fill_resources(
                &mut self.stream,
                &mut self.buffer,
                "OwnedCapturedGraph::instantiate",
            )?;
            let (stream, buffer) = resources.into_parts();
            Ok(OwnedGraphExec {
                native,
                stream: Some(stream),
                buffer: Some(buffer),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            Err(CudaError::unavailable("OwnedCapturedGraph::instantiate"))
        }
    }

    /// Destroys this graph and returns its resources only after the native
    /// close companion proves graph leases were released.
    ///
    /// # Errors
    ///
    /// A native close error consumes the value without returning resources;
    /// raw ownership remains fail-closed rather than becoming reusable.
    pub fn close(mut self) -> CudaResult<OwnedGraphFillResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_fill_resources(
                &mut self.stream,
                &mut self.buffer,
                "OwnedCapturedGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            Err(CudaError::unavailable("OwnedCapturedGraph::close"))
        }
    }
}

/// By-value instantiated fixed-fill CUDA Graph executable.
///
/// It can move into a cold same-thread owner because it stores, rather than
/// borrows, its capture stream and fixed device buffer. Neither is exposed
/// until [`Self::close`] has native resource-release evidence.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphExec>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphExec>();
/// ```
pub struct OwnedGraphExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    stream: Option<CudaStream>,
    buffer: Option<CudaDeviceBuffer>,
    // An attempted launch/completion that returns an error cannot prove this
    // native exec remains safe to close and hand resources back to Rust.
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphExec {
    /// Launches one replay on the exact stream moved into this executable.
    ///
    /// The returned completion owner holds an exclusive mutable borrow of this
    /// value, preventing relaunch or close until the one replay settles.
    ///
    /// # Errors
    ///
    /// Any launch error makes this by-value owner terminal. C05-6 deliberately
    /// prefers retaining native leases over returning resources from a
    /// malformed or deferred launch outcome.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphLaunch<'exec>> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphExec::launch",
                "an earlier owned graph replay left native completion state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            if self.buffer.is_none() {
                self.terminal = true;
                return Err(CudaError::invalid_state(
                    "OwnedGraphExec::launch",
                    "the owned graph exec lost its fixed device buffer",
                ));
            }
            let native = match self.stream.as_mut() {
                Some(stream) => self.native.launch(&mut stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        "OwnedGraphExec::launch",
                        "the owned graph exec lost its capture stream",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphLaunch {
                    native,
                    exec: self,
                    active: true,
                    _not_send_or_sync: PhantomData,
                }),
                Err(error) => {
                    self.terminal = true;
                    Err(error)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.terminal = true;
            Err(CudaError::unavailable("OwnedGraphExec::launch"))
        }
    }

    /// Destroys this executable and returns the captured resources only after
    /// native close proves all graph leases released.
    ///
    /// ```compile_fail
    /// fn cannot_close_an_owned_exec_while_launch_is_live(
    ///     mut exec: riley_cuda::OwnedGraphExec,
    /// ) {
    ///     let launch = exec.launch().unwrap();
    ///     let _ = exec.close();
    ///     drop(launch);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// The consuming error path returns no resources. This preserves native's
    /// one-shot destroy semantics when CUDA destruction or context restoration
    /// may be ambiguous.
    pub fn close(mut self) -> CudaResult<OwnedGraphFillResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphExec::close",
                "an earlier owned graph replay left native completion state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_fill_resources(
                &mut self.stream,
                &mut self.buffer,
                "OwnedGraphExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.buffer);
            Err(CudaError::unavailable("OwnedGraphExec::close"))
        }
    }
}

/// Completion owner for one replay of an [`OwnedGraphExec`].
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphLaunch<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphLaunch<'static>>();
/// ```
pub struct OwnedGraphLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphExec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphLaunch<'_> {
    /// Waits for replay completion exactly once.
    ///
    /// # Errors
    ///
    /// An ambiguous completion makes the retained executable terminal and
    /// never retries CUDA synchronization.
    pub fn finish(mut self) -> CudaResult<()> {
        self.complete_once()
    }

    fn complete_once(&mut self) -> CudaResult<()> {
        let _ = &mut *self.exec;
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            let result = self.native.complete();
            if result.is_err() {
                self.exec.terminal = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.exec.terminal = true;
            Err(CudaError::unavailable("OwnedGraphLaunch::finish"))
        }
    }
}

impl Drop for OwnedGraphLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_fill_resources(
    stream: &mut Option<CudaStream>,
    buffer: &mut Option<CudaDeviceBuffer>,
    operation: &'static str,
) -> CudaResult<OwnedGraphFillResources> {
    let stream = stream.take().ok_or_else(|| {
        CudaError::invalid_state(operation, "the owned graph owner lost its capture stream")
    })?;
    let buffer = buffer.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph owner lost its fixed device buffer",
        )
    })?;
    Ok(OwnedGraphFillResources::new(stream, buffer))
}

#[cfg(feature = "cuda")]
fn validate_graph_fill_capture_preflight(
    stream: &CudaStream,
    buffer: &CudaDeviceBuffer,
    element_count: u64,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, buffer.context_owner(), operation)?;
    buffer.ensure_idle_for_operation(operation)?;
    let required_bytes = element_count
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "element_count overflows the fixed f32 capture byte range",
            )
        })?;
    if required_bytes > buffer.byte_len() {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "element_count={element_count} requires {required_bytes} bytes, but the device buffer has {} bytes",
                buffer.byte_len()
            ),
        ));
    }
    Ok(())
}

impl CudaStream {
    /// Starts one owned thread-local CUDA Graph capture on this stream.
    ///
    /// With CUDA disabled this returns an actionable unavailable error. With
    /// CUDA enabled, success exclusively borrows this stream until the returned
    /// [`GraphCapture`] is explicitly aborted or dropped. No eager fallback is
    /// performed.
    ///
    /// # Errors
    ///
    pub fn begin_graph_capture(
        &mut self,
        mode: CudaGraphCaptureMode,
    ) -> CudaResult<GraphCapture<'_>> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.begin_graph_capture(mode as u32)?;
            begin_deferred_capture_contexts();
            Ok(GraphCapture {
                native,
                stream: self,
                active: true,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (self, mode);
            Err(CudaError::unavailable("CudaStream::begin_graph_capture"))
        }
    }

    /// Begins the C05-5 fixed-fill capture with ownership of both its stream
    /// and device buffer moved into the returned state machine.
    ///
    /// This is the cold-owner counterpart to [`Self::begin_graph_fill_capture`].
    /// It exists so an instantiated graph can be stored by value without a
    /// self-referential Rust borrow. It admits exactly the same fixed-address
    /// f32-fill operation set; it does not add H2D, node updates, model work,
    /// registry lookup, or eager fallback.
    ///
    /// # Errors
    ///
    /// A Rust-side preflight error is returned with the untouched resources in
    /// [`OwnedGraphFillCaptureBeginError::into_resources`]. After native entry,
    /// errors deliberately return no reusable resources because native capture
    /// recovery may retain their raw leases fail-closed.
    pub fn begin_owned_graph_fill_capture(
        self,
        buffer: CudaDeviceBuffer,
        element_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphFillCapture, OwnedGraphFillCaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphFillResources::new(self, buffer);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphFillResources::new(self, buffer);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_fill_capture";
            if let Err(error) = validate_graph_fill_capture_preflight(
                &resources.stream,
                &resources.buffer,
                element_count,
                OPERATION,
            ) {
                return Err(OwnedGraphFillCaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources.stream.native.begin_graph_fill_capture(
                resources.buffer.native_handle(),
                element_count,
                mode as u32,
            ) {
                Ok(native) => native,
                Err(error) => return Err(OwnedGraphFillCaptureBeginError::terminal(error)),
            };
            begin_deferred_capture_contexts();
            let (stream, buffer) = resources.into_parts();
            Ok(OwnedGraphFillCapture {
                native,
                stream: Some(stream),
                buffer: Some(buffer),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (element_count, mode);
            Err(OwnedGraphFillCaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_fill_capture"),
                resources,
            ))
        }
    }

    /// Begins the sole C05-5 capture-admitted operation set: one or more
    /// fixed-shape f32 fills of a caller-preallocated device buffer.
    ///
    /// The returned owner retains mutable borrows of `self` and `buffer` until
    /// the graph/exec is closed, preserving the captured CUDA stream and device
    /// address across replay. Generic eager fills, H2D/D2H chains, cuBLASLt,
    /// and node updates remain outside this slice.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state, range, or native capture-admission error. No
    /// fallback to eager execution is attempted.
    pub fn begin_graph_fill_capture<'stream, 'buffer>(
        &'stream mut self,
        buffer: &'buffer mut CudaDeviceBuffer,
        element_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> CudaResult<GraphFillCapture<'stream, 'buffer>> {
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_graph_fill_capture";
            validate_graph_fill_capture_preflight(self, buffer, element_count, OPERATION)?;
            let native = self.native.begin_graph_fill_capture(
                buffer.native_handle(),
                element_count,
                mode as u32,
            )?;
            begin_deferred_capture_contexts();
            Ok(GraphFillCapture {
                native,
                stream: Some(self),
                buffer: Some(buffer),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (self, buffer, element_count, mode);
            Err(CudaError::unavailable(
                "CudaStream::begin_graph_fill_capture",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_graph_error_info_constructor_is_an_exact_zeroed_v1_record() {
        let raw = RawGraphErrorInfo::new();

        assert_eq!(raw.struct_size(), RawGraphErrorInfo::ABI_SIZE);
        let decoded = decode_graph_failure_info(&raw)
            .expect("an exact zeroed v1 graph companion record must decode");
        assert_eq!(decoded.stage(), None);
        assert_eq!(decoded.capture_id(), None);
        assert_eq!(decoded.exec_id(), None);
        assert!(!decoded.submission_started());
        assert!(!decoded.completion_known());
        assert!(!decoded.resource_release_known());
        assert!(!decoded.poisoned());
    }

    #[test]
    fn lifecycle_accepts_closed_capture_and_replay_path() {
        let mut lifecycle = CudaGraphLifecycle::new();
        lifecycle.begin_capture().unwrap();
        lifecycle.enqueue_capture_operation().unwrap();
        lifecycle.end_capture().unwrap();
        lifecycle.instantiate().unwrap();
        lifecycle.update().unwrap();
        lifecycle.launch().unwrap();
        lifecycle.observe_completion(true, true).unwrap();
        assert_eq!(lifecycle.state(), CudaGraphLifecycleState::Instantiated);
        lifecycle.close().unwrap();
        assert_eq!(lifecycle.state(), CudaGraphLifecycleState::Closed);
    }

    #[test]
    fn lifecycle_rejects_invalid_and_ambiguous_paths_fail_closed() {
        let mut lifecycle = CudaGraphLifecycle::new();
        let error = lifecycle.launch().unwrap_err();
        assert_eq!(error.kind(), CudaErrorKind::InvalidState);
        assert_eq!(error.stage(), CudaErrorStage::Validation);

        lifecycle.begin_capture().unwrap();
        assert_eq!(
            lifecycle.begin_capture().unwrap_err().kind(),
            CudaErrorKind::InvalidState
        );
        assert_eq!(
            lifecycle.close().unwrap_err().kind(),
            CudaErrorKind::InvalidState
        );
        lifecycle.abort_capture().unwrap();
        assert_eq!(lifecycle.state(), CudaGraphLifecycleState::Closed);

        let mut launching = CudaGraphLifecycle::new();
        launching.begin_capture().unwrap();
        launching.end_capture().unwrap();
        launching.instantiate().unwrap();
        launching.launch().unwrap();
        launching.observe_completion(false, false).unwrap();
        assert_eq!(launching.state(), CudaGraphLifecycleState::Poisoned);
        assert_eq!(
            launching.close().unwrap_err().kind(),
            CudaErrorKind::InvalidState
        );
        assert_eq!(
            launching.update().unwrap_err().kind(),
            CudaErrorKind::InvalidState
        );

        let mut unreleased = CudaGraphLifecycle::new();
        unreleased.begin_capture().unwrap();
        unreleased.end_capture().unwrap();
        unreleased.instantiate().unwrap();
        unreleased.launch().unwrap();
        unreleased.observe_completion(true, false).unwrap();
        assert_eq!(unreleased.state(), CudaGraphLifecycleState::Poisoned);
        assert_eq!(
            unreleased.close().unwrap_err().kind(),
            CudaErrorKind::InvalidState
        );
    }

    #[test]
    fn graph_error_record_preserves_unknown_stage_and_conservative_flags() {
        let info = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RawGraphErrorInfo::ABI_SIZE + 8,
            graph_stage: 77,
            capture_id: 41,
            exec_id: 0,
            submission_started: 1,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 1,
            reserved0: 0,
            reserved: [0; 3],
        })
        .unwrap();

        assert_eq!(info.stage(), Some(CudaGraphStage::Unknown(77)));
        assert_eq!(info.capture_id(), NonZeroU64::new(41));
        assert_eq!(info.exec_id(), None);
        assert!(info.submission_started());
        assert!(!info.completion_known());
        assert!(!info.resource_release_known());
        assert!(info.poisoned());
        assert!(!info.is_empty_capture_begin_attempt());
    }

    #[test]
    fn graph_error_record_recognizes_only_the_exact_empty_capture_begin_evidence() {
        let empty = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RawGraphErrorInfo::ABI_SIZE,
            graph_stage: 1,
            capture_id: 0,
            exec_id: 0,
            submission_started: 0,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 0,
            reserved0: 0,
            reserved: [0; 3],
        })
        .unwrap();
        assert!(empty.is_empty_capture_begin_attempt());

        let with_capture_id = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RawGraphErrorInfo::ABI_SIZE,
            graph_stage: 1,
            capture_id: 1,
            exec_id: 0,
            submission_started: 0,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 0,
            reserved0: 0,
            reserved: [0; 3],
        })
        .unwrap();
        assert!(!with_capture_id.is_empty_capture_begin_attempt());
    }

    #[test]
    fn graph_error_record_rejects_malformed_abi_booleans_and_reserved_data() {
        let malformed_flag = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RawGraphErrorInfo::ABI_SIZE,
            graph_stage: 0,
            capture_id: 0,
            exec_id: 0,
            submission_started: 2,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 0,
            reserved0: 0,
            reserved: [0; 3],
        })
        .unwrap_err();
        assert_eq!(malformed_flag.kind(), CudaErrorKind::Internal);

        let reserved = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RawGraphErrorInfo::ABI_SIZE,
            graph_stage: 0,
            capture_id: 0,
            exec_id: 0,
            submission_started: 0,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 0,
            reserved0: 1,
            reserved: [0; 3],
        })
        .unwrap_err();
        assert_eq!(reserved.kind(), CudaErrorKind::Internal);

        let short_prefix = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RawGraphErrorInfo::ABI_SIZE - 1,
            graph_stage: 0,
            capture_id: 0,
            exec_id: 0,
            submission_started: 0,
            completion_known: 0,
            resource_release_known: 0,
            poisoned: 0,
            reserved0: 0,
            reserved: [0; 3],
        })
        .unwrap_err();
        assert_eq!(short_prefix.kind(), CudaErrorKind::Internal);
        assert_eq!(short_prefix.stage(), CudaErrorStage::Validation);
    }
}
