//! CUDA Graph ABI vocabulary and fail-closed lifecycle policy.
//!
//! C05-1 wires one native capture-begin ABI stub while still fixing ownership
//! vocabulary and lifecycle transitions on the CPU. The stub fails closed, so
//! a later successful capture slice cannot silently widen this contract.

use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::rc::Rc;

use crate::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult, CudaStream};

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

/// Borrowed graph-capture owner reserved for the linked native graph ABI.
///
/// This initial ABI foundation never constructs a capture. It still reserves
/// the mutable stream borrow in the safe API so the eventual native capture
/// cannot be introduced as an aliasing-compatible change.
#[derive(Debug)]
pub struct GraphCapture<'stream> {
    _stream: PhantomData<&'stream mut CudaStream>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl CudaStream {
    /// Starts CUDA Graph capture on this stream when the linked native ABI
    /// provides it.
    ///
    /// With CUDA disabled this returns an actionable unavailable error. With
    /// CUDA enabled it calls the linked native ABI, which currently returns
    /// not-supported without mutating stream ownership or falling back to
    /// eager execution.
    ///
    /// # Errors
    ///
    /// Always returns an error until a later native slice can return a safely
    /// owned capture handle.
    pub fn begin_graph_capture(
        &mut self,
        mode: CudaGraphCaptureMode,
    ) -> CudaResult<GraphCapture<'_>> {
        #[cfg(feature = "cuda")]
        {
            self.native.begin_graph_capture(mode as u32)?;
            Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Prepare,
                0,
                "CudaStream::begin_graph_capture",
                "native graph capture returned success without an owned capture handle",
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (self, mode);
            Err(CudaError::unavailable("CudaStream::begin_graph_capture"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

    #[repr(C)]
    struct RawGraphErrorInfo {
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

    const _: () = assert!(size_of::<RawGraphErrorInfo>() == 56);
    const _: () = assert!(align_of::<RawGraphErrorInfo>() == 8);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, struct_size) == 0);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, graph_stage) == 4);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, capture_id) == 8);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, exec_id) == 16);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, submission_started) == 24);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, completion_known) == 25);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, resource_release_known) == 26);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, poisoned) == 27);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, reserved0) == 28);
    const _: () = assert!(offset_of!(RawGraphErrorInfo, reserved) == 32);
    const RAW_GRAPH_ERROR_INFO_SIZE: u32 = 56;

    fn decode_graph_failure_info(raw: &RawGraphErrorInfo) -> CudaResult<CudaGraphFailureInfo> {
        if raw.struct_size < RAW_GRAPH_ERROR_INFO_SIZE {
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
        if decode_abi_bool("submission_started", raw.submission_started)? {
            flags |= CudaGraphFailureInfo::SUBMISSION_STARTED;
        }
        if decode_abi_bool("completion_known", raw.completion_known)? {
            flags |= CudaGraphFailureInfo::COMPLETION_KNOWN;
        }
        if decode_abi_bool("resource_release_known", raw.resource_release_known)? {
            flags |= CudaGraphFailureInfo::RESOURCE_RELEASE_KNOWN;
        }
        if decode_abi_bool("poisoned", raw.poisoned)? {
            flags |= CudaGraphFailureInfo::POISONED;
        }

        Ok(CudaGraphFailureInfo {
            stage: decode_graph_stage(raw.graph_stage),
            capture_id: NonZeroU64::new(raw.capture_id),
            exec_id: NonZeroU64::new(raw.exec_id),
            flags,
        })
    }

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

    fn decode_abi_bool(name: &'static str, value: u8) -> CudaResult<bool> {
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
            struct_size: RAW_GRAPH_ERROR_INFO_SIZE + 8,
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
    }

    #[test]
    fn graph_error_record_rejects_malformed_abi_booleans_and_reserved_data() {
        let malformed_flag = decode_graph_failure_info(&RawGraphErrorInfo {
            struct_size: RAW_GRAPH_ERROR_INFO_SIZE,
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
            struct_size: RAW_GRAPH_ERROR_INFO_SIZE,
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
    }
}
