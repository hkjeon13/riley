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
use crate::Bf16ArgmaxResult;
use crate::batch::PackedBatchHostV1;
#[cfg(feature = "cuda")]
use crate::runtime::{ContextInner, ensure_same_context};
use crate::{
    CudaDeviceBuffer, CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage,
    CudaPinnedHostBuffer, CudaPreparedGemm, CudaResult, CudaStream,
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

/// One exact C05 operation that has an independently reviewed CUDA Graph
/// capture contract.
///
/// This deliberately names only the narrow fixed-address vertical slices in
/// the native ABI. It does not imply that sibling primitives, a whole model
/// forward, or a CUDA stream/context are capture-safe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u32)]
pub enum CudaGraphCaptureOperation {
    /// One or more fixed-shape f32 fills of one caller-owned device buffer.
    FillF32 = 1,
    /// One whole-slab H2D memcpy between retained pinned/device allocations.
    H2D = 2,
    /// One fixed-address, out-of-place BF16 SiLU kernel.
    SiluBf16 = 3,
    /// One fixed-address, out-of-place BF16 activated-gate × up kernel.
    GatedMultiplyBf16 = 4,
    /// One fixed-address, out-of-place BF16 residual-add kernel.
    ResidualAddBf16 = 5,
    /// One fixed-address, out-of-place generic eager BF16 RMSNorm kernel.
    ///
    /// This is deliberately distinct from SmolLM2 and Fixed37 variants, fused
    /// RMSNorm, and C07 executor integration.
    CanonicalRmsNormBf16 = 6,
    /// One fixed-address deterministic BF16 greedy-argmax kernel.
    ///
    /// This retains only the fixed logits and U32 result allocations. It does
    /// not include output-row gather, token/status transfer, completion
    /// handling, C07 executor integration, or a sampling policy beyond this
    /// exact deterministic primitive.
    Bf16Argmax = 7,
    /// One fixed-address, out-of-place BF16 row-gather kernel.
    ///
    /// This retains only the fixed input, U32 index, and output allocations.
    /// It does not include H2D/D2H staging, argmax, token/status transfer,
    /// completion handling, C07 executor integration, or a sampling policy.
    Bf16RowGather = 8,
    /// One fixed-address, two-node BF16 row-gather then deterministic argmax
    /// device-only graph.
    ///
    /// This retains only the fixed input, U32 index, gathered BF16, and U32
    /// result allocations. It excludes H2D/D2H staging, token/status
    /// transfer, completion handling, C07 executor integration, and a
    /// sampling policy beyond the exact deterministic primitive.
    Bf16RowGatherArgmax = 9,
    /// One fixed-address three-node BF16 row-gather -> deterministic argmax
    /// -> pinned-host result-record D2H graph.
    ///
    /// This proves only the exact raw record chain and its fixed D2H lifetime.
    /// It does not validate token/status records, commit scheduler state, or
    /// establish a C07 completion boundary.
    Bf16RowGatherArgmaxD2H = 10,
    /// One fixed-address, out-of-place BF16 indexed RoPE kernel.
    ///
    /// This retains only the fixed input, cosine/sine tables, device
    /// positions, and output allocations. It does not include H2D/D2H,
    /// final normalization, C07 executor integration, or full decode
    /// execution authority.
    IndexedRopeBf16 = 11,
    /// One fixed-address BF16 ragged paged-K/V cache-write kernel.
    ///
    /// This retains only fixed key/value source and pool allocations plus the
    /// five packed device-metadata allocations. It does not include metadata
    /// H2D, projections, attention reads, scheduler commit, C07 executor
    /// integration, or full decode execution authority.
    RaggedPagedKvCacheWriteBf16 = 12,
    /// One fixed-address BF16 grouped-head ragged paged-attention kernel.
    ///
    /// This retains fixed query, K/V pool, output, and packed device-metadata
    /// allocations. It does not include metadata H2D, projection, cache
    /// writes, scheduler commit, C07 executor integration, or full decode
    /// execution authority.
    GroupedRaggedPagedAttentionBf16 = 13,
    /// One fixed-address BF16 embedding validation -> status-D2H graph.
    ///
    /// This retains a fixed BF16 table, U32 token IDs, BF16 output, device
    /// error scratch, and exact pinned status record. It does not include
    /// token H2D, table-residency policy, scheduler commit, C07 executor
    /// integration, or full decode execution authority.
    Bf16EmbeddingStatusD2H = 14,
    /// One fixed-address canonical BF16/F32 cuBLASLt GEMM graph.
    ///
    /// This retains one cold-prepared strict-no-split plan and exact whole
    /// input, weight, output, and workspace allocations. It does not admit
    /// spans, offsets, alternate reduction/epilogue, executor wiring, or C07
    /// projection/LM-head capability evidence.
    CanonicalGemmBf16 = 15,
    /// One fixed-address canonical BF16 RMSNorm -> cuBLASLt GEMM graph.
    ///
    /// The RMSNorm output is the one intentional fixed-allocation dependency
    /// consumed as GEMM input. It retains a cold strict-no-split BF16/F32
    /// plan and every other whole allocation by value; it does not imply a
    /// general graph composer, spans, offsets, node updates, C07 executor
    /// wiring, projection/LM-head evidence, or full decode authority.
    CanonicalRmsNormGemmBf16 = 16,
}

impl CudaGraphCaptureOperation {
    /// Queries this exact operation's native capture admission result.
    ///
    /// The query is a read-only ABI vocabulary lookup: it does not initialize
    /// CUDA, create a context, allocate, or inspect a stream. A result of
    /// [`CudaGraphCaptureCapability::Supported`] is evidence only for this
    /// exact operation and must not be broadened into graph execution
    /// authority without validating the surrounding resources and semantics.
    ///
    /// # Errors
    ///
    /// Returns [`CudaErrorKind::Unavailable`] when this crate was built
    /// without the native CUDA feature, or a translated ABI error otherwise.
    pub fn capture_capability(self) -> CudaResult<CudaGraphCaptureCapability> {
        #[cfg(feature = "cuda")]
        {
            crate::ffi::graph_capture_capability(self as u32)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = self;
            Err(CudaError::unavailable(
                "CudaGraphCaptureOperation::capture_capability",
            ))
        }
    }
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
    /// Staging an exact payload into a graph-retained fixed H2D source before
    /// one replay. This is synchronous host work, not CUDA launch submission.
    InputStage,
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
        10 => Some(CudaGraphStage::InputStage),
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

/// A by-value stream, pinned source, and device destination recovered from a
/// known fixed-address H2D graph release.
///
/// It is not itself a graph capability: its three values are ordinary cold
/// resources again only after native close proves every permanent graph lease
/// has been released.
pub struct OwnedGraphH2DResources {
    stream: CudaStream,
    source: CudaPinnedHostBuffer,
    destination: CudaDeviceBuffer,
}

impl OwnedGraphH2DResources {
    fn new(
        stream: CudaStream,
        source: CudaPinnedHostBuffer,
        destination: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            source,
            destination,
        }
    }

    /// Returns the exact stream, pinned source, and device destination after a
    /// known graph release.
    #[must_use]
    pub fn into_parts(self) -> (CudaStream, CudaPinnedHostBuffer, CudaDeviceBuffer) {
        let Self {
            stream,
            source,
            destination,
        } = self;
        (stream, source, destination)
    }

    /// Explicitly destroys the recovered cold resources.
    ///
    /// The device destination closes before the retained pinned source and
    /// stream, mirroring the ordinary caller-owned copy cleanup order.
    pub fn close(self) -> CudaResult<()> {
        let (stream, source, destination) = self.into_parts();
        destination.close()?;
        source.close()?;
        stream.close()
    }
}

/// Error from beginning a by-value fixed-address H2D graph capture.
///
/// Only Rust preflight failures return the untouched three-resource bundle.
/// Once native capture entry has been attempted, the bundle is intentionally
/// withheld because CUDA may retain its raw source/destination/stream leases
/// while resolving a deferred failure.
#[must_use]
pub struct OwnedGraphH2DCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphH2DResources>,
}

impl OwnedGraphH2DCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphH2DResources) -> Self {
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

    /// Returns untouched values only when native capture ownership was never
    /// entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphH2DResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphH2DCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphH2DCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphH2DCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph H2D capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphH2DCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address whole-slab H2D graph capture.
///
/// The retained pinned source and device destination cannot be independently
/// read, written, closed, or moved while CUDA may reference their addresses.
/// This thread-local capture owner is deliberately `!Send + !Sync`.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphH2DCapture>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphH2DCapture>();
/// ```
pub struct OwnedGraphH2DCapture {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    // Native drops before child resource wrappers. A capture ambiguity can own
    // all three raw addresses and therefore must remain fail-closed first.
    stream: Option<CudaStream>,
    source: Option<CudaPinnedHostBuffer>,
    destination: Option<CudaDeviceBuffer>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphH2DCapture {
    /// Captures the sole fixed-address H2D node.
    ///
    /// There are no user-supplied pointers, offsets, ranges, or payload bytes:
    /// begin fixed the whole source/destination slabs, and replay staging is a
    /// later graph-exec operation.
    pub fn enqueue_h2d(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphH2DCapture::enqueue_h2d";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph H2D capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph H2D enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed H2D graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_h2d();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers all three moved resources into an owned
    /// captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedH2DGraph> {
        const OPERATION: &str = "OwnedGraphH2DCapture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph H2D capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph H2D enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed H2D enqueue",
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
            let resources = take_owned_graph_h2d_resources(
                &mut self.stream,
                &mut self.source,
                &mut self.destination,
                OPERATION,
            )?;
            let (stream, source, destination) = resources.into_parts();
            Ok(OwnedCapturedH2DGraph {
                native,
                stream: Some(stream),
                source: Some(source),
                destination: Some(destination),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.source, &mut self.destination);
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns the moved resources only after known native
    /// recovery releases every graph lease.
    pub fn abort(mut self) -> CudaResult<OwnedGraphH2DResources> {
        self.abort_once()?;
        take_owned_graph_h2d_resources(
            &mut self.stream,
            &mut self.source,
            &mut self.destination,
            "OwnedGraphH2DCapture::abort",
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
            Err(CudaError::unavailable("OwnedGraphH2DCapture::abort"))
        }
    }
}

impl Drop for OwnedGraphH2DCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address H2D CUDA Graph awaiting instantiate or close.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedCapturedH2DGraph>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedCapturedH2DGraph>();
/// ```
pub struct OwnedCapturedH2DGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    stream: Option<CudaStream>,
    source: Option<CudaPinnedHostBuffer>,
    destination: Option<CudaDeviceBuffer>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedH2DGraph {
    /// Instantiates this graph while retaining its exact source, destination,
    /// and stream by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphH2DExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_h2d_resources(
                &mut self.stream,
                &mut self.source,
                &mut self.destination,
                "OwnedCapturedH2DGraph::instantiate",
            )?;
            let (stream, source, destination) = resources.into_parts();
            Ok(OwnedGraphH2DExec {
                native,
                stream: Some(stream),
                source: Some(source),
                destination: Some(destination),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.source, &mut self.destination);
            Err(CudaError::unavailable("OwnedCapturedH2DGraph::instantiate"))
        }
    }

    /// Destroys the graph and returns the three resources only after known
    /// native release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphH2DResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_h2d_resources(
                &mut self.stream,
                &mut self.source,
                &mut self.destination,
                "OwnedCapturedH2DGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.source, &mut self.destination);
            Err(CudaError::unavailable("OwnedCapturedH2DGraph::close"))
        }
    }
}

/// By-value fixed-address H2D CUDA Graph executable.
///
/// It exposes only [`Self::launch_with_source`], which stages one exact whole
/// payload into the retained pinned allocation before the next replay. There is
/// intentionally no safe bare launch that could replay stale source bytes.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphH2DExec>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphH2DExec>();
/// ```
pub struct OwnedGraphH2DExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    stream: Option<CudaStream>,
    source: Option<CudaPinnedHostBuffer>,
    destination: Option<CudaDeviceBuffer>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphH2DExec {
    /// Stages one exact whole-slab payload and launches its fixed-address H2D
    /// graph replay.
    ///
    /// The returned completion owner retains an exclusive mutable borrow of
    /// this exec, so no second stage, relaunch, or close can occur until the
    /// one CUDA replay reaches its completion boundary.
    pub fn launch_with_source<'exec>(
        &'exec mut self,
        bytes: &[u8],
    ) -> CudaResult<OwnedGraphH2DLaunch<'exec>> {
        const OPERATION: &str = "OwnedGraphH2DExec::launch_with_source";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph H2D transition left native state uncertain",
            ));
        }
        let expected_byte_len = self
            .source
            .as_ref()
            .ok_or_else(|| {
                self.terminal = true;
                CudaError::invalid_state(
                    OPERATION,
                    "the owned graph exec lost its pinned H2D source",
                )
            })?
            .byte_len();
        let actual_byte_len = u64::try_from(bytes.len())
            .map_err(|_| CudaError::out_of_range(OPERATION, "payload length does not fit u64"))?;
        if actual_byte_len != expected_byte_len {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!(
                    "payload has {actual_byte_len} bytes, but this fixed H2D graph requires exactly {expected_byte_len}"
                ),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            if self.destination.is_none() {
                self.terminal = true;
                return Err(CudaError::invalid_state(
                    OPERATION,
                    "the owned graph exec lost its fixed device destination",
                ));
            }
            let source = self.source.as_ref().expect("source checked above");
            if let Err(error) = self.native.stage_h2d_source(source.native_handle(), bytes) {
                // Native staging is the only path that marks input fresh. Do
                // not permit any later launch to observe a previous payload
                // after a malformed native stage outcome.
                self.terminal = true;
                return Err(error);
            }
            let native = match self.stream.as_mut() {
                Some(stream) => self.native.launch(&mut stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph exec lost its capture stream",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphH2DLaunch {
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
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns all captured resources only after
    /// native close proves its three permanent leases were released.
    ///
    /// ```compile_fail
    /// fn cannot_close_an_owned_h2d_exec_while_launch_is_live(
    ///     mut exec: riley_cuda::OwnedGraphH2DExec,
    ///     payload: Vec<u8>,
    /// ) {
    ///     let launch = exec.launch_with_source(&payload).unwrap();
    ///     let _ = exec.close();
    ///     drop(launch);
    /// }
    /// ```
    pub fn close(mut self) -> CudaResult<OwnedGraphH2DResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphH2DExec::close",
                "an earlier graph H2D transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_h2d_resources(
                &mut self.stream,
                &mut self.source,
                &mut self.destination,
                "OwnedGraphH2DExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&mut self.stream, &mut self.source, &mut self.destination);
            Err(CudaError::unavailable("OwnedGraphH2DExec::close"))
        }
    }
}

/// Completion owner for one [`OwnedGraphH2DExec`] replay.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphH2DLaunch<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphH2DLaunch<'static>>();
/// ```
pub struct OwnedGraphH2DLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphH2DExec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphH2DLaunch<'_> {
    /// Waits for H2D graph replay completion exactly once.
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
            Err(CudaError::unavailable("OwnedGraphH2DLaunch::finish"))
        }
    }
}

impl Drop for OwnedGraphH2DLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_h2d_resources(
    stream: &mut Option<CudaStream>,
    source: &mut Option<CudaPinnedHostBuffer>,
    destination: &mut Option<CudaDeviceBuffer>,
    operation: &'static str,
) -> CudaResult<OwnedGraphH2DResources> {
    let stream = stream.take().ok_or_else(|| {
        CudaError::invalid_state(operation, "the owned graph owner lost its capture stream")
    })?;
    let source = source.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph owner lost its pinned H2D source",
        )
    })?;
    let destination = destination.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph owner lost its fixed device destination",
        )
    })?;
    Ok(OwnedGraphH2DResources::new(stream, source, destination))
}

#[cfg(feature = "cuda")]
fn validate_graph_h2d_capture_preflight(
    stream: &CudaStream,
    source: &CudaPinnedHostBuffer,
    destination: &CudaDeviceBuffer,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, source.context_owner(), operation)?;
    ensure_same_context(&stream.context, destination.context_owner(), operation)?;
    source.ensure_idle_for_operation(operation)?;
    destination.ensure_idle_for_operation(operation)?;
    if source.byte_len() == 0 || source.byte_len() != destination.byte_len() {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "fixed graph H2D requires equal nonzero whole slabs, but source has {} bytes and destination has {} bytes",
                source.byte_len(),
                destination.byte_len()
            ),
        ));
    }
    Ok(())
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

/// A by-value stream and two fixed device buffers recovered from a known
/// BF16-SiLU graph lifecycle transition.
///
/// This bundle is only resource recovery evidence. It exposes no graph replay,
/// graph-visible pointer, mutable span, or fresh-input staging capability.
pub struct OwnedGraphSiluBf16Resources {
    stream: CudaStream,
    input: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
}

impl OwnedGraphSiluBf16Resources {
    fn new(stream: CudaStream, input: CudaDeviceBuffer, output: CudaDeviceBuffer) -> Self {
        Self {
            stream,
            input,
            output,
        }
    }

    /// Returns the exact stream, fixed input, and fixed output after known
    /// native graph-lease release.
    #[must_use]
    pub fn into_parts(self) -> (CudaStream, CudaDeviceBuffer, CudaDeviceBuffer) {
        let Self {
            stream,
            input,
            output,
        } = self;
        (stream, input, output)
    }

    /// Explicitly destroys both recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, input, output) = self.into_parts();
        output.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning a by-value fixed-address BF16-SiLU graph capture.
///
/// Only Rust-side preflight failures return the untouched resource triple.
/// Once native capture begin is entered, an ambiguous CUDA transition may own
/// raw addresses and resources remain deliberately fail-closed.
#[must_use]
pub struct OwnedGraphSiluBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphSiluBf16Resources>,
}

impl OwnedGraphSiluBf16CaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphSiluBf16Resources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/input/output triple only when native
    /// capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphSiluBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphSiluBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphSiluBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphSiluBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 SiLU capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphSiluBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address, out-of-place BF16 SiLU CUDA
/// Graph capture.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphSiluBf16Capture>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphSiluBf16Capture>();
/// ```
pub struct OwnedGraphSiluBf16Capture {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    resources: Option<OwnedGraphSiluBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphSiluBf16Capture {
    /// Captures the one immutable BF16 SiLU node.
    pub fn enqueue_silu_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphSiluBf16Capture::enqueue_silu_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 SiLU capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 SiLU enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 SiLU graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_silu_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/input/output triple into a
    /// by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedSiluBf16Graph> {
        const OPERATION: &str = "OwnedGraphSiluBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 SiLU capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 SiLU enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 SiLU enqueue",
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
            let resources = take_owned_graph_silu_bf16_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedSiluBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphSiluBf16Resources> {
        self.abort_once()?;
        take_owned_graph_silu_bf16_resources(
            &mut self.resources,
            "OwnedGraphSiluBf16Capture::abort",
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
            Err(CudaError::unavailable("OwnedGraphSiluBf16Capture::abort"))
        }
    }
}

impl Drop for OwnedGraphSiluBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 SiLU CUDA Graph awaiting instantiate or close.
pub struct OwnedCapturedSiluBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphSiluBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedSiluBf16Graph {
    /// Instantiates this graph while retaining its fixed stream/input/output
    /// triple by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphSiluBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_silu_bf16_resources(
                &mut self.resources,
                "OwnedCapturedSiluBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphSiluBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedSiluBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns its resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphSiluBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_silu_bf16_resources(
                &mut self.resources,
                "OwnedCapturedSiluBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable("OwnedCapturedSiluBf16Graph::close"))
        }
    }
}

/// By-value fixed-address BF16 SiLU CUDA Graph executable.
///
/// It only replays the capture-time input allocation. Fresh input staging,
/// mutable spans, node updates, and model/executor wiring remain outside this
/// C05 ownership slice.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphSiluBf16Exec>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphSiluBf16Exec>();
/// ```
pub struct OwnedGraphSiluBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphSiluBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphSiluBf16Exec {
    /// Replays the fixed-address BF16 SiLU graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphSiluBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphSiluBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 SiLU transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 SiLU exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphSiluBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns the fixed resource triple only
    /// after native close proves every graph lease was released.
    ///
    /// ```compile_fail
    /// fn cannot_close_an_owned_silu_exec_while_launch_is_live(
    ///     mut exec: riley_cuda::OwnedGraphSiluBf16Exec,
    /// ) {
    ///     let launch = exec.launch().unwrap();
    ///     let _ = exec.close();
    ///     drop(launch);
    /// }
    /// ```
    pub fn close(mut self) -> CudaResult<OwnedGraphSiluBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphSiluBf16Exec::close",
                "an earlier graph BF16 SiLU transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_silu_bf16_resources(
                &mut self.resources,
                "OwnedGraphSiluBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable("OwnedGraphSiluBf16Exec::close"))
        }
    }
}

/// Completion owner for one [`OwnedGraphSiluBf16Exec`] replay.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphSiluBf16Launch<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphSiluBf16Launch<'static>>();
/// ```
pub struct OwnedGraphSiluBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphSiluBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphSiluBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable("OwnedGraphSiluBf16Launch::finish"))
        }
    }
}

impl Drop for OwnedGraphSiluBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_silu_bf16_resources(
    resources: &mut Option<OwnedGraphSiluBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphSiluBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 SiLU owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_silu_bf16_capture_preflight(
    stream: &CudaStream,
    input: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    element_count: u64,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, input.context_owner(), operation)?;
    ensure_same_context(&stream.context, output.context_owner(), operation)?;
    input.ensure_idle_for_operation(operation)?;
    output.ensure_idle_for_operation(operation)?;
    let required_bytes = element_count
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "element_count overflows the fixed BF16 SiLU capture byte range",
            )
        })?;
    if element_count == 0 || required_bytes > input.byte_len() || required_bytes > output.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "element_count={element_count} requires {required_bytes} BF16 bytes, but input/output capacities are {}/{} bytes",
                input.byte_len(),
                output.byte_len()
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and three fixed device buffers recovered from a known
/// BF16 gated-multiply graph lifecycle transition.
///
/// This bundle is only resource recovery evidence. It exposes no graph replay,
/// graph-visible pointer, mutable span, or fresh-input staging capability.
pub struct OwnedGraphGatedMultiplyBf16Resources {
    stream: CudaStream,
    activated_gate: CudaDeviceBuffer,
    up: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
}

impl OwnedGraphGatedMultiplyBf16Resources {
    fn new(
        stream: CudaStream,
        activated_gate: CudaDeviceBuffer,
        up: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            activated_gate,
            up,
            output,
        }
    }

    /// Returns the exact stream, fixed activated-gate input, fixed up input,
    /// and fixed output after known native graph-lease release.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            activated_gate,
            up,
            output,
        } = self;
        (stream, activated_gate, up, output)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, activated_gate, up, output) = self.into_parts();
        output.close()?;
        up.close()?;
        activated_gate.close()?;
        stream.close()
    }
}

/// Error from beginning a by-value fixed-address BF16 gated-multiply graph
/// capture.
///
/// Only Rust-side preflight failures return the untouched resource quartet.
/// Once native capture begin is entered, an ambiguous CUDA transition may own
/// raw addresses and resources remain deliberately fail-closed.
#[must_use]
pub struct OwnedGraphGatedMultiplyBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphGatedMultiplyBf16Resources>,
}

impl OwnedGraphGatedMultiplyBf16CaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphGatedMultiplyBf16Resources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/input/input/output quartet only when
    /// native capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphGatedMultiplyBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphGatedMultiplyBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphGatedMultiplyBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphGatedMultiplyBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 gated-multiply capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphGatedMultiplyBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address, out-of-place BF16 gated
/// multiply CUDA Graph capture.
pub struct OwnedGraphGatedMultiplyBf16Capture {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    resources: Option<OwnedGraphGatedMultiplyBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphGatedMultiplyBf16Capture {
    /// Captures the one immutable BF16 gated-multiply node.
    pub fn enqueue_gated_multiply_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphGatedMultiplyBf16Capture::enqueue_gated_multiply_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 gated-multiply capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 gated-multiply enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 gated-multiply graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_gated_multiply_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/input/input/output quartet
    /// into a by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedGatedMultiplyBf16Graph> {
        const OPERATION: &str = "OwnedGraphGatedMultiplyBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 gated-multiply capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 gated-multiply enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 gated-multiply enqueue",
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
            let resources =
                take_owned_graph_gated_multiply_bf16_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedGatedMultiplyBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphGatedMultiplyBf16Resources> {
        self.abort_once()?;
        take_owned_graph_gated_multiply_bf16_resources(
            &mut self.resources,
            "OwnedGraphGatedMultiplyBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphGatedMultiplyBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphGatedMultiplyBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 gated-multiply CUDA Graph awaiting instantiate
/// or close.
pub struct OwnedCapturedGatedMultiplyBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphGatedMultiplyBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedGatedMultiplyBf16Graph {
    /// Instantiates this graph while retaining its fixed stream/input/input/
    /// output quartet by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphGatedMultiplyBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_gated_multiply_bf16_resources(
                &mut self.resources,
                "OwnedCapturedGatedMultiplyBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphGatedMultiplyBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedGatedMultiplyBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns its resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphGatedMultiplyBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_gated_multiply_bf16_resources(
                &mut self.resources,
                "OwnedCapturedGatedMultiplyBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedGatedMultiplyBf16Graph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 gated-multiply CUDA Graph executable.
///
/// It only replays the capture-time input allocations. Fresh input staging,
/// mutable spans, node updates, SiLU fusion, and model/executor wiring remain
/// outside this C05 ownership slice.
pub struct OwnedGraphGatedMultiplyBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphGatedMultiplyBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphGatedMultiplyBf16Exec {
    /// Replays the fixed-address BF16 gated-multiply graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphGatedMultiplyBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphGatedMultiplyBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 gated-multiply transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 gated-multiply exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphGatedMultiplyBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns the fixed resource quartet only
    /// after native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphGatedMultiplyBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphGatedMultiplyBf16Exec::close",
                "an earlier graph BF16 gated-multiply transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_gated_multiply_bf16_resources(
                &mut self.resources,
                "OwnedGraphGatedMultiplyBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphGatedMultiplyBf16Exec::close",
            ))
        }
    }
}

/// Completion owner for one [`OwnedGraphGatedMultiplyBf16Exec`] replay.
pub struct OwnedGraphGatedMultiplyBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphGatedMultiplyBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphGatedMultiplyBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphGatedMultiplyBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphGatedMultiplyBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_gated_multiply_bf16_resources(
    resources: &mut Option<OwnedGraphGatedMultiplyBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphGatedMultiplyBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 gated-multiply owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_gated_multiply_bf16_capture_preflight(
    stream: &CudaStream,
    activated_gate: &CudaDeviceBuffer,
    up: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    element_count: u64,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, activated_gate.context_owner(), operation)?;
    ensure_same_context(&stream.context, up.context_owner(), operation)?;
    ensure_same_context(&stream.context, output.context_owner(), operation)?;
    activated_gate.ensure_idle_for_operation(operation)?;
    up.ensure_idle_for_operation(operation)?;
    output.ensure_idle_for_operation(operation)?;
    let required_bytes = element_count
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "element_count overflows the fixed BF16 gated-multiply capture byte range",
            )
        })?;
    if element_count == 0
        || required_bytes > activated_gate.byte_len()
        || required_bytes > up.byte_len()
        || required_bytes > output.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "element_count={element_count} requires {required_bytes} BF16 bytes, but activated_gate/up/output capacities are {}/{}/{} bytes",
                activated_gate.byte_len(),
                up.byte_len(),
                output.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and three fixed device buffers recovered from one known
/// BF16 residual-add graph lifecycle transition.
///
/// This recovery bundle exposes no graph-visible pointer, mutable span, or
/// fresh replay input. The two inputs and output remain distinct throughout
/// capture, graph, and exec ownership.
pub struct OwnedGraphResidualAddBf16Resources {
    stream: CudaStream,
    left: CudaDeviceBuffer,
    right: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
}

impl OwnedGraphResidualAddBf16Resources {
    fn new(
        stream: CudaStream,
        left: CudaDeviceBuffer,
        right: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            left,
            right,
            output,
        }
    }

    /// Returns the exact stream, left input, right input, and output after a
    /// known native graph-lease release.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            left,
            right,
            output,
        } = self;
        (stream, left, right, output)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, left, right, output) = self.into_parts();
        output.close()?;
        right.close()?;
        left.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address BF16 residual-add graph
/// capture.
///
/// Only Rust-side preflight errors recover the untouched resource quartet.
/// Once native begin is attempted, ambiguous CUDA state retains all raw
/// addresses fail-closed.
#[must_use]
pub struct OwnedGraphResidualAddBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphResidualAddBf16Resources>,
}

impl OwnedGraphResidualAddBf16CaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphResidualAddBf16Resources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/input/input/output quartet only when
    /// native capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphResidualAddBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphResidualAddBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphResidualAddBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphResidualAddBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 residual-add capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphResidualAddBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address, out-of-place BF16 residual-add
/// CUDA Graph capture.
pub struct OwnedGraphResidualAddBf16Capture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphResidualAddBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphResidualAddBf16Capture {
    /// Captures the one immutable BF16 residual-add node.
    pub fn enqueue_residual_add_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphResidualAddBf16Capture::enqueue_residual_add_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 residual-add capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 residual-add enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 residual-add graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_residual_add_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/input/input/output quartet
    /// into a by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedResidualAddBf16Graph> {
        const OPERATION: &str = "OwnedGraphResidualAddBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 residual-add capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 residual-add enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 residual-add enqueue",
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
            let resources =
                take_owned_graph_residual_add_bf16_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedResidualAddBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphResidualAddBf16Resources> {
        self.abort_once()?;
        take_owned_graph_residual_add_bf16_resources(
            &mut self.resources,
            "OwnedGraphResidualAddBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphResidualAddBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphResidualAddBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 residual-add CUDA Graph awaiting instantiate or
/// close.
pub struct OwnedCapturedResidualAddBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphResidualAddBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedResidualAddBf16Graph {
    /// Instantiates this graph while retaining its fixed stream/input/input/
    /// output quartet by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphResidualAddBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_residual_add_bf16_resources(
                &mut self.resources,
                "OwnedCapturedResidualAddBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphResidualAddBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedResidualAddBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphResidualAddBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_residual_add_bf16_resources(
                &mut self.resources,
                "OwnedCapturedResidualAddBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedResidualAddBf16Graph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 residual-add CUDA Graph executable.
///
/// It replays only the capture-time input allocations. Fresh input staging,
/// mutable spans, node updates, fused RMSNorm, and model/executor wiring stay
/// outside this narrow C05 ownership slice.
pub struct OwnedGraphResidualAddBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphResidualAddBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphResidualAddBf16Exec {
    /// Replays the fixed-address BF16 residual-add graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphResidualAddBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphResidualAddBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 residual-add transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 residual-add exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphResidualAddBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource quartet only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphResidualAddBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphResidualAddBf16Exec::close",
                "an earlier graph BF16 residual-add transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_residual_add_bf16_resources(
                &mut self.resources,
                "OwnedGraphResidualAddBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphResidualAddBf16Exec::close",
            ))
        }
    }
}

/// Completion owner for one [`OwnedGraphResidualAddBf16Exec`] replay.
pub struct OwnedGraphResidualAddBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphResidualAddBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphResidualAddBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphResidualAddBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphResidualAddBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_residual_add_bf16_resources(
    resources: &mut Option<OwnedGraphResidualAddBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphResidualAddBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 residual-add owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_residual_add_bf16_capture_preflight(
    stream: &CudaStream,
    left: &CudaDeviceBuffer,
    right: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    element_count: u64,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, left.context_owner(), operation)?;
    ensure_same_context(&stream.context, right.context_owner(), operation)?;
    ensure_same_context(&stream.context, output.context_owner(), operation)?;
    left.ensure_idle_for_operation(operation)?;
    right.ensure_idle_for_operation(operation)?;
    output.ensure_idle_for_operation(operation)?;
    let required_bytes = element_count
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "element_count overflows the fixed BF16 residual-add capture byte range",
            )
        })?;
    if element_count == 0
        || required_bytes > left.byte_len()
        || required_bytes > right.byte_len()
        || required_bytes > output.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "element_count={element_count} requires {required_bytes} BF16 bytes, but left/right/output capacities are {}/{}/{} bytes",
                left.byte_len(),
                right.byte_len(),
                output.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and three fixed device buffers recovered from one known
/// canonical BF16 RMSNorm graph lifecycle transition.
///
/// This recovery bundle exposes no graph-visible pointer, mutable span, or
/// fresh replay input. The input, learned weight, and output remain distinct
/// throughout capture, graph, and exec ownership.
pub struct OwnedGraphCanonicalRmsNormBf16Resources {
    stream: CudaStream,
    input: CudaDeviceBuffer,
    weight: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
}

impl OwnedGraphCanonicalRmsNormBf16Resources {
    fn new(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        weight: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            input,
            weight,
            output,
        }
    }

    /// Returns the exact stream, input, weight, and output after a known
    /// native graph-lease release.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            input,
            weight,
            output,
        } = self;
        (stream, input, weight, output)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, input, weight, output) = self.into_parts();
        output.close()?;
        weight.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address canonical BF16 RMSNorm graph
/// capture.
///
/// Only Rust-side preflight errors recover the untouched resource quartet.
/// Once native begin is attempted, ambiguous CUDA state retains all raw
/// addresses fail-closed.
#[must_use]
pub struct OwnedGraphCanonicalRmsNormBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphCanonicalRmsNormBf16Resources>,
}

impl OwnedGraphCanonicalRmsNormBf16CaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphCanonicalRmsNormBf16Resources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/input/weight/output quartet only when
    /// native capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphCanonicalRmsNormBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphCanonicalRmsNormBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphCanonicalRmsNormBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphCanonicalRmsNormBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph canonical BF16 RMSNorm capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphCanonicalRmsNormBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address, out-of-place canonical BF16
/// RMSNorm CUDA Graph capture.
pub struct OwnedGraphCanonicalRmsNormBf16Capture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphCanonicalRmsNormBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalRmsNormBf16Capture {
    /// Captures the one immutable canonical BF16 RMSNorm node.
    pub fn enqueue_canonical_rms_norm_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str =
            "OwnedGraphCanonicalRmsNormBf16Capture::enqueue_canonical_rms_norm_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph canonical BF16 RMSNorm capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph canonical BF16 RMSNorm enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed canonical BF16 RMSNorm graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_canonical_rms_norm_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/input/weight/output quartet
    /// into a by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedCanonicalRmsNormBf16Graph> {
        const OPERATION: &str = "OwnedGraphCanonicalRmsNormBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph canonical BF16 RMSNorm capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph canonical BF16 RMSNorm enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed canonical BF16 RMSNorm enqueue",
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
            let resources =
                take_owned_graph_canonical_rms_norm_bf16_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedCanonicalRmsNormBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Resources> {
        self.abort_once()?;
        take_owned_graph_canonical_rms_norm_bf16_resources(
            &mut self.resources,
            "OwnedGraphCanonicalRmsNormBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalRmsNormBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphCanonicalRmsNormBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address canonical BF16 RMSNorm CUDA Graph awaiting
/// instantiate or close.
pub struct OwnedCapturedCanonicalRmsNormBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphCanonicalRmsNormBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedCanonicalRmsNormBf16Graph {
    /// Instantiates this graph while retaining its fixed stream/input/weight/
    /// output quartet by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_canonical_rms_norm_bf16_resources(
                &mut self.resources,
                "OwnedCapturedCanonicalRmsNormBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphCanonicalRmsNormBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedCanonicalRmsNormBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_canonical_rms_norm_bf16_resources(
                &mut self.resources,
                "OwnedCapturedCanonicalRmsNormBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedCanonicalRmsNormBf16Graph::close",
            ))
        }
    }
}

/// By-value fixed-address canonical BF16 RMSNorm CUDA Graph executable.
///
/// It replays only the capture-time generic eager RMSNorm input and weight
/// allocations. SmolLM2 and Fixed37 profiles, fused RMSNorm, C07 executor
/// integration, fresh input staging, mutable spans, node updates, and profile
/// selection stay outside this narrow C05 ownership slice.
pub struct OwnedGraphCanonicalRmsNormBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphCanonicalRmsNormBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalRmsNormBf16Exec {
    /// Replays the fixed-address canonical BF16 RMSNorm graph once.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphCanonicalRmsNormBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph canonical BF16 RMSNorm transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph canonical BF16 RMSNorm exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphCanonicalRmsNormBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource quartet only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphCanonicalRmsNormBf16Exec::close",
                "an earlier graph canonical BF16 RMSNorm transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_canonical_rms_norm_bf16_resources(
                &mut self.resources,
                "OwnedGraphCanonicalRmsNormBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalRmsNormBf16Exec::close",
            ))
        }
    }
}

/// Completion owner for one [`OwnedGraphCanonicalRmsNormBf16Exec`] replay.
pub struct OwnedGraphCanonicalRmsNormBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphCanonicalRmsNormBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalRmsNormBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalRmsNormBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphCanonicalRmsNormBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_canonical_rms_norm_bf16_resources(
    resources: &mut Option<OwnedGraphCanonicalRmsNormBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph canonical BF16 RMSNorm owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_canonical_rms_norm_bf16_capture_preflight(
    stream: &CudaStream,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, input.context_owner(), operation)?;
    ensure_same_context(&stream.context, weight.context_owner(), operation)?;
    ensure_same_context(&stream.context, output.context_owner(), operation)?;
    input.ensure_idle_for_operation(operation)?;
    weight.ensure_idle_for_operation(operation)?;
    output.ensure_idle_for_operation(operation)?;
    let element_count = row_count.checked_mul(hidden_size).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "row_count * hidden_size overflows the canonical BF16 RMSNorm element range",
        )
    })?;
    let matrix_bytes = element_count
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "row_count * hidden_size overflows the canonical BF16 RMSNorm byte range",
            )
        })?;
    let weight_bytes = hidden_size
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "hidden_size overflows the canonical BF16 RMSNorm weight byte range",
            )
        })?;
    if row_count == 0
        || hidden_size == 0
        || matrix_bytes > input.byte_len()
        || weight_bytes > weight.byte_len()
        || matrix_bytes > output.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "row_count={row_count}, hidden_size={hidden_size} require matrix/weight BF16 capacities {matrix_bytes}/{weight_bytes} bytes, but input/weight/output capacities are {}/{}/{} bytes",
                input.byte_len(),
                weight.byte_len(),
                output.byte_len(),
            ),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(CudaError::invalid_argument(
            operation,
            "epsilon must be finite and greater than zero",
        ));
    }
    Ok(())
}

/// A by-value stream and two fixed device buffers recovered from one known
/// deterministic BF16 argmax graph lifecycle transition.
///
/// This recovery bundle exposes no graph-visible pointer, mutable span, or
/// fresh replay input. The BF16 logits and U32 result records remain distinct
/// throughout capture, graph, and exec ownership.
pub struct OwnedGraphBf16ArgmaxResources {
    stream: CudaStream,
    logits: CudaDeviceBuffer,
    results: CudaDeviceBuffer,
}

impl OwnedGraphBf16ArgmaxResources {
    fn new(stream: CudaStream, logits: CudaDeviceBuffer, results: CudaDeviceBuffer) -> Self {
        Self {
            stream,
            logits,
            results,
        }
    }

    /// Returns the exact stream, logits, and result records after a known
    /// native graph-lease release.
    #[must_use]
    pub fn into_parts(self) -> (CudaStream, CudaDeviceBuffer, CudaDeviceBuffer) {
        let Self {
            stream,
            logits,
            results,
        } = self;
        (stream, logits, results)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, logits, results) = self.into_parts();
        results.close()?;
        logits.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address deterministic BF16 argmax graph
/// capture.
///
/// Only Rust-side preflight errors recover the untouched resource trio. Once
/// native begin is attempted, ambiguous CUDA state retains all raw addresses
/// fail-closed.
#[must_use]
pub struct OwnedGraphBf16ArgmaxCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphBf16ArgmaxResources>,
}

impl OwnedGraphBf16ArgmaxCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphBf16ArgmaxResources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/logits/results trio only when native
    /// capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphBf16ArgmaxResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphBf16ArgmaxCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphBf16ArgmaxCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphBf16ArgmaxCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph deterministic BF16 argmax capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphBf16ArgmaxCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address deterministic BF16 argmax CUDA
/// Graph capture.
pub struct OwnedGraphBf16ArgmaxCapture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphBf16ArgmaxResources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16ArgmaxCapture {
    /// Captures the one immutable deterministic BF16 argmax node.
    pub fn enqueue_bf16_argmax(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphBf16ArgmaxCapture::enqueue_bf16_argmax";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph deterministic BF16 argmax capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph deterministic BF16 argmax enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed deterministic BF16 argmax graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_bf16_argmax();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/logits/results trio into a
    /// by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedBf16ArgmaxGraph> {
        const OPERATION: &str = "OwnedGraphBf16ArgmaxCapture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph deterministic BF16 argmax capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph deterministic BF16 argmax enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed deterministic BF16 argmax enqueue",
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
            let resources = take_owned_graph_bf16_argmax_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedBf16ArgmaxGraph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphBf16ArgmaxResources> {
        self.abort_once()?;
        take_owned_graph_bf16_argmax_resources(
            &mut self.resources,
            "OwnedGraphBf16ArgmaxCapture::abort",
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
            Err(CudaError::unavailable("OwnedGraphBf16ArgmaxCapture::abort"))
        }
    }
}

impl Drop for OwnedGraphBf16ArgmaxCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address deterministic BF16 argmax CUDA Graph awaiting
/// instantiate or close.
pub struct OwnedCapturedBf16ArgmaxGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphBf16ArgmaxResources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedBf16ArgmaxGraph {
    /// Instantiates this graph while retaining its fixed stream/logits/results
    /// trio by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphBf16ArgmaxExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_bf16_argmax_resources(
                &mut self.resources,
                "OwnedCapturedBf16ArgmaxGraph::instantiate",
            )?;
            Ok(OwnedGraphBf16ArgmaxExec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16ArgmaxGraph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16ArgmaxResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_argmax_resources(
                &mut self.resources,
                "OwnedCapturedBf16ArgmaxGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16ArgmaxGraph::close",
            ))
        }
    }
}

/// By-value fixed-address deterministic BF16 argmax CUDA Graph executable.
///
/// It replays only the capture-time BF16 logits and U32 result allocations.
/// Row gather, token/status transfer, completion dependencies, fresh input
/// staging, mutable spans, node updates, sampling policy, and C07 executor
/// integration stay outside this narrow C05 ownership slice.
pub struct OwnedGraphBf16ArgmaxExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphBf16ArgmaxResources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16ArgmaxExec {
    /// Replays the fixed-address deterministic BF16 argmax graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphBf16ArgmaxLaunch<'exec>> {
        const OPERATION: &str = "OwnedGraphBf16ArgmaxExec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph deterministic BF16 argmax transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph deterministic BF16 argmax exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphBf16ArgmaxLaunch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource trio only after native
    /// close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16ArgmaxResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphBf16ArgmaxExec::close",
                "an earlier graph deterministic BF16 argmax transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_argmax_resources(
                &mut self.resources,
                "OwnedGraphBf16ArgmaxExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable("OwnedGraphBf16ArgmaxExec::close"))
        }
    }
}

/// Completion owner for one [`OwnedGraphBf16ArgmaxExec`] replay.
pub struct OwnedGraphBf16ArgmaxLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphBf16ArgmaxExec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16ArgmaxLaunch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable("OwnedGraphBf16ArgmaxLaunch::finish"))
        }
    }
}

impl Drop for OwnedGraphBf16ArgmaxLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_bf16_argmax_resources(
    resources: &mut Option<OwnedGraphBf16ArgmaxResources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphBf16ArgmaxResources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph deterministic BF16 argmax owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_bf16_argmax_capture_preflight(
    stream: &CudaStream,
    logits: &CudaDeviceBuffer,
    results: &CudaDeviceBuffer,
    row_count: u64,
    vocabulary_size: u64,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(&stream.context, logits.context_owner(), operation)?;
    ensure_same_context(&stream.context, results.context_owner(), operation)?;
    logits.ensure_idle_for_operation(operation)?;
    results.ensure_idle_for_operation(operation)?;
    if row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "row_count must be non-zero for a one-node deterministic BF16 argmax graph",
        ));
    }
    if vocabulary_size == 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "vocabulary_size must be non-zero",
        ));
    }
    if vocabulary_size > u64::from(u32::MAX) {
        return Err(CudaError::out_of_range(
            operation,
            "vocabulary_size exceeds the U32 token-id contract",
        ));
    }
    let logit_elements = row_count.checked_mul(vocabulary_size).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "row_count * vocabulary_size overflows the deterministic BF16 argmax element range",
        )
    })?;
    let logits_bytes = logit_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "row_count * vocabulary_size overflows the deterministic BF16 argmax BF16 byte range",
            )
        })?;
    let result_words = row_count.checked_mul(2).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "row_count * two-U32 result records overflows the deterministic BF16 argmax result range",
        )
    })?;
    let results_bytes = result_words
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "row_count * two-U32 result records overflows the deterministic BF16 argmax byte range",
            )
        })?;
    if logits_bytes > logits.byte_len() || results_bytes > results.byte_len() {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "row_count={row_count}, vocabulary_size={vocabulary_size} require BF16/U32 capacities {logits_bytes}/{results_bytes} bytes, but logits/results capacities are {}/{} bytes",
                logits.byte_len(),
                results.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and three fixed device buffers recovered from one known
/// BF16 row-gather graph lifecycle transition.
///
/// This recovery bundle exposes no graph-visible pointer, mutable span, or
/// host-index reference. The BF16 input, U32 indices, and BF16 output remain
/// distinct throughout capture, graph, and exec ownership.
pub struct OwnedGraphBf16RowGatherResources {
    stream: CudaStream,
    input: CudaDeviceBuffer,
    row_indices: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
}

impl OwnedGraphBf16RowGatherResources {
    fn new(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            input,
            row_indices,
            output,
        }
    }

    /// Returns the exact stream, input, row indices, and output after a known
    /// native graph-lease release.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            input,
            row_indices,
            output,
        } = self;
        (stream, input, row_indices, output)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, input, row_indices, output) = self.into_parts();
        output.close()?;
        row_indices.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address BF16 row-gather graph capture.
///
/// Only Rust-side preflight errors recover the untouched resource quartet.
/// Once native begin is attempted, ambiguous CUDA state retains all raw
/// addresses fail-closed.
#[must_use]
pub struct OwnedGraphBf16RowGatherCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphBf16RowGatherResources>,
}

impl OwnedGraphBf16RowGatherCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphBf16RowGatherResources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/input/row-indices/output quartet only
    /// when native capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphBf16RowGatherResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphBf16RowGatherCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphBf16RowGatherCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphBf16RowGatherCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 row-gather capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphBf16RowGatherCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address, out-of-place BF16 row-gather
/// CUDA Graph capture.
pub struct OwnedGraphBf16RowGatherCapture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphBf16RowGatherResources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherCapture {
    /// Captures the one immutable BF16 row-gather node.
    pub fn enqueue_bf16_row_gather(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherCapture::enqueue_bf16_row_gather";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 row-gather capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 row-gather enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 row-gather graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_bf16_row_gather();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/input/row-indices/output
    /// quartet into a by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedBf16RowGatherGraph> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherCapture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 row-gather capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 row-gather enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 row-gather enqueue",
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
            let resources =
                take_owned_graph_bf16_row_gather_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedBf16RowGatherGraph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphBf16RowGatherResources> {
        self.abort_once()?;
        take_owned_graph_bf16_row_gather_resources(
            &mut self.resources,
            "OwnedGraphBf16RowGatherCapture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherCapture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16RowGatherCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 row-gather CUDA Graph awaiting instantiate or
/// close.
pub struct OwnedCapturedBf16RowGatherGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphBf16RowGatherResources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedBf16RowGatherGraph {
    /// Instantiates this graph while retaining its fixed stream/input/
    /// row-indices/output quartet by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphBf16RowGatherExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_bf16_row_gather_resources(
                &mut self.resources,
                "OwnedCapturedBf16RowGatherGraph::instantiate",
            )?;
            Ok(OwnedGraphBf16RowGatherExec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16RowGatherGraph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16RowGatherResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_row_gather_resources(
                &mut self.resources,
                "OwnedCapturedBf16RowGatherGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16RowGatherGraph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 row-gather CUDA Graph executable.
///
/// It replays only the capture-time BF16 input, U32 index, and BF16 output
/// allocations. H2D/D2H staging, fresh inputs, spans, offsets, argmax,
/// token/status transfer, completion dependencies, node updates, sampling,
/// and C07 executor integration stay outside this narrow C05 ownership slice.
pub struct OwnedGraphBf16RowGatherExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphBf16RowGatherResources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherExec {
    /// Replays the fixed-address BF16 row-gather graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphBf16RowGatherLaunch<'exec>> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherExec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 row-gather transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 row-gather exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphBf16RowGatherLaunch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource quartet only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16RowGatherResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphBf16RowGatherExec::close",
                "an earlier graph BF16 row-gather transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_row_gather_resources(
                &mut self.resources,
                "OwnedGraphBf16RowGatherExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable("OwnedGraphBf16RowGatherExec::close"))
        }
    }
}

/// Completion owner for one [`OwnedGraphBf16RowGatherExec`] replay.
pub struct OwnedGraphBf16RowGatherLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphBf16RowGatherExec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherLaunch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherLaunch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16RowGatherLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_bf16_row_gather_resources(
    resources: &mut Option<OwnedGraphBf16RowGatherResources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphBf16RowGatherResources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 row-gather owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_bf16_row_gather_capture_preflight(
    stream: &CudaStream,
    input: &CudaDeviceBuffer,
    row_indices: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    row_indices_host: &[u32],
    input_row_count: u64,
    output_row_count: u64,
    column_count: u64,
    operation: &'static str,
) -> CudaResult<()> {
    if input
        .native_handle()
        .same_allocation(row_indices.native_handle())
        || input
            .native_handle()
            .same_allocation(output.native_handle())
        || row_indices
            .native_handle()
            .same_allocation(output.native_handle())
    {
        return Err(CudaError::invalid_argument(
            operation,
            "input, row_indices, and output must be distinct fixed device allocations",
        ));
    }
    ensure_same_context(&stream.context, input.context_owner(), operation)?;
    ensure_same_context(&stream.context, row_indices.context_owner(), operation)?;
    ensure_same_context(&stream.context, output.context_owner(), operation)?;
    input.ensure_idle_for_operation(operation)?;
    row_indices.ensure_idle_for_operation(operation)?;
    output.ensure_idle_for_operation(operation)?;
    if input_row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "input_row_count must be non-zero for a one-node BF16 row-gather graph",
        ));
    }
    if output_row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "row_indices_host must contain at least one output row for a one-node BF16 row-gather graph",
        ));
    }
    if column_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "column_count must be non-zero for a one-node BF16 row-gather graph",
        ));
    }
    // This uses the exact eager safe-mirror validation and deliberately does
    // not imply that the caller's already-staged device bytes equal the host
    // mirror. The capture owner never retains the host slice.
    crate::batch::validate_gather_indices(row_indices_host, input_row_count)?;
    let input_elements = input_row_count.checked_mul(column_count).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "input_row_count * column_count overflows the BF16 row-gather input element range",
        )
    })?;
    let output_elements = output_row_count.checked_mul(column_count).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "output_row_count * column_count overflows the BF16 row-gather output element range",
        )
    })?;
    let input_bytes = input_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "input_row_count * column_count overflows the BF16 row-gather input byte range",
            )
        })?;
    let row_indices_bytes = output_row_count
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count overflows the BF16 row-gather U32 index byte range",
            )
        })?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count * column_count overflows the BF16 row-gather output byte range",
            )
        })?;
    if input_bytes > input.byte_len()
        || row_indices_bytes > row_indices.byte_len()
        || output_bytes > output.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "input_row_count={input_row_count}, output_row_count={output_row_count}, column_count={column_count} require BF16/U32/BF16 capacities {input_bytes}/{row_indices_bytes}/{output_bytes} bytes, but input/row_indices/output capacities are {}/{}/{} bytes",
                input.byte_len(),
                row_indices.byte_len(),
                output.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and four fixed device buffers recovered from one known
/// BF16 row-gather then deterministic-argmax graph lifecycle transition.
///
/// This recovery bundle exposes no graph-visible pointer, mutable span, or
/// host-index reference. The BF16 input, U32 indices, BF16 gathered logits,
/// and U32 result records remain distinct throughout capture, graph, and exec
/// ownership. `gathered_logits` is intentionally shared by the two captured
/// nodes only; it never crosses into two independent graph owners.
pub struct OwnedGraphBf16RowGatherArgmaxResources {
    stream: CudaStream,
    input: CudaDeviceBuffer,
    row_indices: CudaDeviceBuffer,
    gathered_logits: CudaDeviceBuffer,
    results: CudaDeviceBuffer,
}

impl OwnedGraphBf16RowGatherArgmaxResources {
    fn new(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        gathered_logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            input,
            row_indices,
            gathered_logits,
            results,
        }
    }

    /// Returns the exact stream, input, row indices, gathered logits, and
    /// argmax results after a known native graph-lease release.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            input,
            row_indices,
            gathered_logits,
            results,
        } = self;
        (stream, input, row_indices, gathered_logits, results)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, input, row_indices, gathered_logits, results) = self.into_parts();
        results.close()?;
        gathered_logits.close()?;
        row_indices.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address BF16 row-gather then
/// deterministic-argmax graph capture.
///
/// Only Rust-side preflight errors recover the untouched resource quintet.
/// Once native begin is attempted, ambiguous CUDA state retains all raw
/// addresses fail-closed.
#[must_use]
pub struct OwnedGraphBf16RowGatherArgmaxCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphBf16RowGatherArgmaxResources>,
}

impl OwnedGraphBf16RowGatherArgmaxCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphBf16RowGatherArgmaxResources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns the untouched stream/input/row-indices/gathered/results quintet
    /// only when native capture ownership was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphBf16RowGatherArgmaxResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphBf16RowGatherArgmaxCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphBf16RowGatherArgmaxCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphBf16RowGatherArgmaxCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 row-gather -> argmax capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphBf16RowGatherArgmaxCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address, two-node BF16 row-gather then
/// deterministic-argmax CUDA Graph capture.
pub struct OwnedGraphBf16RowGatherArgmaxCapture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphBf16RowGatherArgmaxResources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherArgmaxCapture {
    /// Captures the one immutable BF16 row-gather then deterministic-argmax
    /// node chain.
    pub fn enqueue_bf16_row_gather_argmax(&mut self) -> CudaResult<()> {
        const OPERATION: &str =
            "OwnedGraphBf16RowGatherArgmaxCapture::enqueue_bf16_row_gather_argmax";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 row-gather -> argmax capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 row-gather -> argmax enqueue failed and this partial capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 row-gather -> argmax graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_bf16_row_gather_argmax();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                // Native may already have recorded the gather node before the
                // argmax node failed. That partial capture is terminal until
                // its one-shot abort establishes known lease release.
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed stream/input/row-indices/
    /// gathered-logits/results quintet into a by-value captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedBf16RowGatherArgmaxGraph> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherArgmaxCapture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 row-gather -> argmax capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 row-gather -> argmax enqueue failed and this partial capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 row-gather -> argmax enqueue",
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
            let resources =
                take_owned_graph_bf16_row_gather_argmax_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedBf16RowGatherArgmaxGraph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxResources> {
        self.abort_once()?;
        take_owned_graph_bf16_row_gather_argmax_resources(
            &mut self.resources,
            "OwnedGraphBf16RowGatherArgmaxCapture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherArgmaxCapture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16RowGatherArgmaxCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 row-gather then deterministic-argmax CUDA
/// Graph awaiting instantiate or close.
pub struct OwnedCapturedBf16RowGatherArgmaxGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphBf16RowGatherArgmaxResources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedBf16RowGatherArgmaxGraph {
    /// Instantiates this graph while retaining its fixed stream/input/
    /// row-indices/gathered-logits/results quintet by value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_bf16_row_gather_argmax_resources(
                &mut self.resources,
                "OwnedCapturedBf16RowGatherArgmaxGraph::instantiate",
            )?;
            Ok(OwnedGraphBf16RowGatherArgmaxExec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16RowGatherArgmaxGraph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_row_gather_argmax_resources(
                &mut self.resources,
                "OwnedCapturedBf16RowGatherArgmaxGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16RowGatherArgmaxGraph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 row-gather then deterministic-argmax CUDA
/// Graph executable.
///
/// It replays only the capture-time BF16 input, U32 index, gathered BF16, and
/// U32 result allocations. H2D/D2H staging, token/status transfer, host
/// completion semantics, fresh inputs, spans, offsets, node updates, sampling,
/// and C07 executor integration stay outside this narrow C05 ownership slice.
pub struct OwnedGraphBf16RowGatherArgmaxExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphBf16RowGatherArgmaxResources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherArgmaxExec {
    /// Replays the fixed-address BF16 row-gather then deterministic-argmax
    /// graph once.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphBf16RowGatherArgmaxLaunch<'exec>> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherArgmaxExec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 row-gather -> argmax transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 row-gather -> argmax exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphBf16RowGatherArgmaxLaunch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource quintet only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphBf16RowGatherArgmaxExec::close",
                "an earlier graph BF16 row-gather -> argmax transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_row_gather_argmax_resources(
                &mut self.resources,
                "OwnedGraphBf16RowGatherArgmaxExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherArgmaxExec::close",
            ))
        }
    }
}

/// Completion owner for one [`OwnedGraphBf16RowGatherArgmaxExec`] replay.
pub struct OwnedGraphBf16RowGatherArgmaxLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphBf16RowGatherArgmaxExec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherArgmaxLaunch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherArgmaxLaunch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16RowGatherArgmaxLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_bf16_row_gather_argmax_resources(
    resources: &mut Option<OwnedGraphBf16RowGatherArgmaxResources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphBf16RowGatherArgmaxResources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 row-gather -> argmax owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_bf16_row_gather_argmax_capture_preflight(
    stream: &CudaStream,
    input: &CudaDeviceBuffer,
    row_indices: &CudaDeviceBuffer,
    gathered_logits: &CudaDeviceBuffer,
    results: &CudaDeviceBuffer,
    row_indices_host: &[u32],
    input_row_count: u64,
    output_row_count: u64,
    vocabulary_size: u64,
    operation: &'static str,
) -> CudaResult<()> {
    if input
        .native_handle()
        .same_allocation(row_indices.native_handle())
        || input
            .native_handle()
            .same_allocation(gathered_logits.native_handle())
        || input
            .native_handle()
            .same_allocation(results.native_handle())
        || row_indices
            .native_handle()
            .same_allocation(gathered_logits.native_handle())
        || row_indices
            .native_handle()
            .same_allocation(results.native_handle())
        || gathered_logits
            .native_handle()
            .same_allocation(results.native_handle())
    {
        return Err(CudaError::invalid_argument(
            operation,
            "input, row_indices, gathered_logits, and results must be distinct fixed device allocations",
        ));
    }
    ensure_same_context(&stream.context, input.context_owner(), operation)?;
    ensure_same_context(&stream.context, row_indices.context_owner(), operation)?;
    ensure_same_context(&stream.context, gathered_logits.context_owner(), operation)?;
    ensure_same_context(&stream.context, results.context_owner(), operation)?;
    input.ensure_idle_for_operation(operation)?;
    row_indices.ensure_idle_for_operation(operation)?;
    gathered_logits.ensure_idle_for_operation(operation)?;
    results.ensure_idle_for_operation(operation)?;
    if input_row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "input_row_count must be non-zero for a two-node BF16 row-gather -> argmax graph",
        ));
    }
    if output_row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "row_indices_host must contain at least one output row for a two-node BF16 row-gather -> argmax graph",
        ));
    }
    if vocabulary_size == 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "vocabulary_size must be non-zero",
        ));
    }
    if vocabulary_size > u64::from(u32::MAX) {
        return Err(CudaError::out_of_range(
            operation,
            "vocabulary_size exceeds the U32 token-id contract",
        ));
    }
    // This uses the exact eager safe-mirror validation and deliberately does
    // not imply that the caller's already-staged device bytes equal the host
    // mirror. The capture owner never retains the host slice.
    crate::batch::validate_gather_indices(row_indices_host, input_row_count)?;
    let input_elements = input_row_count.checked_mul(vocabulary_size).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "input_row_count * vocabulary_size overflows the BF16 row-gather -> argmax input element range",
        )
    })?;
    let gathered_elements = output_row_count
        .checked_mul(vocabulary_size)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count * vocabulary_size overflows the BF16 row-gather -> argmax gathered element range",
            )
        })?;
    let input_bytes = input_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "input_row_count * vocabulary_size overflows the BF16 row-gather -> argmax input byte range",
            )
        })?;
    let row_indices_bytes = output_row_count
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count overflows the BF16 row-gather -> argmax U32 index byte range",
            )
        })?;
    let gathered_bytes = gathered_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count * vocabulary_size overflows the BF16 row-gather -> argmax gathered byte range",
            )
        })?;
    let results_bytes = output_row_count
        .checked_mul(std::mem::size_of::<Bf16ArgmaxResult>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count overflows the BF16 row-gather -> argmax result byte range",
            )
        })?;
    if input_bytes > input.byte_len()
        || row_indices_bytes > row_indices.byte_len()
        || gathered_bytes > gathered_logits.byte_len()
        || results_bytes > results.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "input_row_count={input_row_count}, output_row_count={output_row_count}, vocabulary_size={vocabulary_size} require BF16/U32/BF16/result capacities {input_bytes}/{row_indices_bytes}/{gathered_bytes}/{results_bytes} bytes, but input/row_indices/gathered_logits/results capacities are {}/{}/{}/{} bytes",
                input.byte_len(),
                row_indices.byte_len(),
                gathered_logits.byte_len(),
                results.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream, four fixed device allocations, and one exact pinned
/// result allocation recovered after a known C05-16 graph lifecycle release.
///
/// The pinned allocation is graph-owned rather than a general D2H staging
/// buffer while this bundle is inside capture, graph, or exec ownership.
pub struct OwnedGraphBf16RowGatherArgmaxD2HResources {
    stream: CudaStream,
    input: CudaDeviceBuffer,
    row_indices: CudaDeviceBuffer,
    gathered_logits: CudaDeviceBuffer,
    results: CudaDeviceBuffer,
    pinned_results: CudaPinnedHostBuffer,
}

impl OwnedGraphBf16RowGatherArgmaxD2HResources {
    fn new(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        gathered_logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
        pinned_results: CudaPinnedHostBuffer,
    ) -> Self {
        Self {
            stream,
            input,
            row_indices,
            gathered_logits,
            results,
            pinned_results,
        }
    }

    /// Returns all raw resources only after native graph lease release is
    /// known. Until then neither the result device allocation nor the pinned
    /// destination can be independently reused or closed.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaPinnedHostBuffer,
    ) {
        let Self {
            stream,
            input,
            row_indices,
            gathered_logits,
            results,
            pinned_results,
        } = self;
        (
            stream,
            input,
            row_indices,
            gathered_logits,
            results,
            pinned_results,
        )
    }

    /// Explicitly destroys recovered allocations before the capture stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, input, row_indices, gathered_logits, results, pinned_results) =
            self.into_parts();
        pinned_results.close()?;
        results.close()?;
        gathered_logits.close()?;
        row_indices.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning one by-value fixed-address C05-16 graph capture.
///
/// Only pure Rust preflight errors return the untouched resource sextet.
/// Once native capture entry was attempted, all raw addresses remain
/// fail-closed because CUDA may have retained the live capture owner.
#[must_use]
pub struct OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphBf16RowGatherArgmaxD2HResources>,
}

impl OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphBf16RowGatherArgmaxD2HResources) -> Self {
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

    /// The rejected preflight or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphBf16RowGatherArgmaxD2HResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 row-gather -> argmax -> D2H capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active C05-16 three-node graph capture.
pub struct OwnedGraphBf16RowGatherArgmaxD2HCapture {
    // The native owner drops before child wrappers: on ambiguity it retains
    // every raw address and the Rust resources must not drop independently.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphBf16RowGatherArgmaxD2HResources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherArgmaxD2HCapture {
    /// Records exactly gather -> argmax -> fixed result D2H once.
    pub fn enqueue_bf16_row_gather_argmax_d2h(&mut self) -> CudaResult<()> {
        const OPERATION: &str =
            "OwnedGraphBf16RowGatherArgmaxD2HCapture::enqueue_bf16_row_gather_argmax_d2h";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 row-gather -> argmax -> D2H capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior three-node graph enqueue failed and this partial capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 row-gather -> argmax -> D2H graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_bf16_row_gather_argmax_d2h();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                // A CUDA capture can retain any recorded prefix, including a
                // successfully recorded gather or argmax before D2H fails.
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the resource sextet into a captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedBf16RowGatherArgmaxD2HGraph> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherArgmaxD2HCapture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 row-gather -> argmax -> D2H capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior three-node graph enqueue failed and this partial capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 row-gather -> argmax -> D2H enqueue",
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
            let resources = take_owned_graph_bf16_row_gather_argmax_d2h_resources(
                &mut self.resources,
                OPERATION,
            )?;
            Ok(OwnedCapturedBf16RowGatherArgmaxD2HGraph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after known release evidence.
    pub fn abort(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HResources> {
        self.abort_once()?;
        take_owned_graph_bf16_row_gather_argmax_d2h_resources(
            &mut self.resources,
            "OwnedGraphBf16RowGatherArgmaxD2HCapture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherArgmaxD2HCapture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16RowGatherArgmaxD2HCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value captured C05-16 graph awaiting instantiate or close.
pub struct OwnedCapturedBf16RowGatherArgmaxD2HGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphBf16RowGatherArgmaxD2HResources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedBf16RowGatherArgmaxD2HGraph {
    /// Instantiates while retaining the exact stream/allocation sextet.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_bf16_row_gather_argmax_d2h_resources(
                &mut self.resources,
                "OwnedCapturedBf16RowGatherArgmaxD2HGraph::instantiate",
            )?;
            Ok(OwnedGraphBf16RowGatherArgmaxD2HExec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16RowGatherArgmaxD2HGraph::instantiate",
            ))
        }
    }

    /// Closes the captured graph and returns resources after known lease release.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_row_gather_argmax_d2h_resources(
                &mut self.resources,
                "OwnedCapturedBf16RowGatherArgmaxD2HGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16RowGatherArgmaxD2HGraph::close",
            ))
        }
    }
}

/// By-value executable for one fixed gather -> argmax -> result-D2H graph.
///
/// Raw result records remain inside retained pinned storage. They are exposed
/// only through [`OwnedGraphBf16RowGatherArgmaxD2HCompletion`] after a known
/// graph completion, never as scheduler-ready tokens.
pub struct OwnedGraphBf16RowGatherArgmaxD2HExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphBf16RowGatherArgmaxD2HResources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16RowGatherArgmaxD2HExec {
    /// Launches the fixed three-node graph once.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HLaunch<'exec>> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherArgmaxD2HExec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 row-gather -> argmax -> D2H transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 row-gather -> argmax -> D2H exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphBf16RowGatherArgmaxD2HLaunch {
                    native,
                    exec: Some(self),
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns resources after known graph release.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphBf16RowGatherArgmaxD2HExec::close",
                "an earlier graph BF16 row-gather -> argmax -> D2H transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_row_gather_argmax_d2h_resources(
                &mut self.resources,
                "OwnedGraphBf16RowGatherArgmaxD2HExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherArgmaxD2HExec::close",
            ))
        }
    }
}

/// In-flight completion owner for one C05-16 graph replay.
pub struct OwnedGraphBf16RowGatherArgmaxD2HLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: Option<&'exec mut OwnedGraphBf16RowGatherArgmaxD2HExec>,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl<'exec> OwnedGraphBf16RowGatherArgmaxD2HLaunch<'exec> {
    /// Waits for completion and returns the sole raw-result read receipt.
    ///
    /// The receipt keeps the executable exclusively borrowed, so a new replay
    /// or close cannot race the fixed pinned-host result view.
    pub fn finish(mut self) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HCompletion<'exec>> {
        self.complete_once()?;
        let exec = self.exec.take().ok_or_else(|| {
            CudaError::invalid_state(
                "OwnedGraphBf16RowGatherArgmaxD2HLaunch::finish",
                "the graph launch completion owner was already consumed",
            )
        })?;
        Ok(OwnedGraphBf16RowGatherArgmaxD2HCompletion { exec })
    }

    fn complete_once(&mut self) -> CudaResult<()> {
        let Some(exec) = self.exec.as_deref_mut() else {
            return if self.active {
                Err(CudaError::invalid_state(
                    "OwnedGraphBf16RowGatherArgmaxD2HLaunch::finish",
                    "the graph launch completion owner lost its executable borrow",
                ))
            } else {
                Ok(())
            };
        };
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            let result = self.native.complete();
            if result.is_err() {
                exec.terminal = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            exec.terminal = true;
            Err(CudaError::unavailable(
                "OwnedGraphBf16RowGatherArgmaxD2HLaunch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16RowGatherArgmaxD2HLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

/// Completion-scoped raw read view for C05-16 result records.
///
/// This provides no token/status interpretation and exposes no reusable
/// pinned allocation. Native continues retaining the graph's permanent pinned
/// lease until the executable is explicitly closed.
#[must_use]
pub struct OwnedGraphBf16RowGatherArgmaxD2HCompletion<'exec> {
    exec: &'exec mut OwnedGraphBf16RowGatherArgmaxD2HExec,
}

impl OwnedGraphBf16RowGatherArgmaxD2HCompletion<'_> {
    /// Copies the complete fixed result-record payload into `destination`.
    ///
    /// The destination must have exactly the capture-time result byte length.
    /// This is a byte transport only; callers must not treat its success as
    /// token validation, scheduler commit, or a C07 completion boundary.
    pub fn read_result_bytes(&mut self, destination: &mut [u8]) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphBf16RowGatherArgmaxD2HCompletion::read_result_bytes";
        if self.exec.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the graph executable is terminal after an uncertain transition",
            ));
        }
        let expected_len = match self.exec.resources.as_ref() {
            Some(resources) => resources.pinned_results.byte_len(),
            None => {
                self.exec.terminal = true;
                return Err(CudaError::invalid_state(
                    OPERATION,
                    "the graph executable lost its retained pinned result allocation",
                ));
            }
        };
        let actual_len = u64::try_from(destination.len()).map_err(|_| {
            CudaError::out_of_range(OPERATION, "result destination length does not fit u64")
        })?;
        if actual_len != expected_len {
            return Err(CudaError::out_of_range(
                OPERATION,
                format!(
                    "result destination length {actual_len} must equal the fixed graph D2H result length {expected_len}",
                ),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self
                .exec
                .native
                .read_bf16_row_gather_argmax_d2h_results(destination);
            // Local Rust shape checks above are retryable and deliberately do
            // not enter native code. Once the completion receipt crosses the
            // FFI boundary, any status or metadata failure leaves the raw
            // pinned-result observation uncertain; retain every graph lease
            // fail-closed rather than admitting another replay or close.
            if result.is_err() {
                self.exec.terminal = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = destination;
            Err(CudaError::unavailable(OPERATION))
        }
    }
}

fn take_owned_graph_bf16_row_gather_argmax_d2h_resources(
    resources: &mut Option<OwnedGraphBf16RowGatherArgmaxD2HResources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphBf16RowGatherArgmaxD2HResources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 row-gather -> argmax -> D2H owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_bf16_row_gather_argmax_d2h_capture_preflight(
    stream: &CudaStream,
    input: &CudaDeviceBuffer,
    row_indices: &CudaDeviceBuffer,
    gathered_logits: &CudaDeviceBuffer,
    results: &CudaDeviceBuffer,
    pinned_results: &CudaPinnedHostBuffer,
    row_indices_host: &[u32],
    input_row_count: u64,
    output_row_count: u64,
    vocabulary_size: u64,
    operation: &'static str,
) -> CudaResult<()> {
    if input
        .native_handle()
        .same_allocation(row_indices.native_handle())
        || input
            .native_handle()
            .same_allocation(gathered_logits.native_handle())
        || input
            .native_handle()
            .same_allocation(results.native_handle())
        || row_indices
            .native_handle()
            .same_allocation(gathered_logits.native_handle())
        || row_indices
            .native_handle()
            .same_allocation(results.native_handle())
        || gathered_logits
            .native_handle()
            .same_allocation(results.native_handle())
    {
        return Err(CudaError::invalid_argument(
            operation,
            "input, row_indices, gathered_logits, and results must be distinct fixed device allocations",
        ));
    }
    ensure_same_context(&stream.context, input.context_owner(), operation)?;
    ensure_same_context(&stream.context, row_indices.context_owner(), operation)?;
    ensure_same_context(&stream.context, gathered_logits.context_owner(), operation)?;
    ensure_same_context(&stream.context, results.context_owner(), operation)?;
    ensure_same_context(&stream.context, pinned_results.context_owner(), operation)?;
    input.ensure_idle_for_operation(operation)?;
    row_indices.ensure_idle_for_operation(operation)?;
    gathered_logits.ensure_idle_for_operation(operation)?;
    results.ensure_idle_for_operation(operation)?;
    pinned_results.ensure_idle_for_operation(operation)?;
    if input_row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "input_row_count must be non-zero for a three-node BF16 row-gather -> argmax -> D2H graph",
        ));
    }
    if output_row_count == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "row_indices_host must contain at least one output row for a three-node BF16 row-gather -> argmax -> D2H graph",
        ));
    }
    if vocabulary_size == 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "vocabulary_size must be non-zero",
        ));
    }
    if vocabulary_size > u64::from(u32::MAX) {
        return Err(CudaError::out_of_range(
            operation,
            "vocabulary_size exceeds the U32 token-id contract",
        ));
    }
    // This safe mirror proves shape and eager admissibility only. The graph
    // stores no host slice and reads the caller's independently staged device
    // U32 index bytes during replay.
    crate::batch::validate_gather_indices(row_indices_host, input_row_count)?;
    let input_elements = input_row_count
        .checked_mul(vocabulary_size)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "input_row_count * vocabulary_size overflows the BF16 input element range",
            )
        })?;
    let gathered_elements = output_row_count
        .checked_mul(vocabulary_size)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output_row_count * vocabulary_size overflows the gathered BF16 element range",
            )
        })?;
    let input_bytes = input_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "input element count overflows the BF16 input byte range",
            )
        })?;
    let row_indices_bytes = output_row_count
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output row count overflows the U32 index byte range",
            )
        })?;
    let gathered_bytes = gathered_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "gathered element count overflows the BF16 gathered byte range",
            )
        })?;
    let result_bytes = output_row_count
        .checked_mul(std::mem::size_of::<Bf16ArgmaxResult>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "output row count overflows the result-record byte range",
            )
        })?;
    if input_bytes > input.byte_len()
        || row_indices_bytes > row_indices.byte_len()
        || gathered_bytes > gathered_logits.byte_len()
        || result_bytes > results.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "input_row_count={input_row_count}, output_row_count={output_row_count}, vocabulary_size={vocabulary_size} require BF16/U32/BF16/result capacities {input_bytes}/{row_indices_bytes}/{gathered_bytes}/{result_bytes} bytes, but input/row_indices/gathered_logits/results capacities are {}/{}/{}/{} bytes",
                input.byte_len(),
                row_indices.byte_len(),
                gathered_logits.byte_len(),
                results.byte_len(),
            ),
        ));
    }
    if pinned_results.byte_len() != result_bytes {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "pinned result buffer length {} must exactly equal the fixed result-record length {result_bytes}",
                pinned_results.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and five distinct fixed device buffers recovered from a
/// known BF16 indexed-RoPE graph lifecycle transition.
///
/// The host position mirror is validation-only and is never retained. The
/// input, cosine table, sine table, device positions, and output stay fixed
/// and inaccessible while capture, graph, or exec may retain their addresses.
pub struct OwnedGraphIndexedRopeBf16Resources {
    stream: CudaStream,
    input: CudaDeviceBuffer,
    cos: CudaDeviceBuffer,
    sin: CudaDeviceBuffer,
    positions: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
}

impl OwnedGraphIndexedRopeBf16Resources {
    fn new(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        cos: CudaDeviceBuffer,
        sin: CudaDeviceBuffer,
        positions: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            input,
            cos,
            sin,
            positions,
            output,
        }
    }

    /// Returns the exact fixed stream and device buffers after known native
    /// graph-lease release.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            input,
            cos,
            sin,
            positions,
            output,
        } = self;
        (stream, input, cos, sin, positions, output)
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, input, cos, sin, positions, output) = self.into_parts();
        output.close()?;
        positions.close()?;
        sin.close()?;
        cos.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address BF16 indexed-RoPE graph
/// capture.
///
/// Only Rust-side preflight failures recover the untouched resource sextet.
/// Once native begin is attempted, ambiguous CUDA state retains every raw
/// address fail-closed.
#[must_use]
pub struct OwnedGraphIndexedRopeBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphIndexedRopeBf16Resources>,
}

impl OwnedGraphIndexedRopeBf16CaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphIndexedRopeBf16Resources) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture ownership was never
    /// entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphIndexedRopeBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphIndexedRopeBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphIndexedRopeBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphIndexedRopeBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 indexed-RoPE capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphIndexedRopeBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address BF16 indexed-RoPE CUDA Graph
/// capture.
pub struct OwnedGraphIndexedRopeBf16Capture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphIndexedRopeBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphIndexedRopeBf16Capture {
    /// Captures the one immutable BF16 indexed-RoPE node.
    pub fn enqueue_indexed_rope_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphIndexedRopeBf16Capture::enqueue_indexed_rope_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 indexed-RoPE capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 indexed-RoPE enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 indexed-RoPE graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_indexed_rope_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed resource sextet into a by-value
    /// captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedIndexedRopeBf16Graph> {
        const OPERATION: &str = "OwnedGraphIndexedRopeBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 indexed-RoPE capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 indexed-RoPE enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 indexed-RoPE enqueue",
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
            let resources =
                take_owned_graph_indexed_rope_bf16_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedIndexedRopeBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphIndexedRopeBf16Resources> {
        self.abort_once()?;
        take_owned_graph_indexed_rope_bf16_resources(
            &mut self.resources,
            "OwnedGraphIndexedRopeBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphIndexedRopeBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphIndexedRopeBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 indexed-RoPE CUDA Graph awaiting instantiate
/// or close.
pub struct OwnedCapturedIndexedRopeBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphIndexedRopeBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedIndexedRopeBf16Graph {
    /// Instantiates the graph while retaining its fixed resource sextet by
    /// value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphIndexedRopeBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_indexed_rope_bf16_resources(
                &mut self.resources,
                "OwnedCapturedIndexedRopeBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphIndexedRopeBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedIndexedRopeBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphIndexedRopeBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_indexed_rope_bf16_resources(
                &mut self.resources,
                "OwnedCapturedIndexedRopeBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedIndexedRopeBf16Graph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 indexed-RoPE CUDA Graph executable.
///
/// It replays only capture-time device allocations. Host/device position
/// staging, result transfer, C07 executor wiring, node updates, fresh inputs,
/// sampling, scheduling, and eager fallback stay outside this narrow C05
/// ownership slice.
pub struct OwnedGraphIndexedRopeBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphIndexedRopeBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphIndexedRopeBf16Exec {
    /// Replays the fixed-address BF16 indexed-RoPE graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphIndexedRopeBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphIndexedRopeBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 indexed-RoPE transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 indexed-RoPE exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphIndexedRopeBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource sextet only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphIndexedRopeBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphIndexedRopeBf16Exec::close",
                "an earlier graph BF16 indexed-RoPE transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_indexed_rope_bf16_resources(
                &mut self.resources,
                "OwnedGraphIndexedRopeBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphIndexedRopeBf16Exec::close",
            ))
        }
    }
}

/// Completion owner for one indexed-RoPE graph executable replay.
pub struct OwnedGraphIndexedRopeBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphIndexedRopeBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphIndexedRopeBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphIndexedRopeBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphIndexedRopeBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_indexed_rope_bf16_resources(
    resources: &mut Option<OwnedGraphIndexedRopeBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphIndexedRopeBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 indexed-RoPE owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_indexed_rope_bf16_capture_preflight(
    stream: &CudaStream,
    input: &CudaDeviceBuffer,
    cos: &CudaDeviceBuffer,
    sin: &CudaDeviceBuffer,
    positions: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    positions_host: &[u32],
    active_row_count: u64,
    head_count: u64,
    head_size: u64,
    rotary_dimension: u64,
    table_position_count: u64,
    operation: &'static str,
) -> CudaResult<()> {
    if input.native_handle().same_allocation(cos.native_handle())
        || input.native_handle().same_allocation(sin.native_handle())
        || input
            .native_handle()
            .same_allocation(positions.native_handle())
        || input
            .native_handle()
            .same_allocation(output.native_handle())
        || cos.native_handle().same_allocation(sin.native_handle())
        || cos
            .native_handle()
            .same_allocation(positions.native_handle())
        || cos.native_handle().same_allocation(output.native_handle())
        || sin
            .native_handle()
            .same_allocation(positions.native_handle())
        || sin.native_handle().same_allocation(output.native_handle())
        || positions
            .native_handle()
            .same_allocation(output.native_handle())
    {
        return Err(CudaError::invalid_argument(
            operation,
            "input, cos, sin, positions, and output must be distinct fixed device allocations",
        ));
    }
    ensure_same_context(&stream.context, input.context_owner(), operation)?;
    ensure_same_context(&stream.context, cos.context_owner(), operation)?;
    ensure_same_context(&stream.context, sin.context_owner(), operation)?;
    ensure_same_context(&stream.context, positions.context_owner(), operation)?;
    ensure_same_context(&stream.context, output.context_owner(), operation)?;
    input.ensure_idle_for_operation(operation)?;
    cos.ensure_idle_for_operation(operation)?;
    sin.ensure_idle_for_operation(operation)?;
    positions.ensure_idle_for_operation(operation)?;
    output.ensure_idle_for_operation(operation)?;
    if active_row_count == 0
        || head_count == 0
        || head_size == 0
        || rotary_dimension == 0
        || table_position_count == 0
    {
        return Err(CudaError::out_of_range(
            operation,
            "active_row_count, head_count, head_size, rotary_dimension, and table_position_count must all be non-zero for a one-node BF16 indexed-RoPE graph",
        ));
    }
    if rotary_dimension > head_size || rotary_dimension % 2 != 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "rotary_dimension must be even and no larger than head_size",
        ));
    }
    let mirrored_rows = u64::try_from(positions_host.len()).map_err(|_| {
        CudaError::out_of_range(
            operation,
            "positions_host length exceeds the U64 active-row-count range",
        )
    })?;
    if mirrored_rows != active_row_count {
        return Err(CudaError::invalid_argument(
            operation,
            "positions_host length must exactly equal active_row_count",
        ));
    }
    // This uses eager-equivalent host-mirror validation only. It deliberately
    // makes no claim about the bytes already staged in the fixed device
    // positions allocation, whose raw OOB behavior remains the eager kernel's
    // BF16-NaN sentinel contract.
    crate::batch::validate_indexed_positions(positions_host, table_position_count)?;
    let tensor_elements = active_row_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(head_size))
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "active_row_count * head_count * head_size overflows the BF16 indexed-RoPE tensor element range",
            )
        })?;
    let tensor_bytes = tensor_elements
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(operation, "BF16 indexed-RoPE tensor byte range overflows")
        })?;
    let table_elements = table_position_count
        .checked_mul(rotary_dimension / 2)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "table_position_count * (rotary_dimension / 2) overflows the indexed-RoPE table element range",
            )
        })?;
    let table_bytes = table_elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "indexed-RoPE cosine/sine table byte range overflows",
            )
        })?;
    let positions_bytes = active_row_count
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "active_row_count overflows the indexed-RoPE U32 positions byte range",
            )
        })?;
    if tensor_bytes > input.byte_len()
        || table_bytes > cos.byte_len()
        || table_bytes > sin.byte_len()
        || positions_bytes > positions.byte_len()
        || tensor_bytes > output.byte_len()
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "active_row_count={active_row_count}, head_count={head_count}, head_size={head_size}, rotary_dimension={rotary_dimension}, table_position_count={table_position_count} require input/cos/sin/positions/output capacities {tensor_bytes}/{table_bytes}/{table_bytes}/{positions_bytes}/{tensor_bytes} bytes, but capacities are {}/{}/{}/{}/{} bytes",
                input.byte_len(),
                cos.byte_len(),
                sin.byte_len(),
                positions.byte_len(),
                output.byte_len(),
            ),
        ));
    }
    Ok(())
}

/// A by-value stream and nine distinct fixed device buffers recovered from a
/// known BF16 ragged paged-K/V cache-write graph lifecycle transition.
///
/// The packed host batch is admission evidence only and is never retained.
/// The key/value sources and pools plus every packed device-metadata
/// allocation stay fixed and inaccessible while capture, graph, or exec may
/// retain their addresses.
pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Resources {
    stream: CudaStream,
    key_source: CudaDeviceBuffer,
    value_source: CudaDeviceBuffer,
    key_pool: CudaDeviceBuffer,
    value_pool: CudaDeviceBuffer,
    sequence_block_offsets: CudaDeviceBuffer,
    block_ids: CudaDeviceBuffer,
    valid_tokens: CudaDeviceBuffer,
    row_sequence_slots: CudaDeviceBuffer,
    row_positions: CudaDeviceBuffer,
}

impl OwnedGraphRaggedPagedKvCacheWriteBf16Resources {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: CudaStream,
        key_source: CudaDeviceBuffer,
        value_source: CudaDeviceBuffer,
        key_pool: CudaDeviceBuffer,
        value_pool: CudaDeviceBuffer,
        sequence_block_offsets: CudaDeviceBuffer,
        block_ids: CudaDeviceBuffer,
        valid_tokens: CudaDeviceBuffer,
        row_sequence_slots: CudaDeviceBuffer,
        row_positions: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            key_source,
            value_source,
            key_pool,
            value_pool,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        }
    }

    /// Returns the exact fixed stream and device buffers after known native
    /// graph-lease release.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            key_source,
            value_source,
            key_pool,
            value_pool,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        } = self;
        (
            stream,
            key_source,
            value_source,
            key_pool,
            value_pool,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        )
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (
            stream,
            key_source,
            value_source,
            key_pool,
            value_pool,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        ) = self.into_parts();
        row_positions.close()?;
        row_sequence_slots.close()?;
        valid_tokens.close()?;
        block_ids.close()?;
        sequence_block_offsets.close()?;
        value_pool.close()?;
        key_pool.close()?;
        value_source.close()?;
        key_source.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address BF16 ragged paged-K/V
/// cache-write graph capture.
///
/// Only Rust-side preflight failures recover the untouched resource bundle.
/// Once native begin is attempted, ambiguous CUDA state retains every raw
/// address fail-closed.
#[must_use]
pub struct OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources>,
}

impl OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError {
    fn recoverable(
        error: CudaError,
        resources: OwnedGraphRaggedPagedKvCacheWriteBf16Resources,
    ) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture ownership was never
    /// entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 ragged paged-K/V cache-write capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address BF16 ragged paged-K/V
/// cache-write CUDA Graph capture.
pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Capture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphRaggedPagedKvCacheWriteBf16Capture {
    /// Captures the one immutable BF16 ragged paged-K/V cache-write node.
    pub fn enqueue_ragged_paged_kv_cache_write_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphRaggedPagedKvCacheWriteBf16Capture::enqueue_ragged_paged_kv_cache_write_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 ragged paged-K/V cache-write capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 ragged paged-K/V cache-write enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 ragged paged-K/V cache-write graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_ragged_paged_kv_cache_write_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed resource bundle into a by-value
    /// captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedRaggedPagedKvCacheWriteBf16Graph> {
        const OPERATION: &str = "OwnedGraphRaggedPagedKvCacheWriteBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 ragged paged-K/V cache-write capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 ragged paged-K/V cache-write enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 ragged paged-K/V cache-write enqueue",
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
            let resources = take_owned_graph_ragged_paged_kv_cache_write_bf16_resources(
                &mut self.resources,
                OPERATION,
            )?;
            Ok(OwnedCapturedRaggedPagedKvCacheWriteBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphRaggedPagedKvCacheWriteBf16Resources> {
        self.abort_once()?;
        take_owned_graph_ragged_paged_kv_cache_write_bf16_resources(
            &mut self.resources,
            "OwnedGraphRaggedPagedKvCacheWriteBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphRaggedPagedKvCacheWriteBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphRaggedPagedKvCacheWriteBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 ragged paged-K/V cache-write CUDA Graph
/// awaiting instantiate or close.
pub struct OwnedCapturedRaggedPagedKvCacheWriteBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedRaggedPagedKvCacheWriteBf16Graph {
    /// Instantiates the graph while retaining its fixed resource bundle by
    /// value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphRaggedPagedKvCacheWriteBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_ragged_paged_kv_cache_write_bf16_resources(
                &mut self.resources,
                "OwnedCapturedRaggedPagedKvCacheWriteBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphRaggedPagedKvCacheWriteBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedRaggedPagedKvCacheWriteBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphRaggedPagedKvCacheWriteBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_ragged_paged_kv_cache_write_bf16_resources(
                &mut self.resources,
                "OwnedCapturedRaggedPagedKvCacheWriteBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedRaggedPagedKvCacheWriteBf16Graph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 ragged paged-K/V cache-write CUDA Graph
/// executable.
///
/// It replays only capture-time device allocations. Metadata H2D, projection,
/// attention reads, scheduler commit, C07 executor wiring, node updates,
/// fresh inputs, sampling, and eager fallback stay outside this narrow C05
/// ownership slice.
pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphRaggedPagedKvCacheWriteBf16Exec {
    /// Replays the fixed-address BF16 ragged paged-K/V cache-write graph once.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphRaggedPagedKvCacheWriteBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphRaggedPagedKvCacheWriteBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 ragged paged-K/V cache-write transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 ragged paged-K/V cache-write exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphRaggedPagedKvCacheWriteBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource bundle only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphRaggedPagedKvCacheWriteBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphRaggedPagedKvCacheWriteBf16Exec::close",
                "an earlier graph BF16 ragged paged-K/V cache-write transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_ragged_paged_kv_cache_write_bf16_resources(
                &mut self.resources,
                "OwnedGraphRaggedPagedKvCacheWriteBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphRaggedPagedKvCacheWriteBf16Exec::close",
            ))
        }
    }
}

/// Completion owner for one ragged paged-K/V cache-write graph executable
/// replay.
pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphRaggedPagedKvCacheWriteBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphRaggedPagedKvCacheWriteBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphRaggedPagedKvCacheWriteBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphRaggedPagedKvCacheWriteBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_ragged_paged_kv_cache_write_bf16_resources(
    resources: &mut Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphRaggedPagedKvCacheWriteBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 ragged paged-K/V cache-write owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
fn validate_graph_ragged_paged_kv_cache_write_bf16_capture_preflight(
    stream: &CudaStream,
    key_source: &CudaDeviceBuffer,
    value_source: &CudaDeviceBuffer,
    key_pool: &CudaDeviceBuffer,
    value_pool: &CudaDeviceBuffer,
    sequence_block_offsets: &CudaDeviceBuffer,
    block_ids: &CudaDeviceBuffer,
    valid_tokens: &CudaDeviceBuffer,
    row_sequence_slots: &CudaDeviceBuffer,
    row_positions: &CudaDeviceBuffer,
    batch_host: PackedBatchHostV1<'_>,
    key_value_head_count: u64,
    head_size: u64,
    operation: &'static str,
) -> CudaResult<()> {
    // `CudaDeviceBuffer` is intentionally byte-addressed. The safe ABI dtype
    // contract is therefore the fixed parameter position plus these exact
    // BF16/U32/U16 element widths and checked shape capacities; there is no
    // mutable runtime dtype tag that a graph owner could safely trust.
    const BF16_BYTES: u64 = std::mem::size_of::<u16>() as u64;
    const U32_BYTES: u64 = std::mem::size_of::<u32>() as u64;
    const U16_BYTES: u64 = std::mem::size_of::<u16>() as u64;

    // `PackedBatchHostV1` can only be constructed after it validates the v1
    // CSR, canonical valid-token counts, physical-ID uniqueness/range, and
    // row address invariants. It is deliberately an admission witness only:
    // no host slice is copied into the fixed-address graph owner.
    if batch_host.format_version() != crate::batch::PACKED_BATCH_VERSION {
        return Err(CudaError::invalid_argument(
            operation,
            "the ragged paged-K/V graph requires packed batch format version 1",
        ));
    }
    if batch_host.block_size() != crate::batch::PACKED_BATCH_BLOCK_SIZE {
        return Err(CudaError::invalid_argument(
            operation,
            "the ragged paged-K/V graph requires fixed packed block size 16",
        ));
    }

    let buffers = [
        ("key_source", key_source),
        ("value_source", value_source),
        ("key_pool", key_pool),
        ("value_pool", value_pool),
        ("sequence_block_offsets", sequence_block_offsets),
        ("block_ids", block_ids),
        ("valid_tokens", valid_tokens),
        ("row_sequence_slots", row_sequence_slots),
        ("row_positions", row_positions),
    ];
    for (index, (left_name, left)) in buffers.iter().enumerate() {
        for (right_name, right) in buffers.iter().skip(index + 1) {
            if left.native_handle().same_allocation(right.native_handle()) {
                return Err(CudaError::invalid_argument(
                    operation,
                    format!(
                        "{left_name} and {right_name} must be distinct fixed device allocations",
                    ),
                ));
            }
        }
    }
    for (_, buffer) in &buffers {
        ensure_same_context(&stream.context, buffer.context_owner(), operation)?;
        buffer.ensure_idle_for_operation(operation)?;
    }

    let sequence_count = batch_host.sequence_count();
    let block_count = batch_host.block_count();
    let active_row_count = batch_host.active_row_count();
    let physical_block_count = batch_host.physical_block_count();
    if sequence_count == 0
        || block_count == 0
        || active_row_count == 0
        || physical_block_count == 0
        || key_value_head_count == 0
        || head_size == 0
    {
        return Err(CudaError::out_of_range(
            operation,
            "sequence_count, block_count, active_row_count, physical_block_count, key_value_head_count, and head_size must all be non-zero for a one-node BF16 ragged paged-K/V cache-write graph",
        ));
    }

    let checked_bytes = |name: &'static str, dimensions: &[u64], element_bytes: u64| {
        dimensions
            .iter()
            .try_fold(element_bytes, |bytes, dimension| {
                bytes.checked_mul(*dimension).ok_or_else(|| {
                    CudaError::out_of_range(
                        operation,
                        format!("{name} byte capacity overflows the U64 CUDA ABI range"),
                    )
                })
            })
    };
    let source_bytes = checked_bytes(
        "BF16 key/value source",
        &[active_row_count, key_value_head_count, head_size],
        BF16_BYTES,
    )?;
    let pool_bytes = checked_bytes(
        "BF16 key/value pool",
        &[
            physical_block_count,
            key_value_head_count,
            batch_host.block_size(),
            head_size,
        ],
        BF16_BYTES,
    )?;
    let offset_count = sequence_count.checked_add(1).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "sequence_count + 1 overflows the packed U32 CSR offset count",
        )
    })?;
    let offsets_bytes = checked_bytes("U32 sequence_block_offsets", &[offset_count], U32_BYTES)?;
    let block_ids_bytes = checked_bytes("U32 block_ids", &[block_count], U32_BYTES)?;
    let valid_tokens_bytes = checked_bytes("U16 valid_tokens", &[block_count], U16_BYTES)?;
    let row_sequence_slots_bytes =
        checked_bytes("U32 row_sequence_slots", &[active_row_count], U32_BYTES)?;
    let row_positions_bytes = checked_bytes("U32 row_positions", &[active_row_count], U32_BYTES)?;
    for (name, actual, required) in [
        ("BF16 key_source", key_source.byte_len(), source_bytes),
        ("BF16 value_source", value_source.byte_len(), source_bytes),
        ("BF16 key_pool", key_pool.byte_len(), pool_bytes),
        ("BF16 value_pool", value_pool.byte_len(), pool_bytes),
        (
            "U32 sequence_block_offsets",
            sequence_block_offsets.byte_len(),
            offsets_bytes,
        ),
        ("U32 block_ids", block_ids.byte_len(), block_ids_bytes),
        (
            "U16 valid_tokens",
            valid_tokens.byte_len(),
            valid_tokens_bytes,
        ),
        (
            "U32 row_sequence_slots",
            row_sequence_slots.byte_len(),
            row_sequence_slots_bytes,
        ),
        (
            "U32 row_positions",
            row_positions.byte_len(),
            row_positions_bytes,
        ),
    ] {
        if actual < required {
            return Err(CudaError::out_of_range(
                operation,
                format!(
                    "{name} requires at least {required} bytes for its fixed ABI dtype/shape, but allocation capacity is {actual} bytes",
                ),
            ));
        }
    }
    Ok(())
}

/// A by-value stream and nine distinct fixed device buffers recovered from a
/// known BF16 grouped ragged paged-attention graph lifecycle transition.
///
/// The packed host batch is admission evidence only and is never retained.
/// The query, K/V pools, output, and every packed device-metadata allocation
/// stay fixed and inaccessible while capture, graph, or exec may retain their
/// addresses.
pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Resources {
    stream: CudaStream,
    query: CudaDeviceBuffer,
    key_pool: CudaDeviceBuffer,
    value_pool: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
    sequence_block_offsets: CudaDeviceBuffer,
    block_ids: CudaDeviceBuffer,
    valid_tokens: CudaDeviceBuffer,
    row_sequence_slots: CudaDeviceBuffer,
    row_positions: CudaDeviceBuffer,
}

impl OwnedGraphGroupedRaggedPagedAttentionBf16Resources {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: CudaStream,
        query: CudaDeviceBuffer,
        key_pool: CudaDeviceBuffer,
        value_pool: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        sequence_block_offsets: CudaDeviceBuffer,
        block_ids: CudaDeviceBuffer,
        valid_tokens: CudaDeviceBuffer,
        row_sequence_slots: CudaDeviceBuffer,
        row_positions: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            query,
            key_pool,
            value_pool,
            output,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        }
    }

    /// Returns the exact fixed stream and device buffers after known native
    /// graph-lease release.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            query,
            key_pool,
            value_pool,
            output,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        } = self;
        (
            stream,
            query,
            key_pool,
            value_pool,
            output,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        )
    }

    /// Explicitly destroys recovered device buffers before their stream.
    pub fn close(self) -> CudaResult<()> {
        let (
            stream,
            query,
            key_pool,
            value_pool,
            output,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        ) = self.into_parts();
        row_positions.close()?;
        row_sequence_slots.close()?;
        valid_tokens.close()?;
        block_ids.close()?;
        sequence_block_offsets.close()?;
        output.close()?;
        value_pool.close()?;
        key_pool.close()?;
        query.close()?;
        stream.close()
    }
}

/// Error from beginning an owned fixed-address BF16 grouped ragged
/// paged-attention graph capture.
///
/// Only Rust-side preflight failures recover the untouched resource bundle.
/// Once native begin is attempted, ambiguous CUDA state retains every raw
/// address fail-closed.
#[must_use]
pub struct OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources>,
}

impl OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError {
    fn recoverable(
        error: CudaError,
        resources: OwnedGraphGroupedRaggedPagedAttentionBf16Resources,
    ) -> Self {
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

    /// The underlying validation or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture ownership was never
    /// entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 grouped ragged paged-attention capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active fixed-address BF16 grouped ragged
/// paged-attention CUDA Graph capture.
pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Capture {
    // Native drops before child resource wrappers. A capture ambiguity owns
    // their raw addresses and must remain fail-closed before Rust drops them.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphGroupedRaggedPagedAttentionBf16Capture {
    /// Captures the one immutable BF16 grouped ragged paged-attention node.
    pub fn enqueue_grouped_ragged_paged_attention_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphGroupedRaggedPagedAttentionBf16Capture::enqueue_grouped_ragged_paged_attention_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 grouped ragged paged-attention capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 grouped ragged paged-attention enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 grouped ragged paged-attention graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_grouped_ragged_paged_attention_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the fixed resource bundle into a by-value
    /// captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedGroupedRaggedPagedAttentionBf16Graph> {
        const OPERATION: &str = "OwnedGraphGroupedRaggedPagedAttentionBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 grouped ragged paged-attention capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior graph BF16 grouped ragged paged-attention enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 grouped ragged paged-attention enqueue",
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
            let resources = take_owned_graph_grouped_ragged_paged_attention_bf16_resources(
                &mut self.resources,
                OPERATION,
            )?;
            Ok(OwnedCapturedGroupedRaggedPagedAttentionBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after native recovery proves
    /// every permanent graph lease has been released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphGroupedRaggedPagedAttentionBf16Resources> {
        self.abort_once()?;
        take_owned_graph_grouped_ragged_paged_attention_bf16_resources(
            &mut self.resources,
            "OwnedGraphGroupedRaggedPagedAttentionBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphGroupedRaggedPagedAttentionBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphGroupedRaggedPagedAttentionBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value fixed-address BF16 grouped ragged paged-attention CUDA Graph
/// awaiting instantiate or close.
pub struct OwnedCapturedGroupedRaggedPagedAttentionBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedGroupedRaggedPagedAttentionBf16Graph {
    /// Instantiates the graph while retaining its fixed resource bundle by
    /// value.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphGroupedRaggedPagedAttentionBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_grouped_ragged_paged_attention_bf16_resources(
                &mut self.resources,
                "OwnedCapturedGroupedRaggedPagedAttentionBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphGroupedRaggedPagedAttentionBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedGroupedRaggedPagedAttentionBf16Graph::instantiate",
            ))
        }
    }

    /// Destroys this graph and returns resources only after known native
    /// graph-lease release evidence.
    pub fn close(mut self) -> CudaResult<OwnedGraphGroupedRaggedPagedAttentionBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_grouped_ragged_paged_attention_bf16_resources(
                &mut self.resources,
                "OwnedCapturedGroupedRaggedPagedAttentionBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedGroupedRaggedPagedAttentionBf16Graph::close",
            ))
        }
    }
}

/// By-value fixed-address BF16 grouped ragged paged-attention CUDA Graph
/// executable.
///
/// It replays only capture-time device allocations. Metadata H2D, projection,
/// cache writes, scheduler commit, C07 executor wiring, node updates, fresh
/// inputs, sampling, and eager fallback stay outside this narrow C05
/// ownership slice.
pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphGroupedRaggedPagedAttentionBf16Exec {
    /// Replays the fixed-address BF16 grouped ragged paged-attention graph
    /// once.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphGroupedRaggedPagedAttentionBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphGroupedRaggedPagedAttentionBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 grouped ragged paged-attention transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 grouped ragged paged-attention exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphGroupedRaggedPagedAttentionBf16Launch {
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its resource bundle only after
    /// native close proves every graph lease was released.
    pub fn close(mut self) -> CudaResult<OwnedGraphGroupedRaggedPagedAttentionBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphGroupedRaggedPagedAttentionBf16Exec::close",
                "an earlier graph BF16 grouped ragged paged-attention transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_grouped_ragged_paged_attention_bf16_resources(
                &mut self.resources,
                "OwnedGraphGroupedRaggedPagedAttentionBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphGroupedRaggedPagedAttentionBf16Exec::close",
            ))
        }
    }
}

/// Completion owner for one grouped ragged paged-attention graph executable
/// replay.
pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: &'exec mut OwnedGraphGroupedRaggedPagedAttentionBf16Exec,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphGroupedRaggedPagedAttentionBf16Launch<'_> {
    /// Waits for graph replay completion exactly once.
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
            Err(CudaError::unavailable(
                "OwnedGraphGroupedRaggedPagedAttentionBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphGroupedRaggedPagedAttentionBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn validate_graph_grouped_ragged_paged_attention_bf16_capture_preflight(
    stream: &CudaStream,
    query: &CudaDeviceBuffer,
    key_pool: &CudaDeviceBuffer,
    value_pool: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    sequence_block_offsets: &CudaDeviceBuffer,
    block_ids: &CudaDeviceBuffer,
    valid_tokens: &CudaDeviceBuffer,
    row_sequence_slots: &CudaDeviceBuffer,
    row_positions: &CudaDeviceBuffer,
    batch_host: PackedBatchHostV1<'_>,
    query_head_count: u64,
    key_value_head_count: u64,
    output_row_count: u64,
    scale: f32,
    operation: &'static str,
) -> CudaResult<()> {
    // `CudaDeviceBuffer` is intentionally byte-addressed. The safe ABI dtype
    // contract is therefore the fixed parameter position plus these exact
    // BF16/U32/U16 element widths and checked shape capacities; there is no
    // mutable runtime dtype tag that a graph owner could safely trust.
    const BF16_BYTES: u64 = std::mem::size_of::<u16>() as u64;
    const U32_BYTES: u64 = std::mem::size_of::<u32>() as u64;
    const U16_BYTES: u64 = std::mem::size_of::<u16>() as u64;
    const ATTENTION_HEAD_SIZE: u64 = 64;

    // `PackedBatchHostV1` can only be constructed after it validates the v1
    // CSR, canonical valid-token counts, physical-ID uniqueness/range, and
    // row address invariants. It is deliberately an admission witness only:
    // no host slice is copied into the fixed-address graph owner.
    if batch_host.format_version() != crate::batch::PACKED_BATCH_VERSION {
        return Err(CudaError::invalid_argument(
            operation,
            "the grouped ragged paged-attention graph requires packed batch format version 1",
        ));
    }
    if batch_host.block_size() != crate::batch::PACKED_BATCH_BLOCK_SIZE {
        return Err(CudaError::invalid_argument(
            operation,
            "the grouped ragged paged-attention graph requires fixed packed block size 16",
        ));
    }

    let buffers = [
        ("query", query),
        ("key_pool", key_pool),
        ("value_pool", value_pool),
        ("output", output),
        ("sequence_block_offsets", sequence_block_offsets),
        ("block_ids", block_ids),
        ("valid_tokens", valid_tokens),
        ("row_sequence_slots", row_sequence_slots),
        ("row_positions", row_positions),
    ];
    for (index, (left_name, left)) in buffers.iter().enumerate() {
        for (right_name, right) in buffers.iter().skip(index + 1) {
            if left.native_handle().same_allocation(right.native_handle()) {
                return Err(CudaError::invalid_argument(
                    operation,
                    format!(
                        "{left_name} and {right_name} must be distinct fixed device allocations",
                    ),
                ));
            }
        }
    }
    for (_, buffer) in &buffers {
        ensure_same_context(&stream.context, buffer.context_owner(), operation)?;
        buffer.ensure_idle_for_operation(operation)?;
    }

    let sequence_count = batch_host.sequence_count();
    let block_count = batch_host.block_count();
    let active_row_count = batch_host.active_row_count();
    let physical_block_count = batch_host.physical_block_count();
    if sequence_count == 0
        || block_count == 0
        || active_row_count == 0
        || physical_block_count == 0
        || query_head_count == 0
        || key_value_head_count == 0
    {
        return Err(CudaError::out_of_range(
            operation,
            "sequence_count, block_count, active_row_count, physical_block_count, query_head_count, and key_value_head_count must all be non-zero for a one-node BF16 grouped ragged paged-attention graph",
        ));
    }
    if query_head_count % key_value_head_count != 0 {
        return Err(CudaError::invalid_argument(
            operation,
            "key_value_head_count must divide query_head_count for grouped GQA attention",
        ));
    }
    if output_row_count < active_row_count {
        return Err(CudaError::out_of_range(
            operation,
            "output_row_count must be at least active_row_count for grouped ragged paged attention",
        ));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(CudaError::invalid_argument(
            operation,
            "scale must be finite and greater than zero for grouped ragged paged attention",
        ));
    }

    let checked_bytes = |name: &'static str, dimensions: &[u64], element_bytes: u64| {
        dimensions
            .iter()
            .try_fold(element_bytes, |bytes, dimension| {
                bytes.checked_mul(*dimension).ok_or_else(|| {
                    CudaError::out_of_range(
                        operation,
                        format!("{name} byte capacity overflows the U64 CUDA ABI range"),
                    )
                })
            })
    };
    let query_bytes = checked_bytes(
        "BF16 query",
        &[active_row_count, query_head_count, ATTENTION_HEAD_SIZE],
        BF16_BYTES,
    )?;
    let pool_bytes = checked_bytes(
        "BF16 K/V pool",
        &[
            physical_block_count,
            key_value_head_count,
            batch_host.block_size(),
            ATTENTION_HEAD_SIZE,
        ],
        BF16_BYTES,
    )?;
    let output_bytes = checked_bytes(
        "BF16 output",
        &[output_row_count, query_head_count, ATTENTION_HEAD_SIZE],
        BF16_BYTES,
    )?;
    let offset_count = sequence_count.checked_add(1).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "sequence_count + 1 overflows the packed U32 CSR offset count",
        )
    })?;
    let offsets_bytes = checked_bytes("U32 sequence_block_offsets", &[offset_count], U32_BYTES)?;
    let block_ids_bytes = checked_bytes("U32 block_ids", &[block_count], U32_BYTES)?;
    let valid_tokens_bytes = checked_bytes("U16 valid_tokens", &[block_count], U16_BYTES)?;
    let row_sequence_slots_bytes =
        checked_bytes("U32 row_sequence_slots", &[active_row_count], U32_BYTES)?;
    let row_positions_bytes = checked_bytes("U32 row_positions", &[active_row_count], U32_BYTES)?;
    for (name, actual, required) in [
        ("BF16 query", query.byte_len(), query_bytes),
        ("BF16 key_pool", key_pool.byte_len(), pool_bytes),
        ("BF16 value_pool", value_pool.byte_len(), pool_bytes),
        ("BF16 output", output.byte_len(), output_bytes),
        (
            "U32 sequence_block_offsets",
            sequence_block_offsets.byte_len(),
            offsets_bytes,
        ),
        ("U32 block_ids", block_ids.byte_len(), block_ids_bytes),
        (
            "U16 valid_tokens",
            valid_tokens.byte_len(),
            valid_tokens_bytes,
        ),
        (
            "U32 row_sequence_slots",
            row_sequence_slots.byte_len(),
            row_sequence_slots_bytes,
        ),
        (
            "U32 row_positions",
            row_positions.byte_len(),
            row_positions_bytes,
        ),
    ] {
        if actual < required {
            return Err(CudaError::out_of_range(
                operation,
                format!(
                    "{name} requires at least {required} bytes for its fixed ABI dtype/shape, but allocation capacity is {actual} bytes",
                ),
            ));
        }
    }
    Ok(())
}

fn take_owned_graph_grouped_ragged_paged_attention_bf16_resources(
    resources: &mut Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphGroupedRaggedPagedAttentionBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 grouped ragged paged-attention owner lost its captured resources",
        )
    })
}

// These values and the fixed record width are the native
// `RileyCudaEmbeddingErrorReport` ABI, deliberately kept private to the
// graph owner. Callers receive only a native-validated semantic status after
// launch completion, never a reusable pinned allocation or a raw report
// pointer.
#[cfg(feature = "cuda")]
const BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES: u64 = 32;
#[cfg(feature = "cuda")]
const BF16_EMBEDDING_STATUS_D2H_REPORT_NONE: u32 = 0;
#[cfg(feature = "cuda")]
const BF16_EMBEDDING_STATUS_D2H_REPORT_TOKEN_OUT_OF_RANGE: u32 = 1;

/// Native-validated semantic result of one completed C05-20 embedding graph.
///
/// This is deliberately a status value rather than a CUDA failure. A token
/// outside the fixed vocabulary is the eager embedding primitive's defined
/// fail-before-write outcome, and the graph remains reusable after its report
/// has been observed. Malformed reports and uncertain completion instead
/// fail-close the executable and return [`CudaError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bf16EmbeddingStatusD2HStatus {
    /// Every fixed device token ID was below `vocabulary_size` and the BF16
    /// gather completed.
    Success,
    /// The earliest U32 device token ID outside the fixed vocabulary.
    TokenOutOfRange {
        /// Zero-based position in the fixed device token-ID allocation.
        token_position: u64,
        /// The U32 token ID read at `token_position`, widened losslessly.
        token_id: u64,
    },
}

/// A by-value stream, four fixed device allocations, and one exact pinned
/// embedding-status report allocation recovered after a known C05-20 graph
/// lifecycle release.
///
/// `table`, `token_ids`, `output`, and `device_error_scratch` have immutable
/// capture-time roles and are deliberately not exposed while native may retain
/// their addresses. The pinned report is graph-owned rather than general D2H
/// staging until this bundle is recovered.
pub struct OwnedGraphBf16EmbeddingStatusD2HResources {
    stream: CudaStream,
    table: CudaDeviceBuffer,
    token_ids: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
    device_error_scratch: CudaDeviceBuffer,
    pinned_report: CudaPinnedHostBuffer,
}

impl OwnedGraphBf16EmbeddingStatusD2HResources {
    fn new(
        stream: CudaStream,
        table: CudaDeviceBuffer,
        token_ids: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        device_error_scratch: CudaDeviceBuffer,
        pinned_report: CudaPinnedHostBuffer,
    ) -> Self {
        Self {
            stream,
            table,
            token_ids,
            output,
            device_error_scratch,
            pinned_report,
        }
    }

    /// Returns all fixed resources only after native graph lease release is
    /// known. The pinned report remains unavailable until that point even when
    /// a prior completion receipt has copied its status to the caller.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaPinnedHostBuffer,
    ) {
        let Self {
            stream,
            table,
            token_ids,
            output,
            device_error_scratch,
            pinned_report,
        } = self;
        (
            stream,
            table,
            token_ids,
            output,
            device_error_scratch,
            pinned_report,
        )
    }

    /// Explicitly destroys recovered allocations before the capture stream.
    pub fn close(self) -> CudaResult<()> {
        let (stream, table, token_ids, output, device_error_scratch, pinned_report) =
            self.into_parts();
        pinned_report.close()?;
        device_error_scratch.close()?;
        output.close()?;
        token_ids.close()?;
        table.close()?;
        stream.close()
    }
}

/// Error from beginning one by-value fixed-address C05-20 graph capture.
///
/// Only pure Rust preflight errors return the untouched resource sextet. Once
/// native capture entry was attempted, CUDA may retain every raw address, so
/// the resources remain fail-closed and cannot be recovered through this
/// error value.
#[must_use]
pub struct OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphBf16EmbeddingStatusD2HResources>,
}

impl OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphBf16EmbeddingStatusD2HResources) -> Self {
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

    /// The rejected preflight or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphBf16EmbeddingStatusD2HResources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph BF16 embedding status-D2H capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active C05-20 five-node graph capture.
pub struct OwnedGraphBf16EmbeddingStatusD2HCapture {
    // Native drops before child wrappers: ambiguity retains every raw address
    // and Rust resources must not drop independently.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphBf16EmbeddingStatusD2HResources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16EmbeddingStatusD2HCapture {
    /// Records exactly reset -> validate -> gather -> finalize -> pinned D2H
    /// once. It admits no host callback, fresh token staging, or node update.
    pub fn enqueue_bf16_embedding_status_d2h(&mut self) -> CudaResult<()> {
        const OPERATION: &str =
            "OwnedGraphBf16EmbeddingStatusD2HCapture::enqueue_bf16_embedding_status_d2h";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 embedding status-D2H capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior five-node graph enqueue failed and this partial capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed BF16 embedding status-D2H graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_bf16_embedding_status_d2h();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                // A failed multi-node capture can have retained a recorded
                // prefix. End/instantiate must never observe that prefix.
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the resource sextet into a captured graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedBf16EmbeddingStatusD2HGraph> {
        const OPERATION: &str = "OwnedGraphBf16EmbeddingStatusD2HCapture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned graph BF16 embedding status-D2H capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior five-node graph enqueue failed and this partial capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed BF16 embedding status-D2H enqueue",
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
            let resources = take_owned_graph_bf16_embedding_status_d2h_resources(
                &mut self.resources,
                OPERATION,
            )?;
            Ok(OwnedCapturedBf16EmbeddingStatusD2HGraph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after known native release.
    pub fn abort(mut self) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HResources> {
        self.abort_once()?;
        take_owned_graph_bf16_embedding_status_d2h_resources(
            &mut self.resources,
            "OwnedGraphBf16EmbeddingStatusD2HCapture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphBf16EmbeddingStatusD2HCapture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16EmbeddingStatusD2HCapture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value captured C05-20 graph awaiting instantiate or close.
pub struct OwnedCapturedBf16EmbeddingStatusD2HGraph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphBf16EmbeddingStatusD2HResources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedBf16EmbeddingStatusD2HGraph {
    /// Instantiates while retaining the exact stream/allocation sextet.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HExec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_bf16_embedding_status_d2h_resources(
                &mut self.resources,
                "OwnedCapturedBf16EmbeddingStatusD2HGraph::instantiate",
            )?;
            Ok(OwnedGraphBf16EmbeddingStatusD2HExec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16EmbeddingStatusD2HGraph::instantiate",
            ))
        }
    }

    /// Closes the captured graph and returns resources after known release.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HResources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_embedding_status_d2h_resources(
                &mut self.resources,
                "OwnedCapturedBf16EmbeddingStatusD2HGraph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedBf16EmbeddingStatusD2HGraph::close",
            ))
        }
    }
}

/// By-value executable for one fixed BF16 embedding validation-status graph.
///
/// Its pinned report remains hidden until a matching launch completion creates
/// [`OwnedGraphBf16EmbeddingStatusD2HCompletion`]. A result is not a C07
/// scheduler commit or completion boundary.
pub struct OwnedGraphBf16EmbeddingStatusD2HExec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphBf16EmbeddingStatusD2HResources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphBf16EmbeddingStatusD2HExec {
    /// Launches the fixed five-node graph once.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HLaunch<'exec>> {
        const OPERATION: &str = "OwnedGraphBf16EmbeddingStatusD2HExec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier graph BF16 embedding status-D2H transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned graph BF16 embedding status-D2H exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphBf16EmbeddingStatusD2HLaunch {
                    native,
                    exec: Some(self),
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns resources after known release.
    pub fn close(mut self) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HResources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphBf16EmbeddingStatusD2HExec::close",
                "an earlier graph BF16 embedding status-D2H transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_bf16_embedding_status_d2h_resources(
                &mut self.resources,
                "OwnedGraphBf16EmbeddingStatusD2HExec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphBf16EmbeddingStatusD2HExec::close",
            ))
        }
    }
}

/// In-flight completion owner for one C05-20 graph replay.
pub struct OwnedGraphBf16EmbeddingStatusD2HLaunch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: Option<&'exec mut OwnedGraphBf16EmbeddingStatusD2HExec>,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl<'exec> OwnedGraphBf16EmbeddingStatusD2HLaunch<'exec> {
    /// Waits for completion and returns the sole status-report receipt.
    ///
    /// The receipt keeps the executable exclusively borrowed, so a new replay
    /// or close cannot race the fixed pinned-host report view.
    pub fn finish(mut self) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HCompletion<'exec>> {
        self.complete_once()?;
        let exec = self.exec.take().ok_or_else(|| {
            CudaError::invalid_state(
                "OwnedGraphBf16EmbeddingStatusD2HLaunch::finish",
                "the graph launch completion owner was already consumed",
            )
        })?;
        Ok(OwnedGraphBf16EmbeddingStatusD2HCompletion { exec })
    }

    fn complete_once(&mut self) -> CudaResult<()> {
        let Some(exec) = self.exec.as_deref_mut() else {
            return if self.active {
                Err(CudaError::invalid_state(
                    "OwnedGraphBf16EmbeddingStatusD2HLaunch::finish",
                    "the graph launch completion owner lost its executable borrow",
                ))
            } else {
                Ok(())
            };
        };
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            let result = self.native.complete();
            if result.is_err() {
                exec.terminal = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            exec.terminal = true;
            Err(CudaError::unavailable(
                "OwnedGraphBf16EmbeddingStatusD2HLaunch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphBf16EmbeddingStatusD2HLaunch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

/// Completion-scoped semantic view of C05-20's exact pinned status record.
///
/// Native owns and validates the permanent raw report allocation. This receipt
/// exposes only the two semantic outcomes after a known completion, and does
/// not release or make the pinned allocation independently reusable.
#[must_use]
pub struct OwnedGraphBf16EmbeddingStatusD2HCompletion<'exec> {
    exec: &'exec mut OwnedGraphBf16EmbeddingStatusD2HExec,
}

impl OwnedGraphBf16EmbeddingStatusD2HCompletion<'_> {
    /// Reads one native-validated semantic status from the completed pinned
    /// report.
    ///
    /// A returned [`Bf16EmbeddingStatusD2HStatus::TokenOutOfRange`] is the
    /// intentional eager-equivalent no-output-write result, not an error that
    /// poisons the graph. A native observation, metadata, or report-contract
    /// failure instead retains all resources fail-closed.
    pub fn read_status(&mut self) -> CudaResult<Bf16EmbeddingStatusD2HStatus> {
        const OPERATION: &str = "OwnedGraphBf16EmbeddingStatusD2HCompletion::read_status";
        if self.exec.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the graph executable is terminal after an uncertain transition",
            ));
        }
        if self.exec.resources.is_none() {
            self.exec.terminal = true;
            return Err(CudaError::invalid_state(
                OPERATION,
                "the graph executable lost its retained pinned embedding report allocation",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let report = match self.exec.native.read_bf16_embedding_status_d2h_report() {
                Ok(report) => report,
                Err(error) => {
                    self.exec.terminal = true;
                    return Err(error);
                }
            };
            match report.code {
                BF16_EMBEDDING_STATUS_D2H_REPORT_NONE => {
                    if report.token_position != 0 || report.token_id != 0 {
                        self.exec.terminal = true;
                        return Err(CudaError::new(
                            CudaErrorKind::Internal,
                            CudaErrorDomain::Internal,
                            CudaErrorStage::Synchronize,
                            0,
                            OPERATION,
                            "native embedding graph accepted an inconsistent successful report",
                        ));
                    }
                    Ok(Bf16EmbeddingStatusD2HStatus::Success)
                }
                BF16_EMBEDDING_STATUS_D2H_REPORT_TOKEN_OUT_OF_RANGE => {
                    Ok(Bf16EmbeddingStatusD2HStatus::TokenOutOfRange {
                        token_position: report.token_position,
                        token_id: report.token_id,
                    })
                }
                code => {
                    self.exec.terminal = true;
                    Err(CudaError::new(
                        CudaErrorKind::Internal,
                        CudaErrorDomain::Internal,
                        CudaErrorStage::Synchronize,
                        0,
                        OPERATION,
                        format!("native embedding graph returned unsupported report code {code}"),
                    ))
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.exec.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }
}

fn take_owned_graph_bf16_embedding_status_d2h_resources(
    resources: &mut Option<OwnedGraphBf16EmbeddingStatusD2HResources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphBf16EmbeddingStatusD2HResources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned graph BF16 embedding status-D2H owner lost its captured resources",
        )
    })
}

/// By-value resource bundle for one C05-21 canonical cuBLASLt graph.
///
/// The plan and all four whole allocations remain inaccessible from capture
/// begin until native abort or graph close proves that every permanent lease
/// was released. A zero-byte workspace is still represented by its own device
/// buffer; it is never replaced with an output alias.
pub struct OwnedGraphCanonicalGemmBf16Resources {
    stream: CudaStream,
    plan: CudaPreparedGemm,
    input: CudaDeviceBuffer,
    weight: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
    workspace: CudaDeviceBuffer,
}

impl OwnedGraphCanonicalGemmBf16Resources {
    fn new(
        stream: CudaStream,
        plan: CudaPreparedGemm,
        input: CudaDeviceBuffer,
        weight: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        workspace: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            plan,
            input,
            weight,
            output,
            workspace,
        }
    }

    /// Returns the original plan, stream, and allocations only after native
    /// graph ownership is known to be released.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaPreparedGemm,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            plan,
            input,
            weight,
            output,
            workspace,
        } = self;
        (stream, plan, input, weight, output, workspace)
    }

    /// Explicitly closes recovered resources in the reverse ownership order.
    pub fn close(self) -> CudaResult<()> {
        let (stream, plan, input, weight, output, workspace) = self.into_parts();
        plan.close()?;
        workspace.close()?;
        output.close()?;
        weight.close()?;
        input.close()?;
        stream.close()
    }
}

/// Error from beginning one by-value C05-21 canonical GEMM graph capture.
///
/// Pure Rust preflight errors return the untouched resource bundle. Once the
/// native begin function is entered, all raw plan/allocation addresses remain
/// fail-closed because CUDA may have retained them during capture recovery.
#[must_use]
pub struct OwnedGraphCanonicalGemmBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphCanonicalGemmBf16Resources>,
}

impl OwnedGraphCanonicalGemmBf16CaptureBeginError {
    fn recoverable(error: CudaError, resources: OwnedGraphCanonicalGemmBf16Resources) -> Self {
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

    /// The rejected preflight or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphCanonicalGemmBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphCanonicalGemmBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphCanonicalGemmBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphCanonicalGemmBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph canonical BF16 cuBLASLt GEMM capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphCanonicalGemmBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active C05-21 single-matmul graph capture.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphCanonicalGemmBf16Capture>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphCanonicalGemmBf16Capture>();
/// ```
pub struct OwnedGraphCanonicalGemmBf16Capture {
    // Native drops before child wrappers: ambiguity retains every raw plan and
    // allocation address, so Rust resources must not drop independently.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphCanonicalGemmBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalGemmBf16Capture {
    /// Records exactly one capture-only `cublasLtMatmul` for the cold plan.
    /// It admits no dynamic spans, node updates, command batch, or eager
    /// synchronization.
    pub fn enqueue_canonical_gemm_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "OwnedGraphCanonicalGemmBf16Capture::enqueue_canonical_gemm_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned canonical GEMM graph capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior canonical GEMM graph enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed canonical GEMM graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_canonical_gemm_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends capture and transfers the plan/allocation bundle into a graph.
    pub fn end(mut self) -> CudaResult<OwnedCapturedCanonicalGemmBf16Graph> {
        const OPERATION: &str = "OwnedGraphCanonicalGemmBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned canonical GEMM graph capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior canonical GEMM graph enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed canonical GEMM enqueue",
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
            let resources =
                take_owned_graph_canonical_gemm_bf16_resources(&mut self.resources, OPERATION)?;
            Ok(OwnedCapturedCanonicalGemmBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts capture and returns resources only after known native release.
    pub fn abort(mut self) -> CudaResult<OwnedGraphCanonicalGemmBf16Resources> {
        self.abort_once()?;
        take_owned_graph_canonical_gemm_bf16_resources(
            &mut self.resources,
            "OwnedGraphCanonicalGemmBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalGemmBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphCanonicalGemmBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value captured canonical GEMM graph awaiting instantiate or close.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedCapturedCanonicalGemmBf16Graph>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedCapturedCanonicalGemmBf16Graph>();
/// ```
pub struct OwnedCapturedCanonicalGemmBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphCanonicalGemmBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedCanonicalGemmBf16Graph {
    /// Instantiates while retaining the exact plan, stream, and allocations.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphCanonicalGemmBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_canonical_gemm_bf16_resources(
                &mut self.resources,
                "OwnedCapturedCanonicalGemmBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphCanonicalGemmBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedCanonicalGemmBf16Graph::instantiate",
            ))
        }
    }

    /// Closes the captured graph and returns resources after known release.
    pub fn close(mut self) -> CudaResult<OwnedGraphCanonicalGemmBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_canonical_gemm_bf16_resources(
                &mut self.resources,
                "OwnedCapturedCanonicalGemmBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedCanonicalGemmBf16Graph::close",
            ))
        }
    }
}

/// By-value executable for one canonical capture-only cuBLASLt GEMM graph.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphCanonicalGemmBf16Exec>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphCanonicalGemmBf16Exec>();
/// ```
pub struct OwnedGraphCanonicalGemmBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphCanonicalGemmBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalGemmBf16Exec {
    /// Launches the fixed one-node graph once.
    pub fn launch<'exec>(&'exec mut self) -> CudaResult<OwnedGraphCanonicalGemmBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphCanonicalGemmBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier canonical GEMM graph transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned canonical GEMM graph exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphCanonicalGemmBf16Launch {
                    native,
                    exec: Some(self),
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns resources after known release.
    pub fn close(mut self) -> CudaResult<OwnedGraphCanonicalGemmBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphCanonicalGemmBf16Exec::close",
                "an earlier canonical GEMM graph transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_canonical_gemm_bf16_resources(
                &mut self.resources,
                "OwnedGraphCanonicalGemmBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalGemmBf16Exec::close",
            ))
        }
    }
}

/// In-flight completion owner for one C05-21 graph replay.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphCanonicalGemmBf16Launch<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphCanonicalGemmBf16Launch<'static>>();
/// ```
pub struct OwnedGraphCanonicalGemmBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: Option<&'exec mut OwnedGraphCanonicalGemmBf16Exec>,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalGemmBf16Launch<'_> {
    /// Waits for known graph completion. The output allocation remains owned
    /// by the executable until its later explicit close.
    pub fn finish(mut self) -> CudaResult<()> {
        self.complete_once()
    }

    fn complete_once(&mut self) -> CudaResult<()> {
        let Some(exec) = self.exec.as_deref_mut() else {
            return if self.active {
                Err(CudaError::invalid_state(
                    "OwnedGraphCanonicalGemmBf16Launch::finish",
                    "the graph launch completion owner lost its executable borrow",
                ))
            } else {
                Ok(())
            };
        };
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            let result = self.native.complete();
            if result.is_err() {
                exec.terminal = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            exec.terminal = true;
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalGemmBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphCanonicalGemmBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_canonical_gemm_bf16_resources(
    resources: &mut Option<OwnedGraphCanonicalGemmBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphCanonicalGemmBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned canonical GEMM graph owner lost its captured resources",
        )
    })
}

/// By-value resource bundle for one C05-22 canonical RMSNorm -> GEMM graph.
///
/// `rms_norm_output` is the one deliberate fixed-address dependency: native
/// records it as the canonical GEMM input without a second Rust wrapper or
/// lease. The cold strict-no-split plan and the other five allocations remain
/// inaccessible until a terminal native graph transition proves release.
pub struct OwnedGraphCanonicalRmsNormGemmBf16Resources {
    stream: CudaStream,
    plan: CudaPreparedGemm,
    rms_norm_input: CudaDeviceBuffer,
    rms_norm_weight: CudaDeviceBuffer,
    rms_norm_output: CudaDeviceBuffer,
    gemm_weight: CudaDeviceBuffer,
    gemm_output: CudaDeviceBuffer,
    gemm_workspace: CudaDeviceBuffer,
}

impl OwnedGraphCanonicalRmsNormGemmBf16Resources {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: CudaStream,
        plan: CudaPreparedGemm,
        rms_norm_input: CudaDeviceBuffer,
        rms_norm_weight: CudaDeviceBuffer,
        rms_norm_output: CudaDeviceBuffer,
        gemm_weight: CudaDeviceBuffer,
        gemm_output: CudaDeviceBuffer,
        gemm_workspace: CudaDeviceBuffer,
    ) -> Self {
        Self {
            stream,
            plan,
            rms_norm_input,
            rms_norm_weight,
            rms_norm_output,
            gemm_weight,
            gemm_output,
            gemm_workspace,
        }
    }

    /// Returns the one stream, cold plan, and six exact allocations after a
    /// known native graph-lease release. `rms_norm_output` remains the one
    /// buffer that served both nodes; callers never receive a duplicate alias.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        CudaStream,
        CudaPreparedGemm,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    ) {
        let Self {
            stream,
            plan,
            rms_norm_input,
            rms_norm_weight,
            rms_norm_output,
            gemm_weight,
            gemm_output,
            gemm_workspace,
        } = self;
        (
            stream,
            plan,
            rms_norm_input,
            rms_norm_weight,
            rms_norm_output,
            gemm_weight,
            gemm_output,
            gemm_workspace,
        )
    }

    /// Explicitly closes the recovered plan and buffers in reverse graph-use
    /// order, then releases the retained stream.
    pub fn close(self) -> CudaResult<()> {
        let (
            stream,
            plan,
            rms_norm_input,
            rms_norm_weight,
            rms_norm_output,
            gemm_weight,
            gemm_output,
            gemm_workspace,
        ) = self.into_parts();
        plan.close()?;
        gemm_workspace.close()?;
        gemm_output.close()?;
        gemm_weight.close()?;
        rms_norm_output.close()?;
        rms_norm_weight.close()?;
        rms_norm_input.close()?;
        stream.close()
    }
}

/// Error from beginning one by-value C05-22 canonical RMSNorm -> GEMM graph
/// capture.
///
/// Rust-only preflight errors recover the untouched bundle. Once native begin
/// has been attempted, CUDA may have retained the cold plan and every address,
/// so the safe wrapper fails closed without exposing a possibly-live resource.
#[must_use]
pub struct OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError {
    error: CudaError,
    resources: Option<OwnedGraphCanonicalRmsNormGemmBf16Resources>,
}

impl OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError {
    fn recoverable(
        error: CudaError,
        resources: OwnedGraphCanonicalRmsNormGemmBf16Resources,
    ) -> Self {
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

    /// The rejected preflight or native error.
    #[must_use]
    pub const fn error(&self) -> &CudaError {
        &self.error
    }

    /// Returns moved resources only when native capture was never entered.
    #[must_use]
    pub fn into_resources(self) -> Option<OwnedGraphCanonicalRmsNormGemmBf16Resources> {
        self.resources
    }
}

impl std::fmt::Debug for OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError")
            .field("error", &self.error)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl std::fmt::Display for OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "owned CUDA Graph canonical BF16 RMSNorm -> cuBLASLt GEMM capture begin failed: {}",
            self.error
        )
    }
}

impl std::error::Error for OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// By-value owner of one active C05-22 two-node capture.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphCanonicalRmsNormGemmBf16Capture>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphCanonicalRmsNormGemmBf16Capture>();
/// ```
pub struct OwnedGraphCanonicalRmsNormGemmBf16Capture {
    // Native drops before child wrappers: an ambiguous capture retains the
    // plan and every allocation address, so none may drop independently.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphCaptureHandle,
    resources: Option<OwnedGraphCanonicalRmsNormGemmBf16Resources>,
    active: bool,
    enqueued: bool,
    enqueue_failed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalRmsNormGemmBf16Capture {
    /// Records exactly canonical BF16 RMSNorm followed by the dependent,
    /// capture-only strict-plan cuBLASLt matmul. Dynamic spans, node updates,
    /// fresh inputs, batches, and eager synchronization are not admitted.
    pub fn enqueue_canonical_rms_norm_gemm_bf16(&mut self) -> CudaResult<()> {
        const OPERATION: &str =
            "OwnedGraphCanonicalRmsNormGemmBf16Capture::enqueue_canonical_rms_norm_gemm_bf16";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned canonical RMSNorm -> GEMM graph capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior canonical RMSNorm -> GEMM graph enqueue failed and this capture must be aborted",
            ));
        }
        if self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a fixed canonical RMSNorm -> GEMM graph capture admits exactly one enqueue",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let result = self.native.enqueue_canonical_rms_norm_gemm_bf16();
            if result.is_ok() {
                self.enqueued = true;
            } else {
                // RMSNorm may already be recorded when GEMM admission fails;
                // only one-shot abort can establish native lease release.
                self.enqueue_failed = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Ends the exact two-node capture and transfers its full bundle into a
    /// by-value graph owner.
    pub fn end(mut self) -> CudaResult<OwnedCapturedCanonicalRmsNormGemmBf16Graph> {
        const OPERATION: &str = "OwnedGraphCanonicalRmsNormGemmBf16Capture::end";
        if !self.active {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the owned canonical RMSNorm -> GEMM graph capture was already ended or aborted",
            ));
        }
        if self.enqueue_failed {
            return Err(CudaError::invalid_state(
                OPERATION,
                "a prior canonical RMSNorm -> GEMM graph enqueue failed and this capture must be aborted",
            ));
        }
        if !self.enqueued {
            return Err(CudaError::invalid_state(
                OPERATION,
                "capture end requires the one fixed canonical RMSNorm -> GEMM enqueue",
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
            let resources = take_owned_graph_canonical_rms_norm_gemm_bf16_resources(
                &mut self.resources,
                OPERATION,
            )?;
            Ok(OwnedCapturedCanonicalRmsNormGemmBf16Graph {
                native,
                resources: Some(resources),
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.active = false;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Aborts the capture and returns resources only after native recovery
    /// proves that all permanent graph leases were released.
    pub fn abort(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormGemmBf16Resources> {
        self.abort_once()?;
        take_owned_graph_canonical_rms_norm_gemm_bf16_resources(
            &mut self.resources,
            "OwnedGraphCanonicalRmsNormGemmBf16Capture::abort",
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
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalRmsNormGemmBf16Capture::abort",
            ))
        }
    }
}

impl Drop for OwnedGraphCanonicalRmsNormGemmBf16Capture {
    fn drop(&mut self) {
        let _ = self.abort_once();
    }
}

/// By-value captured C05-22 graph awaiting instantiate or close.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedCapturedCanonicalRmsNormGemmBf16Graph>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedCapturedCanonicalRmsNormGemmBf16Graph>();
/// ```
pub struct OwnedCapturedCanonicalRmsNormGemmBf16Graph {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphHandle,
    resources: Option<OwnedGraphCanonicalRmsNormGemmBf16Resources>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedCapturedCanonicalRmsNormGemmBf16Graph {
    /// Instantiates while retaining the one cold plan and exact allocations.
    pub fn instantiate(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormGemmBf16Exec> {
        #[cfg(feature = "cuda")]
        {
            let native = self.native.instantiate()?;
            let resources = take_owned_graph_canonical_rms_norm_gemm_bf16_resources(
                &mut self.resources,
                "OwnedCapturedCanonicalRmsNormGemmBf16Graph::instantiate",
            )?;
            Ok(OwnedGraphCanonicalRmsNormGemmBf16Exec {
                native,
                resources: Some(resources),
                terminal: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedCanonicalRmsNormGemmBf16Graph::instantiate",
            ))
        }
    }

    /// Closes the captured graph and returns resources after known release.
    pub fn close(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormGemmBf16Resources> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_canonical_rms_norm_gemm_bf16_resources(
                &mut self.resources,
                "OwnedCapturedCanonicalRmsNormGemmBf16Graph::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedCapturedCanonicalRmsNormGemmBf16Graph::close",
            ))
        }
    }
}

/// By-value executable for one fixed canonical RMSNorm -> GEMM graph.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphCanonicalRmsNormGemmBf16Exec>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphCanonicalRmsNormGemmBf16Exec>();
/// ```
pub struct OwnedGraphCanonicalRmsNormGemmBf16Exec {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    resources: Option<OwnedGraphCanonicalRmsNormGemmBf16Resources>,
    terminal: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalRmsNormGemmBf16Exec {
    /// Launches the fixed two-node graph once on its retained stream.
    pub fn launch<'exec>(
        &'exec mut self,
    ) -> CudaResult<OwnedGraphCanonicalRmsNormGemmBf16Launch<'exec>> {
        const OPERATION: &str = "OwnedGraphCanonicalRmsNormGemmBf16Exec::launch";
        if self.terminal {
            return Err(CudaError::invalid_state(
                OPERATION,
                "an earlier canonical RMSNorm -> GEMM graph transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = match self.resources.as_mut() {
                Some(resources) => self.native.launch(&mut resources.stream.native),
                None => {
                    self.terminal = true;
                    return Err(CudaError::invalid_state(
                        OPERATION,
                        "the owned canonical RMSNorm -> GEMM graph exec lost its captured resources",
                    ));
                }
            };
            match native {
                Ok(native) => Ok(OwnedGraphCanonicalRmsNormGemmBf16Launch {
                    native,
                    exec: Some(self),
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
            let _ = &mut self.resources;
            self.terminal = true;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Destroys this executable and returns its bundle after known native
    /// release of every plan/allocation graph lease.
    pub fn close(mut self) -> CudaResult<OwnedGraphCanonicalRmsNormGemmBf16Resources> {
        if self.terminal {
            return Err(CudaError::invalid_state(
                "OwnedGraphCanonicalRmsNormGemmBf16Exec::close",
                "an earlier canonical RMSNorm -> GEMM graph transition left native state uncertain",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()?;
            take_owned_graph_canonical_rms_norm_gemm_bf16_resources(
                &mut self.resources,
                "OwnedGraphCanonicalRmsNormGemmBf16Exec::close",
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self.resources;
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalRmsNormGemmBf16Exec::close",
            ))
        }
    }
}

/// In-flight completion owner for one C05-22 graph replay.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<riley_cuda::OwnedGraphCanonicalRmsNormGemmBf16Launch<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::OwnedGraphCanonicalRmsNormGemmBf16Launch<'static>>();
/// ```
pub struct OwnedGraphCanonicalRmsNormGemmBf16Launch<'exec> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphLaunchHandle,
    exec: Option<&'exec mut OwnedGraphCanonicalRmsNormGemmBf16Exec>,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl OwnedGraphCanonicalRmsNormGemmBf16Launch<'_> {
    /// Waits for known completion exactly once.
    pub fn finish(mut self) -> CudaResult<()> {
        self.complete_once()
    }

    fn complete_once(&mut self) -> CudaResult<()> {
        let Some(exec) = self.exec.as_deref_mut() else {
            return if self.active {
                Err(CudaError::invalid_state(
                    "OwnedGraphCanonicalRmsNormGemmBf16Launch::finish",
                    "the graph launch completion owner lost its executable borrow",
                ))
            } else {
                Ok(())
            };
        };
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            let result = self.native.complete();
            if result.is_err() {
                exec.terminal = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            exec.terminal = true;
            Err(CudaError::unavailable(
                "OwnedGraphCanonicalRmsNormGemmBf16Launch::finish",
            ))
        }
    }
}

impl Drop for OwnedGraphCanonicalRmsNormGemmBf16Launch<'_> {
    fn drop(&mut self) {
        let _ = self.complete_once();
    }
}

fn take_owned_graph_canonical_rms_norm_gemm_bf16_resources(
    resources: &mut Option<OwnedGraphCanonicalRmsNormGemmBf16Resources>,
    operation: &'static str,
) -> CudaResult<OwnedGraphCanonicalRmsNormGemmBf16Resources> {
    resources.take().ok_or_else(|| {
        CudaError::invalid_state(
            operation,
            "the owned canonical RMSNorm -> GEMM graph owner lost its captured resources",
        )
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn validate_graph_canonical_rms_norm_gemm_bf16_capture_preflight(
    plan: &CudaPreparedGemm,
    stream: &CudaStream,
    rms_norm_input: &CudaDeviceBuffer,
    rms_norm_weight: &CudaDeviceBuffer,
    rms_norm_output: &CudaDeviceBuffer,
    gemm_weight: &CudaDeviceBuffer,
    gemm_output: &CudaDeviceBuffer,
    gemm_workspace: &CudaDeviceBuffer,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    operation: &'static str,
) -> CudaResult<()> {
    let config = plan.config();
    if row_count != config.m() || hidden_size != config.k() {
        return Err(CudaError::invalid_argument(
            operation,
            format!(
                "canonical RMSNorm geometry ({row_count}, {hidden_size}) must exactly equal the retained GEMM (M, K) = ({}, {})",
                config.m(),
                config.k(),
            ),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(CudaError::invalid_argument(
            operation,
            "epsilon must be finite and greater than zero",
        ));
    }
    let rms_norm_matrix_bytes = row_count
        .checked_mul(hidden_size)
        .and_then(|element_count| element_count.checked_mul(std::mem::size_of::<u16>() as u64))
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "row_count * hidden_size overflows the canonical BF16 RMSNorm byte range",
            )
        })?;
    let rms_norm_weight_bytes = hidden_size
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .ok_or_else(|| {
            CudaError::out_of_range(
                operation,
                "hidden_size overflows the canonical BF16 RMSNorm weight byte range",
            )
        })?;
    if row_count == 0
        || hidden_size == 0
        || rms_norm_input.byte_len() != rms_norm_matrix_bytes
        || rms_norm_weight.byte_len() != rms_norm_weight_bytes
    {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "canonical RMSNorm input/weight allocations must be exactly {rms_norm_matrix_bytes}/{rms_norm_weight_bytes} BF16 bytes, got {}/{}",
                rms_norm_input.byte_len(),
                rms_norm_weight.byte_len(),
            ),
        ));
    }
    // The plan validates the cold strict-no-split cuBLASLt metadata and exact
    // GEMM intermediate/weight/output/workspace capacities. Passing the one
    // `rms_norm_output` wrapper as GEMM input prevents a duplicate lease.
    plan.validate_canonical_graph_capture(
        stream,
        rms_norm_output,
        gemm_weight,
        gemm_output,
        gemm_workspace,
        operation,
    )?;
    ensure_same_context(&stream.context, rms_norm_input.context_owner(), operation)?;
    ensure_same_context(&stream.context, rms_norm_weight.context_owner(), operation)?;
    rms_norm_input.ensure_idle_for_operation(operation)?;
    rms_norm_weight.ensure_idle_for_operation(operation)?;

    let buffers = [
        ("rms_norm_input", rms_norm_input),
        ("rms_norm_weight", rms_norm_weight),
        ("rms_norm_output", rms_norm_output),
        ("gemm_weight", gemm_weight),
        ("gemm_output", gemm_output),
        ("gemm_workspace", gemm_workspace),
    ];
    for (index, (left_name, left)) in buffers.iter().enumerate() {
        for (right_name, right) in buffers.iter().skip(index + 1) {
            if left.native_handle().same_allocation(right.native_handle()) {
                return Err(CudaError::invalid_argument(
                    operation,
                    format!(
                        "{left_name} and {right_name} must be distinct fixed device allocations; only the one rms_norm_output wrapper is admitted as GEMM input"
                    ),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_graph_bf16_embedding_status_d2h_capture_preflight(
    stream: &CudaStream,
    table: &CudaDeviceBuffer,
    token_ids: &CudaDeviceBuffer,
    output: &CudaDeviceBuffer,
    device_error_scratch: &CudaDeviceBuffer,
    pinned_report: &CudaPinnedHostBuffer,
    token_count: u64,
    vocabulary_size: u64,
    hidden_size: u64,
    operation: &'static str,
) -> CudaResult<()> {
    const BF16_BYTES: u64 = std::mem::size_of::<u16>() as u64;
    const U32_BYTES: u64 = std::mem::size_of::<u32>() as u64;

    // `CudaDeviceBuffer` is byte-addressed. The immutable parameter role is
    // the graph's dtype contract, and this entry point deliberately admits no
    // subspans, offsets, or F32 sibling path.
    let buffers = [
        ("table", table),
        ("token_ids", token_ids),
        ("output", output),
        ("device_error_scratch", device_error_scratch),
    ];
    for (index, (left_name, left)) in buffers.iter().enumerate() {
        for (right_name, right) in buffers.iter().skip(index + 1) {
            if left.native_handle().same_allocation(right.native_handle()) {
                return Err(CudaError::invalid_argument(
                    operation,
                    format!(
                        "{left_name} and {right_name} must be distinct fixed device allocations"
                    ),
                ));
            }
        }
    }
    for (_, buffer) in &buffers {
        ensure_same_context(&stream.context, buffer.context_owner(), operation)?;
        buffer.ensure_idle_for_operation(operation)?;
    }
    ensure_same_context(&stream.context, pinned_report.context_owner(), operation)?;
    pinned_report.ensure_idle_for_operation(operation)?;

    if token_count == 0 || vocabulary_size == 0 || hidden_size == 0 {
        return Err(CudaError::out_of_range(
            operation,
            "token_count, vocabulary_size, and hidden_size must all be non-zero for a five-node BF16 embedding status-D2H graph",
        ));
    }
    let table_elements = vocabulary_size.checked_mul(hidden_size).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "vocabulary_size * hidden_size overflows the BF16 table element range",
        )
    })?;
    let output_elements = token_count.checked_mul(hidden_size).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "token_count * hidden_size overflows the BF16 output element range",
        )
    })?;
    let table_bytes = table_elements.checked_mul(BF16_BYTES).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "BF16 table element count overflows the byte range",
        )
    })?;
    let token_bytes = token_count.checked_mul(U32_BYTES).ok_or_else(|| {
        CudaError::out_of_range(operation, "U32 token count overflows the byte range")
    })?;
    let output_bytes = output_elements.checked_mul(BF16_BYTES).ok_or_else(|| {
        CudaError::out_of_range(
            operation,
            "BF16 output element count overflows the byte range",
        )
    })?;
    for (name, actual, required) in [
        ("BF16 table", table.byte_len(), table_bytes),
        ("U32 token_ids", token_ids.byte_len(), token_bytes),
        ("BF16 output", output.byte_len(), output_bytes),
    ] {
        if actual < required {
            return Err(CudaError::out_of_range(
                operation,
                format!(
                    "{name} requires at least {required} bytes for its fixed ABI dtype/shape, but allocation capacity is {actual} bytes"
                ),
            ));
        }
    }
    if device_error_scratch.byte_len() != BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "device_error_scratch length {} must exactly equal the fixed embedding report length {BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES}",
                device_error_scratch.byte_len(),
            ),
        ));
    }
    if pinned_report.byte_len() != BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES {
        return Err(CudaError::out_of_range(
            operation,
            format!(
                "pinned report length {} must exactly equal the fixed embedding report length {BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES}",
                pinned_report.byte_len(),
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

    /// Begins a by-value C05-7 graph capture containing exactly one
    /// fixed-address whole-slab H2D node.
    ///
    /// The moved pinned source and device destination must have equal nonzero
    /// lengths in this stream's context. Replay payloads are supplied later to
    /// [`OwnedGraphH2DExec::launch_with_source`]; this entry point does not
    /// expose offsets, ranges, arbitrary node updates, eager fallback, model
    /// execution, or C07 registry wiring.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched triple through
    /// [`OwnedGraphH2DCaptureBeginError::into_resources`]. After native entry,
    /// errors return no reusable resources because native capture may retain
    /// the raw source/destination/stream leases fail-closed.
    pub fn begin_owned_graph_h2d_capture(
        self,
        source: CudaPinnedHostBuffer,
        destination: CudaDeviceBuffer,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphH2DCapture, OwnedGraphH2DCaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphH2DResources::new(self, source, destination);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphH2DResources::new(self, source, destination);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_h2d_capture";
            if let Err(error) = validate_graph_h2d_capture_preflight(
                &resources.stream,
                &resources.source,
                &resources.destination,
                OPERATION,
            ) {
                return Err(OwnedGraphH2DCaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources.stream.native.begin_graph_h2d_capture(
                resources.destination.native_handle(),
                resources.source.native_handle(),
                mode as u32,
            ) {
                Ok(native) => native,
                Err(error) => return Err(OwnedGraphH2DCaptureBeginError::terminal(error)),
            };
            begin_deferred_capture_contexts();
            let (stream, source, destination) = resources.into_parts();
            Ok(OwnedGraphH2DCapture {
                native,
                stream: Some(stream),
                source: Some(source),
                destination: Some(destination),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = mode;
            Err(OwnedGraphH2DCaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_h2d_capture"),
                resources,
            ))
        }
    }

    /// Begins a by-value C05-8 capture containing exactly one fixed-address,
    /// out-of-place BF16 SiLU node.
    ///
    /// The moved input and output remain inaccessible until graph close. This
    /// slice deliberately replays capture-time input bytes only: it does not
    /// expose spans, offsets, in-place aliasing, dtype selection, fresh input
    /// staging, node updates, executor wiring, or eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched triple through
    /// [`OwnedGraphSiluBf16CaptureBeginError::into_resources`]. After native
    /// entry, no resources are returned because CUDA may retain their raw
    /// addresses while resolving a deferred capture failure.
    pub fn begin_owned_graph_silu_bf16_capture(
        self,
        input: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        element_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphSiluBf16Capture, OwnedGraphSiluBf16CaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphSiluBf16Resources::new(self, input, output);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphSiluBf16Resources::new(self, input, output);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_silu_bf16_capture";
            if let Err(error) = validate_graph_silu_bf16_capture_preflight(
                &resources.stream,
                &resources.input,
                &resources.output,
                element_count,
                OPERATION,
            ) {
                return Err(OwnedGraphSiluBf16CaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources.stream.native.begin_graph_silu_bf16_capture(
                resources.input.native_handle(),
                resources.output.native_handle(),
                element_count,
                mode as u32,
            ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphSiluBf16CaptureBeginError::terminal(error));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphSiluBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (element_count, mode);
            Err(OwnedGraphSiluBf16CaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_silu_bf16_capture"),
                resources,
            ))
        }
    }

    /// Begins a by-value C05-10 capture containing exactly one fixed-address,
    /// out-of-place BF16 gated-multiply node.
    ///
    /// The moved activated-gate input, up input, and output remain inaccessible
    /// until graph close. This slice deliberately replays capture-time input
    /// bytes only: it does not expose spans, offsets, in-place aliasing, dtype
    /// selection, fresh input staging, node updates, SiLU fusion, executor
    /// wiring, or eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched quartet through
    /// [`OwnedGraphGatedMultiplyBf16CaptureBeginError::into_resources`]. After
    /// native entry, no resources are returned because CUDA may retain their
    /// raw addresses while resolving a deferred capture failure.
    pub fn begin_owned_graph_gated_multiply_bf16_capture(
        self,
        activated_gate: CudaDeviceBuffer,
        up: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        element_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphGatedMultiplyBf16Capture, OwnedGraphGatedMultiplyBf16CaptureBeginError>
    {
        #[cfg(feature = "cuda")]
        let mut resources =
            OwnedGraphGatedMultiplyBf16Resources::new(self, activated_gate, up, output);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphGatedMultiplyBf16Resources::new(self, activated_gate, up, output);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_gated_multiply_bf16_capture";
            if let Err(error) = validate_graph_gated_multiply_bf16_capture_preflight(
                &resources.stream,
                &resources.activated_gate,
                &resources.up,
                &resources.output,
                element_count,
                OPERATION,
            ) {
                return Err(OwnedGraphGatedMultiplyBf16CaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources
                .stream
                .native
                .begin_graph_gated_multiply_bf16_capture(
                    resources.activated_gate.native_handle(),
                    resources.up.native_handle(),
                    resources.output.native_handle(),
                    element_count,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphGatedMultiplyBf16CaptureBeginError::terminal(
                        error,
                    ));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphGatedMultiplyBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (element_count, mode);
            Err(OwnedGraphGatedMultiplyBf16CaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_gated_multiply_bf16_capture"),
                resources,
            ))
        }
    }

    /// Begins a by-value C05-11 capture containing exactly one fixed-address,
    /// out-of-place BF16 residual-add node.
    ///
    /// The moved left input, right input, and output remain inaccessible until
    /// graph close. This slice replays capture-time inputs only; it does not
    /// expose in-place aliases, offsets, fresh inputs, node updates, fused
    /// normalization, executor wiring, or eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched quartet through
    /// [`OwnedGraphResidualAddBf16CaptureBeginError::into_resources`]. After
    /// native entry, no resource is returned because CUDA may retain the raw
    /// addresses while resolving an ambiguous capture failure.
    pub fn begin_owned_graph_residual_add_bf16_capture(
        self,
        left: CudaDeviceBuffer,
        right: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        element_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphResidualAddBf16Capture, OwnedGraphResidualAddBf16CaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphResidualAddBf16Resources::new(self, left, right, output);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphResidualAddBf16Resources::new(self, left, right, output);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_residual_add_bf16_capture";
            if let Err(error) = validate_graph_residual_add_bf16_capture_preflight(
                &resources.stream,
                &resources.left,
                &resources.right,
                &resources.output,
                element_count,
                OPERATION,
            ) {
                return Err(OwnedGraphResidualAddBf16CaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources
                .stream
                .native
                .begin_graph_residual_add_bf16_capture(
                    resources.left.native_handle(),
                    resources.right.native_handle(),
                    resources.output.native_handle(),
                    element_count,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphResidualAddBf16CaptureBeginError::terminal(error));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphResidualAddBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (element_count, mode);
            Err(OwnedGraphResidualAddBf16CaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_residual_add_bf16_capture"),
                resources,
            ))
        }
    }

    /// Begins a by-value C05-12 capture containing exactly one fixed-address,
    /// out-of-place canonical BF16 RMSNorm node.
    ///
    /// The moved input, learned weight, and output remain inaccessible until
    /// graph close. This slice follows only the generic eager
    /// [`crate::rms_norm`] BF16 reduction and normalized-storage-rounding
    /// contract; it excludes
    /// profile-specific SmolLM2 and Fixed37 RMSNorm, fused RMSNorm, C07
    /// executor integration, in-place aliases, spans, offsets, fresh inputs,
    /// node updates, and eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched quartet through
    /// [`OwnedGraphCanonicalRmsNormBf16CaptureBeginError::into_resources`].
    /// After native entry, no resource is returned because CUDA may retain the
    /// raw addresses while resolving an ambiguous capture failure.
    pub fn begin_owned_graph_canonical_rms_norm_bf16_capture(
        self,
        input: CudaDeviceBuffer,
        weight: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        row_count: u64,
        hidden_size: u64,
        epsilon: f32,
        mode: CudaGraphCaptureMode,
    ) -> Result<
        OwnedGraphCanonicalRmsNormBf16Capture,
        OwnedGraphCanonicalRmsNormBf16CaptureBeginError,
    > {
        #[cfg(feature = "cuda")]
        let mut resources =
            OwnedGraphCanonicalRmsNormBf16Resources::new(self, input, weight, output);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphCanonicalRmsNormBf16Resources::new(self, input, weight, output);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_canonical_rms_norm_bf16_capture";
            if let Err(error) = validate_graph_canonical_rms_norm_bf16_capture_preflight(
                &resources.stream,
                &resources.input,
                &resources.weight,
                &resources.output,
                row_count,
                hidden_size,
                epsilon,
                OPERATION,
            ) {
                return Err(
                    OwnedGraphCanonicalRmsNormBf16CaptureBeginError::recoverable(error, resources),
                );
            }
            let native = match resources
                .stream
                .native
                .begin_graph_canonical_rms_norm_bf16_capture(
                    resources.input.native_handle(),
                    resources.weight.native_handle(),
                    resources.output.native_handle(),
                    row_count,
                    hidden_size,
                    epsilon,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphCanonicalRmsNormBf16CaptureBeginError::terminal(
                        error,
                    ));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphCanonicalRmsNormBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (row_count, hidden_size, epsilon, mode);
            Err(
                OwnedGraphCanonicalRmsNormBf16CaptureBeginError::recoverable(
                    CudaError::unavailable(
                        "CudaStream::begin_owned_graph_canonical_rms_norm_bf16_capture",
                    ),
                    resources,
                ),
            )
        }
    }

    /// Begins a by-value C05-13 capture containing exactly one fixed-address,
    /// deterministic BF16 argmax node.
    ///
    /// The moved BF16 logits and U32 result records remain inaccessible until
    /// graph close. This slice follows only [`crate::deterministic_bf16_argmax`]:
    /// finite ties choose the lower token ID, and any non-finite input writes
    /// the fixed invalid-token/non-finite status pair. It excludes row gather,
    /// token/status transfer, completion dependencies, C07 executor
    /// integration, fresh inputs, spans, offsets, node updates, sampling
    /// policy, and eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched resource trio through
    /// [`OwnedGraphBf16ArgmaxCaptureBeginError::into_resources`]. After native
    /// entry, no resource is returned because CUDA may retain the raw addresses
    /// while resolving an ambiguous capture failure.
    pub fn begin_owned_graph_bf16_argmax_capture(
        self,
        logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
        row_count: u64,
        vocabulary_size: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphBf16ArgmaxCapture, OwnedGraphBf16ArgmaxCaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphBf16ArgmaxResources::new(self, logits, results);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphBf16ArgmaxResources::new(self, logits, results);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_bf16_argmax_capture";
            if let Err(error) = validate_graph_bf16_argmax_capture_preflight(
                &resources.stream,
                &resources.logits,
                &resources.results,
                row_count,
                vocabulary_size,
                OPERATION,
            ) {
                return Err(OwnedGraphBf16ArgmaxCaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources.stream.native.begin_graph_bf16_argmax_capture(
                resources.logits.native_handle(),
                resources.results.native_handle(),
                row_count,
                vocabulary_size,
                mode as u32,
            ) {
                Ok(native) => native,
                Err(error) => return Err(OwnedGraphBf16ArgmaxCaptureBeginError::terminal(error)),
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphBf16ArgmaxCapture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (row_count, vocabulary_size, mode);
            Err(OwnedGraphBf16ArgmaxCaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_bf16_argmax_capture"),
                resources,
            ))
        }
    }

    /// Begins a by-value C05-14 capture containing exactly one fixed-address,
    /// out-of-place BF16 row-gather node.
    ///
    /// The moved BF16 input, U32 row-index allocation, and BF16 output remain
    /// inaccessible until graph close. `row_indices_host` is only a temporary
    /// safe-validation mirror: it must be unique and in range like eager
    /// [`crate::row_gather`], is never retained by this owner, and does not
    /// bind the caller's separately staged device index bytes. Raw malformed
    /// device indices retain the eager per-row BF16-NaN behavior. This slice
    /// excludes H2D/D2H, argmax, token/status transfer, completion
    /// dependencies, C07 executor integration, fresh inputs, spans, offsets,
    /// node updates, sampling policy, and eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched resource quartet through
    /// [`OwnedGraphBf16RowGatherCaptureBeginError::into_resources`]. After
    /// native entry, no resource is returned because CUDA may retain the raw
    /// addresses while resolving an ambiguous capture failure.
    pub fn begin_owned_graph_bf16_row_gather_capture(
        self,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        row_indices_host: &[u32],
        input_row_count: u64,
        column_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphBf16RowGatherCapture, OwnedGraphBf16RowGatherCaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphBf16RowGatherResources::new(self, input, row_indices, output);
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphBf16RowGatherResources::new(self, input, row_indices, output);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_bf16_row_gather_capture";
            let output_row_count = match u64::try_from(row_indices_host.len()) {
                Ok(value) => value,
                Err(_) => {
                    return Err(OwnedGraphBf16RowGatherCaptureBeginError::recoverable(
                        CudaError::out_of_range(
                            OPERATION,
                            "row_indices_host length exceeds the U64 row-count range",
                        ),
                        resources,
                    ));
                }
            };
            if let Err(error) = validate_graph_bf16_row_gather_capture_preflight(
                &resources.stream,
                &resources.input,
                &resources.row_indices,
                &resources.output,
                row_indices_host,
                input_row_count,
                output_row_count,
                column_count,
                OPERATION,
            ) {
                return Err(OwnedGraphBf16RowGatherCaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources.stream.native.begin_graph_bf16_row_gather_capture(
                resources.input.native_handle(),
                resources.row_indices.native_handle(),
                resources.output.native_handle(),
                input_row_count,
                output_row_count,
                column_count,
                mode as u32,
            ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphBf16RowGatherCaptureBeginError::terminal(error));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphBf16RowGatherCapture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (row_indices_host, input_row_count, column_count, mode);
            Err(OwnedGraphBf16RowGatherCaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_bf16_row_gather_capture"),
                resources,
            ))
        }
    }

    /// Begins a by-value C05-15 capture containing exactly one fixed-address,
    /// device-only BF16 row-gather then deterministic-argmax node chain.
    ///
    /// The moved BF16 input, U32 row-index allocation, BF16 gathered-logits
    /// allocation, and U32 [`Bf16ArgmaxResult`] allocation remain inaccessible
    /// until graph close. `row_indices_host` is only a temporary safe-
    /// validation mirror: it must be unique and in range like eager
    /// [`crate::row_gather`], is never retained by this owner, and does not
    /// bind the caller's separately staged device index bytes. Raw malformed
    /// device indices retain the eager per-row BF16-NaN behavior, and the
    /// captured argmax then writes the eager non-finite result record. This
    /// slice excludes H2D/D2H, token/status transfer, host completion
    /// semantics, C07 executor integration, fresh inputs, spans, offsets,
    /// node updates, sampling policy, and eager fallback.
    ///
    /// # Errors
    ///
    /// Rust preflight failures return the untouched resource quintet through
    /// [`OwnedGraphBf16RowGatherArgmaxCaptureBeginError::into_resources`].
    /// After native entry, no resource is returned because CUDA may retain the
    /// raw addresses while resolving an ambiguous capture failure.
    pub fn begin_owned_graph_bf16_row_gather_argmax_capture(
        self,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        gathered_logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
        row_indices_host: &[u32],
        input_row_count: u64,
        vocabulary_size: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphBf16RowGatherArgmaxCapture, OwnedGraphBf16RowGatherArgmaxCaptureBeginError>
    {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphBf16RowGatherArgmaxResources::new(
            self,
            input,
            row_indices,
            gathered_logits,
            results,
        );
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphBf16RowGatherArgmaxResources::new(
            self,
            input,
            row_indices,
            gathered_logits,
            results,
        );
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_bf16_row_gather_argmax_capture";
            let output_row_count = match u64::try_from(row_indices_host.len()) {
                Ok(value) => value,
                Err(_) => {
                    return Err(OwnedGraphBf16RowGatherArgmaxCaptureBeginError::recoverable(
                        CudaError::out_of_range(
                            OPERATION,
                            "row_indices_host length exceeds the U64 row-count range",
                        ),
                        resources,
                    ));
                }
            };
            if let Err(error) = validate_graph_bf16_row_gather_argmax_capture_preflight(
                &resources.stream,
                &resources.input,
                &resources.row_indices,
                &resources.gathered_logits,
                &resources.results,
                row_indices_host,
                input_row_count,
                output_row_count,
                vocabulary_size,
                OPERATION,
            ) {
                return Err(OwnedGraphBf16RowGatherArgmaxCaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources
                .stream
                .native
                .begin_graph_bf16_row_gather_argmax_capture(
                    resources.input.native_handle(),
                    resources.row_indices.native_handle(),
                    resources.gathered_logits.native_handle(),
                    resources.results.native_handle(),
                    input_row_count,
                    output_row_count,
                    vocabulary_size,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphBf16RowGatherArgmaxCaptureBeginError::terminal(
                        error,
                    ));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphBf16RowGatherArgmaxCapture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (row_indices_host, input_row_count, vocabulary_size, mode);
            Err(OwnedGraphBf16RowGatherArgmaxCaptureBeginError::recoverable(
                CudaError::unavailable(
                    "CudaStream::begin_owned_graph_bf16_row_gather_argmax_capture",
                ),
                resources,
            ))
        }
    }

    /// Begins one C05-16 fixed-address BF16 row-gather -> argmax -> pinned
    /// result-D2H capture.
    ///
    /// The moved resources remain inaccessible until known graph release. The
    /// temporary `row_indices_host` mirror provides only eager-equivalent
    /// host-side geometry/index validation and is never retained. Completion
    /// later exposes raw result bytes only; no token/status validation,
    /// scheduler commit, C06/C07 dispatch, or eager fallback is implied.
    ///
    /// # Errors
    ///
    /// Rust-side preflight failures return the untouched resource sextet.
    /// Native-entry errors retain all raw addresses fail-closed.
    pub fn begin_owned_graph_bf16_row_gather_argmax_d2h_capture(
        self,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        gathered_logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
        pinned_results: CudaPinnedHostBuffer,
        row_indices_host: &[u32],
        input_row_count: u64,
        vocabulary_size: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<
        OwnedGraphBf16RowGatherArgmaxD2HCapture,
        OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError,
    > {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphBf16RowGatherArgmaxD2HResources::new(
            self,
            input,
            row_indices,
            gathered_logits,
            results,
            pinned_results,
        );
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphBf16RowGatherArgmaxD2HResources::new(
            self,
            input,
            row_indices,
            gathered_logits,
            results,
            pinned_results,
        );
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str =
                "CudaStream::begin_owned_graph_bf16_row_gather_argmax_d2h_capture";
            let output_row_count = match u64::try_from(row_indices_host.len()) {
                Ok(value) => value,
                Err(_) => {
                    return Err(
                        OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError::recoverable(
                            CudaError::out_of_range(
                                OPERATION,
                                "row_indices_host length exceeds the U64 row-count range",
                            ),
                            resources,
                        ),
                    );
                }
            };
            if let Err(error) = validate_graph_bf16_row_gather_argmax_d2h_capture_preflight(
                &resources.stream,
                &resources.input,
                &resources.row_indices,
                &resources.gathered_logits,
                &resources.results,
                &resources.pinned_results,
                row_indices_host,
                input_row_count,
                output_row_count,
                vocabulary_size,
                OPERATION,
            ) {
                return Err(
                    OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError::recoverable(
                        error, resources,
                    ),
                );
            }
            let native = match resources
                .stream
                .native
                .begin_graph_bf16_row_gather_argmax_d2h_capture(
                    resources.input.native_handle(),
                    resources.row_indices.native_handle(),
                    resources.gathered_logits.native_handle(),
                    resources.results.native_handle(),
                    resources.pinned_results.native_handle(),
                    input_row_count,
                    output_row_count,
                    vocabulary_size,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError::terminal(
                        error,
                    ));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphBf16RowGatherArgmaxD2HCapture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (row_indices_host, input_row_count, vocabulary_size, mode);
            Err(
                OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError::recoverable(
                    CudaError::unavailable(
                        "CudaStream::begin_owned_graph_bf16_row_gather_argmax_d2h_capture",
                    ),
                    resources,
                ),
            )
        }
    }

    /// Begins one C05-17 fixed-address BF16 indexed-RoPE graph capture.
    ///
    /// The temporary host position mirror provides eager-equivalent geometry
    /// and bounds validation only; it is not retained and does not attest to
    /// bytes in the fixed device positions allocation. This graph contains
    /// exactly one indexed-RoPE node and deliberately excludes H2D/D2H,
    /// completion receipts, final norm, C07 executor wiring, sampling,
    /// scheduler commit, node updates, and eager fallback.
    ///
    /// # Errors
    ///
    /// Rust-side preflight failures return the untouched resource sextet.
    /// Native-entry errors retain every raw address fail-closed.
    pub fn begin_owned_graph_indexed_rope_bf16_capture(
        self,
        input: CudaDeviceBuffer,
        cos: CudaDeviceBuffer,
        sin: CudaDeviceBuffer,
        positions: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        positions_host: &[u32],
        head_count: u64,
        head_size: u64,
        rotary_dimension: u64,
        table_position_count: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphIndexedRopeBf16Capture, OwnedGraphIndexedRopeBf16CaptureBeginError> {
        #[cfg(feature = "cuda")]
        let mut resources =
            OwnedGraphIndexedRopeBf16Resources::new(self, input, cos, sin, positions, output);
        #[cfg(not(feature = "cuda"))]
        let resources =
            OwnedGraphIndexedRopeBf16Resources::new(self, input, cos, sin, positions, output);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_indexed_rope_bf16_capture";
            let active_row_count = match u64::try_from(positions_host.len()) {
                Ok(value) => value,
                Err(_) => {
                    return Err(OwnedGraphIndexedRopeBf16CaptureBeginError::recoverable(
                        CudaError::out_of_range(
                            OPERATION,
                            "positions_host length exceeds the U64 active-row-count range",
                        ),
                        resources,
                    ));
                }
            };
            if let Err(error) = validate_graph_indexed_rope_bf16_capture_preflight(
                &resources.stream,
                &resources.input,
                &resources.cos,
                &resources.sin,
                &resources.positions,
                &resources.output,
                positions_host,
                active_row_count,
                head_count,
                head_size,
                rotary_dimension,
                table_position_count,
                OPERATION,
            ) {
                return Err(OwnedGraphIndexedRopeBf16CaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources
                .stream
                .native
                .begin_graph_indexed_rope_bf16_capture(
                    resources.input.native_handle(),
                    resources.cos.native_handle(),
                    resources.sin.native_handle(),
                    resources.positions.native_handle(),
                    resources.output.native_handle(),
                    positions_host,
                    active_row_count,
                    head_count,
                    head_size,
                    rotary_dimension,
                    table_position_count,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphIndexedRopeBf16CaptureBeginError::terminal(error));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphIndexedRopeBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                positions_host,
                head_count,
                head_size,
                rotary_dimension,
                table_position_count,
                mode,
            );
            Err(OwnedGraphIndexedRopeBf16CaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_indexed_rope_bf16_capture"),
                resources,
            ))
        }
    }

    /// Begins one C05-18 fixed-address BF16 ragged paged-K/V cache-write graph
    /// capture.
    ///
    /// `batch_host` is a temporary validated admission witness only. Its
    /// packed CSR, canonical valid-token, physical-ID, and logical-row checks
    /// have completed before this call; no host slice is retained and no
    /// device-metadata byte identity is claimed after capture begins. The
    /// graph contains exactly one K/V scatter node and deliberately excludes
    /// metadata H2D, projection, attention read, scheduler commit, C07
    /// executor wiring, sampling, node updates, and eager fallback.
    ///
    /// # Errors
    ///
    /// Rust-side preflight failures return the untouched stream plus nine
    /// allocation resource bundle. Native-entry errors retain every raw
    /// address fail-closed.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture(
        self,
        key_source: CudaDeviceBuffer,
        value_source: CudaDeviceBuffer,
        key_pool: CudaDeviceBuffer,
        value_pool: CudaDeviceBuffer,
        sequence_block_offsets: CudaDeviceBuffer,
        block_ids: CudaDeviceBuffer,
        valid_tokens: CudaDeviceBuffer,
        row_sequence_slots: CudaDeviceBuffer,
        row_positions: CudaDeviceBuffer,
        batch_host: PackedBatchHostV1<'_>,
        key_value_head_count: u64,
        head_size: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<
        OwnedGraphRaggedPagedKvCacheWriteBf16Capture,
        OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError,
    > {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphRaggedPagedKvCacheWriteBf16Resources::new(
            self,
            key_source,
            value_source,
            key_pool,
            value_pool,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        );
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphRaggedPagedKvCacheWriteBf16Resources::new(
            self,
            key_source,
            value_source,
            key_pool,
            value_pool,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        );
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str =
                "CudaStream::begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture";
            if let Err(error) = validate_graph_ragged_paged_kv_cache_write_bf16_capture_preflight(
                &resources.stream,
                &resources.key_source,
                &resources.value_source,
                &resources.key_pool,
                &resources.value_pool,
                &resources.sequence_block_offsets,
                &resources.block_ids,
                &resources.valid_tokens,
                &resources.row_sequence_slots,
                &resources.row_positions,
                batch_host,
                key_value_head_count,
                head_size,
                OPERATION,
            ) {
                return Err(
                    OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError::recoverable(
                        error, resources,
                    ),
                );
            }
            let native = match resources
                .stream
                .native
                .begin_graph_ragged_paged_kv_cache_write_bf16_capture(
                    resources.key_source.native_handle(),
                    resources.value_source.native_handle(),
                    resources.key_pool.native_handle(),
                    resources.value_pool.native_handle(),
                    resources.sequence_block_offsets.native_handle(),
                    resources.block_ids.native_handle(),
                    resources.valid_tokens.native_handle(),
                    resources.row_sequence_slots.native_handle(),
                    resources.row_positions.native_handle(),
                    batch_host.sequence_count(),
                    batch_host.block_count(),
                    batch_host.active_row_count(),
                    batch_host.physical_block_count(),
                    key_value_head_count,
                    head_size,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(
                        OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError::terminal(error),
                    );
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphRaggedPagedKvCacheWriteBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch_host, key_value_head_count, head_size, mode);
            Err(
                OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError::recoverable(
                    CudaError::unavailable(
                        "CudaStream::begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture",
                    ),
                    resources,
                ),
            )
        }
    }

    /// Begins one C05-19 fixed-address BF16 grouped ragged paged-attention
    /// graph capture.
    ///
    /// `batch_host` is a temporary validated admission witness only. Its
    /// packed CSR, canonical valid-token, physical-ID, and logical-row checks
    /// complete before this call; no host slice is retained and no
    /// device-metadata byte identity is claimed after capture begins. The
    /// graph contains exactly one D64 grouped GQA attention node. Metadata
    /// H2D, projection, cache writes, scheduler commit, C07 executor wiring,
    /// sampling, node updates, and eager fallback remain outside this slice.
    ///
    /// # Errors
    ///
    /// Rust-side preflight failures return the untouched stream plus nine
    /// allocation resource bundle. Native-entry errors retain every raw
    /// address fail-closed.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_owned_graph_grouped_ragged_paged_attention_bf16_capture(
        self,
        query: CudaDeviceBuffer,
        key_pool: CudaDeviceBuffer,
        value_pool: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        sequence_block_offsets: CudaDeviceBuffer,
        block_ids: CudaDeviceBuffer,
        valid_tokens: CudaDeviceBuffer,
        row_sequence_slots: CudaDeviceBuffer,
        row_positions: CudaDeviceBuffer,
        batch_host: PackedBatchHostV1<'_>,
        query_head_count: u64,
        key_value_head_count: u64,
        output_row_count: u64,
        scale: f32,
        mode: CudaGraphCaptureMode,
    ) -> Result<
        OwnedGraphGroupedRaggedPagedAttentionBf16Capture,
        OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError,
    > {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphGroupedRaggedPagedAttentionBf16Resources::new(
            self,
            query,
            key_pool,
            value_pool,
            output,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        );
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphGroupedRaggedPagedAttentionBf16Resources::new(
            self,
            query,
            key_pool,
            value_pool,
            output,
            sequence_block_offsets,
            block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
        );
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str =
                "CudaStream::begin_owned_graph_grouped_ragged_paged_attention_bf16_capture";
            if let Err(error) = validate_graph_grouped_ragged_paged_attention_bf16_capture_preflight(
                &resources.stream,
                &resources.query,
                &resources.key_pool,
                &resources.value_pool,
                &resources.output,
                &resources.sequence_block_offsets,
                &resources.block_ids,
                &resources.valid_tokens,
                &resources.row_sequence_slots,
                &resources.row_positions,
                batch_host,
                query_head_count,
                key_value_head_count,
                output_row_count,
                scale,
                OPERATION,
            ) {
                return Err(
                    OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError::recoverable(
                        error, resources,
                    ),
                );
            }
            let native = match resources
                .stream
                .native
                .begin_graph_grouped_ragged_paged_attention_bf16_capture(
                    resources.query.native_handle(),
                    resources.key_pool.native_handle(),
                    resources.value_pool.native_handle(),
                    resources.output.native_handle(),
                    resources.sequence_block_offsets.native_handle(),
                    resources.block_ids.native_handle(),
                    resources.valid_tokens.native_handle(),
                    resources.row_sequence_slots.native_handle(),
                    resources.row_positions.native_handle(),
                    batch_host.sequence_count(),
                    batch_host.block_count(),
                    batch_host.active_row_count(),
                    batch_host.physical_block_count(),
                    query_head_count,
                    key_value_head_count,
                    output_row_count,
                    scale,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(
                        OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError::terminal(error),
                    );
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphGroupedRaggedPagedAttentionBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                batch_host,
                query_head_count,
                key_value_head_count,
                output_row_count,
                scale,
                mode,
            );
            Err(
                OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError::recoverable(
                    CudaError::unavailable(
                        "CudaStream::begin_owned_graph_grouped_ragged_paged_attention_bf16_capture",
                    ),
                    resources,
                ),
            )
        }
    }

    /// Begins one C05-20 fixed-address BF16 embedding validation-status D2H
    /// graph capture.
    ///
    /// No token-ID host mirror is accepted or retained: each replay validates
    /// the fixed U32 device token allocation and publishes either success or
    /// the earliest OOB token through the fixed pinned report. The graph
    /// records exactly reset, validate, BF16 gather, finalize, and D2H. Token
    /// H2D, F32 embedding, table-residency policy, scheduler commit, C07
    /// executor wiring, node updates, and eager fallback remain outside this
    /// narrow ownership slice.
    ///
    /// # Errors
    ///
    /// Rust-side preflight failures return the untouched stream plus four
    /// device allocations and pinned report. Native-entry errors retain every
    /// raw address fail-closed.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_owned_graph_bf16_embedding_status_d2h_capture(
        self,
        table: CudaDeviceBuffer,
        token_ids: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        device_error_scratch: CudaDeviceBuffer,
        pinned_report: CudaPinnedHostBuffer,
        token_count: u64,
        vocabulary_size: u64,
        hidden_size: u64,
        mode: CudaGraphCaptureMode,
    ) -> Result<
        OwnedGraphBf16EmbeddingStatusD2HCapture,
        OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError,
    > {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphBf16EmbeddingStatusD2HResources::new(
            self,
            table,
            token_ids,
            output,
            device_error_scratch,
            pinned_report,
        );
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphBf16EmbeddingStatusD2HResources::new(
            self,
            table,
            token_ids,
            output,
            device_error_scratch,
            pinned_report,
        );
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str =
                "CudaStream::begin_owned_graph_bf16_embedding_status_d2h_capture";
            if let Err(error) = validate_graph_bf16_embedding_status_d2h_capture_preflight(
                &resources.stream,
                &resources.table,
                &resources.token_ids,
                &resources.output,
                &resources.device_error_scratch,
                &resources.pinned_report,
                token_count,
                vocabulary_size,
                hidden_size,
                OPERATION,
            ) {
                return Err(
                    OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError::recoverable(
                        error, resources,
                    ),
                );
            }
            let native = match resources
                .stream
                .native
                .begin_graph_bf16_embedding_status_d2h_capture(
                    resources.table.native_handle(),
                    resources.token_ids.native_handle(),
                    resources.output.native_handle(),
                    resources.device_error_scratch.native_handle(),
                    resources.pinned_report.native_handle(),
                    token_count,
                    vocabulary_size,
                    hidden_size,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError::terminal(
                        error,
                    ));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphBf16EmbeddingStatusD2HCapture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (token_count, vocabulary_size, hidden_size, mode);
            Err(
                OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError::recoverable(
                    CudaError::unavailable(
                        "CudaStream::begin_owned_graph_bf16_embedding_status_d2h_capture",
                    ),
                    resources,
                ),
            )
        }
    }

    /// Begins one C05-21 fixed-address canonical cuBLASLt GEMM graph capture.
    ///
    /// The caller moves a cold-prepared strict-no-split BF16/F32 plan and four
    /// distinct whole allocations into the graph owner. The graph records one
    /// fixed `cublasLtMatmul`; it does not accept spans, offsets, fresh input,
    /// dynamic alpha/beta, alternate reduction, command batches, C07 executor
    /// wiring, or an eager fallback.
    ///
    /// # Errors
    ///
    /// Pure Rust preflight failures return the untouched plan/stream/allocation
    /// bundle. An error after native entry leaves no reusable bundle because
    /// capture recovery may retain every raw address fail-closed.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_owned_graph_canonical_gemm_bf16_capture(
        self,
        plan: CudaPreparedGemm,
        input: CudaDeviceBuffer,
        weight: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
        workspace: CudaDeviceBuffer,
        mode: CudaGraphCaptureMode,
    ) -> Result<OwnedGraphCanonicalGemmBf16Capture, OwnedGraphCanonicalGemmBf16CaptureBeginError>
    {
        #[cfg(feature = "cuda")]
        let mut resources =
            OwnedGraphCanonicalGemmBf16Resources::new(self, plan, input, weight, output, workspace);
        #[cfg(not(feature = "cuda"))]
        let resources =
            OwnedGraphCanonicalGemmBf16Resources::new(self, plan, input, weight, output, workspace);
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str = "CudaStream::begin_owned_graph_canonical_gemm_bf16_capture";
            if let Err(error) = resources.plan.validate_canonical_graph_capture(
                &resources.stream,
                &resources.input,
                &resources.weight,
                &resources.output,
                &resources.workspace,
                OPERATION,
            ) {
                return Err(OwnedGraphCanonicalGemmBf16CaptureBeginError::recoverable(
                    error, resources,
                ));
            }
            let native = match resources.plan.begin_canonical_graph_capture_native(
                &mut resources.stream,
                &resources.input,
                &resources.weight,
                &resources.output,
                &resources.workspace,
                mode as u32,
            ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(OwnedGraphCanonicalGemmBf16CaptureBeginError::terminal(
                        error,
                    ));
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphCanonicalGemmBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = mode;
            Err(OwnedGraphCanonicalGemmBf16CaptureBeginError::recoverable(
                CudaError::unavailable("CudaStream::begin_owned_graph_canonical_gemm_bf16_capture"),
                resources,
            ))
        }
    }

    /// Begins one C05-22 fixed-address canonical BF16 RMSNorm -> cuBLASLt
    /// GEMM graph capture.
    ///
    /// The caller moves one cold-prepared strict-no-split BF16/F32 GEMM plan,
    /// canonical RMSNorm input/weight, the one intermediate allocation, GEMM
    /// weight/final output/workspace, and this stream into the owner. The
    /// intermediate is supplied exactly once: it is the RMSNorm output and
    /// the captured GEMM input. No fresh replay input, spans, offsets, node
    /// updates, command batch, C07 executor wiring, or eager fallback is
    /// exposed by this narrow two-node slice.
    ///
    /// # Errors
    ///
    /// Pure Rust preflight validates exact geometry (`row_count == M`,
    /// `hidden_size == K`), every allocation capacity/context/alias, epsilon,
    /// and the cold strict plan before native entry; such failures recover the
    /// untouched bundle. A native-entry error fails closed because CUDA may
    /// retain all raw addresses during recovery.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_owned_graph_canonical_rms_norm_gemm_bf16_capture(
        self,
        plan: CudaPreparedGemm,
        rms_norm_input: CudaDeviceBuffer,
        rms_norm_weight: CudaDeviceBuffer,
        rms_norm_output: CudaDeviceBuffer,
        gemm_weight: CudaDeviceBuffer,
        gemm_output: CudaDeviceBuffer,
        gemm_workspace: CudaDeviceBuffer,
        row_count: u64,
        hidden_size: u64,
        epsilon: f32,
        mode: CudaGraphCaptureMode,
    ) -> Result<
        OwnedGraphCanonicalRmsNormGemmBf16Capture,
        OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError,
    > {
        #[cfg(feature = "cuda")]
        let mut resources = OwnedGraphCanonicalRmsNormGemmBf16Resources::new(
            self,
            plan,
            rms_norm_input,
            rms_norm_weight,
            rms_norm_output,
            gemm_weight,
            gemm_output,
            gemm_workspace,
        );
        #[cfg(not(feature = "cuda"))]
        let resources = OwnedGraphCanonicalRmsNormGemmBf16Resources::new(
            self,
            plan,
            rms_norm_input,
            rms_norm_weight,
            rms_norm_output,
            gemm_weight,
            gemm_output,
            gemm_workspace,
        );
        #[cfg(feature = "cuda")]
        {
            const OPERATION: &str =
                "CudaStream::begin_owned_graph_canonical_rms_norm_gemm_bf16_capture";
            if let Err(error) = validate_graph_canonical_rms_norm_gemm_bf16_capture_preflight(
                &resources.plan,
                &resources.stream,
                &resources.rms_norm_input,
                &resources.rms_norm_weight,
                &resources.rms_norm_output,
                &resources.gemm_weight,
                &resources.gemm_output,
                &resources.gemm_workspace,
                row_count,
                hidden_size,
                epsilon,
                OPERATION,
            ) {
                return Err(
                    OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError::recoverable(
                        error, resources,
                    ),
                );
            }
            let native = match resources
                .plan
                .begin_canonical_rms_norm_gemm_graph_capture_native(
                    &mut resources.stream,
                    &resources.rms_norm_input,
                    &resources.rms_norm_weight,
                    &resources.rms_norm_output,
                    &resources.gemm_weight,
                    &resources.gemm_output,
                    &resources.gemm_workspace,
                    row_count,
                    hidden_size,
                    epsilon,
                    mode as u32,
                ) {
                Ok(native) => native,
                Err(error) => {
                    return Err(
                        OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError::terminal(error),
                    );
                }
            };
            begin_deferred_capture_contexts();
            Ok(OwnedGraphCanonicalRmsNormGemmBf16Capture {
                native,
                resources: Some(resources),
                active: true,
                enqueued: false,
                enqueue_failed: false,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (row_count, hidden_size, epsilon, mode);
            Err(
                OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError::recoverable(
                    CudaError::unavailable(
                        "CudaStream::begin_owned_graph_canonical_rms_norm_gemm_bf16_capture",
                    ),
                    resources,
                ),
            )
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

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "remote GPU"]
    fn raw_c05_22_alias_preflight_rejects_every_pair_before_capture_and_preserves_resources()
    -> Result<(), Box<dyn std::error::Error>> {
        // The safe by-value entry point cannot manufacture two owners for one
        // device allocation. Exercise every pair in the raw ABI gate with
        // borrowed native handles, without exposing an aliasing API to callers.
        const M: u64 = 1;
        const N: u64 = 576;
        const K: u64 = 576;
        const EPSILON: f32 = 1.0e-5;

        let runtime = crate::CudaRuntime::initialize()?;
        assert!(
            runtime.device_count() > 0,
            "remote GPU runner has no CUDA device"
        );
        let context = runtime.device(0)?.create_context()?;
        let allocation_baseline = context.allocation_stats()?;
        assert!(allocation_baseline.is_zero());

        let config = crate::CudaGemmConfig::new(M, N, K, 8 * 1024 * 1024)?;
        let plan = context.prepare_gemm(config)?;
        let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
        let mut stream = context.create_stream()?;
        let rms_norm_input = context.allocate_device_buffer(config.input_bytes())?;
        let rms_norm_weight = context.allocate_device_buffer(K * 2)?;
        let rms_norm_output = context.allocate_device_buffer(config.input_bytes())?;
        let gemm_weight = context.allocate_device_buffer(config.weight_bytes())?;
        let gemm_output = context.allocate_device_buffer(config.output_bytes())?;
        let workspace = context.allocate_device_buffer(workspace_bytes)?;
        let allocation_with_resources = context.allocation_stats()?;

        let fixed_buffers = [
            &rms_norm_input,
            &rms_norm_weight,
            &rms_norm_output,
            &gemm_weight,
            &gemm_output,
            &workspace,
        ];
        let role_names = [
            "rms_norm_input",
            "rms_norm_weight",
            "rms_norm_output",
            "gemm_weight",
            "gemm_output",
            "gemm_workspace",
        ];
        for left in 0..fixed_buffers.len() {
            for right in (left + 1)..fixed_buffers.len() {
                let mut roles = fixed_buffers;
                roles[right] = roles[left];
                let error = match plan.begin_canonical_rms_norm_gemm_graph_capture_native(
                    &mut stream,
                    roles[0],
                    roles[1],
                    roles[2],
                    roles[3],
                    roles[4],
                    roles[5],
                    M,
                    K,
                    EPSILON,
                    CudaGraphCaptureMode::ThreadLocal as u32,
                ) {
                    Ok(_) => panic!(
                        "raw C05-22 duplicate allocation {} / {} unexpectedly captured",
                        role_names[left], role_names[right]
                    ),
                    Err(error) => error,
                };
                assert_eq!(
                    error.kind(),
                    CudaErrorKind::InvalidArgument,
                    "raw C05-22 alias {} / {} must reject before capture",
                    role_names[left],
                    role_names[right]
                );
                assert_eq!(
                    context.allocation_stats()?,
                    allocation_with_resources,
                    "raw C05-22 alias {} / {} must not alter allocation accounting",
                    role_names[left],
                    role_names[right]
                );
            }
        }

        plan.close()?;
        workspace.close()?;
        gemm_output.close()?;
        gemm_weight.close()?;
        rms_norm_output.close()?;
        rms_norm_weight.close()?;
        rms_norm_input.close()?;
        stream.close()?;
        assert_eq!(context.allocation_stats()?, allocation_baseline);
        context.synchronize()?;
        context.close()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "remote GPU"]
    fn raw_bf16_row_gather_oob_device_index_matches_eager_nan_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        // This is deliberately below the public safe API boundary: safe row
        // gather rejects an OOB host mirror before launch, while the raw C
        // ABI defines an OOB *device* row index as a BF16 NaN output row.
        const INPUT_ROW_COUNT: u64 = 4;
        const OUTPUT_ROW_COUNT: u64 = 3;
        const COLUMN_COUNT: u64 = 5;
        const BF16_BYTES: u64 = 2;
        const U32_BYTES: u64 = 4;

        let runtime = crate::CudaRuntime::initialize()?;
        assert!(
            runtime.device_count() > 0,
            "remote GPU runner has no CUDA device"
        );
        let context = runtime.device(0)?.create_context()?;
        let allocation_baseline = context.allocation_stats()?;
        assert!(allocation_baseline.is_zero());

        let mut eager_stream = context.create_stream()?;
        let mut graph_stream = context.create_stream()?;
        let mut transfer_stream = context.create_stream()?;
        let input_byte_len = INPUT_ROW_COUNT
            .checked_mul(COLUMN_COUNT)
            .and_then(|element_count| element_count.checked_mul(BF16_BYTES))
            .ok_or("raw BF16 row-gather input byte length overflow")?;
        let row_indices_byte_len = OUTPUT_ROW_COUNT
            .checked_mul(U32_BYTES)
            .ok_or("raw BF16 row-gather index byte length overflow")?;
        let output_byte_len = OUTPUT_ROW_COUNT
            .checked_mul(COLUMN_COUNT)
            .and_then(|element_count| element_count.checked_mul(BF16_BYTES))
            .ok_or("raw BF16 row-gather output byte length overflow")?;
        let host_input_bits = [
            0x0000_u16, 0x8000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x3e80, 0xbe80, 0x4080, 0xc080,
            0x3f00, 0xbf00, 0x4040, 0xc040, 0x7fc1, 0xffc1, 0x3f40, 0xbf40, 0x7f80, 0xff80,
        ];
        let host_input: Vec<u8> = host_input_bits
            .iter()
            .flat_map(|&bits| bits.to_ne_bytes())
            .collect();
        // The middle row is deliberately OOB in device memory. There is no
        // host mirror passed to either raw FFI lifecycle below.
        let host_row_indices = [2_u32, INPUT_ROW_COUNT as u32, 0];
        let host_row_index_bytes: Vec<u8> = host_row_indices
            .iter()
            .flat_map(|&index| index.to_ne_bytes())
            .collect();
        assert_eq!(u64::try_from(host_input.len())?, input_byte_len);
        assert_eq!(
            u64::try_from(host_row_index_bytes.len())?,
            row_indices_byte_len
        );
        let output_sentinel = vec![0xa5; usize::try_from(output_byte_len)?];
        let mut staging = context.allocate_pinned_host_buffer(input_byte_len)?;

        let mut eager_input = context.allocate_device_buffer(input_byte_len)?;
        eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        let mut eager_row_indices = context.allocate_device_buffer(row_indices_byte_len)?;
        eager_row_indices.upload_from_slice(
            0,
            &host_row_index_bytes,
            &mut staging,
            &mut eager_stream,
        )?;
        let mut eager_output = context.allocate_device_buffer(output_byte_len)?;
        eager_output.upload_from_slice(0, &output_sentinel, &mut staging, &mut eager_stream)?;

        let graph_input = {
            let mut buffer = context.allocate_device_buffer(input_byte_len)?;
            buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
            buffer
        };
        let graph_row_indices = {
            let mut buffer = context.allocate_device_buffer(row_indices_byte_len)?;
            buffer.upload_from_slice(0, &host_row_index_bytes, &mut staging, &mut eager_stream)?;
            buffer
        };
        let mut graph_output = {
            let mut buffer = context.allocate_device_buffer(output_byte_len)?;
            buffer.upload_from_slice(0, &output_sentinel, &mut staging, &mut eager_stream)?;
            buffer
        };

        // This private FFI call intentionally bypasses `RowGatherParams` and
        // its host-mirror validation to exercise the raw C ABI's OOB rule.
        crate::ffi::row_gather_execute(
            eager_input
                .native_handle()
                .span(crate::ffi::DTYPE_BF16, 0, input_byte_len),
            eager_row_indices
                .native_handle()
                .span(crate::ffi::DTYPE_U32, 0, row_indices_byte_len),
            eager_output
                .native_handle()
                .span(crate::ffi::DTYPE_BF16, 0, output_byte_len),
            INPUT_ROW_COUNT,
            OUTPUT_ROW_COUNT,
            COLUMN_COUNT,
            &mut eager_stream.native,
        )?;

        let allocation_with_resources = context.allocation_stats()?;
        let mut capture = graph_stream.native.begin_graph_bf16_row_gather_capture(
            graph_input.native_handle(),
            graph_row_indices.native_handle(),
            graph_output.native_handle(),
            INPUT_ROW_COUNT,
            OUTPUT_ROW_COUNT,
            COLUMN_COUNT,
            CudaGraphCaptureMode::ThreadLocal as u32,
        )?;
        capture.enqueue_bf16_row_gather()?;
        let transition = capture.end();
        let mut graph = transition.result?;
        let mut exec = graph.instantiate()?;
        let mut launch = exec.launch(&mut graph_stream.native)?;
        launch.complete()?;
        exec.close()?;
        assert_eq!(context.allocation_stats()?, allocation_with_resources);

        let mut eager_output_bytes = vec![0_u8; usize::try_from(output_byte_len)?];
        let mut graph_output_bytes = vec![0_u8; usize::try_from(output_byte_len)?];
        eager_output.download_to_slice(
            0,
            &mut eager_output_bytes,
            &mut staging,
            &mut transfer_stream,
        )?;
        graph_output.download_to_slice(
            0,
            &mut graph_output_bytes,
            &mut staging,
            &mut transfer_stream,
        )?;
        assert_eq!(
            graph_output_bytes, eager_output_bytes,
            "raw graph BF16 row-gather output must match raw eager output byte-for-byte"
        );

        let invalid_row_byte_len = usize::try_from(COLUMN_COUNT * BF16_BYTES)?;
        let invalid_row_start = invalid_row_byte_len;
        let invalid_row_end = invalid_row_start + invalid_row_byte_len;
        let invalid_row = &graph_output_bytes[invalid_row_start..invalid_row_end];
        assert!(
            invalid_row
                .chunks_exact(usize::try_from(BF16_BYTES)?)
                .all(|bytes| {
                    let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
                    (bits & 0x7f80) == 0x7f80 && (bits & 0x007f) != 0
                }),
            "every BF16 element selected by the OOB device index must be NaN"
        );
        assert_eq!(
            invalid_row,
            &eager_output_bytes[invalid_row_start..invalid_row_end],
            "raw graph OOB BF16 NaN row must preserve eager bytes exactly"
        );

        graph_output.close()?;
        graph_row_indices.close()?;
        graph_input.close()?;
        eager_output.close()?;
        eager_row_indices.close()?;
        eager_input.close()?;
        staging.close()?;
        graph_stream.close()?;
        eager_stream.close()?;
        transfer_stream.close()?;
        assert_eq!(context.allocation_stats()?, allocation_baseline);
        context.synchronize()?;
        context.close()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "remote GPU"]
    fn raw_bf16_row_gather_argmax_oob_device_index_matches_eager_nan_result_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        // This deliberately stays below the safe composite-owner boundary.
        // A safe capture rejects the OOB host mirror before native entry, but
        // the raw C contract says the gathered row becomes BF16 NaNs and the
        // following argmax node emits INVALID_TOKEN_ID/NON_FINITE.
        const INPUT_ROW_COUNT: u64 = 4;
        const OUTPUT_ROW_COUNT: u64 = 3;
        const VOCABULARY_SIZE: u64 = 5;
        const BF16_BYTES: u64 = 2;
        const U32_BYTES: u64 = 4;

        let runtime = crate::CudaRuntime::initialize()?;
        assert!(
            runtime.device_count() > 0,
            "remote GPU runner has no CUDA device"
        );
        let context = runtime.device(0)?.create_context()?;
        let allocation_baseline = context.allocation_stats()?;
        assert!(allocation_baseline.is_zero());

        let mut eager_stream = context.create_stream()?;
        let mut graph_stream = context.create_stream()?;
        let mut transfer_stream = context.create_stream()?;
        let input_byte_len = INPUT_ROW_COUNT
            .checked_mul(VOCABULARY_SIZE)
            .and_then(|element_count| element_count.checked_mul(BF16_BYTES))
            .ok_or("raw BF16 row-gather -> argmax input byte length overflow")?;
        let row_indices_byte_len = OUTPUT_ROW_COUNT
            .checked_mul(U32_BYTES)
            .ok_or("raw BF16 row-gather -> argmax index byte length overflow")?;
        let gathered_byte_len = OUTPUT_ROW_COUNT
            .checked_mul(VOCABULARY_SIZE)
            .and_then(|element_count| element_count.checked_mul(BF16_BYTES))
            .ok_or("raw BF16 row-gather -> argmax gathered byte length overflow")?;
        let results_byte_len = OUTPUT_ROW_COUNT
            .checked_mul(std::mem::size_of::<crate::Bf16ArgmaxResult>() as u64)
            .ok_or("raw BF16 row-gather -> argmax result byte length overflow")?;
        let host_input_bits = [
            0x3f80_u16, 0x4000, 0x4040, 0x4080, 0x40a0, // row 0: token 4
            0xbf80, 0xc000, 0xc040, 0xc080, 0xc0a0, // row 1: token 0
            0x0000, 0x3f80, 0x4000, 0x4040, 0x4080, // row 2: token 4
            0x3f00, 0x3f80, 0x4000, 0x4040, 0x4080, // row 3: token 4
        ];
        let host_input: Vec<u8> = host_input_bits
            .iter()
            .flat_map(|&bits| bits.to_ne_bytes())
            .collect();
        // The middle row is deliberately OOB in device memory. No safe host
        // mirror reaches either raw eager or raw graph call below.
        let host_row_indices = [2_u32, INPUT_ROW_COUNT as u32, 0];
        let host_row_index_bytes: Vec<u8> = host_row_indices
            .iter()
            .flat_map(|&index| index.to_ne_bytes())
            .collect();
        assert_eq!(u64::try_from(host_input.len())?, input_byte_len);
        assert_eq!(
            u64::try_from(host_row_index_bytes.len())?,
            row_indices_byte_len
        );
        let gathered_sentinel = vec![0xa5; usize::try_from(gathered_byte_len)?];
        let results_sentinel = vec![0xa5; usize::try_from(results_byte_len)?];
        let mut staging = context.allocate_pinned_host_buffer(input_byte_len)?;

        let mut eager_input = context.allocate_device_buffer(input_byte_len)?;
        eager_input.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
        let mut eager_row_indices = context.allocate_device_buffer(row_indices_byte_len)?;
        eager_row_indices.upload_from_slice(
            0,
            &host_row_index_bytes,
            &mut staging,
            &mut eager_stream,
        )?;
        let mut eager_gathered = context.allocate_device_buffer(gathered_byte_len)?;
        eager_gathered.upload_from_slice(0, &gathered_sentinel, &mut staging, &mut eager_stream)?;
        let mut eager_results = context.allocate_device_buffer(results_byte_len)?;
        eager_results.upload_from_slice(0, &results_sentinel, &mut staging, &mut eager_stream)?;

        let graph_input = {
            let mut buffer = context.allocate_device_buffer(input_byte_len)?;
            buffer.upload_from_slice(0, &host_input, &mut staging, &mut eager_stream)?;
            buffer
        };
        let graph_row_indices = {
            let mut buffer = context.allocate_device_buffer(row_indices_byte_len)?;
            buffer.upload_from_slice(0, &host_row_index_bytes, &mut staging, &mut eager_stream)?;
            buffer
        };
        let mut graph_gathered = {
            let mut buffer = context.allocate_device_buffer(gathered_byte_len)?;
            buffer.upload_from_slice(0, &gathered_sentinel, &mut staging, &mut eager_stream)?;
            buffer
        };
        let mut graph_results = {
            let mut buffer = context.allocate_device_buffer(results_byte_len)?;
            buffer.upload_from_slice(0, &results_sentinel, &mut staging, &mut eager_stream)?;
            buffer
        };

        // These private FFI calls intentionally bypass `RowGatherParams` and
        // `Bf16ArgmaxParams` host-safe descriptors to exercise raw C OOB
        // device-index semantics for the complete two-kernel eager chain.
        crate::ffi::row_gather_execute(
            eager_input
                .native_handle()
                .span(crate::ffi::DTYPE_BF16, 0, input_byte_len),
            eager_row_indices
                .native_handle()
                .span(crate::ffi::DTYPE_U32, 0, row_indices_byte_len),
            eager_gathered
                .native_handle()
                .span(crate::ffi::DTYPE_BF16, 0, gathered_byte_len),
            INPUT_ROW_COUNT,
            OUTPUT_ROW_COUNT,
            VOCABULARY_SIZE,
            &mut eager_stream.native,
        )?;
        crate::ffi::bf16_argmax_execute(
            eager_gathered
                .native_handle()
                .span(crate::ffi::DTYPE_BF16, 0, gathered_byte_len),
            eager_results
                .native_handle()
                .span(crate::ffi::DTYPE_U32, 0, results_byte_len),
            OUTPUT_ROW_COUNT,
            VOCABULARY_SIZE,
            &mut eager_stream.native,
        )?;

        let allocation_with_resources = context.allocation_stats()?;
        let mut capture = graph_stream
            .native
            .begin_graph_bf16_row_gather_argmax_capture(
                graph_input.native_handle(),
                graph_row_indices.native_handle(),
                graph_gathered.native_handle(),
                graph_results.native_handle(),
                INPUT_ROW_COUNT,
                OUTPUT_ROW_COUNT,
                VOCABULARY_SIZE,
                CudaGraphCaptureMode::ThreadLocal as u32,
            )?;
        capture.enqueue_bf16_row_gather_argmax()?;
        let transition = capture.end();
        let mut graph = transition.result?;
        let mut exec = graph.instantiate()?;
        let mut launch = exec.launch(&mut graph_stream.native)?;
        launch.complete()?;
        exec.close()?;
        assert_eq!(context.allocation_stats()?, allocation_with_resources);

        let mut eager_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
        let mut graph_gathered_bytes = vec![0_u8; usize::try_from(gathered_byte_len)?];
        let mut eager_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
        let mut graph_result_bytes = vec![0_u8; usize::try_from(results_byte_len)?];
        eager_gathered.download_to_slice(
            0,
            &mut eager_gathered_bytes,
            &mut staging,
            &mut transfer_stream,
        )?;
        graph_gathered.download_to_slice(
            0,
            &mut graph_gathered_bytes,
            &mut staging,
            &mut transfer_stream,
        )?;
        eager_results.download_to_slice(
            0,
            &mut eager_result_bytes,
            &mut staging,
            &mut transfer_stream,
        )?;
        graph_results.download_to_slice(
            0,
            &mut graph_result_bytes,
            &mut staging,
            &mut transfer_stream,
        )?;
        assert_eq!(
            graph_gathered_bytes, eager_gathered_bytes,
            "raw graph gathered BF16 bytes must match the raw eager chain byte-for-byte"
        );
        assert_eq!(
            graph_result_bytes, eager_result_bytes,
            "raw graph token/status bytes must match the raw eager chain byte-for-byte"
        );

        let invalid_row_byte_len = usize::try_from(VOCABULARY_SIZE * BF16_BYTES)?;
        let invalid_row_start = invalid_row_byte_len;
        let invalid_row_end = invalid_row_start + invalid_row_byte_len;
        let invalid_row = &graph_gathered_bytes[invalid_row_start..invalid_row_end];
        assert!(
            invalid_row
                .chunks_exact(usize::try_from(BF16_BYTES)?)
                .all(|bytes| {
                    let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
                    (bits & 0x7f80) == 0x7f80 && (bits & 0x007f) != 0
                }),
            "every BF16 element selected by the OOB device index must be NaN"
        );
        assert_eq!(
            invalid_row,
            &eager_gathered_bytes[invalid_row_start..invalid_row_end],
            "raw graph OOB BF16 NaN row must preserve raw eager bytes exactly"
        );

        let invalid_result_start = std::mem::size_of::<crate::Bf16ArgmaxResult>();
        let invalid_result_end =
            invalid_result_start + std::mem::size_of::<crate::Bf16ArgmaxResult>();
        let invalid_result = &graph_result_bytes[invalid_result_start..invalid_result_end];
        let invalid_token_id = u32::from_ne_bytes(
            invalid_result[..std::mem::size_of::<u32>()]
                .try_into()
                .expect("one result token field must occupy four bytes"),
        );
        let invalid_status = u32::from_ne_bytes(
            invalid_result[std::mem::size_of::<u32>()..]
                .try_into()
                .expect("one result status field must occupy four bytes"),
        );
        assert_eq!(invalid_token_id, crate::BF16_ARGMAX_INVALID_TOKEN_ID);
        assert_eq!(invalid_status, crate::BF16_ARGMAX_STATUS_NON_FINITE);
        assert_eq!(
            invalid_result,
            &eager_result_bytes[invalid_result_start..invalid_result_end],
            "raw graph OOB argmax record must preserve raw eager bytes exactly"
        );

        graph_results.close()?;
        graph_gathered.close()?;
        graph_row_indices.close()?;
        graph_input.close()?;
        eager_results.close()?;
        eager_gathered.close()?;
        eager_row_indices.close()?;
        eager_input.close()?;
        staging.close()?;
        graph_stream.close()?;
        eager_stream.close()?;
        transfer_stream.close()?;
        assert_eq!(context.allocation_stats()?, allocation_baseline);
        context.synchronize()?;
        context.close()?;
        Ok(())
    }
}
