use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{CudaError, CudaResult};
use crate::runtime::{
    ContextInner, CudaCommandStream, CudaContext, CudaStream, ensure_same_context,
    execution_stream_mut,
};

#[cfg(feature = "cuda")]
use crate::ffi;
#[cfg(feature = "cuda")]
use crate::{CudaErrorDomain, CudaErrorKind, CudaErrorStage};

/// Coherent snapshot of allocations owned by one [`CudaContext`].
///
/// Zero-byte logical buffers contribute one allocation and zero bytes. A
/// destructive CUDA free that returns an ambiguous error remains accounted as
/// live, preventing a false all-clear and making context teardown fail closed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaAllocationStats {
    device_live_bytes: u64,
    device_live_allocations: u64,
    pinned_host_live_bytes: u64,
    pinned_host_live_allocations: u64,
}

/// One-shot native memory-lifecycle fault identifier.
///
/// This API exists only with `cuda-test-fault-injection`. It deliberately
/// creates fail-closed leaks/poisoned contexts and is unsupported in production.
#[cfg(feature = "cuda-test-fault-injection")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CudaMemoryFault {
    DeviceCreateRollbackAmbiguous = 1,
    PinnedCreateRollbackAmbiguous = 2,
    DeviceCloseAmbiguous = 3,
    PinnedCloseAmbiguous = 4,
    CopyDeferredSubmissionError = 5,
    CopyCompletionRestoreAmbiguous = 6,
    /// C05-22 only: fail the dependent cuBLASLt capture submission after its
    /// RMSNorm node was accepted. The capture remains abort-only.
    C05_22GemmSubmissionNotSupported = 7,
}

#[cfg(feature = "cuda-test-fault-injection")]
impl CudaMemoryFault {
    const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            1 => Some(Self::DeviceCreateRollbackAmbiguous),
            2 => Some(Self::PinnedCreateRollbackAmbiguous),
            3 => Some(Self::DeviceCloseAmbiguous),
            4 => Some(Self::PinnedCloseAmbiguous),
            5 => Some(Self::CopyDeferredSubmissionError),
            6 => Some(Self::CopyCompletionRestoreAmbiguous),
            7 => Some(Self::C05_22GemmSubmissionNotSupported),
            _ => None,
        }
    }
}

/// Process-local diagnostic counters for the test-only fault injector.
#[cfg(feature = "cuda-test-fault-injection")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaMemoryFaultStats {
    armed_fault: Option<CudaMemoryFault>,
    faults_fired: u64,
    device_free_attempts: u64,
    pinned_free_attempts: u64,
    copy_use_release_attempts: u64,
}

#[cfg(feature = "cuda-test-fault-injection")]
impl CudaMemoryFaultStats {
    /// Currently armed one-shot fault, if any.
    #[must_use]
    pub const fn armed_fault(self) -> Option<CudaMemoryFault> {
        self.armed_fault
    }

    /// Number of one-shot faults consumed since reset.
    #[must_use]
    pub const fn faults_fired(self) -> u64 {
        self.faults_fired
    }

    /// Number of `cudaFree` attempts observed by this session.
    #[must_use]
    pub const fn device_free_attempts(self) -> u64 {
        self.device_free_attempts
    }

    /// Number of `cudaFreeHost` attempts observed by this session.
    #[must_use]
    pub const fn pinned_free_attempts(self) -> u64 {
        self.pinned_free_attempts
    }

    /// Number of all-or-none native copy reservation releases attempted.
    #[must_use]
    pub const fn copy_use_release_attempts(self) -> u64 {
        self.copy_use_release_attempts
    }
}

impl CudaAllocationStats {
    /// Bytes currently accounted to opaque device buffers.
    #[must_use]
    pub const fn device_live_bytes(self) -> u64 {
        self.device_live_bytes
    }

    /// Number of live opaque device-buffer handles.
    #[must_use]
    pub const fn device_live_allocations(self) -> u64 {
        self.device_live_allocations
    }

    /// Bytes currently accounted to CUDA-pinned host buffers.
    #[must_use]
    pub const fn pinned_host_live_bytes(self) -> u64 {
        self.pinned_host_live_bytes
    }

    /// Number of live CUDA-pinned host-buffer handles.
    #[must_use]
    pub const fn pinned_host_live_allocations(self) -> u64 {
        self.pinned_host_live_allocations
    }

    /// Whether all tracked allocation counts and byte totals are zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.device_live_bytes == 0
            && self.device_live_allocations == 0
            && self.pinned_host_live_bytes == 0
            && self.pinned_host_live_allocations == 0
    }
}

impl CudaContext {
    /// Starts a fresh process-local test fault session for this context.
    ///
    /// # Errors
    ///
    /// Returns a native validation error. Call only in a disposable subprocess.
    #[cfg(feature = "cuda-test-fault-injection")]
    pub fn reset_memory_fault_injection(&self) -> CudaResult<()> {
        self.inner.native.reset_memory_fault_injection()
    }

    /// Arms exactly one destructive test fault. The fault is consumed once.
    ///
    /// # Errors
    ///
    /// Returns invalid-state if the session was not reset for this context or a
    /// previous fault remains armed.
    #[cfg(feature = "cuda-test-fault-injection")]
    pub fn arm_memory_fault(&self, fault: CudaMemoryFault) -> CudaResult<()> {
        self.inner.native.arm_memory_fault(fault as u32)
    }

    /// Returns the current test-only injector counters.
    ///
    /// # Errors
    ///
    /// Returns a native validation error if no session belongs to this context.
    #[cfg(feature = "cuda-test-fault-injection")]
    pub fn memory_fault_stats(&self) -> CudaResult<CudaMemoryFaultStats> {
        let stats = self.inner.native.memory_fault_stats()?;
        Ok(CudaMemoryFaultStats {
            armed_fault: CudaMemoryFault::from_raw(stats.armed_fault),
            faults_fired: stats.faults_fired,
            device_free_attempts: stats.device_free_attempts,
            pinned_free_attempts: stats.pinned_free_attempts,
            copy_use_release_attempts: stats.copy_use_release_attempts,
        })
    }

    /// Allocates an opaque byte-addressed device buffer.
    ///
    /// A zero-byte request returns an owned logical handle and is included in
    /// the allocation count, but performs no `cudaMalloc` call.
    ///
    /// # Errors
    ///
    /// Returns a size, allocation, native context, or accounting error.
    pub fn allocate_device_buffer(&self, byte_len: u64) -> CudaResult<CudaDeviceBuffer> {
        #[cfg(feature = "cuda")]
        {
            let native = ffi::DeviceBufferHandle::create(&self.inner.native, byte_len)?;
            Ok(CudaDeviceBuffer {
                native,
                context: Arc::clone(&self.inner),
                byte_len,
                use_state: BufferUseState::new(),
                _not_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = byte_len;
            Err(CudaError::unavailable(
                "CudaContext::allocate_device_buffer",
            ))
        }
    }

    /// Allocates an opaque CUDA-pinned host byte buffer.
    ///
    /// Pinned storage is optional at the application level: callers choose it
    /// explicitly when asynchronous host/device copies are required. A
    /// zero-byte request still returns one accounted logical handle.
    ///
    /// # Errors
    ///
    /// Returns a size, allocation, native context, or accounting error.
    pub fn allocate_pinned_host_buffer(&self, byte_len: u64) -> CudaResult<CudaPinnedHostBuffer> {
        #[cfg(feature = "cuda")]
        {
            let native = ffi::PinnedHostBufferHandle::create(&self.inner.native, byte_len)?;
            Ok(CudaPinnedHostBuffer {
                native,
                context: Arc::clone(&self.inner),
                byte_len,
                use_state: BufferUseState::new(),
                _not_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = byte_len;
            Err(CudaError::unavailable(
                "CudaContext::allocate_pinned_host_buffer",
            ))
        }
    }

    /// Returns a coherent logical allocation-accounting snapshot.
    ///
    /// # Errors
    ///
    /// Returns a translated native ABI error.
    pub fn allocation_stats(&self) -> CudaResult<CudaAllocationStats> {
        #[cfg(feature = "cuda")]
        {
            let stats = self.inner.native.allocation_stats()?;
            Ok(CudaAllocationStats {
                device_live_bytes: stats.device_live_bytes,
                device_live_allocations: stats.device_live_allocations,
                pinned_host_live_bytes: stats.pinned_host_live_bytes,
                pinned_host_live_allocations: stats.pinned_host_live_allocations,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaContext::allocation_stats"))
        }
    }
}

struct BufferUseState {
    active: AtomicBool,
}

impl BufferUseState {
    #[cfg(any(feature = "cuda", test))]
    const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
        }
    }

    fn begin(&self, operation: &'static str, resource: &'static str) -> CudaResult<()> {
        self.active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| {
                CudaError::invalid_state(
                    operation,
                    format!("{resource} already has an active or forgotten copy token"),
                )
            })
    }

    fn finish(&self) {
        self.active.store(false, Ordering::Release);
    }

    fn ensure_idle(&self, operation: &'static str, resource: &'static str) -> CudaResult<()> {
        if self.active.load(Ordering::Acquire) {
            Err(CudaError::invalid_state(
                operation,
                format!("{resource} still has an active or forgotten copy token"),
            ))
        } else {
            Ok(())
        }
    }
}

/// Opaque byte-addressed CUDA device allocation.
///
/// No device pointer is exposed. The buffer is `Send` but deliberately
/// `!Sync`; one mutable borrow and native active-use guards serialize copies.
/// A forgotten pending copy leaves the buffer permanently busy and safely
/// leaked rather than allowing early free or reuse.
///
/// A live token's exclusive borrows prevent ordinary safe-code reuse:
///
/// ```compile_fail
/// fn cannot_reuse_while_pending(
///     mut device: riley_cuda::CudaDeviceBuffer,
///     mut host: riley_cuda::CudaPinnedHostBuffer,
///     mut stream: riley_cuda::CudaStream,
/// ) {
///     let pending = device
///         .copy_from_pinned_async(0, &mut host, 0, 1, &mut stream)
///         .unwrap();
///     host.write(0, &[1]).unwrap();
///     stream.close().unwrap();
///     drop(pending);
/// }
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::CudaDeviceBuffer>();
/// ```
///
/// ```compile_fail
/// fn cannot_clone(buffer: riley_cuda::CudaDeviceBuffer) {
///     let duplicate = buffer.clone();
///     drop((buffer, duplicate));
/// }
/// ```
pub struct CudaDeviceBuffer {
    #[cfg(feature = "cuda")]
    native: ffi::DeviceBufferHandle,
    // Native must close before releasing the context Arc.
    context: Arc<ContextInner>,
    byte_len: u64,
    use_state: BufferUseState,
    _not_sync: PhantomData<Cell<()>>,
}

impl CudaDeviceBuffer {
    /// Logical buffer length in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn context_owner(&self) -> &Arc<ContextInner> {
        &self.context
    }

    pub(crate) fn ensure_idle_for_operation(&self, operation: &'static str) -> CudaResult<()> {
        self.use_state.ensure_idle(operation, "device buffer")
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn native_handle(&self) -> &ffi::DeviceBufferHandle {
        &self.native
    }

    /// Uploads ordinary host bytes through a caller-owned reusable pinned
    /// staging buffer and waits for every chunk before reusing that staging
    /// storage.
    ///
    /// This helper is intended for model initialization. It performs no device
    /// or pinned allocation; a non-empty source requires a non-empty staging
    /// buffer. Validation happens before the first partial upload.
    ///
    /// # Errors
    ///
    /// Returns a range, context, staging-capacity, copy, or synchronization
    /// error. A failed completion retains the existing fail-closed copy guards.
    pub fn upload_from_slice(
        &mut self,
        destination_offset: u64,
        source: &[u8],
        staging: &mut CudaPinnedHostBuffer,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "CudaDeviceBuffer::upload_from_slice";
        let source_len = u64::try_from(source.len()).map_err(|_| {
            CudaError::out_of_range(OPERATION, "source length does not fit the CUDA ABI")
        })?;
        ensure_same_context(&self.context, &staging.context, OPERATION)?;
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        validate_range(self.byte_len, destination_offset, source_len, OPERATION)?;
        self.use_state.ensure_idle(OPERATION, "device buffer")?;
        staging
            .use_state
            .ensure_idle(OPERATION, "pinned host buffer")?;

        if source.is_empty() {
            return self
                .copy_from_pinned_async(destination_offset, staging, 0, 0, stream)?
                .synchronize();
        }
        if staging.byte_len == 0 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "a non-empty upload requires a non-empty pinned staging buffer",
            ));
        }
        let staging_capacity = usize::try_from(staging.byte_len).unwrap_or(usize::MAX);
        let mut source_offset = 0_usize;
        while source_offset < source.len() {
            let chunk_len = staging_capacity.min(source.len() - source_offset);
            let chunk = &source[source_offset..source_offset + chunk_len];
            staging.write(0, chunk)?;
            let uploaded = u64::try_from(source_offset).map_err(|_| {
                CudaError::out_of_range(OPERATION, "uploaded byte count does not fit the CUDA ABI")
            })?;
            let chunk_len = u64::try_from(chunk_len).map_err(|_| {
                CudaError::out_of_range(OPERATION, "chunk length does not fit the CUDA ABI")
            })?;
            let chunk_destination = destination_offset
                .checked_add(uploaded)
                .ok_or_else(|| CudaError::out_of_range(OPERATION, "destination offset overflow"))?;
            self.copy_from_pinned_async(chunk_destination, staging, 0, chunk_len, stream)?
                .synchronize()?;
            source_offset += usize::try_from(chunk_len).map_err(|_| {
                CudaError::out_of_range(OPERATION, "completed chunk does not fit host usize")
            })?;
        }
        Ok(())
    }

    /// Downloads device bytes into ordinary host storage through a reusable
    /// caller-owned pinned staging buffer.
    ///
    /// The full source range, context ownership, and staging capacity are
    /// validated before the first partial copy. Each chunk is synchronized
    /// before the pinned bytes are read or reused. The successful path performs
    /// no host or device allocation.
    ///
    /// # Errors
    ///
    /// Returns a range, context, staging-capacity, copy, synchronization, or
    /// pinned-host read error. A failed completion retains the existing
    /// fail-closed copy guards.
    pub fn download_to_slice(
        &mut self,
        source_offset: u64,
        destination: &mut [u8],
        staging: &mut CudaPinnedHostBuffer,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "CudaDeviceBuffer::download_to_slice";
        let destination_len = u64::try_from(destination.len()).map_err(|_| {
            CudaError::out_of_range(OPERATION, "destination length does not fit the CUDA ABI")
        })?;
        ensure_same_context(&self.context, &staging.context, OPERATION)?;
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        validate_range(self.byte_len, source_offset, destination_len, OPERATION)?;
        self.use_state.ensure_idle(OPERATION, "device buffer")?;
        staging
            .use_state
            .ensure_idle(OPERATION, "pinned host buffer")?;

        if destination.is_empty() {
            return self
                .copy_to_pinned_async(source_offset, staging, 0, 0, stream)?
                .synchronize();
        }
        if staging.byte_len == 0 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "a non-empty download requires a non-empty pinned staging buffer",
            ));
        }

        let staging_capacity = usize::try_from(staging.byte_len).unwrap_or(usize::MAX);
        let mut downloaded = 0_usize;
        while downloaded < destination.len() {
            let chunk_len = staging_capacity.min(destination.len() - downloaded);
            let downloaded_u64 = u64::try_from(downloaded).map_err(|_| {
                CudaError::out_of_range(
                    OPERATION,
                    "downloaded byte count does not fit the CUDA ABI",
                )
            })?;
            let chunk_len_u64 = u64::try_from(chunk_len).map_err(|_| {
                CudaError::out_of_range(OPERATION, "chunk length does not fit the CUDA ABI")
            })?;
            let chunk_source = source_offset
                .checked_add(downloaded_u64)
                .ok_or_else(|| CudaError::out_of_range(OPERATION, "source offset overflow"))?;
            self.copy_to_pinned_async(chunk_source, staging, 0, chunk_len_u64, stream)?
                .synchronize()?;
            staging.read(0, &mut destination[downloaded..downloaded + chunk_len])?;
            downloaded += chunk_len;
        }
        Ok(())
    }

    /// Enqueues an asynchronous pinned-host-to-device copy in a command batch.
    ///
    /// Unlike [`Self::copy_from_pinned_async`], this allocation-free path is
    /// available only through an active [`CudaCommandStream`]. It creates no
    /// standalone completion token and performs no per-copy synchronization;
    /// [`crate::CudaCommandBatch::finish`] is the completion boundary. Native
    /// batch leases reject CPU access, out-of-batch reuse, or close of either
    /// buffer until that finish confirms completion. A failed or ambiguous
    /// finish keeps the leases live and therefore fails closed.
    ///
    /// The copy direction is fixed to pinned-host-to-device by the typed API.
    /// All handle ownership and byte ranges are validated before submission.
    /// A zero-byte in-range copy is a successful no-op.
    ///
    /// ```compile_fail
    /// fn cannot_drop_copy_resources_before_batch_finish(
    ///     device: riley_cuda::CudaDeviceBuffer,
    ///     source: riley_cuda::CudaPinnedHostBuffer,
    ///     mut stream: riley_cuda::CudaStream,
    /// ) {
    ///     let mut batch = stream.begin_command_batch().unwrap();
    ///     {
    ///         let mut commands = batch.commands();
    ///         device
    ///             .copy_from_pinned_in_command_batch(0, &source, 0, 1, &mut commands)
    ///             .unwrap();
    ///     }
    ///     drop(source);
    ///     batch.finish().unwrap();
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a range, context-ownership, busy-state, command-batch-lifecycle,
    /// or native copy error.
    pub fn copy_from_pinned_in_command_batch<'stream>(
        &'stream self,
        destination_offset: u64,
        source: &'stream CudaPinnedHostBuffer,
        source_offset: u64,
        byte_len: u64,
        commands: &mut CudaCommandStream<'_, 'stream>,
    ) -> CudaResult<()> {
        const OPERATION: &str = "CudaDeviceBuffer::copy_from_pinned_in_command_batch";
        let stream = execution_stream_mut(commands);
        prepare_copy(
            &self.context,
            self.byte_len,
            destination_offset,
            &source.context,
            source.byte_len,
            source_offset,
            byte_len,
            &stream.context,
            OPERATION,
        )?;
        self.use_state.ensure_idle(OPERATION, "device buffer")?;
        source
            .use_state
            .ensure_idle(OPERATION, "pinned host buffer")?;

        #[cfg(feature = "cuda")]
        {
            self.native.copy_from_pinned_in_command_batch(
                destination_offset,
                &source.native,
                source_offset,
                byte_len,
                &mut stream.native,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (destination_offset, source_offset, byte_len, stream);
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Enqueues an asynchronous pinned-host-to-device copy.
    ///
    /// The returned pending token exclusively borrows the originating stream
    /// and both buffers. Native active-use state remains authoritative even if
    /// the token is forgotten, so later CPU access, copies, and closes fail
    /// closed rather than racing DMA.
    ///
    /// # Errors
    ///
    /// Returns a range, context-ownership, busy-state, or native copy error.
    pub fn copy_from_pinned_async<'copy>(
        &'copy mut self,
        destination_offset: u64,
        source: &'copy mut CudaPinnedHostBuffer,
        source_offset: u64,
        byte_len: u64,
        stream: &'copy mut CudaStream,
    ) -> CudaResult<CudaPendingH2D<'copy>> {
        const OPERATION: &str = "CudaDeviceBuffer::copy_from_pinned_async";
        prepare_copy(
            &self.context,
            self.byte_len,
            destination_offset,
            &source.context,
            source.byte_len,
            source_offset,
            byte_len,
            &stream.context,
            OPERATION,
        )?;
        reserve_buffers(&self.use_state, &source.use_state, OPERATION)?;

        #[cfg(feature = "cuda")]
        {
            let native = match ffi::CopyHandle::h2d(
                &self.native,
                destination_offset,
                &source.native,
                source_offset,
                byte_len,
                &stream.native,
            ) {
                Ok(native) => native,
                Err(error) => {
                    release_buffers(&self.use_state, &source.use_state);
                    return Err(error);
                }
            };
            if byte_len == 0 {
                release_buffers(&self.use_state, &source.use_state);
            }
            Ok(CudaPendingH2D {
                inner: PendingCopy {
                    native,
                    _stream: stream,
                    device: self,
                    host: source,
                    released: byte_len == 0,
                    abandoned: false,
                },
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            release_buffers(&self.use_state, &source.use_state);
            let _ = (destination_offset, source_offset, byte_len, stream);
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Enqueues an asynchronous device-to-pinned-host copy.
    ///
    /// # Errors
    ///
    /// Returns a range, context-ownership, busy-state, or native copy error.
    pub fn copy_to_pinned_async<'copy>(
        &'copy mut self,
        source_offset: u64,
        destination: &'copy mut CudaPinnedHostBuffer,
        destination_offset: u64,
        byte_len: u64,
        stream: &'copy mut CudaStream,
    ) -> CudaResult<CudaPendingD2H<'copy>> {
        const OPERATION: &str = "CudaDeviceBuffer::copy_to_pinned_async";
        prepare_copy(
            &self.context,
            self.byte_len,
            source_offset,
            &destination.context,
            destination.byte_len,
            destination_offset,
            byte_len,
            &stream.context,
            OPERATION,
        )?;
        reserve_buffers(&self.use_state, &destination.use_state, OPERATION)?;

        #[cfg(feature = "cuda")]
        {
            let native = match ffi::CopyHandle::d2h(
                &destination.native,
                destination_offset,
                &self.native,
                source_offset,
                byte_len,
                &stream.native,
            ) {
                Ok(native) => native,
                Err(error) => {
                    release_buffers(&self.use_state, &destination.use_state);
                    return Err(error);
                }
            };
            if byte_len == 0 {
                release_buffers(&self.use_state, &destination.use_state);
            }
            Ok(CudaPendingD2H {
                inner: PendingCopy {
                    native,
                    _stream: stream,
                    device: self,
                    host: destination,
                    released: byte_len == 0,
                    abandoned: false,
                },
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            release_buffers(&self.use_state, &destination.use_state);
            let _ = (source_offset, destination_offset, byte_len, stream);
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Explicitly frees this allocation.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for an active/forgotten copy or a translated
    /// single-shot native close error.
    pub fn close(self) -> CudaResult<()> {
        self.use_state
            .ensure_idle("CudaDeviceBuffer::close", "device buffer")?;
        #[cfg(feature = "cuda")]
        {
            let mut this = self;
            this.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = self;
            Err(CudaError::unavailable("CudaDeviceBuffer::close"))
        }
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            // The native handle's Drop runs after this wrapper hook. When a
            // thread-local graph capture is live it may hand the raw close to
            // the native capture owner; retain this Arc until abort drains it.
            let _ = crate::graph::retain_context_for_active_graph_capture(&self.context);
        }
    }
}

/// Opaque CUDA-pinned host allocation used for explicit asynchronous copies.
///
/// CPU reads and writes are synchronous and are rejected while any copy token
/// is active or forgotten. The type is `Send` and deliberately `!Sync`.
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::CudaPinnedHostBuffer>();
/// ```
///
/// ```compile_fail
/// fn cannot_clone(buffer: riley_cuda::CudaPinnedHostBuffer) {
///     let duplicate = buffer.clone();
///     drop((buffer, duplicate));
/// }
/// ```
pub struct CudaPinnedHostBuffer {
    #[cfg(feature = "cuda")]
    native: ffi::PinnedHostBufferHandle,
    context: Arc<ContextInner>,
    byte_len: u64,
    use_state: BufferUseState,
    _not_sync: PhantomData<Cell<()>>,
}

impl CudaPinnedHostBuffer {
    /// Logical buffer length in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn context_owner(&self) -> &Arc<ContextInner> {
        &self.context
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn ensure_idle_for_operation(&self, operation: &'static str) -> CudaResult<()> {
        self.use_state.ensure_idle(operation, "pinned host buffer")
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn native_handle(&self) -> &ffi::PinnedHostBufferHandle {
        &self.native
    }

    /// Copies ordinary host bytes into pinned storage synchronously.
    ///
    /// # Errors
    ///
    /// Returns a range, busy-state, or native validation error.
    pub fn write(&mut self, destination_offset: u64, source: &[u8]) -> CudaResult<()> {
        const OPERATION: &str = "CudaPinnedHostBuffer::write";
        let source_len = u64::try_from(source.len())
            .map_err(|_| CudaError::out_of_range(OPERATION, "source length does not fit u64"))?;
        validate_range(self.byte_len, destination_offset, source_len, OPERATION)?;
        self.use_state
            .ensure_idle(OPERATION, "pinned host buffer")?;
        #[cfg(feature = "cuda")]
        {
            self.native.write(destination_offset, source)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = source;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Copies pinned bytes into ordinary host storage synchronously.
    ///
    /// # Errors
    ///
    /// Returns a range, busy-state, or native validation error.
    pub fn read(&mut self, source_offset: u64, destination: &mut [u8]) -> CudaResult<()> {
        const OPERATION: &str = "CudaPinnedHostBuffer::read";
        let destination_len = u64::try_from(destination.len()).map_err(|_| {
            CudaError::out_of_range(OPERATION, "destination length does not fit u64")
        })?;
        validate_range(self.byte_len, source_offset, destination_len, OPERATION)?;
        self.use_state
            .ensure_idle(OPERATION, "pinned host buffer")?;
        #[cfg(feature = "cuda")]
        {
            self.native.read(source_offset, destination)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = destination;
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Copies this entire pinned allocation into an owned ordinary-host vector.
    ///
    /// # Errors
    ///
    /// Returns a busy-state, host allocation, size-conversion, or native read
    /// error.
    pub fn to_vec(&mut self) -> CudaResult<Vec<u8>> {
        const OPERATION: &str = "CudaPinnedHostBuffer::to_vec";
        self.use_state
            .ensure_idle(OPERATION, "pinned host buffer")?;
        let len = usize::try_from(self.byte_len).map_err(|_| {
            CudaError::out_of_range(OPERATION, "buffer length does not fit host usize")
        })?;
        let mut output = Vec::new();
        output.try_reserve_exact(len).map_err(|error| {
            CudaError::host_allocation(
                OPERATION,
                format!("could not reserve pinned-buffer output: {error}"),
            )
        })?;
        output.resize(len, 0);
        self.read(0, &mut output)?;
        Ok(output)
    }

    /// Explicitly frees this pinned allocation.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for an active/forgotten copy or a translated
    /// single-shot native close error.
    pub fn close(self) -> CudaResult<()> {
        self.use_state
            .ensure_idle("CudaPinnedHostBuffer::close", "pinned host buffer")?;
        #[cfg(feature = "cuda")]
        {
            let mut this = self;
            this.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = self;
            Err(CudaError::unavailable("CudaPinnedHostBuffer::close"))
        }
    }
}

impl Drop for CudaPinnedHostBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let _ = crate::graph::retain_context_for_active_graph_capture(&self.context);
        }
    }
}

fn validate_range(
    total: u64,
    offset: u64,
    byte_len: u64,
    operation: &'static str,
) -> CudaResult<()> {
    if offset <= total && byte_len <= total - offset {
        Ok(())
    } else {
        Err(CudaError::out_of_range(
            operation,
            format!("range offset={offset} byte_len={byte_len} exceeds buffer length {total}"),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_copy(
    device_context: &Arc<ContextInner>,
    device_len: u64,
    device_offset: u64,
    host_context: &Arc<ContextInner>,
    host_len: u64,
    host_offset: u64,
    byte_len: u64,
    stream_context: &Arc<ContextInner>,
    operation: &'static str,
) -> CudaResult<()> {
    ensure_same_context(device_context, host_context, operation)?;
    ensure_same_context(device_context, stream_context, operation)?;
    validate_range(device_len, device_offset, byte_len, operation)?;
    validate_range(host_len, host_offset, byte_len, operation)
}

fn reserve_buffers(
    device: &BufferUseState,
    host: &BufferUseState,
    operation: &'static str,
) -> CudaResult<()> {
    device.begin(operation, "device buffer")?;
    if let Err(error) = host.begin(operation, "pinned host buffer") {
        device.finish();
        return Err(error);
    }
    Ok(())
}

fn release_buffers(device: &BufferUseState, host: &BufferUseState) {
    host.finish();
    device.finish();
}

struct PendingCopy<'copy> {
    #[cfg(feature = "cuda")]
    native: Option<ffi::CopyHandle>,
    _stream: &'copy mut CudaStream,
    #[cfg(feature = "cuda")]
    device: &'copy mut CudaDeviceBuffer,
    #[cfg(feature = "cuda")]
    host: &'copy mut CudaPinnedHostBuffer,
    released: bool,
    abandoned: bool,
}

impl PendingCopy<'_> {
    fn query(&mut self) -> CudaResult<bool> {
        if self.released {
            return Ok(true);
        }
        #[cfg(feature = "cuda")]
        {
            let Some(native) = self.native.as_mut() else {
                return Err(missing_copy_token(
                    "CudaPendingCopy::query",
                    CudaErrorStage::Query,
                ));
            };
            let outcome = native.query();
            self.apply_completion(outcome)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaPendingCopy::query"))
        }
    }

    fn synchronize(&mut self) -> CudaResult<()> {
        if self.released {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            let Some(native) = self.native.as_mut() else {
                return Err(missing_copy_token(
                    "CudaPendingCopy::synchronize",
                    CudaErrorStage::Synchronize,
                ));
            };
            let outcome = native.synchronize();
            match self.apply_completion(outcome) {
                Ok(true) => Ok(()),
                Ok(false) => Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Synchronize,
                    0,
                    "CudaPendingCopy::synchronize",
                    "native synchronize returned success without confirmed completion",
                )),
                Err(error) => Err(error),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaPendingCopy::synchronize"))
        }
    }

    #[cfg(feature = "cuda")]
    fn apply_completion(&mut self, outcome: ffi::CopyCompletion) -> CudaResult<bool> {
        if !outcome.complete {
            return outcome.result.map(|()| false);
        }

        let Some(native) = self.native.as_mut() else {
            return Err(missing_copy_token(
                "CudaPendingCopy::apply_completion",
                CudaErrorStage::Synchronize,
            ));
        };
        let close_result = native.close();
        release_buffers(&self.device.use_state, &self.host.use_state);
        self.released = true;
        self.native = None;
        match outcome.result {
            Ok(()) => close_result.map(|()| true),
            Err(error) => {
                let _ = close_result;
                Err(error)
            }
        }
    }

    fn synchronize_consuming(&mut self) -> CudaResult<()> {
        let result = self.synchronize();
        if result.is_err() && !self.released {
            self.abandon_unconfirmed();
        }
        result
    }

    fn abandon_unconfirmed(&mut self) {
        if self.released || self.abandoned {
            return;
        }
        #[cfg(feature = "cuda")]
        if let Some(native) = self.native.take() {
            // Native still owns active-use references to all borrowed resources.
            // Leaking this token is the fail-closed path: its counters prevent
            // access/free/reuse after the Rust borrows end.
            std::mem::forget(native);
        }
        self.abandoned = true;
    }
}

#[cfg(feature = "cuda")]
fn missing_copy_token(operation: &'static str, stage: CudaErrorStage) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        stage,
        0,
        operation,
        "pending non-zero copy lost its native lifetime token",
    )
}

impl Drop for PendingCopy<'_> {
    fn drop(&mut self) {
        if !self.released && !self.abandoned {
            let _ = self.synchronize();
        }
        if !self.released {
            self.abandon_unconfirmed();
        }
    }
}

/// Pending pinned-host-to-device transfer.
///
/// This value borrows its originating stream and both buffers exclusively.
/// `synchronize` is the explicit error-reporting completion path. Forgetting
/// the value intentionally leaves both Rust and native resources permanently
/// busy/accounted, which is a safe leak rather than a lifetime violation.
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::CudaPendingH2D<'static>>();
/// ```
pub struct CudaPendingH2D<'copy> {
    inner: PendingCopy<'copy>,
}

impl CudaPendingH2D<'_> {
    /// Non-blocking completion query; not-ready maps to `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns a copy/query/context-restoration error. Buffers remain busy
    /// unless native completion and context restoration are both confirmed.
    pub fn query(&mut self) -> CudaResult<bool> {
        self.inner.query()
    }

    /// Waits for completion and releases both buffer-use guards.
    ///
    /// # Errors
    ///
    /// Returns a submission, synchronization, or context-restoration error.
    pub fn synchronize(mut self) -> CudaResult<()> {
        self.inner.synchronize_consuming()
    }
}

/// Pending device-to-pinned-host transfer.
///
/// Pinned bytes become readable only after confirmed completion. The token has
/// the same forget-safe, fail-closed lifetime semantics as [`CudaPendingH2D`].
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<riley_cuda::CudaPendingD2H<'static>>();
/// ```
pub struct CudaPendingD2H<'copy> {
    inner: PendingCopy<'copy>,
}

impl CudaPendingD2H<'_> {
    /// Non-blocking completion query; not-ready maps to `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns a copy/query/context-restoration error.
    pub fn query(&mut self) -> CudaResult<bool> {
        self.inner.query()
    }

    /// Waits for completion, making pinned destination bytes CPU-accessible.
    ///
    /// # Errors
    ///
    /// Returns a submission, synchronization, or context-restoration error.
    pub fn synchronize(mut self) -> CudaResult<()> {
        self.inner.synchronize_consuming()
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferUseState, CudaAllocationStats, validate_range};

    #[test]
    fn fixed_width_range_validation_rejects_overflow_without_cuda() {
        validate_range(32, 32, 0, "test").expect("empty end range is valid");
        assert!(validate_range(32, 31, 2, "test").is_err());
        assert!(validate_range(u64::MAX, u64::MAX, 1, "test").is_err());
    }

    #[test]
    fn forgotten_reservation_remains_busy_fail_closed() {
        struct ForgottenUse<'a> {
            state: &'a BufferUseState,
        }
        impl Drop for ForgottenUse<'_> {
            fn drop(&mut self) {
                self.state.finish();
            }
        }

        let state = BufferUseState::new();
        state.begin("test", "buffer").expect("first use reserves");
        std::mem::forget(ForgottenUse { state: &state });
        assert!(state.ensure_idle("test", "buffer").is_err());
        assert!(state.begin("test", "buffer").is_err());
    }

    #[test]
    fn allocation_stats_zero_contract_is_explicit() {
        assert!(CudaAllocationStats::default().is_zero());
        let logical_zero = CudaAllocationStats {
            device_live_bytes: 0,
            device_live_allocations: 1,
            pinned_host_live_bytes: 0,
            pinned_host_live_allocations: 0,
        };
        assert!(!logical_zero.is_zero());
    }
}
