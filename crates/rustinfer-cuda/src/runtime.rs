use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;

use crate::error::{CudaError, CudaResult};

#[cfg(feature = "cuda")]
use crate::ffi;

/// Initialized CUDA host-runtime view.
///
/// Initialization enumerates through the Driver API but does not create a
/// context until [`CudaDevice::create_context`] is called.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRuntime {
    device_count: u32,
}

impl CudaRuntime {
    /// Initializes the compiled CUDA boundary and snapshots the device count.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CudaErrorKind::Unavailable`] when compiled without the
    /// `cuda` feature, or the translated CUDA initialization error otherwise.
    pub fn initialize() -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let actual_abi = ffi::abi_version();
            validate_runtime_abi(actual_abi)?;
            let device_count = ffi::device_count()?;
            Ok(Self { device_count })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaRuntime::initialize"))
        }
    }

    /// Number of CUDA devices visible at initialization.
    #[must_use]
    pub const fn device_count(&self) -> u32 {
        self.device_count
    }

    /// Returns one visible device and its immutable properties.
    ///
    /// # Errors
    ///
    /// Returns an invalid-device error for an ordinal outside the initialization
    /// snapshot, or a translated native property-query error.
    pub fn device(&self, ordinal: u32) -> CudaResult<CudaDevice> {
        if ordinal >= self.device_count {
            return Err(CudaError::invalid_device(
                "CudaRuntime::device",
                format!(
                    "device ordinal {ordinal} is outside visible range 0..{}",
                    self.device_count
                ),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native_ordinal = i32::try_from(ordinal).map_err(|_| {
                CudaError::invalid_device(
                    "CudaRuntime::device",
                    format!("device ordinal {ordinal} does not fit the native ABI"),
                )
            })?;
            let native = ffi::device_properties(native_ordinal)?;
            Ok(CudaDevice {
                properties: DeviceProperties::from_native(native)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaRuntime::device"))
        }
    }

    /// Returns all visible devices in ordinal order.
    ///
    /// # Errors
    ///
    /// Returns the first translated device-property error.
    pub fn devices(&self) -> CudaResult<Vec<CudaDevice>> {
        (0..self.device_count)
            .map(|ordinal| self.device(ordinal))
            .collect()
    }

    /// Exercises the raw C ABI null-output validation without dereferencing it.
    ///
    /// This diagnostic exists for ABI contract testing; success would indicate
    /// a native validation bug.
    ///
    /// # Errors
    ///
    /// Always returns the native invalid-argument error in a correct CUDA build.
    #[doc(hidden)]
    pub fn diagnose_null_device_output(&self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            ffi::diagnose_null_device_count()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(
                "CudaRuntime::diagnose_null_device_output",
            ))
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn validate_runtime_abi(actual: u32) -> CudaResult<()> {
    if actual == crate::EXPECTED_ABI_VERSION {
        Ok(())
    } else {
        Err(CudaError::new(
            crate::CudaErrorKind::Internal,
            crate::CudaErrorDomain::Internal,
            crate::CudaErrorStage::Initialize,
            0,
            "CudaRuntime::initialize",
            format!(
                "native ABI mismatch: Rust expects {}, native library reports {actual}",
                crate::EXPECTED_ABI_VERSION
            ),
        ))
    }
}

/// One visible CUDA device and its cached immutable metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDevice {
    properties: DeviceProperties,
}

impl CudaDevice {
    /// Device ordinal used by CUDA.
    #[must_use]
    pub const fn ordinal(&self) -> u32 {
        self.properties.ordinal
    }

    /// Cached device and runtime metadata.
    #[must_use]
    pub const fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Retains an owning lease on this device's primary CUDA context.
    ///
    /// # Errors
    ///
    /// Returns a translated driver/runtime context initialization error.
    pub fn create_context(&self) -> CudaResult<CudaContext> {
        #[cfg(feature = "cuda")]
        {
            let ordinal = i32::try_from(self.ordinal()).map_err(|_| {
                CudaError::invalid_device(
                    "CudaDevice::create_context",
                    "device ordinal does not fit the native ABI",
                )
            })?;
            let native = ffi::ContextHandle::create(ordinal)?;
            Ok(CudaContext {
                inner: Arc::new(ContextInner {
                    ordinal: self.ordinal(),
                    compute_capability: self.properties.compute_capability(),
                    native,
                }),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaDevice::create_context"))
        }
    }
}

/// Immutable device properties suitable for benchmark provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceProperties {
    ordinal: u32,
    name: String,
    total_memory_bytes: u64,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multiprocessor_count: u32,
    warp_size: u32,
    max_threads_per_block: u32,
    driver_version: i32,
    runtime_version: i32,
}

impl DeviceProperties {
    #[cfg(feature = "cuda")]
    fn from_native(native: ffi::NativeDeviceProperties) -> CudaResult<Self> {
        let ordinal = u32::try_from(native.ordinal).map_err(|_| {
            CudaError::invalid_device(
                "DeviceProperties::from_native",
                "native device ordinal is negative",
            )
        })?;
        Ok(Self {
            ordinal,
            name: native.name,
            total_memory_bytes: native.total_memory_bytes,
            compute_capability_major: native.compute_capability_major,
            compute_capability_minor: native.compute_capability_minor,
            multiprocessor_count: native.multiprocessor_count,
            warp_size: native.warp_size,
            max_threads_per_block: native.max_threads_per_block,
            driver_version: native.driver_version,
            runtime_version: native.runtime_version,
        })
    }

    /// Device ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// CUDA-reported device name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total global device memory.
    #[must_use]
    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }

    /// Compute capability `(major, minor)`.
    #[must_use]
    pub const fn compute_capability(&self) -> (u32, u32) {
        (self.compute_capability_major, self.compute_capability_minor)
    }

    /// Streaming multiprocessor count.
    #[must_use]
    pub const fn multiprocessor_count(&self) -> u32 {
        self.multiprocessor_count
    }

    /// Native warp size.
    #[must_use]
    pub const fn warp_size(&self) -> u32 {
        self.warp_size
    }

    /// Maximum threads in one block.
    #[must_use]
    pub const fn max_threads_per_block(&self) -> u32 {
        self.max_threads_per_block
    }

    /// CUDA driver version encoded by CUDA, for example `12080`.
    #[must_use]
    pub const fn driver_version(&self) -> i32 {
        self.driver_version
    }

    /// CUDA Runtime version encoded by CUDA.
    #[must_use]
    pub const fn runtime_version(&self) -> i32 {
        self.runtime_version
    }

    /// Stable single-line benchmark evidence marker.
    #[must_use]
    pub fn benchmark_metadata_line(&self) -> String {
        format!(
            "rustinfer-cuda-device-metadata ordinal={} name={} compute_capability={}.{} total_memory_bytes={} multiprocessor_count={} driver_version={} runtime_version={}",
            self.ordinal,
            self.name,
            self.compute_capability_major,
            self.compute_capability_minor,
            self.total_memory_bytes,
            self.multiprocessor_count,
            self.driver_version,
            self.runtime_version
        )
    }
}

pub(crate) struct ContextInner {
    pub(crate) ordinal: u32,
    pub(crate) compute_capability: (u32, u32),
    #[cfg(feature = "cuda")]
    pub(crate) native: ffi::ContextHandle,
}

/// Owning retained lease on a device's primary context.
///
/// `CudaContext` is `Send + Sync`: each native operation pushes and restores
/// the context on the calling host thread. Child resources retain an internal
/// `Arc`, so the context cannot close before them.
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

impl CudaContext {
    /// Device ordinal associated with this context lease.
    #[must_use]
    pub fn device_ordinal(&self) -> u32 {
        self.inner.ordinal
    }

    /// Cached compute capability of the context's device.
    ///
    /// This is captured from [`CudaDevice::properties`] before the context is
    /// created, so cold backend selection does not reinitialize the runtime or
    /// perform another device-properties query.
    #[must_use]
    pub fn compute_capability(&self) -> (u32, u32) {
        self.inner.compute_capability
    }

    /// Creates a non-blocking, non-default CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns a translated native stream creation error.
    pub fn create_stream(&self) -> CudaResult<CudaStream> {
        #[cfg(feature = "cuda")]
        {
            let native = ffi::StreamHandle::create(&self.inner.native)?;
            Ok(CudaStream {
                context: Arc::clone(&self.inner),
                native,
                _not_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaContext::create_stream"))
        }
    }

    /// Creates a timing-enabled CUDA event.
    ///
    /// # Errors
    ///
    /// Returns a translated native event creation error.
    pub fn create_event(&self) -> CudaResult<CudaEvent> {
        #[cfg(feature = "cuda")]
        {
            let native = ffi::EventHandle::create(&self.inner.native)?;
            Ok(CudaEvent {
                context: Arc::clone(&self.inner),
                native,
                _not_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaContext::create_event"))
        }
    }

    /// Returns the statically linked diagnostic fill kernel handle.
    #[must_use]
    pub fn kernel(&self) -> CudaKernel {
        CudaKernel {
            context: Arc::clone(&self.inner),
        }
    }

    /// Synchronizes all work in this context and surfaces late errors.
    ///
    /// # Errors
    ///
    /// Returns a synchronize-stage CUDA error.
    pub fn synchronize(&self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.inner.native.synchronize()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaContext::synchronize"))
        }
    }

    /// Returns `(free_bytes, total_bytes)` for leak-smoke evidence.
    ///
    /// # Errors
    ///
    /// Returns a translated CUDA memory query error.
    pub fn memory_info(&self) -> CudaResult<(u64, u64)> {
        #[cfg(feature = "cuda")]
        {
            self.inner.native.memory_info()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaContext::memory_info"))
        }
    }

    /// Explicitly closes the context after every child resource has closed.
    ///
    /// # Errors
    ///
    /// Returns invalid-state when streams, events, kernels, buffers, or pending
    /// operations still retain it, or a translated native close error.
    pub fn close(self) -> CudaResult<()> {
        let strong_count = Arc::strong_count(&self.inner);
        let inner = Arc::try_unwrap(self.inner).map_err(|_| {
            CudaError::invalid_state(
                "CudaContext::close",
                format!(
                    "context still has {} child/shared references",
                    strong_count.saturating_sub(1)
                ),
            )
        })?;
        #[cfg(feature = "cuda")]
        {
            let mut native = inner.native;
            native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = inner;
            Err(CudaError::unavailable("CudaContext::close"))
        }
    }
}

/// Explicit non-default CUDA stream.
///
/// A stream is `Send` but deliberately `!Sync`; ordered mutations require
/// `&mut self`. Drop is best-effort, while [`Self::synchronize`] and
/// [`Self::close`] surface errors.
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<rustinfer_cuda::CudaStream>();
/// ```
pub struct CudaStream {
    #[cfg(feature = "cuda")]
    pub(crate) native: ffi::StreamHandle,
    // Declared after native so drop closes the child before releasing the
    // context Arc; native context close rejects live children by contract.
    pub(crate) context: Arc<ContextInner>,
    _not_sync: PhantomData<Cell<()>>,
}

impl CudaStream {
    /// Begins a native command batch on this stream.
    ///
    /// Primitive calls made through [`CudaCommandBatch::stream_mut`] remain
    /// ordered on the stream, while their native wrappers may defer per-command
    /// synchronization until the batch ends. The guard holds the stream's
    /// exclusive mutable borrow, preventing direct stream use for its lifetime.
    /// Call [`CudaCommandBatch::finish`] to surface completion errors; dropping
    /// the guard performs the same native end operation on a best-effort basis.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CudaErrorKind::Unavailable`] when compiled without the
    /// `cuda` feature, or a native lifecycle error when the stream cannot enter
    /// command-batch mode (including an already-active batch).
    pub fn begin_command_batch(&mut self) -> CudaResult<CudaCommandBatch<'_>> {
        #[cfg(feature = "cuda")]
        {
            self.native.command_batch_begin()?;
            Ok(CudaCommandBatch {
                stream: self,
                active: true,
                _not_send_or_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaStream::begin_command_batch"))
        }
    }

    /// Non-blocking completion check.
    ///
    /// Native `cudaErrorNotReady` is returned as `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns any query failure other than not-ready. Context restoration
    /// failures take precedence over an otherwise normal not-ready result.
    pub fn query(&mut self) -> CudaResult<bool> {
        #[cfg(feature = "cuda")]
        {
            self.native.query()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaStream::query"))
        }
    }

    /// Blocks for all preceding stream work and surfaces late errors.
    ///
    /// # Errors
    ///
    /// Returns a synchronize-stage CUDA error.
    pub fn synchronize(&mut self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.synchronize()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaStream::synchronize"))
        }
    }

    /// Enqueues a dependency on an event from the same context.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for a context mismatch or a translated record
    /// failure.
    pub fn wait_event(&mut self, event: &CudaEvent) -> CudaResult<()> {
        ensure_same_context(&self.context, &event.context, "CudaStream::wait_event")?;
        #[cfg(feature = "cuda")]
        {
            self.native.wait_event(&event.native)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaStream::wait_event"))
        }
    }

    /// Explicitly destroys this stream.
    ///
    /// Synchronize first when late execution errors must be reported separately.
    ///
    /// # Errors
    ///
    /// Returns a translated native close error.
    pub fn close(self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let mut this = self;
            this.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = self;
            Err(CudaError::unavailable("CudaStream::close"))
        }
    }
}

/// Exclusive RAII guard for one native stream command batch.
///
/// The guard is deliberately `!Send + !Sync` because the native batch is owned
/// by the host thread that began it. Its mutable borrow also prevents the stream
/// from being accessed directly until [`Self::finish`] returns or the guard is
/// dropped.
///
/// ```compile_fail
/// fn cannot_alias(stream: &mut rustinfer_cuda::CudaStream) {
///     let batch = stream.begin_command_batch().unwrap();
///     let _ = stream.synchronize();
///     drop(batch);
/// }
/// ```
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<rustinfer_cuda::CudaCommandBatch<'static>>();
/// ```
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<rustinfer_cuda::CudaCommandBatch<'static>>();
/// ```
#[must_use = "finish the command batch explicitly to observe completion errors"]
pub struct CudaCommandBatch<'stream> {
    stream: &'stream mut CudaStream,
    active: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl CudaCommandBatch<'_> {
    /// Reborrows the exclusively held stream for ordered CUDA operations.
    ///
    /// The returned borrow cannot outlive this guard. Native lifecycle checks
    /// reject attempts to begin a nested command batch.
    pub fn stream_mut(&mut self) -> &mut CudaStream {
        self.stream
    }

    /// Ends the command batch exactly once and reports completion errors.
    ///
    /// The guard marks the end operation consumed before entering native code,
    /// so its destructor never retries an end call that returned an error.
    /// The native boundary owns fail-closed resource retention after ambiguous
    /// completion; retrying the lifecycle transition would be unsafe.
    ///
    /// # Errors
    ///
    /// Returns the translated native end/synchronization error. A dropped guard
    /// cannot report this error, so callers that require reliable completion
    /// diagnostics must call `finish`.
    pub fn finish(mut self) -> CudaResult<()> {
        self.end_once()
    }

    fn end_once(&mut self) -> CudaResult<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        #[cfg(feature = "cuda")]
        {
            self.stream.native.command_batch_end()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaCommandBatch::finish"))
        }
    }
}

impl Drop for CudaCommandBatch<'_> {
    fn drop(&mut self) {
        let _ = self.end_once();
    }
}

/// Timing-enabled CUDA event.
///
/// Events are `Send` but deliberately `!Sync`. Drop is best-effort; explicit
/// synchronize and close methods expose errors.
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<rustinfer_cuda::CudaEvent>();
/// ```
pub struct CudaEvent {
    #[cfg(feature = "cuda")]
    native: ffi::EventHandle,
    // See CudaStream: native child destruction must precede Arc release.
    context: Arc<ContextInner>,
    _not_sync: PhantomData<Cell<()>>,
}

impl CudaEvent {
    /// Records this event after preceding work on `stream`.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for a context mismatch or a native record error.
    pub fn record(&mut self, stream: &mut CudaStream) -> CudaResult<()> {
        ensure_same_context(&self.context, &stream.context, "CudaEvent::record")?;
        #[cfg(feature = "cuda")]
        {
            self.native.record(&mut stream.native)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaEvent::record"))
        }
    }

    /// Non-blocking completion query; not-ready maps to `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns any query failure other than not-ready. Context restoration
    /// failures take precedence over an otherwise normal not-ready result.
    pub fn query(&mut self) -> CudaResult<bool> {
        #[cfg(feature = "cuda")]
        {
            self.native.query()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaEvent::query"))
        }
    }

    /// Waits for this event and surfaces asynchronous errors.
    ///
    /// # Errors
    ///
    /// Returns a synchronize-stage CUDA error.
    pub fn synchronize(&mut self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.synchronize()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaEvent::synchronize"))
        }
    }

    /// Measures elapsed milliseconds between two completed recorded events.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for a context mismatch or the native timing error.
    pub fn elapsed_ms(&self, end: &Self) -> CudaResult<f32> {
        ensure_same_context(&self.context, &end.context, "CudaEvent::elapsed_ms")?;
        #[cfg(feature = "cuda")]
        {
            self.native.elapsed_ms(&end.native)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaEvent::elapsed_ms"))
        }
    }

    /// Explicitly destroys this event.
    ///
    /// # Errors
    ///
    /// Returns a translated native close error.
    pub fn close(self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let mut this = self;
            this.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = self;
            Err(CudaError::unavailable("CudaEvent::close"))
        }
    }
}

/// Statically linked AOT diagnostic fill kernel bound to one context.
///
/// The handle is immutable and `Send + Sync`; launches require an exclusive
/// mutable borrow of a same-context stream.
pub struct CudaKernel {
    context: Arc<ContextInner>,
}

impl CudaKernel {
    /// Allocates diagnostic-only storage and enqueues an asynchronous fill.
    ///
    /// This diagnostic path remains separate from the generic byte-buffer API.
    /// The returned pending value exclusively borrows the stream, preventing
    /// early stream/context/buffer destruction before explicit completion.
    ///
    /// # Errors
    ///
    /// Returns size/ownership validation errors, allocation errors, or an
    /// immediate launch-stage CUDA error.
    pub fn launch_fill<'stream>(
        &self,
        stream: &'stream mut CudaStream,
        element_count: u64,
        value: f32,
    ) -> CudaResult<CudaPendingFill<'stream>> {
        ensure_same_context(&self.context, &stream.context, "CudaKernel::launch_fill")?;
        let host_len = usize::try_from(element_count).map_err(|_| {
            CudaError::out_of_range(
                "CudaKernel::launch_fill",
                "element_count does not fit host usize",
            )
        })?;
        host_len.checked_mul(size_of::<f32>()).ok_or_else(|| {
            CudaError::out_of_range(
                "CudaKernel::launch_fill",
                "element_count overflows host output byte length",
            )
        })?;
        #[cfg(feature = "cuda")]
        {
            let mut buffer = ffi::SmokeHandle::create(&self.context.native, element_count)?;
            buffer.launch(&mut stream.native, value)?;
            Ok(CudaPendingFill {
                buffer,
                stream,
                _context: Arc::clone(&self.context),
                host_len,
                finished: false,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (stream, value, host_len);
            Err(CudaError::unavailable("CudaKernel::launch_fill"))
        }
    }

    /// Issues a deliberately invalid, non-poisoning launch for diagnostics.
    ///
    /// # Errors
    ///
    /// A correct CUDA runtime returns an error with [`crate::CudaErrorStage::Launch`].
    #[doc(hidden)]
    pub fn diagnose_invalid_launch(&self, stream: &mut CudaStream) -> CudaResult<()> {
        ensure_same_context(
            &self.context,
            &stream.context,
            "CudaKernel::diagnose_invalid_launch",
        )?;
        #[cfg(feature = "cuda")]
        {
            stream.native.diagnose_invalid_launch()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable(
                "CudaKernel::diagnose_invalid_launch",
            ))
        }
    }
}

/// In-flight diagnostic fill with an exclusive borrow of its launch stream.
///
/// `finish` is the error-reporting path. Dropping unfinished work performs a
/// best-effort stream synchronize before releasing its buffer.
pub struct CudaPendingFill<'stream> {
    #[cfg(feature = "cuda")]
    buffer: ffi::SmokeHandle,
    stream: &'stream mut CudaStream,
    _context: Arc<ContextInner>,
    host_len: usize,
    finished: bool,
}

impl CudaPendingFill<'_> {
    /// Records an event after the pending fill on its borrowed stream.
    ///
    /// # Errors
    ///
    /// Returns a context mismatch or native event-record error.
    pub fn record_event(&mut self, event: &mut CudaEvent) -> CudaResult<()> {
        event.record(self.stream)
    }

    /// Non-blocking query of the borrowed stream.
    ///
    /// # Errors
    ///
    /// Returns a native query failure other than not-ready.
    pub fn query(&mut self) -> CudaResult<bool> {
        self.stream.query()
    }

    /// Explicitly synchronizes, copies the values to host, and closes storage.
    ///
    /// Kernel completion errors are reported at synchronize stage; a failure
    /// in the subsequent blocking device-to-host copy is reported at copy stage.
    ///
    /// # Errors
    ///
    /// Returns the first synchronize, host allocation, copy, or close error.
    pub fn finish(mut self) -> CudaResult<Vec<f32>> {
        self.stream.synchronize()?;
        let mut output = Vec::new();
        output.try_reserve_exact(self.host_len).map_err(|error| {
            CudaError::host_allocation(
                "CudaPendingFill::finish",
                format!("could not reserve host output: {error}"),
            )
        })?;
        output.resize(self.host_len, 0.0);
        #[cfg(feature = "cuda")]
        {
            self.buffer
                .copy_to_host(&mut self.stream.native, &mut output)?;
            self.buffer.close()?;
        }
        self.finished = true;
        Ok(output)
    }
}

impl Drop for CudaPendingFill<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let _ = self.stream.synchronize();
        #[cfg(feature = "cuda")]
        {
            let _ = self.buffer.close();
        }
    }
}

pub(crate) fn ensure_same_context(
    left: &Arc<ContextInner>,
    right: &Arc<ContextInner>,
    operation: &'static str,
) -> CudaResult<()> {
    if Arc::ptr_eq(left, right) {
        Ok(())
    } else {
        Err(CudaError::invalid_state(
            operation,
            "resources belong to different CUDA context owners",
        ))
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::{CudaRuntime, validate_runtime_abi};
    use crate::{CudaErrorDomain, CudaErrorKind, CudaErrorStage};

    #[test]
    fn feature_off_runtime_has_actionable_error() {
        let error = CudaRuntime::initialize().expect_err("CUDA must remain disabled");
        assert_eq!(error.kind(), CudaErrorKind::Unavailable);
        assert_eq!(error.domain(), CudaErrorDomain::Rust);
        assert_eq!(error.stage(), CudaErrorStage::Initialize);
        assert!(error.message().contains("--features cuda"));
    }

    #[test]
    fn runtime_abi_validation_fails_closed_before_device_use() {
        validate_runtime_abi(crate::EXPECTED_ABI_VERSION)
            .expect("matching ABI version must be accepted");
        let error = validate_runtime_abi(crate::EXPECTED_ABI_VERSION + 1)
            .expect_err("ABI mismatch must fail before device enumeration");
        assert_eq!(error.kind(), CudaErrorKind::Internal);
        assert_eq!(error.domain(), CudaErrorDomain::Internal);
        assert_eq!(error.stage(), CudaErrorStage::Initialize);
    }
}
