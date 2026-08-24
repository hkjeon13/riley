use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::mem::{offset_of, size_of};
use std::ptr::{self, NonNull};

use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};

const STATUS_SUCCESS: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = 1;
const STATUS_INVALID_DEVICE: i32 = 2;
const STATUS_OUT_OF_RANGE: i32 = 3;
const STATUS_NOT_READY: i32 = 4;
const STATUS_OUT_OF_MEMORY: i32 = 5;
const STATUS_DRIVER_ERROR: i32 = 6;
const STATUS_RUNTIME_ERROR: i32 = 7;
const STATUS_INVALID_STATE: i32 = 8;

const DOMAIN_VALIDATION: u32 = 1;
const DOMAIN_DRIVER: u32 = 2;
const DOMAIN_RUNTIME: u32 = 3;

const STAGE_INITIALIZE: u32 = 1;
const STAGE_CREATE: u32 = 3;
const STAGE_LAUNCH: u32 = 4;
const STAGE_SYNCHRONIZE: u32 = 5;
const STAGE_QUERY: u32 = 6;
const STAGE_RECORD: u32 = 7;
const STAGE_COPY: u32 = 8;
const STAGE_CLOSE: u32 = 9;

const ERROR_MESSAGE_CAPACITY: usize = 256;
const DEVICE_NAME_CAPACITY: usize = 256;
const ERROR_INFO_SIZE: u32 = 272;
const DEVICE_PROPERTIES_SIZE: u32 = 320;

#[repr(C)]
struct ErrorInfo {
    struct_size: u32,
    native_code: i32,
    domain: u32,
    stage: u32,
    message: [c_char; ERROR_MESSAGE_CAPACITY],
}

impl ErrorInfo {
    fn new() -> Self {
        Self {
            struct_size: ERROR_INFO_SIZE,
            native_code: 0,
            domain: 0,
            stage: 0,
            message: [0; ERROR_MESSAGE_CAPACITY],
        }
    }
}

#[repr(C)]
struct RawDeviceProperties {
    struct_size: u32,
    ordinal: i32,
    total_memory_bytes: u64,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multiprocessor_count: u32,
    warp_size: u32,
    max_threads_per_block: u32,
    driver_version: i32,
    runtime_version: i32,
    reserved: [u32; 5],
    name: [c_char; DEVICE_NAME_CAPACITY],
}

impl RawDeviceProperties {
    fn new() -> Self {
        Self {
            struct_size: DEVICE_PROPERTIES_SIZE,
            ordinal: 0,
            total_memory_bytes: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            multiprocessor_count: 0,
            warp_size: 0,
            max_threads_per_block: 0,
            driver_version: 0,
            runtime_version: 0,
            reserved: [0; 5],
            name: [0; DEVICE_NAME_CAPACITY],
        }
    }
}

#[derive(Debug)]
pub(super) struct NativeDeviceProperties {
    pub(super) ordinal: i32,
    pub(super) name: String,
    pub(super) total_memory_bytes: u64,
    pub(super) compute_capability_major: u32,
    pub(super) compute_capability_minor: u32,
    pub(super) multiprocessor_count: u32,
    pub(super) warp_size: u32,
    pub(super) max_threads_per_block: u32,
    pub(super) driver_version: i32,
    pub(super) runtime_version: i32,
}

#[repr(C)]
struct RawContext {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawStream {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawEvent {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawSmokeBuffer {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

unsafe extern "C" {
    fn rustinfer_cuda_abi_version() -> u32;
    fn rustinfer_cuda_build_info() -> *const c_char;
    fn rustinfer_cuda_device_count(out_count: *mut u32, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_device_properties(
        ordinal: i32,
        out_properties: *mut RawDeviceProperties,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_create(
        ordinal: i32,
        out_context: *mut *mut RawContext,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_synchronize(context: *mut RawContext, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_context_memory_info(
        context: *mut RawContext,
        out_free_bytes: *mut u64,
        out_total_bytes: *mut u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_close(context: *mut *mut RawContext, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_stream_create(
        context: *mut RawContext,
        out_stream: *mut *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_query(
        stream: *mut RawStream,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_synchronize(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_stream_wait_event(
        stream: *mut RawStream,
        event: *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_close(stream: *mut *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_event_create(
        context: *mut RawContext,
        out_event: *mut *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_record(
        event: *mut RawEvent,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_query(
        event: *mut RawEvent,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_synchronize(event: *mut RawEvent, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_event_elapsed_ms(
        start: *mut RawEvent,
        end: *mut RawEvent,
        out_elapsed_ms: *mut f32,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_close(event: *mut *mut RawEvent, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_smoke_buffer_create(
        context: *mut RawContext,
        element_count: u64,
        out_buffer: *mut *mut RawSmokeBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_fill_launch(
        buffer: *mut RawSmokeBuffer,
        stream: *mut RawStream,
        value: f32,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_copy_to_host(
        buffer: *mut RawSmokeBuffer,
        stream: *mut RawStream,
        host_output: *mut f32,
        host_element_capacity: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_buffer_close(
        buffer: *mut *mut RawSmokeBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_invalid_launch(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
}

pub(super) fn abi_version() -> u32 {
    // SAFETY: the statically linked metadata function takes no arguments and
    // returns a fixed-width value defined by the checked C header.
    unsafe { rustinfer_cuda_abi_version() }
}

pub(super) fn build_info() -> CudaResult<String> {
    // SAFETY: the native ABI returns null or a process-lifetime C string.
    let pointer = unsafe { rustinfer_cuda_build_info() };
    if pointer.is_null() {
        return Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Initialize,
            0,
            "read native build info",
            "native build-info pointer is null",
        ));
    }
    // SAFETY: null was rejected and the ABI promises NUL termination and
    // process lifetime; the bytes are copied before returning.
    let value = unsafe { CStr::from_ptr(pointer) };
    value.to_str().map(str::to_owned).map_err(|error| {
        CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Initialize,
            0,
            "read native build info",
            format!("native build info is not UTF-8: {error}"),
        )
    })
}

pub(super) fn device_count() -> CudaResult<u32> {
    let mut count = 0;
    let mut error = ErrorInfo::new();
    // SAFETY: both output pointers refer to initialized, writable values for
    // the duration of the synchronous C call.
    let status = unsafe { rustinfer_cuda_device_count(&mut count, &mut error) };
    status_result(status, "enumerate CUDA devices", &error)?;
    Ok(count)
}

pub(super) fn diagnose_null_device_count() -> CudaResult<()> {
    let mut error = ErrorInfo::new();
    // SAFETY: null is intentionally supplied to exercise the documented ABI
    // validation path; no memory is dereferenced by contract.
    let status = unsafe { rustinfer_cuda_device_count(ptr::null_mut(), &mut error) };
    status_result(status, "diagnose null device-count output", &error)
}

pub(super) fn device_properties(ordinal: i32) -> CudaResult<NativeDeviceProperties> {
    let mut properties = RawDeviceProperties::new();
    let mut error = ErrorInfo::new();
    // SAFETY: properties and error are correctly sized repr(C) caller buffers.
    let status = unsafe { rustinfer_cuda_device_properties(ordinal, &mut properties, &mut error) };
    status_result(status, "query CUDA device properties", &error)?;
    Ok(NativeDeviceProperties {
        ordinal: properties.ordinal,
        name: c_array_to_string(&properties.name),
        total_memory_bytes: properties.total_memory_bytes,
        compute_capability_major: properties.compute_capability_major,
        compute_capability_minor: properties.compute_capability_minor,
        multiprocessor_count: properties.multiprocessor_count,
        warp_size: properties.warp_size,
        max_threads_per_block: properties.max_threads_per_block,
        driver_version: properties.driver_version,
        runtime_version: properties.runtime_version,
    })
}

pub(super) struct ContextHandle {
    pointer: Option<NonNull<RawContext>>,
}

// SAFETY: native operations push/pop the retained primary context on each
// calling thread, and mutation/destruction requires unique Rust ownership.
unsafe impl Send for ContextHandle {}
// SAFETY: shared context methods are thread-safe CUDA calls; close requires
// `&mut self` and the safe layer prevents it while child Arc references exist.
unsafe impl Sync for ContextHandle {}

impl ContextHandle {
    pub(super) fn create(ordinal: i32) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: pointer and error are writable caller buffers; native returns
        // either null or one newly owned context handle.
        let status = unsafe { rustinfer_cuda_context_create(ordinal, &mut pointer, &mut error) };
        status_result(status, "create CUDA context", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output("create CUDA context", "native context handle is null")
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawContext {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn synchronize(&self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the handle is owned and kept alive by self for this call.
        let status = unsafe { rustinfer_cuda_context_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA context", &error)
    }

    pub(super) fn memory_info(&self) -> CudaResult<(u64, u64)> {
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the live context and both output buffers remain valid for the
        // synchronous native call.
        let status = unsafe {
            rustinfer_cuda_context_memory_info(
                self.as_ptr(),
                &mut free_bytes,
                &mut total_bytes,
                &mut error,
            )
        };
        status_result(status, "query CUDA memory info", &error)?;
        Ok((free_bytes, total_bytes))
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is this handle's unique owned pointer; the native close
        // contract nulls it only after consuming the resource.
        let status = unsafe { rustinfer_cuda_context_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA context", &error)
    }
}

impl Drop for ContextHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct StreamHandle {
    pointer: Option<NonNull<RawStream>>,
}

// SAFETY: CUDA streams may move between host threads because every operation
// restores the current CUDA context. The safe wrapper intentionally is !Sync.
unsafe impl Send for StreamHandle {}

impl StreamHandle {
    pub(super) fn create(context: &ContextHandle) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context stays alive and output/error are writable buffers.
        let status =
            unsafe { rustinfer_cuda_stream_create(context.as_ptr(), &mut pointer, &mut error) };
        status_result(status, "create CUDA stream", &error)?;
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| missing_output("create CUDA stream", "native stream handle is null"))?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawStream {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn query(&mut self) -> CudaResult<bool> {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the stream and output buffers remain live for the call.
        let status =
            unsafe { rustinfer_cuda_stream_query(self.as_ptr(), &mut complete, &mut error) };
        query_result(status, complete, "query CUDA stream", &error)
    }

    pub(super) fn synchronize(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle.
        let status = unsafe { rustinfer_cuda_stream_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA stream", &error)
    }

    pub(super) fn wait_event(&mut self, event: &EventHandle) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both native handles remain alive and the native ABI validates
        // that they belong to the same context.
        let status =
            unsafe { rustinfer_cuda_stream_wait_event(self.as_ptr(), event.as_ptr(), &mut error) };
        status_result(status, "wait for CUDA event", &error)
    }

    pub(super) fn diagnose_invalid_launch(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the function intentionally issues a configuration-invalid
        // launch against this live stream and clears the launch error.
        let status = unsafe { rustinfer_cuda_smoke_invalid_launch(self.as_ptr(), &mut error) };
        status_result(status, "diagnose invalid CUDA launch", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned and native nulls it only on consume.
        let status = unsafe { rustinfer_cuda_stream_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA stream", &error)
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct EventHandle {
    pointer: Option<NonNull<RawEvent>>,
}

// SAFETY: events may move between host threads with context push/pop; the safe
// wrapper is intentionally !Sync and serializes mutable operations.
unsafe impl Send for EventHandle {}

impl EventHandle {
    pub(super) fn create(context: &ContextHandle) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context stays alive and output/error are writable buffers.
        let status =
            unsafe { rustinfer_cuda_event_create(context.as_ptr(), &mut pointer, &mut error) };
        status_result(status, "create CUDA event", &error)?;
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| missing_output("create CUDA event", "native event handle is null"))?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawEvent {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn record(&mut self, stream: &mut StreamHandle) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both handles remain live and the ABI validates their owner.
        let status =
            unsafe { rustinfer_cuda_event_record(self.as_ptr(), stream.as_ptr(), &mut error) };
        status_result(status, "record CUDA event", &error)
    }

    pub(super) fn query(&mut self) -> CudaResult<bool> {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: event and output remain live for the synchronous call.
        let status =
            unsafe { rustinfer_cuda_event_query(self.as_ptr(), &mut complete, &mut error) };
        query_result(status, complete, "query CUDA event", &error)
    }

    pub(super) fn synchronize(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live event handle.
        let status = unsafe { rustinfer_cuda_event_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA event", &error)
    }

    pub(super) fn elapsed_ms(&self, end: &Self) -> CudaResult<f32> {
        let mut elapsed = 0.0;
        let mut error = ErrorInfo::new();
        // SAFETY: both event handles and output remain valid; native validates
        // context identity and recording/completion state.
        let status = unsafe {
            rustinfer_cuda_event_elapsed_ms(self.as_ptr(), end.as_ptr(), &mut elapsed, &mut error)
        };
        status_result(status, "measure CUDA event elapsed time", &error)?;
        Ok(elapsed)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned and native nulls it only on consume.
        let status = unsafe { rustinfer_cuda_event_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA event", &error)
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct SmokeHandle {
    pointer: Option<NonNull<RawSmokeBuffer>>,
}

// SAFETY: diagnostic buffers move only with their borrowed !Sync stream in the
// safe layer; context lifetime is retained separately by Arc.
unsafe impl Send for SmokeHandle {}

impl SmokeHandle {
    pub(super) fn create(context: &ContextHandle, element_count: u64) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context is live and native returns one owned opaque handle.
        let status = unsafe {
            rustinfer_cuda_smoke_buffer_create(
                context.as_ptr(),
                element_count,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "create CUDA smoke buffer", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output("create CUDA smoke buffer", "native smoke handle is null")
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawSmokeBuffer {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn launch(&mut self, stream: &mut StreamHandle, value: f32) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both handles remain live and native validates context/launch.
        let status = unsafe {
            rustinfer_cuda_smoke_fill_launch(self.as_ptr(), stream.as_ptr(), value, &mut error)
        };
        status_result(status, "launch CUDA smoke fill", &error)
    }

    pub(super) fn copy_to_host(
        &mut self,
        stream: &mut StreamHandle,
        output: &mut [f32],
    ) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        let capacity = u64::try_from(output.len()).map_err(|_| {
            CudaError::out_of_range(
                "copy CUDA smoke buffer",
                "host output length does not fit u64",
            )
        })?;
        // SAFETY: output remains exclusively borrowed for the complete native
        // enqueue+stream-synchronize call, and both handles remain live.
        let status = unsafe {
            rustinfer_cuda_smoke_copy_to_host(
                self.as_ptr(),
                stream.as_ptr(),
                output.as_mut_ptr(),
                capacity,
                &mut error,
            )
        };
        status_result(status, "copy CUDA smoke buffer", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned; native synchronizes any in-flight
        // operation before consuming and nulling the handle.
        let status = unsafe { rustinfer_cuda_smoke_buffer_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA smoke buffer", &error)
    }
}

impl Drop for SmokeHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn query_result(
    status: i32,
    complete: u8,
    operation: &'static str,
    error: &ErrorInfo,
) -> CudaResult<bool> {
    if status == STATUS_NOT_READY {
        return Ok(false);
    }
    status_result(status, operation, error)?;
    Ok(complete != 0)
}

fn status_result(status: i32, operation: &'static str, error: &ErrorInfo) -> CudaResult<()> {
    if status == STATUS_SUCCESS {
        return Ok(());
    }
    let kind = match status {
        STATUS_INVALID_ARGUMENT => CudaErrorKind::InvalidArgument,
        STATUS_INVALID_DEVICE => CudaErrorKind::InvalidDevice,
        STATUS_OUT_OF_RANGE => CudaErrorKind::OutOfRange,
        STATUS_NOT_READY => CudaErrorKind::NotReady,
        STATUS_OUT_OF_MEMORY => CudaErrorKind::OutOfMemory,
        STATUS_DRIVER_ERROR => CudaErrorKind::Driver,
        STATUS_RUNTIME_ERROR => CudaErrorKind::Runtime,
        STATUS_INVALID_STATE => CudaErrorKind::InvalidState,
        _ => CudaErrorKind::Internal,
    };
    let domain = match error.domain {
        DOMAIN_VALIDATION => CudaErrorDomain::Validation,
        DOMAIN_DRIVER => CudaErrorDomain::Driver,
        DOMAIN_RUNTIME => CudaErrorDomain::Runtime,
        _ => CudaErrorDomain::Internal,
    };
    let stage = match error.stage {
        STAGE_INITIALIZE => CudaErrorStage::Initialize,
        STAGE_CREATE => CudaErrorStage::Create,
        STAGE_LAUNCH => CudaErrorStage::Launch,
        STAGE_SYNCHRONIZE => CudaErrorStage::Synchronize,
        STAGE_QUERY => CudaErrorStage::Query,
        STAGE_RECORD => CudaErrorStage::Record,
        STAGE_COPY => CudaErrorStage::Copy,
        STAGE_CLOSE => CudaErrorStage::Close,
        _ => CudaErrorStage::Validation,
    };
    let message = c_array_to_string(&error.message);
    Err(CudaError::new(
        kind,
        domain,
        stage,
        error.native_code,
        operation,
        if message.is_empty() {
            format!("native ABI returned undocumented status {status}")
        } else {
            message
        },
    ))
}

fn missing_output(operation: &'static str, message: &'static str) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        CudaErrorStage::Create,
        0,
        operation,
        message,
    )
}

fn c_array_to_string<const N: usize>(bytes: &[c_char; N]) -> String {
    let bytes: Vec<u8> = bytes
        .iter()
        .copied()
        .take_while(|byte| *byte != 0)
        .map(|byte| u8::from_ne_bytes(byte.to_ne_bytes()))
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

const _: () = assert!(size_of::<ErrorInfo>() == 272);
const _: () = assert!(offset_of!(ErrorInfo, message) == 16);
const _: () = assert!(size_of::<RawDeviceProperties>() == 320);
const _: () = assert!(offset_of!(RawDeviceProperties, name) == 64);
