use super::*;
use std::ffi::c_void;

#[repr(C)]
struct RawFa3Spec {
    version: u32,
    mode: u32,
    batch: u32,
    total_q: u32,
    physical_pages: u32,
    q_indptr: *const i32,
    kv_lengths: *const i32,
    page_indptr: *const i32,
    pages: *const i32,
    page_entries: u32,
}
const _: () = assert!(size_of::<RawFa3Spec>() == 64);
const _: () = assert!(offset_of!(RawFa3Spec, q_indptr) == 24);
unsafe extern "C" {
    fn riley_cuda_fa3_owner_create(
        stream: *mut RawStream,
        q: *mut RawDeviceBuffer,
        k: *mut RawDeviceBuffer,
        v: *mut RawDeviceBuffer,
        o: *mut RawDeviceBuffer,
        spec: *const RawFa3Spec,
        out: *mut *mut c_void,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fa3_owner_operation(
        owner: *mut c_void,
        operation: u32,
        error: *mut ErrorInfo,
    ) -> i32;
}
pub(crate) struct Fa3OwnerHandle {
    pointer: Option<NonNull<c_void>>,
}
impl Fa3OwnerHandle {
    pub(crate) fn create(
        batch: &crate::Fa3Batch,
        stream: &StreamHandle,
        buffers: [&DeviceBufferHandle; 4],
    ) -> CudaResult<Self> {
        let spec = RawFa3Spec {
            version: 1,
            mode: u32::from(batch.decode),
            batch: batch.kv_lengths.len() as u32,
            total_q: *batch.q_indptr.last().expect("validated FA3 batch") as u32,
            physical_pages: batch.physical_pages,
            q_indptr: batch.q_indptr.as_ptr(),
            kv_lengths: batch.kv_lengths.as_ptr(),
            page_indptr: batch.page_indptr.as_ptr(),
            pages: batch.pages.as_ptr(),
            page_entries: batch.pages.len() as u32,
        };
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: validated host vectors and exclusively borrowed device/stream owners
        // live for the call; native copies metadata and retains active-use leases.
        let status = unsafe {
            riley_cuda_fa3_owner_create(
                stream.as_ptr(),
                buffers[0].as_ptr(),
                buffers[1].as_ptr(),
                buffers[2].as_ptr(),
                buffers[3].as_ptr(),
                &spec,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "create FA3 owner", &error)?;
        Ok(Self {
            pointer: Some(
                NonNull::new(pointer)
                    .ok_or_else(|| missing_output("create FA3 owner", "null owner"))?,
            ),
        })
    }
    pub(crate) fn operation(&mut self, operation: u32) -> CudaResult<()> {
        let pointer = self
            .pointer
            .ok_or_else(|| CudaError::invalid_state("FA3 owner", "already closed"))?;
        let mut error = ErrorInfo::new();
        // SAFETY: native retains all resources until a successful close. Calls are serialized.
        let status =
            unsafe { riley_cuda_fa3_owner_operation(pointer.as_ptr(), operation, &mut error) };
        status_result(status, "FA3 owner operation", &error)
    }
    pub(crate) fn close(&mut self) -> CudaResult<()> {
        if self.pointer.is_some() {
            self.operation(2)?;
            self.pointer = None;
        }
        Ok(())
    }
}
impl Drop for Fa3OwnerHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
