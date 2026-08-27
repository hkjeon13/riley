mod private {
    pub trait Sealed {}

    impl Sealed for [u8] {}
    impl Sealed for Box<[u8]> {}
    impl Sealed for Vec<u8> {}

    #[cfg(feature = "cuda")]
    impl Sealed for riley_cuda::CudaDeviceBuffer {}

    #[cfg(feature = "cuda")]
    impl Sealed for riley_cuda::CudaPinnedHostBuffer {}
}

/// A backing owner that can anchor a tensor view lifetime.
///
/// This trait is sealed so metadata bounds cannot be forged by downstream
/// implementations. The `cuda` feature adds the opaque device and pinned-host
/// owners without exposing either allocation's raw pointer.
pub trait TensorStorage: private::Sealed {
    /// Returns the accessible storage capacity in bytes.
    fn byte_len(&self) -> u64;
}

impl TensorStorage for [u8] {
    fn byte_len(&self) -> u64 {
        u64::try_from(self.len()).unwrap_or(u64::MAX)
    }
}

impl TensorStorage for Box<[u8]> {
    fn byte_len(&self) -> u64 {
        self.as_ref().byte_len()
    }
}

impl TensorStorage for Vec<u8> {
    fn byte_len(&self) -> u64 {
        self.as_slice().byte_len()
    }
}

#[cfg(feature = "cuda")]
impl TensorStorage for riley_cuda::CudaDeviceBuffer {
    fn byte_len(&self) -> u64 {
        riley_cuda::CudaDeviceBuffer::byte_len(self)
    }
}

#[cfg(feature = "cuda")]
impl TensorStorage for riley_cuda::CudaPinnedHostBuffer {
    fn byte_len(&self) -> u64 {
        riley_cuda::CudaPinnedHostBuffer::byte_len(self)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::TensorStorage;

    #[test]
    fn opaque_cuda_buffers_implement_the_sealed_storage_contract() {
        fn assert_storage<T: TensorStorage>() {}

        assert_storage::<riley_cuda::CudaDeviceBuffer>();
        assert_storage::<riley_cuda::CudaPinnedHostBuffer>();
    }
}
