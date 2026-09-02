use riley_cuda::{
    CudaGraphCaptureCapability, CudaGraphCaptureOperation, EXPECTED_ABI_VERSION, abi_version,
    build_info,
};

#[test]
fn native_symbols_link_without_device_initialization() -> riley_core::Result<()> {
    assert_eq!(abi_version()?, EXPECTED_ABI_VERSION);

    let info = build_info()?;
    assert!(info.starts_with("riley-cuda-native abi=1 nvcc="));

    // The capability query is a pure native vocabulary lookup. It must link
    // and report the exact reviewed C05 operations without creating a device
    // context, so this ABI test intentionally never calls CudaRuntime.
    for operation in [
        CudaGraphCaptureOperation::FillF32,
        CudaGraphCaptureOperation::H2D,
        CudaGraphCaptureOperation::SiluBf16,
        CudaGraphCaptureOperation::GatedMultiplyBf16,
        CudaGraphCaptureOperation::ResidualAddBf16,
        CudaGraphCaptureOperation::CanonicalRmsNormBf16,
    ] {
        assert_eq!(
            operation
                .capture_capability()
                .expect("reviewed C05 capability query must succeed"),
            CudaGraphCaptureCapability::Supported
        );
    }
    Ok(())
}
