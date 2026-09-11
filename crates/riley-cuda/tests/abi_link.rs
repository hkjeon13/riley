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
        CudaGraphCaptureOperation::Bf16Argmax,
        CudaGraphCaptureOperation::Bf16RowGather,
        CudaGraphCaptureOperation::Bf16RowGatherArgmax,
        CudaGraphCaptureOperation::Bf16RowGatherArgmaxD2H,
        CudaGraphCaptureOperation::IndexedRopeBf16,
        CudaGraphCaptureOperation::RaggedPagedKvCacheWriteBf16,
        CudaGraphCaptureOperation::GroupedRaggedPagedAttentionBf16,
        CudaGraphCaptureOperation::Bf16EmbeddingStatusD2H,
        CudaGraphCaptureOperation::CanonicalGemmBf16,
        CudaGraphCaptureOperation::CanonicalRmsNormGemmBf16,
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

#[test]
fn c05_22_ffi_surface_declares_both_native_capture_symbols() {
    let ffi_source = include_str!("../src/ffi.rs");
    assert!(ffi_source.contains("riley_cuda_graph_capture_begin_canonical_rms_norm_gemm_bf16"));
    assert!(ffi_source.contains("riley_cuda_graph_capture_enqueue_canonical_rms_norm_gemm_bf16"));
}
