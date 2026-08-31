#![cfg(not(feature = "cuda"))]

use riley_cuda::{
    CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaGraphCaptureCapability,
    CudaGraphCaptureMode, CudaGraphLifecycle, CudaGraphLifecycleState, CudaGraphStage, CudaResult,
    CudaStream, GraphCapture,
};

#[test]
fn graph_contract_is_additive_and_does_not_claim_native_capture_symbols() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");

    for declaration in [
        "typedef struct RileyCudaGraphCapture RileyCudaGraphCapture;",
        "typedef struct RileyCudaGraph RileyCudaGraph;",
        "typedef struct RileyCudaGraphExec RileyCudaGraphExec;",
        "typedef struct RileyCudaGraphErrorInfo",
        "RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL",
        "RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN",
        "RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_SUPPORTED",
        "RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN",
        "RILEY_CUDA_GRAPH_STAGE_CLOSE",
    ] {
        assert!(
            header.contains(declaration),
            "missing graph ABI: {declaration}"
        );
    }
    assert!(header.contains("rather than a tail extension of RileyCudaErrorInfo"));
    assert!(!header.contains("riley_cuda_graph_capture_begin("));
    assert!(!ffi.contains("riley_cuda_graph_capture_begin"));
}

#[test]
fn graph_public_values_fix_the_cpu_only_contract() {
    assert_eq!(CudaGraphCaptureMode::ThreadLocal as u32, 1);
    assert_eq!(CudaGraphCaptureCapability::Unknown as u32, 0);
    assert_eq!(CudaGraphCaptureCapability::Unsupported as u32, 1);
    assert_eq!(CudaGraphCaptureCapability::Supported as u32, 2);
    assert!(!CudaGraphCaptureCapability::Unknown.admits_capture());
    assert!(!CudaGraphCaptureCapability::Unsupported.admits_capture());
    assert!(CudaGraphCaptureCapability::Supported.admits_capture());
    CudaGraphCaptureCapability::Supported
        .require_capture_admission("graph_cpu::supported")
        .unwrap();
    let unknown = CudaGraphCaptureCapability::Unknown
        .require_capture_admission("graph_cpu::unknown")
        .unwrap_err();
    assert_eq!(unknown.kind(), CudaErrorKind::NotSupported);
    assert_eq!(unknown.stage(), CudaErrorStage::Prepare);
    assert!(matches!(
        CudaGraphStage::CaptureAbort,
        CudaGraphStage::CaptureAbort
    ));
    assert_ne!(CudaGraphStage::Unknown(91), CudaGraphStage::Launch);

    let mut lifecycle = CudaGraphLifecycle::new();
    assert_eq!(lifecycle.state(), CudaGraphLifecycleState::Uninitialized);
    lifecycle.poison().unwrap();
    assert_eq!(lifecycle.state(), CudaGraphLifecycleState::Poisoned);
    let error = lifecycle.close().unwrap_err();
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    assert_eq!(error.domain(), CudaErrorDomain::Rust);
    assert_eq!(error.stage(), CudaErrorStage::Validation);
}

#[test]
fn feature_off_capture_stub_keeps_the_future_mutable_stream_borrow() {
    fn begin(stream: &mut CudaStream) -> CudaResult<GraphCapture<'_>> {
        stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)
    }

    // Naming the function type-checks the public feature-off surface without
    // constructing an unavailable CUDA stream or manufacturing a graph.
    let _ = begin;

    let graph_source = include_str!("../src/graph.rs");
    assert!(graph_source.contains("CudaError::unavailable(\"CudaStream::begin_graph_capture\")"));
    assert!(graph_source.contains("no CUDA Graph capture entry point yet"));
    for forbidden in ["riley_model", "riley_runtime", "riley_server", "llama"] {
        assert!(
            !graph_source.contains(forbidden),
            "graph contract must remain model/runtime independent: {forbidden}"
        );
    }
}
