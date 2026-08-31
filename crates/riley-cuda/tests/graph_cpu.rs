#![cfg(not(feature = "cuda"))]

use riley_cuda::{
    CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaGraphCaptureCapability,
    CudaGraphCaptureMode, CudaGraphLifecycle, CudaGraphLifecycleState, CudaGraphStage, CudaResult,
    CudaStream, GraphCapture,
};

#[test]
fn graph_contract_is_additive_and_declares_the_capture_begin_symbol() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");

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
    assert!(header.contains("riley_cuda_graph_capture_begin("));
    assert!(graph.contains("pub(crate) struct RawGraphErrorInfo"));
    assert!(graph.contains("pub(crate) fn decode_graph_failure_info"));
    assert_eq!(graph.matches("struct RawGraphErrorInfo").count(), 1);
    assert_eq!(graph.matches("fn decode_graph_failure_info").count(), 1);
    assert!(!ffi.contains("struct RawGraphErrorInfo"));
    assert!(ffi.contains("struct RawGraphCapture"));
    assert!(ffi.contains("riley_cuda_graph_capture_begin"));
    assert!(ffi.contains("graph_capture_begin_metadata_is_valid"));
}

#[test]
fn capture_begin_decodes_the_canonical_failure_companion_before_native_status() {
    let ffi = include_str!("../src/ffi.rs");
    let begin = ffi
        .split("pub(super) fn begin_graph_capture")
        .nth(1)
        .expect("FFI must retain the graph-capture begin boundary")
        .split("pub(super) fn wait_event")
        .next()
        .expect("graph-capture begin boundary must end before wait_event");
    let decode_position = begin
        .find("decode_graph_failure_info(&graph_error)?;")
        .expect("capture begin must decode its companion graph evidence");
    let status_position = begin
        .find("status_result(status, OPERATION, &error)?;")
        .expect("capture begin must preserve the native status boundary");

    assert!(
        decode_position < status_position,
        "capture begin must validate graph evidence before returning native status"
    );
    assert!(begin.contains("RawGraphErrorInfo::new()"));
    assert!(begin.contains("graph_capture_begin_metadata_is_valid(&graph_error, &graph_failure)"));
    assert!(ffi.contains("raw.struct_size() == RawGraphErrorInfo::ABI_SIZE"));
    assert!(ffi.contains("decoded.is_empty_capture_begin_attempt()"));
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
    assert!(graph_source.contains("self.native.begin_graph_capture(mode as u32)?;"));
    assert!(graph_source.contains("without an owned capture handle"));
    for forbidden in ["riley_model", "riley_runtime", "riley_server", "llama"] {
        assert!(
            !graph_source.contains(forbidden),
            "graph contract must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn graph_capture_is_a_type_level_exclusive_stream_lease() {
    let graph_source = include_str!("../src/graph.rs");
    let capture_source = graph_source
        .split("/// Borrowed graph-capture owner")
        .nth(1)
        .expect("graph contract must retain the GraphCapture owner declaration")
        .split("impl CudaStream")
        .next()
        .expect("GraphCapture owner declaration must precede CudaStream methods");

    for required in [
        "PhantomData<&'stream mut CudaStream>",
        "PhantomData<Rc<()>>",
        "cannot_query_stream_while_capturing",
        "cannot_synchronize_stream_while_capturing",
        "cannot_begin_a_batch_while_capturing",
        "cannot_close_stream_while_capturing",
        "assert_send::<riley_cuda::GraphCapture<'static>>();",
        "assert_sync::<riley_cuda::GraphCapture<'static>>();",
    ] {
        assert!(
            capture_source.contains(required),
            "GraphCapture must retain the exclusive-lease contract: {required}"
        );
    }
    assert_eq!(
        capture_source.matches("```compile_fail").count(),
        6,
        "GraphCapture must keep four stream-alias and two thread-transfer compile-fail examples"
    );
    for forbidden in ["RawGraphCapture", "ffi::", "unsafe", "*mut", "self.native"] {
        assert!(
            !capture_source.contains(forbidden),
            "C05-3 must remain a type-only owner contract without a native handle: {forbidden}"
        );
    }
}

#[test]
fn native_capture_begin_stub_is_wired_and_fails_closed_without_cuda_work() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let native = include_str!("../../../kernels/src/graph.cu");
    let cmake = include_str!("../../../kernels/CMakeLists.txt");
    let build_script = include_str!("../build.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    assert!(header.contains("out_capture is required and is null on every return"));
    assert!(native.contains("*out_capture = nullptr;"));
    assert!(
        native.contains("clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN)")
    );
    assert!(native.contains("RILEY_CUDA_STATUS_NOT_SUPPORTED"));
    assert!(native.contains("native CUDA Graph capture is not linked into this build"));
    assert!(native.contains("graph_error_reserved_is_zero"));
    assert!(cmake.contains("src/graph.cu"));
    assert!(build_script.contains("kernels_dir.join(\"src/graph.cu\")"));
    assert!(abi_layout.contains("graph_capture_begin_symbol"));

    for forbidden in [
        "cudaStreamBeginCapture",
        "CurrentContext",
        "command_batch",
        "stream->",
    ] {
        assert!(
            !native.contains(forbidden),
            "C05-1 native stub must not begin CUDA work or mutate stream ownership: {forbidden}"
        );
    }
}
