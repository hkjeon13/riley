#![cfg(not(feature = "cuda"))]

use riley_cuda::{
    CapturedGraph, CudaDeviceBuffer, CudaErrorDomain, CudaErrorKind, CudaErrorStage,
    CudaGraphCaptureCapability, CudaGraphCaptureMode, CudaGraphCaptureOperation,
    CudaGraphLifecycle, CudaGraphLifecycleState, CudaGraphStage, CudaPinnedHostBuffer, CudaResult,
    CudaStream, GraphCapture, GraphExec, GraphFillCapture, GraphLaunch,
    OwnedCapturedBf16ArgmaxGraph, OwnedCapturedBf16RowGatherArgmaxGraph,
    OwnedCapturedBf16RowGatherGraph, OwnedCapturedCanonicalRmsNormBf16Graph,
    OwnedCapturedGatedMultiplyBf16Graph, OwnedCapturedGraph, OwnedCapturedH2DGraph,
    OwnedCapturedResidualAddBf16Graph, OwnedCapturedSiluBf16Graph, OwnedGraphBf16ArgmaxCapture,
    OwnedGraphBf16ArgmaxCaptureBeginError, OwnedGraphBf16ArgmaxExec, OwnedGraphBf16ArgmaxLaunch,
    OwnedGraphBf16ArgmaxResources, OwnedGraphBf16RowGatherArgmaxCapture,
    OwnedGraphBf16RowGatherArgmaxCaptureBeginError, OwnedGraphBf16RowGatherArgmaxExec,
    OwnedGraphBf16RowGatherArgmaxLaunch, OwnedGraphBf16RowGatherArgmaxResources,
    OwnedGraphBf16RowGatherCapture, OwnedGraphBf16RowGatherCaptureBeginError,
    OwnedGraphBf16RowGatherExec, OwnedGraphBf16RowGatherLaunch, OwnedGraphBf16RowGatherResources,
    OwnedGraphCanonicalRmsNormBf16Capture, OwnedGraphCanonicalRmsNormBf16CaptureBeginError,
    OwnedGraphCanonicalRmsNormBf16Exec, OwnedGraphCanonicalRmsNormBf16Launch,
    OwnedGraphCanonicalRmsNormBf16Resources, OwnedGraphExec, OwnedGraphFillCapture,
    OwnedGraphFillCaptureBeginError, OwnedGraphFillResources, OwnedGraphGatedMultiplyBf16Capture,
    OwnedGraphGatedMultiplyBf16CaptureBeginError, OwnedGraphGatedMultiplyBf16Exec,
    OwnedGraphGatedMultiplyBf16Launch, OwnedGraphGatedMultiplyBf16Resources, OwnedGraphH2DCapture,
    OwnedGraphH2DCaptureBeginError, OwnedGraphH2DExec, OwnedGraphH2DLaunch, OwnedGraphH2DResources,
    OwnedGraphLaunch, OwnedGraphResidualAddBf16Capture, OwnedGraphResidualAddBf16CaptureBeginError,
    OwnedGraphResidualAddBf16Exec, OwnedGraphResidualAddBf16Launch,
    OwnedGraphResidualAddBf16Resources, OwnedGraphSiluBf16Capture,
    OwnedGraphSiluBf16CaptureBeginError, OwnedGraphSiluBf16Exec, OwnedGraphSiluBf16Launch,
    OwnedGraphSiluBf16Resources,
};

#[test]
fn graph_contract_is_additive_and_declares_the_capture_owner_symbols() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");

    for declaration in [
        "typedef struct RileyCudaGraphCapture RileyCudaGraphCapture;",
        "typedef struct RileyCudaGraph RileyCudaGraph;",
        "typedef struct RileyCudaGraphExec RileyCudaGraphExec;",
        "typedef struct RileyCudaGraphLaunch RileyCudaGraphLaunch;",
        "typedef struct RileyCudaGraphErrorInfo",
        "RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL",
        "RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN",
        "RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_SUPPORTED",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_SILU_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_GATED_MULTIPLY_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_RESIDUAL_ADD_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_CANONICAL_RMS_NORM_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ARGMAX",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ROW_GATHER",
        "RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN",
        "RILEY_CUDA_GRAPH_STAGE_CLOSE",
        "RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE",
    ] {
        assert!(
            header.contains(declaration),
            "missing graph ABI: {declaration}"
        );
    }
    assert!(header.contains("rather than a tail extension of RileyCudaErrorInfo"));
    assert!(header.contains("riley_cuda_graph_capture_begin("));
    assert!(header.contains("riley_cuda_graph_capture_abort("));
    assert!(header.contains("riley_cuda_graph_capture_query_capability("));
    for symbol in [
        "riley_cuda_graph_capture_query_capability",
        "riley_cuda_graph_capture_begin_fill_f32",
        "riley_cuda_graph_capture_enqueue_fill_f32",
        "riley_cuda_graph_capture_begin_h2d",
        "riley_cuda_graph_capture_enqueue_h2d",
        "riley_cuda_graph_capture_begin_silu_bf16",
        "riley_cuda_graph_capture_enqueue_silu_bf16",
        "riley_cuda_graph_capture_begin_gated_multiply_bf16",
        "riley_cuda_graph_capture_enqueue_gated_multiply_bf16",
        "riley_cuda_graph_capture_begin_residual_add_bf16",
        "riley_cuda_graph_capture_enqueue_residual_add_bf16",
        "riley_cuda_graph_capture_begin_canonical_rms_norm_bf16",
        "riley_cuda_graph_capture_enqueue_canonical_rms_norm_bf16",
        "riley_cuda_graph_capture_begin_bf16_argmax",
        "riley_cuda_graph_capture_enqueue_bf16_argmax",
        "riley_cuda_graph_capture_begin_bf16_row_gather",
        "riley_cuda_graph_capture_enqueue_bf16_row_gather",
        "riley_cuda_graph_capture_end",
        "riley_cuda_graph_instantiate",
        "riley_cuda_graph_exec_launch",
        "riley_cuda_graph_exec_stage_h2d_source",
        "riley_cuda_graph_launch_complete",
        "riley_cuda_graph_close",
        "riley_cuda_graph_exec_close",
    ] {
        assert!(
            header.contains(symbol),
            "missing additive C05-5 graph ABI: {symbol}"
        );
        assert!(
            ffi.contains(symbol),
            "missing Rust FFI binding for C05-5 graph ABI: {symbol}"
        );
    }
    assert!(graph.contains("pub(crate) struct RawGraphErrorInfo"));
    assert!(graph.contains("pub(crate) fn decode_graph_failure_info"));
    assert_eq!(graph.matches("struct RawGraphErrorInfo").count(), 1);
    assert_eq!(graph.matches("fn decode_graph_failure_info").count(), 1);
    assert!(!ffi.contains("struct RawGraphErrorInfo"));
    assert!(ffi.contains("struct RawGraphCapture"));
    assert!(ffi.contains("riley_cuda_graph_capture_begin"));
    assert!(ffi.contains("riley_cuda_graph_capture_abort"));
    assert!(ffi.contains("pub(super) fn graph_capture_capability"));
    assert!(ffi.contains("GraphCaptureHandle"));
    assert!(ffi.contains("graph_capture_begin_success_metadata_is_valid"));
    assert!(ffi.contains("graph_capture_abort_metadata_is_valid"));
}

#[test]
fn capture_capability_query_is_pure_and_fail_closed() {
    let native_graph = include_str!("../../../kernels/src/graph.cu");
    let query = native_graph
        .split("extern \"C\" RileyCudaStatus riley_cuda_graph_capture_query_capability")
        .nth(1)
        .expect("native graph capability query must remain exported")
        .split("extern \"C\" RileyCudaStatus riley_cuda_graph_capture_begin")
        .next()
        .expect("capability query must end before graph capture begins");

    assert!(query.contains("clear_error(error);"));
    assert!(query.contains("*out_capability = RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN;"));
    assert!(query.contains("if (out_capability == nullptr)"));
    assert!(query.contains("RILEY_CUDA_STATUS_INVALID_ARGUMENT"));
    for operation in [
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_FILL_F32",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_H2D",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_SILU_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_GATED_MULTIPLY_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_RESIDUAL_ADD_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_CANONICAL_RMS_NORM_BF16",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ARGMAX",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ROW_GATHER",
    ] {
        assert!(
            query.contains(operation),
            "reviewed operation missing from native capability query: {operation}"
        );
    }
    assert!(query.contains("RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_UNKNOWN"));
    assert!(query.contains("default:"));
    assert!(query.contains("RILEY_CUDA_STATUS_SUCCESS"));

    // It is a vocabulary lookup only. Capturing, creating a CUDA context, or
    // allocating here would make the no-device ABI guarantee untrue.
    for forbidden in [
        "cudaStream",
        "cudaMalloc",
        "cudaFree",
        "cuCtx",
        "CurrentContext",
        "try_acquire",
        "std::calloc",
        "std::malloc",
    ] {
        assert!(
            !query.contains(forbidden),
            "capability query must not perform CUDA/resource work: {forbidden}"
        );
    }
}

#[test]
fn deferred_close_abi_is_additive_for_every_safe_wrapper_owner() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");

    // These are deliberately new opt-in entry points. The long-standing raw
    // `*_close` APIs retain their retry behavior; only a safe wrapper that
    // knows it is being dropped during a capture may transfer ownership.
    for symbol in [
        "riley_cuda_context_defer_to_active_capture",
        "riley_cuda_stream_defer_to_active_capture",
        "riley_cuda_event_defer_to_active_capture",
        "riley_cuda_device_buffer_defer_to_active_capture",
        "riley_cuda_pinned_host_buffer_defer_to_active_capture",
        "riley_cuda_hf_prefill_attention_plan_defer_to_active_capture",
        "riley_cuda_gemm_plan_defer_to_active_capture",
    ] {
        assert!(
            header.contains(symbol),
            "missing additive deferred-close ABI declaration: {symbol}"
        );
        assert!(
            ffi.contains(symbol),
            "missing Rust FFI binding for deferred-close ABI: {symbol}"
        );
    }
    assert!(ffi.contains("has_active_graph_capture()"));
}

#[test]
fn capture_begin_decodes_and_recovers_a_non_null_owner_before_returning_error() {
    let ffi = include_str!("../src/ffi.rs");
    let begin = ffi
        .split("pub(super) fn begin_graph_capture")
        .nth(1)
        .expect("FFI must retain the graph-capture begin boundary")
        .split("pub(super) fn wait_event")
        .next()
        .expect("graph-capture begin boundary must end before wait_event");
    let decode_position = begin
        .find("let decoded = decode_graph_failure_info(&graph_error);")
        .expect("capture begin must decode its companion graph evidence");
    let cleanup_position = begin
        .find("let cleanup = pointer.map")
        .expect("capture begin must recover a non-null capture owner");
    let status_position = begin
        .find("status_result(status, OPERATION, &error)")
        .expect("capture begin must preserve the native status boundary");

    assert!(
        decode_position < cleanup_position && cleanup_position < status_position,
        "capture begin must validate graph evidence and recover ownership before returning native status"
    );
    assert!(begin.contains("RawGraphErrorInfo::new()"));
    assert!(begin.contains("owner.abort()"));
    assert!(ffi.contains("decoded.capture_id().is_some()"));
    assert!(ffi.contains("matches!(decoded.stage(), Some(CudaGraphStage::CaptureBegin))"));
}

#[test]
fn successful_abort_requires_nonpoisoned_release_metadata() {
    let ffi = include_str!("../src/ffi.rs");
    let abort_validation = ffi
        .split("fn graph_capture_abort_metadata_is_valid")
        .nth(1)
        .expect("graph-abort metadata validation must remain present")
        .split("#[cfg(feature = \"nvml\")]")
        .next()
        .expect("graph-abort metadata validation must end before NVML helpers");
    assert!(
        abort_validation.contains("decoded.resource_release_known() && !decoded.poisoned()"),
        "a successful graph abort must reject poisoned companion metadata"
    );
}

#[test]
fn graph_public_values_fix_the_cpu_only_contract() {
    assert_eq!(CudaGraphCaptureMode::ThreadLocal as u32, 1);
    assert_eq!(CudaGraphCaptureOperation::FillF32 as u32, 1);
    assert_eq!(CudaGraphCaptureOperation::H2D as u32, 2);
    assert_eq!(CudaGraphCaptureOperation::SiluBf16 as u32, 3);
    assert_eq!(CudaGraphCaptureOperation::GatedMultiplyBf16 as u32, 4);
    assert_eq!(CudaGraphCaptureOperation::ResidualAddBf16 as u32, 5);
    assert_eq!(CudaGraphCaptureOperation::CanonicalRmsNormBf16 as u32, 6);
    assert_eq!(CudaGraphCaptureOperation::Bf16Argmax as u32, 7);
    assert_eq!(CudaGraphCaptureOperation::Bf16RowGather as u32, 8);
    assert_eq!(CudaGraphCaptureOperation::Bf16RowGatherArgmax as u32, 9);
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
    ] {
        assert_eq!(
            operation.capture_capability().unwrap_err().kind(),
            CudaErrorKind::Unavailable,
            "feature-off capability lookup must not fabricate native admission"
        );
    }
    assert!(matches!(
        CudaGraphStage::CaptureAbort,
        CudaGraphStage::CaptureAbort
    ));
    assert!(matches!(
        CudaGraphStage::InputStage,
        CudaGraphStage::InputStage
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

    fn begin_fill<'stream, 'buffer>(
        stream: &'stream mut CudaStream,
        buffer: &'buffer mut CudaDeviceBuffer,
    ) -> CudaResult<GraphFillCapture<'stream, 'buffer>> {
        stream.begin_graph_fill_capture(buffer, 16, CudaGraphCaptureMode::ThreadLocal)
    }

    fn end_fill<'stream, 'buffer>(
        capture: GraphFillCapture<'stream, 'buffer>,
    ) -> CudaResult<CapturedGraph<'stream, 'buffer>> {
        capture.end()
    }

    fn instantiate_fill<'stream, 'buffer>(
        graph: CapturedGraph<'stream, 'buffer>,
    ) -> CudaResult<GraphExec<'stream, 'buffer>> {
        graph.instantiate()
    }

    fn launch_fill<'exec, 'stream, 'buffer>(
        exec: &'exec mut GraphExec<'stream, 'buffer>,
    ) -> CudaResult<GraphLaunch<'exec, 'stream, 'buffer>> {
        exec.launch()
    }

    fn finish_fill<'exec, 'stream, 'buffer>(
        launch: GraphLaunch<'exec, 'stream, 'buffer>,
    ) -> CudaResult<()> {
        launch.finish()
    }

    fn close_exec<'stream, 'buffer>(exec: GraphExec<'stream, 'buffer>) -> CudaResult<()> {
        exec.close()
    }

    fn begin_owned_fill(
        stream: CudaStream,
        buffer: CudaDeviceBuffer,
    ) -> Result<OwnedGraphFillCapture, OwnedGraphFillCaptureBeginError> {
        stream.begin_owned_graph_fill_capture(buffer, 16, CudaGraphCaptureMode::ThreadLocal)
    }

    fn end_owned_fill(capture: OwnedGraphFillCapture) -> CudaResult<OwnedCapturedGraph> {
        capture.end()
    }

    fn instantiate_owned_fill(graph: OwnedCapturedGraph) -> CudaResult<OwnedGraphExec> {
        graph.instantiate()
    }

    fn launch_owned_fill(exec: &mut OwnedGraphExec) -> CudaResult<OwnedGraphLaunch<'_>> {
        exec.launch()
    }

    fn finish_owned_fill(launch: OwnedGraphLaunch<'_>) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_exec(exec: OwnedGraphExec) -> CudaResult<OwnedGraphFillResources> {
        exec.close()
    }

    fn begin_owned_h2d(
        stream: CudaStream,
        source: CudaPinnedHostBuffer,
        destination: CudaDeviceBuffer,
    ) -> Result<OwnedGraphH2DCapture, OwnedGraphH2DCaptureBeginError> {
        stream.begin_owned_graph_h2d_capture(source, destination, CudaGraphCaptureMode::ThreadLocal)
    }

    fn end_owned_h2d(capture: OwnedGraphH2DCapture) -> CudaResult<OwnedCapturedH2DGraph> {
        capture.end()
    }

    fn instantiate_owned_h2d(graph: OwnedCapturedH2DGraph) -> CudaResult<OwnedGraphH2DExec> {
        graph.instantiate()
    }

    fn launch_owned_h2d<'exec>(
        exec: &'exec mut OwnedGraphH2DExec,
        bytes: &[u8],
    ) -> CudaResult<OwnedGraphH2DLaunch<'exec>> {
        exec.launch_with_source(bytes)
    }

    fn finish_owned_h2d(launch: OwnedGraphH2DLaunch<'_>) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_h2d(exec: OwnedGraphH2DExec) -> CudaResult<OwnedGraphH2DResources> {
        exec.close()
    }

    fn begin_owned_silu_bf16(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Result<OwnedGraphSiluBf16Capture, OwnedGraphSiluBf16CaptureBeginError> {
        stream.begin_owned_graph_silu_bf16_capture(
            input,
            output,
            16,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_silu_bf16(
        capture: OwnedGraphSiluBf16Capture,
    ) -> CudaResult<OwnedCapturedSiluBf16Graph> {
        capture.end()
    }

    fn instantiate_owned_silu_bf16(
        graph: OwnedCapturedSiluBf16Graph,
    ) -> CudaResult<OwnedGraphSiluBf16Exec> {
        graph.instantiate()
    }

    fn launch_owned_silu_bf16(
        exec: &mut OwnedGraphSiluBf16Exec,
    ) -> CudaResult<OwnedGraphSiluBf16Launch<'_>> {
        exec.launch()
    }

    fn finish_owned_silu_bf16(launch: OwnedGraphSiluBf16Launch<'_>) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_silu_bf16(
        exec: OwnedGraphSiluBf16Exec,
    ) -> CudaResult<OwnedGraphSiluBf16Resources> {
        exec.close()
    }

    fn begin_owned_gated_multiply_bf16(
        stream: CudaStream,
        activated_gate: CudaDeviceBuffer,
        up: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Result<OwnedGraphGatedMultiplyBf16Capture, OwnedGraphGatedMultiplyBf16CaptureBeginError>
    {
        stream.begin_owned_graph_gated_multiply_bf16_capture(
            activated_gate,
            up,
            output,
            16,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_gated_multiply_bf16(
        capture: OwnedGraphGatedMultiplyBf16Capture,
    ) -> CudaResult<OwnedCapturedGatedMultiplyBf16Graph> {
        capture.end()
    }

    fn instantiate_owned_gated_multiply_bf16(
        graph: OwnedCapturedGatedMultiplyBf16Graph,
    ) -> CudaResult<OwnedGraphGatedMultiplyBf16Exec> {
        graph.instantiate()
    }

    fn launch_owned_gated_multiply_bf16(
        exec: &mut OwnedGraphGatedMultiplyBf16Exec,
    ) -> CudaResult<OwnedGraphGatedMultiplyBf16Launch<'_>> {
        exec.launch()
    }

    fn finish_owned_gated_multiply_bf16(
        launch: OwnedGraphGatedMultiplyBf16Launch<'_>,
    ) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_gated_multiply_bf16(
        exec: OwnedGraphGatedMultiplyBf16Exec,
    ) -> CudaResult<OwnedGraphGatedMultiplyBf16Resources> {
        exec.close()
    }

    fn begin_owned_residual_add_bf16(
        stream: CudaStream,
        left: CudaDeviceBuffer,
        right: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Result<OwnedGraphResidualAddBf16Capture, OwnedGraphResidualAddBf16CaptureBeginError> {
        stream.begin_owned_graph_residual_add_bf16_capture(
            left,
            right,
            output,
            16,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_residual_add_bf16(
        capture: OwnedGraphResidualAddBf16Capture,
    ) -> CudaResult<OwnedCapturedResidualAddBf16Graph> {
        capture.end()
    }

    fn instantiate_owned_residual_add_bf16(
        graph: OwnedCapturedResidualAddBf16Graph,
    ) -> CudaResult<OwnedGraphResidualAddBf16Exec> {
        graph.instantiate()
    }

    fn launch_owned_residual_add_bf16(
        exec: &mut OwnedGraphResidualAddBf16Exec,
    ) -> CudaResult<OwnedGraphResidualAddBf16Launch<'_>> {
        exec.launch()
    }

    fn finish_owned_residual_add_bf16(
        launch: OwnedGraphResidualAddBf16Launch<'_>,
    ) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_residual_add_bf16(
        exec: OwnedGraphResidualAddBf16Exec,
    ) -> CudaResult<OwnedGraphResidualAddBf16Resources> {
        exec.close()
    }

    fn begin_owned_canonical_rms_norm_bf16(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        weight: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Result<
        OwnedGraphCanonicalRmsNormBf16Capture,
        OwnedGraphCanonicalRmsNormBf16CaptureBeginError,
    > {
        stream.begin_owned_graph_canonical_rms_norm_bf16_capture(
            input,
            weight,
            output,
            4,
            16,
            1.0e-5,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_canonical_rms_norm_bf16(
        capture: OwnedGraphCanonicalRmsNormBf16Capture,
    ) -> CudaResult<OwnedCapturedCanonicalRmsNormBf16Graph> {
        capture.end()
    }

    fn instantiate_owned_canonical_rms_norm_bf16(
        graph: OwnedCapturedCanonicalRmsNormBf16Graph,
    ) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Exec> {
        graph.instantiate()
    }

    fn launch_owned_canonical_rms_norm_bf16(
        exec: &mut OwnedGraphCanonicalRmsNormBf16Exec,
    ) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Launch<'_>> {
        exec.launch()
    }

    fn finish_owned_canonical_rms_norm_bf16(
        launch: OwnedGraphCanonicalRmsNormBf16Launch<'_>,
    ) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_canonical_rms_norm_bf16(
        exec: OwnedGraphCanonicalRmsNormBf16Exec,
    ) -> CudaResult<OwnedGraphCanonicalRmsNormBf16Resources> {
        exec.close()
    }

    fn begin_owned_bf16_argmax(
        stream: CudaStream,
        logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
    ) -> Result<OwnedGraphBf16ArgmaxCapture, OwnedGraphBf16ArgmaxCaptureBeginError> {
        stream.begin_owned_graph_bf16_argmax_capture(
            logits,
            results,
            4,
            257,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_bf16_argmax(
        capture: OwnedGraphBf16ArgmaxCapture,
    ) -> CudaResult<OwnedCapturedBf16ArgmaxGraph> {
        capture.end()
    }

    fn instantiate_owned_bf16_argmax(
        graph: OwnedCapturedBf16ArgmaxGraph,
    ) -> CudaResult<OwnedGraphBf16ArgmaxExec> {
        graph.instantiate()
    }

    fn launch_owned_bf16_argmax(
        exec: &mut OwnedGraphBf16ArgmaxExec,
    ) -> CudaResult<OwnedGraphBf16ArgmaxLaunch<'_>> {
        exec.launch()
    }

    fn finish_owned_bf16_argmax(launch: OwnedGraphBf16ArgmaxLaunch<'_>) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_bf16_argmax(
        exec: OwnedGraphBf16ArgmaxExec,
    ) -> CudaResult<OwnedGraphBf16ArgmaxResources> {
        exec.close()
    }

    fn begin_owned_bf16_row_gather(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        output: CudaDeviceBuffer,
    ) -> Result<OwnedGraphBf16RowGatherCapture, OwnedGraphBf16RowGatherCaptureBeginError> {
        // The safe capture owner must validate this temporary host mirror at
        // admission, rather than retaining a borrow into its replay lifetime.
        let row_indices_host = [3_u32, 1, 0, 2];
        stream.begin_owned_graph_bf16_row_gather_capture(
            input,
            row_indices,
            output,
            &row_indices_host,
            4,
            16,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_bf16_row_gather(
        capture: OwnedGraphBf16RowGatherCapture,
    ) -> CudaResult<OwnedCapturedBf16RowGatherGraph> {
        capture.end()
    }

    fn instantiate_owned_bf16_row_gather(
        graph: OwnedCapturedBf16RowGatherGraph,
    ) -> CudaResult<OwnedGraphBf16RowGatherExec> {
        graph.instantiate()
    }

    fn launch_owned_bf16_row_gather(
        exec: &mut OwnedGraphBf16RowGatherExec,
    ) -> CudaResult<OwnedGraphBf16RowGatherLaunch<'_>> {
        exec.launch()
    }

    fn finish_owned_bf16_row_gather(launch: OwnedGraphBf16RowGatherLaunch<'_>) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_bf16_row_gather(
        exec: OwnedGraphBf16RowGatherExec,
    ) -> CudaResult<OwnedGraphBf16RowGatherResources> {
        exec.close()
    }

    fn begin_owned_bf16_row_gather_argmax(
        stream: CudaStream,
        input: CudaDeviceBuffer,
        row_indices: CudaDeviceBuffer,
        gathered_logits: CudaDeviceBuffer,
        results: CudaDeviceBuffer,
    ) -> Result<OwnedGraphBf16RowGatherArgmaxCapture, OwnedGraphBf16RowGatherArgmaxCaptureBeginError>
    {
        // The safe composite owner derives its output count from this
        // temporary eager-compatible mirror, and must not retain it.
        let row_indices_host = [3_u32, 1, 0, 2];
        stream.begin_owned_graph_bf16_row_gather_argmax_capture(
            input,
            row_indices,
            gathered_logits,
            results,
            &row_indices_host,
            4,
            16,
            CudaGraphCaptureMode::ThreadLocal,
        )
    }

    fn end_owned_bf16_row_gather_argmax(
        capture: OwnedGraphBf16RowGatherArgmaxCapture,
    ) -> CudaResult<OwnedCapturedBf16RowGatherArgmaxGraph> {
        capture.end()
    }

    fn instantiate_owned_bf16_row_gather_argmax(
        graph: OwnedCapturedBf16RowGatherArgmaxGraph,
    ) -> CudaResult<OwnedGraphBf16RowGatherArgmaxExec> {
        graph.instantiate()
    }

    fn launch_owned_bf16_row_gather_argmax(
        exec: &mut OwnedGraphBf16RowGatherArgmaxExec,
    ) -> CudaResult<OwnedGraphBf16RowGatherArgmaxLaunch<'_>> {
        exec.launch()
    }

    fn finish_owned_bf16_row_gather_argmax(
        launch: OwnedGraphBf16RowGatherArgmaxLaunch<'_>,
    ) -> CudaResult<()> {
        launch.finish()
    }

    fn close_owned_bf16_row_gather_argmax(
        exec: OwnedGraphBf16RowGatherArgmaxExec,
    ) -> CudaResult<OwnedGraphBf16RowGatherArgmaxResources> {
        exec.close()
    }

    let _ = (
        begin_fill,
        end_fill,
        instantiate_fill,
        launch_fill,
        finish_fill,
        close_exec,
        begin_owned_fill,
        end_owned_fill,
        instantiate_owned_fill,
        launch_owned_fill,
        finish_owned_fill,
        close_owned_exec,
        begin_owned_h2d,
        end_owned_h2d,
        instantiate_owned_h2d,
        launch_owned_h2d,
        finish_owned_h2d,
        close_owned_h2d,
        begin_owned_silu_bf16,
        end_owned_silu_bf16,
        instantiate_owned_silu_bf16,
        launch_owned_silu_bf16,
        finish_owned_silu_bf16,
        close_owned_silu_bf16,
        begin_owned_gated_multiply_bf16,
        end_owned_gated_multiply_bf16,
        instantiate_owned_gated_multiply_bf16,
        launch_owned_gated_multiply_bf16,
        finish_owned_gated_multiply_bf16,
        close_owned_gated_multiply_bf16,
        begin_owned_residual_add_bf16,
        end_owned_residual_add_bf16,
        instantiate_owned_residual_add_bf16,
        launch_owned_residual_add_bf16,
        finish_owned_residual_add_bf16,
        close_owned_residual_add_bf16,
        begin_owned_canonical_rms_norm_bf16,
        end_owned_canonical_rms_norm_bf16,
        instantiate_owned_canonical_rms_norm_bf16,
        launch_owned_canonical_rms_norm_bf16,
        finish_owned_canonical_rms_norm_bf16,
        close_owned_canonical_rms_norm_bf16,
        begin_owned_bf16_argmax,
        end_owned_bf16_argmax,
        instantiate_owned_bf16_argmax,
        launch_owned_bf16_argmax,
        finish_owned_bf16_argmax,
        close_owned_bf16_argmax,
        begin_owned_bf16_row_gather,
        end_owned_bf16_row_gather,
        instantiate_owned_bf16_row_gather,
        launch_owned_bf16_row_gather,
        finish_owned_bf16_row_gather,
        close_owned_bf16_row_gather,
        begin_owned_bf16_row_gather_argmax,
        end_owned_bf16_row_gather_argmax,
        instantiate_owned_bf16_row_gather_argmax,
        launch_owned_bf16_row_gather_argmax,
        finish_owned_bf16_row_gather_argmax,
        close_owned_bf16_row_gather_argmax,
    );

    let graph_source = include_str!("../src/graph.rs");
    assert!(graph_source.contains("CudaError::unavailable(\"CudaStream::begin_graph_capture\")"));
    assert!(graph_source.contains("\"CudaStream::begin_graph_fill_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_fill_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_h2d_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_silu_bf16_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_gated_multiply_bf16_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_residual_add_bf16_capture\""));
    assert!(
        graph_source.contains("\"CudaStream::begin_owned_graph_canonical_rms_norm_bf16_capture\"")
    );
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_bf16_argmax_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_bf16_row_gather_capture\""));
    assert!(
        graph_source.contains("\"CudaStream::begin_owned_graph_bf16_row_gather_argmax_capture\"")
    );
    assert!(graph_source.contains("self.native.begin_graph_capture(mode as u32)?;"));
    assert!(graph_source.contains("native capture handle"));
    for forbidden in ["riley_model", "riley_runtime", "riley_server", "llama"] {
        assert!(
            !graph_source.contains(forbidden),
            "graph contract must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn graph_capture_is_a_real_type_level_exclusive_stream_lease() {
    let graph_source = include_str!("../src/graph.rs");
    let capture_source = graph_source
        .split("/// Borrowed owner of one active thread-local CUDA Graph capture")
        .nth(1)
        .expect("graph contract must retain the GraphCapture owner declaration")
        .split("/// Borrowed owner of the C05-5 fixed-address f32 fill capture")
        .next()
        .expect("GraphCapture owner declaration must precede C05-5 graph owners");

    for required in [
        "stream: &'stream mut CudaStream",
        "native: crate::ffi::GraphCaptureHandle",
        "active: bool",
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
    for forbidden in ["RawGraphCapture", "unsafe", "*mut"] {
        assert!(
            !capture_source.contains(forbidden),
            "GraphCapture must keep raw native pointers outside the safe owner: {forbidden}"
        );
    }
}

#[test]
fn fill_graph_owners_keep_stream_buffer_and_launch_close_ordering_in_safe_types() {
    let graph_source = include_str!("../src/graph.rs");

    for required in [
        "pub struct GraphFillCapture<'stream, 'buffer>",
        "pub struct CapturedGraph<'stream, 'buffer>",
        "pub struct GraphExec<'stream, 'buffer>",
        "pub struct GraphLaunch<'exec, 'stream, 'buffer>",
        "stream: Option<&'stream mut CudaStream>",
        "buffer: Option<&'buffer mut CudaDeviceBuffer>",
        "exec: &'exec mut GraphExec<'stream, 'buffer>",
        "cannot_use_the_capture_stream",
        "cannot_close_the_capture_buffer",
        "cannot_reuse_resources_while_graph_is_live",
        "cannot_close_or_relaunch_an_exec",
        "assert_send::<riley_cuda::GraphFillCapture<'static, 'static>>();",
        "assert_sync::<riley_cuda::GraphExec<'static, 'static>>();",
        "assert_send::<riley_cuda::GraphLaunch<'static, 'static, 'static>>();",
    ] {
        assert!(
            graph_source.contains(required),
            "C05-5 safe lifetime/close-ordering contract is missing: {required}"
        );
    }
}

#[test]
fn owned_fill_graph_owners_move_cold_resources_and_fail_closed_on_uncertain_replay() {
    let graph = include_str!("../src/graph.rs");
    let ffi = include_str!("../src/ffi.rs");

    for required in [
        "pub struct OwnedGraphFillResources",
        "pub struct OwnedGraphFillCaptureBeginError",
        "pub struct OwnedGraphFillCapture",
        "pub struct OwnedCapturedGraph",
        "pub struct OwnedGraphExec",
        "pub struct OwnedGraphLaunch<'exec>",
        "pub fn begin_owned_graph_fill_capture",
        "pub fn into_parts(self) -> (CudaStream, CudaDeviceBuffer)",
        "pub fn close(mut self) -> CudaResult<OwnedGraphFillResources>",
        "exec: &'exec mut OwnedGraphExec",
        "terminal: bool",
        "OwnedGraphFillCaptureBeginError::recoverable",
        "OwnedGraphFillCaptureBeginError::terminal",
    ] {
        assert!(
            graph.contains(required),
            "C05-6 owned graph contract is missing: {required}"
        );
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_fill_capture")
        .nth(1)
        .expect("owned graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned graph capture entry point must end before borrowed capture");
    assert_precedes(
        owned_begin,
        "validate_graph_fill_capture_preflight",
        "resources.stream.native.begin_graph_fill_capture",
        "C05-6 owned capture preflight",
    );
    assert!(
        owned_begin.contains("OwnedGraphFillCaptureBeginError::recoverable"),
        "native-unentered owned begin errors must return the moved resource pair"
    );
    assert!(
        owned_begin.contains("OwnedGraphFillCaptureBeginError::terminal"),
        "native-entered owned begin errors must withhold reusable resources"
    );

    for owner in [
        "pub struct OwnedGraphFillCapture {",
        "pub struct OwnedCapturedGraph {",
        "pub struct OwnedGraphExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain its native handle first"));
        let stream = source
            .find("stream: Option<CudaStream>")
            .unwrap_or_else(|| panic!("{owner} must retain its stream by value"));
        let buffer = source
            .find("buffer: Option<CudaDeviceBuffer>")
            .unwrap_or_else(|| panic!("{owner} must retain its buffer by value"));
        assert!(
            native < stream && stream < buffer,
            "{owner} must drop native graph ownership before its backing resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphExec")
        .nth(1)
        .expect("owned graph exec implementation must remain present")
        .split("/// Completion owner for one replay")
        .next()
        .expect("owned graph exec implementation must end before completion owner");
    assert!(
        owned_exec.contains("self.terminal = true;") && owned_exec.contains("if self.terminal"),
        "any owned replay uncertainty must make resource-returning close terminal"
    );

    for close in ["impl GraphHandle", "impl GraphExecHandle"] {
        let source = ffi
            .split(close)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {close}"))
            .split("impl Drop")
            .next()
            .expect("graph close implementation must end before Drop");
        assert!(
            source.contains("status == STATUS_SUCCESS")
                && source.contains("!graph_resources_released(&graph_failure)"),
            "{close} must reject successful close metadata that cannot prove resource release"
        );
        assert!(
            source.contains("graph_failure.submission_started()")
                && source.contains("graph_failure.completion_known()"),
            "{close} must reject success metadata that still claims an in-flight launch"
        );
    }
    let completion = ffi
        .split("impl GraphLaunchHandle")
        .nth(1)
        .expect("missing graph launch completion owner")
        .split("impl Drop for GraphLaunchHandle")
        .next()
        .expect("graph launch completion implementation must end before Drop");
    assert!(
        completion.contains("!graph_failure.submission_started()")
            && completion.contains("!graph_failure.completion_known()")
            && completion.contains("!graph_resources_released(&graph_failure)"),
        "completion success must prove submission, settled work, and retained graph resource state"
    );
}

#[test]
fn owned_h2d_graph_owners_require_fresh_exact_payloads_and_recover_three_resources() {
    let graph = include_str!("../src/graph.rs");
    let ffi = include_str!("../src/ffi.rs");
    let native = include_str!("../../../kernels/src/graph.cu");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");

    for required in [
        "pub struct OwnedGraphH2DResources",
        "pub struct OwnedGraphH2DCaptureBeginError",
        "pub struct OwnedGraphH2DCapture",
        "pub struct OwnedCapturedH2DGraph",
        "pub struct OwnedGraphH2DExec",
        "pub struct OwnedGraphH2DLaunch<'exec>",
        "pub fn begin_owned_graph_h2d_capture",
        "pub fn enqueue_h2d(&mut self)",
        "pub fn launch_with_source<'exec>",
        "pub fn into_parts(self) -> (CudaStream, CudaPinnedHostBuffer, CudaDeviceBuffer)",
        "pub fn close(mut self) -> CudaResult<OwnedGraphH2DResources>",
        "OwnedGraphH2DCaptureBeginError::recoverable",
        "OwnedGraphH2DCaptureBeginError::terminal",
    ] {
        assert!(
            graph.contains(required),
            "C05-7 owned H2D graph contract is missing: {required}"
        );
    }

    for owner in [
        "pub struct OwnedGraphH2DCapture {",
        "pub struct OwnedCapturedH2DGraph {",
        "pub struct OwnedGraphH2DExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let stream = source
            .find("stream: Option<CudaStream>")
            .unwrap_or_else(|| panic!("{owner} must retain its stream by value"));
        let pinned_source = source
            .find("source: Option<CudaPinnedHostBuffer>")
            .unwrap_or_else(|| panic!("{owner} must retain its pinned source by value"));
        let destination = source
            .find("destination: Option<CudaDeviceBuffer>")
            .unwrap_or_else(|| panic!("{owner} must retain its destination by value"));
        assert!(
            native < stream && stream < pinned_source && pinned_source < destination,
            "{owner} must drop native graph ownership before every captured resource"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_h2d_capture")
        .nth(1)
        .expect("owned H2D graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned H2D graph capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_h2d_capture_preflight",
        "resources.stream.native.begin_graph_h2d_capture",
        "C05-7 owned H2D capture preflight",
    );
    assert!(owned_begin.contains("OwnedGraphH2DCaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphH2DCaptureBeginError::terminal"));

    let owned_exec = graph
        .split("impl OwnedGraphH2DExec")
        .nth(1)
        .expect("owned H2D graph exec implementation must remain present")
        .split("/// Completion owner for one [`OwnedGraphH2DExec`] replay")
        .next()
        .expect("owned H2D graph exec must end before completion owner");
    assert!(owned_exec.contains("pub fn launch_with_source<'exec>"));
    assert!(
        !owned_exec.contains("pub fn launch("),
        "C05-7 must not expose a safe bare graph H2D replay"
    );
    assert_precedes(
        owned_exec,
        "self.native.stage_h2d_source",
        "self.native.launch",
        "C05-7 payload staging",
    );
    assert!(owned_exec.contains("actual_byte_len != expected_byte_len"));
    assert!(owned_exec.contains("self.terminal = true;"));

    for required in [
        "kH2D = 2",
        "h2d_source",
        "h2d_byte_len",
        "h2d_enqueue_count",
        "h2d_source_lease_held",
        "h2d_input_staged",
    ] {
        assert!(
            internal.contains(required),
            "C05-7 native ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_h2d_leases"));
    assert!(native.contains("release_graph_h2d_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_h2d_impl(")
        .nth(1)
        .expect("H2D capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("H2D capture admission helper must end before native exports");
    assert_precedes(
        begin,
        "try_acquire_exclusive_use(source->active_uses)",
        "cudaStreamBeginCapture(stream->stream",
        "C05-7 H2D source lease admission",
    );
    assert_precedes(
        begin,
        "try_acquire_exclusive_use(destination->active_uses)",
        "cudaStreamBeginCapture(stream->stream",
        "C05-7 H2D destination lease admission",
    );
    assert_precedes(
        begin,
        "try_acquire_exclusive_use(stream->active_uses)",
        "cudaStreamBeginCapture(stream->stream",
        "C05-7 H2D stream lease admission",
    );
    assert!(begin.contains("source->byte_len != destination->byte_len"));
    assert!(begin.contains("source->byte_len == 0"));

    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_h2d");
    assert!(enqueue.contains("cudaMemcpyAsync"));
    assert!(enqueue.contains("cudaMemcpyHostToDevice"));
    assert!(enqueue.contains("owner->h2d_enqueue_count != 0"));
    assert!(
        !enqueue.contains("std::calloc") && !enqueue.contains("std::free"),
        "the sole captured H2D node must not allocate or free host bookkeeping"
    );

    let stage = native_export_body(native, "riley_cuda_graph_exec_stage_h2d_source");
    assert!(stage.contains("RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE"));
    assert!(stage.contains("byte_len != exec->h2d_byte_len"));
    assert!(stage.contains("exec->launch_in_flight || exec->h2d_input_staged"));
    assert!(stage.contains("std::memmove(source->host_data, bytes"));
    assert!(stage.contains("exec->h2d_input_staged = true;"));
    assert!(
        !stage.contains("cudaGraphLaunch") && !stage.contains("cudaMemcpyAsync"),
        "private H2D staging must not expose a second CUDA submission path"
    );

    let launch = native_export_body(native, "riley_cuda_graph_exec_launch");
    assert!(launch.contains("!exec->h2d_input_staged"));
    assert!(launch.contains("exec->h2d_input_staged = false;"));
    assert_precedes(
        launch,
        "exec->h2d_input_staged = false;",
        "cudaGraphLaunch(exec->exec, stream->stream)",
        "C05-7 stale-payload replay gate",
    );

    assert!(ffi.contains("riley_cuda_graph_capture_begin_h2d"));
    assert!(ffi.contains("riley_cuda_graph_capture_enqueue_h2d"));
    assert!(ffi.contains("riley_cuda_graph_exec_stage_h2d_source"));
    assert!(ffi.contains("graph_exec_input_stage_metadata_is_valid"));
    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-7 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_silu_graph_uses_fixed_two_buffer_lifecycle_without_model_wiring() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_silu_bf16",
        "riley_cuda_graph_capture_enqueue_silu_bf16",
    ] {
        assert!(header.contains(required), "missing C05-8 ABI: {required}");
        assert!(ffi.contains(required), "missing C05-8 Rust FFI: {required}");
    }
    assert!(abi_layout.contains("graph_capture_begin_silu_bf16_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_silu_bf16_symbol"));

    for required in [
        "kSiluBf16 = 3",
        "silu_input",
        "silu_element_count",
        "silu_enqueue_count",
        "silu_input_lease_held",
        "RileyCudaGraphCaptureOperation operation",
    ] {
        assert!(
            internal.contains(required),
            "C05-8 native graph ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_silu_bf16_leases"));
    assert!(native.contains("release_graph_silu_bf16_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_silu_bf16_impl(")
        .nth(1)
        .expect("BF16 SiLU capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("BF16 SiLU capture admission helper must end before C exports");
    for lease in [
        "try_acquire_exclusive_use(input->active_uses)",
        "try_acquire_exclusive_use(output->active_uses)",
        "try_acquire_exclusive_use(stream->active_uses)",
    ] {
        assert_precedes(
            begin,
            lease,
            "cudaStreamBeginCapture(",
            "C05-8 BF16 SiLU capture lease admission",
        );
    }
    assert!(begin.contains("if (input == output)"));
    assert!(begin.contains("element_count == 0"));
    assert!(begin.contains("RileyCudaGraphCaptureOperation::kSiluBf16"));

    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_silu_bf16");
    assert!(enqueue.contains("graph_silu_bf16<<<"));
    assert!(enqueue.contains("cudaGetLastError"));
    assert!(enqueue.contains("owner->silu_enqueue_count != 0"));
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_silu_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-8 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphSiluBf16Resources",
        "pub struct OwnedGraphSiluBf16CaptureBeginError",
        "pub struct OwnedGraphSiluBf16Capture",
        "pub struct OwnedCapturedSiluBf16Graph",
        "pub struct OwnedGraphSiluBf16Exec",
        "pub struct OwnedGraphSiluBf16Launch<'exec>",
        "pub fn begin_owned_graph_silu_bf16_capture",
        "pub fn enqueue_silu_bf16(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-8 safe owner contract is missing: {required}"
        );
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_silu_bf16_capture")
        .nth(1)
        .expect("owned BF16 SiLU graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned BF16 SiLU capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_silu_bf16_capture_preflight",
        "resources.stream.native.begin_graph_silu_bf16_capture",
        "C05-8 BF16 SiLU Rust preflight",
    );
    assert!(owned_begin.contains("OwnedGraphSiluBf16CaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphSiluBf16CaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphSiluBf16Capture {",
        "pub struct OwnedCapturedSiluBf16Graph {",
        "pub struct OwnedGraphSiluBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphSiluBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphSiluBf16Exec")
        .nth(1)
        .expect("owned BF16 SiLU exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphSiluBf16Exec`] replay")
        .next()
        .expect("owned BF16 SiLU exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    assert!(
        !owned_exec.contains("launch_with_input")
            && !owned_exec.contains("launch_with_source")
            && !owned_exec.contains("CudaBufferSpan"),
        "C05-8 must not expose fresh-input, span, or H2D replay capability"
    );

    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-8 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_gated_multiply_graph_uses_fixed_three_buffer_lifecycle_without_model_wiring() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_gated_multiply_bf16",
        "riley_cuda_graph_capture_enqueue_gated_multiply_bf16",
    ] {
        assert!(header.contains(required), "missing C05-10 ABI: {required}");
        assert!(
            ffi.contains(required),
            "missing C05-10 Rust FFI: {required}"
        );
    }
    assert!(abi_layout.contains("graph_capture_begin_gated_multiply_bf16_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_gated_multiply_bf16_symbol"));

    for required in [
        "kGatedMultiplyBf16 = 4",
        "gated_multiply_activated_gate",
        "gated_multiply_up",
        "gated_multiply_element_count",
        "gated_multiply_enqueue_count",
        "gated_multiply_activated_gate_lease_held",
        "gated_multiply_up_lease_held",
        "RileyCudaGraphCaptureOperation operation",
    ] {
        assert!(
            internal.contains(required),
            "C05-10 native graph ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_gated_multiply_bf16_leases"));
    assert!(native.contains("release_graph_gated_multiply_bf16_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_gated_multiply_bf16_impl(")
        .nth(1)
        .expect("BF16 gated-multiply capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("BF16 gated-multiply capture admission helper must end before C exports");
    for lease in [
        "try_acquire_exclusive_use(activated_gate->active_uses)",
        "try_acquire_exclusive_use(up->active_uses)",
        "try_acquire_exclusive_use(output->active_uses)",
        "try_acquire_exclusive_use(stream->active_uses)",
    ] {
        assert_precedes(
            begin,
            lease,
            "cudaStreamBeginCapture(",
            "C05-10 BF16 gated-multiply capture lease admission",
        );
    }
    assert!(begin.contains("activated_gate == up || activated_gate == output || up == output"));
    assert!(begin.contains("element_count == 0"));
    assert!(begin.contains("RileyCudaGraphCaptureOperation::kGatedMultiplyBf16"));

    let enqueue = native_export_body(
        native,
        "riley_cuda_graph_capture_enqueue_gated_multiply_bf16",
    );
    assert!(enqueue.contains("graph_gated_multiply_bf16<<<"));
    assert!(enqueue.contains("cudaGetLastError"));
    assert!(enqueue.contains("owner->gated_multiply_enqueue_count != 0"));
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_gated_multiply_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-10 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphGatedMultiplyBf16Resources",
        "pub struct OwnedGraphGatedMultiplyBf16CaptureBeginError",
        "pub struct OwnedGraphGatedMultiplyBf16Capture",
        "pub struct OwnedCapturedGatedMultiplyBf16Graph",
        "pub struct OwnedGraphGatedMultiplyBf16Exec",
        "pub struct OwnedGraphGatedMultiplyBf16Launch<'exec>",
        "pub fn begin_owned_graph_gated_multiply_bf16_capture",
        "pub fn enqueue_gated_multiply_bf16(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-10 safe owner contract is missing: {required}"
        );
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_gated_multiply_bf16_capture")
        .nth(1)
        .expect("owned BF16 gated-multiply graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned BF16 gated-multiply capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_gated_multiply_bf16_capture_preflight",
        "begin_graph_gated_multiply_bf16_capture",
        "C05-10 BF16 gated-multiply Rust preflight",
    );
    assert!(owned_begin.contains("OwnedGraphGatedMultiplyBf16CaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphGatedMultiplyBf16CaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphGatedMultiplyBf16Capture {",
        "pub struct OwnedCapturedGatedMultiplyBf16Graph {",
        "pub struct OwnedGraphGatedMultiplyBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphGatedMultiplyBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphGatedMultiplyBf16Exec")
        .nth(1)
        .expect("owned BF16 gated-multiply exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphGatedMultiplyBf16Exec`] replay")
        .next()
        .expect("owned BF16 gated-multiply exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    assert!(
        !owned_exec.contains("launch_with_input")
            && !owned_exec.contains("launch_with_source")
            && !owned_exec.contains("CudaBufferSpan"),
        "C05-10 must not expose fresh-input, span, or H2D replay capability"
    );

    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-10 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_residual_add_graph_uses_fixed_three_buffer_lifecycle_without_model_wiring() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_residual_add_bf16",
        "riley_cuda_graph_capture_enqueue_residual_add_bf16",
    ] {
        assert!(header.contains(required), "missing C05-11 ABI: {required}");
        assert!(
            ffi.contains(required),
            "missing C05-11 Rust FFI: {required}"
        );
    }
    assert!(abi_layout.contains("graph_capture_begin_residual_add_bf16_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_residual_add_bf16_symbol"));

    for required in [
        "kResidualAddBf16 = 5",
        "residual_add_left",
        "residual_add_right",
        "residual_add_element_count",
        "residual_add_enqueue_count",
        "residual_add_left_lease_held",
        "residual_add_right_lease_held",
        "RileyCudaGraphCaptureOperation operation",
    ] {
        assert!(
            internal.contains(required),
            "C05-11 native graph ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_residual_add_bf16_leases"));
    assert!(native.contains("release_graph_residual_add_bf16_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_residual_add_bf16_impl(")
        .nth(1)
        .expect("BF16 residual-add capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("BF16 residual-add capture admission helper must end before C exports");
    for lease in [
        "try_acquire_exclusive_use(left->active_uses)",
        "try_acquire_exclusive_use(right->active_uses)",
        "try_acquire_exclusive_use(output->active_uses)",
        "try_acquire_exclusive_use(stream->active_uses)",
    ] {
        assert_precedes(
            begin,
            lease,
            "cudaStreamBeginCapture(",
            "C05-11 BF16 residual-add capture lease admission",
        );
    }
    assert!(begin.contains("left == right || left == output || right == output"));
    assert!(begin.contains("element_count == 0"));
    assert!(begin.contains("RileyCudaGraphCaptureOperation::kResidualAddBf16"));

    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_residual_add_bf16");
    assert!(enqueue.contains("graph_residual_add_bf16<<<"));
    assert!(enqueue.contains("cudaGetLastError"));
    assert!(enqueue.contains("owner->residual_add_enqueue_count != 0"));
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_residual_add_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-11 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphResidualAddBf16Resources",
        "pub struct OwnedGraphResidualAddBf16CaptureBeginError",
        "pub struct OwnedGraphResidualAddBf16Capture",
        "pub struct OwnedCapturedResidualAddBf16Graph",
        "pub struct OwnedGraphResidualAddBf16Exec",
        "pub struct OwnedGraphResidualAddBf16Launch<'exec>",
        "pub fn begin_owned_graph_residual_add_bf16_capture",
        "pub fn enqueue_residual_add_bf16(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-11 safe owner contract is missing: {required}"
        );
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_residual_add_bf16_capture")
        .nth(1)
        .expect("owned BF16 residual-add graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned BF16 residual-add capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_residual_add_bf16_capture_preflight",
        "begin_graph_residual_add_bf16_capture",
        "C05-11 BF16 residual-add Rust preflight",
    );
    assert!(owned_begin.contains("OwnedGraphResidualAddBf16CaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphResidualAddBf16CaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphResidualAddBf16Capture {",
        "pub struct OwnedCapturedResidualAddBf16Graph {",
        "pub struct OwnedGraphResidualAddBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphResidualAddBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphResidualAddBf16Exec")
        .nth(1)
        .expect("owned BF16 residual-add exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphResidualAddBf16Exec`] replay")
        .next()
        .expect("owned BF16 residual-add exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    assert!(
        !owned_exec.contains("launch_with_input")
            && !owned_exec.contains("launch_with_source")
            && !owned_exec.contains("CudaBufferSpan"),
        "C05-11 must not expose fresh-input, span, or H2D replay capability"
    );

    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-11 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_canonical_bf16_rms_norm_graph_uses_fixed_three_buffer_lifecycle_without_model_wiring() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_canonical_rms_norm_bf16",
        "riley_cuda_graph_capture_enqueue_canonical_rms_norm_bf16",
    ] {
        assert!(header.contains(required), "missing C05-12 ABI: {required}");
        assert!(
            ffi.contains(required),
            "missing C05-12 Rust FFI: {required}"
        );
    }
    assert!(abi_layout.contains("graph_capture_begin_canonical_rms_norm_bf16_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_canonical_rms_norm_bf16_symbol"));

    for required in [
        "kCanonicalRmsNormBf16 = 6",
        "canonical_rms_norm_input",
        "canonical_rms_norm_weight",
        "canonical_rms_norm_row_count",
        "canonical_rms_norm_hidden_size",
        "canonical_rms_norm_epsilon",
        "canonical_rms_norm_enqueue_count",
        "canonical_rms_norm_input_lease_held",
        "canonical_rms_norm_weight_lease_held",
        "RileyCudaGraphCaptureOperation operation",
    ] {
        assert!(
            internal.contains(required),
            "C05-12 native graph ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_canonical_rms_norm_bf16_leases"));
    assert!(native.contains("release_graph_canonical_rms_norm_bf16_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_canonical_rms_norm_bf16_impl(")
        .nth(1)
        .expect("canonical BF16 RMSNorm capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("canonical BF16 RMSNorm capture admission helper must end before C exports");
    for lease in [
        "try_acquire_exclusive_use(input->active_uses)",
        "try_acquire_exclusive_use(weight->active_uses)",
        "try_acquire_exclusive_use(output->active_uses)",
        "try_acquire_exclusive_use(stream->active_uses)",
    ] {
        assert_precedes(
            begin,
            lease,
            "cudaStreamBeginCapture(",
            "C05-12 canonical BF16 RMSNorm capture lease admission",
        );
    }
    assert!(begin.contains("input == weight || input == output || weight == output"));
    assert!(begin.contains("canonical_rms_norm_element_count(row_count, hidden_size"));
    assert!(begin.contains("std::isfinite(epsilon)"));
    assert!(begin.contains("epsilon <= 0.0F"));
    assert!(begin.contains("RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16"));

    let enqueue = native_export_body(
        native,
        "riley_cuda_graph_capture_enqueue_canonical_rms_norm_bf16",
    );
    assert!(enqueue.contains("graph_canonical_rms_norm_bf16<<<"));
    assert!(enqueue.contains("cudaGetLastError"));
    assert!(enqueue.contains("owner->canonical_rms_norm_enqueue_count != 0"));
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_rms_norm_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-12 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphCanonicalRmsNormBf16Resources",
        "pub struct OwnedGraphCanonicalRmsNormBf16CaptureBeginError",
        "pub struct OwnedGraphCanonicalRmsNormBf16Capture",
        "pub struct OwnedCapturedCanonicalRmsNormBf16Graph",
        "pub struct OwnedGraphCanonicalRmsNormBf16Exec",
        "pub struct OwnedGraphCanonicalRmsNormBf16Launch<'exec>",
        "pub fn begin_owned_graph_canonical_rms_norm_bf16_capture",
        "pub fn enqueue_canonical_rms_norm_bf16(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-12 safe owner contract is missing: {required}"
        );
    }
    for required in [
        "generic eager",
        "SmolLM2",
        "Fixed37",
        "fused RMSNorm",
        "C07 executor integration",
    ] {
        assert!(
            graph.contains(required),
            "C05-12 public documentation must preserve its generic-only scope: {required}"
        );
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_canonical_rms_norm_bf16_capture")
        .nth(1)
        .expect("owned canonical BF16 RMSNorm graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned canonical BF16 RMSNorm capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_canonical_rms_norm_bf16_capture_preflight",
        "begin_graph_canonical_rms_norm_bf16_capture",
        "C05-12 canonical BF16 RMSNorm Rust preflight",
    );
    assert!(owned_begin.contains("OwnedGraphCanonicalRmsNormBf16CaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphCanonicalRmsNormBf16CaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphCanonicalRmsNormBf16Capture {",
        "pub struct OwnedCapturedCanonicalRmsNormBf16Graph {",
        "pub struct OwnedGraphCanonicalRmsNormBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphCanonicalRmsNormBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphCanonicalRmsNormBf16Exec")
        .nth(1)
        .expect("owned canonical BF16 RMSNorm exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphCanonicalRmsNormBf16Exec`] replay")
        .next()
        .expect("owned canonical BF16 RMSNorm exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    assert!(
        !owned_exec.contains("launch_with_input")
            && !owned_exec.contains("launch_with_source")
            && !owned_exec.contains("CudaBufferSpan"),
        "C05-12 must not expose fresh-input, span, or H2D replay capability"
    );

    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-12 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_argmax_graph_uses_fixed_two_buffer_lifecycle_without_c07_mapping() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_bf16_argmax",
        "riley_cuda_graph_capture_enqueue_bf16_argmax",
    ] {
        assert!(header.contains(required), "missing C05-13 ABI: {required}");
        assert!(
            ffi.contains(required),
            "missing C05-13 Rust FFI: {required}"
        );
    }
    assert!(abi_layout.contains("graph_capture_begin_bf16_argmax_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_bf16_argmax_symbol"));

    for required in [
        "kBf16Argmax = 7",
        "bf16_argmax_logits",
        "bf16_argmax_row_count",
        "bf16_argmax_vocabulary_size",
        "bf16_argmax_enqueue_count",
        "bf16_argmax_logits_lease_held",
        "RileyCudaGraphCaptureOperation operation",
    ] {
        assert!(
            internal.contains(required),
            "C05-13 native graph ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_bf16_argmax_leases"));
    assert!(native.contains("release_graph_bf16_argmax_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_bf16_argmax_impl(")
        .nth(1)
        .expect("deterministic BF16 argmax capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("deterministic BF16 argmax capture admission helper must end before C exports");
    for lease in [
        "try_acquire_exclusive_use(logits->active_uses)",
        "try_acquire_exclusive_use(results->active_uses)",
        "try_acquire_exclusive_use(stream->active_uses)",
    ] {
        assert_precedes(
            begin,
            lease,
            "cudaStreamBeginCapture(",
            "C05-13 deterministic BF16 argmax capture lease admission",
        );
    }
    for required in [
        "logits == results",
        "row_count == 0",
        "vocabulary_size == 0",
        "vocabulary_size > UINT32_MAX",
        "RileyCudaGraphCaptureOperation::kBf16Argmax",
    ] {
        assert!(
            begin.contains(required),
            "C05-13 capture admission must preserve deterministic BF16 argmax bounds: {required}"
        );
    }

    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_bf16_argmax");
    assert!(enqueue.contains("graph_bf16_argmax_bf16<<<"));
    assert!(enqueue.contains("cudaGetLastError"));
    assert!(enqueue.contains("owner->bf16_argmax_enqueue_count != 0"));
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_bf16_argmax_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-13 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphBf16ArgmaxResources",
        "pub struct OwnedGraphBf16ArgmaxCaptureBeginError",
        "pub struct OwnedGraphBf16ArgmaxCapture",
        "pub struct OwnedCapturedBf16ArgmaxGraph",
        "pub struct OwnedGraphBf16ArgmaxExec",
        "pub struct OwnedGraphBf16ArgmaxLaunch<'exec>",
        "pub fn begin_owned_graph_bf16_argmax_capture",
        "pub fn enqueue_bf16_argmax(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-13 safe owner contract is missing: {required}"
        );
    }
    for required in [
        "lower token ID",
        "non-finite",
        "row gather",
        "C07 executor integration",
    ] {
        assert!(
            graph.contains(required),
            "C05-13 public documentation must preserve its narrow scope: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_bf16_argmax_capture_preflight(")
        .nth(1)
        .expect("C05-13 BF16 argmax preflight must remain present")
        .split("impl CudaStream")
        .next()
        .expect("C05-13 BF16 argmax preflight must precede stream entry points");
    for required in [
        "row_count == 0",
        "vocabulary_size == 0",
        "u64::from(u32::MAX)",
        "row_count.checked_mul(vocabulary_size)",
        "row_count.checked_mul(2)",
        "size_of::<u16>()",
        "size_of::<u32>()",
    ] {
        assert!(
            preflight.contains(required),
            "C05-13 preflight must retain the checked fixed-address geometry: {required}"
        );
    }
    let resources = graph
        .split("impl OwnedGraphBf16ArgmaxResources")
        .nth(1)
        .expect("C05-13 resource bundle must remain present")
        .split("/// Error from beginning an owned fixed-address deterministic BF16 argmax graph")
        .next()
        .expect("C05-13 resource bundle must end before its begin error");
    assert_precedes(
        resources,
        "results.close()?",
        "logits.close()?",
        "C05-13 resource close order",
    );
    assert_precedes(
        resources,
        "logits.close()?",
        "stream.close()",
        "C05-13 resource close order",
    );

    let owned_begin = graph
        .split("pub fn begin_owned_graph_bf16_argmax_capture")
        .nth(1)
        .expect("owned deterministic BF16 argmax graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned deterministic BF16 argmax capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_bf16_argmax_capture_preflight",
        "begin_graph_bf16_argmax_capture",
        "C05-13 deterministic BF16 argmax Rust preflight",
    );
    assert!(owned_begin.contains("OwnedGraphBf16ArgmaxCaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphBf16ArgmaxCaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphBf16ArgmaxCapture {",
        "pub struct OwnedCapturedBf16ArgmaxGraph {",
        "pub struct OwnedGraphBf16ArgmaxExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphBf16ArgmaxResources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphBf16ArgmaxExec")
        .nth(1)
        .expect("owned deterministic BF16 argmax exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphBf16ArgmaxExec`] replay")
        .next()
        .expect("owned deterministic BF16 argmax exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    for forbidden in [
        "launch_with_input",
        "launch_with_source",
        "CudaBufferSpan",
        "row_gather",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !owned_exec.contains(forbidden),
            "C05-13 must not expose C07 or mutable replay capability: {forbidden}"
        );
    }

    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-13 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_row_gather_graph_uses_fixed_three_buffer_lifecycle_without_c07_mapping() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_bf16_row_gather",
        "riley_cuda_graph_capture_enqueue_bf16_row_gather",
    ] {
        assert!(header.contains(required), "missing C05-14 ABI: {required}");
        assert!(
            ffi.contains(required),
            "missing C05-14 Rust FFI: {required}"
        );
    }
    assert!(abi_layout.contains("graph_capture_begin_bf16_row_gather_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_bf16_row_gather_symbol"));

    for required in [
        "kBf16RowGather = 8",
        "bf16_row_gather_input",
        "bf16_row_gather_indices",
        "bf16_row_gather_input_row_count",
        "bf16_row_gather_output_row_count",
        "bf16_row_gather_column_count",
        "bf16_row_gather_enqueue_count",
        "bf16_row_gather_input_lease_held",
        "bf16_row_gather_indices_lease_held",
        "RileyCudaGraphCaptureOperation operation",
    ] {
        assert!(
            internal.contains(required),
            "C05-14 native graph ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_bf16_row_gather_leases"));
    assert!(native.contains("release_graph_bf16_row_gather_leases"));

    let begin = native
        .split("RileyCudaStatus capture_begin_bf16_row_gather_impl(")
        .nth(1)
        .expect("fixed BF16 row-gather capture admission helper must remain present")
        .split("}  // namespace")
        .next()
        .expect("fixed BF16 row-gather admission helper must end before C exports");
    for lease in [
        "try_acquire_exclusive_use(input->active_uses)",
        "try_acquire_exclusive_use(row_indices->active_uses)",
        "try_acquire_exclusive_use(output->active_uses)",
        "try_acquire_exclusive_use(stream->active_uses)",
    ] {
        assert_precedes(
            begin,
            lease,
            "cudaStreamBeginCapture(",
            "C05-14 fixed BF16 row-gather capture lease admission",
        );
    }
    for required in [
        "input == row_indices || input == output || row_indices == output",
        "bf16_row_gather_shape_is_valid(",
        "RileyCudaGraphCaptureOperation::kBf16RowGather",
    ] {
        assert!(
            begin.contains(required),
            "C05-14 capture admission must preserve fixed BF16 row-gather bounds: {required}"
        );
    }

    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_bf16_row_gather");
    assert!(enqueue.contains("graph_bf16_row_gather_bf16<<<"));
    assert!(enqueue.contains("cudaGetLastError"));
    assert!(enqueue.contains("owner->bf16_row_gather_enqueue_count != 0"));
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_row_gather_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-14 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphBf16RowGatherResources",
        "pub struct OwnedGraphBf16RowGatherCaptureBeginError",
        "pub struct OwnedGraphBf16RowGatherCapture",
        "pub struct OwnedCapturedBf16RowGatherGraph",
        "pub struct OwnedGraphBf16RowGatherExec",
        "pub struct OwnedGraphBf16RowGatherLaunch<'exec>",
        "pub fn begin_owned_graph_bf16_row_gather_capture",
        "pub fn enqueue_bf16_row_gather(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-14 safe owner contract is missing: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_bf16_row_gather_capture_preflight(")
        .nth(1)
        .expect("C05-14 BF16 row-gather preflight must remain present")
        .split("impl CudaStream")
        .next()
        .expect("C05-14 BF16 row-gather preflight must precede stream entry points");
    for required in [
        "row_indices_host",
        "output_row_count == 0",
        "input_row_count == 0",
        "column_count == 0",
        "checked_mul",
        "size_of::<u16>()",
        "size_of::<u32>()",
        "crate::batch::validate_gather_indices(row_indices_host, input_row_count)",
    ] {
        assert!(
            preflight.contains(required),
            "C05-14 preflight must retain host-mirror fixed-address validation: {required}"
        );
    }

    let resources = graph
        .split("impl OwnedGraphBf16RowGatherResources")
        .nth(1)
        .expect("C05-14 resource bundle must remain present")
        .split("/// Error from beginning an owned fixed-address BF16 row-gather graph")
        .next()
        .expect("C05-14 resource bundle must end before its begin error");
    assert_precedes(
        resources,
        "output.close()?",
        "row_indices.close()?",
        "C05-14 resource close order",
    );
    assert_precedes(
        resources,
        "row_indices.close()?",
        "input.close()?",
        "C05-14 resource close order",
    );
    assert_precedes(
        resources,
        "input.close()?",
        "stream.close()",
        "C05-14 resource close order",
    );
    assert!(
        !resources.contains("row_indices_host"),
        "the temporary host row-index mirror must not be retained by owned graph resources"
    );

    let owned_begin = graph
        .split("pub fn begin_owned_graph_bf16_row_gather_capture")
        .nth(1)
        .expect("owned BF16 row-gather graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("owned BF16 row-gather graph capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_bf16_row_gather_capture_preflight",
        "begin_graph_bf16_row_gather_capture",
        "C05-14 BF16 row-gather Rust preflight",
    );
    assert!(owned_begin.contains("OwnedGraphBf16RowGatherCaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphBf16RowGatherCaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphBf16RowGatherCapture {",
        "pub struct OwnedCapturedBf16RowGatherGraph {",
        "pub struct OwnedGraphBf16RowGatherExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphBf16RowGatherResources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphBf16RowGatherExec")
        .nth(1)
        .expect("owned BF16 row-gather exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphBf16RowGatherExec`] replay")
        .next()
        .expect("owned BF16 row-gather exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    for forbidden in [
        "launch_with_input",
        "launch_with_source",
        "CudaBufferSpan",
        "CudaBufferSpanMut",
        "GpuGreedy",
        "CompletionBoundary",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !owned_exec.contains(forbidden),
            "C05-14 must not expose C07 or mutable replay capability: {forbidden}"
        );
    }

    for forbidden in ["riley_runtime", "riley_server", "graph_decode", "llama"] {
        assert!(
            !graph.contains(forbidden),
            "C05-14 graph ownership must remain model/runtime independent: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_row_gather_argmax_graph_uses_fixed_four_buffer_lifecycle_without_c07_mapping() {
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");

    for required in [
        "riley_cuda_graph_capture_begin_bf16_row_gather_argmax",
        "riley_cuda_graph_capture_enqueue_bf16_row_gather_argmax",
        "pub(super) fn begin_graph_bf16_row_gather_argmax_capture",
        "pub(super) fn enqueue_bf16_row_gather_argmax",
    ] {
        assert!(
            ffi.contains(required),
            "missing C05-15 Rust FFI: {required}"
        );
    }
    assert!(graph.contains("Bf16RowGatherArgmax = 9"));

    for required in [
        "pub struct OwnedGraphBf16RowGatherArgmaxResources",
        "pub struct OwnedGraphBf16RowGatherArgmaxCaptureBeginError",
        "pub struct OwnedGraphBf16RowGatherArgmaxCapture",
        "pub struct OwnedCapturedBf16RowGatherArgmaxGraph",
        "pub struct OwnedGraphBf16RowGatherArgmaxExec",
        "pub struct OwnedGraphBf16RowGatherArgmaxLaunch<'exec>",
        "pub fn begin_owned_graph_bf16_row_gather_argmax_capture",
        "pub fn enqueue_bf16_row_gather_argmax(&mut self)",
        "pub fn launch<'exec>",
    ] {
        assert!(
            graph.contains(required),
            "C05-15 safe owner contract is missing: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_bf16_row_gather_argmax_capture_preflight(")
        .nth(1)
        .expect("C05-15 BF16 row-gather -> argmax preflight must remain present")
        .split("/// A by-value stream, four fixed device allocations, and one exact pinned")
        .next()
        .expect("C05-15 BF16 row-gather -> argmax preflight must end before C05-16 owners");
    for required in [
        "row_indices_host",
        "output_row_count == 0",
        "input_row_count == 0",
        "vocabulary_size == 0",
        "vocabulary_size > u64::from(u32::MAX)",
        "gathered_logits",
        "results",
        "same_allocation",
        "checked_mul",
        "size_of::<u16>()",
        "size_of::<u32>()",
        "size_of::<Bf16ArgmaxResult>()",
        "crate::batch::validate_gather_indices(row_indices_host, input_row_count)",
    ] {
        assert!(
            preflight.contains(required),
            "C05-15 preflight must retain four-resource validation: {required}"
        );
    }
    assert_eq!(
        preflight.matches("same_allocation").count(),
        6,
        "C05-15 must reject every pairwise alias across its four fixed allocations"
    );
    assert_eq!(
        preflight.matches("ensure_same_context").count(),
        4,
        "C05-15 must bind every fixed allocation to the capture stream context"
    );
    assert_eq!(
        preflight.matches("ensure_idle_for_operation").count(),
        4,
        "C05-15 must require every fixed allocation to be idle before capture"
    );

    let resources = graph
        .split("impl OwnedGraphBf16RowGatherArgmaxResources")
        .nth(1)
        .expect("C05-15 resource bundle must remain present")
        .split("/// Error from beginning an owned fixed-address BF16 row-gather then")
        .next()
        .expect("C05-15 resource bundle must end before its begin error");
    assert_precedes(
        resources,
        "results.close()?",
        "gathered_logits.close()?",
        "C05-15 resource close order",
    );
    assert_precedes(
        resources,
        "gathered_logits.close()?",
        "row_indices.close()?",
        "C05-15 resource close order",
    );
    assert_precedes(
        resources,
        "row_indices.close()?",
        "input.close()?",
        "C05-15 resource close order",
    );
    assert_precedes(
        resources,
        "input.close()?",
        "stream.close()",
        "C05-15 resource close order",
    );
    assert!(
        !resources.contains("row_indices_host"),
        "the temporary host row-index mirror must not be retained by the composite owner"
    );

    let owned_begin = graph
        .split("pub fn begin_owned_graph_bf16_row_gather_argmax_capture")
        .nth(1)
        .expect("owned BF16 row-gather -> argmax graph capture entry point must remain present")
        .split("/// Begins one C05-16 fixed-address BF16 row-gather -> argmax -> pinned")
        .next()
        .expect("owned C05-15 capture must end before C05-16 capture");
    assert_precedes(
        owned_begin,
        "validate_graph_bf16_row_gather_argmax_capture_preflight",
        "begin_graph_bf16_row_gather_argmax_capture",
        "C05-15 BF16 row-gather -> argmax Rust preflight",
    );
    assert!(
        owned_begin.contains("let output_row_count = match u64::try_from(row_indices_host.len())")
    );
    assert!(
        !owned_begin.contains("output_row_count: u64"),
        "safe C05-15 begin must derive output rows only from the temporary host mirror"
    );
    assert!(owned_begin.contains("OwnedGraphBf16RowGatherArgmaxCaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphBf16RowGatherArgmaxCaptureBeginError::terminal"));

    let capture = graph
        .split("impl OwnedGraphBf16RowGatherArgmaxCapture {")
        .nth(1)
        .expect("owned C05-15 capture must remain present")
        .split("impl Drop for OwnedGraphBf16RowGatherArgmaxCapture")
        .next()
        .expect("owned C05-15 capture must end before its drop policy");
    assert!(capture.contains("self.enqueue_failed = true"));
    assert!(capture.contains("this partial capture must be aborted"));
    assert!(
        capture.contains("capture end requires the one fixed BF16 row-gather -> argmax enqueue")
    );

    for owner in [
        "pub struct OwnedGraphBf16RowGatherArgmaxCapture {",
        "pub struct OwnedCapturedBf16RowGatherArgmaxGraph {",
        "pub struct OwnedGraphBf16RowGatherArgmaxExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphBf16RowGatherArgmaxResources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphBf16RowGatherArgmaxExec")
        .nth(1)
        .expect("owned C05-15 exec must remain present")
        .split("/// Completion owner for one [`OwnedGraphBf16RowGatherArgmaxExec`] replay")
        .next()
        .expect("owned C05-15 exec must end before its completion owner");
    assert!(owned_exec.contains("pub fn launch<'exec>"));
    for forbidden in [
        "launch_with_input",
        "launch_with_source",
        "CudaBufferSpan",
        "CudaBufferSpanMut",
        "GpuGreedy",
        "CompletionBoundary",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !owned_exec.contains(forbidden),
            "C05-15 must not expose C07 or mutable replay capability: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_row_gather_argmax_d2h_graph_has_completion_scoped_pinned_result_contract() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    for required in [
        "riley_cuda_graph_capture_begin_bf16_row_gather_argmax_d2h",
        "riley_cuda_graph_capture_enqueue_bf16_row_gather_argmax_d2h",
        "riley_cuda_graph_exec_read_bf16_row_gather_argmax_d2h_results",
    ] {
        assert!(header.contains(required), "missing C05-16 ABI: {required}");
        assert!(
            ffi.contains(required),
            "missing C05-16 Rust FFI: {required}"
        );
    }
    for required in [
        "graph_capture_begin_bf16_row_gather_argmax_d2h_symbol",
        "graph_capture_enqueue_bf16_row_gather_argmax_d2h_symbol",
        "graph_exec_read_bf16_row_gather_argmax_d2h_results_symbol",
    ] {
        assert!(
            abi_layout.contains(required),
            "C05-16 ABI layout/linkage witness is missing: {required}"
        );
    }
    assert!(graph.contains("Bf16RowGatherArgmaxD2H = 10"));
    for required in [
        "kBf16RowGatherArgmaxD2H = 10",
        "bf16_row_gather_argmax_d2h_pinned_results",
        "bf16_row_gather_argmax_d2h_result_byte_len",
        "bf16_row_gather_argmax_d2h_enqueue_count",
        "bf16_row_gather_argmax_d2h_pinned_results_lease_held",
        "bf16_row_gather_argmax_d2h_completion_visible",
    ] {
        assert!(
            internal.contains(required),
            "C05-16 native fixed-address ownership state is missing: {required}"
        );
    }
    assert!(native.contains("release_capture_bf16_row_gather_argmax_d2h_leases"));
    assert!(native.contains("release_graph_bf16_row_gather_argmax_d2h_leases"));

    let enqueue = native_export_body(
        native,
        "riley_cuda_graph_capture_enqueue_bf16_row_gather_argmax_d2h",
    );
    for required in [
        "graph_bf16_row_gather_bf16<<<",
        "graph_bf16_argmax_bf16<<<",
        "cudaMemcpyAsync(",
        "cudaMemcpyDeviceToHost",
        "bf16_row_gather_argmax_d2h_enqueue_count",
    ] {
        assert!(
            enqueue.contains(required),
            "C05-16 capture enqueue must retain its fixed three-node chain: {required}"
        );
    }
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "riley_cuda_row_gather_execute",
        "riley_cuda_bf16_argmax_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-16 capture enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }
    let read = native_export_body(
        native,
        "riley_cuda_graph_exec_read_bf16_row_gather_argmax_d2h_results",
    );
    for required in [
        "bf16_row_gather_argmax_d2h_completion_visible",
        "destination_len != result_byte_len",
        "std::memcpy",
        "true, true, true, false",
    ] {
        assert!(
            read.contains(required),
            "C05-16 read receipt must remain completion-scoped and lease-preserving: {required}"
        );
    }
    assert!(
        !read.contains("release_graph_bf16_row_gather_argmax_d2h_leases"),
        "reading raw result bytes must not release the replayable graph lease"
    );

    for required in [
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HResources",
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError",
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HCapture",
        "pub struct OwnedCapturedBf16RowGatherArgmaxD2HGraph",
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HExec",
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HLaunch<'exec>",
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HCompletion<'exec>",
        "pub fn begin_owned_graph_bf16_row_gather_argmax_d2h_capture",
        "pub fn enqueue_bf16_row_gather_argmax_d2h(&mut self)",
        "pub fn read_result_bytes(&mut self, destination: &mut [u8])",
    ] {
        assert!(
            graph.contains(required),
            "C05-16 safe owner contract is missing: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_bf16_row_gather_argmax_d2h_capture_preflight(")
        .nth(1)
        .expect("C05-16 BF16 row-gather -> argmax -> D2H preflight must remain present")
        .split("pub struct OwnedGraphIndexedRopeBf16Resources")
        .next()
        .expect("C05-16 preflight must end before the C05-17 resource owner");
    for required in [
        "pinned_results",
        "row_indices_host",
        "input_row_count == 0",
        "output_row_count == 0",
        "vocabulary_size == 0",
        "vocabulary_size > u64::from(u32::MAX)",
        "crate::batch::validate_gather_indices(row_indices_host, input_row_count)",
        "size_of::<Bf16ArgmaxResult>()",
        "pinned_results.byte_len() != result_bytes",
    ] {
        assert!(
            preflight.contains(required),
            "C05-16 preflight must retain six-resource validation: {required}"
        );
    }
    assert_eq!(
        preflight.matches("same_allocation").count(),
        6,
        "C05-16 must reject every device-allocation alias across its four fixed device buffers"
    );
    assert_eq!(
        preflight.matches("ensure_same_context").count(),
        5,
        "C05-16 must bind four device allocations and the pinned result allocation to the capture context"
    );
    assert_eq!(
        preflight.matches("ensure_idle_for_operation").count(),
        5,
        "C05-16 must require all four device allocations and the pinned result allocation to be idle"
    );

    let resources = graph
        .split("impl OwnedGraphBf16RowGatherArgmaxD2HResources")
        .nth(1)
        .expect("C05-16 resource sextet must remain present")
        .split("/// Error from beginning one by-value fixed-address C05-16 graph capture")
        .next()
        .expect("C05-16 resource sextet must end before its begin error");
    for (earlier, later) in [
        ("pinned_results.close()?", "results.close()?"),
        ("results.close()?", "gathered_logits.close()?"),
        ("gathered_logits.close()?", "row_indices.close()?"),
        ("row_indices.close()?", "input.close()?"),
        ("input.close()?", "stream.close()"),
    ] {
        assert_precedes(resources, earlier, later, "C05-16 resource close order");
    }
    assert!(
        !resources.contains("row_indices_host"),
        "the temporary host row-index mirror must not survive capture admission"
    );

    let owned_begin = graph
        .split("pub fn begin_owned_graph_bf16_row_gather_argmax_d2h_capture")
        .nth(1)
        .expect("owned C05-16 graph capture entry point must remain present")
        .split("/// Begins one C05-17 fixed-address BF16 indexed-RoPE graph capture.")
        .next()
        .expect("owned C05-16 capture must precede the C05-17 capture");
    assert_precedes(
        owned_begin,
        "validate_graph_bf16_row_gather_argmax_d2h_capture_preflight",
        "begin_graph_bf16_row_gather_argmax_d2h_capture",
        "C05-16 Rust preflight",
    );
    assert!(
        owned_begin.contains("let output_row_count = match u64::try_from(row_indices_host.len())")
    );
    assert!(
        !owned_begin.contains("output_row_count: u64"),
        "C05-16 must derive output rows only from the temporary host mirror"
    );
    assert!(owned_begin.contains("OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError::recoverable"));
    assert!(owned_begin.contains("OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError::terminal"));

    for owner in [
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HCapture {",
        "pub struct OwnedCapturedBf16RowGatherArgmaxD2HGraph {",
        "pub struct OwnedGraphBf16RowGatherArgmaxD2HExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphBf16RowGatherArgmaxD2HResources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before its captured resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let capture = graph
        .split("impl OwnedGraphBf16RowGatherArgmaxD2HCapture {")
        .nth(1)
        .expect("owned C05-16 capture must remain present")
        .split("impl Drop for OwnedGraphBf16RowGatherArgmaxD2HCapture")
        .next()
        .expect("owned C05-16 capture must end before its drop policy");
    assert!(capture.contains("self.enqueue_failed = true"));
    assert!(capture.contains("this partial capture must be aborted"));
    assert!(
        capture.contains(
            "capture end requires the one fixed BF16 row-gather -> argmax -> D2H enqueue"
        )
    );

    let launch = graph
        .split("impl<'exec> OwnedGraphBf16RowGatherArgmaxD2HLaunch<'exec>")
        .nth(1)
        .expect("C05-16 launch completion owner must remain present")
        .split("impl Drop for OwnedGraphBf16RowGatherArgmaxD2HLaunch")
        .next()
        .expect("C05-16 launch completion owner must end before its drop policy");
    assert!(
        launch.contains("CudaResult<OwnedGraphBf16RowGatherArgmaxD2HCompletion<'exec>>"),
        "C05-16 finish must hand out an explicitly completion-scoped receipt"
    );
    let completion = graph
        .split("impl OwnedGraphBf16RowGatherArgmaxD2HCompletion")
        .nth(1)
        .expect("C05-16 completion receipt must remain present")
        .split("fn take_owned_graph_bf16_row_gather_argmax_d2h_resources")
        .next()
        .expect("C05-16 receipt must end before owner-resource recovery");
    for required in [
        "destination must have exactly the capture-time result byte length",
        "pinned_results.byte_len()",
        "read_bf16_row_gather_argmax_d2h_results(destination)",
        "if result.is_err()",
        "self.exec.terminal = true",
    ] {
        assert!(
            completion.contains(required),
            "C05-16 completion receipt must remain raw, exact-length, and op-specific: {required}"
        );
    }
    let native_read = completion
        .split("#[cfg(feature = \"cuda\")]")
        .nth(1)
        .expect("C05-16 completion receipt must retain its CUDA result-read branch")
        .split("#[cfg(not(feature = \"cuda\"))]")
        .next()
        .expect("C05-16 CUDA result-read branch must end before its unavailable branch");
    assert_precedes(
        native_read,
        "read_bf16_row_gather_argmax_d2h_results(destination)",
        "if result.is_err()",
        "C05-16 native result-read must inspect its result before deciding terminality",
    );
    assert_precedes(
        native_read,
        "if result.is_err()",
        "self.exec.terminal = true",
        "C05-16 native result-read failure must terminalize its Rust exec",
    );
    for forbidden in [
        "deterministic_bf16_argmax",
        "row_gather(",
        "GpuGreedy",
        "CompletionBoundary",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !completion.contains(forbidden),
            "C05-16 raw completion receipt must not perform model/runtime work: {forbidden}"
        );
    }
}

#[test]
fn native_graph_owners_are_wired_with_c05_5_fill_capture_replay() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let native = include_str!("../../../kernels/src/graph.cu");
    let cmake = include_str!("../../../kernels/CMakeLists.txt");
    let build_script = include_str!("../build.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");

    assert!(header.contains("riley_cuda_graph_capture_abort("));
    assert!(header.contains("riley_cuda_graph_capture_begin_fill_f32("));
    assert!(header.contains("riley_cuda_graph_capture_enqueue_fill_f32("));
    assert!(header.contains("riley_cuda_graph_capture_begin_h2d("));
    assert!(header.contains("riley_cuda_graph_capture_enqueue_h2d("));
    assert!(header.contains("riley_cuda_graph_capture_end("));
    assert!(header.contains("riley_cuda_graph_instantiate("));
    assert!(header.contains("riley_cuda_graph_exec_launch("));
    assert!(header.contains("riley_cuda_graph_exec_stage_h2d_source("));
    assert!(header.contains("riley_cuda_graph_launch_complete("));
    assert!(header.contains("riley_cuda_graph_exec_close("));
    assert!(native.contains("*out_capture = nullptr;"));
    assert!(
        native.contains("clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN)")
    );
    assert!(native.contains("cudaStreamBeginCapture"));
    assert!(native.contains("cudaStreamEndCapture"));
    assert!(native.contains("cudaGraphDestroy"));
    assert!(native.contains("cudaGraphInstantiate"));
    assert!(native.contains("cudaGraphLaunch"));
    assert!(native.contains("cudaGraphExecDestroy"));
    assert!(native.contains("try_acquire_exclusive_use(stream->active_uses)"));
    assert!(native.contains("native_thread_token"));
    assert!(native.contains("*capture = nullptr;"));
    assert!(native.contains("graph_error_reserved_is_zero"));
    assert!(cmake.contains("src/graph.cu"));
    assert!(build_script.contains("kernels_dir.join(\"src/graph.cu\")"));
    assert!(abi_layout.contains("graph_capture_begin_symbol"));
    assert!(abi_layout.contains("graph_capture_abort_symbol"));
    assert!(abi_layout.contains("graph_capture_begin_fill_f32_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_fill_f32_symbol"));
    assert!(abi_layout.contains("graph_capture_begin_h2d_symbol"));
    assert!(abi_layout.contains("graph_capture_enqueue_h2d_symbol"));
    assert!(abi_layout.contains("graph_capture_end_symbol"));
    assert!(abi_layout.contains("graph_instantiate_symbol"));
    assert!(abi_layout.contains("graph_exec_launch_symbol"));
    assert!(abi_layout.contains("graph_exec_stage_h2d_source_symbol"));
    assert!(abi_layout.contains("graph_launch_complete_symbol"));
    assert!(abi_layout.contains("graph_close_symbol"));
    assert!(abi_layout.contains("graph_exec_close_symbol"));
}

#[test]
fn fill_capture_is_preallocated_and_replay_is_bound_to_its_exact_stream() {
    let native = include_str!("../../../kernels/src/graph.cu");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let graph = include_str!("../src/graph.rs");

    let capture_begin_impl = native
        .split("RileyCudaStatus capture_begin_impl(")
        .nth(1)
        .expect("C05-5 fill-capture admission helper must remain present")
        .split("extern \"C\" RileyCudaStatus")
        .next()
        .expect("fill-capture admission helper must end before C exports");
    assert_precedes(
        capture_begin_impl,
        "try_acquire_exclusive_use(fill_buffer->active_uses)",
        "cudaStreamBeginCapture(stream->stream",
        "C05-5 fixed-buffer capture admission",
    );
    assert_precedes(
        capture_begin_impl,
        "require_stream_capture_idle(stream, error, kBeginFillOperation)",
        "cudaStreamBeginCapture(stream->stream",
        "C05-5 foreign-capture provenance admission",
    );
    assert!(capture_begin_impl.contains("prepared_graph"));
    assert!(
        native.contains("RileyCudaStatus require_stream_capture_idle("),
        "graph begin must prove the stream was not already captured by foreign CUDA work"
    );

    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_fill_f32");
    assert!(enqueue.contains("graph_fill_f32<<<"));
    assert!(enqueue.contains("++owner->fill_enqueue_count"));
    assert!(
        !enqueue.contains("std::calloc") && !enqueue.contains("std::free"),
        "the whitelisted capture enqueue must not allocate or free host bookkeeping"
    );

    let end = native_export_body(native, "riley_cuda_graph_capture_end");
    assert!(end.contains("owner->fill_enqueue_count == 0"));
    assert!(end.contains("owner->prepared_graph"));

    let launch = native_export_body(native, "riley_cuda_graph_exec_launch");
    assert!(launch.contains("stream != exec->stream"));
    assert!(launch.contains("graph exec must launch on its exact captured stream"));

    let complete = native_export_body(native, "riley_cuda_graph_launch_complete");
    assert!(
        complete.contains("exec->poisoned = false;"),
        "a known completion must settle a launch-side deferred error so its exec can close"
    );
    assert!(
        !complete.contains("exec->poisoned ||"),
        "launch completion must remain reachable when launch reported a deferred error"
    );

    let safe_fill_capture = graph
        .split("impl<'stream, 'buffer> GraphFillCapture<'stream, 'buffer>")
        .nth(1)
        .expect("safe fixed-fill graph capture must remain present")
        .split("impl Drop for GraphFillCapture")
        .next()
        .expect("safe fixed-fill graph capture must end before its Drop hook");
    assert_precedes(
        safe_fill_capture,
        "if !self.enqueued",
        "let transition = self.native.end()",
        "zero-enqueue end rejection before native end",
    );
    assert!(safe_fill_capture.contains("Multiple successful enqueue calls are allowed"));
    assert!(
        !safe_fill_capture.contains("enqueue_attempted"),
        "C05-5 must permit the 2-3 fixed-fill nodes captured by its replay regression"
    );

    let safe_exec = graph
        .split("impl<'stream, 'buffer> GraphExec<'stream, 'buffer>")
        .nth(1)
        .expect("safe graph exec must remain present")
        .split("/// Borrowed completion owner")
        .next()
        .expect("safe graph exec must end before graph-launch declaration");
    assert!(
        safe_exec.contains("pub fn launch<'exec>(")
            && safe_exec.contains("self.native.launch(&mut stream.native)?"),
        "safe graph replay must source its stream solely from the retained capture lease"
    );
    assert!(
        !safe_exec.contains("launch(&mut self, stream:"),
        "safe graph replay must not accept a substitutable foreign stream"
    );

    for required in [
        "RileyCudaGraphLaunch",
        "fill_buffer",
        "fill_element_count",
        "fill_enqueue_count",
        "launch_in_flight",
        "owns_capture_leases",
    ] {
        assert!(
            internal.contains(required),
            "missing fixed-buffer graph ownership contract: {required}"
        );
    }
}

#[test]
fn owned_indexed_rope_bf16_graph_has_fixed_five_buffer_lifecycle() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_layout = include_str!("../../../kernels/tests/abi_layout.c");
    let abi_link = include_str!("abi_link.rs");

    for required in [
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_INDEXED_ROPE_BF16",
        "riley_cuda_graph_capture_begin_indexed_rope_bf16",
        "riley_cuda_graph_capture_enqueue_indexed_rope_bf16",
        "positions_mirror_len",
        "active_row_count",
        "rotary_dimension",
        "table_position_count",
    ] {
        assert!(header.contains(required), "missing C05-17 ABI: {required}");
    }
    for required in [
        "riley_cuda_graph_capture_begin_indexed_rope_bf16",
        "riley_cuda_graph_capture_enqueue_indexed_rope_bf16",
        "begin_graph_indexed_rope_bf16_capture",
        "enqueue_indexed_rope_bf16",
    ] {
        assert!(
            ffi.contains(required),
            "missing C05-17 Rust FFI boundary: {required}"
        );
    }
    for required in [
        "graph_capture_begin_indexed_rope_bf16_symbol",
        "graph_capture_enqueue_indexed_rope_bf16_symbol",
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_INDEXED_ROPE_BF16 == 11",
    ] {
        assert!(
            abi_layout.contains(required),
            "C05-17 ABI layout/linkage witness is missing: {required}"
        );
    }
    assert!(abi_link.contains("CudaGraphCaptureOperation::IndexedRopeBf16"));
    assert!(graph.contains("IndexedRopeBf16 = 11"));
    for required in [
        "kIndexedRopeBf16 = 11",
        "indexed_rope_bf16_input",
        "indexed_rope_bf16_cos",
        "indexed_rope_bf16_sin",
        "indexed_rope_bf16_positions",
        "indexed_rope_bf16_enqueue_count",
        "indexed_rope_bf16_input_lease_held",
        "indexed_rope_bf16_positions_lease_held",
    ] {
        assert!(
            internal.contains(required),
            "C05-17 native fixed-address ownership state is missing: {required}"
        );
    }
    assert!(
        !internal.contains("indexed_rope_bf16_positions_mirror"),
        "the temporary host position mirror must not survive native capture admission"
    );
    for required in [
        "release_capture_indexed_rope_bf16_leases",
        "release_graph_indexed_rope_bf16_leases",
        "indexed_rope_bf16_capture_state_is_valid",
        "indexed_rope_bf16_graph_state_is_valid",
        "indexed_rope_bf16_exec_state_is_valid",
    ] {
        assert!(
            native.contains(required),
            "C05-17 native lifecycle is missing: {required}"
        );
    }

    let begin = native
        .split("RileyCudaStatus capture_begin_indexed_rope_bf16_impl(")
        .nth(1)
        .expect("C05-17 native begin implementation must remain present")
        .split("}  // namespace")
        .next()
        .expect("C05-17 native begin implementation must end before exported wrappers");
    for required in [
        "positions_mirror == nullptr || positions_mirror_len != active_row_count",
        "positions_mirror[row]",
        "try_acquire_exclusive_use(input->active_uses)",
        "try_acquire_exclusive_use(cos->active_uses)",
        "try_acquire_exclusive_use(sin->active_uses)",
        "try_acquire_exclusive_use(positions->active_uses)",
        "try_acquire_exclusive_use(output->active_uses)",
        "cudaStreamBeginCapture",
        "RileyCudaGraphCaptureOperation::kIndexedRopeBf16",
    ] {
        assert!(
            begin.contains(required),
            "C05-17 native begin must retain exact validation/ownership: {required}"
        );
    }
    let enqueue = native_export_body(native, "riley_cuda_graph_capture_enqueue_indexed_rope_bf16");
    assert_eq!(
        enqueue.matches("graph_indexed_rope_bf16<<<").count(),
        1,
        "C05-17 capture enqueue must record exactly one indexed-RoPE node"
    );
    for required in [
        "indexed_rope_bf16_enqueue_count != 0",
        "cudaGetLastError",
        "kIndexedRopeBf16EnqueueTerminal",
    ] {
        assert!(
            enqueue.contains(required),
            "C05-17 enqueue lifecycle is missing: {required}"
        );
    }
    for forbidden in [
        "std::calloc",
        "std::free",
        "cudaStreamSynchronize",
        "cudaMemcpyAsync",
        "riley_cuda_indexed_rope_execute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "C05-17 one-node enqueue must stay allocation-free and capture-safe: {forbidden}"
        );
    }

    for required in [
        "pub struct OwnedGraphIndexedRopeBf16Resources",
        "pub struct OwnedGraphIndexedRopeBf16CaptureBeginError",
        "pub struct OwnedGraphIndexedRopeBf16Capture",
        "pub struct OwnedCapturedIndexedRopeBf16Graph",
        "pub struct OwnedGraphIndexedRopeBf16Exec",
        "pub struct OwnedGraphIndexedRopeBf16Launch<'exec>",
        "pub fn begin_owned_graph_indexed_rope_bf16_capture",
        "pub fn enqueue_indexed_rope_bf16(&mut self)",
    ] {
        assert!(
            graph.contains(required),
            "C05-17 safe owner contract is missing: {required}"
        );
    }
    let preflight = graph
        .split("fn validate_graph_indexed_rope_bf16_capture_preflight(")
        .nth(1)
        .expect("C05-17 indexed-RoPE preflight must remain present")
        .split("/// A by-value stream and nine distinct fixed device buffers")
        .next()
        .expect("C05-17 preflight must end before C05-18 resources");
    for required in [
        "positions_host",
        "active_row_count == 0",
        "rotary_dimension > head_size || rotary_dimension % 2 != 0",
        "crate::batch::validate_indexed_positions(positions_host, table_position_count)",
        "positions_host length must exactly equal active_row_count",
        "tensor_bytes",
        "table_bytes",
        "positions_bytes",
    ] {
        assert!(
            preflight.contains(required),
            "C05-17 preflight must retain exact fixed-shape validation: {required}"
        );
    }
    assert_eq!(
        preflight.matches("same_allocation").count(),
        10,
        "C05-17 must reject every alias across five fixed device allocations"
    );
    assert_eq!(
        preflight.matches("ensure_same_context").count(),
        5,
        "C05-17 must bind every fixed device allocation to the capture context"
    );
    assert_eq!(
        preflight.matches("ensure_idle_for_operation").count(),
        5,
        "C05-17 must require every fixed device allocation to be idle"
    );

    let resources = graph
        .split("impl OwnedGraphIndexedRopeBf16Resources")
        .nth(1)
        .expect("C05-17 resource sextet must remain present")
        .split("/// Error from beginning an owned fixed-address BF16 indexed-RoPE graph")
        .next()
        .expect("C05-17 resource sextet must end before its begin error");
    for (earlier, later) in [
        ("output.close()?", "positions.close()?"),
        ("positions.close()?", "sin.close()?"),
        ("sin.close()?", "cos.close()?"),
        ("cos.close()?", "input.close()?"),
        ("input.close()?", "stream.close()"),
    ] {
        assert_precedes(resources, earlier, later, "C05-17 resource close order");
    }
    assert!(
        !resources.contains("positions_host"),
        "the C05-17 temporary host position mirror must not be retained by resources"
    );

    let owned_begin = graph
        .split("pub fn begin_owned_graph_indexed_rope_bf16_capture")
        .nth(1)
        .expect("owned C05-17 graph capture entry point must remain present")
        .split("/// Begins one C05-18 fixed-address BF16 ragged paged-K/V cache-write graph")
        .next()
        .expect("owned C05-17 capture must precede the C05-18 capture");
    assert_precedes(
        owned_begin,
        "validate_graph_indexed_rope_bf16_capture_preflight",
        "begin_graph_indexed_rope_bf16_capture",
        "C05-17 Rust preflight",
    );
    assert!(
        owned_begin.contains("let active_row_count = match u64::try_from(positions_host.len())")
    );
    assert!(
        !owned_begin.contains("active_row_count: u64"),
        "C05-17 must derive active rows only from the temporary host mirror"
    );

    for owner in [
        "pub struct OwnedGraphIndexedRopeBf16Capture {",
        "pub struct OwnedCapturedIndexedRopeBf16Graph {",
        "pub struct OwnedGraphIndexedRopeBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphIndexedRopeBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before fixed resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphIndexedRopeBf16Exec")
        .nth(1)
        .expect("C05-17 owned exec must remain present")
        .split("/// Completion owner for one indexed-RoPE graph executable replay")
        .next()
        .expect("C05-17 owned exec must end before its completion owner");
    for forbidden in [
        "launch_with_input",
        "launch_with_source",
        "CudaBufferSpan",
        "CudaBufferSpanMut",
        "GpuGreedy",
        "CompletionBoundary",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !owned_exec.contains(forbidden),
            "C05-17 must not expose mutable replay or C07 execution capability: {forbidden}"
        );
    }
}

#[test]
fn owned_ragged_paged_kv_cache_write_bf16_graph_has_fixed_nine_buffer_lifecycle() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_link = include_str!("abi_link.rs");

    for required in [
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_RAGGED_PAGED_KV_CACHE_WRITE_BF16",
        "riley_cuda_graph_capture_begin_ragged_paged_kv_cache_write_bf16",
        "riley_cuda_graph_capture_enqueue_ragged_paged_kv_cache_write_bf16",
        "RileyCudaDeviceBuffer* sequence_block_offsets",
        "RileyCudaDeviceBuffer* valid_tokens",
        "uint64_t physical_block_count",
        "uint64_t key_value_head_count",
    ] {
        assert!(header.contains(required), "missing C05-18 ABI: {required}");
    }
    for required in [
        "riley_cuda_graph_capture_begin_ragged_paged_kv_cache_write_bf16",
        "riley_cuda_graph_capture_enqueue_ragged_paged_kv_cache_write_bf16",
        "begin_graph_ragged_paged_kv_cache_write_bf16_capture",
        "enqueue_ragged_paged_kv_cache_write_bf16",
    ] {
        assert!(
            ffi.contains(required),
            "missing C05-18 Rust FFI boundary: {required}"
        );
    }
    assert!(
        abi_link.contains("CudaGraphCaptureOperation::RaggedPagedKvCacheWriteBf16"),
        "C05-18 capability must remain ABI-linked without device initialization"
    );
    assert!(graph.contains("RaggedPagedKvCacheWriteBf16 = 12"));

    for required in [
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Resources",
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError",
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Capture",
        "pub struct OwnedCapturedRaggedPagedKvCacheWriteBf16Graph",
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Exec",
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Launch<'exec>",
        "pub fn begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture",
        "pub fn enqueue_ragged_paged_kv_cache_write_bf16(&mut self)",
    ] {
        assert!(
            graph.contains(required),
            "C05-18 safe owner contract is missing: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_ragged_paged_kv_cache_write_bf16_capture_preflight(")
        .nth(1)
        .expect("C05-18 ragged paged-K/V preflight must remain present")
        .split("impl CudaStream")
        .next()
        .expect("C05-18 preflight must precede stream entry points");
    for required in [
        "PackedBatchHostV1<'_>",
        "batch_host.format_version()",
        "batch_host.block_size()",
        "sequence_count == 0",
        "block_count == 0",
        "active_row_count == 0",
        "physical_block_count == 0",
        "key_value_head_count == 0",
        "head_size == 0",
        "let buffers = [",
        "same_allocation",
        "ensure_same_context",
        "ensure_idle_for_operation",
        "BF16_BYTES",
        "U32_BYTES",
        "U16_BYTES",
        "source_bytes",
        "pool_bytes",
        "offsets_bytes",
        "valid_tokens_bytes",
        "row_sequence_slots_bytes",
        "row_positions_bytes",
    ] {
        assert!(
            preflight.contains(required),
            "C05-18 preflight must retain fixed-address validation: {required}"
        );
    }
    for forbidden in [
        "batch_host.sequence_block_offsets()",
        "batch_host.block_ids()",
        "batch_host.valid_tokens()",
        "batch_host.row_sequence_slots()",
        "batch_host.row_positions()",
    ] {
        assert!(
            !preflight.contains(forbidden),
            "C05-18 host witness must not bind graph metadata identity: {forbidden}"
        );
    }

    let resources = graph
        .split("impl OwnedGraphRaggedPagedKvCacheWriteBf16Resources")
        .nth(1)
        .expect("C05-18 resource bundle must remain present")
        .split("/// Error from beginning an owned fixed-address BF16 ragged paged-K/V")
        .next()
        .expect("C05-18 resource bundle must end before its begin error");
    for required in [
        "key_source",
        "value_source",
        "key_pool",
        "value_pool",
        "sequence_block_offsets",
        "block_ids",
        "valid_tokens",
        "row_sequence_slots",
        "row_positions",
    ] {
        assert!(
            resources.contains(required),
            "C05-18 fixed resource bundle is missing: {required}"
        );
    }
    assert!(
        !resources.contains("PackedBatchHostV1"),
        "the temporary packed host witness must not be retained by C05-18 resources"
    );
    for (earlier, later) in [
        ("row_positions.close()?", "row_sequence_slots.close()?"),
        ("row_sequence_slots.close()?", "valid_tokens.close()?"),
        ("valid_tokens.close()?", "block_ids.close()?"),
        ("block_ids.close()?", "sequence_block_offsets.close()?"),
        ("sequence_block_offsets.close()?", "value_pool.close()?"),
        ("value_pool.close()?", "key_pool.close()?"),
        ("key_pool.close()?", "value_source.close()?"),
        ("value_source.close()?", "key_source.close()?"),
        ("key_source.close()?", "stream.close()"),
    ] {
        assert_precedes(resources, earlier, later, "C05-18 resource close order");
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_ragged_paged_kv_cache_write_bf16_capture")
        .nth(1)
        .expect("owned C05-18 graph capture entry point must remain present")
        .split("/// Begins one C05-19 fixed-address BF16 grouped ragged paged-attention")
        .next()
        .expect("owned C05-18 capture must precede the C05-19 capture");
    assert_precedes(
        owned_begin,
        "validate_graph_ragged_paged_kv_cache_write_bf16_capture_preflight",
        "begin_graph_ragged_paged_kv_cache_write_bf16_capture",
        "C05-18 Rust preflight",
    );
    for required in [
        "batch_host.sequence_count()",
        "batch_host.block_count()",
        "batch_host.active_row_count()",
        "batch_host.physical_block_count()",
    ] {
        assert!(
            owned_begin.contains(required),
            "C05-18 must derive native dimensions from the temporary host witness: {required}"
        );
    }
    assert!(
        !owned_begin.contains("sequence_count: u64"),
        "C05-18 must not accept a separately supplied sequence count"
    );

    for owner in [
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Capture {",
        "pub struct OwnedCapturedRaggedPagedKvCacheWriteBf16Graph {",
        "pub struct OwnedGraphRaggedPagedKvCacheWriteBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphRaggedPagedKvCacheWriteBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before fixed resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphRaggedPagedKvCacheWriteBf16Exec")
        .nth(1)
        .expect("C05-18 owned exec must remain present")
        .split("/// Completion owner for one ragged paged-K/V cache-write graph executable")
        .next()
        .expect("C05-18 owned exec must end before its completion owner");
    for forbidden in [
        "launch_with_input",
        "launch_with_source",
        "CudaBufferSpan",
        "CudaBufferSpanMut",
        "GpuGreedy",
        "CompletionBoundary",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !owned_exec.contains(forbidden),
            "C05-18 must not expose mutable replay or C07 execution capability: {forbidden}"
        );
    }
}

#[test]
fn owned_grouped_ragged_paged_attention_bf16_graph_has_fixed_nine_buffer_lifecycle() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_link = include_str!("abi_link.rs");

    for required in [
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_GROUPED_RAGGED_PAGED_ATTENTION_BF16",
        "riley_cuda_graph_capture_begin_grouped_ragged_paged_attention_bf16",
        "riley_cuda_graph_capture_enqueue_grouped_ragged_paged_attention_bf16",
        "RileyCudaDeviceBuffer* query",
        "RileyCudaDeviceBuffer* output",
        "uint64_t output_row_count",
        "float scale",
    ] {
        assert!(header.contains(required), "missing C05-19 ABI: {required}");
    }
    for required in [
        "riley_cuda_graph_capture_begin_grouped_ragged_paged_attention_bf16",
        "riley_cuda_graph_capture_enqueue_grouped_ragged_paged_attention_bf16",
        "begin_graph_grouped_ragged_paged_attention_bf16_capture",
        "enqueue_grouped_ragged_paged_attention_bf16",
    ] {
        assert!(
            ffi.contains(required),
            "missing C05-19 Rust FFI boundary: {required}"
        );
    }
    assert!(
        abi_link.contains("CudaGraphCaptureOperation::GroupedRaggedPagedAttentionBf16"),
        "C05-19 capability must remain ABI-linked without device initialization"
    );
    assert!(graph.contains("GroupedRaggedPagedAttentionBf16 = 13"));

    for required in [
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Resources",
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError",
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Capture",
        "pub struct OwnedCapturedGroupedRaggedPagedAttentionBf16Graph",
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Exec",
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Launch<'exec>",
        "pub fn begin_owned_graph_grouped_ragged_paged_attention_bf16_capture",
        "pub fn enqueue_grouped_ragged_paged_attention_bf16(&mut self)",
    ] {
        assert!(
            graph.contains(required),
            "C05-19 safe owner contract is missing: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_grouped_ragged_paged_attention_bf16_capture_preflight(")
        .nth(1)
        .expect("C05-19 grouped ragged paged-attention preflight must remain present")
        .split("fn take_owned_graph_grouped_ragged_paged_attention_bf16_resources")
        .next()
        .expect("C05-19 preflight must precede its resource recovery helper");
    for required in [
        "PackedBatchHostV1<'_>",
        "batch_host.format_version()",
        "batch_host.block_size()",
        "ATTENTION_HEAD_SIZE: u64 = 64",
        "sequence_count == 0",
        "block_count == 0",
        "active_row_count == 0",
        "physical_block_count == 0",
        "query_head_count == 0",
        "key_value_head_count == 0",
        "query_head_count % key_value_head_count",
        "output_row_count < active_row_count",
        "!scale.is_finite() || scale <= 0.0",
        "let buffers = [",
        "same_allocation",
        "ensure_same_context",
        "ensure_idle_for_operation",
        "query_bytes",
        "pool_bytes",
        "output_bytes",
        "offsets_bytes",
        "valid_tokens_bytes",
        "row_sequence_slots_bytes",
        "row_positions_bytes",
    ] {
        assert!(
            preflight.contains(required),
            "C05-19 preflight must retain fixed-address D64 GQA validation: {required}"
        );
    }
    for forbidden in [
        "batch_host.sequence_block_offsets()",
        "batch_host.block_ids()",
        "batch_host.valid_tokens()",
        "batch_host.row_sequence_slots()",
        "batch_host.row_positions()",
    ] {
        assert!(
            !preflight.contains(forbidden),
            "C05-19 host witness must not bind graph metadata identity: {forbidden}"
        );
    }

    let resources = graph
        .split("impl OwnedGraphGroupedRaggedPagedAttentionBf16Resources")
        .nth(1)
        .expect("C05-19 resource bundle must remain present")
        .split("/// Error from beginning an owned fixed-address BF16 grouped ragged")
        .next()
        .expect("C05-19 resource bundle must end before its begin error");
    for required in [
        "query",
        "key_pool",
        "value_pool",
        "output",
        "sequence_block_offsets",
        "block_ids",
        "valid_tokens",
        "row_sequence_slots",
        "row_positions",
    ] {
        assert!(
            resources.contains(required),
            "C05-19 fixed resource bundle is missing: {required}"
        );
    }
    assert!(
        !resources.contains("PackedBatchHostV1"),
        "the temporary packed host witness must not be retained by C05-19 resources"
    );
    for (earlier, later) in [
        ("row_positions.close()?", "row_sequence_slots.close()?"),
        ("row_sequence_slots.close()?", "valid_tokens.close()?"),
        ("valid_tokens.close()?", "block_ids.close()?"),
        ("block_ids.close()?", "sequence_block_offsets.close()?"),
        ("sequence_block_offsets.close()?", "output.close()?"),
        ("output.close()?", "value_pool.close()?"),
        ("value_pool.close()?", "key_pool.close()?"),
        ("key_pool.close()?", "query.close()?"),
        ("query.close()?", "stream.close()"),
    ] {
        assert_precedes(resources, earlier, later, "C05-19 resource close order");
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_grouped_ragged_paged_attention_bf16_capture")
        .nth(1)
        .expect("owned C05-19 graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("C05-19 owned capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_grouped_ragged_paged_attention_bf16_capture_preflight",
        "begin_graph_grouped_ragged_paged_attention_bf16_capture",
        "C05-19 Rust preflight",
    );
    for required in [
        "batch_host.sequence_count()",
        "batch_host.block_count()",
        "batch_host.active_row_count()",
        "batch_host.physical_block_count()",
    ] {
        assert!(
            owned_begin.contains(required),
            "C05-19 must derive native dimensions from the temporary host witness: {required}"
        );
    }
    assert!(
        !owned_begin.contains("head_size: u64"),
        "C05-19 must expose exactly the fixed D64 graph ABI rather than a caller-selected head size"
    );

    for owner in [
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Capture {",
        "pub struct OwnedCapturedGroupedRaggedPagedAttentionBf16Graph {",
        "pub struct OwnedGraphGroupedRaggedPagedAttentionBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphGroupedRaggedPagedAttentionBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before fixed resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let owned_exec = graph
        .split("impl OwnedGraphGroupedRaggedPagedAttentionBf16Exec")
        .nth(1)
        .expect("C05-19 owned exec must remain present")
        .split("/// Completion owner for one grouped ragged paged-attention graph executable")
        .next()
        .expect("C05-19 owned exec must end before its completion owner");
    for forbidden in [
        "launch_with_input",
        "launch_with_source",
        "CudaBufferSpan",
        "CudaBufferSpanMut",
        "GpuGreedy",
        "CompletionBoundary",
        "graph_decode",
        "llama",
    ] {
        assert!(
            !owned_exec.contains(forbidden),
            "C05-19 must not expose mutable replay or C07 execution capability: {forbidden}"
        );
    }
}

#[test]
fn owned_bf16_embedding_status_d2h_graph_has_fixed_validation_receipt_lifecycle() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_link = include_str!("abi_link.rs");

    for required in [
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_EMBEDDING_STATUS_D2H",
        "riley_cuda_graph_capture_begin_bf16_embedding_status_d2h",
        "riley_cuda_graph_capture_enqueue_bf16_embedding_status_d2h",
        "riley_cuda_graph_exec_read_bf16_embedding_status_d2h_report",
        "RileyCudaDeviceBuffer* table",
        "RileyCudaDeviceBuffer* token_ids",
        "RileyCudaDeviceBuffer* output",
        "RileyCudaDeviceBuffer* device_error_scratch",
        "RileyCudaPinnedHostBuffer* pinned_report",
        "uint64_t token_count",
        "uint64_t vocabulary_size",
        "uint64_t hidden_size",
    ] {
        assert!(header.contains(required), "missing C05-20 ABI: {required}");
    }
    for required in [
        "riley_cuda_graph_capture_begin_bf16_embedding_status_d2h",
        "riley_cuda_graph_capture_enqueue_bf16_embedding_status_d2h",
        "riley_cuda_graph_exec_read_bf16_embedding_status_d2h_report",
        "begin_graph_bf16_embedding_status_d2h_capture",
        "enqueue_bf16_embedding_status_d2h",
        "read_bf16_embedding_status_d2h_report",
    ] {
        assert!(
            ffi.contains(required),
            "missing C05-20 Rust FFI boundary: {required}"
        );
    }
    assert!(
        abi_link.contains("CudaGraphCaptureOperation::Bf16EmbeddingStatusD2H"),
        "C05-20 capability must remain ABI-linked without device initialization"
    );
    assert!(graph.contains("Bf16EmbeddingStatusD2H = 14"));

    for required in [
        "pub enum Bf16EmbeddingStatusD2HStatus",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HResources",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HCapture",
        "pub struct OwnedCapturedBf16EmbeddingStatusD2HGraph",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HExec",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HLaunch<'exec>",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HCompletion<'exec>",
        "pub fn begin_owned_graph_bf16_embedding_status_d2h_capture",
        "pub fn enqueue_bf16_embedding_status_d2h(&mut self)",
        "pub fn read_status(&mut self) -> CudaResult<Bf16EmbeddingStatusD2HStatus>",
    ] {
        assert!(
            graph.contains(required),
            "C05-20 safe owner contract is missing: {required}"
        );
    }

    let preflight = graph
        .split("fn validate_graph_bf16_embedding_status_d2h_capture_preflight(")
        .nth(1)
        .expect("C05-20 embedding-status preflight must remain present")
        .split("impl CudaStream")
        .next()
        .expect("C05-20 embedding-status preflight must end before CudaStream methods");
    for required in [
        "token_count == 0",
        "vocabulary_size == 0",
        "hidden_size == 0",
        "table_elements",
        "output_elements",
        "table_bytes",
        "token_bytes",
        "output_bytes",
        "let buffers = [",
        "same_allocation",
        "ensure_same_context",
        "ensure_idle_for_operation",
        "device_error_scratch.byte_len() != BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES",
        "pinned_report.byte_len() != BF16_EMBEDDING_STATUS_D2H_REPORT_BYTES",
    ] {
        assert!(
            preflight.contains(required),
            "C05-20 preflight must retain fixed BF16 validation/status admission: {required}"
        );
    }
    assert!(
        !preflight.contains("token_ids_host"),
        "C05-20 must validate raw fixed device IDs instead of binding a host token mirror"
    );

    let resources = graph
        .split("impl OwnedGraphBf16EmbeddingStatusD2HResources")
        .nth(1)
        .expect("C05-20 resource bundle must remain present")
        .split("/// Error from beginning one by-value fixed-address C05-20 graph capture")
        .next()
        .expect("C05-20 resource bundle must end before its begin error");
    for required in [
        "table",
        "token_ids",
        "output",
        "device_error_scratch",
        "pinned_report",
    ] {
        assert!(
            resources.contains(required),
            "C05-20 fixed resource bundle is missing: {required}"
        );
    }
    for (earlier, later) in [
        ("pinned_report.close()?", "device_error_scratch.close()?"),
        ("device_error_scratch.close()?", "output.close()?"),
        ("output.close()?", "token_ids.close()?"),
        ("token_ids.close()?", "table.close()?"),
        ("table.close()?", "stream.close()"),
    ] {
        assert_precedes(resources, earlier, later, "C05-20 resource close order");
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_bf16_embedding_status_d2h_capture")
        .nth(1)
        .expect("owned C05-20 graph capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("C05-20 owned capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_graph_bf16_embedding_status_d2h_capture_preflight",
        "begin_graph_bf16_embedding_status_d2h_capture",
        "C05-20 Rust preflight",
    );
    assert!(
        !owned_begin.contains("token_ids_host"),
        "C05-20 public begin must not accept a host token-ID mirror"
    );

    for owner in [
        "pub struct OwnedGraphBf16EmbeddingStatusD2HCapture {",
        "pub struct OwnedCapturedBf16EmbeddingStatusD2HGraph {",
        "pub struct OwnedGraphBf16EmbeddingStatusD2HExec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphBf16EmbeddingStatusD2HResources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before fixed resources"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }

    let completion = graph
        .split("impl OwnedGraphBf16EmbeddingStatusD2HCompletion")
        .nth(1)
        .expect("C05-20 completion receipt must remain present")
        .split("fn take_owned_graph_bf16_embedding_status_d2h_resources")
        .next()
        .expect("C05-20 completion receipt must precede resource recovery helper");
    for required in [
        "BF16_EMBEDDING_STATUS_D2H_REPORT_NONE",
        "BF16_EMBEDDING_STATUS_D2H_REPORT_TOKEN_OUT_OF_RANGE",
        "Bf16EmbeddingStatusD2HStatus::Success",
        "Bf16EmbeddingStatusD2HStatus::TokenOutOfRange",
        "self.exec.terminal = true",
        "read_bf16_embedding_status_d2h_report",
    ] {
        assert!(
            completion.contains(required),
            "C05-20 receipt must preserve typed status/fail-closed boundary: {required}"
        );
    }
}

#[test]
fn owned_canonical_gemm_graph_has_by_value_plan_and_four_buffer_lifecycle() {
    let header = include_str!("../../../kernels/include/riley_cuda.h");
    let ffi = include_str!("../src/ffi.rs");
    let gemm = include_str!("../src/gemm.rs");
    let graph = include_str!("../src/graph.rs");
    let abi_link = include_str!("abi_link.rs");

    for required in [
        "RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_CANONICAL_GEMM_BF16",
        "riley_cuda_graph_capture_begin_canonical_gemm_bf16",
        "riley_cuda_graph_capture_enqueue_canonical_gemm_bf16",
        "RileyCudaGemmPlan* plan",
        "RileyCudaDeviceBuffer* input",
        "RileyCudaDeviceBuffer* weight",
        "RileyCudaDeviceBuffer* output",
        "RileyCudaDeviceBuffer* workspace",
    ] {
        assert!(header.contains(required), "missing C05-21 ABI: {required}");
    }
    for required in [
        "riley_cuda_graph_capture_begin_canonical_gemm_bf16",
        "riley_cuda_graph_capture_enqueue_canonical_gemm_bf16",
        "begin_graph_canonical_gemm_bf16_capture",
        "enqueue_canonical_gemm_bf16",
    ] {
        assert!(
            ffi.contains(required),
            "missing C05-21 Rust FFI: {required}"
        );
    }
    assert!(
        abi_link.contains("CudaGraphCaptureOperation::CanonicalGemmBf16"),
        "C05-21 capability must remain ABI-linked without device initialization"
    );
    assert!(graph.contains("CanonicalGemmBf16 = 15"));

    for required in [
        "pub struct OwnedGraphCanonicalGemmBf16Resources",
        "pub struct OwnedGraphCanonicalGemmBf16CaptureBeginError",
        "pub struct OwnedGraphCanonicalGemmBf16Capture",
        "pub struct OwnedCapturedCanonicalGemmBf16Graph",
        "pub struct OwnedGraphCanonicalGemmBf16Exec",
        "pub struct OwnedGraphCanonicalGemmBf16Launch<'exec>",
        "pub fn begin_owned_graph_canonical_gemm_bf16_capture",
        "pub fn enqueue_canonical_gemm_bf16(&mut self)",
        "pub fn finish(mut self) -> CudaResult<()>",
    ] {
        assert!(
            graph.contains(required),
            "C05-21 safe owner contract is missing: {required}"
        );
    }
    for required in [
        "fn validate_canonical_graph_capture",
        "fn begin_canonical_graph_capture_native",
        "CudaGemmReductionPolicy::StrictNoSplitV1",
        "allocation must be exactly",
        "must be distinct fixed device allocations",
        "same_allocation",
        "ensure_idle_for_operation",
    ] {
        assert!(
            gemm.contains(required),
            "C05-21 GEMM preflight/helper is missing: {required}"
        );
    }

    let resources = graph
        .split("impl OwnedGraphCanonicalGemmBf16Resources")
        .nth(1)
        .expect("C05-21 resource bundle must remain present")
        .split("/// Error from beginning one by-value C05-21 canonical GEMM graph capture")
        .next()
        .expect("C05-21 resource bundle must end before its begin error");
    for required in ["plan", "input", "weight", "output", "workspace"] {
        assert!(
            resources.contains(required),
            "C05-21 fixed resource bundle is missing: {required}"
        );
    }
    for (earlier, later) in [
        ("plan.close()?", "workspace.close()?"),
        ("workspace.close()?", "output.close()?"),
        ("output.close()?", "weight.close()?"),
        ("weight.close()?", "input.close()?"),
        ("input.close()?", "stream.close()"),
    ] {
        assert_precedes(resources, earlier, later, "C05-21 resource close order");
    }

    let owned_begin = graph
        .split("pub fn begin_owned_graph_canonical_gemm_bf16_capture")
        .nth(1)
        .expect("C05-21 owned capture entry point must remain present")
        .split("/// Begins the sole C05-5 capture-admitted operation set")
        .next()
        .expect("C05-21 owned capture must precede borrowed fill capture");
    assert_precedes(
        owned_begin,
        "validate_canonical_graph_capture",
        "begin_canonical_graph_capture_native",
        "C05-21 Rust preflight",
    );
    for forbidden in ["CudaBufferSpan", "CudaBufferSpanMut", "alpha", "beta"] {
        assert!(
            !owned_begin.contains(forbidden),
            "C05-21 public graph begin must not admit dynamic {forbidden}"
        );
    }

    for owner in [
        "pub struct OwnedGraphCanonicalGemmBf16Capture {",
        "pub struct OwnedCapturedCanonicalGemmBf16Graph {",
        "pub struct OwnedGraphCanonicalGemmBf16Exec {",
    ] {
        let source = graph
            .split(owner)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {owner}"));
        let native_position = source
            .find("native:")
            .unwrap_or_else(|| panic!("{owner} must retain native ownership first"));
        let resources_position = source
            .find("resources: Option<OwnedGraphCanonicalGemmBf16Resources>")
            .unwrap_or_else(|| panic!("{owner} must retain graph resources by value"));
        assert!(
            native_position < resources_position,
            "{owner} must drop native ownership before plan and fixed allocations"
        );
        assert!(
            source.contains("PhantomData<Rc<()>>"),
            "{owner} must remain !Send + !Sync"
        );
    }
}

fn native_export_body<'a>(source: &'a str, symbol: &str) -> &'a str {
    source
        .split(symbol)
        .nth(1)
        .unwrap_or_else(|| panic!("missing native CUDA entrypoint: {symbol}"))
        .split("extern \"C\" RileyCudaStatus")
        .next()
        .expect("native CUDA entrypoint must end before the next export")
}

#[test]
fn native_thread_local_gate_rejects_same_thread_cuda_work_before_driver_entry() {
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let host_runtime = include_str!("../../../kernels/src/host_runtime.cu");
    let current_context = internal
        .split("class CurrentContext final")
        .nth(1)
        .expect("native CurrentContext gate must remain present")
        .split("inline bool same_context")
        .next()
        .expect("CurrentContext gate must end before common helpers");
    let capture_gate = current_context
        .find("thread_has_active_graph_capture()")
        .expect("normal CurrentContext entry must observe the thread-local capture owner");
    let driver_entry = current_context
        .find("cuCtxGetCurrent")
        .expect("CurrentContext must preserve the driver entry boundary");
    assert!(capture_gate < driver_entry);
    assert!(current_context.contains("thread_graph_capture_is_owner(capture_owner)"));
    assert!(internal.contains("static thread_local RileyCudaGraphCapture* owner = nullptr"));
    assert!(internal.contains("clear_thread_graph_capture_owner"));
    assert!(internal.contains("next.compare_exchange_weak"));
    assert!(native.contains("try_publish_thread_graph_capture(capture)"));
    assert!(native.contains("thread_graph_capture_is_owner(owner)"));

    let release_owner = native
        .split("bool release_capture_owner")
        .nth(1)
        .expect("capture owner release helper must remain present")
        .split("bool capture_may_be_active_after_failed_begin")
        .next()
        .expect("capture owner release helper must precede capture observation");
    assert!(
        release_owner
            .find("clear_thread_graph_capture_owner(capture)")
            .expect("known capture recovery must clear its exact TLS owner")
            < release_owner
                .find("std::free(capture)")
                .expect("known capture recovery must free the native owner")
    );

    for symbol in [
        "riley_cuda_device_count",
        "riley_cuda_device_properties",
        "riley_cuda_context_create",
        "riley_cuda_stream_command_batch_begin",
    ] {
        assert!(
            native_export_body(host_runtime, symbol).contains("thread_has_active_graph_capture()"),
            "direct CUDA entrypoint must reject same-thread capture: {symbol}"
        );
    }
    let context_close = host_runtime
        .split("RileyCudaStatus context_close_impl(\n")
        .nth(1)
        .expect("context close implementation must remain present")
        .split("RileyCudaDeferredCloseResult deferred_context_close")
        .next()
        .expect("context close implementation must precede its deferred callback");
    assert!(
        context_close.contains("thread_has_active_graph_capture()"),
        "context-close implementation must reject ordinary same-thread capture before release"
    );
}

fn inline_bool_function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    source
        .split(signature)
        .nth(1)
        .unwrap_or_else(|| panic!("missing inline CUDA admission helper: {signature}"))
        .split("\ninline bool ")
        .next()
        .expect("inline CUDA admission helper must end before the next helper")
}

fn assert_precedes(source: &str, earlier: &str, later: &str, contract: &str) {
    let earlier_position = source
        .find(earlier)
        .unwrap_or_else(|| panic!("missing {contract} prerequisite: {earlier}"));
    let later_position = source
        .find(later)
        .unwrap_or_else(|| panic!("missing {contract} boundary: {later}"));
    assert!(
        earlier_position < later_position,
        "{contract} must keep `{earlier}` before `{later}`"
    );
}

fn assert_primary_context_admission_contract(internal: &str) {
    for required in [
        "struct RileyCudaCaptureDomain",
        "class CaptureDomainAdmissionGuard final",
        "class CaptureDomainControlLease final",
        "capture_domain_for_device",
        "active_captures",
        "pending_smoke_fills",
        "pending_copies",
        "CaptureLifecycleAdmissionGuard",
        "capture_lifecycle_active_captures",
        "capture_lifecycle_pending_lifecycles",
        "try_begin_capture_domain_pending_lifecycle",
        "release_capture_domain_pending_lifecycle",
    ] {
        assert!(
            internal.contains(required),
            "missing primary-context contract: {required}"
        );
    }

    let capture_admission =
        inline_bool_function_body(internal, "inline bool try_begin_capture_domain(\n");
    let capture_release =
        inline_bool_function_body(internal, "inline bool release_capture_domain_capture(\n");
    let pending_admission = inline_bool_function_body(
        internal,
        "inline bool try_begin_capture_domain_pending_lifecycle(\n",
    );
    let pending_release = inline_bool_function_body(
        internal,
        "inline bool release_capture_domain_pending_lifecycle(\n",
    );
    let smoke_admission = inline_bool_function_body(
        internal,
        "inline bool try_begin_capture_domain_smoke_fill(\n",
    );
    let copy_admission = inline_bool_function_body(
        internal,
        "inline bool try_begin_capture_domain_pending_copy(\n",
    );
    let control_admission = internal
        .split("class CaptureDomainControlLease final")
        .nth(1)
        .expect("context-control lease must remain present")
        .split("inline bool same_context")
        .next()
        .expect("context-control lease must precede common helpers");

    // Capture and pending-token acquisition share the same global lock before
    // their per-primary-context lock. This protects safe Drop on a capture
    // thread even when the pending token belongs to another CUDA device.
    for (admission, global_counter, domain_counter, contract) in [
        (
            capture_admission,
            "try_increment_capture_domain_counter(global_active)",
            "try_increment_capture_domain_counter(domain->active_captures)",
            "capture admission",
        ),
        (
            pending_admission,
            "try_increment_capture_domain_counter(global_pending)",
            "try_increment_capture_domain_counter(pending_counter)",
            "pending-lifecycle admission",
        ),
    ] {
        assert_precedes(
            admission,
            "CaptureLifecycleAdmissionGuard lifecycle_admission",
            "CaptureDomainAdmissionGuard admission(domain)",
            contract,
        );
        assert_precedes(
            admission,
            "CaptureDomainAdmissionGuard admission(domain)",
            global_counter,
            contract,
        );
        assert_precedes(admission, global_counter, domain_counter, contract);
    }
    assert!(capture_admission.contains("global_pending.load"));
    assert!(capture_admission.contains("domain->pending_smoke_fills.load"));
    assert!(capture_admission.contains("domain->pending_copies.load"));
    assert!(pending_admission.contains("global_active.load"));
    assert!(pending_admission.contains("domain->active_captures.load"));
    assert!(pending_admission.contains("domain->broad_control_uses.load"));
    assert!(pending_admission.contains("release_capture_domain_counter(global_pending)"));
    assert!(capture_release.contains("release_capture_domain_counter(global_active)"));
    assert!(pending_release.contains("release_capture_domain_counter(global_pending)"));

    assert!(smoke_admission.contains("try_begin_capture_domain_pending_lifecycle"));
    assert!(smoke_admission.contains("domain->pending_smoke_fills"));
    assert!(copy_admission.contains("try_begin_capture_domain_pending_lifecycle"));
    assert!(copy_admission.contains("domain->pending_copies"));

    assert_precedes(
        control_admission,
        "CaptureDomainAdmissionGuard admission(domain)",
        "domain->active_captures.load",
        "primary-context control admission",
    );
    assert!(control_admission.contains("capture_terminated"));
    assert!(control_admission.contains("thread_graph_capture_is_owner"));
}

fn assert_context_controls_precede_cuda(native: &str, host_runtime: &str) {
    assert!(
        native
            .find("try_begin_capture_domain(capture->capture_domain)")
            .expect("capture must reserve its primary-context domain")
            < native
                .find("cudaStreamBeginCapture(stream->stream")
                .expect("native capture begin must remain present")
    );
    for (symbol, cuda_boundary) in [
        ("riley_cuda_context_create", "cuDevicePrimaryCtxRetain"),
        ("riley_cuda_context_synchronize", "CurrentContext scope"),
        ("riley_cuda_context_memory_info", "CurrentContext scope"),
    ] {
        let entry = native_export_body(host_runtime, symbol);
        assert!(
            entry
                .find("CaptureDomainControlLease")
                .expect("primary-context control must take the cross-thread capture lease")
                < entry.find(cuda_boundary).expect("missing CUDA boundary")
        );
    }
    let context_close = host_runtime
        .split("RileyCudaStatus context_close_impl(\n")
        .nth(1)
        .expect("context close implementation must remain present")
        .split("RileyCudaDeferredCloseResult deferred_context_close")
        .next()
        .expect("context close implementation must precede its deferred callback");
    assert_precedes(
        context_close,
        "CaptureDomainControlLease capture_control",
        "cuDevicePrimaryCtxRelease",
        "context-close primary-context control",
    );
}

fn assert_pending_smoke_admission_precedes_cuda() {
    let smoke_fill = include_str!("../../../kernels/src/smoke_fill.cu");
    let smoke_create = native_export_body(smoke_fill, "riley_cuda_smoke_buffer_create");
    let admission_lease = smoke_fill
        .split("class SmokeCaptureAdmissionLease final")
        .nth(1)
        .expect("smoke creation must retain an RAII pending-admission lease")
        .split("__global__ void smoke_fill_f32")
        .next()
        .expect("smoke admission lease must precede the kernel");
    assert!(admission_lease.contains("try_begin_capture_domain_smoke_fill"));
    assert!(admission_lease.contains("release_capture_domain_smoke_fill"));
    assert!(admission_lease.contains("transfer_to_buffer"));
    assert_precedes(
        smoke_create,
        "SmokeCaptureAdmissionLease capture_admission",
        "CurrentContext scope",
        "pending-smoke admission",
    );
    assert_precedes(
        smoke_create,
        "capture_admission.acquire()",
        "CurrentContext scope",
        "pending-smoke admission",
    );
    assert!(smoke_create.contains("element_count, false, true, nullptr"));
    assert!(smoke_create.contains("capture_admission.transfer_to_buffer()"));

    let smoke_launch = native_export_body(smoke_fill, "riley_cuda_smoke_fill_launch");
    assert!(
        smoke_launch.contains("!buffer->capture_admission_held"),
        "non-empty launch must require the allocation-time admission lease"
    );
    assert!(
        !smoke_launch.contains("try_begin_capture_domain_smoke_fill"),
        "a reused native smoke buffer must retain one create-time lease rather than incrementing per launch"
    );
}

fn assert_pending_copy_admission_precedes_cuda() {
    let memory = include_str!("../../../kernels/src/memory.cu");
    let submit = memory
        .split("RileyCudaStatus submit_copy(")
        .nth(1)
        .expect("native pending-copy submission helper must remain present")
        .split("RileyCudaStatus device_buffer_close_impl")
        .next()
        .expect("pending-copy submission must end before buffer close helpers");
    assert_precedes(
        submit,
        "try_begin_capture_domain_pending_copy(device->owner->capture_domain)",
        "void* copy_storage = std::calloc",
        "pending-copy admission",
    );
    assert_precedes(
        submit,
        "try_begin_capture_domain_pending_copy(device->owner->capture_domain)",
        "CurrentContext scope(device->owner)",
        "pending-copy admission",
    );
    assert!(submit.contains("release_capture_domain_pending_copy(device->owner->capture_domain)"));
    assert!(submit.contains("RileyCudaCopy(device->owner, stream, device, host)"));

    let close = native_export_body(memory, "riley_cuda_copy_close");
    assert!(close.contains("const bool held_capture_admission"));
    assert_precedes(
        close,
        "release_capture_domain_pending_copy(owner->capture_domain)",
        "const bool child_released = release_child(owner)",
        "pending-copy consumption",
    );
}

#[test]
fn primary_context_capture_admission_is_linearized_before_cuda() {
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let host_runtime = include_str!("../../../kernels/src/host_runtime.cu");
    assert_primary_context_admission_contract(internal);
    assert_context_controls_precede_cuda(native, host_runtime);
    assert_pending_smoke_admission_precedes_cuda();
    assert_pending_copy_admission_precedes_cuda();
}

#[test]
fn deferred_close_fifo_is_allocation_free_and_drains_after_physical_capture_end() {
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let host_runtime = include_str!("../../../kernels/src/host_runtime.cu");
    let memory = include_str!("../../../kernels/src/memory.cu");
    let gemm = include_str!("../../../kernels/src/gemm.cu");
    let attention = include_str!("../../../kernels/src/attention_cublaslt.cu");
    let rust_graph = include_str!("../src/graph.rs");

    for required in [
        "struct RileyCudaDeferredCloseNode",
        "struct RileyCudaDeferredCloseResult",
        "using RileyCudaDeferredCloseCallback",
        "initialize_capture_deferred_close_node",
        "enqueue_capture_deferred_close",
        "drain_capture_deferred_closes",
        "deferred_close_head",
        "deferred_close_tail",
        "capture_terminated",
    ] {
        assert!(
            internal.contains(required),
            "missing deferred-close FIFO contract: {required}"
        );
    }

    let enqueue = internal
        .split("inline CaptureDeferredCloseEnqueueResult enqueue_capture_deferred_close(")
        .nth(1)
        .expect("deferred-close enqueue helper must remain present")
        .split("\n// A consumed callback")
        .next()
        .expect("deferred-close enqueue must precede drain documentation");
    assert!(!enqueue.contains("malloc"));
    assert!(!enqueue.contains("calloc"));
    assert!(enqueue.contains("node->queued = true"));
    assert!(enqueue.contains("capture->deferred_close_tail->next = node"));
    assert!(enqueue.contains("capture->deferred_close_tail = node"));

    let drain = internal
        .split("inline RileyCudaStatus drain_capture_deferred_closes(")
        .nth(1)
        .expect("deferred-close drain helper must remain present")
        .split("\nclass CurrentContext final")
        .next()
        .expect("deferred-close drain must precede CurrentContext");
    assert_precedes(
        drain,
        "RileyCudaDeferredCloseNode* const next = node->next",
        "node->callback(node, capture, error)",
        "deferred-close FIFO drain",
    );
    assert!(drain.contains("if (result.consumed)"));
    assert!(drain.contains("capture->deferred_close_head = next"));
    assert!(drain.contains("return result.status"));

    let abort = native_export_body(native, "riley_cuda_graph_capture_abort");
    assert_precedes(
        abort,
        "cudaStreamEndCapture(owner->stream->stream, &returned_graph)",
        "owner->capture_terminated = true",
        "deferred-close capture recovery",
    );
    assert_precedes(
        abort,
        "cudaGraphDestroy(returned_graph)",
        "drain_capture_deferred_closes(owner, &deferred_close_error)",
        "deferred-close capture recovery",
    );
    assert_precedes(
        abort,
        "owner->capture_terminated = true",
        "drain_capture_deferred_closes(owner, &deferred_close_error)",
        "deferred-close capture recovery",
    );
    assert_precedes(
        abort,
        "drain_capture_deferred_closes(owner, &deferred_close_error)",
        "release_capture_owner(owner)",
        "deferred-close capture recovery",
    );

    let finish_contexts = rust_graph
        .split("fn finish_deferred_capture_contexts()")
        .nth(1)
        .expect("Rust capture-context ledger cleanup must remain present")
        .split("/// The only CUDA Graph capture mode")
        .next()
        .expect("capture-context ledger cleanup must precede public graph types");
    assert_precedes(
        finish_contexts,
        "ledger.active = false",
        "std::mem::take(&mut ledger.contexts)",
        "Rust capture-context ledger cleanup",
    );
    assert_precedes(
        finish_contexts,
        "std::mem::take(&mut ledger.contexts)",
        "drop(contexts)",
        "Rust capture-context ledger cleanup",
    );

    for (source, symbol, callback) in [
        (
            host_runtime,
            "riley_cuda_context_defer_to_active_capture",
            "deferred_context_close",
        ),
        (
            host_runtime,
            "riley_cuda_stream_defer_to_active_capture",
            "deferred_stream_close",
        ),
        (
            host_runtime,
            "riley_cuda_event_defer_to_active_capture",
            "deferred_event_close",
        ),
        (
            memory,
            "riley_cuda_device_buffer_defer_to_active_capture",
            "deferred_device_buffer_close",
        ),
        (
            memory,
            "riley_cuda_pinned_host_buffer_defer_to_active_capture",
            "deferred_pinned_host_buffer_close",
        ),
        (
            gemm,
            "riley_cuda_gemm_plan_defer_to_active_capture",
            "deferred_gemm_plan_close",
        ),
        (
            attention,
            "riley_cuda_hf_prefill_attention_plan_defer_to_active_capture",
            "deferred_hf_prefill_attention_plan_close",
        ),
    ] {
        let transfer = native_export_body(source, symbol);
        assert!(
            transfer.contains("initialize_capture_deferred_close_node"),
            "{symbol} must initialize its embedded deferred-close node"
        );
        assert!(
            transfer.contains(callback),
            "{symbol} must bind its resource-specific deferred close callback"
        );
        assert!(
            transfer.contains("enqueue_capture_deferred_close"),
            "{symbol} must transfer ownership to the active capture FIFO"
        );
        assert!(
            transfer.contains("CaptureDeferredCloseEnqueueResult::kQueued"),
            "{symbol} must null the caller handle only after FIFO ownership transfers"
        );
    }
}

#[test]
fn in_flight_smoke_close_uses_the_primary_context_control_lease() {
    let smoke_fill = include_str!("../../../kernels/src/smoke_fill.cu");
    let close = native_export_body(smoke_fill, "riley_cuda_smoke_buffer_close");
    let control = close
        .find("CaptureDomainControlLease capture_control")
        .expect("in-flight smoke close must take a primary-context control lease");
    let synchronize = close
        .find("cudaDeviceSynchronize()")
        .expect("in-flight smoke close must retain its device synchronization");
    assert!(
        close.contains("const bool requires_context_control = (*buffer)->in_flight"),
        "only an in-flight smoke close needs the context-wide synchronization lease"
    );
    assert!(
        control < synchronize,
        "smoke-buffer control lease must precede cudaDeviceSynchronize"
    );
    let release = close
        .find("release_capture_domain_smoke_fill")
        .expect("consumed smoke buffers must release their capture admission");
    let free = close
        .find("cudaFree((*buffer)->device_data)")
        .expect("smoke-buffer close must retain the native free boundary");
    assert!(
        free < release,
        "pending smoke admission must remain held through native buffer consumption"
    );
}

#[test]
fn command_batch_tls_gate_blocks_same_thread_capture() {
    let internal = include_str!("../../../kernels/src/ffi_internal.hpp");
    let native = include_str!("../../../kernels/src/graph.cu");
    let host_runtime = include_str!("../../../kernels/src/host_runtime.cu");
    assert!(internal.contains("thread_command_batch_count"));
    assert!(internal.contains("thread_has_active_command_batch"));
    let graph_begin = native
        .split("riley_cuda_graph_capture_begin")
        .nth(1)
        .expect("native graph-capture begin must remain present")
        .split("riley_cuda_graph_capture_abort")
        .next()
        .expect("native graph-capture begin must precede abort");
    assert!(
        graph_begin
            .find("thread_has_active_command_batch()")
            .expect("capture begin must reject a same-thread command batch")
            < graph_begin
                .find("command_batch_is_active(stream)")
                .expect("capture begin must retain its stream-local batch check")
    );

    let batch_begin = native_export_body(host_runtime, "riley_cuda_stream_command_batch_begin");
    assert!(
        batch_begin
            .find("try_publish_thread_command_batch()")
            .expect("command-batch begin must publish its TLS count")
            < batch_begin
                .find("command_batch_owner.compare_exchange_strong")
                .expect("command-batch begin must retain stream ownership publication")
    );
    let batch_end = native_export_body(host_runtime, "riley_cuda_stream_command_batch_end");
    assert!(
        batch_end
            .find("release_exclusive_use(stream->active_uses)")
            .expect("command-batch end must release its stream lease")
            < batch_end
                .rfind("release_thread_command_batch()")
                .expect("command-batch end must clear its TLS count after recovery")
    );
}
