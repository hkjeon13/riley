#![cfg(not(feature = "cuda"))]

use riley_cuda::{
    CapturedGraph, CudaDeviceBuffer, CudaErrorDomain, CudaErrorKind, CudaErrorStage,
    CudaGraphCaptureCapability, CudaGraphCaptureMode, CudaGraphLifecycle, CudaGraphLifecycleState,
    CudaGraphStage, CudaPinnedHostBuffer, CudaResult, CudaStream, GraphCapture, GraphExec,
    GraphFillCapture, GraphLaunch, OwnedCapturedGraph, OwnedCapturedH2DGraph,
    OwnedCapturedSiluBf16Graph, OwnedGraphExec, OwnedGraphFillCapture,
    OwnedGraphFillCaptureBeginError, OwnedGraphFillResources, OwnedGraphH2DCapture,
    OwnedGraphH2DCaptureBeginError, OwnedGraphH2DExec, OwnedGraphH2DLaunch, OwnedGraphH2DResources,
    OwnedGraphLaunch, OwnedGraphSiluBf16Capture, OwnedGraphSiluBf16CaptureBeginError,
    OwnedGraphSiluBf16Exec, OwnedGraphSiluBf16Launch, OwnedGraphSiluBf16Resources,
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
    for symbol in [
        "riley_cuda_graph_capture_begin_fill_f32",
        "riley_cuda_graph_capture_enqueue_fill_f32",
        "riley_cuda_graph_capture_begin_h2d",
        "riley_cuda_graph_capture_enqueue_h2d",
        "riley_cuda_graph_capture_begin_silu_bf16",
        "riley_cuda_graph_capture_enqueue_silu_bf16",
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
    assert!(ffi.contains("GraphCaptureHandle"));
    assert!(ffi.contains("graph_capture_begin_success_metadata_is_valid"));
    assert!(ffi.contains("graph_capture_abort_metadata_is_valid"));
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
    );

    let graph_source = include_str!("../src/graph.rs");
    assert!(graph_source.contains("CudaError::unavailable(\"CudaStream::begin_graph_capture\")"));
    assert!(graph_source.contains("\"CudaStream::begin_graph_fill_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_fill_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_h2d_capture\""));
    assert!(graph_source.contains("\"CudaStream::begin_owned_graph_silu_bf16_capture\""));
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
            "cudaStreamBeginCapture(stream->stream",
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
