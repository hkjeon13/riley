//! Source-level boundary contract for C07's exact-metadata C05 H2D owner.

const OWNER_SOURCE: &str = include_str!("../src/llama/graph_decode_c05_h2d_metadata_owner.rs");
const PINNED_SOURCE: &str = include_str!("../src/llama/graph_decode_exact_pinned_host_slab.rs");
const DEVICE_SOURCE: &str = include_str!("../src/llama/graph_decode_exact_device_slab.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1C05H2DMetadataProvenance",
    "PureDecodeGraphV1C05H2DColdResources",
    "PureDecodeGraphV1C05ExactMetadataH2DOwner",
    "PureDecodeGraphV1ExactHostSlabLease",
    "PureDecodeGraphV1ExactPinnedHostSlab",
    "PureDecodeGraphV1ExactDeviceSlab",
    "PureDecodeGraphV1C05OwnedH2DReplaySlot",
    "PureDecodeGraphV1C06SignatureBinding",
    "pure_decode_graph_v1_exact_metadata_layouts_match",
    "self.pinned.stage_from_host_lease(host_lease)",
    "pinned.into_c05_owned_graph_h2d_source()",
    "device.into_c05_owned_graph_h2d_destination()",
    "stream.begin_owned_graph_h2d_capture(",
    "CudaGraphCaptureMode::ThreadLocal",
    "capture.enqueue_h2d()",
    "capture.end()",
    "captured.instantiate()",
    "PureDecodeGraphV1C05OwnedH2DReplaySlot::new(",
    "self.resolver.resolve(selection)",
    "resolver.close()?",
    "Self::from_known_c05_graph_release(provenance, resources)",
    "provenance.payload_byte_len() != payload_byte_len",
];

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "launch_with_source",
    "exec_for_gpu_test",
    "CudaCommandBatch",
    "CudaCommandStream",
    "begin_command_batch",
    "graph_decode_exact_h2d_submission",
    "graph_decode_exact_h2d_completion",
    "copy_from_pinned",
    "copy_to_pinned",
    "select_registered_execution_graph",
    "GraphDispatchMetrics",
    "observe_pure_decode_graph_v1_c06_registry_dispatch",
    "batch_executor",
    "PreparedLlamaBatchExecutor",
    "CudaDeviceBuffer",
    "CudaPinnedHostBuffer",
    "unsafe",
    "extern \"C\"",
    "as_ptr",
    "as_mut_ptr",
    "Vec",
    "Box",
    "Arc",
    "Mutex",
    "pub use",
];

fn production_source() -> &'static str {
    OWNER_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07-26 owner must retain its unit-test boundary")
}

#[test]
fn graph_decode_c05_h2d_metadata_owner_stays_private_and_linear() {
    let source = production_source();
    for token in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            source.contains(token),
            "C07-26 metadata H2D owner omitted required token {token:?}",
        );
    }
    for token in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !source.contains(token),
            "C07-26 metadata H2D owner crossed its narrow boundary with {token:?}",
        );
    }

    let stage = source
        .find("self.pinned.stage_from_host_lease(host_lease)")
        .expect("C07-26 must synchronously stage the exact host lease");
    let source_move = source
        .find("pinned.into_c05_owned_graph_h2d_source()")
        .expect("C07-26 must move the staged pinned source into C05");
    let begin = source
        .find("stream.begin_owned_graph_h2d_capture(")
        .expect("C07-26 must start C05's owned H2D capture");
    let enqueue = source
        .find("capture.enqueue_h2d()")
        .expect("C07-26 must capture exactly C05's one H2D node");
    let end = source
        .find("capture.end()")
        .expect("C07-26 must end the captured H2D graph");
    let instantiate = source
        .find("captured.instantiate()")
        .expect("C07-26 must instantiate after capture ends");
    assert!(
        stage < source_move
            && source_move < begin
            && begin < enqueue
            && enqueue < end
            && end < instantiate,
        "C07-26 must stage first, then follow one linear C05 begin/enqueue/end/instantiate transition",
    );
    assert_eq!(
        source.matches("CudaGraphCaptureMode::ThreadLocal").count(),
        1,
        "C07-26 must internally fix the only reviewed capture mode exactly once",
    );
    assert!(
        !source.contains("mode: CudaGraphCaptureMode"),
        "C07-26 must not expose a caller-selectable capture mode",
    );
    assert!(
        PINNED_SOURCE.contains("into_c05_owned_graph_h2d_source"),
        "only a narrow pinned-owner consuming helper may supply C05's source",
    );
    assert!(
        PINNED_SOURCE.contains("recover_from_c05_owned_graph_h2d_source"),
        "known C05 close must rewrap the original typed pinned owner",
    );
    assert!(
        DEVICE_SOURCE.contains("into_c05_owned_graph_h2d_destination"),
        "only a narrow device-owner consuming helper may supply C05's destination",
    );
    assert!(
        DEVICE_SOURCE.contains("recover_from_c05_owned_graph_h2d_destination"),
        "known C05 close must rewrap the original typed device owner",
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains(
            "#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-26 binds exact C07 metadata slab provenance to that C05 H2D owner.\nmod graph_decode_c05_h2d_metadata_owner;"
        ),
        "C07-26 metadata H2D owner must stay CUDA-gated and private",
    );
    assert!(
        !LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c05_h2d_metadata_owner"),
        "C07-26 metadata H2D owner must not become a public runtime API",
    );
}
