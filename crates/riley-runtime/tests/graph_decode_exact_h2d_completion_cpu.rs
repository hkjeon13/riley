//! Source-level boundary contract for C07 exact V1 H2D completion authority.

const GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_h2d_completion.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "CudaStream",
    "CudaCommandStream",
    "CudaPendingH2D",
    "CudaPendingD2H",
    "CudaPinnedHostBuffer",
    "begin_command_batch",
    "copy_from_pinned",
    "copy_to_pinned",
    "copy_from_pinned_async",
    "upload_from_slice",
    "download_to_slice",
    "allocate_device_buffer",
    "allocate_pinned_host_buffer",
    "byte_len",
    "write(",
    "read(",
    "to_vec",
    "LlamaPackedBatchMetadata",
    "PureDecodeGraphV1ExactHostSlabLease",
    "PureDecodeGraphV1ExactPinnedHostSlabLease",
    "PureDecodeGraphV1ExactDeviceSlab",
    "LayoutMismatch",
    "GraphCapture",
    "CudaGraph",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "riley_scheduler",
    "riley_server",
    "unsafe",
    "extern \"C\"",
    "*const",
    "*mut",
    "as_ptr",
    "as_mut_ptr",
    "Vec",
    "Box",
    "Arc",
    "map_err",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1ExactH2DEnqueued",
    "PureDecodeGraphV1ExactDeviceFreshLease",
    "PureDecodeGraphV1ExactPinnedDeviceSlabBinding",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataGeometryDigest",
    "CudaCommandBatch",
    "CudaDeviceBuffer",
    "CudaResult",
    "submit_pure_decode_graph_v1_exact_h2d",
    "enqueue_pure_decode_graph_v1_exact_h2d",
    "batch.commands()",
    "CudaResult<PureDecodeGraphV1ExactH2DEnqueued<'stream, 'device, 'pinned>>",
    "Ok(PureDecodeGraphV1ExactH2DEnqueued { binding, batch })",
    "finish_pure_decode_graph_v1_exact_h2d",
    "submitted.batch.finish()?",
    "binding.layout()",
    "binding.geometry_digest()",
    "binding.device_buffer()",
    "'stream",
    "'device",
    "'pinned",
    "'device: 'stream",
    "'pinned: 'stream",
];

#[test]
fn graph_decode_exact_h2d_completion_stays_receipt_bound_and_finish_first() {
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE.contains(forbidden),
            "C07 exact H2D completion crossed its finish-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE.contains(required),
            "C07 exact H2D completion omitted required completion token {required:?}"
        );
    }
    assert_eq!(
        GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
            .matches("batch.commands()")
            .count(),
        1,
        "C07 exact H2D completion must derive one command proxy from its owned batch"
    );
    assert_eq!(
        GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
            .matches("enqueue_pure_decode_graph_v1_exact_h2d(")
            .count(),
        1,
        "C07 exact H2D completion must delegate exactly once to the C07-16 enqueue primitive"
    );
    assert_eq!(
        GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
            .matches("submitted.batch.finish()?")
            .count(),
        1,
        "C07 exact H2D completion must observe its receipt-owned batch exactly once"
    );
    let finish = GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
        .find("submitted.batch.finish()?")
        .expect("C07 exact H2D completion must finish its receipt-owned batch");
    let lease = GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
        .find("Ok(PureDecodeGraphV1ExactDeviceFreshLease {")
        .expect("C07 exact H2D completion must construct a fresh device lease");
    assert!(
        finish < lease,
        "C07 exact H2D completion must not construct a fresh lease before finish succeeds"
    );
    let finish_function = &GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
        [GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE
            .find("fn finish_pure_decode_graph_v1_exact_h2d")
            .expect("C07 exact H2D completion must declare its finish entrypoint")..];
    assert!(
        !finish_function.contains(",\n    batch: CudaCommandBatch"),
        "C07 exact H2D completion must not accept a caller-substituted batch"
    );
    assert!(
        GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE.contains("batch: CudaCommandBatch<'stream>,"),
        "C07 exact H2D receipt must retain its exact batch by value"
    );
    assert!(
        !GRAPH_DECODE_EXACT_H2D_COMPLETION_SOURCE.contains("pub(crate) batch: CudaCommandBatch"),
        "C07 exact H2D receipt must not expose its owned batch outside its module"
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains("#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-17"),
        "C07 exact H2D completion must remain CUDA-feature gated"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_h2d_completion;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_h2d_completion"));
}
