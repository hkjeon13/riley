//! Source-level boundary contract for C07 exact V1 command-batch H2D enqueue.

const GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_h2d_submission.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "CudaStream",
    "CudaCommandBatch",
    "CudaPendingH2D",
    "CudaPendingD2H",
    "CudaDeviceBuffer",
    "CudaPinnedHostBuffer",
    "begin_command_batch",
    "finish(",
    "synchronize",
    "copy_from_pinned_async",
    "copy_to_pinned",
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
    "PureDecodeGraphV1ExactPinnedDeviceSlabBinding",
    "CudaCommandStream",
    "CudaResult",
    "enqueue_pure_decode_graph_v1_exact_h2d",
    "copy_from_pinned_in_command_batch",
    "binding.layout().total_bytes()",
    "'batch",
    "'stream",
    "'device",
    "'pinned",
    "'device: 'stream",
    "'pinned: 'stream",
];

#[test]
fn graph_decode_exact_h2d_submission_stays_one_enqueue_without_completion() {
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE.contains(forbidden),
            "C07 exact H2D submission crossed its enqueue-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE.contains(required),
            "C07 exact H2D submission omitted required enqueue token {required:?}"
        );
    }
    assert_eq!(
        GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE
            .matches("copy_from_pinned_in_command_batch")
            .count(),
        1,
        "C07 exact H2D submission must enqueue exactly one command-batch copy"
    );
    assert!(
        GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE.contains(
            "binding: &'stream PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned>"
        ),
        "C07 exact H2D submission must borrow the two-lifetime binding for the stream lifetime"
    );
    assert!(
        GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE
            .contains("commands: &mut CudaCommandStream<'batch, 'stream>"),
        "C07 exact H2D submission must accept only an active command-stream proxy"
    );
    assert!(
        GRAPH_DECODE_EXACT_H2D_SUBMISSION_SOURCE.contains(
            "0,\n        binding.pinned_host_buffer(),\n        0,\n        binding.layout().total_bytes(),\n        commands,"
        ),
        "C07 exact H2D submission must copy the full exact slab from offset zero to zero"
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains("#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-16"),
        "C07 exact H2D submission must remain CUDA-feature gated"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_h2d_submission;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_h2d_submission"));
}
