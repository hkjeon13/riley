//! Source-level boundary contract for C07 exact V1 pinned host-slab staging.

const GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_pinned_host_slab.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "CudaDeviceBuffer",
    "CudaStream",
    "CudaCommandStream",
    "CudaPendingH2D",
    "copy_from_pinned",
    "copy_to_pinned",
    "upload_from_slice",
    "read(",
    "to_vec",
    "LlamaPackedBatchMetadata",
    "write_pure_decode_graph_v1_exact_metadata_le",
    "pack_pure_decode_graph_metadata_le",
    "project_pure_decode_graph_v1_exact",
    "PureDecodeGraphMetadataFieldSources",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "unsafe",
    "extern \"C\"",
    "*const",
    "*mut",
    "as_ptr",
    "as_mut_ptr",
    "Vec",
    "Box",
    "Arc",
    "bytes_mut",
    "into_bytes",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1ExactHostSlabLease",
    "PureDecodeGraphV1ExactPinnedHostSlab",
    "PureDecodeGraphV1ExactPinnedHostSlabLease",
    "PureDecodeGraphV1ExactPinnedHostSlabStageError",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataGeometryDigest",
    "CudaContext",
    "CudaPinnedHostBuffer",
    "CudaError",
    "CudaResult",
    "prepare(",
    "allocate_pinned_host_buffer",
    "stage_from_host_lease",
    "LayoutMismatch",
    "PinnedWrite",
    "pinned_host_buffer",
    "close(self)",
];

#[test]
fn graph_decode_exact_pinned_host_slab_stays_a_cold_staging_boundary() {
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE.contains(forbidden),
            "C07 exact pinned host slab crossed its staging boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE.contains(required),
            "C07 exact pinned host slab omitted required staging token {required:?}"
        );
    }
    assert_eq!(
        GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE
            .matches("allocate_pinned_host_buffer")
            .count(),
        1,
        "C07 exact pinned host slab must allocate exactly once while cold-preparing"
    );
    assert!(
        GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE
            .contains("context.allocate_pinned_host_buffer(layout.total_bytes())"),
        "C07 exact pinned host slab must cold-allocate exactly its layout byte length"
    );
    assert_eq!(
        GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE
            .matches(".write(")
            .count(),
        1,
        "C07 exact pinned host slab must stage through exactly one pinned write"
    );
    let (_, stage_tail) = GRAPH_DECODE_EXACT_PINNED_HOST_SLAB_SOURCE
        .split_once("pub(crate) fn stage_from_host_lease")
        .expect("C07 exact pinned host slab must expose its staging method");
    for allocation_token in ["allocate_pinned_host_buffer", "Vec", "Box", "Arc"] {
        assert!(
            !stage_tail.contains(allocation_token),
            "C07 exact pinned host slab staging path must not allocate through {allocation_token:?}"
        );
    }
    assert!(
        stage_tail.contains("source.layout() != self.layout"),
        "C07 exact pinned host slab must compare complete layouts before staging"
    );
    assert!(
        stage_tail.contains("self.pinned\n            .write(0, source.bytes())"),
        "C07 exact pinned host slab must write the exact host lease at pinned offset zero"
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains("#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-14"),
        "C07 exact pinned host slab must remain CUDA-feature gated"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_pinned_host_slab;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_pinned_host_slab"));
}
