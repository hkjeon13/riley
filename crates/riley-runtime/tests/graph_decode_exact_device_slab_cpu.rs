//! Source-level boundary contract for C07 exact V1 cold device-slab binding.

const GRAPH_DECODE_EXACT_DEVICE_SLAB_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_device_slab.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "CudaStream",
    "CudaCommandStream",
    "CudaCommandBatch",
    "CudaPendingH2D",
    "CudaPendingD2H",
    "GraphCapture",
    "CudaGraph",
    "begin_command_batch",
    "copy_from_pinned",
    "copy_to_pinned",
    "upload_from_slice",
    "download_to_slice",
    "write(",
    "read(",
    "to_vec",
    "LlamaPackedBatchMetadata",
    "PureDecodeGraphV1ExactHostSlabLease",
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
    "PureDecodeGraphV1ExactPinnedHostSlabLease",
    "PureDecodeGraphV1ExactDeviceSlab",
    "PureDecodeGraphV1ExactPinnedDeviceSlabBinding",
    "PureDecodeGraphV1ExactPinnedDeviceSlabBindingError",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataGeometryDigest",
    "CudaContext",
    "CudaDeviceBuffer",
    "CudaPinnedHostBuffer",
    "CudaResult",
    "prepare(",
    "allocate_device_buffer",
    "bind_pinned_host_lease",
    "LayoutMismatch",
    "device_buffer",
    "pinned_host_buffer",
    "close(self)",
];

#[test]
fn graph_decode_exact_device_slab_stays_a_cold_geometry_binding_boundary() {
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !GRAPH_DECODE_EXACT_DEVICE_SLAB_SOURCE.contains(forbidden),
            "C07 exact device slab crossed its cold binding boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            GRAPH_DECODE_EXACT_DEVICE_SLAB_SOURCE.contains(required),
            "C07 exact device slab omitted required cold-binding token {required:?}"
        );
    }
    assert_eq!(
        GRAPH_DECODE_EXACT_DEVICE_SLAB_SOURCE
            .matches("allocate_device_buffer")
            .count(),
        1,
        "C07 exact device slab must allocate exactly once while cold-preparing"
    );
    assert!(
        GRAPH_DECODE_EXACT_DEVICE_SLAB_SOURCE
            .contains("context.allocate_device_buffer(layout.total_bytes())"),
        "C07 exact device slab must cold-allocate exactly its layout byte length"
    );
    let (_, binding_tail) = GRAPH_DECODE_EXACT_DEVICE_SLAB_SOURCE
        .split_once("pub(crate) fn bind_pinned_host_lease")
        .expect("C07 exact device slab must expose its pinned/device binding method");
    for forbidden in [
        "allocate_device_buffer",
        "copy_from_pinned",
        "copy_to_pinned",
        "upload_from_slice",
        "download_to_slice",
        "write(",
        "read(",
        "Vec",
        "Box",
        "Arc",
    ] {
        assert!(
            !binding_tail.contains(forbidden),
            "C07 exact device slab binding path must not perform {forbidden:?}"
        );
    }
    assert!(
        binding_tail.contains("source.layout() != self.layout"),
        "C07 exact device slab must compare complete layouts before binding"
    );
    assert!(
        binding_tail.contains("device: &self.device"),
        "C07 exact device slab binding must retain the owner's device buffer"
    );
    assert!(
        binding_tail.contains("pinned: source.pinned_host_buffer()"),
        "C07 exact device slab binding must retain the successful pinned lease"
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains("#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-15"),
        "C07 exact device slab must remain CUDA-feature gated"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_device_slab;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_device_slab"));
}
