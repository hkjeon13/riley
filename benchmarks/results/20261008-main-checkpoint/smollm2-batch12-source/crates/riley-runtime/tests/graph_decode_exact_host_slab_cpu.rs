//! Source-level boundary contract for C07 exact V1 cold host-slab ownership.

const GRAPH_DECODE_EXACT_HOST_SLAB_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_host_slab.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "riley_model",
    "riley_tensor",
    "riley_cuda",
    "Cuda",
    "Pinned",
    "Device",
    "H2D",
    "memcpy",
    "copy_from_slice",
    "PreparedLlama",
    "LlamaBatchExecutor",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "riley_scheduler",
    "riley_server",
    "PureDecodeGraphMetadataFieldSources",
    "pack_pure_decode_graph_metadata_le",
    "unsafe",
    "extern \"C\"",
    "*const",
    "*mut",
    "bytes_mut",
    "into_bytes",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1ExactHostSlab",
    "PureDecodeGraphV1ExactHostSlabLease",
    "PureDecodeGraphV1ExactHostSlabPrepareError",
    "PureDecodeGraphV1ExactHostSlabWrite",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataGeometryDigest",
    "LlamaPackedBatchMetadata",
    "prepare(",
    "try_reserve_exact",
    "into_boxed_slice",
    "align_offset",
    "layout.total_bytes()",
    "PureDecodeGraphMetadataLayout::required_base_alignment()",
    "geometry_digest",
    "bytes(&self) -> &[u8]",
    "write_exact_v1",
    "write_exact_v1_leased",
    "write_pure_decode_graph_v1_exact_metadata_le",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_EXACT_HOST_SLAB_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 exact host slab must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 exact host slab must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_exact_host_slab_stays_cold_owned_and_cpu_only() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 exact host slab crossed its cold CPU boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 exact host slab omitted required cold-owner token {required:?}"
        );
    }
    assert_eq!(
        production_source
            .matches("write_pure_decode_graph_v1_exact_metadata_le(")
            .count(),
        1,
        "C07 exact host slab must delegate exactly one write to C07-11"
    );
    assert_eq!(
        production_source.matches("try_reserve_exact").count(),
        1,
        "C07 exact host slab may allocate only while cold-preparing storage"
    );
    let (_, write_tail) = production_source
        .split_once("pub(crate) fn write_exact_v1")
        .expect("C07 exact host slab must expose its checked exact write method");
    for allocation_token in [
        "Vec::new",
        "try_reserve_exact",
        "resize(",
        "into_boxed_slice",
        "align_offset",
    ] {
        assert!(
            !write_tail.contains(allocation_token),
            "C07 exact host slab exact-write path must not allocate through {allocation_token:?}"
        );
    }
    let (_, leased_write_tail) = production_source
        .split_once("pub(crate) fn write_exact_v1_leased")
        .expect("C07 exact host slab must expose its leased exact write method");
    assert_eq!(
        leased_write_tail.matches("self.write_exact_v1(").count(),
        1,
        "C07 exact host slab leased write must delegate exactly once to C07-12"
    );
    assert!(
        !leased_write_tail.contains("write_pure_decode_graph_v1_exact_metadata_le"),
        "C07 exact host slab leased write must not bypass C07-12"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_host_slab;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_host_slab"));
}
