//! Source-level boundary contract for the C07-to-C06 metadata identity bridge.

const GRAPH_DECODE_LAYOUT_SIGNATURE_SOURCE: &str =
    include_str!("../src/llama/graph_decode_layout_signature.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "GraphLayoutSignature",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "GraphCapture",
    "Cuda",
    "LlamaPackedBatchMetadata",
    "PureDecodeGraphV1Exact",
    "graph_decode_packer",
    "pack_pure_decode_graph_metadata_le",
    "write_pure_decode_graph",
    "allocate",
    "Vec",
    "Box",
    "Arc",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "Mutex",
    "RwLock",
    "&mut",
    "write(",
    "read(",
    "unsafe",
    "extern \"C\"",
    "*const",
    "*mut",
    "as_ptr",
    "as_mut_ptr",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "GraphMetadataLayoutSignature",
    "PureDecodeGraphMetadataLayout",
    "pure_decode_graph_v1_metadata_layout_signature",
    "PureDecodeGraphMetadataLayout::schema_version()",
    "layout.geometry_digest().as_bytes()",
    "GraphMetadataLayoutSignature::new",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_LAYOUT_SIGNATURE_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 metadata identity bridge must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 metadata identity bridge must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_layout_signature_stays_a_cold_identity_bridge() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 metadata identity bridge crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 metadata identity bridge omitted required identity token {required:?}"
        );
    }
    assert_eq!(
        production_source
            .matches("GraphMetadataLayoutSignature::new")
            .count(),
        1,
        "C07 metadata identity bridge must derive one generic identity from one C07 layout"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_layout_signature;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_layout_signature"));
}
