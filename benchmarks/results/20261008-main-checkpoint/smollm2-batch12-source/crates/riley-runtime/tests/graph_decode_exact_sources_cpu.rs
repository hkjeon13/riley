//! Source-level boundary contract for C07 exact V1 opaque-region binding.

const GRAPH_DECODE_EXACT_SOURCES_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_sources.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "graph_decode_packer",
    "PureDecodeGraphMetadataFieldSources",
    "pack_pure_decode_graph_metadata_le",
    "LlamaPackedBatchMetadata",
    "PreparedLlamaBatchMetadata",
    "LlamaBatch",
    "project_pure_decode_graph_v1_exact",
    "preflight_pure_decode_graph_v1",
    "bind_pure_decode_graph_v1_preflight",
    "PureDecodeGraphV1LayoutBinding",
    "PureDecodeGraphMetadataBinding",
    "PureDecodeGraphMetadataLayout",
    "riley_model",
    "riley_tensor",
    "riley_cuda",
    "Cuda",
    "PreparedLlama",
    "LlamaBatchExecutor",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "select_pure_decode_graph_bucket",
    "PURE_DECODE_GRAPH_BUCKETS",
    "riley_scheduler",
    "riley_server",
    "Vec<",
    "Vec::new",
    "std::vec::Vec",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "Box<",
    "Box::new",
    "Arc<",
    "Arc::new",
    "Mutex<",
    "RwLock<",
    "String",
    "alloc::",
    "&mut",
    "get_mut",
    "fill(",
    "copy_from_slice",
    "unsafe",
    "extern \"C\"",
    "*const",
    "*mut",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1ExactNativeFields",
    "PureDecodeGraphMetadataField",
    "PureDecodeGraphV1ExactOpaqueField",
    "PureDecodeGraphV1ExactOpaqueSourceError",
    "PureDecodeGraphV1ExactMetadataSources",
    "FieldLengthMismatch",
    "Header",
    "ControlStatus",
    "binding()",
    "layout()",
    "region(",
    "byte_len()",
    "u64::try_from",
    "native_fields",
    "header",
    "control_status",
    "Result<",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_EXACT_SOURCES_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 exact opaque source must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 exact opaque source must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_exact_sources_stay_borrowed_and_outside_packing_or_execution() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 exact opaque source crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 exact opaque source omitted required checked token {required:?}"
        );
    }
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_sources;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_sources"));
}
