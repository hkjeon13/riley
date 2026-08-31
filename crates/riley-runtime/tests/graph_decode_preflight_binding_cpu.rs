//! Source-level boundary contract for C07 V1 candidate-to-layout binding.

const GRAPH_DECODE_PREFLIGHT_BINDING_SOURCE: &str =
    include_str!("../src/llama/graph_decode_preflight_binding.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "LlamaPackedBatchMetadata",
    "preflight_pure_decode_graph_v1(",
    "graph_decode_packer",
    "PureDecodeGraphMetadataFieldSources",
    "pack_pure_decode_graph_metadata_le",
    "plan_pure_decode_graph_padding",
    "select_pure_decode_graph_bucket",
    "PURE_DECODE_GRAPH_BUCKETS",
    "riley_model",
    "riley_tensor",
    "riley_cuda",
    "Cuda",
    "PreparedLlama",
    "LlamaBatchExecutor",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
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
    "PureDecodeGraphV1Preflight",
    "PureDecodeGraphV1Ineligibility",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataBinding",
    "PureDecodeGraphMetadataBindingError",
    "PureDecodeGraphMetadataBinding::try_new",
    "PureDecodeGraphV1LayoutBinding",
    "Bound",
    "Ineligible",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_PREFLIGHT_BINDING_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 preflight binding source must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 preflight binding must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_preflight_binding_stays_value_only_and_outside_execution() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 V1 candidate binding crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 V1 candidate binding omitted required composition token {required:?}"
        );
    }
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_preflight_binding;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_preflight_binding"));
}
