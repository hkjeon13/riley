//! Source-level boundary contract for C07 exact nine-field source composition.

const GRAPH_DECODE_EXACT_FIELD_SOURCES_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_field_sources.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "LlamaPackedBatchMetadata",
    "PreparedLlamaBatchMetadata",
    "LlamaBatch",
    "project_pure_decode_graph_v1_exact",
    "preflight_pure_decode_graph_v1",
    "bind_pure_decode_graph_v1_preflight",
    "PureDecodeGraphV1LayoutBinding",
    "PureDecodeGraphV1ExactProjection",
    "PureDecodeGraphV1ExactOpaqueSourceError",
    "try_new(",
    "pack_pure_decode_graph_metadata_le",
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
    "Result<",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1ExactMetadataSources",
    "PureDecodeGraphMetadataBinding",
    "PureDecodeGraphMetadataFieldSources",
    "PureDecodeGraphV1ExactFieldSourceBinding",
    "compose_pure_decode_graph_v1_exact_field_sources",
    "source.native_fields()",
    "source.header()",
    "token_ids()",
    "position_ids()",
    "row_sequence_slots()",
    "sequence_block_offsets()",
    "physical_block_ids()",
    "valid_tokens()",
    "output_token_indices()",
    "source.control_status()",
    "binding()",
    "PureDecodeGraphMetadataFieldSources::new",
    "'metadata: 'view",
    "'opaque: 'view",
    "field_sources",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_EXACT_FIELD_SOURCES_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 exact field source must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 exact field source must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_exact_field_sources_stay_borrowed_and_outside_packing_or_execution() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 exact field source crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 exact field source omitted required canonical token {required:?}"
        );
    }
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_field_sources;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_field_sources"));
}
