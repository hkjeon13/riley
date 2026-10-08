//! Source-level boundary contract for C07 V1 pure-decode preflight.

const GRAPH_DECODE_PREFLIGHT_SOURCE: &str = include_str!("../src/llama/graph_decode_preflight.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "riley_model",
    "riley_tensor",
    "riley_cuda",
    "Cuda",
    "PreparedLlama",
    "LlamaBatchExecutor",
    "graph_decode_layout",
    "graph_decode_binding",
    "graph_decode_packer",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataBinding",
    "PureDecodeGraphMetadataFieldSources",
    "pack_pure_decode_graph_metadata_le",
    "PackedBatchV1",
    "PackedIterationLayout",
    "pack_iteration_input",
    "GraphMetadataLayoutSignature",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "select_pure_decode_graph_bucket",
    "PURE_DECODE_GRAPH_BUCKETS",
    "LlamaBatchMetadataConfig",
    "PreparedLlamaBatchMetadata",
    "LlamaBatchRow",
    "LlamaBatchBlockTable",
    ".pack(",
    ".schema_version()",
    ".block_table_schema_version()",
    ".sequence_tags()",
    ".row_kind_codes()",
    ".input_row_offsets()",
    ".input_token_ids()",
    ".position_ids()",
    ".row_sequence_slots()",
    ".block_row_offsets()",
    ".physical_block_ids()",
    ".valid_tokens()",
    ".logical_lengths()",
    ".output_slots_by_row()",
    ".output_row_indices()",
    ".output_token_indices()",
    ".prefill_row_indices()",
    ".decode_row_indices()",
    ".input_tokens_for_row(",
    ".positions_for_row(",
    ".physical_blocks_for_row(",
    ".valid_tokens_for_row(",
    ".output_slot_for_row(",
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
    "LlamaPackedBatchMetadata",
    "&LlamaPackedBatchMetadata",
    "row_count()",
    "prefill_row_count()",
    "decode_row_count()",
    "prefill_token_count()",
    "total_input_tokens()",
    "decode_token_count()",
    "output_count()",
    "u32::try_from",
    "plan_pure_decode_graph_padding",
    "PureDecodeGraphPaddingPlan",
    "PureDecodeGraphV1Preflight",
    "Eligible",
    "Ineligible",
    "preflight_counts",
    "UnsupportedActiveRows",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_PREFLIGHT_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 preflight source must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 preflight must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_preflight_stays_read_only_and_outside_graph_execution() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 V1 preflight crossed its read-only boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 V1 preflight omitted required eligibility token {required:?}"
        );
    }
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_preflight;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_preflight"));
}
