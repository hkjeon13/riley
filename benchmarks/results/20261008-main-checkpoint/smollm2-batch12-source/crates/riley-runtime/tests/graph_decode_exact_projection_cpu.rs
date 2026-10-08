//! Source-level boundary contract for C07 exact V1 native-field projection.

const GRAPH_DECODE_EXACT_PROJECTION_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_projection.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "graph_decode_packer",
    "PureDecodeGraphMetadataFieldSources",
    "pack_pure_decode_graph_metadata_le",
    "PackedBatchV1",
    "PackedIterationLayout",
    "pack_iteration_input",
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
    "LlamaBatchMetadataConfig",
    "LlamaBatchRow",
    "LlamaBatchBlockTable",
    ".pack(",
    ".schema_version()",
    ".block_table_schema_version()",
    ".sequence_tags()",
    ".row_kind_codes()",
    ".input_row_offsets()",
    ".total_input_tokens()",
    ".logical_lengths()",
    ".output_slots_by_row()",
    ".output_row_indices()",
    ".output_count()",
    ".prefill_row_indices()",
    ".decode_row_indices()",
    ".prefill_row_count()",
    ".decode_row_count()",
    ".prefill_token_count()",
    ".decode_token_count()",
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
    "preflight_pure_decode_graph_v1",
    "bind_pure_decode_graph_v1_preflight",
    "PureDecodeGraphV1LayoutBinding",
    "PureDecodeGraphMetadataBinding",
    "PureDecodeGraphMetadataBindingError",
    "PureDecodeGraphMetadataLayout",
    "PureDecodeGraphMetadataField",
    "padding_plan()",
    "padding_rows()",
    "input_token_ids()",
    "position_ids()",
    "row_sequence_slots()",
    "block_row_offsets()",
    "physical_block_ids()",
    "valid_tokens()",
    "output_token_indices()",
    "PaddingRequired",
    "NativeFieldLengthMismatch",
    "Projected",
    "Ineligible",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_EXACT_PROJECTION_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 exact projection source must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 exact projection must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_exact_projection_stays_borrowed_and_outside_execution() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 exact V1 projection crossed its borrowed-field boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 exact V1 projection omitted required checked token {required:?}"
        );
    }
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_projection;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_projection"));
}
