//! Source-level boundary contract for C07 exact V1 caller-owned slab writing.

const GRAPH_DECODE_EXACT_SLAB_WRITER_SOURCE: &str =
    include_str!("../src/llama/graph_decode_exact_slab_writer.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const TEST_MODULE_SENTINEL: &str = "\n#[cfg(test)]\nmod tests {";

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphMetadataFieldSources",
    "PureDecodeGraphMetadataField::",
    ".input_token_ids()",
    ".position_ids()",
    ".row_sequence_slots()",
    ".block_row_offsets()",
    ".physical_block_ids()",
    ".valid_tokens()",
    ".output_token_indices()",
    "plan_pure_decode_graph_padding",
    "select_pure_decode_graph_bucket",
    "PURE_DECODE_GRAPH_BUCKETS",
    "LlamaBatchMetadataConfig",
    "LlamaBatchRow",
    "LlamaBatchBlockTable",
    "PreparedLlamaBatchMetadata",
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
    "get_mut",
    ".fill(",
    "copy_from_slice",
    "unsafe",
    "extern \"C\"",
    "*const",
    "*mut",
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "LlamaPackedBatchMetadata",
    "PureDecodeGraphMetadataLayout",
    "project_pure_decode_graph_v1_exact",
    "PureDecodeGraphV1ExactProjection",
    "PureDecodeGraphV1ExactProjectionIneligibility",
    "PureDecodeGraphV1ExactMetadataSources",
    "compose_pure_decode_graph_v1_exact_field_sources",
    "pack_pure_decode_graph_metadata_le",
    "PureDecodeGraphMetadataBindingError",
    "PureDecodeGraphV1ExactOpaqueSourceError",
    "PureDecodeGraphMetadataPackError",
    "LayoutBinding",
    "OpaqueSource",
    "Pack",
    "Written",
    "Ineligible",
    "destination: &mut [u8]",
    "source.field_sources()",
    "PureDecodeGraphV1ExactSlabWrite",
];

fn production_source() -> &'static str {
    let (production_source, test_source) = GRAPH_DECODE_EXACT_SLAB_WRITER_SOURCE
        .split_once(TEST_MODULE_SENTINEL)
        .expect("C07 exact slab writer must separate its test module");
    assert!(
        !test_source.contains(TEST_MODULE_SENTINEL),
        "C07 exact slab writer must keep one test module boundary"
    );
    production_source
}

#[test]
fn graph_decode_exact_slab_writer_stays_a_single_checked_write_boundary() {
    let production_source = production_source();
    for forbidden in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !production_source.contains(forbidden),
            "C07 exact slab writer crossed its exact-write boundary with {forbidden:?}"
        );
    }
    for required in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            production_source.contains(required),
            "C07 exact slab writer omitted required checked token {required:?}"
        );
    }
    assert_eq!(
        production_source
            .matches("pack_pure_decode_graph_metadata_le(")
            .count(),
        1,
        "C07 exact slab writer must delegate exactly one mutation to C07-5"
    );
    assert_eq!(
        production_source.matches("&mut").count(),
        1,
        "C07 exact slab writer may mutate only its caller-owned destination"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_exact_slab_writer;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_exact_slab_writer"));
}
