//! Source-level boundary contract for C07 fixed metadata slab writing.

const GRAPH_DECODE_PACKER_SOURCE: &str = include_str!("../src/llama/graph_decode_packer.rs");

#[test]
fn graph_decode_packer_stays_value_only_and_separate_from_batch_graph_execution() {
    for forbidden in [
        "riley_model",
        "riley_tensor",
        "riley_cuda",
        "Cuda",
        "LlamaBatch",
        "Packed",
        "V1",
        "LlamaBatchMetadataConfig",
        "PreparedLlama",
        "LlamaBatchExecutor",
        "GraphMetadataLayoutSignature",
        "GraphSignature",
        "GraphRegistry",
        "select_execution_graph",
        "select_pure_decode_graph_bucket",
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
        "unsafe",
        "extern \"C\"",
        "*const",
        "*mut",
        "to_ne_bytes",
        "to_be_bytes",
    ] {
        assert!(
            !GRAPH_DECODE_PACKER_SOURCE.contains(forbidden),
            "C07 metadata packer crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in [
        "PureDecodeGraphMetadataBinding",
        "PureDecodeGraphMetadataFieldSources",
        "pack_pure_decode_graph_metadata_le",
        "validate_pack_layout",
        "destination.fill(0)",
        "to_le_bytes",
        "checked_add",
    ] {
        assert!(
            GRAPH_DECODE_PACKER_SOURCE.contains(required),
            "C07 metadata packer omitted required fixed-slab token {required:?}"
        );
    }
}
