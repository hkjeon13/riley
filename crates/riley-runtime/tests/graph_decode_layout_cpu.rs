//! Source-level boundary contract for C07 fixed layout geometry.

const GRAPH_DECODE_LAYOUT_SOURCE: &str = include_str!("../src/llama/graph_decode_layout.rs");

#[test]
fn graph_decode_layout_stays_value_only_and_separate_from_current_batch_execution() {
    for forbidden in [
        "riley_model",
        "riley_tensor",
        "riley_cuda",
        "Cuda",
        "PreparedLlama",
        "LlamaBatchExecutor",
        "LlamaPackedBatchMetadata",
        "LlamaBatchMetadataConfig",
        "PackedBatchV1",
        "PackedIterationLayout",
        "GraphSignature",
        "GraphRegistry",
        "select_execution_graph",
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
    ] {
        assert!(
            !GRAPH_DECODE_LAYOUT_SOURCE.contains(forbidden),
            "C07 layout descriptor crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in [
        "PURE_DECODE_GRAPH_BUCKETS",
        "PureDecodeGraphMetadataField",
        "PureDecodeGraphMetadataGeometryDigest",
        "PURE_DECODE_GRAPH_METADATA_GEOMETRY_DIGEST_FIELD_COUNT",
        "Sha256",
        "checked_mul",
        "checked_add",
        "to_le_bytes",
        "BlockEntryCapacityTooSmall",
    ] {
        assert!(
            GRAPH_DECODE_LAYOUT_SOURCE.contains(required),
            "C07 layout descriptor omitted required geometry token {required:?}"
        );
    }
}
