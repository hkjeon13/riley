//! Source-level boundary contract for C07 metadata-layout binding.

const GRAPH_DECODE_BINDING_SOURCE: &str = include_str!("../src/llama/graph_decode_binding.rs");

#[test]
fn graph_decode_binding_stays_value_only_and_separate_from_batch_graph_execution() {
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
    ] {
        assert!(
            !GRAPH_DECODE_BINDING_SOURCE.contains(forbidden),
            "C07 metadata binding crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in [
        "PureDecodeGraphMetadataLayout",
        "PureDecodeGraphPaddingPlan",
        "geometry_digest",
        "LayoutPaddingBucketMismatch",
    ] {
        assert!(
            GRAPH_DECODE_BINDING_SOURCE.contains(required),
            "C07 metadata binding omitted required identity token {required:?}"
        );
    }
}
