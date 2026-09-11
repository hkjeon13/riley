//! Source-level boundary contract for C07 trailing-padding topology.

const GRAPH_DECODE_PADDING_SOURCE: &str = include_str!("../src/llama/graph_decode_padding.rs");

#[test]
fn graph_decode_padding_stays_value_only_and_separate_from_batch_graph_execution() {
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
            !GRAPH_DECODE_PADDING_SOURCE.contains(forbidden),
            "C07 padding plan crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in [
        "select_pure_decode_graph_bucket",
        "checked_sub",
        "PureDecodeGraphPaddingPlan",
        "PureDecodeGraphPaddingLane",
    ] {
        assert!(
            GRAPH_DECODE_PADDING_SOURCE.contains(required),
            "C07 padding plan omitted required topology token {required:?}"
        );
    }
}
