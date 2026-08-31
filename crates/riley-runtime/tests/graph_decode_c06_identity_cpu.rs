const GRAPH_DECODE_C06_IDENTITY_SOURCE: &str =
    include_str!("../src/llama/graph_decode_c06_identity.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const REQUIRED_TOKENS: &[&str] = &[
    "PureDecodeGraphV1LayoutBinding",
    "PureDecodeGraphV1Ineligibility",
    "PureDecodeGraphMetadataBinding",
    "pure_decode_graph_v1_metadata_layout_signature",
    "GraphMetadataLayoutSignature",
    "GraphIterationSignature",
    "GraphWorkloadStage::PureDecode",
    "layout.bucket_rows()",
    "sampling_backend",
];

const FORBIDDEN_TOKENS: &[&str] = &[
    "GraphLayoutSignature",
    "GraphStaticSignature",
    "GraphSignature",
    "GraphRegistry",
    "select_execution_graph",
    "select_registered_execution_graph",
    "GraphCapture",
    "Cuda",
    "LlamaPackedBatchMetadata",
    "PureDecodeGraphV1Exact",
    "GraphDispatch",
    "Vec<",
    "Box<",
    "String",
    "std::collections",
    "unsafe",
    "as_ptr",
];

fn production_source() -> &'static str {
    GRAPH_DECODE_C06_IDENTITY_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07 C06 identity source must have a unit-test boundary")
}

#[test]
fn graph_decode_c06_identity_stays_a_value_only_partial_identity_bridge() {
    let source = production_source();

    for token in REQUIRED_TOKENS {
        assert!(
            source.contains(token),
            "C07 C06 identity bridge must retain {token}",
        );
    }
    for token in FORBIDDEN_TOKENS {
        assert!(
            !source.contains(token),
            "C07 C06 identity bridge must not introduce {token}",
        );
    }
    assert_eq!(
        source
            .matches("pure_decode_graph_v1_metadata_layout_signature")
            .count(),
        2,
        "C07 C06 identity bridge must import and derive the metadata identity once",
    );
    assert_eq!(
        source.matches("GraphIterationSignature::new").count(),
        1,
        "C07 C06 identity bridge must construct exactly one iteration identity",
    );
    assert!(source.contains("PureDecodeGraphV1LayoutBinding::Bound"));
    assert!(source.contains("PureDecodeGraphV1LayoutBinding::Ineligible"));
    assert!(source.contains("PureDecodeGraphV1C06Identity::Bound"));
    assert!(source.contains("PureDecodeGraphV1C06Identity::Ineligible"));
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_c06_identity;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c06_identity"));
}
