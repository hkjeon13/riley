const C06_GRAPH_SOURCE: &str = include_str!("../src/llama/executor/graph.rs");
const GRAPH_DECODE_C06_SIGNATURE_SOURCE: &str =
    include_str!("../src/llama/graph_decode_c06_signature.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const REQUIRED_C07_TOKENS: &[&str] = &[
    "PureDecodeGraphV1C06Identity",
    "PureDecodeGraphV1C06IdentityBinding",
    "GraphStaticSignature",
    "GraphSignature",
    "GraphStaticMetadataLayoutMismatch",
    "compose_graph_signature_checked_metadata",
    "identity.metadata_layout_signature()",
    "identity.iteration_signature()",
    "PureDecodeGraphV1C06Identity::Bound",
    "PureDecodeGraphV1C06Identity::Ineligible",
    "PureDecodeGraphV1C06Signature::Bound",
    "PureDecodeGraphV1C06Signature::Ineligible",
];

const FORBIDDEN_C07_TOKENS: &[&str] = &[
    "GraphSignature::new",
    "GraphRegistry",
    "select_execution_graph",
    "select_registered_execution_graph",
    "GraphDispatch",
    "GraphCapture",
    "Cuda",
    "riley_cuda",
    "H2D",
    "buffer",
    "lease",
    "executor",
    "LlamaPackedBatchMetadata",
    "PureDecodeGraphMetadataLayout",
    "GraphLayoutSignature",
    "GraphIterationSignature::new",
    "with_metadata_layout",
    "Vec<",
    "Box<",
    "String",
    "std::collections",
    "unsafe",
    "as_ptr",
];

fn production_source() -> &'static str {
    GRAPH_DECODE_C06_SIGNATURE_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07 complete-signature source must have a unit-test boundary")
}

#[test]
fn graph_decode_complete_signature_stays_a_checked_value_only_bridge() {
    let source = production_source();

    for token in REQUIRED_C07_TOKENS {
        assert!(
            source.contains(token),
            "C07 complete-signature bridge must retain {token}",
        );
    }
    for token in FORBIDDEN_C07_TOKENS {
        assert!(
            !source.contains(token),
            "C07 complete-signature bridge must not introduce {token}",
        );
    }
    assert_eq!(
        source
            .matches("compose_graph_signature_checked_metadata")
            .count(),
        2,
        "C07 must import and delegate complete-key composition exactly once",
    );
    assert!(source.contains("PureDecodeGraphV1C06Signature::Bound"));
    assert!(source.contains("PureDecodeGraphV1C06Signature::Ineligible"));
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_c06_signature;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c06_signature"));
}

#[test]
fn c06_helper_checks_metadata_before_constructing_without_rebinding_static_identity() {
    let helper = C06_GRAPH_SOURCE
        .split("pub(crate) fn compose_graph_signature_checked_metadata")
        .nth(1)
        .expect("C06 must retain the checked complete-signature helper")
        .split("impl GraphModelArchitecture")
        .next()
        .expect("C06 helper must end before fingerprint implementation details");

    let check_position = helper
        .find("if static_metadata_layout != expected_metadata_layout")
        .expect("C06 helper must compare full metadata identities");
    let construct_position = helper
        .find("GraphSignature::new(static_signature, iteration_signature)")
        .expect("C06 helper must construct only after an exact metadata match");

    assert!(check_position < construct_position);
    assert!(helper.contains("static_signature.layout.metadata_layout"));
    for forbidden in [
        "GraphStaticSignature::new",
        "GraphLayoutSignature::new",
        "static_signature.layout.metadata_layout =",
        "with_metadata_layout",
        "GraphRegistry",
        "GraphDispatch",
        "Cuda",
        "unsafe",
    ] {
        assert!(
            !helper.contains(forbidden),
            "C06 helper must not rebuild, rebind, or execute graph state: {forbidden}"
        );
    }
}
