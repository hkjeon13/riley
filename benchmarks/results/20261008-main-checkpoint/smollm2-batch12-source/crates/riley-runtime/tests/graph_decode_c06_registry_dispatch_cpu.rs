const GRAPH_DECODE_C06_REGISTRY_DISPATCH_SOURCE: &str =
    include_str!("../src/llama/graph_decode_c06_registry_dispatch.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const REQUIRED_TOKENS: &[&str] = &[
    "PureDecodeGraphV1C06Signature",
    "PureDecodeGraphV1C06SignatureBinding",
    "PureDecodeGraphV1Ineligibility",
    "GraphDispatchRequest",
    "GraphDispatchError",
    "GraphRegistry",
    "GraphRegistryDispatchDecision",
    "select_registered_execution_graph",
    "PureDecodeGraphV1C06Signature::Ineligible",
    "PureDecodeGraphV1C06Signature::Bound",
    "PureDecodeGraphV1C06RegistryDispatch::Ineligible",
    "PureDecodeGraphV1C06RegistryDispatch::Bound",
];

const FORBIDDEN_TOKENS: &[&str] = &[
    "GraphDispatchRequest::new",
    ".with_eligibility",
    ".with_inventory",
    "GraphDispatchEligibility::new",
    "GraphCaptureSafety::new",
    "GraphSignature::new",
    "GraphStaticSignature",
    "GraphIterationSignature::new",
    "GraphLayoutSignature",
    "registry.lookup",
    "replay_slot()",
    "GraphRegistryDispatchDecision::FullGraph",
    "GraphRegistryDispatchDecision::PiecewiseGraph",
    "GraphRegistryDispatchDecision::ExactEager",
    "GraphDispatchMetrics",
    "GraphCapture",
    "Cuda",
    "riley_cuda",
    "H2D",
    "buffer",
    "lease",
    "LlamaPackedBatchMetadata",
    "Vec<",
    "Box<",
    "String",
    "std::collections",
    "unsafe",
    "as_ptr",
];

fn production_source() -> &'static str {
    GRAPH_DECODE_C06_REGISTRY_DISPATCH_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07 registry-dispatch source must have a unit-test boundary")
}

fn selection_function_source() -> &'static str {
    production_source()
        .split("pub(crate) fn select_pure_decode_graph_v1_c06_registry_dispatch")
        .nth(1)
        .expect("C07 must retain its registry-selection boundary")
}

#[test]
fn graph_decode_registry_dispatch_is_a_single_pass_through_to_c06() {
    let source = production_source();
    let selection = selection_function_source();

    for token in REQUIRED_TOKENS {
        assert!(
            source.contains(token),
            "C07 registry-dispatch bridge must retain {token}",
        );
    }
    for token in FORBIDDEN_TOKENS {
        assert!(
            !source.contains(token),
            "C07 registry-dispatch bridge must not introduce {token}",
        );
    }
    assert_eq!(
        source.matches("select_registered_execution_graph").count(),
        2,
        "C07 must import and delegate to C06 exactly once",
    );
    let ineligible_position = selection
        .find("PureDecodeGraphV1C06Signature::Ineligible")
        .expect("C07 must preserve its ineligible branch");
    let delegate_position = selection
        .find("select_registered_execution_graph(request, binding.signature(), registry)")
        .expect("C07 Bound branch must delegate its original request and exact signature");
    assert!(
        ineligible_position < delegate_position,
        "C07 ineligible candidates must return before C06 request or registry delegation"
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_c06_registry_dispatch;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c06_registry_dispatch"));
}
