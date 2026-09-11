const GRAPH_DECODE_C06_REGISTRY_OBSERVATION_SOURCE: &str =
    include_str!("../src/llama/graph_decode_c06_registry_observation.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const REQUIRED_TOKENS: &[&str] = &[
    "PureDecodeGraphV1C06RegistryDispatch",
    "PureDecodeGraphV1C06RegistryDispatchResult",
    "GraphDispatchMetrics",
    "PureDecodeGraphV1C06RegistryDispatch::Ineligible",
    "PureDecodeGraphV1C06RegistryDispatch::Bound",
    "metrics.record_decision(binding.decision())",
    "metrics.record_error(error)",
];

const FORBIDDEN_TOKENS: &[&str] = &[
    "GraphDispatchRequest",
    "GraphRegistry",
    "select_registered_execution_graph",
    "registry.",
    "replay_slot()",
    "GraphRegistryDispatchDecision::",
    "GraphSignature",
    "GraphStaticSignature",
    "GraphIterationSignature",
    "GraphDispatchMetricsSnapshot",
    "GraphCapture",
    "Cuda",
    "riley_cuda",
    "H2D",
    "buffer",
    "lease",
    "LlamaPackedBatchMetadata",
    "LlamaBatchExecutor",
    "Vec<",
    "Box<",
    "String",
    "std::collections",
    "unsafe",
    "as_ptr",
];

fn production_source() -> &'static str {
    GRAPH_DECODE_C06_REGISTRY_OBSERVATION_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07 observation source must have a unit-test boundary")
}

fn observation_function_source() -> &'static str {
    production_source()
        .split("pub(crate) fn observe_pure_decode_graph_v1_c06_registry_dispatch")
        .nth(1)
        .expect("C07 must retain its registry-observation boundary")
}

#[test]
fn graph_decode_registry_observation_preserves_c07_and_c06_outcome_boundaries() {
    let source = production_source();
    let observation = observation_function_source();

    for token in REQUIRED_TOKENS {
        assert!(
            source.contains(token),
            "C07 registry observation must retain {token}",
        );
    }
    for token in FORBIDDEN_TOKENS {
        assert!(
            !source.contains(token),
            "C07 registry observation must not introduce {token}",
        );
    }
    assert_eq!(
        source
            .matches("metrics.record_decision(binding.decision())")
            .count(),
        1,
        "a bound C07 result must record its opaque C06 decision exactly once",
    );
    assert_eq!(
        source.matches("metrics.record_error(error)").count(),
        1,
        "a C06 require error must be recorded exactly once",
    );

    let ineligible_start = observation
        .find("Ok(PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason))")
        .expect("C07 must preserve its ineligible branch");
    let bound_start = observation
        .find("Ok(PureDecodeGraphV1C06RegistryDispatch::Bound(binding))")
        .expect("C07 must retain its bound branch");
    let ineligible_arm = &observation[ineligible_start..bound_start];
    assert!(
        !ineligible_arm.contains("metrics."),
        "ineligible C07 candidates must not enter C06 metrics",
    );
    assert!(LLAMA_MODULE_SOURCE.contains("mod graph_decode_c06_registry_observation;"));
    assert!(!LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c06_registry_observation"));
}
