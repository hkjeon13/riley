//! Source-level boundary contract for C07's C05-to-C06 evidence binding.

const BINDING_SOURCE: &str =
    include_str!("../src/llama/graph_decode_c05_capture_capability_dispatch_binding.rs");
const GRAPH_SOURCE: &str = include_str!("../src/llama/executor/graph.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const EXECUTION_SOURCES: &[(&str, &str)] = &[
    (
        "batch executor",
        include_str!("../src/llama/batch_executor.rs"),
    ),
    ("decode", include_str!("../src/llama/decode.rs")),
    ("forward", include_str!("../src/llama/forward.rs")),
    ("generation", include_str!("../src/llama/generation.rs")),
    (
        "executor dispatch",
        include_str!("../src/llama/executor/dispatch.rs"),
    ),
    (
        "generic registry dispatcher",
        include_str!("../src/llama/executor/graph_registry_dispatch.rs"),
    ),
    (
        "C06 registry dispatcher",
        include_str!("../src/llama/graph_decode_c06_registry_dispatch.rs"),
    ),
    (
        "C06 registry observation",
        include_str!("../src/llama/graph_decode_c06_registry_observation.rs"),
    ),
    (
        "C05 metadata owner",
        include_str!("../src/llama/graph_decode_c05_h2d_metadata_owner.rs"),
    ),
    (
        "C05 fill resolver",
        include_str!("../src/llama/graph_decode_c05_owned_exec_resolver.rs"),
    ),
    (
        "C05 H2D resolver",
        include_str!("../src/llama/graph_decode_c05_owned_h2d_exec_resolver.rs"),
    ),
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "GraphDispatchRequest",
    "PureDecodeGraphV1CaptureCapabilityInventory",
    "pub(crate) fn bind_pure_decode_graph_v1_capture_capability_inventory",
    "pub(crate) fn bind_pure_decode_graph_v1_exact_c05_capture_evidence",
    "pure_decode_graph_v1_c05_capture_capability_evidence",
    "GraphWorkloadStage::PureDecode",
    "inventory.operator_capability()",
    "GraphOperatorCapability::Unknown",
    ".with_eligibility(",
    "CudaResult<GraphDispatchRequest>",
];

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "GraphDispatchRequest::new",
    "GraphDispatchEligibility::new",
    "GraphCaptureSafety::new",
    "select_execution_graph",
    "select_registered_execution_graph",
    "GraphRegistry",
    "GraphDispatchMetrics",
    "CudaGraphCaptureOperation",
    "CudaRuntime",
    "CudaContext",
    "CudaStream",
    "CudaDevice",
    "CudaPinnedHostBuffer",
    "OwnedGraph",
    "begin_graph_capture",
    ".enqueue_",
    ".end(",
    ".instantiate(",
    ".launch(",
    "Llama",
    "unsafe",
    "extern \"C\"",
    "Vec",
    "Box",
    "Arc",
    "Mutex",
    "pub use",
];

const EXECUTION_WIRING_TOKENS: &[&str] = &[
    "graph_decode_c05_capture_capability_dispatch_binding",
    "bind_pure_decode_graph_v1_exact_c05_capture_evidence",
];

fn production_source() -> &'static str {
    BINDING_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07-30 binding must retain its unit-test boundary")
}

fn inventory_binding_body() -> &'static str {
    production_source()
        .split("pub(crate) fn bind_pure_decode_graph_v1_capture_capability_inventory")
        .nth(1)
        .expect("C07-30 must retain the inventory binding")
        .split("/// Queries exact C05 primitive evidence")
        .next()
        .expect("inventory binding must end before native evidence binding")
}

fn exact_evidence_binding_body() -> &'static str {
    production_source()
        .split("pub(crate) fn bind_pure_decode_graph_v1_exact_c05_capture_evidence")
        .nth(1)
        .expect("C07-30 must retain the exact C05 evidence binding")
}

#[test]
fn c05_to_c06_binding_stays_private_exact_and_nonexecuting() {
    let source = production_source();
    for token in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            source.contains(token),
            "C07-30 capability binding omitted required token {token:?}",
        );
    }
    for token in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !source.contains(token),
            "C07-30 capability binding crossed its value-only boundary with {token:?}",
        );
    }
    for &(name, execution_source) in EXECUTION_SOURCES {
        for token in EXECUTION_WIRING_TOKENS {
            assert!(
                !execution_source.contains(token),
                "C07-30 capability binding must not wire into {name} through {token:?}",
            );
        }
    }

    let inventory_binding = inventory_binding_body();
    assert!(
        inventory_binding.contains("inventory.operator_capability()"),
        "C07-30 must bind the inventory aggregate rather than individual primitive facts"
    );
    assert!(
        inventory_binding.contains("== GraphWorkloadStage::PureDecode")
            && inventory_binding.contains("GraphOperatorCapability::Unknown"),
        "C07-30 must apply pure-decode evidence only to pure-decode requests and fail closed elsewhere",
    );
    assert!(
        !inventory_binding.contains("pure_decode_graph_v1_c05_capture_capability_evidence"),
        "the pure inventory binding must not query native evidence"
    );

    let exact_binding = exact_evidence_binding_body();
    assert_eq!(
        exact_binding
            .matches("pure_decode_graph_v1_c05_capture_capability_evidence()")
            .count(),
        1,
        "C07-30 must query C07-29's exact C05 evidence once before binding"
    );
    assert!(
        exact_binding.contains("bind_pure_decode_graph_v1_capture_capability_inventory"),
        "the native-evidence path must delegate to the same pure aggregate binding"
    );

    assert!(
        GRAPH_SOURCE.contains("pub(crate) const fn with_operator_capability")
            && GRAPH_SOURCE.contains("self.capture_safety.sampling_backend")
            && GRAPH_SOURCE.contains("self.capture_safety.backend_capture_safe"),
        "C06 eligibility must retain every non-operator fact when replacing evidence",
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains(
            "#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-30 binds exact C05 evidence to C06 policy without enabling execution.\nmod graph_decode_c05_capture_capability_dispatch_binding;"
        ),
        "C07-30 binding must remain a private CUDA-gated module",
    );
    assert!(
        !LLAMA_MODULE_SOURCE
            .contains("pub use graph_decode_c05_capture_capability_dispatch_binding"),
        "C07-30 binding must not become a public runtime API",
    );
}
