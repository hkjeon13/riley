//! Source-level boundary contract for C07's pure-decode capability inventory.

const INVENTORY_SOURCE: &str = include_str!("../src/llama/graph_decode_capture_inventory.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const EXECUTION_SOURCES: &[(&str, &str)] = &[
    (
        "batch executor",
        include_str!("../src/llama/batch_executor.rs"),
    ),
    ("decode", include_str!("../src/llama/decode.rs")),
    (
        "executor dispatch",
        include_str!("../src/llama/executor/dispatch.rs"),
    ),
    (
        "generic registry dispatcher",
        include_str!("../src/llama/executor/graph_registry_dispatch.rs"),
    ),
    (
        "generic graph policy",
        include_str!("../src/llama/executor/graph.rs"),
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
];

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT",
    "PureDecodeGraphV1CaptureOperation",
    "PureDecodeGraphV1CaptureCapabilityInventory",
    "pub(crate) const ALL",
    "pub(crate) const fn index",
    "pub(crate) const fn capability_for",
    "pub(crate) const fn with_capability",
    "pub(crate) fn operator_capability",
    "GraphOperatorCapability::Unsupported",
    "GraphOperatorCapability::Supported",
    "GraphOperatorCapability::Unknown",
];

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "riley_cuda",
    "CudaStream",
    "CudaCommandBatch",
    "CudaCommandStream",
    "OwnedGraph",
    "begin_graph_capture",
    "launch_with_source",
    "exec_for_gpu_test",
    "select_registered_execution_graph",
    "select_execution_graph",
    "GraphDispatchRequest",
    "GraphDispatchEligibility",
    "GraphCaptureSafety",
    "PureDecodeGraphV1C06",
    "GraphDispatchMetrics",
    "GraphRegistry",
    "batch_executor",
    "PreparedLlamaBatchExecutor",
    "unsafe",
    "extern \"C\"",
    "Vec",
    "Box",
    "Arc",
    "Mutex",
    "pub use",
];

const EXECUTION_WIRING_TOKENS: &[&str] = &[
    "graph_decode_capture_inventory",
    "PureDecodeGraphV1CaptureCapabilityInventory",
    "PureDecodeGraphV1CaptureOperation",
];

const CANONICAL_OPERATIONS: &[&str] = &[
    "MetadataH2d",
    "Embedding",
    "Norm",
    "LayerProjectionGemm",
    "Rope",
    "KvWrite",
    "Attention",
    "Mlp",
    "Residual",
    "FinalNorm",
    "LmHead",
    "GpuGreedy",
    "CompletionBoundary",
];

fn production_source() -> &'static str {
    INVENTORY_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07-28 inventory must retain its unit-test boundary")
}

#[test]
fn graph_decode_capture_inventory_stays_private_value_only_and_fail_closed() {
    let source = production_source();
    for token in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            source.contains(token),
            "C07-28 capture inventory omitted required token {token:?}",
        );
    }
    for token in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !source.contains(token),
            "C07-28 capture inventory crossed its read-only boundary with {token:?}",
        );
    }
    for &(name, execution_source) in EXECUTION_SOURCES {
        for token in EXECUTION_WIRING_TOKENS {
            assert!(
                !execution_source.contains(token),
                "C07-28 capture inventory must not wire into {name} through {token:?}",
            );
        }
    }

    let mut previous = 0;
    for operation in CANONICAL_OPERATIONS {
        let position = source
            .find(operation)
            .expect("every C07-28 canonical operation must be represented");
        assert!(
            position > previous,
            "C07-28 canonical operation order must remain stable at {operation:?}",
        );
        previous = position;
    }
    assert!(
        source.contains("any(|capability| *capability == GraphOperatorCapability::Unsupported)")
            && source
                .contains("all(|capability| *capability == GraphOperatorCapability::Supported)"),
        "C07-28 must preserve unsupported-over-unknown fail-closed reduction",
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains(
            "#[allow(dead_code)] // C07-28 records cold pure-decode capture evidence before any real graph owner.\nmod graph_decode_capture_inventory;"
        ),
        "C07-28 inventory must remain a private CPU-only module",
    );
    assert!(
        !LLAMA_MODULE_SOURCE.contains("pub use graph_decode_capture_inventory"),
        "C07-28 inventory must not become a public runtime API",
    );
}
