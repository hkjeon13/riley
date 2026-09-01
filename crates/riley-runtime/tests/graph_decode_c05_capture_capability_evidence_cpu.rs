//! Source-level boundary contract for C07's exact C05 primitive evidence.

const EVIDENCE_SOURCE: &str =
    include_str!("../src/llama/graph_decode_c05_capture_capability_evidence.rs");
const INVENTORY_SOURCE: &str = include_str!("../src/llama/graph_decode_capture_inventory.rs");
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
        "generic graph policy",
        include_str!("../src/llama/executor/graph.rs"),
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
    "CudaGraphCaptureOperation::H2D",
    "CudaGraphCaptureOperation::SiluBf16",
    "CudaGraphCaptureCapability::Unknown",
    "CudaGraphCaptureCapability::Unsupported",
    "CudaGraphCaptureCapability::Supported",
    "PureDecodeGraphV1CaptureOperation::MetadataH2d",
    "PureDecodeGraphV1CaptureOperation::MlpSiluBf16",
    "PureDecodeGraphV1CaptureCapabilityInventory::default()",
    "fn map_exact_c05_capability",
    "pub(crate) fn pure_decode_graph_v1_c05_capture_capability_evidence",
    "CudaResult<PureDecodeGraphV1CaptureCapabilityInventory>",
];

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "CudaRuntime",
    "CudaContext",
    "CudaStream",
    "CudaDevice",
    "CudaPinnedHostBuffer",
    "OwnedGraph",
    "begin_graph_capture",
    ".enqueue_silu_bf16(",
    ".end(",
    ".instantiate(",
    ".launch(",
    "launch_with_source",
    "GraphRegistry",
    "select_registered_execution_graph",
    "select_execution_graph",
    "GraphDispatchRequest",
    "GraphDispatchEligibility",
    "GraphDispatchMetrics",
    "unsafe",
    "extern \"C\"",
    "Vec",
    "Box",
    "Arc",
    "Mutex",
    "pub use",
];

const EXECUTION_WIRING_TOKENS: &[&str] = &[
    "graph_decode_c05_capture_capability_evidence",
    "pure_decode_graph_v1_c05_capture_capability_evidence",
];

fn production_source() -> &'static str {
    EVIDENCE_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07-29 evidence adapter must retain its unit-test boundary")
}

fn query_body() -> &'static str {
    production_source()
        .split("pub(crate) fn pure_decode_graph_v1_c05_capture_capability_evidence")
        .nth(1)
        .expect("C07-29 must retain its native capability query boundary")
}

#[test]
fn c05_capture_capability_evidence_stays_exact_private_and_cold() {
    let source = production_source();
    for token in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            source.contains(token),
            "C07-29 capability evidence omitted required token {token:?}",
        );
    }
    for token in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !source.contains(token),
            "C07-29 capability evidence crossed its read-only boundary with {token:?}",
        );
    }
    for &(name, execution_source) in EXECUTION_SOURCES {
        for token in EXECUTION_WIRING_TOKENS {
            assert!(
                !execution_source.contains(token),
                "C07-29 capability evidence must not wire into {name} through {token:?}",
            );
        }
    }

    let body = query_body();
    assert!(
        !body.contains("CudaGraphCaptureOperation::FillF32"),
        "C05 fill evidence has no semantically identical C07 decode slot"
    );
    assert!(
        body.contains("CudaGraphCaptureOperation::H2D.capture_capability()?"),
        "metadata H2D must be queried through the exact C05 operation"
    );
    assert!(
        body.contains("CudaGraphCaptureOperation::SiluBf16.capture_capability()?"),
        "MLP SiLU must be queried through the exact C05 operation"
    );

    assert!(
        INVENTORY_SOURCE.contains("MlpSiluBf16") && INVENTORY_SOURCE.contains("MlpGatedMultiply"),
        "C07 inventory must keep C05 SiLU evidence distinct from unreviewed gated multiplication",
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains(
            "#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-29 maps only exact reviewed C05 primitives into that cold inventory.\nmod graph_decode_c05_capture_capability_evidence;"
        ),
        "C07-29 evidence must remain a private CUDA-gated module",
    );
    assert!(
        !LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c05_capture_capability_evidence"),
        "C07-29 evidence must not become a public runtime API",
    );
}
