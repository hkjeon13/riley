//! Source-level boundary contract for C07's private C05 owned-exec resolver.

const RESOLVER_SOURCE: &str = include_str!("../src/llama/graph_decode_c05_owned_exec_resolver.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");

const REQUIRED_PRODUCTION_TOKENS: &[&str] = &[
    "PureDecodeGraphV1C05OwnedReplaySlot",
    "PureDecodeGraphV1C05ReplayResolution",
    "PureDecodeGraphV1C05ResolvedExec",
    "PureDecodeGraphV1C05ReplayResolveError",
    "PureDecodeGraphV1C06RegistryDispatchBinding",
    "PureDecodeGraphV1C06SignatureBinding",
    "GraphSignature",
    "GraphReplaySlot",
    "OwnedGraphExec",
    "OwnedGraphFillResources",
    "selection.signature_binding().signature()",
    "selection.decision()",
    "GraphRegistryDispatchDecision::FullGraph",
    "GraphRegistryDispatchDecision::PiecewiseGraph",
    "GraphRegistryDispatchDecision::ExactEager",
    "selected_signature != owned_signature",
    "replay_slot != owned_slot",
    "self.exec.close()",
];

const FORBIDDEN_PRODUCTION_TOKENS: &[&str] = &[
    "select_registered_execution_graph",
    "GraphDispatchMetrics",
    "observe_pure_decode_graph_v1_c06_registry_dispatch",
    "CudaStream",
    "CudaCommandBatch",
    "CudaCommandStream",
    "CudaDeviceBuffer",
    "CudaPinnedHostBuffer",
    "graph_decode_exact_",
    "begin_command_batch",
    "copy_from_pinned",
    "copy_to_pinned",
    "upload_from_slice",
    "download_to_slice",
    ".launch(",
    ".finish(",
    "Vec",
    "Box",
    "Arc",
    "Mutex",
    "unsafe",
    "extern \"C\"",
    "as_ptr",
    "as_mut_ptr",
    "fingerprint()",
];

fn production_source() -> &'static str {
    RESOLVER_SOURCE
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("C07 resolver must retain its unit-test boundary")
}

#[test]
fn graph_decode_c05_owned_exec_resolver_stays_private_and_metadata_only() {
    let source = production_source();
    for token in REQUIRED_PRODUCTION_TOKENS {
        assert!(
            source.contains(token),
            "C07 C05 owner resolver omitted required token {token:?}",
        );
    }
    for token in FORBIDDEN_PRODUCTION_TOKENS {
        assert!(
            !source.contains(token),
            "C07 C05 owner resolver crossed its narrow boundary with {token:?}",
        );
    }
    assert_eq!(
        source.matches("validate_selected_full_graph(").count(),
        2,
        "C07 must have one metadata resolver call and one helper definition",
    );
    assert!(
        source.contains("exec: &mut self.exec"),
        "successful resolution must retain an exclusive mutable C05 owner borrow",
    );
    assert!(
        source.contains("pub(crate) fn close(self) -> CudaResult<OwnedGraphFillResources>"),
        "only the enclosing owner may directly delegate C05 close",
    );
    assert!(
        !source.contains("pub(crate) fn exec("),
        "production C07 resolver must not expose an executable accessor",
    );
    assert!(
        LLAMA_MODULE_SOURCE.contains(
            "#[cfg(feature = \"cuda\")]\n#[allow(dead_code)] // C07-24 binds one selected full-graph identity to a C05 owner.\nmod graph_decode_c05_owned_exec_resolver;"
        ),
        "C07 C05 owner resolver must stay CUDA-gated and private",
    );
    assert!(
        !LLAMA_MODULE_SOURCE.contains("pub use graph_decode_c05_owned_exec_resolver"),
        "C07 C05 owner resolver must not become a public runtime API",
    );
}
