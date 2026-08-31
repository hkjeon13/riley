use riley_runtime::llama::{
    ExecutionGraphPolicy, ExecutionMode, GraphCaptureSafety, GraphComputeType, GraphDataType,
    GraphDeviceSignature, GraphDispatchError, GraphDispatchRequest, GraphEntryFootprint,
    GraphFallbackReason, GraphGemmPlanSetId, GraphGeometrySignature, GraphImplementationId,
    GraphImplementationSignature, GraphInventoryState, GraphIterationSignature,
    GraphLayoutSignature, GraphMetadataLayoutSignature, GraphModelArchitecture,
    GraphModelSignature, GraphOperatorCapability, GraphReductionPolicyId, GraphRegistry,
    GraphRegistryDispatchDecision, GraphRegistryEntry, GraphRegistryEntryState,
    GraphRegistryLimits, GraphReplayMode, GraphReplaySlot, GraphRevisionFingerprint,
    GraphSamplingBackend, GraphSignature, GraphStaticSignature, GraphTensorSignature,
    GraphWorkloadStage, select_registered_execution_graph,
};

const GRAPH_REGISTRY_DISPATCH_SOURCE: &str =
    include_str!("../src/llama/executor/graph_registry_dispatch.rs");
const DEVICE: GraphDeviceSignature = GraphDeviceSignature::new(8, 9, 12_804, 12_804, 1);
const TENSORS: GraphTensorSignature = GraphTensorSignature::new(
    GraphDataType::BFloat16,
    GraphDataType::BFloat16,
    GraphComputeType::Float32,
);
const GEOMETRY: GraphGeometrySignature =
    GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 128);
const METADATA_LAYOUT: GraphMetadataLayoutSignature =
    GraphMetadataLayoutSignature::new(1, [0xA1; 32]);
const LAYOUT: GraphLayoutSignature = GraphLayoutSignature::new(8_192, 16, 1, METADATA_LAYOUT);
const IMPLEMENTATIONS: GraphImplementationSignature = GraphImplementationSignature::new(
    GraphImplementationId::new(1),
    GraphImplementationId::new(2),
    GraphImplementationId::new(3),
    GraphImplementationId::new(4),
    GraphGemmPlanSetId::new(5),
    GraphReductionPolicyId::new(6),
);

const fn signature(
    stage: GraphWorkloadStage,
    sampling_backend: GraphSamplingBackend,
    bucket: u32,
) -> GraphSignature {
    let model = GraphModelSignature::new(
        GraphModelArchitecture::LlamaDecoder,
        1,
        GraphRevisionFingerprint::from_bytes([1; 32]),
        1,
    );
    GraphSignature::new(
        GraphStaticSignature::new(model, DEVICE, TENSORS, GEOMETRY, LAYOUT, IMPLEMENTATIONS),
        GraphIterationSignature::new(stage, bucket, sampling_backend),
    )
}

const fn request(
    policy: ExecutionGraphPolicy,
    stage: GraphWorkloadStage,
    sampling_backend: GraphSamplingBackend,
    active_row_bucket_supported: bool,
) -> GraphDispatchRequest {
    GraphDispatchRequest::new(
        policy,
        riley_runtime::llama::GraphDispatchEligibility::new(
            stage,
            active_row_bucket_supported,
            true,
            true,
            GraphCaptureSafety::new(sampling_backend, GraphOperatorCapability::Supported, true),
        ),
        GraphInventoryState::NotPrepared,
    )
}

const fn entry(
    signature: GraphSignature,
    replay_mode: GraphReplayMode,
    replay_slot: u32,
    state: GraphRegistryEntryState,
) -> GraphRegistryEntry {
    GraphRegistryEntry::new(
        signature,
        replay_mode,
        GraphReplaySlot::new(replay_slot),
        state,
        GraphEntryFootprint::new(1, 1),
    )
}

const fn limits(
    maximum_graph_count: usize,
    maximum_full_graph_count: usize,
    maximum_piecewise_graph_count: usize,
) -> GraphRegistryLimits {
    GraphRegistryLimits::new(
        maximum_graph_count,
        maximum_full_graph_count,
        maximum_piecewise_graph_count,
        16,
        16,
    )
}

#[test]
fn adapter_selects_exact_full_and_piecewise_entries_with_slots() {
    let full_signature = signature(
        GraphWorkloadStage::PureDecode,
        GraphSamplingBackend::GpuGreedy,
        1,
    );
    let piecewise_signature = signature(
        GraphWorkloadStage::Mixed,
        GraphSamplingBackend::GpuGreedy,
        2,
    );
    let registry = GraphRegistry::<2>::try_new(
        limits(2, 1, 1),
        &[
            entry(
                full_signature,
                GraphReplayMode::FullGraph,
                10,
                GraphRegistryEntryState::Prepared,
            ),
            entry(
                piecewise_signature,
                GraphReplayMode::PiecewiseGraph,
                11,
                GraphRegistryEntryState::Prepared,
            ),
        ],
    )
    .expect("prepared exact entries should fit the cold registry");

    let full = select_registered_execution_graph(
        request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphSamplingBackend::GpuGreedy,
            true,
        ),
        full_signature,
        &registry,
    );
    assert_eq!(
        full,
        Ok(GraphRegistryDispatchDecision::FullGraph {
            replay_slot: GraphReplaySlot::new(10),
        })
    );
    assert_eq!(
        full.expect("full graph must be selected").mode(),
        ExecutionMode::FullGraph
    );

    assert_eq!(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::Mixed,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            piecewise_signature,
            &registry,
        ),
        Ok(GraphRegistryDispatchDecision::PiecewiseGraph {
            replay_slot: GraphReplaySlot::new(11),
        })
    );
}

#[test]
fn adapter_never_reuses_a_mismatched_signature_stage_or_sampling_backend() {
    let full_signature = signature(
        GraphWorkloadStage::PureDecode,
        GraphSamplingBackend::GpuGreedy,
        1,
    );
    let mixed_signature = signature(
        GraphWorkloadStage::Mixed,
        GraphSamplingBackend::GpuGreedy,
        2,
    );
    let unsupported_sampling_signature = signature(
        GraphWorkloadStage::PureDecode,
        GraphSamplingBackend::Unsupported,
        3,
    );
    let registry = GraphRegistry::<3>::try_new(
        limits(3, 2, 1),
        &[
            entry(
                full_signature,
                GraphReplayMode::FullGraph,
                10,
                GraphRegistryEntryState::Prepared,
            ),
            entry(
                mixed_signature,
                GraphReplayMode::PiecewiseGraph,
                11,
                GraphRegistryEntryState::Prepared,
            ),
            entry(
                unsupported_sampling_signature,
                GraphReplayMode::FullGraph,
                12,
                GraphRegistryEntryState::Prepared,
            ),
        ],
    )
    .expect("distinct cold entries should fit the registry");

    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            signature(
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                9,
            ),
            &registry,
        )
        .expect("an exact miss must fall back under auto"),
        GraphFallbackReason::NotPrepared,
    );
    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::Prefill,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            mixed_signature,
            &registry,
        )
        .expect("stage mismatch must stay eager"),
        GraphFallbackReason::SignatureMismatch,
    );
    assert_eq!(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Require,
                GraphWorkloadStage::Prefill,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            full_signature,
            &registry,
        ),
        Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::SignatureMismatch,
        })
    );
    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            unsupported_sampling_signature,
            &registry,
        )
        .expect("sampling mismatch must stay eager"),
        GraphFallbackReason::SignatureMismatch,
    );
}

#[test]
fn adapter_maps_poison_capacity_and_require_to_closed_outcomes() {
    let exact = signature(
        GraphWorkloadStage::PureDecode,
        GraphSamplingBackend::GpuGreedy,
        1,
    );
    let poisoned = GraphRegistry::<1>::try_new(
        limits(1, 1, 0),
        &[entry(
            exact,
            GraphReplayMode::FullGraph,
            10,
            GraphRegistryEntryState::Poisoned,
        )],
    )
    .expect("a poisoned entry is retained in a valid registry");

    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            exact,
            &poisoned,
        )
        .expect("auto must not replay a poisoned entry"),
        GraphFallbackReason::GraphPoisoned,
    );
    assert_eq!(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Require,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            exact,
            &poisoned,
        ),
        Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::GraphPoisoned,
        })
    );

    let disabled_capacity = GraphRegistry::<1>::capacity_disabled();
    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            exact,
            &disabled_capacity,
        )
        .expect("auto must report disabled registry capacity as eager"),
        GraphFallbackReason::CapacityDisabled,
    );
    assert_eq!(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Require,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
            ),
            exact,
            &disabled_capacity,
        ),
        Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::CapacityDisabled,
        })
    );
}

#[test]
fn adapter_keeps_prelookup_rejections_and_disabled_policy_outside_registry_selection() {
    let exact = signature(
        GraphWorkloadStage::PureDecode,
        GraphSamplingBackend::GpuGreedy,
        1,
    );
    let prepared = GraphRegistry::<1>::try_new(
        limits(1, 1, 0),
        &[entry(
            exact,
            GraphReplayMode::FullGraph,
            10,
            GraphRegistryEntryState::Prepared,
        )],
    )
    .expect("prepared registry should be valid");

    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                false,
            ),
            exact,
            &prepared,
        )
        .expect("unsupported shape must remain eager"),
        GraphFallbackReason::UnsupportedShape,
    );
    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::Unsupported,
                true,
            ),
            signature(
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::Unsupported,
                1,
            ),
            &prepared,
        )
        .expect("unsupported sampling must remain eager"),
        GraphFallbackReason::UnsupportedSampling,
    );
    assert_exact_eager(
        select_registered_execution_graph(
            GraphDispatchRequest::new(
                ExecutionGraphPolicy::Auto,
                riley_runtime::llama::GraphDispatchEligibility::new(
                    GraphWorkloadStage::PureDecode,
                    true,
                    true,
                    false,
                    GraphCaptureSafety::new(
                        GraphSamplingBackend::GpuGreedy,
                        GraphOperatorCapability::Supported,
                        true,
                    ),
                ),
                GraphInventoryState::NotPrepared,
            ),
            exact,
            &prepared,
        )
        .expect("caller-disabled inventory must remain eager"),
        GraphFallbackReason::CapacityDisabled,
    );
    assert_exact_eager(
        select_registered_execution_graph(
            request(
                ExecutionGraphPolicy::Disabled,
                GraphWorkloadStage::Unsupported,
                GraphSamplingBackend::Unsupported,
                false,
            ),
            exact,
            &prepared,
        )
        .expect("disabled must ignore all graph facts"),
        GraphFallbackReason::PolicyDisabled,
    );
}

#[test]
fn registry_dispatch_stays_cpu_only_and_keeps_the_lookup_hook_private() {
    for forbidden in [
        "riley_model",
        "riley_tensor",
        "riley_cuda",
        "PreparedLlama",
        "LlamaBatchExecutor",
        "Cuda",
        "extern \"C\"",
        "unsafe",
        "Vec<",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "Box<",
        "Arc<",
        "Mutex<",
        "RwLock<",
        "String",
        "alloc::",
        "*const",
        "*mut",
    ] {
        assert!(
            !GRAPH_REGISTRY_DISPATCH_SOURCE.contains(forbidden),
            "registry dispatch crossed its CPU-only boundary with {forbidden:?}"
        );
    }
    assert!(
        !GRAPH_REGISTRY_DISPATCH_SOURCE.contains("pub trait GraphInventorySource"),
        "the fake lookup seam must remain private to registry dispatch"
    );
    let disabled = GRAPH_REGISTRY_DISPATCH_SOURCE
        .find("ExecutionGraphPolicy::Disabled")
        .expect("disabled branch must remain explicit");
    let lookup = GRAPH_REGISTRY_DISPATCH_SOURCE
        .find("inventory_source.lookup")
        .expect("registry lookup must remain centralized");
    assert!(
        disabled < lookup,
        "disabled must return before registry lookup is reachable"
    );
}

fn assert_exact_eager(
    decision: GraphRegistryDispatchDecision,
    expected_reason: GraphFallbackReason,
) {
    assert_eq!(decision.mode(), ExecutionMode::ExactEager);
    assert_eq!(decision.fallback_reason(), Some(expected_reason));
    assert_eq!(decision.replay_slot(), None);
}
