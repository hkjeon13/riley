use std::collections::HashSet;

use riley_runtime::llama::{
    ExecutionGraphPolicy, ExecutionMode, GRAPH_SIGNATURE_SCHEMA_VERSION, GraphCaptureSafety,
    GraphComputeType, GraphDataType, GraphDeviceSignature, GraphDispatchDecision,
    GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest, GraphFallbackReason,
    GraphGemmPlanSetId, GraphGeometrySignature, GraphImplementationId,
    GraphImplementationSignature, GraphInventoryState, GraphIterationSignature,
    GraphLayoutSignature, GraphMetadataLayoutSignature, GraphModelArchitecture,
    GraphModelSignature, GraphOperatorCapability, GraphReductionPolicyId, GraphRevisionFingerprint,
    GraphSamplingBackend, GraphSignature, GraphStaticSignature, GraphTensorSignature,
    GraphWorkloadStage, select_execution_graph,
};

const GRAPH_DISPATCH_SOURCE: &str = include_str!("../src/llama/executor/graph.rs");
const MODEL_REVISION: GraphRevisionFingerprint = GraphRevisionFingerprint::from_bytes([1; 32]);
const MODEL: GraphModelSignature =
    GraphModelSignature::new(GraphModelArchitecture::LlamaDecoder, 1, MODEL_REVISION, 1);
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
const ATTENTION: GraphImplementationId = GraphImplementationId::new(1);
const PROJECTIONS: GraphImplementationId = GraphImplementationId::new(2);
const MLP: GraphImplementationId = GraphImplementationId::new(3);
const OUTPUT: GraphImplementationId = GraphImplementationId::new(4);
const GEMM_PLAN_SET: GraphGemmPlanSetId = GraphGemmPlanSetId::new(5);
const REDUCTION: GraphReductionPolicyId = GraphReductionPolicyId::new(6);
const IMPLEMENTATIONS: GraphImplementationSignature = GraphImplementationSignature::new(
    ATTENTION,
    PROJECTIONS,
    MLP,
    OUTPUT,
    GEMM_PLAN_SET,
    REDUCTION,
);
const STATIC_SIGNATURE: GraphStaticSignature =
    GraphStaticSignature::new(MODEL, DEVICE, TENSORS, GEOMETRY, LAYOUT, IMPLEMENTATIONS);
const ITERATION_SIGNATURE: GraphIterationSignature = GraphIterationSignature::new(
    GraphWorkloadStage::PureDecode,
    1,
    GraphSamplingBackend::GpuGreedy,
);
const SIGNATURE: GraphSignature = GraphSignature::new(STATIC_SIGNATURE, ITERATION_SIGNATURE);

const MODEL_VARIANTS: [GraphModelSignature; 3] = [
    GraphModelSignature::new(GraphModelArchitecture::LlamaDecoder, 2, MODEL_REVISION, 1),
    GraphModelSignature::new(
        GraphModelArchitecture::LlamaDecoder,
        1,
        GraphRevisionFingerprint::from_bytes([2; 32]),
        1,
    ),
    GraphModelSignature::new(GraphModelArchitecture::LlamaDecoder, 1, MODEL_REVISION, 2),
];
const DEVICE_VARIANTS: [GraphDeviceSignature; 5] = [
    GraphDeviceSignature::new(9, 9, 12_804, 12_804, 1),
    GraphDeviceSignature::new(8, 8, 12_804, 12_804, 1),
    GraphDeviceSignature::new(8, 9, 12_805, 12_804, 1),
    GraphDeviceSignature::new(8, 9, 12_804, 12_805, 1),
    GraphDeviceSignature::new(8, 9, 12_804, 12_804, 2),
];
const TENSOR_VARIANTS: [GraphTensorSignature; 3] = [
    GraphTensorSignature::new(
        GraphDataType::Float16,
        GraphDataType::BFloat16,
        GraphComputeType::Float32,
    ),
    GraphTensorSignature::new(
        GraphDataType::BFloat16,
        GraphDataType::Float16,
        GraphComputeType::Float32,
    ),
    GraphTensorSignature::new(
        GraphDataType::BFloat16,
        GraphDataType::BFloat16,
        GraphComputeType::TensorFloat32,
    ),
];
const GEOMETRY_VARIANTS: [GraphGeometrySignature; 7] = [
    GraphGeometrySignature::new(33, 4_096, 11_008, 32_000, 32, 8, 128),
    GraphGeometrySignature::new(32, 4_097, 11_008, 32_000, 32, 8, 128),
    GraphGeometrySignature::new(32, 4_096, 11_009, 32_000, 32, 8, 128),
    GraphGeometrySignature::new(32, 4_096, 11_008, 32_001, 32, 8, 128),
    GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 33, 8, 128),
    GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 9, 128),
    GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 129),
];
const METADATA_LAYOUT_SCHEMA_VARIANT: GraphMetadataLayoutSignature =
    GraphMetadataLayoutSignature::new(2, [0xA1; 32]);
const METADATA_LAYOUT_DIGEST_VARIANT: GraphMetadataLayoutSignature =
    GraphMetadataLayoutSignature::new(1, [0xA2; 32]);
const LAYOUT_VARIANTS: [GraphLayoutSignature; 5] = [
    GraphLayoutSignature::new(8_193, 16, 1, METADATA_LAYOUT),
    GraphLayoutSignature::new(8_192, 17, 1, METADATA_LAYOUT),
    GraphLayoutSignature::new(8_192, 16, 2, METADATA_LAYOUT),
    GraphLayoutSignature::new(8_192, 16, 1, METADATA_LAYOUT_SCHEMA_VARIANT),
    GraphLayoutSignature::new(8_192, 16, 1, METADATA_LAYOUT_DIGEST_VARIANT),
];
const IMPLEMENTATION_VARIANTS: [GraphImplementationSignature; 6] = [
    GraphImplementationSignature::new(
        GraphImplementationId::new(99),
        PROJECTIONS,
        MLP,
        OUTPUT,
        GEMM_PLAN_SET,
        REDUCTION,
    ),
    GraphImplementationSignature::new(
        ATTENTION,
        GraphImplementationId::new(99),
        MLP,
        OUTPUT,
        GEMM_PLAN_SET,
        REDUCTION,
    ),
    GraphImplementationSignature::new(
        ATTENTION,
        PROJECTIONS,
        GraphImplementationId::new(99),
        OUTPUT,
        GEMM_PLAN_SET,
        REDUCTION,
    ),
    GraphImplementationSignature::new(
        ATTENTION,
        PROJECTIONS,
        MLP,
        GraphImplementationId::new(99),
        GEMM_PLAN_SET,
        REDUCTION,
    ),
    GraphImplementationSignature::new(
        ATTENTION,
        PROJECTIONS,
        MLP,
        OUTPUT,
        GraphGemmPlanSetId::new(99),
        REDUCTION,
    ),
    GraphImplementationSignature::new(
        ATTENTION,
        PROJECTIONS,
        MLP,
        OUTPUT,
        GEMM_PLAN_SET,
        GraphReductionPolicyId::new(99),
    ),
];
const ITERATION_VARIANTS: [GraphIterationSignature; 3] = [
    GraphIterationSignature::new(
        GraphWorkloadStage::Mixed,
        1,
        GraphSamplingBackend::GpuGreedy,
    ),
    GraphIterationSignature::new(
        GraphWorkloadStage::PureDecode,
        2,
        GraphSamplingBackend::GpuGreedy,
    ),
    GraphIterationSignature::new(
        GraphWorkloadStage::PureDecode,
        1,
        GraphSamplingBackend::Unsupported,
    ),
];

const fn admitted_safety() -> GraphCaptureSafety {
    GraphCaptureSafety::new(
        GraphSamplingBackend::GpuGreedy,
        GraphOperatorCapability::Supported,
        true,
    )
}

const fn request(
    policy: ExecutionGraphPolicy,
    stage: GraphWorkloadStage,
    inventory: GraphInventoryState,
) -> GraphDispatchRequest {
    GraphDispatchRequest::new(
        policy,
        GraphDispatchEligibility::new(stage, true, true, true, admitted_safety()),
        inventory,
    )
}

#[test]
fn graph_dispatch_is_cpu_only_and_keeps_disabled_as_exact_eager() {
    let disabled = select_execution_graph(GraphDispatchRequest::new(
        ExecutionGraphPolicy::Disabled,
        GraphDispatchEligibility::new(
            GraphWorkloadStage::Unsupported,
            false,
            false,
            false,
            GraphCaptureSafety::new(
                GraphSamplingBackend::Unsupported,
                GraphOperatorCapability::Unknown,
                false,
            ),
        ),
        GraphInventoryState::Poisoned,
    ))
    .expect("disabled must choose the exact eager path without inspecting facts");

    assert_eq!(disabled.mode(), ExecutionMode::ExactEager);
    assert_eq!(
        disabled.fallback_reason(),
        Some(GraphFallbackReason::PolicyDisabled)
    );
}

#[test]
fn auto_selects_only_exactly_matching_graph_entries() {
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedFull,
        )),
        Ok(GraphDispatchDecision::FullGraph)
    );
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::Mixed,
            GraphInventoryState::PreparedPiecewise,
        )),
        Ok(GraphDispatchDecision::PiecewiseGraph)
    );
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedPiecewise,
        )),
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::NotPrepared
        ))
    );
    assert_eq!(
        select_execution_graph(GraphDispatchRequest::new(
            ExecutionGraphPolicy::Auto,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::Mixed,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::Unsupported,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphInventoryState::PreparedPiecewise,
        )),
        Ok(GraphDispatchDecision::PiecewiseGraph)
    );
}

#[test]
fn unknown_capability_and_poison_never_select_a_graph() {
    let unknown = GraphDispatchRequest::new(
        ExecutionGraphPolicy::Auto,
        GraphDispatchEligibility::new(
            GraphWorkloadStage::PureDecode,
            true,
            true,
            true,
            GraphCaptureSafety::new(
                GraphSamplingBackend::GpuGreedy,
                GraphOperatorCapability::Unknown,
                true,
            ),
        ),
        GraphInventoryState::PreparedFull,
    );
    assert_eq!(
        select_execution_graph(unknown),
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::OperatorCapabilityUnknown
        ))
    );
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::Prefill,
            GraphInventoryState::Poisoned,
        )),
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::GraphPoisoned
        ))
    );
}

#[test]
fn require_rejects_instead_of_silently_running_eager() {
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Require,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::NotPrepared,
        )),
        Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::NotPrepared,
        })
    );
}

#[test]
fn graph_signature_is_versioned_deterministic_and_hashable() {
    assert_eq!(SIGNATURE.schema_version(), GRAPH_SIGNATURE_SCHEMA_VERSION);
    assert_eq!(
        SIGNATURE,
        GraphSignature::new(STATIC_SIGNATURE, ITERATION_SIGNATURE)
    );
    assert_eq!(
        SIGNATURE.fingerprint(),
        GraphSignature::new(STATIC_SIGNATURE, ITERATION_SIGNATURE).fingerprint()
    );
    assert_eq!(
        SIGNATURE.fingerprint().as_bytes(),
        &[
            253, 252, 227, 175, 212, 136, 18, 9, 37, 172, 52, 185, 211, 201, 188, 21, 244, 142,
            145, 54, 159, 58, 119, 51, 84, 231, 135, 221, 99, 75, 67, 164,
        ]
    );
    assert_eq!(SIGNATURE.static_signature(), STATIC_SIGNATURE);
    assert_eq!(SIGNATURE.iteration(), ITERATION_SIGNATURE);
    assert_eq!(METADATA_LAYOUT.schema_version(), 1);
    assert_eq!(METADATA_LAYOUT.digest(), &[0xA1; 32]);

    let canonical = GraphRevisionFingerprint::from_canonical_bytes(b"canonical-model-v1");
    assert_eq!(
        canonical,
        GraphRevisionFingerprint::from_canonical_bytes(b"canonical-model-v1")
    );
    assert_ne!(
        canonical,
        GraphRevisionFingerprint::from_canonical_bytes(b"canonical-model-v2")
    );
    assert_eq!(
        canonical.as_bytes(),
        &[
            55, 135, 43, 30, 86, 22, 133, 173, 61, 57, 46, 198, 20, 66, 118, 209, 111, 103, 189,
            63, 162, 5, 171, 34, 230, 235, 207, 62, 20, 33, 72, 4,
        ]
    );
}

#[test]
fn graph_signature_misses_for_every_model_device_and_tensor_field() {
    let model_candidates = MODEL_VARIANTS.map(signature_with_model);
    let device_candidates = DEVICE_VARIANTS.map(signature_with_device);
    let tensor_candidates = TENSOR_VARIANTS.map(signature_with_tensors);

    assert_signature_misses(&model_candidates);
    assert_signature_misses(&device_candidates);
    assert_signature_misses(&tensor_candidates);
}

#[test]
fn graph_signature_misses_for_every_geometry_layout_and_plan_field() {
    let geometry_candidates = GEOMETRY_VARIANTS.map(signature_with_geometry);
    let layout_candidates = LAYOUT_VARIANTS.map(signature_with_layout);
    let implementation_candidates = IMPLEMENTATION_VARIANTS.map(signature_with_implementations);

    assert_signature_misses(&geometry_candidates);
    assert_signature_misses(&layout_candidates);
    assert_signature_misses(&implementation_candidates);
}

#[test]
fn graph_signature_misses_for_every_iteration_field() {
    let candidates = ITERATION_VARIANTS.map(signature_with_iteration);
    assert_signature_misses(&candidates);
}

#[test]
fn auto_reports_every_pre_lookup_eligibility_miss_with_a_closed_reason() {
    let cases = [
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::Unsupported,
                true,
                true,
                true,
                admitted_safety(),
            ),
            GraphFallbackReason::UnsupportedStage,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                false,
                true,
                true,
                admitted_safety(),
            ),
            GraphFallbackReason::UnsupportedShape,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                false,
                true,
                admitted_safety(),
            ),
            GraphFallbackReason::LayoutMismatch,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::Unsupported,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphFallbackReason::UnsupportedSampling,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::GpuGreedy,
                    GraphOperatorCapability::Unsupported,
                    true,
                ),
            ),
            GraphFallbackReason::BackendNotCaptureSafe,
        ),
    ];

    for (eligibility, reason) in cases {
        assert_eq!(
            select_execution_graph(GraphDispatchRequest::new(
                ExecutionGraphPolicy::Auto,
                eligibility,
                GraphInventoryState::PreparedFull,
            )),
            Ok(GraphDispatchDecision::ExactEager(reason))
        );
    }
}

#[test]
fn graph_dispatch_policy_does_not_own_cuda_or_model_execution() {
    for forbidden in [
        "riley_model",
        "riley_tensor",
        "riley_cuda",
        "batch_executor",
        "PreparedLlama",
        "LlamaBatchExecutor",
        "CudaContext",
        "extern \"C\"",
        "unsafe",
        "Vec<",
        "HashMap",
        "HashSet",
        "usize",
    ] {
        assert!(
            !GRAPH_DISPATCH_SOURCE.contains(forbidden),
            "graph dispatch crossed its scalar-only boundary with {forbidden:?}"
        );
    }
}

const fn static_signature(
    model: GraphModelSignature,
    device: GraphDeviceSignature,
    tensors: GraphTensorSignature,
    geometry: GraphGeometrySignature,
    layout: GraphLayoutSignature,
    implementations: GraphImplementationSignature,
) -> GraphStaticSignature {
    GraphStaticSignature::new(model, device, tensors, geometry, layout, implementations)
}

const fn signature_with_model(model: GraphModelSignature) -> GraphSignature {
    GraphSignature::new(
        static_signature(model, DEVICE, TENSORS, GEOMETRY, LAYOUT, IMPLEMENTATIONS),
        ITERATION_SIGNATURE,
    )
}

const fn signature_with_device(device: GraphDeviceSignature) -> GraphSignature {
    GraphSignature::new(
        static_signature(MODEL, device, TENSORS, GEOMETRY, LAYOUT, IMPLEMENTATIONS),
        ITERATION_SIGNATURE,
    )
}

const fn signature_with_tensors(tensors: GraphTensorSignature) -> GraphSignature {
    GraphSignature::new(
        static_signature(MODEL, DEVICE, tensors, GEOMETRY, LAYOUT, IMPLEMENTATIONS),
        ITERATION_SIGNATURE,
    )
}

const fn signature_with_geometry(geometry: GraphGeometrySignature) -> GraphSignature {
    GraphSignature::new(
        static_signature(MODEL, DEVICE, TENSORS, geometry, LAYOUT, IMPLEMENTATIONS),
        ITERATION_SIGNATURE,
    )
}

const fn signature_with_layout(layout: GraphLayoutSignature) -> GraphSignature {
    GraphSignature::new(
        static_signature(MODEL, DEVICE, TENSORS, GEOMETRY, layout, IMPLEMENTATIONS),
        ITERATION_SIGNATURE,
    )
}

const fn signature_with_implementations(
    implementations: GraphImplementationSignature,
) -> GraphSignature {
    GraphSignature::new(
        static_signature(MODEL, DEVICE, TENSORS, GEOMETRY, LAYOUT, implementations),
        ITERATION_SIGNATURE,
    )
}

const fn signature_with_iteration(iteration: GraphIterationSignature) -> GraphSignature {
    GraphSignature::new(STATIC_SIGNATURE, iteration)
}

fn assert_signature_misses(candidates: &[GraphSignature]) {
    let mut keys = HashSet::from([SIGNATURE]);
    for &candidate in candidates {
        assert_ne!(SIGNATURE, candidate);
        assert_ne!(SIGNATURE.fingerprint(), candidate.fingerprint());
        keys.insert(candidate);
    }
    assert_eq!(keys.len(), candidates.len() + 1);
}
