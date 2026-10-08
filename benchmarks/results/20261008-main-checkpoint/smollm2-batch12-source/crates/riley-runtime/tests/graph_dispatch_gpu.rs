//! GPU-only C06 dispatcher acceptance using a deliberately tiny synthetic C05 owner.
//!
//! The owner table and fixed-fill graph in this test are not C07 model wiring.
//! They prove only that C06's exact logical replay-slot selection can drive a
//! resource owner without allowing `auto`, `require`, or `disabled` fallback
//! paths to launch that owner.

use std::error::Error;

use riley_cuda::{
    CudaContext, CudaGraphCaptureMode, CudaResult, CudaRuntime, CudaStream, OwnedGraphExec,
    OwnedGraphFillResources,
};
use riley_runtime::llama::{
    ExecutionGraphPolicy, GraphCaptureSafety, GraphComputeType, GraphDataType,
    GraphDeviceSignature, GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest,
    GraphEntryFootprint, GraphFallbackReason, GraphGemmPlanSetId, GraphGeometrySignature,
    GraphImplementationId, GraphImplementationSignature, GraphInventoryState,
    GraphIterationSignature, GraphLayoutSignature, GraphMetadataLayoutSignature,
    GraphModelArchitecture, GraphModelSignature, GraphOperatorCapability, GraphReductionPolicyId,
    GraphRegistry, GraphRegistryDispatchDecision, GraphRegistryEntry, GraphRegistryEntryState,
    GraphRegistryLimits, GraphReplayMode, GraphReplaySlot, GraphRevisionFingerprint,
    GraphSamplingBackend, GraphSignature, GraphStaticSignature, GraphTensorSignature,
    GraphWorkloadStage, select_registered_execution_graph,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ELEMENT_COUNT: u64 = 256;
const REPLAYS: usize = 64;
const FINAL_VALUE: f32 = -7.25;
const FULL_SLOT: GraphReplaySlot = GraphReplaySlot::new(17);

const DEVICE: GraphDeviceSignature = GraphDeviceSignature::new(8, 9, 12_804, 12_804, 1);
const TENSORS: GraphTensorSignature = GraphTensorSignature::new(
    GraphDataType::BFloat16,
    GraphDataType::BFloat16,
    GraphComputeType::Float32,
);
const GEOMETRY: GraphGeometrySignature =
    GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 128);
const METADATA_LAYOUT: GraphMetadataLayoutSignature =
    GraphMetadataLayoutSignature::new(1, [0xC6; 32]);
const LAYOUT: GraphLayoutSignature = GraphLayoutSignature::new(8_192, 16, 1, METADATA_LAYOUT);
const IMPLEMENTATIONS: GraphImplementationSignature = GraphImplementationSignature::new(
    GraphImplementationId::new(1),
    GraphImplementationId::new(2),
    GraphImplementationId::new(3),
    GraphImplementationId::new(4),
    GraphGemmPlanSetId::new(5),
    GraphReductionPolicyId::new(6),
);

/// Test-only C06 logical-slot owner. It deliberately retains the C05 graph
/// independently of the immutable registry, just like a future owner must.
struct SyntheticFullGraphOwner {
    signature: GraphSignature,
    replay_slot: GraphReplaySlot,
    exec: OwnedGraphExec,
    replay_count: usize,
}

impl SyntheticFullGraphOwner {
    fn new(
        context: &CudaContext,
        signature: GraphSignature,
        replay_slot: GraphReplaySlot,
    ) -> TestResult<Self> {
        let capture_stream = context.create_stream()?;
        let byte_len = ELEMENT_COUNT
            .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
            .ok_or("C06 synthetic graph byte length overflow")?;
        let buffer = context.allocate_device_buffer(byte_len)?;
        let mut capture = capture_stream.begin_owned_graph_fill_capture(
            buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        capture.enqueue_fill(FINAL_VALUE)?;
        let exec = capture.end()?.instantiate()?;
        Ok(Self {
            signature,
            replay_slot,
            exec,
            replay_count: 0,
        })
    }

    fn replay(
        &mut self,
        selected_signature: GraphSignature,
        selected_slot: GraphReplaySlot,
    ) -> TestResult {
        if selected_signature != self.signature {
            return Err("C06 synthetic owner rejected a mismatched full signature".into());
        }
        if selected_slot != self.replay_slot {
            return Err("C06 synthetic owner rejected a mismatched replay slot".into());
        }
        self.exec.launch()?.finish()?;
        self.replay_count = self
            .replay_count
            .checked_add(1)
            .ok_or("C06 synthetic replay counter overflow")?;
        Ok(())
    }

    fn replay_count(&self) -> usize {
        self.replay_count
    }

    fn close(self) -> CudaResult<OwnedGraphFillResources> {
        self.exec.close()
    }
}

fn signature(bucket: u32) -> GraphSignature {
    let model = GraphModelSignature::new(
        GraphModelArchitecture::LlamaDecoder,
        1,
        GraphRevisionFingerprint::from_bytes([1; 32]),
        1,
    );
    GraphSignature::new(
        GraphStaticSignature::new(model, DEVICE, TENSORS, GEOMETRY, LAYOUT, IMPLEMENTATIONS),
        GraphIterationSignature::new(
            GraphWorkloadStage::PureDecode,
            bucket,
            GraphSamplingBackend::GpuGreedy,
        ),
    )
}

fn request(policy: ExecutionGraphPolicy) -> GraphDispatchRequest {
    GraphDispatchRequest::new(
        policy,
        GraphDispatchEligibility::new(
            GraphWorkloadStage::PureDecode,
            true,
            true,
            true,
            GraphCaptureSafety::new(
                GraphSamplingBackend::GpuGreedy,
                GraphOperatorCapability::Supported,
                true,
            ),
        ),
        GraphInventoryState::NotPrepared,
    )
}

fn registry(signature: GraphSignature) -> TestResult<GraphRegistry<1>> {
    Ok(GraphRegistry::try_new(
        GraphRegistryLimits::new(1, 1, 0, 4_096, 4_096),
        &[GraphRegistryEntry::new(
            signature,
            GraphReplayMode::FullGraph,
            FULL_SLOT,
            GraphRegistryEntryState::Prepared,
            GraphEntryFootprint::new(1, 1),
        )],
    )?)
}

fn eager_fill(context: &CudaContext, stream: &mut CudaStream, value: f32) -> TestResult<Vec<f32>> {
    let kernel = context.kernel();
    let values = kernel.launch_fill(stream, ELEMENT_COUNT, value)?.finish()?;
    drop(kernel);
    Ok(values)
}

fn download_graph_values(
    context: &CudaContext,
    buffer: &mut riley_cuda::CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult<Vec<f32>> {
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("C06 synthetic graph download byte length overflow")?;
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;
    let mut bytes = vec![0_u8; usize::try_from(byte_len)?];
    buffer.download_to_slice(0, &mut bytes, &mut staging, stream)?;
    staging.close()?;
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn assert_f32_bits_equal(actual: &[f32], expected: &[f32], message: &str) {
    assert_eq!(actual.len(), expected.len(), "{message}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{message}: bit mismatch at element {index}"
        );
    }
}

fn first_context() -> TestResult<CudaContext> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(context)
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn auto_full_graph_dispatch_resolves_one_exact_slot_and_matches_eager_after_64_replays()
-> TestResult {
    let context = first_context()?;
    let allocation_baseline = context.allocation_stats()?;
    let exact_signature = signature(1);
    let registry = registry(exact_signature)?;
    let mut owner = SyntheticFullGraphOwner::new(&context, exact_signature, FULL_SLOT)?;
    let allocations_before_replay = context.allocation_stats()?;

    assert!(
        owner
            .replay(exact_signature, GraphReplaySlot::new(FULL_SLOT.value() + 1))
            .is_err(),
        "a stale logical slot must not launch the retained synthetic graph"
    );
    assert_eq!(owner.replay_count(), 0);

    for _ in 0..REPLAYS {
        let decision = select_registered_execution_graph(
            request(ExecutionGraphPolicy::Auto),
            exact_signature,
            &registry,
        )?;
        match decision {
            GraphRegistryDispatchDecision::FullGraph { replay_slot } => {
                owner.replay(exact_signature, replay_slot)?;
            }
            other => {
                return Err(format!("exact C06 full graph unexpectedly selected {other:?}").into());
            }
        }
    }
    assert_eq!(owner.replay_count(), REPLAYS);
    assert_eq!(
        context.allocation_stats()?,
        allocations_before_replay,
        "C06 dispatch/replay must not change retained CUDA allocation accounting"
    );

    let mut eager_stream = context.create_stream()?;
    let eager_values = eager_fill(&context, &mut eager_stream, FINAL_VALUE)?;
    let resources = owner.close()?;
    let (mut graph_stream, mut graph_buffer) = resources.into_parts();
    let graph_values = download_graph_values(&context, &mut graph_buffer, &mut graph_stream)?;
    assert_f32_bits_equal(
        &graph_values,
        &eager_values,
        "C06 exact full-graph dispatch must match eager fixed-fill bytes",
    );

    graph_buffer.close()?;
    graph_stream.close()?;
    eager_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn auto_require_and_disabled_fallbacks_never_launch_the_synthetic_full_graph() -> TestResult {
    let context = first_context()?;
    let allocation_baseline = context.allocation_stats()?;
    let exact_signature = signature(1);
    let missing_signature = signature(2);
    let registry = registry(exact_signature)?;
    let owner = SyntheticFullGraphOwner::new(&context, exact_signature, FULL_SLOT)?;
    let allocations_after_prepare = context.allocation_stats()?;
    let mut eager_stream = context.create_stream()?;
    let mut eager_runs = 0_usize;

    let auto = select_registered_execution_graph(
        request(ExecutionGraphPolicy::Auto),
        missing_signature,
        &registry,
    )?;
    assert_eq!(
        auto,
        GraphRegistryDispatchDecision::ExactEager {
            reason: GraphFallbackReason::NotPrepared,
        }
    );
    let eager_values = eager_fill(&context, &mut eager_stream, FINAL_VALUE)?;
    eager_runs += 1;
    assert_f32_bits_equal(
        &eager_values,
        &vec![FINAL_VALUE; usize::try_from(ELEMENT_COUNT)?],
        "C06 auto fallback must execute exact eager work",
    );

    assert_eq!(
        select_registered_execution_graph(
            request(ExecutionGraphPolicy::Require),
            missing_signature,
            &registry,
        ),
        Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::NotPrepared,
        }),
        "require must fail closed instead of launching graph or eager work"
    );

    let disabled = select_registered_execution_graph(
        request(ExecutionGraphPolicy::Disabled),
        exact_signature,
        &registry,
    )?;
    assert_eq!(
        disabled,
        GraphRegistryDispatchDecision::ExactEager {
            reason: GraphFallbackReason::PolicyDisabled,
        }
    );
    let disabled_eager_values = eager_fill(&context, &mut eager_stream, FINAL_VALUE)?;
    eager_runs += 1;
    assert_f32_bits_equal(
        &disabled_eager_values,
        &vec![FINAL_VALUE; usize::try_from(ELEMENT_COUNT)?],
        "C06 disabled policy must execute exact eager work",
    );

    assert_eq!(eager_runs, 2);
    assert_eq!(
        owner.replay_count(),
        0,
        "fallback decisions must never resolve or launch the synthetic graph owner"
    );
    assert_eq!(
        context.allocation_stats()?,
        allocations_after_prepare,
        "C06 eager fallbacks must not retain CUDA allocations beside the cold graph"
    );

    let resources = owner.close()?;
    let (graph_stream, graph_buffer) = resources.into_parts();
    graph_buffer.close()?;
    graph_stream.close()?;
    eager_stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)
}
