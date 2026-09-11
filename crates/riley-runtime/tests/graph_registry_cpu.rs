use riley_runtime::llama::{
    GraphComputeType, GraphDataType, GraphDeviceSignature, GraphEntryFootprint, GraphGemmPlanSetId,
    GraphGeometrySignature, GraphImplementationId, GraphImplementationSignature,
    GraphIterationSignature, GraphLayoutSignature, GraphMetadataLayoutSignature,
    GraphModelArchitecture, GraphModelSignature, GraphReductionPolicyId, GraphRegistry,
    GraphRegistryAvailability, GraphRegistryBuildError, GraphRegistryEntry,
    GraphRegistryEntryState, GraphRegistryLimits, GraphRegistryLookup, GraphRegistryUsage,
    GraphReplayMode, GraphReplaySlot, GraphRevisionFingerprint, GraphSamplingBackend,
    GraphSignature, GraphStaticSignature, GraphTensorSignature, GraphWorkloadStage,
};

const GRAPH_REGISTRY_SOURCE: &str = include_str!("../src/llama/executor/graph_registry.rs");
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

const fn signature(model_revision_byte: u8, bucket: u32) -> GraphSignature {
    let model = GraphModelSignature::new(
        GraphModelArchitecture::LlamaDecoder,
        1,
        GraphRevisionFingerprint::from_bytes([model_revision_byte; 32]),
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

const fn entry(
    signature: GraphSignature,
    replay_mode: GraphReplayMode,
    replay_slot: u32,
    state: GraphRegistryEntryState,
    host_bytes: u64,
    device_bytes: u64,
) -> GraphRegistryEntry {
    GraphRegistryEntry::new(
        signature,
        replay_mode,
        GraphReplaySlot::new(replay_slot),
        state,
        GraphEntryFootprint::new(host_bytes, device_bytes),
    )
}

const fn limits(
    maximum_graph_count: usize,
    maximum_full_graph_count: usize,
    maximum_piecewise_graph_count: usize,
    maximum_retained_host_bytes: u64,
    maximum_retained_device_bytes: u64,
) -> GraphRegistryLimits {
    GraphRegistryLimits::new(
        maximum_graph_count,
        maximum_full_graph_count,
        maximum_piecewise_graph_count,
        maximum_retained_host_bytes,
        maximum_retained_device_bytes,
    )
}

#[test]
fn registry_uses_full_signature_and_replay_mode_as_its_exact_key() {
    let exact = signature(1, 1);
    let entries = [
        entry(
            exact,
            GraphReplayMode::FullGraph,
            10,
            GraphRegistryEntryState::Prepared,
            4,
            5,
        ),
        entry(
            exact,
            GraphReplayMode::PiecewiseGraph,
            11,
            GraphRegistryEntryState::Prepared,
            6,
            7,
        ),
    ];
    let registry = GraphRegistry::<2>::try_new(limits(2, 1, 1, 10, 12), &entries)
        .expect("exact entries should fit their cold limits");

    assert_prepared(
        registry.lookup(exact, GraphReplayMode::FullGraph),
        GraphReplaySlot::new(10),
    );
    assert_prepared(
        registry.lookup(exact, GraphReplayMode::PiecewiseGraph),
        GraphReplaySlot::new(11),
    );
    assert_eq!(
        registry.lookup(signature(2, 1), GraphReplayMode::FullGraph),
        GraphRegistryLookup::NotPrepared
    );
    assert_eq!(
        registry.lookup(signature(1, 2), GraphReplayMode::FullGraph),
        GraphRegistryLookup::NotPrepared
    );
}

#[test]
fn registry_keeps_poisoned_entries_exact_and_retained() {
    let poisoned = entry(
        signature(1, 1),
        GraphReplayMode::FullGraph,
        10,
        GraphRegistryEntryState::Poisoned,
        4,
        5,
    );
    let prepared = entry(
        signature(2, 1),
        GraphReplayMode::FullGraph,
        11,
        GraphRegistryEntryState::Prepared,
        6,
        7,
    );
    let registry = GraphRegistry::<2>::try_new(limits(2, 2, 0, 10, 12), &[poisoned, prepared])
        .expect("poisoned entries remain accounted for in a snapshot");

    assert_poisoned(
        registry.lookup(signature(1, 1), GraphReplayMode::FullGraph),
        GraphReplaySlot::new(10),
    );
    assert_prepared(
        registry.lookup(signature(2, 1), GraphReplayMode::FullGraph),
        GraphReplaySlot::new(11),
    );
    assert_eq!(
        registry.lookup(signature(1, 1), GraphReplayMode::PiecewiseGraph),
        GraphRegistryLookup::NotPrepared
    );

    assert_eq!(registry.usage().entry_count(), 2);
    assert_eq!(registry.usage().full_graph_count(), 2);
    assert_eq!(registry.usage().piecewise_graph_count(), 0);
    assert_eq!(registry.usage().retained_host_bytes(), 10);
    assert_eq!(registry.usage().retained_device_bytes(), 12);
}

#[test]
fn disabled_capacity_is_distinct_from_an_exact_not_prepared_miss() {
    let disabled = GraphRegistry::<2>::capacity_disabled();
    assert_eq!(
        disabled.availability(),
        GraphRegistryAvailability::CapacityDisabled
    );
    assert_eq!(
        disabled.lookup(signature(1, 1), GraphReplayMode::FullGraph),
        GraphRegistryLookup::CapacityDisabled
    );

    let disabled_limits = limits(0, 0, 0, 123, 456);
    let disabled_from_limits = GraphRegistry::<2>::try_new(disabled_limits, &[])
        .expect("zero capacity with no entries must form a disabled snapshot");
    assert_eq!(
        disabled_from_limits.lookup(signature(1, 1), GraphReplayMode::FullGraph),
        GraphRegistryLookup::CapacityDisabled
    );
    assert_eq!(disabled_from_limits.limits(), disabled_limits);
}

#[test]
fn registry_rejects_count_mode_and_retained_byte_quota_violations() {
    let first = entry(
        signature(1, 1),
        GraphReplayMode::FullGraph,
        10,
        GraphRegistryEntryState::Prepared,
        8,
        8,
    );
    let second = entry(
        signature(2, 1),
        GraphReplayMode::FullGraph,
        11,
        GraphRegistryEntryState::Prepared,
        3,
        3,
    );
    let piecewise = entry(
        signature(3, 1),
        GraphReplayMode::PiecewiseGraph,
        12,
        GraphRegistryEntryState::Prepared,
        1,
        1,
    );

    assert_eq!(
        GraphRegistry::<3>::try_new(limits(1, 1, 1, 20, 20), &[first, second]),
        Err(GraphRegistryBuildError::EntryCountExceedsMaximum {
            entry_count: 2,
            maximum_graph_count: 1,
        })
    );
    assert_eq!(
        GraphRegistry::<3>::try_new(limits(3, 1, 2, 20, 20), &[first, second]),
        Err(GraphRegistryBuildError::FullGraphCountExceedsMaximum {
            full_graph_count: 2,
            maximum_full_graph_count: 1,
        })
    );
    assert_eq!(
        GraphRegistry::<3>::try_new(limits(3, 2, 0, 20, 20), &[piecewise]),
        Err(GraphRegistryBuildError::PiecewiseGraphCountExceedsMaximum {
            piecewise_graph_count: 1,
            maximum_piecewise_graph_count: 0,
        })
    );
    assert_eq!(
        GraphRegistry::<3>::try_new(limits(3, 2, 1, 10, 20), &[first, second]),
        Err(GraphRegistryBuildError::RetainedHostBytesExceedMaximum {
            maximum_retained_host_bytes: 10,
        })
    );
    assert_eq!(
        GraphRegistry::<3>::try_new(limits(3, 2, 1, 20, 10), &[first, second]),
        Err(GraphRegistryBuildError::RetainedDeviceBytesExceedMaximum {
            maximum_retained_device_bytes: 10,
        })
    );
}

#[test]
fn registry_rejects_overflow_duplicate_keys_slots_and_storage_mismatch() {
    let maximum_host = entry(
        signature(1, 1),
        GraphReplayMode::FullGraph,
        10,
        GraphRegistryEntryState::Prepared,
        u64::MAX,
        0,
    );
    let one_host_byte = entry(
        signature(2, 1),
        GraphReplayMode::PiecewiseGraph,
        11,
        GraphRegistryEntryState::Prepared,
        1,
        0,
    );
    let maximum_device = entry(
        signature(3, 1),
        GraphReplayMode::FullGraph,
        13,
        GraphRegistryEntryState::Prepared,
        0,
        u64::MAX,
    );
    let one_device_byte = entry(
        signature(4, 1),
        GraphReplayMode::PiecewiseGraph,
        14,
        GraphRegistryEntryState::Prepared,
        0,
        1,
    );
    let duplicate_key = entry(
        signature(1, 1),
        GraphReplayMode::FullGraph,
        12,
        GraphRegistryEntryState::Prepared,
        0,
        0,
    );
    let duplicate_slot = entry(
        signature(2, 1),
        GraphReplayMode::PiecewiseGraph,
        10,
        GraphRegistryEntryState::Prepared,
        0,
        0,
    );

    assert_eq!(
        GraphRegistry::<1>::try_new(limits(2, 1, 1, 1, 1), &[]),
        Err(GraphRegistryBuildError::MaximumGraphCountExceedsStorage {
            maximum_graph_count: 2,
            storage_capacity: 1,
        })
    );
    assert_eq!(
        GraphRegistry::<2>::try_new(limits(2, 3, 0, 0, 0), &[]),
        Err(GraphRegistryBuildError::ModeGraphCountExceedsMaximum {
            replay_mode: GraphReplayMode::FullGraph,
            maximum_mode_graph_count: 3,
            maximum_graph_count: 2,
        })
    );
    assert_eq!(
        GraphRegistry::<2>::try_new(limits(2, 0, 0, 0, 0), &[]),
        Err(GraphRegistryBuildError::NoReplayModeCapacity {
            maximum_graph_count: 2,
        })
    );
    assert_eq!(
        GraphRegistry::<2>::try_new(limits(2, 1, 1, u64::MAX, 0), &[maximum_host, one_host_byte]),
        Err(GraphRegistryBuildError::RetainedHostBytesOverflow)
    );
    assert_eq!(
        GraphRegistry::<2>::try_new(
            limits(2, 1, 1, 0, u64::MAX),
            &[maximum_device, one_device_byte]
        ),
        Err(GraphRegistryBuildError::RetainedDeviceBytesOverflow)
    );
    assert_eq!(
        GraphRegistry::<2>::try_new(limits(2, 2, 0, u64::MAX, 0), &[maximum_host, duplicate_key]),
        Err(GraphRegistryBuildError::DuplicateKey {
            existing_index: 0,
            duplicate_index: 1,
        })
    );
    assert_eq!(
        GraphRegistry::<2>::try_new(
            limits(2, 1, 1, u64::MAX, 0),
            &[maximum_host, duplicate_slot]
        ),
        Err(GraphRegistryBuildError::DuplicateReplaySlot {
            existing_index: 0,
            duplicate_index: 1,
        })
    );
}

#[test]
fn registry_is_fixed_storage_send_sync_and_keeps_ownership_boundaries_closed() {
    assert_send_sync::<GraphRegistry<2>>();

    let registry = GraphRegistry::<3>::try_new(limits(3, 2, 1, 20, 20), &[])
        .expect("empty enabled registry should be valid");
    assert_eq!(registry.availability(), GraphRegistryAvailability::Enabled);
    assert_eq!(registry.storage_capacity(), 3);
    assert_eq!(registry.usage(), GraphRegistryUsage::default());
    assert_eq!(registry.limits().maximum_graph_count(), 3);
    assert_eq!(registry.limits().maximum_full_graph_count(), 2);
    assert_eq!(registry.limits().maximum_piecewise_graph_count(), 1);
    assert_eq!(registry.limits().maximum_retained_host_bytes(), 20);
    assert_eq!(registry.limits().maximum_retained_device_bytes(), 20);

    for forbidden in [
        "riley_model",
        "riley_tensor",
        "riley_cuda",
        "PreparedLlama",
        "LlamaBatchExecutor",
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
            !GRAPH_REGISTRY_SOURCE.contains(forbidden),
            "graph registry crossed its metadata-only boundary with {forbidden:?}"
        );
    }
    assert!(
        !GRAPH_REGISTRY_SOURCE.contains(".fingerprint("),
        "registry must use full signature equality instead of fingerprint authority"
    );
    assert!(
        GRAPH_REGISTRY_SOURCE.contains("entry.signature == signature"),
        "registry must compare full graph signatures before replay selection"
    );
}

fn assert_prepared(lookup: GraphRegistryLookup<'_>, expected_slot: GraphReplaySlot) {
    match lookup {
        GraphRegistryLookup::Prepared(entry) => {
            assert_eq!(entry.replay_slot(), expected_slot);
            assert_eq!(entry.state(), GraphRegistryEntryState::Prepared);
        }
        other => panic!("expected prepared exact registry entry, got {other:?}"),
    }
}

fn assert_poisoned(lookup: GraphRegistryLookup<'_>, expected_slot: GraphReplaySlot) {
    match lookup {
        GraphRegistryLookup::Poisoned(entry) => {
            assert_eq!(entry.replay_slot(), expected_slot);
            assert_eq!(entry.state(), GraphRegistryEntryState::Poisoned);
        }
        other => panic!("expected poisoned exact registry entry, got {other:?}"),
    }
}

fn assert_send_sync<T: Send + Sync>() {}
