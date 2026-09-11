use riley_runtime::llama::{
    GraphDispatchError, GraphDispatchMetrics, GraphDispatchMetricsSnapshot, GraphFallbackReason,
    GraphRegistryDispatchDecision, GraphReplaySlot,
};

const GRAPH_DISPATCH_METRICS_SOURCE: &str = include_str!("../src/llama/executor/graph_metrics.rs");

#[test]
fn metrics_keep_graph_selections_and_exact_eager_separate() {
    let mut metrics = GraphDispatchMetrics::new();

    metrics.record_outcome(Ok(GraphRegistryDispatchDecision::FullGraph {
        replay_slot: GraphReplaySlot::new(10),
    }));
    metrics.record_decision(GraphRegistryDispatchDecision::PiecewiseGraph {
        replay_slot: GraphReplaySlot::new(11),
    });
    metrics.record_decision(GraphRegistryDispatchDecision::ExactEager {
        reason: GraphFallbackReason::NotPrepared,
    });

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.full_graph_selected_count(), 1);
    assert_eq!(snapshot.piecewise_graph_selected_count(), 1);
    assert_eq!(snapshot.exact_eager_count(), 1);
    assert_eq!(snapshot.required_graph_rejection_count(), 0);
    assert_eq!(
        snapshot.fallback_reason_count(GraphFallbackReason::NotPrepared),
        1
    );
    assert_eq!(
        snapshot.fallback_reason_count(GraphFallbackReason::GraphPoisoned),
        0
    );
}

#[test]
fn metrics_keep_require_rejections_out_of_exact_eager_counts() {
    let mut metrics = GraphDispatchMetrics::default();

    metrics.record_decision(GraphRegistryDispatchDecision::ExactEager {
        reason: GraphFallbackReason::PolicyDisabled,
    });
    metrics.record_error(GraphDispatchError::RequiredGraphUnavailable {
        reason: GraphFallbackReason::SignatureMismatch,
    });
    metrics.record_outcome(Err(GraphDispatchError::RequiredGraphUnavailable {
        reason: GraphFallbackReason::CapacityDisabled,
    }));

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.exact_eager_count(), 1);
    assert_eq!(snapshot.required_graph_rejection_count(), 2);
    assert_eq!(
        snapshot.fallback_reason_count(GraphFallbackReason::PolicyDisabled),
        1
    );
    assert_eq!(
        snapshot.fallback_reason_count(GraphFallbackReason::SignatureMismatch),
        1
    );
    assert_eq!(
        snapshot.fallback_reason_count(GraphFallbackReason::CapacityDisabled),
        1
    );
}

#[test]
fn every_closed_fallback_reason_has_an_independent_fixed_counter() {
    let reasons = [
        GraphFallbackReason::PolicyDisabled,
        GraphFallbackReason::NotPrepared,
        GraphFallbackReason::UnsupportedStage,
        GraphFallbackReason::UnsupportedShape,
        GraphFallbackReason::UnsupportedSampling,
        GraphFallbackReason::LayoutMismatch,
        GraphFallbackReason::SignatureMismatch,
        GraphFallbackReason::BackendNotCaptureSafe,
        GraphFallbackReason::GraphPoisoned,
        GraphFallbackReason::CapacityDisabled,
        GraphFallbackReason::OperatorCapabilityUnknown,
    ];
    let mut metrics = GraphDispatchMetrics::new();

    for reason in reasons {
        metrics.record_decision(GraphRegistryDispatchDecision::ExactEager { reason });
    }
    let eager_snapshot = metrics.snapshot();
    assert_eq!(eager_snapshot.exact_eager_count(), 11);
    assert_eq!(eager_snapshot.required_graph_rejection_count(), 0);
    for reason in reasons {
        assert_eq!(eager_snapshot.fallback_reason_count(reason), 1);
        metrics.record_error(GraphDispatchError::RequiredGraphUnavailable { reason });
    }

    let rejected_snapshot = metrics.snapshot();
    assert_eq!(rejected_snapshot.exact_eager_count(), 11);
    assert_eq!(rejected_snapshot.required_graph_rejection_count(), 11);
    for reason in reasons {
        assert_eq!(rejected_snapshot.fallback_reason_count(reason), 2);
    }
}

#[test]
fn snapshots_are_copyable_values_and_do_not_reset_counters() {
    fn assert_copy<T: Copy>() {}

    assert_copy::<GraphDispatchMetricsSnapshot>();
    assert_eq!(
        GraphDispatchMetricsSnapshot::default(),
        GraphDispatchMetricsSnapshot::new()
    );

    let mut metrics = GraphDispatchMetrics::new();
    let before = metrics.snapshot();
    metrics.record_decision(GraphRegistryDispatchDecision::FullGraph {
        replay_slot: GraphReplaySlot::new(1),
    });

    assert_eq!(before.full_graph_selected_count(), 0);
    assert_eq!(metrics.snapshot().full_graph_selected_count(), 1);
}

#[test]
fn dispatch_metrics_stay_value_only_and_do_not_retain_graph_identity() {
    let production_source = GRAPH_DISPATCH_METRICS_SOURCE
        .split("\n#[cfg(test)]")
        .next()
        .expect("metrics source must keep test-only saturation coverage separate");
    for forbidden in [
        "riley_model",
        "riley_tensor",
        "riley_cuda",
        "Cuda",
        "PreparedLlama",
        "LlamaBatchExecutor",
        "GraphSignature",
        "GraphReplaySlot",
        "fingerprint",
        "request_id",
        "RequestId",
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
        "format!",
        "alloc::",
        "*const",
        "*mut",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "dispatch metrics crossed its value-only boundary with {forbidden:?}"
        );
    }
    for required in [
        "GraphRegistryDispatchDecision",
        "GraphDispatchError",
        "record_outcome",
        "saturating_add(1)",
    ] {
        assert!(
            production_source.contains(required),
            "dispatch metrics omitted required outcome token {required:?}"
        );
    }
}
