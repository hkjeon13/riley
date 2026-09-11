//! Allocation-free observation of closed graph-dispatch outcomes.
//!
//! These counters describe policy selection only. They own no graph resource,
//! never retain a replay slot or signature, and do not imply that native graph
//! work was launched or completed.

use super::graph::{
    GRAPH_FALLBACK_REASON_METRIC_COUNT, GraphDispatchDecision, GraphDispatchError,
    GraphFallbackReason,
};
use super::graph_registry_dispatch::GraphRegistryDispatchDecision;

const FALLBACK_REASON_METRIC_COUNT: usize = GRAPH_FALLBACK_REASON_METRIC_COUNT as usize;

/// Immutable snapshot of cumulative graph-dispatch observations.
///
/// A graph selection is intentionally not called a replay or a hit: C07 owns
/// native launch and completion evidence. A required-policy rejection records
/// a reason but is distinct from an exact-eager selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphDispatchMetricsSnapshot {
    full_graph_selected_count: u64,
    piecewise_graph_selected_count: u64,
    exact_eager_count: u64,
    required_graph_rejection_count: u64,
    fallback_reason_counts: [u64; FALLBACK_REASON_METRIC_COUNT],
}

impl Default for GraphDispatchMetricsSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphDispatchMetricsSnapshot {
    /// Creates an empty fixed-size observation snapshot.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            full_graph_selected_count: 0,
            piecewise_graph_selected_count: 0,
            exact_eager_count: 0,
            required_graph_rejection_count: 0,
            fallback_reason_counts: [0; FALLBACK_REASON_METRIC_COUNT],
        }
    }

    /// Returns the number of full-graph policy selections.
    #[must_use]
    pub const fn full_graph_selected_count(self) -> u64 {
        self.full_graph_selected_count
    }

    /// Returns the number of piecewise-graph policy selections.
    #[must_use]
    pub const fn piecewise_graph_selected_count(self) -> u64 {
        self.piecewise_graph_selected_count
    }

    /// Returns the number of exact-eager policy selections.
    #[must_use]
    pub const fn exact_eager_count(self) -> u64 {
        self.exact_eager_count
    }

    /// Returns the number of required-policy rejections.
    #[must_use]
    pub const fn required_graph_rejection_count(self) -> u64 {
        self.required_graph_rejection_count
    }

    /// Returns the closed-reason count across eager selections and rejections.
    #[must_use]
    pub const fn fallback_reason_count(self, reason: GraphFallbackReason) -> u64 {
        self.fallback_reason_counts[reason.metric_index() as usize]
    }
}

/// Mutable in-process counters for one graph-dispatch observation stream.
///
/// Callers record each returned dispatch outcome exactly once. Recording only
/// changes these scalar counters; it cannot alter graph policy, registry,
/// resource ownership, or execution.
#[derive(Debug, Eq, PartialEq)]
pub struct GraphDispatchMetrics {
    snapshot: GraphDispatchMetricsSnapshot,
}

impl Default for GraphDispatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphDispatchMetrics {
    /// Creates empty graph-dispatch counters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            snapshot: GraphDispatchMetricsSnapshot::new(),
        }
    }

    /// Returns a value snapshot without resetting or mutating the counters.
    #[must_use]
    pub const fn snapshot(&self) -> GraphDispatchMetricsSnapshot {
        self.snapshot
    }

    /// Records one result returned by the registry dispatch adapter.
    pub fn record_outcome(
        &mut self,
        outcome: Result<GraphRegistryDispatchDecision, GraphDispatchError>,
    ) {
        match outcome {
            Ok(decision) => self.record_decision(decision),
            Err(error) => self.record_error(error),
        }
    }

    /// Records one successful graph-policy decision without retaining its slot.
    pub fn record_decision(&mut self, decision: GraphRegistryDispatchDecision) {
        match decision.decision() {
            GraphDispatchDecision::FullGraph => {
                self.snapshot.full_graph_selected_count =
                    self.snapshot.full_graph_selected_count.saturating_add(1);
            }
            GraphDispatchDecision::PiecewiseGraph => {
                self.snapshot.piecewise_graph_selected_count = self
                    .snapshot
                    .piecewise_graph_selected_count
                    .saturating_add(1);
            }
            GraphDispatchDecision::ExactEager(reason) => {
                self.snapshot.exact_eager_count = self.snapshot.exact_eager_count.saturating_add(1);
                self.record_fallback_reason(reason);
            }
        }
    }

    /// Records one fail-closed required-policy rejection.
    pub fn record_error(&mut self, error: GraphDispatchError) {
        match error {
            GraphDispatchError::RequiredGraphUnavailable { reason } => {
                self.snapshot.required_graph_rejection_count = self
                    .snapshot
                    .required_graph_rejection_count
                    .saturating_add(1);
                self.record_fallback_reason(reason);
            }
        }
    }

    fn record_fallback_reason(&mut self, reason: GraphFallbackReason) {
        let count = &mut self.snapshot.fallback_reason_counts[reason.metric_index() as usize];
        *count = count.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{FALLBACK_REASON_METRIC_COUNT, GraphDispatchMetrics, GraphDispatchMetricsSnapshot};
    use crate::llama::graph::{GraphDispatchError, GraphFallbackReason};
    use crate::llama::graph_registry::GraphReplaySlot;
    use crate::llama::graph_registry_dispatch::GraphRegistryDispatchDecision;

    #[test]
    fn counters_saturate_without_wrapping_or_retaining_graph_identity() {
        let mut metrics = GraphDispatchMetrics {
            snapshot: GraphDispatchMetricsSnapshot {
                full_graph_selected_count: u64::MAX,
                piecewise_graph_selected_count: u64::MAX,
                exact_eager_count: u64::MAX,
                required_graph_rejection_count: u64::MAX,
                fallback_reason_counts: [u64::MAX; FALLBACK_REASON_METRIC_COUNT],
            },
        };

        metrics.record_decision(GraphRegistryDispatchDecision::FullGraph {
            replay_slot: GraphReplaySlot::new(1),
        });
        metrics.record_decision(GraphRegistryDispatchDecision::PiecewiseGraph {
            replay_slot: GraphReplaySlot::new(2),
        });
        metrics.record_decision(GraphRegistryDispatchDecision::ExactEager {
            reason: GraphFallbackReason::NotPrepared,
        });
        metrics.record_error(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::GraphPoisoned,
        });

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.full_graph_selected_count(), u64::MAX);
        assert_eq!(snapshot.piecewise_graph_selected_count(), u64::MAX);
        assert_eq!(snapshot.exact_eager_count(), u64::MAX);
        assert_eq!(snapshot.required_graph_rejection_count(), u64::MAX);
        assert_eq!(
            snapshot.fallback_reason_count(GraphFallbackReason::NotPrepared),
            u64::MAX
        );
        assert_eq!(
            snapshot.fallback_reason_count(GraphFallbackReason::GraphPoisoned),
            u64::MAX
        );
    }
}
