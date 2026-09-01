//! C07 V1 observation of opaque C06 registry-selection outcomes.
//!
//! C07-22 observes only C07-21's closed selection result. A bound C06
//! decision is recorded once while the caller retains its result unchanged; a
//! C07 ineligibility is not a C06 dispatch outcome and therefore leaves the
//! metrics untouched. This module never resolves a replay slot or owns an
//! execution resource.

use super::graph_decode_c06_registry_dispatch::{
    PureDecodeGraphV1C06RegistryDispatch, PureDecodeGraphV1C06RegistryDispatchResult,
};
use super::graph_metrics::GraphDispatchMetrics;

/// Records one C06 selection outcome without changing its C07 meaning.
///
/// A bound result records C06's opaque decision exactly once. A `require`
/// rejection records its original C06 error exactly once. C07-ineligible
/// candidates did not enter C06 selection, so this function leaves their typed
/// reason and the metrics untouched. The caller retains the result to preserve
/// its exact identity without copying the large closed value.
pub(crate) fn observe_pure_decode_graph_v1_c06_registry_dispatch(
    outcome: &PureDecodeGraphV1C06RegistryDispatchResult,
    metrics: &mut GraphDispatchMetrics,
) {
    match *outcome {
        Ok(PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason)) => {
            let _ = reason;
        }
        Ok(PureDecodeGraphV1C06RegistryDispatch::Bound(binding)) => {
            metrics.record_decision(binding.decision());
        }
        Err(error) => {
            metrics.record_error(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::observe_pure_decode_graph_v1_c06_registry_dispatch;
    use crate::llama::graph::{GraphDispatchError, GraphFallbackReason};
    use crate::llama::graph_decode_c06_registry_dispatch::{
        PureDecodeGraphV1C06RegistryDispatch, PureDecodeGraphV1C06RegistryDispatchResult,
    };
    use crate::llama::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
    use crate::llama::graph_metrics::GraphDispatchMetrics;

    #[test]
    fn ineligible_candidate_preserves_the_closed_reason_without_metric_observation() {
        let reason = PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 };
        let mut metrics = GraphDispatchMetrics::new();
        let before = metrics.snapshot();
        let outcome: PureDecodeGraphV1C06RegistryDispatchResult =
            Ok(PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason));

        observe_pure_decode_graph_v1_c06_registry_dispatch(&outcome, &mut metrics);
        assert_eq!(
            outcome,
            Ok(PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason)),
        );
        assert_eq!(metrics.snapshot(), before);
    }

    #[test]
    fn require_error_is_observed_once_and_retained_unchanged() {
        let error = GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::SignatureMismatch,
        };
        let mut metrics = GraphDispatchMetrics::new();

        let outcome: PureDecodeGraphV1C06RegistryDispatchResult = Err(error);
        observe_pure_decode_graph_v1_c06_registry_dispatch(&outcome, &mut metrics);
        assert_eq!(outcome, Err(error));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.full_graph_selected_count(), 0);
        assert_eq!(snapshot.piecewise_graph_selected_count(), 0);
        assert_eq!(snapshot.exact_eager_count(), 0);
        assert_eq!(snapshot.required_graph_rejection_count(), 1);
        assert_eq!(
            snapshot.fallback_reason_count(GraphFallbackReason::SignatureMismatch),
            1,
        );
    }
}
