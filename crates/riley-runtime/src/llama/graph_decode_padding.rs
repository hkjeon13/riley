//! Trailing lane topology for one pure-decode graph bucket.
//!
//! This C07-3 value contract only turns an already validated active-row count
//! into the C07 bucket and its trailing placeholder lanes. It never maps a
//! caller row, request, output, or KV block; it neither writes sentinels nor
//! packs metadata or owns graph runtime resources.

use super::graph::select_pure_decode_graph_bucket;

/// Role of one lane in a fixed pure-decode graph bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphPaddingLane {
    /// A lane in the caller-provided active-row prefix.
    Active,
    /// A trailing fixed-shape placeholder lane.
    Padding,
}

/// Checked trailing-padding topology for one pure-decode graph iteration.
///
/// The plan contains counts only. A future owner must still prove all row
/// mapping, padding data, and kernel-mask semantics before it can materialize
/// a placeholder lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PureDecodeGraphPaddingPlan {
    active: u32,
    bucket: u32,
    padding: u32,
}

impl PureDecodeGraphPaddingPlan {
    /// Returns the validated active-row count at the beginning of the bucket.
    #[must_use]
    pub(crate) const fn active_rows(self) -> u32 {
        self.active
    }

    /// Returns the exact C07 bucket selected for this iteration.
    #[must_use]
    pub(crate) const fn bucket_rows(self) -> u32 {
        self.bucket
    }

    /// Returns the number of trailing placeholder lanes.
    #[must_use]
    pub(crate) const fn padding_rows(self) -> u32 {
        self.padding
    }

    /// Classifies one bucket-relative lane without mapping it to caller data.
    #[must_use]
    pub(crate) const fn lane_at(self, bucket_row: u32) -> Option<PureDecodeGraphPaddingLane> {
        if bucket_row < self.active {
            Some(PureDecodeGraphPaddingLane::Active)
        } else if bucket_row < self.bucket {
            Some(PureDecodeGraphPaddingLane::Padding)
        } else {
            None
        }
    }
}

/// Plans trailing padding for an already validated pure-decode active-row count.
///
/// Zero and counts outside the fixed C07 catalog have no plan. This never
/// substitutes the largest bucket for an unsupported count.
#[must_use]
pub(crate) fn plan_pure_decode_graph_padding(
    active_rows: u32,
) -> Option<PureDecodeGraphPaddingPlan> {
    let bucket_rows = select_pure_decode_graph_bucket(active_rows)?;
    let padding_rows = bucket_rows.checked_sub(active_rows)?;
    Some(PureDecodeGraphPaddingPlan {
        active: active_rows,
        bucket: bucket_rows,
        padding: padding_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::{PureDecodeGraphPaddingLane, plan_pure_decode_graph_padding};
    use crate::llama::graph::PURE_DECODE_GRAPH_BUCKETS;

    #[test]
    fn every_supported_active_count_has_one_contiguous_trailing_padding_plan() {
        for active_rows in 1..=32 {
            let plan = plan_pure_decode_graph_padding(active_rows)
                .expect("every C07 active row count must select one bucket");
            assert!(PURE_DECODE_GRAPH_BUCKETS.contains(&plan.bucket_rows()));
            assert_eq!(plan.active_rows() + plan.padding_rows(), plan.bucket_rows());
            for lane in 0..plan.bucket_rows() {
                let expected = if lane < active_rows {
                    PureDecodeGraphPaddingLane::Active
                } else {
                    PureDecodeGraphPaddingLane::Padding
                };
                assert_eq!(plan.lane_at(lane), Some(expected));
            }
            assert_eq!(plan.lane_at(plan.bucket_rows()), None);
        }
    }

    #[test]
    fn boundary_and_exact_buckets_preserve_the_reviewed_trailing_shape() {
        let cases = [(1, 1, 0), (3, 4, 1), (5, 8, 3), (17, 32, 15), (32, 32, 0)];
        for (active_rows, bucket_rows, padding_rows) in cases {
            let plan = plan_pure_decode_graph_padding(active_rows).expect("supported C07 count");
            assert_eq!(plan.active_rows(), active_rows);
            assert_eq!(plan.bucket_rows(), bucket_rows);
            assert_eq!(plan.padding_rows(), padding_rows);
        }
    }

    #[test]
    fn unsupported_counts_have_no_maximum_bucket_fallback() {
        for active_rows in [0, 33, u32::MAX] {
            assert_eq!(plan_pure_decode_graph_padding(active_rows), None);
        }
    }

    #[test]
    fn padding_plan_is_a_copyable_value_only_topology() {
        let plan = plan_pure_decode_graph_padding(3).expect("supported C07 count");
        let copied_plan = plan;
        assert_eq!(copied_plan, plan);
        assert_eq!(
            copied_plan.lane_at(2),
            Some(PureDecodeGraphPaddingLane::Active)
        );
        assert_eq!(
            copied_plan.lane_at(3),
            Some(PureDecodeGraphPaddingLane::Padding)
        );
        assert_eq!(copied_plan.lane_at(4), None);
    }
}
