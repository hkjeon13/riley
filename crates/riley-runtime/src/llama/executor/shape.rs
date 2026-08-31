//! Allocation-free dense-row selection and accounting for Llama batch shapes.
//!
//! This module owns host-only shape policy, cold bucket validation, and success
//! accounting. It deliberately describes only scalar dense-row facts; the
//! enclosing batch executor retains model, plan, and runtime-resource ownership.

use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResult};
use super::metrics::{LlamaBatchShapeBucketHit, LlamaBatchShapeObservation};

const ACTIVE_ROW_BUCKETS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Maximum number of cold-prepared dense-row shapes, including the configured
/// maximum catch-all shape.
pub const MAX_LLAMA_BATCH_SHAPE_BUCKETS: usize = ACTIVE_ROW_BUCKETS.len() + 1;

pub(crate) const fn batch_shape_policy_id(policy: LlamaBatchShapePolicy) -> &'static str {
    match policy {
        LlamaBatchShapePolicy::FixedMaximum => "fixed-max",
        LlamaBatchShapePolicy::ActiveRowBuckets => "power-of-two",
    }
}

/// Dense-row shape selection for one continuous-batch execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LlamaBatchShapePolicy {
    /// Preserve the established rollback graph with `M = max_input_tokens`.
    #[default]
    FixedMaximum,
    /// Select the smallest prepared `1..=256` power-of-two bucket, then the
    /// configured maximum as the final catch-all shape.
    ActiveRowBuckets,
}

impl LlamaBatchShapePolicy {
    /// Selects an exact dense row count for `active_rows` within `maximum_rows`.
    ///
    /// # Errors
    ///
    /// Returns when the active row count is zero or exceeds the prepared
    /// maximum. Selection is host-only and is safe before device dispatch.
    pub fn select_dense_rows(
        self,
        active_rows: usize,
        maximum_rows: usize,
    ) -> LlamaBatchExecutorResult<usize> {
        if active_rows == 0 {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "active_rows",
                reason: "must be greater than zero",
            });
        }
        if active_rows > maximum_rows {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "active_rows",
                reason: "exceeds the prepared dense-row maximum",
            });
        }
        if self == Self::FixedMaximum {
            return Ok(maximum_rows);
        }
        Ok(ACTIVE_ROW_BUCKETS
            .into_iter()
            .find(|&rows| rows >= active_rows && rows < maximum_rows)
            .unwrap_or(maximum_rows))
    }
}

/// Fixed-capacity cold representation of active-row execution shapes.
///
/// Keeping the values inline makes shape selection and hit accounting
/// allocation-free on the iteration hot path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LlamaBatchShapeBuckets {
    values: [usize; MAX_LLAMA_BATCH_SHAPE_BUCKETS],
    len: usize,
}

impl LlamaBatchShapeBuckets {
    pub(crate) const fn automatic(maximum_rows: usize) -> Self {
        let mut values = [0; MAX_LLAMA_BATCH_SHAPE_BUCKETS];
        let mut source_index = 0;
        let mut len = 0;
        while source_index < ACTIVE_ROW_BUCKETS.len() {
            let rows = ACTIVE_ROW_BUCKETS[source_index];
            if rows >= maximum_rows {
                break;
            }
            values[len] = rows;
            len += 1;
            source_index += 1;
        }
        values[len] = maximum_rows;
        len += 1;
        Self { values, len }
    }

    pub(crate) fn custom(buckets: &[usize], maximum_rows: usize) -> LlamaBatchExecutorResult<Self> {
        validate_shape_buckets(buckets, maximum_rows)?;
        let mut values = [0; MAX_LLAMA_BATCH_SHAPE_BUCKETS];
        values[..buckets.len()].copy_from_slice(buckets);
        Ok(Self {
            values,
            len: buckets.len(),
        })
    }

    pub(crate) const fn as_slice(&self) -> &[usize] {
        self.values.split_at(self.len).0
    }

    pub(crate) fn select(self, active_rows: usize) -> LlamaBatchExecutorResult<usize> {
        if active_rows == 0 {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "active_rows",
                reason: "must be greater than zero",
            });
        }
        self.as_slice()
            .iter()
            .copied()
            .find(|&rows| rows >= active_rows)
            .ok_or(LlamaBatchExecutorError::InvalidBatch {
                field: "active_rows",
                reason: "exceeds the prepared dense-row maximum",
            })
    }
}

pub(crate) fn validate_shape_buckets(
    buckets: &[usize],
    maximum_rows: usize,
) -> LlamaBatchExecutorResult<()> {
    if buckets.is_empty() {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "shape_buckets",
            reason: "must contain at least one bucket",
        });
    }
    if buckets.len() > MAX_LLAMA_BATCH_SHAPE_BUCKETS {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "shape_buckets",
            reason: "contains too many buckets",
        });
    }
    if buckets[0] != 1 {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "shape_buckets",
            reason: "the first bucket must be exactly one",
        });
    }
    for pair in buckets.windows(2) {
        if pair[0] >= pair[1] {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "shape_buckets",
                reason: "buckets must be strictly increasing",
            });
        }
    }
    if buckets.last().copied() != Some(maximum_rows) {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "shape_buckets",
            reason: "the final bucket must equal max_input_tokens",
        });
    }
    Ok(())
}

/// Chooses the smallest successfully prepared bucket, falling back to the
/// exact maximum plan when an optional smaller bucket was unavailable.
///
/// The caller validates its policy/configuration first, then provides only
/// scalar rows from successfully prepared optional variants. This keeps
/// runtime selection independent of the batch owner and CUDA plan handles.
pub(crate) fn select_prepared_dense_rows(
    active_rows: usize,
    maximum_rows: usize,
    prepared_rows: impl Iterator<Item = usize>,
) -> usize {
    prepared_rows
        .filter(|&dense_rows| dense_rows >= active_rows)
        .min()
        .unwrap_or(maximum_rows)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LlamaBatchShapeHistory {
    entries: [LlamaBatchShapeBucketHit; MAX_LLAMA_BATCH_SHAPE_BUCKETS],
    len: usize,
    last_success: Option<LlamaBatchShapeObservation>,
}

impl LlamaBatchShapeHistory {
    pub(crate) fn new(
        policy: LlamaBatchShapePolicy,
        buckets: &[usize],
        maximum_rows: usize,
    ) -> LlamaBatchExecutorResult<Self> {
        let mut entries = [LlamaBatchShapeBucketHit::default(); MAX_LLAMA_BATCH_SHAPE_BUCKETS];
        let len = if policy == LlamaBatchShapePolicy::FixedMaximum {
            entries[0] = LlamaBatchShapeBucketHit::new(maximum_rows);
            1
        } else {
            validate_shape_buckets(buckets, maximum_rows)?;
            for (entry, &dense_rows) in entries.iter_mut().zip(buckets) {
                *entry = LlamaBatchShapeBucketHit::new(dense_rows);
            }
            buckets.len()
        };
        Ok(Self {
            entries,
            len,
            last_success: None,
        })
    }

    pub(crate) const fn entries(&self) -> &[LlamaBatchShapeBucketHit] {
        self.entries.split_at(self.len).0
    }

    pub(crate) const fn last_success(&self) -> Option<LlamaBatchShapeObservation> {
        self.last_success
    }

    pub(crate) fn bucket_index(&self, dense_rows: usize) -> LlamaBatchExecutorResult<usize> {
        self.entries()
            .iter()
            .position(|entry| entry.dense_rows() == dense_rows)
            .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                field: "shape_history",
                reason: "selected dense-row bucket is not tracked",
            })
    }

    pub(crate) fn record_success(
        &mut self,
        bucket_index: usize,
        active_rows: usize,
        dense_rows: usize,
    ) {
        debug_assert!(active_rows <= dense_rows);
        debug_assert_eq!(self.entries[bucket_index].dense_rows(), dense_rows);
        self.entries[bucket_index].record_hit();
        self.last_success = Some(LlamaBatchShapeObservation::new(
            active_rows,
            dense_rows,
            dense_rows - active_rows,
        ));
    }

    pub(crate) fn retain_prepared_variants(
        &mut self,
        maximum_rows: usize,
        mut is_prepared_variant: impl FnMut(usize) -> bool,
    ) {
        let mut retained = [LlamaBatchShapeBucketHit::default(); MAX_LLAMA_BATCH_SHAPE_BUCKETS];
        let mut retained_len = 0;
        for entry in self.entries() {
            if entry.dense_rows() == maximum_rows || is_prepared_variant(entry.dense_rows()) {
                retained[retained_len] = *entry;
                retained_len += 1;
            }
        }
        debug_assert!(retained_len != 0);
        self.entries = retained;
        self.len = retained_len;
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaBatchShapeHistory, LlamaBatchShapePolicy, select_prepared_dense_rows};

    #[test]
    fn history_retains_supported_buckets_and_the_maximum_fallback() {
        let mut history =
            LlamaBatchShapeHistory::new(LlamaBatchShapePolicy::ActiveRowBuckets, &[1, 4, 8], 8)
                .expect("valid shape history");
        let supported_bucket = history.bucket_index(4).expect("tracked bucket");
        history.record_success(supported_bucket, 3, 4);

        history.retain_prepared_variants(8, |dense_rows| dense_rows == 4);

        let hits = history.entries();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].dense_rows(), 4);
        assert_eq!(hits[0].hit_count(), 1);
        assert_eq!(hits[1].dense_rows(), 8);
        assert_eq!(hits[1].hit_count(), 0);
    }

    #[test]
    fn unavailable_anchored_shape_uses_the_next_prepared_bucket_or_exact_maximum() {
        let prepared = [2, 8, 64];
        assert_eq!(select_prepared_dense_rows(1, 256, prepared.into_iter()), 2);
        assert_eq!(select_prepared_dense_rows(3, 256, prepared.into_iter()), 8);
        assert_eq!(select_prepared_dense_rows(9, 256, prepared.into_iter()), 64);
        assert_eq!(
            select_prepared_dense_rows(65, 256, prepared.into_iter()),
            256
        );
    }
}
