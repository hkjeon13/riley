//! Allocation-free value types published by the Llama batch executor.
//!
//! These types describe completed shape selection only.  They own neither
//! execution resources nor policy, so later executor components can publish
//! the same facts without depending on the batch-owner implementation.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

/// Shape facts from the most recent successfully completed iteration.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaBatchShapeObservation {
    active_rows: usize,
    selected_dense_rows: usize,
    padding_rows: usize,
}

impl LlamaBatchShapeObservation {
    #[must_use]
    pub(crate) const fn new(
        active_rows: usize,
        selected_dense_rows: usize,
        padding_rows: usize,
    ) -> Self {
        Self {
            active_rows,
            selected_dense_rows,
            padding_rows,
        }
    }

    #[must_use]
    pub const fn active_rows(self) -> usize {
        self.active_rows
    }

    #[must_use]
    pub const fn selected_dense_rows(self) -> usize {
        self.selected_dense_rows
    }

    #[must_use]
    pub const fn padding_rows(self) -> usize {
        self.padding_rows
    }
}

/// Allocation-free cumulative hit counter for one cold-prepared shape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LlamaBatchShapeBucketHit {
    dense_rows: usize,
    hit_count: u64,
}

impl LlamaBatchShapeBucketHit {
    #[must_use]
    pub(crate) const fn new(dense_rows: usize) -> Self {
        Self {
            dense_rows,
            hit_count: 0,
        }
    }

    pub(crate) fn record_hit(&mut self) {
        self.hit_count = self.hit_count.saturating_add(1);
    }

    #[must_use]
    pub const fn dense_rows(self) -> usize {
        self.dense_rows
    }

    #[must_use]
    pub const fn hit_count(self) -> u64 {
        self.hit_count
    }
}
