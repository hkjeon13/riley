//! Scoped execution authority derived from live scheduler reservations.
use crate::plan::{IterationPlan, OwnedBlockTable};
use crate::{RequestId, Scheduler};
/// One scheduler-validated request reservation; values cannot be constructed by callers.
pub struct AuthorizedExecutionRow {
    pub(crate) request_id: RequestId,
    pub(crate) committed_length: usize,
    pub(crate) generated_index: usize,
    pub(crate) max_output_tokens: usize,
    pub(crate) table: OwnedBlockTable,
}
impl AuthorizedExecutionRow {
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }
    pub fn committed_length(&self) -> usize {
        self.committed_length
    }
    pub fn generated_index(&self) -> usize {
        self.generated_index
    }
    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }
    pub fn table(&self) -> &OwnedBlockTable {
        &self.table
    }
}
/// Holds an immutable scheduler borrow until execution has completed. While this
/// token exists no reservation can be settled, cancelled or replaced through a
/// mutable scheduler call. Dropping it performs no commit or rollback.
pub struct AuthorizedExecution<'a> {
    pub(crate) scheduler: &'a Scheduler,
    pub(crate) plan: &'a IterationPlan,
    pub(crate) rows: Vec<AuthorizedExecutionRow>,
    pub(crate) block_owners: Vec<(u32, RequestId)>,
}
impl AuthorizedExecution<'_> {
    pub fn plan(&self) -> &IterationPlan {
        self.plan
    }
    pub fn rows(&self) -> &[AuthorizedExecutionRow] {
        &self.rows
    }
    /// Includes off-batch resident requests and tentative destinations.
    pub fn block_owners(&self) -> &[(u32, RequestId)] {
        &self.block_owners
    }
    pub fn physical_block_count(&self) -> usize {
        self.scheduler.pool_physical_block_count()
    }
}
