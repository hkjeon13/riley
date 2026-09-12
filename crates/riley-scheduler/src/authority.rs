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

impl AuthorizedExecution<'_> {
    /// Only the execution adapter can form this packet expectation. Its owner
    /// identity and monotonically assigned cookies must come from the retained
    /// runtime owner; request/KV fields always come from this scheduler borrow.
    #[allow(dead_code)]
    pub(crate) fn descriptor_expectation(
        &self,
        owner: crate::descriptor::OwnerExpectation,
        replay: u64,
        cookies: &[u64],
    ) -> crate::descriptor::Result<crate::descriptor::SubmissionExpectation> {
        self.descriptor_expectation_with_mode(owner, replay, cookies, crate::descriptor::ResultMode::FullLogits)
    }
    pub(crate) fn descriptor_expectation_with_mode(
        &self, owner: crate::descriptor::OwnerExpectation, replay: u64, cookies: &[u64],
        mode: crate::descriptor::ResultMode,
    ) -> crate::descriptor::Result<crate::descriptor::SubmissionExpectation> {
        use crate::descriptor::{
            BlockOwnership, Error, ReservationExpectation, Stage, SubmissionExpectation,
        };
        let bad = || Error {
            field: "scheduler authority",
            reason: "incompatible live reservation or owner geometry",
        };
        if owner.physical_block_count as usize != self.physical_block_count()
            || cookies.len() != self.rows.len()
            || (!self.plan.prefill_items().is_empty() && !self.plan.decode_items().is_empty())
        {
            return Err(bad());
        }
        let stage = if self.plan.prefill_items().is_empty() {
            Stage::Decode
        } else {
            Stage::Prefill128
        };
        let to32 = |v: usize| u32::try_from(v).map_err(|_| bad());
        let mut rows = Vec::new();
        for ((work, row), cookie) in self
            .plan
            .prefill_items()
            .iter()
            .chain(self.plan.decode_items())
            .zip(&self.rows)
            .zip(cookies)
        {
            if work.request_id() != row.request_id {
                return Err(bad());
            }
            rows.push(ReservationExpectation {
                sequence_tag: row.request_id.get(),
                cookie: *cookie,
                input_tokens: work.input_tokens().to_vec(),
                committed_length: to32(row.committed_length)?,
                target_length: to32(work.target_logical_length())?,
                output_slot: work.output_slot().ok_or_else(bad)?.get(),
                generated_index: to32(row.generated_index)?,
                max_output_tokens: to32(row.max_output_tokens)?,
                physical_ids: row.table.physical_block_ids().to_vec(),
                valid_tokens: row.table.valid_tokens().to_vec(),
            });
        }
        let expectation = SubmissionExpectation {
            owner,
            replay_id: replay,
            iteration_id: self.plan.iteration_id().get(),
            stage,
            mode,
            rows,
            block_ownership: self
                .block_owners
                .iter()
                .map(|(id, tag)| BlockOwnership {
                    physical_id: *id,
                    sequence_tag: tag.get(),
                })
                .collect(),
        };
        crate::descriptor::validate_expectations(&expectation)?;
        Ok(expectation)
    }
}

#[cfg(test)]
mod tests {
    use crate::descriptor::{
        CatalogEntry, OwnerExpectation, ResultMode, Stage, decode_request, encode_request,
    };
    use crate::{
        ExecutionShapePolicy, OverloadPolicy, RequestDescriptor, Scheduler, SchedulerConfig,
    };
    #[test]
    fn live_scheduler_authority_encodes_exact_request_and_rejects_foreign_geometry() {
        let mut scheduler = Scheduler::new_with_execution_shape(
            SchedulerConfig {
                max_waiting_requests: 8,
                max_waiting_prompt_tokens: 1024,
                max_active_sequences: 4,
                max_sequence_tokens: 160,
                iteration_token_budget: 128,
                max_prefill_chunk_tokens: 128,
                aging_threshold_ns: 1,
                overload_policy: OverloadPolicy::Wait,
                admission_timeout_ns: None,
                max_promised_kv_blocks: 40,
                metrics_window_samples: 16,
            },
            riley_runtime::paged_kv::KvLayout::checked(30, 40, 3, 64).unwrap(),
            ExecutionShapePolicy::CompletePrefill128DecodeN,
        )
        .unwrap();
        let id = scheduler
            .submit(RequestDescriptor::new(vec![17; 128], 32), 0)
            .unwrap()
            .request_id();
        let plan = scheduler.plan_iteration(1).unwrap().into_parts().0.unwrap();
        let authority = scheduler.authorize_execution(&plan).unwrap();
        let owner = OwnerExpectation {
            generation: 1,
            last_accepted_replay: 0,
            catalog_digest: [0x5a; 32],
            max_active_rows: 4,
            physical_block_count: 40,
            catalog: vec![CatalogEntry {
                stage: Stage::Prefill128,
                bucket: 1,
                mode: ResultMode::FullLogits,
            }],
        };
        let expectation = authority
            .descriptor_expectation(owner.clone(), 1, &[1])
            .unwrap();
        assert_eq!(expectation.rows[0].sequence_tag, id.get());
        assert_eq!(expectation.rows[0].committed_length, 0);
        assert_eq!(expectation.rows[0].generated_index, 0);
        let packet = encode_request(&expectation).unwrap();
        assert_eq!(packet.len(), 1792);
        decode_request(&packet, &expectation).unwrap();
        let mut other = owner;
        other.physical_block_count = 41;
        assert!(authority.descriptor_expectation(other, 1, &[1]).is_err());
        let mut missing_ownership = expectation;
        missing_ownership.block_ownership.clear();
        assert!(encode_request(&missing_ownership).is_err());
        drop(authority);
        scheduler
            .abort_iteration(plan.iteration_id(), crate::ExecutionAbort::NotDispatched, 2)
            .unwrap();
        scheduler.close(3, None).unwrap();
    }
}
