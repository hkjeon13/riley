//! Scoped execution authority derived from live scheduler reservations.
use crate::plan::{IterationPlan, OwnedBlockTable};
use crate::{RequestId, Scheduler};
/// One scheduler-validated request reservation; values cannot be constructed by callers.
pub struct AuthorizedExecutionRow {
    pub(crate) request_id: RequestId,
    pub(crate) committed_length: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_index: usize,
    pub(crate) max_output_tokens: usize,
    pub(crate) table: OwnedBlockTable,
}
impl AuthorizedExecutionRow {
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }
    pub fn prompt_tokens(&self) -> usize { self.prompt_tokens }
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
            let progress=crate::descriptor::shape_progress::Progress {
                prompt_tokens:to32(row.prompt_tokens)?, output_limit:to32(row.max_output_tokens)?, context_tokens:160,
                committed_tokens:to32(row.committed_length)?, input_tokens:to32(work.input_tokens().len())?, generated_index:to32(row.generated_index)?,
                stage:if stage==Stage::Prefill128 { crate::descriptor::shape_progress::InputStage::Prefill } else { crate::descriptor::shape_progress::InputStage::Decode },
            }.validate()?;
            // v2 has no prompt-length field: keep it restricted until the new wire
            // and native executor carry this scheduler-owned value end to end.
            if row.prompt_tokens!=128 || progress.target_tokens!=to32(work.target_logical_length())? || progress.logits_input_row.is_none() { return Err(bad()); }
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


/// Geometry/identity supplied only by the retained V3 execution adapter.
/// This is not a catalog authorization or a GPU completion receipt.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct VariableOwnerGeometry {
    pub generation: u64,
    pub last_accepted_replay: u64,
    pub catalog_digest: [u8;32],
    pub max_active_rows: u32,
    pub physical_block_count: u32,
    pub context_tokens: u32,
    pub packed_prefill: bool,
    pub mixed_execution: bool,
}
impl AuthorizedExecution<'_> {
    #[allow(dead_code)]
    pub(crate) fn variable_descriptor_expectation(&self,owner:&VariableOwnerGeometry,replay:u64,cookies:&[u64],mode:crate::descriptor::ResultMode)->crate::descriptor::Result<crate::descriptor::variable_wire::Expectation>{self.variable_descriptor_expectation_rows::<8>(owner,replay,cookies,mode)}
    pub(crate) fn variable_descriptor_expectation_rows<const ROWS:usize>(
        &self, owner: &VariableOwnerGeometry, replay: u64, cookies: &[u64],
        mode: crate::descriptor::ResultMode,
    ) -> crate::descriptor::Result<crate::descriptor::variable_wire::Expectation<ROWS>> {
        use crate::descriptor::{Error, BlockOwnership, shape_progress::{Progress, InputStage}, variable_wire::{self, Row, Expectation}};
        let bad = || Error { field: "scheduler V3 authority", reason: "incompatible live reservation or owner geometry" };
        let prefill = self.plan.prefill_items();
        let decode = self.plan.decode_items();
        if owner.physical_block_count as usize != self.physical_block_count()
            || cookies.len()!=self.rows.len() || prefill.len()+decode.len()!=self.rows.len()
            || (!owner.mixed_execution && !prefill.is_empty() && !decode.is_empty()) { return Err(bad()); }
        let stage = if prefill.is_empty() { InputStage::Decode } else { InputStage::Prefill };
        let to32 = |v:usize| u32::try_from(v).map_err(|_| bad());
        let mut rows = Vec::with_capacity(self.rows.len());
        let mut partial_slot=self.plan.output_slots().len() as u32;
        for ((work,row),cookie) in prefill.iter().chain(decode).zip(&self.rows).zip(cookies) {
            if work.request_id()!=row.request_id { return Err(bad()); }
            let progress = Progress {
                prompt_tokens:to32(row.prompt_tokens)?, output_limit:to32(row.max_output_tokens)?,
                context_tokens:owner.context_tokens, committed_tokens:to32(row.committed_length)?,
                input_tokens:to32(work.input_tokens().len())?, generated_index:to32(row.generated_index)?, stage:if work.kind()==crate::WorkKind::Prefill{InputStage::Prefill}else{InputStage::Decode},
            };
            let checked = progress.validate()?;
            if checked.target_tokens != to32(work.target_logical_length())?
                || checked.logits_input_row.is_some()!=work.output_slot().is_some() { return Err(bad()); }
            // V3 retains a dense row slot even when partial prefill publishes no output.
            // Such a slot must never be converted to an IterationOutput.
            let output_slot = work.output_slot().map(|s|s.get()).unwrap_or_else(||{let slot=partial_slot;partial_slot+=1;slot});
            rows.push(Row { sequence_tag:row.request_id.get(), cookie:*cookie, output_slot, progress,
                input_tokens:work.input_tokens().to_vec(), physical_ids:row.table.physical_block_ids().to_vec(),
                valid_tokens:row.table.valid_tokens().to_vec() });
        }
        let e = Expectation { mixed_execution:owner.mixed_execution, packed_prefill:owner.packed_prefill, owner_generation:owner.generation, last_accepted_replay:owner.last_accepted_replay,
            replay_id:replay, iteration_id:self.plan.iteration_id().get(), catalog_digest:owner.catalog_digest,
            physical_block_count:owner.physical_block_count, max_active_rows:owner.max_active_rows,
            stage, mode, rows, block_ownership:self.block_owners.iter().map(|(id,tag)| BlockOwnership {
                physical_id:*id, sequence_tag:tag.get() }).collect() };
        variable_wire::validate(&e)?;
        Ok(e)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn variable_policy_batches_live_decode_and_retries_without_false_progress() {
        use crate::{IterationResult,IterationOutput,ExecutionAbort,ExecutionShapePolicy};
        use crate::descriptor::{shape_progress::InputStage,variable_wire};
        for cancel_last in [false,true] {
        let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig {
            max_waiting_requests:8,max_waiting_prompt_tokens:4096,max_active_sequences:4,
            max_sequence_tokens:1024,iteration_token_budget:73,max_prefill_chunk_tokens:73,
            aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,
            max_promised_kv_blocks:256,metrics_window_samples:16,
        },riley_runtime::paged_kv::KvLayout::checked(30,256,3,64).unwrap(),ExecutionShapePolicy::VariablePrefillDecodeN).unwrap();
        let ids:Vec<_>=[16,128,398,73].into_iter().map(|n|scheduler.submit(RequestDescriptor::new(vec![17;n],32),0).unwrap().request_id()).collect();
        let mut owner=super::VariableOwnerGeometry{mixed_execution:false,packed_prefill:false,generation:1,last_accepted_replay:0,catalog_digest:[9;32],max_active_rows:4,physical_block_count:256,context_tokens:1024};
        let (mut replay,mut now,mut widest,mut retried,mut partial)=(0u64,0u64,0usize,false,false);
        let mut last=None;
        loop {
            now+=2;
            let Some(plan)=scheduler.plan_iteration(now).unwrap().into_parts().0 else {break};
            let prefill=!plan.prefill_items().is_empty();
            assert!(plan.prefill_items().is_empty() || plan.decode_items().is_empty());
            if prefill {assert_eq!(plan.prefill_items().len(),1);assert!(plan.total_tokens()<=73);}
            widest=widest.max(plan.decode_items().len());
            let signature:Vec<_>=plan.prefill_items().iter().chain(plan.decode_items()).map(|w|(w.request_id(),w.input_tokens().to_vec(),w.target_logical_length())).collect();
            if !retried {
                scheduler.abort_iteration(plan.iteration_id(),ExecutionAbort::NotDispatched,now+1).unwrap();
                last=Some(signature);retried=true;continue;
            }
            if let Some(expected)=last.take(){assert_eq!(signature,expected);}
            replay+=1;
            let authority=scheduler.authorize_execution(&plan).unwrap();
            let cookies:Vec<_>=(0..plan.batch_size()).map(|i|replay*8+i as u64+1).collect();
            let expectation=authority.variable_descriptor_expectation(&owner,replay,&cookies,ResultMode::FullLogits).unwrap();
            assert_eq!(expectation.stage,if prefill{InputStage::Prefill}else{InputStage::Decode});
            let mut bytes=vec![0;variable_wire::REQUEST_BYTES];variable_wire::encode_into(&mut bytes,&expectation).unwrap();
            variable_wire::validate_packet(&bytes,&mut vec![0;bytes.len()],&expectation).unwrap();
            partial |= prefill && plan.output_slots().is_empty();
            let outputs=plan.output_slots().iter().map(|&slot|IterationOutput::new(slot,23,false)).collect();
            drop(authority);
            let result=IterationResult::new(plan.iteration_id(),outputs,0,0).unwrap();
            assert!(scheduler.complete_iteration(&result,now+1).unwrap().settlement_failures().is_empty());
            owner.last_accepted_replay=replay;
            if cancel_last && replay==1 {scheduler.cancel(ids[3],now+1).unwrap();}
            assert!(replay<256);
        }
        assert!(partial && retried && widest>=3);
        for (i,id) in ids.into_iter().enumerate() {assert_eq!(scheduler.request_state(id),Some(if cancel_last && i==3 {crate::RequestState::Cancelled}else{crate::RequestState::Finished}));}
        scheduler.close(now+2,None).unwrap();
        }
    }

    #[test]
    fn packed_policy_batches_prefill_and_preserves_retry_partial_cancel() {
        use crate::{IterationResult,IterationOutput,ExecutionAbort,ExecutionShapePolicy};
        use crate::descriptor::{shape_progress::InputStage,variable_wire};
        for cancel_last in [false,true] {
        let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig {
            max_waiting_requests:8,max_waiting_prompt_tokens:4096,max_active_sequences:4,
            max_sequence_tokens:1024,iteration_token_budget:256,max_prefill_chunk_tokens:128,
            aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,
            max_promised_kv_blocks:256,metrics_window_samples:16,
        },riley_runtime::paged_kv::KvLayout::checked(30,256,3,64).unwrap(),ExecutionShapePolicy::PackedPrefillDecode32).unwrap();
        let ids:Vec<_>=[16,128,398,73].into_iter().map(|n|scheduler.submit(RequestDescriptor::new(vec![17;n],32),0).unwrap().request_id()).collect();
        let mut owner=super::VariableOwnerGeometry{mixed_execution:false,packed_prefill:true,generation:1,last_accepted_replay:0,catalog_digest:[9;32],max_active_rows:4,physical_block_count:256,context_tokens:1024};
        let (mut replay,mut now,mut widest,mut retried,mut partial)=(0u64,0u64,0usize,false,false);
        let mut last=None;let mut prefill_width=0;
        loop {
            now+=2;
            let Some(plan)=scheduler.plan_iteration(now).unwrap().into_parts().0 else {break};
            let prefill=!plan.prefill_items().is_empty();
            assert!(plan.prefill_items().is_empty() || plan.decode_items().is_empty());
            if prefill {prefill_width=prefill_width.max(plan.prefill_items().len());assert!(plan.prefill_items().len()<=4);assert!(plan.total_tokens()<=256);}
            widest=widest.max(plan.decode_items().len());
            let signature:Vec<_>=plan.prefill_items().iter().chain(plan.decode_items()).map(|w|(w.request_id(),w.input_tokens().to_vec(),w.target_logical_length())).collect();
            if !retried {
                scheduler.abort_iteration(plan.iteration_id(),ExecutionAbort::NotDispatched,now+1).unwrap();
                last=Some(signature);retried=true;continue;
            }
            if let Some(expected)=last.take(){assert_eq!(signature,expected);}
            replay+=1;
            let authority=scheduler.authorize_execution(&plan).unwrap();
            let cookies:Vec<_>=(0..plan.batch_size()).map(|i|replay*8+i as u64+1).collect();
            let expectation=authority.variable_descriptor_expectation_rows::<32>(&owner,replay,&cookies,ResultMode::FullLogits).unwrap();
            assert_eq!(expectation.stage,if prefill{InputStage::Prefill}else{InputStage::Decode});
            let mut bytes=vec![0;variable_wire::Layout::<32>::REQUEST_BYTES];variable_wire::encode_into(&mut bytes,&expectation).unwrap();
            variable_wire::validate_packet(&bytes,&mut vec![0;bytes.len()],&expectation).unwrap();
            partial |= prefill && plan.output_slots().len()<plan.prefill_items().len();
            let outputs=plan.output_slots().iter().map(|&slot|IterationOutput::new(slot,23,false)).collect();
            drop(authority);
            let result=IterationResult::new(plan.iteration_id(),outputs,0,0).unwrap();
            assert!(scheduler.complete_iteration(&result,now+1).unwrap().settlement_failures().is_empty());
            owner.last_accepted_replay=replay;
            if cancel_last && replay==1 {scheduler.cancel(ids[3],now+1).unwrap();}
            assert!(replay<256);
        }
        assert!(partial && retried && widest>=3 && prefill_width>=3);
        for (i,id) in ids.into_iter().enumerate() {assert_eq!(scheduler.request_state(id),Some(if cancel_last && i==3 {crate::RequestState::Cancelled}else{crate::RequestState::Finished}));}
        scheduler.close(now+2,None).unwrap();
        }
    }

    #[test]
    fn mixed_policy_budget_stage_partial_retry_and_cancel() {
        use crate::{IterationResult,IterationOutput,ExecutionAbort,ExecutionShapePolicy};
        use crate::descriptor::{shape_progress::InputStage,variable_wire};
        for cancel_last in [false,true] {
        let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig {
            max_waiting_requests:8,max_waiting_prompt_tokens:4096,max_active_sequences:4,
            max_sequence_tokens:1024,iteration_token_budget:256,max_prefill_chunk_tokens:128,
            aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,
            max_promised_kv_blocks:256,metrics_window_samples:16,
        },riley_runtime::paged_kv::KvLayout::checked(30,256,3,64).unwrap(),ExecutionShapePolicy::MixedPrefillDecode32).unwrap();
        let ids:Vec<_>=[16,128,398,73].into_iter().map(|n|scheduler.submit(RequestDescriptor::new(vec![17;n],32),0).unwrap().request_id()).collect();
        let mut owner=super::VariableOwnerGeometry{mixed_execution:true,packed_prefill:true,generation:1,last_accepted_replay:0,catalog_digest:[9;32],max_active_rows:4,physical_block_count:256,context_tokens:1024};
        let (mut replay,mut now,mut widest,mut retried,mut partial)=(0u64,0u64,0usize,false,false);
        let mut last=None;let mut prefill_width=0;let mut saw_mixed=false;
        loop {
            now+=2;
            let Some(plan)=scheduler.plan_iteration(now).unwrap().into_parts().0 else {break};
            let prefill=!plan.prefill_items().is_empty();
            saw_mixed|=!plan.prefill_items().is_empty()&&!plan.decode_items().is_empty();
            if prefill {prefill_width=prefill_width.max(plan.prefill_items().len());assert!(plan.prefill_items().len()<=4);assert!(plan.total_tokens()<=256);}
            widest=widest.max(plan.decode_items().len());
            let signature:Vec<_>=plan.prefill_items().iter().chain(plan.decode_items()).map(|w|(w.request_id(),w.input_tokens().to_vec(),w.target_logical_length())).collect();
            if !retried {
                scheduler.abort_iteration(plan.iteration_id(),ExecutionAbort::NotDispatched,now+1).unwrap();
                last=Some(signature);retried=true;continue;
            }
            if let Some(expected)=last.take(){assert_eq!(signature,expected);}
            replay+=1;
            let authority=scheduler.authorize_execution(&plan).unwrap();
            let cookies:Vec<_>=(0..plan.batch_size()).map(|i|replay*8+i as u64+1).collect();
            let expectation=authority.variable_descriptor_expectation_rows::<32>(&owner,replay,&cookies,ResultMode::FullLogits).unwrap();
            assert_eq!(expectation.stage,if prefill{InputStage::Prefill}else{InputStage::Decode});
            let mut bytes=vec![0;variable_wire::MIXED_REQUEST_BYTES];variable_wire::encode_into(&mut bytes,&expectation).unwrap();
            variable_wire::validate_packet(&bytes,&mut vec![0;bytes.len()],&expectation).unwrap();
            partial |= prefill && plan.output_slots().len()<plan.prefill_items().len();
            let outputs=plan.output_slots().iter().map(|&slot|IterationOutput::new(slot,23,false)).collect();
            drop(authority);
            let result=IterationResult::new(plan.iteration_id(),outputs,0,0).unwrap();
            assert!(scheduler.complete_iteration(&result,now+1).unwrap().settlement_failures().is_empty());
            owner.last_accepted_replay=replay;
            if cancel_last && replay==1 {scheduler.cancel(ids[3],now+1).unwrap();}
            assert!(replay<256);
        }
        assert!(partial && retried && widest>=3 && prefill_width>=3 && saw_mixed);
        for (i,id) in ids.into_iter().enumerate() {assert_eq!(scheduler.request_state(id),Some(if cancel_last && i==3 {crate::RequestState::Cancelled}else{crate::RequestState::Finished}));}
        scheduler.close(now+2,None).unwrap();
        }
    }

    #[test]
    fn variable_authority_tracks_chunked_prefill_and_decode_from_live_scheduler() {
        use crate::{IterationResult, IterationOutput};
        use crate::descriptor::{shape_progress::InputStage, variable_wire};
        for (prompt,limit) in [(16,32),(128,64),(398,128)] {
            let mut scheduler = Scheduler::new(SchedulerConfig {
                max_waiting_requests:8, max_waiting_prompt_tokens:4096, max_active_sequences:1,
                max_sequence_tokens:1024, iteration_token_budget:73, max_prefill_chunk_tokens:73,
                aging_threshold_ns:1, overload_policy:OverloadPolicy::Wait, admission_timeout_ns:None,
                max_promised_kv_blocks:64, metrics_window_samples:16,
            }, riley_runtime::paged_kv::KvLayout::checked(30,64,3,64).unwrap()).unwrap();
            let id=scheduler.submit(RequestDescriptor::new(vec![17;prompt],limit),0).unwrap().request_id();
            let mut owner=super::VariableOwnerGeometry { mixed_execution:false, packed_prefill:false, generation:1,last_accepted_replay:0,catalog_digest:[7;32],
                max_active_rows:1,physical_block_count:64,context_tokens:1024 };
            let (mut committed,mut generated,mut replay)=(0,0,0u64);
            while generated<limit {
                replay+=1;
                let plan=scheduler.plan_iteration(replay*2).unwrap().into_parts().0.unwrap();
                let a=scheduler.authorize_execution(&plan).unwrap();
                let e=a.variable_descriptor_expectation(&owner,replay,&[replay],ResultMode::FullLogits).unwrap();
                assert_eq!(e.rows[0].sequence_tag,id.get());
                let p=e.rows[0].progress;
                assert_eq!((p.prompt_tokens,p.committed_tokens,p.generated_index),(prompt as u32,committed,generated as u32));
                assert_eq!(p.stage,if committed<prompt as u32 {InputStage::Prefill}else{InputStage::Decode});
                let v=p.validate().unwrap();
                let mut packet=vec![0;variable_wire::REQUEST_BYTES];
                variable_wire::encode_into(&mut packet,&e).unwrap();
                variable_wire::validate_packet(&packet,&mut vec![0;variable_wire::REQUEST_BYTES],&e).unwrap();
                assert!(a.variable_descriptor_expectation(&owner,replay+1,&[replay],ResultMode::FullLogits).is_err());
                assert!(a.variable_descriptor_expectation(&owner,replay,&[],ResultMode::FullLogits).is_err());
                let mut foreign=e.clone(); foreign.block_ownership.clear();
                assert!(variable_wire::encode_into(&mut packet,&foreign).is_err());
                let outputs=plan.prefill_items().iter().chain(plan.decode_items()).filter_map(|w|w.output_slot())
                    .map(|slot|IterationOutput::new(slot,23,false)).collect();
                drop(a);
                let result=IterationResult::new(plan.iteration_id(),outputs,0,0).unwrap();
                let updates=scheduler.complete_iteration(&result,replay*2+1).unwrap();
                assert!(updates.settlement_failures().is_empty());
                committed=v.target_tokens;
                if v.logits_input_row.is_some() {generated+=1;}
                owner.last_accepted_replay=replay;
            }
            assert_eq!(committed,(prompt+limit-1) as u32);
            scheduler.close(replay*2+2,None).unwrap();
        }
    }

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

/// Retains an immutable scheduler borrow across both dependent executions.
/// Neither embedded single-step authority is exposed to ordinary executors.
/// ```compile_fail
/// use riley_scheduler::{Scheduler,RequestId};
/// use riley_scheduler::plan::DecodeWindowPlan;
/// fn cannot_cancel_while_authorized(s:&mut Scheduler,w:&DecodeWindowPlan) {
///     let authority=s.authorize_decode_window(w).unwrap();
///     s.cancel(RequestId::new(1).unwrap(),0).unwrap();
///     drop(authority);
/// }
/// ```
pub struct AuthorizedDecodeWindow<'a> {
    pub(crate) first: AuthorizedExecution<'a>,
    pub(crate) second: AuthorizedExecution<'a>,
}
impl AuthorizedDecodeWindow<'_> {
    #[allow(dead_code)]
    pub(crate) fn prepare_wire(&self,owner:&VariableOwnerGeometry,replay:u64,first_cookies:&[u64],second_cookies:&[u64])->crate::descriptor::Result<crate::descriptor::future_token::PreparedFutureWindow> {
        let first=self.first.variable_descriptor_expectation_rows::<32>(owner,replay,first_cookies,crate::descriptor::ResultMode::Greedy)?;
        let mut structural_owner=owner.clone();structural_owner.last_accepted_replay=replay;
        let next=replay.checked_add(1).ok_or(crate::descriptor::Error{field:"window replay",reason:"overflow"})?;
        let mut second=self.second.variable_descriptor_expectation_rows::<32>(&structural_owner,next,second_cookies,crate::descriptor::ResultMode::Greedy)?;
        second.last_accepted_replay=owner.last_accepted_replay;
        let sources=(0..second.rows.len()).map(|i|crate::descriptor::future_token::TokenSource::PreviousRow(i as u32)).collect::<Vec<_>>();
        crate::descriptor::future_token::PreparedFutureWindow::new(first,second,&sources)
    }
}
