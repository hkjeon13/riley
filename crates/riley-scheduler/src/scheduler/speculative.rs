//! Experimental multi-token scheduling. GPU adapter integration is separate.
use super::*;
use riley_runtime::speculative::{prompt_lookup, settle_completed_greedy, verify_greedy, Stop};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeculativeRow {
    request_id: RequestId,
    start: u32,
    inputs: Vec<u32>,
    table: OwnedBlockTable,
}
impl SpeculativeRow {
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }
    pub fn committed_tokens(&self) -> u32 {
        self.start
    }
    pub fn inputs(&self) -> &[u32] {
        &self.inputs
    }
    pub fn table(&self) -> &OwnedBlockTable {
        &self.table
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeculativePlan {
    wide:bool,
    iteration_id: IterationId,
    rows: Vec<SpeculativeRow>,
    eos: Option<u32>,
}
impl SpeculativePlan {
    pub fn iteration_id(&self) -> IterationId {
        self.iteration_id
    }
    pub fn rows(&self) -> &[SpeculativeRow] {
        &self.rows
    }
}
/// Retains an immutable scheduler borrow throughout device execution.
pub struct AuthorizedSpeculative<'a> {
    scheduler: &'a Scheduler,
    plan: &'a SpeculativePlan,
}
impl AuthorizedSpeculative<'_> {
    pub fn plan(&self) -> &SpeculativePlan {
        self.plan
    }
    pub(crate) fn verification_expectation(&self,owner:&crate::authority::VariableOwnerGeometry,replay:u64,cookies:&[u64])->crate::descriptor::Result<crate::descriptor::variable_wire::Expectation<32>> {
        use crate::descriptor::{variable_wire::{Expectation,Row},shape_progress::{InputStage,Progress},BlockOwnership,ResultMode,Error};
        let bad=||Error{field:"speculative authority",reason:"owner or progress differs from live reservation"};
        if !owner.mixed_execution || !owner.packed_prefill || (owner.shared_prefixes && !self.plan.wide) || owner.max_active_rows!=32 || owner.physical_block_count as usize!=self.physical_block_count() || cookies.len()!=self.plan.rows.len(){return Err(bad());}
        let mut rows=Vec::new();
        for (slot,(row,&cookie)) in self.plan.rows.iter().zip(cookies).enumerate(){
            let record=&self.scheduler.requests[self.scheduler.record_index(row.request_id).ok_or_else(bad)?];
            rows.push(Row{sequence_tag:row.request_id.get(),cookie,output_slot:slot as u32,
                progress:Progress{prompt_tokens:record.descriptor.prompt_token_ids.len() as u32,output_limit:record.descriptor.max_new_tokens as u32,context_tokens:owner.context_tokens,committed_tokens:row.start,input_tokens:row.inputs.len() as u32,generated_index:record.generated_token_ids.len() as u32,stage:InputStage::Verification},
                input_tokens:row.inputs.clone(),physical_ids:row.table.physical_block_ids().to_vec(),valid_tokens:row.table.valid_tokens().to_vec()});
        }
        let e=Expectation{owner_generation:owner.generation,last_accepted_replay:owner.last_accepted_replay,replay_id:replay,iteration_id:self.plan.iteration_id.get(),catalog_digest:owner.catalog_digest,physical_block_count:owner.physical_block_count,max_active_rows:32,stage:InputStage::Verification,mode:if self.plan.wide{ResultMode::Greedy}else{ResultMode::FullLogits},rows,block_ownership:self.scheduler.execution_block_owners().map_err(|_|bad())?.into_iter().map(|(physical_id,id)|BlockOwnership{physical_id,sequence_tag:id.get()}).collect(),packed_prefill:true,mixed_execution:true,shared_prefixes:owner.shared_prefixes};
        crate::descriptor::variable_wire::validate(&e)?;Ok(e)
    }
    pub fn physical_block_count(&self) -> usize {
        self.scheduler.pool_physical_block_count()
    }
}
fn invalid() -> SchedulerError {
    SchedulerError::InvalidPlan {
        field: "speculative round",
        reason: "result or plan differs from live bounded append",
    }
}
impl Scheduler {
    pub fn authorize_speculative_execution<'a>(
        &'a self,
        plan: &'a SpeculativePlan,
    ) -> SchedulerResult<AuthorizedSpeculative<'a>> {
        self.validate_inflight_reservations()?;
        if self.inflight.as_ref().and_then(|p| p.speculative.as_ref()) != Some(plan) {
            return Err(invalid());
        }
        Ok(AuthorizedSpeculative {
            scheduler: self,
            plan,
        })
    }

    /// Try bounded prompt lookup after ordinary admission/prefill. None selects
    /// ordinary scheduling. The narrow path excludes prefix cache; the wide path
    /// admits immutable full-page prefixes with private writable tails. Never reinterpret these rows as ordinary prompt prefill.
    pub fn plan_speculative_iteration(&mut self,now_ns:u64,eos:Option<u32>)->SchedulerResult<Option<SpeculativePlan>>{self.plan_speculative_internal(now_ns,eos,false)}
    pub fn plan_wide_speculative_iteration(&mut self,now_ns:u64,eos:Option<u32>)->SchedulerResult<Option<SpeculativePlan>>{self.plan_speculative_internal(now_ns,eos,true)}
    fn plan_speculative_internal(
        &mut self,
        now_ns: u64,
        eos: Option<u32>, wide:bool,
    ) -> SchedulerResult<Option<SpeculativePlan>> {
        let started = Instant::now();
        self.ensure_completion_backlog_empty()?;
        self.observe_now(now_ns)?;
        if let Some(p) = &self.inflight {
            return Err(SchedulerError::IterationInFlight {
                iteration_id: p.iteration_id,
            });
        }
        if eos.is_some_and(|x| x >= 49152) {
            return Err(invalid());
        }
        if (self.prefix_cache.is_some() && !wide)
            || !self.waiting.is_empty()
            || self.execution_shape_policy != ExecutionShapePolicy::MixedPrefillDecode32
        {
            return Ok(None);
        }
        let (candidates, aging) = self.select_candidates(now_ns)?;
        if candidates.is_empty() || candidates.iter().any(|c| c.kind != WorkKind::Decode) {
            return Ok(None);
        }
        let mut drafts = Vec::new();
        let mut budget = self.config.iteration_token_budget;
        for c in &candidates {
            if drafts.len()==if wide{32}else{4} || budget<1 {break;}
            let record = &self.requests[self.record_index(c.request_id).ok_or_else(invalid)?];
            let remaining = record.descriptor.max_new_tokens - record.generated_token_ids.len();
            if remaining < if wide{1}else{2} {
                continue;
            }
            let mut history = Vec::new();
            let total = record.descriptor.prompt_token_ids.len() + record.generated_token_ids.len();
            try_reserve_exact(&mut history, total.min(4096), "speculative lookup history")?;
            history.extend(
                record
                    .descriptor
                    .prompt_token_ids
                    .iter()
                    .chain(&record.generated_token_ids)
                    .skip(total.saturating_sub(4096))
                    .copied(),
            );
            let found = prompt_lookup(&history, 8, 4096);
            let count = found
                .tokens()
                .len()
                .min(7)
                .min(remaining - 1)
                .min(budget.saturating_sub(if wide{candidates.len()-drafts.len()}else{1}));
            if count == 0 && !wide {
                continue;
            }
            let mut input = Vec::new();
            try_reserve_exact(&mut input, count + 1, "speculative inputs")?;
            input.push(*record.generated_token_ids.last().ok_or_else(invalid)?);
            input.extend_from_slice(&found.tokens()[..count]);
            budget -= input.len();
            drafts.push((c.request_id, input));
        }
        if drafts.is_empty() || (wide && drafts.iter().all(|(_,input)|input.len()==1)){return Ok(None);}
        let id = self.peek_iteration_id()?;
        let next = self
            .next_iteration_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "iteration" })?;
        let mut rows = Vec::new();
        let mut items = Vec::new();
        try_reserve_exact(&mut rows, drafts.len(), "speculative rows")?;
        try_reserve_exact(&mut items, drafts.len(), "speculative reservations")?;
        for (request_id, inputs) in drafts {
            let index = self.record_index(request_id).ok_or_else(invalid)?;
            let record = &mut self.requests[index];
            let sequence = record.sequence.as_mut().ok_or_else(invalid)?;
            let start = sequence.logical_length();
            let target = start as usize + inputs.len();
            let reservation = match sequence.reserve_to(&mut self.pool, target) {
                Ok(r) => r,
                Err(e) => return self.rollback_prepared_plan(items, e.into()),
            };
            items.push(InflightItem {
                request_id,
                kind: WorkKind::Decode,
                target_logical_length: target,
                output_slot: None,
                previous_state: record.state,
                previous_ready_since_ns: record.ready_since_ns,
                reservation,
            });
            let table = match sequence
                .execution_block_table(&self.pool, Some(&items.last().unwrap().reservation))
                .map_err(SchedulerError::from)
                .and_then(|t| OwnedBlockTable::copy_from_v1(request_id, t))
            {
                Ok(t) => t,
                Err(e) => return self.rollback_prepared_plan(items, e),
            };
            rows.push(SpeculativeRow {
                request_id,
                start,
                inputs,
                table,
            });
        }
        let plan = SpeculativePlan {
            wide,
            iteration_id: id,
            rows,
            eos,
        };
        self.inflight = Some(InflightPlan {
            speculative: Some(plan.clone()),
            mixed_cost_bucket: 0,
            iteration_id: id,
            successor: None,
            prefix_settlement: None,
            device_progress: false,
            prefill_tokens: 0,
            decode_tokens: plan.rows.iter().map(|r| r.inputs.len()).sum(),
            prefill_count: 0,
            decode_count: items.len(),
            scheduler_cpu_ns: elapsed_ns(started),
            expected_output_slots: Vec::new(),
            items,
        });
        self.next_iteration_id = next;
        self.aging_override_last_iteration = aging;
        self.refresh_metric_gauges();
        Ok(Some(plan))
    }
    /// Validate the complete target result before detokenization stages output.
    pub fn preview_speculative_iteration(&self,plan:&SpeculativePlan,targets:&[Vec<u32>])->SchedulerResult<Vec<riley_runtime::speculative::GreedyDecision>> {
        self.validate_inflight_reservations()?;
        if self.inflight.as_ref().and_then(|p|p.speculative.as_ref())!=Some(plan) || targets.len()!=plan.rows.len(){return Err(invalid());}
        let mut decisions = Vec::new();
        try_reserve_exact(&mut decisions, targets.len(), "speculative decisions")?;
        for (row, target) in plan.rows.iter().zip(targets) {
            let record = &self.requests[self.record_index(row.request_id).ok_or_else(invalid)?];
            decisions.push(
                verify_greedy(
                    &row.inputs[1..],
                    target,
                    49152,
                    record.descriptor.max_new_tokens - record.generated_token_ids.len(),
                    plan.eos,
                    record.cancellation_deferred,
                )
                .map_err(|_| invalid())?,
            );
        }
        Ok(decisions)
    }
    /// The caller must establish GPU completion and append-only writes before
    /// calling. Unknown writes use quiesced abort. Target rows are untrusted and
    /// must include every input endpoint, even after mismatch or cancellation.
    /// No publication occurs before every row and reservation is validated.
    pub fn complete_speculative_iteration(
        &mut self,
        plan: &SpeculativePlan,
        targets: &[Vec<u32>],
        now_ns: u64,
        timing: crate::IterationTiming,
    ) -> SchedulerResult<IterationUpdates> {
        self.complete_speculative_iteration_with_stops(plan,targets,&vec![None;targets.len()],now_ns,timing)
    }
    /// Stop positions count emitted tokens, including the stopping token. They
    /// cannot extend the verified prefix or bypass complete target validation.
    pub fn complete_speculative_iteration_with_stops(&mut self,plan:&SpeculativePlan,targets:&[Vec<u32>],stops:&[Option<usize>],now_ns:u64,timing:crate::IterationTiming)->SchedulerResult<IterationUpdates> {
        let started = Instant::now();
        self.ensure_completion_backlog_empty()?;
        self.validate_now(now_ns)?;
        self.validate_inflight_reservations()?;
        let pending = self
            .inflight
            .as_ref()
            .ok_or(SchedulerError::NoIterationInFlight)?;
        if pending.speculative.as_ref() != Some(plan) || targets.len() != plan.rows.len() {
            return Err(invalid());
        }
        let mut decisions=self.preview_speculative_iteration(plan,targets)?;
        if stops.len()!=decisions.len(){return Err(invalid());}
        for (decision,stop) in decisions.iter_mut().zip(stops) {
            if let Some(count)=stop {decision.stop_after(*count).map_err(|_|invalid())?;}
        }
        let mut updates = IterationUpdates::empty();
        try_reserve_exact(&mut updates.token_events, 256, "speculative token events")?;
        try_reserve_exact(
            &mut updates.completions,
            plan.rows.len(),
            "speculative completions",
        )?;
        try_reserve_exact(
            &mut updates.settlement_failures,
            plan.rows.len(),
            "speculative failures",
        )?;
        self.ensure_completion_capacity(plan.rows.len())?;
        self.last_now_ns = Some(now_ns);
        let inflight = self.inflight.take().unwrap();
        self.record_dispatched_shape(&inflight);
        let metric = IterationMetricSample {
            batch_size: inflight.items.len(),
            prefill_tokens: 0,
            decode_tokens: inflight.decode_tokens,
            scheduler_cpu_ns: inflight.scheduler_cpu_ns,
            gpu_execution_ns: timing.gpu_execution_ns(),
            gpu_idle_gap_ns: timing.gpu_idle_gap_ns(),
        };
        #[allow(unused_variables)]
        let mut settled_count = 0;
        for ((item, row), decision) in inflight.items.into_iter().zip(&plan.rows).zip(&decisions) {
            let index = self
                .record_index(item.request_id)
                .expect("prevalidated owner");
            #[cfg(test)]
            let forced_failure = self.take_test_post_validation_commit_fault(settled_count);
            #[cfg(not(test))]
            let forced_failure = false;
            let result = if forced_failure {
                Err(riley_runtime::paged_kv::PagedKvError::ReservationMismatch)
            } else {
                settle_completed_greedy(
                    self.requests[index].sequence.as_mut().unwrap(),
                    &mut self.pool,
                    item.reservation,
                    row.start,
                    decision,
                )
            };
            if result.is_ok() {
                settled_count += 1;
            }
            if let Err(error) = result {
                let error =
                    self.contain_live_request_failure(item.request_id, now_ns, error.into());
                updates.settlement_failures.push(RequestSettlementFailure {
                    request_id: item.request_id,
                    error,
                });
            }
        }
        if !updates.settlement_failures.is_empty() {
            for row in &plan.rows {
                if self.record_index(row.request_id).is_some() {
                    if let Err(e) = self.finish_live_request(
                        row.request_id,
                        RequestFinishReason::ExecutorFailure,
                        now_ns,
                    ) {
                        let error = self.contain_live_request_failure(row.request_id, now_ns, e);
                        updates.settlement_failures.push(RequestSettlementFailure {
                            request_id: row.request_id,
                            error,
                        });
                    }
                }
            }
        } else {
            for (row, decision) in plan.rows.iter().zip(&decisions) {
                let index = self.record_index(row.request_id).unwrap();
                let record = &mut self.requests[index];
                for &token in decision.tokens() {
                    let generated_index = record.generated_token_ids.len();
                    record.generated_token_ids.push(token);
                    updates.token_events.push(TokenEvent {
                        request_id: row.request_id,
                        token_id: token,
                        generated_index,
                    });
                }
                record.state = RequestState::Decoding;
                record.ready_since_ns = now_ns;
                let reason = match decision.stop() {
                    Some(Stop::Eos | Stop::External) => Some(RequestFinishReason::Stop),
                    Some(Stop::Length) => Some(RequestFinishReason::Length),
                    Some(Stop::Cancelled) => Some(RequestFinishReason::Cancelled),
                    None => None,
                };
                if let Some(reason) = reason {
                    if let Err(e) = self.finish_live_request(row.request_id, reason, now_ns) {
                        let error = self.contain_live_request_failure(row.request_id, now_ns, e);
                        updates.settlement_failures.push(RequestSettlementFailure {
                            request_id: row.request_id,
                            error,
                        });
                    }
                }
            }
        }
        self.drain_completion_outbox_into(&mut updates.completions);
        if updates.settlement_failures.is_empty() {
            let metric = IterationMetricSample {
                scheduler_cpu_ns: metric.scheduler_cpu_ns.saturating_add(elapsed_ns(started)),
                ..metric
            };
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_iteration(metric),
                "speculative completed iteration",
            );
            updates.iteration_metric = Some(metric);
        } else {
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_aborted_iteration(),
                "speculative failed settlement",
            );
        }
        self.refresh_metric_gauges();
        Ok(updates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riley_runtime::paged_kv::KvLayout;
    fn ready(prompt:usize,limit:usize)->(Scheduler,Vec<RequestId>){ready_owners(prompt,limit,4)}
    fn ready_owners(prompt: usize, limit: usize, owners:usize) -> (Scheduler, Vec<RequestId>) {
        let config = SchedulerConfig {
            max_waiting_requests: 8,
            max_waiting_prompt_tokens: 1024,
            max_active_sequences: owners,
            max_sequence_tokens: 128,
            iteration_token_budget: 512,
            max_prefill_chunk_tokens: 128,
            aging_threshold_ns: 1000,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: None,
            max_promised_kv_blocks: if owners==4{64}else{256},
            metrics_window_samples: 8,
        };
        let mut s = Scheduler::new_with_execution_shape(
            config,
            KvLayout::checked(30, if owners==4{64}else{256}, 3, 64).unwrap(),
            ExecutionShapePolicy::MixedPrefillDecode32,
        )
        .unwrap();
        let ids = (0..owners)
            .map(|_| {
                s.submit(
                    RequestDescriptor::new((0..prompt).map(|i| i as u32 % 3 + 1).collect(), limit),
                    0,
                )
                .unwrap()
                .request_id()
            })
            .collect();
        while s.requests.iter().any(|r|r.generated_token_ids.is_empty()){
        let p = s.plan_iteration(1).unwrap().into_parts().0.unwrap();
        let result = IterationResult::new(
            p.iteration_id(),
            p.output_slots()
                .iter()
                .map(|&slot|{let id=p.prefill_items().iter().chain(p.decode_items()).find(|w|w.output_slot()==Some(slot)).unwrap().request_id();let generated=s.requests[s.record_index(id).unwrap()].generated_token_ids.len();IterationOutput::new(slot,(prompt+generated) as u32%3+1,false)})
                .collect(),
            0,
            0,
        )
        .unwrap();
        assert!(s
            .complete_iteration(&result, 1)
            .unwrap()
            .settlement_failures()
            .is_empty());
        }
        (s, ids)
    }
    #[test]
    fn external_stop_truncates_verified_output_and_releases_kv_at_every_position() {
        for prompt in [16,17,31,32,33] {for stop in 1..=8 {
            let (mut s,ids)=ready(prompt,32);let p=s.plan_wide_speculative_iteration(3,None).unwrap().unwrap();
            let targets=p.rows().iter().map(|r|{let mut t=r.inputs()[1..].to_vec();t.push(77);t}).collect::<Vec<_>>();
            let before=s.pool_stats().allocated_block_count();
            let preview=s.preview_speculative_iteration(&p,&targets).unwrap();assert!(preview.iter().all(|d|d.tokens().len()==8));
            for invalid_count in [0,9] {
                assert!(s.complete_speculative_iteration_with_stops(&p,&targets,&[Some(invalid_count);4],4,crate::IterationTiming::default()).is_err());
                assert_eq!(s.pool_stats().allocated_block_count(),before);
                assert_eq!(s.preview_speculative_iteration(&p,&targets).unwrap(),preview);
            }
            let u=s.complete_speculative_iteration_with_stops(&p,&targets,&[Some(stop),None,None,None],5,crate::IterationTiming::default()).unwrap();
            assert!(u.settlement_failures().is_empty());
            let tokens=u.token_events().iter().filter(|e|e.request_id()==ids[0]).map(|e|e.token_id()).collect::<Vec<_>>();
            assert_eq!(tokens,&targets[0][..stop]);assert_eq!(u.completions().len(),1);
            assert!(s.record_index(ids[0]).is_none());
            for id in &ids[1..]{assert_eq!(s.requests[s.record_index(*id).unwrap()].sequence.as_ref().unwrap().logical_length(),prompt as u32+8);}
            s.close(6,None).unwrap();
        }}
    }

    #[test]
    fn speculative_scheduler_commits_only_accepted_prefix_and_routes_all_tokens() {
        for prompt in [16, 17, 31, 32, 33] {
            for mismatch in 0..=7 {
                let (mut s, ids) = ready(prompt, 32);
                let p = s.plan_speculative_iteration(3, None).unwrap().unwrap();
                assert_eq!(p.rows.len(), 4);
                let authority = s.authorize_speculative_execution(&p).unwrap();
                assert_eq!(authority.plan(), &p);
                drop(authority);
                assert!(s.plan_iteration(4).is_err());
                let mut targets = Vec::new();
                let mut counts = Vec::new();
                for (i, row) in p.rows.iter().enumerate() {
                    assert_eq!(row.inputs.len(), 8);
                    let mut t = row.inputs[1..].to_vec();
                    t.push(100 + i as u32);
                    let first = (mismatch + i) % 8;
                    if first < 7 {
                        t[first] = 200 + i as u32;
                    }
                    counts.push(first + 1);
                    targets.push(t);
                }
                let before: Vec<_> = ids
                    .iter()
                    .map(|id| s.request_snapshot(*id).unwrap().logical_kv_tokens())
                    .collect();
                let mut corrupt = targets.clone();
                corrupt[3][7] = 49152;
                assert!(s
                    .complete_speculative_iteration(
                        &p,
                        &corrupt,
                        5,
                        crate::IterationTiming::new(0, 0)
                    )
                    .is_err());
                for (id, n) in ids.iter().zip(before) {
                    assert_eq!(s.request_snapshot(*id).unwrap().logical_kv_tokens(), n);
                }
                let ordinary = IterationResult::new(p.iteration_id(), Vec::new(), 0, 0).unwrap();
                assert!(s.complete_iteration(&ordinary, 5).is_err());
                let updates = s
                    .complete_speculative_iteration(
                        &p,
                        &targets,
                        5,
                        crate::IterationTiming::new(0, 0),
                    )
                    .unwrap();
                assert!(updates.settlement_failures().is_empty());
                assert_eq!(updates.token_events().len(), counts.iter().sum::<usize>());
                for ((row, target), count) in p.rows.iter().zip(&targets).zip(&counts) {
                    let snapshot = s.request_snapshot(row.request_id).unwrap();
                    assert_eq!(snapshot.logical_kv_tokens(), prompt + count);
                    assert_eq!(snapshot.generated_tokens(), 1 + count);
                    let events: Vec<_> = updates
                        .token_events()
                        .iter()
                        .filter(|e| e.request_id() == row.request_id)
                        .collect();
                    for (i, event) in events.iter().enumerate() {
                        assert_eq!(event.token_id(), target[i]);
                        assert_eq!(event.generated_index(), i + 1);
                    }
                }
                assert!(s
                    .complete_speculative_iteration(
                        &p,
                        &targets,
                        6,
                        crate::IterationTiming::new(0, 0)
                    )
                    .is_err());
                let next = s.plan_iteration(6).unwrap().into_parts().0.unwrap();
                let authority = s.authorize_execution(&next).unwrap();
                drop(authority);
                for item in next.decode_items() {
                    let i = p
                        .rows
                        .iter()
                        .position(|r| r.request_id == item.request_id())
                        .unwrap();
                    assert_eq!(item.input_tokens(), &targets[i][counts[i] - 1..counts[i]]);
                }
                s.abort_iteration(next.iteration_id(), ExecutionAbort::NotDispatched, 7)
                    .unwrap();
                s.close(8, None).unwrap();
            }
        }
    }
    #[test]
    fn speculative_scheduler_cancellation_eos_length_and_abort() {
        for limit in 3..=9 {
            let (mut s, ids) = ready(32, limit);
            let p = s.plan_speculative_iteration(3, Some(99)).unwrap().unwrap();
            s.cancel(ids[0], 4).unwrap();
            let mut targets: Vec<_> = p
                .rows
                .iter()
                .map(|r| {
                    let mut t = r.inputs[1..].to_vec();
                    t.push(100);
                    t
                })
                .collect();
            targets[1][0] = 99;
            let updates = s
                .complete_speculative_iteration(&p, &targets, 5, crate::IterationTiming::new(0, 0))
                .unwrap();
            assert!(updates.settlement_failures().is_empty());
            assert!(!updates
                .token_events()
                .iter()
                .any(|e| e.request_id() == ids[0]));
            assert_eq!(updates.completions().len(), 4);
            assert_eq!(s.pool.stats().allocated_block_count(), 0);
            s.close(6, None).unwrap();
        }
        for abort in [
            ExecutionAbort::NotDispatched,
            ExecutionAbort::DeviceQuiescedMutationUnknown,
        ] {
            let (mut s, _) = ready(32, 32);
            let p = s.plan_speculative_iteration(3, None).unwrap().unwrap();
            assert!(s
                .abort_iteration(p.iteration_id(), abort, 4)
                .unwrap()
                .settlement_failures()
                .is_empty());
            s.close(5, None).unwrap();
        }
    }
    #[test]
    fn speculative_scheduler_rejects_foreign_plans_and_contains_commit_fault() {
        let (mut s, _) = ready(32, 32);
        let p = s.plan_speculative_iteration(3, None).unwrap().unwrap();
        let targets: Vec<_> = p
            .rows
            .iter()
            .map(|r| {
                let mut t = r.inputs[1..].to_vec();
                t.push(100);
                t
            })
            .collect();
        let mut foreign = p.clone();
        foreign.rows[0].inputs[0] += 1;
        assert!(s.authorize_speculative_execution(&foreign).is_err());
        assert!(s
            .complete_speculative_iteration(
                &foreign,
                &targets,
                4,
                crate::IterationTiming::new(0, 0)
            )
            .is_err());
        s.test_post_validation_commit_fault_after_successful_commits = Some(1);
        let result = s
            .complete_speculative_iteration(&p, &targets, 4, crate::IterationTiming::new(0, 0))
            .unwrap();
        assert!(result.token_events().is_empty());
        assert_eq!(result.completions().len(), 4);
        assert_eq!(result.settlement_failures().len(), 1);
        assert!(result.iteration_metric().is_none());
        assert_eq!(s.pool.stats().allocated_block_count(), 0);
        s.close(5, None).unwrap();
        let (mut s, _) = ready(32, 2);
        assert!(s.plan_speculative_iteration(3, None).unwrap().is_none());
        let ordinary = s.plan_iteration(4).unwrap().into_parts().0.unwrap();
        s.abort_iteration(ordinary.iteration_id(), ExecutionAbort::NotDispatched, 5)
            .unwrap();
        s.close(6, None).unwrap();
    }
    #[test]
    fn speculative_explicit_progress_preserves_original_request_geometry(){
        use crate::descriptor::{shape_progress::InputStage,variable_wire};
        let(mut s,_)=ready(32,32);let plan=s.plan_speculative_iteration(3,None).unwrap().unwrap();
        let authority=s.authorize_speculative_execution(&plan).unwrap();
        let owner=crate::authority::VariableOwnerGeometry{generation:1,last_accepted_replay:0,catalog_digest:[7;32],max_active_rows:32,physical_block_count:64,context_tokens:128,packed_prefill:true,mixed_execution:true,shared_prefixes:false};
        let e=authority.verification_expectation(&owner,1,&[1,2,3,4]).unwrap();
        assert_eq!(e.stage,InputStage::Verification);
        for row in &e.rows{assert_eq!(row.progress.prompt_tokens,32);assert_eq!(row.progress.generated_index,1);assert_eq!(row.progress.committed_tokens,32);assert_eq!(row.progress.input_tokens,8);}
        let mut bytes=vec![0;variable_wire::MIXED_REQUEST_BYTES];variable_wire::encode_into(&mut bytes,&e).unwrap();
        variable_wire::validate_packet(&bytes,&mut vec![0;bytes.len()],&e).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()),3);
        if let Ok(path)=std::env::var("RILEY_SPECULATIVE_PACKET"){std::fs::write(path,&bytes).unwrap();}
        for change in 0..6 {let mut bad=e.clone();match change {
            0=>bad.rows[0].progress.generated_index=0,
            1=>bad.rows[0].progress.committed_tokens=33,
            2=>bad.rows[0].progress.input_tokens=9,
            3=>bad.rows[0].progress.output_limit=8,
            4=>bad.rows[0].progress.stage=InputStage::Prefill,
            _=>bad.shared_prefixes=true,
        }assert!(variable_wire::validate(&bad).is_err());}
        drop(authority);s.abort_iteration(plan.iteration_id(),ExecutionAbort::NotDispatched,4).unwrap();s.close(5,None).unwrap();
    }

    #[test]
    fn speculative_candidates_skip_missing_drafts_without_hiding_later_owners(){
        let(mut s,ids)=ready(32,32);
        for id in &ids[..3]{let i=s.record_index(*id).unwrap();s.requests[i].descriptor.prompt_token_ids=(100..132).collect();s.requests[i].generated_token_ids[0]=999;}
        let plan=s.plan_speculative_iteration(3,None).unwrap().unwrap();assert_eq!(plan.rows.len(),1);assert_eq!(plan.rows[0].request_id,ids[3]);
        s.abort_iteration(plan.iteration_id(),ExecutionAbort::NotDispatched,4).unwrap();s.close(5,None).unwrap();
    }

    #[test]
    fn wide_verification_preserves_active_batch_and_reserves_budget_for_every_owner(){
        for owners in [8,16,32]{for budget in [owners,owners+1,256]{
            let(mut s,ids)=ready_owners(32,32,owners);s.config.iteration_token_budget=budget;
            let p=s.plan_wide_speculative_iteration(3,None).unwrap();
            if budget==owners{assert!(p.is_none());s.close(4,None).unwrap();continue;}
            let p=p.unwrap();assert_eq!(p.rows.len(),owners);assert_eq!(p.rows.iter().map(|r|r.inputs.len()).sum::<usize>(),budget.min(owners*8));
            let authority=s.authorize_speculative_execution(&p).unwrap();
            let owner=crate::authority::VariableOwnerGeometry{generation:1,last_accepted_replay:0,catalog_digest:[7;32],max_active_rows:32,physical_block_count:256,context_tokens:128,packed_prefill:true,mixed_execution:true,shared_prefixes:false};
            let e=authority.verification_expectation(&owner,1,&(1..=owners as u64).collect::<Vec<_>>()).unwrap();
            assert_eq!(e.mode,crate::descriptor::ResultMode::Greedy);
            let mut packet=vec![0;crate::descriptor::variable_wire::MIXED_REQUEST_BYTES];crate::descriptor::variable_wire::encode_into(&mut packet,&e).unwrap();
            if owners==32 && budget==256{if let Ok(path)=std::env::var("RILEY_SPECULATIVE_WIDE_PACKET"){std::fs::write(path,packet).unwrap();}}
            drop(authority);
            let targets:Vec<_>=p.rows.iter().map(|r|{let mut t=r.inputs[1..].to_vec();t.push(123);t}).collect();
            let result=s.complete_speculative_iteration(&p,&targets,4,crate::IterationTiming::new(0,0)).unwrap();
            assert!(result.settlement_failures().is_empty());assert_eq!(result.token_events().len(),budget.min(owners*8));
            for id in ids{assert!(result.token_events().iter().any(|e|e.request_id()==id));}s.close(5,None).unwrap();
        }}
    }

}
