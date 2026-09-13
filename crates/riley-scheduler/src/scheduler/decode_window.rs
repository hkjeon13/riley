//! Opt-in two-decode reservation and settlement; no CUDA dispatch is implied.
use super::*;
use crate::plan::DecodeWindowPlan;

impl Scheduler {
    /// Plan two dependent decode iterations with one final reservation.
    /// Returns None when the current selection needs prefill or has fewer than
    /// two outputs remaining; the caller should use ordinary planning then.
    /// Abort by the first ID: NotDispatched means neither step was dispatched;
    /// DeviceQuiescedMutationUnknown requires both steps to be quiescent.
    pub fn plan_decode_window(&mut self,now_ns:u64)->SchedulerResult<Option<DecodeWindowPlan>> {
        let cpu_started=Instant::now();
        self.ensure_completion_backlog_empty()?;self.observe_now(now_ns)?;
        if let Some(p)=&self.inflight {return Err(SchedulerError::IterationInFlight{iteration_id:p.iteration_id});}
        if self.execution_shape_policy!=ExecutionShapePolicy::MixedPrefillDecode32 {
            return Err(SchedulerError::InvalidPlan{field:"decode window",reason:"requires mixed32 execution policy"});
        }
        // Ordinary planning performs admission and timeout completion handling.
        if !self.waiting.is_empty() && (self.active_sequences<self.config.max_active_sequences || self.config.admission_timeout_ns.is_some()) {return Ok(None);}
        let(candidates,aging)=self.select_candidates(now_ns)?;
        if candidates.is_empty(){return Ok(None);}
        for c in &candidates {
            let r=&self.requests[self.record_index(c.request_id).ok_or(SchedulerError::UnknownRequest{request_id:c.request_id})?];
            if c.kind!=WorkKind::Decode || r.descriptor.max_new_tokens.saturating_sub(r.generated_token_ids.len())<2 {return Ok(None);}
        }
        let first_id=self.peek_iteration_id()?;
        let following=self.next_iteration_id.checked_add(2).ok_or(SchedulerError::IdentifierExhausted{kind:"iteration"})?;
        let second_id=IterationId::new(first_id.get()+1).ok_or(SchedulerError::IdentifierExhausted{kind:"iteration"})?;
        let first=self.prepare_plan(first_id,&candidates,now_ns,Some(second_id))?;
        let second=(|| {
            let p=self.inflight.as_ref().ok_or(SchedulerError::NoIterationInFlight)?;
            let mut items=Vec::new();let mut tables=Vec::new();
            try_reserve_exact(&mut items,p.items.len(),"successor items")?;
            try_reserve_exact(&mut tables,p.items.len(),"successor tables")?;
            for item in &p.items {
                let record=&self.requests[self.record_index(item.request_id).ok_or(SchedulerError::UnknownRequest{request_id:item.request_id})?];
                let seq=record.sequence.as_ref().ok_or(SchedulerError::NoIterationInFlight)?;
                let index=tables.len();tables.push(OwnedBlockTable::copy_from_v1(item.request_id,seq.reserved_block_table(&item.reservation)?)?);
                items.push(WorkItem::new(item.request_id,WorkKind::Decode,copy_tokens(&[0],"future token placeholder")?,item.target_logical_length,index,item.output_slot)?);
            }
            IterationPlan::new(second_id,Vec::new(),items,tables)
        })();
        let second=match second {Ok(p)=>p,Err(error)=>{
            let pending=self.inflight.take().expect("prepared window");
            return self.rollback_prepared_plan(pending.items,error);
        }};
        self.inflight.as_mut().expect("prepared window").scheduler_cpu_ns=elapsed_ns(cpu_started);
        self.next_iteration_id=following;self.aging_override_last_iteration=aging;self.refresh_metric_gauges();
        Ok(Some(DecodeWindowPlan{first,second}))
    }

    /// Binds both immutable plans to the actual shared scheduler reservations.
    pub fn authorize_decode_window<'a>(&'a self,window:&'a DecodeWindowPlan)->SchedulerResult<crate::AuthorizedDecodeWindow<'a>> {
        self.validate_inflight_reservations()?;
        let pending=self.inflight.as_ref().ok_or(SchedulerError::NoIterationInFlight)?;
        let bad=||SchedulerError::InvalidPlan{field:"window authority",reason:"plans differ from live shared reservation"};
        if pending.successor!=Some(window.second.iteration_id()) || pending.iteration_id!=window.first.iteration_id() || !window.first.prefill_items().is_empty() || !window.second.prefill_items().is_empty() || window.first.batch_size()!=pending.items.len() || window.second.batch_size()!=pending.items.len(){return Err(bad());}
        let mut first_rows=Vec::new();let mut second_rows=Vec::new();
        try_reserve_exact(&mut first_rows,pending.items.len(),"window first authority")?;
        try_reserve_exact(&mut second_rows,pending.items.len(),"window second authority")?;
        for ((item,a),b) in pending.items.iter().zip(window.first.decode_items()).zip(window.second.decode_items()) {
            let record=&self.requests[self.record_index(item.request_id).ok_or_else(bad)?];let seq=record.sequence.as_ref().ok_or_else(bad)?;
            let prefix=item.target_logical_length-1;
            let(ta,tb)=OwnedBlockTable::copy_reserved_window(item.request_id,seq,&item.reservation,prefix)?;
            if a.request_id()!=item.request_id || b.request_id()!=item.request_id || a.output_slot()!=item.output_slot || b.output_slot()!=item.output_slot || a.target_logical_length()!=prefix || b.target_logical_length()!=item.target_logical_length || record.generated_token_ids.last().map(std::slice::from_ref)!=Some(a.input_tokens()) || b.input_tokens()!=[0] || window.first.block_tables().get(a.block_table_index())!=Some(&ta) || window.second.block_tables().get(b.block_table_index())!=Some(&tb){return Err(bad());}
            first_rows.push(crate::AuthorizedExecutionRow{request_id:item.request_id,committed_length:seq.logical_length() as usize,prompt_tokens:record.descriptor.prompt_token_ids.len(),generated_index:record.generated_token_ids.len(),max_output_tokens:record.descriptor.max_new_tokens,table:ta});
            second_rows.push(crate::AuthorizedExecutionRow{request_id:item.request_id,committed_length:prefix,prompt_tokens:record.descriptor.prompt_token_ids.len(),generated_index:record.generated_token_ids.len()+1,max_output_tokens:record.descriptor.max_new_tokens,table:tb});
        }
        let mut owners=Vec::new();try_reserve_exact(&mut owners,self.pool.layout().physical_block_count(),"window block ownership")?;
        for record in &self.requests {
            if let Some(seq)=&record.sequence {
                let table=if let Some(item)=pending.items.iter().find(|i|i.request_id==record.request_id){seq.reserved_block_table(&item.reservation)?}else{seq.block_table()?};
                owners.extend(table.physical_block_ids().iter().map(|&id|(id,record.request_id)));
            }
        }
        Ok(crate::AuthorizedDecodeWindow {
            first:crate::AuthorizedExecution{scheduler:self,plan:&window.first,rows:first_rows,block_owners:owners.clone()},
            second:crate::AuthorizedExecution{scheduler:self,plan:&window.second,rows:second_rows,block_owners:owners},
        })
    }

    /// Settle after BOTH executions are quiescent and have passed full runtime
    /// identity/numerical validation. This is not a CUDA completion fence.
    /// A first-step stop/cancel suppresses the successor and releases its append
    /// only after this caller-supplied window-wide drain evidence.
    pub fn complete_decode_window_after_drain(&mut self,first:&IterationResult,second:&IterationResult,now_ns:u64)->SchedulerResult<IterationUpdates> {
        let cpu_started=Instant::now();
        self.ensure_completion_backlog_empty()?;self.validate_now(now_ns)?;self.validate_inflight_reservations()?;
        let p=self.inflight.as_ref().ok_or(SchedulerError::NoIterationInFlight)?;
        let second_id=p.successor.ok_or(SchedulerError::InvalidPlan{field:"decode window",reason:"ordinary iteration is not a window"})?;
        for(result,id)in [(first,p.iteration_id),(second,second_id)] {
            if result.iteration_id()!=id {return Err(SchedulerError::UnexpectedIteration{expected:id,actual:result.iteration_id()});}
            if result.outputs().len()!=p.expected_output_slots.len() || result.outputs().iter().any(|o|o.token_id()>=49152 || !p.expected_output_slots.contains(&o.slot())) {
                return Err(SchedulerError::InvalidIterationResult{field:"window outputs",reason:"both complete slot sets and vocabulary-valid tokens are required"});
            }
        }
        let count=p.items.len();let mut updates=IterationUpdates::empty();
        try_reserve_exact(&mut updates.token_events,count*2,"window token events")?;
        try_reserve_exact(&mut updates.completions,count,"window completions")?;
        try_reserve_exact(&mut updates.settlement_failures,count,"window failures")?;
        self.ensure_completion_capacity(count)?;
        // No output or ownership is mutated before both result sets validate.
        self.last_now_ns=Some(now_ns);let pending=self.inflight.take().expect("validated window");
        self.record_dispatched_shape(&pending);let planning_cpu_ns=pending.scheduler_cpu_ns;
        for mut item in pending.items {
            let id=item.request_id;
            let outcome=(||->SchedulerResult<()> {
                let slot=item.output_slot.ok_or(SchedulerError::InvalidPlan{field:"window slot",reason:"decode requires output"})?;
                let a=*first.outputs().iter().find(|o|o.slot()==slot).expect("validated first slot");
                let b=*second.outputs().iter().find(|o|o.slot()==slot).expect("validated second slot");
                let index=self.record_index(id).ok_or(SchedulerError::UnknownRequest{request_id:id})?;
                let prefix=item.target_logical_length-1;
                let terminal=self.requests[index].cancellation_deferred || a.stop();
                let seq=self.requests[index].sequence.as_mut().ok_or(SchedulerError::NoIterationInFlight)?;
                seq.commit_prefix(&mut self.pool,&mut item.reservation,prefix)?;
                if terminal {
                    seq.discard_completed_append(&mut self.pool,item.reservation)?;
                    return self.publish_committed_item(SettledInflightItem{request_id:id,kind:WorkKind::Decode,target_logical_length:prefix,output:Some(a)},now_ns,&mut updates);
                }
                self.publish_committed_item(SettledInflightItem{request_id:id,kind:WorkKind::Decode,target_logical_length:prefix,output:Some(a)},now_ns,&mut updates)?;
                let index=self.record_index(id).ok_or(SchedulerError::UnknownRequest{request_id:id})?;
                self.requests[index].sequence.as_mut().ok_or(SchedulerError::NoIterationInFlight)?.commit(&mut self.pool,item.reservation)?;
                self.publish_committed_item(SettledInflightItem{request_id:id,kind:WorkKind::Decode,target_logical_length:item.target_logical_length,output:Some(b)},now_ns,&mut updates)
            })();
            if let Err(error)=outcome {
                let error=self.contain_live_request_failure(id,now_ns,error);
                updates.settlement_failures.push(RequestSettlementFailure{request_id:id,error});
            }
        }
        self.drain_completion_outbox_into(&mut updates.completions);
        if updates.settlement_failures.is_empty() {
            for (step,result) in [first,second].into_iter().enumerate() {
                let sample=IterationMetricSample{batch_size:count,prefill_tokens:0,decode_tokens:count,scheduler_cpu_ns:if step==0{planning_cpu_ns.saturating_add(elapsed_ns(cpu_started))}else{0},gpu_execution_ns:result.gpu_execution_ns(),gpu_idle_gap_ns:result.gpu_idle_gap_ns()};
                observe_metric(&mut self.metrics_degraded,self.metrics.record_iteration(sample),"completed decode window step");
                // The singular result field cannot describe two iterations.
                // Both samples are recorded in scheduler metrics above.
            }
        } else {observe_metric(&mut self.metrics_degraded,self.metrics.record_aborted_iteration(),"failed decode window settlement");}
        self.refresh_metric_gauges();Ok(updates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready(max_new:usize)->(Scheduler,Vec<RequestId>){
        let config=SchedulerConfig{max_waiting_requests:8,max_waiting_prompt_tokens:1024,max_active_sequences:2,max_sequence_tokens:64,iteration_token_budget:128,max_prefill_chunk_tokens:64,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:16,metrics_window_samples:16};
        let mut s=Scheduler::new_with_execution_shape(config,KvLayout::checked(30,16,3,64).unwrap(),ExecutionShapePolicy::MixedPrefillDecode32).unwrap();
        let ids=(0..2).map(|_|s.submit(RequestDescriptor::new(vec![3;15],max_new),0).unwrap().request_id()).collect();
        assert!(s.plan_decode_window(0).unwrap().is_none());
        let p=s.plan_iteration(1).unwrap().into_parts().0.unwrap();let r=result(&p,10,None);
        s.complete_iteration(&r,2).unwrap();(s,ids)
    }
    fn result(p:&IterationPlan,base:u32,stop:Option<usize>)->IterationResult{
        IterationResult::new(p.iteration_id(),p.prefill_items().iter().chain(p.decode_items()).enumerate().filter_map(|(i,w)|w.output_slot().map(|slot|IterationOutput::new(slot,base+i as u32,stop==Some(i)))).collect(),100,5).unwrap()
    }
    #[test] fn paired_decode_commits_two_tokens_and_preserves_request_progress(){
        let(mut s,ids)=ready(4);let w=s.plan_decode_window(3).unwrap().unwrap();
        assert_eq!(w.first().block_tables()[0].valid_tokens(),[16]);assert_eq!(w.second().block_tables()[0].valid_tokens(),[16,1]);
        assert_eq!(s.pool.stats().allocated_block_count(),4);assert!(s.authorize_execution(w.first()).is_err());assert!(s.authorize_execution(w.second()).is_err());
        let a=result(w.first(),20,None);let b=result(w.second(),30,None);
        {let authority=s.authorize_decode_window(&w).unwrap();let owner=crate::authority::VariableOwnerGeometry{generation:7,last_accepted_replay:1,catalog_digest:[9;32],max_active_rows:2,physical_block_count:16,context_tokens:64,packed_prefill:true,mixed_execution:true};
        let(first,second,future)=authority.prepare_wire(&owner,2,&[100,101],&[102,103]).unwrap();assert_eq!(first.rows[0].progress.generated_index,1);assert_eq!(second.rows[0].progress.generated_index,2);assert_eq!(future.committed_replay(),1);assert_eq!(future.references().len(),4480);}

        assert!(s.complete_iteration(&a,4).is_err());
        let updates=s.complete_decode_window_after_drain(&a,&b,4).unwrap();assert_eq!(updates.token_events().len(),4);assert!(updates.settlement_failures().is_empty());
        for(i,id)in ids.iter().enumerate(){let r=&s.requests[s.record_index(*id).unwrap()];assert_eq!(r.generated_token_ids,vec![10+i as u32,20+i as u32,30+i as u32]);assert_eq!(r.sequence.as_ref().unwrap().logical_length(),17);}
        assert!(s.plan_decode_window(5).unwrap().is_none(),"one output remaining uses ordinary planning");
        assert!(s.complete_decode_window_after_drain(&a,&b,5).is_err());s.close(6,None).unwrap();
    }
    #[test] fn prefix_stop_and_deferred_cancel_drop_successor_after_drain(){
        let(mut s,ids)=ready(4);let w=s.plan_decode_window(3).unwrap().unwrap();
        let a=result(w.first(),20,Some(0));let b=result(w.second(),30,None);
        s.cancel(ids[1],4).unwrap();assert_eq!(s.pool.stats().allocated_block_count(),4);
        let u=s.complete_decode_window_after_drain(&a,&b,5).unwrap();assert!(u.settlement_failures().is_empty());
        assert_eq!(u.token_events().len(),1);assert_eq!(u.token_events()[0].token_id(),20);
        assert_eq!(u.completions().len(),2);assert_eq!(s.pool.stats().allocated_block_count(),0);
        let stopped=u.completions().iter().find(|c|c.request_id()==ids[0]).unwrap();assert_eq!(stopped.generated_token_ids(),[10,20]);
        let cancelled=u.completions().iter().find(|c|c.request_id()==ids[1]).unwrap();assert_eq!(cancelled.generated_token_ids(),[11]);s.close(6,None).unwrap();
    }
    #[test] fn second_step_length_limit_and_new_admission_use_normal_lifecycle(){
        let(mut s,_)=ready(3);let w=s.plan_decode_window(3).unwrap().unwrap();
        let a=result(w.first(),20,None);let b=result(w.second(),30,None);
        let u=s.complete_decode_window_after_drain(&a,&b,4).unwrap();assert_eq!(u.completions().len(),2);assert_eq!(u.token_events().len(),4);assert!(u.completions().iter().all(|c|c.reason()==RequestFinishReason::Length));assert_eq!(s.pool.stats().allocated_block_count(),0);s.close(5,None).unwrap();
        let(mut s,ids)=ready(8);let w=s.plan_decode_window(3).unwrap().unwrap();
        let waiting=s.submit(RequestDescriptor::new(vec![4;7],4),4).unwrap().request_id();s.cancel(ids[0],4).unwrap();
        s.complete_decode_window_after_drain(&result(w.first(),20,None),&result(w.second(),30,None),5).unwrap();
        assert!(s.plan_decode_window(6).unwrap().is_none());
        let p=s.plan_iteration(6).unwrap().into_parts().0.unwrap();assert!(p.prefill_items().iter().any(|i|i.request_id()==waiting));
        s.abort_iteration(p.iteration_id(),ExecutionAbort::NotDispatched,7).unwrap();s.close(8,None).unwrap();
    }

    #[test] fn window_oom_rolls_back_already_reserved_rows_and_retries_same_ids(){
        let(mut s,ids)=ready(8);let next_id=s.next_iteration_id;
        let mut pressure=s.pool.create_sequence(208).unwrap();let hold=pressure.reserve_to(&mut s.pool,208).unwrap();pressure.commit(&mut s.pool,hold).unwrap();
        let before=s.pool.stats();assert!(s.plan_decode_window(3).is_err());assert!(s.inflight.is_none());assert_eq!(s.next_iteration_id,next_id);assert_eq!(s.pool.stats().allocated_block_count(),before.allocated_block_count());assert_eq!(s.pool.stats().free_block_count(),before.free_block_count());
        for id in ids {let r=&s.requests[s.record_index(id).unwrap()];assert_eq!(r.sequence.as_ref().unwrap().logical_length(),15);assert_eq!(r.state,RequestState::Decoding);}
        pressure.close(&mut s.pool).unwrap();let w=s.plan_decode_window(4).unwrap().unwrap();assert_eq!(w.first().iteration_id().get(),next_id);
        s.abort_iteration(w.first().iteration_id(),ExecutionAbort::NotDispatched,5).unwrap();s.close(6,None).unwrap();
    }

    #[test] fn full_c32_window_handles_ragged_progress_and_dense_output_slots(){
        let config=SchedulerConfig{max_active_sequences:32,max_sequence_tokens:128,iteration_token_budget:1024,max_prefill_chunk_tokens:64,max_promised_kv_blocks:256,..SchedulerConfig::default()};
        let mut s=Scheduler::new_with_execution_shape(config,KvLayout::checked(30,256,3,64).unwrap(),ExecutionShapePolicy::MixedPrefillDecode32).unwrap();
        let ids:Vec<_>=(0..32).map(|_|s.submit(RequestDescriptor::new(vec![3;15],64),0).unwrap().request_id()).collect();
        for tick in 1..40 {if ids.iter().all(|&id|s.request_state(id)==Some(RequestState::Decoding)){break;}
            let p=s.plan_iteration(tick*2).unwrap().into_parts().0.unwrap();s.complete_iteration(&result(&p,10,None),tick*2+1).unwrap();}
        assert!(ids.iter().all(|&id|s.request_state(id)==Some(RequestState::Decoding)));
        let w=s.plan_decode_window(100).unwrap().unwrap();assert_eq!(w.first().batch_size(),32);assert_eq!(s.current_gauges().outstanding_iterations,2);
        assert!(s.authorize_decode_window(&w).is_ok());
        let u=s.complete_decode_window_after_drain(&result(w.first(),100,None),&result(w.second(),200,None),101).unwrap();
        assert_eq!(u.token_events().len(),64);assert!(u.settlement_failures().is_empty());assert_eq!(s.current_gauges().outstanding_iterations,0);
        s.close(102,None).unwrap();
    }

    #[test] fn invalid_successor_is_atomic_and_not_dispatched_abort_restores_pages(){
        let(mut s,ids)=ready(4);let w=s.plan_decode_window(3).unwrap().unwrap();let before=s.pool.stats();
        let a=result(w.first(),20,None);let bad=result(w.second(),49152,None);
        assert!(s.complete_decode_window_after_drain(&a,&bad,4).is_err());assert_eq!(s.pool.stats(),before);
        for id in ids {let r=&s.requests[s.record_index(id).unwrap()];assert_eq!(r.generated_token_ids.len(),1);assert_eq!(r.sequence.as_ref().unwrap().logical_length(),15);}
        // No executor was called in this host test.
        assert!(s.abort_iteration(w.first().iteration_id(),ExecutionAbort::NotDispatched,4).unwrap().settlement_failures().is_empty());
        assert_eq!(s.pool.stats().allocated_block_count(),2);let next=s.plan_decode_window(5).unwrap().unwrap();assert!(next.first().iteration_id().get()>w.second().iteration_id().get());
        s.abort_iteration(next.first().iteration_id(),ExecutionAbort::DeviceQuiescedMutationUnknown,6).unwrap();assert_eq!(s.pool.stats().allocated_block_count(),0);s.close(7,None).unwrap();
    }
}
