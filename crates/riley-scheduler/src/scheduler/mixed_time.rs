//! Bounded wall-time feedback, independent of CUDA and disabled by default.
//! Eight workload classes isolate context ranges and decode coexistence. Cold
//! classes retain the configured budget. Updates need eight saturated samples;
//! short batches cannot spuriously reduce the budget. This is not a hard SLO.
#[derive(Debug)]
pub(super) struct MixedTimePolicy {
    target_ns: u64,
    maximum: usize,
    buckets: [Bucket; 8],
}
#[derive(Clone, Copy, Debug)]
struct Bucket { limit: usize, samples: u32, sum: u128, adjustments: u64 }
impl MixedTimePolicy {
    pub(super) fn new(target_ns:u64, maximum:usize)->Self {
        Self{target_ns,maximum,buckets:[Bucket{limit:maximum,samples:0,sum:0,adjustments:0};8]}
    }
    pub(super) fn class(context:usize,decode:bool)->usize {
        let c=if context<=128{0}else if context<=256{1}else if context<=512{2}else{3};c*2+usize::from(decode)
    }
    pub(super) fn limit(&self,class:usize)->usize {self.buckets[class].limit}
    pub(super) fn observe(&mut self,class:usize,prefill_tokens:usize,wall_ns:u64) {
        let b=&mut self.buckets[class];
        if wall_ns==0 || prefill_tokens<b.limit.saturating_sub(64).max(1){return;}
        b.samples+=1;b.sum+=u128::from(wall_ns);
        if b.samples<8{return;}
        let mean=b.sum/8;b.sum=0;b.samples=0;
        let previous=b.limit;
        if mean>u128::from(self.target_ns)*6/5 {b.limit=b.limit.saturating_sub(64).max(128.min(self.maximum));}
        else if mean<u128::from(self.target_ns)*4/5 {b.limit=b.limit.saturating_add(64).min(self.maximum);}
        if previous!=b.limit {b.adjustments=b.adjustments.saturating_add(1);}
    }
    pub(super) fn state(&self)->[(usize,u64);8] {self.buckets.map(|b|(b.limit,b.adjustments))}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn cold_unsaturated_and_classes_are_isolated(){
        let mut p=MixedTimePolicy::new(1000,512);
        for _ in 0..32 {p.observe(0,64,100000);}
        assert_eq!(p.limit(0),512);
        for _ in 0..7 {p.observe(0,480,2000);}assert_eq!(p.limit(0),512);
        p.observe(0,480,2000);assert_eq!(p.limit(0),448);assert_eq!(p.limit(1),512);
    }
    #[test] fn bounded_recovery_and_deadband(){
        let mut p=MixedTimePolicy::new(1000,512);
        for _ in 0..200 {p.observe(0,512,2000);}assert_eq!(p.limit(0),128);
        for _ in 0..8 {p.observe(0,512,1000);}assert_eq!(p.limit(0),128);
        for _ in 0..200 {p.observe(0,512,500);}assert_eq!(p.limit(0),512);
    }
    #[test] fn context_boundaries(){
        assert_eq!([128,129,256,257,512,513].map(|c|MixedTimePolicy::class(c,true)),[1,3,3,5,5,7]);
    }
}

#[cfg(test)]
mod scheduler_tests {
    use super::super::*;
    fn scheduler()->Scheduler {
        Scheduler::new_with_execution_shape(SchedulerConfig{max_waiting_requests:8,max_waiting_prompt_tokens:8192,max_active_sequences:4,max_sequence_tokens:1024,iteration_token_budget:512,max_prefill_chunk_tokens:512,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,max_promised_kv_blocks:256,metrics_window_samples:16},KvLayout::checked(30,256,3,64).unwrap(),ExecutionShapePolicy::MixedPrefillDecode32).unwrap()
    }
    #[test]
    fn mixed_time_abort_and_commit_preserve_decode_and_prefill_progress() {
        let mut s=scheduler();s.enable_mixed_time_budget(1_000_000).unwrap();
        for class in 0..8 {for _ in 0..80 {s.mixed_time.as_mut().unwrap().observe(class,512,2_000_000);}}
        let ids:Vec<_>=[1,398,398,398].map(|n|s.submit(RequestDescriptor::new(vec![17;n],4),0).unwrap().request_id()).into_iter().collect();
        assert!(s.enable_mixed_time_budget(1_000_000).is_err());
        let mut steps=0;let mut retried=false;let mut mixed=false;
        loop {
            let ready_decode=s.requests.iter().filter(|r|r.state==RequestState::Decoding).count();
            let Some(plan)=s.plan_iteration(steps*3+1).unwrap().into_parts().0 else {break};
            assert_eq!(plan.decode_items().len(),ready_decode);
            assert!(plan.prefill_items().iter().map(|w|w.input_tokens().len()).sum::<usize>()<=128);
            assert!(plan.total_tokens()<=512);
            mixed|=!plan.prefill_items().is_empty()&&!plan.decode_items().is_empty();
            let before=s.mixed_time_budget_state();
            if !retried {
                let wrong=IterationResult::new(IterationId::new(999).unwrap(),vec![],0,0).unwrap();
                assert!(s.complete_iteration_with_execution_wall(&wrong,steps*3+2,9_000_000).is_err());
                assert_eq!(before,s.mixed_time_budget_state());
                s.abort_iteration(plan.iteration_id(),ExecutionAbort::NotDispatched,steps*3+2).unwrap();
                assert_eq!(before,s.mixed_time_budget_state());retried=true;steps+=1;continue;
            }
            let outputs=plan.output_slots().iter().map(|&slot|IterationOutput::new(slot,23,false)).collect();
            let result=IterationResult::new(plan.iteration_id(),outputs,0,0).unwrap();
            assert!(s.complete_iteration_with_execution_wall(&result,steps*3+2,1_000_000).unwrap().settlement_failures().is_empty());
            steps+=1;assert!(steps<64);
        }
        assert!(mixed);assert_eq!(s.active_sequence_count(),0);
        for id in ids {assert_eq!(s.request_state(id),Some(RequestState::Finished));}
        assert_eq!(s.pool_stats().allocated_block_count(),0);
        s.close(steps*3+3,None).unwrap();
    }
}
