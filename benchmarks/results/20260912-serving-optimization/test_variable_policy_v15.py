from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-scheduler/src/authority.rs');s=p.read_text();a='    #[test]\n    fn variable_authority_tracks_chunked_prefill_and_decode_from_live_scheduler()'
b='''    #[test]
    fn variable_policy_batches_live_decode_and_retries_without_false_progress() {
        use crate::{IterationResult,IterationOutput,ExecutionAbort,ExecutionShapePolicy};
        use crate::descriptor::{shape_progress::InputStage,variable_wire};
        let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig {
            max_waiting_requests:8,max_waiting_prompt_tokens:4096,max_active_sequences:4,
            max_sequence_tokens:1024,iteration_token_budget:73,max_prefill_chunk_tokens:73,
            aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,admission_timeout_ns:None,
            max_promised_kv_blocks:256,metrics_window_samples:16,
        },riley_runtime::paged_kv::KvLayout::checked(30,256,3,64).unwrap(),ExecutionShapePolicy::VariablePrefillDecodeN).unwrap();
        let ids:Vec<_>=[16,128,398,73].into_iter().map(|n|scheduler.submit(RequestDescriptor::new(vec![17;n],32),0).unwrap().request_id()).collect();
        let mut owner=super::VariableOwnerGeometry{generation:1,last_accepted_replay:0,catalog_digest:[9;32],max_active_rows:4,physical_block_count:256,context_tokens:1024};
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
            assert!(replay<256);
        }
        assert!(partial && retried && widest>=3);
        for id in ids {assert_eq!(scheduler.request_state(id),Some(crate::RequestState::Finished));}
        scheduler.close(now+2,None).unwrap();
    }

'''+a
assert a in s;s=s.replace(a,b);p.write_text(s)
