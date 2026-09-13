#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_runtime::llama::variable_session::VariableSessionIdentity;
use riley_scheduler::{Scheduler,SchedulerConfig,ExecutionShapePolicy,RequestDescriptor,IterationResult,IterationOutput,OutputSlot,IterationTiming,SampledIterationToken,IterationId};
type TestResult<T> = Result<T,Box<dyn std::error::Error>>;
fn owner(i:VariableSessionIdentity)->riley_scheduler::VariableOwnerGeometry {
    riley_scheduler::VariableOwnerGeometry{generation:i.generation,last_accepted_replay:i.last_accepted_replay,catalog_digest:i.catalog_digest,physical_block_count:i.physical_block_count,context_tokens:i.context_tokens,max_active_rows:32,packed_prefill:i.packed_prefill,mixed_execution:i.mixed_execution,shared_prefixes:i.shared_prefixes}
}
fn result(id:IterationId,rows:Vec<(u32,u32)>)->TestResult<IterationResult> {
    Ok(IterationResult::new(id,rows.into_iter().map(|(s,t)|IterationOutput::new(OutputSlot::new(s),t,false)).collect(),0,0)?)
}
#[test]
#[ignore="requires real SmolLM2 checkpoint and CUDA GPU"]
fn rolling_tickets_match_serial_generation_and_cancel_after_prefix()->TestResult<()> {
    let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
    let context=CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut reference=None;
    for mode in 0..4 {
        let mut stream=context.create_stream()?;
        let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,64,1,256)?,PreparedLlamaForwardConfig::default());
        let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
        let mut session=executor.into_owned_buffered_variable_mixed_session(&context,128,true)?;
        let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{max_active_sequences:4,max_sequence_tokens:128,iteration_token_budget:128,max_prefill_chunk_tokens:32,max_promised_kv_blocks:256,..SchedulerConfig::default()},riley_runtime::paged_kv::KvLayout::checked(30,256,3,64)?,ExecutionShapePolicy::MixedPrefillDecode32)?;
        let ids:Vec<_>=(0..4).map(|r|scheduler.submit(RequestDescriptor::new((0..15+r).map(|i|17+((i*37+r*73)%900)as u32).collect(),24),0).unwrap().request_id()).collect();
        let mut tokens=vec![Vec::new();4];let mut completed=0;let mut now=1;let mut rolls=0;let mut cancelled=false;
        while completed<4 {
            let mut updates=Vec::new();
            if let Some(mut window)=if mode>0 {scheduler.plan_decode_window(now)?}else{None} {
                let (identity,replay,a,mut b)=session.issue_decode_window(window.first().batch_size())?;
                let prepared=scheduler.authorize_decode_window(&window)?.prepare_wire(&owner(identity),replay,&a,&b)?;
                session.submit_prepared_decode_window(prepared)?;
                loop {
                    let first=result(window.first().iteration_id(),session.wait_decode_window_first()?)?;now+=1;
                    let prefix=scheduler.complete_decode_window_prefix(&first,now)?;
                    let published=prefix.is_some();if let Some(u)=prefix {updates.push(u);}
                    if mode==2 && rolls==3 && !cancelled {scheduler.cancel(ids[0],now)?;cancelled=true;}
                    if published {
                        if let Some(next)=scheduler.roll_decode_window(now)? {
                            let(identity,replay,cookies)=session.promote_decode_window_successor(window.first().iteration_id().get(),next.first().iteration_id().get())?;
                            let prepared=scheduler.authorize_decode_window(&next)?.prepare_wire(&owner(identity),replay-1,&b,&cookies)?;
                            if mode==3 && rolls==3 {
                                let mut bad=prepared.second().clone();bad.rows[0].cookie=0;
                                assert!(session.submit_decode_window_successor(||Ok(bad)).is_err());
                                assert!(session.issue_rows(1).is_err());
                                session.close()?; // establishes quiescence before any KV retirement
                                let u=scheduler.abort_iteration(next.first().iteration_id(),riley_scheduler::ExecutionAbort::DeviceQuiescedMutationUnknown,now)?;
                                assert!(u.settlement_failures().is_empty());assert_eq!(u.completions().len(),4);
                                let expected:&Vec<Vec<u32>>=reference.as_ref().unwrap();
                                for c in u.completions() {let row=ids.iter().position(|id|*id==c.request_id()).unwrap();assert_eq!(c.generated_token_ids(),&expected[row][..c.generated_token_ids().len()]);}
                                scheduler.close(now+1,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());
                                println!("ROLLING_GPU mode=3 injected_cookie_failure=true retained_ticket_drained=true completions=4 allocation_zero=true");
                                context.close()?;return Ok(());
                            }
                            session.submit_decode_window_successor(||Ok(prepared.second().clone()))?;
                            b=cookies;window=next;rolls+=1;
                            // Host fixture collection is deliberately outside GPU timing claims.
                            for u in updates.drain(..) {assert!(u.settlement_failures().is_empty());for e in u.token_events(){tokens[ids.iter().position(|id|*id==e.request_id()).unwrap()].push(e.token_id());}completed+=u.completions().len();}
                            continue;
                        }
                    }
                    let second=result(window.second().iteration_id(),session.wait_decode_window_second()?)?;now+=1;
                    updates.push(scheduler.complete_decode_window_after_drain(&first,&second,now)?);
                    session.confirm_decode_window_commit(window.first().iteration_id().get(),window.second().iteration_id().get())?;break;
                }
            } else {
                let plan=scheduler.plan_iteration(now)?.into_parts().0.ok_or("no progress")?;
                let d=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&scheduler.authorize_execution(&plan)?,&mut session,true).unwrap();
                let samples=d.greedy_token_ids().iter().map(|&t|SampledIterationToken::new(t,false)).collect::<Vec<_>>();
                let r=d.into_result(&samples,IterationTiming::new(0,0))?;now+=1;
                updates.push(scheduler.complete_iteration(&r,now)?);session.confirm_scheduler_commit(plan.iteration_id().get())?;
            }
            for u in updates {assert!(u.settlement_failures().is_empty());for e in u.token_events(){tokens[ids.iter().position(|id|*id==e.request_id()).unwrap()].push(e.token_id());}completed+=u.completions().len();}
            now+=1;assert!(now<512);
        }
        if let Some(expected)=&reference {let expected:&Vec<Vec<u32>>=expected;for(a,b)in tokens.iter().zip(expected){assert_eq!(a,&b[..a.len()]);}assert!(rolls>=4);}
        else {reference=Some(tokens.clone());}
        if mode<2 {assert!(tokens.iter().all(|t|t.len()==24));}else{assert!(cancelled);assert!(tokens[0].len()<24);assert!(tokens[1..].iter().all(|t|t.len()==24));}
        session.close()?;scheduler.close(now,None)?;stream.close()?;assert!(context.allocation_stats()?.is_zero());
        println!("ROLLING_GPU mode={mode} rolls={rolls} lengths={:?} allocation_zero=true",tokens.iter().map(Vec::len).collect::<Vec<_>>());
    }
    context.close()?;Ok(())
}
