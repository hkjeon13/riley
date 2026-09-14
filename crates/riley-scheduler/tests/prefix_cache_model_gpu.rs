#![cfg(feature="cuda")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadedModel,LoadLimits};
use riley_runtime::llama::{PreparedLlamaBatchExecutor,PreparedLlamaBatchExecutorConfig,PreparedLlamaForwardConfig,LlamaBatchMetadataConfig};
use riley_scheduler::{Scheduler,SchedulerConfig,RequestDescriptor,OverloadPolicy,SampledIterationToken,IterationTiming,ExecutionShapePolicy};
type TestResult<T> = Result<T,Box<dyn std::error::Error>>;

fn run(cached:bool,prompts:&[Vec<u32>],attention:u8)->TestResult<Vec<Vec<u8>>> {
    let model=LoadedModel::load(std::path::Path::new(&std::env::var("RILEY_REAL_CHECKPOINT")?),LoadLimits::default().with_weight_byte_limits(1<<30,1<<30)?)?;
    let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;
    let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,8,1,64)?,PreparedLlamaForwardConfig::default());
    let executor=PreparedLlamaBatchExecutor::prepare(&model,&context,&mut stream,config)?;
    let mut session=if attention==4 {executor.into_owned_variable_ffn_adaptive_session(&context,512,false,false,cached)?}else if attention==3 {executor.into_owned_variable_projection_pipeline_session(&context,512,false,false,cached)?}else if attention==2 {executor.into_owned_variable_gqa_staging_session(&context,512,false,false,cached)?}else if attention==1 {executor.into_owned_variable_query_reuse_session(&context,512,false,false,cached)?}else if cached {executor.into_owned_variable_prefix_session(&context,512,false,false,false,false)?}
        else {executor.into_owned_variable_mixed_session(&context,512,false)?};
    let mut scheduler=Scheduler::new_with_execution_shape(SchedulerConfig{
        max_waiting_requests:4,max_waiting_prompt_tokens:512,max_active_sequences:2,max_sequence_tokens:1024,
        iteration_token_budget:512,max_prefill_chunk_tokens:512,aging_threshold_ns:1,overload_policy:OverloadPolicy::Wait,
        admission_timeout_ns:None,max_promised_kv_blocks:64,metrics_window_samples:8,
    },riley_runtime::paged_kv::KvLayout::checked(30,64,3,64)?,ExecutionShapePolicy::MixedPrefillDecode32)?;
    if cached {scheduler.enable_prefix_cache(session.prefix_cache_identity()?,4,8)?;}
    let mut output=Vec::new();let mut now=0;let mut prefill_tokens=0;
    for prompt in prompts {
        scheduler.submit(RequestDescriptor::new(prompt.clone(),3),now)?;
        let mut request=Vec::new();
        loop {
            now+=1;
            let Some(plan)=scheduler.plan_iteration(now)?.into_parts().0 else{break};
            prefill_tokens+=plan.prefill_items().iter().map(|w|w.input_tokens().len()).sum::<usize>();
            let authority=scheduler.authorize_execution(&plan)?;
            let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,false)?;
            request.extend_from_slice(downloaded.logits_bf16_native());
            // Identical teacher-forced inputs isolate KV reuse from sampling.
            let samples=vec![SampledIterationToken::new(7,false);downloaded.output_count()];
            drop(authority);let result=downloaded.into_result(&samples,IterationTiming::new(0,0))?;
            now+=1;assert!(scheduler.complete_iteration(&result,now)?.settlement_failures().is_empty());
            session.confirm_scheduler_commit(plan.iteration_id().get())?;
        }
        assert_eq!(request.len(),3*49152*2);output.push(request);
    }
    let stats=scheduler.prefix_cache_stats();
    if cached {assert!(stats.2>=3);assert!(stats.3>=80);}else{assert_eq!(stats,(0,0,0,0));}
    println!("automatic-prefix-cache enabled={cached} prefill_tokens={prefill_tokens} entries={} pages={} hits={} reused_tokens={}",stats.0,stats.1,stats.2,stats.3);
    session.close()?;scheduler.close(now+1,None)?;stream.close()?;
    assert!(context.allocation_stats()?.is_zero());context.close()?;Ok(output)
}

#[test]
#[ignore="requires CUDA13 SM89 and real SmolLM2 checkpoint"]
fn automatic_cache_matches_uncached_model_logits()->TestResult<()> {
    let mut suffix=vec![17;33];suffix[32]=18;
    let mut shorter=vec![17;25];shorter[20]=19;
    let prompts=vec![vec![17;33],vec![17;33],suffix,shorter,vec![18;33]];
    let cold=run(false,&prompts,0)?;let cached=run(true,&prompts,0)?;
    for (index,(a,b)) in cold.iter().zip(&cached).enumerate(){assert_eq!(a,b,"cached request {index} changed BF16 logits");}
    println!("automatic-prefix-cache exact_logits_bytes={} requests={} outputs_per_request=3",cold.iter().map(Vec::len).sum::<usize>(),prompts.len());Ok(())
}

#[test]
#[ignore="requires CUDA13 SM89 and real SmolLM2 checkpoint"]
fn query_reuse_matches_full_model_logits()->TestResult<()> {
    let mut suffix=vec![17;398];suffix[397]=18;
    let prompts=vec![vec![17;398],suffix,vec![19;47],vec![20;512]];
    let cold=run(false,&prompts,0)?;
    let candidate=run(false,&prompts,1)?;
    for (a,b) in cold.iter().zip(&candidate){assert_eq!(a,b,"query reuse changed full model logits");}
    // Repeated long prefixes cover cache-only owners and shorter suffix prefill.
    let shared=vec![vec![17;128];4];
    let baseline=run(true,&shared,0)?;let reuse=run(true,&shared,1)?;
    for (a,b) in baseline.iter().zip(&reuse){assert_eq!(a,b,"cached query reuse changed full model logits");}
    println!("query-reuse exact_logits_bytes={}",cold.iter().chain(&baseline).map(Vec::len).sum::<usize>());Ok(())
}

#[test]
#[ignore="requires CUDA13 SM89 and real SmolLM2 checkpoint"]
fn gqa_staging_matches_full_model_logits()->TestResult<()> {
    let mut suffix=vec![17;398];suffix[397]=18;
    let prompts=vec![vec![17;398],suffix,vec![19;47],vec![20;512]];
    let cold=run(false,&prompts,0)?;
    let candidate=run(false,&prompts,2)?;
    for (a,b) in cold.iter().zip(&candidate){assert_eq!(a,b,"GQA staging changed full model logits");}
    // Repeated long prefixes cover cache-only owners and shorter suffix prefill.
    let shared=vec![vec![17;128];4];
    let baseline=run(true,&shared,0)?;let reuse=run(true,&shared,2)?;
    for (a,b) in baseline.iter().zip(&reuse){assert_eq!(a,b,"cached GQA staging changed full model logits");}
    println!("gqa-staging exact_logits_bytes={}",cold.iter().chain(&baseline).map(Vec::len).sum::<usize>());Ok(())
}

#[test]
#[ignore="requires CUDA13 SM89 and real SmolLM2 checkpoint"]
fn projection_pipeline_matches_full_model_logits()->TestResult<()> {
    let mut suffix=vec![17;398];suffix[397]=18;
    let prompts=vec![vec![17;398],suffix,vec![19;47],vec![20;512]];
    let cold=run(false,&prompts,0)?;
    let candidate=run(false,&prompts,3)?;
    for (a,b) in cold.iter().zip(&candidate){assert_eq!(a,b,"projection pipeline changed full model logits");}
    // Repeated long prefixes cover cache-only owners and shorter suffix prefill.
    let shared=vec![vec![17;128];4];
    let baseline=run(true,&shared,0)?;let reuse=run(true,&shared,3)?;
    for (a,b) in baseline.iter().zip(&reuse){assert_eq!(a,b,"cached projection pipeline changed full model logits");}
    println!("projection-pipeline exact_logits_bytes={}",cold.iter().chain(&baseline).map(Vec::len).sum::<usize>());Ok(())
}

#[test]
#[ignore="requires CUDA13 SM89 and real SmolLM2 checkpoint"]
fn ffn_adaptive_matches_full_model_logits()->TestResult<()> {
    let mut suffix=vec![17;398];suffix[397]=18;
    let prompts=vec![vec![17;398],suffix,vec![19;47],vec![20;512],vec![21;191],vec![22;192],vec![23;193]];
    let cold=run(false,&prompts,3)?;
    let candidate=run(false,&prompts,4)?;
    for (a,b) in cold.iter().zip(&candidate){assert_eq!(a,b,"adaptive FFN changed full model logits");}
    // Repeated long prefixes cover cache-only owners and shorter suffix prefill.
    let shared=vec![vec![17;128];4];
    let baseline=run(true,&shared,3)?;let reuse=run(true,&shared,4)?;
    for (a,b) in baseline.iter().zip(&reuse){assert_eq!(a,b,"cached adaptive FFN changed full model logits");}
    println!("ffn-adaptive exact_logits_bytes={}",cold.iter().chain(&baseline).map(Vec::len).sum::<usize>());Ok(())
}
