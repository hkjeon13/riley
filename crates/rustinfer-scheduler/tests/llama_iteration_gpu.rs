//! Remote-only end-to-end gate for scheduler plan -> CUDA batch -> scheduler commit.

#![cfg(feature = "cuda")]

use std::convert::Infallible;
use std::error::Error;
use std::path::PathBuf;

use rustinfer_model::{LoadLimits, LoadedModel};
use rustinfer_runtime::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig,
    PreparedLlamaForwardConfig,
};
use rustinfer_runtime::paged_kv::KvLayout;
use rustinfer_runtime::sampling::{
    SamplingParams, SamplingRng, SamplingWorkspace, TokenConstraints,
};
use rustinfer_runtime::{CudaContext, CudaRuntime, CudaStream};
use rustinfer_scheduler::{
    IterationTiming, OverloadPolicy, RequestDescriptor, RequestFinishReason, RequestState,
    SampledIterationToken, Scheduler, SchedulerConfig, execute_llama_iteration,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1 << 30;
const PHYSICAL_BLOCKS: usize = 8;
const PROMPT: [u32; 3] = [504, 2_365, 6_354];

struct NoRng;

impl SamplingRng for NoRng {
    type Error = Infallible;

    fn next_u32(&mut self) -> Result<u32, Self::Error> {
        unreachable!("temperature-zero sampling must not consume RNG")
    }
}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((context, stream))
}

fn scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: 2,
        max_waiting_prompt_tokens: 32,
        max_active_sequences: 1,
        max_sequence_tokens: 32,
        iteration_token_budget: 8,
        max_prefill_chunk_tokens: 8,
        aging_threshold_ns: 10,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(100),
        max_promised_kv_blocks: PHYSICAL_BLOCKS,
        metrics_window_samples: 8,
    }
}

fn greedy_sample(
    downloaded: &rustinfer_scheduler::DownloadedLlamaIteration,
    workspace: &mut SamplingWorkspace,
) -> TestResult<SampledIterationToken> {
    let logits = downloaded.logits_for_slot(rustinfer_scheduler::OutputSlot::new(0))?;
    let params = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let distribution =
        workspace.process_bf16_native(logits, TokenConstraints::AllowAll, &[], params)?;
    let token = distribution
        .sample(&mut NoRng)
        .expect("greedy sampling is infallible and consumes no RNG")
        .token_id();
    Ok(SampledIterationToken::new(token, false))
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn scheduler_plan_executes_and_commits_prefill_then_decode() -> TestResult {
    let limits = LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?;
    let model = LoadedModel::load(&checkpoint_path(), limits)?;
    let attention = model
        .spec()
        .blocks()
        .first()
        .expect("validated model has at least one decoder block")
        .attention();
    let layout = KvLayout::checked(
        model.spec().blocks().len(),
        PHYSICAL_BLOCKS,
        attention.key_value_heads(),
        attention.head_dimension(),
    )?;
    let mut scheduler = Scheduler::new(scheduler_config(), layout)?;
    let submission = scheduler.submit(RequestDescriptor::new(PROMPT.to_vec(), 2), 0)?;
    assert_eq!(submission.state(), RequestState::Admitted);

    let (context, mut stream) = first_context()?;
    let executor_config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 8, 8, 1, PHYSICAL_BLOCKS)?,
        PreparedLlamaForwardConfig::default(),
    );
    let mut executor =
        PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, executor_config)?;
    let mut sampler = SamplingWorkspace::new(executor.vocabulary_size())?;

    let first_planning = scheduler.plan_iteration(1)?;
    assert!(first_planning.completions().is_empty());
    let first_plan = first_planning.into_parts().0.expect("prefill plan");
    assert_eq!(first_plan.prefill_items().len(), 1);
    assert!(first_plan.decode_items().is_empty());
    assert_eq!(first_plan.prefill_items()[0].input_tokens(), PROMPT);
    let first_download = execute_llama_iteration(&first_plan, &mut executor, &mut stream)?;
    let first_sample = greedy_sample(&first_download, &mut sampler)?;
    let first_result = first_download
        .into_result(&[first_sample], IterationTiming::default())
        .map_err(|failure| -> Box<dyn Error> { Box::new(failure) })?;
    let first_updates = scheduler.complete_iteration(&first_result, 2)?;
    assert_eq!(first_updates.token_events().len(), 1);
    assert!(first_updates.completions().is_empty());
    assert!(first_updates.settlement_failures().is_empty());
    assert_eq!(
        scheduler.request_state(submission.request_id()),
        Some(RequestState::Decoding)
    );

    let second_planning = scheduler.plan_iteration(3)?;
    assert!(second_planning.completions().is_empty());
    let second_plan = second_planning.into_parts().0.expect("decode plan");
    assert!(second_plan.prefill_items().is_empty());
    assert_eq!(second_plan.decode_items().len(), 1);
    assert_eq!(
        second_plan.decode_items()[0].input_tokens(),
        &[first_sample.token_id()]
    );
    let second_download = execute_llama_iteration(&second_plan, &mut executor, &mut stream)?;
    let second_sample = greedy_sample(&second_download, &mut sampler)?;
    let second_result = second_download
        .into_result(&[second_sample], IterationTiming::default())
        .map_err(|failure| -> Box<dyn Error> { Box::new(failure) })?;
    let second_updates = scheduler.complete_iteration(&second_result, 4)?;
    assert_eq!(second_updates.token_events().len(), 1);
    assert!(second_updates.settlement_failures().is_empty());
    assert_eq!(second_updates.completions().len(), 1);
    let completion = &second_updates.completions()[0];
    assert_eq!(completion.request_id(), submission.request_id());
    assert_eq!(completion.reason(), RequestFinishReason::Length);
    assert_eq!(
        completion.generated_token_ids(),
        &[first_sample.token_id(), second_sample.token_id()]
    );
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);

    let closed = scheduler
        .close(5, None)
        .map_err(|failure| -> Box<dyn Error> { Box::new(failure) })?;
    assert!(closed.completions().is_empty());
    assert!(closed.settlement_failures().is_empty());
    executor.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
