//! Remote-only end-to-end gate for scheduler plan -> CUDA batch -> scheduler commit.

#![cfg(feature = "cuda")]

use std::convert::Infallible;
use std::error::Error;
use std::path::PathBuf;

use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig,
    PreparedLlamaForwardConfig,
};
use riley_runtime::paged_kv::KvLayout;
use riley_runtime::sampling::{SamplingParams, SamplingRng, SamplingWorkspace, TokenConstraints};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
use riley_scheduler::{
    IterationPlan, IterationTiming, OutputSlot, OverloadPolicy, RequestDescriptor,
    RequestFinishReason, RequestId, RequestState, SampledIterationToken, Scheduler,
    SchedulerConfig, execute_llama_iteration, execute_llama_iteration_greedy,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1 << 30;
const PROMPT: [u32; 3] = [504, 2_365, 6_354];
const BF16_BYTES: usize = 2;
const GREEDY_RESULT_BYTES: usize = 8;

#[derive(Default)]
struct CountingRng {
    draws: usize,
}

impl SamplingRng for CountingRng {
    type Error = Infallible;

    fn next_u32(&mut self) -> Result<u32, Self::Error> {
        self.draws += 1;
        Ok(0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputBackend {
    FallbackLogits,
    GpuGreedy,
}

#[derive(Debug, Eq, PartialEq)]
struct CompletionTrace {
    request_id: u64,
    reason: RequestFinishReason,
    generated_token_ids: Vec<u32>,
}

#[derive(Debug, Eq, PartialEq)]
struct ScenarioTrace {
    slot_routes: Vec<Vec<(u32, u64)>>,
    token_ids: Vec<Vec<u32>>,
    full_logit_bytes: Vec<usize>,
    greedy_record_bytes: Vec<usize>,
    completions: Vec<CompletionTrace>,
    rng_draws: usize,
    stop_exercised: bool,
    cancellation_before_commit_exercised: bool,
}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RILEY_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((context, stream))
}

fn scheduler_config(concurrency: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: concurrency,
        max_waiting_prompt_tokens: PROMPT.len() * concurrency,
        max_active_sequences: concurrency,
        max_sequence_tokens: 16,
        iteration_token_budget: PROMPT.len() * concurrency,
        max_prefill_chunk_tokens: PROMPT.len(),
        aging_threshold_ns: 10,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: concurrency,
        metrics_window_samples: 8,
    }
}

fn cpu_greedy_tokens(
    downloaded: &riley_scheduler::DownloadedLlamaIteration,
    workspace: &mut SamplingWorkspace,
    rng: &mut CountingRng,
) -> TestResult<Vec<u32>> {
    let mut tokens = Vec::with_capacity(downloaded.output_count());
    for raw_slot in 0..u32::try_from(downloaded.output_count())? {
        let logits = downloaded.logits_for_slot(OutputSlot::new(raw_slot))?;
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let distribution =
            workspace.process_bf16_native(logits, TokenConstraints::AllowAll, &[], params)?;
        tokens.push(
            distribution
                .sample(rng)
                .expect("temperature-zero sampling is infallible")
                .token_id(),
        );
    }
    Ok(tokens)
}

fn prompt_for(request_index: usize) -> TestResult<Vec<u32>> {
    let mut prompt = PROMPT.to_vec();
    prompt[2] = prompt[2]
        .checked_add(u32::try_from(request_index)?)
        .ok_or("test prompt token overflow")?;
    Ok(prompt)
}

fn dense_slot_routes(plan: &IterationPlan) -> Vec<(u32, u64)> {
    let mut routes = plan
        .prefill_items()
        .iter()
        .chain(plan.decode_items())
        .filter_map(|item| {
            item.output_slot()
                .map(|slot| (slot.get(), item.request_id().get()))
        })
        .collect::<Vec<_>>();
    routes.sort_unstable_by_key(|(slot, _)| *slot);
    for (index, (slot, _)) in routes.iter().enumerate() {
        assert_eq!(usize::try_from(*slot).ok(), Some(index));
    }
    routes
}

fn record_completions(
    destination: &mut Vec<CompletionTrace>,
    completions: &[riley_scheduler::RequestCompletion],
) {
    destination.extend(completions.iter().map(|completion| CompletionTrace {
        request_id: completion.request_id().get(),
        reason: completion.reason(),
        generated_token_ids: completion.generated_token_ids().to_vec(),
    }));
}

#[allow(clippy::too_many_lines)]
fn run_scenario(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    concurrency: usize,
    backend: OutputBackend,
) -> TestResult<ScenarioTrace> {
    let attention = model
        .spec()
        .blocks()
        .first()
        .expect("validated model has at least one decoder block")
        .attention();
    let layout = KvLayout::checked(
        model.spec().blocks().len(),
        concurrency,
        attention.key_value_heads(),
        attention.head_dimension(),
    )?;
    let mut scheduler = Scheduler::new(scheduler_config(concurrency), layout)?;
    let mut submissions = Vec::with_capacity(concurrency);
    for request_index in 0..concurrency {
        let submission =
            scheduler.submit(RequestDescriptor::new(prompt_for(request_index)?, 2), 0)?;
        assert_eq!(submission.state(), RequestState::Admitted);
        submissions.push(submission);
    }

    let max_input_tokens = PROMPT
        .len()
        .checked_mul(concurrency)
        .ok_or("test batch token capacity overflow")?;
    let executor_config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(
            concurrency,
            max_input_tokens,
            concurrency,
            concurrency,
            concurrency,
        )?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_iteration_batch_completion()
    .with_active_row_buckets();
    let mut executor =
        PreparedLlamaBatchExecutor::prepare(model, context, stream, executor_config)?;
    let stable = context.allocation_stats()?;
    let mut workspace = match backend {
        OutputBackend::FallbackLogits => Some(SamplingWorkspace::new(executor.vocabulary_size())?),
        OutputBackend::GpuGreedy => None,
    };
    let mut rng = CountingRng::default();
    let mut trace = ScenarioTrace {
        slot_routes: Vec::new(),
        token_ids: Vec::new(),
        full_logit_bytes: Vec::new(),
        greedy_record_bytes: Vec::new(),
        completions: Vec::new(),
        rng_draws: 0,
        stop_exercised: false,
        cancellation_before_commit_exercised: false,
    };

    while trace.completions.len() < concurrency {
        let iteration_index = trace.token_ids.len();
        assert!(
            iteration_index < 3,
            "scenario did not converge in two iterations"
        );
        let planning_now = 10 + u64::try_from(iteration_index)? * 10;
        let planning = scheduler.plan_iteration(planning_now)?;
        assert!(planning.completions().is_empty());
        let plan = planning
            .into_parts()
            .0
            .expect("active requests produce a plan");
        let routes = dense_slot_routes(&plan);
        assert!(!routes.is_empty());

        let downloaded = match backend {
            OutputBackend::FallbackLogits => execute_llama_iteration(&plan, &mut executor, stream)?,
            OutputBackend::GpuGreedy => {
                execute_llama_iteration_greedy(&plan, &mut executor, stream)?
            }
        };
        assert_eq!(downloaded.iteration_id(), plan.iteration_id());
        assert_eq!(downloaded.output_count(), routes.len());
        assert_eq!(context.allocation_stats()?, stable);

        let (tokens, full_logit_bytes, greedy_record_bytes) = match backend {
            OutputBackend::FallbackLogits => {
                assert!(downloaded.greedy_token_ids().is_empty());
                assert_eq!(
                    downloaded.logits_bf16_native().len(),
                    routes.len() * executor.vocabulary_size() * BF16_BYTES
                );
                assert!(
                    downloaded
                        .greedy_token_for_slot(OutputSlot::new(0))
                        .is_err()
                );
                let tokens = cpu_greedy_tokens(
                    &downloaded,
                    workspace
                        .as_mut()
                        .expect("fallback path owns a CPU sampling workspace"),
                    &mut rng,
                )?;
                (tokens, downloaded.logits_bf16_native().len(), 0)
            }
            OutputBackend::GpuGreedy => {
                assert!(workspace.is_none());
                assert!(downloaded.logits_bf16_native().is_empty());
                assert_eq!(downloaded.greedy_token_ids().len(), routes.len());
                assert!(downloaded.logits_for_slot(OutputSlot::new(0)).is_err());
                let mut tokens = Vec::with_capacity(routes.len());
                for raw_slot in 0..u32::try_from(routes.len())? {
                    tokens.push(downloaded.greedy_token_for_slot(OutputSlot::new(raw_slot))?);
                }
                assert!(
                    downloaded
                        .greedy_token_for_slot(OutputSlot::new(u32::try_from(routes.len())?))
                        .is_err()
                );
                let record_bytes = executor.greedy_result_byte_len_for(routes.len())?;
                assert_eq!(record_bytes, routes.len() * GREEDY_RESULT_BYTES);
                (tokens, 0, record_bytes)
            }
        };
        assert_eq!(rng.draws, 0, "greedy decoding consumed RNG");

        let stop_first = concurrency == 8 && iteration_index == 0;
        let samples = tokens
            .iter()
            .enumerate()
            .map(|(slot, &token)| SampledIterationToken::new(token, stop_first && slot == 0))
            .collect::<Vec<_>>();
        trace.stop_exercised |= stop_first;

        let cancel_request = if concurrency == 8 && iteration_index == 0 {
            assert_eq!(routes.len(), 8);
            assert_eq!(routes[1].1, submissions[1].request_id().get());
            let request_id = RequestId::new(routes[1].1).expect("scheduler IDs are nonzero");
            let outcome = scheduler.cancel(request_id, planning_now + 1)?;
            assert_eq!(outcome.request_id(), request_id);
            assert!(outcome.deferred_until_iteration_settles());
            assert!(!outcome.already_terminal());
            assert!(outcome.completion().is_none());
            trace.cancellation_before_commit_exercised = true;
            Some(request_id)
        } else {
            None
        };

        let result = downloaded
            .into_result(&samples, IterationTiming::default())
            .map_err(|failure| -> Box<dyn Error> { Box::new(failure) })?;
        let updates = scheduler.complete_iteration(&result, planning_now + 2)?;
        assert!(updates.settlement_failures().is_empty());
        assert_eq!(
            updates.token_events().len(),
            routes.len() - usize::from(cancel_request.is_some())
        );
        record_completions(&mut trace.completions, updates.completions());
        trace.slot_routes.push(routes);
        trace.token_ids.push(tokens);
        trace.full_logit_bytes.push(full_logit_bytes);
        trace.greedy_record_bytes.push(greedy_record_bytes);
    }

    trace.completions.sort_unstable_by_key(|row| row.request_id);
    trace.rng_draws = rng.draws;
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let closed = scheduler
        .close(100, None)
        .map_err(|failure| -> Box<dyn Error> { Box::new(failure) })?;
    assert!(closed.completions().is_empty());
    assert!(closed.settlement_failures().is_empty());
    executor.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(trace)
}

fn assert_cpu_gpu_semantic_parity(cpu: &ScenarioTrace, gpu: &ScenarioTrace) {
    assert_eq!(gpu.slot_routes, cpu.slot_routes);
    assert_eq!(gpu.token_ids, cpu.token_ids);
    assert_eq!(gpu.completions, cpu.completions);
    assert_eq!(cpu.rng_draws, 0);
    assert_eq!(gpu.rng_draws, 0);
    assert!(cpu.full_logit_bytes.iter().all(|&bytes| bytes > 0));
    assert!(cpu.greedy_record_bytes.iter().all(|&bytes| bytes == 0));
    assert!(gpu.full_logit_bytes.iter().all(|&bytes| bytes == 0));
    assert_eq!(gpu.greedy_record_bytes.len(), gpu.token_ids.len());
    for (bytes, tokens) in gpu.greedy_record_bytes.iter().zip(&gpu.token_ids) {
        assert_eq!(*bytes, tokens.len() * GREEDY_RESULT_BYTES);
    }
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
#[allow(clippy::similar_names)]
fn scheduler_plan_executes_and_commits_prefill_then_decode() -> TestResult {
    let limits = LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?;
    let model = LoadedModel::load(&checkpoint_path(), limits)?;
    let (context, mut stream) = first_context()?;
    let c1_cpu = run_scenario(
        &model,
        &context,
        &mut stream,
        1,
        OutputBackend::FallbackLogits,
    )?;
    let c1_gpu = run_scenario(&model, &context, &mut stream, 1, OutputBackend::GpuGreedy)?;
    assert_cpu_gpu_semantic_parity(&c1_cpu, &c1_gpu);
    assert_eq!(c1_gpu.slot_routes.len(), 2);
    assert!(
        c1_gpu
            .slot_routes
            .iter()
            .all(|routes| routes.len() == 1 && routes[0].0 == 0)
    );
    assert_eq!(c1_gpu.completions.len(), 1);
    assert_eq!(c1_gpu.completions[0].reason, RequestFinishReason::Length);
    assert!(!c1_gpu.stop_exercised);
    assert!(!c1_gpu.cancellation_before_commit_exercised);

    let c8_cpu = run_scenario(
        &model,
        &context,
        &mut stream,
        8,
        OutputBackend::FallbackLogits,
    )?;
    let c8_gpu = run_scenario(&model, &context, &mut stream, 8, OutputBackend::GpuGreedy)?;
    assert_cpu_gpu_semantic_parity(&c8_cpu, &c8_gpu);
    assert_eq!(
        c8_gpu.slot_routes.iter().map(Vec::len).collect::<Vec<_>>(),
        [8, 6]
    );
    assert!(c8_gpu.stop_exercised);
    assert!(c8_gpu.cancellation_before_commit_exercised);
    assert_eq!(
        c8_gpu
            .completions
            .iter()
            .filter(|row| row.reason == RequestFinishReason::Stop)
            .count(),
        1
    );
    assert_eq!(
        c8_gpu
            .completions
            .iter()
            .filter(|row| row.reason == RequestFinishReason::Cancelled)
            .count(),
        1
    );
    assert_eq!(
        c8_gpu
            .completions
            .iter()
            .filter(|row| row.reason == RequestFinishReason::Length)
            .count(),
        6
    );

    println!(
        "pr16-gpu-greedy-scheduler-integration schema_version=1 \
         c1_iterations=2 c8_iterations=2 dense_slot_mismatches=0 \
         token_id_mismatches=0 rng_draws=0 fallback_logits=true \
         stop=true cancellation_before_commit=true bytes_per_output=8 status=passed"
    );
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
