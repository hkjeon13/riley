//! Candidate-bound CUDA replay target for the fixed C03-B routing corpus.
//!
//! This target is intentionally ignored: source inclusion before a candidate
//! freeze is allowed, while GPU execution and formal C03-B acceptance require
//! the separate C02 qualification and operational authorization boundary.

#![cfg(feature = "cuda")]

#[path = "support/gpu_fixed_corpus.rs"]
#[allow(dead_code)]
mod gpu_fixed_corpus;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::error::Error;
use std::path::PathBuf;

use gpu_fixed_corpus::{
    GpuFixedCorpusCase, GpuFixedPhase, GpuFixedSettlement, GpuFixedTerminalReason,
    GpuFixedWorkKind, gpu_fixed_corpus,
};
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig,
    PreparedLlamaForwardConfig,
};
use riley_runtime::paged_kv::KvLayout;
use riley_runtime::sampling::{SamplingParams, SamplingRng, SamplingWorkspace, TokenConstraints};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
use riley_scheduler::{
    DownloadedLlamaIteration, ExecutionAbort, IterationAdapterError, IterationPlan,
    IterationTiming, OutputSlot, OverloadPolicy, RequestCompletion, RequestDescriptor,
    RequestFinishReason, RequestId, RequestState, SampledIterationToken, Scheduler,
    SchedulerCloseOutput, SchedulerConfig, WorkItem, WorkKind, execute_llama_iteration,
    execute_llama_iteration_greedy,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1 << 30;
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
    label: u8,
    reason: RequestFinishReason,
    generated_token_ids: Vec<u32>,
}

#[derive(Debug, Eq, PartialEq)]
struct CaseTrace {
    phases: Vec<(GpuFixedPhase, Vec<(u8, usize, usize)>, Vec<u32>)>,
    events: Vec<(u8, usize, u32)>,
    completions: Vec<CompletionTrace>,
    rng_draws: usize,
}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RILEY_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_REAL_CHECKPOINT must name the candidate-bound checkpoint directory")
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((context, stream))
}

fn scheduler_config(case: &GpuFixedCorpusCase) -> SchedulerConfig {
    let maximum_prompt_tokens = case.decoder_prompt_token_count() * case.primed_labels().len()
        + case.final_prefill_prompt_token_count() * case.final_prefill_labels().len();
    SchedulerConfig {
        max_waiting_requests: case.concurrency(),
        max_waiting_prompt_tokens: maximum_prompt_tokens,
        max_active_sequences: case.concurrency(),
        max_sequence_tokens: case.maximum_sequence_tokens(),
        iteration_token_budget: case.maximum_iteration_input_tokens(),
        max_prefill_chunk_tokens: case.decoder_prompt_token_count(),
        aging_threshold_ns: u64::MAX,
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: None,
        max_promised_kv_blocks: case.promised_kv_blocks(),
        metrics_window_samples: 8,
    }
}

fn new_scheduler(case: &GpuFixedCorpusCase) -> Scheduler {
    let layout =
        KvLayout::checked(1, case.promised_kv_blocks(), 1, 8).expect("fixed C03-B GPU KV layout");
    Scheduler::new(scheduler_config(case), layout).expect("fixed C03-B GPU scheduler configuration")
}

fn prompt_for(label: u8, token_count: usize) -> Vec<u32> {
    let first = 512_u32 + u32::from(label) * 32;
    (0..token_count)
        .map(|index| first + u32::try_from(index).expect("small fixed prompt index"))
        .collect()
}

fn submit_labels(
    scheduler: &mut Scheduler,
    labels: &[u8],
    prompt_token_count: usize,
    max_new_tokens: usize,
    now_ns: u64,
    request_ids: &mut BTreeMap<u8, RequestId>,
) {
    for &label in labels {
        let submission = scheduler
            .submit(
                RequestDescriptor::new(prompt_for(label, prompt_token_count), max_new_tokens),
                now_ns,
            )
            .expect("fixed C03-B GPU submission");
        assert_eq!(submission.state(), RequestState::Admitted);
        assert!(request_ids.insert(label, submission.request_id()).is_none());
    }
}

fn plan_now(scheduler: &mut Scheduler, now_ns: u64) -> IterationPlan {
    let planning = scheduler
        .plan_iteration(now_ns)
        .expect("fixed C03-B GPU plan");
    let (plan, completions) = planning.into_parts();
    assert!(completions.is_empty());
    plan.expect("fixed C03-B GPU fixture has planned live work")
}

fn assert_work_item(
    phase: GpuFixedPhase,
    plan: &IterationPlan,
    item: &WorkItem,
    expected: gpu_fixed_corpus::GpuFixedRoute,
    request_ids: &BTreeMap<u8, RequestId>,
    previous_tokens: &BTreeMap<(u8, usize), u32>,
) {
    let expected_kind = match expected.work_kind {
        GpuFixedWorkKind::Prefill => WorkKind::Prefill,
        GpuFixedWorkKind::Decode => WorkKind::Decode,
    };
    assert_eq!(
        item.kind(),
        expected_kind,
        "{phase:?} label {}",
        expected.label
    );
    assert_eq!(
        item.request_id(),
        *request_ids
            .get(&expected.label)
            .expect("descriptor label was submitted"),
    );
    let expected_input = match expected.work_kind {
        GpuFixedWorkKind::Prefill => prompt_for(expected.label, expected.input_token_count),
        GpuFixedWorkKind::Decode => vec![
            *previous_tokens
                .get(&(
                    expected.label,
                    expected
                        .generation_step
                        .checked_sub(1)
                        .expect("decode follows a sampled output"),
                ))
                .expect("prior phase recorded the decoder token"),
        ],
    };
    assert_eq!(item.input_tokens(), expected_input.as_slice());
    assert_eq!(item.target_logical_length(), expected.target_logical_length);
    assert_eq!(
        item.output_slot(),
        Some(OutputSlot::new(
            u32::try_from(expected.output_slot).expect("fixed output slot fits u32"),
        )),
    );
    let table = plan
        .block_tables()
        .get(item.block_table_index())
        .expect("work item has a copied block table");
    assert_eq!(table.request_id(), item.request_id());
    assert_eq!(
        usize::try_from(table.logical_length()).expect("fixed logical length"),
        expected.target_logical_length,
    );
    assert_eq!(
        table.valid_tokens(),
        GpuFixedCorpusCase::valid_tokens_for(expected.target_logical_length).as_slice(),
    );
}

fn assert_phase_plan(
    case: &GpuFixedCorpusCase,
    phase: GpuFixedPhase,
    plan: &IterationPlan,
    request_ids: &BTreeMap<u8, RequestId>,
    previous_tokens: &BTreeMap<(u8, usize), u32>,
) {
    let routes = case.routes_for_phase(phase);
    let decode_routes = routes
        .iter()
        .copied()
        .filter(|route| route.work_kind == GpuFixedWorkKind::Decode)
        .collect::<Vec<_>>();
    let prefill_routes = routes
        .iter()
        .copied()
        .filter(|route| route.work_kind == GpuFixedWorkKind::Prefill)
        .collect::<Vec<_>>();
    assert_eq!(plan.decode_items().len(), decode_routes.len());
    assert_eq!(plan.prefill_items().len(), prefill_routes.len());
    assert_eq!(plan.batch_size(), routes.len());
    assert_eq!(
        plan.total_tokens(),
        routes
            .iter()
            .map(|route| route.input_token_count)
            .sum::<usize>()
    );
    assert_eq!(
        plan.output_slots(),
        routes
            .iter()
            .map(|route| OutputSlot::new(u32::try_from(route.output_slot).expect("slot")))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    for (item, expected) in plan.decode_items().iter().zip(&decode_routes) {
        assert_work_item(phase, plan, item, *expected, request_ids, previous_tokens);
    }
    for (item, expected) in plan.prefill_items().iter().zip(&prefill_routes) {
        assert_work_item(phase, plan, item, *expected, request_ids, previous_tokens);
    }
}

fn prepare_executor(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    case: &GpuFixedCorpusCase,
) -> TestResult<PreparedLlamaBatchExecutor> {
    let metadata = LlamaBatchMetadataConfig::new(
        case.concurrency(),
        case.maximum_iteration_input_tokens(),
        case.maximum_plan_block_entries(),
        case.concurrency(),
        case.promised_kv_blocks(),
    )?;
    let config =
        PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
            .with_iteration_batch_completion()
            .with_active_row_buckets();
    Ok(PreparedLlamaBatchExecutor::prepare(
        model, context, stream, config,
    )?)
}

fn cpu_greedy_tokens(
    downloaded: &DownloadedLlamaIteration,
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

fn execute_download(
    plan: &IterationPlan,
    executor: &mut PreparedLlamaBatchExecutor,
    stream: &mut CudaStream,
    backend: OutputBackend,
) -> TestResult<DownloadedLlamaIteration> {
    Ok(match backend {
        OutputBackend::FallbackLogits => execute_llama_iteration(plan, executor, stream)?,
        OutputBackend::GpuGreedy => execute_llama_iteration_greedy(plan, executor, stream)?,
    })
}

fn downloaded_tokens(
    downloaded: &DownloadedLlamaIteration,
    executor: &PreparedLlamaBatchExecutor,
    backend: OutputBackend,
    workspace: &mut Option<SamplingWorkspace>,
    rng: &mut CountingRng,
) -> TestResult<Vec<u32>> {
    match backend {
        OutputBackend::FallbackLogits => {
            assert!(downloaded.greedy_token_ids().is_empty());
            assert_eq!(
                downloaded.logits_bf16_native().len(),
                downloaded.output_count() * executor.vocabulary_size() * BF16_BYTES,
            );
            cpu_greedy_tokens(
                downloaded,
                workspace
                    .as_mut()
                    .expect("fallback backend owns a CPU sampling workspace"),
                rng,
            )
        }
        OutputBackend::GpuGreedy => {
            assert!(workspace.is_none());
            assert!(downloaded.logits_bf16_native().is_empty());
            assert_eq!(
                downloaded.greedy_token_ids().len(),
                downloaded.output_count()
            );
            let mut tokens = Vec::with_capacity(downloaded.output_count());
            for raw_slot in 0..u32::try_from(downloaded.output_count())? {
                tokens.push(downloaded.greedy_token_for_slot(OutputSlot::new(raw_slot))?);
            }
            assert!(
                downloaded
                    .greedy_token_for_slot(OutputSlot::new(u32::try_from(
                        downloaded.output_count()
                    )?))
                    .is_err()
            );
            assert_eq!(
                executor.greedy_result_byte_len_for(downloaded.output_count())?,
                downloaded.output_count() * GREEDY_RESULT_BYTES,
            );
            Ok(tokens)
        }
    }
}

fn label_for(request_ids: &BTreeMap<u8, RequestId>, request_id: RequestId) -> u8 {
    request_ids
        .iter()
        .find_map(|(&label, &candidate)| (candidate == request_id).then_some(label))
        .expect("every public request ID has a fixed corpus label")
}

fn record_updates(
    updates: &riley_scheduler::IterationUpdates,
    request_ids: &BTreeMap<u8, RequestId>,
    events: &mut BTreeMap<(u8, usize), u32>,
    completions: &mut BTreeMap<u8, RequestCompletion>,
) {
    assert!(updates.settlement_failures().is_empty());
    for event in updates.token_events() {
        let label = label_for(request_ids, event.request_id());
        assert!(
            events
                .insert((label, event.generated_index()), event.token_id())
                .is_none(),
            "one fixed route may publish at most one token"
        );
    }
    for completion in updates.completions() {
        let label = label_for(request_ids, completion.request_id());
        assert!(
            completions.insert(label, completion.clone()).is_none(),
            "one fixed request may terminally complete only once"
        );
    }
}

fn assert_closed_quiescent(closed: &SchedulerCloseOutput) {
    assert!(closed.completions().is_empty());
    assert!(closed.settlement_failures().is_empty());
    let gauges = closed.final_metrics().gauges;
    assert_eq!(gauges.waiting_requests, 0);
    assert_eq!(gauges.waiting_prompt_tokens, 0);
    assert_eq!(gauges.active_sequences, 0);
    assert_eq!(gauges.promised_kv_blocks, 0);
    assert_eq!(gauges.allocated_kv_blocks, 0);
    assert_eq!(gauges.pending_completions, 0);
    assert_eq!(gauges.outstanding_iterations, 0);
}

fn expected_normal_history(case: &GpuFixedCorpusCase, label: u8) -> usize {
    if matches!(
        case.settlement(),
        GpuFixedSettlement::DeferredCancel {
            label: cancelled_label
        } if *cancelled_label == label
    ) {
        1
    } else if case.primed_labels().contains(&label) {
        case.decoder_max_new_tokens()
    } else {
        case.final_prefill_max_new_tokens()
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_and_commit_phase(
    case: &GpuFixedCorpusCase,
    phase: GpuFixedPhase,
    plan: IterationPlan,
    scheduler: &mut Scheduler,
    executor: &mut PreparedLlamaBatchExecutor,
    stream: &mut CudaStream,
    backend: OutputBackend,
    workspace: &mut Option<SamplingWorkspace>,
    rng: &mut CountingRng,
    request_ids: &BTreeMap<u8, RequestId>,
    events: &mut BTreeMap<(u8, usize), u32>,
    completions: &mut BTreeMap<u8, RequestCompletion>,
    deferred_cancel_label: Option<u8>,
    now_ns: u64,
) -> TestResult<(Vec<(u8, usize, usize)>, Vec<u32>)> {
    let routes = case.routes_for_phase(phase);
    let downloaded = execute_download(&plan, executor, stream, backend)?;
    assert_eq!(downloaded.iteration_id(), plan.iteration_id());
    assert_eq!(downloaded.output_count(), routes.len());
    let tokens = downloaded_tokens(&downloaded, executor, backend, workspace, rng)?;
    assert_eq!(tokens.len(), routes.len());
    if let Some(label) = deferred_cancel_label {
        let request_id = *request_ids
            .get(&label)
            .expect("cancelled label was submitted");
        let outcome = scheduler.cancel(request_id, now_ns - 1)?;
        assert_eq!(outcome.request_id(), request_id);
        assert!(outcome.deferred_until_iteration_settles());
        assert!(!outcome.already_terminal());
        assert!(outcome.completion().is_none());
    }
    let samples = tokens
        .iter()
        .copied()
        .map(|token| SampledIterationToken::new(token, false))
        .collect::<Vec<_>>();
    let result = downloaded.into_result(&samples, IterationTiming::default())?;
    let updates = scheduler.complete_iteration(&result, now_ns)?;
    record_updates(&updates, request_ids, events, completions);
    for route in &routes {
        let cancelled = deferred_cancel_label == Some(route.label);
        if !cancelled {
            assert_eq!(
                events.get(&(route.label, route.generation_step)),
                tokens.get(route.output_slot),
                "dense GPU slot must publish to its descriptor request/step",
            );
        }
    }
    Ok((
        routes
            .iter()
            .map(|route| (route.label, route.generation_step, route.output_slot))
            .collect(),
        tokens,
    ))
}

fn replay_normal_case(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    case: &GpuFixedCorpusCase,
    backend: OutputBackend,
) -> TestResult<CaseTrace> {
    assert!(!case.is_commit_data_assembly_failure());
    let mut scheduler = new_scheduler(case);
    let mut executor = prepare_executor(model, context, stream, case)?;
    let stable_allocation = context.allocation_stats()?;
    let mut workspace = match backend {
        OutputBackend::FallbackLogits => Some(SamplingWorkspace::new(executor.vocabulary_size())?),
        OutputBackend::GpuGreedy => None,
    };
    let mut rng = CountingRng::default();
    let mut request_ids = BTreeMap::new();
    let mut events = BTreeMap::new();
    let mut completions = BTreeMap::new();
    let mut phases = Vec::new();

    submit_labels(
        &mut scheduler,
        case.primed_labels(),
        case.decoder_prompt_token_count(),
        case.decoder_max_new_tokens(),
        0,
        &mut request_ids,
    );
    let prime = plan_now(&mut scheduler, 1);
    assert_phase_plan(case, GpuFixedPhase::Prime, &prime, &request_ids, &events);
    phases.push(execute_and_commit_phase(
        case,
        GpuFixedPhase::Prime,
        prime,
        &mut scheduler,
        &mut executor,
        stream,
        backend,
        &mut workspace,
        &mut rng,
        &request_ids,
        &mut events,
        &mut completions,
        None,
        2,
    )?);
    assert_eq!(context.allocation_stats()?, stable_allocation);

    submit_labels(
        &mut scheduler,
        case.final_prefill_labels(),
        case.final_prefill_prompt_token_count(),
        case.final_prefill_max_new_tokens(),
        3,
        &mut request_ids,
    );
    let mixed = plan_now(&mut scheduler, 4);
    assert_phase_plan(case, GpuFixedPhase::Mixed, &mixed, &request_ids, &events);
    let deferred_cancel_label = match case.settlement() {
        GpuFixedSettlement::DeferredCancel { label } => Some(*label),
        GpuFixedSettlement::Commit => None,
        GpuFixedSettlement::AbortAfterInvalidSampleCount => unreachable!("normal corpus only"),
    };
    phases.push(execute_and_commit_phase(
        case,
        GpuFixedPhase::Mixed,
        mixed,
        &mut scheduler,
        &mut executor,
        stream,
        backend,
        &mut workspace,
        &mut rng,
        &request_ids,
        &mut events,
        &mut completions,
        deferred_cancel_label,
        6,
    )?);
    assert_eq!(context.allocation_stats()?, stable_allocation);

    if case.requires_boundary_decode() {
        let boundary = plan_now(&mut scheduler, 7);
        assert_phase_plan(
            case,
            GpuFixedPhase::BoundaryDecode,
            &boundary,
            &request_ids,
            &events,
        );
        phases.push(execute_and_commit_phase(
            case,
            GpuFixedPhase::BoundaryDecode,
            boundary,
            &mut scheduler,
            &mut executor,
            stream,
            backend,
            &mut workspace,
            &mut rng,
            &request_ids,
            &mut events,
            &mut completions,
            None,
            8,
        )?);
        assert_eq!(context.allocation_stats()?, stable_allocation);
    }

    assert_eq!(rng.draws, 0, "temperature-zero path consumed RNG");
    assert_eq!(completions.len(), case.concurrency());
    for label in case.all_labels() {
        let completion = completions
            .get(&label)
            .expect("every normal fixed request completes exactly once");
        let expected_reason = match case.normal_terminal_reason(label) {
            GpuFixedTerminalReason::Cancelled => RequestFinishReason::Cancelled,
            GpuFixedTerminalReason::Length => RequestFinishReason::Length,
        };
        assert_eq!(completion.reason(), expected_reason);
        assert_eq!(
            completion.generated_token_ids().len(),
            expected_normal_history(case, label),
        );
    }
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let closed = scheduler.close(9, None)?;
    assert_closed_quiescent(&closed);
    executor.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(CaseTrace {
        phases,
        events: events
            .into_iter()
            .map(|((label, generation_step), token)| (label, generation_step, token))
            .collect(),
        completions: completions
            .into_iter()
            .map(|(label, completion)| CompletionTrace {
                label,
                reason: completion.reason(),
                generated_token_ids: completion.generated_token_ids().to_vec(),
            })
            .collect(),
        rng_draws: rng.draws,
    })
}

fn replay_commit_data_assembly_failure_case(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    case: &GpuFixedCorpusCase,
) -> TestResult {
    assert!(case.is_commit_data_assembly_failure());
    let mut scheduler = new_scheduler(case);
    let mut executor = prepare_executor(model, context, stream, case)?;
    let stable_allocation = context.allocation_stats()?;
    let mut workspace = None;
    let mut rng = CountingRng::default();
    let mut request_ids = BTreeMap::new();
    let mut events = BTreeMap::new();
    let mut completions = BTreeMap::new();

    submit_labels(
        &mut scheduler,
        case.primed_labels(),
        case.decoder_prompt_token_count(),
        case.decoder_max_new_tokens(),
        0,
        &mut request_ids,
    );
    let prime = plan_now(&mut scheduler, 1);
    assert_phase_plan(case, GpuFixedPhase::Prime, &prime, &request_ids, &events);
    let _ = execute_and_commit_phase(
        case,
        GpuFixedPhase::Prime,
        prime,
        &mut scheduler,
        &mut executor,
        stream,
        OutputBackend::GpuGreedy,
        &mut workspace,
        &mut rng,
        &request_ids,
        &mut events,
        &mut completions,
        None,
        2,
    )?;
    assert_eq!(context.allocation_stats()?, stable_allocation);

    submit_labels(
        &mut scheduler,
        case.final_prefill_labels(),
        case.final_prefill_prompt_token_count(),
        case.final_prefill_max_new_tokens(),
        3,
        &mut request_ids,
    );
    let mixed = plan_now(&mut scheduler, 4);
    assert_phase_plan(case, GpuFixedPhase::Mixed, &mixed, &request_ids, &events);
    let downloaded = execute_download(&mixed, &mut executor, stream, OutputBackend::GpuGreedy)?;
    assert_eq!(downloaded.output_count(), case.concurrency());
    assert_eq!(context.allocation_stats()?, stable_allocation);
    let tokens = downloaded_tokens(
        &downloaded,
        &executor,
        OutputBackend::GpuGreedy,
        &mut workspace,
        &mut rng,
    )?;
    let samples = tokens
        .iter()
        .copied()
        .map(|token| SampledIterationToken::new(token, false))
        .collect::<Vec<_>>();
    let failure = downloaded
        .into_result(
            samples
                .get(..samples.len().checked_sub(1).expect("eight output rows"))
                .expect("drop exactly one dense sample"),
            IterationTiming::default(),
        )
        .expect_err("seven samples for eight downloaded rows must fail assembly");
    assert!(matches!(
        failure.error(),
        IterationAdapterError::InvalidSampleCount {
            expected: 8,
            actual: 7
        }
    ));
    assert_eq!(failure.iteration().greedy_token_ids(), tokens.as_slice());
    let (iteration_id, abort) = failure.abort_data();
    assert_eq!(abort, ExecutionAbort::DeviceQuiescedMutationUnknown);
    let updates = scheduler.abort_iteration(iteration_id, abort, 5)?;
    assert!(updates.token_events().is_empty());
    record_updates(&updates, &request_ids, &mut events, &mut completions);
    assert_eq!(events.len(), case.primed_labels().len());
    assert_eq!(completions.len(), case.concurrency());
    for label in case.all_labels() {
        let completion = completions
            .get(&label)
            .expect("poisoned mixed plan terminally closes every request");
        assert_eq!(completion.reason(), RequestFinishReason::ExecutorFailure);
        let expected_history_len = usize::from(case.primed_labels().contains(&label));
        assert_eq!(completion.generated_token_ids().len(), expected_history_len);
    }
    assert_eq!(rng.draws, 0);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let closed = scheduler.close(6, None)?;
    assert_closed_quiescent(&closed);
    executor.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(())
}

#[test]
#[ignore = "requires C02-qualified candidate, approved CUDA execution, and RILEY_REAL_CHECKPOINT"]
fn fixed_c03_b_gpu_corpus_matches_cpu_fallback_and_greedy_output() -> TestResult {
    let limits = LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?;
    let model = LoadedModel::load(&checkpoint_path(), limits)?;
    let (context, mut stream) = first_context()?;
    for case in gpu_fixed_corpus()
        .iter()
        .filter(|case| !case.is_commit_data_assembly_failure())
    {
        let cpu = replay_normal_case(
            &model,
            &context,
            &mut stream,
            case,
            OutputBackend::FallbackLogits,
        )?;
        let gpu = replay_normal_case(
            &model,
            &context,
            &mut stream,
            case,
            OutputBackend::GpuGreedy,
        )?;
        assert_eq!(gpu, cpu, "{} CPU/GPU routing parity", case.case_id());
    }
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "requires C02-qualified candidate, approved CUDA execution, and RILEY_REAL_CHECKPOINT"]
fn fixed_c03_b_gpu_corpus_contains_commit_data_assembly_abort() -> TestResult {
    let limits = LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?;
    let model = LoadedModel::load(&checkpoint_path(), limits)?;
    let (context, mut stream) = first_context()?;
    let case = gpu_fixed_corpus()
        .into_iter()
        .find(GpuFixedCorpusCase::is_commit_data_assembly_failure)
        .expect("fixed C=8 commit-data assembly failure fixture");
    replay_commit_data_assembly_failure_case(&model, &context, &mut stream, &case)?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
