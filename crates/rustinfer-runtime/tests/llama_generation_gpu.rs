//! Remote-only PR11 end-to-end Llama generation validation.

#![cfg(feature = "cuda")]
#![allow(clippy::similar_names, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::PathBuf;

use rustinfer_cuda::{CudaAllocationStats, CudaContext, CudaRuntime, CudaStream};
use rustinfer_model::{LoadLimits, LoadedModel, Tokenizer};
use rustinfer_runtime::generation::{FinishReason, GenerationRequest, GenerationState};
use rustinfer_runtime::llama::{
    GenerationModelStage, GenerationTokenTiming, LlamaDecodePhase, LlamaGenerationEvent,
    LlamaGenerationFailure, LlamaGenerationTimingSummary, PreparedLlamaDecodeConfig,
    PreparedLlamaGeneration,
};
use rustinfer_runtime::rng::PHILOX4X32_10_ALGORITHM_ID;
use rustinfer_runtime::sampling::SamplingParams;
use serde_json::Value;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const BF16_BYTES: usize = 2;
const EXPECTED_VOCABULARY_SIZE: usize = 49_152;
const EXPECTED_GOLDEN_CASES: usize = 31;
const EXPECTED_GOLDEN_OUTPUT_CAPACITY: usize = 16;
const STOCHASTIC_OUTPUT_CAPACITY: usize = 8;
const PINNED_PROMPT: [u32; 7] = [504, 2_365, 6_354, 16_438, 11_139, 253, 1_890];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GoldenFinish {
    Length,
    Eos,
}

impl GoldenFinish {
    const fn runtime(self) -> FinishReason {
        match self {
            Self::Length => FinishReason::Length,
            Self::Eos => FinishReason::Eos,
        }
    }
}

#[derive(Debug)]
struct GoldenCase {
    index: usize,
    prompt_token_ids: Vec<u32>,
    cache_on_token_ids: Vec<u32>,
    cache_off_token_ids: Vec<u32>,
    finish: GoldenFinish,
}

#[derive(Debug)]
struct GoldenFixture {
    cases_by_prompt_length: BTreeMap<usize, Vec<GoldenCase>>,
    eos_token_ids: Vec<u32>,
    max_new_tokens: usize,
}

#[derive(Debug, Default)]
struct CallbackTrace {
    token_ids: Vec<u32>,
    token_logprobs: Vec<f32>,
    text: String,
    finish_reasons: Vec<Option<FinishReason>>,
    timings: Vec<GenerationTokenTiming>,
    cancellation_deltas: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ConsumerStopped;

impl fmt::Display for ConsumerStopped {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the PR11 test consumer stopped")
    }
}

impl Error for ConsumerStopped {}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn golden_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/reference/smollm2-135m-bf16.json")
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote runner has no CUDA device"
    );
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((context, stream))
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

fn json_u32_array(value: &Value, field: &'static str) -> TestResult<Vec<u32>> {
    let values = value.as_array().ok_or(field)?;
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let raw = value.as_u64().ok_or(field)?;
        output.push(u32::try_from(raw)?);
    }
    Ok(output)
}

fn parse_golden_fixture() -> TestResult<GoldenFixture> {
    let document: Value = serde_json::from_slice(&fs::read(golden_fixture_path())?)?;
    assert_eq!(document["schema_version"], "1.0.0");
    assert_eq!(
        document["contract"]["model_id"],
        "HuggingFaceTB/SmolLM2-135M"
    );
    assert_eq!(document["generation"]["strategy"], "greedy");
    assert_eq!(
        document["generation"]["cache_modes"],
        serde_json::json!(["on", "off"])
    );
    assert_eq!(document["rng"]["algorithm_id"], PHILOX4X32_10_ALGORITHM_ID);
    assert_eq!(document["rng"]["domain"], "token-sampling");
    assert_eq!(document["rng"]["greedy_draws_consumed"], 0);

    let eos_token_ids = json_u32_array(
        &document["generation"]["eos_token_ids"],
        "generation.eos_token_ids must be a U32 array",
    )?;
    let max_new_tokens = usize::try_from(
        document["generation"]["max_new_tokens"]
            .as_u64()
            .ok_or("generation.max_new_tokens must be an unsigned integer")?,
    )?;
    assert_eq!(max_new_tokens, EXPECTED_GOLDEN_OUTPUT_CAPACITY);

    let cases = document["cases"]
        .as_array()
        .ok_or("cases must be an array")?;
    assert_eq!(cases.len(), EXPECTED_GOLDEN_CASES);
    assert_eq!(
        document["corpus"]["prompt_count"].as_u64(),
        Some(u64::try_from(cases.len())?)
    );

    let mut cases_by_prompt_length: BTreeMap<usize, Vec<GoldenCase>> = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        let prompt_token_ids = json_u32_array(
            &case["input"]["token_ids"],
            "cases[].input.token_ids must be a U32 array",
        )?;
        assert_eq!(
            case["input"]["token_count"].as_u64(),
            Some(u64::try_from(prompt_token_ids.len())?),
            "golden case {index} prompt length"
        );
        let cache_on_token_ids = json_u32_array(
            &case["greedy"]["cache_on_token_ids"],
            "cases[].greedy.cache_on_token_ids must be a U32 array",
        )?;
        let cache_off_token_ids = json_u32_array(
            &case["greedy"]["cache_off_token_ids"],
            "cases[].greedy.cache_off_token_ids must be a U32 array",
        )?;
        assert_eq!(
            cache_on_token_ids, cache_off_token_ids,
            "golden case {index} cache-on/off fixture mismatch"
        );
        assert_eq!(
            case["greedy"]["exact_match"].as_bool(),
            Some(true),
            "golden case {index} fixture exact-match declaration"
        );
        let finish = match case["greedy"]["stop_reason"].as_str() {
            Some("max_new_tokens") => GoldenFinish::Length,
            Some("eos") => GoldenFinish::Eos,
            other => return Err(format!("unsupported golden stop reason {other:?}").into()),
        };
        match finish {
            GoldenFinish::Length => assert_eq!(
                cache_on_token_ids.len(),
                max_new_tokens,
                "golden case {index} length finish"
            ),
            GoldenFinish::Eos => assert!(
                cache_on_token_ids.len() <= max_new_tokens,
                "golden case {index} EOS finish exceeds its cap"
            ),
        }

        cases_by_prompt_length
            .entry(prompt_token_ids.len())
            .or_default()
            .push(GoldenCase {
                index,
                prompt_token_ids,
                cache_on_token_ids,
                cache_off_token_ids,
                finish,
            });
    }

    Ok(GoldenFixture {
        cases_by_prompt_length,
        eos_token_ids,
        max_new_tokens,
    })
}

fn capture_event(trace: &mut CallbackTrace, event: LlamaGenerationEvent<'_>) {
    match event {
        LlamaGenerationEvent::Token { token, timing } => {
            trace.token_ids.push(token.token_id());
            trace.token_logprobs.push(
                token
                    .token_logprob()
                    .expect("CPU sampling records log-probability"),
            );
            trace.text.push_str(token.text_delta());
            trace.finish_reasons.push(token.finish_reason());
            trace.timings.push(timing);
        }
        LlamaGenerationEvent::Cancelled { text_delta } => {
            trace.cancellation_deltas.push(text_delta.to_owned());
            trace.text.push_str(text_delta);
        }
    }
}

fn assert_idle_owner(owner: &PreparedLlamaGeneration<'_>) {
    assert!(!owner.is_terminal());
    assert_eq!(owner.decode_phase(), Some(LlamaDecodePhase::Empty));
    let pool = owner
        .paged_pool_stats()
        .expect("PR11 generation tests use the paged KV cache");
    assert!(pool.physical_block_count() > 0);
    assert_eq!(pool.allocated_block_count(), 0);
    assert_eq!(pool.free_block_count(), pool.physical_block_count());
}

fn assert_recovered_owner(owner: &PreparedLlamaGeneration<'_>) {
    assert_idle_owner(owner);
    let pool = owner
        .paged_pool_stats()
        .expect("PR11 generation tests use the paged KV cache");
    assert!(pool.high_water_mark() > 0);
    assert!(pool.lifetime_allocation_count() > 0);
}

fn assert_allocation_report_matches_context(
    owner: &PreparedLlamaGeneration<'_>,
    stats: CudaAllocationStats,
) {
    let report = owner
        .decode_allocation_report()
        .expect("a healthy generation owner retains its decoder");
    assert_eq!(report.total_device_bytes(), stats.device_live_bytes());
    assert_eq!(
        report.device_allocation_count(),
        stats.device_live_allocations()
    );
    assert_eq!(report.pinned_host_bytes(), stats.pinned_host_live_bytes());
    assert_eq!(
        report.pinned_host_allocation_count(),
        stats.pinned_host_live_allocations()
    );
}

fn assert_token_trace(
    trace: &CallbackTrace,
    expected_token_ids: &[u32],
    expected_finish: FinishReason,
    logits_row_bytes: usize,
) {
    assert_eq!(trace.token_ids, expected_token_ids);
    assert_eq!(trace.token_logprobs.len(), expected_token_ids.len());
    assert!(trace.token_logprobs.iter().all(|value| value.is_finite()));
    assert!(trace.cancellation_deltas.is_empty());
    assert_eq!(trace.finish_reasons.len(), expected_token_ids.len());
    assert_eq!(trace.timings.len(), expected_token_ids.len());

    for (token_index, (&finish, timing)) in
        trace.finish_reasons.iter().zip(&trace.timings).enumerate()
    {
        let expected_stage = if token_index == 0 {
            GenerationModelStage::Prefill
        } else {
            GenerationModelStage::Decode
        };
        assert_eq!(timing.model_stage(), expected_stage);
        assert!(timing.model_gpu_milliseconds().is_finite());
        assert!(timing.model_gpu_milliseconds() >= 0.0);
        assert_eq!(timing.logits_download_bytes(), logits_row_bytes);
        if token_index + 1 == expected_token_ids.len() {
            assert_eq!(finish, Some(expected_finish));
        } else {
            assert_eq!(finish, None);
        }
    }
}

fn assert_timing_summary(
    summary: LlamaGenerationTimingSummary,
    sampled_tokens: usize,
    logits_row_bytes: usize,
) -> TestResult {
    assert_eq!(summary.sampled_tokens(), sampled_tokens);
    assert_eq!(summary.prefill_tokens(), usize::from(sampled_tokens > 0));
    assert_eq!(summary.decode_tokens(), sampled_tokens.saturating_sub(1));
    let expected_download_bytes = sampled_tokens
        .checked_mul(logits_row_bytes)
        .ok_or("summary logits-download byte count overflow")?;
    assert_eq!(
        summary.logits_download_bytes(),
        u64::try_from(expected_download_bytes)?
    );
    assert!(summary.model_gpu_milliseconds().is_finite());
    assert!(summary.model_gpu_milliseconds() >= 0.0);
    assert!(summary.request_wall() >= summary.token_wall());
    Ok(())
}

fn generation_request(
    request_id: Vec<u8>,
    seed: u64,
    prompt_token_ids: Vec<u32>,
    sampling_params: SamplingParams,
    max_new_tokens: usize,
    eos_token_ids: Vec<u32>,
) -> GenerationRequest {
    GenerationRequest {
        request_id,
        seed,
        prompt_token_ids,
        sampling_params,
        min_new_tokens: 0,
        max_new_tokens,
        eos_token_ids,
        stop_token_ids: Vec::new(),
        stop_strings: Vec::new(),
    }
}

fn new_state(model: &LoadedModel, request: GenerationRequest) -> TestResult<GenerationState> {
    Ok(GenerationState::new(
        request,
        model.spec().embedding().vocabulary_size(),
        model.tokenizer().maximum_decoded_token_bytes(),
    )?)
}

#[test]
#[ignore = "remote-only 31-case Gate C golden generation on server-4096"]
fn pinned_smollm2_all_greedy_sequences_match_cache_on_and_cache_off_golden() -> TestResult {
    let fixture = parse_golden_fixture()?;
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    assert_eq!(
        model.spec().embedding().vocabulary_size(),
        EXPECTED_VOCABULARY_SIZE
    );
    assert_eq!(model.spec().special_tokens().eos(), fixture.eos_token_ids);
    let (context, mut stream) = first_context()?;
    let logits_row_bytes = EXPECTED_VOCABULARY_SIZE
        .checked_mul(BF16_BYTES)
        .ok_or("logits row byte count overflow")?;
    let prompt_shape_count = fixture.cases_by_prompt_length.len();
    let mut executed_cases = 0_usize;

    for (prompt_length, cases) in fixture.cases_by_prompt_length {
        let mut owner = PreparedLlamaGeneration::prepare(
            &model,
            &context,
            &mut stream,
            prompt_length,
            fixture.max_new_tokens,
            PreparedLlamaDecodeConfig::default()
                .with_paged_kv_cache()
                .with_optimized_decode_attention(),
        )?;
        assert_eq!(owner.prompt_length(), prompt_length);
        assert_eq!(owner.output_capacity(), fixture.max_new_tokens);
        assert_eq!(owner.vocabulary_size(), EXPECTED_VOCABULARY_SIZE);
        assert_eq!(owner.logits_row_bytes(), logits_row_bytes);
        assert_idle_owner(&owner);
        let stable_allocations = context.allocation_stats()?;
        assert_allocation_report_matches_context(&owner, stable_allocations);

        for case in cases {
            let request = generation_request(
                format!("pr11-golden-case-{:02}", case.index).into_bytes(),
                0,
                case.prompt_token_ids,
                SamplingParams {
                    temperature: 0.0,
                    top_k: Some(10),
                    top_p: None,
                    repetition_penalty: 1.0,
                },
                fixture.max_new_tokens,
                fixture.eos_token_ids.clone(),
            );
            let mut state = new_state(&model, request)?;
            let mut trace = CallbackTrace::default();
            let summary = owner.generate(
                &mut state,
                &mut stream,
                || false,
                |event| {
                    capture_event(&mut trace, event);
                    Ok::<(), Infallible>(())
                },
            )?;

            assert_eq!(
                case.cache_on_token_ids, case.cache_off_token_ids,
                "golden case {} fixture cache parity",
                case.index
            );
            assert_eq!(
                state.generated_token_ids(),
                case.cache_on_token_ids,
                "golden case {} adapter/cache-on exact",
                case.index
            );
            assert_eq!(
                state.generated_token_ids(),
                case.cache_off_token_ids,
                "golden case {} adapter/cache-off exact",
                case.index
            );
            assert_eq!(state.finish_reason(), Some(case.finish.runtime()));
            assert_eq!(state.rng_draws(), 0);
            assert_eq!(state.rng_algorithm_id(), PHILOX4X32_10_ALGORITHM_ID);
            assert_eq!(trace.text, state.text());
            assert_token_trace(
                &trace,
                &case.cache_on_token_ids,
                case.finish.runtime(),
                logits_row_bytes,
            );
            assert_timing_summary(summary, case.cache_on_token_ids.len(), logits_row_bytes)?;
            assert_recovered_owner(&owner);
            assert_eq!(context.allocation_stats()?, stable_allocations);
            executed_cases += 1;

            println!(
                "pr11-generation-golden schema_version=1 case={} prompt_length={} \
sampled_tokens={} prefill_calls={} decode_calls={} rng_draws=0 finish_reason={} \
cache_on_off_fixture_exact=true adapter_cache_on_exact=true adapter_cache_off_exact=true \
logits_download_bytes={} owner_reused_by_prompt_shape=true status=passed",
                case.index,
                prompt_length,
                case.cache_on_token_ids.len(),
                summary.prefill_tokens(),
                summary.decode_tokens(),
                case.finish.runtime(),
                summary.logits_download_bytes(),
            );
        }

        owner.close()?;
        assert!(context.allocation_stats()?.is_zero());
    }

    assert_eq!(executed_cases, EXPECTED_GOLDEN_CASES);
    println!(
        "pr11-generation-golden-summary schema_version=1 cases={executed_cases} \
prompt_shapes={prompt_shape_count} \
cache_modes=on,off greedy_exact=true rng_draws=0 n_tokens_requires_n_minus_1_decode=true \
per_token_cpu_gpu_timing=true full_bf16_logits_d2h_bytes_per_token={logits_row_bytes} \
kv_pool_reset=true cuda_allocation_zero_after_close=true status=passed",
    );
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote-only PR11 stochastic and generation lifecycle validation on server-4096"]
fn pinned_smollm2_fixed_seed_cancellation_callback_and_reuse_are_deterministic() -> TestResult {
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    assert_eq!(
        model.spec().embedding().vocabulary_size(),
        EXPECTED_VOCABULARY_SIZE
    );
    let (context, mut stream) = first_context()?;
    let mut owner = PreparedLlamaGeneration::prepare(
        &model,
        &context,
        &mut stream,
        PINNED_PROMPT.len(),
        STOCHASTIC_OUTPUT_CAPACITY,
        PreparedLlamaDecodeConfig::default()
            .with_paged_kv_cache()
            .with_optimized_decode_attention(),
    )?;
    let logits_row_bytes = owner.logits_row_bytes();
    assert_eq!(logits_row_bytes, EXPECTED_VOCABULARY_SIZE * BF16_BYTES);
    assert_idle_owner(&owner);
    let stable_allocations = context.allocation_stats()?;
    assert_allocation_report_matches_context(&owner, stable_allocations);

    let stochastic_params = SamplingParams {
        temperature: 0.8,
        top_k: Some(32),
        top_p: Some(0.9),
        repetition_penalty: 1.05,
    };

    let mut pre_model_state = new_state(
        &model,
        generation_request(
            b"pr11-cancel-before-prefill".to_vec(),
            7,
            PINNED_PROMPT.to_vec(),
            stochastic_params,
            STOCHASTIC_OUTPUT_CAPACITY,
            Vec::new(),
        ),
    )?;
    let mut pre_model_checks = 0_usize;
    let mut pre_model_trace = CallbackTrace::default();
    let pre_model_summary = owner.generate(
        &mut pre_model_state,
        &mut stream,
        || {
            pre_model_checks += 1;
            true
        },
        |event| {
            capture_event(&mut pre_model_trace, event);
            Ok::<(), Infallible>(())
        },
    )?;
    assert_eq!(pre_model_checks, 1);
    assert_eq!(
        pre_model_state.finish_reason(),
        Some(FinishReason::Cancelled)
    );
    assert!(pre_model_state.generated_token_ids().is_empty());
    assert_eq!(pre_model_state.rng_draws(), 0);
    assert!(pre_model_trace.token_ids.is_empty());
    assert_eq!(pre_model_trace.cancellation_deltas, [String::new()]);
    assert_timing_summary(pre_model_summary, 0, logits_row_bytes)?;
    assert_idle_owner(&owner);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let mut post_model_state = new_state(
        &model,
        generation_request(
            b"pr11-cancel-after-prefill".to_vec(),
            7,
            PINNED_PROMPT.to_vec(),
            stochastic_params,
            STOCHASTIC_OUTPUT_CAPACITY,
            Vec::new(),
        ),
    )?;
    let mut post_model_checks = 0_usize;
    let mut post_model_trace = CallbackTrace::default();
    let post_model_summary = owner.generate(
        &mut post_model_state,
        &mut stream,
        || {
            post_model_checks += 1;
            post_model_checks == 2
        },
        |event| {
            capture_event(&mut post_model_trace, event);
            Ok::<(), Infallible>(())
        },
    )?;
    assert_eq!(
        post_model_checks, 2,
        "the second check occurs after the in-flight model stage"
    );
    assert_eq!(
        post_model_state.finish_reason(),
        Some(FinishReason::Cancelled)
    );
    assert!(post_model_state.generated_token_ids().is_empty());
    assert_eq!(post_model_state.rng_draws(), 0);
    assert!(post_model_trace.token_ids.is_empty());
    assert_eq!(post_model_trace.cancellation_deltas, [String::new()]);
    assert_timing_summary(post_model_summary, 0, logits_row_bytes)?;
    assert_recovered_owner(&owner);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let mut callback_state = new_state(
        &model,
        generation_request(
            b"pr11-consumer-error".to_vec(),
            11,
            PINNED_PROMPT.to_vec(),
            stochastic_params,
            STOCHASTIC_OUTPUT_CAPACITY,
            Vec::new(),
        ),
    )?;
    let mut callback_calls = 0_usize;
    let callback_error = owner
        .generate(
            &mut callback_state,
            &mut stream,
            || false,
            |event| {
                callback_calls += 1;
                assert!(matches!(event, LlamaGenerationEvent::Token { .. }));
                Err(ConsumerStopped)
            },
        )
        .expect_err("the consumer error must propagate after the first accepted token");
    match callback_error.failure() {
        LlamaGenerationFailure::Callback(source) => assert_eq!(*source, ConsumerStopped),
        other => panic!("expected callback failure, got {other}"),
    }
    assert!(callback_error.first_cleanup_failure().is_none());
    assert_eq!(callback_error.additional_cleanup_failures(), 0);
    assert_eq!(callback_calls, 1);
    assert_eq!(callback_state.finish_reason(), Some(FinishReason::Error));
    assert_eq!(callback_state.generated_token_ids().len(), 1);
    assert_eq!(callback_state.rng_draws(), 1);
    assert_recovered_owner(&owner);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let repeat_request = || {
        generation_request(
            b"pr11-fixed-seed-repeat".to_vec(),
            0x5eed_cafe_d00d_f00d,
            PINNED_PROMPT.to_vec(),
            stochastic_params,
            STOCHASTIC_OUTPUT_CAPACITY,
            Vec::new(),
        )
    };
    let mut first_state = new_state(&model, repeat_request())?;
    let mut first_trace = CallbackTrace::default();
    let first_summary = owner.generate(
        &mut first_state,
        &mut stream,
        || false,
        |event| {
            capture_event(&mut first_trace, event);
            Ok::<(), Infallible>(())
        },
    )?;
    assert_eq!(first_state.finish_reason(), Some(FinishReason::Length));
    assert_eq!(
        first_state.generated_token_ids().len(),
        STOCHASTIC_OUTPUT_CAPACITY
    );
    assert_eq!(
        first_state.rng_draws(),
        u128::try_from(STOCHASTIC_OUTPUT_CAPACITY)?
    );
    assert_eq!(first_state.rng_algorithm_id(), PHILOX4X32_10_ALGORITHM_ID);
    assert_eq!(first_trace.text, first_state.text());
    assert_token_trace(
        &first_trace,
        first_state.generated_token_ids(),
        FinishReason::Length,
        logits_row_bytes,
    );
    assert_timing_summary(first_summary, STOCHASTIC_OUTPUT_CAPACITY, logits_row_bytes)?;
    assert_recovered_owner(&owner);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let mut second_state = new_state(&model, repeat_request())?;
    let mut second_trace = CallbackTrace::default();
    let second_summary = owner.generate(
        &mut second_state,
        &mut stream,
        || false,
        |event| {
            capture_event(&mut second_trace, event);
            Ok::<(), Infallible>(())
        },
    )?;
    assert_eq!(second_state.finish_reason(), Some(FinishReason::Length));
    assert_eq!(
        second_state.generated_token_ids(),
        first_state.generated_token_ids()
    );
    assert_eq!(second_state.text(), first_state.text());
    assert_eq!(second_state.rng_snapshot(), first_state.rng_snapshot());
    assert_eq!(
        second_state.rng_draws(),
        u128::try_from(STOCHASTIC_OUTPUT_CAPACITY)?
    );
    assert_eq!(second_trace.token_ids, first_trace.token_ids);
    assert_eq!(second_trace.text, first_trace.text);
    assert_token_trace(
        &second_trace,
        second_state.generated_token_ids(),
        FinishReason::Length,
        logits_row_bytes,
    );
    assert_timing_summary(second_summary, STOCHASTIC_OUTPUT_CAPACITY, logits_row_bytes)?;
    assert_recovered_owner(&owner);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    println!(
        "pr11-generation-lifecycle schema_version=1 stochastic_tokens={} stochastic_rng_draws={} \
fixed_seed_repeat_exact=true rng_snapshot_repeat_exact=true pre_model_cancel_rng_draws=0 \
post_model_cancel_rng_draws=0 callback_error_rng_draws=1 callback_error_recovered=true \
n_tokens_requires_n_minus_1_decode=true prefill_calls={} decode_calls={} \
full_bf16_logits_d2h_bytes_per_token={} per_token_cpu_gpu_timing=true \
healthy_owner_reused=true kv_pool_reset=true status=passed",
        STOCHASTIC_OUTPUT_CAPACITY,
        first_state.rng_draws(),
        first_summary.prefill_tokens(),
        first_summary.decode_tokens(),
        logits_row_bytes,
    );

    owner.close()?;
    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    close_context(context)
}
