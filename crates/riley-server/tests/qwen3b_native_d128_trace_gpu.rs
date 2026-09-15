//! Remote-only Qwen2.5-3B teacher-forced native-D128 numerical trace.
//!
//! This test deliberately replays the production scheduler/executor seam rather
//! than opening an HTTP listener.  It uses the same C8/M32, canonical, eager
//! native-D128 configuration as the serving smoke and feeds the externally
//! recorded Hugging Face *cache-off* tokens back through the scheduler after
//! every authoritative commit.  Cache-off is explicit here: the older external
//! cache-on generation artifact records its first cached token at position 2049
//! instead of the server's 2048, so it is not used as this trace's decode-row
//! reference.
//!
//! The test emits one compact JSON receipt only after all staged rows have
//! committed.  The receipt is a numerical diagnostic, never a serving
//! performance result or a C02 generation-audit record.

#![cfg(feature = "cuda")]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use riley_model::{LoadLimits, LoadedModel, ModelArchitecture, ModelFamily};
use riley_runtime::llama::{
    LlamaBatchMetadataConfig, LlamaProjectionBiasMode, LlamaReductionProfile,
    PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
use riley_scheduler::{
    DownloadedLlamaIteration, ExecutionAbort, IterationId, LlamaIterationCudaTimer, OverloadPolicy,
    RequestDescriptor, RequestFinishReason, SampledIterationToken, Scheduler, SchedulerConfig,
    execute_llama_iteration_timed,
};
use riley_tensor::DType;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const QWEN3B_MODEL_ID: &str = "Qwen/Qwen2.5-3B-Instruct";
const QWEN3B_REVISION: &str = "aa8e72537993ba99e69dfaafa59ed015b17504d1";
const QWEN3B_WORKLOAD_SCHEMA: &str = "riley.n06a-d128-serving-workload.v1";
const QWEN3B_WORKLOAD_CASE: &str = "qwen3b-c8-p2048-o128";
const QWEN3B_SERVING_WORKLOAD_SHA256: &str =
    "7a0a8fec31d45e397e1ec57335fa1c9de2d3da7daa9a32e9002a62763c05261e";
const HF_GENERATION_ORACLE_SCHEMA: &str = "riley.qwen3b-hf-eager-generation.v1";
const HF_CACHE_OFF_MODE: &str = "cache-off";
const EXPECTED_SOURCE_ARCHITECTURE: &str = "Qwen2ForCausalLM";
const EXPECTED_LAYER_COUNT: usize = 36;
const EXPECTED_HIDDEN_SIZE: usize = 2_048;
const EXPECTED_QUERY_HEADS: usize = 16;
const EXPECTED_KEY_VALUE_HEADS: usize = 2;
const EXPECTED_HEAD_DIMENSION: usize = 128;
const EXPECTED_VOCABULARY_SIZE: usize = 151_936;
const EXPECTED_ADDRESSABLE_TOKEN_COUNT: usize = 151_665;
const EXPECTED_MAX_SEQUENCE_LENGTH: usize = 32_768;
const MAX_WEIGHT_BYTES: u64 = 8 * ONE_GIB;
const SERVER_MAX_ACTIVE_SEQUENCES: usize = 8;
const SERVER_MAX_WAITING_REQUESTS: usize = 64;
const SERVER_MAX_SEQUENCE_TOKENS: usize = 2_176;
const SERVER_MAX_OUTPUT_TOKENS: usize = 128;
const SERVER_BATCH_TOKEN_BUDGET: usize = 32;
const SERVER_PREFILL_CHUNK_TOKENS: usize = 32;
const SERVER_PHYSICAL_KV_BLOCKS: usize = 1_088;
const TRACE_OUTPUT_ROWS: usize = 9;
const TOP_K: usize = 32;
const NATIVE_D128_BACKEND_ID: &str = "riley.cuda.ragged-paged-attention.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2";

#[derive(Debug)]
struct ServingWorkload {
    prompt_token_ids: Vec<u32>,
    output_token_ids: Vec<u32>,
}

#[derive(Debug)]
struct HfStep {
    step: usize,
    input_token_count: usize,
    attention_mask_token_count: usize,
    position_start: usize,
    position_end: usize,
    raw_logit_sha256: String,
    raw_argmax_token_id: u32,
    selected_token_id: u32,
    selected_logit: f32,
    top_token_ids: Vec<u32>,
    top_values: Vec<f32>,
}

#[derive(Debug)]
struct HfCacheOffOracle {
    artifact_sha256: String,
    selected_token_ids: Vec<u32>,
    steps: Vec<HfStep>,
}

#[derive(Debug)]
struct TraceRow {
    step: usize,
    scheduler_iteration_id: u64,
    kind: &'static str,
    iteration_input_token_count: usize,
    target_logical_length: usize,
    row_bf16_le_sha256: String,
    addressable_bf16_le_sha256: String,
    raw_argmax_token_id: u32,
    selected_token_id: u32,
    selected_logit: f32,
    top_token_ids: Vec<u32>,
    top_values: Vec<f32>,
    hf: HfStep,
}

#[derive(Debug)]
struct ModeTrace {
    projection_bias_backend: &'static str,
    rows: Vec<TraceRow>,
    prefill_iteration_count: usize,
    decode_iteration_count: usize,
}

fn required_path(name: &'static str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must name the remote Qwen3B artifact"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn json_u32(value: &Value, label: &'static str) -> TestResult<u32> {
    Ok(u32::try_from(value.as_u64().ok_or(label)?)?)
}

fn json_usize(value: &Value, label: &'static str) -> TestResult<usize> {
    Ok(usize::try_from(value.as_u64().ok_or(label)?)?)
}

fn json_f32(value: &Value, label: &'static str) -> TestResult<f32> {
    let number: f32 = serde_json::from_value(value.clone())?;
    if !number.is_finite() {
        return Err(format!("{label} must be finite").into());
    }
    Ok(number)
}

fn json_u32_array(value: &Value, label: &'static str) -> TestResult<Vec<u32>> {
    let values = value.as_array().ok_or(label)?;
    values.iter().map(|value| json_u32(value, label)).collect()
}

fn json_f32_array(value: &Value, label: &'static str) -> TestResult<Vec<f32>> {
    let values = value.as_array().ok_or(label)?;
    values.iter().map(|value| json_f32(value, label)).collect()
}

fn require_exact_fields(value: &Value, expected: &[&str], label: &'static str) -> TestResult {
    let object = value.as_object().ok_or(label)?;
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual != expected {
        return Err(
            format!("{label} fields differ: actual={actual:?} expected={expected:?}").into(),
        );
    }
    Ok(())
}

fn load_workload() -> TestResult<ServingWorkload> {
    let path = required_path("RILEY_QWEN_SERVING_WORKLOAD");
    let payload = fs::read(&path)?;
    assert_eq!(sha256_hex(&payload), QWEN3B_SERVING_WORKLOAD_SHA256);
    let document: Value = serde_json::from_slice(&payload)?;
    require_exact_fields(
        &document,
        &[
            "schema_version",
            "case",
            "model_id",
            "model_revision",
            "prompt",
            "prompt_token_ids",
            "output_token_ids",
            "output_text",
            "finish_reason",
            "sampling",
        ],
        "serving workload",
    )?;
    assert_eq!(
        document["schema_version"].as_str(),
        Some(QWEN3B_WORKLOAD_SCHEMA)
    );
    assert_eq!(document["case"].as_str(), Some(QWEN3B_WORKLOAD_CASE));
    assert_eq!(document["model_id"].as_str(), Some(QWEN3B_MODEL_ID));
    assert_eq!(document["model_revision"].as_str(), Some(QWEN3B_REVISION));
    assert_eq!(document["finish_reason"].as_str(), Some("length"));
    assert_eq!(document["sampling"]["temperature"].as_f64(), Some(0.0));
    assert_eq!(document["sampling"]["top_p"].as_f64(), Some(1.0));
    let prompt_token_ids = json_u32_array(&document["prompt_token_ids"], "prompt_token_ids")?;
    let output_token_ids = json_u32_array(&document["output_token_ids"], "output_token_ids")?;
    assert_eq!(prompt_token_ids.len(), 2_048);
    assert_eq!(output_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
    assert!(
        prompt_token_ids
            .iter()
            .chain(&output_token_ids)
            .all(|&token| usize::try_from(token)
                .is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT))
    );
    Ok(ServingWorkload {
        prompt_token_ids,
        output_token_ids,
    })
}

fn parse_hf_step(value: &Value) -> TestResult<HfStep> {
    require_exact_fields(
        value,
        &[
            "attention_mask_token_count",
            "input_token_count",
            "logits_bf16_le_sha256",
            "position_end",
            "position_start",
            "raw_argmax_token_id",
            "selected_logit_bf16_as_f32",
            "selected_token_id",
            "step",
            "top_token_ids",
            "top_values_f32",
        ],
        "HF generation step",
    )?;
    let raw_logit_sha256 = value["logits_bf16_le_sha256"]
        .as_str()
        .ok_or("HF step logits hash must be a string")?
        .to_owned();
    assert_eq!(raw_logit_sha256.len(), 64);
    assert!(
        raw_logit_sha256
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    );
    let top_token_ids = json_u32_array(&value["top_token_ids"], "HF top_token_ids")?;
    let top_values = json_f32_array(&value["top_values_f32"], "HF top_values_f32")?;
    assert_eq!(top_token_ids.len(), TOP_K);
    assert_eq!(top_values.len(), TOP_K);
    assert!(top_token_ids.iter().all(|&token| {
        usize::try_from(token).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    }));
    assert!(top_values.windows(2).all(|window| window[0] >= window[1]));
    let selected_token_id = json_u32(&value["selected_token_id"], "HF selected_token_id")?;
    assert!(
        usize::try_from(selected_token_id).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    );
    assert_eq!(top_token_ids[0], selected_token_id);
    let selected_logit = json_f32(
        &value["selected_logit_bf16_as_f32"],
        "HF selected_logit_bf16_as_f32",
    )?;
    assert_eq!(top_values[0], selected_logit);
    let raw_argmax_token_id = json_u32(&value["raw_argmax_token_id"], "HF raw_argmax_token_id")?;
    assert!(usize::try_from(raw_argmax_token_id).is_ok_and(|id| id < EXPECTED_VOCABULARY_SIZE));
    Ok(HfStep {
        step: json_usize(&value["step"], "HF step")?,
        input_token_count: json_usize(&value["input_token_count"], "HF input_token_count")?,
        attention_mask_token_count: json_usize(
            &value["attention_mask_token_count"],
            "HF attention_mask_token_count",
        )?,
        position_start: json_usize(&value["position_start"], "HF position_start")?,
        position_end: json_usize(&value["position_end"], "HF position_end")?,
        raw_logit_sha256,
        raw_argmax_token_id,
        selected_token_id,
        selected_logit,
        top_token_ids,
        top_values,
    })
}

fn load_hf_cache_off_oracle() -> TestResult<HfCacheOffOracle> {
    let path = required_path("RILEY_QWEN_HF_GENERATION_ORACLE");
    let payload = fs::read(&path)?;
    let artifact_sha256 = sha256_hex(&payload);
    let document: Value = serde_json::from_slice(&payload)?;
    require_exact_fields(
        &document,
        &[
            "artifact_kind",
            "created_at_utc",
            "execution",
            "generation",
            "model",
            "performance_claim_eligible",
            "producer",
            "schema_version",
            "source",
            "workload",
        ],
        "HF generation artifact",
    )?;
    assert_eq!(
        document["schema_version"].as_str(),
        Some(HF_GENERATION_ORACLE_SCHEMA)
    );
    assert_eq!(
        document["performance_claim_eligible"].as_bool(),
        Some(false)
    );
    assert_eq!(document["model"]["id"].as_str(), Some(QWEN3B_MODEL_ID));
    assert_eq!(
        document["model"]["revision"].as_str(),
        Some(QWEN3B_REVISION)
    );
    assert_eq!(
        document["workload"]["sha256"].as_str(),
        Some(QWEN3B_SERVING_WORKLOAD_SHA256)
    );
    let cache_off = &document["generation"]["cache_off"];
    require_exact_fields(
        cache_off,
        &[
            "fixed_output_token_count",
            "mode",
            "selected_token_ids",
            "selected_token_ids_le_u32_sha256",
            "steps",
            "text",
            "text_utf8_sha256",
        ],
        "HF cache-off generation",
    )?;
    assert_eq!(cache_off["mode"].as_str(), Some(HF_CACHE_OFF_MODE));
    assert_eq!(cache_off["fixed_output_token_count"].as_u64(), Some(128));
    let selected_token_ids = json_u32_array(
        &cache_off["selected_token_ids"],
        "HF cache-off selected_token_ids",
    )?;
    assert_eq!(selected_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
    let expected_token_hash = sha256_hex(
        &selected_token_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        cache_off["selected_token_ids_le_u32_sha256"].as_str(),
        Some(expected_token_hash.as_str())
    );
    let steps_value = cache_off["steps"].as_array().ok_or("HF cache-off steps")?;
    assert_eq!(steps_value.len(), SERVER_MAX_OUTPUT_TOKENS);
    let steps: Vec<_> = steps_value
        .iter()
        .map(parse_hf_step)
        .collect::<TestResult<_>>()?;
    for (index, step) in steps.iter().enumerate() {
        assert_eq!(step.step, index);
        assert_eq!(step.selected_token_id, selected_token_ids[index]);
        assert_eq!(step.input_token_count, 2_048 + index);
        assert_eq!(step.attention_mask_token_count, 2_048 + index);
        assert_eq!(step.position_start, 0);
        assert_eq!(step.position_end, 2_047 + index);
    }
    Ok(HfCacheOffOracle {
        artifact_sha256,
        selected_token_ids,
        steps,
    })
}

fn load_model() -> TestResult<LoadedModel> {
    let checkpoint = required_path("RILEY_QWEN3B_CHECKPOINT");
    let model = LoadedModel::load(
        &checkpoint,
        LoadLimits::default().with_weight_byte_limits(MAX_WEIGHT_BYTES, MAX_WEIGHT_BYTES)?,
    )?;
    assert_eq!(model.config().family(), ModelFamily::Qwen2);
    assert_eq!(model.provenance().source_model(), QWEN3B_MODEL_ID);
    assert_eq!(model.provenance().source_revision(), QWEN3B_REVISION);
    let spec = model.spec();
    assert_eq!(spec.architecture(), ModelArchitecture::Llama);
    assert_eq!(spec.source_architecture(), EXPECTED_SOURCE_ARCHITECTURE);
    assert_eq!(spec.dtype(), DType::BF16);
    assert_eq!(spec.blocks().len(), EXPECTED_LAYER_COUNT);
    assert_eq!(spec.embedding().hidden_size(), EXPECTED_HIDDEN_SIZE);
    assert_eq!(spec.embedding().vocabulary_size(), EXPECTED_VOCABULARY_SIZE);
    assert_eq!(spec.max_sequence_length(), EXPECTED_MAX_SEQUENCE_LENGTH);
    assert_eq!(
        model.tokenizer().addressable_token_count(),
        EXPECTED_ADDRESSABLE_TOKEN_COUNT
    );
    for (index, block) in spec.blocks().iter().enumerate() {
        assert_eq!(block.index(), index);
        let attention = block.attention();
        assert_eq!(attention.query_heads(), EXPECTED_QUERY_HEADS);
        assert_eq!(attention.key_value_heads(), EXPECTED_KEY_VALUE_HEADS);
        assert_eq!(attention.head_dimension(), EXPECTED_HEAD_DIMENSION);
        assert!(attention.bias().query());
        assert!(attention.bias().key());
        assert!(attention.bias().value());
        assert!(!attention.bias().output());
    }
    Ok(model)
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

fn scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: SERVER_MAX_WAITING_REQUESTS,
        max_waiting_prompt_tokens: SERVER_MAX_WAITING_REQUESTS * SERVER_MAX_SEQUENCE_TOKENS,
        max_active_sequences: SERVER_MAX_ACTIVE_SEQUENCES,
        max_sequence_tokens: SERVER_MAX_SEQUENCE_TOKENS,
        iteration_token_budget: SERVER_BATCH_TOKEN_BUDGET,
        max_prefill_chunk_tokens: SERVER_PREFILL_CHUNK_TOKENS,
        aging_threshold_ns: 100_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(30_000_000_000),
        max_promised_kv_blocks: SERVER_PHYSICAL_KV_BLOCKS,
        metrics_window_samples: 1_024,
    }
}

fn executor_config(mode: LlamaProjectionBiasMode) -> TestResult<PreparedLlamaBatchExecutorConfig> {
    let forward = PreparedLlamaForwardConfig::default().with_projection_bias_mode(mode);
    let metadata = LlamaBatchMetadataConfig::new(
        SERVER_MAX_ACTIVE_SEQUENCES,
        SERVER_BATCH_TOKEN_BUDGET,
        SERVER_PHYSICAL_KV_BLOCKS,
        SERVER_MAX_ACTIVE_SEQUENCES,
        SERVER_PHYSICAL_KV_BLOCKS,
    )?;
    Ok(PreparedLlamaBatchExecutorConfig::new(metadata, forward)
        .with_separate_residual_norm()
        .with_iteration_batch_completion()
        .with_synchronous_metadata()
        .with_fixed_maximum_shape()
        .with_reduction_profile(LlamaReductionProfile::CanonicalV1)
        .with_native_bf16_paged_split_gqa_d128_two_stage())
}

fn decode_bf16_scalar(bytes: &[u8]) -> f32 {
    assert_eq!(bytes.len(), BF16_BYTES);
    f32::from_bits(u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])) << 16)
}

fn bf16_le_bytes(native: &[u8]) -> TestResult<Vec<u8>> {
    if native.is_empty() || native.len() % BF16_BYTES != 0 {
        return Err("native BF16 row has an invalid byte length".into());
    }
    let mut output = Vec::with_capacity(native.len());
    for lane in native.chunks_exact(BF16_BYTES) {
        output.extend_from_slice(&u16::from_ne_bytes([lane[0], lane[1]]).to_le_bytes());
    }
    Ok(output)
}

fn top_k(logits: &[u8], count: usize) -> TestResult<Vec<u32>> {
    if logits.len() % BF16_BYTES != 0 || count > logits.len() / BF16_BYTES {
        return Err("invalid BF16 top-k row shape".into());
    }
    let values: Vec<_> = logits
        .chunks_exact(BF16_BYTES)
        .map(decode_bf16_scalar)
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("native logit row contains a non-finite BF16 value".into());
    }
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    let mut output = Vec::with_capacity(indices.len());
    for index in indices {
        output.push(u32::try_from(index)?);
    }
    Ok(output)
}

fn logit_at(logits: &[u8], token_id: u32) -> TestResult<f32> {
    let start = usize::try_from(token_id)?
        .checked_mul(BF16_BYTES)
        .ok_or("logit offset overflow")?;
    let lane = logits
        .get(start..start + BF16_BYTES)
        .ok_or("token outside logits")?;
    Ok(decode_bf16_scalar(lane))
}

fn overlap_count(left: &[u32], right: &[u32]) -> usize {
    let left: BTreeSet<_> = left.iter().copied().collect();
    let right: BTreeSet<_> = right.iter().copied().collect();
    left.intersection(&right).count()
}

fn abort_scheduler_iteration(
    scheduler: &mut Scheduler,
    iteration_id: IterationId,
    abort: ExecutionAbort,
    now_ns: u64,
    reason: &'static str,
) -> TestResult {
    let updates = scheduler
        .abort_iteration(iteration_id, abort, now_ns)
        .map_err(|error| format!("{reason}: scheduler abort failed: {error}"))?;
    if !updates.settlement_failures().is_empty() {
        return Err(format!(
            "{reason}: scheduler abort had settlement failures: {:?}",
            updates.settlement_failures()
        )
        .into());
    }
    if !updates.token_events().is_empty() {
        return Err(format!("{reason}: scheduler abort unexpectedly emitted tokens").into());
    }
    Ok(())
}

fn abort_downloaded_iteration(
    scheduler: &mut Scheduler,
    downloaded: &DownloadedLlamaIteration,
    now_ns: u64,
    reason: &'static str,
) -> TestResult {
    let (iteration_id, abort) = downloaded.abort_data();
    abort_scheduler_iteration(scheduler, iteration_id, abort, now_ns, reason)
}

fn close_trace_resources(
    mut scheduler: Scheduler,
    timer: LlamaIterationCudaTimer,
    executor: PreparedLlamaBatchExecutor,
    mut stream: CudaStream,
    context: CudaContext,
    now_ns: u64,
    require_empty_scheduler_close: bool,
) -> TestResult {
    let mut failures = Vec::new();
    if let Some(iteration_id) = scheduler.inflight_iteration_id() {
        match stream.synchronize() {
            Ok(()) => {
                if let Err(error) = abort_scheduler_iteration(
                    &mut scheduler,
                    iteration_id,
                    ExecutionAbort::DeviceQuiescedMutationUnknown,
                    now_ns,
                    "trace cleanup",
                ) {
                    failures.push(error.to_string());
                }
            }
            Err(error) => failures.push(format!(
                "trace cleanup could not synchronize an in-flight iteration: {error}"
            )),
        }
    }
    match scheduler.close(now_ns, None) {
        Ok(close) => {
            if !close.settlement_failures().is_empty() {
                failures.push(format!(
                    "scheduler close had settlement failures: {:?}",
                    close.settlement_failures()
                ));
            }
            if require_empty_scheduler_close && !close.completions().is_empty() {
                failures.push(
                    "successful trace scheduler close unexpectedly emitted completions".to_owned(),
                );
            }
        }
        Err(error) => failures.push(format!("scheduler close failed: {}", error.error())),
    }
    if let Err(error) = timer.close() {
        failures.push(format!("trace timer close failed: {error}"));
    }
    if let Err(error) = executor.close() {
        failures.push(format!("trace executor close failed: {error}"));
    }
    if let Err(error) = stream.close() {
        failures.push(format!("trace stream close failed: {error}"));
    }
    if let Err(error) = close_context(context) {
        failures.push(format!("trace context close failed: {error}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
    }
}

fn trace_mode(
    model: &LoadedModel,
    workload: &ServingWorkload,
    hf: &HfCacheOffOracle,
    mode: LlamaProjectionBiasMode,
) -> TestResult<ModeTrace> {
    let (context, mut stream) = first_context()?;
    let mut executor =
        PreparedLlamaBatchExecutor::prepare(model, &context, &mut stream, executor_config(mode)?)?;
    assert_eq!(executor.batch_token_budget(), SERVER_BATCH_TOKEN_BUDGET);
    assert_eq!(executor.projection_bias_backend_id(), mode.id());
    assert_eq!(
        executor.decode_attention_implementation_id(),
        NATIVE_D128_BACKEND_ID
    );
    assert_eq!(
        executor.config().reduction_profile(),
        LlamaReductionProfile::CanonicalV1
    );
    assert!(executor.config().reduction_profile_is_coherent());
    let mut scheduler = Scheduler::new(scheduler_config(), executor.kv_layout())?;
    let request = scheduler.submit(
        RequestDescriptor::new(workload.prompt_token_ids.clone(), TRACE_OUTPUT_ROWS),
        0,
    )?;
    let mut timer = LlamaIterationCudaTimer::prepare(&context)?;
    let mut now_ns = 1_u64;
    let run = (|| -> TestResult<ModeTrace> {
        let mut rows = Vec::with_capacity(TRACE_OUTPUT_ROWS);
        let mut prefill_iteration_count = 0_usize;
        let mut decode_iteration_count = 0_usize;
        let mut terminal_token_ids = None;

        while rows.len() < TRACE_OUTPUT_ROWS {
            let planning = scheduler.plan_iteration(now_ns)?;
            assert!(planning.completions().is_empty());
            let plan = planning
                .plan()
                .ok_or("trace scheduler unexpectedly has no plan")?;
            assert_eq!(plan.batch_size(), 1);
            let (kind, target_logical_length) = if let Some(item) = plan.prefill_items().first() {
                assert!(plan.decode_items().is_empty());
                assert_eq!(plan.prefill_items().len(), 1);
                assert_eq!(item.input_tokens().len(), SERVER_PREFILL_CHUNK_TOKENS);
                assert_eq!(plan.total_tokens(), SERVER_PREFILL_CHUNK_TOKENS);
                prefill_iteration_count += 1;
                ("prefill", item.target_logical_length())
            } else {
                assert_eq!(plan.decode_items().len(), 1);
                let item = &plan.decode_items()[0];
                assert_eq!(item.input_tokens().len(), 1);
                assert_eq!(plan.total_tokens(), 1);
                decode_iteration_count += 1;
                ("decode", item.target_logical_length())
            };
            let (downloaded, timing) =
                match execute_llama_iteration_timed(plan, &mut executor, &mut stream, &mut timer) {
                    Ok(execution) => execution,
                    Err(error) => {
                        if let Some((iteration_id, abort)) = error.abort_data() {
                            abort_scheduler_iteration(
                                &mut scheduler,
                                iteration_id,
                                abort,
                                now_ns,
                                "trace execution failure",
                            )?;
                        }
                        return Err(format!("trace execution failed: {error}").into());
                    }
                };
            let trace_step = rows.len();
            let staged = match (|| -> TestResult<Option<TraceRow>> {
                if downloaded.output_count() == 0 {
                    assert_eq!(kind, "prefill");
                    assert!(plan.output_slots().is_empty());
                    return Ok(None);
                }
                assert_eq!(downloaded.output_count(), 1);
                assert_eq!(plan.output_slots().len(), 1);
                let native = downloaded.logits_bf16_native();
                assert_eq!(native.len(), EXPECTED_VOCABULARY_SIZE * BF16_BYTES);
                let addressable_len = EXPECTED_ADDRESSABLE_TOKEN_COUNT * BF16_BYTES;
                let hf_step = hf
                    .steps
                    .get(trace_step)
                    .ok_or("HF trace has too few steps")?;
                let top_token_ids = top_k(&native[..addressable_len], TOP_K)?;
                let top_values = top_token_ids
                    .iter()
                    .map(|&token| logit_at(native, token))
                    .collect::<TestResult<Vec<_>>>()?;
                Ok(Some(TraceRow {
                    step: trace_step,
                    scheduler_iteration_id: downloaded.iteration_id().get(),
                    kind,
                    iteration_input_token_count: plan.total_tokens(),
                    target_logical_length,
                    row_bf16_le_sha256: sha256_hex(&bf16_le_bytes(native)?),
                    addressable_bf16_le_sha256: sha256_hex(&bf16_le_bytes(
                        &native[..addressable_len],
                    )?),
                    raw_argmax_token_id: top_k(native, 1)?[0],
                    selected_token_id: top_token_ids[0],
                    selected_logit: top_values[0],
                    top_token_ids,
                    top_values,
                    hf: HfStep {
                        step: hf_step.step,
                        input_token_count: hf_step.input_token_count,
                        attention_mask_token_count: hf_step.attention_mask_token_count,
                        position_start: hf_step.position_start,
                        position_end: hf_step.position_end,
                        raw_logit_sha256: hf_step.raw_logit_sha256.clone(),
                        raw_argmax_token_id: hf_step.raw_argmax_token_id,
                        selected_token_id: hf_step.selected_token_id,
                        selected_logit: hf_step.selected_logit,
                        top_token_ids: hf_step.top_token_ids.clone(),
                        top_values: hf_step.top_values.clone(),
                    },
                }))
            })() {
                Ok(staged) => staged,
                Err(error) => {
                    abort_downloaded_iteration(
                        &mut scheduler,
                        &downloaded,
                        now_ns,
                        "trace row staging failure",
                    )?;
                    return Err(format!("trace row staging failed: {error}").into());
                }
            };
            let samples = match staged.as_ref() {
                Some(_) => vec![SampledIterationToken::new(
                    hf.selected_token_ids[trace_step],
                    false,
                )],
                None => Vec::new(),
            };
            let result = match downloaded.into_result(&samples, timing) {
                Ok(result) => result,
                Err(error) => {
                    let (iteration_id, abort) = error.abort_data();
                    abort_scheduler_iteration(
                        &mut scheduler,
                        iteration_id,
                        abort,
                        now_ns,
                        "trace result construction failure",
                    )?;
                    return Err(format!("trace result construction failed: {error}").into());
                }
            };
            let iteration_id = result.iteration_id();
            let updates = match scheduler.complete_iteration(&result, now_ns) {
                Ok(updates) => updates,
                Err(error) => {
                    if scheduler.inflight_iteration_id() == Some(iteration_id) {
                        abort_scheduler_iteration(
                            &mut scheduler,
                            iteration_id,
                            ExecutionAbort::DeviceQuiescedMutationUnknown,
                            now_ns,
                            "trace scheduler commit failure",
                        )?;
                    }
                    return Err(format!("trace scheduler commit failed: {error}").into());
                }
            };
            assert!(updates.settlement_failures().is_empty());
            if !updates.completions().is_empty() {
                assert_eq!(updates.completions().len(), 1);
                assert!(terminal_token_ids.is_none());
                assert_eq!(updates.completions()[0].request_id(), request.request_id());
                assert_eq!(
                    updates.completions()[0].reason(),
                    RequestFinishReason::Length
                );
                terminal_token_ids = Some(updates.completions()[0].generated_token_ids().to_vec());
            }
            if let Some(staged) = staged {
                assert_eq!(updates.token_events().len(), 1);
                assert_eq!(updates.token_events()[0].request_id(), request.request_id());
                assert_eq!(
                    updates.token_events()[0].token_id(),
                    hf.selected_token_ids[trace_step]
                );
                rows.push(staged);
            } else {
                assert!(updates.token_events().is_empty());
            }
            now_ns = now_ns.checked_add(1).ok_or("trace clock overflow")?;
        }
        assert_eq!(prefill_iteration_count, 2_048 / SERVER_PREFILL_CHUNK_TOKENS);
        assert_eq!(decode_iteration_count, TRACE_OUTPUT_ROWS - 1);
        assert!(scheduler.inflight_iteration_id().is_none());
        assert_eq!(
            terminal_token_ids.as_deref(),
            Some(&hf.selected_token_ids[..TRACE_OUTPUT_ROWS])
        );
        Ok(ModeTrace {
            projection_bias_backend: mode.id(),
            rows,
            prefill_iteration_count,
            decode_iteration_count,
        })
    })();
    let cleanup = close_trace_resources(
        scheduler,
        timer,
        executor,
        stream,
        context,
        now_ns,
        run.is_ok(),
    );
    match (run, cleanup) {
        (Ok(trace), Ok(())) => Ok(trace),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => Err(format!(
            "trace execution failed: {run_error}; trace cleanup also failed: {cleanup_error}"
        )
        .into()),
    }
}

fn trace_json(mode: &ModeTrace) -> Value {
    let rows: Vec<_> = mode
        .rows
        .iter()
        .map(|row| {
            json!({
                "step": row.step,
                "scheduler_iteration_id": row.scheduler_iteration_id,
                "kind": row.kind,
                "iteration_input_token_count": row.iteration_input_token_count,
                "target_logical_length": row.target_logical_length,
                "row_bf16_le_sha256": row.row_bf16_le_sha256,
                "addressable_bf16_le_sha256": row.addressable_bf16_le_sha256,
                "raw_argmax_token_id": row.raw_argmax_token_id,
                "selected_token_id": row.selected_token_id,
                "selected_logit_bf16_as_f32": row.selected_logit,
                "top_token_ids": row.top_token_ids,
                "top_values_f32": row.top_values,
                "hf_cache_off": {
                    "step": row.hf.step,
                    "input_token_count": row.hf.input_token_count,
                    "attention_mask_token_count": row.hf.attention_mask_token_count,
                    "position_start": row.hf.position_start,
                    "position_end": row.hf.position_end,
                    "logits_bf16_le_sha256": row.hf.raw_logit_sha256,
                    "raw_argmax_token_id": row.hf.raw_argmax_token_id,
                    "selected_token_id": row.hf.selected_token_id,
                    "selected_logit_bf16_as_f32": row.hf.selected_logit,
                    "top_token_ids": row.hf.top_token_ids,
                    "top_values_f32": row.hf.top_values,
                    "raw_hash_matches": row.row_bf16_le_sha256 == row.hf.raw_logit_sha256,
                    "selected_token_matches": row.selected_token_id == row.hf.selected_token_id,
                    "raw_argmax_matches": row.raw_argmax_token_id == row.hf.raw_argmax_token_id,
                    "top32_order_matches_tie_sensitive": row.top_token_ids == row.hf.top_token_ids,
                    "top32_set_overlap": overlap_count(&row.top_token_ids, &row.hf.top_token_ids),
                }
            })
        })
        .collect();
    let first_selected_token_mismatch = mode
        .rows
        .iter()
        .find_map(|row| (row.selected_token_id != row.hf.selected_token_id).then_some(row.step));
    json!({
        "projection_bias_backend": mode.projection_bias_backend,
        "prefill_iteration_count": mode.prefill_iteration_count,
        "decode_iteration_count": mode.decode_iteration_count,
        "first_hf_cache_off_selected_token_mismatch": first_selected_token_mismatch,
        "rows": rows,
    })
}

#[test]
#[ignore = "remote-only Qwen2.5-3B C8/M32 native-D128 teacher-forced numerical trace"]
fn qwen3b_native_d128_teacher_forced_trace_is_scheduler_committed() -> TestResult {
    let workload = load_workload()?;
    let hf = load_hf_cache_off_oracle()?;
    assert_eq!(workload.output_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
    assert_eq!(hf.selected_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
    let model = load_model()?;
    let strict = trace_mode(
        &model,
        &workload,
        &hf,
        LlamaProjectionBiasMode::StrictStagedV1,
    )?;
    let fused = trace_mode(
        &model,
        &workload,
        &hf,
        LlamaProjectionBiasMode::CublasLtBiasEpilogueExperimentalV1,
    )?;
    let document = json!({
        "schema_version": "riley.qwen3b-native-d128-teacher-forced-logit-trace.v1",
        "artifact_kind": "qwen2.5-3b-native-d128-scheduler-committed-teacher-forced-logit-trace",
        "performance_claim_eligible": false,
        "trace_contract": {
            "scheduler_executor_replay": true,
            "http_transport_replayed": false,
            "sampling": "teacher-forced-hf-cache-off-selected-token-after-scheduler-commit",
            "hf_reference": "full-prefix-cache-off; cache-on excluded because its recorded decode position starts at 2049",
            "projection_bias_modes": [strict.projection_bias_backend, fused.projection_bias_backend],
            "native_d128_backend": NATIVE_D128_BACKEND_ID,
            "max_active_sequences": SERVER_MAX_ACTIVE_SEQUENCES,
            "batch_token_budget": SERVER_BATCH_TOKEN_BUDGET,
            "prefill_chunk_tokens": SERVER_PREFILL_CHUNK_TOKENS,
            "max_sequence_tokens": SERVER_MAX_SEQUENCE_TOKENS,
            "physical_kv_blocks": SERVER_PHYSICAL_KV_BLOCKS,
            "trace_output_rows": TRACE_OUTPUT_ROWS,
            "submitted_request_count": 1,
            "scheduled_batch_size": 1,
            "capacity_configuration_only": true,
            "request_max_new_tokens": TRACE_OUTPUT_ROWS,
            "execution_graph_policy": "disabled",
            "residual_rmsnorm": "separate",
            "execution_completion": "iteration-batch",
            "metadata_transport": "synchronous",
            "batch_shape_policy": "fixed-maximum",
            "reduction_profile": "canonical-v1",
            "top32_order_comparison": "observational-tie-sensitive",
        },
        "model": {"id": QWEN3B_MODEL_ID, "revision": QWEN3B_REVISION},
        "workload": {
            "sha256": QWEN3B_SERVING_WORKLOAD_SHA256,
            "case": QWEN3B_WORKLOAD_CASE,
            "prompt_token_count": workload.prompt_token_ids.len(),
            "hf_teacher_token_ids": &hf.selected_token_ids[..TRACE_OUTPUT_ROWS],
        },
        "hf_generation_oracle": {
            "schema_version": HF_GENERATION_ORACLE_SCHEMA,
            "artifact_sha256": hf.artifact_sha256,
            "mode": HF_CACHE_OFF_MODE,
        },
        "modes": [trace_json(&strict), trace_json(&fused)],
    });
    println!(
        "RILEY_QWEN3B_NATIVE_D128_LOGIT_TRACE={}",
        serde_json::to_string(&document)?
    );
    Ok(())
}
