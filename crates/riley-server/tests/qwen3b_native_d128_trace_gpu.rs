//! Remote-only Qwen2.5-3B teacher-forced native-D128 numerical trace.
//!
//! This test deliberately replays the production scheduler/executor seam rather
//! than opening an HTTP listener. It consumes an externally validated offline
//! Hugging Face teacher-forced artifact: cache-off derives the forced teacher
//! stream, while cache-on is the scheduler reference. Cache-on row zero is the
//! P2048 prefill; every later row consumes `teacher_token_ids[step - 1]` at
//! absolute position `2048 + step - 1`.
//!
//! The test binds the manifest and both raw-BF16 safetensors sidecars by
//! basename and complete SHA-256, then rederives each row's numerical metadata
//! from canonical little-endian BF16 before it emits one compact JSON receipt.
//! The receipt is a numerical diagnostic, never a serving performance result
//! or a C02 generation-audit record.

#![cfg(feature = "cuda")]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

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
const HF_TEACHER_FORCED_ORACLE_SCHEMA: &str = "riley.qwen3b-hf-eager-teacher-forced-generation.v1";
const HF_TEACHER_FORCED_ARTIFACT_KIND: &str =
    "qwen2.5-3b-hf-eager-bf16-p2048-teacher-forced-generation";
const HF_CACHE_OFF_MODE: &str = "cache-off";
const HF_CACHE_ON_MODE: &str = "cache-on";
const HF_SIDECAR_TENSOR_KEY: &str = "teacher_forced/logits";
const MAX_SAFETENSORS_HEADER_BYTES: usize = 1_048_576;
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

#[derive(Debug, Clone)]
struct HfLogits {
    raw_logit_sha256: String,
    addressable_logit_sha256: String,
    non_addressable_logit_sha256: String,
    raw_argmax_token_id: u32,
    selected_token_id: u32,
    cache_off_teacher_token_id: u32,
    selection_matches_cache_off_teacher: bool,
    selected_logit: f32,
    top_token_ids: Vec<u32>,
    top_values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct HfStep {
    step: usize,
    call_input_token_count: usize,
    call_input_token_ids_le_sha256: String,
    context_token_count: usize,
    attention_mask_token_count: usize,
    position_start: usize,
    position_end: usize,
    teacher_input_token_id: Option<u32>,
    cache_length_before: Option<usize>,
    cache_length_after: Option<usize>,
    logits: HfLogits,
}

#[derive(Debug)]
struct HfSidecarBinding {
    basename: String,
    sha256: String,
    bytes: Vec<u8>,
    tensor_data_start: usize,
    tensor_data_end: usize,
}

#[derive(Debug)]
struct HfTeacherForcedOracle {
    artifact_sha256: String,
    teacher_token_ids: Vec<u32>,
    cache_off_steps: Vec<HfStep>,
    cache_on_steps: Vec<HfStep>,
    cache_off_sidecar: HfSidecarBinding,
    cache_on_sidecar: HfSidecarBinding,
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
    hf_cache_off: HfStep,
    hf_cache_on: HfStep,
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

fn json_sha256(value: &Value, label: &str) -> TestResult<String> {
    let value = value
        .as_str()
        .ok_or_else(|| format!("{label} must be a SHA-256 string"))?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!("{label} must be 64 lowercase hexadecimal characters").into());
    }
    Ok(value.to_owned())
}

fn json_optional_u32(value: &Value, label: &str) -> TestResult<Option<u32>> {
    if value.is_null() {
        Ok(None)
    } else {
        value
            .as_u64()
            .ok_or_else(|| format!("{label} must be an integer or null"))
            .and_then(|value| {
                u32::try_from(value).map_err(|error| format!("{label} is outside u32: {error}"))
            })
            .map(Some)
            .map_err(Into::into)
    }
}

fn json_optional_usize(value: &Value, label: &str) -> TestResult<Option<usize>> {
    if value.is_null() {
        Ok(None)
    } else {
        value
            .as_u64()
            .ok_or_else(|| format!("{label} must be an integer or null"))
            .and_then(|value| {
                usize::try_from(value).map_err(|error| format!("{label} is outside usize: {error}"))
            })
            .map(Some)
            .map_err(Into::into)
    }
}

fn token_ids_sha256(token_ids: &[u32]) -> String {
    let bytes: Vec<_> = token_ids
        .iter()
        .flat_map(|token_id| token_id.to_le_bytes())
        .collect();
    sha256_hex(&bytes)
}

fn parse_hf_logits(
    value: &Value,
    step: usize,
    teacher_token: u32,
    require_teacher_match: bool,
) -> TestResult<HfLogits> {
    require_exact_fields(
        value,
        &[
            "addressable_bf16_le_bytes",
            "addressable_bf16_le_sha256",
            "cache_off_teacher_token_id",
            "canonical_byte_order",
            "dtype",
            "element_count",
            "non_addressable_bf16_le_bytes",
            "non_addressable_bf16_le_sha256",
            "raw_argmax_token_id",
            "raw_bf16_le_bytes",
            "raw_bf16_le_sha256",
            "selected_logit_bf16_as_f32",
            "selected_token_id",
            "selection_matches_cache_off_teacher",
            "sidecar_row_index",
            "top_k",
            "top_token_ids",
            "top_values_bf16_as_f32",
        ],
        "HF teacher-forced logits",
    )?;
    assert_eq!(
        json_usize(&value["sidecar_row_index"], "HF sidecar row index")?,
        step
    );
    assert_eq!(value["dtype"].as_str(), Some("bfloat16"));
    assert_eq!(
        value["element_count"].as_u64(),
        Some(EXPECTED_VOCABULARY_SIZE as u64)
    );
    assert_eq!(
        value["canonical_byte_order"].as_str(),
        Some("little-endian-u16")
    );
    assert_eq!(
        value["raw_bf16_le_bytes"].as_u64(),
        Some((EXPECTED_VOCABULARY_SIZE * BF16_BYTES) as u64)
    );
    assert_eq!(
        value["addressable_bf16_le_bytes"].as_u64(),
        Some((EXPECTED_ADDRESSABLE_TOKEN_COUNT * BF16_BYTES) as u64)
    );
    assert_eq!(
        value["non_addressable_bf16_le_bytes"].as_u64(),
        Some(((EXPECTED_VOCABULARY_SIZE - EXPECTED_ADDRESSABLE_TOKEN_COUNT) * BF16_BYTES) as u64)
    );
    assert_eq!(value["top_k"].as_u64(), Some(TOP_K as u64));
    let raw_logit_sha256 = json_sha256(&value["raw_bf16_le_sha256"], "HF raw logit hash")?;
    let addressable_logit_sha256 = json_sha256(
        &value["addressable_bf16_le_sha256"],
        "HF addressable logit hash",
    )?;
    let non_addressable_logit_sha256 = json_sha256(
        &value["non_addressable_bf16_le_sha256"],
        "HF non-addressable logit hash",
    )?;
    let raw_argmax_token_id = json_u32(&value["raw_argmax_token_id"], "HF raw argmax token")?;
    assert!(usize::try_from(raw_argmax_token_id).is_ok_and(|id| id < EXPECTED_VOCABULARY_SIZE));
    let selected_token_id = json_u32(&value["selected_token_id"], "HF selected token")?;
    assert!(
        usize::try_from(selected_token_id).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    );
    let cache_off_teacher_token_id = json_u32(
        &value["cache_off_teacher_token_id"],
        "HF cache-off teacher token",
    )?;
    assert_eq!(cache_off_teacher_token_id, teacher_token);
    let selection_matches_cache_off_teacher = value["selection_matches_cache_off_teacher"]
        .as_bool()
        .ok_or("HF cache-off teacher selection flag must be a boolean")?;
    assert_eq!(
        selection_matches_cache_off_teacher,
        selected_token_id == teacher_token
    );
    if require_teacher_match {
        assert_eq!(selected_token_id, teacher_token);
    }
    let selected_logit = json_f32(
        &value["selected_logit_bf16_as_f32"],
        "HF selected BF16 logit",
    )?;
    let top_token_ids = json_u32_array(&value["top_token_ids"], "HF top token IDs")?;
    let top_values = json_f32_array(&value["top_values_bf16_as_f32"], "HF top BF16 values")?;
    assert_eq!(top_token_ids.len(), TOP_K);
    assert_eq!(top_values.len(), TOP_K);
    assert!(top_token_ids.iter().all(|&token| {
        usize::try_from(token).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    }));
    assert_eq!(
        top_token_ids.len(),
        top_token_ids.iter().collect::<BTreeSet<_>>().len()
    );
    assert!(top_values.windows(2).all(|window| window[0] >= window[1]));
    assert_eq!(top_token_ids[0], selected_token_id);
    assert_eq!(top_values[0], selected_logit);
    Ok(HfLogits {
        raw_logit_sha256,
        addressable_logit_sha256,
        non_addressable_logit_sha256,
        raw_argmax_token_id,
        selected_token_id,
        cache_off_teacher_token_id,
        selection_matches_cache_off_teacher,
        selected_logit,
        top_token_ids,
        top_values,
    })
}

fn parse_hf_step(
    value: &Value,
    mode: &str,
    step: usize,
    teacher_token_ids: &[u32],
    workload: &ServingWorkload,
) -> TestResult<HfStep> {
    require_exact_fields(
        value,
        &[
            "attention_mask_token_count",
            "cache_length_after",
            "cache_length_before",
            "call_input_token_count",
            "call_input_token_ids_le_u32_sha256",
            "context_token_count",
            "logits",
            "position_end",
            "position_start",
            "step",
            "teacher_input_token_id",
        ],
        "HF teacher-forced step",
    )?;
    assert_eq!(json_usize(&value["step"], "HF step")?, step);
    let teacher_token = *teacher_token_ids
        .get(step)
        .ok_or("HF teacher-forced trace has too few teacher tokens")?;
    let context_token_count = workload
        .prompt_token_ids
        .len()
        .checked_add(step)
        .ok_or("HF context length overflow")?;
    let (
        expected_call_input_token_ids,
        expected_position_start,
        expected_position_end,
        expected_teacher_input_token_id,
        expected_cache_length_before,
        expected_cache_length_after,
    ) = if mode == HF_CACHE_OFF_MODE {
        let mut input = workload.prompt_token_ids.clone();
        input.extend_from_slice(&teacher_token_ids[..step]);
        (input, 0, context_token_count - 1, None, None, None)
    } else {
        assert_eq!(mode, HF_CACHE_ON_MODE);
        if step == 0 {
            (
                workload.prompt_token_ids.clone(),
                0,
                workload.prompt_token_ids.len() - 1,
                None,
                Some(0),
                Some(workload.prompt_token_ids.len()),
            )
        } else {
            let input_token = teacher_token_ids[step - 1];
            (
                vec![input_token],
                context_token_count - 1,
                context_token_count - 1,
                Some(input_token),
                Some(context_token_count - 1),
                Some(context_token_count),
            )
        }
    };
    assert_eq!(
        json_usize(&value["call_input_token_count"], "HF call input count")?,
        expected_call_input_token_ids.len()
    );
    assert_eq!(
        json_sha256(
            &value["call_input_token_ids_le_u32_sha256"],
            "HF call input token hash",
        )?,
        token_ids_sha256(&expected_call_input_token_ids)
    );
    assert_eq!(
        json_usize(&value["context_token_count"], "HF context token count")?,
        context_token_count
    );
    assert_eq!(
        json_usize(
            &value["attention_mask_token_count"],
            "HF attention-mask token count",
        )?,
        context_token_count
    );
    assert_eq!(
        json_usize(&value["position_start"], "HF position start")?,
        expected_position_start
    );
    assert_eq!(
        json_usize(&value["position_end"], "HF position end")?,
        expected_position_end
    );
    assert_eq!(
        json_optional_u32(&value["teacher_input_token_id"], "HF teacher input token")?,
        expected_teacher_input_token_id
    );
    assert_eq!(
        json_optional_usize(&value["cache_length_before"], "HF cache length before")?,
        expected_cache_length_before
    );
    assert_eq!(
        json_optional_usize(&value["cache_length_after"], "HF cache length after")?,
        expected_cache_length_after
    );
    Ok(HfStep {
        step,
        call_input_token_count: expected_call_input_token_ids.len(),
        call_input_token_ids_le_sha256: token_ids_sha256(&expected_call_input_token_ids),
        context_token_count,
        attention_mask_token_count: context_token_count,
        position_start: expected_position_start,
        position_end: expected_position_end,
        teacher_input_token_id: expected_teacher_input_token_id,
        cache_length_before: expected_cache_length_before,
        cache_length_after: expected_cache_length_after,
        logits: parse_hf_logits(
            &value["logits"],
            step,
            teacher_token,
            mode == HF_CACHE_OFF_MODE,
        )?,
    })
}

fn regular_non_symlink_file(path: &Path, label: &str) -> TestResult<PathBuf> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {path:?}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file").into());
    }
    Ok(path.canonicalize()?)
}

fn parse_hf_sidecar_tensor(bytes: &[u8]) -> TestResult<(usize, usize)> {
    if bytes.len() < 8 {
        return Err("HF sidecar is shorter than its safetensors header prefix".into());
    }
    let header_bytes = usize::try_from(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))?;
    if header_bytes == 0 || header_bytes > MAX_SAFETENSORS_HEADER_BYTES {
        return Err("HF sidecar safetensors header length differs".into());
    }
    let tensor_data_start = 8_usize
        .checked_add(header_bytes)
        .ok_or("HF sidecar safetensors header length overflows")?;
    if tensor_data_start > bytes.len() {
        return Err("HF sidecar safetensors header is truncated".into());
    }
    let header: Value = serde_json::from_slice(&bytes[8..tensor_data_start])?;
    require_exact_fields(
        &header,
        &[HF_SIDECAR_TENSOR_KEY],
        "HF sidecar safetensors header",
    )?;
    let tensor = &header[HF_SIDECAR_TENSOR_KEY];
    require_exact_fields(
        tensor,
        &["data_offsets", "dtype", "shape"],
        "HF sidecar logits tensor",
    )?;
    assert_eq!(tensor["dtype"].as_str(), Some("BF16"));
    assert_eq!(
        tensor["shape"],
        json!([SERVER_MAX_OUTPUT_TOKENS, EXPECTED_VOCABULARY_SIZE])
    );
    let offsets = tensor["data_offsets"]
        .as_array()
        .ok_or("HF sidecar logits offsets must be an array")?;
    if offsets.len() != 2 {
        return Err("HF sidecar logits offsets must have two elements".into());
    }
    let offset_start = usize::try_from(
        offsets[0]
            .as_u64()
            .ok_or("HF sidecar logits offset start must be unsigned")?,
    )?;
    let offset_end = usize::try_from(
        offsets[1]
            .as_u64()
            .ok_or("HF sidecar logits offset end must be unsigned")?,
    )?;
    let expected_data_bytes = SERVER_MAX_OUTPUT_TOKENS
        .checked_mul(EXPECTED_VOCABULARY_SIZE)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .ok_or("HF sidecar logits byte count overflows")?;
    let tensor_data_end = tensor_data_start
        .checked_add(offset_end)
        .ok_or("HF sidecar logits range overflows")?;
    if offset_start != 0 || offset_end != expected_data_bytes || tensor_data_end != bytes.len() {
        return Err("HF sidecar logits range differs".into());
    }
    Ok((tensor_data_start, tensor_data_end))
}

fn parse_hf_sidecar(
    value: &Value,
    sidecar_path: &Path,
    label: &'static str,
) -> TestResult<HfSidecarBinding> {
    require_exact_fields(
        value,
        &[
            "canonical_byte_order",
            "dtype",
            "format",
            "path",
            "raw_bf16_le_bytes",
            "sha256",
            "shape",
            "tensor_count",
            "tensor_key",
        ],
        label,
    )?;
    let basename = value["path"]
        .as_str()
        .ok_or("HF sidecar basename must be a string")?
        .to_owned();
    assert!(
        Path::new(&basename)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == basename)
    );
    assert!(basename.ends_with(".safetensors"));
    assert_eq!(value["format"].as_str(), Some("safetensors"));
    assert_eq!(value["tensor_key"].as_str(), Some(HF_SIDECAR_TENSOR_KEY));
    assert_eq!(value["tensor_count"].as_u64(), Some(1));
    assert_eq!(value["dtype"].as_str(), Some("bfloat16"));
    assert_eq!(
        value["canonical_byte_order"].as_str(),
        Some("little-endian-u16")
    );
    assert_eq!(
        value["shape"],
        json!([SERVER_MAX_OUTPUT_TOKENS, EXPECTED_VOCABULARY_SIZE])
    );
    let expected_data_bytes = SERVER_MAX_OUTPUT_TOKENS
        .checked_mul(EXPECTED_VOCABULARY_SIZE)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .ok_or("HF sidecar logits byte count overflows")?;
    assert_eq!(
        value["raw_bf16_le_bytes"].as_u64(),
        Some(u64::try_from(expected_data_bytes)?)
    );
    let sha256 = json_sha256(&value["sha256"], "HF sidecar SHA-256")?;
    let actual_sidecar = regular_non_symlink_file(sidecar_path, label)?;
    let actual_basename = actual_sidecar
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or("HF sidecar environment path lacks a UTF-8 basename")?;
    assert_eq!(basename, actual_basename);
    let bytes = fs::read(&actual_sidecar)?;
    assert_eq!(sha256_hex(&bytes), sha256);
    let (tensor_data_start, tensor_data_end) = parse_hf_sidecar_tensor(&bytes)?;
    Ok(HfSidecarBinding {
        basename,
        sha256,
        bytes,
        tensor_data_start,
        tensor_data_end,
    })
}

fn load_hf_teacher_forced_oracle(workload: &ServingWorkload) -> TestResult<HfTeacherForcedOracle> {
    let path = required_path("RILEY_QWEN_HF_TEACHER_FORCED_ORACLE");
    let cache_off_sidecar_path = required_path("RILEY_QWEN_HF_CACHE_OFF_SIDECAR");
    let cache_on_sidecar_path = required_path("RILEY_QWEN_HF_CACHE_ON_SIDECAR");
    let payload = fs::read(&path)?;
    let artifact_sha256 = sha256_hex(&payload);
    let document: Value = serde_json::from_slice(&payload)?;
    require_exact_fields(
        &document,
        &[
            "artifact_kind",
            "contract",
            "created_at",
            "generation",
            "model",
            "performance_claim_eligible",
            "producer",
            "provenance",
            "schema_version",
        ],
        "HF teacher-forced artifact",
    )?;
    assert_eq!(
        document["schema_version"].as_str(),
        Some(HF_TEACHER_FORCED_ORACLE_SCHEMA)
    );
    assert_eq!(
        document["artifact_kind"].as_str(),
        Some(HF_TEACHER_FORCED_ARTIFACT_KIND)
    );
    assert_eq!(
        document["performance_claim_eligible"].as_bool(),
        Some(false)
    );
    assert!(document["created_at"].as_str().is_some());
    assert!(document["model"].is_object());
    assert!(document["producer"].is_object());
    assert!(document["provenance"].is_object());
    let contract = &document["contract"];
    require_exact_fields(
        contract,
        &[
            "execution",
            "model_id",
            "model_revision",
            "teacher_forced_generation",
            "workload",
        ],
        "HF teacher-forced artifact contract",
    )?;
    assert_eq!(contract["model_id"].as_str(), Some(QWEN3B_MODEL_ID));
    assert_eq!(contract["model_revision"].as_str(), Some(QWEN3B_REVISION));
    assert!(contract["execution"].is_object());
    assert_eq!(
        contract["workload"],
        json!({
            "schema_version": QWEN3B_WORKLOAD_SCHEMA,
            "case": QWEN3B_WORKLOAD_CASE,
            "source_sha256": QWEN3B_SERVING_WORKLOAD_SHA256,
            "prompt_token_count": workload.prompt_token_ids.len(),
            "prompt_token_ids_le_u32_sha256": token_ids_sha256(&workload.prompt_token_ids),
            "workload_output_token_count": workload.output_token_ids.len(),
            "workload_output_token_ids_le_u32_sha256": token_ids_sha256(&workload.output_token_ids),
            "workload_output_token_ids_used_as_teacher": false,
        })
    );
    assert_eq!(
        contract["teacher_forced_generation"],
        json!({
            "fixed_output_token_count": SERVER_MAX_OUTPUT_TOKENS,
            "teacher_token_derivation": "cache-off-addressable-greedy-argmax",
            "teacher_token_source_mode": HF_CACHE_OFF_MODE,
            "addressable_token_count": EXPECTED_ADDRESSABLE_TOKEN_COUNT,
            "vocabulary_size": EXPECTED_VOCABULARY_SIZE,
            "cache_modes": [HF_CACHE_OFF_MODE, HF_CACHE_ON_MODE],
            "cache_on_first_decode_position": workload.prompt_token_ids.len(),
            "cache_on_decode_position_rule": "prompt_token_count+step-1",
            "cache_on_decode_input_rule": "teacher_token_ids[step-1]",
            "cache_position_argument": "omitted-transformers-5.15.1",
        })
    );
    let generation = &document["generation"];
    require_exact_fields(
        generation,
        &[
            "cache_off",
            "cache_on",
            "teacher_token_ids",
            "teacher_token_ids_le_u32_sha256",
        ],
        "HF teacher-forced generation",
    )?;
    let teacher_token_ids = json_u32_array(
        &generation["teacher_token_ids"],
        "HF teacher-forced teacher tokens",
    )?;
    assert_eq!(teacher_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
    assert!(teacher_token_ids.iter().all(|&token| {
        usize::try_from(token).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    }));
    assert_eq!(
        json_sha256(
            &generation["teacher_token_ids_le_u32_sha256"],
            "HF teacher-forced teacher token hash",
        )?,
        token_ids_sha256(&teacher_token_ids)
    );
    let cache_off = &generation["cache_off"];
    let cache_on = &generation["cache_on"];
    require_exact_fields(
        cache_off,
        &["mode", "sidecar", "steps"],
        "HF cache-off mode",
    )?;
    require_exact_fields(cache_on, &["mode", "sidecar", "steps"], "HF cache-on mode")?;
    assert_eq!(cache_off["mode"].as_str(), Some(HF_CACHE_OFF_MODE));
    assert_eq!(cache_on["mode"].as_str(), Some(HF_CACHE_ON_MODE));
    let cache_off_sidecar = parse_hf_sidecar(
        &cache_off["sidecar"],
        &cache_off_sidecar_path,
        "HF cache-off sidecar",
    )?;
    let cache_on_sidecar = parse_hf_sidecar(
        &cache_on["sidecar"],
        &cache_on_sidecar_path,
        "HF cache-on sidecar",
    )?;
    let cache_off_steps_value = cache_off["steps"].as_array().ok_or("HF cache-off steps")?;
    let cache_on_steps_value = cache_on["steps"].as_array().ok_or("HF cache-on steps")?;
    assert_eq!(cache_off_steps_value.len(), SERVER_MAX_OUTPUT_TOKENS);
    assert_eq!(cache_on_steps_value.len(), SERVER_MAX_OUTPUT_TOKENS);
    let cache_off_steps = cache_off_steps_value
        .iter()
        .enumerate()
        .map(|(step, value)| {
            parse_hf_step(value, HF_CACHE_OFF_MODE, step, &teacher_token_ids, workload)
        })
        .collect::<TestResult<Vec<_>>>()?;
    let cache_on_steps = cache_on_steps_value
        .iter()
        .enumerate()
        .map(|(step, value)| {
            parse_hf_step(value, HF_CACHE_ON_MODE, step, &teacher_token_ids, workload)
        })
        .collect::<TestResult<Vec<_>>>()?;
    verify_hf_sidecar_rows(&cache_off_sidecar, &cache_off_steps, "HF cache-off sidecar")?;
    verify_hf_sidecar_rows(&cache_on_sidecar, &cache_on_steps, "HF cache-on sidecar")?;
    Ok(HfTeacherForcedOracle {
        artifact_sha256,
        teacher_token_ids,
        cache_off_steps,
        cache_on_steps,
        cache_off_sidecar,
        cache_on_sidecar,
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

const BF16_EXPONENT_MASK: u16 = 0x7f80;
const BF16_SIGN_MASK: u16 = 0x8000;

fn bf16_numeric_sort_key(bits: u16) -> TestResult<u16> {
    if bits & BF16_EXPONENT_MASK == BF16_EXPONENT_MASK {
        return Err("BF16 logit row contains a non-finite value".into());
    }
    let bits = if bits == BF16_SIGN_MASK { 0 } else { bits };
    Ok(if bits & BF16_SIGN_MASK != 0 {
        !bits
    } else {
        bits | BF16_SIGN_MASK
    })
}

#[derive(Debug)]
struct Bf16LogitAnalysis {
    raw_argmax_token_id: u32,
    selected_token_id: u32,
    selected_logit: f32,
    top_token_ids: Vec<u32>,
    top_values: Vec<f32>,
}

fn bf16_entry_order(
    left_rank: u16,
    left_token_id: usize,
    right_rank: u16,
    right_token_id: usize,
) -> std::cmp::Ordering {
    right_rank
        .cmp(&left_rank)
        .then_with(|| left_token_id.cmp(&right_token_id))
}

fn decode_bf16_le_scalar(bytes: &[u8]) -> TestResult<f32> {
    if bytes.len() != BF16_BYTES {
        return Err("canonical BF16 scalar byte count differs".into());
    }
    let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
    if !value.is_finite() {
        return Err("canonical BF16 scalar is non-finite".into());
    }
    Ok(value)
}

fn analyze_hf_bf16_le_logits(raw: &[u8]) -> TestResult<Bf16LogitAnalysis> {
    let expected_bytes = EXPECTED_VOCABULARY_SIZE
        .checked_mul(BF16_BYTES)
        .ok_or("canonical BF16 row byte count overflows")?;
    if raw.len() != expected_bytes {
        return Err("canonical BF16 logit row byte count differs".into());
    }
    let mut raw_best: Option<(u16, usize)> = None;
    let mut selected_best: Option<(u16, usize)> = None;
    let mut top_entries: Vec<(u16, usize, u16)> = Vec::with_capacity(TOP_K);
    for (token_id, bytes) in raw.chunks_exact(BF16_BYTES).enumerate() {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let rank = bf16_numeric_sort_key(bits)?;
        if raw_best.is_none_or(|(best_rank, best_token_id)| {
            bf16_entry_order(rank, token_id, best_rank, best_token_id).is_lt()
        }) {
            raw_best = Some((rank, token_id));
        }
        if token_id >= EXPECTED_ADDRESSABLE_TOKEN_COUNT {
            continue;
        }
        if selected_best.is_none_or(|(best_rank, best_token_id)| {
            bf16_entry_order(rank, token_id, best_rank, best_token_id).is_lt()
        }) {
            selected_best = Some((rank, token_id));
        }
        if top_entries.len() < TOP_K {
            top_entries.push((rank, token_id, bits));
            if top_entries.len() == TOP_K {
                top_entries.sort_unstable_by(|left, right| {
                    bf16_entry_order(left.0, left.1, right.0, right.1)
                });
            }
        } else if bf16_entry_order(
            rank,
            token_id,
            top_entries[TOP_K - 1].0,
            top_entries[TOP_K - 1].1,
        )
        .is_lt()
        {
            top_entries[TOP_K - 1] = (rank, token_id, bits);
            top_entries
                .sort_unstable_by(|left, right| bf16_entry_order(left.0, left.1, right.0, right.1));
        }
    }
    let (_raw_rank, raw_token_id) = raw_best.ok_or("canonical BF16 row lacks raw logits")?;
    let (selected_rank, selected_token_id) =
        selected_best.ok_or("canonical BF16 row lacks addressable logits")?;
    if top_entries.len() != TOP_K
        || top_entries
            .first()
            .is_none_or(|entry| (entry.0, entry.1) != (selected_rank, selected_token_id))
    {
        return Err("canonical BF16 selected token differs from top-k".into());
    }
    let top_token_ids = top_entries
        .iter()
        .map(|entry| u32::try_from(entry.1).map_err(Into::into))
        .collect::<TestResult<Vec<_>>>()?;
    let top_values = top_entries
        .iter()
        .map(|entry| decode_bf16_le_scalar(&entry.2.to_le_bytes()))
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Bf16LogitAnalysis {
        raw_argmax_token_id: u32::try_from(raw_token_id)?,
        selected_token_id: u32::try_from(selected_token_id)?,
        selected_logit: top_values[0],
        top_token_ids,
        top_values,
    })
}

#[test]
fn hf_sidecar_bf16_analysis_orders_equal_values_by_lower_token_id() -> TestResult {
    let mut raw = vec![0_u8; EXPECTED_VOCABULARY_SIZE * BF16_BYTES];
    for lane in raw.chunks_exact_mut(BF16_BYTES) {
        lane.copy_from_slice(&0xbf80_u16.to_le_bytes()); // -1.0 BF16
    }
    let set = |raw: &mut [u8], token_id: usize, bits: u16| {
        let start = token_id * BF16_BYTES;
        raw[start..start + BF16_BYTES].copy_from_slice(&bits.to_le_bytes());
    };
    set(&mut raw, 8, 0x4000); // +2.0
    set(&mut raw, 12, 0x4000); // Equal +2.0: ID 8 must win the tie.
    set(&mut raw, 4, 0x0000); // +0.0
    set(&mut raw, 5, 0x8000); // -0.0: numerically equal to +0.0.
    set(&mut raw, EXPECTED_ADDRESSABLE_TOKEN_COUNT + 1, 0x4040); // +3.0 raw-only.

    let analysis = analyze_hf_bf16_le_logits(&raw)?;
    assert_eq!(
        analysis.raw_argmax_token_id,
        u32::try_from(EXPECTED_ADDRESSABLE_TOKEN_COUNT + 1)?
    );
    assert_eq!(analysis.selected_token_id, 8);
    assert_eq!(&analysis.top_token_ids[..6], &[8, 12, 4, 5, 0, 1]);
    assert_eq!(analysis.selected_logit, 2.0);
    assert_eq!(&analysis.top_values[..2], &[2.0, 2.0]);
    assert_eq!(analysis.top_values[2], 0.0);
    assert!(analysis.top_values[3].is_sign_negative());
    Ok(())
}

fn verify_hf_sidecar_rows(sidecar: &HfSidecarBinding, steps: &[HfStep], label: &str) -> TestResult {
    let row_bytes = EXPECTED_VOCABULARY_SIZE
        .checked_mul(BF16_BYTES)
        .ok_or("HF sidecar row byte count overflows")?;
    let expected_data_bytes = SERVER_MAX_OUTPUT_TOKENS
        .checked_mul(row_bytes)
        .ok_or("HF sidecar data byte count overflows")?;
    if steps.len() != SERVER_MAX_OUTPUT_TOKENS
        || sidecar
            .tensor_data_end
            .checked_sub(sidecar.tensor_data_start)
            != Some(expected_data_bytes)
    {
        return Err(format!("{label} row layout differs").into());
    }
    for (index, step) in steps.iter().enumerate() {
        if step.step != index {
            return Err(format!("{label} step ordering differs").into());
        }
        let raw_start = sidecar
            .tensor_data_start
            .checked_add(
                index
                    .checked_mul(row_bytes)
                    .ok_or("HF sidecar row offset overflows")?,
            )
            .ok_or("HF sidecar row range overflows")?;
        let raw_end = raw_start
            .checked_add(row_bytes)
            .ok_or("HF sidecar row range overflows")?;
        let raw = sidecar
            .bytes
            .get(raw_start..raw_end)
            .ok_or("HF sidecar row is truncated")?;
        let addressable_bytes = EXPECTED_ADDRESSABLE_TOKEN_COUNT
            .checked_mul(BF16_BYTES)
            .ok_or("HF sidecar addressable byte count overflows")?;
        let addressable = raw
            .get(..addressable_bytes)
            .ok_or("HF sidecar addressable row is truncated")?;
        let non_addressable = raw
            .get(addressable_bytes..)
            .ok_or("HF sidecar non-addressable row is truncated")?;
        let logits = &step.logits;
        if sha256_hex(raw) != logits.raw_logit_sha256
            || sha256_hex(addressable) != logits.addressable_logit_sha256
            || sha256_hex(non_addressable) != logits.non_addressable_logit_sha256
        {
            return Err(format!("{label} row {index} SHA-256 differs from manifest").into());
        }
        let analysis = analyze_hf_bf16_le_logits(raw)?;
        if analysis.raw_argmax_token_id != logits.raw_argmax_token_id
            || analysis.selected_token_id != logits.selected_token_id
            || analysis.selected_logit != logits.selected_logit
            || analysis.top_token_ids != logits.top_token_ids
            || analysis.top_values != logits.top_values
        {
            return Err(
                format!("{label} row {index} BF16 metadata differs from raw sidecar").into(),
            );
        }
    }
    Ok(())
}

fn top_k(logits: &[u8], count: usize) -> TestResult<Vec<u32>> {
    if logits.len() % BF16_BYTES != 0 || count > logits.len() / BF16_BYTES {
        return Err("invalid BF16 top-k row shape".into());
    }
    let mut entries = Vec::with_capacity(logits.len() / BF16_BYTES);
    for (index, lane) in logits.chunks_exact(BF16_BYTES).enumerate() {
        let bits = u16::from_ne_bytes([lane[0], lane[1]]);
        entries.push((bf16_numeric_sort_key(bits)?, index));
    }
    entries.sort_unstable_by(|(left_rank, left_index), (right_rank, right_index)| {
        bf16_entry_order(*left_rank, *left_index, *right_rank, *right_index)
    });
    entries.truncate(count);
    entries
        .into_iter()
        .map(|(_, index)| u32::try_from(index).map_err(Into::into))
        .collect()
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
    hf: &HfTeacherForcedOracle,
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
                let hf_cache_off = hf
                    .cache_off_steps
                    .get(trace_step)
                    .ok_or("HF cache-off trace has too few steps")?;
                let hf_cache_on = hf
                    .cache_on_steps
                    .get(trace_step)
                    .ok_or("HF cache-on trace has too few steps")?;
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
                    hf_cache_off: hf_cache_off.clone(),
                    hf_cache_on: hf_cache_on.clone(),
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
                    hf.teacher_token_ids[trace_step],
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
                    hf.teacher_token_ids[trace_step]
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
            Some(&hf.teacher_token_ids[..TRACE_OUTPUT_ROWS])
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

fn hf_step_json(step: &HfStep, mode: &'static str, row: &TraceRow) -> Value {
    let logits = &step.logits;
    json!({
        "mode": mode,
        "step": step.step,
        "call_input_token_count": step.call_input_token_count,
        "call_input_token_ids_le_u32_sha256": step.call_input_token_ids_le_sha256,
        "context_token_count": step.context_token_count,
        "attention_mask_token_count": step.attention_mask_token_count,
        "position_start": step.position_start,
        "position_end": step.position_end,
        "teacher_input_token_id": step.teacher_input_token_id,
        "cache_length_before": step.cache_length_before,
        "cache_length_after": step.cache_length_after,
        "logits_bf16_le_sha256": logits.raw_logit_sha256,
        "addressable_bf16_le_sha256": logits.addressable_logit_sha256,
        "raw_argmax_token_id": logits.raw_argmax_token_id,
        "selected_token_id": logits.selected_token_id,
        "cache_off_teacher_token_id": logits.cache_off_teacher_token_id,
        "selection_matches_cache_off_teacher": logits.selection_matches_cache_off_teacher,
        "selected_logit_bf16_as_f32": logits.selected_logit,
        "top_token_ids": logits.top_token_ids,
        "top_values_bf16_as_f32": logits.top_values,
        "raw_hash_matches": row.row_bf16_le_sha256 == logits.raw_logit_sha256,
        "selected_token_matches": row.selected_token_id == logits.selected_token_id,
        "raw_argmax_matches": row.raw_argmax_token_id == logits.raw_argmax_token_id,
        "top32_order_matches_bf16_numeric_tie_break": row.top_token_ids == logits.top_token_ids,
        "top32_set_overlap": overlap_count(&row.top_token_ids, &logits.top_token_ids),
    })
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
                "top_values_bf16_as_f32": row.top_values,
                "hf_cache_off": hf_step_json(&row.hf_cache_off, HF_CACHE_OFF_MODE, row),
                "hf_cache_on": hf_step_json(&row.hf_cache_on, HF_CACHE_ON_MODE, row),
            })
        })
        .collect();
    let first_cache_off_selected_token_mismatch = mode.rows.iter().find_map(|row| {
        (row.selected_token_id != row.hf_cache_off.logits.selected_token_id).then_some(row.step)
    });
    let first_cache_on_selected_token_mismatch = mode.rows.iter().find_map(|row| {
        (row.selected_token_id != row.hf_cache_on.logits.selected_token_id).then_some(row.step)
    });
    json!({
        "projection_bias_backend": mode.projection_bias_backend,
        "prefill_iteration_count": mode.prefill_iteration_count,
        "decode_iteration_count": mode.decode_iteration_count,
        "first_hf_cache_off_selected_token_mismatch": first_cache_off_selected_token_mismatch,
        "first_hf_cache_on_selected_token_mismatch": first_cache_on_selected_token_mismatch,
        "rows": rows,
    })
}

#[test]
#[ignore = "remote-only Qwen2.5-3B C8/M32 native-D128 teacher-forced numerical trace"]
fn qwen3b_native_d128_teacher_forced_trace_is_scheduler_committed() -> TestResult {
    let workload = load_workload()?;
    let hf = load_hf_teacher_forced_oracle(&workload)?;
    assert_eq!(workload.output_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
    assert_eq!(hf.teacher_token_ids.len(), SERVER_MAX_OUTPUT_TOKENS);
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
    let trace_teacher_tokens = &hf.teacher_token_ids[..TRACE_OUTPUT_ROWS];
    let document = json!({
        "schema_version": "riley.qwen3b-native-d128-teacher-forced-logit-trace.v2",
        "artifact_kind": "qwen2.5-3b-native-d128-scheduler-committed-teacher-forced-logit-trace",
        "performance_claim_eligible": false,
        "trace_contract": {
            "scheduler_executor_replay": true,
            "http_transport_replayed": false,
            "sampling": "teacher-forced-cache-off-addressable-greedy-argmax-after-scheduler-commit",
            "cache_on_scheduler_reference": "step0-p2048-prefill;step>0-teacher_token_ids[step-1]-at-position-2048+step-1",
            "cache_off_control_reference": "full-prefix-cache-off-at-position-0-through-2047+step",
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
            "top32_order_comparison": "bf16-numeric-descending-token-id-ascending-tie-break",
        },
        "model": {"id": QWEN3B_MODEL_ID, "revision": QWEN3B_REVISION},
        "workload": {
            "sha256": QWEN3B_SERVING_WORKLOAD_SHA256,
            "case": QWEN3B_WORKLOAD_CASE,
            "prompt_token_count": workload.prompt_token_ids.len(),
            "teacher_token_ids": trace_teacher_tokens,
            "teacher_token_ids_le_u32_sha256": token_ids_sha256(trace_teacher_tokens),
        },
        "hf_teacher_forced_oracle": {
            "schema_version": HF_TEACHER_FORCED_ORACLE_SCHEMA,
            "artifact_kind": HF_TEACHER_FORCED_ARTIFACT_KIND,
            "artifact_sha256": hf.artifact_sha256,
            "teacher_token_ids_le_u32_sha256": token_ids_sha256(&hf.teacher_token_ids),
            "cache_off_sidecar": {
                "basename": hf.cache_off_sidecar.basename,
                "sha256": hf.cache_off_sidecar.sha256,
            },
            "cache_on_sidecar": {
                "basename": hf.cache_on_sidecar.basename,
                "sha256": hf.cache_on_sidecar.sha256,
            },
        },
        "modes": [trace_json(&strict), trace_json(&fused)],
    });
    println!(
        "RILEY_QWEN3B_NATIVE_D128_LOGIT_TRACE={}",
        serde_json::to_string(&document)?
    );
    Ok(())
}
