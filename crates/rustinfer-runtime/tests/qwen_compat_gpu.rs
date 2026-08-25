//! Remote-only PR12 Qwen2.5 compatibility and cross-model CUDA lifecycle gate.
//!
//! `RUSTINFER_QWEN_GOLDEN` names the committed
//! `rustinfer-qwen2-compat-v1` fixture. Its English, Korean, and code cases
//! carry rendered prompt IDs, BF16 raw-logit probes/top tokens, and exact
//! greedy cache-on/cache-off token sequences.

#![cfg(feature = "cuda")]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rustinfer_cuda::{CudaContext, CudaRuntime, CudaStream};
use rustinfer_model::{EncodeOptions, LoadLimits, LoadedModel, ModelArchitecture, ModelFamily};
use rustinfer_runtime::generation::{FinishReason, GenerationRequest, GenerationState};
use rustinfer_runtime::llama::{
    LlamaDecodePhase, LlamaGenerationEvent, PreparedLlamaDecode, PreparedLlamaDecodeConfig,
    PreparedLlamaForward, PreparedLlamaForwardConfig, PreparedLlamaGeneration,
};
use rustinfer_runtime::sampling::{
    SamplingParams, SamplingRng, SamplingWorkspace, TokenConstraints,
};
use rustinfer_tensor::DType;
use serde_json::Value;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const GOLDEN_SCHEMA: &str = "rustinfer-qwen2-compat-v1";
const QWEN_MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const QWEN_REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
const EXPECTED_SOURCE_ARCHITECTURE: &str = "Qwen2ForCausalLM";
const EXPECTED_HIDDEN_SIZE: usize = 896;
const EXPECTED_INTERMEDIATE_SIZE: usize = 4_864;
const EXPECTED_LAYER_COUNT: usize = 24;
const EXPECTED_QUERY_HEADS: usize = 14;
const EXPECTED_KEY_VALUE_HEADS: usize = 2;
const EXPECTED_HEAD_DIMENSION: usize = 64;
const EXPECTED_VOCABULARY_SIZE: usize = 151_936;
const EXPECTED_ADDRESSABLE_TOKEN_COUNT: usize = 151_665;
const EXPECTED_MAX_SEQUENCE_LENGTH: usize = 32_768;
const EXPECTED_BOS_TOKEN_ID: u32 = 151_643;
const EXPECTED_EOS_TOKEN_ID: u32 = 151_645;
const EXPECTED_LOGICAL_WEIGHT_COUNT: usize = EXPECTED_LAYER_COUNT * 12 + 3;
const EXPECTED_GOLDEN_CASE_NAMES: [&str; 3] = ["english", "korean", "code"];
const EXPECTED_GOLDEN_OUTPUT_TOKENS: usize = 8;
const EXPECTED_RAW_TOP_TOKENS: usize = 10;
// Reuse the immutable PR01 E0 v2 final-logits max-absolute-error bound. The
// committed Qwen fixture intentionally stores sparse logits rather than the
// full row, so its cosine and mean-absolute-error gates cannot be recomputed
// here. Greedy top-1 and the top-10 token set remain exact semantic gates.
const FINAL_LOGIT_MAX_ABS_TOLERANCE: f64 = 5.852_936_458_587_647;

#[derive(Debug)]
struct GoldenFixture {
    cases: Vec<GoldenCase>,
}

#[derive(Debug)]
struct GoldenCase {
    name: String,
    rendered_chat: String,
    prompt_token_ids: Vec<u32>,
    raw_top_token_ids: Vec<u32>,
    raw_top_values: Vec<f32>,
    raw_probe_values: BTreeMap<u32, f32>,
    max_new_tokens: usize,
    cache_on_token_ids: Vec<u32>,
    cache_off_token_ids: Vec<u32>,
}

#[derive(Debug, Default)]
struct GenerationTrace {
    token_ids: Vec<u32>,
    finish_reasons: Vec<Option<FinishReason>>,
}

#[derive(Clone, Copy, Debug)]
struct NoGreedyDraw;

impl SamplingRng for NoGreedyDraw {
    type Error = Infallible;

    fn next_u32(&mut self) -> Result<u32, Self::Error> {
        panic!("temperature-zero sampling must not draw an RNG word")
    }
}

fn qwen_checkpoint_path() -> PathBuf {
    std::env::var_os("RUSTINFER_QWEN_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_QWEN_CHECKPOINT must name the remote Qwen checkpoint directory")
}

fn qwen_golden_path() -> PathBuf {
    std::env::var_os("RUSTINFER_QWEN_GOLDEN")
        .map(PathBuf::from)
        .expect("RUSTINFER_QWEN_GOLDEN must name the remote Qwen golden fixture")
}

fn qwen_load_limits() -> TestResult<LoadLimits> {
    Ok(LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?)
}

fn exact_qwen_decode_config() -> PreparedLlamaDecodeConfig {
    // The optimized online attention backends are tolerance-tested performance
    // paths, not bit-exact oracles for the staged-BF16 eager attention golden.
    // Keep both prefill and decode on the explicit reference policy here.
    PreparedLlamaDecodeConfig::new(PreparedLlamaForwardConfig::default().with_reference_attention())
        .with_paged_kv_cache()
        .with_reference_decode_attention()
}

fn load_qwen() -> TestResult<LoadedModel> {
    Ok(LoadedModel::load(
        &qwen_checkpoint_path(),
        qwen_load_limits()?,
    )?)
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

fn json_u32(value: &Value, field: &'static str) -> TestResult<u32> {
    Ok(u32::try_from(value.as_u64().ok_or(field)?)?)
}

fn json_usize(value: &Value, field: &'static str) -> TestResult<usize> {
    Ok(usize::try_from(value.as_u64().ok_or(field)?)?)
}

fn json_u32_array(value: &Value, field: &'static str) -> TestResult<Vec<u32>> {
    let values = value.as_array().ok_or(field)?;
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        output.push(json_u32(value, field)?);
    }
    Ok(output)
}

fn json_f32_array(value: &Value, field: &'static str) -> TestResult<Vec<f32>> {
    let values = value.as_array().ok_or(field)?;
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let number: f32 = serde_json::from_value(value.clone())?;
        if !number.is_finite() {
            return Err(format!("{field} must contain only finite values").into());
        }
        output.push(number);
    }
    Ok(output)
}

fn parse_probe_values(value: &Value) -> TestResult<BTreeMap<u32, f32>> {
    let values = value
        .as_object()
        .ok_or("raw_last_logits.probe_values_f32 must be an object")?;
    let mut output = BTreeMap::new();
    for (token_id, value) in values {
        let token_id = token_id.parse::<u32>()?;
        let number: f32 = serde_json::from_value(value.clone())?;
        if !number.is_finite() {
            return Err("raw_last_logits.probe_values_f32 must be finite".into());
        }
        if output.insert(token_id, number).is_some() {
            return Err(format!("duplicate raw-logit probe token {token_id}").into());
        }
    }
    Ok(output)
}

fn parse_golden_case(value: &Value) -> TestResult<GoldenCase> {
    let name = value["name"]
        .as_str()
        .ok_or("cases[].name must be a string")?
        .to_owned();
    let rendered_chat = value["rendered_chat"]
        .as_str()
        .ok_or("cases[].rendered_chat must be a string")?
        .to_owned();
    let prompt_token_ids = json_u32_array(
        &value["prompt_token_ids"],
        "cases[].prompt_token_ids must be a U32 array",
    )?;
    let raw_top_token_ids = json_u32_array(
        &value["raw_last_logits"]["top_token_ids"],
        "cases[].raw_last_logits.top_token_ids must be a U32 array",
    )?;
    let raw_top_values = json_f32_array(
        &value["raw_last_logits"]["top_values_f32"],
        "cases[].raw_last_logits.top_values_f32 must be an F32 array",
    )?;
    let raw_probe_values = parse_probe_values(&value["raw_last_logits"]["probe_values_f32"])?;
    let max_new_tokens = json_usize(
        &value["greedy"]["max_new_tokens"],
        "cases[].greedy.max_new_tokens must be unsigned",
    )?;
    let addressable_token_count = json_usize(
        &value["greedy"]["addressable_token_count"],
        "cases[].greedy.addressable_token_count must be unsigned",
    )?;
    assert_eq!(
        addressable_token_count, EXPECTED_ADDRESSABLE_TOKEN_COUNT,
        "golden case {name} addressable domain"
    );
    let cache_on_token_ids = json_u32_array(
        &value["greedy"]["cache_on_token_ids"],
        "cases[].greedy.cache_on_token_ids must be a U32 array",
    )?;
    let cache_off_token_ids = json_u32_array(
        &value["greedy"]["cache_off_token_ids"],
        "cases[].greedy.cache_off_token_ids must be a U32 array",
    )?;

    let case = GoldenCase {
        name,
        rendered_chat,
        prompt_token_ids,
        raw_top_token_ids,
        raw_top_values,
        raw_probe_values,
        max_new_tokens,
        cache_on_token_ids,
        cache_off_token_ids,
    };
    validate_golden_case(&case);
    Ok(case)
}

fn parse_golden_fixture() -> TestResult<GoldenFixture> {
    let document: Value = serde_json::from_slice(&fs::read(qwen_golden_path())?)?;
    assert_eq!(document["schema_version"], GOLDEN_SCHEMA);
    assert_eq!(document["model"]["id"], QWEN_MODEL_ID);
    assert_eq!(document["model"]["revision"], QWEN_REVISION);
    assert_eq!(document["model"]["dtype"], "bfloat16");
    assert_eq!(document["model"]["attention"], "eager");

    let values = document["cases"]
        .as_array()
        .ok_or("cases must be an array")?;
    assert_eq!(values.len(), EXPECTED_GOLDEN_CASE_NAMES.len());
    let mut cases = Vec::with_capacity(values.len());
    for (expected_name, value) in EXPECTED_GOLDEN_CASE_NAMES.iter().zip(values) {
        let case = parse_golden_case(value)?;
        assert_eq!(&case.name, expected_name);
        cases.push(case);
    }
    Ok(GoldenFixture { cases })
}

fn validate_golden_case(case: &GoldenCase) {
    assert!(!case.rendered_chat.is_empty());
    assert!(!case.prompt_token_ids.is_empty());
    assert_eq!(case.max_new_tokens, EXPECTED_GOLDEN_OUTPUT_TOKENS);
    assert_eq!(case.cache_on_token_ids.len(), case.max_new_tokens);
    assert_eq!(case.cache_on_token_ids, case.cache_off_token_ids);
    assert!(case.prompt_token_ids.iter().all(|&token| {
        usize::try_from(token).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    }));
    assert!(case.cache_on_token_ids.iter().all(|&token| {
        usize::try_from(token).is_ok_and(|id| id < EXPECTED_ADDRESSABLE_TOKEN_COUNT)
    }));
    assert!(!case.cache_on_token_ids.contains(&EXPECTED_EOS_TOKEN_ID));
    assert_eq!(case.raw_top_token_ids.len(), EXPECTED_RAW_TOP_TOKENS);
    assert_eq!(case.raw_top_values.len(), EXPECTED_RAW_TOP_TOKENS);
    assert!(
        case.raw_top_token_ids
            .iter()
            .all(|&token| { usize::try_from(token).is_ok_and(|id| id < EXPECTED_VOCABULARY_SIZE) })
    );
    let mut distinct_top_ids = case.raw_top_token_ids.clone();
    distinct_top_ids.sort_unstable();
    distinct_top_ids.dedup();
    assert_eq!(distinct_top_ids.len(), EXPECTED_RAW_TOP_TOKENS);
    assert!(
        case.raw_top_values
            .windows(2)
            .all(|values| values[0] >= values[1])
    );
    assert!(
        case.raw_probe_values
            .keys()
            .all(|&token| { usize::try_from(token).is_ok_and(|id| id < EXPECTED_VOCABULARY_SIZE) })
    );
    assert_eq!(case.raw_top_token_ids[0], case.cache_on_token_ids[0]);
    assert!(case.raw_probe_values.contains_key(&0));
    assert!(
        case.raw_probe_values
            .contains_key(&u32::try_from(EXPECTED_ADDRESSABLE_TOKEN_COUNT).unwrap())
    );
    assert!(
        case.raw_probe_values
            .contains_key(&u32::try_from(EXPECTED_VOCABULARY_SIZE - 1).unwrap())
    );
}

fn assert_model_contract(model: &LoadedModel) {
    assert_eq!(model.config().family(), ModelFamily::Qwen2);
    assert_eq!(model.provenance().source_model(), QWEN_MODEL_ID);
    assert_eq!(model.provenance().source_revision(), QWEN_REVISION);

    let spec = model.spec();
    assert_eq!(spec.architecture(), ModelArchitecture::Llama);
    assert_eq!(spec.source_architecture(), EXPECTED_SOURCE_ARCHITECTURE);
    assert_eq!(spec.dtype(), DType::BF16);
    assert_eq!(spec.embedding().hidden_size(), EXPECTED_HIDDEN_SIZE);
    assert_eq!(spec.embedding().vocabulary_size(), EXPECTED_VOCABULARY_SIZE);
    assert_eq!(spec.max_sequence_length(), EXPECTED_MAX_SEQUENCE_LENGTH);
    assert_eq!(spec.blocks().len(), EXPECTED_LAYER_COUNT);
    assert_eq!(spec.lm_head().hidden_size(), EXPECTED_HIDDEN_SIZE);
    assert_eq!(spec.lm_head().vocabulary_size(), EXPECTED_VOCABULARY_SIZE);
    assert!(spec.lm_head().tied_to_embedding());
    assert_eq!(spec.special_tokens().bos(), Some(EXPECTED_BOS_TOKEN_ID));
    assert_eq!(spec.special_tokens().eos(), &[EXPECTED_EOS_TOKEN_ID]);
    assert_eq!(
        model.tokenizer().addressable_token_count(),
        EXPECTED_ADDRESSABLE_TOKEN_COUNT
    );
    assert!(model.qwen2_tokenizer_config().is_some());

    for (index, block) in spec.blocks().iter().enumerate() {
        assert_eq!(block.index(), index);
        let attention = block.attention();
        assert_eq!(attention.hidden_size(), EXPECTED_HIDDEN_SIZE);
        assert_eq!(attention.query_heads(), EXPECTED_QUERY_HEADS);
        assert_eq!(attention.key_value_heads(), EXPECTED_KEY_VALUE_HEADS);
        assert_eq!(attention.head_dimension(), EXPECTED_HEAD_DIMENSION);
        assert!(attention.bias().query());
        assert!(attention.bias().key());
        assert!(attention.bias().value());
        assert!(!attention.bias().output());
        assert!(!block.mlp().has_bias());
    }
}

fn assert_plan_contract(owner: &PreparedLlamaForward) {
    let plan = owner.plan();
    let dimensions = plan.dimensions();
    assert_eq!(dimensions.hidden_size(), EXPECTED_HIDDEN_SIZE);
    assert_eq!(dimensions.intermediate_size(), EXPECTED_INTERMEDIATE_SIZE);
    assert_eq!(dimensions.vocabulary_size(), EXPECTED_VOCABULARY_SIZE);
    assert_eq!(dimensions.query_heads(), EXPECTED_QUERY_HEADS);
    assert_eq!(dimensions.key_value_heads(), EXPECTED_KEY_VALUE_HEADS);
    assert_eq!(dimensions.head_dimension(), EXPECTED_HEAD_DIMENSION);
    assert_eq!(
        dimensions.key_value_width(),
        EXPECTED_KEY_VALUE_HEADS * EXPECTED_HEAD_DIMENSION
    );
    assert_eq!(
        dimensions.group_size(),
        EXPECTED_QUERY_HEADS / EXPECTED_KEY_VALUE_HEADS
    );
    assert_eq!(plan.layers().len(), EXPECTED_LAYER_COUNT);
    assert_eq!(plan.logical_weight_count(), EXPECTED_LOGICAL_WEIGHT_COUNT);
    assert_eq!(plan.final_norm_epsilon(), 1.0e-6_f32);
    assert_eq!(plan.rope_theta(), 1_000_000.0_f32);
}

fn logits_row_bytes() -> TestResult<usize> {
    EXPECTED_VOCABULARY_SIZE
        .checked_mul(BF16_BYTES)
        .ok_or_else(|| "Qwen logits row byte count overflow".into())
}

fn decode_bf16_scalar(bytes: &[u8]) -> f32 {
    assert_eq!(bytes.len(), BF16_BYTES);
    let bits = u16::from_ne_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn top1(logits: &[u8]) -> u32 {
    assert!(!logits.is_empty());
    assert_eq!(logits.len() % BF16_BYTES, 0);
    let mut values = logits.chunks_exact(BF16_BYTES).map(decode_bf16_scalar);
    let first = values.next().expect("validated non-empty logits row");
    assert!(first.is_finite());
    let mut best_id = 0_usize;
    let mut best_value = first;
    for (offset, value) in values.enumerate() {
        assert!(value.is_finite());
        if value.total_cmp(&best_value).is_gt() {
            best_id = offset + 1;
            best_value = value;
        }
    }
    u32::try_from(best_id).expect("Qwen vocabulary IDs fit U32")
}

fn top_k(logits: &[u8], count: usize) -> Vec<u32> {
    assert!(count <= logits.len() / BF16_BYTES);
    let values: Vec<_> = logits
        .chunks_exact(BF16_BYTES)
        .map(decode_bf16_scalar)
        .collect();
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    indices
        .into_iter()
        .map(|index| u32::try_from(index).expect("Qwen vocabulary IDs fit U32"))
        .collect()
}

fn addressable_logits(logits: &[u8]) -> &[u8] {
    let byte_len = EXPECTED_ADDRESSABLE_TOKEN_COUNT
        .checked_mul(BF16_BYTES)
        .expect("Qwen addressable logits byte count");
    &logits[..byte_len]
}

fn logit_at(logits: &[u8], token_id: u32) -> f32 {
    let index = usize::try_from(token_id).expect("Qwen token ID fits usize");
    let offset = index
        .checked_mul(BF16_BYTES)
        .expect("Qwen logit offset does not overflow");
    decode_bf16_scalar(&logits[offset..offset + BF16_BYTES])
}

fn assert_raw_prefill_logits(logits: &[u8], case: &GoldenCase) {
    assert_eq!(logits.len(), EXPECTED_VOCABULARY_SIZE * BF16_BYTES);
    for (&token_id, &expected) in &case.raw_probe_values {
        let actual = logit_at(logits, token_id);
        let absolute_error = f64::from((actual - expected).abs());
        assert!(
            absolute_error <= FINAL_LOGIT_MAX_ABS_TOLERANCE,
            "case={} raw probe token={} actual={} expected={} absolute_error={} tolerance={}",
            case.name,
            token_id,
            actual,
            expected,
            absolute_error,
            FINAL_LOGIT_MAX_ABS_TOLERANCE
        );
    }
    for (&token_id, &expected) in case.raw_top_token_ids.iter().zip(&case.raw_top_values) {
        let actual = logit_at(logits, token_id);
        let absolute_error = f64::from((actual - expected).abs());
        assert!(
            absolute_error <= FINAL_LOGIT_MAX_ABS_TOLERANCE,
            "case={} raw top token={} actual={} expected={} absolute_error={} tolerance={}",
            case.name,
            token_id,
            actual,
            expected,
            absolute_error,
            FINAL_LOGIT_MAX_ABS_TOLERANCE
        );
    }

    assert_eq!(
        top1(logits),
        case.raw_top_token_ids[0],
        "case={} raw top-1 token mismatch",
        case.name
    );
    let actual_top: BTreeSet<_> = top_k(logits, EXPECTED_RAW_TOP_TOKENS).into_iter().collect();
    let expected_top: BTreeSet<_> = case.raw_top_token_ids.iter().copied().collect();
    assert_eq!(
        actual_top, expected_top,
        "case={} raw top-10 token set mismatch",
        case.name
    );
}

fn bf16_bytes(value: f32) -> [u8; BF16_BYTES] {
    let bits = u16::try_from(value.to_bits() >> 16).expect("upper F32 bits fit U16");
    bits.to_ne_bytes()
}

fn assert_padded_sampling_domain_contract() -> TestResult {
    let mut logits = vec![0_u8; logits_row_bytes()?];
    let baseline = bf16_bytes(1.0);
    for value in logits.chunks_exact_mut(BF16_BYTES) {
        value.copy_from_slice(&baseline);
    }
    let addressable_winner = 42_usize;
    logits[addressable_winner * BF16_BYTES..(addressable_winner + 1) * BF16_BYTES]
        .copy_from_slice(&bf16_bytes(8.0));
    for value in
        logits[EXPECTED_ADDRESSABLE_TOKEN_COUNT * BF16_BYTES..].chunks_exact_mut(BF16_BYTES)
    {
        value.copy_from_slice(&bf16_bytes(64.0));
    }

    let greedy = SamplingParams {
        temperature: 0.0,
        top_k: None,
        top_p: None,
        repetition_penalty: 1.0,
    };
    let mut workspace = SamplingWorkspace::new(EXPECTED_VOCABULARY_SIZE)?;
    let mut no_draw = NoGreedyDraw;
    let unmasked =
        workspace.process_bf16_native(&logits, TokenConstraints::AllowAll, &[], greedy)?;
    assert_eq!(
        unmasked.sample(&mut no_draw)?.token_id(),
        u32::try_from(EXPECTED_ADDRESSABLE_TOKEN_COUNT)?,
        "the synthetic raw row must be won by the first padded ID"
    );

    let mut allowed = vec![false; EXPECTED_VOCABULARY_SIZE];
    allowed[..EXPECTED_ADDRESSABLE_TOKEN_COUNT].fill(true);
    let masked = workspace.process_bf16_native(
        &logits,
        TokenConstraints::AllowedMask(&allowed),
        &[],
        greedy,
    )?;
    let sampled = masked.sample(&mut no_draw)?.token_id();
    assert_eq!(sampled, u32::try_from(addressable_winner)?);
    for token_id in EXPECTED_ADDRESSABLE_TOKEN_COUNT..EXPECTED_VOCABULARY_SIZE {
        assert_eq!(masked.probabilities()[token_id], 0.0);
        assert_eq!(masked.log_probabilities()[token_id], f64::NEG_INFINITY);
        assert_eq!(masked.processed_logits()[token_id], f64::NEG_INFINITY);
    }
    Ok(())
}

fn capture_generation_event(trace: &mut GenerationTrace, event: LlamaGenerationEvent<'_>) {
    match event {
        LlamaGenerationEvent::Token { token, .. } => {
            trace.token_ids.push(token.token_id());
            trace.finish_reasons.push(token.finish_reason());
        }
        LlamaGenerationEvent::Cancelled { .. } => {
            panic!("the fixed Qwen golden generation must not be cancelled")
        }
    }
}

#[test]
#[ignore = "remote-only pinned Qwen2.5 compatibility gate on server-4096"]
fn pinned_qwen_prefill_decode_and_generation_match_golden() -> TestResult {
    let fixture = parse_golden_fixture()?;
    let model = load_qwen()?;
    assert_model_contract(&model);
    assert_padded_sampling_domain_contract()?;
    let (context, mut stream) = first_context()?;

    for case in &fixture.cases {
        run_golden_case(&model, &context, &mut stream, case)?;
        assert!(context.allocation_stats()?.is_zero());
    }

    println!(
        "pr12-qwen-compat-summary schema_version=1 cases={} case_names=english,korean,code \
raw_logits_sparse_max_abs_tolerance={} raw_top1_exact=true raw_top10_set_exact=true \
cache_on_off_fixture_exact=true cached_generation_exact=true \
vocabulary={} addressable_tokens={} padded_tokens={} qkv_bias_layers={} status=passed",
        fixture.cases.len(),
        FINAL_LOGIT_MAX_ABS_TOLERANCE,
        EXPECTED_VOCABULARY_SIZE,
        EXPECTED_ADDRESSABLE_TOKEN_COUNT,
        EXPECTED_VOCABULARY_SIZE - EXPECTED_ADDRESSABLE_TOKEN_COUNT,
        EXPECTED_LAYER_COUNT,
    );
    stream.close()?;
    close_context(context)
}

fn run_golden_case(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    case: &GoldenCase,
) -> TestResult {
    let encoded = model.tokenizer().encode(
        &case.rendered_chat,
        EncodeOptions {
            add_special_tokens: false,
        },
    )?;
    assert_eq!(
        encoded, case.prompt_token_ids,
        "case={} rendered chat tokenizer IDs",
        case.name
    );

    let decode_calls = case.cache_on_token_ids.len().saturating_sub(1);
    let decode_capacity = case
        .prompt_token_ids
        .len()
        .checked_add(decode_calls)
        .ok_or("Qwen decode capacity overflow")?;
    let mut decode = PreparedLlamaDecode::prepare(
        model,
        context,
        stream,
        case.prompt_token_ids.len(),
        decode_capacity,
        exact_qwen_decode_config(),
    )?;
    decode.prefill(&case.prompt_token_ids, stream)?;
    assert_eq!(decode.phase(), LlamaDecodePhase::Prefilled);
    assert_eq!(decode.logical_length(), case.prompt_token_ids.len());
    let mut logits = vec![0_u8; logits_row_bytes()?];
    decode.download_last_logits(&mut logits, stream)?;
    assert_raw_prefill_logits(&logits, case);

    for (step, &expected_token) in case.cache_on_token_ids.iter().enumerate() {
        let actual_token = top1(addressable_logits(&logits));
        assert_eq!(
            actual_token, expected_token,
            "case={} greedy cache-on step={} actual_token={} expected_token={}",
            case.name, step, actual_token, expected_token
        );
        if step < decode_calls {
            decode.decode(expected_token, stream)?;
            decode.download_last_logits(&mut logits, stream)?;
        }
    }
    assert_eq!(decode.phase(), LlamaDecodePhase::Decoding);
    assert_eq!(decode.logical_length(), decode_capacity);
    decode.close()?;
    assert!(context.allocation_stats()?.is_zero());

    let mut owner = PreparedLlamaGeneration::prepare(
        model,
        context,
        stream,
        case.prompt_token_ids.len(),
        case.max_new_tokens,
        exact_qwen_decode_config(),
    )?;
    assert_eq!(owner.vocabulary_size(), EXPECTED_VOCABULARY_SIZE);
    assert_eq!(
        owner.addressable_token_count(),
        EXPECTED_ADDRESSABLE_TOKEN_COUNT
    );
    let request = GenerationRequest {
        request_id: format!("pr12-qwen-golden-{}", case.name).into_bytes(),
        seed: 0,
        prompt_token_ids: case.prompt_token_ids.clone(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        },
        min_new_tokens: 0,
        max_new_tokens: case.max_new_tokens,
        eos_token_ids: vec![EXPECTED_EOS_TOKEN_ID],
        stop_token_ids: Vec::new(),
        stop_strings: Vec::new(),
    };
    let mut state = GenerationState::new(
        request,
        EXPECTED_VOCABULARY_SIZE,
        model.tokenizer().maximum_decoded_token_bytes(),
    )?;
    let mut trace = GenerationTrace::default();
    let summary = owner.generate(
        &mut state,
        stream,
        || false,
        |event| {
            capture_generation_event(&mut trace, event);
            Ok::<(), Infallible>(())
        },
    )?;

    assert_eq!(
        case.cache_on_token_ids, case.cache_off_token_ids,
        "case={} fixture cache-on/off parity",
        case.name
    );
    assert_eq!(
        state.generated_token_ids(),
        case.cache_on_token_ids,
        "case={} runtime cached generation vs cache-on golden",
        case.name
    );
    assert_eq!(
        state.generated_token_ids(),
        case.cache_off_token_ids,
        "case={} runtime cached generation vs cache-off golden",
        case.name
    );
    assert_eq!(trace.token_ids, case.cache_on_token_ids);
    assert_eq!(state.finish_reason(), Some(FinishReason::Length));
    assert_eq!(state.rng_draws(), 0);
    assert_eq!(summary.sampled_tokens(), case.cache_on_token_ids.len());
    assert_eq!(summary.prefill_tokens(), 1);
    assert_eq!(
        summary.decode_tokens(),
        case.cache_on_token_ids.len().saturating_sub(1)
    );
    assert!(
        trace
            .token_ids
            .iter()
            .all(|&token| usize::try_from(token)
                .is_ok_and(|id| id < owner.addressable_token_count())),
        "generation sampled a padded vocabulary ID"
    );
    assert_eq!(trace.finish_reasons.len(), trace.token_ids.len());
    for finish in &trace.finish_reasons[..trace.finish_reasons.len() - 1] {
        assert_eq!(*finish, None);
    }
    assert_eq!(
        trace.finish_reasons.last().copied().flatten(),
        Some(FinishReason::Length)
    );
    assert_eq!(owner.decode_phase(), Some(LlamaDecodePhase::Empty));

    println!(
        "pr12-qwen-compat schema_version=1 case={} prompt_tokens={} generated_tokens={} \
raw_top_tokens={} raw_probes={} prefill_top1={} decode_calls={} cache_on_off_exact=true \
runtime_cached_exact=true addressable_mask=true rng_draws=0 status=passed",
        case.name,
        case.prompt_token_ids.len(),
        case.cache_on_token_ids.len(),
        case.raw_top_token_ids.len(),
        case.raw_probe_values.len(),
        case.cache_on_token_ids[0],
        decode_calls,
    );

    owner.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(())
}

fn execute_single_token_owner(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    inspect: Option<fn(&PreparedLlamaForward)>,
) -> TestResult {
    let mut owner = PreparedLlamaForward::prepare(
        model,
        context,
        stream,
        1,
        PreparedLlamaForwardConfig::default().with_optimized_attention(),
    )?;
    if let Some(inspect) = inspect {
        inspect(&owner);
    }
    owner.forward(&[0], stream)?;
    owner.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(())
}

#[test]
#[ignore = "remote-only sequential SmolLM2 then Qwen CUDA owner lifecycle gate"]
fn smollm2_then_qwen_cuda_owners_close_in_one_process() -> TestResult {
    let Some(smol_checkpoint) = std::env::var_os("RUSTINFER_REAL_CHECKPOINT") else {
        println!(
            "pr12-cross-model-owner schema_version=1 status=skipped \
reason=RUSTINFER_REAL_CHECKPOINT_not_set"
        );
        return Ok(());
    };

    let (context, mut stream) = first_context()?;
    let smol_model = LoadedModel::load(&PathBuf::from(smol_checkpoint), LoadLimits::default())?;
    assert_eq!(smol_model.config().family(), ModelFamily::Llama);
    assert_eq!(smol_model.spec().embedding().vocabulary_size(), 49_152);
    execute_single_token_owner(&smol_model, &context, &mut stream, None)?;
    drop(smol_model);

    let qwen_model = load_qwen()?;
    assert_model_contract(&qwen_model);
    execute_single_token_owner(
        &qwen_model,
        &context,
        &mut stream,
        Some(assert_plan_contract),
    )?;
    drop(qwen_model);

    println!(
        "pr12-cross-model-owner schema_version=1 order=smollm2,qwen2.5 \
executed_each=true cuda_allocation_zero_after_each_close=true status=passed"
    );
    stream.close()?;
    close_context(context)
}
