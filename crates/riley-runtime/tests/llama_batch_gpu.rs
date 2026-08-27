//! Remote-only PR13 fixed-M continuous-batch GPU gates.

#![cfg(feature = "cuda")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use riley_cuda::{
    AttentionReductionProfile, CudaAllocationStats, CudaContext, CudaRuntime, CudaStream,
};
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{
    ExecutionCompletionImplementation, LlamaBatchBlockTable, LlamaBatchMetadataConfig,
    LlamaBatchRow, LlamaBatchRowKind, LlamaReductionProfile, PreparedLlamaBatchExecutor,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaDecode, PreparedLlamaDecodeConfig,
    PreparedLlamaForward, PreparedLlamaForwardConfig, ResidualNormImplementation,
};
use riley_runtime::paged_kv::{BLOCK_TABLE_V1_VERSION, KV_BLOCK_SIZE};
use serde_json::Value;
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const TOKENS_A: [u32; 7] = [504, 2_365, 6_354, 16_438, 11_139, 253, 1_890];
const TOKENS_B: [u32; 4] = [504, 2_365, 42, 43];
const BF16_BYTES: usize = 2;
const ONE_GIB: u64 = 1 << 30;
const EXPECTED_GOLDEN_CASES: usize = 31;
const EXPECTED_GOLDEN_STEPS: usize = 481;
const EXPECTED_GOLDEN_EXACT_WINDOW: usize = 16;
const EXPECTED_GOLDEN_FIXTURE_SHA256: &str =
    "87333a1859be45a2f8e7563d898dde5e64256ccc03ca4da3cab90def07dd3c95";
const EXPECTED_GOLDEN_TOKEN_IDS_SHA256: &str =
    "9e38488c0d41dae4a28e7e262baf772f2c643e9f8a9c57941a9e47aaec77ac5c";
// Reuse the immutable PR01 E0 v2 full-corpus final-logit bounds. These are
// conservative secondary guards beside exact greedy top-1 and are not fitted
// to the PR13 differential runs.
const BATCH_LOGIT_COSINE_MIN: f64 = 0.997_903_530_549_539_3;
const BATCH_LOGIT_MAX_ABS_MAX: f64 = 5.852_936_458_587_647;
const BATCH_LOGIT_MEAN_ABS_MAX: f64 = 1.151_280_319_263_363;

#[derive(Clone, Copy, Debug)]
struct LogitMetrics {
    cosine: f64,
    max_abs: f64,
    mean_abs: f64,
}

#[derive(Debug)]
struct GreedyExecutionTrace {
    generated_token_ids: Vec<u32>,
    logits_by_iteration: Vec<Vec<u8>>,
    cuda_live_allocation_delta: i128,
    owner_close_live_allocation_count: u64,
}

#[derive(Debug)]
struct Fixed37BatchGoldenCase {
    index: usize,
    prompt_id: String,
    prompt_token_ids: Vec<u32>,
    golden_token_ids: Vec<u32>,
    fixed_cached_logits: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct Fixed37BatchGoldenFixture {
    cases_by_prompt_length: BTreeMap<usize, Vec<Fixed37BatchGoldenCase>>,
    total_generated_steps: usize,
    exact_window: usize,
    fixture_sha256: String,
    generated_token_ids_sha256: String,
}

fn prepared_decode_profile_trace(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    config: PreparedLlamaDecodeConfig,
    decode_steps: usize,
) -> TestResult<(Vec<u32>, Vec<Vec<u8>>)> {
    let profile = config.reduction_profile();
    let maximum_length = TOKENS_A
        .len()
        .checked_add(decode_steps)
        .ok_or("maximum decode length overflow")?;
    let mut decode = PreparedLlamaDecode::prepare(
        model,
        context,
        stream,
        TOKENS_A.len(),
        maximum_length,
        config,
    )?;
    assert_eq!(decode.reduction_profile(), profile);
    let expected_attention_profile = match profile {
        LlamaReductionProfile::CanonicalV1 => AttentionReductionProfile::CanonicalV1,
        LlamaReductionProfile::FixedContiguous37BalancedV1 => {
            AttentionReductionProfile::FixedContiguous37BalancedV1
        }
    };
    assert_eq!(
        decode
            .prepared_attention()
            .selection_trace()
            .reduction_profile(),
        expected_attention_profile
    );

    decode.prefill(&TOKENS_A, stream)?;
    let mut logits = vec![0_u8; vocabulary_row_bytes(model)?];
    decode.download_last_logits(&mut logits, stream)?;
    let stable = context.allocation_stats()?;
    let mut token_ids = Vec::with_capacity(decode_steps);
    let mut logits_by_iteration = Vec::with_capacity(decode_steps + 1);
    logits_by_iteration.push(logits.clone());
    for step in 0..decode_steps {
        let token = u32::try_from(top1(&logits))?;
        token_ids.push(token);
        decode.decode(token, stream)?;
        decode.download_last_logits(&mut logits, stream)?;
        logits_by_iteration.push(logits.clone());
        assert_eq!(
            context.allocation_stats()?,
            stable,
            "prepared decode allocation changed at step {}",
            step + 1
        );
    }
    decode.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((token_ids, logits_by_iteration))
}

fn live_allocation_count(stats: CudaAllocationStats) -> u64 {
    stats
        .device_live_allocations()
        .checked_add(stats.pinned_host_live_allocations())
        .expect("CUDA live allocation count overflow")
}

fn live_allocation_delta(before: CudaAllocationStats, after: CudaAllocationStats) -> i128 {
    i128::from(live_allocation_count(after)) - i128::from(live_allocation_count(before))
}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RILEY_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn checkpoint_load_limits() -> TestResult<LoadLimits> {
    Ok(LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?)
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
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

fn batch_config(
    max_rows: usize,
    max_input_tokens: usize,
    max_block_entries: usize,
    max_output_slots: usize,
    physical_block_count: usize,
) -> TestResult<PreparedLlamaBatchExecutorConfig> {
    Ok(PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(
            max_rows,
            max_input_tokens,
            max_block_entries,
            max_output_slots,
            physical_block_count,
        )?,
        PreparedLlamaForwardConfig::default(),
    ))
}

fn block_table(length: usize, first_physical: u32) -> TestResult<(Vec<u32>, Vec<u16>)> {
    let block_count = length.div_ceil(KV_BLOCK_SIZE);
    let mut ids = Vec::with_capacity(block_count);
    let mut valid = Vec::with_capacity(block_count);
    for block in 0..block_count {
        ids.push(
            first_physical
                .checked_add(u32::try_from(block)?)
                .ok_or("physical block ID overflow")?,
        );
        let remaining = length - block * KV_BLOCK_SIZE;
        valid.push(u16::try_from(remaining.min(KV_BLOCK_SIZE))?);
    }
    Ok((ids, valid))
}

fn row<'a>(
    tag: u64,
    kind: LlamaBatchRowKind,
    tokens: &'a [u32],
    target_length: usize,
    ids: &'a [u32],
    valid: &'a [u16],
    output_slot: Option<u32>,
) -> TestResult<LlamaBatchRow<'a>> {
    Ok(LlamaBatchRow::new(
        tag,
        kind,
        tokens,
        u32::try_from(target_length)?,
        LlamaBatchBlockTable::new(
            BLOCK_TABLE_V1_VERSION,
            ids,
            valid,
            u32::try_from(target_length)?,
        ),
        output_slot,
    ))
}

fn vocabulary_row_bytes(model: &LoadedModel) -> TestResult<usize> {
    Ok(model
        .spec()
        .embedding()
        .vocabulary_size()
        .checked_mul(BF16_BYTES)
        .ok_or("vocabulary row byte length overflow")?)
}

fn golden_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/reference/smollm2-135m-bf16.json")
}

fn json_u32_array(value: &Value, field: &'static str) -> TestResult<Vec<u32>> {
    let values = value.as_array().ok_or(field)?;
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        output.push(u32::try_from(value.as_u64().ok_or(field)?)?);
    }
    Ok(output)
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn parse_fixed37_batch_golden_fixture() -> TestResult<Fixed37BatchGoldenFixture> {
    assert!(
        std::env::var_os("RILEY_GROWING_PREFIX_PROMPT_ID").is_none(),
        "the release gate forbids prompt-filtered golden execution"
    );
    let fixture_bytes = fs::read(golden_fixture_path())?;
    let fixture_sha256 = lowercase_hex(&Sha256::digest(&fixture_bytes));
    assert_eq!(fixture_sha256, EXPECTED_GOLDEN_FIXTURE_SHA256);
    let document: Value = serde_json::from_slice(&fixture_bytes)?;
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
    let exact_window = usize::try_from(
        document["generation"]["max_new_tokens"]
            .as_u64()
            .ok_or("generation.max_new_tokens must be an unsigned integer")?,
    )?;
    assert_eq!(exact_window, EXPECTED_GOLDEN_EXACT_WINDOW);
    let cases = document["cases"]
        .as_array()
        .ok_or("cases must be an array")?;
    assert_eq!(cases.len(), EXPECTED_GOLDEN_CASES);
    assert_eq!(
        document["corpus"]["prompt_count"].as_u64(),
        Some(u64::try_from(cases.len())?)
    );

    let mut token_hasher = Sha256::new();
    let mut total_generated_steps = 0_usize;
    let mut cases_by_prompt_length: BTreeMap<usize, Vec<Fixed37BatchGoldenCase>> = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        let prompt_id = case["prompt_id"]
            .as_str()
            .ok_or("cases[].prompt_id must be a string")?
            .to_owned();
        let prompt_token_ids = json_u32_array(
            &case["input"]["token_ids"],
            "cases[].input.token_ids must be a U32 array",
        )?;
        assert_eq!(
            case["input"]["token_count"].as_u64(),
            Some(u64::try_from(prompt_token_ids.len())?)
        );
        let golden_token_ids = json_u32_array(
            &case["greedy"]["cache_on_token_ids"],
            "cases[].greedy.cache_on_token_ids must be a U32 array",
        )?;
        let cache_off_token_ids = json_u32_array(
            &case["greedy"]["cache_off_token_ids"],
            "cases[].greedy.cache_off_token_ids must be a U32 array",
        )?;
        assert_eq!(golden_token_ids, cache_off_token_ids);
        assert_eq!(case["greedy"]["exact_match"].as_bool(), Some(true));
        assert!(!golden_token_ids.is_empty());
        assert!(golden_token_ids.len() <= exact_window);
        total_generated_steps = total_generated_steps
            .checked_add(golden_token_ids.len())
            .ok_or("golden generated-step count overflow")?;
        for token_id in &golden_token_ids {
            token_hasher.update(token_id.to_le_bytes());
        }
        cases_by_prompt_length
            .entry(prompt_token_ids.len())
            .or_default()
            .push(Fixed37BatchGoldenCase {
                index,
                prompt_id,
                prompt_token_ids,
                golden_token_ids,
                fixed_cached_logits: Vec::new(),
            });
    }
    assert_eq!(total_generated_steps, EXPECTED_GOLDEN_STEPS);
    let generated_token_ids_sha256 = lowercase_hex(&token_hasher.finalize());
    assert_eq!(generated_token_ids_sha256, EXPECTED_GOLDEN_TOKEN_IDS_SHA256);
    Ok(Fixed37BatchGoldenFixture {
        cases_by_prompt_length,
        total_generated_steps,
        exact_window,
        fixture_sha256,
        generated_token_ids_sha256,
    })
}

fn production_batch_config(
    max_input_tokens: usize,
    maximum_length: usize,
    profile: LlamaReductionProfile,
) -> TestResult<PreparedLlamaBatchExecutorConfig> {
    let physical_blocks = maximum_length.div_ceil(KV_BLOCK_SIZE);
    Ok(
        batch_config(1, max_input_tokens, physical_blocks, 1, physical_blocks)?
            .with_separate_residual_norm()
            .with_iteration_batch_completion()
            .with_reduction_profile(profile),
    )
}

fn run_cached_batch_golden_group(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    prompt_length: usize,
    cases: &mut [Fixed37BatchGoldenCase],
    profile: LlamaReductionProfile,
    capture_fixed_logits: bool,
) -> TestResult<usize> {
    let maximum_steps = cases
        .iter()
        .map(|case| case.golden_token_ids.len())
        .max()
        .ok_or("golden prompt-length group is empty")?;
    let maximum_length = prompt_length
        .checked_add(maximum_steps)
        .ok_or("cached batch maximum length overflow")?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        model,
        context,
        stream,
        production_batch_config(prompt_length, maximum_length, profile)?,
    )?;
    assert_eq!(batch.reduction_profile(), profile);
    assert!(batch.reduction_profile_is_coherent());
    assert_eq!(
        batch.config().residual_norm_implementation(),
        ResidualNormImplementation::Separate
    );
    assert_eq!(
        batch.config().execution_completion_implementation(),
        ExecutionCompletionImplementation::IterationBatch
    );
    let stable = context.allocation_stats()?;
    let row_bytes = vocabulary_row_bytes(model)?;
    let mut logits = vec![0_u8; row_bytes];
    let mut compared_steps = 0_usize;
    for case in cases {
        let (prompt_ids, prompt_valid) = block_table(prompt_length, 0)?;
        let prompt_rows = [row(
            u64::try_from(case.index + 1)?,
            LlamaBatchRowKind::Prefill,
            &case.prompt_token_ids,
            prompt_length,
            &prompt_ids,
            &prompt_valid,
            Some(0),
        )?];
        batch.execute(&prompt_rows, stream)?;
        assert_eq!(context.allocation_stats()?, stable);
        if capture_fixed_logits {
            case.fixed_cached_logits.clear();
            case.fixed_cached_logits
                .reserve(case.golden_token_ids.len());
        }
        for (step, &expected_token_id) in case.golden_token_ids.iter().enumerate() {
            batch.download_logits(&mut logits, stream)?;
            let actual_token_id = u32::try_from(top1(&logits))?;
            assert_eq!(
                actual_token_id, expected_token_id,
                "{profile:?} cached production batch differs from golden prompt={} step={step}",
                case.prompt_id
            );
            if capture_fixed_logits {
                case.fixed_cached_logits.push(logits.clone());
            }
            compared_steps = compared_steps
                .checked_add(1)
                .ok_or("cached compared-step count overflow")?;
            if step + 1 < case.golden_token_ids.len() {
                let target_length = prompt_length
                    .checked_add(step + 1)
                    .ok_or("cached decode target length overflow")?;
                let (ids, valid) = block_table(target_length, 0)?;
                let tokens = [actual_token_id];
                let decode_rows = [row(
                    u64::try_from(case.index + 1)?,
                    LlamaBatchRowKind::Decode,
                    &tokens,
                    target_length,
                    &ids,
                    &valid,
                    Some(0),
                )?];
                batch.execute(&decode_rows, stream)?;
                assert_eq!(context.allocation_stats()?, stable);
            }
        }
    }
    batch.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(compared_steps)
}

fn run_fixed37_growing_prefix_group(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    prompt_length: usize,
    cases: &[Fixed37BatchGoldenCase],
) -> TestResult<(usize, LogitMetrics)> {
    let profile = LlamaReductionProfile::FixedContiguous37BalancedV1;
    let maximum_steps = cases
        .iter()
        .map(|case| case.golden_token_ids.len())
        .max()
        .ok_or("golden prompt-length group is empty")?;
    let row_bytes = vocabulary_row_bytes(model)?;
    let mut compared_steps = 0_usize;
    let mut worst_metrics = LogitMetrics {
        cosine: 1.0,
        max_abs: 0.0,
        mean_abs: 0.0,
    };
    for step in 0..maximum_steps {
        let active = cases
            .iter()
            .filter(|case| step < case.golden_token_ids.len())
            .collect::<Vec<_>>();
        if active.is_empty() {
            continue;
        }
        let sequence_length = prompt_length
            .checked_add(step)
            .ok_or("growing-prefix sequence length overflow")?;
        let mut batch = PreparedLlamaBatchExecutor::prepare(
            model,
            context,
            stream,
            production_batch_config(sequence_length, sequence_length, profile)?,
        )?;
        assert_eq!(batch.reduction_profile(), profile);
        assert!(batch.reduction_profile_is_coherent());
        let stable = context.allocation_stats()?;
        let mut logits = vec![0_u8; row_bytes];
        for case in active {
            let mut prefix = Vec::with_capacity(sequence_length);
            prefix.extend_from_slice(&case.prompt_token_ids);
            prefix.extend_from_slice(&case.golden_token_ids[..step]);
            let (ids, valid) = block_table(sequence_length, 0)?;
            let rows = [row(
                u64::try_from(case.index + 1)?,
                LlamaBatchRowKind::Prefill,
                &prefix,
                sequence_length,
                &ids,
                &valid,
                Some(0),
            )?];
            batch.execute(&rows, stream)?;
            assert_eq!(context.allocation_stats()?, stable);
            batch.download_logits(&mut logits, stream)?;
            let label = format!(
                "fixed37-cached-growing-prompt-{}-step-{step}",
                case.prompt_id
            );
            let metrics = assert_semantic_parity(&label, &logits, &case.fixed_cached_logits[step]);
            worst_metrics.cosine = worst_metrics.cosine.min(metrics.cosine);
            worst_metrics.max_abs = worst_metrics.max_abs.max(metrics.max_abs);
            worst_metrics.mean_abs = worst_metrics.mean_abs.max(metrics.mean_abs);
            if step == 0 {
                // Both sides are fixed37 production-batch prefill. Decode versus
                // growing-prefix prefill uses different attention paths, so only
                // this structurally identical path carries a raw-byte contract.
                assert_exact_bytes(&label, &logits, &case.fixed_cached_logits[step]);
            }
            assert_eq!(
                top1(&logits),
                top1(&case.fixed_cached_logits[step]),
                "fixed37 cached/growing production batch top-1 differs prompt={} step={step}",
                case.prompt_id
            );
            assert_eq!(
                u32::try_from(top1(&logits))?,
                case.golden_token_ids[step],
                "fixed37 growing-prefix production batch differs from golden prompt={} step={step}",
                case.prompt_id
            );
            compared_steps = compared_steps
                .checked_add(1)
                .ok_or("growing-prefix compared-step count overflow")?;
        }
        batch.close()?;
        assert!(context.allocation_stats()?.is_zero());
    }
    Ok((compared_steps, worst_metrics))
}

fn independent_last_logits(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    tokens: &[u32],
) -> TestResult<Vec<u8>> {
    let mut forward = PreparedLlamaForward::prepare(
        model,
        context,
        stream,
        tokens.len(),
        PreparedLlamaForwardConfig::default(),
    )?;
    forward.forward(tokens, stream)?;
    let mut logits = vec![0_u8; vocabulary_row_bytes(model)?];
    forward.download_last_logits(&mut logits, stream)?;
    forward.close()?;
    Ok(logits)
}

fn bf16(value: &[u8]) -> f64 {
    let bits = u32::from(u16::from_ne_bytes([value[0], value[1]])) << 16;
    f64::from(f32::from_bits(bits))
}

fn top1(logits: &[u8]) -> usize {
    logits
        .chunks_exact(BF16_BYTES)
        .enumerate()
        .max_by(|(left_id, left), (right_id, right)| {
            bf16(left)
                .total_cmp(&bf16(right))
                .then_with(|| right_id.cmp(left_id))
        })
        .map_or(0, |(token, _)| token)
}

fn assert_semantic_parity(label: &str, actual: &[u8], expected: &[u8]) -> LogitMetrics {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    assert_eq!(actual.len() % BF16_BYTES, 0);
    let actual_top1 = top1(actual);
    let expected_top1 = top1(expected);
    assert_eq!(actual_top1, expected_top1, "{label} top-1 differs");
    let mut dot = 0.0_f64;
    let mut actual_norm = 0.0_f64;
    let mut expected_norm = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for (actual, expected) in actual
        .chunks_exact(BF16_BYTES)
        .zip(expected.chunks_exact(BF16_BYTES))
    {
        let actual = bf16(actual);
        let expected = bf16(expected);
        assert!(actual.is_finite(), "{label} actual logit is not finite");
        assert!(expected.is_finite(), "{label} expected logit is not finite");
        dot += actual * expected;
        actual_norm += actual * actual;
        expected_norm += expected * expected;
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
    }
    assert!(actual_norm > 0.0, "{label} actual logits have zero norm");
    assert!(
        expected_norm > 0.0,
        "{label} expected logits have zero norm"
    );
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    let mean_abs = sum_abs / ((actual.len() / BF16_BYTES) as f64);
    let actual_top1_value = bf16(&actual[actual_top1 * BF16_BYTES..][..BF16_BYTES]);
    let expected_top1_value = bf16(&expected[expected_top1 * BF16_BYTES..][..BF16_BYTES]);
    let metrics = LogitMetrics {
        cosine,
        max_abs,
        mean_abs,
    };
    assert!(
        cosine.is_finite() && max_abs.is_finite() && mean_abs.is_finite(),
        "{label} metrics are not finite: {metrics:?}"
    );
    println!(
        "pr13-batch-parity schema_version=1 label={label} cosine={cosine:.12} \
max_abs={max_abs:.9} mean_abs={mean_abs:.9} actual_top1={actual_top1} \
expected_top1={expected_top1} actual_top1_value={actual_top1_value:.6} \
expected_top1_value={expected_top1_value:.6} top1_exact=true \
cosine_min={BATCH_LOGIT_COSINE_MIN} max_abs_max={BATCH_LOGIT_MAX_ABS_MAX} \
mean_abs_max={BATCH_LOGIT_MEAN_ABS_MAX}"
    );
    assert!(
        cosine >= BATCH_LOGIT_COSINE_MIN,
        "{label} failed the predeclared batch cosine gate: {metrics:?}"
    );
    assert!(
        max_abs <= BATCH_LOGIT_MAX_ABS_MAX,
        "{label} failed the predeclared batch max-abs gate: {metrics:?}"
    );
    assert!(
        mean_abs <= BATCH_LOGIT_MEAN_ABS_MAX,
        "{label} failed the predeclared batch mean-abs gate: {metrics:?}"
    );
    metrics
}

fn assert_report_matches_context(
    report: riley_runtime::llama::PreparedLlamaBatchAllocationReport,
    stats: CudaAllocationStats,
) {
    assert_eq!(report.total_device_bytes(), stats.device_live_bytes());
    assert_eq!(
        report.total_device_allocation_count(),
        stats.device_live_allocations()
    );
    assert_eq!(report.pinned_host_bytes(), stats.pinned_host_live_bytes());
    assert_eq!(
        report.pinned_host_allocation_count(),
        stats.pinned_host_live_allocations()
    );
}

fn assert_exact_bytes(label: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len(), "{label} byte length differs");
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected);
    assert!(
        mismatch.is_none(),
        "{label} differs at byte {}: actual={} reference={}",
        mismatch.map_or(0, |(index, _)| index),
        mismatch.map_or(0, |(_, (actual, _))| *actual),
        mismatch.map_or(0, |(_, (_, expected))| *expected),
    );
}

fn greedy_execution_trace(
    model: &LoadedModel,
    residual_norm: ResidualNormImplementation,
    execution_completion: ExecutionCompletionImplementation,
    reduction_profile: LlamaReductionProfile,
    ragged_attention_reduction_profile: Option<AttentionReductionProfile>,
    decode_steps: usize,
) -> TestResult<GreedyExecutionTrace> {
    let maximum_length = TOKENS_A
        .len()
        .checked_add(decode_steps)
        .ok_or("maximum sequence length overflow")?;
    let physical_blocks = maximum_length.div_ceil(KV_BLOCK_SIZE);
    let (context, mut stream) = first_context()?;
    let config = match residual_norm {
        ResidualNormImplementation::Separate => {
            batch_config(1, TOKENS_A.len(), physical_blocks, 1, physical_blocks)?
                .with_separate_residual_norm()
        }
        ResidualNormImplementation::Fused => {
            batch_config(1, TOKENS_A.len(), physical_blocks, 1, physical_blocks)?
                .with_fused_residual_norm()
        }
    };
    let config = match execution_completion {
        ExecutionCompletionImplementation::PerOperation => config.with_per_operation_completion(),
        ExecutionCompletionImplementation::IterationBatch => {
            config.with_iteration_batch_completion()
        }
    }
    .with_reduction_profile(reduction_profile);
    let config = ragged_attention_reduction_profile.map_or(config, |profile| {
        config.with_ragged_attention_reduction_profile(profile)
    });
    let mut batch = PreparedLlamaBatchExecutor::prepare(model, &context, &mut stream, config)?;
    assert_eq!(batch.reduction_profile(), reduction_profile);
    if ragged_attention_reduction_profile.is_none() {
        assert!(batch.reduction_profile_is_coherent());
    }
    let expected_ragged_profile =
        ragged_attention_reduction_profile.unwrap_or(match reduction_profile {
            LlamaReductionProfile::CanonicalV1 => AttentionReductionProfile::CanonicalV1,
            LlamaReductionProfile::FixedContiguous37BalancedV1 => {
                AttentionReductionProfile::FixedContiguous37BalancedV1
            }
        });
    assert_eq!(
        batch.config().ragged_attention_reduction_profile(),
        expected_ragged_profile
    );
    let (prompt_ids, prompt_valid) = block_table(TOKENS_A.len(), 0)?;
    let prompt_rows = [row(
        15,
        LlamaBatchRowKind::Prefill,
        &TOKENS_A,
        TOKENS_A.len(),
        &prompt_ids,
        &prompt_valid,
        Some(0),
    )?];
    batch.execute(&prompt_rows, &mut stream)?;

    let mut logits = vec![0_u8; vocabulary_row_bytes(model)?];
    batch.download_logits(&mut logits, &mut stream)?;
    let stable = context.allocation_stats()?;
    let mut trace = GreedyExecutionTrace {
        generated_token_ids: Vec::with_capacity(decode_steps),
        logits_by_iteration: Vec::with_capacity(decode_steps + 1),
        cuda_live_allocation_delta: 0,
        owner_close_live_allocation_count: 0,
    };
    trace.logits_by_iteration.push(logits.clone());

    for step in 0..decode_steps {
        let token = u32::try_from(top1(&logits))?;
        trace.generated_token_ids.push(token);
        let target_length = TOKENS_A.len() + step + 1;
        let (ids, valid) = block_table(target_length, 0)?;
        let tokens = [token];
        let rows = [row(
            15,
            LlamaBatchRowKind::Decode,
            &tokens,
            target_length,
            &ids,
            &valid,
            Some(0),
        )?];
        batch.execute(&rows, &mut stream)?;
        batch.download_logits(&mut logits, &mut stream)?;
        trace.logits_by_iteration.push(logits.clone());
        let current = context.allocation_stats()?;
        assert_eq!(
            current,
            stable,
            "{residual_norm:?}/{execution_completion:?}/{reduction_profile:?}/{ragged_attention_reduction_profile:?} allocation changed after committed decode step {}",
            step + 1,
        );
        trace.cuda_live_allocation_delta = live_allocation_delta(stable, current);
    }

    batch.close()?;
    let after_owner_close = context.allocation_stats()?;
    trace.owner_close_live_allocation_count = live_allocation_count(after_owner_close);
    assert!(
        after_owner_close.is_zero(),
        "{residual_norm:?}/{execution_completion:?}/{reduction_profile:?}/{ragged_attention_reduction_profile:?} executor close leaked CUDA allocations"
    );
    stream.close()?;
    close_context(context)?;
    Ok(trace)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn concurrency_one_matches_the_single_request_forward() -> TestResult {
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let (context, mut stream) = first_context()?;
    let expected = independent_last_logits(&model, &context, &mut stream, &TOKENS_A)?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(1, 8, 8, 1, 32)?,
    )?;
    assert_report_matches_context(batch.allocation_report(), context.allocation_stats()?);
    let (ids, valid) = block_table(TOKENS_A.len(), 0)?;
    let rows = [row(
        1,
        LlamaBatchRowKind::Prefill,
        &TOKENS_A,
        TOKENS_A.len(),
        &ids,
        &valid,
        Some(0),
    )?];
    batch.execute(&rows, &mut stream)?;
    let mut actual = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut actual, &mut stream)?;
    assert_semantic_parity("concurrency-one", &actual, &expected);
    batch.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn fused_residual_norm_matches_separate_multi_step_greedy_exactly() -> TestResult {
    const DECODE_STEPS: usize = 16;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let separate = greedy_execution_trace(
        &model,
        ResidualNormImplementation::Separate,
        ExecutionCompletionImplementation::PerOperation,
        LlamaReductionProfile::CanonicalV1,
        Some(AttentionReductionProfile::CanonicalV1),
        DECODE_STEPS,
    )?;
    let fused = greedy_execution_trace(
        &model,
        ResidualNormImplementation::Fused,
        ExecutionCompletionImplementation::PerOperation,
        LlamaReductionProfile::CanonicalV1,
        Some(AttentionReductionProfile::CanonicalV1),
        DECODE_STEPS,
    )?;

    assert_eq!(
        &fused.generated_token_ids, &separate.generated_token_ids,
        "fused and separate greedy token IDs differ"
    );
    assert_eq!(
        fused.logits_by_iteration.len(),
        separate.logits_by_iteration.len()
    );
    for (iteration, (fused_logits, separate_logits)) in fused
        .logits_by_iteration
        .iter()
        .zip(&separate.logits_by_iteration)
        .enumerate()
    {
        assert_exact_bytes(
            &format!("residual-norm iteration {iteration}"),
            fused_logits,
            separate_logits,
        );
        assert_eq!(
            top1(fused_logits),
            top1(separate_logits),
            "iteration {iteration} top-1 differs"
        );
    }
    println!(
        "pr15-residual-rmsnorm-parity schema_version=1 decode_steps={DECODE_STEPS} \
committed_iterations={} raw_logit_mismatches=0 generated_token_ids={:?} status=passed",
        fused.logits_by_iteration.len() - 1,
        fused.generated_token_ids,
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly() -> TestResult {
    const DECODE_STEPS: usize = 16;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let per_operation = greedy_execution_trace(
        &model,
        ResidualNormImplementation::Separate,
        ExecutionCompletionImplementation::PerOperation,
        LlamaReductionProfile::CanonicalV1,
        Some(AttentionReductionProfile::CanonicalV1),
        DECODE_STEPS,
    )?;
    let iteration_batch = greedy_execution_trace(
        &model,
        ResidualNormImplementation::Separate,
        ExecutionCompletionImplementation::IterationBatch,
        LlamaReductionProfile::CanonicalV1,
        Some(AttentionReductionProfile::CanonicalV1),
        DECODE_STEPS,
    )?;

    let token_id_mismatches = iteration_batch
        .generated_token_ids
        .iter()
        .zip(&per_operation.generated_token_ids)
        .filter(|(actual, expected)| actual != expected)
        .count()
        + iteration_batch
            .generated_token_ids
            .len()
            .abs_diff(per_operation.generated_token_ids.len());
    assert_eq!(token_id_mismatches, 0);
    assert_eq!(
        iteration_batch.logits_by_iteration.len(),
        per_operation.logits_by_iteration.len()
    );
    let mut raw_logit_mismatches = 0_usize;
    for (iteration, (batched_logits, per_operation_logits)) in iteration_batch
        .logits_by_iteration
        .iter()
        .zip(&per_operation.logits_by_iteration)
        .enumerate()
    {
        raw_logit_mismatches = raw_logit_mismatches
            .checked_add(
                batched_logits
                    .iter()
                    .zip(per_operation_logits)
                    .filter(|(actual, expected)| actual != expected)
                    .count(),
            )
            .and_then(|count| {
                count.checked_add(batched_logits.len().abs_diff(per_operation_logits.len()))
            })
            .expect("raw logit mismatch count overflow");
        assert_exact_bytes(
            &format!("execution-completion iteration {iteration}"),
            batched_logits,
            per_operation_logits,
        );
        assert_eq!(
            top1(batched_logits),
            top1(per_operation_logits),
            "iteration {iteration} top-1 differs"
        );
    }
    assert_eq!(raw_logit_mismatches, 0);
    println!(
        "pr15-execution-completion-parity schema_version=1 decode_steps={DECODE_STEPS} \
committed_iterations={} raw_logit_mismatches={raw_logit_mismatches} \
token_id_mismatches={token_id_mismatches} \
cuda_live_allocation_delta={} owner_close_live_allocation_count={} \
generated_token_ids={:?} status=passed",
        iteration_batch.logits_by_iteration.len() - 1,
        iteration_batch.cuda_live_allocation_delta,
        iteration_batch.owner_close_live_allocation_count,
        iteration_batch.generated_token_ids,
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn fixed37_ragged_attention_completion_modes_match_multi_step_greedy_exactly() -> TestResult {
    const DECODE_STEPS: usize = 16;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let per_operation = greedy_execution_trace(
        &model,
        ResidualNormImplementation::Separate,
        ExecutionCompletionImplementation::PerOperation,
        LlamaReductionProfile::CanonicalV1,
        Some(AttentionReductionProfile::FixedContiguous37BalancedV1),
        DECODE_STEPS,
    )?;
    let iteration_batch = greedy_execution_trace(
        &model,
        ResidualNormImplementation::Separate,
        ExecutionCompletionImplementation::IterationBatch,
        LlamaReductionProfile::CanonicalV1,
        Some(AttentionReductionProfile::FixedContiguous37BalancedV1),
        DECODE_STEPS,
    )?;

    assert_eq!(
        iteration_batch.generated_token_ids, per_operation.generated_token_ids,
        "fixed37 completion modes generated different token IDs"
    );
    assert_eq!(
        iteration_batch.logits_by_iteration.len(),
        per_operation.logits_by_iteration.len()
    );
    for (iteration, (batched_logits, per_operation_logits)) in iteration_batch
        .logits_by_iteration
        .iter()
        .zip(&per_operation.logits_by_iteration)
        .enumerate()
    {
        assert_exact_bytes(
            &format!("fixed37 execution-completion iteration {iteration}"),
            batched_logits,
            per_operation_logits,
        );
        assert_eq!(
            top1(batched_logits),
            top1(per_operation_logits),
            "fixed37 iteration {iteration} top-1 differs"
        );
    }
    assert_eq!(iteration_batch.cuda_live_allocation_delta, 0);
    assert_eq!(iteration_batch.owner_close_live_allocation_count, 0);
    println!(
        "pr16-fixed37-ragged-completion-parity schema_version=1 decode_steps={DECODE_STEPS} \
committed_iterations={} raw_logit_mismatches=0 token_id_mismatches=0 \
cuda_live_allocation_delta={} owner_close_live_allocation_count={} \
generated_token_ids={:?} status=passed",
        iteration_batch.logits_by_iteration.len() - 1,
        iteration_batch.cuda_live_allocation_delta,
        iteration_batch.owner_close_live_allocation_count,
        iteration_batch.generated_token_ids,
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn fixed37_production_batch_growing_prefix_matches_golden_exactly() -> TestResult {
    let mut fixture = parse_fixed37_batch_golden_fixture()?;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let fixed_profile = LlamaReductionProfile::FixedContiguous37BalancedV1;
    let canonical_profile = LlamaReductionProfile::CanonicalV1;
    let (context, mut stream) = first_context()?;
    let mut fixed_cached_steps = 0_usize;
    let mut canonical_cached_steps = 0_usize;
    let mut fixed_growing_steps = 0_usize;
    let mut fixed_cached_growing_worst = LogitMetrics {
        cosine: 1.0,
        max_abs: 0.0,
        mean_abs: 0.0,
    };
    for (&prompt_length, cases) in &mut fixture.cases_by_prompt_length {
        fixed_cached_steps = fixed_cached_steps
            .checked_add(run_cached_batch_golden_group(
                &model,
                &context,
                &mut stream,
                prompt_length,
                cases,
                fixed_profile,
                true,
            )?)
            .ok_or("fixed cached step count overflow")?;
        canonical_cached_steps = canonical_cached_steps
            .checked_add(run_cached_batch_golden_group(
                &model,
                &context,
                &mut stream,
                prompt_length,
                cases,
                canonical_profile,
                false,
            )?)
            .ok_or("canonical cached step count overflow")?;
        let (group_steps, group_metrics) =
            run_fixed37_growing_prefix_group(&model, &context, &mut stream, prompt_length, cases)?;
        fixed_growing_steps = fixed_growing_steps
            .checked_add(group_steps)
            .ok_or("fixed growing-prefix step count overflow")?;
        fixed_cached_growing_worst.cosine =
            fixed_cached_growing_worst.cosine.min(group_metrics.cosine);
        fixed_cached_growing_worst.max_abs = fixed_cached_growing_worst
            .max_abs
            .max(group_metrics.max_abs);
        fixed_cached_growing_worst.mean_abs = fixed_cached_growing_worst
            .mean_abs
            .max(group_metrics.mean_abs);
    }
    assert_eq!(fixed_cached_steps, fixture.total_generated_steps);
    assert_eq!(canonical_cached_steps, fixture.total_generated_steps);
    assert_eq!(fixed_growing_steps, fixture.total_generated_steps);
    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    close_context(context)?;
    println!(
        "pr16-fixed37-production-batch-e0-v1 schema_version=1 \
fixture_sha256={} generated_token_ids_sha256={} cases={} compared_steps={} \
exact_window={} fixed_profile={} canonical_profile={} residual_rmsnorm=separate \
execution_completion=iteration-batch fixed_prefill_raw_logit_mismatches=0 \
fixed_cached_growing_token_id_mismatches=0 \
fixed_cached_growing_cosine_min={BATCH_LOGIT_COSINE_MIN} \
fixed_cached_growing_max_abs_max={BATCH_LOGIT_MAX_ABS_MAX} \
fixed_cached_growing_mean_abs_max={BATCH_LOGIT_MEAN_ABS_MAX} \
fixed_cached_growing_worst_cosine={:.17} fixed_cached_growing_worst_max_abs={:.9} \
fixed_cached_growing_worst_mean_abs={:.9} fixed_cached_growing_threshold_violations=0 \
fixed_golden_token_id_mismatches=0 \
canonical_golden_token_id_mismatches=0 cuda_live_allocation_delta=0 \
owner_close_live_allocation_count=0 status=passed",
        fixture.fixture_sha256,
        fixture.generated_token_ids_sha256,
        EXPECTED_GOLDEN_CASES,
        fixture.total_generated_steps,
        fixture.exact_window,
        fixed_profile.id(),
        canonical_profile.id(),
        fixed_cached_growing_worst.cosine,
        fixed_cached_growing_worst.max_abs,
        fixed_cached_growing_worst.mean_abs,
    );
    Ok(())
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn fixed37_whole_profile_selects_forward_and_both_decode_cache_paths() -> TestResult {
    const DECODE_STEPS: usize = 16;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let profile = LlamaReductionProfile::FixedContiguous37BalancedV1;
    let attention_profile = AttentionReductionProfile::FixedContiguous37BalancedV1;
    let (context, mut stream) = first_context()?;

    let mut forward = PreparedLlamaForward::prepare(
        &model,
        &context,
        &mut stream,
        TOKENS_A.len(),
        PreparedLlamaForwardConfig::default().with_reduction_profile(profile),
    )?;
    assert_eq!(forward.reduction_profile(), profile);
    assert_eq!(
        forward.attention_selection().reduction_profile(),
        attention_profile
    );
    forward.forward(&TOKENS_A, &mut stream)?;
    let mut forward_logits = vec![0_u8; vocabulary_row_bytes(&model)?];
    forward.download_last_logits(&mut forward_logits, &mut stream)?;
    forward.close()?;
    assert!(context.allocation_stats()?.is_zero());

    let contiguous = prepared_decode_profile_trace(
        &model,
        &context,
        &mut stream,
        PreparedLlamaDecodeConfig::default()
            .with_reduction_profile(profile)
            .with_contiguous_kv_cache(),
        DECODE_STEPS,
    )?;
    let paged = prepared_decode_profile_trace(
        &model,
        &context,
        &mut stream,
        PreparedLlamaDecodeConfig::default()
            .with_reduction_profile(profile)
            .with_paged_kv_cache(),
        DECODE_STEPS,
    )?;

    assert_eq!(
        paged.0, contiguous.0,
        "fixed37 decode cache token IDs differ"
    );
    assert_eq!(paged.1.len(), contiguous.1.len());
    for (iteration, (paged_logits, contiguous_logits)) in
        paged.1.iter().zip(&contiguous.1).enumerate()
    {
        assert_exact_bytes(
            &format!("fixed37 paged/contiguous decode iteration {iteration}"),
            paged_logits,
            contiguous_logits,
        );
    }
    println!(
        "pr16-fixed37-whole-profile-runtime-selection schema_version=1 profile={} \
decode_steps={DECODE_STEPS} forward_attention_profile={attention_profile:?} \
decode_cache_paths=contiguous,paged raw_logit_mismatches=0 token_id_mismatches=0 \
generated_token_ids={:?} status=passed",
        profile.id(),
        paged.0,
    );
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned checkpoint and CUDA GPU on server-4096"]
fn concurrency_one_matches_prepared_decode_for_thirty_two_steps() -> TestResult {
    const DECODE_STEPS: usize = 32;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let maximum_length = TOKENS_A.len() + DECODE_STEPS;
    let physical_blocks = maximum_length.div_ceil(KV_BLOCK_SIZE);
    let (context, mut stream) = first_context()?;
    let mut reference = PreparedLlamaDecode::prepare(
        &model,
        &context,
        &mut stream,
        TOKENS_A.len(),
        maximum_length,
        PreparedLlamaDecodeConfig::default(),
    )?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(1, 8, physical_blocks, 1, physical_blocks)?,
    )?;

    reference.prefill(&TOKENS_A, &mut stream)?;
    let (prompt_ids, prompt_valid) = block_table(TOKENS_A.len(), 0)?;
    let prompt_rows = [row(
        71,
        LlamaBatchRowKind::Prefill,
        &TOKENS_A,
        TOKENS_A.len(),
        &prompt_ids,
        &prompt_valid,
        Some(0),
    )?];
    batch.execute(&prompt_rows, &mut stream)?;
    let row_bytes = vocabulary_row_bytes(&model)?;
    let mut reference_logits = vec![0_u8; row_bytes];
    let mut batch_logits = vec![0_u8; row_bytes];
    reference.download_last_logits(&mut reference_logits, &mut stream)?;
    batch.download_logits(&mut batch_logits, &mut stream)?;
    let prefill_metrics =
        assert_semantic_parity("decode-step-prefill", &batch_logits, &reference_logits);
    let mut worst_cosine = prefill_metrics.cosine;
    let mut worst_max_abs = prefill_metrics.max_abs;
    let mut worst_mean_abs = prefill_metrics.mean_abs;
    let stable = context.allocation_stats()?;

    for step in 0..DECODE_STEPS {
        let token = u32::try_from(top1(&reference_logits))?;
        reference.decode(token, &mut stream)?;
        let target_length = TOKENS_A.len() + step + 1;
        let (ids, valid) = block_table(target_length, 0)?;
        let tokens = [token];
        let rows = [row(
            71,
            LlamaBatchRowKind::Decode,
            &tokens,
            target_length,
            &ids,
            &valid,
            Some(0),
        )?];
        batch.execute(&rows, &mut stream)?;
        reference.download_last_logits(&mut reference_logits, &mut stream)?;
        batch.download_logits(&mut batch_logits, &mut stream)?;
        let metrics = assert_semantic_parity(
            &format!("decode-step-{}", step + 1),
            &batch_logits,
            &reference_logits,
        );
        worst_cosine = worst_cosine.min(metrics.cosine);
        worst_max_abs = worst_max_abs.max(metrics.max_abs);
        worst_mean_abs = worst_mean_abs.max(metrics.mean_abs);
        assert_eq!(context.allocation_stats()?, stable, "decode step {step}");
    }
    println!(
        "pr13-batch-parity-summary schema_version=1 fixed_m=8 parity_rows={} \
top1_mismatches=0 worst_cosine={worst_cosine:.12} worst_max_abs={worst_max_abs:.9} \
worst_mean_abs={worst_mean_abs:.9} status=passed",
        DECODE_STEPS + 1,
    );

    batch.close()?;
    reference.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn multiple_independent_requests_and_permuted_output_slots_match() -> TestResult {
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let (context, mut stream) = first_context()?;
    let expected_a = independent_last_logits(&model, &context, &mut stream, &TOKENS_A[..3])?;
    let expected_b = independent_last_logits(&model, &context, &mut stream, &TOKENS_B)?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(2, 8, 8, 2, 32)?,
    )?;
    let (ids_a, valid_a) = block_table(3, 0)?;
    let (ids_b, valid_b) = block_table(TOKENS_B.len(), 1)?;
    let rows = [
        row(
            11,
            LlamaBatchRowKind::Prefill,
            &TOKENS_A[..3],
            3,
            &ids_a,
            &valid_a,
            Some(1),
        )?,
        row(
            22,
            LlamaBatchRowKind::Prefill,
            &TOKENS_B,
            TOKENS_B.len(),
            &ids_b,
            &valid_b,
            Some(0),
        )?,
    ];
    batch.execute(&rows, &mut stream)?;
    let row_bytes = vocabulary_row_bytes(&model)?;
    let mut actual = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut actual, &mut stream)?;
    assert_semantic_parity("permuted-slot-b", &actual[..row_bytes], &expected_b);
    assert_semantic_parity("permuted-slot-a", &actual[row_bytes..], &expected_a);
    batch.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn mixed_prefill_chunk_and_decode_match_independent_full_sequences() -> TestResult {
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let (context, mut stream) = first_context()?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(2, 8, 8, 2, 32)?,
    )?;
    let (ids_a3, valid_a3) = block_table(3, 0)?;
    let (ids_b2, valid_b2) = block_table(2, 1)?;
    let initial = [
        row(
            31,
            LlamaBatchRowKind::Prefill,
            &TOKENS_A[..3],
            3,
            &ids_a3,
            &valid_a3,
            Some(0),
        )?,
        row(
            32,
            LlamaBatchRowKind::Prefill,
            &TOKENS_B[..2],
            2,
            &ids_b2,
            &valid_b2,
            Some(1),
        )?,
    ];
    batch.execute(&initial, &mut stream)?;
    let row_bytes = vocabulary_row_bytes(&model)?;
    let mut initial_logits = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut initial_logits, &mut stream)?;
    let decode_token = u32::try_from(top1(&initial_logits[..row_bytes]))?;
    let continuation = &TOKENS_B[2..4];
    let (ids_a4, valid_a4) = block_table(4, 0)?;
    let (ids_b4, valid_b4) = block_table(4, 1)?;
    let mixed = [
        row(
            31,
            LlamaBatchRowKind::Decode,
            std::slice::from_ref(&decode_token),
            4,
            &ids_a4,
            &valid_a4,
            Some(1),
        )?,
        row(
            32,
            LlamaBatchRowKind::Prefill,
            continuation,
            4,
            &ids_b4,
            &valid_b4,
            Some(0),
        )?,
    ];
    batch.execute(&mixed, &mut stream)?;
    let mut actual = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut actual, &mut stream)?;
    let mut full_a = TOKENS_A[..3].to_vec();
    full_a.push(decode_token);
    let expected_a = independent_last_logits(&model, &context, &mut stream, &full_a)?;
    let expected_b = independent_last_logits(&model, &context, &mut stream, &TOKENS_B)?;
    assert_semantic_parity("mixed-prefill-b", &actual[..row_bytes], &expected_b);
    assert_semantic_parity("mixed-decode-a", &actual[row_bytes..], &expected_a);
    batch.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote 1000-iteration allocation-accounting gate on server-4096"]
fn one_thousand_iterations_do_not_allocate_or_leak() -> TestResult {
    if !env_enabled("RILEY_PR13_LONG_STEPS") {
        eprintln!("pr13-long-gate-skipped env=RILEY_PR13_LONG_STEPS expected=true");
        return Ok(());
    }
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    assert!(model.spec().max_sequence_length() >= 1_001);
    let (context, mut stream) = first_context()?;
    let physical_blocks = 1_001_usize.div_ceil(KV_BLOCK_SIZE);
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(1, 1, physical_blocks, 1, physical_blocks)?,
    )?;
    let stable = context.allocation_stats()?;
    let mut token = TOKENS_A[0];
    let maximum_blocks = 1_001_usize.div_ceil(KV_BLOCK_SIZE);
    let ids: Vec<u32> = (0..maximum_blocks)
        .map(u32::try_from)
        .collect::<Result<_, _>>()?;
    let mut valid = vec![u16::try_from(KV_BLOCK_SIZE)?; maximum_blocks];
    let mut logits = vec![0_u8; vocabulary_row_bytes(&model)?];
    for target_length in 1_usize..=1_001 {
        let block_count = target_length.div_ceil(KV_BLOCK_SIZE);
        valid[block_count - 1] = u16::try_from(target_length - (block_count - 1) * KV_BLOCK_SIZE)?;
        let kind = if target_length == 1 {
            LlamaBatchRowKind::Prefill
        } else {
            LlamaBatchRowKind::Decode
        };
        let tokens = [token];
        let rows = [row(
            99,
            kind,
            &tokens,
            target_length,
            &ids[..block_count],
            &valid[..block_count],
            Some(0),
        )?];
        batch.execute(&rows, &mut stream)?;
        batch.download_logits(&mut logits, &mut stream)?;
        token = u32::try_from(top1(&logits))?;
        assert_eq!(
            context.allocation_stats()?,
            stable,
            "iteration={target_length}"
        );
    }
    batch.close()?;
    stream.close()?;
    close_context(context)
}
