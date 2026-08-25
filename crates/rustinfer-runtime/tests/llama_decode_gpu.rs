//! Remote-only PR09 contiguous-KV Llama decode validation and evidence markers.

#![cfg(feature = "cuda")]
#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use rustinfer_cuda::{
    CudaAllocationStats, CudaContext, CudaRuntime, CudaStream, DecodeAttentionBackend,
};
use rustinfer_model::{LoadLimits, LoadedModel};
use rustinfer_runtime::llama::{
    LlamaDecodeError, LlamaDecodePhase, PreparedLlamaAllocationReport, PreparedLlamaDecode,
    PreparedLlamaDecodeAllocationReport, PreparedLlamaDecodeConfig, PreparedLlamaForward,
    PreparedLlamaForwardConfig,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const PINNED_TOKENS_A: [u32; 7] = [504, 2_365, 6_354, 16_438, 11_139, 253, 1_890];
const PINNED_TOKENS_B: [u32; 7] = [504, 2_365, 6_354, 16_438, 42, 43, 44];
const BF16_BYTES: usize = 2;
const DEFAULT_PARITY_DECODE_CALLS: usize = 32;
const LONG_PARITY_DECODE_CALLS: usize = 128;
const REFERENCE_DECODE_IMPLEMENTATION: &str = "rustinfer.cuda.materialized-gqa-decode.bf16";
const OPTIMIZED_DECODE_IMPLEMENTATION: &str = "rustinfer.cuda.chunked-online-gqa-decode.bf16.d64";

#[derive(Clone, Copy, Debug)]
struct NumericMetrics {
    cosine: f64,
    max_abs: f64,
    mean_abs: f64,
}

struct CachedChainOutcome {
    consumed_tokens: Vec<u32>,
    logits: Vec<u8>,
    row_bytes: usize,
    implementation_id: &'static str,
}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory")
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

fn row_bytes(model: &LoadedModel) -> TestResult<usize> {
    model
        .spec()
        .embedding()
        .vocabulary_size()
        .checked_mul(BF16_BYTES)
        .ok_or_else(|| "logit row byte length overflow".into())
}

fn decode_bf16_scalar(bytes: &[u8]) -> f32 {
    assert_eq!(bytes.len(), BF16_BYTES);
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn numeric_metrics(actual: &[u8], expected: &[u8]) -> NumericMetrics {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    assert_eq!(actual.len() % BF16_BYTES, 0);
    let mut dot = 0.0_f64;
    let mut actual_norm = 0.0_f64;
    let mut expected_norm = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut all_equal = true;
    for (actual, expected) in actual
        .chunks_exact(BF16_BYTES)
        .zip(expected.chunks_exact(BF16_BYTES))
    {
        let actual = decode_bf16_scalar(actual);
        let expected = decode_bf16_scalar(expected);
        assert!(
            actual.is_finite() && expected.is_finite(),
            "PR09 logits must remain finite"
        );
        all_equal &= actual == expected;
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        dot = actual.mul_add(expected, dot);
        actual_norm = actual.mul_add(actual, actual_norm);
        expected_norm = expected.mul_add(expected, expected_norm);
    }
    let cosine = if actual_norm == 0.0 || expected_norm == 0.0 {
        if all_equal { 1.0 } else { 0.0 }
    } else {
        dot / (actual_norm.sqrt() * expected_norm.sqrt())
    };
    NumericMetrics {
        cosine,
        max_abs,
        mean_abs: sum_abs / (actual.len() / BF16_BYTES) as f64,
    }
}

fn top1(bytes: &[u8]) -> usize {
    assert_eq!(bytes.len() % BF16_BYTES, 0);
    let mut values = bytes.chunks_exact(BF16_BYTES).map(decode_bf16_scalar);
    let first = values
        .next()
        .expect("the model vocabulary must be non-empty");
    assert!(first.is_finite());
    let mut best_index = 0;
    let mut best_value = first;
    for (offset, value) in values.enumerate() {
        assert!(value.is_finite());
        if value.total_cmp(&best_value).is_gt() {
            best_index = offset + 1;
            best_value = value;
        }
    }
    best_index
}

fn assert_semantic_logits(label: &str, actual: &[u8], expected: &[u8]) -> NumericMetrics {
    let metrics = numeric_metrics(actual, expected);
    assert_eq!(top1(actual), top1(expected), "{label} top-1 token");
    metrics
}

fn assert_decode_report_matches_context(
    report: PreparedLlamaDecodeAllocationReport,
    stats: CudaAllocationStats,
) {
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
    assert_eq!(
        report.additional_device_bytes(),
        report.kv_cache_bytes()
            + report.rope_table_bytes()
            + report.attention_workspace_bytes()
            + report.decode_gemm_workspace_bytes()
    );
    assert_eq!(
        report.total_device_bytes(),
        report.forward().total_device_bytes() + report.additional_device_bytes()
    );
}

fn assert_forward_report_matches_context(
    report: PreparedLlamaAllocationReport,
    stats: CudaAllocationStats,
) {
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

fn decode_config(reference: bool) -> PreparedLlamaDecodeConfig {
    let config = PreparedLlamaDecodeConfig::new(
        PreparedLlamaForwardConfig::default().with_optimized_attention(),
    );
    if reference {
        config.with_reference_decode_attention()
    } else {
        config.with_optimized_decode_attention()
    }
}

fn percentile_nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    assert!(!sorted.is_empty());
    let rank = percentile
        .checked_mul(sorted.len())
        .expect("percentile rank fits usize")
        .div_ceil(100);
    sorted[rank - 1]
}

fn run_cached_chain(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    prompt: &[u32],
    decode_calls: usize,
    reference_decode: bool,
    teacher_tokens: Option<&[u32]>,
) -> TestResult<CachedChainOutcome> {
    assert!(context.allocation_stats()?.is_zero());
    if let Some(tokens) = teacher_tokens {
        assert_eq!(tokens.len(), decode_calls);
    }
    let maximum_length = prompt
        .len()
        .checked_add(decode_calls)
        .ok_or("decode capacity overflow")?;
    let mut decode = PreparedLlamaDecode::prepare(
        model,
        context,
        stream,
        prompt.len(),
        maximum_length,
        decode_config(reference_decode),
    )?;
    let trace = decode.prepared_attention().selection_trace();
    let expected_backend = if reference_decode {
        assert_eq!(
            decode.prepared_attention().backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        REFERENCE_DECODE_IMPLEMENTATION
    } else {
        assert_eq!(
            decode.prepared_attention().backend(),
            DecodeAttentionBackend::ChunkedOnline
        );
        OPTIMIZED_DECODE_IMPLEMENTATION
    };
    assert_eq!(trace.implementation_id(), expected_backend);
    let report = decode.allocation_report();
    assert_decode_report_matches_context(report, context.allocation_stats()?);
    assert_eq!(decode.prompt_length(), prompt.len());
    assert_eq!(decode.maximum_length(), maximum_length);
    assert_eq!(decode.logical_length(), 0);
    assert_eq!(decode.phase(), LlamaDecodePhase::Empty);

    println!(
        "pr09-llama-decode-metadata schema_version=1 implementation_id={} \
implementation_version={} native_dependency={} compiled_architectures={} \
device_ordinal={} compute_capability={}.{} prompt_length={} maximum_length={} \
decode_calls={} workspace_bytes={} materialized_score_bytes={} partial_state_bytes={} \
partial_state_capacity={} tokens_per_partition={} kv_cache_bytes={} rope_table_bytes={} \
decode_gemm_workspace_bytes={} total_device_bytes={} device_allocation_count={} \
timing_boundary=decode_plus_stream_synchronize sampling=greedy_test_harness",
        trace.implementation_id(),
        trace.implementation_version(),
        trace.native_dependency(),
        trace.compiled_architectures(),
        trace.device_ordinal(),
        trace.compute_capability().0,
        trace.compute_capability().1,
        prompt.len(),
        maximum_length,
        decode_calls,
        trace.workspace_bytes(),
        trace.materialized_score_bytes(),
        trace.partial_state_bytes(),
        trace.partial_state_capacity(),
        trace.tokens_per_partition(),
        report.kv_cache_bytes(),
        report.rope_table_bytes(),
        report.decode_gemm_workspace_bytes(),
        report.total_device_bytes(),
        report.device_allocation_count(),
    );

    let prefill_started = Instant::now();
    decode.prefill(prompt, stream)?;
    stream.synchronize()?;
    let prefill_latency_ns = u64::try_from(prefill_started.elapsed().as_nanos())?;
    let stable_allocations = context.allocation_stats()?;
    assert_decode_report_matches_context(report, stable_allocations);
    assert_eq!(decode.logical_length(), prompt.len());
    assert_eq!(decode.phase(), LlamaDecodePhase::Prefilled);
    println!(
        "pr09-llama-prefill-raw schema_version=1 implementation_id={} prompt_length={} \
latency_ns={} timing_boundary=prefill_cache_write_plus_stream_synchronize",
        trace.implementation_id(),
        prompt.len(),
        prefill_latency_ns,
    );

    let row_bytes = row_bytes(model)?;
    let row_count = decode_calls
        .checked_add(1)
        .ok_or("cached logit row count overflow")?;
    let mut logits = vec![
        0_u8;
        row_count
            .checked_mul(row_bytes)
            .ok_or("logit byte overflow")?
    ];
    decode.download_last_logits(&mut logits[..row_bytes], stream)?;
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let mut consumed_tokens = Vec::with_capacity(decode_calls);
    let mut latencies_ns = Vec::with_capacity(decode_calls);
    for decode_call in 0..decode_calls {
        let current_start = decode_call * row_bytes;
        let current_end = current_start + row_bytes;
        let token_id = teacher_tokens.map_or_else(
            || u32::try_from(top1(&logits[current_start..current_end])),
            |tokens| Ok(tokens[decode_call]),
        )?;
        consumed_tokens.push(token_id);
        let logical_before = decode.logical_length();
        let started = Instant::now();
        decode.decode(token_id, stream)?;
        stream.synchronize()?;
        let latency_ns = u64::try_from(started.elapsed().as_nanos())?;
        latencies_ns.push(latency_ns);
        let next_start = current_end;
        let next_end = next_start + row_bytes;
        decode.download_last_logits(&mut logits[next_start..next_end], stream)?;
        assert_eq!(decode.logical_length(), logical_before + 1);
        assert_eq!(decode.phase(), LlamaDecodePhase::Decoding);
        assert_eq!(
            context.allocation_stats()?,
            stable_allocations,
            "decode call {} changed CUDA allocation accounting",
            decode_call + 1
        );
        println!(
            "pr09-llama-decode-raw schema_version=1 implementation_id={} decode_call={} \
logical_before={} logical_after={} token_id={} latency_ns={} \
timing_boundary=decode_plus_stream_synchronize logits_download_excluded=true",
            trace.implementation_id(),
            decode_call + 1,
            logical_before,
            decode.logical_length(),
            token_id,
            latency_ns,
        );
    }
    assert_eq!(decode.logical_length(), maximum_length);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    if !latencies_ns.is_empty() {
        let mut sorted = latencies_ns;
        sorted.sort_unstable();
        println!(
            "pr09-llama-decode-summary schema_version=1 implementation_id={} samples={} \
median_ns={} p95_ns={} first_logical_length={} final_logical_length={} \
timing_boundary=decode_plus_stream_synchronize logits_download_excluded=true",
            trace.implementation_id(),
            sorted.len(),
            percentile_nearest_rank(&sorted, 50),
            percentile_nearest_rank(&sorted, 95),
            prompt.len(),
            maximum_length,
        );
    }

    let implementation_id = trace.implementation_id();
    decode.close()?;
    assert!(
        context.allocation_stats()?.is_zero(),
        "explicit decode close must release weights, cache, workspaces, and staging"
    );
    Ok(CachedChainOutcome {
        consumed_tokens,
        logits,
        row_bytes,
        implementation_id,
    })
}

fn run_cache_free_forward(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    tokens: &[u32],
) -> TestResult<Vec<u8>> {
    assert!(context.allocation_stats()?.is_zero());
    let mut forward = PreparedLlamaForward::prepare(
        model,
        context,
        stream,
        tokens.len(),
        PreparedLlamaForwardConfig::default().with_optimized_attention(),
    )?;
    let report = forward.allocation_report();
    assert_forward_report_matches_context(report, context.allocation_stats()?);
    forward.forward(tokens, stream)?;
    let row_bytes = row_bytes(model)?;
    let mut logits = vec![
        0_u8;
        tokens
            .len()
            .checked_mul(row_bytes)
            .ok_or("logit overflow")?
    ];
    forward.download_logits(&mut logits, stream)?;
    forward.close()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok(logits)
}

fn assert_efficient_cache_parity(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    decode_calls: usize,
) -> TestResult {
    let cached = run_cached_chain(
        model,
        context,
        stream,
        &PINNED_TOKENS_A,
        decode_calls,
        false,
        None,
    )?;
    let same_shape_cache_free = run_cache_free_forward(model, context, stream, &PINNED_TOKENS_A)?;
    assert_eq!(
        same_shape_cache_free.len(),
        cached.row_bytes * PINNED_TOKENS_A.len()
    );
    let same_shape_last = (PINNED_TOKENS_A.len() - 1) * cached.row_bytes;
    assert_eq!(
        &cached.logits[..cached.row_bytes],
        &same_shape_cache_free[same_shape_last..same_shape_last + cached.row_bytes],
        "writing the prefill K/V cache must not alter same-shape prefill logits"
    );
    println!(
        "pr09-llama-prefill-parity schema_version=1 implementation_id={} prompt_length={} \
same_shape_cache_free=true byte_exact=true cached_top1={} cache_free_top1={}",
        cached.implementation_id,
        PINNED_TOKENS_A.len(),
        top1(&cached.logits[..cached.row_bytes]),
        top1(&same_shape_cache_free[same_shape_last..same_shape_last + cached.row_bytes]),
    );

    let mut full_tokens = Vec::with_capacity(PINNED_TOKENS_A.len() + decode_calls);
    full_tokens.extend_from_slice(&PINNED_TOKENS_A);
    full_tokens.extend_from_slice(&cached.consumed_tokens);
    let cache_free = run_cache_free_forward(model, context, stream, &full_tokens)?;
    assert_eq!(cached.row_bytes, row_bytes(model)?);

    for causal_row in 0..=decode_calls {
        let cached_start = causal_row * cached.row_bytes;
        let cached_end = cached_start + cached.row_bytes;
        let full_row = PINNED_TOKENS_A.len() - 1 + causal_row;
        let full_start = full_row * cached.row_bytes;
        let full_end = full_start + cached.row_bytes;
        let cached_logits = &cached.logits[cached_start..cached_end];
        let cache_free_logits = &cache_free[full_start..full_end];
        let metrics = assert_semantic_logits(
            &format!("cache-on/cache-off causal row {causal_row}"),
            cached_logits,
            cache_free_logits,
        );
        println!(
            "pr09-llama-cache-parity schema_version=1 implementation_id={} decode_calls_target={} \
causal_row={} full_forward_row={} cosine={:.12} max_abs={:.9} mean_abs={:.9} \
cached_top1={} cache_free_top1={} semantic_gate=top1-exact \
numeric_metrics=diagnostic-only reason=different-gemm-shape-reduction oracle_recertification=false",
            cached.implementation_id,
            decode_calls,
            causal_row,
            full_row,
            metrics.cosine,
            metrics.max_abs,
            metrics.mean_abs,
            top1(cached_logits),
            top1(cache_free_logits),
        );
        if causal_row < decode_calls {
            assert_eq!(
                top1(cache_free_logits),
                usize::try_from(cached.consumed_tokens[causal_row])?,
                "the cache-free causal row must reproduce cached greedy token {}",
                causal_row + 1
            );
        }
    }

    for milestone in [1, 2, 32, 128] {
        if milestone <= decode_calls {
            println!(
                "pr09-llama-decode-milestone schema_version=1 implementation_id={} \
decode_calls={} parity_rows={} greedy_tokens={} status=passed",
                cached.implementation_id,
                milestone,
                milestone + 1,
                milestone,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn pinned_smollm2_reference_and_optimized_decode_preserve_logits_and_top1() -> TestResult {
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let (context, mut stream) = first_context()?;

    let reference = run_cached_chain(
        &model,
        &context,
        &mut stream,
        &PINNED_TOKENS_A,
        DEFAULT_PARITY_DECODE_CALLS,
        true,
        None,
    )?;
    let optimized = run_cached_chain(
        &model,
        &context,
        &mut stream,
        &PINNED_TOKENS_A,
        DEFAULT_PARITY_DECODE_CALLS,
        false,
        Some(&reference.consumed_tokens),
    )?;
    assert_eq!(reference.row_bytes, optimized.row_bytes);
    assert_eq!(reference.consumed_tokens, optimized.consumed_tokens);
    for row in 0..=DEFAULT_PARITY_DECODE_CALLS {
        let start = row * reference.row_bytes;
        let end = start + reference.row_bytes;
        let metrics = assert_semantic_logits(
            &format!("optimized/reference decode row {row}"),
            &optimized.logits[start..end],
            &reference.logits[start..end],
        );
        println!(
            "pr09-llama-backend-parity schema_version=1 reference_implementation={} \
optimized_implementation={} decode_call={} cosine={:.12} max_abs={:.9} mean_abs={:.9} \
reference_top1={} optimized_top1={}",
            reference.implementation_id,
            optimized.implementation_id,
            row,
            metrics.cosine,
            metrics.max_abs,
            metrics.mean_abs,
            top1(&reference.logits[start..end]),
            top1(&optimized.logits[start..end]),
        );
    }

    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn pinned_smollm2_cache_prefill_and_1_2_32_decode_calls_match_one_full_forward() -> TestResult {
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let (context, mut stream) = first_context()?;
    assert_efficient_cache_parity(&model, &context, &mut stream, DEFAULT_PARITY_DECODE_CALLS)?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote long-running 128-call SmolLM2 cache parity on server-4096"]
fn pinned_smollm2_128_decode_calls_match_one_full_forward_when_enabled() -> TestResult {
    if !env_enabled("RUSTINFER_PR09_LONG_STEPS") {
        eprintln!("pr09-llama-long-parity-skipped env=RUSTINFER_PR09_LONG_STEPS expected=true");
        return Ok(());
    }
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let (context, mut stream) = first_context()?;
    assert_efficient_cache_parity(&model, &context, &mut stream, LONG_PARITY_DECODE_CALLS)?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn pinned_smollm2_capacity_reset_and_prompt_reuse_have_no_contamination() -> TestResult {
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let (context, mut stream) = first_context()?;
    let maximum_length = PINNED_TOKENS_A.len() + 2;
    let mut decode = PreparedLlamaDecode::prepare(
        &model,
        &context,
        &mut stream,
        PINNED_TOKENS_A.len(),
        maximum_length,
        decode_config(false),
    )?;
    let report = decode.allocation_report();
    let stable_allocations = context.allocation_stats()?;
    assert_decode_report_matches_context(report, stable_allocations);
    let row_bytes = row_bytes(&model)?;

    let mut a_prefill = vec![0_u8; row_bytes];
    let mut a_decode = vec![0_u8; row_bytes];
    let mut latest_at_capacity = vec![0_u8; row_bytes];
    decode.prefill(&PINNED_TOKENS_A, &mut stream)?;
    decode.download_last_logits(&mut a_prefill, &mut stream)?;
    let a_first_token = u32::try_from(top1(&a_prefill))?;
    decode.decode(a_first_token, &mut stream)?;
    decode.download_last_logits(&mut a_decode, &mut stream)?;
    let a_second_token = u32::try_from(top1(&a_decode))?;
    decode.decode(a_second_token, &mut stream)?;
    decode.download_last_logits(&mut latest_at_capacity, &mut stream)?;
    assert_eq!(decode.logical_length(), maximum_length);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let error = decode
        .decode(u32::try_from(top1(&latest_at_capacity))?, &mut stream)
        .expect_err("the first token beyond fixed capacity must fail before mutation");
    assert!(matches!(
        error,
        LlamaDecodeError::CapacityExceeded {
            logical_length,
            maximum_length: actual_maximum,
        } if logical_length == maximum_length && actual_maximum == maximum_length
    ));
    let mut after_capacity_error = vec![0_u8; row_bytes];
    decode.download_last_logits(&mut after_capacity_error, &mut stream)?;
    assert_eq!(after_capacity_error, latest_at_capacity);
    assert_eq!(decode.logical_length(), maximum_length);
    assert_eq!(context.allocation_stats()?, stable_allocations);

    decode.reset()?;
    assert_eq!(decode.logical_length(), 0);
    assert_eq!(decode.phase(), LlamaDecodePhase::Empty);
    assert!(matches!(
        decode.download_last_logits(&mut after_capacity_error, &mut stream),
        Err(LlamaDecodeError::OutputNotReady)
    ));
    assert_eq!(context.allocation_stats()?, stable_allocations);

    let mut b_prefill_reused = vec![0_u8; row_bytes];
    let mut b_decode_reused = vec![0_u8; row_bytes];
    decode.prefill(&PINNED_TOKENS_B, &mut stream)?;
    decode.download_last_logits(&mut b_prefill_reused, &mut stream)?;
    assert_ne!(b_prefill_reused, a_prefill);
    let b_token = u32::try_from(top1(&b_prefill_reused))?;
    decode.decode(b_token, &mut stream)?;
    decode.download_last_logits(&mut b_decode_reused, &mut stream)?;
    assert_eq!(context.allocation_stats()?, stable_allocations);

    decode.reset()?;
    decode.prefill(&PINNED_TOKENS_A, &mut stream)?;
    let mut a_prefill_reused = vec![0_u8; row_bytes];
    let mut a_decode_reused = vec![0_u8; row_bytes];
    decode.download_last_logits(&mut a_prefill_reused, &mut stream)?;
    decode.decode(a_first_token, &mut stream)?;
    decode.download_last_logits(&mut a_decode_reused, &mut stream)?;
    assert_eq!(a_prefill_reused, a_prefill);
    assert_eq!(a_decode_reused, a_decode);
    assert_eq!(context.allocation_stats()?, stable_allocations);
    decode.close()?;
    assert!(context.allocation_stats()?.is_zero());

    let mut fresh_b = PreparedLlamaDecode::prepare(
        &model,
        &context,
        &mut stream,
        PINNED_TOKENS_B.len(),
        maximum_length,
        decode_config(false),
    )?;
    assert_decode_report_matches_context(fresh_b.allocation_report(), context.allocation_stats()?);
    fresh_b.prefill(&PINNED_TOKENS_B, &mut stream)?;
    let mut b_prefill_fresh = vec![0_u8; row_bytes];
    let mut b_decode_fresh = vec![0_u8; row_bytes];
    fresh_b.download_last_logits(&mut b_prefill_fresh, &mut stream)?;
    fresh_b.decode(b_token, &mut stream)?;
    fresh_b.download_last_logits(&mut b_decode_fresh, &mut stream)?;
    assert_eq!(b_prefill_fresh, b_prefill_reused);
    assert_eq!(b_decode_fresh, b_decode_reused);
    fresh_b.close()?;
    assert!(context.allocation_stats()?.is_zero());

    let dropped_owner = PreparedLlamaDecode::prepare(
        &model,
        &context,
        &mut stream,
        PINNED_TOKENS_A.len(),
        maximum_length,
        decode_config(false),
    )?;
    assert_decode_report_matches_context(
        dropped_owner.allocation_report(),
        context.allocation_stats()?,
    );
    drop(dropped_owner);
    assert!(
        context.allocation_stats()?.is_zero(),
        "implicit request Drop must return CUDA allocation accounting to zero"
    );

    println!(
        "pr09-llama-lifecycle schema_version=1 capacity={maximum_length} capacity_error_pre_mutation=true \
reset_allocation_stable=true same_prompt_replay_byte_exact=true different_prompt_fresh_byte_exact=true \
implicit_drop_accounting_zero=true"
    );
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote near-model-limit cache boundary on server-4096"]
fn pinned_smollm2_near_limit_reaches_capacity_and_fails_next_call_when_enabled() -> TestResult {
    if !env_enabled("RUSTINFER_PR09_NEAR_LIMIT") {
        eprintln!("pr09-llama-near-limit-skipped env=RUSTINFER_PR09_NEAR_LIMIT expected=true");
        return Ok(());
    }
    let checkpoint = checkpoint_path();
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let maximum_length = model.spec().max_sequence_length();
    let prompt_length = maximum_length
        .checked_sub(LONG_PARITY_DECODE_CALLS)
        .ok_or("model context is shorter than the PR09 near-limit decode suffix")?;
    let prompt: Vec<u32> = (0..prompt_length)
        .map(|index| PINNED_TOKENS_A[index % PINNED_TOKENS_A.len()])
        .collect();
    let (context, mut stream) = first_context()?;
    let mut decode = PreparedLlamaDecode::prepare(
        &model,
        &context,
        &mut stream,
        prompt_length,
        maximum_length,
        decode_config(false),
    )?;
    let report = decode.allocation_report();
    assert_decode_report_matches_context(report, context.allocation_stats()?);
    decode.prefill(&prompt, &mut stream)?;
    let stable_allocations = context.allocation_stats()?;
    for decode_call in 0..LONG_PARITY_DECODE_CALLS {
        let token_id = PINNED_TOKENS_A[decode_call % PINNED_TOKENS_A.len()];
        let logical_before = decode.logical_length();
        let started = Instant::now();
        decode.decode(token_id, &mut stream)?;
        stream.synchronize()?;
        let latency_ns = u64::try_from(started.elapsed().as_nanos())?;
        assert_eq!(decode.logical_length(), logical_before + 1);
        assert_eq!(context.allocation_stats()?, stable_allocations);
        println!(
            "pr09-llama-near-limit-raw schema_version=1 decode_call={} logical_before={} \
logical_after={} token_id={} latency_ns={} timing_boundary=decode_plus_stream_synchronize",
            decode_call + 1,
            logical_before,
            decode.logical_length(),
            token_id,
            latency_ns,
        );
    }
    assert_eq!(decode.logical_length(), maximum_length);
    let row_bytes = row_bytes(&model)?;
    let mut latest = vec![0_u8; row_bytes];
    decode.download_last_logits(&mut latest, &mut stream)?;
    let error = decode
        .decode(PINNED_TOKENS_A[0], &mut stream)
        .expect_err("decode call 129 beyond model capacity must fail safely");
    assert!(matches!(error, LlamaDecodeError::CapacityExceeded { .. }));
    let mut after = vec![0_u8; row_bytes];
    decode.download_last_logits(&mut after, &mut stream)?;
    assert_eq!(after, latest);
    assert_eq!(decode.logical_length(), maximum_length);
    assert_eq!(context.allocation_stats()?, stable_allocations);
    decode.close()?;
    assert!(context.allocation_stats()?.is_zero());
    println!(
        "pr09-llama-near-limit-summary schema_version=1 prompt_length={prompt_length} \
decode_calls={LONG_PARITY_DECODE_CALLS} final_logical_length={maximum_length} \
capacity_error_pre_mutation=true allocation_stable=true"
    );
    stream.close()?;
    close_context(context)
}
