//! Correctness-only Attention and residual normalization correctness evidence.
#![cfg(feature = "cuda")]

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaRuntime, RopeParams, RopeTableParams, rope,
    rope_table,
};
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{LlamaTracePoint, PreparedLlamaForward, PreparedLlamaForwardConfig};
use std::{collections::BTreeMap, error::Error, fs, path::PathBuf};

#[test]
#[ignore = "requires CUDA, checkpoint, explicit prefix cases and output directory"]
fn export_first_layer_attention_trace() -> Result<(), Box<dyn Error>> {
    let checkpoint = PathBuf::from(std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint")?);
    let cases = PathBuf::from(std::env::var_os("RILEY_TRACE_CASES").ok_or("cases")?);
    let output = PathBuf::from(std::env::var_os("RILEY_TRACE_OUTPUT").ok_or("output")?);
    let cases: BTreeMap<String, Vec<u32>> = serde_json::from_slice(&fs::read(cases)?)?;
    fs::create_dir_all(&output)?;
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let points = [
        LlamaTracePoint::Embedding,
        LlamaTracePoint::Layer0QueryProjection,
        LlamaTracePoint::Layer0KeyProjection,
        LlamaTracePoint::Layer0ValueProjection,
        LlamaTracePoint::Layer0AttentionProbabilities,
        LlamaTracePoint::Layer0AttentionContext,
        LlamaTracePoint::Layer0AfterAttentionResidual,
        LlamaTracePoint::Layer0PostAttentionNorm,
    ];
    for (name, tokens) in cases {
        assert!(!name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'));
        let mut forward = PreparedLlamaForward::prepare(
            &model,
            &context,
            &mut stream,
            tokens.len(),
            PreparedLlamaForwardConfig::default().with_reference_attention(),
        )?;
        let mut trace = forward.prepare_trace_points(&points)?;
        forward.upload_tokens(&tokens, &mut stream)?;
        forward.execute_traced(&mut stream, &mut trace)?;
        assert_eq!(trace.captured_count(), u32::try_from(points.len())?);
        for point in points {
            fs::write(
                output.join(format!("{name}-{}.bf16", point.name())),
                trace.tensor(point).ok_or("missing trace tensor")?,
            )?;
        }
        forward.close()?;
        // Same reviewed SmolLM2 angle construction as build_rope_angles.
        let rows = tokens.len() as u64;
        let angles: Vec<u8> = (0..rows)
            .flat_map(|position| {
                (0..32).flat_map(move |pair| {
                    let inverse_frequency = 1.0_f32 / 100_000_f32.powf((2 * pair) as f32 / 64.0);
                    (position as f32 * inverse_frequency).to_le_bytes()
                })
            })
            .collect();
        let table_bytes = angles.len() as u64;
        let mut staging = context.allocate_pinned_host_buffer(rows * 576 * 2)?;
        let mut cos = context.allocate_device_buffer(table_bytes)?;
        let mut sin = context.allocate_device_buffer(table_bytes)?;
        cos.upload_from_slice(0, &angles, &mut staging, &mut stream)?;
        rope_table(
            &mut RopeTableParams {
                angles_cos: CudaBufferSpanMut::new(&mut cos, CudaDType::F32, 0, table_bytes)?,
                sin: CudaBufferSpanMut::new(&mut sin, CudaDType::F32, 0, table_bytes)?,
                element_count: rows * 32,
            },
            &mut stream,
        )?;
        for (label, heads) in [("q", 9_u64), ("k", 3_u64)] {
            let raw = fs::read(output.join(format!("{name}-layer0.{label}_proj.bf16")))?;
            let bytes = raw.len() as u64;
            let mut input = context.allocate_device_buffer(bytes)?;
            let mut rotated = context.allocate_device_buffer(bytes)?;
            input.upload_from_slice(0, &raw, &mut staging, &mut stream)?;
            rope(
                &mut RopeParams {
                    input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, bytes)?,
                    cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, table_bytes)?,
                    sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, table_bytes)?,
                    output: CudaBufferSpanMut::new(&mut rotated, CudaDType::BF16, 0, bytes)?,
                    token_count: rows,
                    head_count: heads,
                    head_size: 64,
                    rotary_dimension: 64,
                    table_position_count: rows,
                    position_offset: 0,
                },
                &mut stream,
            )?;
            let mut result = vec![0; raw.len()];
            rotated.download_to_slice(0, &mut result, &mut staging, &mut stream)?;
            fs::write(
                output.join(format!("{name}-layer0.{label}_rope.bf16")),
                result,
            )?;
            input.close()?;
            rotated.close()?;
        }
        cos.close()?;
        sin.close()?;
        staging.close()?;
        println!(
            "ATTENTION_TRACE case={name} tokens={} captured_attention_points=8 performance_trials=0",
            tokens.len()
        );
    }
    stream.close()?;
    let stats = context.allocation_stats()?;
    assert_eq!(stats.device_live_allocations(), 0);
    assert_eq!(stats.pinned_host_live_allocations(), 0);
    context.close()?;
    Ok(())
}
