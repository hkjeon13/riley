//! Correctness-only MLP tail evidence with controlled operation inputs.
#![cfg(feature = "cuda")]

use riley_cuda::CudaRuntime;
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{LlamaTracePoint, PreparedLlamaForward, PreparedLlamaForwardConfig};
use std::{collections::BTreeMap, error::Error, fs, path::PathBuf};

#[test]
#[ignore = "requires CUDA, checkpoint, explicit prefix cases and output directory"]
fn export_first_layer_tail_trace() -> Result<(), Box<dyn Error>> {
    let checkpoint = PathBuf::from(std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint")?);
    let cases = PathBuf::from(std::env::var_os("RILEY_TRACE_CASES").ok_or("cases")?);
    let output = PathBuf::from(std::env::var_os("RILEY_TRACE_OUTPUT").ok_or("output")?);
    let cases: BTreeMap<String, Vec<u32>> = serde_json::from_slice(&fs::read(cases)?)?;
    fs::create_dir_all(&output)?;
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let points = [
        LlamaTracePoint::Layer0AfterAttentionResidual,
        LlamaTracePoint::Layer0PostAttentionNorm,
        LlamaTracePoint::Layer0GateProjection,
        LlamaTracePoint::Layer0UpProjection,
        LlamaTracePoint::Layer0Gated,
        LlamaTracePoint::Layer0DownProjection,
        LlamaTracePoint::Layer0Output,
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
        println!(
            "TAIL_TRACE case={name} tokens={} captured_tail_points=7 performance_trials=0",
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
