//! Remote-only regression for the canonical empty-prompt cached decode path.

#![cfg(feature = "cuda")]

use std::error::Error;
use std::path::PathBuf;

use riley_cuda::{CudaContext, CudaRuntime, CudaStream, DecodeAttentionBackend};
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::llama::{PreparedLlamaDecode, PreparedLlamaDecodeConfig};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1 << 30;
const PROMPT: [u32; 1] = [0];
const STEPS: usize = 32;
const HF_CACHE_ON_TOKENS: [u32; STEPS] = [
    198, 198, 504, 216, 33, 41, 40, 32, 99, 436, 253, 655, 282, 1109, 1363, 281, 260, 905, 282,
    2477, 30, 378, 216, 33, 41, 40, 32, 99, 436, 253, 655, 282,
];

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RILEY_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    Ok((context, stream))
}

fn ranked_two(logits: &[u8]) -> [(u32, f32); 2] {
    let mut ranked = [(u32::MAX, f32::NEG_INFINITY); 2];
    for (index, bytes) in logits.chunks_exact(2).enumerate() {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
        let token = u32::try_from(index).expect("vocabulary token fits u32");
        let row = (token, value);
        if value > ranked[0].1 || (value == ranked[0].1 && token < ranked[0].0) {
            ranked[1] = ranked[0];
            ranked[0] = row;
        } else if value > ranked[1].1 || (value == ranked[1].1 && token < ranked[1].0) {
            ranked[1] = row;
        }
    }
    ranked
}

fn trace(
    label: &str,
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    config: PreparedLlamaDecodeConfig,
) -> TestResult<Vec<u32>> {
    let mut decode = PreparedLlamaDecode::prepare(
        model,
        context,
        stream,
        PROMPT.len(),
        PROMPT.len() + STEPS,
        config,
    )?;
    println!(
        "{label} backend={:?}",
        decode.prepared_attention().backend()
    );
    assert_eq!(
        decode.prepared_attention().backend(),
        DecodeAttentionBackend::ChunkedOnline
    );
    decode.prefill(&PROMPT, stream)?;
    let vocabulary = model.spec().embedding().vocabulary_size();
    let mut logits = vec![0_u8; vocabulary * 2];
    let mut generated = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        decode.download_last_logits(&mut logits, stream)?;
        let ranked = ranked_two(&logits);
        generated.push(ranked[0].0);
        if step == 17 {
            assert_eq!(ranked[0], (905, 16.625));
            assert_eq!(ranked[1], (1797, 16.625));
        }
        if step + 1 < STEPS {
            decode.decode(ranked[0].0, stream)?;
        }
    }
    decode.close()?;
    println!("{label} tokens={generated:?}");
    Ok(generated)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn empty_prompt_production_cache_matches_hf_eager_for_thirty_two_steps() -> TestResult {
    let model = LoadedModel::load(
        &checkpoint_path(),
        LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?,
    )?;
    let (context, mut stream) = context()?;

    let production_paged = trace(
        "production-paged",
        &model,
        &context,
        &mut stream,
        PreparedLlamaDecodeConfig::default(),
    )?;
    let production_contiguous = trace(
        "production-contiguous",
        &model,
        &context,
        &mut stream,
        PreparedLlamaDecodeConfig::default().with_contiguous_kv_cache(),
    )?;

    assert_eq!(production_paged, HF_CACHE_ON_TOKENS);
    assert_eq!(production_contiguous, HF_CACHE_ON_TOKENS);
    assert_eq!(production_paged, production_contiguous);
    println!(
        "pr16-empty-cache-on schema_version=1 steps={STEPS} first_divergence_boundary=17 \
hf_eager_exact=true paged_contiguous_exact=true status=passed"
    );

    stream.close()?;
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
