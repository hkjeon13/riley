//! Real-weight integration parity against the retained M1 profile, not timing.
use super::*;
use crate::llama::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_cuda::{CudaContext, CudaStridedGemmConfig};
use riley_model::{LoadLimits, LoadedModel};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
fn prepare(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
) -> Result<OwnedLlamaDecodeExecutor> {
    let config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 128, 10, 1, 40)?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_grouped_ragged_attention_heads()
    .with_separate_residual_norm()
    .with_iteration_batch_completion()
    .with_packed_async_metadata()
    .with_vllm_smol_p128_graph();
    let mut executor = PreparedLlamaBatchExecutor::prepare(model, context, stream, config)?;
    let bytes = executor.owner.layout.bytes_per_kind();
    let zeros = vec![0; bytes as usize];
    let mut io = context.allocate_pinned_host_buffer(1 << 20)?;
    executor
        .owner
        .key_cache
        .upload_from_slice(0, &zeros, &mut io, stream)?;
    executor
        .owner
        .value_cache
        .upload_from_slice(0, &zeros, &mut io, stream)?;
    Ok(executor.into_owned_decode_graph(context)?)
}
fn run(
    owner: &mut OwnedLlamaDecodeExecutor,
    row: usize,
    tokens: &[u32],
    length: usize,
) -> Result<Vec<u8>> {
    let mapping: Vec<u32> = (0..10).map(|i| ((row * 10 + i) * 7 % 40) as u32).collect();
    let live = length.div_ceil(16);
    let mut valid = vec![16u16; live];
    valid[live - 1] = ((length - 1) % 16 + 1) as u16;
    let row = LlamaBatchRow::new(
        row as u64 + 1,
        if tokens.len() == 128 {
            LlamaBatchRowKind::Prefill
        } else {
            LlamaBatchRowKind::Decode
        },
        tokens,
        length as u32,
        LlamaBatchBlockTable::new(
            crate::paged_kv::BLOCK_TABLE_V1_VERSION,
            &mapping[..live],
            &valid,
            length as u32,
        ),
        Some(0),
    );
    Ok(owner.execute(&[row])?.to_vec())
}
fn put32(p: &mut [u8], at: usize, v: u32) {
    p[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(p: &mut [u8], at: usize, v: u64) {
    p[at..at + 8].copy_from_slice(&v.to_le_bytes());
}
fn packet(bucket: u32, active: u32, positions: &[usize], tokens: &[u32], replay: u64) -> Vec<u8> {
    let mut p = vec![0; 640 + bucket as usize * 98304];
    for (at, v) in [
        (0, 0x31444d52),
        (4, 0x00800001),
        (8, 1280),
        (12, 128),
        (16, 1),
        (20, bucket),
        (24, active),
        (28, active),
        (32, 4),
        (36, 10),
        (40, 49152),
        (44, 40),
        (104, 1),
    ] {
        put32(&mut p, at, v);
    }
    put64(&mut p, 48, 1);
    put64(&mut p, 56, replay);
    put64(&mut p, 64, replay);
    p[72..104].fill(0x5a);
    for r in 0..active as usize {
        let b = 128 + r * 128;
        let pos = positions[r];
        let live = pos / 16 + 1;
        for (at, v) in [
            (0, tokens[r]),
            (4, pos as u32),
            (12, live as u32),
            (96, pos as u32 + 1),
            (100, active - 1 - r as u32),
            (104, 1),
            (108, pos as u32 - 127),
            (112, 1),
            (116, 1),
        ] {
            put32(&mut p, b + at, v);
        }
        for i in 0..live {
            put32(&mut p, b + 16 + 4 * i, ((r * 10 + i) * 7 % 40) as u32);
            let v = if i + 1 < live {
                16u16
            } else {
                (pos % 16 + 1) as u16
            };
            p[b + 56 + 2 * i..b + 58 + 2 * i].copy_from_slice(&v.to_le_bytes());
        }
        put64(&mut p, b + 80, r as u64 + 1);
        put64(&mut p, b + 88, 100 + r as u64);
    }
    p
}
#[test]
#[ignore = "requires SmolLM2 checkpoint and qualified CUDA runtime"]
fn multisequence_full_model_logits_and_kv_match_m1() -> Result {
    let path = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
    let model = LoadedModel::load(std::path::Path::new(&path), LoadLimits::default())?;
    let context = riley_cuda::CudaRuntime::initialize()?
        .device(0)?
        .create_context()?;
    let mut stream = context.create_stream()?;
    let mut io = context.allocate_pinned_host_buffer(1 << 20)?;
    let mut total_rows = 0;
    for bucket in [2u32, 4] {
        let mut oracle = prepare(&model, &context, &mut stream)?;
        let mut candidate = prepare(&model, &context, &mut stream)?;
        let mut tokens = vec![0; bucket as usize];
        let mut positions = vec![128usize; bucket as usize];
        for r in 0..bucket as usize {
            let prompt: Vec<u32> = (0..128)
                .map(|i| ((i * 337 + r * 719 + 11) % 49152) as u32)
                .collect();
            let expected = run(&mut oracle, r, &prompt, 128)?;
            assert_eq!(run(&mut candidate, r, &prompt, 128)?, expected);
            tokens[r] = oracle.greedy_token()?;
        }
        let mut parents = candidate.graph.close()?;
        let b = u64::from(bucket);
        let result = 640 + bucket as usize * 98304;
        let mut scratch = Vec::new();
        for size in [
            1152 * b,
            1152 * b,
            1920 * b,
            1152 * b,
            1152 * b,
            1152 * b,
            2304 * b,
            6144 * b,
            3072 * b,
            1152 * b,
            98304 * b,
            1280,
            result as u64,
            8 * b,
        ] {
            scratch.push(context.allocate_device_buffer(size)?);
        }
        let mut staging = context.allocate_pinned_host_buffer((2 * result) as u64)?;
        let mut plans = Vec::new();
        for (n, k) in [
            (960, 576),
            (3072, 576),
            (576, 576),
            (576, 1536),
            (49152, 576),
        ] {
            plans.push(
                context.prepare_strided_gemm(CudaStridedGemmConfig::new(n, k, bucket, false)?)?,
            );
        }
        let owner = &mut parents.executor.owner;
        let f = &mut owner.forward;
        let mut weights = vec![
            f.plan.embedding_weight().index(),
            f.plan.final_norm_weight().index(),
            f.plan.lm_head_weight().index(),
        ];
        for l in f.plan.layers() {
            weights.extend(
                [
                    l.input_norm_weight(),
                    l.query_weight(),
                    l.key_weight(),
                    l.value_weight(),
                    l.output_weight(),
                    l.post_attention_norm_weight(),
                    l.gate_weight(),
                    l.up_weight(),
                    l.down_weight(),
                ]
                .map(|w| w.index()),
            );
        }
        let mut devices: Vec<_> = f.weights.borrow_graph_weight_parents().collect();
        let packed_start = devices.len();
        devices.extend(
            parents
                .packed
                .as_mut()
                .ok_or("packed missing")?
                .buffers
                .iter_mut()
                .take(60),
        );
        weights.extend(packed_start..packed_start + 60);
        let start = devices.len();
        devices.extend(scratch.iter_mut());
        devices.extend([
            &mut owner.key_cache,
            &mut owner.value_cache,
            &mut owner.absolute_rope_cos,
            &mut owner.absolute_rope_sin,
        ]);
        let mut graph = riley_cuda::BorrowedGraphResourceReservation::reserve_with_strided(
            riley_cuda::BorrowedGraphResourceParents {
                stream: &mut parents.stream,
                devices,
                pinned: vec![&mut staging],
                plans: vec![],
            },
            plans.iter_mut().collect(),
        )?;
        graph.record_multisequence_decode(
            &std::array::from_fn(|i| start + i),
            &weights,
            &[0, 1, 2, 3, 4],
            0,
            bucket,
            40,
            true,
        )?;
        for step in 0..31 {
            let active = if bucket == 4 && step % 3 == 1 {
                3
            } else {
                bucket
            };
            let p = packet(bucket, active, &positions, &tokens, step + 1);
            let mut expected = Vec::new();
            for r in 0..active as usize {
                expected.push(run(&mut oracle, r, &[tokens[r]], positions[r] + 1)?);
                tokens[r] = oracle.greedy_token()?;
            }
            graph.replay_transfer(&p)?;
            let mut actual = vec![0; result];
            graph.read_transfer(&mut actual)?;
            for r in 0..active as usize {
                let logits = &actual[640 + r * 98304..640 + (r + 1) * 98304];
                let mismatches = logits
                    .chunks_exact(2)
                    .zip(expected[r][..98304].chunks_exact(2))
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "bucket={bucket} step={step} row={r} position={}",
                    positions[r]
                );
                assert_eq!(
                    &actual[128 + r * 128 + 48..128 + r * 128 + 52],
                    &tokens[r].to_le_bytes()
                );
                positions[r] += 1;
                total_rows += 1;
            }
            assert!(
                actual[128 + active as usize * 128..640]
                    .iter()
                    .all(|b| *b == 0)
            );
            assert!(
                actual[640 + active as usize * 98304..]
                    .iter()
                    .all(|b| *b == 0)
            );
        }
        graph.close()?;
        let mut oracle_parents = oracle.graph.close()?;
        let bytes = parents.executor.owner.layout.bytes_per_kind() as usize;
        for (actual, expected) in [
            (
                &mut parents.executor.owner.key_cache,
                &mut oracle_parents.executor.owner.key_cache,
            ),
            (
                &mut parents.executor.owner.value_cache,
                &mut oracle_parents.executor.owner.value_cache,
            ),
        ] {
            let mut a = vec![0; bytes];
            let mut e = vec![0; bytes];
            actual.download_to_slice(0, &mut a, &mut io, &mut stream)?;
            expected.download_to_slice(0, &mut e, &mut io, &mut stream)?;
            assert_eq!(
                a.iter().zip(&e).filter(|(x, y)| x != y).count(),
                0,
                "full physical KV pool differs, bucket={bucket}"
            );
        }
    }
    drop(io);
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    println!(
        "MULTI_MODEL full_logits_exact=true full_physical_kv_exact=true buckets=2,4 replays=62 rows={total_rows} zero_allocations=true"
    );
    Ok(())
}
