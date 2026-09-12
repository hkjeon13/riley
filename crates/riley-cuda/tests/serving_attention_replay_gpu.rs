//! Replays actual vLLM backend inputs through Riley paged attention.
#![cfg(feature = "cuda")]
use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaRuntime, PackedBatchHostV1, PackedBatchV1,
    RaggedPagedAttentionParams, grouped_ragged_paged_attention,
};
use std::{error::Error, fs, path::PathBuf};
#[test]
#[ignore = "requires CUDA and captured serving tensors"]
fn replay_actual_serving_attention() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(std::env::var_os("RILEY_SERVING_TRACE").ok_or("trace root")?);
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1024 * 1024)?;
    for layer in 0..30 {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for call in 0_usize..9 {
            keys.extend(fs::read(
                root.join(format!("layer{layer}-call{call}-k.bf16")),
            )?);
            values.extend(fs::read(
                root.join(format!("layer{layer}-call{call}-v.bf16")),
            )?);
            let rows = if call == 0 { 128_usize } else { 1 };
            let len = 128 + call;
            let blocks = len.div_ceil(16);
            assert_eq!(keys.len(), len * 192 * 2);
            let mut kp = vec![0_u8; blocks * 3 * 16 * 64 * 2];
            let mut vp = kp.clone();
            for token in 0..len {
                for head in 0..3 {
                    let src = (token * 3 + head) * 64 * 2;
                    let dst = (((token / 16) * 3 + head) * 16 + token % 16) * 64 * 2;
                    kp[dst..dst + 128].copy_from_slice(&keys[src..src + 128]);
                    vp[dst..dst + 128].copy_from_slice(&values[src..src + 128]);
                }
            }
            let offsets = [0_u32, blocks as u32];
            let ids: Vec<u32> = (0..blocks as u32).collect();
            let mut valid = vec![16_u16; blocks];
            valid[blocks - 1] = ((len - 1) % 16 + 1) as u16;
            let slots = vec![0_u32; rows];
            let positions: Vec<u32> = if call == 0 {
                (0..128).collect()
            } else {
                vec![(len - 1) as u32]
            };
            let rawq = fs::read(root.join(format!("layer{layer}-call{call}-q.bf16")))?;
            assert_eq!(rawq.len(), rows * 576 * 2);
            let host =
                PackedBatchHostV1::new(&offsets, &ids, &valid, &slots, &positions, blocks as u64)?;
            let payloads = [
                rawq,
                kp,
                vp,
                offsets.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
                valid.iter().flat_map(|v| v.to_le_bytes()).collect(),
                slots.iter().flat_map(|v| v.to_le_bytes()).collect(),
                positions.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ];
            let mut bufs = Vec::new();
            for bytes in &payloads {
                let mut b = context.allocate_device_buffer(bytes.len() as u64)?;
                b.upload_from_slice(0, bytes, &mut staging, &mut stream)?;
                bufs.push(b);
            }
            let span =
                |i: usize, dtype| CudaBufferSpan::new(&bufs[i], dtype, 0, bufs[i].byte_len());
            let batch = PackedBatchV1::new(
                host,
                span(3, CudaDType::U32)?,
                span(4, CudaDType::U32)?,
                span(5, CudaDType::U16)?,
                span(6, CudaDType::U32)?,
                span(7, CudaDType::U32)?,
            )?;
            let bytes = (rows * 576 * 2) as u64;
            let mut output = context.allocate_device_buffer(bytes)?;
            grouped_ragged_paged_attention(
                &mut RaggedPagedAttentionParams {
                    query: span(0, CudaDType::BF16)?,
                    key_pool: span(1, CudaDType::BF16)?,
                    value_pool: span(2, CudaDType::BF16)?,
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, bytes)?,
                    batch,
                    query_head_count: 9,
                    key_value_head_count: 3,
                    head_size: 64,
                    output_row_count: rows as u64,
                    scale: 0.125,
                },
                &mut stream,
            )?;
            let mut raw = vec![0; bytes as usize];
            output.download_to_slice(0, &mut raw, &mut staging, &mut stream)?;
            fs::write(
                root.join(format!("layer{layer}-call{call}-riley-context.bf16")),
                raw,
            )?;
            output.close()?;
            for b in bufs {
                b.close()?;
            }
        }
    }
    staging.close()?;
    stream.close()?;
    let stats = context.allocation_stats()?;
    assert_eq!(stats.device_live_allocations(), 0);
    assert_eq!(stats.pinned_host_live_allocations(), 0);
    context.close()?;
    println!("SERVING_REPLAY cases=270 zero_allocations=true performance_trials=0");
    Ok(())
}
