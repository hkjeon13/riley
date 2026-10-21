#![cfg(feature = "cuda-cublas-gemm-probe")]

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaExecutionStream, CudaGemmConfig, CudaPinnedHostBuffer,
    CudaPreparedCublasGemmProbe, CudaRuntime, CudaStream,
};
use std::error::Error;
type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;

fn upload(c: &CudaContext, s: &mut CudaStream, h: &mut CudaPinnedHostBuffer,
          data: &[u8]) -> Result<CudaDeviceBuffer> {
    let mut b = c.allocate_device_buffer(data.len() as u64)?;
    b.upload_from_slice(0, data, h, s)?;
    Ok(b)
}

fn finite_bf16(count: usize, seed: u32) -> Vec<u8> {
    (0..count).flat_map(|i| {
        let bits = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed * 1_013);
        let x = ((bits % 2_047) as f32 - 1_023.0) / 1_024.0;
        let f = x.to_bits();
        (((f + 0x7fff + ((f >> 16) & 1)) >> 16) as u16).to_le_bytes()
    }).collect()
}

#[allow(clippy::too_many_arguments)]
fn attention<S: CudaExecutionStream + ?Sized>(
    plan: &mut CudaPreparedCublasGemmProbe, query: &CudaDeviceBuffer,
    key: &CudaDeviceBuffer, value: &CudaDeviceBuffer, ids: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer, scratch: &mut [CudaDeviceBuffer; 8],
    m: u64, t: u64, physical: u64, query_offset: u64, output_offset: u64,
    stream: &mut S,
) -> Result {
    let qbytes = m * 32 * 128 * 2;
    let pool = physical * 8 * 16 * 128 * 2;
    plan.execute_hf_attention(
        CudaBufferSpan::new(query, CudaDType::BF16, query_offset, qbytes)?,
        CudaBufferSpan::new(key, CudaDType::BF16, 0, pool)?,
        CudaBufferSpan::new(value, CudaDType::BF16, 0, pool)?,
        CudaBufferSpan::new(ids, CudaDType::U32, 0, t.div_ceil(16) * 4)?,
        CudaBufferSpanMut::new(output, CudaDType::BF16, output_offset, qbytes)?,
        scratch, m, t, physical, 32, 8, 128, stream,
    )?;
    Ok(())
}

// Compare the qualified compact route and its unaligned-span fallback with an
// independently prepared original-layout attention plan. Sentinel bytes prove
// that the view, rather than its complete backing allocation, is written.
#[test]
#[ignore = "A100 SM80/CUDA12.4: compact attention and unaligned BF16 views"]
fn compact_prefill_preserves_original_bytes_and_unaligned_view_bounds() -> Result {
    let runtime = CudaRuntime::initialize()?;
    let c = runtime.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut staging = c.allocate_pinned_host_buffer(4 * 1024 * 1024)?;
    let mut checked = 0;
    for (m, t) in [(31_u64, 129_u64), (128, 128), (129, 257), (256, 512)] {
        let physical = t.div_ceil(16) + 3;
        let qbytes = (m * 32 * 128 * 2) as usize;
        let pool = (physical * 8 * 16 * 128 * 2) as usize;
        let score = (32 * m * t * 2) as usize;
        let tile = (32 * m.min(128) * t * 4) as usize;
        let ids: Vec<u8> = (0..t.div_ceil(16))
            .flat_map(|i| ((physical - 1 - i) as u32).to_le_bytes()).collect();
        let table = upload(&c, &mut stream, &mut staging, &ids)?;
        let mut query = upload(&c, &mut stream, &mut staging, &vec![0xa5; qbytes + 16])?;
        let mut output = upload(&c, &mut stream, &mut staging, &vec![0xa5; qbytes + 16])?;
        let mut key = upload(&c, &mut stream, &mut staging, &vec![0; pool])?;
        let mut value = upload(&c, &mut stream, &mut staging, &vec![0; pool])?;
        let lengths = [qbytes, (t * 32 * 128 * 2) as usize,
            (t * 32 * 128 * 2) as usize, score, tile, tile, qbytes, 16 * 1024 * 1024];
        let mut buffers = Vec::new();
        for len in lengths { buffers.push(c.allocate_device_buffer(len as u64)?); }
        let mut scratch: [CudaDeviceBuffer; 8] = buffers.try_into().map_err(|_| "scratch count")?;
        let config = CudaGemmConfig::new(m, t, 128, 16 * 1024 * 1024)?;
        let mut original = c.prepare_owned_hf_gemm(config)?;
        let mut compact = c.prepare_owned_hf_gemm(config)?;
        compact.prepare_sm80_attention()?;
        let stable = c.allocation_stats()?;
        for seed in 1..=3 {
            let q = finite_bf16(qbytes / 2, seed);
            key.upload_from_slice(0, &finite_bf16(pool / 2, seed + 1), &mut staging, &mut stream)?;
            value.upload_from_slice(0, &finite_bf16(pool / 2, seed + 2), &mut staging, &mut stream)?;
            query.upload_from_slice(0, &q, &mut staging, &mut stream)?;
            attention(&mut original, &query, &key, &value, &table, &mut output,
                &mut scratch, m, t, physical, 0, 0, &mut stream)?;
            let mut expected = vec![0; qbytes];
            output.download_to_slice(0, &mut expected, &mut staging, &mut stream)?;
            for (qo, oo) in [(0_u64, 0_u64), (2, 0), (0, 2), (2, 6), (14, 14)] {
                let mut padded_query = vec![0xa5; qbytes + 16];
                padded_query[qo as usize..qo as usize + qbytes].copy_from_slice(&q);
                query.upload_from_slice(0, &padded_query, &mut staging, &mut stream)?;
                for batched in [false, true] {
                    output.upload_from_slice(0, &vec![0xa5; qbytes + 16], &mut staging, &mut stream)?;
                    if batched {
                        let mut batch = stream.begin_command_batch()?;
                        attention(&mut compact, &query, &key, &value, &table, &mut output,
                            &mut scratch, m, t, physical, qo, oo, &mut batch.commands())?;
                        batch.finish()?;
                    } else {
                        attention(&mut compact, &query, &key, &value, &table, &mut output,
                            &mut scratch, m, t, physical, qo, oo, &mut stream)?;
                    }
                    let mut actual = vec![0; qbytes + 16];
                    output.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                    let start = oo as usize;
                    assert_eq!(&actual[start..start + qbytes], expected.as_slice(),
                        "M{m} T{t} seed{seed} query+{qo} output+{oo} batch={batched}");
                    assert!(actual[..start].iter().chain(&actual[start + qbytes..]).all(|&b| b == 0xa5));
                    assert_eq!(c.allocation_stats()?, stable, "no execution CUDA allocation");
                    checked += 1;
                }
            }
        }
        original.close()?; compact.close()?;
        for b in scratch { b.close()?; }
        value.close()?; key.close()?; output.close()?; query.close()?; table.close()?;
    }
    eprintln!("SM80_PREFILL_ALIGNMENT exact_view_cases={checked} sentinel_bounds=true hot_cuda_allocations=0");
    assert_eq!(checked, 120);
    staging.close()?; stream.close()?; c.close()?;
    Ok(())
}
