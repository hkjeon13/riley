//! Peer all-reduce on two or more GPUs of one host.
//!
//! `RILEY_PEER_DEVICES` lists the CUDA ordinals in rank order (default
//! `0,1`). Each rank runs on its own host thread and stream, as tensor
//! parallelism will.

use std::error::Error;
use std::thread;

use riley_cuda::{
    CudaBufferSpanMut, CudaContext, CudaDType, CudaPeerAllReduceGroup, CudaRuntime,
    peer_all_reduce_bf16_in_place,
};

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

const HIDDEN: u64 = 5_120;
// Row counts cover the vector path (whole rows of 5120) and a scalar tail.
const ROW_COUNTS: [u64; 7] = [1, 2, 3, 8, 32, 37, 256];
const REPEATS: usize = 20;

fn devices() -> Vec<u32> {
    std::env::var("RILEY_PEER_DEVICES")
        .unwrap_or_else(|_| "0,1".to_owned())
        .split(',')
        .map(|ordinal| ordinal.trim().parse().expect("RILEY_PEER_DEVICES ordinal"))
        .collect()
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    u16::try_from((bits + rounding) >> 16).expect("bf16 bits")
}

// A value that depends on (rank, row, column) only, never on the row count,
// with exponents spread over a few binades and both signs.
fn input(rank: u32, row: u64, column: u64) -> u16 {
    let mut state = (u64::from(rank) << 48) ^ (row << 24) ^ column ^ 0x9e37_79b9_7f4a_7c15;
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    let mantissa = f32::from(u16::try_from(state & 0xffff).expect("16 bits")) / 65_536.0;
    let scale = [0.25_f32, 1.0, 4.0, 16.0][usize::try_from((state >> 16) & 3).expect("index")];
    let sign = if state & (1 << 20) == 0 { 1.0 } else { -1.0 };
    f32_to_bf16(sign * mantissa * scale)
}

fn expected(ranks: u32, row: u64, column: u64) -> u16 {
    let mut sum = 0.0_f32;
    for rank in 0..ranks {
        sum += bf16_to_f32(input(rank, row, column));
    }
    f32_to_bf16(sum)
}

fn run_rank(
    group: &CudaPeerAllReduceGroup,
    context: &CudaContext,
    rank: u32,
    columns: u64,
) -> TestResult<Vec<Vec<u16>>> {
    let mut stream = context.create_stream()?;
    let max_bytes = ROW_COUNTS.iter().max().copied().unwrap_or(1) * columns * 2;
    let mut staging = context.allocate_pinned_host_buffer(max_bytes)?;
    let mut buffer = context.allocate_device_buffer(max_bytes)?;
    let mut outputs = Vec::new();
    for _ in 0..REPEATS {
        for rows in ROW_COUNTS {
            let elements = rows * columns;
            let host: Vec<u8> = (0..rows)
                .flat_map(|row| (0..columns).map(move |column| input(rank, row, column)))
                .flat_map(u16::to_le_bytes)
                .collect();
            buffer.upload_from_slice(0, &host, &mut staging, &mut stream)?;
            {
                let mut span = CudaBufferSpanMut::new(&mut buffer, CudaDType::BF16, 0, elements * 2)?;
                peer_all_reduce_bf16_in_place(group, rank, &mut span, elements, &mut stream)?;
            }
            let mut result = vec![0_u8; usize::try_from(elements * 2)?];
            buffer.download_to_slice(0, &mut result, &mut staging, &mut stream)?;
            outputs.push(
                result
                    .chunks_exact(2)
                    .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                    .collect(),
            );
        }
    }
    Ok(outputs)
}

fn check(columns: u64) -> TestResult {
    let ordinals = devices();
    let runtime = CudaRuntime::initialize()?;
    let contexts = ordinals
        .iter()
        .map(|ordinal| runtime.device(*ordinal)?.create_context())
        .collect::<Result<Vec<_>, _>>()?;
    let references: Vec<&CudaContext> = contexts.iter().collect();
    let max_elements = ROW_COUNTS.iter().max().copied().unwrap_or(1) * columns;
    let group = CudaPeerAllReduceGroup::new(&references, max_elements)?;
    let ranks = group.rank_count();

    let results = thread::scope(|scope| {
        let handles: Vec<_> = contexts
            .iter()
            .enumerate()
            .map(|(rank, context)| {
                let group = &group;
                scope.spawn(move || {
                    run_rank(group, context, u32::try_from(rank).expect("rank"), columns)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("rank thread"))
            .collect::<Result<Vec<_>, _>>()
    })?;

    for (call, rank0) in results[0].iter().enumerate() {
        let rows = ROW_COUNTS[call % ROW_COUNTS.len()];
        for (rank, outputs) in results.iter().enumerate() {
            assert_eq!(&outputs[call], rank0, "rank {rank} differs from rank 0 at call {call}");
        }
        for row in 0..rows {
            for column in 0..columns {
                let index = usize::try_from(row * columns + column)?;
                assert_eq!(
                    rank0[index],
                    expected(ranks, row, column),
                    "rows {rows} row {row} column {column}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn rank_ordered_sum_is_identical_on_every_rank_and_independent_of_rows() -> TestResult {
    check(HIDDEN)
}

#[test]
fn scalar_path_handles_widths_that_are_not_vector_multiples() -> TestResult {
    check(7)
}

#[test]
fn group_rejects_a_repeated_device() -> TestResult {
    let ordinals = devices();
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(ordinals[0])?.create_context()?;
    assert!(CudaPeerAllReduceGroup::new(&[&context, &context], 16).is_err());
    Ok(())
}
