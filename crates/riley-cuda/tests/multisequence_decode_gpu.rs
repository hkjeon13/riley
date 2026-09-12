#![cfg(feature = "cuda")]
use riley_cuda::{
    BorrowedGraphResourceParents, BorrowedGraphResourceReservation, CudaRuntime,
    CudaStridedGemmConfig,
};
fn put32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
fn packet(bucket: u32, active: u32, full: bool, transfer: usize, replay: u64) -> Vec<u8> {
    let mut p = vec![0; transfer];
    for (at, value) in [
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
        (104, u32::from(full)),
    ] {
        put32(&mut p, at, value);
    }
    put64(&mut p, 48, 1);
    put64(&mut p, 56, replay);
    put64(&mut p, 64, replay);
    p[72..104].fill(0x5a);
    for row in 0..active as usize {
        let base = 128 + row * 128;
        let pos = 128 + row as u32;
        put32(&mut p, base, 17 + row as u32);
        put32(&mut p, base + 4, pos);
        put32(&mut p, base + 12, 9);
        for i in 0..9 {
            put32(&mut p, base + 16 + 4 * i, (row * 10 + i) as u32);
            let valid = if i < 8 { 16u16 } else { pos as u16 % 16 + 1 };
            p[base + 56 + 2 * i..base + 58 + 2 * i].copy_from_slice(&valid.to_le_bytes());
        }
        put64(&mut p, base + 80, 100 + row as u64);
        put64(&mut p, base + 88, 200 + row as u64);
        for (at, value) in [
            (96, pos + 1),
            (100, active - 1 - row as u32),
            (104, 1),
            (108, pos - 127),
            (112, 1),
            (116, 1),
        ] {
            put32(&mut p, base + at, value);
        }
    }
    p
}
#[test]
#[ignore = "requires qualified RTX 4090; zero-weight full-DAG lifecycle smoke, not model parity"]
fn full_multisequence_decode_capture_replay_and_packet_rejection()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut upload = context.allocate_pinned_host_buffer(1 << 20)?;
    for bucket in [2u32, 4] {
        for full in [false, true] {
            let result = 640 + if full { bucket as usize * 98304 } else { 0 };
            let transfer = result.max(1280);
            let b = u64::from(bucket);
            let sizes = [
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
                384 * 16 * 40 * 30,
                384 * 16 * 40 * 30,
                161 * 128,
                161 * 128,
                49152 * 1152,
                1152,
                960 * 1152,
                576 * 1152,
                3072 * 1152,
                576 * 3072,
            ];
            let mut buffers = Vec::new();
            for bytes in sizes {
                let mut buffer = context.allocate_device_buffer(bytes)?;
                buffer.upload_from_slice(0, &vec![0; bytes as usize], &mut upload, &mut stream)?;
                buffers.push(buffer);
            }
            let mut plans = Vec::new();
            for (n, k) in [
                (960, 576),
                (3072, 576),
                (576, 576),
                (576, 1536),
                (49152, 576),
            ] {
                plans.push(
                    context
                        .prepare_strided_gemm(CudaStridedGemmConfig::new(n, k, bucket, false)?)?,
                );
            }
            let mut staging = context.allocate_pinned_host_buffer((transfer * 2) as u64)?;
            let mut weights = vec![19usize; 273];
            weights[0] = 18;
            weights[2] = 18;
            for l in 0..30 {
                weights[7 + 9 * l] = 21;
                weights[11 + 9 * l] = 23;
                weights.extend([20, 22]);
            }
            let mut owner = BorrowedGraphResourceReservation::reserve_with_strided(
                BorrowedGraphResourceParents {
                    stream: &mut stream,
                    devices: buffers.iter_mut().collect(),
                    pinned: vec![&mut staging],
                    plans: vec![],
                },
                plans.iter_mut().collect(),
            )?;
            owner.record_multisequence_decode(
                &std::array::from_fn(|i| i),
                &weights,
                &[0, 1, 2, 3, 4],
                0,
                bucket,
                40,
                full,
            )?;
            for (iteration, active) in [bucket, if bucket == 4 { 3 } else { 2 }, bucket]
                .into_iter()
                .enumerate()
            {
                let p = packet(bucket, active, full, transfer, iteration as u64 + 1);
                let mut bad = p.clone();
                bad[128 + 128 + 16..128 + 128 + 20].copy_from_slice(&0u32.to_le_bytes());
                assert!(owner.replay_transfer(&bad).is_err());
                let mut out = vec![0xff; transfer];
                assert!(owner.read_transfer(&mut out).is_err());
                owner.replay_transfer(&p)?;
                owner.read_transfer(&mut out)?;
                assert_eq!(&out[..4], &0x314f4d52u32.to_le_bytes());
                assert_eq!(&out[8..12], &(result as u32).to_le_bytes());
                assert_eq!(&out[24..28], &active.to_le_bytes());
                assert!(out[108..112].iter().all(|b| *b == 0));
                for row in 0..active as usize {
                    let base = 128 + row * 128;
                    assert_eq!(&out[base..base + 8], &(100 + row as u64).to_le_bytes());
                    assert_eq!(
                        &out[base + 36..base + 40],
                        &(active - 1 - row as u32).to_le_bytes()
                    );
                    assert!(out[base + 48..base + 56].iter().all(|b| *b == 0));
                    assert_eq!(&out[base + 56..base + 60], &32u32.to_le_bytes());
                    assert_eq!(&out[base + 88..base + 92], &1u32.to_le_bytes());
                }
                assert!(
                    out[128 + active as usize * 128..640]
                        .iter()
                        .all(|b| *b == 0)
                );
                assert!(out[640..].iter().all(|b| *b == 0));
            }
            owner.close()?;
            for plan in plans {
                plan.close()?;
            }
        }
    }
    eprintln!(
        "MULTI_FULL_DAG zero_weight_cases=4 replays=12 malformed_packets_rejected=12 model_parity=false"
    );
    Ok(())
}
