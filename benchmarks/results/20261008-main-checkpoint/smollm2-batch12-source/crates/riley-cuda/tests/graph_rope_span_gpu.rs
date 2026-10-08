#![cfg(feature = "cuda")]
use riley_cuda::{
    BorrowedRopeGraph, BorrowedRopeResources, CudaBufferSpan, CudaBufferSpanMut, CudaDType,
    CudaRuntime, IndexedRopeParams, RopeGraphGeometry, indexed_rope,
};

#[test]
#[ignore = "requires CUDA; packed position parent span and lifecycle"]
fn borrowed_rope_position_span_matches_eager_and_preserves_parent()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(32768)?;
    for (rows, heads, offset) in [(1_u64, 3_u64, 0_u64), (1, 9, 64), (3, 3, 128), (3, 9, 64)] {
        let bytes = rows * heads * 64 * 2;
        let input_bytes: Vec<u8> = (0..bytes / 2 + 8)
            .flat_map(|i| (0x3e00_u16 + (i % 127) as u16).to_le_bytes())
            .collect();
        let cos_bytes: Vec<u8> = (0..8 * 32)
            .flat_map(|i| ((i as f32 * 0.03).cos()).to_le_bytes())
            .collect();
        let sin_bytes: Vec<u8> = (0..8 * 32)
            .flat_map(|i| ((i as f32 * 0.03).sin()).to_le_bytes())
            .collect();
        let positions_host: Vec<u32> = (0..rows).map(|r| r as u32 + 2).collect();
        let mut parent_bytes = vec![0xa5; (offset + rows * 4 + 32) as usize];
        for (i, p) in positions_host.iter().enumerate() {
            let start = offset as usize + i * 4;
            parent_bytes[start..start + 4].copy_from_slice(&p.to_le_bytes());
        }
        let mut input = context.allocate_device_buffer(input_bytes.len() as u64)?;
        let mut cos = context.allocate_device_buffer(cos_bytes.len() as u64)?;
        let mut sin = context.allocate_device_buffer(sin_bytes.len() as u64)?;
        let mut positions = context.allocate_device_buffer(parent_bytes.len() as u64)?;
        let mut output = context.allocate_device_buffer(input_bytes.len() as u64)?;
        input.upload_from_slice(0, &input_bytes, &mut staging, &mut stream)?;
        cos.upload_from_slice(0, &cos_bytes, &mut staging, &mut stream)?;
        sin.upload_from_slice(0, &sin_bytes, &mut staging, &mut stream)?;
        positions.upload_from_slice(0, &parent_bytes, &mut staging, &mut stream)?;
        output.upload_from_slice(0, &vec![0x5a; input_bytes.len()], &mut staging, &mut stream)?;
        indexed_rope(
            &mut IndexedRopeParams {
                input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, bytes)?,
                cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
                sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
                positions: CudaBufferSpan::new(&positions, CudaDType::U32, offset, rows * 4)?,
                positions_host: &positions_host,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, bytes)?,
                head_count: heads,
                head_size: 64,
                rotary_dimension: 64,
                table_position_count: 8,
            },
            &mut stream,
        )?;
        let mut expected = vec![0; input_bytes.len()];
        output.download_to_slice(0, &mut expected, &mut staging, &mut stream)?;
        let shape = RopeGraphGeometry {
            rows,
            heads,
            head_size: 64,
            rotary_dimension: 64,
            table_positions: 8,
            positions_byte_offset: offset,
        };
        let stats = context.allocation_stats()?;
        for bad_offset in [1, positions.byte_len(), u64::MAX] {
            assert!(
                BorrowedRopeGraph::prepare(
                    BorrowedRopeResources {
                        stream: &mut stream,
                        input: &mut input,
                        cos: &mut cos,
                        sin: &mut sin,
                        positions: &mut positions,
                        output: &mut output
                    },
                    RopeGraphGeometry {
                        positions_byte_offset: bad_offset,
                        ..shape
                    },
                    &positions_host
                )
                .is_err()
            );
            assert_eq!(context.allocation_stats()?, stats);
        }
        for explicit_close in [true, false] {
            let mut graph = BorrowedRopeGraph::prepare(
                BorrowedRopeResources {
                    stream: &mut stream,
                    input: &mut input,
                    cos: &mut cos,
                    sin: &mut sin,
                    positions: &mut positions,
                    output: &mut output,
                },
                shape,
                &positions_host,
            )?;
            for _ in 0..32 {
                graph.replay()?;
            }
            if explicit_close {
                graph.close()?;
            } else {
                drop(graph);
            }
            let mut actual = vec![0; input_bytes.len()];
            output.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
            assert_eq!(actual, expected);
            assert_eq!(&actual[bytes as usize..], &[0x5a; 16]);
            input.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
            assert_eq!(actual, input_bytes);
            let mut parent = vec![0; parent_bytes.len()];
            positions.download_to_slice(0, &mut parent, &mut staging, &mut stream)?;
            assert_eq!(parent, parent_bytes);
            let mut table = vec![0; cos_bytes.len()];
            cos.download_to_slice(0, &mut table, &mut staging, &mut stream)?;
            assert_eq!(table, cos_bytes);
            sin.download_to_slice(0, &mut table, &mut staging, &mut stream)?;
            assert_eq!(table, sin_bytes);
            assert_eq!(context.allocation_stats()?, stats);
        }
        input.close()?;
        cos.close()?;
        sin.close()?;
        positions.close()?;
        output.close()?;
    }
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
