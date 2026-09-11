#![cfg(feature = "cuda")]
use riley_cuda::{
    Bf16EmbeddingStatusD2HStatus as Status, BorrowedEmbeddingGraph, BorrowedEmbeddingResources,
    CudaRuntime, EmbeddingGraphGeometry,
};
#[test]
#[ignore = "requires CUDA; actual borrowed embedding status and preservation"]
fn borrowed_embedding_reports_invalid_tokens_without_output_writes()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    let mut report = context.allocate_pinned_host_buffer(32)?;
    let table_bytes: Vec<u8> = (0..7 * 64 + 8)
        .flat_map(|i| (0x3e00_u16 + (i % 511) as u16).to_le_bytes())
        .collect();
    let mut table = context.allocate_device_buffer(table_bytes.len() as u64)?;
    let mut tokens = context.allocate_device_buffer(64)?;
    let mut output = context.allocate_device_buffer(400)?;
    let mut scratch = context.allocate_device_buffer(32)?;
    table.upload_from_slice(0, &table_bytes, &mut staging, &mut stream)?;
    let geometry = EmbeddingGraphGeometry {
        tokens: 3,
        vocabulary: 7,
        hidden: 64,
    };
    let stats = context.allocation_stats()?;
    for (ids, expected_status, explicit) in [
        ([1_u32, 2, 6], Status::Success, true),
        (
            [1, 9, 10],
            Status::TokenOutOfRange {
                token_position: 1,
                token_id: 9,
            },
            false,
        ),
        ([0, 3, 5], Status::Success, false),
    ] {
        let mut payload = vec![0xa5; 64];
        for (i, id) in ids.iter().enumerate() {
            payload[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
        }
        tokens.upload_from_slice(0, &payload, &mut staging, &mut stream)?;
        output.upload_from_slice(0, &[0x5a; 400], &mut staging, &mut stream)?;
        let mut short_report = context.allocate_pinned_host_buffer(31)?;
        assert!(
            BorrowedEmbeddingGraph::prepare(
                BorrowedEmbeddingResources {
                    stream: &mut stream,
                    table: &mut table,
                    token_ids: &mut tokens,
                    output: &mut output,
                    error_scratch: &mut scratch,
                    report: &mut short_report
                },
                geometry
            )
            .is_err()
        );
        short_report.close()?;
        let mut graph = BorrowedEmbeddingGraph::prepare(
            BorrowedEmbeddingResources {
                stream: &mut stream,
                table: &mut table,
                token_ids: &mut tokens,
                output: &mut output,
                error_scratch: &mut scratch,
                report: &mut report,
            },
            geometry,
        )?;
        for _ in 0..32 {
            assert_eq!(graph.replay()?, expected_status);
        }
        if explicit {
            graph.close()?;
        } else {
            drop(graph);
        }
        let mut actual = vec![0; 400];
        output.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
        let mut expected = vec![0x5a; 400];
        if expected_status == Status::Success {
            for (i, id) in ids.iter().enumerate() {
                expected[i * 128..i * 128 + 128]
                    .copy_from_slice(&table_bytes[*id as usize * 128..*id as usize * 128 + 128]);
            }
        }
        assert_eq!(actual, expected);
        let mut after = vec![0; table_bytes.len()];
        table.download_to_slice(0, &mut after, &mut staging, &mut stream)?;
        assert_eq!(after, table_bytes);
        let mut after = vec![0; 64];
        tokens.download_to_slice(0, &mut after, &mut staging, &mut stream)?;
        assert_eq!(after, payload);
        assert_eq!(context.allocation_stats()?, stats);
    }
    table.close()?;
    tokens.close()?;
    output.close()?;
    scratch.close()?;
    report.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
