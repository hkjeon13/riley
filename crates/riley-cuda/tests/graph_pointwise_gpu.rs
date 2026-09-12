#![cfg(feature = "cuda")]
use riley_cuda::{
    BorrowedPointwiseGraph, BorrowedPointwiseResources, CudaBufferSpan, CudaBufferSpanMut,
    CudaDType, CudaRuntime, GatedMultiplyParams, PointwiseGraphOperation as Op, ResidualAddParams,
    SiluParams, gated_multiply, residual_add, silu,
};
#[test]
#[ignore = "requires CUDA; borrowed pointwise parity, tails and lifecycle"]
fn borrowed_pointwise_matches_eager_and_preserves_owners() -> Result<(), Box<dyn std::error::Error>>
{
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(16384)?;
    for elements in [1_u64, 257, 2048] {
        let a: Vec<u8> = (0..elements + 8)
            .flat_map(|i| (0x3e00_u16 + (i % 250) as u16).to_le_bytes())
            .collect();
        let b: Vec<u8> = (0..elements + 8)
            .flat_map(|i| (0xbf00_u16 + (i % 127) as u16).to_le_bytes())
            .collect();
        let mut input = context.allocate_device_buffer(a.len() as u64)?;
        let mut other = context.allocate_device_buffer(b.len() as u64)?;
        let mut output = context.allocate_device_buffer(a.len() as u64)?;
        input.upload_from_slice(0, &a, &mut staging, &mut stream)?;
        other.upload_from_slice(0, &b, &mut staging, &mut stream)?;
        for op in [Op::Silu, Op::GatedMultiply, Op::ResidualAdd] {
            output.upload_from_slice(0, &vec![0xa5; a.len()], &mut staging, &mut stream)?;
            let left = CudaBufferSpan::new(&input, CudaDType::BF16, 0, elements * 2)?;
            let right = CudaBufferSpan::new(&other, CudaDType::BF16, 0, elements * 2)?;
            let out = CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, elements * 2)?;
            match op {
                Op::Silu => silu(
                    &mut SiluParams {
                        input: left,
                        output: out,
                        element_count: elements,
                    },
                    &mut stream,
                )?,
                Op::GatedMultiply => gated_multiply(
                    &mut GatedMultiplyParams {
                        activated_gate: left,
                        up: right,
                        output: out,
                        element_count: elements,
                    },
                    &mut stream,
                )?,
                Op::ResidualAdd => residual_add(
                    &mut ResidualAddParams {
                        left,
                        right,
                        output: out,
                        element_count: elements,
                    },
                    &mut stream,
                )?,
            }
            let mut expected = vec![0; a.len()];
            output.download_to_slice(0, &mut expected, &mut staging, &mut stream)?;
            let stats = context.allocation_stats()?;
            for n in [0, u64::MAX, elements + 9] {
                assert!(
                    BorrowedPointwiseGraph::prepare(
                        BorrowedPointwiseResources {
                            stream: &mut stream,
                            input: &mut input,
                            other: if op == Op::Silu {
                                None
                            } else {
                                Some(&mut other)
                            },
                            output: &mut output
                        },
                        op,
                        n
                    )
                    .is_err()
                );
                assert_eq!(stats, context.allocation_stats()?);
            }
            assert!(
                BorrowedPointwiseGraph::prepare(
                    BorrowedPointwiseResources {
                        stream: &mut stream,
                        input: &mut input,
                        other: if op == Op::Silu {
                            Some(&mut other)
                        } else {
                            None
                        },
                        output: &mut output
                    },
                    op,
                    elements
                )
                .is_err()
            );
            for explicit in [true, false] {
                let mut graph = BorrowedPointwiseGraph::prepare(
                    BorrowedPointwiseResources {
                        stream: &mut stream,
                        input: &mut input,
                        other: if op == Op::Silu {
                            None
                        } else {
                            Some(&mut other)
                        },
                        output: &mut output,
                    },
                    op,
                    elements,
                )?;
                for _ in 0..32 {
                    graph.replay()?;
                }
                if explicit {
                    graph.close()?;
                } else {
                    drop(graph);
                }
                let mut actual = vec![0; a.len()];
                output.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                assert_eq!(actual, expected);
                assert_eq!(&actual[elements as usize * 2..], &[0xa5; 16]);
                input.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                assert_eq!(actual, a);
                other.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                assert_eq!(actual, b);
                assert_eq!(stats, context.allocation_stats()?);
            }
        }
        input.close()?;
        other.close()?;
        output.close()?;
    }
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
