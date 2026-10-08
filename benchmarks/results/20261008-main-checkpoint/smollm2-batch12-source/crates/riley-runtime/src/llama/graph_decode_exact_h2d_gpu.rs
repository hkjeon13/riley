//! Remote CUDA parity regression for the closed C07 exact V1 H2D chain.
//!
//! This test intentionally lives outside the production C07 boundaries.  It
//! drives the existing host-slab -> pinned-slab -> device-slab -> command-batch
//! path and then reads the opaque device bytes back only after the completed
//! fresh lease has ended.  It neither captures a CUDA Graph nor resolves a C06
//! replay slot.

use std::error::Error;

use riley_cuda::{CudaContext, CudaRuntime, CudaStream};

use super::batch::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchMetadata,
};
use super::graph_decode_exact_device_slab::PureDecodeGraphV1ExactDeviceSlab;
use super::graph_decode_exact_h2d_completion::{
    finish_pure_decode_graph_v1_exact_h2d, submit_pure_decode_graph_v1_exact_h2d,
};
use super::graph_decode_exact_host_slab::{
    PureDecodeGraphV1ExactHostSlab, PureDecodeGraphV1ExactHostSlabWrite,
};
use super::graph_decode_exact_pinned_host_slab::PureDecodeGraphV1ExactPinnedHostSlab;
use super::graph_decode_layout::{
    PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
};
use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((context, stream))
}

/// Runs the strict M=1/B=1 C07 H2D chain on a real GPU and reads it back.
///
/// The test is intentionally ignored on developer/CI machines without a CUDA
/// device.  It validates transport/lifetime parity only; graph capture,
/// registry selection, executor mutation, and performance claims remain out
/// of scope.
#[test]
#[ignore = "requires a remote CUDA GPU"]
fn c07_23_exact_h2d_is_byte_exact_and_releases_every_cuda_allocation() -> TestResult {
    let (context, mut stream) = first_context()?;
    let layout =
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(1, 1, 3, 5))?;
    let mut prepared =
        PreparedLlamaBatchMetadata::prepare(LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)?)?;
    let token_ids = [0x0102_0304_u32];
    let physical_block_ids = [0_u32];
    let valid_tokens = [1_u16];
    let rows = [LlamaBatchRow::new(
        0xfeed_beef,
        LlamaBatchRowKind::Decode,
        &token_ids,
        1,
        LlamaBatchBlockTable::new(
            BLOCK_TABLE_V1_VERSION,
            &physical_block_ids,
            &valid_tokens,
            1,
        ),
        Some(0),
    )];
    let metadata = prepared.pack(&rows)?;
    let mut host_slab = PureDecodeGraphV1ExactHostSlab::prepare(layout)?;
    let mut pinned_slab = PureDecodeGraphV1ExactPinnedHostSlab::prepare(&context, layout)?;
    let mut device_slab = PureDecodeGraphV1ExactDeviceSlab::prepare(&context, layout)?;
    let stable_allocations = context.allocation_stats()?;
    let mut expected = None;
    for iteration in 0_u8..32 {
        let header = [0xa0_u8, 0xa1, iteration];
        let control_status = [0xc0_u8, 0xc1, 0xc2, 0xc3, iteration];
        let expected_bytes = {
            let host_lease = match host_slab
                .write_exact_v1_leased(&metadata, &header, &control_status)
                .map_err(|error| {
                    std::io::Error::other(format!("C07 exact host write failed: {error:?}"))
                })? {
                PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
                PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                    return Err(format!("strict M=1/B=1 fixture was ineligible: {reason:?}").into());
                }
            };
            let expected = host_lease.bytes().to_vec();
            let pinned_lease = pinned_slab.stage_from_host_lease(host_lease)?;
            let binding = device_slab.bind_pinned_host_lease(pinned_lease)?;
            let batch = stream.begin_command_batch()?;
            let submitted = submit_pure_decode_graph_v1_exact_h2d(&binding, batch)?;
            let fresh = finish_pure_decode_graph_v1_exact_h2d(submitted)?;
            assert_eq!(fresh.layout(), layout);
            assert_eq!(fresh.geometry_digest(), layout.geometry_digest());
            assert_eq!(fresh.device_buffer().byte_len(), layout.total_bytes());
            drop(fresh);
            expected
        };
        assert_eq!(
            context.allocation_stats()?,
            stable_allocations,
            "C07 exact H2D changed cold CUDA allocation accounting on iteration {iteration}"
        );
        expected = Some(expected_bytes);
    }
    let expected = expected.expect("C07 exact H2D fixture must execute at least once");

    let mut readback_staging = context.allocate_pinned_host_buffer(layout.total_bytes())?;
    device_slab
        .device_buffer_for_gpu_test()
        .copy_to_pinned_async(
            0,
            &mut readback_staging,
            0,
            layout.total_bytes(),
            &mut stream,
        )?
        .synchronize()?;
    assert_eq!(readback_staging.to_vec()?, expected);
    readback_staging.close()?;
    device_slab.close()?;
    pinned_slab.close()?;
    drop(host_slab);
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
