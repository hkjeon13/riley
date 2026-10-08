#![cfg(feature = "cuda")]
use riley_cuda::{
    AttentionParentLayer, CudaRuntime, PackedAttentionMetadataLayout, PackedBatchHostV1,
    PackedParentKvWriteGraph, PackedParentKvWriteResources,
};

#[test]
#[ignore = "requires CUDA; packed write parent/layer isolation"]
fn packed_kv_write_preserves_other_layers_metadata_and_sources()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(32768)?;
    let batch = PackedBatchHostV1::new(&[0, 2], &[1, 0], &[16, 2], &[0, 0], &[1, 17], 2)?;
    let mut payload = vec![0xa5; 64];
    for (offset, words) in [
        (0, &[0_u32, 0][..]),
        (8, &[0, 2][..]),
        (24, &[1, 0][..]),
        (48, &[1, 17][..]),
    ] {
        for (i, w) in words.iter().enumerate() {
            payload[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    payload[40..44].copy_from_slice(&[16, 0, 2, 0]);
    let metadata =
        PackedAttentionMetadataLayout::new(64, [(8, 8), (24, 8), (40, 4), (0, 8), (48, 8)])?;
    let k: Vec<u8> = (0..264)
        .flat_map(|i| (0x3e00_u16 + (i % 127) as u16).to_le_bytes())
        .collect();
    let v: Vec<u8> = (0..264)
        .flat_map(|i| (0x3f00_u16 + (i % 127) as u16).to_le_bytes())
        .collect();
    let geometry = AttentionParentLayer::new(3, 0, 2, 2)?;
    let mut key = context.allocate_device_buffer(k.len() as u64)?;
    let mut value = context.allocate_device_buffer(v.len() as u64)?;
    let mut kp = context.allocate_device_buffer(geometry.parent_byte_len())?;
    let mut vp = context.allocate_device_buffer(geometry.parent_byte_len())?;
    let mut slab = context.allocate_device_buffer(64)?;
    key.upload_from_slice(0, &k, &mut staging, &mut stream)?;
    value.upload_from_slice(0, &v, &mut staging, &mut stream)?;
    slab.upload_from_slice(0, &payload, &mut staging, &mut stream)?;
    let stats = context.allocation_stats()?;
    for layer in 0..3 {
        let geometry = AttentionParentLayer::new(3, layer, 2, 2)?;
        let original = vec![0x5a; geometry.parent_byte_len() as usize];
        kp.upload_from_slice(0, &original, &mut staging, &mut stream)?;
        vp.upload_from_slice(0, &original, &mut staging, &mut stream)?;
        for explicit in [true, false] {
            let mut graph = PackedParentKvWriteGraph::prepare(
                PackedParentKvWriteResources {
                    stream: &mut stream,
                    key_source: &mut key,
                    value_source: &mut value,
                    key_parent: &mut kp,
                    value_parent: &mut vp,
                    metadata_slab: &mut slab,
                },
                geometry,
                metadata,
                batch,
                &mut staging,
            )?;
            for _ in 0..32 {
                graph.replay()?;
            }
            if explicit {
                graph.close()?;
            } else {
                drop(graph);
            }
            for (pool, source) in [(&mut kp, &k), (&mut vp, &v)] {
                let mut expected = original.clone();
                for (row, block) in [(0, 1), (1, 0)] {
                    for head in 0..2 {
                        let dst = geometry.byte_offset() as usize
                            + (block * 2 * 16 * 64 + head * 16 * 64 + 64) * 2;
                        let src = (row * 2 + head) * 64 * 2;
                        expected[dst..dst + 128].copy_from_slice(&source[src..src + 128]);
                    }
                }
                let mut actual = vec![0; expected.len()];
                pool.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                assert_eq!(
                    actual, expected,
                    "layer {layer} write range differs from CPU scatter"
                );
            }
            for (buffer, expected) in [(&mut key, &k), (&mut value, &v), (&mut slab, &payload)] {
                let mut actual = vec![0; expected.len()];
                buffer.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                assert_eq!(&actual, expected);
            }
            assert_eq!(context.allocation_stats()?, stats);
        }
    }
    let stale = PackedBatchHostV1::new(&[0, 2], &[1, 0], &[16, 2], &[0, 0], &[2, 17], 2)?;
    assert!(
        PackedParentKvWriteGraph::prepare(
            PackedParentKvWriteResources {
                stream: &mut stream,
                key_source: &mut key,
                value_source: &mut value,
                key_parent: &mut kp,
                value_parent: &mut vp,
                metadata_slab: &mut slab
            },
            geometry,
            metadata,
            stale,
            &mut staging
        )
        .is_err()
    );
    let mut recovery = PackedParentKvWriteGraph::prepare(
        PackedParentKvWriteResources {
            stream: &mut stream,
            key_source: &mut key,
            value_source: &mut value,
            key_parent: &mut kp,
            value_parent: &mut vp,
            metadata_slab: &mut slab,
        },
        geometry,
        metadata,
        batch,
        &mut staging,
    )?;
    recovery.replay()?;
    recovery.close()?;
    key.close()?;
    value.close()?;
    kp.close()?;
    vp.close()?;
    slab.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
