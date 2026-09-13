use super::*;

#[test]
fn range_import_binds_source_tokens_and_survives_source_and_cache_release() {
    let mut pool=pool(4);let mut source=pool.create_sequence(64).unwrap();
    let reservation=source.reserve_to(&mut pool,32).unwrap();source.commit(&mut pool,reservation).unwrap();
    let expected=descriptor(&pool,32);let mut export=pool.export_prefix(&source,expected).unwrap();
    source.close(&mut pool).unwrap();
    let tokens:Vec<_>=(0..32).collect();let mut wrong=tokens.clone();wrong[31]=0;
    let mut target=pool.create_sequence(16).unwrap();
    assert!(target.import_prefix_range(&mut pool,&export,&wrong,16).is_err());
    assert!(target.import_prefix_range(&mut pool,&export,&tokens,0).is_err());
    assert!(target.import_prefix_range(&mut pool,&export,&tokens,33).is_err());
    assert_eq!(target.logical_length(),0);
    target.import_prefix_range(&mut pool,&export,&tokens,16).unwrap();
    assert_eq!(target.logical_length(),16);
    pool.release_prefix(&mut export).unwrap();
    assert_eq!(pool.stats().allocated_block_count(),1);
    assert_eq!(target.execution_block_table(&pool,None).unwrap().physical_block_ids().len(),1);
    target.close(&mut pool).unwrap();assert_eq!(pool.stats().allocated_block_count(),0);
}

fn pool(blocks: usize) -> KvBlockPool {
    KvBlockPool::new(KvLayout::checked(2, blocks, 2, 8).unwrap()).unwrap()
}

fn descriptor(pool: &KvBlockPool, length: usize) -> PrefixDescriptor {
    PrefixDescriptor::new(
        KvIdentity {
            model_revision: [1; 32],
            numerical_profile: [2; 32],
            position_encoding: [3; 32],
            partition: [4; 32],
            layout: pool.layout.into(),
        },
        0,
        &(0..length as u32).collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn descriptor_rejects_missing_identity_and_invalid_position_ranges() {
    let pool = pool(4);
    let expected = descriptor(&pool, 1);
    assert!(PrefixDescriptor::new(expected.identity.clone(), 0, &[]).is_err());
    assert!(PrefixDescriptor::new(expected.identity.clone(), u32::MAX, &[1]).is_err());
    let mut missing = expected.identity.clone();
    missing.numerical_profile = [0; 32];
    assert!(PrefixDescriptor::new(missing, 0, &[1]).is_err());
    let larger_pool = KvLayout::checked(2, 100, 2, 8).unwrap();
    assert_eq!(
        expected.identity.layout,
        KvPageLayout::from(larger_pool),
        "physical capacity does not change page representation identity"
    );
    let split_a = PrefixDescriptor::new(expected.identity.clone(), 0, &[1, 256]).unwrap();
    let split_b = PrefixDescriptor::new(expected.identity, 0, &[257, 0]).unwrap();
    assert_ne!(split_a, split_b);
}

fn committed(pool: &mut KvBlockPool, length: usize) -> SequenceState {
    let mut sequence = pool.create_sequence(64).unwrap();
    let reservation = sequence.reserve_to(pool, length).unwrap();
    sequence.commit(pool, reservation).unwrap();
    sequence
}

#[test]
fn prefix_identity_binds_every_semantic_field_before_import() {
    let mut pool = pool(8);
    let mut source = committed(&mut pool, 17);
    let expected = descriptor(&pool, 17);
    let mut export = pool.export_prefix(&source, expected.clone()).unwrap();
    let mut target = pool.create_sequence(64).unwrap();
    for field in 0..9 {
        let mut wrong = expected.clone();
        match field {
            0 => wrong.identity.model_revision[0] ^= 1,
            1 => wrong.identity.numerical_profile[0] ^= 1,
            2 => wrong.identity.position_encoding[0] ^= 1,
            3 => wrong.identity.partition[0] ^= 1,
            4 => wrong.identity.layout.dimension += 1,
            5 => wrong.identity.layout.format_version += 1,
            6 => wrong.token_digest[0] ^= 1,
            7 => wrong.start_position += 1,
            _ => wrong.tokens -= 1,
        }
        let before = pool.stats();
        assert!(target.import_prefix(&mut pool, &export, &wrong).is_err());
        assert_eq!(pool.stats(), before);
        assert_eq!(target.logical_length(), 0);
    }
    target.import_prefix(&mut pool, &export, &expected).unwrap();
    assert_eq!(source.block_id(0), target.block_id(0));
    assert_eq!(pool.stats().allocated_block_count(), 2);
    let after = pool.stats();
    assert!(target.import_prefix(&mut pool, &export, &expected).is_err());
    assert_eq!(pool.stats(), after);
    source.close(&mut pool).unwrap();
    pool.release_prefix(&mut export).unwrap();
    target.close(&mut pool).unwrap();
    assert_eq!(pool.stats().allocated_block_count(), 0);
}

#[test]
fn shared_owners_survive_source_reclaim_cache_eviction_and_arbitrary_close_order() {
    for reverse in [false, true] {
        let mut pool = pool(8);
        let source = committed(&mut pool, 17);
        let expected = descriptor(&pool, 17);
        let mut export = pool.export_prefix(&source, expected.clone()).unwrap();
        let mut targets = Vec::new();
        for _ in 0..4 {
            let mut target = pool.create_sequence(64).unwrap();
            target.import_prefix(&mut pool, &export, &expected).unwrap();
            targets.push(target);
        }
        let reclaim = source.abandon_for_reclaim();
        assert_eq!(pool.reclaim_sequence(&reclaim).unwrap(), 2);
        pool.release_prefix(&mut export).unwrap();
        assert_eq!(pool.stats().allocated_block_count(), 2);
        for target in &mut targets {
            for index in 0..2 {
                pool.validate_block(target.block_id(index).unwrap(), target.sequence_id())
                    .unwrap();
            }
            assert!(matches!(
                target.reserve_to(&mut pool, 18),
                Err(PagedKvError::ImmutableBlock { .. })
            ));
        }
        if reverse {
            targets.reverse();
        }
        while let Some(mut target) = targets.pop() {
            if targets.is_empty() {
                // Last remaining owner is exclusive after cache eviction.
                let reservation = target.reserve_to(&mut pool, 18).unwrap();
                target.commit(&mut pool, reservation).unwrap();
            }
            target.close(&mut pool).unwrap();
        }
        assert_eq!(pool.stats().allocated_block_count(), 0);
        assert_eq!(pool.reclaim_sequence(&reclaim).unwrap(), 0);
    }
}

#[test]
fn import_from_retained_orphan_and_released_export_fails_closed() {
    let mut pool = pool(4);
    let mut source = committed(&mut pool, 16);
    let expected = descriptor(&pool, 16);
    let mut export = pool.export_prefix(&source, expected.clone()).unwrap();
    source.close(&mut pool).unwrap();
    let mut target = pool.create_sequence(64).unwrap();
    target.import_prefix(&mut pool, &export, &expected).unwrap();
    pool.release_prefix(&mut export).unwrap();
    let mut next = pool.create_sequence(64).unwrap();
    assert!(next.import_prefix(&mut pool, &export, &expected).is_err());
    target.close(&mut pool).unwrap();
    assert_eq!(pool.stats().allocated_block_count(), 0);
}

#[test]
fn completed_cow_copies_all_valid_kv_slices_before_publication() {
    for length in [1, 15, 17, 31, 33] {
        let mut pool = pool(8);
        let mut source = committed(&mut pool, length);
        let expected = descriptor(&pool, length);
        let mut export = pool.export_prefix(&source, expected.clone()).unwrap();
        let mut target = pool.create_sequence(64).unwrap();
        target.import_prefix(&mut pool, &export, &expected).unwrap();
        let original = target.block_id((length - 1) / 16).unwrap();
        let mut ticket = target.begin_copy_on_write(&mut pool).unwrap().unwrap();
        let (from, to, valid) = ticket.copy_range().unwrap();
        assert_eq!(from, original);
        assert_ne!(from, to);
        assert_eq!(valid as usize, length % 16);
        assert!(target.block_table().is_err());
        assert!(target.reserve_to(&mut pool, length + 1).is_err());
        let bytes = pool.layout.bytes_per_kind() as usize;
        let mut buffers = [vec![0xa5; bytes], vec![0x5a; bytes]];
        let regions = ticket.copy_regions().collect::<Vec<_>>();
        assert_eq!(regions.len(), 8); // 2 kinds x 2 layers x 2 heads
                                      // Independent dense BF16 layout oracle for this 2-layer/8-page/2-head
                                      // fixture, rather than computing expected addresses with the API under test.
        for (ordinal, region) in regions.iter().enumerate() {
            let layer = (ordinal % 4) / 2;
            let head = ordinal % 2;
            let offset = |page: u32| (((layer * 8 + page as usize) * 2 + head) * 16 * 8 * 2) as u64;
            assert_eq!(region.value, ordinal >= 4);
            assert_eq!(region.source_offset, offset(from.physical_index()));
            assert_eq!(region.destination_offset, offset(to.physical_index()));
            assert_eq!(region.bytes, u64::from(valid) * 8 * 2);
        }
        let mut expected_buffers = buffers.clone();
        for (ordinal, region) in regions.iter().enumerate() {
            let buffer = &mut buffers[usize::from(region.value)];
            for offset in 0..region.bytes as usize {
                buffer[region.source_offset as usize + offset] = (ordinal * 13 + offset) as u8;
            }
        }
        for region in &regions {
            let kind = usize::from(region.value);
            let from = region.source_offset as usize;
            let to = region.destination_offset as usize;
            let bytes = region.bytes as usize;
            expected_buffers[kind][from..from + bytes]
                .copy_from_slice(&buffers[kind][from..from + bytes]);
            expected_buffers[kind][to..to + bytes]
                .copy_from_slice(&buffers[kind][from..from + bytes]);
            buffers[kind].copy_within(from..from + bytes, to);
        }
        assert_eq!(
            buffers, expected_buffers,
            "valid slices copied; all guard bytes retained"
        );
        assert_eq!(
            target
                .complete_copy_on_write(&mut pool, &mut ticket, true)
                .unwrap(),
            CopyOutcome::Published
        );
        assert_eq!(target.block_id((length - 1) / 16), Some(to));
        assert_eq!(source.block_id((length - 1) / 16), Some(original));
        assert!(ticket.copy_regions().next().is_none());
        assert_eq!(
            target
                .complete_copy_on_write(&mut pool, &mut ticket, true)
                .unwrap(),
            CopyOutcome::AlreadyCompleted
        );
        let reservation = target.reserve_to(&mut pool, length + 1).unwrap();
        target.commit(&mut pool, reservation).unwrap();
        target.close(&mut pool).unwrap();
        source.close(&mut pool).unwrap();
        pool.release_prefix(&mut export).unwrap();
        assert_eq!(pool.stats().allocated_block_count(), 0);
    }
}

#[test]
fn failed_cow_and_late_completion_do_not_publish_or_mutate_new_transactions() {
    for reset in [false, true] {
        let mut pool = pool(8);
        let mut source = committed(&mut pool, 17);
        let expected = descriptor(&pool, 17);
        let mut export = pool.export_prefix(&source, expected.clone()).unwrap();
        let mut target = pool.create_sequence(64).unwrap();
        target.import_prefix(&mut pool, &export, &expected).unwrap();
        let original = target.block_id(1).unwrap();
        let mut ticket = target.begin_copy_on_write(&mut pool).unwrap().unwrap();
        let next = if reset {
            target.reset(&mut pool).unwrap();
            Some(target.reserve_to(&mut pool, 1).unwrap())
        } else {
            None
        };
        assert_eq!(
            target
                .complete_copy_on_write(&mut pool, &mut ticket, reset)
                .unwrap(),
            CopyOutcome::Discarded
        );
        if let Some(reservation) = next {
            assert!(
                target.block_table().is_err(),
                "new reservation remains pending"
            );
            target.commit(&mut pool, reservation).unwrap();
            assert_eq!(target.logical_length(), 1);
        } else {
            assert_eq!(target.block_id(1), Some(original));
            assert_eq!(target.block_table().unwrap().logical_length(), 17);
        }
        target.close(&mut pool).unwrap();
        source.close(&mut pool).unwrap();
        pool.release_prefix(&mut export).unwrap();
        assert_eq!(pool.stats().allocated_block_count(), 0);
    }
}

#[test]
fn orphan_copy_retains_both_buffers_until_cancelled_transfer_quiesces() {
    let mut pool = pool(4);
    let mut source = committed(&mut pool, 1);
    let expected = descriptor(&pool, 1);
    let mut export = pool.export_prefix(&source, expected).unwrap();
    let mut ticket = source.begin_copy_on_write(&mut pool).unwrap().unwrap();
    let reclaim = source.abandon_for_reclaim();
    pool.reclaim_sequence(&reclaim).unwrap();
    pool.release_prefix(&mut export).unwrap();
    assert_eq!(pool.stats().allocated_block_count(), 2);
    assert_eq!(
        pool.discard_completed_copy(&mut ticket).unwrap(),
        CopyOutcome::Discarded
    );
    assert_eq!(pool.stats().allocated_block_count(), 0);
    assert_eq!(
        pool.discard_completed_copy(&mut ticket).unwrap(),
        CopyOutcome::AlreadyCompleted
    );
}

#[test]
fn cow_oom_and_wrong_target_preserve_source_and_ticket() {
    let mut pool = pool(1);
    let mut source = committed(&mut pool, 1);
    let mut export = pool.export_prefix(&source, descriptor(&pool, 1)).unwrap();
    assert!(matches!(
        source.begin_copy_on_write(&mut pool),
        Err(PagedKvError::OutOfBlocks { .. })
    ));
    assert_eq!(source.block_table().unwrap().logical_length(), 1);
    assert_eq!(pool.stats().allocated_block_count(), 1);
    pool.release_prefix(&mut export).unwrap();
    assert!(source.begin_copy_on_write(&mut pool).unwrap().is_none());
    source.close(&mut pool).unwrap();

    let mut pool = super::tests::pool(4);
    let mut source = committed(&mut pool, 1);
    let mut export = pool.export_prefix(&source, descriptor(&pool, 1)).unwrap();
    let mut ticket = source.begin_copy_on_write(&mut pool).unwrap().unwrap();
    let mut wrong = pool.create_sequence(64).unwrap();
    assert!(matches!(
        wrong.complete_copy_on_write(&mut pool, &mut ticket, true),
        Err(PagedKvError::ReservationMismatch)
    ));
    assert!(ticket.copy_range().is_some());
    source
        .complete_copy_on_write(&mut pool, &mut ticket, false)
        .unwrap();
    source.close(&mut pool).unwrap();
    pool.release_prefix(&mut export).unwrap();
    assert_eq!(pool.stats().allocated_block_count(), 0);
}
