use super::*;

const GOLDEN_REQUEST: &[u8] = include_bytes!("fixtures/request-n3-v2.bin");
const GOLDEN_RESULT: &[u8] = include_bytes!("fixtures/result-n3-v2.bin");

fn catalog(max: u32) -> Vec<CatalogEntry> {
    let mut result = Vec::new();
    for mode in [ResultMode::Greedy, ResultMode::FullLogits] {
        result.push(CatalogEntry {
            stage: Stage::Prefill128,
            bucket: 1,
            mode,
        });
        for bucket in [1, 2, 4, 8].into_iter().filter(|b| *b <= max) {
            result.push(CatalogEntry {
                stage: Stage::Decode,
                bucket,
                mode,
            });
        }
    }
    result
}
fn row(
    tag: u64,
    cookie: u64,
    token: u32,
    pos: u32,
    slot: u32,
    ids: Vec<u32>,
) -> ReservationExpectation {
    let mut valid = vec![16; ids.len()];
    *valid.last_mut().unwrap() = (pos % 16 + 1) as u16;
    ReservationExpectation {
        sequence_tag: tag,
        cookie,
        input_tokens: vec![token],
        committed_length: pos,
        target_length: pos + 1,
        output_slot: slot,
        generated_index: pos - 127,
        max_output_tokens: 32,
        physical_ids: ids,
        valid_tokens: valid,
    }
}
fn expectation() -> SubmissionExpectation {
    let rows = vec![
        row(101, 5001, 28, 128, 2, vec![7, 2, 19, 5, 11, 23, 0, 16, 9]),
        row(
            202,
            5002,
            339,
            144,
            0,
            vec![34, 28, 31, 24, 36, 25, 39, 29, 33, 21],
        ),
        row(
            303,
            5003,
            5248,
            158,
            1,
            vec![3, 4, 6, 8, 10, 12, 13, 14, 15, 17],
        ),
    ];
    let mut block_ownership: Vec<_> = rows
        .iter()
        .flat_map(|r| {
            r.physical_ids.iter().map(move |&id| BlockOwnership {
                physical_id: id,
                sequence_tag: r.sequence_tag,
            })
        })
        .collect();
    // A real off-batch resident request: in-pool does not mean available.
    block_ownership.push(BlockOwnership {
        physical_id: 1,
        sequence_tag: 404,
    });
    SubmissionExpectation {
        owner: OwnerExpectation {
            generation: 1,
            last_accepted_replay: 6,
            catalog_digest: [0x5a; 32],
            max_active_rows: 4,
            physical_block_count: 40,
            catalog: catalog(4),
        },
        replay_id: 7,
        iteration_id: 42,
        stage: Stage::Decode,
        mode: ResultMode::FullLogits,
        rows,
        block_ownership,
    }
}
fn prefill() -> SubmissionExpectation {
    let mut e = expectation();
    e.stage = Stage::Prefill128;
    e.rows.truncate(1);
    let r = &mut e.rows[0];
    r.input_tokens = (0..128).collect();
    r.committed_length = 0;
    r.target_length = 128;
    r.generated_index = 0;
    r.output_slot = 0;
    r.physical_ids.truncate(8);
    r.valid_tokens = vec![16; 8];
    e
}
fn shape(active: usize, mode: ResultMode) -> SubmissionExpectation {
    let mut e = expectation();
    e.mode = mode;
    if active >= 4 {
        e.owner.max_active_rows = if active > 4 { 8 } else { 4 };
        e.owner.physical_block_count = e.owner.max_active_rows * 10;
        e.owner.catalog = catalog(e.owner.max_active_rows);
        // Distinct four-request fixture, canonical nine-block targets.
        e.rows = (0..active as u64)
            .map(|i| {
                row(
                    100 + i,
                    6000 + i,
                    30 + i as u32,
                    128,
                    i as u32,
                    ((i * 10) as u32..(i * 10 + 9) as u32).collect(),
                )
            })
            .collect();
        e.block_ownership = e
            .rows
            .iter()
            .flat_map(|r| {
                r.physical_ids.iter().map(move |&id| BlockOwnership {
                    physical_id: id,
                    sequence_tag: r.sequence_tag,
                })
            })
            .collect();
    } else {
        e.rows.truncate(active);
        for (i, row) in e.rows.iter_mut().enumerate() {
            row.output_slot = i as u32;
        }
    }
    e
}
fn synthetic_result(e: &SubmissionExpectation) -> Vec<u8> {
    validate_expectations(e).unwrap();
    let mut bytes = vec![0; result_bytes(bucket(e.rows.len()).unwrap(), e.mode).unwrap()];
    write_header(&mut bytes, e, true).unwrap();
    for (i, r) in e.rows.iter().enumerate() {
        let b = 128 + 128 * i;
        for (off, value) in [
            (0, r.sequence_tag),
            (8, r.cookie),
            (16, e.replay_id),
            (24, e.iteration_id),
        ] {
            put64(&mut bytes, b + off, value).unwrap();
        }
        for (off, value) in [
            (32, i as u32),
            (36, r.output_slot),
            (40, r.target_length - 1),
            (44, r.generated_index),
            (48, 0),
            (52, 0),
            (56, 32),
            (88, 1),
        ] {
            put32(&mut bytes, b + off, value).unwrap();
        }
        if e.mode == ResultMode::FullLogits {
            put64(&mut bytes, b + 96, (1152 + i * LOGITS_BYTES) as u64).unwrap();
            put64(&mut bytes, b + 104, LOGITS_BYTES as u64).unwrap();
        }
    }
    bytes
}

#[test]
fn independent_golden_request_exact_and_result_scatter() {
    let e = expectation();
    let encoded = encode_request(&e).unwrap();
    assert_eq!(encoded.as_slice(), GOLDEN_REQUEST);
    assert_eq!(synthetic_result(&e), GOLDEN_RESULT);
    let decoded = decode_request(GOLDEN_REQUEST, &e).unwrap();
    assert_eq!(decoded.bucket, 4);
    assert_eq!(decoded.rows[0].physical_ids[6], 0);
    assert_eq!(
        decoded
            .rows
            .iter()
            .map(|r| r.output_slot)
            .collect::<Vec<_>>(),
        [2, 0, 1]
    );
    let output = validate_bulk_result(GOLDEN_RESULT, &e, CompletionEvidence::Quiesced).unwrap();
    assert_eq!(
        output
            .rows_by_slot()
            .iter()
            .map(|r| r.sequence_tag())
            .collect::<Vec<_>>(),
        [202, 303, 101]
    );
    assert_eq!(
        output
            .rows_by_slot()
            .iter()
            .map(|r| r.execution_row())
            .collect::<Vec<_>>(),
        [1, 2, 0]
    );
    assert_eq!(output.replay_id(), 7);
    assert_eq!(output.iteration_id(), 42);
    assert_eq!(output.owner_generation(), 1);
}

#[test]
fn every_request_byte_is_bound_in_both_stages() {
    for e in [expectation(), prefill()] {
        let original = encode_request(&e).unwrap();
        for offset in 0..REQUEST_BYTES {
            let mut changed = original;
            changed[offset] ^= 1;
            assert!(
                decode_request(&changed, &e).is_err(),
                "unbound byte {offset}"
            );
        }
    }
}

#[test]
fn p128_compatibility_view_and_all_prompt_tokens() {
    let e = prefill();
    let bytes = encode_request(&e).unwrap();
    assert_eq!(&bytes[1152..1232], &bytes[128..208]);
    assert_eq!(read32(&bytes, 1232).unwrap(), 0x50313238);
    assert_eq!(read32(&bytes, 1236).unwrap(), 128);
    assert_eq!(read32(&bytes, 1240).unwrap(), 0);
    assert_eq!(read32(&bytes, 1748).unwrap(), 127);
    assert_eq!(decode_request(&bytes, &e).unwrap().rows[0].input_count, 128);
    validate_bulk_result(&synthetic_result(&e), &e, CompletionEvidence::Quiesced).unwrap();
}

#[test]
fn all_exact_bucket_and_output_mode_sizes() {
    for active in 1..=8 {
        for mode in [ResultMode::Greedy, ResultMode::FullLogits] {
            let e = shape(active, mode);
            let request = encode_request(&e).unwrap();
            let b = if active <= 2 { active } else if active <= 4 { 4 } else { 8 };
            assert_eq!(decode_request(&request, &e).unwrap().bucket, b as u32);
            let bytes = synthetic_result(&e);
            assert_eq!(
                bytes.len(),
                1152 + if mode == ResultMode::FullLogits {
                    b * 98304
                } else {
                    0
                }
            );
            let out = validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).unwrap();
            assert_eq!(out.rows_by_slot().len(), active);
            assert_eq!(
                out.rows_by_slot()[0].logits().is_some(),
                mode == ResultMode::FullLogits
            );
        }
    }
    assert!(result_bytes(3, ResultMode::Greedy).is_err());
}

#[test]
fn unknown_stage_mode_version_and_lengths_rejected() {
    let e = expectation();
    let good = encode_request(&e).unwrap();
    for (offset, value) in [
        (0, 0),
        (4, 1),
        (6, 127),
        (8, (REQUEST_BYTES - 1) as u32),
        (12, 64),
        (16, 2),
        (20, 3),
        (24, 0),
        (28, 4),
        (104, 2),
        (108, 1),
    ] {
        let mut bytes = good;
        if offset == 4 || offset == 6 {
            put16(&mut bytes, offset, value as u16).unwrap();
        } else {
            put32(&mut bytes, offset, value).unwrap();
        }
        assert!(decode_request(&bytes, &e).is_err(), "accepted {offset}");
    }
    assert!(decode_request(&good[..REQUEST_BYTES - 1], &e).is_err());
    let mut extra = good.to_vec();
    extra.push(0);
    assert!(decode_request(&extra, &e).is_err());
    let result = synthetic_result(&e);
    assert!(
        validate_bulk_result(
            &result[..result.len() - 1],
            &e,
            CompletionEvidence::Quiesced
        )
        .is_err()
    );
}

#[test]
fn explicit_ownership_blocks_forged_off_batch_reassignment() {
    let mut e = expectation();
    e.rows[0].physical_ids[0] = 1;
    assert_eq!(encode_request(&e).unwrap_err().field, "block_ownership");
    let mut e = expectation();
    e.block_ownership.retain(|owner| owner.physical_id != 0);
    assert_eq!(encode_request(&e).unwrap_err().field, "block_ownership");
    let mut e = expectation();
    e.block_ownership.push(e.block_ownership[0]);
    assert!(encode_request(&e).is_err());
    // A structurally plausible forged packet cannot amend the external ledger.
    let e = expectation();
    let mut forged = encode_request(&e).unwrap();
    put32(&mut forged, 144, 1).unwrap();
    assert!(decode_request(&forged, &e).is_err());
}

#[test]
fn duplicate_physical_tags_cookies_slots_and_sparse_output_rejected() {
    let mut variants = Vec::new();
    let mut e = expectation();
    e.rows[1].sequence_tag = e.rows[0].sequence_tag;
    variants.push(e);
    let mut e = expectation();
    e.rows[1].cookie = e.rows[0].cookie;
    variants.push(e);
    let mut e = expectation();
    e.rows[1].output_slot = e.rows[0].output_slot;
    variants.push(e);
    let mut e = expectation();
    e.rows[1].output_slot = 3;
    variants.push(e);
    let mut e = expectation();
    e.rows[1].physical_ids[0] = e.rows[0].physical_ids[0];
    variants.push(e);
    let mut e = expectation();
    e.rows[0].physical_ids[1] = e.rows[0].physical_ids[0];
    variants.push(e);
    let mut e = expectation();
    e.rows[0].physical_ids[0] = 40;
    variants.push(e);
    for e in variants {
        assert!(encode_request(&e).is_err());
    }
}

#[test]
fn canonical_target_table_and_generation_bounds() {
    let mut variants = Vec::new();
    let mut e = expectation();
    e.rows[0].valid_tokens[0] = 15;
    variants.push(e);
    let mut e = expectation();
    *e.rows[0].valid_tokens.last_mut().unwrap() = 0;
    variants.push(e);
    let mut e = expectation();
    e.rows[0].physical_ids.pop();
    variants.push(e);
    let mut e = expectation();
    e.rows[0].committed_length -= 1;
    variants.push(e);
    let mut e = expectation();
    e.rows[0].generated_index += 1;
    variants.push(e);
    let mut e = expectation();
    e.rows[0].max_output_tokens = 1;
    variants.push(e);
    let mut e = expectation();
    e.rows[0].input_tokens.push(3);
    variants.push(e);
    let mut e = expectation();
    e.rows[0].input_tokens[0] = 49152;
    variants.push(e);
    let mut e = prefill();
    e.rows[0].input_tokens.pop();
    variants.push(e);
    let mut e = prefill();
    e.rows[0].committed_length = 1;
    e.rows[0].target_length = 129;
    variants.push(e);
    for e in variants {
        assert!(encode_request(&e).is_err());
    }
    // p159 fits160 KV positions but cannot issue output index32 under O<=32.
    let mut e = expectation();
    let r = &mut e.rows[2];
    r.committed_length = 159;
    r.target_length = 160;
    r.generated_index = 32;
    *r.valid_tokens.last_mut().unwrap() = 16;
    assert_eq!(encode_request(&e).unwrap_err().field, "generation_bound");
}

#[test]
fn checked_overflow_and_no_catalog_fallback() {
    let mut e = expectation();
    e.owner.last_accepted_replay = u64::MAX;
    e.replay_id = 0;
    assert_eq!(encode_request(&e).unwrap_err().field, "replay_id");
    let mut e = expectation();
    e.rows[0].committed_length = u32::MAX;
    assert_eq!(encode_request(&e).unwrap_err().field, "target_length");
    let mut e = expectation();
    e.owner.catalog.retain(|entry| entry.bucket != 4);
    assert_eq!(encode_request(&e).unwrap_err().field, "catalog");
    let mut e = expectation();
    e.owner
        .catalog
        .retain(|entry| entry.mode == ResultMode::Greedy);
    assert_eq!(encode_request(&e).unwrap_err().field, "catalog");
    let mut e = expectation();
    e.owner.max_active_rows = 2;
    assert!(encode_request(&e).is_err());
    assert!(span(&[0], usize::MAX, 2).is_err());
}

#[test]
fn result_header_every_byte_and_row_identity_rejected() {
    let e = expectation();
    let good = GOLDEN_RESULT;
    for offset in 0..128 {
        let mut bytes = good.to_vec();
        bytes[offset] ^= 1;
        assert!(
            validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).is_err(),
            "header {offset}"
        );
    }
    for base in [128, 256, 384] {
        for off in [
            0, 8, 16, 24, 32, 36, 40, 44, 52, 56, 60, 64, 72, 80, 88, 92, 96, 104, 112, 127,
        ] {
            let mut bytes = good.to_vec();
            bytes[base + off] ^= 1;
            assert!(
                validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).is_err(),
                "row {base} field {off}"
            );
        }
    }
}

#[test]
fn malformed_last_row_never_exposes_earlier_rows() {
    let e = expectation();
    let request = encode_request(&e).unwrap();
    let mut tx = Transaction::admit(&request, e.clone()).unwrap();
    let mut bytes = GOLDEN_RESULT.to_vec();
    put32(&mut bytes, 384 + 52, 1).unwrap();
    assert!(
        tx.accept_result(&bytes, CompletionEvidence::Quiesced)
            .is_err()
    );
    assert!(tx.result().is_err());
    assert_eq!(
        tx.settlement_advice(),
        SettlementAdvice::ContainAffectedRequests
    );
    assert_eq!(tx.retained_expectation(), &e);
}

#[test]
fn inactive_rows_logits_and_arbitrary_spans_rejected() {
    let e = expectation();
    for offset in [512, 592, 639, 296064, GOLDEN_RESULT.len() - 1] {
        let mut bytes = GOLDEN_RESULT.to_vec();
        bytes[offset] = 1;
        assert!(validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).is_err());
    }
    let mut bytes = GOLDEN_RESULT.to_vec();
    put64(&mut bytes, 128 + 96, u64::MAX - 1).unwrap();
    assert!(validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).is_err());
    let mut e = expectation();
    e.mode = ResultMode::Greedy;
    let mut bytes = synthetic_result(&e);
    bytes[128 + 96] = 1;
    assert!(validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).is_err());
}

#[test]
fn bf16_bits_preserved_and_nonfinite_rejected() {
    let e = expectation();
    let mut bytes = GOLDEN_RESULT.to_vec();
    put16(&mut bytes, 1152, 0x8000).unwrap();
    put16(&mut bytes, 1154, 0x0001).unwrap();
    let validated = validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).unwrap();
    assert_eq!(
        &validated.rows_by_slot()[2].logits().unwrap()[..4],
        &[0, 128, 1, 0]
    );
    for word in [0x7f80, 0xff80, 0x7fc1, 0x7f81] {
        put16(&mut bytes, 1152, word).unwrap();
        assert_eq!(
            validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced)
                .unwrap_err()
                .field,
            "logits"
        );
    }
}

#[test]
fn greedy_tokens_scatter_independently_from_execution_rows() {
    let mut e = expectation();
    e.mode = ResultMode::Greedy;
    let mut bytes = synthetic_result(&e);
    for (i, token) in [11, 22, 33].iter().enumerate() {
        put32(&mut bytes, 128 + 128 * i + 48, *token).unwrap();
    }
    let output = validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).unwrap();
    assert_eq!(
        output
            .rows_by_slot()
            .iter()
            .map(|r| r.token_id())
            .collect::<Vec<_>>(),
        [22, 33, 11]
    );
    for invalid in [49152, u32::MAX] {
        put32(&mut bytes, 128 + 48, invalid).unwrap();
        assert!(validate_bulk_result(&bytes, &e, CompletionEvidence::Quiesced).is_err());
    }
}

#[test]
fn unknown_completion_retains_snapshot_no_late_publication() {
    let e = expectation();
    let mut owner = CodecOwner::new(e.owner.clone());
    owner.admit(GOLDEN_REQUEST, e.clone()).unwrap();
    assert!(
        owner
            .accept_result(GOLDEN_RESULT, CompletionEvidence::Unknown)
            .is_err()
    );
    assert_eq!(owner.settlement_advice(), Some(SettlementAdvice::RetainAll));
    assert_eq!(owner.retained_expectation(), Some(&e));
    assert!(owner.result().is_err());
    assert!(owner.settle_failure(true).is_err());
    assert!(
        owner
            .accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
            .is_err()
    );
    assert!(owner.result().is_err());
    owner
        .establish_quiescence_after_failure(CompletionEvidence::Quiesced)
        .unwrap();
    assert_eq!(
        owner.settlement_advice(),
        Some(SettlementAdvice::ContainAffectedRequests)
    );
    owner.settle_failure(true).unwrap();
    let mut next = e;
    next.owner = owner.expectation().clone();
    next.replay_id = 8;
    assert!(owner.admit(&encode_request(&next).unwrap(), next).is_err());
}

#[test]
fn any_repeated_result_attempt_invalidates_ready_output() {
    let e = expectation();
    let mut owner = CodecOwner::new(e.owner.clone());
    owner.admit(GOLDEN_REQUEST, e).unwrap();
    owner
        .accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
        .unwrap();
    assert!(owner.result().is_ok());
    assert!(
        owner
            .accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
            .is_err()
    );
    assert!(owner.result().is_err());
    assert!(owner.settle_success(true).is_err());
}

#[test]
fn owner_preflight_failure_does_not_advance_and_stale_replay_cannot_reenter() {
    let e = expectation();
    let mut owner = CodecOwner::new(e.owner.clone());
    let mut bad = GOLDEN_REQUEST.to_vec();
    bad[0] ^= 1;
    assert!(owner.admit(&bad, e.clone()).is_err());
    assert_eq!(owner.expectation().last_accepted_replay, 6);
    assert!(owner.retained_expectation().is_none());
    assert!(owner.result().is_err());
    owner.admit(GOLDEN_REQUEST, e.clone()).unwrap();
    owner
        .accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
        .unwrap();
    assert!(owner.settle_success(false).is_err());
    let result = owner.settle_success(true).unwrap();
    assert_eq!(result.rows_by_slot().len(), 3);
    assert!(owner.result().is_err());
    assert!(owner.admit(GOLDEN_REQUEST, e).is_err());
    assert_eq!(owner.expectation().last_accepted_replay, 7);
}

#[test]
fn unsettled_ledger_cannot_be_replaced_or_published_after_busy_attempt() {
    for ready in [false, true] {
        let e = expectation();
        let mut owner = CodecOwner::new(e.owner.clone());
        owner.admit(GOLDEN_REQUEST, e.clone()).unwrap();
        if ready {
            owner
                .accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
                .unwrap();
        }
        let mut next = e.clone();
        next.owner = owner.expectation().clone();
        next.replay_id = 8;
        next.iteration_id = 43;
        assert!(owner.admit(&encode_request(&next).unwrap(), next).is_err());
        assert_eq!(owner.retained_expectation(), Some(&e));
        assert!(owner.result().is_err());
        if !ready {
            assert!(
                owner
                    .accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
                    .is_err()
            );
        }
        assert_eq!(
            owner.settlement_advice(),
            Some(SettlementAdvice::ContainAffectedRequests)
        );
        assert!(owner.settle_success(true).is_err());
        owner.settle_failure(true).unwrap();
    }
}

#[test]
fn shrinking_and_growing_buckets_need_fresh_replay_and_result_shape() {
    let initial = shape(4, ResultMode::Greedy);
    let mut owner = CodecOwner::new(initial.owner.clone());
    for (offset, active) in [4, 3, 1, 2].into_iter().enumerate() {
        let mut e = shape(active, ResultMode::Greedy);
        e.owner = owner.expectation().clone();
        e.replay_id = e.owner.last_accepted_replay + 1;
        e.iteration_id = 42 + offset as u64;
        let request = encode_request(&e).unwrap();
        let response = synthetic_result(&e);
        owner.admit(&request, e).unwrap();
        assert!(owner.result().is_err());
        owner
            .accept_result(&response, CompletionEvidence::Quiesced)
            .unwrap();
        assert_eq!(
            owner.settle_success(true).unwrap().rows_by_slot().len(),
            active
        );
    }
}

#[test]
fn caller_changes_cannot_rewrite_admitted_snapshot() {
    let mut e = expectation();
    let original = e.clone();
    let mut tx = Transaction::admit(GOLDEN_REQUEST, e.clone()).unwrap();
    // A cancellation/routing change in outside state cannot rewrite the native
    // reservation snapshot. Actual cancellation publication remains scheduler work.
    e.rows.clear();
    e.block_ownership.clear();
    assert_eq!(tx.retained_expectation(), &original);
    tx.accept_result(GOLDEN_RESULT, CompletionEvidence::Quiesced)
        .unwrap();
    assert_eq!(tx.result().unwrap().rows_by_slot().len(), 3);
}

#[test]
fn two_row_owner_has_exact_catalog_capacity_and_pool_bounds() {
    let mut e = shape(4, ResultMode::Greedy);
    e.rows.truncate(2);
    e.owner.max_active_rows = 2;
    e.owner.physical_block_count = 20;
    e.owner.catalog.retain(|entry| entry.bucket <= 2);
    e.block_ownership.retain(|entry| entry.physical_id < 20);
    let bytes = encode_request(&e).unwrap();
    assert_eq!(decode_request(&bytes, &e).unwrap().bucket, 2);
    validate_bulk_result(&synthetic_result(&e), &e, CompletionEvidence::Quiesced).unwrap();

    let mut missing = e.clone();
    missing.owner.catalog.retain(|entry| entry.bucket != 2);
    assert_eq!(encode_request(&missing).unwrap_err().field, "catalog");
    let mut invalid_catalog = e.clone();
    invalid_catalog.owner.catalog.push(CatalogEntry {
        stage: Stage::Decode,
        bucket: 4,
        mode: ResultMode::Greedy,
    });
    assert_eq!(
        encode_request(&invalid_catalog).unwrap_err().field,
        "catalog"
    );
    let mut undersized = e.clone();
    undersized.owner.physical_block_count = 19;
    assert_eq!(
        encode_request(&undersized).unwrap_err().field,
        "physical_block_count"
    );
    let mut over_capacity = e;
    over_capacity
        .rows
        .push(shape(4, ResultMode::Greedy).rows[2].clone());
    assert_eq!(
        encode_request(&over_capacity).unwrap_err().field,
        "active_rows"
    );
}

#[test]
fn prefill_then_decode_requires_new_binding_and_respects_o1_terminal() {
    let mut p = prefill();
    p.mode = ResultMode::Greedy;
    let mut owner = CodecOwner::new(p.owner.clone());
    owner
        .admit(&encode_request(&p).unwrap(), p.clone())
        .unwrap();
    let p_result = synthetic_result(&p);
    owner
        .accept_result(&p_result, CompletionEvidence::Quiesced)
        .unwrap();
    owner.settle_success(true).unwrap();

    let mut d = shape(1, ResultMode::Greedy);
    d.owner = owner.expectation().clone();
    d.replay_id = 8;
    d.iteration_id = 43;
    d.rows[0].cookie = p.rows[0].cookie + 1;
    let bytes = encode_request(&d).unwrap();
    assert!(bytes[1152..1752].iter().all(|&x| x == 0));
    assert!(validate_bulk_result(&p_result, &d, CompletionEvidence::Quiesced).is_err());
    owner.admit(&bytes, d.clone()).unwrap();
    owner
        .accept_result(&synthetic_result(&d), CompletionEvidence::Quiesced)
        .unwrap();
    assert_eq!(
        owner.settle_success(true).unwrap().rows_by_slot()[0].generated_index(),
        1
    );

    let mut terminal = prefill();
    terminal.rows[0].max_output_tokens = 1;
    encode_request(&terminal).unwrap();
    d.rows[0].max_output_tokens = 1;
    assert_eq!(encode_request(&d).unwrap_err().field, "generation_bound");
}

#[test]
fn old_wire_version_is_not_admitted() {
 let e=expectation();
 assert!(decode_request(include_bytes!("fixtures/request-n3.bin"),&e).is_err());
 assert!(validate_bulk_result(include_bytes!("fixtures/result-n3.bin"),&e,CompletionEvidence::Quiesced).is_err());
}
#[test]
fn eighth_row_identity_and_cross_row_block_alias_are_rejected() {
 let e=shape(8,ResultMode::FullLogits);
 let good=encode_request(&e).unwrap();
 for offset in [128+7*128+80,128+7*128+88,128+7*128+100] {
  let mut bytes=good;bytes[offset]^=1;assert!(decode_request(&bytes,&e).is_err());
 }
 let mut alias=e.clone();alias.rows[7].physical_ids[0]=alias.rows[0].physical_ids[0];
 assert!(encode_request(&alias).is_err());
 let mut output=synthetic_result(&e);output[128+7*128]^=1;
 assert!(validate_bulk_result(&output,&e,CompletionEvidence::Quiesced).is_err());
}
