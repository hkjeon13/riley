//! Opt-in byte evidence shared by the original and split-stage profile2 graphs.
//!
//! Run the same ignored test in both isolated source copies, then compare the
//! binary files. The manifest alone is not a correctness or performance claim.
use super::*;
use crate::llama::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_model::{LoadLimits, LoadedModel};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    result
}

struct ByteEvidence {
    output: BufWriter<std::fs::File>,
    digest: Sha256,
    offset: u64,
    records: Vec<Value>,
}

impl ByteEvidence {
    fn append(&mut self, bytes: &[u8], mut record: Value) -> std::io::Result<()> {
        record["offset"] = json!(self.offset);
        record["bytes"] = json!(bytes.len());
        record["sha256"] = json!(hex(&Sha256::digest(bytes)));
        self.output.write_all(bytes)?;
        self.digest.update(bytes);
        self.offset += bytes.len() as u64;
        self.records.push(record);
        Ok(())
    }
}

#[test]
#[ignore = "requires SM89 CUDA13, SmolLM2 checkpoint, and new RILEY_STAGE_PARITY_OUTPUT path"]
fn owned_profile2_stage_parity_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = PathBuf::from(
        std::env::var_os("RILEY_STAGE_PARITY_OUTPUT")
            .ok_or("RILEY_STAGE_PARITY_OUTPUT must name a new binary evidence file")?,
    );
    let mut manifest_name = output_path.as_os_str().to_os_string();
    manifest_name.push(".json");
    let manifest_path = PathBuf::from(manifest_name);
    if output_path.exists() || manifest_path.exists() {
        return Err("refusing to overwrite stage parity binary or manifest".into());
    }
    let mut evidence = ByteEvidence {
        output: BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output_path)?,
        ),
        digest: Sha256::new(),
        offset: 0,
        records: Vec::new(),
    };
    // A failed run leaves an empty manifest, never a completed success receipt.
    let mut manifest = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)?;
    let checkpoint = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
    let model = LoadedModel::load(std::path::Path::new(&checkpoint), LoadLimits::default())?;
    let hello = model
        .tokenizer()
        .encode(&"Hello".repeat(128), riley_model::EncodeOptions::default())?;
    assert_eq!(hello.len(), 128);
    let context = riley_cuda::CudaRuntime::initialize()?
        .device(0)?
        .create_context()?;
    let mut stream = context.create_stream()?;
    let config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16)?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_grouped_ragged_attention_heads()
    .with_separate_residual_norm()
    .with_iteration_batch_completion()
    .with_packed_async_metadata()
    .with_vllm_smol_p128_graph();
    let mut prepared = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
    let vocabulary_size = prepared.vocabulary_size();
    let cache_bytes = usize::try_from(prepared.owner.layout.bytes_per_kind())?;
    let mut io = context.allocate_pinned_host_buffer(cache_bytes as u64)?;
    let mut cache = vec![0; cache_bytes];
    // Initialize untouched padding too: full-cache comparison must never depend
    // on allocator contents left by an earlier test or graph preparation.
    prepared
        .owner
        .key_cache
        .upload_from_slice(0, &cache, &mut io, &mut stream)?;
    prepared
        .owner
        .value_cache
        .upload_from_slice(0, &cache, &mut io, &mut stream)?;
    let mut candidate = prepared.into_owned_decode_graph(&context)?;
    assert_eq!(candidate.numerical_profile_id(), "vllm-smol-p128-v1");
    candidate.validate_request_shape(128, 32)?;
    let diverse_a: Vec<u32> = (0_u32..128)
        .map(|i| (i * 337 + 11) % vocabulary_size as u32)
        .collect();
    let diverse_b: Vec<u32> = (0_u32..128)
        .map(|i| (i * i * 31 + i * 719 + 101) % vocabulary_size as u32)
        .collect();
    let requests = [
        ("hello_complete", &hello, 159_usize),
        ("hello_cancel_prefill", &hello, 23),
        ("hello_cancel_output", &hello, 150),
        ("diverse_a_complete", &diverse_a, 159),
        ("diverse_b_complete", &diverse_b, 159),
        ("hello_reuse_complete", &hello, 159),
    ];
    let mut request_records = Vec::new();
    let mut expected_hello = None;
    for (request, (label, prompt, limit)) in requests.into_iter().enumerate() {
        let mapping: Vec<u32> = (0..16)
            .map(|i| ((i * 7 + request * 3) % 16) as u32)
            .collect();
        let mut token = prompt[0];
        let mut output_tokens = Vec::new();
        for position in 0..limit {
            if position < 128 {
                token = prompt[position];
            }
            let live = position / 16 + 1;
            let mut valid = vec![16; live];
            valid[live - 1] = (position % 16 + 1) as u16;
            let input = [token];
            let kind = if position < 128 {
                LlamaBatchRowKind::Prefill
            } else {
                LlamaBatchRowKind::Decode
            };
            let rows = [LlamaBatchRow::new(
                (request + 1) as u64,
                kind,
                &input,
                (position + 1) as u32,
                LlamaBatchBlockTable::new(
                    crate::paged_kv::BLOCK_TABLE_V1_VERSION,
                    &mapping[..live],
                    &valid,
                    (position + 1) as u32,
                ),
                Some(0),
            )];
            let before = context.allocation_stats()?;
            let logits = candidate.execute(&rows)?;
            let logits_bytes = logits.len();
            let logits_sha256 = hex(&Sha256::digest(logits));
            assert_eq!(logits_bytes, vocabulary_size * 2);
            assert_eq!(
                context.allocation_stats()?,
                before,
                "replay allocated CUDA memory"
            );
            token = candidate.greedy_token()?;
            assert!(token < vocabulary_size as u32);
            // Completed raw transfer includes all BF16 logits, argmax status,
            // embedding status and initialized padding, without duplicate data.
            evidence.append(
                &candidate.output,
                json!({
                    "kind": "completed_output", "request": request, "position": position,
                    "stage": if position < 128 { "prefill" } else { "decode" },
                    "input_token": input[0], "greedy_token": token,
                    "logits_offset_within_record": 0, "logits_bytes": logits_bytes,
                    "logits_sha256": logits_sha256,
                }),
            )?;
            if position >= 127 {
                output_tokens.push(token);
            }
        }
        if label == "hello_complete" {
            expected_hello = Some(output_tokens.clone());
        }
        if label == "hello_reuse_complete" {
            assert_eq!(
                Some(&output_tokens),
                expected_hello.as_ref(),
                "cancelled/diverse requests contaminated reused KV mappings"
            );
        }
        request_records.push(json!({
            "request": request, "label": label, "prompt_tokens": prompt,
            "positions_executed": limit, "physical_block_mapping": mapping,
            "output_tokens": output_tokens,
        }));
    }
    assert_eq!(candidate.replay_count(), 809);
    let replay_count = candidate.replay_count();
    // Close the aggregate graph first; only then are actual KV parents available
    // to download and close. Both source versions follow this same lifecycle.
    let mut parents = candidate.graph.close()?;
    parents
        .executor
        .owner
        .key_cache
        .download_to_slice(0, &mut cache, &mut io, &mut stream)?;
    evidence.append(
        &cache,
        json!({"kind": "final_key_cache", "initialized_to_zero": true}),
    )?;
    parents
        .executor
        .owner
        .value_cache
        .download_to_slice(0, &mut cache, &mut io, &mut stream)?;
    evidence.append(
        &cache,
        json!({"kind": "final_value_cache", "initialized_to_zero": true}),
    )?;
    parents.executor.close()?;
    parents.metadata.close()?;
    parents.result.close()?;
    parents.staging.close()?;
    parents.stream.close()?;
    io.close()?;
    stream.close()?;
    let allocations = context.allocation_stats()?;
    assert_eq!(allocations.device_live_allocations(), 0);
    assert_eq!(allocations.pinned_host_live_allocations(), 0);
    context.close()?;
    evidence.output.flush()?;
    evidence.output.get_ref().sync_all()?;
    let digest = hex(&evidence.digest.finalize());
    serde_json::to_writer_pretty(
        &mut manifest,
        &json!({
            "schema_version": "riley.owned-stage-parity-bytes.v1",
            "status": "complete", "numerical_profile": "vllm-smol-p128-v1",
            "performance_claim_eligible": false, "comparison_required": true,
            "replays": replay_count, "requests": request_records,
            "bytes": evidence.offset, "sha256": digest,
            "zero_device_and_pinned_allocations_after_close": true,
            "records": evidence.records,
        }),
    )?;
    manifest.write_all(b"\n")?;
    manifest.sync_all()?;
    println!(
        "STAGE_PARITY_BYTES replays={replay_count} requests=6 bytes={} sha256={digest} zero_allocations=true output={}",
        evidence.offset,
        output_path.display()
    );
    Ok(())
}
