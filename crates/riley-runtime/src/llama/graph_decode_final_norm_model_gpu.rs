//! G02C checks the actual executor's selected profile before opening capture.
use super::PreparedLlamaBatchExecutor;
use crate::llama::forward::LlamaRmsNormProfile;
use crate::llama::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use crate::paged_kv::BLOCK_TABLE_V1_VERSION;
use riley_cuda::CudaRuntime;
use riley_model::{LoadLimits, LoadedModel};

#[test]
#[ignore = "requires CUDA and RILEY_REAL_CHECKPOINT pinned SmolLM2"]
fn c07_final_norm_model_hf_owner_preserves_norm_and_continuation()
-> Result<(), Box<dyn std::error::Error>> {
    model_case("RILEY_REAL_CHECKPOINT", false, false)
}

#[test]
#[ignore = "requires CUDA and RILEY_CANONICAL_CHECKPOINT synthetic canonical fixture"]
fn c07_final_norm_model_canonical_owner_preserves_norm_and_continuation()
-> Result<(), Box<dyn std::error::Error>> {
    model_case("RILEY_CANONICAL_CHECKPOINT", true, false)
}

#[test]
#[ignore = "requires CUDA and RILEY_REAL_CHECKPOINT"]
fn c07_layer_norm_model_hf_all_layers_and_continuation() -> Result<(), Box<dyn std::error::Error>> {
    model_case("RILEY_REAL_CHECKPOINT", false, true)
}

#[test]
#[ignore = "requires CUDA and RILEY_CANONICAL_CHECKPOINT"]
fn c07_layer_norm_model_canonical_all_layers_and_continuation()
-> Result<(), Box<dyn std::error::Error>> {
    model_case("RILEY_CANONICAL_CHECKPOINT", true, true)
}

#[test]
#[ignore = "requires CUDA and RILEY_ODD_CHECKPOINT synthetic three-layer fixture"]
fn c07_hidden_bindings_odd_layer_continuation() -> Result<(), Box<dyn std::error::Error>> {
    model_case("RILEY_ODD_CHECKPOINT", true, false)
}

fn model_case(
    checkpoint_env: &str,
    canonical: bool,
    layer_audit: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = std::env::var_os(checkpoint_env).ok_or("checkpoint required")?;
    let model = LoadedModel::load(std::path::Path::new(&checkpoint), LoadLimits::default())?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut embedding_report = context.allocate_pinned_host_buffer(32)?;
    let bounds = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)?;
    let config =
        PreparedLlamaBatchExecutorConfig::new(bounds, PreparedLlamaForwardConfig::default())
            .with_grouped_ragged_attention_heads()
            .with_iteration_batch_completion()
            .with_packed_async_metadata();
    let mut baseline = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
    let mut candidate = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
    assert_eq!(
        candidate.owner.forward.rms_norm_profile(),
        if canonical {
            LlamaRmsNormProfile::Canonical
        } else {
            LlamaRmsNormProfile::HuggingFaceSmolLm2
        }
    );
    if layer_audit {
        let stats = context.allocation_stats()?;
        candidate.config = config.with_fused_residual_norm();
        assert!(candidate.audit_c07_layer_norms(&mut stream).is_err());
        assert!(!candidate.owner.poisoned);
        assert_eq!(stats, context.allocation_stats()?);
        candidate.config = config;
        assert!(candidate.audit_c07_layer_norms(&mut stream).is_err());
        assert!(candidate.audit_c07_lm_head(&mut stream).is_err());
        assert!(candidate.audit_c07_greedy(&mut stream).is_err());
        assert!(candidate.audit_c07_h2d(&[], &mut stream).is_err());
        assert!(!candidate.owner.poisoned);
    }
    let mut evidence = Vec::new();
    let mut token = 504_u32;
    let mut predictions = Vec::new();
    for step in 0..4 {
        let tokens = [token];
        let valid = [(step + 1) as u16];
        let rows = [LlamaBatchRow::new(
            1,
            if step == 0 {
                LlamaBatchRowKind::Prefill
            } else {
                LlamaBatchRowKind::Decode
            },
            &tokens,
            step + 1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &[0], &valid, step + 1),
            Some(0),
        )];
        baseline.execute(&rows, &mut stream)?;
        candidate.execute(&rows, &mut stream)?;
        let mut expected = vec![0; baseline.output_byte_len()?];
        let mut actual = vec![0; candidate.output_byte_len()?];
        baseline.download_logits(&mut expected, &mut stream)?;
        let stats = context.allocation_stats()?;
        let (devices, plans) = candidate.audit_c07_resource_reservation(&mut stream)?;
        println!(
            "G03_RESOURCES canonical={canonical} step={step} device_parents={devices} plans={plans} pinned=2 close=passed"
        );
        candidate.audit_c07_attention_chain(&rows, &mut stream)?;
        println!(
            "G03_ATTENTION_CHAIN canonical={canonical} step={step} replays=32 eager_exact=true"
        );
        candidate.audit_c07_qkv_kv(&rows, &mut stream)?;
        println!("G03_QKV_KV canonical={canonical} step={step} replays=32 cpu_scatter_exact=true");
        candidate.audit_c07_qkv_rope(&rows, &mut stream)?;
        println!("G03_QKV_ROPE canonical={canonical} step={step} replays=32 eager_exact=true");
        candidate.audit_c07_norm_qkv(&mut stream)?;
        println!("G03_NORM_QKV canonical={canonical} step={step} replays=32 eager_exact=true");
        candidate.audit_c07_layer_tail(&mut stream)?;
        println!("G03_LAYER_TAIL canonical={canonical} step={step} replays=32 eager_exact=true");
        candidate.audit_c07_norm_mlp_chain(&mut stream)?;
        println!("G03_NORM_MLP canonical={canonical} step={step} replays=32 eager_exact=true");
        candidate.audit_c07_mlp_chain(&mut stream)?;
        println!("G03_MLP canonical={canonical} step={step} replays=32 eager_exact=true");
        candidate.audit_c07_swiglu_chain(&mut stream)?;
        println!("G03_SWIGLU canonical={canonical} step={step} replays=32 eager_exact=true");
        assert_eq!(stats, context.allocation_stats()?);
        {
            use crate::llama::graph::GraphOperatorCapability;
            use crate::llama::graph_decode_capture_inventory::{
                PureDecodeGraphV1CaptureCapabilityInventory, PureDecodeGraphV1CaptureOperation,
            };
            use crate::llama::graph_decode_final_norm_owner::FinalNormBinding;
            let plan = &candidate.owner.forward.plan;
            let binding = FinalNormBinding::new(
                candidate.owner.forward.rms_norm_profile(),
                1,
                plan.dimensions().hidden_size() as u64,
                plan.final_norm_epsilon(),
                plan.final_norm_weight(),
            )
            .unwrap();
            let mut before =
                vec![0; candidate.owner.forward.buffers.hidden_norm.byte_len() as usize];
            candidate
                .owner
                .forward
                .buffers
                .hidden_norm
                .download_to_slice(
                    0,
                    &mut before,
                    &mut candidate.owner.forward.io_staging,
                    &mut stream,
                )?;
            if layer_audit {
                if step == 0 {
                    candidate.config = config.with_fused_residual_norm();
                    assert!(candidate.audit_c07_pointwise(&mut stream).is_err());
                    assert!(!candidate.owner.poisoned);
                    assert_eq!(context.allocation_stats()?, stats);
                    candidate.config = config;
                }
                if step == 0 {
                    let f = &candidate.owner.forward;
                    for (name, plan) in [
                        ("hidden", &f.gemms.hidden),
                        ("key_value", &f.gemms.key_value),
                        ("intermediate", &f.gemms.intermediate),
                        ("down", &f.gemms.down),
                        ("lm_head", &f.gemms.lm_head),
                    ] {
                        let c = plan.config();
                        println!(
                            "G02B plan canonical={canonical} name={name} m={} n={} k={} policy={} workspace={} shared_workspace={:?}",
                            c.m(),
                            c.n(),
                            c.k(),
                            c.reduction_policy().id(),
                            plan.workspace_bytes(),
                            f.buffers
                                .gemm_workspace
                                .as_ref()
                                .map(riley_cuda::CudaDeviceBuffer::byte_len)
                        );
                        if let crate::llama::forward::PreparedLlamaGemm::Canonical(p) = plan {
                            println!(
                                "G02B algorithm canonical={canonical} name={name} metadata={:?}",
                                p.algorithm_metadata()
                            );
                        }
                    }
                }
                candidate.audit_c07_h2d(&rows, &mut stream)?;
                println!(
                    "G02P1 canonical={canonical} step={step} actual token/metadata H2D graph and full slab parity passed"
                );
                let projections = candidate.audit_c07_gemms(&mut stream)?;
                assert_eq!(projections, candidate.owner.forward.plan.layers().len() * 7);
                println!(
                    "G02B selected canonical={canonical} step={step} projections={projections} actual plans/optional workspace graph parity and restore passed; no sentinel"
                );
                let graph_logits = candidate.audit_c07_lm_head(&mut stream)?;
                assert_eq!(
                    graph_logits, expected,
                    "LM head graph must match independent baseline logits"
                );
                println!(
                    "G02D canonical={canonical} step={step} actual LM head graph/logits parity and restoration passed"
                );
                candidate.audit_c07_embedding(&rows, &mut stream, &mut embedding_report)?;
                eprintln!(
                    "G02A canonical={canonical} step={step} actual table/token/output parity and restore passed"
                );
                candidate.audit_c07_pointwise(&mut stream)?;
                eprintln!(
                    "G02P5/P6/P7 canonical={canonical} step={step} SiLU/multiply/attention-residual/MLP-residual parity passed"
                );
                candidate.audit_c07_kv_write(&rows, &mut stream)?;
                eprintln!(
                    "G02P4 canonical={canonical} step={step} all-layer KV write and restoration passed"
                );
                candidate.audit_c07_rope(&rows, &mut stream)?;
                eprintln!(
                    "G02P3 canonical={canonical} step={step} actual packed positions and Q/K parity passed"
                );
                use crate::llama::graph_decode_final_norm_owner::NormBindingSite;
                let layer_count = candidate.owner.forward.plan.layers().len();
                let receipts = candidate.audit_c07_layer_norms(&mut stream)?;
                assert_eq!(receipts.len(), layer_count * 2);
                for (layer, pair) in receipts.chunks_exact(2).enumerate() {
                    assert_eq!(pair[0].site, NormBindingSite::Input(layer));
                    assert_eq!(pair[1].site, NormBindingSite::PostAttention(layer));
                    for receipt in pair {
                        eprintln!(
                            "G02P2 canonical={canonical} step={step} site={:?} hashes={:02x?}",
                            receipt.site, receipt.hashes
                        );
                    }
                }
            } else {
                let mut owner = candidate.owner.prepare_c07_final_norm_graph(&mut stream)?;
                let inventory = owner.bind_inventory(
                    PureDecodeGraphV1CaptureCapabilityInventory::default(),
                    binding,
                );
                assert_eq!(
                    inventory.capability_for(PureDecodeGraphV1CaptureOperation::FinalNorm),
                    GraphOperatorCapability::Supported
                );
                assert_eq!(
                    inventory.operator_capability(),
                    GraphOperatorCapability::Unknown
                );
                for _ in 0..32 {
                    owner.replay()?;
                }
                eprintln!(
                    "executor canonical={canonical} step={step} input/weight/output SHA256={:02x?}",
                    owner.parity_hashes()
                );
                owner.close()?;
            }
            let mut after = vec![0; before.len()];
            candidate
                .owner
                .forward
                .buffers
                .hidden_norm
                .download_to_slice(
                    0,
                    &mut after,
                    &mut candidate.owner.forward.io_staging,
                    &mut stream,
                )?;
            assert_eq!(
                before, after,
                "actual final norm owner changed eager output"
            );
        }
        assert!(!candidate.owner.poisoned);
        assert_eq!(context.allocation_stats()?, stats);
        candidate.download_logits(&mut actual, &mut stream)?;
        assert_eq!(
            actual, expected,
            "final norm intervention changed logits/continuation at {step}"
        );
        evidence.extend_from_slice(&expected);
        let kv = candidate.kv_layout();
        for (left, right) in [
            (
                &mut baseline.owner.key_cache,
                &mut candidate.owner.key_cache,
            ),
            (
                &mut baseline.owner.value_cache,
                &mut candidate.owner.value_cache,
            ),
        ] {
            let mut left_bytes = vec![0; kv.bytes_per_kind() as usize];
            let mut right_bytes = vec![0; kv.bytes_per_kind() as usize];
            left.download_to_slice(
                0,
                &mut left_bytes,
                &mut baseline.owner.forward.io_staging,
                &mut stream,
            )?;
            right.download_to_slice(
                0,
                &mut right_bytes,
                &mut candidate.owner.forward.io_staging,
                &mut stream,
            )?;
            for layer in 0..kv.layer_count() {
                for head in 0..kv.key_value_head_count() {
                    let offset = (kv.layer_byte_offset(layer).unwrap()
                        + head as u64 * kv.head_stride_bytes())
                        as usize;
                    let end = offset + usize::try_from(step + 1)? * 64 * 2;
                    evidence.extend_from_slice(&left_bytes[offset..end]);
                    assert_eq!(
                        left_bytes[offset..end],
                        right_bytes[offset..end],
                        "norm intervention changed KV layer={layer} head={head} step={step}"
                    );
                }
            }
        }
        let mut best = (f32::NEG_INFINITY, 0_u32);
        for (index, bytes) in expected.chunks_exact(2).enumerate() {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
            if value > best.0 {
                best = (value, index as u32);
            }
        }
        if layer_audit {
            let output_token = candidate.audit_c07_output(&rows, &mut stream)?;
            assert_eq!(output_token, best.1);
            println!(
                "G02F canonical={canonical} step={step} token={output_token} actual packed gather/argmax/D2H completion passed"
            );
            let selected = candidate.audit_c07_greedy(&mut stream)?;
            assert_eq!(
                selected, best.1,
                "GPU graph token must match independent CPU baseline argmax"
            );
            assert_eq!(stats, context.allocation_stats()?);
            assert!(!candidate.owner.poisoned);
            println!(
                "G02E canonical={canonical} step={step} token={selected} graph completion, D2H status and restore passed"
            );
        }
        token = best.1;
        predictions.push(token);
    }
    {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(&evidence)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        println!(
            "G03_BINDING_PARITY checkpoint={checkpoint_env} layers={} bytes={} sha256={digest} tokens={predictions:?}",
            candidate.owner.forward.plan.layers().len(),
            evidence.len()
        );
    }
    eprintln!(
        "G02C/P2 layer_audit={layer_audit} canonical={canonical}; continuation={predictions:?}"
    );
    if layer_audit {
        let tokens = [504_u32];
        let valid = [1_u16];
        let stale = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Prefill,
            &tokens,
            1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &[0], &valid, 1),
            Some(0),
        )];
        let stats = context.allocation_stats()?;
        // Corrupt the baseline's actual device row map; the host mirror stays 0.
        let packed = baseline.owner.metadata.pack(&stale)?;
        let layout =
            crate::llama::executor::metadata::PackedIterationLayout::for_batch(&packed, 1)?;
        if let crate::llama::executor::buffers::BatchDeviceInput::IterationBatch { slab } =
            &mut baseline.owner.device_input
        {
            slab.upload_from_slice(
                layout.output_token_indices.offset as u64,
                &1_u32.to_ne_bytes(),
                &mut baseline.owner.forward.io_staging,
                &mut stream,
            )?;
        } else {
            panic!("packed owner required");
        }
        let error = baseline
            .audit_c07_output(&stale, &mut stream)
            .expect_err("stale output map must fail");
        assert!(matches!(
            error,
            super::LlamaBatchExecutorError::InvalidConfiguration {
                field: "C07 output audit",
                reason: "stale device output map"
            }
        ));
        assert!(baseline.owner.poisoned);
        assert_eq!(context.allocation_stats()?, stats);
        assert!(candidate.audit_c07_rope(&stale, &mut stream).is_err());
        assert!(candidate.owner.poisoned);
        assert_eq!(context.allocation_stats()?, stats);
    }
    baseline.close()?;
    candidate.close()?;
    embedding_report.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
