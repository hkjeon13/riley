//! Source-bound teacher-forced diagnostics; numerical metrics do not imply acceptance.
#![cfg(feature = "cuda-cublas-gemm-probe")]
use riley_cuda::CudaRuntime;
use riley_model::{LoadLimits, LoadedModel};
use riley_runtime::{
    llama::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
    },
    paged_kv::{BLOCK_TABLE_V1_VERSION, KV_BLOCK_SIZE},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};
type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn ids(v: &Value) -> Result<Vec<u32>> {
    v.as_array()
        .ok_or("IDs missing")?
        .iter()
        .map(|v| Ok(u32::try_from(v.as_u64().ok_or("ID not unsigned")?)?))
        .collect()
}
fn leaf(s: &str) -> Result<&str> {
    if s.is_empty()
        || s.len() > 128
        || s.contains("..")
        || !s
            .bytes()
            .all(|v| v.is_ascii_alphanumeric() || b"-_.".contains(&v))
    {
        Err("invalid diagnostic leaf".into())
    } else {
        Ok(s)
    }
}
fn values(bytes: &[u8]) -> Result<Vec<f64>> {
    if bytes.len() % 2 != 0 {
        return Err("unaligned BF16 bytes".into());
    }
    bytes
        .chunks_exact(2)
        .map(|v| {
            let f = f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16);
            if f.is_finite() {
                Ok(f64::from(f))
            } else {
                Err("nonfinite logits".into())
            }
        })
        .collect()
}
fn metrics(a: &[u8], b: &[u8]) -> Result<Value> {
    if a.len() != b.len() {
        return Err("logit row lengths differ".into());
    }
    let (a, b) = (values(a)?, values(b)?);
    let top = |v: &[f64]| {
        let mut best = 0;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best
    };
    let (at, bt) = (top(&a), top(&b));
    let (mut dot, mut aa, mut bb, mut mae, mut maximum) = (0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
        mae += (x - y).abs();
        maximum = maximum.max((x - y).abs());
    }
    let lse = |v: &[f64]| {
        let max = v[top(v)];
        max + v.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(&a), lse(&b));
    let kl = b
        .iter()
        .zip(&a)
        .map(|(&y, &x)| (y - lb).exp() * ((y - lb) - (x - la)))
        .sum::<f64>();
    Ok(
        json!({"native_top1":at,"hf_top1":bt,"top1_equal":at==bt,"cosine":dot/(aa*bb).sqrt(),"mean_abs":mae/a.len() as f64,"max_abs":maximum,"hf_to_native_kl":kl,"native_choice_hf_rank":1+b.iter().filter(|&&x|x>b[at]).count(),"hf_margin_over_native_choice":b[bt]-b[at],"hf_choice_native_logprob":a[bt]-la,"hf_choice_hf_logprob":b[bt]-lb,"verdict":"diagnostic only; no fitted tolerance or quality acceptance"}),
    )
}
fn table(length: usize) -> Result<(Vec<u32>, Vec<u16>)> {
    let n = length.div_ceil(KV_BLOCK_SIZE);
    Ok((
        (0..n)
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()?,
        (0..n)
            .map(|i| u16::try_from((length - i * KV_BLOCK_SIZE).min(KV_BLOCK_SIZE)))
            .collect::<std::result::Result<_, _>>()?,
    ))
}
#[test]
#[ignore = "GPU2/3 only; checksum-bound HF diagnostic fixture required"]
fn owned_batch_teacher_forced_logits_are_finite_and_reported() -> Result {
    let root = std::path::PathBuf::from(std::env::var("RILEY_DIVERGENCE_FIXTURE_ROOT")?);
    let output = std::path::PathBuf::from(std::env::var("RILEY_DIVERGENCE_NATIVE_OUTPUT")?);
    fs::create_dir(&output)?;
    assert!(root.is_absolute() && output.is_absolute());
    let fixture_bytes = fs::read(root.join("fixture.json"))?;
    let fixture: Value = serde_json::from_slice(&fixture_bytes)?;
    let rt = CudaRuntime::initialize()?;
    assert_eq!(rt.device_count(), 1);
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut reports = Vec::new();
    let only_model = std::env::var("RILEY_DIVERGENCE_ONLY_MODEL").ok();
    let graph_mode = std::env::var("RILEY_DIVERGENCE_MLP_GRAPH").as_deref() == Ok("1");
    let profile = std::env::var("RILEY_DIVERGENCE_PROFILE").unwrap_or_else(|_| "owned".into());
    assert!(
        matches!(profile.as_str(), "owned" | "fast-merged-v1"),
        "unknown numerical diagnostic profile"
    );
    assert!(
        !graph_mode || profile == "owned",
        "MLP graph diagnostic requires the owned profile"
    );
    for model_case in fixture["models"].as_array().ok_or("models missing")? {
        if only_model
            .as_deref()
            .is_some_and(|tag| model_case["tag"].as_str() != Some(tag))
        {
            continue;
        }
        let path = Path::new(model_case["path"].as_str().ok_or("model path missing")?);
        assert_eq!(
            sha(&fs::read(path.join("riley-checkpoint.json"))?),
            model_case["checkpoint_manifest_sha256"]
                .as_str()
                .ok_or("manifest hash missing")?
        );
        let model = LoadedModel::load(
            path,
            LoadLimits::default().with_weight_byte_limits(32 << 30, 32 << 30)?,
        )?;
        let forward = if profile == "fast-merged-v1" {
            PreparedLlamaForwardConfig::default()
                .with_merged_qkv_projection()
                .with_merged_gate_up_projection()
                .with_first_heuristic_gemm_reductions()
                .with_bit_equivalent_gemm_selection()
        } else {
            PreparedLlamaForwardConfig::default()
        };
        let input_bound = if profile == "fast-merged-v1" {
            256
        } else {
            512
        };
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, input_bound, 64, 1, 64)?,
            forward,
        )
        .with_active_row_buckets()
        .with_serving_sequence_token_bound(1024);
        let config = if profile == "fast-merged-v1" {
            config
                .with_iteration_batch_completion()
                .with_fused_residual_norm()
                .with_native_bf16_paged_split_gqa_d128_two_stage()
                .with_d128_global_merge()
                .with_d128_tensor_core_producer()
                .with_fused_silu_multiply()
                .with_fused_qkv_rope()
                .with_fused_kv_write()
                .with_fused_mlp_residual_norm()
                .with_output_row_lm_head()
                .with_decode_step_capture()
        } else {
            config.with_owned_hf_eager_bf16()
        };
        let config = if graph_mode {
            config
                .with_iteration_batch_completion()
                .with_fused_silu_multiply()
                .with_owned_mlp_graphs()
        } else {
            config
        };
        let mut owner = PreparedLlamaBatchExecutor::prepare(&model, &c, &mut stream, config)?;
        let stable = c.allocation_stats()?;
        let mut bytes = vec![0; model.spec().embedding().vocabulary_size() * 2];
        let mut cases = Vec::new();
        let operation = (|| -> Result {
            for (i, case) in model_case["cases"]
                .as_array()
                .ok_or("cases missing")?
                .iter()
                .enumerate()
            {
                let prompt = ids(&case["prompt_ids"])?;
                let teacher = ids(&case["teacher_ids"])?;
                let steps = case["steps"].as_array().ok_or("steps missing")?;
                assert!(
                    !prompt.is_empty()
                        && prompt.len() <= 512
                        && prompt.len() + teacher.len() <= 1024
                        && !teacher.is_empty()
                );
                let mut rows = Vec::new();
                // Match the serving path's bounded prompt chunks. Cache positions
                // describe the whole prefix, while each call submits at most the
                // native attention backend's 256-row budget.
                let tail_start = (prompt.len() - 1) / input_bound * input_bound;
                for start in (0..tail_start).step_by(input_bound) {
                    let length = start + input_bound;
                    let (blocks, valid) = table(length)?;
                    let row = LlamaBatchRow::new(
                        i as u64 + 1,
                        LlamaBatchRowKind::Prefill,
                        &prompt[start..length],
                        u32::try_from(length)?,
                        LlamaBatchBlockTable::new(
                            BLOCK_TABLE_V1_VERSION,
                            &blocks,
                            &valid,
                            u32::try_from(length)?,
                        ),
                        Some(0),
                    );
                    owner.execute(&[row], &mut stream)?;
                }
                for step in 0..teacher.len() {
                    let token = [if step == 0 { 0 } else { teacher[step - 1] }];
                    let tokens = if step == 0 {
                        &prompt[tail_start..]
                    } else {
                        token.as_slice()
                    };
                    let length = prompt.len() + step;
                    let (blocks, valid) = table(length)?;
                    let row = LlamaBatchRow::new(
                        i as u64 + 1,
                        if step == 0 {
                            LlamaBatchRowKind::Prefill
                        } else {
                            LlamaBatchRowKind::Decode
                        },
                        tokens,
                        u32::try_from(length)?,
                        LlamaBatchBlockTable::new(
                            BLOCK_TABLE_V1_VERSION,
                            &blocks,
                            &valid,
                            u32::try_from(length)?,
                        ),
                        Some(0),
                    );
                    owner.execute(&[row], &mut stream)?;
                    if let Some(expected) = steps
                        .iter()
                        .find(|v| v["step"].as_u64() == Some(step as u64))
                    {
                        owner.download_logits(&mut bytes, &mut stream)?;
                        let filename = leaf(expected["file"].as_str().ok_or("HF file missing")?)?;
                        let hf = fs::read(root.join(filename))?;
                        let contract = &fixture["files"][filename];
                        assert_eq!(Some(hf.len() as u64), contract["bytes"].as_u64());
                        assert_eq!(
                            sha(&hf),
                            contract["sha256"].as_str().ok_or("HF SHA missing")?
                        );
                        let file = format!(
                            "{}-{}-s{step}.native.bf16",
                            leaf(model_case["tag"].as_str().ok_or("tag missing")?)?,
                            leaf(case["id"].as_str().ok_or("case ID missing")?)?
                        );
                        fs::write(output.join(&file), &bytes)?;
                        let mut m = metrics(&bytes, &hf)?;
                        m["step"] = json!(step);
                        m["file"] = json!(file);
                        m["sha256"] = json!(sha(&bytes));
                        rows.push(m);
                    }
                    assert_eq!(
                        c.allocation_stats()?,
                        stable,
                        "teacher execution allocated CUDA buffers"
                    );
                }
                println!(
                    "TEACHER {} {} {:?}",
                    model_case["tag"],
                    case["id"],
                    rows.last()
                );
                cases.push(json!({"id":case["id"],"divergence":case["divergence"],"rows":rows}));
            }
            Ok(())
        })();
        let graph_stats = owner.owned_mlp_graph_stats();
        owner.close()?;
        operation?;
        if graph_mode {
            let stats = graph_stats.ok_or("graph owner missing")?;
            assert!(
                stats.captures > 0 && stats.hits > 0,
                "managed model graph must actually capture and replay"
            );
        }
        assert!(c.allocation_stats()?.is_zero());
        reports.push(json!({"tag":model_case["tag"],"cases":cases,"mlp_graph":graph_stats.map(|s|json!({"captures":s.captures,"hits":s.hits,"fallbacks":s.eager_fallbacks}))}));
    }
    stream.close()?;
    c.close()?;
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"teacher-forced numerical diagnostic, not quality acceptance","fixture_sha256":sha(&fixture_bytes),"profile":profile,"models":reports}),
        )?,
    )?;
    Ok(())
}
