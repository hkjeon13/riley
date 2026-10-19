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
    let completion = std::env::var("RILEY_DIVERGENCE_COMPLETION")
        .unwrap_or_else(|_| "per-operation".to_owned());
    assert!(matches!(completion.as_str(), "per-operation" | "iteration-batch"));
    let copies:usize=std::env::var("RILEY_TEACHER_COPIES").unwrap_or_else(|_|"1".into()).parse()?;
    assert!(matches!(copies,1|7|8));
    let rt = CudaRuntime::initialize()?;
    assert_eq!(rt.device_count(), 1);
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut reports = Vec::new();
    for model_case in fixture["models"].as_array().ok_or("models missing")? {
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
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(copies, 512*copies, 64*copies, copies, 64*copies)?,
            PreparedLlamaForwardConfig::default(),
        )
        .with_active_row_buckets()
        .with_serving_sequence_token_bound(1024)
        .with_owned_hf_eager_bf16();
        let config = if completion == "iteration-batch" {
            config.with_iteration_batch_completion()
        } else {
            config
        };
        let config = if std::env::var("RILEY_OWNED_GRAPH").as_deref() == Ok("segments-v1") {
            config.with_owned_decode_graph()
        } else { config };
        let config = match std::env::var("RILEY_OWNED_BATCH").as_deref() {
            Ok("grouped-v1") => config.with_owned_decode_batch(),
            Ok("lt-mlp-v2") => config.with_owned_mlp_decode_batch(),
            _ => config,
        };
        let mut owner = PreparedLlamaBatchExecutor::prepare(&model, &c, &mut stream, config)?;
        let stable = c.allocation_stats()?;
        let row_bytes = model.spec().embedding().vocabulary_size() * 2;
        let mut bytes = vec![0; row_bytes*copies];
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
                for step in 0..teacher.len() {
                    let token = [if step == 0 { 0 } else { teacher[step - 1] }];
                    let tokens = if step == 0 {
                        prompt.as_slice()
                    } else {
                        token.as_slice()
                    };
                    let length = prompt.len() + step;
                    let (blocks, valid) = table(length)?;
                    let block_tables=(0..copies).map(|copy| blocks.iter().map(|b| *b+copy as u32*64).collect::<Vec<_>>()).collect::<Vec<_>>();
                    let batch_rows=(0..copies).map(|copy| LlamaBatchRow::new(
                        (i*copies+copy+1) as u64,
                        if step==0 { LlamaBatchRowKind::Prefill } else { LlamaBatchRowKind::Decode },
                        tokens,u32::try_from(length).expect("bounded sequence"),
                        LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION,&block_tables[copy],&valid,u32::try_from(length).expect("bounded sequence")),
                        Some(copy as u32),
                    )).collect::<Vec<_>>();
                    owner.execute(&batch_rows, &mut stream)?;
                    if let Some(expected) = steps
                        .iter()
                        .find(|v| v["step"].as_u64() == Some(step as u64))
                    {
                        owner.download_logits(&mut bytes, &mut stream)?;
                        for row in bytes.chunks_exact(row_bytes) { assert_eq!(row,&bytes[..row_bytes],"replicated request logits differ"); }
                        let bytes=&bytes[..row_bytes];

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
        owner.close()?;
        operation?;
        assert!(c.allocation_stats()?.is_zero());
        reports.push(json!({"tag":model_case["tag"],"cases":cases}));
    }
    stream.close()?;
    c.close()?;
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"teacher-forced numerical diagnostic, not quality acceptance","completion":completion,"copies":copies,"fixture_sha256":sha(&fixture_bytes),"models":reports}),
        )?,
    )?;
    Ok(())
}
