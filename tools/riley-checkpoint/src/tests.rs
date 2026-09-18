use super::*;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);
pub(crate) struct Temp(pub(crate) PathBuf);
impl Temp {
    pub(crate) fn new() -> Self {
        let root = std::env::temp_dir().join(format!("riley-checkpoint-test-{}-{}", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
        create_directory(&root).unwrap();
        Self(root)
    }
}
impl Drop for Temp { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

fn fixture(root: &Path) -> Vec<u8> {
    let config = json!({
        "architectures":["LlamaForCausalLM"], "attention_bias":false,
        "bos_token_id":0,"eos_token_id":0,"hidden_act":"silu","hidden_size":4,
        "intermediate_size":8,"max_position_embeddings":16,"model_type":"llama",
        "num_attention_heads":2,"num_hidden_layers":1,"num_key_value_heads":1,
        "rms_norm_eps":0.00001,"rope_scaling":null,"rope_theta":10000,
        "tie_word_embeddings":true,"torch_dtype":"bfloat16","vocab_size":8
    });
    let tokenizer = json!({
        "version":"1.0","truncation":null,"padding":null,
        "added_tokens":[{"id":0,"content":"<|endoftext|>","single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"Digits","individual_digits":true},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]},
        "post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{"type":"BPE","dropout":null,"unk_token":null,
            "continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,
            "byte_fallback":false,"ignore_merges":false,
            "vocab":{"<|endoftext|>":0,"h":1,"e":2,"l":3,"o":4,"Ġ":5,"1":6,"2":7},"merges":[]}
    });
    let tensors: &[(&str, &[usize])] = &[
        ("model.embed_tokens.weight", &[8,4]),
        ("model.layers.0.input_layernorm.weight", &[4]),
        ("model.layers.0.self_attn.q_proj.weight", &[4,4]),
        ("model.layers.0.self_attn.k_proj.weight", &[2,4]),
        ("model.layers.0.self_attn.v_proj.weight", &[2,4]),
        ("model.layers.0.self_attn.o_proj.weight", &[4,4]),
        ("model.layers.0.post_attention_layernorm.weight", &[4]),
        ("model.layers.0.mlp.gate_proj.weight", &[8,4]),
        ("model.layers.0.mlp.up_proj.weight", &[8,4]),
        ("model.layers.0.mlp.down_proj.weight", &[4,8]),
        ("model.norm.weight", &[4]),
    ];
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let start = data.len();
        data.resize(start + shape.iter().product::<usize>() * 2, 0);
        header.insert((*name).to_owned(), json!({"dtype":"BF16","shape":shape,"data_offsets":[start,data.len()]}));
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut weights = (header.len() as u64).to_le_bytes().to_vec();
    weights.extend(header);
    weights.extend(data);
    let payloads = [
        ("config.json", serde_json::to_vec(&config).unwrap()),
        ("tokenizer.json", serde_json::to_vec(&tokenizer).unwrap()),
        ("model.safetensors", weights),
    ];
    let mut assertions = Vec::new();
    for (name, bytes) in payloads {
        fs::write(root.join(name), &bytes).unwrap();
        assertions.push(json!({"path":name,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    serde_json::to_vec(&json!({"format":"riley-checkpoint-v1","source_model":"fixture/tiny-llama",
        "source_revision":"1111111111111111111111111111111111111111",
        "converter_revision":null,"dtype":"bf16","transforms":[],"files":assertions})).unwrap()
}

#[test]
fn offline_prepare_passes_real_loader_and_copies_not_external_hardlinks() {
    let source = Temp::new();
    let parent = Temp::new();
    let bytes = fixture(&source.0);
    let plan = Plan::from_bytes(bytes.clone()).unwrap();
    let target = parent.0.join("ready");
    prepare(&plan, &source.0, &target).unwrap();
    verify(&target).unwrap();
    assert_eq!(fs::read(target.join(PROVENANCE_FILENAME)).unwrap(), bytes);
    assert!(!target.join(STAGING).exists());
    fs::write(source.0.join("config.json"), b"changed source").unwrap();
    verify(&target).unwrap();
}

#[test]
fn checksum_failure_never_publishes_and_removes_only_owned_destination() {
    let source = Temp::new();
    let parent = Temp::new();
    let plan = Plan::from_bytes(fixture(&source.0)).unwrap();
    let path = source.0.join("config.json");
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(path, bytes).unwrap();
    let target = parent.0.join("ready");
    assert!(prepare(&plan, &source.0, &target).is_err());
    assert!(!target.exists());
    assert!(source.0.join("config.json").exists());
}

#[test]
fn short_file_and_interrupted_source_leave_no_commit_marker() {
    let source = Temp::new();
    let parent = Temp::new();
    let plan = Plan::from_bytes(fixture(&source.0)).unwrap();
    let target = parent.0.join("ready");
    let mut calls = 0;
    assert!(publish_with(&plan, &target, |_, entry| {
        calls += 1;
        if calls == 2 { Err(invalid("injected transport interruption")) }
        else { Ok(source.0.join(entry.path())) }
    }).is_err());
    assert_eq!(calls, 2);
    assert!(!target.exists());
    fs::write(source.0.join("config.json"), b"{}").unwrap();
    assert!(prepare(&plan, &source.0, &target).is_err());
    assert!(!target.exists());
}

#[test]
fn loader_rejects_hash_correct_but_semantically_invalid_model() {
    let source = Temp::new();
    let parent = Temp::new();
    let mut manifest: Value = serde_json::from_slice(&fixture(&source.0)).unwrap();
    let broken = b"{\"model_type\":\"not-supported\"}";
    fs::write(source.0.join("config.json"), broken).unwrap();
    let entry = manifest["files"].as_array_mut().unwrap().iter_mut().find(|v| v["path"] == "config.json").unwrap();
    entry["bytes"] = json!(broken.len());
    entry["sha256"] = json!(format!("{:x}", Sha256::digest(broken)));
    let plan = Plan::from_bytes(serde_json::to_vec(&manifest).unwrap()).unwrap();
    let target = parent.0.join("ready");
    assert!(prepare(&plan, &source.0, &target).is_err());
    assert!(!target.exists());
}

#[test]
fn existing_partial_destination_is_never_overwritten_or_cleaned() {
    let source = Temp::new();
    let existing = Temp::new();
    let plan = Plan::from_bytes(fixture(&source.0)).unwrap();
    fs::write(existing.0.join("sentinel"), b"keep me").unwrap();
    assert!(prepare(&plan, &source.0, &existing.0).is_err());
    assert_eq!(fs::read(existing.0.join("sentinel")).unwrap(), b"keep me");
    assert!(verify(&existing.0).is_err());
}

#[test]
fn simultaneous_publish_has_exactly_one_winner() {
    let source = Temp::new();
    let parent = Temp::new();
    let bytes = fixture(&source.0);
    let target = parent.0.join("ready");
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let run = || {
            let plan = Plan::from_bytes(bytes.clone()).unwrap();
            barrier.wait();
            prepare(&plan, &source.0, &target).is_ok()
        };
        let left = scope.spawn(run);
        let right = scope.spawn(run);
        assert_ne!(left.join().unwrap(), right.join().unwrap());
    });
    verify(&target).unwrap();
}

#[test]
fn manifest_validation_is_strict_and_revision_is_immutable() {
    let source = Temp::new();
    let bytes = fixture(&source.0);
    let original: Value = serde_json::from_slice(&bytes).unwrap();
    for revision in ["main", "v1.0", "abc123", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] {
        let mut value = original.clone();
        value["source_revision"] = json!(revision);
        assert!(Plan::from_bytes(serde_json::to_vec(&value).unwrap()).is_err());
    }
    for path in ["../config.json", "/config.json", "https://evil/file", "riley-checkpoint.json", "config\\evil.json"] {
        let mut value = original.clone();
        value["files"][0]["path"] = json!(path);
        assert!(Plan::from_bytes(serde_json::to_vec(&value).unwrap()).is_err());
    }
    let mut unknown = original;
    unknown["extra"] = json!(true);
    assert!(Plan::from_bytes(serde_json::to_vec(&unknown).unwrap()).is_err());
    let duplicate = String::from_utf8(bytes).unwrap().replacen('{', "{\"format\":\"riley-checkpoint-v1\",", 1);
    assert!(Plan::from_bytes(duplicate.into_bytes()).is_err());
}

#[cfg(unix)]
#[test]
fn symlink_files_source_ancestors_and_destinations_are_rejected() {
    use std::os::unix::fs::symlink;
    let source = Temp::new();
    let parent = Temp::new();
    let plan = Plan::from_bytes(fixture(&source.0)).unwrap();
    let alias = parent.0.join("source-alias");
    symlink(&source.0, &alias).unwrap();
    assert!(prepare(&plan, &alias, &parent.0.join("ready")).is_err());
    let config = source.0.join("config.json");
    fs::rename(&config, source.0.join("real.json")).unwrap();
    symlink("real.json", &config).unwrap();
    assert!(prepare(&plan, &source.0, &parent.0.join("ready")).is_err());
    assert!(!parent.0.join("ready").exists());
    symlink(&source.0, parent.0.join("ready")).unwrap();
    assert!(prepare(&plan, &source.0, &parent.0.join("ready")).is_err());
    assert!(fs::symlink_metadata(parent.0.join("ready")).unwrap().file_type().is_symlink());
}
