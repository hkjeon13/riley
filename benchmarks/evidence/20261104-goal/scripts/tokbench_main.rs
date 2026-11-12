use riley_model::{EncodeOptions, LoadLimits, Qwen2Tokenizer, SmolLm2Tokenizer, Tokenizer};
use std::time::Instant;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1]; let kind = &args[2]; let corpus_dir = &args[3]; let out = &args[4];
    let json = std::fs::read(format!("{dir}/tokenizer.json")).unwrap();
    let tok: Box<dyn Tokenizer> = match kind.as_str() {
        "qwen" => Box::new(Qwen2Tokenizer::from_json_slice(&json).unwrap()),
        "llama" => Box::new(SmolLm2Tokenizer::from_llama3_json_slice_with_limits(&json, LoadLimits::default()).unwrap()),
        _ => Box::new(SmolLm2Tokenizer::from_json_slice(&json).unwrap()),
    };
    let mut files: Vec<_> = std::fs::read_dir(corpus_dir).unwrap().map(|e| e.unwrap().path()).collect(); files.sort();
    let mut dump = String::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let opts = EncodeOptions { add_special_tokens: false };
        let ids = tok.encode(&text, opts).unwrap();
        let reps = (200_000 / ids.len().max(1)).clamp(1, 50);
        let t = Instant::now();
        for _ in 0..reps { std::hint::black_box(tok.encode(&text, opts).unwrap()); }
        let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("{:>28} {:>7} tokens {:>10.1} us {:>6.3} us/token", f.file_name().unwrap().to_string_lossy(), ids.len(), us, us / ids.len() as f64);
        dump.push_str(&format!("{}:{:?}\n", f.file_name().unwrap().to_string_lossy(), ids));
    }
    std::fs::write(out, dump).unwrap();
}
