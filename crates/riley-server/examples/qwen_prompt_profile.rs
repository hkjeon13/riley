//! Render and encode Qwen or Llama 3 chat requests from JSON lines for offline prefix audits.
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use riley_model::{
    ChatTemplateSource, EncodeOptions, LoadLimits, Qwen2Tokenizer, SmolLm2Tokenizer, Tokenizer,
};
use riley_server::chat::{ChatTemplate, normalize_chat_request};
use riley_server::domain::RequestLimits;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected model directory")?,
    );
    let mut source = ChatTemplateSource::from_tokenizer_config(&std::fs::read(
        dir.join("tokenizer_config.json"),
    )?)?;
    let standalone = dir.join("chat_template.jinja");
    if standalone.try_exists()? {
        source.template = std::fs::read_to_string(standalone)?;
    }
    let template = ChatTemplate::compile(&source)?;
    let bytes = std::fs::read(dir.join("tokenizer.json"))?;
    let mut llama = false;
    let mut interactive = false;
    for arg in std::env::args().skip(2) {
        match arg.as_str() {
            "--llama3" => llama = true,
            "--interactive" => interactive = true,
            _ => {
                return Err(
                    "expected tokenizer directory and optional --llama3 or --interactive".into(),
                );
            }
        }
    }
    let tokenizer: Box<dyn Tokenizer> = if llama {
        Box::new(SmolLm2Tokenizer::from_llama3_json_slice_with_limits(
            &bytes,
            LoadLimits::default(),
        )?)
    } else {
        Box::new(Qwen2Tokenizer::from_json_slice(&bytes)?)
    };
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    if interactive {
        writeln!(stdout, "{{\"ready\":true}}")?;
        stdout.flush()?;
    }
    for line in io::stdin().lock().lines() {
        let line = line?;
        let row = match profile(&line, &template, tokenizer.as_ref()) {
            Ok(value) => value,
            Err(error) if interactive => json!({"error": error.to_string()}),
            Err(error) => return Err(error),
        };
        serde_json::to_writer(&mut stdout, &row)?;
        writeln!(stdout)?;
        if interactive {
            stdout.flush()?;
        }
    }
    stdout.flush()?;
    Ok(())
}

fn profile(
    line: &str,
    template: &ChatTemplate,
    tokenizer: &dyn Tokenizer,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let incoming = serde_json::from_str(line)?;
    let started = Instant::now();
    let request = normalize_chat_request(incoming, template, RequestLimits::default())
        .map_err(|err| format!("invalid request: {err:?}"))?;
    let normalized = Instant::now();
    let ids = tokenizer.encode(
        &request.prompt,
        EncodeOptions {
            add_special_tokens: request.add_special_tokens,
        },
    )?;
    let encoded = Instant::now();
    Ok(json!({
        "normalize_ns": (normalized-started).as_nanos(),
        "encode_ns": (encoded-normalized).as_nanos(),
        "ids": ids,
    }))
}
