//! Remote-only HTTP gate for the HF-exact Qwen P2048 cache-on numerics profile.
//!
//! A real loopback HTTP server, engine worker and backend serve the pinned
//! `qwen3b-c8-p2048-o128` workload. Every generated token and every published
//! BF16 logit row must equal the pinned Hugging Face eager cache-on reference
//! byte for byte. The same server then proves cancellation, re-request and
//! shutdown reclamation. Nothing here measures serving performance.

#![cfg(feature = "hf-exact-probe")]
#![allow(clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riley_model::{DecodeOptions, EncodeOptions, LoadLimits, LoadedModel};
use riley_server::domain::{ModelMetadata, RequestLimits};
use riley_server::engine::{
    EngineConfig, HF_EXACT_MAX_OUTPUT_TOKENS, HF_EXACT_PROFILE_ID, HF_EXACT_PROMPT_TOKENS,
    HfExactEngineResources, InferenceEngine, LogitObservationSink,
};
use riley_server::http::HttpLimits;
use riley_server::service::{CompletionBackend, ServerConfig, start_server};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const MODEL_ID: &str = "qwen3b-hf-exact-gate";
const VOCABULARY_SIZE: usize = 151_936;
const ROW_BYTES: usize = VOCABULARY_SIZE * 2;
const MAX_WEIGHT_BYTES: u64 = 8 << 30;
const CLIENT_IO_TIMEOUT: Duration = Duration::from_secs(300);
const STATE_TIMEOUT: Duration = Duration::from_secs(120);
// Pinned Hugging Face eager references, as bound by the native exact gates.
const TEACHER_ORACLE_SHA256: &str =
    "35d5d9b153d73b4badc67e5eedf2e8226ca2b55c7e30eb0970cfb741031edd35";
const CACHE_ON_SIDECAR_SHA256: &str =
    "d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39";

fn required_path(variable: &str) -> TestResult<PathBuf> {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{variable} must be set").into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn u32_array(value: &Value, name: &str) -> TestResult<Vec<u32>> {
    value
        .as_array()
        .ok_or_else(|| format!("{name} is not an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{name} holds a non-token value").into())
        })
        .collect()
}

#[derive(Default)]
struct RowSink {
    rows: Mutex<BTreeMap<String, Vec<Vec<u8>>>>,
}

impl LogitObservationSink for RowSink {
    fn observe(
        &self,
        request_id: &str,
        generated_index: usize,
        logits_bf16_native: &[u8],
    ) -> Result<(), String> {
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| "row sink poisoned".to_owned())?;
        let request = rows.entry(request_id.to_owned()).or_default();
        if request.len() != generated_index {
            return Err(format!(
                "row {generated_index} observed after {} rows",
                request.len()
            ));
        }
        request.push(logits_bf16_native.to_vec());
        Ok(())
    }
}

struct Completion {
    status: u16,
    id: String,
    token_ids: Vec<u32>,
    prompt_token_ids: Option<Vec<u32>>,
    text: String,
    finish_reason: Option<String>,
    usage: Option<Value>,
}

fn completion_body(prompt: &str, max_tokens: usize, stream: bool, temperature: f64) -> Vec<u8> {
    let mut body = json!({
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": stream,
        "return_token_ids": true
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    serde_json::to_vec(&body).expect("request body serializes")
}

fn open_completion(address: SocketAddr, body: &[u8]) -> io::Result<TcpStream> {
    let mut request = Vec::with_capacity(body.len() + 256);
    write!(
        request,
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    request.extend_from_slice(body);
    let mut stream = TcpStream::connect(address)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(CLIENT_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(CLIENT_IO_TIMEOUT))?;
    stream.write_all(&request)?;
    stream.flush()?;
    Ok(stream)
}

fn split_response(bytes: &[u8]) -> TestResult<(u16, &[u8])> {
    let header_offset = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or("HTTP response omitted the header terminator")?;
    let status = std::str::from_utf8(&bytes[..header_offset])?
        .lines()
        .next()
        .and_then(|line| line.split_ascii_whitespace().nth(1))
        .ok_or("HTTP response omitted a status")?
        .parse::<u16>()?;
    Ok((status, &bytes[header_offset + 4..]))
}

fn send_completion(address: SocketAddr, body: &[u8], stream: bool) -> TestResult<Completion> {
    let mut connection = open_completion(address, body)?;
    let mut bytes = Vec::new();
    connection.read_to_end(&mut bytes)?;
    let (status, payload) = split_response(&bytes)?;
    let mut completion = Completion {
        status,
        id: String::new(),
        token_ids: Vec::new(),
        prompt_token_ids: None,
        text: String::new(),
        finish_reason: None,
        usage: None,
    };
    if status != 200 {
        return Ok(completion);
    }
    let mut absorb = |value: &Value| -> TestResult {
        if let Some(id) = value["id"].as_str() {
            if !completion.id.is_empty() && completion.id != id {
                return Err("response identifier changed mid-stream".into());
            }
            completion.id = id.to_owned();
        }
        for choice in value["choices"].as_array().into_iter().flatten() {
            completion
                .text
                .push_str(choice["text"].as_str().unwrap_or(""));
            if !choice["token_ids"].is_null() {
                completion
                    .token_ids
                    .extend(u32_array(&choice["token_ids"], "token_ids")?);
            }
            if !choice["prompt_token_ids"].is_null() {
                completion.prompt_token_ids =
                    Some(u32_array(&choice["prompt_token_ids"], "prompt_token_ids")?);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                completion.finish_reason = Some(reason.to_owned());
            }
        }
        if !value["usage"].is_null() {
            completion.usage = Some(value["usage"].clone());
        }
        Ok(())
    };
    if stream {
        let mut done = false;
        for line in std::str::from_utf8(payload)?.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                done = true;
                continue;
            }
            if done {
                return Err("SSE event followed [DONE]".into());
            }
            absorb(&serde_json::from_str(data)?)?;
        }
        if !done {
            return Err("SSE stream omitted [DONE]".into());
        }
    } else {
        absorb(&serde_json::from_slice(payload)?)?;
    }
    Ok(completion)
}

fn wait_until(mut predicate: impl FnMut() -> bool) -> io::Result<()> {
    let deadline = Instant::now() + STATE_TIMEOUT;
    while Instant::now() < deadline {
        if predicate() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(5));
    }
    Err(io::Error::other(
        "timed out waiting for bounded server state",
    ))
}

fn read_http_head(reader: &mut BufReader<TcpStream>) -> io::Result<u16> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let status = line
        .split_ascii_whitespace()
        .nth(1)
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| io::Error::other("streaming response status line was invalid"))?;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::other("streaming response headers ended early"));
        }
        if line == "\r\n" {
            return Ok(status);
        }
    }
}

fn read_sse_payload(reader: &mut BufReader<TcpStream>) -> io::Result<String> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::other("SSE stream ended early"));
        }
        if let Some(payload) = line.strip_prefix("data: ") {
            return Ok(payload.trim_end_matches(['\r', '\n']).to_owned());
        }
    }
}

/// Compares every observed row of one request with the reference prefix.
fn exact_rows(sink: &RowSink, request_id: &str, expected: &[u8], rows: usize) -> TestResult<Value> {
    let observed = sink
        .rows
        .lock()
        .map_err(|_| "row sink poisoned")?
        .get(request_id)
        .cloned()
        .ok_or("request published no logit rows")?;
    if observed.len() != rows {
        return Err(format!(
            "request published {} rows instead of {rows}",
            observed.len()
        )
        .into());
    }
    let mut records = Vec::with_capacity(rows);
    let mut first_non_exact = None;
    for (row, actual) in observed.iter().enumerate() {
        let reference = &expected[row * ROW_BYTES..(row + 1) * ROW_BYTES];
        let exact = actual.as_slice() == reference;
        if !exact && first_non_exact.is_none() {
            first_non_exact = Some(row);
        }
        records.push(json!({
            "row": row,
            "bf16_exact": exact,
            "actual_sha256": sha256_hex(actual),
            "expected_sha256": sha256_hex(reference),
        }));
    }
    Ok(json!({"rows": records, "first_non_exact_row": first_non_exact}))
}

#[test]
#[ignore = "requires the pinned Qwen2.5-3B checkpoint, HF references and CUDA"]
fn qwen3b_http_hf_exact_c1_tokens_logits_and_lifecycle() -> TestResult {
    let workload: Value =
        serde_json::from_slice(&fs::read(required_path("RILEY_QWEN_SERVING_WORKLOAD")?)?)?;
    let prompt = workload["prompt"]
        .as_str()
        .ok_or("workload omits its prompt")?
        .to_owned();
    let prompt_token_ids = u32_array(&workload["prompt_token_ids"], "prompt_token_ids")?;
    let output_token_ids = u32_array(&workload["output_token_ids"], "output_token_ids")?;
    if workload["case"] != "qwen3b-c8-p2048-o128"
        || workload["finish_reason"] != "length"
        || workload["sampling"]["temperature"] != 0.0
        || workload["sampling"]["top_p"] != 1.0
        || prompt_token_ids.len() != HF_EXACT_PROMPT_TOKENS
        || output_token_ids.len() != HF_EXACT_MAX_OUTPUT_TOKENS
    {
        return Err("workload differs from the qwen3b-c8-p2048-o128 contract".into());
    }
    let oracle_bytes = fs::read(required_path("RILEY_QWEN_HF_TEACHER_FORCED_ORACLE")?)?;
    if sha256_hex(&oracle_bytes) != TEACHER_ORACLE_SHA256 {
        return Err("teacher-forced oracle differs from the pinned reference".into());
    }
    let oracle: Value = serde_json::from_slice(&oracle_bytes)?;
    let teacher_token_ids = u32_array(
        &oracle["generation"]["teacher_token_ids"],
        "teacher_token_ids",
    )?;
    if teacher_token_ids.len() != HF_EXACT_MAX_OUTPUT_TOKENS {
        return Err("HF teacher stream is not 128 tokens".into());
    }
    // The workload file also records an output stream from another producer.
    // The Hugging Face reference is the teacher stream; the difference is
    // reported, not reconciled.
    let workload_output_matches_hf = output_token_ids == teacher_token_ids;
    let sidecar = fs::read(required_path("RILEY_QWEN_HF_CACHE_ON_SIDECAR")?)?;
    if sha256_hex(&sidecar) != CACHE_ON_SIDECAR_SHA256 {
        return Err("cache-on logits sidecar differs from the pinned reference".into());
    }
    let header_length = usize::try_from(u64::from_le_bytes(sidecar[..8].try_into()?))?;
    let expected = &sidecar[8 + header_length..];
    if expected.len() != HF_EXACT_MAX_OUTPUT_TOKENS * ROW_BYTES {
        return Err("cache-on logits sidecar extent differs".into());
    }

    let limits =
        LoadLimits::default().with_weight_byte_limits(MAX_WEIGHT_BYTES, MAX_WEIGHT_BYTES)?;
    let model = LoadedModel::load(&required_path("RILEY_QWEN3B_CHECKPOINT")?, limits)?;
    let encode = |text: &str| {
        model.tokenizer().encode(
            text,
            EncodeOptions {
                add_special_tokens: true,
            },
        )
    };
    if encode(&prompt)? != prompt_token_ids {
        return Err("server tokenization of the workload prompt differs from its token IDs".into());
    }
    // Streamed text must equal this tokenizer's decoding of the HF tokens.
    let output_text = model.tokenizer().decode(
        &teacher_token_ids,
        DecodeOptions {
            skip_special_tokens: true,
        },
    )?;
    let short_prompt = prompt
        .strip_suffix(" word")
        .ok_or("workload prompt has an unexpected shape")?
        .to_owned();
    if encode(&short_prompt)?.len() != HF_EXACT_PROMPT_TOKENS - 1 {
        return Err("shortened prompt is not one token shorter".into());
    }

    let sink = Arc::new(RowSink::default());
    let metadata = ModelMetadata {
        model_id: MODEL_ID.to_owned(),
        created_unix_seconds: 0,
        owned_by: "riley".to_owned(),
        context_window_tokens: HF_EXACT_PROMPT_TOKENS + HF_EXACT_MAX_OUTPUT_TOKENS,
        max_output_tokens: HF_EXACT_MAX_OUTPUT_TOKENS,
    };
    let resources = HfExactEngineResources::prepare(metadata, model, 0, 2)?
        .with_logit_observation(sink.clone());
    let engine = Arc::new(InferenceEngine::start_hf_exact(
        resources,
        EngineConfig {
            command_queue_capacity: 4,
            event_channel_capacity: 32,
            max_inflight_requests: 3,
            admission_timeout: Duration::from_secs(30),
            idle_poll_interval: Duration::from_millis(1),
            publish_after_submit: false,
        },
    )?);
    let backend: Arc<dyn CompletionBackend> = engine.clone();
    let server = start_server(
        ServerConfig {
            bind_address: SocketAddr::from(([127, 0, 0, 1], 0)),
            worker_threads: 6,
            connection_queue_capacity: 8,
            read_timeout: Duration::from_secs(10),
            framing_timeout: Duration::from_secs(15),
            write_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(300),
            shutdown_grace: Duration::from_secs(60),
            http_limits: HttpLimits::default(),
            request_limits: RequestLimits {
                max_output_tokens: HF_EXACT_MAX_OUTPUT_TOKENS,
                ..RequestLimits::default()
            },
            maximum_non_streaming_bytes: 1024 * 1024,
            observation_capacity: 32,
        },
        backend,
    )?;
    let address = server.local_address();
    let drained = || {
        wait_until(|| {
            let status = engine.status();
            status.active_requests == 0 && status.waiting_requests == 0
        })
    };

    // Requests outside the qualified profile are refused, not approximated.
    let wrong_length = send_completion(
        address,
        &completion_body(&short_prompt, 8, false, 0.0),
        false,
    )?;
    let wrong_sampling = send_completion(address, &completion_body(&prompt, 8, false, 0.7), false)?;
    if wrong_length.status != 400 || wrong_sampling.status != 400 {
        return Err(format!(
            "out-of-profile requests returned HTTP {} and {} instead of 400",
            wrong_length.status, wrong_sampling.status
        )
        .into());
    }
    drained()?;

    // 1. Streaming contract request: tokens and all 128 logit rows.
    let full = HF_EXACT_MAX_OUTPUT_TOKENS;
    let streaming = send_completion(address, &completion_body(&prompt, full, true, 0.0), true)?;
    if streaming.status != 200 {
        return Err(format!("streaming request returned HTTP {}", streaming.status).into());
    }
    let streaming_rows = exact_rows(&sink, &streaming.id, expected, full)?;
    drained()?;

    // 2. Disconnect while the single-pass prefill is the active step.
    let prefill_client = open_completion(address, &completion_body(&prompt, full, false, 0.0))?;
    wait_until(|| engine.status().active_requests > 0)?;
    prefill_client.shutdown(Shutdown::Both)?;
    drop(prefill_client);
    drained()?;

    // 3. Disconnect after three streamed tokens.
    let decode_client = open_completion(address, &completion_body(&prompt, full, true, 0.0))?;
    let mut reader = BufReader::new(decode_client);
    if read_http_head(&mut reader)? != 200 {
        return Err("decode-cancel stream did not return HTTP 200".into());
    }
    let mut cancelled_prefix = Vec::new();
    while cancelled_prefix.len() < 3 {
        let payload = read_sse_payload(&mut reader)?;
        if payload == "[DONE]" {
            return Err("decode-cancel stream finished before cancellation".into());
        }
        let chunk: Value = serde_json::from_str(&payload)?;
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            if !choice["token_ids"].is_null() {
                cancelled_prefix.extend(u32_array(&choice["token_ids"], "token_ids")?);
            }
        }
    }
    reader.get_mut().shutdown(Shutdown::Both)?;
    drop(reader);
    drained()?;
    let cancelled_prefix_exact =
        cancelled_prefix.as_slice() == &teacher_token_ids[..cancelled_prefix.len()];

    // 4. Re-request after both cancellations reuses the same KV owner.
    let rerequest = send_completion(address, &completion_body(&prompt, full, false, 0.0), false)?;
    if rerequest.status != 200 {
        return Err(format!("re-request returned HTTP {}", rerequest.status).into());
    }
    let rerequest_rows = exact_rows(&sink, &rerequest.id, expected, full)?;
    drained()?;

    // 5. Two queued requests are served in turn on the single active slot.
    let queued = (0..2)
        .map(|_| {
            let body = completion_body(&prompt, 8, false, 0.0);
            thread::spawn(move || send_completion(address, &body, false).map_err(|e| e.to_string()))
        })
        .collect::<Vec<_>>();
    let mut queued_exact = true;
    for client in queued {
        let completion = client.join().map_err(|_| "queued client panicked")??;
        queued_exact &= completion.status == 200
            && completion.token_ids == teacher_token_ids[..8]
            && exact_rows(&sink, &completion.id, expected, 8)?["first_non_exact_row"].is_null();
    }
    drained()?;

    // 6. Shutdown with an active request, then verify reclamation.
    let shutdown_body = completion_body(&prompt, full, false, 0.0);
    let shutdown_client = thread::spawn(move || {
        send_completion(address, &shutdown_body, false).map_err(|e| e.to_string())
    });
    wait_until(|| engine.status().active_requests > 0)?;
    server.shutdown()?;
    let shutdown_status = shutdown_client
        .join()
        .map_err(|_| "shutdown client panicked")??
        .status;
    let status = engine.status();
    let allocation = engine
        .final_metrics_snapshot()
        .and_then(|metrics| metrics.allocation)
        .ok_or("final allocation metrics missing")?;

    let usage_exact = |completion: &Completion| {
        completion.usage.as_ref().is_some_and(|usage| {
            usage["prompt_tokens"] == HF_EXACT_PROMPT_TOKENS
                && usage["completion_tokens"] == HF_EXACT_MAX_OUTPUT_TOKENS
        })
    };
    let request_exact = |completion: &Completion, rows: &Value| {
        completion.token_ids == teacher_token_ids
            && completion.prompt_token_ids.as_deref() == Some(prompt_token_ids.as_slice())
            && completion.text == output_text
            && completion.finish_reason.as_deref() == Some("length")
            && usage_exact(completion)
            && rows["first_non_exact_row"].is_null()
    };
    let streaming_exact = request_exact(&streaming, &streaming_rows);
    let rerequest_exact = request_exact(&rerequest, &rerequest_rows);
    let reclaimed = shutdown_status == 503
        && !status.ready
        && !status.accepting
        && status.active_requests == 0
        && status.waiting_requests == 0
        && allocation.device_live_allocations == 0
        && allocation.device_live_bytes == 0
        && allocation.pinned_host_live_allocations == 0
        && allocation.pinned_host_live_bytes == 0;
    let pass =
        streaming_exact && rerequest_exact && cancelled_prefix_exact && queued_exact && reclaimed;
    let receipt = json!({
        "schema_version": "riley.qwen3b-http-hf-exact-c1.v1",
        "numerics_profile": HF_EXACT_PROFILE_ID,
        "transport": "loopback HTTP/1.1 POST /v1/completions through the service, engine worker and backend",
        "contract": {
            "case": workload["case"],
            "model_revision": workload["model_revision"],
            "prompt_tokens": HF_EXACT_PROMPT_TOKENS,
            "output_tokens": HF_EXACT_MAX_OUTPUT_TOKENS,
            "teacher_oracle_sha256": TEACHER_ORACLE_SHA256,
            "cache_on_sidecar_sha256": CACHE_ON_SIDECAR_SHA256,
            "equality": "token IDs equal and BF16 logit rows byte-identical; no tolerance",
            "workload_recorded_output_matches_hf_teacher": workload_output_matches_hf,
        },
        "out_of_profile_rejections": {
            "prompt_2047_tokens_http_status": wrong_length.status,
            "temperature_0_7_http_status": wrong_sampling.status,
        },
        "streaming_request": {
            "pass": streaming_exact,
            "token_ids": streaming.token_ids,
            "tokens_match_hf": streaming.token_ids == teacher_token_ids,
            "text_matches_decoded_hf_tokens": streaming.text == output_text,
            "finish_reason": streaming.finish_reason,
            "usage": streaming.usage,
            "logits": streaming_rows,
        },
        "cancel_during_prefill": {"drained": true},
        "cancel_after_tokens": {
            "tokens_seen": cancelled_prefix,
            "tokens_match_hf_prefix": cancelled_prefix_exact,
            "drained": true,
        },
        "rerequest_after_cancellations": {
            "pass": rerequest_exact,
            "streaming": false,
            "tokens_match_hf": rerequest.token_ids == teacher_token_ids,
            "finish_reason": rerequest.finish_reason,
            "logits": rerequest_rows,
        },
        "queued_requests": {"count": 2, "output_tokens": 8, "tokens_and_logits_exact": queued_exact},
        "shutdown": {
            "active_request_http_status": shutdown_status,
            "device_live_allocations": allocation.device_live_allocations,
            "device_live_bytes": allocation.device_live_bytes,
            "pinned_host_live_allocations": allocation.pinned_host_live_allocations,
            "pinned_host_live_bytes": allocation.pinned_host_live_bytes,
            "reclaimed": reclaimed,
        },
        "scope_limits": [
            "one prompt, one active request, greedy sampling, at most 128 output tokens",
            "the scheduler/paged-KV route is unchanged and remains not HF-exact",
            "no serving performance is measured",
        ],
        "pass": pass,
    });
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(required_path("RILEY_QWEN_HTTP_HF_EXACT_OUTPUT")?)?;
    output.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    output.sync_all()?;
    if !pass {
        return Err(format!(
            "HTTP HF-exact gate failed: streaming={streaming_exact} rerequest={rerequest_exact} cancelled_prefix={cancelled_prefix_exact} queued={queued_exact} reclaimed={reclaimed}"
        )
        .into());
    }
    Ok(())
}
