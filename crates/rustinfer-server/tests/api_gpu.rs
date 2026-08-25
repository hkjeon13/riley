//! Remote-only HTTP -> engine -> scheduler -> CUDA lifecycle gate.

#![cfg(feature = "cuda")]
#![allow(clippy::similar_names, clippy::too_many_lines)]

use std::error::Error;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rustinfer_model::{EncodeOptions, LoadLimits, LoadedModel};
use rustinfer_runtime::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use rustinfer_runtime::paged_kv::KV_BLOCK_SIZE;
use rustinfer_scheduler::{OverloadPolicy, SchedulerConfig};
use rustinfer_server::domain::{ModelMetadata, RequestLimits, ServiceErrorClass};
use rustinfer_server::engine::{
    CudaBackendConfig, CudaEngineResources, EngineConfig, InferenceEngine,
};
use rustinfer_server::http::HttpLimits;
use rustinfer_server::openai::{CompletionChunk, CompletionFinishReason, CompletionResponse};
use rustinfer_server::service::{
    CompletionBackend, RequestObservationStatus, ServerConfig, ServerHandle, start_server,
};
use serde_json::json;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ONE_GIB: u64 = 1 << 30;
const CLIENT_IO_TIMEOUT: Duration = Duration::from_secs(120);
const STATE_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_SEQUENCE_TOKENS: usize = 256;
const MAX_OUTPUT_TOKENS: usize = 32;
const MAX_ACTIVE_REQUESTS: usize = 2;
const MAX_WAITING_REQUESTS: usize = 4;
const ITERATION_TOKEN_BUDGET: usize = 16;
const MAX_PREFILL_CHUNK_TOKENS: usize = 1;
const PREFILL_PROMPT_TARGET_TOKENS: usize = 192;
const PHYSICAL_BLOCKS: usize = MAX_ACTIVE_REQUESTS * MAX_SEQUENCE_TOKENS.div_ceil(KV_BLOCK_SIZE);
const MODEL_ID: &str = "rustinfer-real-gpu-gate";

#[derive(Debug)]
struct RawHttpResponse {
    status: u16,
    body: Vec<u8>,
}

#[derive(Debug)]
struct CompletionResult {
    text: String,
    finish_reason: CompletionFinishReason,
}

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn multi_iteration_prompt(model: &LoadedModel) -> TestResult<String> {
    let mut prompt = String::new();
    for _ in 0..64 {
        prompt.push_str("The quick brown fox jumps over the lazy dog. ");
        let token_count = model
            .tokenizer()
            .encode(&prompt, EncodeOptions::default())?
            .len();
        if token_count >= PREFILL_PROMPT_TARGET_TOKENS {
            if token_count <= MAX_PREFILL_CHUNK_TOKENS {
                return Err(io::Error::other("prefill prompt did not span multiple chunks").into());
            }
            if token_count + MAX_OUTPUT_TOKENS > MAX_SEQUENCE_TOKENS {
                return Err(io::Error::other("prefill prompt exceeded the test context").into());
            }
            return Ok(prompt);
        }
    }
    Err(io::Error::other("could not construct a bounded multi-iteration prompt").into())
}

fn cuda_config() -> TestResult<CudaBackendConfig> {
    let scheduler = SchedulerConfig {
        max_waiting_requests: MAX_WAITING_REQUESTS,
        max_waiting_prompt_tokens: MAX_WAITING_REQUESTS * MAX_SEQUENCE_TOKENS,
        max_active_sequences: MAX_ACTIVE_REQUESTS,
        max_sequence_tokens: MAX_SEQUENCE_TOKENS,
        iteration_token_budget: ITERATION_TOKEN_BUDGET,
        max_prefill_chunk_tokens: MAX_PREFILL_CHUNK_TOKENS,
        aging_threshold_ns: 10_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(60_000_000_000),
        max_promised_kv_blocks: PHYSICAL_BLOCKS,
        metrics_window_samples: 32,
    };
    let executor = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(
            MAX_ACTIVE_REQUESTS,
            ITERATION_TOKEN_BUDGET,
            PHYSICAL_BLOCKS,
            MAX_ACTIVE_REQUESTS,
            PHYSICAL_BLOCKS,
        )?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_fused_residual_norm();
    Ok(CudaBackendConfig {
        device_ordinal: 0,
        scheduler,
        executor,
    })
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        command_queue_capacity: 8,
        event_channel_capacity: 8,
        max_inflight_requests: MAX_ACTIVE_REQUESTS + MAX_WAITING_REQUESTS,
        admission_timeout: Duration::from_secs(30),
        idle_poll_interval: Duration::from_millis(1),
    }
}

fn server_config() -> ServerConfig {
    ServerConfig {
        bind_address: SocketAddr::from(([127, 0, 0, 1], 0)),
        worker_threads: 6,
        connection_queue_capacity: 8,
        read_timeout: Duration::from_secs(10),
        write_timeout: Duration::from_secs(30),
        request_timeout: Duration::from_secs(120),
        shutdown_grace: Duration::from_secs(60),
        http_limits: HttpLimits::default(),
        request_limits: RequestLimits {
            max_prompt_bytes: 64 * 1024,
            max_output_tokens: MAX_OUTPUT_TOKENS,
            ..RequestLimits::default()
        },
        maximum_non_streaming_bytes: 64 * 1024,
        observation_capacity: 32,
    }
}

fn completion_body(prompt: &str, max_tokens: usize, stream: bool) -> serde_json::Result<Vec<u8>> {
    serde_json::to_vec(&json!({
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 7,
        "stream": stream
    }))
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

fn parse_http_response(bytes: &[u8]) -> io::Result<RawHttpResponse> {
    let header_offset = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| io::Error::other("HTTP response omitted the header terminator"))?;
    let header_end = header_offset + 4;
    let head = std::str::from_utf8(&bytes[..header_offset])
        .map_err(|_| io::Error::other("HTTP response headers were not UTF-8"))?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_ascii_whitespace().nth(1))
        .ok_or_else(|| io::Error::other("HTTP response omitted a status"))?
        .parse::<u16>()
        .map_err(|_| io::Error::other("HTTP response status was invalid"))?;
    Ok(RawHttpResponse {
        status,
        body: bytes[header_end..].to_vec(),
    })
}

fn send_completion(address: SocketAddr, body: &[u8]) -> io::Result<RawHttpResponse> {
    let mut stream = open_completion(address, body)?;
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes)?;
    parse_http_response(&bytes)
}

fn parse_non_streaming(response: &RawHttpResponse) -> TestResult<CompletionResult> {
    if response.status != 200 {
        return Err(io::Error::other("non-streaming completion did not return HTTP 200").into());
    }
    let response: CompletionResponse = serde_json::from_slice(&response.body)?;
    if response.choices.len() != 1 || response.usage.completion_tokens == 0 {
        return Err(io::Error::other("non-streaming completion shape was invalid").into());
    }
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| io::Error::other("non-streaming completion omitted its choice"))?;
    Ok(CompletionResult {
        text: choice.text,
        finish_reason: choice.finish_reason,
    })
}

fn parse_streaming(response: &RawHttpResponse) -> TestResult<CompletionResult> {
    if response.status != 200 {
        return Err(io::Error::other("streaming completion did not return HTTP 200").into());
    }
    let body = std::str::from_utf8(&response.body)?;
    let frames = body
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect::<Vec<_>>();
    if frames.len() < 2 || frames.last().copied() != Some("data: [DONE]") {
        return Err(io::Error::other("SSE stream did not end with [DONE]").into());
    }

    let mut text = String::new();
    let mut finish = None;
    let mut finish_index = None;
    for (index, frame) in frames[..frames.len() - 1].iter().enumerate() {
        let payload = frame
            .strip_prefix("data: ")
            .ok_or_else(|| io::Error::other("SSE frame omitted its data prefix"))?;
        let chunk: CompletionChunk = serde_json::from_str(payload)?;
        if chunk.choices.len() != 1 {
            return Err(io::Error::other("SSE chunk did not contain exactly one choice").into());
        }
        let choice = chunk
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| io::Error::other("SSE chunk omitted its choice"))?;
        if let Some(reason) = choice.finish_reason {
            if finish.is_some() || !choice.text.is_empty() {
                return Err(io::Error::other("SSE finish chunk was malformed").into());
            }
            finish = Some(reason);
            finish_index = Some(index);
        } else {
            if finish.is_some() {
                return Err(io::Error::other("SSE emitted a delta after finish").into());
            }
            text.push_str(&choice.text);
        }
    }
    if finish_index != Some(frames.len() - 2) {
        return Err(io::Error::other("SSE finish was not immediately before [DONE]").into());
    }
    Ok(CompletionResult {
        text,
        finish_reason: finish.ok_or_else(|| io::Error::other("SSE omitted its finish chunk"))?,
    })
}

fn read_http_head(reader: &mut BufReader<TcpStream>) -> io::Result<u16> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(io::Error::other(
            "streaming response omitted its status line",
        ));
    }
    let status = line
        .split_ascii_whitespace()
        .nth(1)
        .ok_or_else(|| io::Error::other("streaming response status line was invalid"))?
        .parse::<u16>()
        .map_err(|_| io::Error::other("streaming response status was invalid"))?;
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
            return Err(io::Error::other("SSE stream ended before its first delta"));
        }
        if let Some(payload) = line.strip_prefix("data: ") {
            return Ok(payload.trim_end_matches(['\r', '\n']).to_owned());
        }
    }
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

fn wait_for_disconnect_observation(
    server: &ServerHandle,
    previous_count: usize,
    expect_first_token: bool,
) -> io::Result<()> {
    wait_until(|| {
        server
            .observations()
            .observations
            .into_iter()
            .skip(previous_count)
            .any(|observation| {
                observation.status == RequestObservationStatus::ClientDisconnected
                    && observation.error_class == Some(ServiceErrorClass::Cancelled)
                    && observation.time_to_first_token.is_some() == expect_first_token
                    && (expect_first_token || observation.tokens_generated == 0)
            })
    })
}

#[test]
#[ignore = "requires pinned checkpoint and CUDA on server-4096"]
fn real_cuda_http_lifecycle_is_bounded_and_consistent() -> TestResult {
    let limits = LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?;
    let model = LoadedModel::load(&checkpoint_path(), limits)?;
    if model.spec().max_sequence_length() < MAX_SEQUENCE_TOKENS {
        return Err(io::Error::other("checkpoint context is smaller than the GPU gate").into());
    }
    let parity_prompt = "Once upon a time, a careful engineer";
    let parity_prompt_tokens = model
        .tokenizer()
        .encode(parity_prompt, EncodeOptions::default())?
        .len();
    if parity_prompt_tokens + MAX_OUTPUT_TOKENS > MAX_SEQUENCE_TOKENS {
        return Err(io::Error::other("parity prompt exceeded the test context").into());
    }
    let long_prompt = multi_iteration_prompt(&model)?;

    let metadata = ModelMetadata {
        model_id: MODEL_ID.to_owned(),
        created_unix_seconds: 0,
        owned_by: "rustinfer".to_owned(),
        context_window_tokens: MAX_SEQUENCE_TOKENS,
        max_output_tokens: MAX_OUTPUT_TOKENS,
    };
    let resources = CudaEngineResources::prepare(metadata, model, cuda_config()?)?;
    let engine = Arc::new(InferenceEngine::start_cuda(resources, engine_config())?);
    let backend: Arc<dyn CompletionBackend> = engine.clone();
    let server = start_server(server_config(), backend)?;
    let address = server.local_address();

    let non_streaming_body = completion_body(parity_prompt, 8, false)?;
    let non_streaming = parse_non_streaming(&send_completion(address, &non_streaming_body)?)?;
    let streaming_body = completion_body(parity_prompt, 8, true)?;
    let streaming = parse_streaming(&send_completion(address, &streaming_body)?)?;
    if streaming.text != non_streaming.text {
        return Err(io::Error::other("streaming and non-streaming text differed").into());
    }
    if streaming.finish_reason != non_streaming.finish_reason {
        return Err(io::Error::other("streaming and non-streaming finish reasons differed").into());
    }
    wait_until(|| {
        let status = engine.status();
        status.active_requests == 0 && status.waiting_requests == 0
    })?;

    let prefill_observations = server.observations().observations.len();
    let prefill_body = completion_body(&long_prompt, MAX_OUTPUT_TOKENS, false)?;
    let prefill_client = open_completion(address, &prefill_body)?;
    wait_until(|| engine.status().active_requests > 0)?;
    prefill_client.shutdown(Shutdown::Both)?;
    drop(prefill_client);
    wait_for_disconnect_observation(&server, prefill_observations, false)?;
    wait_until(|| {
        let status = engine.status();
        status.active_requests == 0 && status.waiting_requests == 0
    })?;

    let decode_observations = server.observations().observations.len();
    let decode_body = completion_body(parity_prompt, MAX_OUTPUT_TOKENS, true)?;
    let decode_client = open_completion(address, &decode_body)?;
    let mut decode_reader = BufReader::new(decode_client);
    if read_http_head(&mut decode_reader)? != 200 {
        return Err(io::Error::other("decode stream did not return HTTP 200").into());
    }
    let first_payload = read_sse_payload(&mut decode_reader)?;
    if first_payload == "[DONE]" {
        return Err(io::Error::other("decode stream finished before its first delta").into());
    }
    let first_chunk: CompletionChunk = serde_json::from_str(&first_payload)?;
    if first_chunk.choices.len() != 1
        || first_chunk.choices[0].finish_reason.is_some()
        || first_chunk.choices[0].text.is_empty()
    {
        return Err(io::Error::other("decode stream first event was not a text delta").into());
    }
    decode_reader.get_mut().shutdown(Shutdown::Both)?;
    drop(decode_reader);
    wait_for_disconnect_observation(&server, decode_observations, true)?;
    wait_until(|| {
        let status = engine.status();
        status.active_requests == 0 && status.waiting_requests == 0
    })?;

    let concurrent_body = completion_body(parity_prompt, 4, false)?;
    let clients = (0..3)
        .map(|_| {
            let body = concurrent_body.clone();
            thread::spawn(move || send_completion(address, &body))
        })
        .collect::<Vec<_>>();
    for client in clients {
        let response = client
            .join()
            .map_err(|_| io::Error::other("concurrent HTTP client panicked"))??;
        let _ = parse_non_streaming(&response)?;
    }
    wait_until(|| {
        let status = engine.status();
        status.active_requests == 0 && status.waiting_requests == 0
    })?;

    let shutdown_body = completion_body(&long_prompt, MAX_OUTPUT_TOKENS, false)?;
    let shutdown_client = thread::spawn(move || send_completion(address, &shutdown_body));
    wait_until(|| engine.status().active_requests > 0)?;
    server.shutdown()?;
    let shutdown_response = shutdown_client
        .join()
        .map_err(|_| io::Error::other("shutdown HTTP client panicked"))??;
    if shutdown_response.status != 503 {
        return Err(io::Error::other(format!(
            "active shutdown request returned HTTP {} instead of 503",
            shutdown_response.status
        ))
        .into());
    }
    let status = engine.status();
    if status.ready
        || status.accepting
        || status.active_requests != 0
        || status.waiting_requests != 0
    {
        return Err(io::Error::other("engine did not reach a drained shutdown state").into());
    }
    Ok(())
}
