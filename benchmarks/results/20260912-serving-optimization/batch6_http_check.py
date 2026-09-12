#!/usr/bin/env python3
"""Correctness-only HTTP qualification of the frozen batch6 release binary."""

import argparse
import http.client
import json
from pathlib import Path
import socket
import subprocess
import time

from run_batch6_tests import (
    DEFAULT_BASE, DEFAULT_ROOT, MODEL, environment, read, require, sha, verify_snapshot, write_new,
)


def send(port, prompt, outputs=32, stream=False):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        connection.request("POST", "/v1/completions", json.dumps({
            "model": "g04-smol", "prompt": prompt, "temperature": 0,
            "max_tokens": outputs, "stream": stream,
        }), {"Content-Type": "application/json"})
        return connection, connection.getresponse()
    except BaseException:
        connection.close()
        raise


def complete(port, prompt, expected, *, stream=False):
    connection, response = send(port, prompt, stream=stream)
    try:
        require(response.status == 200, "completion HTTP status differs: " + str(response.status))
        if not stream:
            value = json.loads(response.read())
            require(value["usage"]["prompt_tokens"] == 128
                    and value["usage"]["completion_tokens"] == 32, "HTTP token usage differs")
            require(len(value["choices"]) == 1 and value["choices"][0]["finish_reason"] == "length",
                    "completion choice or finish reason differs")
            text = value["choices"][0]["text"]
        else:
            text, finish, done = "", None, False
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                value = json.loads(data)
                require("error" not in value and len(value.get("choices", [])) == 1,
                        "SSE error or unexpected choices")
                choice = value["choices"][0]
                text += choice.get("text", "")
                finish = choice.get("finish_reason") or finish
            require(done and finish == "length", "SSE terminal contract differs")
        require(text == expected, "HTTP output text differs from exact token reference")
    finally:
        response.close()
        connection.close()


def wait_ready(process, port):
    deadline = time.monotonic() + 120
    while process.poll() is None and time.monotonic() < deadline:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            body = response.read()
            if response.status == 200 and any(item.get("id") == "g04-smol"
                                              for item in json.loads(body).get("data", [])):
                require(process.poll() is None, "server exited during readiness")
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise RuntimeError("owned batch6 HTTP server did not become ready")


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def main():
    # Import only when executing HTTP checks; --help and static validation need
    # no remote tokenizers installation.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    from tokenizers import Tokenizer

    root, base = args.root.resolve(), args.base.resolve()
    runner_sha256 = sha(Path(__file__))
    shared_runner = Path(__file__).with_name("run_batch6_tests.py")
    shared_runner_sha256 = sha(shared_runner)
    build_path = root / "batch6-build.json"
    build, build_sha256 = read(build_path), sha(build_path)
    initial = verify_snapshot(root, build, build_sha256)
    binary = root / "batch6-target/release/riley"
    binding_path, request_path = base / "native-binding.json", base / "request.json"
    binding, request = read(binding_path), read(request_path)
    references = {str(path): sha(path) for path in (binding_path, request_path)}
    model_files = {str(MODEL / name): binding["workload"][field]
                   for name, field in (("model.safetensors", "weights_sha256"),
                                       ("tokenizer.json", "tokenizer_sha256"))}
    require(all(sha(path) == expected for path, expected in model_files.items()), "model data pin differs")
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    prompt, tokens = request["prompt"], binding["generated_token_ids"]
    require(tokenizer.encode(prompt, add_special_tokens=True).ids == binding["input_token_ids"]
            == request["prompt_token_ids"] and len(binding["input_token_ids"]) == 128
            and len(tokens) == 32 and request["requested_output_tokens"] == 32,
            "fixed HTTP request/reference token contract differs")
    expected = tokenizer.decode(tokens, skip_special_tokens=True)
    require(bool(expected), "reference text is empty")
    output = root / "batch6-http-correctness"
    output.mkdir(exist_ok=False)
    results = []
    env = environment(root)
    for sampling in ("cpu", "gpu-greedy"):
        require(verify_snapshot(root, build, build_sha256) == initial, "snapshot changed before HTTP lane")
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        argv = [str(binary), "serve", "--model", str(MODEL), "--model-id", "g04-smol",
                "--bind", f"127.0.0.1:{port}", "--max-active-sequences", "1",
                "--max-waiting-requests", "4", "--batch-token-budget", "128",
                "--prefill-chunk-tokens", "128", "--max-sequence-tokens", "160",
                "--max-output-tokens", "32", "--kv-blocks", "10", "--residual-rmsnorm", "separate",
                "--execution-completion", "iteration-batch", "--metadata-transport", "packed-async",
                "--execution-graph-policy", "require", "--graph-numerics", "vllm-smol-p128-v1",
                "--sampling-backend", sampling]
        log_path = output / ("http-" + sampling + ".log")
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=log, env=env, cwd=root / "batch6-source")
            try:
                wait_ready(process, port)
                complete(port, prompt, expected)
                complete(port, prompt, expected, stream=True)
                rejected = []
                for bad, outputs in (("Hello" * 127, 32), ("Hello" * 129, 31), (prompt, 33)):
                    connection, response = send(port, bad, outputs)
                    try:
                        response.read()
                        require(response.status == 400, "unsupported shape did not return HTTP 400")
                        rejected.append(response.status)
                    finally:
                        response.close()
                        connection.close()
                for after_text in (False, True):
                    connection, response = send(port, prompt, stream=True)
                    try:
                        require(response.status == 200, "disconnect probe was not admitted")
                        if after_text:
                            seen = False
                            for line in response:
                                if not line.startswith(b"data: "):
                                    continue
                                data = line[6:].strip()
                                if data == b"[DONE]":
                                    break
                                if any(choice.get("text") for choice in json.loads(data).get("choices", [])):
                                    seen = True
                                    break
                            require(seen, "disconnect probe did not observe text before closing")
                    finally:
                        response.close()
                        connection.close()
                    complete(port, prompt, expected)
                process.terminate()
                process.wait(timeout=30)
                require(process.returncode == 0, "HTTP server did not shut down gracefully")
            finally:
                stop_owned(process)
        require(verify_snapshot(root, build, build_sha256) == initial, "snapshot changed during HTTP lane")
        results.append({"sampling": sampling, "full_text_exact": True, "streaming_exact": True,
                        "invalid_request_statuses": rejected, "post_disconnect_reuse_exact": True,
                        "disconnect_points": ["after_headers_before_reading_text", "after_first_nonempty_text"],
                        "graceful_exit_code": 0, "argv": argv,
                        "log": {"path": str(log_path), "sha256": sha(log_path)}})
    require(all(sha(path) == expected for path, expected in {**references, **model_files}.items()),
            "reference or model files changed during HTTP qualification")
    require(verify_snapshot(root, build, build_sha256) == initial, "final HTTP source identity differs")
    require(sha(Path(__file__)) == runner_sha256 and sha(shared_runner) == shared_runner_sha256,
            "HTTP checker source changed during qualification")
    result = {"profile": "vllm-smol-p128-v1", "binary_sha256": initial["binaries"][str(binary)],
              "source_commit": initial["source_commit"], "source_clean": True,
              "build_sha256": build_sha256, "reference_tokens": tokens,
              "reference_binding_sha256": references[str(binding_path)], "references": references,
              "model_files": model_files, "results": results, "performance_trials": 0,
              "runner_sha256": runner_sha256, "shared_runner_sha256": shared_runner_sha256,
              "desktop_session_policy": "left untouched"}
    write_new(output / "http-validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
