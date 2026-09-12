#!/usr/bin/env python3
"""Correctness-only HTTP check of batch3's new binary; no timing trials.

Requires the completed build receipt. GPU component tests remain batch2
evidence; these requests exercise the new HTTP binary with the frozen runtime.
"""

import argparse
import http.client
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from build_batch3 import (SERVICE_SHA, component_parity, evidence, git, read,
                          require, sha, validate_frozen, write_new)


def check(root, model, binding_path):
    from tokenizers import Tokenizer

    root, model, binding_path = root.resolve(), model.resolve(), binding_path.resolve()
    build_path = root / "batch3-build.json"
    build, baseline = read(build_path), read(root / "batch2-build.json")
    validate_frozen(root, baseline)
    parity = component_parity(root, baseline)
    require(build["batch2_component_parity"]["files"] == parity, "component parity receipt differs")
    source, binary = root / "batch3-source", root / "batch3-target/release/riley"
    commit = git(source, "rev-parse", "HEAD")
    require(commit == build["source_commit"], "HTTP source differs from build")
    require(sha(binary) == build["binaries"][str(binary)], "HTTP binary differs from build")
    binding = read(binding_path)
    reference = binding["generated_token_ids"]
    require(len(reference) == 32, "expected exactly 32 reference tokens")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    expected = tokenizer.decode(reference, skip_special_tokens=True)
    prompt = "Hello" * 128
    require(tokenizer.encode(prompt, add_special_tokens=True).ids == binding["input_token_ids"],
            "HTTP prompt differs from fixed reference tokens")
    require(len(binding["input_token_ids"]) == 128, "expected P128 reference")
    directory = root / "batch3-http-correctness"
    directory.mkdir()

    def send(port, text=prompt, outputs=32, stream=False):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        connection.request("POST", "/v1/completions", json.dumps({
            "model": "g04-smol", "prompt": text, "temperature": 0,
            "max_tokens": outputs, "stream": stream,
        }), {"Content-Type": "application/json"})
        return connection, connection.getresponse()

    def complete(port, stream=False):
        connection, response = send(port, stream=stream)
        try:
            require(response.status == 200, "completion returned HTTP " + str(response.status))
            if not stream:
                body = json.loads(response.read())
                require(body["usage"]["prompt_tokens"] == 128
                        and body["usage"]["completion_tokens"] == 32, "completion usage differs")
                require(body["choices"][0]["finish_reason"] == "length", "completion length differs")
                text = body["choices"][0]["text"]
            else:
                text, done, finish = "", False, None
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    for choice in json.loads(data).get("choices", []):
                        text += choice.get("text", "")
                        finish = choice.get("finish_reason") or finish
                require(done and finish == "length", "SSE did not finish at the fixed length")
            require(text == expected, "HTTP output differs from the complete reference text")
        finally:
            response.close()
            connection.close()

    results = []
    for sampling in ("cpu", "gpu-greedy"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        argv = [str(binary), "serve", "--model", str(model), "--model-id", "g04-smol",
                "--bind", f"127.0.0.1:{port}", "--max-active-sequences", "1",
                "--max-waiting-requests", "4", "--batch-token-budget", "128",
                "--prefill-chunk-tokens", "128", "--max-sequence-tokens", "160",
                "--max-output-tokens", "32", "--kv-blocks", "10", "--residual-rmsnorm", "separate",
                "--execution-completion", "iteration-batch", "--metadata-transport", "packed-async",
                "--execution-graph-policy", "require", "--graph-numerics", "vllm-smol-p128-v1",
                "--sampling-backend", sampling]
        env = dict(os.environ, LD_LIBRARY_PATH="/data/riley-g04-cuda13/lib")
        log_path = directory / f"http-{sampling}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=log, env=env, cwd=source)
            try:
                ready, deadline = False, time.monotonic() + 120
                while process.poll() is None and time.monotonic() < deadline:
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    try:
                        connection.request("GET", "/v1/models")
                        response = connection.getresponse()
                        response.read()
                        if response.status == 200:
                            ready = True
                            break
                    except OSError:
                        pass
                    finally:
                        connection.close()
                    time.sleep(0.1)
                require(ready, "HTTP server did not become ready")
                complete(port)
                complete(port, True)
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
                for after_token in (False, True):
                    connection, response = send(port, stream=True)
                    require(response.status == 200, "disconnect request was rejected")
                    try:
                        if after_token:
                            found_token = False
                            for line in response:
                                if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                                    choices = json.loads(line[6:]).get("choices", [])
                                    if any(choice.get("text") for choice in choices):
                                        found_token = True
                                        break
                            require(found_token, "disconnect probe did not observe text")
                    finally:
                        connection.close()
                        response.close()
                    complete(port)
                process.terminate()
                process.wait(timeout=30)
                require(process.returncode == 0, "server did not shut down cleanly")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        results.append({
            "sampling": sampling, "full_text_exact": True, "streaming_exact": True,
            "invalid_request_statuses": rejected, "post_disconnect_reuse_exact": True,
            "argv": argv, "server_exit_code": process.returncode, "server_log": evidence(log_path),
        })
    require(git(source, "rev-parse", "HEAD") == commit, "source commit changed during HTTP check")
    require(component_parity(root, baseline) == parity, "source changed during HTTP check")
    require(sha(binary) == build["binaries"][str(binary)], "binary changed during HTTP check")
    result = {
        "passed": True, "profile": "vllm-smol-p128-v1", "source_commit": commit,
        "service_sha256": SERVICE_SHA, "build": evidence(build_path),
        "binary_sha256": sha(binary), "reference_binding": evidence(binding_path),
        "reference_tokens": reference, "results": results, "performance_trials": 0,
        "gpu_component_tests_rerun": False,
    }
    write_new(directory / "http-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/riley-opt-260912"))
    parser.add_argument("--model", type=Path, default=Path("/data/riley-benchmark/20260827T051948Z-d7ad713a/model"))
    parser.add_argument("--binding", type=Path, default=Path("/tmp/riley-g04-vllm-profile-260911/native-binding.json"))
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.model, args.binding), indent=2))


if __name__ == "__main__":
    main()
