"""Loopback SSE reference for the real Coder checkpoint, used only to check final-answer delivery."""

import json
import os
import subprocess
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent_workflow import fingerprint, replay, stream
from coder_catalog_trial import tool_output


def record_tool(seed, receipts, invocations, index, call, docs):
    invocations.append({"index": index, "id": call["id"], "function": call["function"]})
    return tool_output("selected", False, seed, receipts, index, call, docs)


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    root = Path.cwd()
    reg = json.loads((root / "named-cpu-registration.json").read_text())
    checkpoint = root.parents[2] / "models" / reg["checkpoint"]
    profiler = root.parent / reg["native_profiler"]
    if "http_budget" in reg or reg.get("report_prefix"):
        profiler = root / reg["native_profiler"]
    assert not os.environ.get("CUDA_VISIBLE_DEVICES") and not torch.cuda.is_available()
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    assert next(model.parameters()).device.type == "cpu"
    observations = []
    external_final = threading.Event()

    def encode(req):
        encoded = subprocess.run(
            [profiler, checkpoint],
            input=json.dumps(req) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(encoded.stdout)["ids"]

    prep = None
    if "first_tool_kv" in reg:
        from cpu_first_tool_kv import FirstToolKV

        prep = FirstToolKV(model, encode, reg["first_tool_kv"])

    wire = None
    if "http_budget" in reg:
        from cpu_bounded_http import HTTPPreview

        wire = HTTPPreview(model, tokenizer, reg["http_budget"])

    class Reference(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert (
                req["tool_choice"] == "none"
                and req["stream"]
                and req["temperature"] == 0
            )
            texts = [m for m in req["messages"] if m.get("role") == "tool"]
            is_preview = wire is not None and req["max_tokens"] == 1
            assert is_preview or len(texts) == 2
            # Use the native tokenizer and renderer from the earlier Riley source build.
            if prep:
                prep.stop_preparation()
            ids = encode(req)
            inputs = torch.tensor([ids], dtype=torch.long)
            rid = f"cpu-{len(observations)}"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def emit(obj):
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.flush()

            if is_preview:
                wire.serve(req, ids, emit)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            class SSE(TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    if text:
                        emit(
                            {
                                "id": rid,
                                "choices": [{"index": 0, "delta": {"content": text}}],
                            }
                        )

            extra = {}
            hook = None
            owner = prep or wire
            if owner:
                cache = owner.take(ids)
                if cache is not None:
                    extra["past_key_values"] = cache

                def observe(_, args, kwargs):
                    values = kwargs["input_ids"]
                    owner.records[-1].setdefault(
                        "first_forward_tokens", values.shape[1]
                    )

                hook = model.register_forward_pre_hook(observe, with_kwargs=True)
            with torch.inference_mode():
                result = model.generate(
                    inputs,
                    attention_mask=torch.ones_like(inputs),
                    **extra,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=req["max_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=SSE(
                        tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                )
            if hook:
                hook.remove()
            generated = result[0, len(ids) :].tolist()
            eos = model.generation_config.eos_token_id
            if isinstance(eos, int):
                eos = [eos]
            finish = "stop" if generated[-1] in eos else "length"
            usage = {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated),
                "total_tokens": len(ids) + len(generated),
            }
            observations.append(
                {
                    "id": rid,
                    "input_sha256": fingerprint(req),
                    "prompt_ids_sha256": fingerprint(ids),
                    "output_ids": generated,
                    "decoded": tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "finish": finish,
                    "usage": usage,
                }
            )
            emit(
                {
                    "id": rid,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": usage,
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            if reg.get("external_client"):
                external_final.set()

    server = HTTPServer(("127.0.0.1", 0), Reference)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    (root / "ready.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "threads": torch.get_num_threads(),
                "dtype": str(next(model.parameters()).dtype),
                "device": "cpu",
            }
        )
        + "\n"
    )

    def send(address, request, row, event):
        if reg.get("layout_citations"):
            from multi_router_trial import with_layout

            request = with_layout(request)
        elif reg.get("explicit_citations"):
            from verify_agent_citations import with_citations

            request = with_citations(request)
        if reg.get("report_prefix"):
            from report_answer_prefix import prefix_request

            request = prefix_request(
                request, guide_continuation=reg.get("guide_continuation", False)
            )
            prefix = request.get("riley_assistant_prefix")
            if prefix:
                row["application_prefix"] = prefix
                suffix, calls = stream(address, request, row, event, timeout_s=600)
                row["generated_suffix"] = suffix
                return prefix + suffix, calls
        return stream(address, request, row, event, timeout_s=600)

    rows = []
    try:
        if reg.get("external_client"):
            assert wire and len(reg["jobs"]) == 1
            (root / "address.json").write_text(
                json.dumps({"port": server.server_port}) + "\n"
            )
            if not external_final.wait(600):
                raise TimeoutError("external workflow did not complete")
            (root / "inference.json").write_text(json.dumps(observations) + "\n")
            (root / "external-preview.json").write_text(json.dumps(wire.records) + "\n")
            return
        for seed, history in reg["jobs"]:
            receipts = []
            invocations = []

            if wire:
                from budgeted_tool_prefix import PromptProfiler, budgeted_preview
                from tool_wait_trial import measure

                with PromptProfiler(profiler, checkpoint) as planner:
                    trial = measure(
                        f"127.0.0.1:{server.server_port}",
                        True,
                        "bounded",
                        seed,
                        reg["delay_s"],
                        include_completed_call=True,
                        first_fixed_tool=True,
                        cancel_on_final=True,
                        preview_planner=partial(
                            budgeted_preview,
                            budget=reg["http_budget"],
                            encode=planner.ids,
                        ),
                        document_pairs=2,
                        history=history,
                        execute_tool=partial(record_tool, seed, receipts, invocations),
                    )
                work = trial["target"]
                (root / "http-preview.json").write_text(
                    json.dumps({"client": trial["prefix"], "server": wire.records})
                    + "\n"
                )
            else:
                work = replay(
                    f"127.0.0.1:{server.server_port}",
                    "m",
                    seed,
                    2,
                    reg["delay_s"],
                    f"cpu-{seed}",
                    history=history,
                    on_tool_start=prep.start if prep else None,
                    execute_tool=partial(record_tool, seed, receipts, invocations),
                    dispatch_fixed_tools=True,
                    send=send,
                )
            rows.append(
                {"work": work, "receipts": receipts, "invocations": invocations}
            )
            if prep:
                (root / "preparation.json").write_text(json.dumps(prep.records) + "\n")
            (root / "client.json").write_text(json.dumps(rows) + "\n")
            (root / "inference.json").write_text(json.dumps(observations) + "\n")
    finally:
        server.shutdown()
        serving.join()
        server.server_close()


if __name__ == "__main__":
    main()
