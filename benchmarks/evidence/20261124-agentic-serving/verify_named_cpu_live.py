"""Check the streamed response, tokenizer fingerprint and committed tool calls independently."""

import argparse
import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path

from agent_workflow import (
    answer_format,
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)
from coder_catalog_trial import catalog
from multi_router_trial import with_layout
from verify_agent_citations import with_citations
from verify_coder_catalog import answer_checks
from verify_model_decode import natural, read


def verify(root, profiler, tokenizer):
    from tokenizers import Tokenizer

    assert (root / "exit.log").read_text().strip() == "0"
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha256((root / name).read_bytes()).hexdigest() == digest
    assert (
        sha256((root / "model.sha256").read_bytes()).hexdigest()
        == "43351e19aa3f53fc3447b2a5f4e29a7f05f7a329b6f283ed09414abd8964da7b"
    )
    model = {
        k: v
        for v, k in (
            line.split(maxsplit=1)
            for line in (root / "model.sha256").read_text().splitlines()
        )
    }
    assert len([p for p in model if p.endswith(".safetensors")]) == 4
    for p in ("tokenizer.json", "tokenizer_config.json"):
        assert sha256((tokenizer / p).read_bytes()).hexdigest() == model[p]
    assert not (tokenizer / "chat_template.jinja").exists()
    decoder = Tokenizer.from_file(str(tokenizer / "tokenizer.json"))
    eos = decoder.token_to_id(read(tokenizer / "tokenizer_config.json")["eos_token"])
    assert eos is not None
    reg = read(root / "named-cpu-registration.json")
    transform = None
    if reg.get("report_prefix"):
        from report_answer_prefix import prefix_request

        def transform(req):
            configured = prefix_request(
                req, guide_continuation=reg.get("guide_continuation", False)
            )
            assert configured["riley_assistant_prefix"]
            return configured

    elif reg.get("layout_citations"):
        assert not reg.get("explicit_citations")
        transform = with_layout
    elif reg.get("explicit_citations"):
        transform = with_citations
    engine = read(root / "ready.json")
    assert engine["device"] == "cpu" and engine["dtype"] == "torch.float32"
    rows = []
    for (seed, history), obs, k in zip(
        reg["jobs"],
        read(root / "inference.json"),
        read(root / "client.json"),
        strict=True,
    ):
        w = k["work"]
        tools, prompt, docs = fixture(seed, 2)
        facts = natural(
            w,
            seed,
            2,
            history,
            delay=reg["delay_s"],
            application_tools=True,
            final_transform=transform,
        )
        messages = history_messages(seed, history) + [
            {"role": "user", "content": prompt}
        ]
        assert len(k["invocations"]) == 2
        for idx, (turn, inv) in enumerate(
            zip(w["turns"][:2], k["invocations"], strict=True)
        ):
            assert inv == {
                "id": turn["call"]["id"],
                "function": turn["call"]["function"],
                "index": idx,
            }
            assert turn["tool_ready_ns"] - turn["tool_start_ns"] >= reg["delay_s"] * 1e9
            append_tool_result(messages, [turn["call"]], docs, idx)
        assert k["invocations"][0]["id"] != k["invocations"][1]["id"]
        assert len(k["receipts"]) == 1
        receipt = k["receipts"][0]

        assert receipt == {
            "source_sha256": fingerprint(catalog(seed, docs)),
            "result_sha256": fingerprint(docs),
            "source_count": 64,
            "result_count": 2,
            "requested_ids": ["report-A", "report-B"],
        }
        req = turn_request("m", messages, tools, None)
        if transform:
            req = transform(req)
        final = w["turns"][-1]
        assert (
            obs["id"] == final["response_id"]
            and fingerprint(req) == final["input_sha256"] == obs["input_sha256"]
        )
        ids = json.loads(
            subprocess.run(
                [profiler, tokenizer],
                input=json.dumps(req) + "\n",
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )["ids"]
        assert fingerprint(ids) == obs["prompt_ids_sha256"]
        assert (
            obs["usage"] == final["usage"]
            and len(ids) == final["usage"]["prompt_tokens"]
        )
        assert len(obs["output_ids"]) == final["usage"]["completion_tokens"]
        assert (
            decoder.decode(obs["output_ids"], skip_special_tokens=True)
            == obs["decoded"]
        )
        prefix = req.get("riley_assistant_prefix", "")
        if prefix:
            assert prefix == final["application_prefix"]
            assert obs["decoded"] == final["generated_suffix"]
        assert (
            prefix + obs["decoded"] == final["answer"]
            and obs["finish"] == final["finish"]
        )
        if obs["finish"] == "stop":
            assert obs["output_ids"][-1] == eos
        assert (
            len(final["content_ns"]) > 1
            and final["sent_ns"]
            <= final["content_ns"][0]
            <= final["content_ns"][-1]
            <= final["done_ns"]
        )
        assert final["answer_format"] == answer_format(final["answer"], docs)
        paragraphs = re.split(
            r"\n\s*\n", final["answer"].replace("**", "").strip(), maxsplit=1
        )
        explanation = paragraphs[1] if len(paragraphs) == 2 else ""
        adjacent = {
            doc["id"]: bool(
                re.search(
                    rf"\b{doc['p95_ms']}\s*(?:ms|milliseconds)\s*\(\s*{re.escape(doc['id'])}\s*\)",
                    explanation,
                    re.IGNORECASE,
                )
            )
            for doc in docs
        }
        rows.append(
            {
                "seed": seed,
                "history": history,
                "tool_invocations": len(k["invocations"]),
                "model_requests": 1,
                "valid": facts["valid"],
                "usage": obs["usage"],
                "finish": obs["finish"],
                "explanation_words": final["answer_format"]["explanation_words"],
                "visible_chunks": len(final["content_ns"]),
                **(
                    {
                        "application_prefix": prefix,
                        "generated_suffix": obs["decoded"],
                        "submission_to_first_generated_chunk_ms": facts[
                            "final_submission_to_first_ms"
                        ],
                    }
                    if prefix
                    else {}
                ),
                "immediately_attributed": adjacent,
                "strict_opening": final["answer_format"]["exact_numeric_opening"],
                "answer": final["answer"],
                **answer_checks(w),
            }
        )
    return {"engine": engine, "jobs": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "profiler", "tokenizer", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(verify(a.root, a.profiler, a.tokenizer), indent=2) + "\n"
    )
