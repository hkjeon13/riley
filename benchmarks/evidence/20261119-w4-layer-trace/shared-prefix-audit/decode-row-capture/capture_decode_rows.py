"""Run the selected W4 requests and read their actual BF16 logits by token ID."""

from argparse import ArgumentParser
import json
import math
from hashlib import sha256
from pathlib import Path
import re
import struct

from long_accuracy import complete, read_capture, read_workload, same_completion


MARKER = re.compile(r"^W4_DECODE_LOGITS prompt=(\d+) offset=(\d+) gpu_token=(none|\d+) vocabulary=(\d+) addressable=(\d+)$", re.MULTILINE)
CONTROL_CASES = {"long": (9,), "holdout": ()}


def require(condition, issue):
    if not condition:
        raise ValueError(issue)


def files(root):
    for suite in ("long", "holdout"):
        directory = root / f"20261119-w4-{suite}"
        workload = read_workload(directory / "holdout.json.gz" if suite == "holdout" else None)
        original = read_capture(directory / "riley/capture.json.gz", workload)
        reference = json.loads((directory / "hf/reference.json").read_bytes())
        require(reference.get("workload_sha256") == workload.sha256, f"{suite}: reference inputs differ")
        rows = zip(workload.cases, original["baseline"], original["diagnostic"], reference["cases"], strict=True)
        for count, (case, base, diag, hf) in enumerate(rows):
            divergence = hf["first_divergence"]
            if divergence is not None or count in CONTROL_CASES[suite]:
                require(same_completion(base, diag) and len(base["prompt_tokens"]) == hf["prompt_tokens"],
                        f"{suite}/{count}: source diagnostic or input differs")
                offset = divergence["offset"] if divergence is not None else 0
                other = hf["hf_tokens"][offset]
                yield suite, case, base, hf, offset, other


def score_row(data, vocabulary, addressable, selected, reference):
    require(len(data) == vocabulary * 2 and 1 < addressable <= vocabulary,
            "unexpected vocabulary extent in BF16 row")
    require(0 <= selected < addressable and 0 <= reference < addressable, "token ID outside addressable vocabulary")
    values = [struct.unpack("<f", b"\x00\x00" + struct.pack("<H", bits))[0]
              for (bits,) in struct.iter_unpack("<H", data[:addressable * 2])]
    require(all(math.isfinite(value) for value in values), "nonfinite addressable logit")
    highest = max(values)
    tied = [index for index, value in enumerate(values) if value == highest]
    require(selected in tied, "the served token was not among the highest unmasked raw logits")
    return {"sha256": sha256(data).hexdigest(), "addressable_vocabulary": addressable,
            "captured_vocabulary": vocabulary, "riley_logit": values[selected], "hf_choice_logit": values[reference],
            "riley_minus_hf_logit": values[selected] - values[reference], "tied_maximum_ids": tied}


def capture(root, address, model, rows, log):
    result = []
    for suite, case, base, hf, offset, other in files(root):
        response = complete(address, model, case.prompt, diagnostic=True, expected=base["prompt_tokens"], limit=case.limit)
        require(same_completion(base, response) and base["usage"] == response["usage"],
                f"{suite}/{case.label}: probe changed the original response")
        prompt = len(base["prompt_tokens"])
        captured = rows / f"{prompt}-{offset}.bf16"
        markers = [(gpu, int(vocab), int(addressable)) for plen, step, gpu, vocab, addressable in MARKER.findall(log.read_text())
                   if (int(plen), int(step)) == (prompt, offset)]
        require(len(markers) == 1, f"{suite}/{case.label}: expected one row receipt")
        gpu, vocab, addressable = markers[0]
        selected = response["tokens"][offset]
        require(gpu == "none" or int(gpu) == selected, f"{suite}/{case.label}: device and served IDs differ")
        result.append({"suite": suite, "case": case.label, "prompt_len": prompt, "offset": offset,
                       "riley_id": selected, "hf_id": other,
                       "captured_backend": "cpu" if gpu == "none" else "gpu_greedy",
                       "observed_output_hash": sha256(json.dumps(response["tokens"]).encode()).hexdigest(),
                       "row": score_row(captured.read_bytes(), vocab, addressable, selected, other)})
    require(len(result) == 8, "expected seven divergences and one matching control")
    return {"scope": "Actual Riley BF16 output rows on the original C1 path; output IDs, bytes, usage and finish reproduced.",
            "cases": result}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("rows", type=Path)
    parser.add_argument("server_log", type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.evidence, args.address, args.model, args.rows, args.server_log), indent=2, allow_nan=False))
