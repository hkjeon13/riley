"""Replay two observed output-head scores using the captured last-token norm."""
from argparse import ArgumentParser
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
import struct


TOKEN_IDS = (264, 5257)


def digest(data):
    return sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_payload(directory, name, expected):
    path = directory / name
    data = path.read_bytes() if path.exists() else gzip.decompress(path.with_name(name + ".gz").read_bytes())
    require(digest(data) == expected, f"{directory.name}/{name}: receipt digest differs")
    return data


def bf16(data):
    values = [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word, in struct.iter_unpack("<H", data)]
    require(all(map(math.isfinite, values)), "nonfinite BF16 data")
    return values


def round_bf16(value):
    if value == 0:
        return value
    _, exponent = math.frexp(value)
    quantum = math.ldexp(1.0, max(exponent - 8, -133))
    return round(value / quantum) * quantum


def read_weight_rows(model, hidden, vocabulary):
    index = (model / "model.safetensors.index.json").read_bytes()
    name = json.loads(index)["weight_map"]["lm_head.weight"]
    require(Path(name).name == name and name.endswith(".safetensors"), "unexpected weight shard path")
    shard = model / name
    with shard.open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        require(0 < length <= 16 * 1024**2, "invalid safetensors header length")
        header = stream.read(length)
        entry = json.loads(header)["lm_head.weight"]
        start, end = entry["data_offsets"]
        require(entry["dtype"] == "BF16" and entry["shape"] == [vocabulary, hidden]
                and end - start == 2 * hidden * vocabulary
                and 0 <= start < end <= shard.stat().st_size - length - 8, "unexpected output head geometry")
        raw = {}
        for token in TOKEN_IDS:
            require(token < vocabulary, "requested token outside vocabulary")
            stream.seek(8 + length + start + token * hidden * 2)
            raw[token] = stream.read(hidden * 2)
            require(len(raw[token]) == hidden * 2, "truncated output head row")
    return raw, {"shard": name, "index_sha256": digest(index), "header_sha256": digest(header),
                 "tokens": {str(token): digest(data) for token, data in raw.items()}}


def read_arms(trace, fixture, case, hidden, vocabulary, checkpoint_sha):
    source = json.loads(fixture)["baseline"][case]
    native = trace / f"native{case}"
    reference = trace / f"hf{case}"
    metadata, traces = {}, {}
    offsets = None
    for label, directory in (("riley", native), ("hf", reference)):
        receipt = json.loads((directory / "receipt.json").read_bytes())
        raw_hashes = receipt["result_sha256"]
        require(receipt["status"] == "passed" and receipt["case"] == case, f"{label}: trace did not finish")
        saved_inputs = {Path(path).name: value for path, value in receipt["sha256"].items()}
        require(saved_inputs["riley-checkpoint.json"] == checkpoint_sha
                and saved_inputs["legacy.json"] == digest(fixture), f"{label}: pinned inputs differ")
        data_dir = directory if label == "riley" else directory / "hf-rows"
        raw = read_payload(data_dir, "rows.bf16", raw_hashes["rows.bf16"])
        if label == "riley":
            index = read_payload(data_dir, "index.tsv", raw_hashes["index.tsv"]).decode().splitlines()
            head = index[0].split()
            require(tuple(map(int, head[1:3])) == (len(source["prompt_tokens"]) - 1, source["prompt_tokens"][-1]),
                    "Riley row differs from final input token")
            norm, logits = [row.split("\t") for row in index[-2:]]
            require(norm[:2] == ["final_norm", "-"] and logits[:2] == ["logits", "-"], "missing final rows")
            start, norm_bytes, logit_start, logit_bytes = map(int, norm[2:] + logits[2:])
            require(norm_bytes == 2 * hidden and logit_bytes == 2 * vocabulary
                    and start + norm_bytes == logit_start and logit_start + logit_bytes == len(raw),
                    "unexpected final trace extents")
            offsets = (start, logit_start, len(raw))
            response = json.loads(read_payload(data_dir, "response.json", raw_hashes["response.json"]))
            require(response["all_original_ids_text_and_finish"], "Riley generation changed under trace")
            selected = source["tokens"][0]
        else:
            meta = json.loads(read_payload(data_dir, "metadata.json", raw_hashes["metadata.json"]))
            require((meta["case"], meta["prompt_sha256"], meta["last_token_id"])
                    == (case, source["prompt_sha256"], source["prompt_tokens"][-1]), "HF row source differs")
            selected = meta["selected_id"]
        start, logit_start, extent = offsets
        require(len(raw) == extent, f"{label}: trace extent differs")
        final_norm = raw[start:logit_start]
        logits = bf16(raw[logit_start:])
        maximum = max(logits)
        traces[label] = bf16(final_norm)
        metadata[label] = {"captured_rows_sha256": digest(raw), "final_norm_sha256": digest(final_norm),
                           "selected_id": selected, "observed_pair": {str(token): logits[token] for token in TOKEN_IDS},
                           "observed_argmax_ties": [i for i, value in enumerate(logits) if value == maximum]}
    return traces, metadata


def replay(model, trace, fixture):
    config = json.loads((model / "config.json").read_bytes())
    require(config["tie_word_embeddings"] is False, "expected an explicit LM-head weight")
    hidden, vocabulary = config["hidden_size"], config["vocab_size"]
    checkpoint_sha = digest((model / "riley-checkpoint.json").read_bytes())
    raw, weights_metadata = read_weight_rows(model, hidden, vocabulary)
    weights = {token: bf16(row) for token, row in raw.items()}
    reports = {}
    for case in (3, 9):
        norms, engines = read_arms(trace, fixture, case, hidden, vocabulary, checkpoint_sha)
        for label, norm in norms.items():
            values = {str(token): math.fsum(x * w for x, w in zip(norm, row, strict=True))
                      for token, row in weights.items()}
            rounded = {token: round_bf16(value) for token, value in values.items()}
            engines[label]["fp64_dot_products"] = values
            engines[label]["fp64_margin_5257_minus_264"] = values["5257"] - values["264"]
            engines[label]["rounded_bf16_matches_observed_pair"] = rounded == engines[label]["observed_pair"]
            engines[label]["rounded_bf16_scores"] = rounded
        reports[str(case)] = {"engines": engines,
                              "changed_norm_bf16_elements": sum(a != b for a, b in zip(*norms.values(), strict=True))}
    return {"checkpoint_manifest_sha256": checkpoint_sha, "weights": weights_metadata, "cases": reports}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.model, args.trace, (args.trace / "fixtures/legacy.json").read_bytes())
    if args.output == Path("-"):
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
