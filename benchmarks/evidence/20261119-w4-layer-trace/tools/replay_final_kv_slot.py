"""Compare a captured final KV slot with the same row's pre-bias projection."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import struct

from replay_final_head import bf16, digest, read_payload, require, round_bf16

WIDTH = 3584
KV_WIDTH = 512
HEAD = 128
PAGE = 16


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bias_head(model, weight_map, projection):
    tensor = f"model.layers.0.self_attn.{projection}_proj.bias"
    filename = weight_map[tensor]
    require(Path(filename).name == filename and filename.endswith(".safetensors"), "invalid bias shard")
    with (model / filename).open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        require(0 < length <= 16 * 1024**2, "invalid bias shard header length")
        header = stream.read(length)
        entry = json.loads(header)[tensor]
        start, end = entry["data_offsets"]
        require(entry["dtype"] == "BF16" and entry["shape"] == [KV_WIDTH] and end - start == KV_WIDTH * 2
                and 0 <= start < end <= (model / filename).stat().st_size - 8 - length, "invalid KV projection bias")
        stream.seek(8 + length + start)
        data = stream.read(KV_WIDTH * 2)
    require(len(data) == KV_WIDTH * 2, "short KV bias data")
    return bf16(data)[:HEAD], {"file": filename, "header_sha256": digest(header), "bias_sha256": digest(data)}


def rotate(key, theta, position):
    half = HEAD // 2
    first, second = [], []
    for index, (a, b) in enumerate(zip(key[:half], key[half:], strict=True)):
        inverse_freq = f32(1 / f32(theta ** (index / half)))
        angle = f32(position * inverse_freq)
        cos, sin = round_bf16(f32(math.cos(angle))), round_bf16(f32(math.sin(angle)))
        first.append(round_bf16(round_bf16(a * cos) - round_bf16(b * sin)))
        second.append(round_bf16(round_bf16(b * cos) + round_bf16(a * sin)))
    return first + second


def compare_case(trace, fixture, case, checkpoint, theta, biases):
    source = json.loads(fixture)["baseline"][case]
    directory = trace / f"native{case}"
    receipt = json.loads((directory / "receipt.json").read_bytes())
    require(receipt["status"] == "passed" and receipt["server_exit"] == 0
            and receipt["mode"] == f"native{case}", "incomplete native KV capture")
    inputs = {Path(path).name: value for path, value in receipt["sha256"].items()}
    require(inputs["riley-checkpoint.json"] == checkpoint and inputs["legacy.json"] == digest(fixture),
            "KV capture does not bind the same model and input")
    hashes = receipt["result_sha256"]
    response = json.loads(read_payload(directory, "response.json", hashes["response.json"]))
    require(response["all_original_ids_text_and_finish"] and response["response"]["prompt_tokens"] == source["prompt_tokens"],
            "diagnostic request or output changed")

    index, *rows = read_payload(directory, "index.tsv", hashes["index.tsv"]).decode().splitlines()
    marker, position, token, *_ = index.split()
    position = int(position)
    require(marker == "#" and position == len(source["prompt_tokens"]) - 1
            and int(token) == source["prompt_tokens"][-1], "captured projection row does not match the input")
    header, *table = read_payload(directory, "kv-pages.tsv", hashes["kv-pages.tsv"]).decode().splitlines()
    marker, _, last, page_size, width, kv_head = header.split()
    require((marker, int(last), int(page_size), int(width), int(kv_head)) == ("#", position, PAGE, HEAD, 0),
            "unexpected page-table shape")
    logical, slot = divmod(position, PAGE)
    ordinal, physical, valid, used = map(int, table[logical].split())
    require(len(table) == logical + 1 and ordinal == logical and physical >= 0 and 0 < slot
            and used == slot + 1 <= valid <= PAGE,
            "final logical position is outside its captured KV page")
    extents = {(stage, index): (int(start), int(size)) for stage, index, start, size in (row.split("\t") for row in rows)}
    require(len(extents) == len(rows), "duplicate trace row")
    data = read_payload(directory, "rows.bf16", hashes["rows.bf16"])

    def vector(stage, index, count):
        start, size = extents[(stage, str(index))]
        require(size == 2 * count and 0 <= start <= len(data) - size, "inconsistent BF16 vector extent")
        return bf16(data[start:start + size])

    projected = vector("qkv_prebias", 0, WIDTH + 2 * KV_WIDTH)
    key = [round_bf16(a + b) for a, b in zip(projected[WIDTH:WIDTH + HEAD], biases["k"], strict=True)]
    value_begin = WIDTH + KV_WIDTH
    value = [round_bf16(a + b) for a, b in zip(projected[value_begin:value_begin + HEAD], biases["v"], strict=True)]
    expected = {"kv_key": rotate(key, theta, position), "kv_value": value}
    results = {}
    for stage, prediction in expected.items():
        page = vector(stage, logical, used * HEAD)
        actual = page[slot * HEAD:(slot + 1) * HEAD]
        previous = page[(slot - 1) * HEAD:slot * HEAD]
        results[stage] = {"compared": HEAD, "matches": sum(a == b for a, b in zip(actual, prediction, strict=True)),
                          "previous_slot_matches": sum(a == b for a, b in zip(previous, prediction, strict=True))}
    return {"position": position, "logical_page": logical, "physical_page": physical, "slot": slot,
            "capture_sha256": hashes["rows.bf16"], "page_table_sha256": hashes["kv-pages.tsv"], "vectors": results}


def replay(model, trace):
    config = json.loads((model / "config.json").read_bytes())
    require((config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"])
            == (WIDTH, WIDTH // HEAD, KV_WIDTH // HEAD) and not config.get("rope_scaling"), "unexpected model geometry")
    theta = config["rope_theta"]
    require(theta > 0 and math.isfinite(theta), "invalid RoPE theta")
    checkpoint = digest((model / "riley-checkpoint.json").read_bytes())
    index = (model / "model.safetensors.index.json").read_bytes()
    weights = json.loads(index)["weight_map"]
    loaded = {projection: bias_head(model, weights, projection) for projection in ("k", "v")}
    fixture = (trace / "fixtures/legacy.json").read_bytes()
    return {"checkpoint_manifest_sha256": checkpoint, "weight_index_sha256": digest(index), "layer": 0, "kv_head": 0,
            "bias_inputs": {name: value[1] for name, value in loaded.items()},
            "cases": {str(case): compare_case(trace, fixture, case, checkpoint, theta,
                                             {name: value[0] for name, value in loaded.items()}) for case in (3, 9)}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.model, args.trace), indent=2, allow_nan=False))
