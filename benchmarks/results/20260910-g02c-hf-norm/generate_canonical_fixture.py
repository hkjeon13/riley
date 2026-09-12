"""Generate a deterministic synthetic Llama checkpoint for owner wiring tests.

Uses only the pinned tokenizer/config template; no pretrained weights are reused.
This fixture provides correctness plumbing evidence, never model quality evidence.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

source, destination = map(Path, sys.argv[1:])
destination.mkdir(exist_ok=False)
tokenizer = (source / "tokenizer.json").read_bytes()
assert hashlib.sha256(tokenizer).hexdigest() == "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
config = json.loads((source / "config.json").read_bytes())
config.update(hidden_size=64, intermediate_size=128, num_attention_heads=1,
              num_key_value_heads=1, num_hidden_layers=2, max_position_embeddings=64,
              tie_word_embeddings=True)
config.pop("head_dim", None)
config.pop("_name_or_path", None)
(destination / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
(destination / "tokenizer.json").write_bytes(tokenizer)
shapes = {"model.embed_tokens.weight": [config["vocab_size"], 64], "model.norm.weight": [64]}
for layer in range(2):
    prefix = f"model.layers.{layer}."
    for name, shape in [
        ("input_layernorm.weight", [64]), ("post_attention_layernorm.weight", [64]),
        ("self_attn.q_proj.weight", [64, 64]), ("self_attn.k_proj.weight", [64, 64]),
        ("self_attn.v_proj.weight", [64, 64]), ("self_attn.o_proj.weight", [64, 64]),
        ("mlp.gate_proj.weight", [128, 64]), ("mlp.up_proj.weight", [128, 64]),
        ("mlp.down_proj.weight", [64, 128]),
    ]:
        shapes[prefix + name] = shape
header, data = {}, bytearray()
for seed, (name, shape) in enumerate(sorted(shapes.items())):
    start = len(data)
    for i in range(math.prod(shape)):
        # Normalization weights are near one; other weights are finite signed
        # BF16 values near 1/128, with distinct deterministic tensor patterns.
        word = 0x3F80 + (i % 13) if len(shape) == 1 else 0x3B00 + ((i * 17 + seed * 31) % 256)
        if len(shape) == 2 and (i + seed) % 3 == 0:
            word |= 0x8000
        data.extend(struct.pack("<H", word))
    header[name] = dict(dtype="BF16", shape=shape, data_offsets=[start, len(data)])
encoded = json.dumps(header, sort_keys=True).encode()
encoded += b" " * (-len(encoded) % 8)
(destination / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
files = []
for name in ["config.json", "model.safetensors", "tokenizer.json"]:
    payload = (destination / name).read_bytes()
    files.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
manifest = dict(format="riley-checkpoint-v1", source_model="fixture/g02c-canonical-llama-h64-l2",
                source_revision="1" * 40, converter_revision=None, transforms=[], dtype="bf16", files=files)
(destination / "riley-checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
