"""Build the paired fifth-wave phase diagnostic without changing request bodies."""

REQUEST = 45
ORDER = ("riley-first", "qkv-first", "qkv-second", "riley-second")
OFFSETS_MS = {"wave-early": -12, "wave-nominal": 0, "wave-late": 12}


def phase_cells(model, arm):
    if model != "qwen14" or arm not in ORDER:
        raise ValueError("phase diagnostic requires the pinned 14B model and ABBA order")
    offsets = tuple(OFFSETS_MS.items())
    if arm.endswith("-second"):
        offsets = offsets[::-1]
    return {
        name: {"prompt_words": [1500, 3000], "max_tokens": 128, "concurrency": 8,
               "requests": 40, "warmup": 8, "seed": 1, "stagger_ms": 50,
               "request_offsets_ms": {str(REQUEST): offset}}
        for name, offset in offsets
    }


def dispatch_seconds(cell, index, first):
    return ((index - first) * cell["stagger_ms"] + cell.get("request_offsets_ms", {}).get(str(index), 0)) / 1000
