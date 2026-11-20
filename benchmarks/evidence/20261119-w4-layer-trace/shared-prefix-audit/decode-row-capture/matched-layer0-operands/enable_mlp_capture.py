"""Add layer-zero MLP rows to the existing diagnostic trace."""
from argparse import ArgumentParser
from pathlib import Path


def change(path, replacements):
    source = path.read_text()
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"{path}: source anchor changed: {old.splitlines()[0]}")
        source = source.replace(old, new)
    path.write_text(source)


p = ArgumentParser(description=__doc__)
p.add_argument("stage", type=Path)
stage = p.parse_args().stage
source = stage / "src/crates/riley-runtime/src/llama/executor/dispatch.rs"
change(source, [
    ('        "layer_residual" => layer != 27,\n        "post_attention_norm" | "mlp_down" => true,',
     '        "layer_residual" => layer != 0 && layer != 27,'),
    ('''                trace_row(
                    &mut long_trace,
                    "mlp_down",''', '''                trace_row(
                    &mut long_trace,
                    "mlp_gated",
                    Some(layer_index),
                    &buffers.gated_product,
                    plan.dimensions().intermediate_size() as u64,
                    false,
                    stream,
                )?;
                trace_row(
                    &mut long_trace,
                    "mlp_down",'''),
])
change(stage / "tools/trace_decode_layers.py", [
    ('    ("attention_residual", "0", HIDDEN), ("layer_residual", "27", HIDDEN),',
     '''    ("attention_residual", "0", HIDDEN), ("post_attention_norm", "0", HIDDEN),
    ("mlp_gated", "0", 18944), ("mlp_down", "0", HIDDEN),
    ("layer_residual", "0", HIDDEN), ("layer_residual", "27", HIDDEN),'''),
])
