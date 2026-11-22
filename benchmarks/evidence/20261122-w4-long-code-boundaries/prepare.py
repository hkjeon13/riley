"""Compose the two archived plan switches and select the pinned long-code rows."""

from pathlib import Path
import sys


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"unexpected source selector: {before.splitlines()[0]}")
    return text.replace(before, after)


root = Path(sys.argv[1])
attention = Path(sys.argv[2])
down = Path(sys.argv[3])


def native_source(path):
    code, separator, _ = path.read_text().partition('\ndriver = root / "tools/run_accuracy.py"')
    if not separator:
        raise ValueError(f"archived serving script changed: {path}")
    return code


sys.argv[1:] = [str(root)]
exec(compile(native_source(attention), str(attention), "exec"), {})

old = """    status = anchor != nullptr
                 ? select_anchored_deterministic_algorithm(plan, anchor, error)"""
after_o31 = """    status = w4_attention_o31_requested(plan)
                 ? choose_w4_attention_o31(plan, error)
                 : anchor != nullptr
                 ? select_anchored_deterministic_algorithm(plan, anchor, error)"""
down_source = replace(native_source(down), 'anchor = """' + old + '"""', 'anchor = """' + after_o31 + '"""')
down_source = replace(down_source, """                 ? choose_w4_down_fp32(plan, error)
                 : anchor != nullptr""", """                 ? choose_w4_down_fp32(plan, error)
                 : w4_attention_o31_requested(plan)
                 ? choose_w4_attention_o31(plan, error)
                 : anchor != nullptr""")
exec(compile(down_source, str(down), "exec"), {})

dispatch = root / "src/crates/riley-runtime/src/llama/executor/dispatch.rs"
source = replace(dispatch.read_text(), '''        let target = match (position, token_id) {
            (677, 264) => "669-9",
            (1089, 3208) => "1023-67",
            _ => return Ok(None),
        };
        if packed.row_kind_codes().get(*sequence_row as usize)
            != Some(&LlamaBatchRowKind::Decode.code())
''', '''        let (target, kind) = match (position, token_id) {
            (7835, 25) => ("prompt-final", LlamaBatchRowKind::Prefill),
            (7842, 198) => ("decode-7", LlamaBatchRowKind::Decode),
            _ => return Ok(None),
        };
        if packed.row_kind_codes().get(*sequence_row as usize) != Some(&kind.code())
''')
source = replace(source, '''        if tensor_parallel
            || active_rows != 1
            || row != 0
            || packed.prefill_token_count() != 0
            || config.execution_completion_implementation()''',
    '''        if tensor_parallel
            || row >= active_rows
            || (target == "decode-7"
                && (active_rows != 1 || row != 0 || packed.prefill_token_count() != 0))
            || config.execution_completion_implementation()''')
dispatch.write_text(source)
