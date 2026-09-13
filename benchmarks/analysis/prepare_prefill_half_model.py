"""Prepare an isolated model experiment from the verified native adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
for name in ('source', 'output', 'adapter'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
source, target = a.source.resolve(), a.output.absolute()
if source == target.resolve() or source in target.resolve().parents:
    raise ValueError('output must be outside source')
adapter = a.adapter.read_bytes()
if hashlib.sha256(adapter).hexdigest() != '8900e10e7d8f51c6ce3da9cbd9afd14a4089875a8052f3d9ba6ca66193ce0b83':
    raise ValueError('adapter must match native v3 validation')
shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', 'target', '__pycache__', '._*'))
changed = []
def edit(name, replacements):
    path = target/name
    text = path.read_text()
    for old, new in replacements:
        if old not in text:
            raise ValueError('missing anchor in '+name+': '+old)
        text = text.replace(old, new)
    path.write_text(text)
    changed.append(name)

edit('kernels/optional/prepare_flashinfer_prefill_overlay.py', [
    ('from verify_flashinfer import verify, header_digest',
     'from verify_flashinfer import verify, header_digest\nfrom experimental_prefill_half import transform'),
    ('    if hashlib.sha256(patched.encode()).hexdigest()',
     '    patched = transform(patched, zero_padding=True)\n    if hashlib.sha256(patched.encode()).hexdigest()'),
    ('996253b7c64caaa3ed86540fb5d5ca44482298c9e8c9e3665b91ba5310b0e975',
     '52e30676f420ea9bb86e8bb091de48f4e4c4250ec94f4cebf94f40ba51f35ca0'),
    ('one warp barrier before write_o_reg_gmem cross-lane shared reads',
     'warp barrier and explicit FTZ FP16 register inputs, BF16 storage, FP32 accumulation; diagnostic')])
edit('kernels/optional/flashinfer.cmake', [('flashinfer-prefill-overlay-v1', 'flashinfer-prefill-half-register-zero-padding-v2')])
edit('crates/riley-runtime/src/llama/variable_session.rs', [('if prefill_only{33996}', 'if prefill_only{34008}')])
edit('crates/riley-runtime/src/llama/graph_decode_full.rs', [
    ('flashinfer-0.6.16.post3.prefill-only.v1', 'flashinfer-0.6.16.post3.prefill-half-register.zero-padding.diagnostic.v2')])
for name in ('crates/riley-server/src/main.rs', 'crates/riley-server/src/engine.rs'):
    replacements = [('flashinfer-smol-experimental-v2', 'flashinfer-prefill-half-register-zero-padding-diagnostic-v2')]
    if name.endswith('engine.rs'):
        replacements.append(('into_owned_variable_flashinfer_experimental_session', 'into_owned_variable_flashinfer_prefill_session'))
    edit(name, replacements)
name = 'kernels/optional/flashinfer_prefill.cu'
(target/name).write_bytes(adapter)
changed.extend([name, 'kernels/optional/experimental_prefill_half.py'])
receipt = {name: {'source_sha256': hashlib.sha256((source/name).read_bytes()).hexdigest(),
                  'diagnostic_sha256': hashlib.sha256((target/name).read_bytes()).hexdigest()}
           for name in changed}
(target/'half-register-experiment.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(target)
