"""Prepare a separate model experiment; never mutate the serving checkout."""
from pathlib import Path
import argparse,hashlib,json,shutil
parser=argparse.ArgumentParser(description='Create an isolated diagnostic residual-prefill source copy')
parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();source=args.source.resolve();target=args.output.absolute()
if source==target.resolve() or source in target.resolve().parents:raise ValueError('output must be outside source')
shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','target','__pycache__','._*'))
p=target/'kernels/optional/prepare_flashinfer_prefill_overlay.py'
s=p.read_text().replace('from verify_flashinfer import verify, header_digest','from verify_flashinfer import verify, header_digest\nfrom experimental_prefill_residual import transform')
s=s.replace('    if hashlib.sha256(patched.encode()).hexdigest()', '    patched = transform(patched)\n    if hashlib.sha256(patched.encode()).hexdigest()')
s=s.replace('996253b7c64caaa3ed86540fb5d5ca44482298c9e8c9e3665b91ba5310b0e975','e467d96c13eb05688488e68d5b408d5c37f6ed9251af15484080d2cb86c47e1e')
s=s.replace('one warp barrier before write_o_reg_gmem cross-lane shared reads','warp barrier plus FP32 denominator and two BF16 probability products; diagnostic only')
p.write_text(s)
p=target/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text().replace('flashinfer-0.6.16.post3.prefill-only.v1','flashinfer-0.6.16.post3.prefill-residual.diagnostic.v1');p.write_text(s)
for name in ['crates/riley-server/src/main.rs','crates/riley-server/src/engine.rs']:
 p=target/name;s=p.read_text().replace('flashinfer-smol-experimental-v2','flashinfer-prefill-residual-diagnostic-v1')
 if name.endswith('engine.rs'):s=s.replace('into_owned_variable_flashinfer_experimental_session','into_owned_variable_flashinfer_prefill_session')
 p.write_text(s)
files=['kernels/optional/prepare_flashinfer_prefill_overlay.py','crates/riley-runtime/src/llama/graph_decode_full.rs','kernels/optional/experimental_prefill_residual.py','crates/riley-server/src/main.rs','crates/riley-server/src/engine.rs']
receipt={name:{'source_sha256':hashlib.sha256((source/name).read_bytes()).hexdigest(),'diagnostic_sha256':hashlib.sha256((target/name).read_bytes()).hexdigest()} for name in files}
(target/'residual-experiment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(target)
