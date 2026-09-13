"""Offline BF16-storage/FP16-register diagnostic. Never changes installed headers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
for name in ('source', 'output', 'headers', 'nvcc'):
    parser.add_argument('--'+name, type=Path, required=True)
parser.add_argument('--zero-padding', action='store_true')
args = parser.parse_args()
source = args.source.resolve()
out = args.output.absolute()
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(source/'kernels/optional'))
from prepare_flashinfer_prefill_overlay import prepare
from experimental_prefill_half import transform
from verify_flashinfer import header_digest

receipt = prepare(args.headers, out/'headers')
header = out/'headers/include/flashinfer/attention/prefill.cuh'
header.write_text(transform(header.read_text(), zero_padding=args.zero_padding))
# The original receipt describes only the synchronized BF16 overlay. Retain it
# as provenance, and replace it with an explicit diagnostic receipt.
receipt = {'base_overlay': receipt,
           'profile': 'BF16 storage, FP16 register inputs with explicit FTZ, FP32 accumulation',
           'numerical_error_status_bit': 32, 'zero_padding': args.zero_padding,
           'prefill_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
           'header_tree_sha256': header_digest(out/'headers')}
(out/'headers/overlay.json').write_text(json.dumps(receipt, indent=2)+'\n')

adapter = (source/'kernels/optional/flashinfer_prefill.cu').read_text()
def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('adapter anchor changed: '+old)
    return text.replace(old, new)
adapter = replace_once(adapter, '  int chunk;', '  int chunk;\n  unsigned* numerical_status;')
adapter = replace_once(adapter, '  m->chunk=4096;', '  m->chunk=4096;\n  m->numerical_status=status;')
adapter = replace_once(adapter,
    'using Params=BatchPrefillPagedParams<__nv_bfloat16,__nv_bfloat16,__nv_bfloat16,int>;',
    'using Params=riley_flashinfer_prefill::HalfParams;')
adapter = replace_once(adapter, 'namespace riley_flashinfer_prefill {',
    'namespace riley_flashinfer_prefill {\nstruct HalfParams : flashinfer::BatchPrefillPagedParams<__nv_bfloat16,__nv_bfloat16,__nv_bfloat16,int> { unsigned** riley_status_slot; };')
adapter = replace_once(adapter, '    Params p;', '    Params p;\n    p.riley_status_slot=&m->numerical_status;')
(out/'adapter.cu').write_text(adapter)
probe = (source/'benchmarks/analysis/flashinfer_prefill_probe.cu').read_text()
probe = replace_once(probe, ' CHECK(cudaMemcpy(q,hq.data()', '''
 if(argc>4){
  auto* values=std::strcmp(argv[3],"q")==0?&hq:std::strcmp(argv[3],"k")==0?&hk:std::strcmp(argv[3],"v")==0?&hv:nullptr;
  if(!values)return 17;
  std::fill(values->begin(),values->end(),__float2bfloat16_rn(std::atof(argv[4])));
 }
 printf("workspace_bytes=%llu\\n",static_cast<unsigned long long>(bytes));
 CHECK(cudaMemcpy(q,hq.data()''')
probe = replace_once(probe, '  if(error)return 5;', '''
  if(argc>4){if(error!=32)return 16;printf("case=%u numerical_status=%u detected=true\\n",test,error);continue;}
  if(error)return 5;''')
(out/'probe.cu').write_text(probe)
includes = [out/'headers/include', args.headers/'cccl/libcudacxx/include',
            args.headers/'cccl/cub', args.headers/'cccl/thrust', source/'kernels/optional']
command = [str(args.nvcc), '-std=c++17', '-O3', '-lineinfo', '-arch=sm_89',
           *[f'-I{x}' for x in includes],
           str(out/'probe.cu'),
           str(out/'adapter.cu'), '-o', str(out/'probe')]
run = subprocess.run(command, capture_output=True, text=True)
(out/'build.log').write_text(run.stdout+run.stderr)
receipt.update(command=command, build_exit_code=run.returncode,
               probe_sha256=hashlib.sha256(probe.encode()).hexdigest(),
               adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest())
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
if run.returncode:
    sys.exit(run.stderr[-6000:])
print(json.dumps(receipt, indent=2))
