"""Bind actual correctness receipts to the isolated, clean source candidate."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

root = Path(__file__).resolve().parent
previous = Path('/tmp/riley-g04-readiness-260911')
source = Path('/tmp/riley-g04-followup-source-260911')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(name):
    return json.loads((root/name).read_text())

def write(name, value):
    (root/name).write_text(json.dumps(value, indent=2)+'\n')

def tokens(log):
    text = (root/log).read_text()
    assert 'test result: ok. 1 passed; 0 failed;' in text
    return ast.literal_eval(re.search(r'output_tokens=(\[[^\n]+\])', text)[1])

revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'], text=True).strip()
assert not subprocess.check_output(['git','-C',str(source),'diff','2cfd76c27ad1f25f5f41cfea93c2d3f5420e8a9b','--','kernels/src/primitives.cu'], text=True).strip()
runtime = read('runtime-host.json')
c = json.loads((previous/'candidate.json').read_text())
c.update(source_root=str(source), source_commit=revision, source_clean=True)
c['host']['ram_bytes'] = runtime['memtotal_kib']*1024
c['environment_id'] = 'rtx4090-ubuntu22-driver580-20260911-v2'
c['software'].update(cuda_runtime_version=str(runtime['cuda_runtime_version']),
    cuda_toolkit_version=runtime['nvcc'].splitlines()[-2], cublas_version='13.1.1')
for name in c['binaries']:
    path = Path('/tmp/riley-g04-followup-target13/release')/name
    c['binaries'][name] = {'path': str(path), 'sha256': sha(path)}
assert read('release-http-verification.json')['release_binary_sha256'] == c['binaries']['riley']['sha256']
write('candidate.json', c)
manifest = {}
with tarfile.open(root/'source.tar.gz', 'r:gz') as archive:
    for item in archive.getmembers():
        if item.isfile():
            digest = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
            assert sha(source/item.name) == digest
            manifest[item.name] = digest
write('source-manifest.json', manifest)
write('source-bundle.json', {'source_commit': revision, 'source_archive_sha256': sha(root/'source.tar.gz'),
    'source_manifest_sha256': sha(root/'source-manifest.json'), 'file_count': len(manifest),
    'original_checkout_committed': False, 'diagnostic_norm_patch_included': False})
base = read('parity.json')
t13, t128 = tokens('cuda13-owned.log'), tokens('cuda128-regression.log')
experimental = tokens('norm-only-experiment.log')
eager = read('vllm-eager-probe.json')['tokens']
assert t13 == t128 == base['riley_output_tokens']
assert experimental == eager
assert t13 == read('hf-probe.json')['hf_prefill_chunk_128']['tokens']
def first(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x != y), None)
diagnosis = {'performance_trials': 0, 'candidate_rounding_unchanged': True,
    'cuda_128_vs_130_exact_tokens': True, 'riley_matches_hf_eager_prefill128': True,
    'first_embedding_norm': read('hf-probe.json')['first_embedding_norm'],
    'isolated_norm_change_matches_vllm_eager_32_tokens': True,
    'isolated_norm_change_in_candidate': False,
    'riley_vs_default_vllm_first_mismatch': first(t13, base['vllm_output_tokens']),
    'vllm_eager_vs_default_first_mismatch': first(eager, base['vllm_output_tokens']),
    'riley_tokens': t13, 'diagnostic_norm_only_tokens': experimental,
    'vllm_eager_tokens': eager, 'vllm_default_tokens': base['vllm_output_tokens'],
    'limitation': 'Single p128/o32 diagnostic; compiled-vs-eager operator divergence is not localized. No alternative profile is qualified.'}
write('diagnosis.json', diagnosis)
assert 'test result: ok. 2 passed; 0 failed;' in (root/'http13-lifecycle.log').read_text()
assert '259 passed; 0 failed;' in (root/'runtime-cpu.log').read_text()
assert 'Ran 45 tests' in (root/'competitive-tests.log').read_text()
assert 'Ran 3 tests' in (root/'preflight-tests.log').read_text()
write('verification.json', {'source_commit': revision, 'implementation_checks_passed': True,
    'competitive_qualification_passed': False, 'performance_trials': 0,
    'qualified_scope': 'SmolLM2 c1 M=1 full decode, CUDA runtime 13.0',
    'checks': {'cuda13_graph_eager_all_logits_exact': True, 'cuda128_regression': True,
        'replays_per_owner_test': 341, 'zero_final_allocations': True,
        'release_http_policies': ['require-M1','auto-M1','disabled-M1','reject-require-M2'],
        'cpu_gpu_greedy_http_lifecycle_tests': 2, 'cpu_runtime_tests':259,
        'preflight_tests': 3, 'competitive_contract_tests':45},
    'resolved': ['CUDA runtime mismatch for the qualified M1 path', 'exact RAM snapshot identity'],
    'blockers': ['default vLLM cross-engine token mismatch', 'unrelated GPU compute sessions; preflight fails'],
    'not_qualified': ['general dense HF prefill on CUDA 13', 'CUDA13 Auto-M2 fallback',
        'diagnostic norm rounding variant', 'M4/M5', 'performance comparison'],
    'evidence': {name: sha(root/name) for name in ['diagnosis.json','cuda13-owned.log',
        'cuda128-regression.log','http13-lifecycle.log','release-http-verification.json',
        'runtime-host.json','preflight-v2.stderr','source-bundle.json']}})
print(json.dumps({'source_commit': revision, 'source_files':len(manifest), 'performance_trials':0,
                  'competitive_qualification_passed':False}))
