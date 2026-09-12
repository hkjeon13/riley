"""Materialize reviewable invocation arguments; never execute a benchmark."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    c=json.loads((ROOT/'candidate.json').read_text()); e=json.loads((ROOT/'environment.json').read_text())
    p=json.loads((ROOT/'parity.json').read_text()); root=c['source_root']; gpu=c['gpu']; host=c['host']; sw=c['software']
    seed=json.loads((Path(root)/'benchmarks/prompts.jsonl').read_text().splitlines()[0])
    seed.update(prompt_id='g04-hello-p128',text='Hello'*128,target_prompt_tokens=128)
    (ROOT/'prompts.jsonl').write_text(json.dumps(seed)+'\n')
    model=e['riley_checkpoint']; revision=e['model_revision']; modelid=e['model_id']
    common_env={'LD_LIBRARY_PATH':'/data/cuda-12.8.1/lib64'}
    venv={'HF_HOME':'/data/riley-vllm-interim.CfrT9T/hf','HF_HUB_OFFLINE':'1',
          'HF_HUB_DISABLE_TELEMETRY':'1','TRANSFORMERS_OFFLINE':'1','VLLM_DO_NOT_TRACK':'1',
          'VLLM_NO_USAGE_STATS':'1','VLLM_USE_FLASHINFER_SAMPLER':'0','DO_NOT_TRACK':'1'}
    flags={
      '--model':model,'--prompts':str(ROOT/'prompts.jsonl'),'--output':'{output}/native-profile.json',
      '--role':'candidate','--pair-index':'{index}','--run-id':'g04-riley-{index}',
      '--recorded-at-utc':'{started_at_utc}','--git-commit':c['source_commit'],'--git-dirty':'false',
      '--executable-sha256':c['binaries']['riley-profile']['sha256'],'--implementation-id':'g04-owned-full-graph',
      '--runtime-flag-name':'execution_graph_policy','--runtime-flag-value':'require','--semantic-class':'E0',
      '--correctness-gate-id':'g04-full-decode-p128-o32','--correctness-report-sha256':sha(ROOT/'verification.json'),
      '--gpu-model':gpu['model'],'--gpu-uuid':gpu['uuid'],'--device-index':'0','--gpu-pci-bus-id':gpu['pci_bus_id'],
      '--gpu-compute-capability':gpu['compute_capability'],'--gpu-vram-bytes':str(gpu['vram_bytes']),
      '--environment-id':'g04-sm89-pinned-native-host','--cpu-model':host['cpu_model'],
      '--physical-core-count':str(host['physical_core_count']),'--logical-core-count':str(host['logical_core_count']),
      '--ram-bytes':str(host['ram_bytes']),'--os-release':host['os_release'],'--kernel-release':host['kernel_release'],
      '--architecture':host['architecture'],'--nvidia-driver-version':gpu['driver'],
      '--cuda-runtime-version':'12.8','--cuda-toolkit-version':sw['cuda_toolkit_version'],
      '--cublas-version':sw['cublas_version'],'--container-image-sha256':'none',
      '--workload-id':'g04-c1-p128-o32','--model-id':modelid,'--model-revision':revision,
      '--weights-sha256':e['files']['model.safetensors']['riley'],
      '--tokenizer-sha256':e['files']['tokenizer.json']['riley'],'--dtype':'bf16',
      '--concurrency':'1','--prompt-tokens':'128','--output-tokens':'32','--warmups':'5',
      '--measured-iterations':'30','--sampling-id':'greedy','--seed':'none'}
    riley=[c['binaries']['riley-profile']['path']]+[v for item in flags.items() for v in item]
    vllm=[e['executable'],str(ROOT/'vllm_lane.py'),'--matrix',root+'/benchmarks/matrix.yaml',
      '--prompts',str(ROOT/'prompts.jsonl'),'--result-dir','{output}/vllm','--run-index','{index}',
      '--run-id','g04-vllm-{index}','--warm-state','warm','--concurrency','1','--prompt-tokens','128','--output-tokens','32']
    riley_http=[c['binaries']['riley']['path'],'serve','--model',model,'--model-id','g04-smol',
      '--bind','127.0.0.1:19341','--max-active-sequences','1','--batch-token-budget','1','--prefill-chunk-tokens','1',
      '--max-sequence-tokens','160','--max-output-tokens','32','--kv-blocks','10',
      '--residual-rmsnorm','separate','--execution-completion','iteration-batch','--metadata-transport','packed-async',
      '--execution-graph-policy','require','--sampling-backend','gpu-greedy']
    vllm_http=[e['vllm_executable'],'serve',e['vllm_checkpoint'],'--tokenizer',e['vllm_checkpoint'],
      '--served-model-name','g04-smol','--host','127.0.0.1','--port','19342','--dtype','bfloat16',
      '--max-model-len','160','--max-num-seqs','1','--max-num-batched-tokens','128',
      '--gpu-memory-utilization','0.3','--no-enable-prefix-caching','--seed','0']
    immutable={item['path']:item['sha256'] for item in c['binaries'].values()}
    immutable[e['vllm_executable']]=e['vllm_executable_sha256']
    for name in ['model.safetensors','tokenizer.json']:
      immutable[str(Path(model)/name)]=e['files'][name]['riley']
      immutable[str(Path(e['vllm_checkpoint'])/name)]=e['files'][name]['vllm']
    for name in ['candidate.json','environment.json','dependencies.lock','request.json','parity.json',
                 'prompts.jsonl','verification.json','measure.py','vllm_lane.py','source.tar.gz']:
      immutable[str(ROOT/name)]=sha(ROOT/name)
    plan={'schema_version':'riley.g04.measurement-preparation.v1','measurement_started':False,
      'source_root':root,'source_commit':c['source_commit'],'immutable_files':immutable,
      'qualification_blockers':['cross-engine output token mismatch at zero-based index 8',
                                'Riley CUDA 12.8 differs from vLLM CUDA 13.0'],
      'live_preflight_last_failure':'RAM differs by 4096 bytes from the existing canonical host contract; GPU also has other compute processes',
      'comparison_status':'blocked; no M4/M5 or vLLM performance claim',
      'engine_eos_policy':'ignore-eos','http_eos_policy':'natural-eos with fixed-length receipt',
      'pairs':[{'index':i,'order':['riley','vllm'] if i%2 else ['vllm','riley']} for i in range(1,6)],
      'warmups_per_process':5,'measured_requests_per_process':30,'fresh_process_per_lane_per_pair':True,
      'engine_lanes':{'riley':{'argv':riley,'env':common_env},'vllm':{'argv':vllm,'env':venv}},
      'http_lanes':{'riley':{'argv':riley_http,'env':common_env,'port':19341,'expected_output_text':p['riley_output_text']},
                    'vllm':{'argv':vllm_http,'env':venv,'port':19342,'expected_output_text':p['vllm_output_text']}}}
    (ROOT/'measurement-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('G04_PLAN materialized=true measurement_started=false competitive_qualification=false')

if __name__=='__main__':main()
