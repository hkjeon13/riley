"""Audit full frozen HTTP request/launch contracts, independently of summaries."""
import argparse,hashlib,json,re,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL='/data/riley-serving-260913-recovery/runtime-assets-20260915/model'
EXPECTED_ENV={'HOME':'/home/psyche',
 'PATH':'/data/riley-serving-261007/vllm0271-venv/bin:/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13/bin:/usr/bin:/bin',
 'CUDA_HOME':'/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13',
 'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','CPATH':'/data/riley-serving-261007/curand-only-include',
 'CUDA_VISIBLE_DEVICES':'0','VLLM_BATCH_INVARIANT':'0',
 'LD_LIBRARY_PATH':'/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13/lib'}
BIN={
 'v52':('/data/riley-serving-261007/v52-target/release/riley','6f424278437f82a462ee141c829914d02ccb8c02ed604fa166c70c3c70bb16f6'),
 'candidate':('/data/riley-serving-261007/kernel-batch01-target/release/riley','7a2b53ad33bbd106f3e140c6b750e8f78a5f5c46f696d61a0e0e1a334eeb3c6f'),
 'vllm':('/data/riley-serving-261007/vllm0271-venv/bin/vllm','fe00035d5bd98fd57d20fe31aa6270907afbd68e92663dbc6ac9f793258e7e27')}
CANDIDATES={
 'da5ec52aefbcb4216d087ef2053974a720b54601':('/data/riley-serving-261007/kernel-batch12-target-attempt01/release/riley','cc23b7a4317c64df66c2655892702c143fc3dc6026efb57744af87fc5883e84b'),
 '73003e2d8f1ce797b4e669c813c43261b30a46a2':('/data/riley-serving-261007/kernel-batch07-target-attempt01/release/riley','be2d848f574c02b695a98e4c2f780b68bdf1b4d6b1f3ce1d31c3771cc3391f7c'),
 '61402ebc42e9d0ea5bde4e6415a1a9c25e3db8a1':('/data/riley-serving-261007/kernel-batch05-target-attempt01/release/riley','23820eb2df5006c2a522625083a86247a46731f8312cee6072be35be1be26fd6'),
 '7a8409177c9efc210b7271963e109a26a62022b8':('/data/riley-serving-261007/kernel-batch04-target-attempt01/release/riley','89d294d2be5f593607778c43111a797d70c45e7a3450bd02194478e83a461a85'),
 'b3fde5c7':BIN['candidate'],
 '3e9c979feff54b0525e198e569885e0ac0988e7f':('/data/riley-serving-261007/kernel-batch02-target/release/riley','72a1ed7eb48e4efc029ad23c37d574f723a4737f6d69fa3add3d5d88f0c759f7'),
 'ab5486ebb6cd341bdd350f927807e7dd81590202':('/data/riley-serving-261007/kernel-batch03-target-attempt01/release/riley','5353e2ab9bc9f7510122b3cba570cf7c63766e5503c0ff36fb92d11d638a5939')}
def audit(archive,source_commit):
    assert source_commit in CANDIDATES, 'candidate source not independently reviewed'
    binaries={**BIN,'candidate':CANDIDATES[source_commit]}
    metadata={};counts={};requests=0
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile():continue
            name=Path(member.name).name
            if name in ['preparation.json','plan-snapshot.json','completion.json'] or name.endswith('-launch.json'):
                assert name not in metadata;metadata[name]=json.load(tar.extractfile(member))
            match=re.fullmatch(r'(c\d+-(?:fixed|natural)-r\d+)-(v52|candidate|vllm)-(warmup|retained)-rows.json',name)
            if not match:continue
            case,engine,phase=match.groups();rows=json.load(tar.extractfile(member));actual=[]
            assert len(rows)==(96 if phase=='warmup' else 384), 'phase missing requests'
            for row in rows:
                req=row['request'];assert req['model']=='g04-smol' and req['temperature']==0
                assert req['stream'] is True and req['return_token_ids'] is True and req['stream_options']=={'include_usage':True}
                actual.append((row['corpus_id'],len(row['prompt_token_ids']),len(row['token_ids'])))
            counts.setdefault((case,phase),{})[engine]=actual;requests+=len(rows)
    prep=metadata['preparation.json'];plan=metadata['plan-snapshot.json'];assert prep['plan']==plan
    terminal=metadata['completion.json'];assert terminal['failure'] is None and terminal['all_lanes_complete'] is True and len(terminal['records'])==96
    if 'candidate_source_commit' in plan:assert plan['candidate_source_commit']==source_commit
    if 'candidate_binary_sha256' in plan:assert plan['candidate_binary_sha256']==binaries['candidate'][1]
    assert plan['concurrency']==[1,8,16,32] and plan['workloads']==['fixed','natural'] and plan['repeats']==4
    assert plan['orders']==[['v52','candidate','vllm'],['vllm','candidate','v52']]*2
    assert plan['warmup']==96 and plan['retained']==384 and plan['token_budget']==512
    assert plan['prefix_cache'] is False and plan['gpu_operating_cap_bytes']==21474836480
    assert plan['gpu_uuid']=='GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0'
    assert prep['source_commits']=={'v52':'06302d8d8396b8f2f4996fec8595bbfa1dcd7450','candidate':source_commit}
    for path,digest in binaries.values():assert prep['pins'][path]==digest
    assert prep['pins'][MODEL+'/model.safetensors']=='80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
    assert prep['pins'][MODEL+'/tokenizer.json']=='9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'
    assert prep['pins']['/data/riley-serving-261007/vllm0271-packages.txt']=='784965e0ac101c6469b50649c60ac31249ad1a5333bd6e0710824c25d1693182'
    assert hashlib.sha256(prep['software_packages'].encode()).hexdigest()==prep['pins']['/data/riley-serving-261007/vllm0271-packages.txt']
    launches=0
    for name,launch in metadata.items():
        match=re.fullmatch(r'c(\d+)-(fixed|natural)-r(\d+)-(v52|candidate|vllm)-launch.json',name)
        if not match:continue
        c,w,r,e=match.groups();c=int(c);natural=w=='natural';a=launch['argv'];path,digest=binaries[e]
        assert a[0]==path and launch['binary_sha256']==digest
        if e!='vllm':
            bind=a[a.index('--bind')+1];assert re.fullmatch(r'127\.0\.0\.1:\d+',bind)
            expected=[path,'serve','--model',MODEL,'--model-id','g04-smol','--bind',bind,
             '--max-active-sequences',str(c),'--max-waiting-requests','64','--batch-token-budget','512',
             '--prefill-chunk-tokens',str(512 if natural else 128),'--max-sequence-tokens',str(1024 if natural else 160),
             '--max-output-tokens',str(128 if natural else 32),'--kv-blocks',str(c*(64 if natural else 10)),
             '--residual-rmsnorm','separate','--execution-completion','iteration-batch','--metadata-transport','synchronous',
             '--execution-graph-policy','require','--graph-numerics','variable-smol-v7','--sampling-backend','gpu-greedy']
        else:
            port=a[a.index('--port')+1];assert port.isdecimal()
            expected=[path,'serve',MODEL,'--tokenizer',MODEL,'--served-model-name','g04-smol','--host','127.0.0.1','--port',port,
             '--dtype','bfloat16','--max-model-len',str(1024 if natural else 160),'--max-num-seqs',str(c),
             '--max-num-batched-tokens','512','--gpu-memory-utilization','0.3','--no-enable-prefix-caching','--seed','0']
        assert a==expected,'launch differs '+name
        assert launch['env']==EXPECTED_ENV,'environment differs from frozen launch contract'
        assert launch['host']['gpu'].split(',')[0].strip()==plan['gpu_uuid'],'GPU changed'
        launches+=1
    assert launches==96 and len(counts)==64 and requests==46080
    for engine_counts in counts.values():
        assert set(engine_counts)==set(binaries)
        assert engine_counts['v52']==engine_counts['candidate']==engine_counts['vllm'],'per-request token counts differ'
    return {'launches_audited':launches,'requests_audited':requests,'request_counts_equal_by_index':True,
      'frozen_cli_env_binary_model_tokenizer_software_pins_passed':True,
      'historical_vllm_dependency_parity':'unproven; this is rebuilt0.27.1 only',
      'pressure_policy':prep['pressure_policy'],'candidate_source_commit':source_commit,'candidate_binary_sha256':binaries['candidate'][1],'adopted':False,'goal_achieved':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--source-commit',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=audit(a.archive,a.source_commit);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
