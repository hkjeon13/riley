"""Correctness only: no request latency or throughput measurements."""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def request(port, prompt, stream, outputs=32):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    c.request('POST','/v1/completions',json.dumps({'model':'g04-smol','prompt':prompt,
        'temperature':0,'max_tokens':outputs,'stream':stream}),{'Content-Type':'application/json'})
    r=c.getresponse(); assert r.status==200,(r.status,r.read())
    if not stream:
        obj=json.loads(r.read()); c.close(); return obj
    text=''; finish=None; done=False
    for line in r:
        if not line.startswith(b'data: '):continue
        data=line[6:].strip()
        if data==b'[DONE]':done=True;break
        for choice in json.loads(data).get('choices',[]):
            text+=choice.get('text','')
            if choice.get('finish_reason'):finish=choice['finish_reason']
    c.close(); assert done and finish=='length'
    return text


def main():
    candidate=json.loads((ROOT/'candidate.json').read_text())
    environment=json.loads((ROOT/'environment.json').read_text())
    prompt=json.loads((ROOT/'request.json').read_text())['prompt']
    parity=json.loads((ROOT/'parity.json').read_text())
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/data/riley-g04-cuda13/lib'
    outcomes=[]
    for policy,budget,expect_ready in [('require',1,True),('auto',1,True),('disabled',1,True),('require',2,False)]:
        with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        argv=[candidate['binaries']['riley']['path'],'serve','--model',environment['riley_checkpoint'],
          '--model-id','g04-smol','--bind',f'127.0.0.1:{port}','--max-active-sequences',str(budget),
          '--batch-token-budget',str(budget),'--prefill-chunk-tokens','1','--max-sequence-tokens','160',
          '--max-output-tokens','32','--kv-blocks',str(10*budget),'--residual-rmsnorm','separate',
          '--execution-completion','iteration-batch','--metadata-transport','packed-async',
          '--execution-graph-policy',policy,'--sampling-backend','gpu-greedy']
        logpath=ROOT/f'release-http-{policy}-m{budget}.log'
        with logpath.open('w') as log:
            process=subprocess.Popen(argv,env=env,stdout=log,stderr=log,cwd=candidate['source_root'])
            try:
                deadline=time.monotonic()+120;ready=False
                while process.poll() is None and time.monotonic()<deadline:
                    try:
                        c=http.client.HTTPConnection('127.0.0.1',port,timeout=1);c.request('GET','/v1/models')
                        r=c.getresponse();r.read();c.close()
                        if r.status==200:ready=True;break
                    except OSError:pass
                    time.sleep(.1)
                assert ready==expect_ready,(policy,budget,logpath.read_text())
                if ready:
                    result=request(port,prompt,False)
                    assert result['usage']['prompt_tokens']==128
                    assert result['usage']['completion_tokens']==32
                    if policy=='require':
                        assert result['choices'][0]['text']==parity['riley_output_text']
                        assert request(port,prompt,True)==result['choices'][0]['text']
                    process.terminate();process.wait(timeout=30)
                    assert process.returncode==0,process.returncode
                else:
                    assert process.poll() is not None and process.returncode!=0
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=30)
        output=logpath.read_text()
        if ready:
            assert f'prepared={str(policy!="disabled").lower()}' in output
        outcomes.append({'policy':policy,'maximum_rows':budget,'ready':ready,'passed':True,'argv':argv})
    (ROOT/'release-http-verification.json').write_text(json.dumps({'correctness_only':True,
        'performance_trials':0,'outcomes':outcomes,'release_binary_sha256':candidate['binaries']['riley']['sha256']},indent=2)+'\n')
    print('G04_RELEASE_HTTP require=true auto_supported=true disabled=true unsupported_require_rejected=true')


if __name__=='__main__':main()
