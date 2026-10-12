"""Diagnostic-only single-lane HTTP run (optionally with perf on the engine thread). Not a qualifying measurement."""
import argparse,json,os,signal,socket,subprocess,sys,time,http.client
from pathlib import Path
root=Path("/data/riley-serving-261007");code=root/"kernel-batch12-controller-attempt01";sys.path.insert(0,str(code))
from serving_token_client_v2 import TokenHttpClient, TokenReference
from mixed_phase_v34 import mixed_phase
a=argparse.ArgumentParser();a.add_argument("--binary",required=True);a.add_argument("--workload",required=True);a.add_argument("--concurrency",type=int,required=True);a.add_argument("--label",required=True);a.add_argument("--perf",action="store_true");a.add_argument("--retained",type=int,default=384)
args=a.parse_args();out=Path(__file__).parent/args.label;out.mkdir()
fixtures=json.loads((code/"fixtures.json").read_text());data=fixtures[args.workload];corpus=data["corpus"]
refs=[TokenReference("g04-smol",tuple(b["choices"][0]["prompt_token_ids"]) if "prompt_token_ids" in b["choices"][0] else tuple(c["prompt_token_ids"]),tuple(b["choices"][0]["token_ids"]),b["choices"][0]["text"],b["choices"][0]["finish_reason"]) for c,b in zip(corpus,data["responses"])]
model="/data/riley-serving-260913-recovery/runtime-assets-20260915/model";tool="/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13"
env={"HOME":"/home/psyche","PATH":"/usr/bin:/bin","CUDA_VISIBLE_DEVICES":"0","LD_LIBRARY_PATH":tool+"/lib"}
c=args.concurrency;nat=args.workload=="natural"
with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
argv=[args.binary,"serve","--model",model,"--model-id","g04-smol","--bind",f"127.0.0.1:{port}","--max-active-sequences",str(c),"--max-waiting-requests","64","--batch-token-budget","512","--prefill-chunk-tokens",str(512 if nat else 128),"--max-sequence-tokens",str(1024 if nat else 160),"--max-output-tokens",str(128 if nat else 32),"--kv-blocks",str((64 if nat else 10)*c),"--residual-rmsnorm","separate","--execution-completion","iteration-batch","--metadata-transport","synchronous","--execution-graph-policy","require","--graph-numerics","variable-smol-v7","--sampling-backend","gpu-greedy"]
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
log=(out/"server.log").open("w");p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);perf=None
try:
    deadline=time.monotonic()+300
    while True:
        assert p.poll() is None and time.monotonic()<deadline,"startup"
        k=http.client.HTTPConnection("127.0.0.1",port,timeout=1)
        try:
            k.request("GET","/v1/models");r=k.getresponse();r.read()
            if r.status==200:break
        except (OSError,http.client.HTTPException):pass
        finally:k.close()
        time.sleep(.1)
    with TokenHttpClient() as client:
        def one(i):
            x=corpus[i%len(corpus)];row=client.request(port,{"model":"g04-smol","prompt":x["prompt"],"max_tokens":x["max_tokens"],"temperature":0},refs[i%len(corpus)],streaming=True,mode="observe");row["corpus_id"]=x["id"];return row
        for phase,count in [("warmup",96),("retained",args.retained)]:
            if phase=="retained" and args.perf:
                tid=next(t for t in os.listdir(f"/proc/{p.pid}/task") if open(f"/proc/{p.pid}/task/{t}/comm").read().strip()=="riley-engine")
                perf=subprocess.Popen(["perf","record","-F","20000","-g","-t",tid,"-o",str(out/"perf.data")],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(.5)
            rows,account,summary=mixed_phase(client,one,concurrency=c,count=count,phase=phase,corpus_ids=[x["id"] for x in corpus])
            assert account["completed"] and account["strict_reference_pass"],"exact reference failed"
        if perf:perf.send_signal(signal.SIGINT);perf.wait()
        (out/"summary.json").write_text(json.dumps(summary,indent=1))
        med=lambda k:summary[k].get("p50",summary[k].get("median")) if isinstance(summary[k],dict) else summary[k]
        print(args.label,"tok/s",round(summary["successful_output_tokens_per_wall_second"],1),"ttft",summary["token_ttft_ms"],"tpot",summary["token_tpot_ms"],flush=True)
finally:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    p.wait(timeout=40)
