"""Independent recomputation of the Batch14 gate-v1 matched96 attempt from retained request rows. No lane or sample is excluded."""
import json,statistics as st,sys,hashlib
from pathlib import Path
R=Path("/data/riley-serving-261007/kernel-batch14-gatev1-quiet-attempt01");O=Path("/data/riley-serving-261007/kernel-batch14-gatev1-independent-analysis-attempt01");O.mkdir(exist_ok=True)
def q(v,p):
    v=sorted(v);k=(len(v)-1)*p;f=int(k);c=min(f+1,len(v)-1);return v[f]+(v[c]-v[f])*(k-f)
engines=["batch13","candidate","vllm"];cells=[f"c{c}-{w}" for c in (1,8,16,32) for w in ("fixed","natural")]
lanes={};mismatch=[]
for cell in cells:
    for e in engines:
        for r in range(4):
            p=f"{cell}-r{r}-{e}";rows=json.loads((R/f"{p}-retained-rows.json").read_text());summ=json.loads((R/f"{p}-summary.json").read_text())
            ok=[x for x in rows if x["status"]=="success"];err=len(rows)-len(ok)
            ttft=[];tpot=[];e2e=[];tokens=0
            for x in ok:
                a=x["token_arrival_ns"];n=len(x["token_ids"]);tokens+=n
                ttft.append((a[0]-x["started_ns"])/1e6);e2e.append((x["finished_ns"]-x["started_ns"])/1e6)
                if n>1:tpot.append((a[-1]-a[0])/1e6/(n-1))
            wall=(max(x["finished_ns"] for x in rows)-min(x["started_ns"] for x in rows))/1e9
            host=json.loads((R/f"{p}-host.json").read_text());gate=json.loads((R/f"{p}-retained-host-gate.json").read_text())
            gpu=max(float(h["gpu"].split(",")[4]) for h in host);rss=max(h["server_process_memory"]["rss_sum_KiB"] for h in host)/1024
            L={"rows":len(rows),"errors":err,"tok_s":tokens/wall,"ttft":{k:q(ttft,v) for k,v in (("p50",.5),("p95",.95),("p99",.99))},"tpot":{k:q(tpot,v) for k,v in (("p50",.5),("p95",.95),("p99",.99))},"e2e":{k:q(e2e,v) for k,v in (("p50",.5),("p95",.95),("p99",.99))},
               "gpu_peak_MiB":gpu,"rss_peak_MiB":rss,"reference_all":all(x.get("reference_match") for x in rows),"exit":json.loads((R/f"{p}-exit.json").read_text())["exit"]}
            lanes[p]=L
            for name,mine,theirs in (("tok_s",L["tok_s"],summ["successful_output_tokens_per_wall_second"]),("tpot_p50",L["tpot"]["p50"],summ["token_tpot_ms"]["median"]),("ttft_p50",L["ttft"]["p50"],summ["token_ttft_ms"]["median"])):
                if abs(mine-theirs)>1e-6*max(1,abs(theirs)):mismatch.append((p,name,mine,theirs))
agg={}
for cell in cells:
    for e in engines:
        ls=[lanes[f"{cell}-r{r}-{e}"] for r in range(4)];g=lambda f:[f(l) for l in ls]
        m=lambda f:{"mean":st.mean(g(f)),"sd":st.stdev(g(f)),"min":min(g(f)),"max":max(g(f))}
        agg[(cell,e)]={"tok_s":m(lambda l:l["tok_s"]),**{f"{a}_{b}":m(lambda l,a=a,b=b:l[a][b]) for a in ("ttft","tpot","e2e") for b in ("p50","p95","p99")},"gpu":max(g(lambda l:l["gpu_peak_MiB"])),"rss":max(g(lambda l:l["rss_peak_MiB"])),"errors":sum(g(lambda l:l["errors"])),"reference_all":all(g(lambda l:l["reference_all"]))}
out=["| cell | engine | tok/s (sd) | TTFT p50/p95/p99 ms | TPOT p50 (sd) /p95/p99 ms | E2E p50/p95/p99 ms | err | GPU MiB | RSS MiB |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for cell in cells:
    for e in engines:
        a=agg[(cell,e)];out.append(f"| {cell} | {e} | {a['tok_s']['mean']:.1f} ({a['tok_s']['sd']:.1f}) | {a['ttft_p50']['mean']:.2f}/{a['ttft_p95']['mean']:.2f}/{a['ttft_p99']['mean']:.2f} | {a['tpot_p50']['mean']:.4f} ({a['tpot_p50']['sd']:.4f}) /{a['tpot_p95']['mean']:.3f}/{a['tpot_p99']['mean']:.3f} | {a['e2e_p50']['mean']:.1f}/{a['e2e_p95']['mean']:.1f}/{a['e2e_p99']['mean']:.1f} | {a['errors']} | {a['gpu']:.0f} | {a['rss']:.0f} |")
out+=["","| cell | cand/vLLM tok/s | TTFT p50 | TPOT p50 | worst TTFT/TPOT/E2E P95,P99 ratio | minimum | challenge | cand/Batch13 tok/s | TPOT p50 | worst-repeat cand TPOT vs best vLLM |","|---|---:|---:|---:|---:|:-:|:-:|---:|---:|---:|"]
verdict={}
for cell in cells:
    c,v,b=agg[(cell,"candidate")],agg[(cell,"vllm")],agg[(cell,"batch13")];r=lambda k:c[k]["mean"]/v[k]["mean"]
    tails=max(r(f"{a}_{p}") for a in ("ttft","tpot","e2e") for p in ("p95","p99"))
    mn=r("tok_s")>=1 and r("ttft_p50")<=1 and r("tpot_p50")<=1 and tails<=1;ch=r("tok_s")>=1.15 and r("ttft_p50")<=.9 and r("tpot_p50")<=.9
    verdict[cell]={"minimum":mn,"challenge":ch,"tok":r("tok_s"),"ttft":r("ttft_p50"),"tpot":r("tpot_p50"),"tails":tails}
    out.append(f"| {cell} | {r('tok_s'):.3f} | {r('ttft_p50'):.3f} | {r('tpot_p50'):.3f} | {tails:.3f} | {'pass' if mn else 'FAIL'} | {'pass' if ch else 'miss'} | {c['tok_s']['mean']/b['tok_s']['mean']:.3f} | {c['tpot_p50']['mean']/b['tpot_p50']['mean']:.3f} | {c['tpot_p50']['max']/v['tpot_p50']['min']:.3f} |")
gates=[json.loads(p.read_text()) for p in sorted(R.glob("*-host-gate.json"))]
out+=["",f"lanes {len(lanes)}; request errors {sum(l['errors'] for l in lanes.values())}; Riley reference all-match {all(l['reference_all'] for p,l in lanes.items() if not p.endswith('vllm'))}; nonzero exits {[p for p,l in lanes.items() if l['exit'] not in (0,-15)]}; controller-summary mismatches {len(mismatch)}; host gate records {len(gates)}"]
(O/"report.md").write_text("\n".join(out)+"\n");(O/"analysis.json").write_text(json.dumps({"lanes":lanes,"cells":{f"{k[0]}|{k[1]}":v for k,v in agg.items()},"verdict":verdict,"controller_mismatch":mismatch},indent=1))
print("\n".join(out));print(mismatch[:5])
