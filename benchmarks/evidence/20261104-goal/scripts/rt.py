import glob,json,os,sys
root=sys.argv[1]
cols=[("reason-c8","output_tok_s"),("reason-c32","output_tok_s"),("reason-c8","tpot_p50_ms"),("reason-c32","tpot_p50_ms"),("rag-c1","ttft_p50_ms")]
def m(p,k):
    v=[json.load(open(x)).get(k) for x in p]; v=[a for a in v if a is not None]; return sum(v)/len(v) if v else None
print("model | r/v tok/s c8 | c32 | TPOT r/v c8 | c32 | rag-c1 TTFT r/v | errors")
for mo in sorted(os.listdir(root)):
    c=[];e=0
    for s,k in cols:
        r=m(glob.glob(f"{root}/{mo}/r[0-9]-{s}.json"),k); v=m(glob.glob(f"{root}/{mo}/v[0-9]-{s}.json"),k)
        c.append("-" if r is None or v is None else f"{r/v:.3f}")
    for p in glob.glob(f"{root}/{mo}/[rv][0-9]-*.json"): e+=json.load(open(p)).get("errors",0)
    print(mo,"|"," | ".join(c),"|",e)
