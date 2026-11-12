import json,glob,sys,os
d=sys.argv[1]
ms=sorted({os.path.basename(p).split("-base-")[0] for p in glob.glob(d+"/*-base-*-rag-c1.json")})
for m in ms:
  out=[m]
  for s,k in [("rag-c1","ttft_p50_ms"),("rag6k-c1","ttft_p50_ms"),("rag-c8","ttft_p95_ms"),("rag-c8","tpot_p50_ms"),("rag-c8","output_tok_s")]:
    v={}
    for c in ["base","fine"]:
      xs=[json.load(open(p))[k] for p in glob.glob(f"{d}/{m}-{c}-[12]-{s}.json")]
      v[c]=sum(xs)/len(xs) if xs else None
    if v["base"] and v["fine"]: out.append(f"{s}:{k.split('_')[0]} {v['base']:.1f}->{v['fine']:.1f} ({v['fine']/v['base']:.3f})")
  print(" | ".join(out))
