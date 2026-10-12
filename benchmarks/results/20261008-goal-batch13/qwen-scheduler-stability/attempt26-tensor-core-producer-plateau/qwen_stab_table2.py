import json
rows=[]
for c in (1,8,16,32):
    for w in ("fixed","natural"):
        d=json.load(open("/data/riley-serving-261007/qwen-scheduler-stability-c%d-%s-attempt%s/result.json"%(c,w,__import__("sys").argv[1]))); s=d["sustained"]; m=d["memory"]; ca=d["cancellation"]
        t=s["tok_per_s_first_last_min"]; p=s["tpot_p50_first_last"]; g=m["gpu_MiB_first_last_quarter"]; r=m["rss_MiB_first_last_quarter"]
        rows.append("| c%d-%s | %s | %d | %d | %d | %.1f / %.1f (min %.1f) | %.2f / %.2f | %.0f / %.0f | %.1f / %.1f | %s / %s | %s |"%(c,w,"통과" if d["pass"] else "실패",s["requests"],s["errors"],s["inconsistent"],t[0],t[1],t[2],p[0],p[1],g[0],g[1],r[0],r[1],ca["idle"],ca["under_load"],s["kv_after_rounds"]))
        if c==32 and w=="natural":
            extra={k:(v if len(json.dumps(v))<200 else "...") for k,v in d.items() if k not in ("sustained","memory","cancellation")}; cx={k:v for k,v in ca.items() if not isinstance(v,list)}
print("| 조건 | 결과 | 요청 | 오류 | 불일치 | tok/s 첫 / 끝 round (최소) | TPOT p50 첫 / 끝 ms | GPU MiB 첫¼ / 끝¼ | RSS MiB 첫¼ / 끝¼ | 취소 idle / 부하 | round 뒤 KV 블록 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
print("\n".join(rows)); print("EXTRA",json.dumps(extra)[:700]); print("CANCEL",json.dumps(cx)[:500])
