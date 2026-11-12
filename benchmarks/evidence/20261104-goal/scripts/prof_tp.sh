#!/bin/bash
# prof_tp.sh: nsys of Qwen2.5-32B TP=2 (GPUs 6,7), 8 concurrent generations of 256 tokens, Riley lp11.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=$W/models/Qwen2.5-32B-Instruct; P=18285; O=evidence/proftp
NS=$W/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
export TMPDIR=$W/tmp-prof CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
mkdir -p tmp-prof $O; rm -f $O/*
setsid nohup $NS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/riley $W/bin/riley-lp11 serve --model $M --model-id m --bind 127.0.0.1:$P --inference-profile native-bf16-varlen-v1 --tensor-parallel-devices 0,1 --max-active-sequences 32 --max-output-tokens 512 --max-weight-bytes 68719476736 > logs/proftp.log 2>&1 < /dev/null &
for i in $(seq 1 1200); do curl -sf -m 2 http://127.0.0.1:$P/readyz >/dev/null && break; sleep 1; done
sleep 8
python3 - <<PY
import concurrent.futures, json, urllib.request
def call(i):
    body = json.dumps({"model": "m", "prompt": f"Write a long essay about topic number {i}.\n\n", "max_tokens": 256, "temperature": 0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$P/v1/completions", body, {"Content-Type": "application/json"}), timeout=900))["usage"]["completion_tokens"]
with concurrent.futures.ThreadPoolExecutor(8) as pool: print(list(pool.map(call, range(8))))
PY
V=$(ps -u nlp -o pid,cmd | grep "^ *[0-9]* $W/bin/riley-lp11 serve .*$P" | awk '{print $1}' | head -1); kill -INT $V
for i in $(seq 1 300); do [ -s $O/riley.nsys-rep ] && ! kill -0 $V 2>/dev/null && break; sleep 4; done; sleep 20
$NS export --type=sqlite --force-overwrite=true -o $O/riley.sqlite $O/riley.nsys-rep > /dev/null 2>&1; echo EXPORT $?
python3 - <<PY
import sqlite3, re, collections
db = sqlite3.connect("$O/riley.sqlite")
rows = db.execute("select k.start,k.end,k.deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
i = max(j for j in range(1, len(rows)) if rows[j][0] - rows[j-1][1] > 5e9); rows = rows[i:]
steps = sum(1 for r in rows if "argmax" in r[3] and r[2] == min(x[2] for x in rows))
names = collections.defaultdict(lambda: [0, 0])
for s, e, d, n in rows:
    if d != min(x[2] for x in rows[:100]): continue
    k = re.sub(r"\(.*", "", n)[:80]; names[k][0] += e - s; names[k][1] += 1
span = (rows[-1][1] - rows[0][0]) / 1e6
print("steps", steps, "span ms", round(span), "per step ms", round(span / max(steps, 1), 3))
for k, (t, c) in sorted(names.items(), key=lambda x: -x[1][0])[:14]:
    print("%8.3f ms/step %6.1f us/call %6.1f calls/step  %s" % (t / steps / 1e6, t / c / 1e3, c / steps, k))
PY
rm -rf tmp-prof; echo PROFTP_DONE
