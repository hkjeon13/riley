"""Derive the Batch13 matched128 controller and predeclared plan from the pinned Batch12 controller."""
import json, hashlib, time
from pathlib import Path

R = Path("/data/riley-serving-261007")
C = R / "kernel-batch13-controller-attempt01"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
s = (R / "kernel-batch12-controller-attempt01/run.py").read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


rep("code = root/'kernel-batch12-controller-attempt01'", "code = root/'kernel-batch13-controller-attempt01'")
rep("out = root/f'kernel-batch12-{args.mode}", "out = root/f'kernel-batch13-{args.mode}")
# gate v2: cpu/memory thresholds stay hard; system-wide IO pressure is awaited, then recorded instead of aborting
old = s[s.index("def quiet_host_gate(prefix):"):s.index("def host_gate(prefix):")]
new = '''def quiet_host_gate(prefix):
    policy = plan['host_start_gate']; records=[]; streak=0; hard=0
    start=time.monotonic(); deadline = start+policy['timeout_seconds']
    while time.monotonic()<deadline:
        h=host(); records.append(h)
        def avg(resource, kind):
            row=next(x for x in h['pressure'][resource].splitlines() if x.startswith(kind+' '))
            return float(row.split()[1].split('=')[1])
        hard_quiet = avg('cpu','some')<=policy['cpu_some_max'] and avg('memory','full')<=policy['memory_full_max']
        io_quiet = avg('io','full')<=policy['io_full_max']
        hard=hard+1 if hard_quiet else 0
        streak=streak+1 if (hard_quiet and io_quiet) else 0
        if streak>=policy['consecutive']:
            write(prefix+'-host-gate.json', {'passed':True,'io_elevated':False,'policy':policy,'samples':records});return
        if hard>=policy['consecutive'] and time.monotonic()-start>=policy['io_soft_wait_seconds']:
            write(prefix+'-host-gate.json', {'passed':True,'io_elevated':True,'io_full_avg10_at_admission':avg('io','full'),'policy':policy,'samples':records});return
        time.sleep(policy['interval_seconds'])
    write(prefix+'-host-gate.json', {'passed':False,'policy':policy,'samples':records})
    raise RuntimeError('whole attempt aborted: host start gate timeout '+prefix)

'''
s = s.replace(old, new)
rep("binary={'v52':root/'v52-target/release/riley','candidate':root/'kernel-batch12-target-attempt01/release/riley',",
    "binary={'v52':root/'v52-target/release/riley','batch12':root/'kernel-batch12-target-attempt01/release/riley','candidate':root/'kernel-batch13-target-attempt01/release/riley',")
rep("code/'kernel-batch12-native-http-independent-verification-attempt01.json',root/'vllm0271-packages.txt'",
    "code/'kernel-batch13-correctness-attempt01.json',root/'vllm0271-packages.txt'")
a = s.index("validation=json.loads(")
b = s.index("write('validation-snapshot.json',validation)")
s = s[:a] + '''validation=json.loads((root/'kernel-batch13-server-validation-attempt01/completion.json').read_text())
http_screen=json.loads((root/'kernel-batch13-http-screen-attempt01/completion.json').read_text())
binary_receipt=json.loads((root/'kernel-batch13-server-validation-attempt01/binary-receipt.json').read_text())
assert validation['failure'] is None, 'candidate server/build/HTTP validation incomplete'
primitive=json.loads((root/'kernel-batch13-primitive-validation-attempt01/completion.json').read_text())
native=json.loads((root/'kernel-batch13-primitive-validation-attempt01/native-process.json').read_text())
assert primitive['failure'] is None and native['exit']==0 and native['native_result']['passed'] and native['native_result']['cases']==3161, 'candidate primitive validation failed'
proof=json.loads((code/'kernel-batch13-correctness-attempt01.json').read_text())
assert sha(code/'kernel-batch13-correctness-attempt01.json')==plan['correctness_sha256'], 'correctness record changed'
assert proof['source_commit']==plan['candidate_source_commit'] and proof['HTTP_exact_requests']==60 and proof['native_exact_cases']==3161, 'native/HTTP qualification incomplete'
write('primitive-validation-snapshot.json',primitive)
write('native-process-snapshot.json',native)
(out/'correctness-snapshot.json').write_bytes((code/'kernel-batch13-correctness-attempt01.json').read_bytes())
assert http_screen['failure'] is None and http_screen['requests']==60 and len(http_screen['records'])==8 and all(x['strict_reference_pass'] for x in http_screen['records']), 'candidate HTTP exact screen incomplete'
assert binary_receipt['source_commit']==plan['candidate_source_commit'] and binary_receipt['sha256']==pins[str(binary['candidate'])], 'validated candidate binary identity differs'
assert binary_receipt['sha256']==plan['candidate_binary_sha256'], 'predeclared candidate digest mismatch'
assert pins[str(binary['v52'])]=='6f424278437f82a462ee141c829914d02ccb8c02ed604fa166c70c3c70bb16f6', 'reviewed V52 digest mismatch'
assert pins[str(binary['batch12'])]=='cc23b7a4317c64df66c2655892702c143fc3dc6026efb57744af87fc5883e84b', 'Batch12 digest mismatch'
''' + s[b:]
rep("'source_commits':{'v52':'06302d8d8396b8f2f4996fec8595bbfa1dcd7450','candidate':plan['candidate_source_commit']}",
    "'source_commits':{'v52':'06302d8d8396b8f2f4996fec8595bbfa1dcd7450','batch12':'da5ec52aefbcb4216d087ef2053974a720b54601','candidate':plan['candidate_source_commit']}")
rep("if name in ('v52','candidate'):", "if name in ('v52','batch12','candidate'):", 2)
rep("len(records)==96", "len(records)==128")
(C / "run.py").write_text(s)

binr = json.loads((R / "kernel-batch13-server-validation-attempt01/binary-receipt.json").read_text())
native = json.loads((R / "kernel-batch13-primitive-validation-attempt01/native-process.json").read_text())
http = json.loads((R / "kernel-batch13-http-screen-attempt01/completion.json").read_text())
proof = {"source_commit": binr["source_commit"], "server_binary_sha256": binr["sha256"],
         "native_exact_cases": native["native_result"]["cases"], "native_binary_sha256": native["binary_after"],
         "native_log_sha256": native["log_sha256"], "BF16_and_FP32_exact": native["native_result"]["BF16_and_FP32_exact"],
         "HTTP_exact_requests": http["requests"], "HTTP_cells": [x["case"] for x in http["records"]],
         "source_receipt_sha256": sha(R / "kernel-batch13-source-receipt-attempt01.json"),
         "scope": "correctness only; native log is byte-identical to Batch12 (same sha256); device kernel sources unchanged vs Batch12"}
(C / "kernel-batch13-correctness-attempt01.json").write_text(json.dumps(proof, indent=1) + "\n")

plan = json.loads((R / "kernel-batch12-controller-attempt01/batch-plan.json").read_text())
for k in ["independent_correctness_sha256", "measured_motivation", "host_gate_revision_reason", "host_pressure_policy", "execution_state"]:
    plan.pop(k)
A = ["v52", "batch12", "candidate", "vllm"]
plan.update({
    "schema": "riley.kernel-batch13.matched128.v1",
    "scope": "All eight cells with V52 baseline / Batch12 / Batch13 candidate / reconstructed vLLM 0.27.1 interleaved in one attempt",
    "orders": [A, A[::-1], A, A[::-1]],
    "host_start_gate": {"timeout_seconds": 300, "interval_seconds": 2, "consecutive": 3, "cpu_some_max": 5,
                        "io_full_max": 5, "memory_full_max": 0.5, "io_soft_wait_seconds": 60},
    "host_gate_revision": "gate v2, a separately labelled comparison condition: CPU and memory pressure thresholds are unchanged and still abort; system-wide IO full pressure is awaited for 60 s and then recorded per lane (io_elevated) instead of aborting. Reason: Batch12 attempts 01-03 aborted on IO pressure from unrelated shared-host services; across 288 lanes of Batch09-11 the retained-phase IO pressure (0-10) showed Pearson r=0.056 with throughput deviation. No lane is excluded; io_elevated lanes are reported and compared.",
    "comparability": "V52, Batch12, Batch13 and vLLM are all re-measured under gate v2 within this attempt; results are not merged with gate v1 attempts",
    "batch_changes": ["Dense-table block ownership/aliasing validation in wire validate, block tables and execution authority (replaces BTree/HashSet/quadratic scan)",
                      "Prompt encoding on the submitting thread while the engine has active requests",
                      "Word-wise padding scan in the native shape packet validator"],
    "measured_motivation": {"source": "Batch11 nsys profiles c32-natural/c32-fixed/c1-fixed r0 candidate",
                            "finding": "host time between graph replays 16-17% of C32 wall (natural 332us, fixed 121us per decode iteration), ~40% of it prompt tokenization in admit; C1 byte-wise validator scan",
                            "limits": "diagnostic only"},
    "candidate_source_commit": binr["source_commit"], "candidate_binary_sha256": binr["sha256"],
    "correctness_sha256": sha(C / "kernel-batch13-correctness-attempt01.json"),
    "diagnostic_runs_before_plan": "kernel-batch13-diagnostic-attempt01 single-lane A/B runs guided development; they are not qualifying measurements and are preserved",
    "post_screen_required": ["independent recomputation from retained rows", "sustained load", "cancellation", "re-request", "resource reclamation"],
    "predeclared_time_ns": time.time_ns()})
(C / "batch-plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print(sha(C / "run.py"), sha(C / "batch-plan.json"))
