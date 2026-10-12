"""Independent recomputation of a matched serving attempt from retained raw rows.

Recomputes every metric from per-request rows (not from controller summaries), rechecks
exact token/text references against the frozen fixtures, and applies the minimum and
challenge criteria. No lane, repeat or request is excluded.
"""
import argparse, hashlib, json, math, statistics as st, tarfile
from pathlib import Path

a = argparse.ArgumentParser()
a.add_argument("--raw", required=True)
a.add_argument("--fixtures", required=True)
a.add_argument("--out", required=True)
a.add_argument("--candidate", default="candidate")
a.add_argument("--engines", default="v52,batch12,candidate,vllm")
a.add_argument("--title", required=True)
args = a.parse_args()
raw = Path(args.raw); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
engines = args.engines.split(",")
fixtures = json.loads(Path(args.fixtures).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(values, q):
    # linear interpolation between closest ranks
    v = sorted(values); k = (len(v) - 1) * q; lo = math.floor(k); hi = math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def io_full(h):
    return float(h["pressure"]["io"].splitlines()[1].split()[1].split("=")[1])


terminal = json.loads((raw / "completion.json").read_text())
plan = json.loads((raw / "plan-snapshot.json").read_text())
manifest = {str(p.relative_to(raw)): sha(p) for p in sorted(raw.rglob("*")) if p.is_file()}
archive = out / (raw.name + ".tar.gz")
if not archive.exists():
    with tarfile.open(archive, "w:gz") as t:
        t.add(raw, arcname=raw.name)
assert manifest == {str(p.relative_to(raw)): sha(p) for p in sorted(raw.rglob("*")) if p.is_file()}, "raw changed during archive"

lanes = []; problems = []
for conc in plan["concurrency"]:
    for workload in plan["workloads"]:
        data = fixtures[workload]
        ref = {c["id"]: (b["choices"][0]["token_ids"], b["choices"][0]["text"], b["choices"][0]["finish_reason"])
               for c, b in zip(data["corpus"], data["responses"])}
        for repeat in range(plan["repeats"]):
            for engine in engines:
                case = f"c{conc}-{workload}-r{repeat}-{engine}"
                path = raw / (case + "-retained-rows.json")
                if not path.exists():
                    problems.append({"case": case, "problem": "missing retained rows"}); continue
                rows = json.loads(path.read_text())
                ok = [r for r in rows if r["status"] == "success" and r["http_status"] == 200 and r["protocol_valid"] and r["transport_complete"]]
                exact = sum(1 for r in ok if (r["token_ids"], r["text"], r["finish_reason"]) == (ref[r["corpus_id"]][0], ref[r["corpus_id"]][1], ref[r["corpus_id"]][2]))
                tokens = sum(len(r["token_ids"]) for r in ok)
                wall = (max(r["finished_ns"] for r in rows) - min(r["started_ns"] for r in rows)) / 1e9
                ttft = [(r["token_arrival_ns"][0] - r["started_ns"]) / 1e6 for r in ok]
                tpot = [(r["token_arrival_ns"][-1] - r["token_arrival_ns"][0]) / 1e6 / (len(r["token_ids"]) - 1) for r in ok if len(r["token_ids"]) > 1]
                e2e = [(r["finished_ns"] - r["started_ns"]) / 1e6 for r in ok]
                host = json.loads((raw / (case + "-host.json")).read_text())
                gate = {p: json.loads((raw / f"{case}-{p}-host-gate.json").read_text()) for p in ["start", "warmup", "retained"]}
                summary = json.loads((raw / (case + "-summary.json")).read_text())
                lane = {"case": case, "cell": f"c{conc}-{workload}", "repeat": repeat, "engine": engine,
                        "requests": len(rows), "errors": len(rows) - len(ok), "error_rate": (len(rows) - len(ok)) / len(rows),
                        "exact_reference_matches": exact, "input_tokens": sum(len(r["prompt_token_ids"]) for r in ok), "output_tokens": tokens,
                        "throughput": tokens / wall,
                        "ttft_p50": st.median(ttft), "ttft_p95": pct(ttft, .95), "ttft_p99": pct(ttft, .99),
                        "tpot_p50": st.median(tpot), "tpot_p95": pct(tpot, .95), "tpot_p99": pct(tpot, .99),
                        "e2e_p50": st.median(e2e), "e2e_p95": pct(e2e, .95), "e2e_p99": pct(e2e, .99),
                        "gpu_peak_MiB": max(float(h["gpu"].split(",")[4]) for h in host),
                        "rss_peak_MiB": max(h["server_process_memory"]["rss_sum_KiB"] for h in host) / 1024,
                        "io_elevated_gates": [p for p, g in gate.items() if g.get("io_elevated")],
                        "io_full_avg10_max_during_lane": max(io_full(h) for h in host),
                        "controller_summary_crosscheck": {
                            "throughput_rel_diff": tokens / wall / summary["successful_output_tokens_per_wall_second"] - 1,
                            "ttft_p50_diff_ms": st.median(ttft) - summary["token_ttft_ms"]["median"],
                            "tpot_p50_diff_ms": st.median(tpot) - summary["token_tpot_ms"]["median"],
                            "e2e_p50_diff_ms": st.median(e2e) - summary["e2e_ms"]["median"]}}
                if engine != "vllm" and exact != len(rows):
                    problems.append({"case": case, "problem": f"Riley exact reference {exact}/{len(rows)}"})
                if lane["errors"]:
                    problems.append({"case": case, "problem": f"{lane['errors']} request errors"})
                lanes.append(lane)

METRICS = ["throughput", "ttft_p50", "tpot_p50", "e2e_p50", "ttft_p95", "ttft_p99", "tpot_p95", "tpot_p99", "e2e_p95", "e2e_p99", "error_rate", "gpu_peak_MiB", "rss_peak_MiB"]
cells = []
for conc in plan["concurrency"]:
    for workload in plan["workloads"]:
        cell = f"c{conc}-{workload}"; stats = {}
        for engine in engines:
            sel = [l for l in lanes if l["cell"] == cell and l["engine"] == engine]
            stats[engine] = {"repeats": len(sel), **{m: {"mean": st.mean(l[m] for l in sel), "sd": st.stdev(l[m] for l in sel) if len(sel) > 1 else None,
                                                         "values": [l[m] for l in sel]} for m in METRICS}} if sel else {"repeats": 0}
        verdict = None
        if stats[args.candidate]["repeats"] == plan["repeats"] and stats["vllm"]["repeats"] == plan["repeats"]:
            c, v = stats[args.candidate], stats["vllm"]
            checks = {"throughput>=vllm": c["throughput"]["mean"] >= v["throughput"]["mean"]}
            for m in ["ttft_p50", "tpot_p50", "ttft_p95", "ttft_p99", "tpot_p95", "tpot_p99", "e2e_p95", "e2e_p99"]:
                checks[m + "<=vllm"] = c[m]["mean"] <= v[m]["mean"]
            ratios = {m: c[m]["mean"] / v[m]["mean"] for m in ["throughput", "ttft_p50", "tpot_p50", "e2e_p50", "ttft_p95", "ttft_p99", "tpot_p95", "tpot_p99", "e2e_p95", "e2e_p99"]}
            # every-repeat view: worst candidate repeat against best vLLM repeat
            strict = {"throughput": min(c["throughput"]["values"]) >= max(v["throughput"]["values"]),
                      "ttft_p50": max(c["ttft_p50"]["values"]) <= min(v["ttft_p50"]["values"]),
                      "tpot_p50": max(c["tpot_p50"]["values"]) <= min(v["tpot_p50"]["values"])}
            verdict = {"checks": checks, "minimum_pass": all(checks.values()), "ratios_candidate_over_vllm": ratios,
                       "challenge_pass": ratios["throughput"] >= 1.15 and ratios["ttft_p50"] <= 0.90 and ratios["tpot_p50"] <= 0.90,
                       "worst_candidate_repeat_vs_best_vllm_repeat": strict}
        cells.append({"cell": cell, "stats": stats, "verdict": verdict})

complete = terminal["failure"] is None and len(lanes) == len(plan["concurrency"]) * len(plan["workloads"]) * plan["repeats"] * len(engines)
result = {"title": args.title, "raw": str(raw), "raw_files": len(manifest), "archive": str(archive), "archive_sha256": sha(archive),
          "controller_terminal_failure": terminal["failure"], "lanes": len(lanes), "matrix_complete": complete, "problems": problems,
          "exclusions": "none", "candidate": args.candidate,
          "all_cells_minimum_pass": complete and not problems and all(c["verdict"] and c["verdict"]["minimum_pass"] for c in cells),
          "all_cells_challenge_pass": complete and not problems and all(c["verdict"] and c["verdict"]["challenge_pass"] for c in cells),
          "io_elevated_lanes": [l["case"] for l in lanes if l["io_elevated_gates"]],
          "max_abs_controller_crosscheck": {k: max(abs(l["controller_summary_crosscheck"][k]) for l in lanes) for k in ["throughput_rel_diff", "ttft_p50_diff_ms", "tpot_p50_diff_ms", "e2e_p50_diff_ms"]} if lanes else None,
          "stability": "별도 검증", "cells": cells, "lane_records": lanes}
(out / "analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
(out / "raw-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

label = {"v52": "V52 기준", "batch12": "Batch12", "candidate": "새 후보", "vllm": "vLLM 0.27.1"}
f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
ms = lambda s, m, d=3: f"{f(s[m]['mean'], d)} ± {f(s[m]['sd'], d)}"
L = [f"# {args.title}", "",
     f"각 수치는 {plan['repeats']}회 반복의 평균 ± 표본 표준편차이며 모든 반복·요청을 유지했습니다(제외 없음). 분위수는 각 반복의 통계치를 평균한 값입니다.",
     f"원자료 {len(manifest)}개 파일, archive SHA256 `{result['archive_sha256']}`. 요청 행으로부터 독립 재계산했고 controller 요약과의 최대 차이: {json.dumps(result['max_abs_controller_crosscheck'])}.",
     f"matrix 완료: {complete} / 문제: {len(problems)}건 / 최소 기준 전체 통과: {result['all_cells_minimum_pass']} / 도전 목표 전체 통과: {result['all_cells_challenge_pass']}",
     f"IO pressure 상승 상태로 입장한 lane(gate v2, 제외하지 않음): {len(result['io_elevated_lanes'])}개", "",
     "| 조건 | 엔진 | 출력 tok/s | TTFT p50 ms | TPOT p50 ms | E2E p50 ms |", "|---|---|---:|---:|---:|---:|"]
for c in cells:
    for e in engines:
        s = c["stats"][e]
        if s["repeats"]: L.append(f"| {c['cell']} | {label.get(e, e)} | {ms(s, 'throughput', 1)} | {ms(s, 'ttft_p50')} | {ms(s, 'tpot_p50')} | {ms(s, 'e2e_p50')} |")
L += ["", "| 조건 | 엔진 | TTFT P95 | TTFT P99 | TPOT P95 | TPOT P99 | E2E P95 | E2E P99 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for c in cells:
    for e in engines:
        s = c["stats"][e]
        if s["repeats"]: L.append(f"| {c['cell']} | {label.get(e, e)} | " + " | ".join(ms(s, m, 2) for m in ["ttft_p95", "ttft_p99", "tpot_p95", "tpot_p99", "e2e_p95", "e2e_p99"]) + " |")
L += ["", "| 조건 | 엔진 | 오류율 | GPU peak MiB | RSS peak MiB |", "|---|---|---:|---:|---:|"]
for c in cells:
    for e in engines:
        s = c["stats"][e]
        if s["repeats"]: L.append(f"| {c['cell']} | {label.get(e, e)} | {ms(s, 'error_rate', 4)} | {ms(s, 'gpu_peak_MiB', 0)} | {ms(s, 'rss_peak_MiB', 1)} |")
L += ["", "새 후보 / vLLM 비율과 판정 (처리량은 ≥1, 지연은 ≤1 이어야 최소 기준 통과; 도전 목표는 처리량 ≥1.15, TTFT·TPOT ≤0.90):", "",
      "| 조건 | 처리량 | TTFT p50 | TPOT p50 | TTFT P95/P99 | TPOT P95/P99 | E2E P95/P99 | 최소 | 도전 | 실패 항목 |", "|---|---:|---:|---:|---:|---:|---:|:-:|:-:|---|"]
for c in cells:
    v = c["verdict"]
    if not v: L.append(f"| {c['cell']} | 미완료 | | | | | | | | |"); continue
    r = v["ratios_candidate_over_vllm"]
    L.append(f"| {c['cell']} | {f(r['throughput'])} | {f(r['ttft_p50'])} | {f(r['tpot_p50'])} | {f(r['ttft_p95'], 2)}/{f(r['ttft_p99'], 2)} | {f(r['tpot_p95'], 2)}/{f(r['tpot_p99'], 2)} | {f(r['e2e_p95'], 2)}/{f(r['e2e_p99'], 2)} | {'통과' if v['minimum_pass'] else '미달'} | {'통과' if v['challenge_pass'] else '미달'} | {', '.join(k for k, ok in v['checks'].items() if not ok)} |")
if problems:
    L += ["", "문제 목록:"] + [f"- {p['case']}: {p['problem']}" for p in problems]
(out / "report.md").write_text("\n".join(L) + "\n")
print(json.dumps({k: result[k] for k in ["matrix_complete", "lanes", "all_cells_minimum_pass", "all_cells_challenge_pass", "controller_terminal_failure"]}, ensure_ascii=False), len(problems))
