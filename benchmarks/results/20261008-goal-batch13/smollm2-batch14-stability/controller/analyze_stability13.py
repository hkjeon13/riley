"""Summarize stability lanes from their retained evidence: errors, exactness, drains, memory trend, reclamation."""
import json, statistics as st, sys
from pathlib import Path

R = Path('/data/riley-serving-261007'); attempt = int(sys.argv[1])
plan = json.loads((R / 'kernel-batch14-stability-controller-attempt01/stability-plan.json').read_text())
out = R / f'kernel-batch14-stability-analysis-attempt{attempt:02d}'; out.mkdir(exist_ok=True)
rows = []; L = ['# Batch14 안정성 검증', '',
                f"지속 부하 {plan['sustained_seconds']}s, idle 취소 {plan['cancellation_trials_each_mode']}×2(early/decode, 매회 전체 corpus exact 재요청), 부하 중 취소 {plan['loaded_cancellations']}회. 제외 없음.", '',
                '| 조건 | 엔진 | 결과 | 지속 요청 | 오류 | exact | 처리량 첫/끝 round | TPOT p50 첫/끝 | GPU MiB 첫¼/끝¼ | RSS MiB 첫¼/끝¼ | RSS 기울기 MiB/min | drain 횟수(최대 ms) | 취소(idle/부하) | 종료 회수 |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for case, engine in plan['lanes']:
    d = R / f"{plan['output_prefix']}-{case}-{engine}-attempt{attempt:02d}"
    if not (d / 'completion.json').exists():
        rows.append({'case': case, 'engine': engine, 'state': '미실행'}); L.append(f'| {case} | {engine} | 미실행 | | | | | | | | | | | |'); continue
    c = json.loads((d / 'completion.json').read_text()); s = c['summary']; samples = json.loads((d / 'host-samples.json').read_text()); drains = json.loads((d / 'drains.json').read_text())
    rec = {'case': case, 'engine': engine, 'failure': c['failure'], 'reclaimed': c['reclaimed'], 'exit_code': c['exit_code'], 'native_close': c['native_close'], 'after_stop': c['after_stop']}
    if 'sustained' in s and 'sample_marks' in s:
        a, b = s['sample_marks']['sustained_first_sample'], s['sample_marks']['sustained_last_sample']; w = samples[a:b]; q = max(1, len(w) // 4)
        gpu = [float(h['gpu'].split(',')[4]) for h in w]; rss = [h['rss']['rss_sum_KiB'] / 1024 for h in w]; t = [(h['time_ns'] - w[0]['time_ns']) / 60e9 for h in w]
        mt, mr = st.mean(t), st.mean(rss); slope = sum((x - mt) * (y - mr) for x, y in zip(t, rss)) / sum((x - mt) ** 2 for x in t)
        rounds = s['sustained']['rounds']
        rec.update(sustained_seconds=s['sustained']['seconds'], sustained_requests=s['sustained']['requests'], sustained_rounds=len(rounds),
                   exact=sum(r['reference_matches'] for r in rounds), throughput_first=rounds[0]['throughput'], throughput_last=rounds[-1]['throughput'],
                   throughput_min=min(r['throughput'] for r in rounds), tpot_first=rounds[0]['tpot_p50'], tpot_last=rounds[-1]['tpot_p50'],
                   gpu_first_quarter=st.mean(gpu[:q]), gpu_last_quarter=st.mean(gpu[-q:]), gpu_max=max(gpu), rss_first_quarter=st.mean(rss[:q]), rss_last_quarter=st.mean(rss[-q:]),
                   rss_max=max(rss), rss_slope_MiB_per_min=slope, kv_after_rounds=sorted({r.get('kv_allocated_blocks_after_round') for r in rounds if 'kv_allocated_blocks_after_round' in r}),
                   drains=len(drains), drain_max_ms=max((x['elapsed_ms'] for x in drains), default=None))
    if 'cancellations' in s:
        k = s['cancellations']
        rec.update(idle_cancels=len(k['idle']), idle_after_token=sum(x['after_first_token'] for x in k['idle']), idle_rerequests_exact=sum(x['re_requests_exact'] for x in k['idle']),
                   loaded_cancels=len(k['under_load']), loaded_after_token=sum(x['after_first_token'] for x in k['under_load']),
                   loaded_exact_requests=sum(r['reference_matches'] for r in k['load_rounds']), loaded_requests=sum(r['requests'] for r in k['load_rounds']),
                   server_counters=s.get('server_counters'))
    rows.append(rec)
    f = lambda k, d=1: '—' if rec.get(k) is None else f'{rec[k]:.{d}f}'
    L.append(f"| {case} | {engine} | {'통과' if c['failure'] is None else '실패: ' + c['failure']['message'][:60]} | {rec.get('sustained_requests', '—')} | 0 | {rec.get('exact', '—')} | {f('throughput_first')}/{f('throughput_last')} | {f('tpot_first', 3)}/{f('tpot_last', 3)} | {f('gpu_first_quarter', 0)}/{f('gpu_last_quarter', 0)} | {f('rss_first_quarter')}/{f('rss_last_quarter')} | {f('rss_slope_MiB_per_min', 3)} | {rec.get('drains', '—')} ({f('drain_max_ms')}) | {rec.get('idle_cancels', '—')}/{rec.get('loaded_cancels', '—')} | {rec['reclaimed']} |")
ok = all(r.get('failure', 1) is None and r.get('reclaimed') for r in rows)
(out / 'analysis.json').write_text(json.dumps({'all_lanes_pass': ok, 'lanes': rows, 'plan': plan}, indent=1, ensure_ascii=False) + '\n')
L += ['', f'전체 lane 통과: {ok}', '', '한계: ' + '; '.join(plan['limits'])]
(out / 'report.md').write_text('\n'.join(L) + '\n'); print('all_lanes_pass', ok)
