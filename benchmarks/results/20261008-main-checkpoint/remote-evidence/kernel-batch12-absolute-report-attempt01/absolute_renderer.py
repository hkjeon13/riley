"""Absolute metrics only, from a complete independently replayed matrix."""
import json,math
ENGINES=[('vllm','vLLM 0.27.1'),('v52','V52 기준'),('candidate','Batch12 후보')]
def fmt(stats):
 assert stats['n']==4 and math.isfinite(stats['mean']) and math.isfinite(stats['sample_sd'])
 return f"{stats['mean']:.3f} ± {stats['sample_sd']:.3f}"
def table(summary,columns):
 lines=['| 조건 | 엔진 | '+' | '.join(label for key,label in columns)+' |','|---|---|'+'|'.join(['---:']*len(columns))+'|']
 for cell in summary['cells']:
  assert cell['complete_four_three_engine_repeats']
  for name,label in ENGINES:
   stats=cell['all_available_repeat_statistics'][name]
   lines.append('| '+cell['cell']+' | '+label+' | '+' | '.join(fmt(stats[key]) for key,_ in columns)+' |')
 return '\n'.join(lines)
def render(summary,evidence):
 assert summary['matrix_complete'] and summary['verified_lane_count']==96 and summary['failure'] is None
 assert len(summary['cells'])==8
 lines=['Batch12 실제 HTTP serving 비교입니다. 각 수치는 4회 반복의 평균 ± 표본 표준편차이며, 모든 반복과 요청을 유지했습니다.',
 'TTFT/TPOT/E2E 중앙값과 P95/P99는 각 반복의 해당 통계치를 평균한 값입니다. 요청을 합쳐 계산한 분위수나 신뢰구간은 아닙니다.',
 'vLLM 0.27.1 재구성 비교이며 이전 vLLM 버전의 역사적 기준과 구분합니다. 안정성 검증은 미실행이고 후보 채택·목표 달성은 미확정입니다.','']
 lines.append(table(summary,[('throughput_tokens_s','출력 tok/s'),('ttft_ms_median','TTFT 중앙값 ms'),('tpot_ms_median','TPOT 중앙값 ms'),('e2e_ms_median','E2E 중앙값 ms')]))
 lines+=['','지연 tail (ms):','']
 lines.append(table(summary,[(m+'_'+s,label+' '+s.upper()+' ms') for m,label in [('ttft_ms','TTFT'),('tpot_ms','TPOT'),('e2e_ms','E2E')] for s in ['p95','p99']]))
 lines+=['','오류율은 실패/요청 비율입니다. 메모리는 각 반복에서 관측한 peak의 평균 ± 표본 표준편차입니다. GPU는 장치 전체 사용량이며 RSS는 process group 합계로 공유 페이지가 중복 집계될 수 있습니다. 샘플링 간격 사이의 순간 최대는 증명하지 않습니다.','']
 lines.append(table(summary,[('error_rate','오류율 (0–1)'),('gpu_peak_observed_MiB','GPU peak MiB'),('cpu_process_group_peak_observed_RSS_MiB','Process group RSS peak MiB')]))
 lines+=['','실제 token 수 및 반복 기록:','','| 조건 | 반복 | 엔진별 입력 token 합계 | 엔진별 출력 token 합계 |','|---|---:|---:|---:|']
 for cell in summary['cells']:
  for pair in cell['per_repeat_records']:
   lines.append(f"| {cell['cell']} | {pair['repeat']} | {pair['input_tokens_each']} | {pair['output_tokens_each']} |")
 lines+=['','평균 방향 기준 screen: '+str(summary['all_core_minimum_mean_screen'])+'. 이는 전체 안정성·정확성 완료 판정이 아닙니다.',
 'Source revisions: '+json.dumps(summary['source_commits'],ensure_ascii=False),
 'Raw archive (ssh ai-assistant): '+evidence['archive_path'],
 'Archive SHA256: '+evidence['archive_sha256'],
 'Summary SHA256: '+evidence['summary_sha256'],
 'Raw replay SHA256: '+evidence['replay_sha256'],
 'Host audit completion SHA256: '+evidence['host_completion_sha256'],
 '모든 launch/model/tokenizer/software pin, 제외 없는 반복별 결과는 동일 원자료 archive와 독립 검증 결과에 보존되어 있습니다.']
 return '\n'.join(lines)+'\n'
