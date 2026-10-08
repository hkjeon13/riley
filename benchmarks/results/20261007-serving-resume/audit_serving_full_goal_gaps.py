"""Describe every core gap from immutable independent summaries; never qualify adoption."""
import argparse,json,hashlib
from pathlib import Path
from summarize_kernel_batch01 import METRICS,direction_pass,summarize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def audit(replay_path,summary_path):
 replay=json.loads(replay_path.read_bytes());summary=json.loads(summary_path.read_bytes());assert summarize(replay)==summary,'summary does not reproduce from independent raw replay'
 expected={f'c{c}-{w}' for c in [1,8,16,32] for w in ['fixed','natural']};assert {x['cell'] for x in summary['cells']}==expected
 rows=[]
 for cell in summary['cells']:
  complete=cell['complete_four_three_engine_repeats'];comparisons=cell['complete_cell_comparisons'];mean_gaps=[];repeat_gaps=[]
  if complete:
   for metric in METRICS:
    x=comparisons['candidate_vs_vllm'][metric];change=x['change_of_four_repeat_means_percent']
    if not direction_pass(metric,change):mean_gaps.append({'metric':metric,'four_repeat_mean_change_percent':change})
    for repeat,change in enumerate(x['per_repeat_change_percent']):
     if not direction_pass(metric,change):repeat_gaps.append({'metric':metric,'repeat':repeat,'change_percent':change})
  rows.append({'cell':cell['cell'],'complete':complete,'mean_minimum_screen_pass':cell['candidate_minimum_mean_screen'],'mean_challenge_screen_pass':cell['candidate_challenge_mean_screen'],'mean_gaps_vs_vllm':mean_gaps,'individual_repeat_regressions_vs_vllm':repeat_gaps,'candidate_vs_v52_all_metrics':comparisons.get('candidate_vs_v52'),'absolute_metrics_and_repeat_spread':cell['all_available_repeat_statistics']})
 mean_pass=summary['matrix_complete'] and summary['failure'] is None and summary['pressure_policy']=='quiet' and summary['all_core_minimum_mean_screen'] is True
 return {'bindings':{'raw_replay_sha256':sha(replay_path),'summary_sha256':sha(summary_path)},'source_commits':summary['source_commits'],'full_matrix':summary['matrix_complete'],'failure':summary['failure'],'all8_cells':rows,'eligible_for_sustained_load_screen':mean_pass,'decision':'continue to actual all8 sustained/cancel/re-request/reclamation checks; per-repeat regressions remain visible and no final qualification inferred' if mean_pass else 'retain last validated reference; incomplete/failed mean screen candidate cannot be adopted','repeat_scope':'all four repeats and all rows retained; per-repeat regressions are separate descriptive evidence, not automatically erased by mean gains; no confidence inference from four samples','remaining_completion_evidence':['actual full matched launch/model/tokenizer/software contract audit','actual all8 sustained load and correctness','actual cancellation and golden re-request','actual native resource reclamation','reproducible tail nonregression considering recorded repeat variation','separate Qwen cache-on full128 correctness and free-running scope'],'serving_adoption':False,'goal_achieved':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('replay',type=Path);p.add_argument('summary',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();j=audit(a.replay,a.summary);a.output.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'full_matrix':j['full_matrix'],'eligible_for_sustained_load_screen':j['eligible_for_sustained_load_screen'],'mean_gap_counts':{x['cell']:len(x['mean_gaps_vs_vllm']) for x in j['all8_cells']},'goal_achieved':False}))
