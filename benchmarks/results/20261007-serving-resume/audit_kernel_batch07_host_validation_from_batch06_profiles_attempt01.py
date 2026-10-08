"""Source-bound CPU stack presence audit; sampling counts are not serving costs."""
import pathlib,json,hashlib,sqlite3,collections,subprocess
R=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/Users/psyche/.codex/worktrees/serving-attention-batch07/riley');C=R/'kernel-batch06-cpu-capture-gap-profile-independent-collection-attempt01';P=R/'kernel-batch06-cpu-capture-gap-profile-phase-analysis-attempt01/summary.json'
old=json.loads((R/'kernel-batch06-source-receipt-attempt02.json').read_bytes());new=json.loads((R/'kernel-batch07-source-receipt-attempt01.json').read_bytes());names=[n for n in old['files'] if n.startswith('crates/')];assert len(names)==342 and all(old['files'][n]==new['files'][n] for n in names)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();paths=['crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-runtime/src/llama/variable_session.rs','crates/riley-scheduler/src/scheduler.rs'];assert all(sha(W/n)==new['files'][n] for n in paths)
assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()==new['source_commit'];assert not subprocess.check_output(['git','-C',str(W),'status','--porcelain','--untracked-files=no'])
reports=[]
for lane in json.loads(P.read_bytes())['reports']:
 folder=C/'kernel-batch06-cpu-capture-gap-profile-attempt01'/lane['case'];path=folder/'trace.sqlite';assert sha(path)==lane['sqlite_sha256'];lo,hi=next(p for p in lane['phases'] if p['phase']=='retained')['conservative_trace_window_ns'];owner=json.loads((folder/'owned-target-shutdown.json').read_bytes())['owned_pids'][0]
 with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
  strings=dict(db.execute('select id,value from StringIds'));gid=db.execute('select globalPid from PROCESSES where pid=?',(owner,)).fetchone()[0]
  samples={r[0] for r in db.execute('select id from COMPOSITE_EVENTS where start>=? and start<=? and globalTid>=? and globalTid<?',(lo,hi,gid,gid+(1<<24)))}
  symbols=collections.defaultdict(set);leaf=collections.defaultdict(set);classes=collections.defaultdict(set)
  for sample,sym,depth in db.execute('select id,symbol,stackDepth from SAMPLING_CALLCHAINS'):
   if sample not in samples:continue
   name=strings[sym]
   keys=[]
   if 'variable_wire8validate' in name:keys.append('wire_validate')
   if 'btree' in name and ('insert' in name or 'search' in name):keys.append('btree_insert_or_search')
   if 'authorize_execution' in name:keys.append('scheduler_authorize_execution')
   if name in ['__libc_malloc','_int_malloc','malloc_consolidate']:keys.append('malloc_functions')
   for key in keys:classes[key].add(sample)
   if keys:
    symbols[name].add(sample)
    if depth==0:leaf[name].add(sample)
  reports.append({'case':lane['case'],'sqlite_sha256':lane['sqlite_sha256'],'retained_target_CPU_samples':len(samples),'distinct_samples_by_class':{k:{'samples':len(v),'sample_presence_percent':100*len(v)/len(samples)} for k,v in classes.items()},'symbols':{k:{'unique_inclusive_samples':len(v),'unique_leaf_samples':len(leaf[k])} for k,v in symbols.items()}})
wire=(W/paths[0]).read_text();session=(W/paths[1]).read_text();assert 'validate(e)?;packet.fill(0);' in wire and 'pub fn validate_compact_result' in wire and 'wire::encode_into(&mut self.input,&e)?' in session and 'self.retained=Some(e)' in session and 'self.poisoned=true' in session
result={'candidate_source_commit':new['source_commit'],'profile_candidate_source_commit':old['source_commit'],'Rust_crate_files_unchanged_SHA256':342,'source_files':{n:new['files'][n] for n in paths},'CPU_profiles':reports,'phase_analysis_sha256':sha(P),'source_contract_findings':['Request encoding validates authority before clearing/writing packet.','Production execute_rows owns Expectation in self.retained through replay/read_transfer; completion validators validate authority again before publication.','Validation constructs one ownership BTreeMap and four BTreeSets per call; ROWS supported8/16/32, pool bounded1..4096.','Invalid native execution/completion poisons session and retains owner; scheduler commit confirms exact completed iteration before releasing expectation.'],'next_experiment_if_full_screen_fails':'Measure bounded ownership lookup and duplicate-set allocation alternatives with unchanged validation order/errors/packet bytes; preserve all pre-submit/pre-publication/cancel/reuse guards. No candidate implemented or adopted by this audit.','limits':['Distinct sample presence is not execution time; classes overlap and are never added.','LBR stacks may be incomplete/unresolved; no CPU/GPU causal attribution or speedup estimate.','Root diagnostic profiling UID0 differs from actual serving UID1000.','All original raw traces and samples retained; sampling presence does not qualify serving performance.'],'serving_performance':'미실행; source/profiling audit only','goal_achieved':False}
(R/'kernel-batch07-host-validation-from-batch06-profiles-attempt01.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps({'profiles':len(reports),'unchanged_Rust_files':342,'wire_presence':[{'case':r['case'],'samples':r['distinct_samples_by_class'].get('wire_validate')} for r in reports]}))
