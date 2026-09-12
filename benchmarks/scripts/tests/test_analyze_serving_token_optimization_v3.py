"""Saved synthetic raw responses; no process, network, model or GPU execution."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_serving_token_optimization_v3 as analyzer
from test_analyze_serving_token_optimization_v2 import campaign_fixture,phase_fixture,put

c,t=analyzer.v2.controller,analyzer.v2.tokens


def read(path):
    return json.loads(path.read_text())


def partial_initial_campaign(root):
    """Five actual synthetic1000-response pairs; the next declared C2 cell never ran."""
    campaign=campaign_fixture(root)
    plan=read(root/'plan.json');prepared=read(campaign/'preparation.json')
    old_setting=plan['settings'][0]
    setting={'id':'c1','offered_concurrency':1,'vllm_token_budget':128}
    plan['settings']=[setting,old_setting]
    plan['workload'].update(purpose='initial-measurement',retained_requests_per_process=1000)
    prepared.update(workload=plan['workload'],settings=plan['settings'])
    binding=read(Path(plan['reference']['binding']['path']))
    parent=read(Path(plan['reference']['parent_c1_plan']['path']))
    reference=t.TokenReference('g04-smol',tuple(binding['input_token_ids']),tuple(binding['generated_token_ids']),
                               parent['http_lanes']['riley']['expected_output_text'])
    comparison=plan['comparisons'][0]
    entries=[]
    for index in range(1,6):
        source=campaign/f"c2-{comparison['id']}-pair{index:02d}"
        pair_dir=campaign/f"c1-{comparison['id']}-pair{index:02d}"
        source.rename(pair_dir)
        order=[comparison['left'],comparison['right']]
        if index%2==0:order.reverse()
        saved={}
        for position,name in enumerate(order):
            directory=pair_dir/name;lane=plan['lanes'][name]
            launch=read(directory/'launch.json')
            launch.update(setting=setting,argv=c.lane_argv(lane,setting,directory))
            put(directory/'launch.json',launch)
            pid=read(directory/'process.json')['pid']
            for number,phase in enumerate(('warmup-nonstream','warmup-stream','retained')):
                summary=phase_fixture(directory,reference,phase,1000 if phase=='retained' else 5,1,
                                      index*10_000_000+position*4_000_000+number*1_000_000,
                                      2 if name=='baseline' else 1,str(pid))
            exit_record=read(directory/'process-exit.json')
            exit_record['cleanup']['log']=c.evidence(directory/'server.log')
            put(directory/'process-exit.json',exit_record)
            saved[name]={'lane':name,'kind':'riley','completed':True,'setting':setting,'summary':summary,
                         'cleanup':exit_record['cleanup'],'performance_claim':False,
                         'raw':c.evidence(directory/'retained.jsonl')}
            put(directory/'summary.json',saved[name])
        result=c.pair_result(comparison,index,order,saved)
        entries.append({'setting':'c1','pair':put(pair_dir/'pair.json',result),'ratios':result['ratios']})
    prepared['plan']=put(root/'plan.json',plan)
    put(campaign/'preparation.json',prepared)
    restored=read(root/'restored.json');restored['host_or_global_configuration_modified']=False
    final={'completed_pairs':entries,'failure':{'type':'SyntheticStop','message':'next predeclared C2 cell not run'},
           'restoration':put(root/'restored.json',restored)}
    put(campaign/'finalization.json',final)
    (campaign/'completion.json').unlink()
    return campaign


class CompletedCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name).resolve()
        cls.campaign=partial_initial_campaign(cls.root)
        cls.result=analyzer.analyze(cls.campaign)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_raw_validation_emits_full_cell_without_accepting_partial_campaign(self):
        result=self.result
        self.assertTrue(result['cleanup_gate']['validated'],result['errors'])
        self.assertFalse(result['whole_campaign_completed'])
        self.assertEqual(result['campaign_analysis_v2']['status'],'incomplete')
        self.assertEqual(len(result['completed_cells']),1)
        cell=result['completed_cells'][0]
        self.assertEqual(cell['setting']['id'],'c1')
        self.assertEqual(cell['validated_pairs'],5)
        self.assertEqual(cell['retained_requests_per_process'],1000)
        self.assertEqual(cell['retained_requests_per_lane'],5000)
        self.assertEqual(cell['fresh_server_processes'],10)
        self.assertEqual(cell['paired_ratios']['token_tpot_ms_right_over_left']['median']['samples'],5)
        self.assertEqual(cell['paired_ratios']['token_tpot_ms_right_over_left']['median']['median'],.5)
        self.assertFalse(result['overall_acceptance']);self.assertFalse(result['performance_claim'])
        self.assertIsNone(result['winner'])
        self.assertTrue(all(not group['comparison_eligible'] and group['paired_ratios'] is None
                            for group in result['campaign_analysis_v2']['groups']))

    def test_unstarted_cell_and_missing_repeat_are_excluded_without_imputation(self):
        self.assertEqual(len(self.result['excluded_cells']),1)
        self.assertEqual(self.result['excluded_cells'][0]['setting']['id'],'c2')
        self.assertEqual(self.result['excluded_cells'][0]['validated_pairs'],0)
        group=copy.deepcopy(self.result['campaign_analysis_v2']['groups'][0])
        group['pairs'].pop();group['validated_pairs']=4
        core,_=analyzer.v2.private_core()
        with self.assertRaisesRegex(Exception,'all five prescribed'):
            analyzer.completed_cell(group,self.result['campaign_analysis_v2']['workload'],core.aggregate_pairs)

    def test_smaller_screen_or_selected_pair_order_cannot_be_promoted(self):
        group=copy.deepcopy(self.result['campaign_analysis_v2']['groups'][0])
        workload=copy.deepcopy(self.result['campaign_analysis_v2']['workload'])
        core,_=analyzer.v2.private_core()
        workload['retained_requests_per_process']=999
        with self.assertRaisesRegex(Exception,'1000'):
            analyzer.completed_cell(group,workload,core.aggregate_pairs)
        workload['retained_requests_per_process']=1000
        group['pairs'][1]['order'].reverse()
        with self.assertRaisesRegex(Exception,'pair order'):
            analyzer.completed_cell(group,workload,core.aggregate_pairs)

    def test_copied_campaign_keeps_original_references_and_completed_cell_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied=Path(temporary).resolve()/'copy'
            shutil.copytree(self.root,copied)
            result=analyzer.analyze(copied/'campaign',[(str(self.root),str(copied))])
            self.assertEqual(len(result['completed_cells']),1,result['errors'])
            self.assertFalse(result['whole_campaign_completed'])
            self.assertEqual(result['cleanup_gate']['restoration']['path'],str(self.root/'restored.json'))
            self.assertTrue(any(row['original_path']==str(self.root/'restored.json')
                                and row['local_path']==str(copied/'restored.json') for row in result['inputs']))

    def envelope(self):
        core,_=analyzer.v2.private_core()
        return analyzer.cleanup_envelope(self.campaign,self.result['campaign_analysis_v2'],core.Evidence())

    def test_missing_finalization_or_restoration_blocks_all_cells(self):
        path=self.campaign/'finalization.json';original=path.read_bytes()
        try:
            path.unlink()
            with self.assertRaisesRegex(Exception,'finalization'):self.envelope()
            data=json.loads(original);data['restoration']=None;put(path,data)
            with self.assertRaisesRegex(Exception,'restoration'):self.envelope()
            put(path,json.loads(original))
            restored=self.root/'restored.json';saved=restored.read_bytes()
            try:
                data=json.loads(saved);data['alive_and_listening']=False
                final=json.loads(original);final['restoration']=put(restored,data);put(path,final)
                with self.assertRaisesRegex(Exception,'restoration is incomplete'):self.envelope()
            finally:restored.write_bytes(saved)
        finally:path.write_bytes(original)

    def test_missing_exit_and_incomplete_cleanup_even_for_later_failed_lane_block(self):
        directory=self.campaign/'c2-candidate-baseline-pair01/baseline';directory.mkdir(parents=True)
        setting=self.result['campaign_analysis_v2']['groups'][1]['setting']
        launch=read(self.campaign/'c1-candidate-baseline-pair01/baseline/launch.json')
        launch['setting']=setting
        put(directory/'launch.json',launch)
        put(directory/'process.json',{'pid':9999,'session_id':9999})
        try:
            with self.assertRaisesRegex(Exception,'exit receipt'):self.envelope()
            log=directory/'server.log';log.write_text('synthetic failed later lane, no performance result')
            cleanup={'cleanup_verified':True,'remaining_owned_pids':[9999],'log':c.evidence(log)}
            put(directory/'process-exit.json',{'cleanup':cleanup,'failure':{'type':'SyntheticFailure'}})
            with self.assertRaisesRegex(Exception,'cleanup is incomplete'):self.envelope()
            cleanup['remaining_owned_pids']=[]
            put(directory/'process-exit.json',{'cleanup':cleanup,'failure':{'type':'SyntheticFailure'}})
            self.assertEqual(self.envelope()[1]['owned_processes_with_verified_cleanup'],11)
        finally:
            for path in directory.iterdir():path.unlink()
            directory.rmdir();directory.parent.rmdir()

    def test_reordered_or_extra_finalized_pairs_and_wrong_restore_scope_block(self):
        path=self.campaign/'finalization.json';original=path.read_bytes()
        try:
            final=json.loads(original);final['completed_pairs'][0],final['completed_pairs'][1]=final['completed_pairs'][1],final['completed_pairs'][0]
            put(path,final)
            with self.assertRaisesRegex(Exception,'predeclared completed-pair prefix'):self.envelope()
            restored=self.root/'restored.json';saved=restored.read_bytes()
            try:
                data=json.loads(saved);data['processes'][0]['port']=1
                final=json.loads(original);final['restoration']=put(restored,data);put(path,final)
                with self.assertRaisesRegex(Exception,'identity/port'):self.envelope()
            finally:restored.write_bytes(saved)
        finally:path.write_bytes(original)


if __name__=='__main__':
    unittest.main()
