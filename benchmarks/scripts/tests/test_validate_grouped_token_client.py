"""Replay the immutable captured failure; tests never contact HTTP or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import validate_grouped_token_client as validator

ROOT=SCRIPTS.parent/'results/20260912-serving-optimization'
RAW=ROOT/'raw/token-serving-round13/c1-b128-baseline-vllm-pair01/vllm'


class GroupedQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=validator.module(validator.evidence(SCRIPTS/'serving_token_client.py'),'test_qualification_v1')
        cls.new=validator.module(validator.evidence(SCRIPTS/'serving_token_client_v2.py'),'test_qualification_v2')
        binding=validator.read(SCRIPTS.parent/'results/20260911-g04-vllm-profile/native-binding.json')
        parent=validator.read(ROOT/'raw/batch7-http-plan.json')
        cls.reference={'model':'g04-smol','prompt_token_ids':binding['input_token_ids'],
                       'output_token_ids':binding['generated_token_ids'],'text':parent['http_lanes']['riley']['expected_output_text'],
                       'finish_reason':'length'}
        cls.files={phase:validator.evidence(RAW/(phase+'.jsonl')) for phase in validator.RAW_PHASES}

    def test_actual_successes_and_partial_group_replay_without_invented_completion(self):
        result=validator.replay_files(self.old,self.new,self.reference,self.files)
        self.assertEqual(result['successful_requests_replayed'],133)
        self.assertEqual(result['retained_successes_replayed'],123)
        partial=result['partial'][0]
        self.assertEqual(partial['replayed_prefix_tokens'],23)
        self.assertEqual(partial['group_metadata']['within_frame_zero_itl_count'],1)
        self.assertFalse(partial['completion_recovered'])
        self.assertFalse(partial['full_request_correctness_qualified'])
        self.assertFalse(result['performance_claim'])

    def test_modified_group_and_success_metrics_reject_even_with_new_file_hash(self):
        for kind in ('group','metrics'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                rows=[json.loads(line) for line in (RAW/'retained.jsonl').read_text().splitlines()]
                if kind=='group':
                    value=json.loads(rows[-1]['frames'][-1]['data']);value['choices'][0]['token_ids']=[314,999]
                    rows[-1]['frames'][-1]['data']=json.dumps(value)
                else:rows[0]['metrics']['token_tpot_ns']+=1
                path=Path(temporary)/'retained.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                files={**self.files,'retained':validator.evidence(path)}
                with self.assertRaises(ValueError):validator.replay_files(self.old,self.new,self.reference,files)


    def test_cpu_suite_is_bound_to_declared_imported_clients(self):
        refs={"client_v1":validator.evidence(SCRIPTS/'serving_token_client.py'),
              "client_v2":validator.evidence(SCRIPTS/'serving_token_client_v2.py'),
              "tests":validator.evidence(SCRIPTS/'tests/test_serving_token_client_v2.py')}
        validator.validate_client_inputs(refs)
        wrong=copy.deepcopy(refs);wrong['tests']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'26-test'):validator.validate_client_inputs(wrong)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);tests=root/'other/tests/test_serving_token_client_v2.py';tests.parent.mkdir(parents=True)
            tests.write_bytes(Path(refs['tests']['path']).read_bytes())
            wrong={**refs,'tests':validator.evidence(tests)}
            with self.assertRaises((ValueError,FileNotFoundError)):validator.validate_client_inputs(wrong)

    def test_group_metadata_cannot_disagree_with_raw_frames(self):
        row=json.loads((RAW/'warmup-stream.jsonl').read_text().splitlines()[0])
        reference=self.new.TokenReference(**self.reference)
        result,_=validator.replay(self.new,reference,row)
        result['token_delivery_groups']['within_frame_zero_itl_count']=1
        with self.assertRaisesRegex(ValueError,'group metadata'):
            validator.group_summary(self.new,result)


if __name__=='__main__':unittest.main()
