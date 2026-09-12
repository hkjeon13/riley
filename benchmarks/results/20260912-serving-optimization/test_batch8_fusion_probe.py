"""CPU-only rejection tests; generated records are not GPU evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('fusion',Path(__file__).with_name('batch8_fusion_probe.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def rows():
    schema='riley.batch8-fusion-native.v1'
    return [dict(schema=schema,kind='device',compute_major=8,compute_minor=9,ordinal=0,runtime_version=13000,uuid_hex=p.EXPECTED_UUID)]+[
        dict(schema=schema,kind='case',passed=True,**case,**{k:True for k in p.FLAGS},compared_bytes=[1152,1152,98304,98304],mismatch_words=[0,0,0,0]) for case in p.cases()]+[
        dict(schema=schema,kind='summary',passed=True,**p.COUNTS,device_allocations_created=103680,device_allocations_freed=103680,failed_cases=0,live_device_allocations=0,live_device_bytes=0,cleanup_errors=0,all_allocations_freed=True,stream_destroyed=True,performance_measured=False,performance_claim_eligible=False,error='')]


def validate(data):
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'synthetic.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in data))
        return p.validate_raw(path,{})


class Contracts(unittest.TestCase):
    def test_complete_synthetic_record_schema(self):
        self.assertEqual(validate(rows())[1]['cases'],8640)

    def test_rejects_incomplete_or_inexact_proof(self):
        mutations=[lambda r:r.pop(100),lambda r:r[10].update(case_id=8),
            lambda r:r[1].update(mismatch_words=[0,0,1,0]),lambda r:r[1].update(compared_bytes=[1152,1152,384,384]),
            lambda r:r[1].update(inactive_kv_unchanged=False),lambda r:r[1].update(current_values_raw_exact=False),
            lambda r:r[-1].update(device_allocations_freed=103679),lambda r:r[0].update(uuid_hex='0'*32),
            lambda r:r[-1].update(performance_measured=True)]
        for mutate in mutations:
            data=rows();mutate(data)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):validate(data)

    def test_tu_flags_remain_separate_and_both_linked(self):
        options={'fast_math':{'flags':['--generate-code=arch=compute_89,code=sm_89','--use_fast_math']},'precise':{'flags':['--generate-code=arch=compute_89,code=sm_89']}}
        manifest={'source_root':'/frozen','candidate_entry':'enqueue_compiled_packed_decode_rope_attention','probe_source':{'path':'/probe.cu'}}
        commands=p.compile_commands(manifest,Path('/nvcc'),options,Path('/build'),Path('/lib/libcudart.so'))
        self.assertIn('--use_fast_math',commands[0]);self.assertNotIn('--use_fast_math',commands[1])
        self.assertIn('/frozen/'+p.PRECISE,commands[1]);self.assertIn('/build/attention.o',commands[3]);self.assertIn('/build/precise.o',commands[3])

    def test_helpers_pinned_and_candidate_keeps_precise_oracle(self):
        self.assertEqual(set(p.helper_evidence()),set(p.HELPERS))
        source=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as d:
            oracle=Path(d)/'oracle.cu';candidate=(source/p.SOURCE).read_bytes();oracle.write_bytes(candidate[:13564])
            p.source_contract(source,oracle,'enqueue_compiled_packed_decode_rope_attention',p.sha(source/p.SOURCE))
            with self.assertRaises(ValueError):p.source_contract(source,oracle,'enqueue_compiled_packed_decode_attention_two_warp',p.sha(source/p.SOURCE))


if __name__=='__main__':unittest.main()
