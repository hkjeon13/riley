import copy,importlib.util,sys,unittest
from pathlib import Path
from test_check_native_profile_pair import ProfilePairFixture
path=Path(__file__).resolve().parents[1]/'check_vllm_profile_run.py'
spec=importlib.util.spec_from_file_location('check_vllm_profile_run',path);checker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=checker;spec.loader.exec_module(checker)
class VllmProfileTests(unittest.TestCase):
 def fixture(self):
  run=ProfilePairFixture._run(None,'candidate',1)
  run['schema_version']='riley.vllm-profile-run.v1';run['source']['runtime_flag']={'name':'graph_numerics','value':'vllm-smol-p128-v1'};run['source']['semantic_class']='VLLM_REFERENCE';run['source']['correctness_gate_id']='g04-vllm-smol-p128-v1';run['aggregate']['cuda']['stream_span_ns']={'validity':'unmeasured','value':None}
  binding={key:copy.deepcopy(run[key]) for key in ['source','environment','workload']};binding.update(input_token_ids=[19556]*128,generated_token_ids=list(range(32)))
  for row in run['requests']:row['prompt_u32le_sha256']=checker.token_hash(binding['input_token_ids']);row['generated_u32le_sha256']=checker.token_hash(binding['generated_token_ids'])
  return run,binding
 def test_valid_reference_run(self):
  run,binding=self.fixture();self.assertTrue(checker.validate(run,binding)['valid'])
 def test_rejects_e0_and_changed_provenance(self):
  for mutation in [lambda x:x['source'].__setitem__('semantic_class','E0'),lambda x:x['source'].__setitem__('git_commit','c'*40),lambda x:x.__setitem__('schema_version','riley.native-profile-run.v1')]:
   run,binding=self.fixture();mutation(run)
   with self.assertRaises(Exception):checker.validate(run,binding)
 def test_rejects_token_mismatch_missing_records_and_unqualified_timing(self):
  for mutation in [lambda x:x['requests'][0].__setitem__('generated_u32le_sha256','0'*64),lambda x:x['requests'].pop(),lambda x:x['trace'].__setitem__('dropped_records',1),lambda x:x['aggregate']['cuda'].__setitem__('stream_span_ns',{'validity':'measured','value':1})]:
   run,binding=self.fixture();mutation(run)
   with self.assertRaises(ValueError):checker.validate(run,binding)
if __name__=='__main__':unittest.main()
