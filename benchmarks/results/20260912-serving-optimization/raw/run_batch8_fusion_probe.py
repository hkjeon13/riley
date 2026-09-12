"""Execute frozen batch8 standalone GPU correctness; desktop remains untouched."""
import importlib.util
import json
from pathlib import Path
import hashlib
import time
ROOT=Path('/tmp/riley-opt-260912')
PINS={'batch8_fusion_probe.py': 'c025519755c4ce73bf61d2aa80a3fc36ec6ae745843fb0995058c312a8b74904', 'remote_session_round13_v2.py': '33101f68f4b1d8d1c3e04fa2db236c22bc3edbb2f687b7eea633f0931deed65b'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name):
 p=ROOT/(name+'.py');assert sha(p)==PINS[p.name]
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
probe=module('batch8_fusion_probe')
session=module('remote_session_round13_v2')
runtime=session.validate_runtime()
before=session.check(runtime)
binding=ROOT.parent/'riley-g04-vllm-profile-260911/native-binding.json'
expected=json.loads(binding.read_text())['environment']['gpu']['uuid'].removeprefix('GPU-').replace('-','')
assert expected==probe.EXPECTED_UUID
started=time.time()
try:
 result=probe.run_probe(ROOT/'batch8-fusion-probe/fixtures.json',session.COMPUTE/'extracted/usr/lib/x86_64-linux-gnu')
 assert result['device']['uuid_hex']==expected
 assert probe.validate_receipt(ROOT/'batch8-fusion-probe/receipt.json',ROOT/'batch8-build.json')==result
finally:
 after=session.check()
 assert before==after
assert runtime==session.validate_runtime()
receipt={'completed':True,'gpu_tests_executed':True,'performance_measured':False,'performance_claim_eligible':False,
 'started_unix':started,'finished_unix':time.time(),'probe_receipt':probe.evidence(ROOT/'batch8-fusion-probe/receipt.json'),
 'native_binding':probe.evidence(binding),'runtime_receipts':runtime['receipts'] if 'receipts' in runtime else [probe.evidence(session.COMPUTE/'receipt.json')],
 'helpers':{name:probe.evidence(ROOT/name) for name in PINS},'blender_unchanged':True,
 'verified_pids':[row['pid'] for row in after],'candidate_device_uuid':expected}
probe.write_new(ROOT/'batch8-fusion-probe/execution-context.json',receipt)
print(json.dumps(receipt))
