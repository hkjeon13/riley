"""Verify request evidence and per-prefill row census from bounded diagnostic runs."""
import hashlib,json,pathlib,re,sys,tarfile,tempfile
from ffn_row_census_analysis import parse
from serving_evidence_validation import validate_row
root=pathlib.Path(sys.argv[1]);archive=root/'evidence.tar.gz';prefix='ffn-row-census-v2/'
with tarfile.open(archive) as tar:
    members=tar.getmembers();assert len(members)==24 and len({m.name for m in members})==24
    for m in members:
        assert m.isfile() and m.name.startswith(prefix) and '/' not in m.name[len(prefix):] and pathlib.Path(m.name).suffix in ('.json','.log')
        data=tar.extractfile(m).read()
        assert not re.search(rb'(?:Bearer\s+[A-Za-z0-9._-]{20,}|--token\s+[A-Za-z0-9._-]{20,}|-----BEGIN .*PRIVATE KEY)',data)
    def read(name):return tar.extractfile(prefix+name).read()
    def load(name):return json.loads(read(name))
    sources=load('sources.json')
    assert sources['ffn_row_census.py']==hashlib.sha256(pathlib.Path(__file__).with_name('ffn_row_census.py').read_bytes()).hexdigest()
    assert sources['engine.rs']==hashlib.sha256(pathlib.Path(__file__).resolve().parents[2].joinpath('crates/riley-server/src/engine.rs').read_bytes()).hexdigest()
    assert sources['paired_decode_serving_screen.py']=='adf9930078b8c9fd4a83ef0fd5936749fb9088a5114c47644883458d5dd9cf7e'
    assert load('execution.json')==[{'name':'profile','exit_code':0}] and load('blender-restored.json')['restored']
    fixtures=load('fixtures.json');result={};total=0
    for name in ('control-shared','adaptive-shared','adaptive-unique','control-unique'):
        launch=load(name+'-launch.json');receipt=load(name+'-receipt.json')
        assert launch['binary_sha256']=='716efc2fa85e75dfde265da5dc89ff009a778e16d3892f55b947d6288005ff40'
        assert launch['controller_sha256']==sources['ffn_row_census.py'] and launch['client_sha256']==sources['paired_decode_serving_screen.py']
        assert launch['fixture_sha256']==hashlib.sha256(read('fixtures.json')).hexdigest()
        assert launch['projection_pipeline'] and launch['ffn_adaptive']==name.startswith('adaptive')
        assert launch['client_concurrency']==32 and launch['active_capacity']==32 and launch['requests']==256 and launch['kv_payload_bytes']==754974720
        assert receipt['reference_exact'] and receipt['exit_code']==0 and not receipt['remaining_owned_pids'] and 'limit_exceeded' not in receipt
        by_id={f['id']:f for f in fixtures[name.split('-')[1]]}
        for phase,count in [('warmup',64),('rows',256)]:
            rows=load(name+'-'+phase+'.json');assert len(rows)==count
            assert all(all(validate_row(r,by_id[r['id']]).values()) and len(r['token_ids'])==32 for r in rows)
            g=receipt['client_gc'][phase];assert g['policy']=='disabled' and g['before']==g['after'] and g['restored_enabled'];total+=count
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp)/'lane.log';p.write_bytes(read(name+'.log'));result[name]=parse(p)
    assert total==1280
(root/'receipt.json').write_text(json.dumps({'verified_requests':total,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'results':result},indent=2)+'\n')
for name,r in result.items():print(name,r['packed_ffn_steps'],r['m32_eligible_steps'],r['m32_eligible_step_percent'])
