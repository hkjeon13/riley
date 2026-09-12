"""Prepare Round16 from unchanged Round14 lifecycle and verified Round15 successors."""
import ast
import json
from pathlib import Path
import sys
import prepare_remote_session_round15 as previous_factory

ROOT = Path('/tmp/riley-opt-260912')
LIFECYCLE_SHA = previous_factory.PREDECESSOR_SHA
PREDECESSOR_SHA = 'a7e8b194bd0d0bccd4ef0631dc8b295a718d8478266882cc0eeebb2716a222d2'
BUILDER_SHA = '57a62964d4d656a8caf873cf98285656486e39ea410d6e395896f041149a54b5'
FACTORY_SHA = 'e7ee002a0c2d77d76e2262b24703501fe9b9ea305491894d31460879096ca876'
require, sha, read, load = previous_factory.require, previous_factory.sha, previous_factory.read, previous_factory.load_pinned


def inputs():
    require(sha(ROOT/'prepare_remote_session_round15.py') == FACTORY_SHA, 'previous factory changed')
    builder = load(ROOT/'build_fusion_history_candidate_v1.py', BUILDER_SHA, '_round16_gate')
    require(__debug__ and builder.ROOT == ROOT, 'gate context differs')
    gate = builder.gate()
    completion = read(ROOT/'paired-rope-attention-round15/completion.json')
    paths = [ROOT/name for name in ('remote_session.py', 'remote_session_round14.py',
        'remote_session_round15.py', 'prepare_remote_session_round15.py',
        'build_fusion_history_candidate_v1.py', 'blender-session.json',
        'paired-rope-attention-round15/completion.json', 'paired-rope-attention-round15/finalization.json')]
    paths += [ROOT/'blender-round15'/name for name in ('session.json', 'runtime.json', 'verified.json')]
    paths += [Path(x['completion']['path']) for x in completion['processes']]
    require(all(p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink() for p in paths), 'nonregular generation input')
    pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    require(pins['remote_session_round14.py'] == LIFECYCLE_SHA and pins['remote_session_round15.py'] == PREDECESSOR_SHA
            and pins['remote_session.py'] == previous_factory.BASE_SHA, 'lifecycle helper changed')
    return gate, pins


def render(source, gate, pins):
    require(previous_factory.hashlib.sha256(source.encode()).hexdigest() == LIFECYCLE_SHA, 'unknown lifecycle')
    chain = previous_factory.CHAIN_SOURCE.replace('round15', 'round16').replace('round14', 'round15')
    chain = chain.replace('build_decode_profile_batch8.py', 'build_fusion_history_candidate_v1.py').replace('builder.campaign_gate()', 'builder.gate()')
    replacements = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'qualified_successors':
            replacements.append((node.lineno-1,node.end_lineno,chain+'\n'))
        elif isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            values={'ROOT': "ROOT = CAMPAIGN / 'blender-round16'\n", 'PREVIOUS': "PREVIOUS = CAMPAIGN / 'blender-round15'\n",
                'PREVIOUS_PINS': 'PREVIOUS_PINS = '+repr({n:pins['blender-round15/'+n] for n in ('session.json','runtime.json','verified.json')})+'\n'
                +'PREDECESSOR_HELPER_SHA = '+repr(PREDECESSOR_SHA)+'\n'
                +'PREDECESSOR_BASE_SHA = '+repr(previous_factory.BASE_SHA)+'\n'
                +'GENERATION_PINS = '+repr(pins)+'\n'+'GENERATION_GATE = '+repr(gate)+'\n'}
            if name in values: replacements.append((node.lineno-1,node.end_lineno,values[name]))
    require(len(replacements)==4, 'lifecycle anchors differ')
    lines=source.splitlines(keepends=True)
    for first,last,text in sorted(replacements,reverse=True): lines[first:last]=[text]
    result=''.join(lines).replace("return {'schema_version': 'riley.round14-private-runtime.v1'", "return {'schema_version': 'riley.round16-private-runtime.v1'",1)
    result=result.replace('Pause only the three verified round13 successors of the authorized first round.', 'Pause only the three verified round15 successors of the authorized first round.',1).replace('State is private to round14.', 'State is private to round16.',1)
    old,new=previous_factory.functions(source),previous_factory.functions(result)
    for name in old.keys()-{'qualified_successors','validate_runtime'}: require(old[name]==new[name], 'lifecycle changed: '+name)
    normalized=result.replace("return {'schema_version': 'riley.round16-private-runtime.v1'", "return {'schema_version': 'riley.round14-private-runtime.v1'",1)
    require(old['validate_runtime']==previous_factory.functions(normalized)['validate_runtime'], 'runtime lifecycle changed')
    names=[n.name for n in ast.parse(result).body if isinstance(n,ast.FunctionDef)]
    require(len(names)==len(set(names)), 'duplicate function definitions')
    return result


def main():
    output,receipt=ROOT/'remote_session_round16.py',ROOT/'round16-session-preparation.json'
    require(not output.exists() and not receipt.exists(), 'refusing overwrite')
    gate,pins=inputs()
    predecessor=load(ROOT/'remote_session_round15.py',PREDECESSOR_SHA,'_round16_predecessor')
    require(predecessor.bound_runtime()==read(ROOT/'blender-round15/runtime.json'), 'predecessor runtime changed')
    source=render((ROOT/'remote_session_round14.py').read_text(),gate,pins)
    namespace={'__name__':'_round16_preflight','__file__':str(output)}
    exec(compile(source,str(output),'exec'),namespace)
    contexts=read(ROOT/'driver-runtime-gui-probe-v2/completion.json')['blender_after']
    successors=namespace['qualified_successors'](contexts)
    require(inputs()==(gate,pins), 'generation inputs changed')
    with output.open('x') as f: f.write(source)
    result=dict(schema_version='riley.round16-session-preparation.v1',generator={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        helper={'path':str(output),'sha256':sha(output)},predecessor_inputs=pins,campaign_gate=gate,proven_successors=successors,
        lifecycle_ast_identical=True,live_process_check_executed=False,session_action_executed=False,performance_or_acceptance_inferred=False)
    with receipt.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'prepared':True,'helper':result['helper'],'session_action_executed':False}))

if __name__=='__main__': main()
