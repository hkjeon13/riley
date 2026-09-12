"""Generate, but never run, Round15's session helper after Round14 restoration.

The frozen Round14 lifecycle is copied without AST changes. Only the new state
paths, runtime schema and verified predecessor-chain resolver differ. A failed
serving campaign can pass the cleanup gate; this is not performance acceptance.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

CAMPAIGN = Path('/tmp/riley-opt-260912')
PREDECESSOR_SHA = 'aad60fca63db57251ba7c75f7a286b4b066c24ed2a8d8dbdf5941a30c9de585e'
BUILDER_SHA = 'c21db7598264a10735b9f38fd3dba5c69233f63fa94ebf2a00c5bafe7380b67a'
BASE_SHA = '047b133460322099851213149c623be4ef7f67a4b1ebfa7c2fedb4a70ef24712'

# This source is also executed by the CPU fixtures. It reads archived evidence;
# live PID, command, environment and mapping checks remain in frozen check().
CHAIN_SOURCE = '''def predecessor_module():
    import importlib.util
    path = CAMPAIGN / 'remote_session_round14.py'
    ensure(digest(path) == PREDECESSOR_HELPER_SHA, 'frozen round14 helper changed')
    base = Path(sys.modules['remote_session'].__file__).resolve()
    ensure(base == (CAMPAIGN / 'remote_session.py').resolve()
           and digest(base) == PREDECESSOR_BASE_SHA, 'base session helper differs')
    spec = importlib.util.spec_from_file_location('_round15_frozen_round14', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure(module.CAMPAIGN == CAMPAIGN and module.ROOT == PREVIOUS,
           'predecessor helper has different state roots')
    return module


def archived_vendor_maps(after, runtime, preserved):
    proof = after['driver_runtime']
    lines = proof['driver_mappings']
    ensure(lines, 'successor has no archived vendor mappings')
    observed = {}
    for line in lines:
        fields = line.split(None, 5)
        ensure(len(fields) == 6 and not fields[5].endswith(' (deleted)'),
               'invalid archived vendor mapping')
        path = fields[5]
        ensure(Path(path).name.startswith(VENDOR_PREFIXES) and path in runtime['files'],
               'successor mapped an unpinned vendor file')
        value = {'sha256': runtime['files'][path], 'device': fields[3], 'inode': int(fields[4])}
        ensure(value['inode'] > 0 and (path not in observed or observed[path] == value),
               'inconsistent archived vendor file identity')
        observed[path] = value
    names = {Path(path).name for path in observed}
    ensure(any(name.startswith(('libGLX_nvidia.so', 'libEGL_nvidia.so')) for name in names)
           and any(name.startswith(('libnvidia-glcore.so', 'libnvidia-eglcore.so')) for name in names),
           'successor lacks archived private GL vendor/core mappings')
    if preserved:
        ensure(proof['private_runtime_launch_verified'] is False,
               'preserved original falsely claims a new private-runtime launch')
    else:
        ensure(proof['ready'] is True and proof['all_observed_vendor_mappings_pinned'] is True
               and proof['private_vendor_files'] == observed,
               'successor private mapping inventory differs')


def qualified_successors(context_sessions):
    """Original GUI scope -> frozen14 resolver -> verified14 restoration."""
    for name, expected in GENERATION_PINS.items():
        ensure(digest(CAMPAIGN / name) == expected, 'round14 generation input changed: ' + name)
    for name, expected in PREVIOUS_PINS.items():
        ensure(digest(PREVIOUS / name) == expected, 'round14 predecessor receipt changed: ' + name)
    import importlib.util
    spec = importlib.util.spec_from_file_location('_round15_cleanup_gate', CAMPAIGN / 'build_decode_profile_batch8.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    ensure(__debug__ and builder.ROOT == CAMPAIGN and builder.campaign_gate() == GENERATION_GATE,
           'round14 cleanup gate changed since helper generation')
    predecessor = predecessor_module()
    proven = predecessor.qualified_successors(context_sessions)
    before = read(PREVIOUS / 'session.json')
    restored = read(PREVIOUS / 'verified.json')
    runtime = read(PREVIOUS / 'runtime.json')
    ensure(runtime['schema_version'] == 'riley.round14-private-runtime.v1'
           and runtime['proven_sessions'] == proven,
           'round14 predecessor-chain provenance differs')
    ensure(runtime['helpers'][str(CAMPAIGN / 'remote_session_round14.py')] == PREDECESSOR_HELPER_SHA
           and runtime['helpers'][str(CAMPAIGN / 'remote_session.py')] == PREDECESSOR_BASE_SHA,
           'round14 runtime helper binding differs')
    manifest = evidence(PREVIOUS / 'runtime.json')
    ensure(restored['runtime_manifest'] == manifest
           and restored['all_relaunched_processes_have_pinned_vendor_maps'] is True
           and restored['alive_and_listening'] is True
           and restored['commands_and_gui_environment_match'] is True
           and restored['host_or_global_configuration_modified'] is False,
           'round14 private runtime restoration incomplete')
    canonical = read(CAMPAIGN / 'blender-session.json')
    rows = restored['processes']
    ensure(len(before) == len(rows) == len(proven) == len(canonical) == 3
           and [row['pid'] for row in canonical] == PIDS
           and len({row['pid'] for row in before}) == 3
           and len({row['new_pid'] for row in rows}) == 3,
           'round14 successor inventory differs')
    result = []
    for first, prior, after, proof, port in zip(canonical, before, rows, proven, PORTS):
        ensure(same_command(first, prior)
               and (prior['pid'], prior['start'], prior['port'], prior['env'])
               == (proof['pid'], proof['start'], proof['port'], proof['gui_env'])
               and after['original_pid'] == prior['pid'] and after['port'] == port,
               'round14 successor scope differs from proven predecessor')
        ensure(type(after['new_pid']) is int and after['new_pid'] > 0
               and isinstance(after['start'], str) and after['start'].isdecimal()
               and after['runtime_manifest'] == manifest,
               'round14 successor identity/runtime binding differs')
        preserved = after['new_pid'] == prior['pid']
        if preserved:
            ensure(after['start'] == prior['start']
                   and after['runtime_mode'] == 'preserved_original_not_relaunched',
                   'preserved original identity differs')
        else:
            ensure(after['runtime_mode'] == 'validated_private_compute_and_gl_environment'
                   and isinstance(after['tag'], str) and bool(after['tag']),
                   'round14 successor launch identity differs')
        archived_vendor_maps(after, runtime, preserved)
        result.append(dict(pid=after['new_pid'], start=after['start'], port=port,
                           commands_and_gui_environment_match=True, gui_env=prior['env']))
    return result
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def load_pinned(path, expected, name):
    require(sha(path) == expected, 'frozen helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(source):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def render(source, previous_pins, generation_pins, generation_gate):
    require(hashlib.sha256(source.encode()).hexdigest() == PREDECESSOR_SHA,
            'cannot generate from an unknown predecessor source')
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'qualified_successors':
            replacements.append((node.lineno - 1, node.end_lineno, CHAIN_SOURCE + '\n'))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('ROOT', 'PREVIOUS', 'PREVIOUS_PINS'):
                value = {'ROOT': "ROOT = CAMPAIGN / 'blender-round15'\n",
                         'PREVIOUS': "PREVIOUS = CAMPAIGN / 'blender-round14'\n",
                         'PREVIOUS_PINS': 'PREVIOUS_PINS = ' + repr(dict(sorted(previous_pins.items()))) + '\n'
                           + 'PREDECESSOR_HELPER_SHA = ' + repr(PREDECESSOR_SHA) + '\n'
                           + 'PREDECESSOR_BASE_SHA = ' + repr(BASE_SHA) + '\n'
                           + 'GENERATION_PINS = ' + repr(dict(sorted(generation_pins.items()))) + '\n'
                           + 'GENERATION_GATE = ' + repr(generation_gate) + '\n'}[name]
                replacements.append((node.lineno - 1, node.end_lineno, value))
    require(len(replacements) == 4, 'unknown predecessor anchors')
    for start, end, text in sorted(replacements, reverse=True):
        lines[start:end] = [text]
    result = ''.join(lines)
    require(result.count("'riley.round14-private-runtime.v1'") == 2, 'runtime schema anchors changed')
    result = result.replace("return {'schema_version': 'riley.round14-private-runtime.v1'",
                            "return {'schema_version': 'riley.round15-private-runtime.v1'", 1)
    result = result.replace('Pause only the three verified round13 successors of the authorized first round.',
                            'Pause only the three verified round14 successors of the authorized first round.', 1)
    result = result.replace('State is private to round14.', 'State is private to round15.', 1)
    old, new = functions(source), functions(result)
    for name in old.keys() - {'qualified_successors', 'validate_runtime'}:
        require(old[name] == new[name], 'lifecycle AST changed: ' + name)
    normalized = result.replace("return {'schema_version': 'riley.round15-private-runtime.v1'",
                                "return {'schema_version': 'riley.round14-private-runtime.v1'", 1)
    require(old['validate_runtime'] == functions(normalized)['validate_runtime'], 'runtime lifecycle changed')
    compile(result, 'remote_session_round15.py', 'exec')
    return result


def gate_inputs(root, builder):
    require(__debug__, 'optimized Python disables the frozen builder assertions')
    require(builder.ROOT == root, 'campaign gate root differs')
    gate = builder.campaign_gate()
    paths = [root / 'token-serving-round14-plan.json', root / 'token-serving-round14/preparation.json',
             root / 'token-serving-round14/finalization.json', root / 'blender-session.json',
             root / 'build_decode_profile_batch8.py', root / 'remote_session_round14.py', root / 'remote_session.py']
    paths += [Path(path) for path in gate['process_exit_receipts']]
    paths += [root / 'blender-round14' / name for name in ('session.json', 'runtime.json', 'verified.json')]
    require(all(path.is_relative_to(root) and path.is_file() and not path.is_symlink() for path in paths),
            'generation inputs must be existing regular campaign files')
    pins = {str(path.relative_to(root)): sha(path) for path in paths}
    require(pins['remote_session_round14.py'] == PREDECESSOR_SHA
            and pins['remote_session.py'] == BASE_SHA
            and pins['build_decode_profile_batch8.py'] == BUILDER_SHA, 'generation helper identity differs')
    require(gate['preparation_sha256'] == pins['token-serving-round14/preparation.json']
            and gate['finalization_sha256'] == pins['token-serving-round14/finalization.json']
            and gate['restoration'] == {'path': str(root / 'blender-round14/verified.json'),
                                       'sha256': pins['blender-round14/verified.json']}
            and all(pins[str(Path(path).relative_to(root))] == digest
                    for path, digest in gate['process_exit_receipts'].items()), 'gate inputs changed during read')
    return gate, pins


def prepare(root=CAMPAIGN):
    root = Path(root).resolve()
    require(root == CAMPAIGN, 'factory is restricted to the authorized campaign root')
    output, receipt = root / 'remote_session_round15.py', root / 'round15-session-preparation.json'
    require(not output.exists() and not receipt.exists(), 'Round15 helper outputs must be fresh')
    builder = load_pinned(root / 'build_decode_profile_batch8.py', BUILDER_SHA, '_round15_campaign_gate')
    gate, pins = gate_inputs(root, builder)
    # Import only after the original session helper has been content-bound.
    sys.path.insert(0, str(root))
    predecessor = load_pinned(root / 'remote_session_round14.py', PREDECESSOR_SHA, '_round15_predecessor')
    require(Path(sys.modules['remote_session'].__file__).resolve() == root / 'remote_session.py',
            'base session helper import is shadowed')
    runtime = predecessor.bound_runtime()  # Read-only files/kernel identity; no GPU API or process signal.
    require(runtime == read(root / 'blender-round14/runtime.json'), 'Round14 runtime snapshot differs')
    previous_pins = {name: pins['blender-round14/' + name] for name in ('session.json', 'runtime.json', 'verified.json')}
    source = render((root / 'remote_session_round14.py').read_text(), previous_pins, pins, gate)
    namespace = {'__name__': '_round15_generated_preflight', '__file__': str(output)}
    exec(compile(source, str(output), 'exec'), namespace)
    contexts = read(root / 'driver-runtime-gui-probe-v2/completion.json')['blender_after']
    successors = namespace['qualified_successors'](contexts)
    require(gate_inputs(root, builder) == (gate, pins), 'generation inputs changed during preparation')
    with output.open('x') as stream:
        stream.write(source)
    result = {'schema_version': 'riley.round15-session-preparation.v1',
              'generator': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
              'helper': {'path': str(output), 'sha256': sha(output)}, 'predecessor_inputs': pins,
              'campaign_gate': gate, 'proven_successors': successors,
              'lifecycle_ast_identical': True, 'live_process_check_executed': False,
              'session_action_executed': False, 'performance_or_acceptance_inferred': False}
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=CAMPAIGN)
    print(json.dumps(prepare(parser.parse_args().root), indent=2))
