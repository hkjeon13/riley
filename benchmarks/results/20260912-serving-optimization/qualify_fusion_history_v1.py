"""Fresh model qualification for fusion-history-v1; frozen helpers stay unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import qualify_batch8 as frozen
import batch8_fusion_probe as probe
import build_fusion_history_candidate_v1 as builder

ROOT = builder.ROOT
COMMIT = '4c5bcff43d1942fdd3c396b2b9bcd9df3bf63593'
require, read, sha, evidence = frozen.require, frozen.read, frozen.sha, frozen.evidence

def inputs():
    parent = frozen.validate_build(ROOT)
    require(parent['source_commit'] == builder.BASE_COMMIT, 'parent commit differs')
    path = ROOT / 'fusion-history-build-v1.json'
    build = read(path)
    source = ROOT / 'fusion-history-source-v1'
    target = ROOT / 'fusion-history-target-v1'
    require(build['source_commit'] == COMMIT and build['source_root'] == str(source), 'candidate identity differs')
    require(build['parent_source_commit'] == builder.BASE_COMMIT and
            build['parent_build_sha256'] == sha(ROOT / 'batch8-build.json') == builder.BASE_SHA, 'parent binding differs')
    require(build['source_files'] == {**parent['source_files'], builder.GRAPH: builder.CANDIDATE_SHA}
            and build['changed_files'] == [builder.GRAPH], 'candidate scope differs')
    frozen.check_tree(source, COMMIT, build['source_files'])
    frozen.exact_diff(source, builder.BASE_COMMIT, {builder.GRAPH})
    require(frozen.git(source, 'rev-parse', 'HEAD^') == builder.BASE_COMMIT, 'candidate is not direct child')
    env = {**parent['build_environment'], 'CARGO_TARGET_DIR': str(target)}
    require(build['build_environment'] == env and build['build_argv'] == frozen.build_commands(), 'build command differs')
    require(build['builder_sha256'] == sha(Path(builder.__file__)) and
            build['build_log_sha256'] == sha(ROOT / 'fusion-history-build-v1.log'), 'build evidence changed')
    require(build['binaries'] == {str(target / 'release' / n): sha(target / 'release' / n)
                                 for n in ('riley', 'riley-profile')}, 'binary changed')
    fusion = probe.validate_receipt(ROOT / 'fusion-history-probe-v1/receipt.json', path)
    require(all(fusion[k] == build[k] for k in ('source_root', 'source_commit', 'source_files', 'binaries')), 'probe build mismatch')
    binding, refs = frozen.reference(frozen.BASE)
    library = ROOT / 'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu'
    runtime = frozen.verify_runtime(ROOT, library)
    fixed = {**env, 'LD_LIBRARY_PATH': str(library) + ':/data/riley-g04-cuda13/lib',
             'RILEY_REAL_CHECKPOINT': str(frozen.frozen.MODEL), 'CUDA_VISIBLE_DEVICES': '0', 'CARGO_TERM_COLOR': 'never'}
    helpers = {name: evidence(Path(module.__file__).resolve()) for name, module in
               [('runner', __import__(__name__)), ('model_contract', frozen), ('test_contract', frozen.frozen),
                ('validation', frozen.prior), ('probe', probe), ('builder', builder)]}
    return dict(build=evidence(path), source_root=str(source), source_commit=COMMIT,
                source_files=build['source_files'], binaries=build['binaries'], references=refs,
                fusion_probe=evidence(ROOT / 'fusion-history-probe-v1/receipt.json'),
                runtime=runtime, runtime_environment=fixed, helpers=helpers), binding

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'validate'])
    args = parser.parse_args()
    context, binding = inputs()
    path = ROOT / 'fusion-history-model-tests-v1.json'
    contracts = frozen.commands()
    logs = [ROOT / ('fusion-history-v1-' + name + '.log') for name, _, _ in contracts]
    if args.command == 'run':
        require(not os.environ.get('LD_PRELOAD'), 'unexpected preload')
        require(not any(p.exists() for p in [path, *logs]), 'refusing to replace evidence')
        env = {**os.environ, **context['runtime_environment']}
        env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
        gpu = frozen.gpu_identity(context['runtime'], binding, env)
        results = []
        for log, (name, argv, is_gpu) in zip(logs, contracts):
            require(inputs() == (context, binding), 'inputs changed before test')
            with log.open('x') as out:
                subprocess.run(argv, cwd=context['source_root'], env=env, stdout=out, stderr=out, check=True)
            counts = frozen.frozen.validate_test_log(log, name if is_gpu else None)
            require(inputs() == (context, binding), 'inputs changed after test')
            require(frozen.gpu_identity(context['runtime'], binding, env) == gpu, 'GPU changed')
            results.append(dict(name=name, argv=argv, gpu_test=is_gpu, log=evidence(log), counts=counts))
        result = dict(schema_version='riley.fusion-history-model-tests.v1', passed=True, **context,
                      gpu=gpu, checks=results, gpu_tests_executed=True, http_correctness_qualified=False,
                      performance_measured=False, performance_claim_eligible=False)
        frozen.write_new(path, result)
    result = read(path)
    require(result['schema_version'] == 'riley.fusion-history-model-tests.v1' and result['passed'] is True, 'receipt failed')
    require(all(result[k] == v for k, v in context.items()), 'receipt input mismatch')
    require(len(result['checks']) == len(contracts), 'test count differs')
    for item, log, (name, argv, is_gpu) in zip(result['checks'], logs, contracts):
        require(item == dict(name=name, argv=argv, gpu_test=is_gpu, log=evidence(log),
                            counts=frozen.frozen.validate_test_log(log, name if is_gpu else None)), 'test evidence mismatch')
    require(result['gpu_tests_executed'] is True and all(result[k] is False for k in
            ('http_correctness_qualified', 'performance_measured', 'performance_claim_eligible')), 'scope differs')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
