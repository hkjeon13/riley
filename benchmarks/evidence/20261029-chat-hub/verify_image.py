"""Verify the riley distribution image on one A100 (run under the GPU-6 holder lease).

A: CLI smoke without GPU.  B: fresh preset download into a new named volume, serve,
restart (must reuse).  C: every preset dir from EFS served by the image with the
preset defaults, and by the measured host binary of the same commit; greedy outputs
must be identical.
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

IMG = sys.argv[1]
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
GPU = '6'
EFS = '/mnt/efs/nlp/psyche/riley-261013/models'
HOST_BIN = '/mnt/efs/nlp/psyche/riley-261013/bin/riley-mc-b74361d3'
PRESETS = json.loads(Path(sys.argv[3]).read_text())  # name -> EFS dir
ONLY = sys.argv[4].split(',') if len(sys.argv) > 4 else None
PROMPTS = ['The capital of France is', 'def fibonacci(n):']
report = {}

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def running(name):
    return sh(['docker', 'inspect', '-f', '{{.State.Running}}', name]).stdout.strip() == 'true'

def ready(port, deadline, proc=None, name=None):
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        if name is not None and not running(name):
            return False
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/readyz', timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def complete(port, model):
    texts = []
    for prompt in PROMPTS:
        body = json.dumps({'model': model, 'prompt': prompt, 'max_tokens': 32,
                           'temperature': 0}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', body,
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r:
            texts.append(json.loads(r.read())['choices'][0]['text'])
    return texts

def container(name, args, volumes, port):
    sh(['docker', 'rm', '-f', name])
    cmd = ['docker', 'run', '-d', '--name', name, '--gpus', f'"device={GPU}"',
           '-p', f'127.0.0.1:{port}:8080']
    for v in volumes:
        cmd += ['-v', v]
    r = sh(cmd + [IMG] + args)
    assert r.returncode == 0, r.stderr
    return name

def finish(name, tag):
    (OUT / f'{tag}.log').write_text(sh(['docker', 'logs', name]).stdout
                                    + sh(['docker', 'logs', name]).stderr)
    t0 = time.time(); stop = sh(['docker', 'stop', '-t', '30', name])
    code = sh(['docker', 'inspect', '-f', '{{.State.ExitCode}}', name]).stdout.strip()
    sh(['docker', 'rm', '-f', name])
    return {'stop_s': round(time.time() - t0, 1), 'exit_code': code}

# A: CLI without GPU
for gpu, args in (([], ['help']), ([], ['presets']), ([], ['riley-checkpoint', '--help']),
                  (['--gpus', f'"device={GPU}"'], ['--version'])):
    r = sh(['docker', 'run', '--rm'] + gpu + [IMG] + args)
    report['cli ' + ' '.join(args)] = {'rc': r.returncode, 'out': (r.stdout + r.stderr)[-1500:]}
    log('cli', args, r.returncode)

# B: fresh download into a new named volume, then a restart that must reuse it
if ONLY is None or 'download' in ONLY:
    vol = f'riley-verify-{int(time.time())}'
    for attempt in ('first', 'restart'):
        t0 = time.time()
        name = container('riley-verify-dl', ['serve', '--preset', 'qwen2.5-0.5b'],
                         [f'{vol}:/models'], 18080)
        ok = ready(18080, time.time() + 900, name=name)
        row = {'ready': ok, 'ready_s': round(time.time() - t0, 1)}
        if ok:
            row['texts'] = complete(18080, 'qwen2.5-0.5b')
        row.update(finish(name, f'download-{attempt}'))
        row['downloaded'] = 'downloading preset' in (OUT / f'download-{attempt}.log').read_text()
        report[f'download {attempt}'] = row
        log('download', attempt, row)
    sh(['docker', 'volume', 'rm', vol])

# C: every preset, image vs measured host binary
for preset, directory in PRESETS.items():
    if ONLY is not None and preset not in ONLY:
        continue
    plan = json.loads(Path(f'{EFS}/{directory}/riley-checkpoint.json').read_text())
    weight = str(sum(f['bytes'] for f in plan['files']))
    flags = ['--model-id', preset, '--inference-profile', 'native-bf16-varlen-v1',
             '--max-weight-bytes', weight]
    row = {}
    t0 = time.time()
    name = container('riley-verify-c', ['serve', '--model', f'/efs/{directory}'] + flags,
                     [f'{EFS}:/efs:ro'], 18082)
    row['image_ready'] = ready(18082, time.time() + 1800, name=name)
    row['image_ready_s'] = round(time.time() - t0, 1)
    if row['image_ready']:
        row['image'] = complete(18082, preset)
    row['image_stop'] = finish(name, f'image-{preset}')

    t0 = time.time()
    env = {'CUDA_VISIBLE_DEVICES': GPU, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'LD_LIBRARY_PATH': '/usr/local/cuda-12.4/lib64', 'PATH': '/usr/bin:/bin'}
    with open(OUT / f'host-{preset}.log', 'w') as hl:
        proc = subprocess.Popen([HOST_BIN, 'serve', '--model', f'{EFS}/{directory}',
                                 '--bind', '127.0.0.1:18083', '--device', '0'] + flags,
                                env=env, stdout=hl, stderr=subprocess.STDOUT)
        row['host_ready'] = ready(18083, time.time() + 1800, proc)
        row['host_ready_s'] = round(time.time() - t0, 1)
        if row['host_ready']:
            row['host'] = complete(18083, preset)
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    row['identical'] = row.get('image') is not None and row.get('image') == row.get('host')
    report[f'preset {preset}'] = row
    log(preset, {k: v for k, v in row.items() if k not in ('image', 'host')})
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))

(OUT / 'report.json').write_text(json.dumps(report, indent=2))
log('done')
