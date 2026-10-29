"""Verify Hub-id serving and /v1/chat/completions of the riley image on one A100.

For each model: `serve --model <Hub id or preset>` downloads into an EFS-backed
/models, then for several conversations the prompt token IDs Riley feeds the
model (return_token_ids) must equal Transformers' apply_chat_template IDs, and
streamed chat content must equal the non-streamed content.
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

from transformers import AutoTokenizer

IMG, OUT, MODELS = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
GPU, PORT = '6', 18090
CASES = [
    ('Qwen/Qwen2.5-0.5B-Instruct', {}),
    ('Qwen/Qwen3-0.6B', {}),
    ('Qwen/Qwen3-0.6B', {'enable_thinking': False}),
    ('HuggingFaceTB/SmolLM2-360M-Instruct', {}),
    ('unsloth/Llama-3.2-1B-Instruct', {}),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', {}),
    ('Qwen/Qwen2.5-Coder-1.5B-Instruct', {}),
    ('llama-3.2-1b', {}),  # bundled preset with its new chat files
]
CONVERSATIONS = [
    [{'role': 'user', 'content': 'What is the capital of France?'}],
    [{'role': 'system', 'content': 'You are a terse assistant.'},
     {'role': 'user', 'content': 'Name three prime numbers.'}],
    [{'role': 'user', 'content': 'Hi'},
     {'role': 'assistant', 'content': 'Hello! How can I help?'},
     {'role': 'user', 'content': 'Tell me a short joke.'}],
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def post(path, body, stream=False):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read().decode()
            return r.status, raw if stream else json.loads(raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def running(name):
    return sh(['docker', 'inspect', '-f', '{{.State.Running}}', name]).stdout.strip() == 'true'


def ready(name, deadline):
    while time.time() < deadline:
        if not running(name):
            return False
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/readyz', timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def hf_ids(directory, messages, kwargs):
    tokenizer = AutoTokenizer.from_pretrained(directory)
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                        **kwargs)
    if isinstance(ids, dict) or hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    return list(ids)


report = {}
for model, kwargs in CASES:
    key = model + (' ' + json.dumps(kwargs) if kwargs else '')
    row = {}
    name = 'riley-verify-chat'
    sh(['docker', 'rm', '-f', name])
    t0 = time.time()
    r = sh(['docker', 'run', '-d', '--name', name, '--gpus', f'"device={GPU}"',
            '--user', '1003:1003', '-v', f'{MODELS}:/models', '-p', f'127.0.0.1:{PORT}:8080',
            IMG, 'serve', '--model', model, '--max-output-tokens', '64'])
    assert r.returncode == 0, r.stderr
    row['ready'] = ready(name, time.time() + 3600)
    row['ready_s'] = round(time.time() - t0, 1)
    logs = sh(['docker', 'logs', name])
    (OUT / f"{model.replace('/', '--')}.log").write_text(logs.stdout + logs.stderr)
    if row['ready']:
        directory = sh(['docker', 'exec', name, 'sh', '-c',
                        'ls -d /models/hub/*/* /models/* 2>/dev/null']).stdout
        served = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models').read())
        model_id = served['data'][0]['id']
        row['model_id'] = model_id
        # Host path of the checkpoint the container serves.
        inspect = sh(['docker', 'inspect', '-f', '{{json .Args}}', name]).stdout
        cases = []
        for messages in CONVERSATIONS:
            body = {'model': model_id, 'messages': messages, 'max_tokens': 32, 'temperature': 0,
                    'return_token_ids': True}
            if kwargs:
                body['chat_template_kwargs'] = kwargs
            status, response = post('/v1/chat/completions', body)
            case = {'status': status}
            if status == 200:
                choice = response['choices'][0]
                case['content'] = choice['message']['content']
                case['finish_reason'] = choice['finish_reason']
                case['object'] = response['object']
                case['riley_prompt_ids'] = choice.get('prompt_token_ids')
                status_s, streamed = post('/v1/chat/completions',
                                          {**{k: v for k, v in body.items() if k != 'return_token_ids'},
                                           'stream': True}, stream=True)
                content = ''
                for frame in streamed.split('\n\n'):
                    if frame.startswith('data: ') and frame != 'data: [DONE]':
                        chunk = json.loads(frame[6:])
                        content += chunk['choices'][0]['delta'].get('content', '') if chunk.get('choices') else ''
                case['stream_matches'] = status_s == 200 and content == case['content']
            else:
                case['error'] = response
            cases.append(case)
        row['cases'] = cases
        status, plain = post('/v1/completions', {'model': model_id, 'prompt': 'The capital of France is',
                                                 'max_tokens': 8, 'temperature': 0})
        row['completions_ok'] = status == 200
    sh(['docker', 'stop', '-t', '30', name])
    row['exit_code'] = sh(['docker', 'inspect', '-f', '{{.State.ExitCode}}', name]).stdout.strip()
    sh(['docker', 'rm', '-f', name])

    # Reference IDs from the very files the container verified and served.
    if row.get('ready'):
        if '/' in model:
            root = Path(MODELS) / 'hub' / model.replace('/', '--')
            directory = max((p for p in root.iterdir() if (p / 'riley-checkpoint.json').exists()),
                            key=lambda p: p.stat().st_mtime)
        else:
            directory = Path(MODELS) / model
        row['checkpoint'] = str(directory)
        for case, messages in zip(row['cases'], CONVERSATIONS):
            if case.get('riley_prompt_ids') is not None:
                reference = hf_ids(str(directory), messages, kwargs)
                case['hf_prompt_ids_match'] = reference == case['riley_prompt_ids']
                if not case['hf_prompt_ids_match']:
                    case['hf_prompt_ids'] = reference
    report[key] = row
    summary = {k: v for k, v in row.items() if k != 'cases'}
    summary['cases'] = [{k: c.get(k) for k in ('status', 'hf_prompt_ids_match', 'stream_matches',
                                               'finish_reason')} for c in row.get('cases', [])]
    log(key, summary)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
log('done')
