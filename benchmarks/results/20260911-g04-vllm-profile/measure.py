"""Preparation by default. No process/model is launched without --measure."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(plan):
    for path, expected in plan['immutable_files'].items():
        if digest(path) != expected:
            raise RuntimeError('immutable artifact changed: ' + path)
    revision = subprocess.check_output(['git', '-C', plan['source_root'], 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', plan['source_root'], 'status', '--porcelain'], text=True)
    if revision != plan['source_commit'] or dirty:
        raise RuntimeError('candidate source is not the pinned clean snapshot')


def preflight(plan, directory):
    env = os.environ.copy()
    env['RILEY_PREFLIGHT_OUTPUT_ROOT'] = str(directory)
    env['RILEY_PREFLIGHT_ENVIRONMENT_ID'] = plan['preflight_environment_id']
    # Never lower the reviewed canonical preflight thresholds to make a run pass.
    with (directory/'preflight.stdout').open('x') as out, (directory/'preflight.stderr').open('x') as err:
        subprocess.run(['bash', str(Path(plan['source_root'])/'benchmarks/scripts/preflight.sh')],
                       cwd=plan['source_root'], env=env, stdout=out, stderr=err, check=True)
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if rows:
        raise RuntimeError('GPU is not exclusive; existing compute PIDs: ' + rows)


def http_request(port, request, streaming):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    body = {'model': 'g04-smol', 'prompt': request['prompt'], 'max_tokens': 32,
            'temperature': 0, 'top_p': 1, 'stream': streaming}
    started = time.perf_counter_ns()
    connection.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f'HTTP {response.status}: {response.read()[:300]!r}')
    events = []
    text = ''
    finish = None
    if streaming:
        done = False
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                done = True
                break
            value = json.loads(payload)
            for choice in value.get('choices', []):
                delta = choice.get('text', '')
                if delta:
                    events.append(time.perf_counter_ns())
                    text += delta
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
        if not done or finish != 'length':
            raise RuntimeError('stream did not complete exactly at its length bound')
    else:
        value = json.loads(response.read())
        if value['usage']['prompt_tokens'] != 128 or value['usage']['completion_tokens'] != 32:
            raise RuntimeError('HTTP token counts differ from the fixed cell')
        text = value['choices'][0]['text']
        finish = value['choices'][0]['finish_reason']
    ended = time.perf_counter_ns()
    connection.close()
    return {'started_ns': started, 'finished_ns': ended, 'event_ns': events,
            'text': text, 'finish_reason': finish}


def run_http(plan, lane, directory):
    request = json.loads((ROOT/'request.json').read_text())
    port = lane['port']
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    env = os.environ.copy(); env.update(lane['env'])
    with (directory/'server.log').open('x') as log:
        process = subprocess.Popen(lane['argv'], env=env, cwd=plan['source_root'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    connection.request('GET', '/v1/models')
                    response = connection.getresponse(); response.read(); connection.close()
                    if response.status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('server startup timeout')
                time.sleep(0.2)
            # Functionality and token-count check, outside retained timing records.
            checked = http_request(port, request, False)
            expected = lane['expected_output_text']
            if checked['text'] != expected:
                raise RuntimeError('HTTP output differs from pinned correctness output')
            for _ in range(5):
                http_request(port, request, True)
            with (directory/'http-streaming.jsonl').open('x') as out:
                for index in range(30):
                    row = http_request(port, request, True)
                    if row['text'] != expected:
                        raise RuntimeError('streamed output changed')
                    # SSE events are not necessarily individual tokens. Preserve raw
                    # event timestamps; never label them per-token ITL without routing proof.
                    row.update(index=index, prompt_tokens=128, output_tokens=32,
                               mode='http-streaming', timestamp_unit='ns', event_unit='sse-text-event')
                    out.write(json.dumps(row)+'\n'); out.flush()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    # Kill only the process created by this invocation.
                    process.kill(); process.wait()


def check_vllm_rows(rows, binding):
    def token_hash(ids):
        return hashlib.sha256(b''.join(struct.pack('<I', x) for x in ids)).hexdigest()
    if len(rows) != 30 or [row['trial_index'] for row in rows] != list(range(1, 31)):
        raise RuntimeError('vLLM measured request set differs from the fixed cell')
    for row in rows:
        if row['status'] != 'success' or row['failure_count'] or len(row['requests']) != 1:
            raise RuntimeError('vLLM request failed or concurrency changed')
        req = row['requests'][0]
        if req['status'] != 'success' or (req['prompt_tokens'], req['requested_output_tokens'], req['generated_tokens']) != (128, 32, 32):
            raise RuntimeError('vLLM token count changed')
        if req['prompt_token_ids_sha256'] != token_hash(binding['input_token_ids']) or req['generated_token_ids_sha256'] != token_hash(binding['generated_token_ids']):
            raise RuntimeError('vLLM output differs from the pinned numerical reference')
    return {'valid': True, 'requests': 30, 'tokens_exact': True, 'speedup_claim': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--mode', choices=['engine-only','http-streaming'], default='engine-only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plan = json.loads((ROOT/'measurement-plan.json').read_text())
    validate(plan)
    if not args.measure:
        print(json.dumps({'artifacts_valid': True, 'measurement_started': False,
                          'modes': ['engine-only','http-streaming'], 'pairs':5,
                          'qualification_blockers':plan['qualification_blockers'], 'measurement_environment_blockers':plan['measurement_environment_blockers'], 'next_gate':'exclusive GPU and live preflight'}))
        return
    if plan['qualification_blockers']:
        raise RuntimeError('measurement remains blocked: ' + '; '.join(plan['qualification_blockers']))
    if args.output is None:
        parser.error('--measure requires a new --output directory')
    args.output.mkdir(parents=False, exist_ok=False)
    for pair in plan['pairs']:
        for role in pair['order']:
            directory = args.output/f'pair-{pair["index"]:02d}-{role}'
            directory.mkdir()
            validate(plan)
            preflight(plan, directory)
            if args.mode == 'http-streaming':
                run_http(plan, plan['http_lanes'][role], directory)
            else:
                lane = plan['engine_lanes'][role]
                values = {'index': str(pair['index']), 'output': str(directory),
                          'started_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
                argv = [x.format(**values) for x in lane['argv']]
                env = os.environ.copy(); env.update(lane['env'])
                with (directory/'stdout.log').open('x') as out, (directory/'stderr.log').open('x') as err:
                    subprocess.run(argv, cwd=plan['source_root'], env=env, stdout=out, stderr=err, check=True)
                if role == 'riley':
                    with (directory/'reference-check.json').open('x') as checked:
                        subprocess.run([plan['reference_checker_python'],str(Path(plan['source_root'])/'benchmarks/scripts/check_vllm_profile_run.py'),str(directory/'native-profile.json'),'--binding',str(ROOT/'native-binding.json')],stdout=checked,check=True)
                else:
                    rows = [json.loads(line) for line in (directory/'vllm/raw.jsonl').read_text().splitlines() if line.strip()]
                    checked = check_vllm_rows(rows, json.loads((ROOT/'native-binding.json').read_text()))
                    with (directory/'reference-check.json').open('x') as out:
                        json.dump(checked, out)

            (directory/'execution-complete.json').write_text(json.dumps({'completed':True,'mode':args.mode}))


if __name__ == '__main__':
    main()
