"""Cross-check host riley-checkpoint.json manifests against Hub metadata and emit presets.

For every file: size must equal the Hub size; LFS files must match the Hub's LFS sha256;
non-LFS files must match the Hub git blob id (sha1 of "blob <n>\\0" + bytes) computed
from the local file, and the local sha256 must equal the manifest's.
"""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi

MODELS = Path('/mnt/efs/nlp/psyche/riley-261013/models')
OUT = Path(sys.argv[1])
PRESETS = {
    'smollm2-135m': 'SmolLM2-135M',
    'smollm2-360m': 'SmolLM2-360M',
    'smollm2-1.7b': 'SmolLM2-1.7B',
    'qwen2.5-0.5b': 'Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b': 'Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1',
    'qwen2.5-7b': 'Qwen2.5-7B-Instruct',
    'qwen2.5-14b': 'Qwen2.5-14B-Instruct',
    'qwen2.5-coder-7b': 'Qwen2.5-Coder-7B-Instruct',
    'qwen3-0.6b': 'Qwen3-0.6B',
    'qwen3-1.7b': 'Qwen3-1.7B',
    'qwen3-4b': 'Qwen3-4B',
    'qwen3-8b': 'Qwen3-8B',
    'llama-3.2-1b': 'Llama-3.2-1B-Instruct',
    'llama-3.2-3b': 'Llama-3.2-3B-Instruct',
    'llama-3.1-8b': 'Meta-Llama-3.1-8B-Instruct',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()

api = HfApi()
OUT.mkdir(parents=True, exist_ok=True)
for name, directory in PRESETS.items():
    manifest = json.loads((MODELS / directory / 'riley-checkpoint.json').read_text())
    info = api.model_info(manifest['source_model'], revision=manifest['source_revision'],
                          files_metadata=True)
    assert info.sha == manifest['source_revision'], (name, info.sha)
    hub = {s.rfilename: s for s in info.siblings}
    for row in manifest['files']:
        s = hub[row['path']]
        assert s.size == row['bytes'], (name, row['path'], s.size, row['bytes'])
        if s.lfs:
            assert s.lfs.sha256 == row['sha256'], (name, row['path'], 'lfs sha256')
            how = 'hub-lfs-sha256'
        else:
            local = MODELS / directory / row['path']
            assert git_blob(local) == s.blob_id, (name, row['path'], 'git blob')
            assert sha256(local) == row['sha256'], (name, row['path'], 'local sha256')
            how = 'hub-git-blob+local-sha256'
        print(name, row['path'], how, flush=True)
    (OUT / f'{name}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, 'OK', manifest['source_model'], manifest['source_revision'],
          sum(r['bytes'] for r in manifest['files']), 'gated' if info.gated else 'open', flush=True)
