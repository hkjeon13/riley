"""Derive a three-layer canonical fixture without touching the two-layer source."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
source = Path('/tmp/riley-g02c-canonical-fixture')
target = Path('/tmp/riley-g03-odd-canonical-fixture')
target.mkdir(exist_ok=False)
config = json.loads((source / 'config.json').read_text())
config['num_hidden_layers'] = 3
(target / 'config.json').write_text(json.dumps(config) + '\n')
shutil.copyfile(source / 'tokenizer.json', target / 'tokenizer.json')
blob = (source / 'model.safetensors').read_bytes()
length = struct.unpack('<Q', blob[:8])[0]
header = json.loads(blob[8:8+length])
data = blob[8+length:]
result = {k:v for k,v in header.items()}
for name, value in header.items():
    if '.layers.1.' in name:
        result[name.replace('.layers.1.', '.layers.2.')] = value.copy()
output = bytearray()
for name,value in result.items():
    if name == '__metadata__':
        continue
    start,end = value['data_offsets']
    chunk = data[start:end]
    value['data_offsets'] = [len(output),len(output)+len(chunk)]
    output.extend(chunk)
encoded = json.dumps(result, separators=(',', ':')).encode()
encoded += b' ' * (-len(encoded) % 8)
(target / 'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+output)
manifest = json.loads((source / 'riley-checkpoint.json').read_text())
manifest['source_model'] = 'fixture/g03-canonical-llama-h64-l3'
for entry in manifest['files']:
    payload = (target / entry['path']).read_bytes()
    entry.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
(target / 'riley-checkpoint.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
