#!/usr/bin/env python3
"""Build the optional adapter from explicitly supplied, already installed headers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--nvcc',type=Path,required=True)
parser.add_argument('--flashinfer-data',type=Path,required=True)
parser.add_argument('--architecture',action='append',required=True)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
for architecture in args.architecture:
    if not re.fullmatch(r'[0-9]{2,3}[af]?',architecture):
        parser.error('architecture must be an explicit CUDA AOT target such as 89 or 90a')
includes=[args.flashinfer_data/'include',args.flashinfer_data/'cccl/libcudacxx/include',
          args.flashinfer_data/'cccl/cub',args.flashinfer_data/'cccl/thrust']
for path in includes:
    if not path.is_dir():parser.error(f'missing dependency include directory: {path}')
header_digest=hashlib.sha256()
for base in includes:
    for path in sorted(base.rglob('*')):
        if path.is_file():
            header_digest.update(str(path.relative_to(args.flashinfer_data)).encode()+b'\0')
            header_digest.update(hashlib.sha256(path.read_bytes()).digest())
command=[str(args.nvcc),'-std=c++17','-O3','-shared','-Xcompiler','-fPIC']
for arch in args.architecture:command+=['-gencode',f'arch=compute_{arch},code=sm_{arch}']
command += [f'-I{p}' for p in includes]+[str(args.source),'-o',str(args.output)]
args.output.parent.mkdir(parents=True,exist_ok=True)
result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
args.output.with_suffix('.build.log').write_text(result.stdout)
manifest={'command':command,'compiler':subprocess.check_output([str(args.nvcc),'--version'],text=True),
          'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
          'dependency_headers_sha256':header_digest.hexdigest(),'exit_code':result.returncode,
          'library_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest() if result.returncode==0 else None,
          'production_registration':False}
args.output.with_suffix('.build.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
raise SystemExit(result.returncode)
